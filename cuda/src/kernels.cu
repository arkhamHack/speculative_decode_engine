#include "kernels.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>

#include <curand_kernel.h>

namespace cg = cooperative_groups;

// Ampere+ may require raising the dynamic shared-memory cap when scratch > 48 KiB.
template<typename K>
static inline void cuda_configure_kernel_dynamic_smem(K kernel, size_t smem_bytes) {
    constexpr size_t kDefaultDynSmemCap = 49152;
    if (smem_bytes > kDefaultDynSmemCap) {
        CUDA_CHECK(cudaFuncSetAttribute(
            (void*)kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes));
    }
}

// ============================================================================
// Shared memory layout (dynamic — sized at launch via compute_smem_bytes())
//
//   shared[0          .. d_model)         hidden      [d_model]
//   shared[d_model .. end]   scratch for layers + final RMSNorm
//                               (streaming attention + tiled MLP — see model.cu)
//
// Logits are written to a global-memory buffer (g_logits) allocated by the
// host wrapper.  This removes the vocab_size limit from shared memory and
// allows real models with vocab_size >> 256.
// ============================================================================

// Forward pass only: populate g_logits (no argmax).
__global__ void single_token_forward_logits_kernel(
        ModelWeights model, KVCache kv,
        int token_id, int seq_len,
        float* g_logits) {
    extern __shared__ float shared[];
    int d = model.cfg.d_model;
    float* hidden = shared;
    float* smem   = shared + d;
    model_forward_logits(model, kv, token_id, seq_len, hidden, g_logits, smem);
}

// -----------------------------------------------------------------------------
// Leviathan/Chen stochastic helpers (single block, logits in global memory)
//
// Scratch smem reused from the normed region prefix (>= warp reduction slots via
// d_model floats is typical). After forward logits, hidden/norm/etc. are disposable.
// -----------------------------------------------------------------------------

// Read-only softmax mass at logits[idx]/temp (does not mutate logits array).
__device__ float logits_softmax_prob_at_global(const float* logits, int V, int idx,
                                               float temperature, float* smem) {
    if (temperature < 1e-6f)
        temperature = 1e-6f;

    int tid = threadIdx.x;
    float   local_max = -FLT_MAX;
    for (int i = tid; i < V; i += blockDim.x) {
        float z = logits[i] / temperature;
        local_max = fmaxf(local_max, z);
    }
    float max_val = block_reduce_max(local_max, smem);

    float local_sum = 0.f;
    for (int i = tid; i < V; i += blockDim.x)
        local_sum += expf(logits[i] / temperature - max_val);
    float sum_val = block_reduce_sum(local_sum, smem);

    if (tid == 0)
        smem[0] = expf(logits[idx] / temperature - max_val) / sum_val;
    __syncthreads();

    return smem[0];
}

// Convert logits[T] → softmax(logits/T) in place using block_softmax_inplace.
__device__ void logits_inplace_softmax_temp(float* logits, int V, float temperature,
                                           float* smem_reduction) {
    if (temperature < 1e-6f) temperature = 1e-6f;
    int tid = threadIdx.x;
    for (int i = tid; i < V; i += blockDim.x)
        logits[i] /= temperature;
    __syncthreads();

    block_softmax_inplace(logits, V, smem_reduction);
    __syncthreads();
}

// Draft forward + temperature-softmax logits + categorical sample (thread 0 only).
__global__ void stochastic_draft_forward_sample_kernel(
        ModelWeights draft_model,
        KVCache kv,
        int token_id,
        int seq_len,
        float temperature,
        curandState* rng,
        float* g_logits,
        int* sampled_id,
        float* sampled_q_prob // device float[1], q(token) mass under tempered softmax
        ) {
    extern __shared__ float shared[];

    int d = draft_model.cfg.d_model;
    int V = draft_model.cfg.vocab_size;
    float* hidden = shared;
    float* smem   = shared + d;

    model_forward_logits(draft_model, kv, token_id, seq_len, hidden, g_logits,
                         smem);
    __syncthreads();

    float* softmax_scratch = hidden; // repurpose scratch after forward [d floats]
    logits_inplace_softmax_temp(g_logits, V, temperature, softmax_scratch);

    if (threadIdx.x == 0) {
        float u     = curand_uniform(rng);
        float cdf   = 0.f;
        int   choice = V - 1;
        for (int i = 0; i < V; i++) {
            cdf += g_logits[i];
            if (u <= cdf || i == V - 1) {
                choice = i;
                break;
            }
        }

        sampled_id[0]       = choice;
        sampled_q_prob[0]   = g_logits[choice];
        // restore raw logits unnecessary — next sampling path overwrites logits.
    }

    __syncthreads();
}

// Device-pointer variant: reads token_id from device memory so consecutive draft
// steps can be chained without a CPU roundtrip between them.
__global__ void stochastic_draft_forward_sample_dptr_kernel(
        ModelWeights draft_model,
        KVCache kv,
        const int* d_token_id,   // ← device pointer (output of previous step)
        int seq_len,
        float temperature,
        curandState* rng,
        float* g_logits,
        int* sampled_id,
        float* sampled_q_prob) {
    extern __shared__ float shared[];

    int token_id = d_token_id[0];
    int d = draft_model.cfg.d_model;
    int V = draft_model.cfg.vocab_size;
    float* hidden = shared;
    float* smem   = shared + d;

    model_forward_logits(draft_model, kv, token_id, seq_len, hidden, g_logits, smem);
    __syncthreads();

    float* softmax_scratch = hidden;
    logits_inplace_softmax_temp(g_logits, V, temperature, softmax_scratch);

    if (threadIdx.x == 0) {
        float u      = curand_uniform(rng);
        float cdf    = 0.f;
        int   choice = V - 1;
        for (int i = 0; i < V; i++) {
            cdf += g_logits[i];
            if (u <= cdf || i == V - 1) { choice = i; break; }
        }
        sampled_id[0]     = choice;
        sampled_q_prob[0] = g_logits[choice];
    }
    __syncthreads();
}

// Forward target, then softmax p(idx) without destroying raw logits buffer.
__global__ void target_forward_prob_mass_kernel(ModelWeights target_model,
                                                KVCache kv,
                                                int token_id,
                                                int seq_len,
                                                int idx,
                                                float* g_logits,
                                                float* mass_out_device // float[1]
                                                ) {
    extern __shared__ float shared[];
    int d = target_model.cfg.d_model;
    int V = target_model.cfg.vocab_size;

    float* hidden  = shared;
    float* lay_smem = shared + d;

    model_forward_logits(target_model, kv, token_id, seq_len, hidden,
                         g_logits, lay_smem);
    __syncthreads();

    float* softmax_reduction = hidden; // repurposed — forward complete
    float  pm                = logits_softmax_prob_at_global(
        g_logits, V, idx, 1.f, softmax_reduction);
    __syncthreads();

    if (threadIdx.x == 0)
        mass_out_device[0] = pm;
}

__global__ void stochastic_accept_gate_kernel(float               p_mass,
                                              float               q_mass,
                                               curandState* rng_state,
                                               int* accepted_flag // 0/1
                                              ) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    float q_eff = fmaxf(q_mass, 1e-36f);
    float cap   = fminf(1.f, p_mass / q_eff);
    accepted_flag[0] = (curand_uniform(rng_state) <= cap) ? 1 : 0;
}

// Fused kernel: target forward pass + p-mass computation + stochastic acceptance gate.
// Replaces target_forward_prob_mass_kernel + stochastic_accept_gate_kernel (two launches,
// two syncs, two memcpys) with a single launch + one sync + one memcpy per verify step.
// g_logits holds raw target logits on exit (not mutated) so corrected_sample can follow.
__global__ void target_fwd_prob_and_accept_kernel(
        ModelWeights target_model,
        KVCache kv,
        int token_id, int seq_len,
        int draft_token,    // token whose mass under target we test
        float q_mass,       // draft probability mass for draft_token
        float* g_logits,    // [vocab_size] — raw target logits on exit
        curandState* rng,
        int* accepted_flag) // 0 or 1
{
    extern __shared__ float shared[];
    int d = target_model.cfg.d_model;
    int V = target_model.cfg.vocab_size;
    float* hidden   = shared;
    float* lay_smem = shared + d;

    model_forward_logits(target_model, kv, token_id, seq_len, hidden, g_logits, lay_smem);
    __syncthreads();

    // logits_softmax_prob_at_global is read-only; raw logits are preserved for
    // corrected_sample_adjusted_logits_kernel that may follow a rejection.
    float* softmax_reduction = hidden;
    float pm = logits_softmax_prob_at_global(g_logits, V, draft_token, 1.f, softmax_reduction);
    __syncthreads();

    if (threadIdx.x == 0) {
        float q_eff = fmaxf(q_mass, 1e-36f);
        float cap   = fminf(1.f, pm / q_eff);
        accepted_flag[0] = (curand_uniform(rng) <= cap) ? 1 : 0;
    }
}

// Shared device helpers — megakernel + multi-kernel use the same acceptance / sampling maths.
__device__ __forceinline__ bool device_stochastic_accept_mass(float    p_mass,
                                                               float    q_mass,
                                                               curandState* rng_state) {
    float q_eff = fmaxf(q_mass, 1e-36f);
    float cap   = fminf(1.f, p_mass / q_eff);
    return curand_uniform(rng_state) <= cap;
}

// threadIdx.x == 0 only; caller wraps with barriers.
__device__ int device_softmax_sample_logits_temp_inplace(float* logits,
                                                         int               V,
                                                        float temperature,
                                                         curandState* rng_state) {
    float inv_t = temperature < 1e-6f ? 1e6f : 1.f / temperature;

    float mx = -FLT_MAX;
    for (int i = 0; i < V; i++)
        mx = fmaxf(mx, logits[i] * inv_t);
    float s = 0.f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] * inv_t - mx);
        s += logits[i];
    }

    float u_fix = curand_uniform(rng_state);
    float cdf   = 0.f;
    for (int i = 0; i < V; i++) {
        cdf += logits[i] / s;
        if (cdf >= u_fix || i == V - 1)
            return i;
    }
    return V - 1;
}

__device__ int device_corrected_adjusted_sample(float* logits_p,
                                                float* logits_q,
                                                 int               V,
                                                curandState* rng_state,
                                                float* work,
                                                 float tp,
                                                 float tq) {
    float* wp = work;
    float* wq = work + V;
    float* wr = work + 2 * V;

    float inv_tp = tp < 1e-6f ? 1e6f : 1.f / tp;
    float inv_tq = tq < 1e-6f ? 1e6f : 1.f / tq;

    float max_p = -FLT_MAX;
    float max_q = -FLT_MAX;
    for (int i = 0; i < V; i++)
        max_p = fmaxf(max_p, logits_p[i] * inv_tp);
    for (int i = 0; i < V; i++)
        max_q = fmaxf(max_q, logits_q[i] * inv_tq);

    float sum_p = 0.f;
    float sum_q = 0.f;
    for (int i = 0; i < V; i++) {
        wp[i] = expf(logits_p[i] * inv_tp - max_p);
        wq[i] = expf(logits_q[i] * inv_tq - max_q);
        sum_p += wp[i];
        sum_q += wq[i];
    }

    float sum_corr = 0.f;
    for (int i = 0; i < V; i++) {
        float pi = wp[i] / sum_p;
        float qi = wq[i] / sum_q;
        float dj = pi - qi;
        if (dj > 0.f) {
            wr[i] = dj;
            sum_corr += dj;
        } else
            wr[i] = 0.f;
    }

    if (sum_corr <= 1e-20f) {
        float u_fix = curand_uniform(rng_state);
        float cdf  = 0.f;
        for (int i = 0; i < V; i++) {
            cdf += wp[i] / sum_p;
            if (cdf >= u_fix || i == V - 1)
                return i;
        }
        return V - 1;
    }

    float u_fix = curand_uniform(rng_state);
    float cdf   = 0.f;
    for (int i = 0; i < V; i++) {
        cdf += wr[i] / sum_corr;
        if (cdf >= u_fix || i == V - 1)
            return i;
    }
    return V - 1;
}

// Corrected sampler: logits_p and logits_q are RAW logits vectors (already filled).
// Computes p̂=max(0,softmax(p)-softmax(q)) then samples correction token (~Leviathan).
__global__ void corrected_sample_adjusted_logits_kernel(float* logits_p,
                                                        float* logits_q,
                                                         int               V,
                                                        curandState* rng,
                                                        int* out_token,
                                                        float* work,
                                                        float temperature_p,
                                                        float temperature_q) {
    // Single-thread GPU kernel acceptable for prototyping (V vocab-wide scalar loops).
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float* wp = work;
    float* wq = work + V;
    float* wr = work + 2 * V;

    float inv_tp = temperature_p < 1e-6f ? 1e6f : 1.f / temperature_p;
    float inv_tq = temperature_q < 1e-6f ? 1e6f : 1.f / temperature_q;

    float max_p = -FLT_MAX;
    float max_q = -FLT_MAX;
    for (int i = 0; i < V; i++) {
        float z = logits_p[i] * inv_tp;
        max_p = fmaxf(max_p, z);
    }
    for (int i = 0; i < V; i++) {
        float z = logits_q[i] * inv_tq;
        max_q = fmaxf(max_q, z);
    }

    float sum_p = 0.f;
    float sum_q = 0.f;
    for (int i = 0; i < V; i++) {
        wp[i] = expf(logits_p[i] * inv_tp - max_p);
        wq[i] = expf(logits_q[i] * inv_tq - max_q);
        sum_p += wp[i];
        sum_q += wq[i];
    }

    float sum_corr = 0.f;
    for (int i = 0; i < V; i++) {
        float pi = wp[i] / sum_p;
        float qi = wq[i] / sum_q;
        float dj = pi - qi;
        if (dj > 0.f) {
            wr[i] = dj;
            sum_corr += dj;
        } else
            wr[i] = 0.f;
    }

    if (sum_corr <= 1e-20f) {
        float u_fix = curand_uniform(rng);
        float cdf  = 0.f;
        for (int i = 0; i < V; i++) {
            cdf += wp[i] / sum_p;
            if (cdf >= u_fix || i == V - 1) {
                out_token[0] = i;
                return;
            }
        }
        out_token[0] = V - 1;
        return;
    }

    float u_fix = curand_uniform(rng);
    float cdf   = 0.f;
    for (int i = 0; i < V; i++) {
        cdf += wr[i] / sum_corr;
        if (cdf >= u_fix || i == V - 1) {
            out_token[0] = i;
            return;
        }
    }
    out_token[0] = V - 1;
}

__global__ void softmax_sample_temperature_kernel(float* logits,
                                                   int               V,
                                                  float temperature,
                                                  curandState* rng,
                                                  int* out_token) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float inv_t = temperature < 1e-6f ? 1e6f : 1.f / temperature;

    float mx = -FLT_MAX;
    for (int i = 0; i < V; i++)
        mx = fmaxf(mx, logits[i] * inv_t);
    float s = 0.f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] * inv_t - mx);
        s += logits[i];
    }

    float u_fix = curand_uniform(rng);
    float cdf   = 0.f;
    for (int i = 0; i < V; i++) {
        cdf += logits[i] / s;
        if (cdf >= u_fix || i == V - 1) {
            out_token[0] = i;
            return;
        }
    }
    out_token[0] = V - 1;
}

__global__ void rng_init_kernel(curandState* state,
                                unsigned long long seed_worker) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        curand_init(seed_worker, 0ULL, 0ULL, state);
}

// ============================================================================
// Standalone single-block kernels for intra-layer Q/K/V + gate/up stream overlap
//
// These are launched individually on separate CUDA streams so that Q, K, V
// projections (and gate, up projections) execute concurrently on the GPU
// when the device has sufficient SM resources.
//
// All take global-memory pointers; no shared-memory coupling between kernels.
// ============================================================================

// Embedding lookup → global memory hidden state.
__global__ void embed_global_kernel(
    ModelWeights model, int token_id, float* g_hidden)
{
    model_embed(model, token_id, g_hidden);
}

// RMSNorm: g_x → g_out, using static smem for the warp reduction.
__global__ void rmsnorm_global_kernel(
    const float* __restrict__ g_x,
    const half*  __restrict__ g_weight,
    float*       g_out,
    int d)
{
    __shared__ float s_red[BLOCK_THREADS / WARP_SIZE];
    device_rmsnorm(g_x, g_weight, g_out, d, s_red);
}

// Full GEMV: g_x @ g_W → g_out (single block, stride loop over d_out).
__global__ void gemv_global_kernel(
    const float* __restrict__ g_x,
    const half*  __restrict__ g_W,
    float*       g_out,
    int d_in, int d_out)
{
    device_matvec(g_x, g_W, g_out, d_in, d_out);
}

// Multi-block GEMV: each block handles one BLOCK_THREADS-wide column tile.
// Launch with <<<(d_out+BLOCK_THREADS-1)/BLOCK_THREADS, BLOCK_THREADS, 0, stream>>>.
// Uses L1 read-only cache (__ldg) for weight matrix; input vector fits in L2.
// Compared to gemv_global_kernel (1 SM) this fans out across all available SMs.
__global__ void gemv_mb_kernel(
    const float* __restrict__ g_x,
    const half*  __restrict__ g_W,
    float*       g_out,
    int d_in, int d_out)
{
    int col = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    if (col >= d_out) return;
    float acc = 0.f;
    for (int row = 0; row < d_in; row++)
        acc += g_x[row] * __half2float(__ldg(&g_W[row * d_out + col]));
    g_out[col] = acc;
}

// In-place RoPE on Q and K stored in global memory.
__global__ void rope_qk_global_kernel(
    float* g_q, float* g_k,
    int n_heads, int d_head_per, int seq_pos, float rope_theta)
{
    rope_apply_heads_qk_inplace(g_q, g_k, n_heads, d_head_per,
                                seq_pos, rope_theta);
}

// Write pre-computed (and RoPE'd) K and V into the paged KV cache.
__global__ void kv_write_kernel(
    KVCache kv, int layer_idx,
    const float* g_k, const float* g_v,
    int seq_pos)
{
    kv_cache_append(kv, layer_idx, g_k, g_v, seq_pos);
}

// Flash attention with Q supplied from global memory.
// Uses causal mask: attends to positions [0 .. total_len).
// Static shared memory only — no dynamic smem argument needed.
__global__ void flash_attn_global_kernel(
    const float* g_q,       // [d_model] Q (with RoPE)
    float*       g_attn_out,// [d_model] output (unnormalised accumulator)
    KVCache kv, int layer_idx,
    int d, int n_heads, int total_len, float scale)
{
    __shared__ float blk_logits[KV_BLOCK_SIZE];
    __shared__ float s_m, s_l, s_m_tile, s_alpha, s_inv_den;

    const int tid = threadIdx.x;
    const int dph = d / n_heads;
    const int n_log_blks = (total_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    for (int i = tid; i < d; i += blockDim.x) g_attn_out[i] = 0.f;
    __syncthreads();

    for (int h = 0; h < n_heads; h++) {
        const int ho = h * dph;

        if (tid == 0) { s_m = -FLT_MAX; s_l = 0.f; }
        for (int ei = tid; ei < dph; ei += blockDim.x)
            g_attn_out[ho + ei] = 0.f;
        __syncthreads();

        for (int lb = 0; lb < n_log_blks; lb++) {
            const int p0 = lb * KV_BLOCK_SIZE;

            for (int sl = tid; sl < KV_BLOCK_SIZE; sl += blockDim.x) {
                int p = p0 + sl;
                float logit = -FLT_MAX;
                if (p < total_len) {
                    int bidx = p / KV_BLOCK_SIZE, sl_kv = p % KV_BLOCK_SIZE;
                    int pb   = kv.layers[layer_idx].block_table[bidx];
                    const half* base =
                        kv.pool + (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                    const half* k_src = base + sl_kv * kv.d_head + ho;
                    float dot = 0.f;
                    for (int e = 0; e < dph; e++)
                        dot += g_q[ho + e] * __half2float(k_src[e]);
                    logit = dot * scale;
                }
                blk_logits[sl] = logit;
            }
            __syncthreads();

            if (tid == 0) {
                float mt = -FLT_MAX;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++)
                    if (p0 + sl < total_len) mt = fmaxf(mt, blk_logits[sl]);
                float mp = s_m, mn = fmaxf(mp, mt);
                float au = (lb > 0) ? expf(mp - mn) : 1.f;
                float lt = 0.f;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++)
                    if (p0 + sl < total_len) lt += expf(blk_logits[sl] - mn);
                s_m = mn;  s_l = au * s_l + lt;
                s_m_tile = mn;  s_alpha = au;
            }
            __syncthreads();

            float rp = s_alpha;
            for (int ei = tid; ei < dph; ei += blockDim.x)
                g_attn_out[ho + ei] *= rp;
            __syncthreads();

            float mh = s_m_tile;
            for (int ei = tid; ei < dph; ei += blockDim.x) {
                int ge = ho + ei;
                float acc = 0.f;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++) {
                    int p = p0 + sl;
                    if (p >= total_len) continue;
                    float w = expf(blk_logits[sl] - mh);
                    int bidx = p / KV_BLOCK_SIZE, sl_kv = p % KV_BLOCK_SIZE;
                    int pb   = kv.layers[layer_idx].block_table[bidx];
                    const half* base =
                        kv.pool + (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                    const half* v_src =
                        base + KV_BLOCK_SIZE * kv.d_head + sl_kv * kv.d_head;
                    acc += w * __half2float(v_src[ge]);
                }
                g_attn_out[ge] += acc;
            }
            __syncthreads();
        }

        if (tid == 0)
            s_inv_den = (s_l > 1e-20f && isfinite(s_l)) ? (1.f / s_l) : 0.f;
        __syncthreads();
        float nd = s_inv_den;
        for (int ei = tid; ei < dph; ei += blockDim.x)
            g_attn_out[ho + ei] *= nd;
        __syncthreads();
    }
}

// Element-wise residual add: g_x[i] += g_delta[i].
__global__ void residual_add_kernel(float* g_x, const float* g_delta, int d)
{
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        g_x[i] += g_delta[i];
    __syncthreads();
}

// Zero a float buffer of length n.
__global__ void zero_buf_kernel(float* g_buf, int n)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        g_buf[i] = 0.f;
    __syncthreads();
}

// Full gate or up projection: g_x @ g_W → g_out  (same as gemv_global_kernel;
// kept as a named alias for readability in the MLP overlap path).
__global__ void proj_global_kernel(
    const float* __restrict__ g_x,
    const half*  __restrict__ g_W,
    float*       g_out,
    int d_in, int d_out)
{
    device_matvec(g_x, g_W, g_out, d_in, d_out);
}

// Fused SwiGLU + W_down GEMV, writing result to g_out (overwrites, not accumulates).
// g_gate is modified in-place to hold the activated values; g_up is read-only.
__global__ void swiglu_down_kernel(
    float*       g_gate,    // [dff]  in/out: gate logits → SwiGLU(gate)*up
    const float* g_up,      // [dff]
    const half*  W_down,    // [dff, d] row-major
    float*       g_out,     // [d]
    int d, int dff)
{
    int tid = threadIdx.x;

    // Pass 1: SwiGLU activation in-place on g_gate.
    for (int i = tid; i < dff; i += blockDim.x) {
        float g  = g_gate[i];
        g_gate[i] = (g / (1.f + expf(-g))) * g_up[i];
    }
    __syncthreads();

    // Pass 2: W_down @ g_gate_activated → g_out.
    // g_gate (activated, d_ff floats) fits in L2 cache on all target GPUs.
    for (int oc = tid; oc < d; oc += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < dff; i++)
            acc += g_gate[i] * __half2float(__ldg(&W_down[i * d + oc]));
        g_out[oc] = acc;
    }
    __syncthreads();
}

// Final RMSNorm + output projection + greedy argmax in one block.
// smem must be at least 2*BLOCK_THREADS floats (for global_argmax scratch).
__global__ void output_argmax_global_kernel(
    ModelWeights model,
    const float* g_hidden,
    float*       g_normed,   // scratch for normalised hidden (global memory)
    float*       g_logits,   // [vocab_size]
    int*         d_next)
{
    extern __shared__ float smem[];
    const int d = model.cfg.d_model;
    const int V = model.cfg.vocab_size;

    device_rmsnorm(g_hidden, model.rms_final_weight, g_normed, d, smem);
    device_matvec(g_normed, model.output_proj, g_logits, d, V);
    int next = global_argmax(g_logits, V, smem);
    if (threadIdx.x == 0) *d_next = next;
}

// ============================================================================
// launch_prefill_overlapped
//
// Runs a complete single-token forward pass through `model` using separate
// CUDA streams so that projections that are data-independent (Q||K||V and
// gate||up) execute concurrently on the GPU.
//
// Stream assignment:
//   s_main    — sequential work (embed, RMSNorm, RoPE, attention, residuals,
//               output head, Q projection, gate projection)
//   s_k       — K projection (parallel with Q)
//   s_v_gate  — V projection (parallel with Q+K); up projection (parallel
//               with gate in the MLP sub-layer)
//
// The three streams are synchronised with ovl_events[0..2] at each
// data-dependency boundary.  Results are written to the named regions of
// d_ovl_buf (see InferenceEngine layout in model.h).
//
// After return, d_next_token contains the greedy next-token argmax.
// The caller must set_seq_len_kernel and/or sync s_main as needed.
//
// Only activated for models with d_model >= OVERLAP_MIN_D.
// ============================================================================

static constexpr int OVERLAP_MIN_D = 1024;  // skip tiny/dummy models

static void launch_prefill_overlapped(
    const ModelWeights& model, KVCache& kv,
    int token_id, int seq_pos,
    float* d_ovl_buf,       // [7*d + 2*dff] work buffer for this model
    float* g_logits,        // [vocab_size]  output logits
    int*   d_next_token,    // output: greedy argmax
    InferenceEngine& eng,
    cudaStream_t s_main,    // primary stream
    cudaStream_t s_k,       // K-projection auxiliary
    cudaStream_t s_v_gate)  // V-projection + up-projection auxiliary
{
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int nh  = model.cfg.n_heads;
    const int dph = d / nh;

    // Named views into d_ovl_buf (see layout in model.h)
    float* g_hidden  = d_ovl_buf;
    float* g_normed  = d_ovl_buf + d;
    float* g_q       = d_ovl_buf + 2 * d;
    float* g_k       = d_ovl_buf + 3 * d;
    float* g_v       = d_ovl_buf + 4 * d;
    float* g_attn    = d_ovl_buf + 5 * d;
    float* g_mlp_acc = d_ovl_buf + 6 * d;
    float* g_gate    = d_ovl_buf + 7 * d;
    float* g_up      = d_ovl_buf + 7 * d + dff;

    cudaEvent_t ev0 = eng.ovl_events[0];   // RMSNorm done / Q done
    cudaEvent_t ev1 = eng.ovl_events[1];   // K done / gate done
    cudaEvent_t ev2 = eng.ovl_events[2];   // V done / up  done

    // smem for RMSNorm kernels (warp-reduction scratch)
    const size_t smem_red = (BLOCK_THREADS / WARP_SIZE) * sizeof(float);

    // Block counts for multi-block GEMV: one thread per output column.
    // Falls back to 1 block for tiny dummy models (d < BLOCK_THREADS).
    const int nb_d   = (d   + BLOCK_THREADS - 1) / BLOCK_THREADS;  // QKV / Wo
    const int nb_dff = (dff + BLOCK_THREADS - 1) / BLOCK_THREADS;  // gate / up

    // ---- Embedding lookup ----
    embed_global_kernel<<<1, BLOCK_THREADS, 0, s_main>>>(
        model, token_id, g_hidden);

    for (int l = 0; l < model.cfg.n_layers; l++) {
        const LayerWeights& lw = model.layers[l];

        // ================================================================
        // Attention sub-layer
        // ================================================================

        // -- RMSNorm (attn) --
        rmsnorm_global_kernel<<<1, BLOCK_THREADS, smem_red, s_main>>>(
            g_hidden, lw.rms_attn_weight, g_normed, d);
        CUDA_CHECK(cudaEventRecord(ev0, s_main));    // normed ready

        // Tell K and V streams to wait for normed
        CUDA_CHECK(cudaStreamWaitEvent(s_k,      ev0, 0));
        CUDA_CHECK(cudaStreamWaitEvent(s_v_gate, ev0, 0));

        // -- Q || K || V projections (three-way parallel, multi-block) --
        gemv_mb_kernel<<<nb_d, BLOCK_THREADS, 0, s_main>>>(
            g_normed, lw.Wq, g_q, d, d);                       // Q on s_main
        gemv_mb_kernel<<<nb_d, BLOCK_THREADS, 0, s_k>>>(
            g_normed, lw.Wk, g_k, d, d);                       // K on s_k
        gemv_mb_kernel<<<nb_d, BLOCK_THREADS, 0, s_v_gate>>>(
            g_normed, lw.Wv, g_v, d, d);                       // V on s_v_gate

        CUDA_CHECK(cudaEventRecord(ev1, s_k));       // K done
        CUDA_CHECK(cudaEventRecord(ev2, s_v_gate));  // V done

        // -- RoPE: needs Q (s_main) and K (ev1) done --
        CUDA_CHECK(cudaStreamWaitEvent(s_main, ev1, 0));
        rope_qk_global_kernel<<<1, BLOCK_THREADS, 0, s_main>>>(
            g_q, g_k, nh, dph, seq_pos, model.cfg.rope_theta);

        // -- KV cache write: needs RoPE'd K and V (ev2) done --
        CUDA_CHECK(cudaStreamWaitEvent(s_main, ev2, 0));
        kv_write_kernel<<<1, BLOCK_THREADS, 0, s_main>>>(
            kv, l, g_k, g_v, seq_pos);

        // -- Flash attention (Q from global memory, causal mask to seq_pos+1) --
        flash_attn_global_kernel<<<1, BLOCK_THREADS, 0, s_main>>>(
            g_q, g_attn, kv, l, d, nh, seq_pos + 1,
            rsqrtf((float)dph));

        // -- Wo projection + residual add (multi-block) --
        gemv_mb_kernel<<<nb_d, BLOCK_THREADS, 0, s_main>>>(
            g_attn, lw.Wo, g_normed, d, d);        // g_normed = Wo * attn_out
        residual_add_kernel<<<1, BLOCK_THREADS, 0, s_main>>>(
            g_hidden, g_normed, d);                 // hidden += Wo_out

        // ================================================================
        // MLP sub-layer
        // ================================================================

        // -- RMSNorm (MLP) --
        rmsnorm_global_kernel<<<1, BLOCK_THREADS, smem_red, s_main>>>(
            g_hidden, lw.rms_mlp_weight, g_normed, d);
        CUDA_CHECK(cudaEventRecord(ev0, s_main));    // MLP normed ready

        // Tell up stream to wait for normed before starting its projection
        CUDA_CHECK(cudaStreamWaitEvent(s_v_gate, ev0, 0));

        // -- Gate || Up projections (two-way parallel, multi-block) --
        gemv_mb_kernel<<<nb_dff, BLOCK_THREADS, 0, s_main>>>(
            g_normed, lw.W_gate, g_gate, d, dff);   // gate on s_main
        gemv_mb_kernel<<<nb_dff, BLOCK_THREADS, 0, s_v_gate>>>(
            g_normed, lw.W_up,   g_up,  d, dff);    // up on s_v_gate

        CUDA_CHECK(cudaEventRecord(ev2, s_v_gate));  // up done

        // -- Fused SwiGLU + W_down → g_mlp_acc (needs gate and up done) --
        CUDA_CHECK(cudaStreamWaitEvent(s_main, ev2, 0));
        swiglu_down_kernel<<<1, BLOCK_THREADS, 0, s_main>>>(
            g_gate, g_up, lw.W_down, g_mlp_acc, d, dff);

        // -- Residual add --
        residual_add_kernel<<<1, BLOCK_THREADS, 0, s_main>>>(
            g_hidden, g_mlp_acc, d);
    }

    // ---- Output head: final RMSNorm → output_proj → argmax ----
    const size_t smem_out = 2 * BLOCK_THREADS * sizeof(float);
    cuda_configure_kernel_dynamic_smem(output_argmax_global_kernel, smem_out);
    output_argmax_global_kernel<<<1, BLOCK_THREADS, smem_out, s_main>>>(
        model, g_hidden, g_normed, g_logits, d_next_token);
}

// ============================================================================
//  MULTI-KERNEL PATH
// ============================================================================

// Single-token decode kernel.
// g_logits: pre-allocated global-memory buffer of vocab_size floats.
__global__ void single_token_decode_kernel(
        ModelWeights model, KVCache kv,
        int token_id, int seq_len,
        float* g_logits, int* out) {
    extern __shared__ float shared[];
    int d = model.cfg.d_model;
    float* hidden = shared;
    float* smem   = shared + d;

    int next =
        model_forward(model, kv, token_id, seq_len, hidden, g_logits, smem);
    if (threadIdx.x == 0) *out = next;
}

// Update seq_len counter on device.
__global__ void set_seq_len_kernel(int* seq_len_ptr, int val) {
    *seq_len_ptr = val;
}

// Copy GenerationResult fields from device temp storage to result struct.
__global__ void write_result_kernel(GenerationResult* result,
                                    const int* tokens, int n,
                                    int proposed, int accepted, int iters) {
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x)
        result->output_tokens[i] = tokens[i];
    if (tid == 0) {
        result->n_generated     = n;
        result->draft_proposed  = proposed;
        result->draft_accepted  = accepted;
        result->spec_iterations = iters;
    }
}

// ---- Batched draft/verify kernels ----
//
// These process k (draft) or k+1 (verify) tokens sequentially inside one GPU
// kernel, eliminating the O(k) CPU-GPU round trips that made the per-token
// multi-kernel loop slower than baseline.  Compute is unchanged; what we save
// is k kernel-launch round trips (~50–200 µs each on Windows) per round.

// Autoregressive draft: k tokens chained inside one kernel.
// Caller updates draft_kv.seq_len via set_seq_len_kernel after this returns.
__global__ void batch_draft_greedy_kernel(
        ModelWeights model, KVCache kv,
        int first_token, int k, int start_seq_len,
        float* g_logits,   // [vocab_size] global scratch, reused per step
        int*   out_tokens) // [k] predicted draft tokens
{
    extern __shared__ float shared[];
    int d      = model.cfg.d_model;
    float* hidden = shared;
    float* smem   = shared + d;

    __shared__ int s_cur;
    if (threadIdx.x == 0) s_cur = first_token;
    __syncthreads();

    for (int i = 0; i < k; i++) {
        int nxt = model_forward(model, kv, s_cur, start_seq_len + i,
                                hidden, g_logits, smem);
        if (threadIdx.x == 0) { out_tokens[i] = nxt; s_cur = nxt; }
        __syncthreads();
    }
}

// Sequential target verify: batch_size tokens in one kernel.
// verify_tokens[0]          = last accepted token (context anchor)
// verify_tokens[1..k]       = draft tokens to verify
// out_tokens[i]             = model's argmax prediction at position start+i
// Caller updates target_kv.seq_len to start_seq_len + batch_size afterward.
__global__ void batch_target_verify_kernel(
        ModelWeights model, KVCache kv,
        const int* verify_tokens, int batch_size, int start_seq_len,
        float* g_logits,   // [vocab_size] global scratch, reused per step
        int*   out_tokens) // [batch_size] predicted next tokens per position
{
    extern __shared__ float shared[];
    int d      = model.cfg.d_model;
    float* hidden = shared;
    float* smem   = shared + d;

    for (int i = 0; i < batch_size; i++) {
        int nxt = model_forward(model, kv, verify_tokens[i],
                                start_seq_len + i, hidden, g_logits, smem);
        if (threadIdx.x == 0) out_tokens[i] = nxt;
        __syncthreads();
    }
}

// ---- Baseline (multi-kernel) ----

// Forward declaration — full definition appears later in this file.
static void launch_cooperative_decode_step(
    const ModelWeights& model, KVCache& kv,
    int token_id, int seq_len,
    float* g_coop_hidden,
    float* g_coop_scratch,
    float* g_logits,
    int*   d_next_token,
    int    max_coop_blocks,
    cudaStream_t stream = nullptr);

void multikernel_baseline(const ModelWeights& target_model,
                          KVCache& target_kv,
                          const int* h_prompt, int prompt_len,
                          GenerationResult* d_result,
                          const GenerationParams& params,
                          InferenceEngine* eng) {
    // h_prompt is a HOST pointer — values are read on CPU for each kernel launch.
    int max_new = params.max_new_tokens;

    bool use_coop = (eng != nullptr && eng->coop_supported);

    // Use intra-layer Q/K/V and gate/up stream overlap for prefill when the
    // engine is available and the model is large enough to benefit.
    const bool use_ovl_prefill = (eng != nullptr &&
                                  target_model.cfg.d_model >= OVERLAP_MIN_D);

    // Dynamic smem and logits buffer sized for the target model
    size_t smem_bytes = compute_smem_bytes(target_model.cfg);
    if (!use_coop && !use_ovl_prefill)
        cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, smem_bytes);

    float* g_logits;
    CUDA_CHECK(cudaMalloc(&g_logits,
                          (size_t)target_model.cfg.vocab_size * sizeof(float)));

    int* d_next_token;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, MAX_SEQ_LEN * sizeof(int)));

    kv_cache_reset(target_kv);
    int seq_len = 0;

    // Prefill: process each prompt token
    // Overlapped path: Q, K, V on streams[0,2,3]; gate, up share streams[0,3].
    // Cooperative path: multi-block output-projection parallelism (decode phase).
    // Fallback: single-block sequential kernel.
    for (int i = 0; i < prompt_len; i++) {
        if (use_ovl_prefill) {
            launch_prefill_overlapped(
                target_model, target_kv, h_prompt[i], seq_len,
                eng->d_ovl_buf[0],          // target uses slot 0
                g_logits, d_next_token, *eng,
                eng->streams[0],            // s_main
                eng->streams[2],            // s_k  (K projection)
                eng->streams[3]);           // s_v_gate (V + up)
            seq_len++;
            set_seq_len_kernel<<<1, 1, 0, eng->streams[0]>>>(
                target_kv.seq_len, seq_len);
        } else if (use_coop) {
            launch_cooperative_decode_step(
                target_model, target_kv, h_prompt[i], seq_len,
                eng->d_coop_hidden, eng->d_coop_scratch,
                g_logits, d_next_token, eng->max_coop_blocks);
            seq_len++;
            set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, seq_len);
        } else {
            single_token_decode_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
                target_model, target_kv, h_prompt[i], seq_len,
                g_logits, d_next_token);
            seq_len++;
            set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, seq_len);
        }
    }

    // Drain the overlap streams before any D2H transfer.
    if (use_ovl_prefill) {
        CUDA_CHECK(cudaStreamSynchronize(eng->streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(eng->streams[2]));
        CUDA_CHECK(cudaStreamSynchronize(eng->streams[3]));
    }

    int generated = 0;
    int current_token;
    CUDA_CHECK(cudaMemcpy(&current_token, d_next_token, sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Decode loop
    for (int step = 0; step < max_new; step++) {
        CUDA_CHECK(cudaMemcpy(d_output + generated, &current_token,
                              sizeof(int), cudaMemcpyHostToDevice));
        generated++;
        if ((params.eos_token >= 0 && current_token == params.eos_token) || generated >= max_new) break;

        if (use_coop) {
            launch_cooperative_decode_step(
                target_model, target_kv, current_token, seq_len,
                eng->d_coop_hidden, eng->d_coop_scratch,
                g_logits, d_next_token, eng->max_coop_blocks);
        } else {
            single_token_decode_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
                target_model, target_kv, current_token, seq_len,
                g_logits, d_next_token);
        }
        seq_len++;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, seq_len);
        CUDA_CHECK(cudaMemcpy(&current_token, d_next_token, sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    write_result_kernel<<<1, BLOCK_THREADS>>>(
        d_result, d_output, generated, 0, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(g_logits);
    cudaFree(d_next_token);
    cudaFree(d_output);
}

// ============================================================================
// Cooperative parallel verify kernel
//
// Launches batch_size = k+1 thread blocks simultaneously.  Each block b
// processes verify_tokens[b] at KV position start_seq_len + b.
//
// Per layer, execution proceeds in two grid-wide phases separated by
// cooperative_groups::this_grid().sync():
//   Phase 1 (all blocks in parallel): K/V projections + RoPE + KV cache write.
//             Each block writes to a distinct cache slot — no conflicts.
//   Phase 2 (all blocks in parallel): causal attention over [0..start+b] + FFN.
//             Block b reads K/V up to its own position (causal mask via total_len).
//
// This replaces the sequential for-loop in batch_target_verify_kernel with true
// GPU parallelism: k+1 tokens are processed in the time of ~1 sequential token
// (for large-enough models where compute >> launch overhead).
//
// Requires cudaDevAttrCooperativeLaunch support (SM 6.0+, checked at runtime).
// Must be launched with cudaLaunchCooperativeKernel.
//
// g_logits_batch: device buffer of shape [batch_size × vocab_size], one row per block.
// out_tokens:     device buffer of shape [batch_size], written with argmax per block.
// ============================================================================
__global__ void batch_target_verify_coop_kernel(
        ModelWeights model, KVCache kv,
        const int* verify_tokens,   // [batch_size] device ptr
        int batch_size,
        int start_seq_len,
        float* g_logits_batch,      // [batch_size × vocab_size] device ptr
        int* out_tokens)            // [batch_size] device ptr
{
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    int b = (int)blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    int d = model.cfg.d_model;
    float* hidden = shared;        // [d_model]  — residual stream for this block
    float* smem   = shared + d;    // scratch for layer ops

    int seq_pos = start_seq_len + b;

    // Embed this block's input token into the hidden state.
    model_embed(model, verify_tokens[b], hidden);

    for (int l = 0; l < model.cfg.n_layers; l++) {
        // ---- Phase 1: K/V write ----
        // All blocks project and cache their K,V simultaneously.
        // Each block writes to a unique cache position — no race conditions.
        model_layer_kv_phase(model, l, hidden, kv, seq_pos, smem);

        // Barrier: every block must have committed K,V before any block reads
        // another block's entries in the attention phase.
        grid.sync();

        // ---- Phase 2: Attention + FFN ----
        // Block b attends to positions [0 .. start_seq_len + b] (causal).
        // All k+1 blocks run this phase simultaneously.
        model_layer_attn_mlp_phase(model, l, hidden, kv, seq_pos, smem);

        // Barrier: all hidden states updated before next layer's K/V write.
        grid.sync();
    }

    // Final output head: RMSNorm → logits → argmax.
    float* my_logits = g_logits_batch + (size_t)b * model.cfg.vocab_size;
    model_output(model, hidden, my_logits, smem);
    int next = global_argmax(my_logits, model.cfg.vocab_size, smem);
    if (threadIdx.x == 0) out_tokens[b] = next;
}

// ============================================================================
// cooperative_decode_kernel
//
// Multi-block single-token forward pass using cooperative groups:
//
//   Block 0:    full single-token forward pass (embed → layers → final RMSNorm)
//               writes the normed hidden vector to g_scratch[0..d_model)
//   All blocks: split the output-projection GEMV across GEMV_COL_TILE-wide
//               column stripes, writing g_logits[0..vocab_size) in parallel
//   Block 0:    greedy argmax over g_logits → *d_next_token
//
// For large vocabulary models (32K+), the output projection GEMV is the
// dominant cost — splitting it across all SMs gives near-linear speedup.
//
// g_hidden  : [d_model]  residual stream (written by block 0 in Phase 1)
// g_scratch : [d_model]  normed final hidden (written by block 0, read by all)
// g_logits  : [vocab_size]  output logits (written by all blocks in Phase 2)
// ============================================================================
__global__ void cooperative_decode_kernel(
        ModelWeights model, KVCache kv,
        int token_id, int current_seq_len,
        float* g_hidden,    // [d_model]
        float* g_scratch,   // [d_model]  normed hidden (inter-phase buffer)
        float* g_logits,    // [vocab_size]
        int*   d_next_token)
{
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int d   = model.cfg.d_model;
    const int V   = model.cfg.vocab_size;

    // ---- Phase 1: Block 0 runs the full single-token forward pass ----
    if (blockIdx.x == 0) {
        // smem layout: [0..d) hidden temp | [d..end) layer scratch
        float* scratch = smem + d;   // used by model_layer_forward internals

        model_embed(model, token_id, g_hidden);
        for (int l = 0; l < model.cfg.n_layers; l++)
            model_layer_forward(model, l, g_hidden, kv, current_seq_len, scratch);

        // Final RMSNorm → write normed hidden to g_scratch (global, visible to all)
        device_rmsnorm(g_hidden, model.rms_final_weight, g_scratch, d, scratch);
        __syncthreads();   // make sure all threads of block 0 have written g_scratch
    }

    // Grid barrier: all blocks wait for g_scratch to be ready
    grid.sync();

    // ---- Phase 2: All blocks split the output projection GEMV ----
    // Each iteration covers GEMV_COL_TILE columns, strided by gridDim.x tiles.
    for (int ct = (int)blockIdx.x; ct * GEMV_COL_TILE < V; ct += (int)gridDim.x) {
        int cs = ct * GEMV_COL_TILE;
        int cc = (cs + GEMV_COL_TILE <= V) ? GEMV_COL_TILE : (V - cs);
        device_matvec_partial(g_scratch, model.output_proj, g_logits, d, V, cs, cc);
    }

    // Grid barrier: all logit columns written before argmax
    grid.sync();

    // ---- Phase 3: Block 0 computes the greedy argmax ----
    if (blockIdx.x == 0) {
        int next = global_argmax(g_logits, V, smem);
        if (tid == 0) *d_next_token = next;
    }
}

// Second declaration (no new default arguments; first decl is before
// multikernel_baseline, which already specifies stream = nullptr).
static void launch_cooperative_decode_step(
    const ModelWeights& model, KVCache& kv,
    int token_id, int seq_len,
    float* g_coop_hidden,
    float* g_coop_scratch,
    float* g_logits,
    int*   d_next_token,
    int    max_coop_blocks,
    cudaStream_t stream);

// ---- Speculative (multi-kernel) ----

void multikernel_speculative(const ModelWeights& draft_model,
                             const ModelWeights& target_model,
                             KVCache& draft_kv,
                             KVCache& target_kv,
                             const int* h_prompt, int prompt_len,
                             GenerationResult* d_result,
                             const GenerationParams& params,
                             InferenceEngine* eng) {
    // h_prompt is a HOST pointer.
    int max_new = params.max_new_tokens;
    int k       = params.spec_k;

    bool use_coop_draft  = (eng != nullptr && eng->coop_supported);
    bool use_coop_target = (eng != nullptr && eng->coop_supported);
    bool use_coop_verify = (eng != nullptr && eng->coop_supported);

    // Each model gets its own smem budget and logits buffer
    size_t draft_smem  = compute_smem_bytes(draft_model.cfg);
    size_t target_smem = compute_smem_bytes(target_model.cfg);
    size_t max_decode_smem = draft_smem > target_smem ? draft_smem : target_smem;

    if (!use_coop_draft && !use_coop_target)
        cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, max_decode_smem);
    else if (!use_coop_draft)
        cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, draft_smem);
    else if (!use_coop_target)
        cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, target_smem);

    float* g_logits_draft;
    float* g_logits_target;
    CUDA_CHECK(cudaMalloc(&g_logits_draft,
                          (size_t)draft_model.cfg.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_logits_target,
                          (size_t)target_model.cfg.vocab_size * sizeof(float)));

    int* d_next;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_next, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, MAX_SEQ_LEN * sizeof(int)));

    kv_cache_reset(draft_kv);
    kv_cache_reset(target_kv);

    int draft_seq  = 0;
    int target_seq = 0;

    // ---- Prefill: draft (stream[0]) and target (stream[1]) run in parallel ----
    // Each prompt token still has a sequential dependency within each model
    // (token i+1 needs KV from token i), but draft[i] and target[i] are
    // independent of each other and can run concurrently.
    //
    // When the model is large enough we further overlap Q‖K‖V and gate‖up
    // projections inside each token step using intra-layer stream overlap.
    // Draft  uses streams[0](main) / streams[2](K) / streams[3](V+up).
    // Target uses streams[1](main) / streams[2](K) / streams[3](V+up).
    // Since draft and target run sequentially within each prompt position
    // (gated by the cross-stream sync events), sharing streams[2/3] is safe.
    cudaStream_t stream_d = (eng != nullptr) ? eng->streams[0] : nullptr;
    cudaStream_t stream_t = (eng != nullptr) ? eng->streams[1] : nullptr;

    const bool use_ovl_draft  = (eng != nullptr &&
                                  draft_model.cfg.d_model  >= OVERLAP_MIN_D);
    const bool use_ovl_target = (eng != nullptr &&
                                  target_model.cfg.d_model >= OVERLAP_MIN_D);

    for (int i = 0; i < prompt_len; i++) {
        // Draft prefill on stream_d; uses d_ovl_buf[1] / d_coop_hidden2
        // (separate buffer from target to avoid aliasing).
        if (use_ovl_draft) {
            launch_prefill_overlapped(
                draft_model, draft_kv, h_prompt[i], draft_seq,
                eng->d_ovl_buf[1],          // draft uses slot 1
                g_logits_draft, d_next, *eng,
                eng->streams[0],            // s_main
                eng->streams[2],            // s_k
                eng->streams[3]);           // s_v_gate
            draft_seq++;
            set_seq_len_kernel<<<1, 1, 0, stream_d>>>(draft_kv.seq_len, draft_seq);
        } else if (use_coop_draft) {
            launch_cooperative_decode_step(
                draft_model, draft_kv, h_prompt[i], draft_seq,
                eng->d_coop_hidden2, eng->d_coop_scratch2,
                g_logits_draft, d_next, eng->max_coop_blocks, stream_d);
            draft_seq++;
            set_seq_len_kernel<<<1, 1, 0, stream_d>>>(draft_kv.seq_len, draft_seq);
        } else {
            single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem, stream_d>>>(
                draft_model, draft_kv, h_prompt[i], draft_seq,
                g_logits_draft, d_next);
            draft_seq++;
            set_seq_len_kernel<<<1, 1, 0, stream_d>>>(draft_kv.seq_len, draft_seq);
        }

        // Target prefill on stream_t (concurrent with draft's main stream).
        // Intra-layer overlap for target also uses streams[2,3]; since draft's
        // last kernel on those streams records ev1/ev2 and target's first
        // use waits for ev0 (which fires AFTER the cross-stream sync below),
        // there is no hazard as long as each step's sync fires in order.
        if (use_ovl_target) {
            launch_prefill_overlapped(
                target_model, target_kv, h_prompt[i], target_seq,
                eng->d_ovl_buf[0],          // target uses slot 0
                g_logits_target, d_next, *eng,
                eng->streams[1],            // s_main
                eng->streams[2],            // s_k
                eng->streams[3]);           // s_v_gate
            target_seq++;
            set_seq_len_kernel<<<1, 1, 0, stream_t>>>(target_kv.seq_len, target_seq);
        } else if (use_coop_target) {
            launch_cooperative_decode_step(
                target_model, target_kv, h_prompt[i], target_seq,
                eng->d_coop_hidden, eng->d_coop_scratch,
                g_logits_target, d_next, eng->max_coop_blocks, stream_t);
            target_seq++;
            set_seq_len_kernel<<<1, 1, 0, stream_t>>>(target_kv.seq_len, target_seq);
        } else {
            single_token_decode_kernel<<<1, BLOCK_THREADS, target_smem, stream_t>>>(
                target_model, target_kv, h_prompt[i], target_seq,
                g_logits_target, d_next);
            target_seq++;
            set_seq_len_kernel<<<1, 1, 0, stream_t>>>(target_kv.seq_len, target_seq);
        }

        // Cross-stream sync: token i+1 can't start until both models finished token i.
        // For the overlapped path the relevant "done" signal is on the s_main streams
        // (streams[0] for draft, streams[1] for target); these carry all residual-add
        // and seq_len writes, so recording ev[0/1] on them is sufficient.
        if (eng != nullptr) {
            CUDA_CHECK(cudaEventRecord(eng->sync_events[0], stream_d));
            CUDA_CHECK(cudaEventRecord(eng->sync_events[1], stream_t));
            CUDA_CHECK(cudaStreamWaitEvent(stream_d, eng->sync_events[1], 0));
            CUDA_CHECK(cudaStreamWaitEvent(stream_t, eng->sync_events[0], 0));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    // Drain all streams before entering the decode loop.
    if (eng != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream_d));
        CUDA_CHECK(cudaStreamSynchronize(stream_t));
        if (use_ovl_draft || use_ovl_target) {
            CUDA_CHECK(cudaStreamSynchronize(eng->streams[2]));
            CUDA_CHECK(cudaStreamSynchronize(eng->streams[3]));
        }
    }

    int last_token;
    CUDA_CHECK(cudaMemcpy(&last_token, d_next, sizeof(int),
                          cudaMemcpyDeviceToHost));

    int generated       = 0;
    int total_proposed  = 0;
    int total_accepted  = 0;
    int iterations      = 0;

    // Write the first token (prefill output) to match baseline
    CUDA_CHECK(cudaMemcpy(d_output, &last_token, sizeof(int),
                          cudaMemcpyHostToDevice));
    generated = 1;

    int* h_draft_tokens = new int[k];

    // ---------------------------------------------------------------------
    // Stochastic speculative decoding (multi-kernel only; distribution-level
    // acceptance + adjusted rejection sampling vs greedy token identity).
    // ---------------------------------------------------------------------
    if (params.stochastic_spec_decode) {
        if (draft_model.cfg.vocab_size != target_model.cfg.vocab_size) {
            fprintf(stderr,
                    "multikernel_speculative: stochastic mode requires identical "
                    "draft/target vocab_size (draft=%d target=%d)\n",
                    draft_model.cfg.vocab_size,
                    target_model.cfg.vocab_size);
            exit(EXIT_FAILURE);
        }

        int V_vocab = target_model.cfg.vocab_size;

        cuda_configure_kernel_dynamic_smem(stochastic_draft_forward_sample_kernel,
                                             draft_smem);
        cuda_configure_kernel_dynamic_smem(stochastic_draft_forward_sample_dptr_kernel,
                                             draft_smem);
        cuda_configure_kernel_dynamic_smem(single_token_forward_logits_kernel,
                                           max_decode_smem);
        cuda_configure_kernel_dynamic_smem(target_forward_prob_mass_kernel,
                                           target_smem);
        cuda_configure_kernel_dynamic_smem(target_fwd_prob_and_accept_kernel,
                                           target_smem);
        cuda_configure_kernel_dynamic_smem(softmax_sample_temperature_kernel,
                                           0);
        cuda_configure_kernel_dynamic_smem(corrected_sample_adjusted_logits_kernel,
                                             0);
        cuda_configure_kernel_dynamic_smem(stochastic_accept_gate_kernel, 0);
        cuda_configure_kernel_dynamic_smem(rng_init_kernel, 0);

        curandState* d_rng = nullptr;
        float*       d_p_mass_out = nullptr;
        float*       d_q_mass_out = nullptr;
        float*       d_corr_work = nullptr;
        int*         d_accept = nullptr;

        CUDA_CHECK(cudaMalloc(&d_rng, sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_p_mass_out, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q_mass_out, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_accept, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_corr_work,
                              (size_t)V_vocab * 3 * sizeof(float)));

        rng_init_kernel<<<1, 1>>>(d_rng,
                                 (unsigned long long)params.stochastic_rng_seed);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_q_probs = new float[(size_t)k];
        float  draft_temp_dyn = params.draft_temperature;
        float  ewma_accept    = -1.f;

        // Device-side buffers for chained draft sampling — avoids k CPU roundtrips
        // per speculation round by keeping tokens on device until all k are ready.
        int*   d_draft_tokens_dev = nullptr;
        float* d_q_probs_dev      = nullptr;
        int*   d_ctx_seed         = nullptr;   // bootstrap: device copy of last_token
        CUDA_CHECK(cudaMalloc(&d_draft_tokens_dev, (size_t)k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_q_probs_dev,      (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ctx_seed,          sizeof(int)));

        while (generated < max_new) {
            iterations++;
            int remaining = max_new - generated;
            int current_k = (k < remaining) ? k : remaining;

            int draft_seq_save  = draft_seq;
            int target_seq_save = target_seq;
            total_proposed += current_k;

            // ---- Draft phase: fully chained on GPU, no per-step sync ----
            // Seed: copy last_token to device so step 0 can read it as a pointer.
            CUDA_CHECK(cudaMemcpy(d_ctx_seed, &last_token, sizeof(int),
                                  cudaMemcpyHostToDevice));
            const int* d_ctx_ptr = d_ctx_seed;

            for (int di = 0; di < current_k; di++) {
                stochastic_draft_forward_sample_dptr_kernel
                    <<<1, BLOCK_THREADS, draft_smem>>>(
                        draft_model, draft_kv, d_ctx_ptr, draft_seq,
                        draft_temp_dyn, d_rng, g_logits_draft,
                        d_draft_tokens_dev + di, d_q_probs_dev + di);
                // Chain: next step reads the token this step just wrote on device.
                d_ctx_ptr = d_draft_tokens_dev + di;
                draft_seq++;
                set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            }
            // One sync + one batch copy for all k draft tokens and probabilities.
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_draft_tokens, d_draft_tokens_dev,
                                  (size_t)current_k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_q_probs, d_q_probs_dev,
                                  (size_t)current_k * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            int n_accept_round = 0;
            int bonus          = -1; // sentinel: "no bonus token yet"
            bool broke_early    = false;
            int  target_roll    = target_seq_save;

            for (int vi = 0; vi < current_k; vi++) {
                int inp = (vi == 0) ? last_token : h_draft_tokens[vi - 1];

                // Fused target forward + acceptance gate: one launch, one sync,
                // one memcpy per verification step (was two launches + two syncs).
                target_fwd_prob_and_accept_kernel
                    <<<1, BLOCK_THREADS, target_smem>>>(
                        target_model, target_kv, inp, target_roll,
                        h_draft_tokens[vi], h_q_probs[vi],
                        g_logits_target, d_rng, d_accept);
                CUDA_CHECK(cudaDeviceSynchronize());

                int acc_flag = 0;
                CUDA_CHECK(cudaMemcpy(&acc_flag, d_accept, sizeof(int),
                                      cudaMemcpyDeviceToHost));

                target_roll++;
                set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_roll);

                if (acc_flag) {
                    n_accept_round++;
                    continue;
                }

                broke_early = true;
                set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len,
                                             draft_seq_save + vi);
                set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len,
                                             target_seq_save + vi);

                single_token_forward_logits_kernel
                    <<<1, BLOCK_THREADS, draft_smem>>>(
                        draft_model, draft_kv, inp, draft_seq_save + vi,
                        g_logits_draft);
                CUDA_CHECK(cudaDeviceSynchronize());

                corrected_sample_adjusted_logits_kernel<<<1, 1>>>(
                    g_logits_target, g_logits_draft, V_vocab, d_rng, d_next,
                    d_corr_work, 1.f, draft_temp_dyn);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&bonus, d_next, sizeof(int),
                                      cudaMemcpyDeviceToHost));

                single_token_forward_logits_kernel
                    <<<1, BLOCK_THREADS, target_smem>>>(
                        target_model, target_kv, bonus,
                        target_seq_save + n_accept_round, g_logits_target);
                CUDA_CHECK(cudaDeviceSynchronize());

                break;
            }

            if (!broke_early) {
                single_token_forward_logits_kernel
                    <<<1, BLOCK_THREADS, target_smem>>>(
                        target_model, target_kv, h_draft_tokens[current_k - 1],
                        target_roll, g_logits_target);
                CUDA_CHECK(cudaDeviceSynchronize());

                softmax_sample_temperature_kernel<<<1, 1>>>(
                    g_logits_target, V_vocab, 1.f, d_rng, d_next);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&bonus, d_next, sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }

            total_accepted += n_accept_round;

            for (int wi = 0; wi < n_accept_round && generated < max_new; wi++) {
                CUDA_CHECK(cudaMemcpy(d_output + generated, &h_draft_tokens[wi],
                                      sizeof(int), cudaMemcpyHostToDevice));
                generated++;
            }
            if (generated < max_new) {
                CUDA_CHECK(cudaMemcpy(d_output + generated, &bonus,
                                      sizeof(int), cudaMemcpyHostToDevice));
                generated++;
            }
            last_token = bonus;

            target_seq = target_seq_save + n_accept_round + 1;
            set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);

            draft_seq = draft_seq_save + n_accept_round;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);

            if (n_accept_round == current_k) {
                cuda_configure_kernel_dynamic_smem(single_token_decode_kernel,
                                                   draft_smem);
                single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                    draft_model, draft_kv, h_draft_tokens[current_k - 1],
                    draft_seq, g_logits_draft, d_next);
                draft_seq++;
                set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            }

            cuda_configure_kernel_dynamic_smem(single_token_decode_kernel,
                                               draft_smem);
            single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                draft_model, draft_kv, last_token, draft_seq, g_logits_draft,
                d_next);
            draft_seq++;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (params.adaptive_draft_temperature && current_k > 0) {
                float mix = params.stochastic_adapt_ewma_mix;
                if (mix < 1e-6f)
                    mix = 1e-6f;
                else if (mix > 1.f - 1e-6f)
                    mix = 1.f - 1e-6f;
                float round_rate =
                    (float)n_accept_round / (float)current_k;
                if (ewma_accept < 0.f)
                    ewma_accept = round_rate;
                else
                    ewma_accept = (1.f - mix) * ewma_accept + mix * round_rate;
                draft_temp_dyn +=
                    params.stochastic_adapt_temp_gain *
                    (ewma_accept - params.stochastic_adapt_target_accept);
                draft_temp_dyn =
                    fmaxf(params.min_draft_temperature,
                          fminf(params.max_draft_temperature, draft_temp_dyn));
            }

            if ((params.eos_token >= 0 && bonus == params.eos_token) || generated >= max_new)
                break;
        }

        delete[] h_q_probs;
        cudaFree(d_ctx_seed);
        cudaFree(d_q_probs_dev);
        cudaFree(d_draft_tokens_dev);
        cudaFree(d_corr_work);
        cudaFree(d_accept);
        cudaFree(d_q_mass_out);
        cudaFree(d_p_mass_out);
        cudaFree(d_rng);

        write_result_kernel<<<1, BLOCK_THREADS>>>(
            d_result, d_output, generated, total_proposed, total_accepted,
            iterations);
        CUDA_CHECK(cudaDeviceSynchronize());

        delete[] h_draft_tokens;
        cudaFree(g_logits_draft);
        cudaFree(g_logits_target);
        cudaFree(d_next);
        cudaFree(d_output);
        return;
    }

    int* h_target_tokens = new int[k + 1];
    int* h_verify_tokens = new int[k + 1]; // host build: [last_token, draft_0..draft_{k-1}]

    // Device buffers for batched draft/verify — replaces O(k) per-round CPU-GPU syncs
    int* d_batch_draft;
    int* d_batch_verify_in;
    int* d_batch_verify_out;
    CUDA_CHECK(cudaMalloc(&d_batch_draft,      (size_t)(k + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_verify_in,  (size_t)(k + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_verify_out, (size_t)(k + 1) * sizeof(int)));

    // Configure batch kernels once before the loop
    cuda_configure_kernel_dynamic_smem(batch_draft_greedy_kernel, draft_smem);
    if (!use_coop_verify)
        cuda_configure_kernel_dynamic_smem(batch_target_verify_kernel, target_smem);
    if (use_coop_verify)
        cuda_configure_kernel_dynamic_smem(batch_target_verify_coop_kernel, target_smem);

    // Pre-allocate a logits buffer for cooperative verify ([MAX_VERIFY_BATCH × vocab_size])
    float* g_logits_verify_batch = nullptr;
    if (use_coop_verify)
        CUDA_CHECK(cudaMalloc(&g_logits_verify_batch,
                              (size_t)MAX_VERIFY_BATCH * target_model.cfg.vocab_size
                              * sizeof(float)));

    while (generated < max_new) {
        iterations++;
        int remaining = max_new - generated;
        int current_k = (k < remaining) ? k : remaining;

        int draft_seq_save  = draft_seq;
        int target_seq_save = target_seq;
        total_proposed += current_k;

        // ---- Draft phase: one batched kernel (k separate launches → 1) ----
        batch_draft_greedy_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
            draft_model, draft_kv, last_token, current_k, draft_seq,
            g_logits_draft, d_batch_draft);
        draft_seq += current_k;
        set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
        // D2H memcpy provides implicit sync; no separate DeviceSync needed
        CUDA_CHECK(cudaMemcpy(h_draft_tokens, d_batch_draft,
                              current_k * sizeof(int), cudaMemcpyDeviceToHost));

        // ---- Verify phase ----
        h_verify_tokens[0] = last_token;
        for (int i = 0; i < current_k; i++) h_verify_tokens[i + 1] = h_draft_tokens[i];
        int batch_size = current_k + 1;
        CUDA_CHECK(cudaMemcpy(d_batch_verify_in, h_verify_tokens,
                              batch_size * sizeof(int), cudaMemcpyHostToDevice));

        if (use_coop_verify) {
            // Cooperative: k+1 blocks run in parallel, one block per token
            void* coop_args[] = {
                (void*)&target_model,
                (void*)&target_kv,
                (void*)&d_batch_verify_in,
                (void*)&batch_size,
                (void*)&target_seq,
                (void*)&g_logits_verify_batch,
                (void*)&d_batch_verify_out
            };
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)batch_target_verify_coop_kernel,
                dim3(batch_size), dim3(BLOCK_THREADS),
                coop_args, target_smem, nullptr));
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            // Fallback: sequential single-block pass through all k+1 tokens
            batch_target_verify_kernel<<<1, BLOCK_THREADS, target_smem>>>(
                target_model, target_kv, d_batch_verify_in, batch_size, target_seq,
                g_logits_target, d_batch_verify_out);
        }
        target_seq += batch_size;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);
        CUDA_CHECK(cudaMemcpy(h_target_tokens, d_batch_verify_out,
                              batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        // ---- Accept/reject: greedy match ----
        int n_accepted = 0;
        for (int i = 0; i < current_k; i++) {
            if (h_target_tokens[i] == h_draft_tokens[i]) n_accepted++;
            else break;
        }
        total_accepted += n_accepted;

        // Per-round acceptance rate for diagnosis (stderr avoids web-parser conflicts)
        fprintf(stderr, "[spec round %d] k=%d accepted=%d/%d  alpha=%.3f\n",
                iterations, current_k, n_accepted, current_k,
                current_k > 0 ? (float)n_accepted / current_k : 0.f);

        for (int i = 0; i < n_accepted && generated < max_new; i++) {
            CUDA_CHECK(cudaMemcpy(d_output + generated, &h_draft_tokens[i],
                                  sizeof(int), cudaMemcpyHostToDevice));
            generated++;
        }

        int bonus = h_target_tokens[n_accepted];
        if (generated < max_new) {
            CUDA_CHECK(cudaMemcpy(d_output + generated, &bonus,
                                  sizeof(int), cudaMemcpyHostToDevice));
            generated++;
        }
        last_token = bonus;

        // Rollback caches to the accepted prefix
        target_seq = target_seq_save + n_accepted + 1;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);

        draft_seq = draft_seq_save + n_accepted;
        set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);

        // When ALL k drafts were accepted, draft[k-1] was only a prediction —
        // its K/V was never written.  One extra forward fills that slot.
        // After this draft_seq == target_seq; the bonus-sync below must NOT
        // run, or draft_seq would overshoot target_seq by 1 and corrupt RoPE
        // positions in every subsequent round (accumulating alignment errors).
        if (n_accepted == current_k) {
            if (use_coop_draft) {
                launch_cooperative_decode_step(
                    draft_model, draft_kv, h_draft_tokens[current_k - 1],
                    draft_seq, eng->d_coop_hidden2, eng->d_coop_scratch2,
                    g_logits_draft, d_next, eng->max_coop_blocks);
            } else {
                single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                    draft_model, draft_kv, h_draft_tokens[current_k - 1],
                    draft_seq, g_logits_draft, d_next);
            }
            draft_seq++;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            // draft_seq == target_seq — do NOT run the bonus sync.
        } else {
            // Partial accept: write bonus token's K/V at the rollback position
            // so that both caches are aligned before the next spec round.
            if (use_coop_draft) {
                launch_cooperative_decode_step(
                    draft_model, draft_kv, last_token, draft_seq,
                    eng->d_coop_hidden2, eng->d_coop_scratch2,
                    g_logits_draft, d_next, eng->max_coop_blocks);
            } else {
                single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                    draft_model, draft_kv, last_token, draft_seq,
                    g_logits_draft, d_next);
            }
            draft_seq++;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        if ((params.eos_token >= 0 && bonus == params.eos_token) || generated >= max_new) break;
    }

    write_result_kernel<<<1, BLOCK_THREADS>>>(
        d_result, d_output, generated,
        total_proposed, total_accepted, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (g_logits_verify_batch) cudaFree(g_logits_verify_batch);
    delete[] h_draft_tokens;
    delete[] h_target_tokens;
    delete[] h_verify_tokens;
    cudaFree(d_batch_draft);
    cudaFree(d_batch_verify_in);
    cudaFree(d_batch_verify_out);
    cudaFree(g_logits_draft);
    cudaFree(g_logits_target);
    cudaFree(d_next);
    cudaFree(d_output);
}

// ============================================================================
// launch_cooperative_decode_step  (definition; forward-declared above)
//
// Launches cooperative_decode_kernel with a grid sized to cover all vocab
// columns in parallel, capped to the device's cooperative-launch limit.
// Falls back gracefully to 1 block if max_coop_blocks == 0.
// ============================================================================
static void launch_cooperative_decode_step(
        const ModelWeights& model, KVCache& kv,
        int token_id, int seq_len,
        float* g_coop_hidden,
        float* g_coop_scratch,
        float* g_logits,
        int*   d_next_token,
        int    max_coop_blocks,
        cudaStream_t stream) {
    const int V = model.cfg.vocab_size;
    size_t smem = compute_smem_bytes(model.cfg);
    cuda_configure_kernel_dynamic_smem(cooperative_decode_kernel, smem);

    int n_blocks_needed = (V + GEMV_COL_TILE - 1) / GEMV_COL_TILE;
    int n_blocks = (n_blocks_needed < max_coop_blocks) ? n_blocks_needed
                                                        : max_coop_blocks;
    if (n_blocks < 1) n_blocks = 1;

    void* args[] = {
        (void*)&model,
        (void*)&kv,
        (void*)&token_id,
        (void*)&seq_len,
        (void*)&g_coop_hidden,
        (void*)&g_coop_scratch,
        (void*)&g_logits,
        (void*)&d_next_token
    };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)cooperative_decode_kernel,
        dim3(n_blocks), dim3(BLOCK_THREADS),
        args, smem, stream));
}

// ============================================================================
//  PERSISTENT MEGAKERNEL PATH
//
//  A single kernel runs the entire generation loop on the GPU.
//  g_logits is a global-memory scratch buffer for intermediate logits.
//
//  TODO: For multi-block / larger models, extend with cooperative-group
//        grid-level barriers instead of __syncthreads().
// ============================================================================

// ---- Megakernel: Baseline ----

// ModelWeights/KVCache are large (e.g. layers[MAX_LAYERS]); passing them by value
// overflows the 4 KiB CUDA kernel parameter limit. Pass device pointers instead.
__global__ void megakernel_baseline_kernel(
        const ModelWeights* p_target_model, KVCache* p_target_kv,
        const int* prompt, int prompt_len, int max_new_tokens,
        float* g_logits, GenerationResult* result, int eos_token) {
    const ModelWeights& target_model = *p_target_model;
    KVCache&            target_kv    = *p_target_kv;
    extern __shared__ float shared[];
    int d = target_model.cfg.d_model;
    float* hidden = shared;       // [d_model]
    float* smem   = shared + d;   // scratch

    int tid = threadIdx.x;

    __shared__ int s_current_token;
    __shared__ int s_seq_len;
    __shared__ int s_generated;

    if (tid == 0) { s_seq_len = 0; s_generated = 0; }
    __syncthreads();

    // Prefill
    for (int i = 0; i < prompt_len; i++) {
        int next = model_forward(target_model, target_kv, prompt[i],
                                 s_seq_len, hidden, g_logits, smem);
        if (tid == 0) {
            s_current_token = next;
            s_seq_len++;
            *target_kv.seq_len = s_seq_len;
        }
        __syncthreads();
    }

    // Decode loop
    while (s_generated < max_new_tokens) {
        if (tid == 0) {
            result->output_tokens[s_generated] = s_current_token;
            s_generated++;
        }
        __syncthreads();

        if ((eos_token >= 0 && s_current_token == eos_token) || s_generated >= max_new_tokens)
            break;

        int next = model_forward(target_model, target_kv, s_current_token,
                                 s_seq_len, hidden, g_logits, smem);
        if (tid == 0) {
            s_current_token = next;
            s_seq_len++;
            *target_kv.seq_len = s_seq_len;
        }
        __syncthreads();
    }

    if (tid == 0) {
        result->n_generated     = s_generated;
        result->draft_proposed  = 0;
        result->draft_accepted  = 0;
        result->spec_iterations = 0;
    }
}

// ---- Megakernel: Speculative ----
// s_draft/target_tokens are sized for spec_k up to MAX_MEGA_K.
constexpr int MAX_MEGA_K = 16;

__global__ void megakernel_speculative_kernel(
        const ModelWeights* p_draft_model, const ModelWeights* p_target_model,
        KVCache* p_draft_kv, KVCache* p_target_kv,
        const int* prompt, int prompt_len,
        int max_new_tokens, int spec_k,
        float* g_logits,
        float* g_batch_hidden,
        float* g_batch_work,
        float* g_batch_logits,
        GenerationResult* result, int eos_token) {
    const ModelWeights& draft_model  = *p_draft_model;
    const ModelWeights& target_model = *p_target_model;
    KVCache&            draft_kv     = *p_draft_kv;
    KVCache&            target_kv    = *p_target_kv;
    extern __shared__ float shared[];
    int d = target_model.cfg.d_model;
    float* hidden = shared;
    float* smem   = shared + d;

    int tid = threadIdx.x;

    __shared__ int s_last_token;
    __shared__ int s_draft_seq;
    __shared__ int s_target_seq;
    __shared__ int s_generated;
    __shared__ int s_total_proposed;
    __shared__ int s_total_accepted;
    __shared__ int s_iterations;
    __shared__ int s_draft_tokens [MAX_MEGA_K];
    __shared__ int s_target_tokens[MAX_MEGA_K + 1];
    __shared__ int s_draft_seq_save;
    __shared__ int s_target_seq_save;
    __shared__ int s_n_accepted;

    if (tid == 0) {
        s_draft_seq = 0; s_target_seq = 0; s_generated = 0;
        s_total_proposed = 0; s_total_accepted = 0; s_iterations = 0;
    }
    __syncthreads();

    // ---- Prefill both models ----
    for (int i = 0; i < prompt_len; i++) {
        model_forward(draft_model, draft_kv, prompt[i],
                      s_draft_seq, hidden, g_logits, smem);
        if (tid == 0) { s_draft_seq++; *draft_kv.seq_len = s_draft_seq; }
        __syncthreads();

        int next_t = model_forward(target_model, target_kv, prompt[i],
                                   s_target_seq, hidden, g_logits, smem);
        if (tid == 0) {
            s_last_token = next_t;
            s_target_seq++;
            *target_kv.seq_len = s_target_seq;
        }
        __syncthreads();
    }

    // Write prefill output (matches baseline token[0])
    if (tid == 0) {
        result->output_tokens[0] = s_last_token;
        s_generated = 1;
    }
    __syncthreads();

    // ---- Speculative decode loop ----
    while (s_generated < max_new_tokens) {
        if (tid == 0) {
            s_iterations++;
            int current_k = spec_k;
            if (s_generated + current_k + 1 > max_new_tokens)
                current_k = max_new_tokens - s_generated - 1;
            if (current_k < 1) current_k = 1;
            s_draft_seq_save  = s_draft_seq;
            s_target_seq_save = s_target_seq;
            s_total_proposed += current_k;
        }
        __syncthreads();

        // Use a shared variable for current_k so all threads agree
        __shared__ int s_current_k;
        if (tid == 0) {
            int ck = spec_k;
            if (s_generated + ck + 1 > max_new_tokens)
                ck = max_new_tokens - s_generated - 1;
            if (ck < 1) ck = 1;
            s_current_k = ck;
        }
        __syncthreads();

        int current_k = s_current_k;

        // ---- Draft phase ----
        __shared__ int s_draft_input;
        if (tid == 0) s_draft_input = s_last_token;
        __syncthreads();

        for (int di = 0; di < current_k; di++) {
            int next = model_forward(draft_model, draft_kv, s_draft_input,
                                     s_draft_seq, hidden, g_logits, smem);
            if (tid == 0) {
                s_draft_tokens[di] = next;
                s_draft_input = next;
                s_draft_seq++;
                *draft_kv.seq_len = s_draft_seq;
            }
            __syncthreads();
        }

        // ---- Batched verify phase (reads target weights ONCE for all k+1 tokens) ----
        {
            __shared__ int s_verify_toks[MAX_MEGA_K + 1];
            if (tid == 0) {
                s_verify_toks[0] = s_last_token;
                for (int i = 0; i < current_k; i++)
                    s_verify_toks[i + 1] = s_draft_tokens[i];
            }
            __syncthreads();

            model_batch_forward_logits(
                target_model, target_kv,
                s_verify_toks, s_target_seq_save,
                current_k + 1,
                g_batch_hidden, g_batch_work, g_batch_logits,
                smem);

            int V = target_model.cfg.vocab_size;
            for (int vi = 0; vi <= current_k; vi++) {
                int next = global_argmax(
                    g_batch_logits + vi * V, V, smem);
                if (tid == 0)
                    s_target_tokens[vi] = next;
                __syncthreads();
            }

            if (tid == 0) {
                s_target_seq = s_target_seq_save + current_k + 1;
                *target_kv.seq_len = s_target_seq;
            }
            __syncthreads();
        }

        // ---- Accept/reject (greedy) and cache rollback ----
        if (tid == 0) {
            int n_acc = 0;
            for (int i = 0; i < current_k; i++) {
                if (s_target_tokens[i] == s_draft_tokens[i]) n_acc++;
                else break;
            }
            s_n_accepted = n_acc;
            s_total_accepted += n_acc;

            for (int i = 0; i < n_acc && s_generated < max_new_tokens; i++) {
                result->output_tokens[s_generated++] = s_draft_tokens[i];
            }
            int bonus = s_target_tokens[n_acc];
            if (s_generated < max_new_tokens)
                result->output_tokens[s_generated++] = bonus;
            s_last_token = bonus;

            s_target_seq = s_target_seq_save + n_acc + 1;
            *target_kv.seq_len = s_target_seq;
            s_draft_seq = s_draft_seq_save + n_acc;
            *draft_kv.seq_len = s_draft_seq;
        }
        __syncthreads();

        // All-accepted: draft[k-1] K/V was never written; fill it now.
        // After the extra forward draft_seq == target_seq — do NOT also run
        // the bonus sync or draft_seq would overshoot by 1 (RoPE misalignment).
        if (s_n_accepted == current_k) {
            model_forward(draft_model, draft_kv, s_draft_tokens[current_k - 1],
                          s_draft_seq, hidden, g_logits, smem);
            if (tid == 0) { s_draft_seq++; *draft_kv.seq_len = s_draft_seq; }
            __syncthreads();
        } else {
            // Partial accept: sync draft cache to the bonus token position.
            model_forward(draft_model, draft_kv, s_last_token,
                          s_draft_seq, hidden, g_logits, smem);
            if (tid == 0) { s_draft_seq++; *draft_kv.seq_len = s_draft_seq; }
            __syncthreads();
        }

        if ((eos_token >= 0 && s_last_token == eos_token) || s_generated >= max_new_tokens)
            break;
    }

    if (tid == 0) {
        result->n_generated     = s_generated;
        result->draft_proposed  = s_total_proposed;
        result->draft_accepted  = s_total_accepted;
        result->spec_iterations = s_iterations;
    }
}

// Persistent megakernel: stochastic speculative (same acceptance/sampling maths as multi-kernel).
__global__ void megakernel_speculative_stochastic_kernel(
        const ModelWeights* p_draft_model,
        const ModelWeights* p_target_model,
        KVCache* p_draft_kv,
        KVCache* p_target_kv,
        const int* prompt,
        int          prompt_len,
        int          max_new_tokens,
        int          spec_k_param,
        float* logits_draft,
        float* logits_target,
        float* corr_workspace,
        float* g_batch_hidden,
        float* g_batch_work,
        float* g_batch_logits,
        float draft_temp_initial,
        int adaptive_enabled,
        float min_draft_temp,
        float max_draft_temp,
        float adapt_tgt_accept,
        float adapt_gain,
        float adapt_ewma_mix,
        unsigned long long rng_seed,
        GenerationResult* result,
        int eos_token) {

    const ModelWeights& draft_model  = *p_draft_model;
    const ModelWeights& target_model = *p_target_model;
    KVCache&            draft_kv    = *p_draft_kv;
    KVCache&            target_kv   = *p_target_kv;

    const int V_vocab = target_model.cfg.vocab_size;

    extern __shared__ float shared[];
    int                     dim_t = target_model.cfg.d_model;
    float*                  hidden = shared;
    float* scratch            = shared + dim_t;

    const int tid = threadIdx.x;

    __shared__ curandState s_rng;
    __shared__ float s_dyn_dt;
    __shared__ float s_ewma;

    __shared__ int s_last_token;
    __shared__ int s_draft_seq;
    __shared__ int s_target_seq;
    __shared__ int s_generated;
    __shared__ int s_total_proposed;
    __shared__ int s_total_accepted;
    __shared__ int s_iterations;

    __shared__ int s_draft_ids[MAX_MEGA_K];
    __shared__ float s_q_probs[MAX_MEGA_K];
    __shared__ int s_ds_save;
    __shared__ int s_ts_save;
    __shared__ int s_n_accepted;
    __shared__ int s_ck;
    __shared__ int s_tr_roll;
    __shared__ int s_bonus_token;
    __shared__ int acc_cnt;
    __shared__ int s_ok_gate;
    __shared__ int s_cur_in;

    if (tid == 0) {
        s_draft_seq = s_target_seq = s_generated = s_total_proposed =
            s_total_accepted = s_iterations = 0;
        s_dyn_dt          = draft_temp_initial;
        s_ewma            = -1.f;
        curand_init(rng_seed, 0ULL, 0ULL, &s_rng);
    }
    __syncthreads();

    for (int pi = 0; pi < prompt_len; pi++) {
        model_forward(draft_model, draft_kv, prompt[pi],
                      s_draft_seq, hidden, logits_draft, scratch);
        if (tid == 0) {
            s_draft_seq++;
            *draft_kv.seq_len = s_draft_seq;
        }
        __syncthreads();

        int tgt_id = model_forward(target_model, target_kv, prompt[pi],
                                   s_target_seq, hidden,
                                   logits_target, scratch);
        if (tid == 0) {
            s_last_token = tgt_id;
            s_target_seq++;
            *target_kv.seq_len = s_target_seq;
        }
        __syncthreads();
    }

    if (tid == 0) {
        result->output_tokens[0] = s_last_token;
        s_generated              = 1;
    }
    __syncthreads();

    while (s_generated < max_new_tokens) {

        if (tid == 0) {
            s_iterations++;
            int ck = spec_k_param;
            if (s_generated + ck + 1 > max_new_tokens)
                ck = max_new_tokens - s_generated - 1;
            if (ck < 1)
                ck = 1;
            s_ck = ck;
            s_total_proposed += ck;
            s_ds_save = s_draft_seq;
            s_ts_save = s_target_seq;
        }
        __syncthreads();

        const int ck = s_ck;

        if (tid == 0)
            s_cur_in = s_last_token;
        __syncthreads();

        for (int di = 0; di < ck; di++) {
            model_forward_logits(draft_model,
                                 draft_kv,
                                 s_cur_in,
                                 s_draft_seq,
                                 hidden,
                                 logits_draft,
                                 scratch);
            __syncthreads();

            logits_inplace_softmax_temp(logits_draft,
                                        V_vocab,
                                        s_dyn_dt,
                                        scratch);
            __syncthreads();

            if (tid == 0) {
                float cu = curand_uniform(&s_rng);
                float acc = 0.f;
                int   pick_w = V_vocab - 1;
                for (int z = 0; z < V_vocab; z++) {
                    acc += logits_draft[z];
                    if (cu <= acc || z == V_vocab - 1) {
                        pick_w = z;
                        break;
                    }
                }
                s_draft_ids[di] = pick_w;
                s_q_probs[di]   = logits_draft[pick_w];
                s_cur_in        = pick_w;
                s_draft_seq++;
                *draft_kv.seq_len = s_draft_seq;
            }
            __syncthreads();
        }

        // ---- Batched target verification (reads weights ONCE for all k+1 tokens) ----
        {
            __shared__ int s_verify_toks[MAX_MEGA_K + 1];
            if (tid == 0) {
                s_verify_toks[0] = s_last_token;
                for (int i = 0; i < ck; i++)
                    s_verify_toks[i + 1] = s_draft_ids[i];
                acc_cnt = 0;
            }
            __syncthreads();

            model_batch_forward_logits(
                target_model, target_kv,
                s_verify_toks, s_ts_save,
                ck + 1,
                g_batch_hidden, g_batch_work, g_batch_logits,
                scratch);

            bool inner_done = false;
            for (int vi = 0; vi < ck; vi++) {
                float pv = logits_softmax_prob_at_global(
                    g_batch_logits + vi * V_vocab,
                    V_vocab, s_draft_ids[vi], 1.f, scratch);
                __syncthreads();

                if (tid == 0) {
                    s_ok_gate = device_stochastic_accept_mass(
                        pv, s_q_probs[vi], &s_rng) ? 1 : 0;
                }
                __syncthreads();

                if (s_ok_gate) {
                    if (tid == 0) acc_cnt++;
                    __syncthreads();
                    continue;
                }

                if (tid == 0) {
                    *draft_kv.seq_len  = s_ds_save + vi;
                    *target_kv.seq_len = s_ts_save + vi;
                    s_tr_roll    = s_ts_save + vi;
                    s_draft_seq  = s_ds_save + vi;
                    s_target_seq = s_tr_roll;
                }
                __syncthreads();

                int in_t = (vi == 0) ? s_last_token : s_draft_ids[vi - 1];
                model_forward_logits(draft_model, draft_kv, in_t,
                                     s_ds_save + vi, hidden,
                                     logits_draft, scratch);
                __syncthreads();

                if (tid == 0) {
                    int bon_c = device_corrected_adjusted_sample(
                        g_batch_logits + vi * V_vocab,
                        logits_draft, V_vocab, &s_rng,
                        corr_workspace, 1.f, s_dyn_dt);
                    s_bonus_token = bon_c;
                }
                __syncthreads();

                model_forward_logits(target_model, target_kv,
                                     s_bonus_token, s_ts_save + acc_cnt,
                                     hidden, logits_target, scratch);
                __syncthreads();

                if (tid == 0) {
                    s_target_seq       = s_ts_save + acc_cnt + 1;
                    s_tr_roll          = s_target_seq;
                    *target_kv.seq_len = s_target_seq;
                }
                inner_done = true;
                __syncthreads();
                break;
            }

            __syncthreads();

            if (!inner_done && ck > 0) {
                for (int i = tid; i < V_vocab; i += blockDim.x)
                    logits_target[i] = g_batch_logits[ck * V_vocab + i];
                __syncthreads();

                if (tid == 0) {
                    int bon_full =
                        device_softmax_sample_logits_temp_inplace(
                            logits_target, V_vocab, 1.f, &s_rng);
                    s_bonus_token = bon_full;
                }
                __syncthreads();

                model_forward_logits(target_model, target_kv,
                                     s_bonus_token, s_ts_save + ck,
                                     hidden, logits_target, scratch);
                __syncthreads();

                if (tid == 0) {
                    acc_cnt        = ck;
                    s_target_seq   = s_ts_save + ck + 1;
                    s_tr_roll      = s_target_seq;
                    *target_kv.seq_len = s_target_seq;
                }
                __syncthreads();
            }
        }

        __syncthreads();

        if (tid == 0) {
            s_n_accepted           = acc_cnt;
            s_total_accepted += acc_cnt;

            for (int w = 0; w < acc_cnt && s_generated < max_new_tokens; w++)
                result->output_tokens[s_generated++] = s_draft_ids[w];

            if (s_generated < max_new_tokens)
                result->output_tokens[s_generated++] = s_bonus_token;

            s_last_token                   = s_bonus_token;

            s_target_seq = s_ts_save + s_n_accepted + 1;

            *target_kv.seq_len = s_target_seq;
            s_draft_seq =
                s_ds_save + s_n_accepted;

            *draft_kv.seq_len = s_draft_seq;
        }

        __syncthreads();

        // All-accepted stochastic: fill missing K/V for draft[ck-1].
        // Afterward draft_seq == target_seq; skip the bonus sync to prevent
        // a one-step RoPE misalignment that compounds across rounds.
        if (s_n_accepted == ck && ck > 0) {
            model_forward(draft_model,
                          draft_kv,
                          s_draft_ids[ck - 1],
                          s_draft_seq,
                          hidden,
                          logits_draft,
                          scratch);
            if (tid == 0) {
                s_draft_seq++;
                *draft_kv.seq_len = s_draft_seq;
            }
            __syncthreads();
        } else {
            // Partial accept: sync draft cache to bonus token position.
            model_forward(draft_model,
                          draft_kv,
                          s_last_token,
                          s_draft_seq,
                          hidden,
                          logits_draft,
                          scratch);
            if (tid == 0) {
                s_draft_seq++;
                *draft_kv.seq_len = s_draft_seq;
            }
            __syncthreads();
        }

        __syncthreads();

        if (adaptive_enabled && ck > 0 && tid == 0) {
            float mix = adapt_ewma_mix;
            if (mix < 1e-6f)
                mix = 1e-6f;
            else if (mix > 1.f - 1e-6f)
                mix = 1.f - 1e-6f;
            float rr = (float)acc_cnt / (float)ck;
            if (s_ewma < 0.f)
                s_ewma = rr;
            else
                s_ewma = (1.f - mix) * s_ewma + mix * rr;
            s_dyn_dt +=
                adapt_gain * (s_ewma - adapt_tgt_accept);
            s_dyn_dt = fmaxf(min_draft_temp,
                             fminf(max_draft_temp, s_dyn_dt));
        }

        __syncthreads();

        if ((eos_token >= 0 && s_last_token == eos_token) || s_generated >= max_new_tokens)
            break;
    }

    if (tid == 0) {
        result->n_generated      = s_generated;
        result->draft_proposed  = s_total_proposed;
        result->draft_accepted  = s_total_accepted;
        result->spec_iterations = s_iterations;
    }
}

// ============================================================================
// Host wrappers for megakernel launches
// ============================================================================

void megakernel_baseline(const ModelWeights& target_model,
                         KVCache& target_kv,
                         const int* prompt, int prompt_len,
                         GenerationResult* d_result,
                         const GenerationParams& params) {
    float* g_logits;
    CUDA_CHECK(cudaMalloc(&g_logits,
                          (size_t)target_model.cfg.vocab_size * sizeof(float)));

    kv_cache_reset(target_kv);
    size_t smem_bytes = compute_smem_bytes(target_model.cfg);
    cuda_configure_kernel_dynamic_smem(megakernel_baseline_kernel, smem_bytes);

    ModelWeights* d_target_model;
    KVCache*      d_target_kv;
    CUDA_CHECK(cudaMalloc(&d_target_model, sizeof(ModelWeights)));
    CUDA_CHECK(cudaMalloc(&d_target_kv, sizeof(KVCache)));
    CUDA_CHECK(cudaMemcpy(d_target_model, &target_model, sizeof(ModelWeights),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_kv, &target_kv, sizeof(KVCache),
                          cudaMemcpyHostToDevice));

    megakernel_baseline_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
        d_target_model, d_target_kv, prompt, prompt_len,
        params.max_new_tokens, g_logits, d_result, params.eos_token);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_target_model);
    cudaFree(d_target_kv);
    cudaFree(g_logits);
}

void megakernel_speculative(const ModelWeights& draft_model,
                            const ModelWeights& target_model,
                            KVCache& draft_kv,
                            KVCache& target_kv,
                            const int* prompt, int prompt_len,
                            GenerationResult* d_result,
                            const GenerationParams& params) {
    int vocab = (target_model.cfg.vocab_size > draft_model.cfg.vocab_size)
                ? target_model.cfg.vocab_size
                : draft_model.cfg.vocab_size;

    kv_cache_reset(draft_kv);
    kv_cache_reset(target_kv);

    ModelWeights* d_draft_model;
    ModelWeights* d_target_model;
    KVCache*      d_draft_kv;
    KVCache*      d_target_kv;
    CUDA_CHECK(cudaMalloc(&d_draft_model, sizeof(ModelWeights)));
    CUDA_CHECK(cudaMalloc(&d_target_model, sizeof(ModelWeights)));
    CUDA_CHECK(cudaMalloc(&d_draft_kv, sizeof(KVCache)));
    CUDA_CHECK(cudaMalloc(&d_target_kv, sizeof(KVCache)));
    CUDA_CHECK(cudaMemcpy(d_draft_model, &draft_model, sizeof(ModelWeights),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_model, &target_model, sizeof(ModelWeights),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft_kv, &draft_kv, sizeof(KVCache),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_kv, &target_kv, sizeof(KVCache),
                          cudaMemcpyHostToDevice));

    size_t smem_bytes = compute_smem_bytes(target_model.cfg);

    if (params.stochastic_spec_decode) {
        if (draft_model.cfg.vocab_size != target_model.cfg.vocab_size) {
            fprintf(stderr,
                    "megakernel_speculative: stochastic mode requires identical "
                    "draft/target vocab_size (draft=%d target=%d)\n",
                    draft_model.cfg.vocab_size,
                    target_model.cfg.vocab_size);
            exit(EXIT_FAILURE);
        }

        float* logits_d = nullptr;
        float* logits_t = nullptr;
        float* corr_ws  = nullptr;
        CUDA_CHECK(cudaMalloc(&logits_d, (size_t)vocab * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&logits_t, (size_t)vocab * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&corr_ws, (size_t)vocab * 3 * sizeof(float)));

        int max_B = params.spec_k + 1;
        int d_t   = target_model.cfg.d_model;
        int V_t   = target_model.cfg.vocab_size;

        float* g_batch_hidden_s;
        float* g_batch_work_s;
        float* g_batch_logits_s;
        CUDA_CHECK(cudaMalloc(&g_batch_hidden_s,
                              (size_t)max_B * d_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_work_s,
                              (size_t)(7 * d_t + MLP_FF_TILE) * max_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_logits_s,
                              (size_t)max_B * V_t * sizeof(float)));

        cuda_configure_kernel_dynamic_smem(
            megakernel_speculative_stochastic_kernel,
            smem_bytes);

        megakernel_speculative_stochastic_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            d_draft_model, d_target_model, d_draft_kv, d_target_kv,
            prompt, prompt_len, params.max_new_tokens, params.spec_k,
            logits_d, logits_t, corr_ws,
            g_batch_hidden_s, g_batch_work_s, g_batch_logits_s,
            params.draft_temperature,
            params.adaptive_draft_temperature ? 1 : 0,
            params.min_draft_temperature,
            params.max_draft_temperature,
            params.stochastic_adapt_target_accept,
            params.stochastic_adapt_temp_gain,
            params.stochastic_adapt_ewma_mix,
            (unsigned long long)params.stochastic_rng_seed,
            d_result,
            params.eos_token);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(logits_d);
        cudaFree(logits_t);
        cudaFree(corr_ws);
        cudaFree(g_batch_hidden_s);
        cudaFree(g_batch_work_s);
        cudaFree(g_batch_logits_s);
    } else {
        float* g_logits;
        CUDA_CHECK(cudaMalloc(&g_logits, (size_t)vocab * sizeof(float)));

        int max_B = params.spec_k + 1;
        int d_t   = target_model.cfg.d_model;
        int V_t   = target_model.cfg.vocab_size;

        float* g_batch_hidden;
        float* g_batch_work;
        float* g_batch_logits;
        CUDA_CHECK(cudaMalloc(&g_batch_hidden,
                              (size_t)max_B * d_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_work,
                              (size_t)(7 * d_t + MLP_FF_TILE) * max_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_logits,
                              (size_t)max_B * V_t * sizeof(float)));

        cuda_configure_kernel_dynamic_smem(megakernel_speculative_kernel,
                                           smem_bytes);

        megakernel_speculative_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            d_draft_model, d_target_model, d_draft_kv, d_target_kv,
            prompt, prompt_len, params.max_new_tokens, params.spec_k,
            g_logits,
            g_batch_hidden, g_batch_work, g_batch_logits,
            d_result, params.eos_token);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(g_logits);
        cudaFree(g_batch_hidden);
        cudaFree(g_batch_work);
        cudaFree(g_batch_logits);
    }

    cudaFree(d_draft_model);
    cudaFree(d_target_model);
    cudaFree(d_draft_kv);
    cudaFree(d_target_kv);
}
