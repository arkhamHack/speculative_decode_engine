#include "kernels.h"
#include "attention.h"
#include "gemm_backend.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
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

// Shared memory layout (dynamic — sized at launch via compute_smem_bytes())
//
//   shared[0          .. d_model)         hidden      [d_model]
//   shared[d_model .. end]   scratch for layers + final RMSNorm
//                               (streaming attention + tiled MLP — see model.cu)
//
// Logits are written to a global-memory buffer (g_logits) allocated by the
// host wrapper.  This removes the vocab_size limit from shared memory and
// allows real models with vocab_size >> 256.

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

// Leviathan/Chen stochastic helpers (single block, logits in global memory)
//
// Scratch smem reused from the normed region prefix (>= warp reduction slots via
// d_model floats is typical). After forward logits, hidden/norm/etc. are disposable.

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

// Block-parallel inverse-CDF sample from a probability array p[0..V) (sums ~1).
// Draws u once on thread 0, then all threads cooperatively find the sampled index
// (first i where prefix_sum(p[0..i]) >= u): each thread sums a contiguous slice,
// an exclusive prefix over the blockDim slice-sums locates the slice holding u,
// and that thread walks it. ~blockDim-fold faster than the serial O(V) walk;
// equivalent up to float rounding at bin boundaries (a non-issue for sampling).
__device__ __forceinline__ int device_sample_inverse_cdf(const float* p, int V,
                                                         curandState* rng) {
    __shared__ float s_seg[BLOCK_THREADS];   // per-thread slice sum, then prefix
    __shared__ float s_u;
    __shared__ int   s_pick;
    int tid = threadIdx.x;
    int nt  = blockDim.x;

    if (tid == 0) { s_u = curand_uniform(rng); s_pick = V - 1; }
    __syncthreads();

    int seg = (V + nt - 1) / nt;             // contiguous slice per thread
    int lo  = tid * seg;
    int hi  = (lo + seg < V) ? lo + seg : V;

    float local = 0.f;
    for (int i = lo; i < hi; i++) local += p[i];
    s_seg[tid] = local;
    __syncthreads();

    if (tid == 0) {                          // exclusive prefix over blockDim sums
        float run = 0.f;
        for (int t = 0; t < nt; t++) { float v = s_seg[t]; s_seg[t] = run; run += v; }
    }
    __syncthreads();

    float base = s_seg[tid];                 // = sum(p[0 .. lo-1])
    float u    = s_u;
    if (lo < hi && u > base && u <= base + local) {
        float acc = base;
        for (int i = lo; i < hi; i++) { acc += p[i]; if (u <= acc) { s_pick = i; break; } }
    }
    __syncthreads();
    return s_pick;
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

    int choice = device_sample_inverse_cdf(g_logits, V, rng);
    if (threadIdx.x == 0) {
        sampled_id[0]     = choice;
        sampled_q_prob[0] = g_logits[choice];
    }
    __syncthreads();
}

// Self-spec stochastic draft: same as above but saves prefix activation for verify.
__global__ void stochastic_draft_forward_sample_save_dptr_kernel(
        ModelWeights target_model,
        KVCache kv,
        const int* d_token_id,
        int seq_len,
        int n_prefix_layers,
        float temperature,
        curandState* rng,
        float* g_logits,
        float* g_saved_hidden,
        int* sampled_id,
        float* sampled_q_prob) {
    extern __shared__ float shared[];

    int token_id = d_token_id[0];
    int d = target_model.cfg.d_model;
    int V = target_model.cfg.vocab_size;
    float* hidden = shared;
    float* smem   = shared + d;

    model_forward_draft_logits(target_model, kv, token_id, seq_len,
                               n_prefix_layers, hidden, g_logits,
                               g_saved_hidden, smem);
    __syncthreads();

    float* softmax_scratch = hidden;
    logits_inplace_softmax_temp(g_logits, V, temperature, softmax_scratch);

    int choice = device_sample_inverse_cdf(g_logits, V, rng);
    if (threadIdx.x == 0) {
        sampled_id[0]     = choice;
        sampled_q_prob[0] = g_logits[choice];
    }
    __syncthreads();
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

// Self-spec verify: reuse draft-saved activation when use_saved != 0.
__global__ void target_fwd_prob_and_accept_selfspec_kernel(
        ModelWeights target_model,
        KVCache kv,
        int token_id, int seq_len,
        int layer_start,
        const float* saved_hidden,
        int use_saved,
        int draft_token,
        float q_mass,
        float* g_logits,
        curandState* rng,
        int* accepted_flag)
{
    extern __shared__ float shared[];
    int d = target_model.cfg.d_model;
    int V = target_model.cfg.vocab_size;
    float* hidden   = shared;
    float* lay_smem = shared + d;

    if (use_saved)
        model_forward_verify_from_hidden_logits(
            target_model, kv, saved_hidden, layer_start, seq_len,
            hidden, g_logits, lay_smem);
    else
        model_forward_logits(
            target_model, kv, token_id, seq_len,
            hidden, g_logits, lay_smem);
    __syncthreads();

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
    // Softmax is block-parallel (dominant cost at large vocab); the inverse-CDF
    // draw stays on thread 0 so the RNG sequence is unchanged.
    __shared__ float red[BLOCK_THREADS];
    logits_inplace_softmax_temp(logits, V, temperature, red);   // logits → probs
    if (threadIdx.x == 0) {
        float u_fix = curand_uniform(rng);
        float cdf   = 0.f;
        for (int i = 0; i < V; i++) {
            cdf += logits[i];
            if (cdf >= u_fix || i == V - 1) { out_token[0] = i; return; }
        }
        out_token[0] = V - 1;
    }
}

// Sample from raw logits and also return the chosen token's probability mass.
// Mutates logits in-place to softmax probabilities.
__global__ void sample_from_logits_q_kernel(float* logits, int V,
                                            float temperature,
                                            curandState* rng,
                                            int* sampled_id,
                                            float* sampled_q_prob) {
    // Block-parallel softmax (dominant cost at large vocab), then a thread-0
    // inverse-CDF draw — same RNG sequence as the single-thread version.
    __shared__ float red[BLOCK_THREADS];
    logits_inplace_softmax_temp(logits, V, temperature, red);   // logits → probs
    if (threadIdx.x == 0) {
        float u   = curand_uniform(rng);
        float cdf = 0.f;
        int choice = V - 1;
        for (int i = 0; i < V; i++) {
            cdf += logits[i];
            if (u <= cdf || i == V - 1) { choice = i; break; }
        }
        sampled_id[0]     = choice;
        sampled_q_prob[0] = logits[choice];
    }
}

// Stochastic accept gate from raw (unnormalized) target logits.
// Does not mutate logits (read-only mass computation).
__global__ void accept_from_logits_kernel(float* g_logits, int V,
                                          int draft_token, float q_mass,
                                          curandState* rng,
                                          int* accepted_flag) {
    extern __shared__ float shared[];
    float* scratch = shared;
    float pm = logits_softmax_prob_at_global(g_logits, V, draft_token, 1.f, scratch);
    __syncthreads();
    if (threadIdx.x == 0) {
        float q_eff = fmaxf(q_mass, 1e-36f);
        float cap   = fminf(1.f, pm / q_eff);
        *accepted_flag = (curand_uniform(rng) < cap) ? 1 : 0;
    }
}

__global__ void rng_init_kernel(curandState* state,
                                unsigned long long seed_worker) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        curand_init(seed_worker, 0ULL, 0ULL, state);
}

//  MULTI-KERNEL PATH

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

// Batched stochastic accept gate: one launch computes the accept flag for all k
// draft positions (vs one kernel + host sync per position). Each iteration does a
// block-parallel softmax prob-mass of the drafted token in the target dist p, then
// thread 0 applies the min(1, p/q) gate with a fresh RNG draw. The host then keeps
// the longest accepted prefix. Replaces the per-token launch/sync/copy loop that
// re-introduced the overhead the batched verify had just amortized away.
__global__ void batch_stochastic_accept_kernel(
        const float* g_logits, int V,
        const int* draft_tokens, const float* q_probs, int k,
        curandState* rng, int* accept_flags) {
    extern __shared__ float shared[];
    for (int vi = 0; vi < k; vi++) {
        float pm = logits_softmax_prob_at_global(
            g_logits + (size_t)vi * V, V, draft_tokens[vi], 1.f, shared);
        __syncthreads();
        if (threadIdx.x == 0) {
            float q_eff = fmaxf(q_probs[vi], 1e-36f);
            float cap   = fminf(1.f, pm / q_eff);
            accept_flags[vi] = (curand_uniform(rng) < cap) ? 1 : 0;
        }
        __syncthreads();
    }
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

// Prefix-only save for the verify bonus slot (position k in a k+1 batch).
__global__ void selfspec_bonus_prefix_save_dptr_kernel(
        ModelWeights target_model, KVCache kv,
        const int* d_token_id, int seq_len, int n_prefix_layers,
        float* g_saved_hidden) {
    extern __shared__ float shared[];
    int token_id = d_token_id[0];
    int d = target_model.cfg.d_model;
    float* hidden = shared;
    float* smem   = shared + d;
    model_forward_draft_prefix_save(
        target_model, kv, token_id, seq_len, n_prefix_layers,
        hidden, g_saved_hidden, smem);
}

// Self-spec draft: k tokens + bonus prefix save, saving post-prefix activations.
__global__ void batch_draft_selfspec_kernel(
        ModelWeights target_model, KVCache kv,
        int first_token, int k, int start_seq_len,
        int n_prefix_layers,
        float* g_logits,
        float* g_saved_hiddens,  // [(k + 1) * d_model]
        int*   out_tokens)
{
    extern __shared__ float shared[];
    int d      = target_model.cfg.d_model;
    int V      = target_model.cfg.vocab_size;
    float* hidden = shared;
    float* smem   = shared + d;

    __shared__ int s_cur;
    if (threadIdx.x == 0) s_cur = first_token;
    __syncthreads();

    for (int i = 0; i < k; i++) {
        model_forward_draft_logits(
            target_model, kv, s_cur, start_seq_len + i,
            n_prefix_layers, hidden, g_logits,
            g_saved_hiddens + (size_t)i * d, smem);
        int nxt = global_argmax(g_logits, V, smem);
        if (threadIdx.x == 0) { out_tokens[i] = nxt; s_cur = nxt; }
        __syncthreads();
    }

    if (k > 0) {
        model_forward_draft_prefix_save(
            target_model, kv, s_cur, start_seq_len + k,
            n_prefix_layers, hidden,
            g_saved_hiddens + (size_t)k * d, smem);
    }
}

// Self-spec verify: all positions load saved[b] and run suffix layers.
__global__ void batch_target_verify_selfspec_kernel(
        ModelWeights model, KVCache kv,
        const int* verify_tokens, int batch_size, int start_seq_len,
        int layer_start,
        const float* g_saved_hiddens,  // [batch_size * d_model]
        float* g_hidden, float* g_work,
        float* g_logits_batch,         // [batch_size * vocab_size]
        int*   out_tokens)
{
    extern __shared__ float shared[];
    float* smem = shared;

    model_batch_forward_selfspec_verify_logits(
        model, kv, verify_tokens, start_seq_len, batch_size, layer_start,
        g_saved_hiddens, g_hidden, g_work, g_logits_batch, smem);

    int V = model.cfg.vocab_size;
    for (int i = 0; i < batch_size; i++) {
        int nxt = global_argmax(g_logits_batch + (size_t)i * V, V, smem);
        if (threadIdx.x == 0) out_tokens[i] = nxt;
        __syncthreads();
    }
}

// Cooperative self-spec verify: one block per token; suffix from saved[b].
__global__ void batch_target_verify_coop_selfspec_kernel(
        ModelWeights model, KVCache kv,
        const int* verify_tokens,
        int batch_size,
        int start_seq_len,
        int layer_start,
        const float* g_saved_hiddens,  // [batch_size * d_model]
        float* g_logits_batch,
        int* out_tokens)
{
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    int b = (int)blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    int d = model.cfg.d_model;
    float* hidden = shared;
    float* smem   = shared + d;

    int seq_pos = start_seq_len + b;

    const float* src = g_saved_hiddens + (size_t)b * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        hidden[i] = src[i];
    __syncthreads();

    for (int l = layer_start; l < model.cfg.n_layers; l++) {
        model_layer_kv_phase(model, l, hidden, kv, seq_pos, smem);
        grid.sync();
        model_layer_attn_mlp_phase(model, l, hidden, kv, seq_pos, smem);
        grid.sync();
    }

    float* my_logits = g_logits_batch + (size_t)b * model.cfg.vocab_size;
    model_output(model, hidden, my_logits, smem);
    int next = global_argmax(my_logits, model.cfg.vocab_size, smem);
    if (threadIdx.x == 0) out_tokens[b] = next;
}

// cuBLAS prefill utility kernels  (global kernels on global-memory buffers)
//
// These are ONLY used by the host-orchestrated cuBLAS prefill path.
// The legacy device-internal paths use the __device__ functions in model.cu.

// Batched embedding: g_hidden[M, d] from token_ids[M]
__global__ void cublas_embed_kernel(const half* token_embedding, const int* token_ids,
                                    float* g_hidden, int d, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / d, col = idx % d;
    if (m < M)
        g_hidden[m * d + col] = __half2float(token_embedding[token_ids[m] * d + col]);
}

// Batched RMSNorm: g_out[M, d] = RMSNorm(g_in[M, d], weight[d])
// One block per token. smem: input[d] + reduction[WARP_SIZE]
__global__ void cublas_rmsnorm_kernel(const float* g_in, const half* weight,
                                      float* g_out, int d, int M) {
    int m = blockIdx.x;
    if (m >= M) return;
    const float* x = g_in + (size_t)m * d;
    float* out = g_out + (size_t)m * d;
    extern __shared__ float s_rms[];
    float* s_red = s_rms + d;

    for (int i = threadIdx.x; i < d; i += blockDim.x)
        s_rms[i] = x[i];
    __syncthreads();

    device_rmsnorm(s_rms, weight, out, d, s_red);
}

// Batched RoPE: apply RoPE in-place on g_x[M, n_heads * dph] with positions [pos_base, pos_base+M)
__global__ void cublas_rope_kernel(float* g_x, int n_heads_x, int dph,
                                   int pos_base, int M,
                                   float rope_theta,
                                   RopeScalingType scaling_type,
                                   float scaling_factor) {
    int m = blockIdx.x;
    if (m >= M) return;
    float* x = g_x + m * n_heads_x * dph;
    extern __shared__ float s_rope[];

    int total = n_heads_x * dph;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        s_rope[i] = x[i];
    __syncthreads();

    rope_apply_inplace(s_rope, n_heads_x, dph, pos_base + m,
                       rope_theta, scaling_type, scaling_factor);

    for (int i = threadIdx.x; i < total; i += blockDim.x)
        x[i] = s_rope[i];
}

// Convert fp32 → fp16 for cuBLAS input
__global__ void f32_to_f16_kernel(const float* src, half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

// Batched KV cache append: write K[M, kd] and V[M, kd] to cache at positions [pos_base..pos_base+M)
__global__ void cublas_kv_append_kernel(KVCache kv, int layer,
                                        const float* g_k, const float* g_v,
                                        int kd, int pos_base, int M) {
    int m = blockIdx.x;
    if (m >= M) return;
    int pos = pos_base + m;
    int bidx = pos / KV_BLOCK_SIZE, sl = pos % KV_BLOCK_SIZE;
    int pb = kv.layers[layer].block_table[bidx];
    half* base = kv.pool + (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
    half* k_dst = base + sl * kv.d_head;
    half* v_dst = base + KV_BLOCK_SIZE * kv.d_head + sl * kv.d_head;

    const float* k_src = g_k + m * kd;
    const float* v_src = g_v + m * kd;
    for (int i = threadIdx.x; i < kd; i += blockDim.x) {
        k_dst[i] = __float2half(k_src[i]);
        v_dst[i] = __float2half(v_src[i]);
    }
}

// Batched residual add: g_hidden[M, d] += g_residual[M, d]
__global__ void cublas_residual_add_kernel(float* g_hidden, const float* g_residual,
                                           int d, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * d;
    if (idx < total)
        g_hidden[idx] += g_residual[idx];
}

// Broadcast-add bias: g_x[m, i] += bias[i]  (no-op when bias == nullptr)
__global__ void cublas_add_bias_kernel(float* g_x, const half* bias, int M, int n) {
    if (!bias) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * n;
    if (idx < total)
        g_x[idx] += __half2float(bias[idx % n]);
}

// Batched SwiGLU: out[i] = silu(gate[i]) * up[i]
__global__ void cublas_swiglu_kernel(const float* gate, const float* up,
                                     float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        out[idx] = (g / (1.0f + expf(-g))) * up[idx];
    }
}

// Argmax over g_logits[V] → d_out (single token, last position)
__global__ void cublas_argmax_kernel(const float* g_logits, int V, int* d_out) {
    extern __shared__ float s_arg[];
    float* s_val = s_arg;
    int* s_idx = (int*)(s_arg + blockDim.x);

    float best_val = -FLT_MAX;
    int best_idx = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = g_logits[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    s_val[threadIdx.x] = best_val;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_val[threadIdx.x + s] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + s];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *d_out = s_idx[0];
}

// GQA-aware flash attention on a single token's Q against the KV cache.
// One block per token (blockIdx.x = token index m).
__global__ void cublas_attention_kernel(
    const float* g_Q, float* g_attn_out,
    KVCache kv, int layer,
    int nh, int nkv, int dph, int kd,
    int pos_base, int M)
{
    int m = blockIdx.x;
    if (m >= M) return;

    int tid = threadIdx.x;
    int d   = nh * dph;
    int total_len = pos_base + m + 1;
    float scale   = rsqrtf((float)dph);

    extern __shared__ float smem_a[];
    float* s_q   = smem_a;
    float* s_out = smem_a + d;
    float* s_blk = smem_a + 2 * d;

    const float* q_src = g_Q + (size_t)m * d;
    for (int i = tid; i < d; i += blockDim.x)
        s_q[i] = q_src[i];
    for (int i = tid; i < d; i += blockDim.x)
        s_out[i] = 0.f;
    __syncthreads();

    __shared__ float s_m, s_l, s_tm, s_alpha, s_inv;

    device_flash_attention_gqa(
        s_q, s_out, kv, layer,
        nh, nkv, dph, total_len,
        s_blk, s_m, s_l, s_tm, s_alpha, s_inv);

    float* out_dst = g_attn_out + (size_t)m * d;
    for (int i = tid; i < d; i += blockDim.x)
        out_dst[i] = s_out[i];
}

// Lazy-growth scratch allocator for cuBLAS forward paths.
// Allocates on first call, grows (never shrinks) when M increases.

static void cublas_ensure_scratch(InferenceEngine& eng,
                                  const ModelConfig& cfg, int M) {
    const int d   = cfg.d_model;
    const int kd  = kv_dim(cfg);
    const int dff = cfg.d_ff;

    size_t need_f = (size_t)M * (4*d + 2*kd + 2*dff);
    size_t need16 = (size_t)M * d;
    if ((size_t)M * dff > need16) need16 = (size_t)M * dff;

    if (eng.cublas_buf_floats < need_f) {
        if (eng.d_cublas_buf) { cudaFree(eng.d_cublas_buf); eng.d_cublas_buf = nullptr; }
        CUDA_CHECK(cudaMalloc(&eng.d_cublas_buf, need_f * sizeof(float)));
        eng.cublas_buf_floats = need_f;
    }
    if (eng.cublas_x16_halves < need16) {
        if (eng.d_cublas_x16) { cudaFree(eng.d_cublas_x16); eng.d_cublas_x16 = nullptr; }
        CUDA_CHECK(cudaMalloc(&eng.d_cublas_x16, need16 * sizeof(half)));
        eng.cublas_x16_halves = need16;
    }
}

// Host-orchestrated cuBLAS forward pass.
//
// Handles prefill (M = prompt_len), decode (M = 1), and batched verify
// (M = spec_k + 1).  Tokens are placed at KV positions
// [seq_base, seq_base + M).
//
// When all_token_logits == false (prefill / decode):
//   Final RMSNorm + output GEMM + argmax on the LAST token only.
//   g_logits must hold vocab_size floats.
//   d_out_tokens must hold 1 int.
//
// When all_token_logits == true (verify):
//   Final RMSNorm + output GEMM + argmax on ALL M tokens.
//   g_logits must hold M * vocab_size floats.
//   d_out_tokens must hold M ints.

static void cublas_forward(const ModelWeights& model, KVCache& kv,
                           const int* d_tokens, int M,
                           int seq_base,
                           float* g_logits, int* d_out_tokens,
                           InferenceEngine& eng,
                           bool all_token_logits = false) {
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int nh  = model.cfg.n_heads;
    const int nkv = model.cfg.n_kv_heads;
    const int dph = d / nh;
    const int kd  = nkv * dph;
    const int V   = model.cfg.vocab_size;

    cudaStream_t stream = eng.streams[0];
    GemmBackend gb;
    gb.init(eng.cublas, stream);

    cublas_ensure_scratch(eng, model.cfg, M);

    float* d_buf = eng.d_cublas_buf;
    half*  d_x16 = eng.d_cublas_x16;

    float* g_hidden = d_buf;
    float* g_normed = g_hidden + (size_t)M * d;
    float* g_Q      = g_normed + (size_t)M * d;
    float* g_K      = g_Q      + (size_t)M * d;
    float* g_V      = g_K      + (size_t)M * kd;
    float* g_attn   = g_V      + (size_t)M * kd;
    float* g_gate   = g_attn   + (size_t)M * d;
    float* g_up     = g_gate   + (size_t)M * dff;

    size_t rms_smem  = ((size_t)d + WARP_SIZE) * sizeof(float);
    size_t attn_smem = ((size_t)2 * d + KV_BLOCK_SIZE) * sizeof(float) + 16;
    int bk = 256;

    auto cvt = [&](const float* s, half* dst, int n) {
        f32_to_f16_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(s, dst, n);
    };

    // Embed
    {
        int n = M * d;
        cublas_embed_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
            model.token_embedding, d_tokens, g_hidden, d, M);
    }

    // Layer loop
    for (int l = 0; l < model.cfg.n_layers; l++) {
        const LayerWeights& lw = model.layers[l];

        cublas_rmsnorm_kernel<<<M, BLOCK_THREADS, rms_smem, stream>>>(
            g_hidden, lw.rms_attn_weight, g_normed, d, M);

        cvt(g_normed, d_x16, M * d);
        gemm_backend_gemm(gb, M, d,  d, d_x16, lw.Wq, g_Q);
        gemm_backend_gemm(gb, M, kd, d, d_x16, lw.Wk, g_K);
        gemm_backend_gemm(gb, M, kd, d, d_x16, lw.Wv, g_V);
        if (lw.Wq_bias) {
            int n = M * d;
            cublas_add_bias_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(g_Q, lw.Wq_bias, M, d);
        }
        if (lw.Wk_bias) {
            int n = M * kd;
            cublas_add_bias_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(g_K, lw.Wk_bias, M, kd);
        }
        if (lw.Wv_bias) {
            int n = M * kd;
            cublas_add_bias_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(g_V, lw.Wv_bias, M, kd);
        }

        size_t rope_q_smem = (size_t)d  * sizeof(float);
        size_t rope_k_smem = (size_t)kd * sizeof(float);
        cublas_rope_kernel<<<M, BLOCK_THREADS, rope_q_smem, stream>>>(
            g_Q, nh, dph, seq_base, M, model.cfg.rope_theta,
            model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);
        cublas_rope_kernel<<<M, BLOCK_THREADS, rope_k_smem, stream>>>(
            g_K, nkv, dph, seq_base, M, model.cfg.rope_theta,
            model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);

        cublas_kv_append_kernel<<<M, BLOCK_THREADS, 0, stream>>>(
            kv, l, g_K, g_V, kd, seq_base, M);

        cublas_attention_kernel<<<M, BLOCK_THREADS, attn_smem, stream>>>(
            g_Q, g_attn, kv, l, nh, nkv, dph, kd, seq_base, M);

        cvt(g_attn, d_x16, M * d);
        gemm_backend_gemm(gb, M, d, d, d_x16, lw.Wo, g_normed);
        {
            int n = M * d;
            cublas_residual_add_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_hidden, g_normed, d, M);
        }

        cublas_rmsnorm_kernel<<<M, BLOCK_THREADS, rms_smem, stream>>>(
            g_hidden, lw.rms_mlp_weight, g_normed, d, M);

        cvt(g_normed, d_x16, M * d);
        gemm_backend_gemm(gb, M, dff, d, d_x16, lw.W_gate, g_gate);
        gemm_backend_gemm(gb, M, dff, d, d_x16, lw.W_up,   g_up);

        {
            int n = M * dff;
            cublas_swiglu_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_gate, g_up, g_gate, n);
        }

        cvt(g_gate, d_x16, M * dff);
        gemm_backend_gemm(gb, M, d, dff, d_x16, lw.W_down, g_normed);
        {
            int n = M * d;
            cublas_residual_add_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_hidden, g_normed, d, M);
        }
    }

    // Final output
    size_t arg_smem = (size_t)BLOCK_THREADS * (sizeof(float) + sizeof(int));

    if (all_token_logits) {
        // All M tokens: RMSNorm + output GEMM + per-token argmax
        cublas_rmsnorm_kernel<<<M, BLOCK_THREADS, rms_smem, stream>>>(
            g_hidden, model.rms_final_weight, g_normed, d, M);
        cvt(g_normed, d_x16, M * d);
        gemm_backend_gemm(gb, M, V, d, d_x16, model.output_proj, g_logits);

        for (int m = 0; m < M; m++) {
            cublas_argmax_kernel<<<1, BLOCK_THREADS, arg_smem, stream>>>(
                g_logits + (size_t)m * V, V, d_out_tokens + m);
        }
    } else {
        // Last token only
        cublas_rmsnorm_kernel<<<1, BLOCK_THREADS, rms_smem, stream>>>(
            g_hidden + (size_t)(M - 1) * d, model.rms_final_weight, g_normed, d, 1);
        cvt(g_normed, d_x16, d);
        gemm_backend_gemm(gb, 1, V, d, d_x16, model.output_proj, g_logits);

        cublas_argmax_kernel<<<1, BLOCK_THREADS, arg_smem, stream>>>(
            g_logits, V, d_out_tokens);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void cublas_forward_from_hidden(
    const ModelWeights& model, KVCache& kv,
    const float* g_saved_hiddens, int M,
    int seq_base, int layer_start,
    float* g_logits, int* d_out_tokens,
    InferenceEngine& eng);

// Run transformer layers [layer_start, layer_end) on g_hidden[M, d].
// Caller must embed tokens into g_hidden when layer_start == 0.
static void cublas_run_layer_range(
    const ModelWeights& model, KVCache& kv,
    int M, int seq_base, int layer_start, int layer_end,
    float* g_hidden, float* g_normed, float* g_Q, float* g_K, float* g_V,
    float* g_attn, float* g_gate, float* g_up,
    half* d_x16, GemmBackend& gb, cudaStream_t stream, int bk)
{
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int nh  = model.cfg.n_heads;
    const int nkv = model.cfg.n_kv_heads;
    const int dph = d / nh;
    const int kd  = nkv * dph;

    size_t rms_smem  = ((size_t)d + WARP_SIZE) * sizeof(float);
    size_t attn_smem = ((size_t)2 * d + KV_BLOCK_SIZE) * sizeof(float) + 16;

    auto cvt = [&](const float* s, half* dst, int n) {
        f32_to_f16_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(s, dst, n);
    };

    for (int l = layer_start; l < layer_end; l++) {
        const LayerWeights& lw = model.layers[l];

        cublas_rmsnorm_kernel<<<M, BLOCK_THREADS, rms_smem, stream>>>(
            g_hidden, lw.rms_attn_weight, g_normed, d, M);

        cvt(g_normed, d_x16, M * d);
        gemm_backend_gemm(gb, M, d,  d, d_x16, lw.Wq, g_Q);
        gemm_backend_gemm(gb, M, kd, d, d_x16, lw.Wk, g_K);
        gemm_backend_gemm(gb, M, kd, d, d_x16, lw.Wv, g_V);
        if (lw.Wq_bias) {
            int n = M * d;
            cublas_add_bias_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_Q, lw.Wq_bias, M, d);
        }
        if (lw.Wk_bias) {
            int n = M * kd;
            cublas_add_bias_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_K, lw.Wk_bias, M, kd);
        }
        if (lw.Wv_bias) {
            int n = M * kd;
            cublas_add_bias_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_V, lw.Wv_bias, M, kd);
        }

        size_t rope_q_smem = (size_t)d  * sizeof(float);
        size_t rope_k_smem = (size_t)kd * sizeof(float);
        cublas_rope_kernel<<<M, BLOCK_THREADS, rope_q_smem, stream>>>(
            g_Q, nh, dph, seq_base, M, model.cfg.rope_theta,
            model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);
        cublas_rope_kernel<<<M, BLOCK_THREADS, rope_k_smem, stream>>>(
            g_K, nkv, dph, seq_base, M, model.cfg.rope_theta,
            model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);

        cublas_kv_append_kernel<<<M, BLOCK_THREADS, 0, stream>>>(
            kv, l, g_K, g_V, kd, seq_base, M);

        cublas_attention_kernel<<<M, BLOCK_THREADS, attn_smem, stream>>>(
            g_Q, g_attn, kv, l, nh, nkv, dph, kd, seq_base, M);

        cvt(g_attn, d_x16, M * d);
        gemm_backend_gemm(gb, M, d, d, d_x16, lw.Wo, g_normed);
        {
            int n = M * d;
            cublas_residual_add_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_hidden, g_normed, d, M);
        }

        cublas_rmsnorm_kernel<<<M, BLOCK_THREADS, rms_smem, stream>>>(
            g_hidden, lw.rms_mlp_weight, g_normed, d, M);

        cvt(g_normed, d_x16, M * d);
        gemm_backend_gemm(gb, M, dff, d, d_x16, lw.W_gate, g_gate);
        gemm_backend_gemm(gb, M, dff, d, d_x16, lw.W_up,   g_up);

        {
            int n = M * dff;
            cublas_swiglu_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_gate, g_up, g_gate, n);
        }

        cvt(g_gate, d_x16, M * dff);
        gemm_backend_gemm(gb, M, d, dff, d_x16, lw.W_down, g_normed);
        {
            int n = M * d;
            cublas_residual_add_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
                g_hidden, g_normed, d, M);
        }
    }
}

// Embed + prefix layers [0, n_prefix_layers); write post-prefix hidden to g_saved.
static void cublas_prefix_save(
    const ModelWeights& model, KVCache& kv,
    const int* d_tokens, int M, int seq_base, int n_prefix_layers,
    float* g_saved_hidden, InferenceEngine& eng)
{
    const int d = model.cfg.d_model;

    cudaStream_t stream = eng.streams[0];
    GemmBackend gb;
    gb.init(eng.cublas, stream);

    cublas_ensure_scratch(eng, model.cfg, M);

    float* d_buf = eng.d_cublas_buf;
    half*  d_x16 = eng.d_cublas_x16;

    float* g_hidden = d_buf;
    float* g_normed = g_hidden + (size_t)M * d;
    float* g_Q      = g_normed + (size_t)M * d;
    float* g_K      = g_Q      + (size_t)M * d;
    const int kd    = kv_dim(model.cfg);
    float* g_V      = g_K      + (size_t)M * kd;
    float* g_attn   = g_V      + (size_t)M * kd;
    float* g_gate   = g_attn   + (size_t)M * d;
    float* g_up     = g_gate   + (size_t)M * model.cfg.d_ff;

    int bk = 256;
    {
        int n = M * d;
        cublas_embed_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(
            model.token_embedding, d_tokens, g_hidden, d, M);
    }

    cublas_run_layer_range(model, kv, M, seq_base, 0, n_prefix_layers,
                           g_hidden, g_normed, g_Q, g_K, g_V, g_attn,
                           g_gate, g_up, d_x16, gb, stream, bk);

    CUDA_CHECK(cudaMemcpyAsync(g_saved_hidden, g_hidden,
                               (size_t)M * d * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Self-spec greedy draft via cuBLAS: k chained proposals + bonus prefix save.
static void cublas_selfspec_draft_greedy(
    const ModelWeights& model, KVCache& kv,
    int first_token, int k, int start_seq_len, int n_prefix_layers,
    float* g_logits_scratch, float* d_saved_hiddens, int* d_draft_tokens,
    int* d_tok_buf, InferenceEngine& eng)
{
    int cur = first_token;
    for (int i = 0; i < k; i++) {
        CUDA_CHECK(cudaMemcpy(d_tok_buf, &cur, sizeof(int),
                              cudaMemcpyHostToDevice));
        cublas_prefix_save(model, kv, d_tok_buf, 1, start_seq_len + i,
                           n_prefix_layers,
                           d_saved_hiddens + (size_t)i * model.cfg.d_model,
                           eng);
        cublas_forward_from_hidden(
            model, kv,
            d_saved_hiddens + (size_t)i * model.cfg.d_model,
            1, start_seq_len + i, n_prefix_layers,
            g_logits_scratch, d_draft_tokens + i, eng);
        CUDA_CHECK(cudaMemcpy(&cur, d_draft_tokens + i, sizeof(int),
                              cudaMemcpyDeviceToHost));
    }
    if (k > 0) {
        CUDA_CHECK(cudaMemcpy(d_tok_buf, &cur, sizeof(int),
                              cudaMemcpyHostToDevice));
        cublas_prefix_save(model, kv, d_tok_buf, 1, start_seq_len + k,
                           n_prefix_layers,
                           d_saved_hiddens + (size_t)k * model.cfg.d_model,
                           eng);
    }
}

// Self-spec cuBLAS verify: starts from saved hidden states, runs layers
// [layer_start, n_layers), then output head on all M tokens.
static void cublas_forward_from_hidden(
    const ModelWeights& model, KVCache& kv,
    const float* g_saved_hiddens, int M,
    int seq_base, int layer_start,
    float* g_logits, int* d_out_tokens,
    InferenceEngine& eng) {
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int nh  = model.cfg.n_heads;
    const int nkv = model.cfg.n_kv_heads;
    const int dph = d / nh;
    const int kd  = nkv * dph;
    const int V   = model.cfg.vocab_size;

    cudaStream_t stream = eng.streams[0];
    GemmBackend gb;
    gb.init(eng.cublas, stream);

    cublas_ensure_scratch(eng, model.cfg, M);

    float* d_buf = eng.d_cublas_buf;
    half*  d_x16 = eng.d_cublas_x16;

    float* g_hidden = d_buf;
    float* g_normed = g_hidden + (size_t)M * d;
    float* g_Q      = g_normed + (size_t)M * d;
    float* g_K      = g_Q      + (size_t)M * d;
    float* g_V      = g_K      + (size_t)M * kd;
    float* g_attn   = g_V      + (size_t)M * kd;
    float* g_gate   = g_attn   + (size_t)M * d;
    float* g_up     = g_gate   + (size_t)M * dff;

    size_t rms_smem  = ((size_t)d + WARP_SIZE) * sizeof(float);
    int bk = 256;

    auto cvt = [&](const float* s, half* dst, int n) {
        f32_to_f16_kernel<<<(n+bk-1)/bk, bk, 0, stream>>>(s, dst, n);
    };

    // Load saved hidden states
    CUDA_CHECK(cudaMemcpyAsync(g_hidden, g_saved_hiddens,
                               (size_t)M * d * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    cublas_run_layer_range(model, kv, M, seq_base, layer_start,
                           model.cfg.n_layers,
                           g_hidden, g_normed, g_Q, g_K, g_V, g_attn,
                           g_gate, g_up, d_x16, gb, stream, bk);

    // Output on all M tokens
    cublas_rmsnorm_kernel<<<M, BLOCK_THREADS, rms_smem, stream>>>(
        g_hidden, model.rms_final_weight, g_normed, d, M);
    cvt(g_normed, d_x16, M * d);
    gemm_backend_gemm(gb, M, V, d, d_x16, model.output_proj, g_logits);

    size_t arg_smem = (size_t)BLOCK_THREADS * (sizeof(float) + sizeof(int));
    for (int m = 0; m < M; m++) {
        cublas_argmax_kernel<<<1, BLOCK_THREADS, arg_smem, stream>>>(
            g_logits + (size_t)m * V, V, d_out_tokens + m);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

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

    bool use_cublas = (eng != nullptr && eng->gemm_backend == GEMM_BACKEND_CUBLAS);
    bool use_coop   = (eng != nullptr && eng->coop_supported) && !use_cublas;

    // Dynamic smem and logits buffer sized for the target model
    size_t smem_bytes = compute_smem_bytes(target_model.cfg);
    if (!use_coop && !use_cublas)
        cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, smem_bytes);

    float* g_logits;
    CUDA_CHECK(cudaMalloc(&g_logits,
                          (size_t)target_model.cfg.vocab_size * sizeof(float)));

    int* d_next_token;
    int* d_output;
    int* d_tok_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, MAX_SEQ_LEN * sizeof(int)));
    if (use_cublas)
        CUDA_CHECK(cudaMalloc(&d_tok_buf, sizeof(int)));

    kv_cache_reset(target_kv);
    int seq_len = 0;

    // Prefill: cuBLAS path batches all prompt tokens through tensor-core GEMMs;
    // legacy path processes one token at a time.
    if (use_cublas && prompt_len > 0) {
        int* d_prompt;
        CUDA_CHECK(cudaMalloc(&d_prompt, prompt_len * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt, prompt_len * sizeof(int),
                              cudaMemcpyHostToDevice));

        cublas_forward(target_model, target_kv, d_prompt, prompt_len,
                       0, g_logits, d_next_token, *eng);
        seq_len = prompt_len;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, seq_len);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_prompt);
    } else {
        for (int i = 0; i < prompt_len; i++) {
            if (use_coop) {
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

        if (use_cublas) {
            CUDA_CHECK(cudaMemcpy(d_tok_buf, &current_token, sizeof(int),
                                  cudaMemcpyHostToDevice));
            cublas_forward(target_model, target_kv, d_tok_buf, 1,
                           seq_len, g_logits, d_next_token, *eng);
        } else if (use_coop) {
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
    if (d_tok_buf) cudaFree(d_tok_buf);
}

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

    // cuBLAS when requested (baseline uses multikernel_baseline separately).
    bool use_cublas = (eng != nullptr && eng->gemm_backend == GEMM_BACKEND_CUBLAS);
    bool use_coop_draft  = (eng != nullptr && eng->coop_supported) && !use_cublas;
    bool use_coop_target = (eng != nullptr && eng->coop_supported) && !use_cublas;
    bool use_coop_verify = (eng != nullptr && eng->coop_supported) && !use_cublas;

    // Each model gets its own smem budget and logits buffer
    size_t draft_smem  = compute_smem_bytes(draft_model.cfg);
    size_t target_smem = compute_smem_bytes(target_model.cfg);
    size_t max_decode_smem = draft_smem > target_smem ? draft_smem : target_smem;

    if (!use_cublas) {
        if (!use_coop_draft && !use_coop_target)
            cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, max_decode_smem);
        else if (!use_coop_draft)
            cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, draft_smem);
        else if (!use_coop_target)
            cuda_configure_kernel_dynamic_smem(single_token_decode_kernel, target_smem);
    }

    float* g_logits_draft;
    float* g_logits_target;
    CUDA_CHECK(cudaMalloc(&g_logits_draft,
                          (size_t)draft_model.cfg.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_logits_target,
                          (size_t)target_model.cfg.vocab_size * sizeof(float)));

    int* d_next;
    int* d_output;
    int* d_tok_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, MAX_SEQ_LEN * sizeof(int)));
    if (use_cublas)
        CUDA_CHECK(cudaMalloc(&d_tok_buf, sizeof(int)));

    kv_cache_reset(draft_kv);
    if (!params.self_speculative)
        kv_cache_reset(target_kv);

    int draft_seq  = 0;
    int target_seq = 0;

    // ---- Prefill ----
    // Self-spec uses one shared cache: target full prefill only (draft layers are
    // a prefix of the same KV entries written during verify each round).
    if (params.self_speculative) {
        if (use_cublas && prompt_len > 0) {
            int* d_prompt;
            CUDA_CHECK(cudaMalloc(&d_prompt, prompt_len * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt, prompt_len * sizeof(int),
                                  cudaMemcpyHostToDevice));
            cublas_forward(target_model, target_kv, d_prompt, prompt_len,
                           0, g_logits_target, d_next, *eng);
            target_seq = prompt_len;
            set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaFree(d_prompt);
        } else if (!use_cublas) {
            cudaStream_t stream_t = (eng != nullptr) ? eng->streams[1] : nullptr;
            const bool use_coop_tgt = (eng != nullptr && eng->coop_supported);
            for (int i = 0; i < prompt_len; i++) {
                if (use_coop_tgt) {
                    launch_cooperative_decode_step(
                        target_model, target_kv, h_prompt[i], target_seq,
                        eng->d_coop_hidden, eng->d_coop_scratch,
                        g_logits_target, d_next, eng->max_coop_blocks, stream_t);
                } else {
                    single_token_decode_kernel<<<1, BLOCK_THREADS, target_smem, stream_t>>>(
                        target_model, target_kv, h_prompt[i], target_seq,
                        g_logits_target, d_next);
                }
                target_seq++;
                set_seq_len_kernel<<<1, 1, 0, stream_t>>>(target_kv.seq_len, target_seq);
            }
            if (eng != nullptr)
                CUDA_CHECK(cudaStreamSynchronize(stream_t));
        }
        draft_seq = target_seq;
    } else if (use_cublas && prompt_len > 0) {
        // Sequential cuBLAS prefill: draft then target (both use stream[0]).
        int* d_prompt;
        CUDA_CHECK(cudaMalloc(&d_prompt, prompt_len * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt, prompt_len * sizeof(int),
                              cudaMemcpyHostToDevice));

        cublas_forward(draft_model, draft_kv, d_prompt, prompt_len,
                       0, g_logits_draft, d_next, *eng);
        draft_seq = prompt_len;
        set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);

        cublas_forward(target_model, target_kv, d_prompt, prompt_len,
                       0, g_logits_target, d_next, *eng);
        target_seq = prompt_len;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_prompt);
    } else {
        // ---- Prefill: draft (stream[0]) and target (stream[1]) run in parallel ----
        cudaStream_t stream_d = (eng != nullptr) ? eng->streams[0] : nullptr;
        cudaStream_t stream_t = (eng != nullptr) ? eng->streams[1] : nullptr;

        for (int i = 0; i < prompt_len; i++) {
        // Draft prefill on stream_d; uses d_coop_hidden2
        // (separate buffer from target to avoid aliasing).
        if (use_coop_draft) {
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
        if (use_coop_target) {
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

    // Stochastic speculative decoding (multi-kernel only; distribution-level
    // acceptance + adjusted rejection sampling vs greedy token identity).
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

        cuda_configure_kernel_dynamic_smem(stochastic_draft_forward_sample_dptr_kernel,
                                             draft_smem);
        if (params.self_speculative)
            cuda_configure_kernel_dynamic_smem(
                stochastic_draft_forward_sample_save_dptr_kernel, draft_smem);
        cuda_configure_kernel_dynamic_smem(single_token_forward_logits_kernel,
                                           max_decode_smem);
        cuda_configure_kernel_dynamic_smem(target_fwd_prob_and_accept_kernel,
                                           target_smem);
        if (params.self_speculative)
            cuda_configure_kernel_dynamic_smem(
                target_fwd_prob_and_accept_selfspec_kernel, target_smem);
        cuda_configure_kernel_dynamic_smem(softmax_sample_temperature_kernel,
                                           0);
        cuda_configure_kernel_dynamic_smem(corrected_sample_adjusted_logits_kernel,
                                             0);
        cuda_configure_kernel_dynamic_smem(rng_init_kernel, 0);

        curandState* d_rng = nullptr;
        float*       d_corr_work = nullptr;
        int*         d_accept = nullptr;

        CUDA_CHECK(cudaMalloc(&d_rng, sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_accept, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_corr_work,
                              (size_t)V_vocab * 3 * sizeof(float)));

        rng_init_kernel<<<1, 1>>>(d_rng,
                                 (unsigned long long)params.stochastic_rng_seed);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_q_probs = new float[(size_t)k];
        int*   h_accept_flags = new int[(size_t)k];   // batched accept flags (host)
        float  draft_temp_dyn = params.draft_temperature;
        float  ewma_accept    = -1.f;

        // Device-side buffers for chained draft sampling — avoids k CPU roundtrips
        // per speculation round by keeping tokens on device until all k are ready.
        int*   d_draft_tokens_dev = nullptr;
        float* d_q_probs_dev      = nullptr;
        int*   d_ctx_seed         = nullptr;   // bootstrap: device copy of last_token
        float* d_selfspec_hiddens_local = nullptr;
        float* d_selfspec_hiddens = (eng != nullptr) ? eng->d_selfspec_hiddens : nullptr;
        if (params.self_speculative && d_selfspec_hiddens == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_selfspec_hiddens_local,
                                  (size_t)(k + 1) * target_model.cfg.d_model *
                                      sizeof(float)));
            d_selfspec_hiddens = d_selfspec_hiddens_local;
        }
        CUDA_CHECK(cudaMalloc(&d_draft_tokens_dev, (size_t)k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_q_probs_dev,      (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ctx_seed,          sizeof(int)));

        // cuBLAS self-spec (stochastic): batched suffix-verify logits for all k+1
        // tokens in one pass, plus an (ignored) argmax scratch for the shared helper.
        int    dim_t = target_model.cfg.d_model;
        float* g_logits_verify_batch = nullptr;
        int*   d_batch_verify_out    = nullptr;
        if (use_cublas && params.self_speculative) {
            CUDA_CHECK(cudaMalloc(&g_logits_verify_batch,
                                  (size_t)(k + 1) * V_vocab * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_batch_verify_out, (size_t)(k + 1) * sizeof(int)));
        }

        while (generated < max_new) {
            iterations++;
            int remaining = max_new - generated;
            int current_k = (k < remaining) ? k : remaining;

            int draft_seq_save  = draft_seq;
            int target_seq_save = target_seq;
            total_proposed += current_k;

            // ---- Draft phase ----
            CUDA_CHECK(cudaMemcpy(d_ctx_seed, &last_token, sizeof(int),
                                  cudaMemcpyHostToDevice));
            const int* d_ctx_ptr = d_ctx_seed;

            if (use_cublas && params.self_speculative) {
                // Self-spec draft on cuBLAS: run the first n_draft_layers (prefix)
                // and save the hidden, then finish the remaining layers to get logits,
                // then sample stochastically. Mirrors the greedy cuBLAS draft but
                // samples (with q-mass) instead of taking the argmax.
                int cur = last_token;
                for (int di = 0; di < current_k; di++) {
                    CUDA_CHECK(cudaMemcpy(d_ctx_seed, &cur, sizeof(int),
                                          cudaMemcpyHostToDevice));
                    cublas_prefix_save(target_model, draft_kv, d_ctx_seed, 1,
                                       draft_seq, params.n_draft_layers,
                                       d_selfspec_hiddens + (size_t)di * dim_t, *eng);
                    cublas_forward_from_hidden(
                        target_model, draft_kv,
                        d_selfspec_hiddens + (size_t)di * dim_t, 1,
                        draft_seq, params.n_draft_layers,
                        g_logits_draft, d_next, *eng);
                    sample_from_logits_q_kernel<<<1, BLOCK_THREADS>>>(
                        g_logits_draft, V_vocab, draft_temp_dyn, d_rng,
                        d_draft_tokens_dev + di, d_q_probs_dev + di);
                    CUDA_CHECK(cudaMemcpy(&cur, d_draft_tokens_dev + di,
                                          sizeof(int), cudaMemcpyDeviceToHost));
                    draft_seq++;
                }
                // Bonus prefix hidden (index current_k) for the batched verify.
                if (current_k > 0) {
                    CUDA_CHECK(cudaMemcpy(d_ctx_seed, &cur, sizeof(int),
                                          cudaMemcpyHostToDevice));
                    cublas_prefix_save(target_model, draft_kv, d_ctx_seed, 1,
                                       draft_seq, params.n_draft_layers,
                                       d_selfspec_hiddens +
                                           (size_t)current_k * dim_t, *eng);
                }
            } else if (use_cublas && !params.self_speculative) {
                for (int di = 0; di < current_k; di++) {
                    cublas_forward(draft_model, draft_kv, d_ctx_ptr, 1,
                                   draft_seq, g_logits_draft,
                                   d_draft_tokens_dev + di, *eng);
                    sample_from_logits_q_kernel<<<1, BLOCK_THREADS>>>(
                        g_logits_draft, V_vocab, draft_temp_dyn, d_rng,
                        d_draft_tokens_dev + di, d_q_probs_dev + di);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    d_ctx_ptr = d_draft_tokens_dev + di;
                    draft_seq++;
                    set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
                }
            } else {
                for (int di = 0; di < current_k; di++) {
                    if (params.self_speculative) {
                        stochastic_draft_forward_sample_save_dptr_kernel
                            <<<1, BLOCK_THREADS, draft_smem>>>(
                                target_model, draft_kv, d_ctx_ptr, draft_seq,
                                params.n_draft_layers,
                                draft_temp_dyn, d_rng, g_logits_draft,
                                d_selfspec_hiddens +
                                    (size_t)di * target_model.cfg.d_model,
                                d_draft_tokens_dev + di, d_q_probs_dev + di);
                    } else {
                        stochastic_draft_forward_sample_dptr_kernel
                            <<<1, BLOCK_THREADS, draft_smem>>>(
                                draft_model, draft_kv, d_ctx_ptr, draft_seq,
                                draft_temp_dyn, d_rng, g_logits_draft,
                                d_draft_tokens_dev + di, d_q_probs_dev + di);
                    }
                    d_ctx_ptr = d_draft_tokens_dev + di;
                    draft_seq++;
                    set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
                }
            }
            if (!use_cublas && params.self_speculative && current_k > 0) {
                selfspec_bonus_prefix_save_dptr_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                    target_model, draft_kv,
                    d_draft_tokens_dev + (current_k - 1), draft_seq,
                    params.n_draft_layers,
                    d_selfspec_hiddens +
                        (size_t)current_k * target_model.cfg.d_model);
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

            if (use_cublas && params.self_speculative) {
                // Batched cuBLAS self-spec verify: one suffix pass over all k+1
                // tokens from their saved prefix hiddens, then stochastic accept /
                // adjusted-rejection sampling on the precomputed logits. KV commit
                // and rollback follow the shared (greedy-style) logic below.
                cublas_forward_from_hidden(
                    target_model, target_kv, d_selfspec_hiddens, current_k + 1,
                    target_seq_save, params.n_draft_layers,
                    g_logits_verify_batch, d_batch_verify_out, *eng);

                // One batched pass computes every accept flag (no per-token sync).
                size_t accept_smem = (size_t)BLOCK_THREADS * sizeof(float);
                batch_stochastic_accept_kernel<<<1, BLOCK_THREADS, accept_smem>>>(
                    g_logits_verify_batch, V_vocab,
                    d_draft_tokens_dev, d_q_probs_dev, current_k,
                    d_rng, d_batch_verify_out);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_accept_flags, d_batch_verify_out,
                                      (size_t)current_k * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                for (int vi = 0; vi < current_k; vi++) {
                    if (h_accept_flags[vi]) { n_accept_round++; continue; }

                    // Reject: recover draft dist q at vi from its saved prefix hidden,
                    // then sample the correction norm(max(0, p - q)).
                    broke_early = true;
                    cublas_forward_from_hidden(
                        target_model, draft_kv,
                        d_selfspec_hiddens + (size_t)vi * dim_t, 1,
                        draft_seq_save + vi, params.n_draft_layers,
                        g_logits_draft, d_next, *eng);
                    corrected_sample_adjusted_logits_kernel<<<1, 1>>>(
                        g_logits_verify_batch + (size_t)vi * V_vocab,
                        g_logits_draft, V_vocab, d_rng, d_next, d_corr_work,
                        1.f, draft_temp_dyn);
                    CUDA_CHECK(cudaMemcpy(&bonus, d_next, sizeof(int),
                                          cudaMemcpyDeviceToHost));
                    break;
                }
                if (!broke_early) {
                    // All k accepted: bonus is sampled from the last verify position.
                    softmax_sample_temperature_kernel<<<1, BLOCK_THREADS>>>(
                        g_logits_verify_batch + (size_t)current_k * V_vocab,
                        V_vocab, 1.f, d_rng, d_next);
                    CUDA_CHECK(cudaMemcpy(&bonus, d_next, sizeof(int),
                                          cudaMemcpyDeviceToHost));
                }
            } else
            for (int vi = 0; vi < current_k; vi++) {
                int inp = (vi == 0) ? last_token : h_draft_tokens[vi - 1];

                if (use_cublas && !params.self_speculative) {
                    CUDA_CHECK(cudaMemcpy(d_tok_buf, &inp, sizeof(int),
                                          cudaMemcpyHostToDevice));
                    cublas_forward(target_model, target_kv, d_tok_buf, 1,
                                   target_roll, g_logits_target, d_next, *eng);
                    size_t accept_smem = (size_t)BLOCK_THREADS * sizeof(float);
                    accept_from_logits_kernel<<<1, BLOCK_THREADS, accept_smem>>>(
                        g_logits_target, V_vocab,
                        h_draft_tokens[vi], h_q_probs[vi],
                        d_rng, d_accept);
                } else if (params.self_speculative) {
                    target_fwd_prob_and_accept_selfspec_kernel
                        <<<1, BLOCK_THREADS, target_smem>>>(
                            target_model, target_kv, inp, target_roll,
                            params.n_draft_layers,
                            d_selfspec_hiddens +
                                (size_t)vi * target_model.cfg.d_model,
                            1,
                            h_draft_tokens[vi], h_q_probs[vi],
                            g_logits_target, d_rng, d_accept);
                } else {
                    target_fwd_prob_and_accept_kernel
                        <<<1, BLOCK_THREADS, target_smem>>>(
                            target_model, target_kv, inp, target_roll,
                            h_draft_tokens[vi], h_q_probs[vi],
                            g_logits_target, d_rng, d_accept);
                }
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
                if (params.self_speculative) {
                    set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len,
                                                 target_seq_save + vi);
                } else {
                    set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len,
                                                 draft_seq_save + vi);
                    set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len,
                                                 target_seq_save + vi);
                }

                if (use_cublas && !params.self_speculative) {
                    CUDA_CHECK(cudaMemcpy(d_tok_buf, &inp, sizeof(int),
                                          cudaMemcpyHostToDevice));
                    cublas_forward(draft_model, draft_kv, d_tok_buf, 1,
                                   draft_seq_save + vi, g_logits_draft,
                                   d_next, *eng);
                } else {
                    single_token_forward_logits_kernel
                        <<<1, BLOCK_THREADS, draft_smem>>>(
                            draft_model, draft_kv, inp, draft_seq_save + vi,
                            g_logits_draft);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                corrected_sample_adjusted_logits_kernel<<<1, 1>>>(
                    g_logits_target, g_logits_draft, V_vocab, d_rng, d_next,
                    d_corr_work, 1.f, draft_temp_dyn);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&bonus, d_next, sizeof(int),
                                      cudaMemcpyDeviceToHost));

                if (use_cublas && !params.self_speculative) {
                    CUDA_CHECK(cudaMemcpy(d_tok_buf, &bonus, sizeof(int),
                                          cudaMemcpyHostToDevice));
                    cublas_forward(target_model, target_kv, d_tok_buf, 1,
                                   target_seq_save + n_accept_round,
                                   g_logits_target, d_next, *eng);
                } else {
                    single_token_forward_logits_kernel
                        <<<1, BLOCK_THREADS, target_smem>>>(
                            target_model, target_kv, bonus,
                            target_seq_save + n_accept_round, g_logits_target);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                break;
            }

            if (!broke_early) {
                if (use_cublas && !params.self_speculative) {
                    CUDA_CHECK(cudaMemcpy(d_tok_buf, &h_draft_tokens[current_k - 1],
                                          sizeof(int), cudaMemcpyHostToDevice));
                    cublas_forward(target_model, target_kv, d_tok_buf, 1,
                                   target_roll, g_logits_target, d_next, *eng);
                } else {
                    single_token_forward_logits_kernel
                        <<<1, BLOCK_THREADS, target_smem>>>(
                            target_model, target_kv, h_draft_tokens[current_k - 1],
                            target_roll, g_logits_target);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                softmax_sample_temperature_kernel<<<1, BLOCK_THREADS>>>(
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

            if (params.self_speculative) {
                draft_seq = target_seq;
            } else {
                draft_seq = draft_seq_save + n_accept_round;
                set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);

                if (n_accept_round == current_k) {
                    if (use_cublas) {
                        CUDA_CHECK(cudaMemcpy(d_tok_buf, &h_draft_tokens[current_k - 1],
                                              sizeof(int), cudaMemcpyHostToDevice));
                        cublas_forward(draft_model, draft_kv, d_tok_buf, 1,
                                       draft_seq, g_logits_draft, d_next, *eng);
                    } else {
                        cuda_configure_kernel_dynamic_smem(single_token_decode_kernel,
                                                           draft_smem);
                        single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                            draft_model, draft_kv, h_draft_tokens[current_k - 1],
                            draft_seq, g_logits_draft, d_next);
                    }
                    draft_seq++;
                    set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
                }

                if (use_cublas) {
                    CUDA_CHECK(cudaMemcpy(d_tok_buf, &last_token, sizeof(int),
                                          cudaMemcpyHostToDevice));
                    cublas_forward(draft_model, draft_kv, d_tok_buf, 1,
                                   draft_seq, g_logits_draft, d_next, *eng);
                } else {
                    cuda_configure_kernel_dynamic_smem(single_token_decode_kernel,
                                                       draft_smem);
                    single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                        draft_model, draft_kv, last_token, draft_seq, g_logits_draft,
                        d_next);
                }
                draft_seq++;
                set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            }
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
        delete[] h_accept_flags;
        cudaFree(d_ctx_seed);
        cudaFree(d_q_probs_dev);
        cudaFree(d_draft_tokens_dev);
        if (d_selfspec_hiddens_local) cudaFree(d_selfspec_hiddens_local);
        if (g_logits_verify_batch) cudaFree(g_logits_verify_batch);
        if (d_batch_verify_out)    cudaFree(d_batch_verify_out);
        cudaFree(d_corr_work);
        cudaFree(d_accept);
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
        if (d_tok_buf) cudaFree(d_tok_buf);
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

    float* d_selfspec_hiddens_local = nullptr;
    float* d_selfspec_hiddens = (eng != nullptr) ? eng->d_selfspec_hiddens : nullptr;
    if (params.self_speculative && d_selfspec_hiddens == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_selfspec_hiddens_local,
                              (size_t)(k + 1) * target_model.cfg.d_model *
                                  sizeof(float)));
        d_selfspec_hiddens = d_selfspec_hiddens_local;
    }

    float* g_batch_hidden = nullptr;
    float* g_batch_work   = nullptr;

    // Configure batch kernels once before the loop
    cuda_configure_kernel_dynamic_smem(batch_draft_greedy_kernel, draft_smem);
    if (!use_coop_verify)
        cuda_configure_kernel_dynamic_smem(batch_target_verify_kernel, target_smem);
    if (use_coop_verify)
        cuda_configure_kernel_dynamic_smem(batch_target_verify_coop_kernel, target_smem);
    if (params.self_speculative && !use_cublas) {
        cuda_configure_kernel_dynamic_smem(batch_draft_selfspec_kernel, draft_smem);
        cuda_configure_kernel_dynamic_smem(batch_target_verify_selfspec_kernel,
                                           target_smem);
        if (use_coop_verify)
            cuda_configure_kernel_dynamic_smem(batch_target_verify_coop_selfspec_kernel,
                                               target_smem);
        int max_B = k + 1;
        int d_t   = target_model.cfg.d_model;
        CUDA_CHECK(cudaMalloc(&g_batch_hidden,
                              (size_t)max_B * d_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_work,
                              (size_t)(7 * d_t + MLP_FF_TILE) * max_B * sizeof(float)));
    } else if (params.self_speculative) {
        cuda_configure_kernel_dynamic_smem(batch_draft_selfspec_kernel, draft_smem);
    }

    // Batched verify logits (cooperative, self-spec, or cuBLAS verify)
    float* g_logits_verify_batch = nullptr;
    if (use_coop_verify || params.self_speculative || use_cublas)
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

        // ---- Draft phase ----
        if (use_cublas && params.self_speculative) {
            cublas_selfspec_draft_greedy(
                target_model, draft_kv, last_token, current_k, draft_seq,
                params.n_draft_layers, g_logits_draft, d_selfspec_hiddens,
                d_batch_draft, d_tok_buf, *eng);
            draft_seq += current_k;
        } else if (use_cublas && !params.self_speculative) {
            // Chain k single-token cuBLAS forwards on the draft model.
            CUDA_CHECK(cudaMemcpy(d_tok_buf, &last_token, sizeof(int),
                                  cudaMemcpyHostToDevice));
            const int* d_ctx = d_tok_buf;
            for (int di = 0; di < current_k; di++) {
                cublas_forward(draft_model, draft_kv, d_ctx, 1,
                               draft_seq, g_logits_draft,
                               d_batch_draft + di, *eng);
                d_ctx = d_batch_draft + di;
                draft_seq++;
                set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            }
        } else if (params.self_speculative) {
            batch_draft_selfspec_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                target_model, draft_kv, last_token, current_k, draft_seq,
                params.n_draft_layers,
                g_logits_draft, d_selfspec_hiddens, d_batch_draft);
            draft_seq += current_k;
            // Shared KV: leave device seq_len at target_seq until verify completes.
        } else {
            batch_draft_greedy_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                draft_model, draft_kv, last_token, current_k, draft_seq,
                g_logits_draft, d_batch_draft);
            draft_seq += current_k;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
        }
        // D2H memcpy provides implicit sync
        CUDA_CHECK(cudaMemcpy(h_draft_tokens, d_batch_draft,
                              current_k * sizeof(int), cudaMemcpyDeviceToHost));

        // ---- Verify phase ----
        h_verify_tokens[0] = last_token;
        for (int i = 0; i < current_k; i++) h_verify_tokens[i + 1] = h_draft_tokens[i];
        int batch_size = current_k + 1;
        CUDA_CHECK(cudaMemcpy(d_batch_verify_in, h_verify_tokens,
                              batch_size * sizeof(int), cudaMemcpyHostToDevice));

        if (use_cublas && params.self_speculative) {
            cublas_forward_from_hidden(
                target_model, target_kv, d_selfspec_hiddens, batch_size,
                target_seq, params.n_draft_layers,
                g_logits_verify_batch, d_batch_verify_out, *eng);
        } else if (use_cublas) {
            cublas_forward(target_model, target_kv, d_batch_verify_in, batch_size,
                           target_seq, g_logits_verify_batch, d_batch_verify_out,
                           *eng, /*all_token_logits=*/true);
        } else if (params.self_speculative) {
            int n_dl = params.n_draft_layers;
            if (use_coop_verify) {
                void* coop_args[] = {
                    (void*)&target_model,
                    (void*)&target_kv,
                    (void*)&d_batch_verify_in,
                    (void*)&batch_size,
                    (void*)&target_seq,
                    (void*)&n_dl,
                    (void*)&d_selfspec_hiddens,
                    (void*)&g_logits_verify_batch,
                    (void*)&d_batch_verify_out
                };
                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)batch_target_verify_coop_selfspec_kernel,
                    dim3(batch_size), dim3(BLOCK_THREADS),
                    coop_args, target_smem, nullptr));
                CUDA_CHECK(cudaDeviceSynchronize());
            } else {
                batch_target_verify_selfspec_kernel<<<1, BLOCK_THREADS, target_smem>>>(
                    target_model, target_kv, d_batch_verify_in, batch_size,
                    target_seq, n_dl, d_selfspec_hiddens,
                    g_batch_hidden, g_batch_work, g_logits_verify_batch,
                    d_batch_verify_out);
            }
        } else if (use_coop_verify) {
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

        // Rollback cache to the accepted prefix
        target_seq = target_seq_save + n_accepted + 1;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);

        if (params.self_speculative) {
            // Shared KV: verify wrote full-depth K/V at each position; no draft-only sync.
            draft_seq = target_seq;
        } else {
            draft_seq = draft_seq_save + n_accepted;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);

            // When ALL k drafts were accepted, draft[k-1] was only a prediction —
            // its K/V was never written.  One extra forward fills that slot.
            if (n_accepted == current_k) {
                if (use_cublas) {
                    CUDA_CHECK(cudaMemcpy(d_tok_buf, &h_draft_tokens[current_k - 1],
                                          sizeof(int), cudaMemcpyHostToDevice));
                    cublas_forward(draft_model, draft_kv, d_tok_buf, 1,
                                   draft_seq, g_logits_draft, d_next, *eng);
                } else if (use_coop_draft) {
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
            } else {
                // Partial accept: write bonus token's K/V at the rollback position.
                if (use_cublas) {
                    CUDA_CHECK(cudaMemcpy(d_tok_buf, &last_token, sizeof(int),
                                          cudaMemcpyHostToDevice));
                    cublas_forward(draft_model, draft_kv, d_tok_buf, 1,
                                   draft_seq, g_logits_draft, d_next, *eng);
                } else if (use_coop_draft) {
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
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        if ((params.eos_token >= 0 && bonus == params.eos_token) || generated >= max_new) break;
    }

    write_result_kernel<<<1, BLOCK_THREADS>>>(
        d_result, d_output, generated,
        total_proposed, total_accepted, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (g_logits_verify_batch) cudaFree(g_logits_verify_batch);
    if (g_batch_hidden) cudaFree(g_batch_hidden);
    if (g_batch_work) cudaFree(g_batch_work);
    if (d_selfspec_hiddens_local) cudaFree(d_selfspec_hiddens_local);
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
    if (d_tok_buf) cudaFree(d_tok_buf);
}

// launch_cooperative_decode_step  (definition; forward-declared above)
//
// Launches cooperative_decode_kernel with a grid sized to cover all vocab
// columns in parallel, capped to the device's cooperative-launch limit.
// Falls back gracefully to 1 block if max_coop_blocks == 0.
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

    // Query the true per-SM occupancy AFTER setting the smem attribute so the
    // calculator sees the real shared-memory footprint.  On consumer GPUs
    // (RTX 3050: 48 KB/SM, sm_86) large models can limit blocks_per_sm to 1,
    // giving a cooperative-launch limit of only n_sm × 1 blocks.  Exceeding
    // that limit causes cudaErrorInvalidConfiguration at launch.
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        (void*)cooperative_decode_kernel,
        BLOCK_THREADS, smem);
    if (blocks_per_sm < 1) blocks_per_sm = 1;

    int dev = 0;  cudaGetDevice(&dev);
    int sm_count = 1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int hard_limit = sm_count * blocks_per_sm;   // max blocks for coop launch

    int n_blocks_needed = (V + GEMV_COL_TILE - 1) / GEMV_COL_TILE;
    int n_blocks = n_blocks_needed;
    if (n_blocks > max_coop_blocks) n_blocks = max_coop_blocks;
    if (n_blocks > hard_limit)      n_blocks = hard_limit;
    if (n_blocks < 1)               n_blocks = 1;

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

//  PERSISTENT MEGAKERNEL PATH
//
//  A single kernel runs the entire generation loop on the GPU.
//  g_logits is a global-memory scratch buffer for intermediate logits.
//
//  TODO: For multi-block / larger models, extend with cooperative-group
//        grid-level barriers instead of __syncthreads().

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

// ---- Megakernel: Baseline, persistent + cooperative (multi-SM) ----
//
// The whole generation loop lives in one cooperative launch: block 0 runs the
// transformer body, then every block splits the output-projection GEMV across
// SMs. g_ctrl holds the loop state in global memory so all blocks agree after
// each grid.sync(): [0]=cur_token [1]=seq_len [2]=generated [3]=done.
//
// Every block executes the same barrier sequence every iteration (the done
// flag gates work, never a grid.sync) so the grid can never deadlock.
// Cooperative (multi-SM) single-token transformer layer. Every block streams a
// column-stripe of each weight matrix (Wq/Wk/Wv/Wo/gate/up/down) via the
// device_matvec_partial helpers, so the bandwidth-dominant projections span all
// SMs instead of one block. RMSNorm is recomputed locally per block (cheap, O(d),
// avoids a global buffer + barrier); RoPE, KV append and attention (KV-bound,
// small at low context) run on block 0. grid.sync() separates dependent phases.
//
// Global scratch (all [.] resident across the layer): g_hidden [d] residual,
// g_q [d], g_k/g_v [kd], g_attn [d], g_gate [d_ff].  smem holds this block's
// local normed vector + reduction/attention scratch.
__device__ void coop_layer_forward(
        const ModelWeights& model, int l, KVCache& kv, int seq,
        float* g_hidden, float* g_q, float* g_k, float* g_v,
        float* g_attn, float* g_gate, float* smem,
        cg::grid_group& grid) {
    const LayerWeights& lw = model.layers[l];
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int nh  = model.cfg.n_heads;
    const int nkv = model.cfg.n_kv_heads;
    const int dph = d / nh;
    const int kd  = nkv * dph;
    const int nb  = (int)gridDim.x;
    const int bx  = (int)blockIdx.x;
    const int tid = threadIdx.x;

    float* s_normed  = smem;       // [d] this block's normed input
    float* s_scratch = smem + d;   // reduction / attention scratch

    // ---- attention RMSNorm (recomputed locally by every block) ----
    device_rmsnorm(g_hidden, lw.rms_attn_weight, s_normed, d, s_scratch);

    // ---- Q / K / V projections, output columns split across blocks ----
    for (int ct = bx; ct * GEMV_COL_TILE < d; ct += nb) {
        int cs = ct * GEMV_COL_TILE;
        int cc = (cs + GEMV_COL_TILE <= d) ? GEMV_COL_TILE : (d - cs);
        device_matvec_partial(s_normed, lw.Wq, g_q, d, d, cs, cc);
        if (lw.Wq_bias)
            for (int i = tid; i < cc; i += blockDim.x)
                g_q[cs + i] += __half2float(lw.Wq_bias[cs + i]);
    }
    for (int ct = bx; ct * GEMV_COL_TILE < kd; ct += nb) {
        int cs = ct * GEMV_COL_TILE;
        int cc = (cs + GEMV_COL_TILE <= kd) ? GEMV_COL_TILE : (kd - cs);
        device_matvec_partial(s_normed, lw.Wk, g_k, d, kd, cs, cc);
        device_matvec_partial(s_normed, lw.Wv, g_v, d, kd, cs, cc);
        for (int i = tid; i < cc; i += blockDim.x) {
            if (lw.Wk_bias) g_k[cs + i] += __half2float(lw.Wk_bias[cs + i]);
            if (lw.Wv_bias) g_v[cs + i] += __half2float(lw.Wv_bias[cs + i]);
        }
    }
    grid.sync();

    // ---- RoPE + KV append + attention on block 0 (KV-bound, cheap) ----
    if (bx == 0) {
        rope_apply_inplace(g_q, nh, dph, seq, model.cfg.rope_theta,
                           model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);
        rope_apply_inplace(g_k, nkv, dph, seq, model.cfg.rope_theta,
                           model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);
        kv_cache_append(kv, l, g_k, g_v, seq);
        for (int i = tid; i < d; i += blockDim.x) g_attn[i] = 0.f;
        __syncthreads();
        __shared__ float s_m, s_l, s_tm, s_al, s_iv;
        device_attention_dispatch(model.cfg.attention_type, g_q, g_attn, kv, l,
                                  nh, nkv, dph, seq + 1, s_scratch,
                                  s_m, s_l, s_tm, s_al, s_iv);
    }
    grid.sync();

    // ---- output projection: g_hidden += Wo(attn), split across blocks ----
    for (int ct = bx; ct * GEMV_COL_TILE < d; ct += nb) {
        int cs = ct * GEMV_COL_TILE;
        int cc = (cs + GEMV_COL_TILE <= d) ? GEMV_COL_TILE : (d - cs);
        device_matvec_partial_accum(g_attn, lw.Wo, g_hidden, d, d, cs, cc);
    }
    grid.sync();

    // ---- MLP RMSNorm (local) ----
    device_rmsnorm(g_hidden, lw.rms_mlp_weight, s_normed, d, s_scratch);

    // ---- gate/up + SwiGLU fused, d_ff split across blocks ----
    for (int ct = bx; ct * GEMV_COL_TILE < dff; ct += nb) {
        int cs = ct * GEMV_COL_TILE;
        int cc = (cs + GEMV_COL_TILE <= dff) ? GEMV_COL_TILE : (dff - cs);
        for (int oc = tid; oc < cc; oc += blockDim.x) {
            int c = cs + oc;
            float g = 0.f, u = 0.f;
            #pragma unroll 4
            for (int r = 0; r < d; r++) {
                float xr = s_normed[r];
                g += xr * __half2float(__ldg(&lw.W_gate[r * dff + c]));
                u += xr * __half2float(__ldg(&lw.W_up[r * dff + c]));
            }
            g_gate[c] = (g / (1.0f + expf(-g))) * u;   // silu(gate) * up
        }
    }
    grid.sync();

    // ---- down projection: g_hidden += W_down(swiglu), split across blocks ----
    for (int ct = bx; ct * GEMV_COL_TILE < d; ct += nb) {
        int cs = ct * GEMV_COL_TILE;
        int cc = (cs + GEMV_COL_TILE <= d) ? GEMV_COL_TILE : (d - cs);
        device_matvec_partial_accum(g_gate, lw.W_down, g_hidden, dff, d, cs, cc);
    }
    grid.sync();
}

__global__ void megakernel_baseline_coop_kernel(
        const ModelWeights* p_model, KVCache* p_kv,
        const int* prompt, int prompt_len, int max_new_tokens,
        float* g_hidden, float* g_scratch, float* g_logits,
        int* g_ctrl, GenerationResult* result, int eos_token,
        float* g_q, float* g_k, float* g_v, float* g_attn, float* g_gate) {
    const ModelWeights& model = *p_model;
    KVCache&            kv    = *p_kv;
    auto grid = cg::this_grid();

    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int d   = model.cfg.d_model;
    const int V   = model.cfg.vocab_size;

    if (blockIdx.x == 0 && tid == 0) {
        g_ctrl[0] = 0; g_ctrl[1] = 0; g_ctrl[2] = 0; g_ctrl[3] = 0;
    }
    grid.sync();

    int total_fwd = prompt_len + (max_new_tokens > 0 ? max_new_tokens - 1 : 0);

    for (int f = 0; f < total_fwd; f++) {
        int done = g_ctrl[3];
        int seq  = g_ctrl[1];

        // Embed the input token into the resident hidden vector (block 0).
        if (!done && blockIdx.x == 0) {
            int input_tok = (f < prompt_len) ? prompt[f] : g_ctrl[0];
            model_embed(model, input_tok, g_hidden);
        }
        grid.sync();

        // Per-layer forward, every projection streamed across all SMs.
        if (!done) {
            for (int l = 0; l < model.cfg.n_layers; l++)
                coop_layer_forward(model, l, kv, seq, g_hidden,
                                   g_q, g_k, g_v, g_attn, g_gate, smem, grid);
        }

        // Final RMSNorm (block 0) feeds the split vocab projection below.
        if (!done && blockIdx.x == 0)
            device_rmsnorm(g_hidden, model.rms_final_weight, g_scratch, d, smem + d);
        grid.sync();

        if (!done) {
            for (int ct = (int)blockIdx.x; ct * GEMV_COL_TILE < V; ct += (int)gridDim.x) {
                int cs = ct * GEMV_COL_TILE;
                int cc = (cs + GEMV_COL_TILE <= V) ? GEMV_COL_TILE : (V - cs);
                device_matvec_partial(g_scratch, model.output_proj, g_logits, d, V, cs, cc);
            }
        }
        grid.sync();

        if (!done && blockIdx.x == 0) {
            int next = global_argmax(g_logits, V, smem);
            if (tid == 0) {
                int out_idx = f - prompt_len + 1;
                if (out_idx >= 0 && out_idx < max_new_tokens) {
                    result->output_tokens[out_idx] = next;
                    g_ctrl[2] = out_idx + 1;
                    if ((eos_token >= 0 && next == eos_token) ||
                        out_idx + 1 >= max_new_tokens)
                        g_ctrl[3] = 1;
                }
                g_ctrl[0]   = next;
                g_ctrl[1]   = seq + 1;
                *kv.seq_len = seq + 1;
            }
        }
        grid.sync();
    }

    if (blockIdx.x == 0 && tid == 0) {
        result->n_generated     = g_ctrl[2];
        result->draft_proposed  = 0;
        result->draft_accepted  = 0;
        result->spec_iterations = 0;
    }
}

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
        float* g_selfspec_hiddens,
        GenerationResult* result, int eos_token, bool shared_kv) {
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

    // ---- Prefill ----
    if (shared_kv) {
        for (int i = 0; i < prompt_len; i++) {
            int next_t = model_forward(target_model, target_kv, prompt[i],
                                       s_target_seq, hidden, g_logits, smem);
            if (tid == 0) {
                s_last_token = next_t;
                s_target_seq++;
                s_draft_seq = s_target_seq;
                *target_kv.seq_len = s_target_seq;
            }
            __syncthreads();
        }
    } else {
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
            int next;
            if (shared_kv) {
                int n_prefix = draft_model.cfg.n_layers;
                model_forward_draft_logits(
                    target_model, draft_kv, s_draft_input, s_draft_seq,
                    n_prefix, hidden, g_logits,
                    g_selfspec_hiddens + (size_t)di * d, smem);
                __syncthreads();
                next = global_argmax(g_logits, target_model.cfg.vocab_size, smem);
            } else {
                next = model_forward(draft_model, draft_kv, s_draft_input,
                                     s_draft_seq, hidden, g_logits, smem);
            }
            if (tid == 0) {
                s_draft_tokens[di] = next;
                s_draft_input = next;
                s_draft_seq++;
                *draft_kv.seq_len = s_draft_seq;
            }
            __syncthreads();
        }

        if (shared_kv && current_k > 0) {
            model_forward_draft_prefix_save(
                target_model, target_kv, s_draft_tokens[current_k - 1],
                s_draft_seq, draft_model.cfg.n_layers,
                hidden, g_selfspec_hiddens + (size_t)current_k * d, smem);
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

            if (shared_kv) {
                model_batch_forward_selfspec_verify_logits(
                    target_model, target_kv,
                    s_verify_toks, s_target_seq_save,
                    current_k + 1,
                    draft_model.cfg.n_layers,
                    g_selfspec_hiddens,
                    g_batch_hidden, g_batch_work, g_batch_logits,
                    smem);
            } else {
            model_batch_forward_logits(
                target_model, target_kv,
                s_verify_toks, s_target_seq_save,
                current_k + 1,
                g_batch_hidden, g_batch_work, g_batch_logits,
                smem);
            }

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
            if (shared_kv) {
                s_draft_seq = s_target_seq;
            } else {
                s_draft_seq = s_draft_seq_save + n_acc;
                *draft_kv.seq_len = s_draft_seq;
            }
        }
        __syncthreads();

        if (!shared_kv) {
        // All-accepted: draft[k-1] K/V was never written; fill it now.
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
        float* g_selfspec_hiddens,
        float draft_temp_initial,
        int adaptive_enabled,
        float min_draft_temp,
        float max_draft_temp,
        float adapt_tgt_accept,
        float adapt_gain,
        float adapt_ewma_mix,
        unsigned long long rng_seed,
        GenerationResult* result,
        int eos_token,
        bool shared_kv) {

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
        if (!shared_kv) {
            model_forward(draft_model, draft_kv, prompt[pi],
                          s_draft_seq, hidden, logits_draft, scratch);
            if (tid == 0) {
                s_draft_seq++;
                *draft_kv.seq_len = s_draft_seq;
            }
            __syncthreads();
        }

        int tgt_id = model_forward(target_model, target_kv, prompt[pi],
                                   s_target_seq, hidden,
                                   logits_target, scratch);
        if (tid == 0) {
            s_last_token = tgt_id;
            s_target_seq++;
            if (shared_kv)
                s_draft_seq = s_target_seq;
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
            if (shared_kv) {
                int n_prefix = draft_model.cfg.n_layers;
                model_forward_draft_logits(target_model, draft_kv, s_cur_in,
                                           s_draft_seq, n_prefix,
                                           hidden, logits_draft,
                                           g_selfspec_hiddens + (size_t)di * dim_t,
                                           scratch);
            } else {
                model_forward_logits(draft_model, draft_kv, s_cur_in,
                                     s_draft_seq, hidden, logits_draft, scratch);
            }
            __syncthreads();

            logits_inplace_softmax_temp(logits_draft,
                                        V_vocab,
                                        s_dyn_dt,
                                        scratch);
            __syncthreads();

            int pick_w = device_sample_inverse_cdf(logits_draft, V_vocab, &s_rng);
            if (tid == 0) {
                s_draft_ids[di] = pick_w;
                s_q_probs[di]   = logits_draft[pick_w];
                s_cur_in        = pick_w;
                s_draft_seq++;
                *draft_kv.seq_len = s_draft_seq;
            }
            __syncthreads();
        }

        if (shared_kv && ck > 0) {
            model_forward_draft_prefix_save(
                target_model, target_kv, s_draft_ids[ck - 1],
                s_draft_seq, draft_model.cfg.n_layers,
                hidden, g_selfspec_hiddens + (size_t)ck * dim_t, scratch);
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

            if (shared_kv) {
                model_batch_forward_selfspec_verify_logits(
                    target_model, target_kv,
                    s_verify_toks, s_ts_save,
                    ck + 1,
                    draft_model.cfg.n_layers,
                    g_selfspec_hiddens,
                    g_batch_hidden, g_batch_work, g_batch_logits,
                    scratch);
            } else {
            model_batch_forward_logits(
                target_model, target_kv,
                s_verify_toks, s_ts_save,
                ck + 1,
                g_batch_hidden, g_batch_work, g_batch_logits,
                scratch);
            }

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
            if (shared_kv) {
                s_draft_seq = s_target_seq;
            } else {
                s_draft_seq = s_ds_save + s_n_accepted;
                *draft_kv.seq_len = s_draft_seq;
            }
        }

        __syncthreads();

        if (!shared_kv) {
        // All-accepted stochastic: fill missing K/V for draft[ck-1].
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

// Persistent cooperative baseline: whole decode loop in one multi-SM launch.
// Falls back to the single-block megakernel when the device lacks cooperative
// launch support.
void megakernel_baseline_coop(const ModelWeights& target_model,
                              KVCache& target_kv,
                              const int* prompt, int prompt_len,
                              GenerationResult* d_result,
                              const GenerationParams& params) {
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    int coop_attr = 0;
    cudaDeviceGetAttribute(&coop_attr, cudaDevAttrCooperativeLaunch, dev);
    if (!coop_attr) {
        megakernel_baseline(target_model, target_kv, prompt, prompt_len,
                            d_result, params);
        return;
    }

    const int d   = target_model.cfg.d_model;
    const int V   = target_model.cfg.vocab_size;
    const int kd  = target_model.cfg.n_kv_heads * (d / target_model.cfg.n_heads);
    const int dff = target_model.cfg.d_ff;

    kv_cache_reset(target_kv);
    size_t smem_bytes = compute_smem_bytes(target_model.cfg);
    cuda_configure_kernel_dynamic_smem(megakernel_baseline_coop_kernel, smem_bytes);

    float *g_hidden, *g_scratch, *g_logits;
    int   *g_ctrl;
    CUDA_CHECK(cudaMalloc(&g_hidden,  (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_scratch, (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_logits,  (size_t)V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_ctrl,    4 * sizeof(int)));

    // Per-layer cooperative scratch (resident across the whole decode loop).
    float *g_q, *g_k, *g_v, *g_attn, *g_gate;
    CUDA_CHECK(cudaMalloc(&g_q,    (size_t)d   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_k,    (size_t)kd  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_v,    (size_t)kd  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_attn, (size_t)d   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_gate, (size_t)dff * sizeof(float)));

    ModelWeights* d_model; KVCache* d_kv;
    CUDA_CHECK(cudaMalloc(&d_model, sizeof(ModelWeights)));
    CUDA_CHECK(cudaMalloc(&d_kv,    sizeof(KVCache)));
    CUDA_CHECK(cudaMemcpy(d_model, &target_model, sizeof(ModelWeights),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kv, &target_kv, sizeof(KVCache),
                          cudaMemcpyHostToDevice));

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, (void*)megakernel_baseline_coop_kernel,
        BLOCK_THREADS, smem_bytes);
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    int sm_count = 1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

    int n_blocks = (V + GEMV_COL_TILE - 1) / GEMV_COL_TILE;
    int hard_limit = sm_count * blocks_per_sm;
    if (n_blocks > hard_limit) n_blocks = hard_limit;
    if (n_blocks < 1)          n_blocks = 1;

    void* args[] = {
        (void*)&d_model, (void*)&d_kv, (void*)&prompt, (void*)&prompt_len,
        (void*)&params.max_new_tokens, (void*)&g_hidden, (void*)&g_scratch,
        (void*)&g_logits, (void*)&g_ctrl, (void*)&d_result, (void*)&params.eos_token,
        (void*)&g_q, (void*)&g_k, (void*)&g_v, (void*)&g_attn, (void*)&g_gate
    };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)megakernel_baseline_coop_kernel,
        dim3(n_blocks), dim3(BLOCK_THREADS), args, smem_bytes, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_model); cudaFree(d_kv);
    cudaFree(g_hidden); cudaFree(g_scratch); cudaFree(g_logits); cudaFree(g_ctrl);
    cudaFree(g_q); cudaFree(g_k); cudaFree(g_v); cudaFree(g_attn); cudaFree(g_gate);
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
    if (!params.self_speculative)
        kv_cache_reset(target_kv);

    ModelWeights* d_draft_model;
    ModelWeights* d_target_model;
    KVCache*      d_draft_kv;
    KVCache*      d_target_kv;
    KVCache*      d_shared_kv = nullptr;

    CUDA_CHECK(cudaMalloc(&d_draft_model, sizeof(ModelWeights)));
    CUDA_CHECK(cudaMalloc(&d_target_model, sizeof(ModelWeights)));

    if (params.self_speculative) {
        CUDA_CHECK(cudaMalloc(&d_shared_kv, sizeof(KVCache)));
        CUDA_CHECK(cudaMemcpy(d_shared_kv, &target_kv, sizeof(KVCache),
                              cudaMemcpyHostToDevice));
        d_draft_kv  = d_shared_kv;
        d_target_kv = d_shared_kv;
    } else {
        CUDA_CHECK(cudaMalloc(&d_draft_kv, sizeof(KVCache)));
        CUDA_CHECK(cudaMalloc(&d_target_kv, sizeof(KVCache)));
        CUDA_CHECK(cudaMemcpy(d_draft_kv, &draft_kv, sizeof(KVCache),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target_kv, &target_kv, sizeof(KVCache),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_draft_model, &draft_model, sizeof(ModelWeights),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_model, &target_model, sizeof(ModelWeights),
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
        float* g_selfspec_hiddens_s = nullptr;
        CUDA_CHECK(cudaMalloc(&g_batch_hidden_s,
                              (size_t)max_B * d_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_work_s,
                              (size_t)(7 * d_t + MLP_FF_TILE) * max_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_logits_s,
                              (size_t)max_B * V_t * sizeof(float)));
        if (params.self_speculative)
            CUDA_CHECK(cudaMalloc(&g_selfspec_hiddens_s,
                                  (size_t)(params.spec_k + 1) * d_t *
                                      sizeof(float)));

        cuda_configure_kernel_dynamic_smem(
            megakernel_speculative_stochastic_kernel,
            smem_bytes);

        megakernel_speculative_stochastic_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            d_draft_model, d_target_model, d_draft_kv, d_target_kv,
            prompt, prompt_len, params.max_new_tokens, params.spec_k,
            logits_d, logits_t, corr_ws,
            g_batch_hidden_s, g_batch_work_s, g_batch_logits_s,
            g_selfspec_hiddens_s,
            params.draft_temperature,
            params.adaptive_draft_temperature ? 1 : 0,
            params.min_draft_temperature,
            params.max_draft_temperature,
            params.stochastic_adapt_target_accept,
            params.stochastic_adapt_temp_gain,
            params.stochastic_adapt_ewma_mix,
            (unsigned long long)params.stochastic_rng_seed,
            d_result,
            params.eos_token,
            params.self_speculative ? 1 : 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(logits_d);
        cudaFree(logits_t);
        cudaFree(corr_ws);
        cudaFree(g_batch_hidden_s);
        cudaFree(g_batch_work_s);
        cudaFree(g_batch_logits_s);
        if (g_selfspec_hiddens_s) cudaFree(g_selfspec_hiddens_s);
    } else {
        float* g_logits;
        CUDA_CHECK(cudaMalloc(&g_logits, (size_t)vocab * sizeof(float)));

        int max_B = params.spec_k + 1;
        int d_t   = target_model.cfg.d_model;
        int V_t   = target_model.cfg.vocab_size;

        float* g_batch_hidden;
        float* g_batch_work;
        float* g_batch_logits;
        float* g_selfspec_hiddens = nullptr;
        CUDA_CHECK(cudaMalloc(&g_batch_hidden,
                              (size_t)max_B * d_t * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_work,
                              (size_t)(7 * d_t + MLP_FF_TILE) * max_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_batch_logits,
                              (size_t)max_B * V_t * sizeof(float)));
        if (params.self_speculative)
            CUDA_CHECK(cudaMalloc(&g_selfspec_hiddens,
                                  (size_t)(params.spec_k + 1) * d_t *
                                      sizeof(float)));

        cuda_configure_kernel_dynamic_smem(megakernel_speculative_kernel,
                                           smem_bytes);

        megakernel_speculative_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            d_draft_model, d_target_model, d_draft_kv, d_target_kv,
            prompt, prompt_len, params.max_new_tokens, params.spec_k,
            g_logits,
            g_batch_hidden, g_batch_work, g_batch_logits,
            g_selfspec_hiddens,
            d_result, params.eos_token, params.self_speculative ? 1 : 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(g_logits);
        cudaFree(g_batch_hidden);
        cudaFree(g_batch_work);
        cudaFree(g_batch_logits);
        if (g_selfspec_hiddens) cudaFree(g_selfspec_hiddens);
    }

    cudaFree(d_draft_model);
    cudaFree(d_target_model);
    if (params.self_speculative) {
        cudaFree(d_shared_kv);
    } else {
        cudaFree(d_draft_kv);
        cudaFree(d_target_kv);
    }
}
