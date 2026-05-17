#include "kernels.h"
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
    for (int i = 0; i < V && cdf < u_fix - 1e-8f; i++) {
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
        for (int i = 0; i < V && cdf < u_fix - 1e-8f; i++) {
            cdf += wp[i] / sum_p;
            if (cdf >= u_fix || i == V - 1)
                return i;
        }
        return V - 1;
    }

    float u_fix = curand_uniform(rng_state);
    float cdf   = 0.f;
    for (int i = 0; i < V && cdf < u_fix - 1e-8f; i++) {
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
        for (int i = 0; i < V && cdf < u_fix - 1e-8f; i++) {
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
    for (int i = 0; i < V && cdf < u_fix - 1e-8f; i++) {
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
    for (int i = 0; i < V && cdf < u_fix - 1e-8f; i++) {
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

// ---- Baseline (multi-kernel) ----

void multikernel_baseline(const ModelWeights& target_model,
                          KVCache& target_kv,
                          const int* h_prompt, int prompt_len,
                          GenerationResult* d_result,
                          const GenerationParams& params) {
    // h_prompt is a HOST pointer — values are read on CPU for each kernel launch.
    int max_new = params.max_new_tokens;

    // Dynamic smem and logits buffer sized for the target model
    size_t smem_bytes = compute_smem_bytes(target_model.cfg);
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
    for (int i = 0; i < prompt_len; i++) {
        single_token_decode_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            target_model, target_kv, h_prompt[i], seq_len,
            g_logits, d_next_token);
        seq_len++;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, seq_len);
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
        if (current_token == EOS_TOKEN || generated >= max_new) break;

        single_token_decode_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            target_model, target_kv, current_token, seq_len,
            g_logits, d_next_token);
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

// ---- Speculative (multi-kernel) ----

void multikernel_speculative(const ModelWeights& draft_model,
                             const ModelWeights& target_model,
                             KVCache& draft_kv,
                             KVCache& target_kv,
                             const int* h_prompt, int prompt_len,
                             GenerationResult* d_result,
                             const GenerationParams& params) {
    // h_prompt is a HOST pointer.
    int max_new = params.max_new_tokens;
    int k       = params.spec_k;

    // Each model gets its own smem budget and logits buffer
    size_t draft_smem  = compute_smem_bytes(draft_model.cfg);
    size_t target_smem = compute_smem_bytes(target_model.cfg);
    size_t max_decode_smem =
        draft_smem > target_smem ? draft_smem : target_smem;
    cuda_configure_kernel_dynamic_smem(single_token_decode_kernel,
                                       max_decode_smem);

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

    // Prefill both models
    for (int i = 0; i < prompt_len; i++) {
        single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
            draft_model, draft_kv, h_prompt[i], draft_seq,
            g_logits_draft, d_next);
        draft_seq++;
        set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);

        single_token_decode_kernel<<<1, BLOCK_THREADS, target_smem>>>(
            target_model, target_kv, h_prompt[i], target_seq,
            g_logits_target, d_next);
        target_seq++;
        set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);
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
        cuda_configure_kernel_dynamic_smem(single_token_forward_logits_kernel,
                                           max_decode_smem);
        cuda_configure_kernel_dynamic_smem(target_forward_prob_mass_kernel,
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

        while (generated < max_new) {
            iterations++;
            int remaining = max_new - generated;
            int current_k = (k < remaining) ? k : remaining;

            int draft_seq_save  = draft_seq;
            int target_seq_save = target_seq;
            total_proposed += current_k;

            int draft_ctx = last_token;
            for (int di = 0; di < current_k; di++) {
                stochastic_draft_forward_sample_kernel
                    <<<1, BLOCK_THREADS, draft_smem>>>(
                        draft_model, draft_kv, draft_ctx, draft_seq,
                        draft_temp_dyn, d_rng, g_logits_draft, d_next,
                        d_q_mass_out);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&h_draft_tokens[di], d_next,
                                      sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&h_q_probs[di], d_q_mass_out,
                                      sizeof(float), cudaMemcpyDeviceToHost));

                draft_ctx = h_draft_tokens[di];
                draft_seq++;
                set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            }

            int n_accept_round = 0;
            int bonus          = EOS_TOKEN;
            bool broke_early    = false;
            int  target_roll    = target_seq_save;

            for (int vi = 0; vi < current_k; vi++) {
                int inp = (vi == 0) ? last_token : h_draft_tokens[vi - 1];

                cuda_configure_kernel_dynamic_smem(target_forward_prob_mass_kernel,
                                                   target_smem);
                target_forward_prob_mass_kernel
                    <<<1, BLOCK_THREADS, target_smem>>>(
                        target_model, target_kv, inp, target_roll,
                        h_draft_tokens[vi], g_logits_target, d_p_mass_out);
                CUDA_CHECK(cudaDeviceSynchronize());

                float p_mass = 0.f;
                CUDA_CHECK(cudaMemcpy(&p_mass, d_p_mass_out, sizeof(float),
                                      cudaMemcpyDeviceToHost));

                cuda_configure_kernel_dynamic_smem(stochastic_accept_gate_kernel,
                                                   0);
                stochastic_accept_gate_kernel<<<1, 1>>>(
                    p_mass, h_q_probs[vi], d_rng, d_accept);
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

                cuda_configure_kernel_dynamic_smem(single_token_forward_logits_kernel,
                                                     draft_smem);
                single_token_forward_logits_kernel
                    <<<1, BLOCK_THREADS, draft_smem>>>(
                        draft_model, draft_kv, inp, draft_seq_save + vi,
                        g_logits_draft);
                CUDA_CHECK(cudaDeviceSynchronize());

                cuda_configure_kernel_dynamic_smem(corrected_sample_adjusted_logits_kernel,
                                                     0);
                corrected_sample_adjusted_logits_kernel<<<1, 1>>>(
                    g_logits_target, g_logits_draft, V_vocab, d_rng, d_next,
                    d_corr_work, 1.f, draft_temp_dyn);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&bonus, d_next, sizeof(int),
                                      cudaMemcpyDeviceToHost));

                cuda_configure_kernel_dynamic_smem(single_token_forward_logits_kernel,
                                                   target_smem);
                single_token_forward_logits_kernel
                    <<<1, BLOCK_THREADS, target_smem>>>(
                        target_model, target_kv, bonus,
                        target_seq_save + n_accept_round, g_logits_target);
                CUDA_CHECK(cudaDeviceSynchronize());

                break;
            }

            if (!broke_early) {
                cuda_configure_kernel_dynamic_smem(single_token_forward_logits_kernel,
                                                     target_smem);
                single_token_forward_logits_kernel
                    <<<1, BLOCK_THREADS, target_smem>>>(
                        target_model, target_kv, h_draft_tokens[current_k - 1],
                        target_roll, g_logits_target);
                CUDA_CHECK(cudaDeviceSynchronize());

                cuda_configure_kernel_dynamic_smem(softmax_sample_temperature_kernel,
                                                     0);
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

            if (bonus == EOS_TOKEN || generated >= max_new)
                break;
        }

        delete[] h_q_probs;
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

    while (generated < max_new) {
        iterations++;
        int remaining = max_new - generated;
        int current_k = (k < remaining) ? k : remaining;

        int draft_seq_save  = draft_seq;
        int target_seq_save = target_seq;
        total_proposed += current_k;

        // ---- Draft phase ----
        int draft_token = last_token;
        for (int di = 0; di < current_k; di++) {
            single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                draft_model, draft_kv, draft_token, draft_seq,
                g_logits_draft, d_next);
            draft_seq++;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
            CUDA_CHECK(cudaMemcpy(&draft_token, d_next, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            h_draft_tokens[di] = draft_token;
        }

        // ---- Verify phase: feed [last_token, draft_0..draft_{k-1}] to target ----
        for (int vi = 0; vi <= current_k; vi++) {
            int tok = (vi == 0) ? last_token : h_draft_tokens[vi - 1];
            single_token_decode_kernel<<<1, BLOCK_THREADS, target_smem>>>(
                target_model, target_kv, tok, target_seq,
                g_logits_target, d_next);
            target_seq++;
            set_seq_len_kernel<<<1, 1>>>(target_kv.seq_len, target_seq);
            CUDA_CHECK(cudaMemcpy(&h_target_tokens[vi], d_next, sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }

        // ---- Accept/reject: greedy match ----
        int n_accepted = 0;
        for (int i = 0; i < current_k; i++) {
            if (h_target_tokens[i] == h_draft_tokens[i]) n_accepted++;
            else break;
        }
        total_accepted += n_accepted;

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

        // If ALL k drafts were accepted, draft[k-1] was only an output —
        // its K,V was never appended.  Run an extra forward pass to sync.
        if (n_accepted == current_k) {
            single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
                draft_model, draft_kv, h_draft_tokens[current_k - 1],
                draft_seq, g_logits_draft, d_next);
            draft_seq++;
            set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
        }

        // Sync draft cache with the bonus/correction token
        single_token_decode_kernel<<<1, BLOCK_THREADS, draft_smem>>>(
            draft_model, draft_kv, last_token, draft_seq,
            g_logits_draft, d_next);
        draft_seq++;
        set_seq_len_kernel<<<1, 1>>>(draft_kv.seq_len, draft_seq);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (bonus == EOS_TOKEN || generated >= max_new) break;
    }

    write_result_kernel<<<1, BLOCK_THREADS>>>(
        d_result, d_output, generated,
        total_proposed, total_accepted, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());

    delete[] h_draft_tokens;
    delete[] h_target_tokens;
    cudaFree(g_logits_draft);
    cudaFree(g_logits_target);
    cudaFree(d_next);
    cudaFree(d_output);
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
        float* g_logits, GenerationResult* result) {
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

        if (s_current_token == EOS_TOKEN || s_generated >= max_new_tokens)
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
        GenerationResult* result) {
    const ModelWeights& draft_model  = *p_draft_model;
    const ModelWeights& target_model = *p_target_model;
    KVCache&            draft_kv     = *p_draft_kv;
    KVCache&            target_kv    = *p_target_kv;
    extern __shared__ float shared[];
    // Use the larger model's d_model for the hidden state layout.
    // Both models share the same smem; the larger model's scratch
    // fully covers the smaller model's needs.
    int d = target_model.cfg.d_model;
    float* hidden = shared;       // [d_model_target]
    float* smem   = shared + d;   // scratch

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

        // ---- Verify phase ----
        for (int vi = 0; vi <= current_k; vi++) {
            int tok = (vi == 0) ? s_last_token : s_draft_tokens[vi - 1];
            int next = model_forward(target_model, target_kv, tok,
                                     s_target_seq, hidden, g_logits, smem);
            if (tid == 0) {
                s_target_tokens[vi] = next;
                s_target_seq++;
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

        // Sync draft cache: if ALL drafts accepted, the last draft's K,V
        // was never stored — run an extra forward pass to fix that.
        if (s_n_accepted == current_k) {
            model_forward(draft_model, draft_kv, s_draft_tokens[current_k - 1],
                          s_draft_seq, hidden, g_logits, smem);
            if (tid == 0) { s_draft_seq++; *draft_kv.seq_len = s_draft_seq; }
            __syncthreads();
        }

        // Sync draft cache with bonus token
        model_forward(draft_model, draft_kv, s_last_token,
                      s_draft_seq, hidden, g_logits, smem);
        if (tid == 0) { s_draft_seq++; *draft_kv.seq_len = s_draft_seq; }
        __syncthreads();

        if (s_last_token == EOS_TOKEN || s_generated >= max_new_tokens)
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
        float draft_temp_initial,
        int adaptive_enabled,
        float min_draft_temp,
        float max_draft_temp,
        float adapt_tgt_accept,
        float adapt_gain,
        float adapt_ewma_mix,
        unsigned long long rng_seed,
        GenerationResult* result) {

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

        if (tid == 0) {
            s_tr_roll = s_ts_save;
            acc_cnt   = 0;
        }
        __syncthreads();

        bool inner_done = false;
        for (int vi = 0; vi < ck; vi++) {
            int in_t =
                (vi == 0) ? s_last_token : s_draft_ids[vi - 1];

            model_forward_logits(target_model,
                                 target_kv,
                                 in_t,
                                 s_tr_roll,
                                 hidden,
                                 logits_target,
                                 scratch);
            __syncthreads();

            if (tid == 0) {
                s_tr_roll++;
                *target_kv.seq_len = s_tr_roll;
            }
            __syncthreads();

            float pv = logits_softmax_prob_at_global(logits_target,
                                                     V_vocab,
                                                     s_draft_ids[vi],
                                                     1.f,
                                                     scratch);
            __syncthreads();

            if (tid == 0) {
                s_ok_gate = device_stochastic_accept_mass(pv,
                                                          s_q_probs[vi],
                                                          &s_rng)
                                ? 1
                                : 0;
            }
            __syncthreads();

            if (s_ok_gate) {
                if (tid == 0)
                    acc_cnt++;
                __syncthreads();
                continue;
            }

            if (tid == 0) {
                *draft_kv.seq_len          = s_ds_save + vi;
                *target_kv.seq_len          = s_ts_save + vi;
                s_tr_roll                  = s_ts_save + vi;
                s_draft_seq                = s_ds_save + vi;
                s_target_seq               = s_tr_roll;
            }
            __syncthreads();

            model_forward_logits(draft_model,
                                 draft_kv,
                                 in_t,
                                 s_ds_save + vi,
                                 hidden,
                                 logits_draft,
                                 scratch);
            __syncthreads();

            if (tid == 0) {
                int bon_c = device_corrected_adjusted_sample(logits_target,
                                                              logits_draft,
                                                              V_vocab,
                                                              &s_rng,
                                                              corr_workspace,
                                                              1.f,
                                                              s_dyn_dt);
                s_bonus_token = bon_c;
            }

            __syncthreads();

            model_forward_logits(target_model,
                                 target_kv,
                                 s_bonus_token,
                                 s_ts_save + acc_cnt,
                                 hidden,
                                 logits_target,
                                 scratch);
            __syncthreads();

            if (tid == 0) {
                s_target_seq = s_ts_save + acc_cnt + 1;
                s_tr_roll    = s_target_seq;
                *target_kv.seq_len = s_target_seq;
            }
            inner_done = true;
            __syncthreads();
            break;
        }

        __syncthreads();

        if (!inner_done && ck > 0) {
            model_forward_logits(target_model,
                                 target_kv,
                                 s_draft_ids[ck - 1],
                                 s_ts_save + ck,
                                 hidden,
                                 logits_target,
                                 scratch);
            __syncthreads();

            if (tid == 0) {
                int bon_full =
                    device_softmax_sample_logits_temp_inplace(logits_target,
                                                              V_vocab,
                                                              1.f,
                                                              &s_rng);
                s_bonus_token = bon_full;
            }

            __syncthreads();

            model_forward_logits(target_model,
                                 target_kv,
                                 s_bonus_token,
                                 s_ts_save + ck,
                                 hidden,
                                 logits_target,
                                 scratch);

            __syncthreads();

            if (tid == 0) {
                acc_cnt           = ck;
                s_target_seq      = s_ts_save + ck + 1;
                s_tr_roll         = s_target_seq;
                *target_kv.seq_len = s_target_seq;
            }
            __syncthreads();
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
        }

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

        if (s_last_token == EOS_TOKEN || s_generated >= max_new_tokens)
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
        params.max_new_tokens, g_logits, d_result);
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

        cuda_configure_kernel_dynamic_smem(
            megakernel_speculative_stochastic_kernel,
            smem_bytes);

        megakernel_speculative_stochastic_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            d_draft_model, d_target_model, d_draft_kv, d_target_kv,
            prompt, prompt_len, params.max_new_tokens, params.spec_k,
            logits_d, logits_t, corr_ws,
            params.draft_temperature,
            params.adaptive_draft_temperature ? 1 : 0,
            params.min_draft_temperature,
            params.max_draft_temperature,
            params.stochastic_adapt_target_accept,
            params.stochastic_adapt_temp_gain,
            params.stochastic_adapt_ewma_mix,
            (unsigned long long)params.stochastic_rng_seed,
            d_result);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(logits_d);
        cudaFree(logits_t);
        cudaFree(corr_ws);
    } else {
        float* g_logits;
        CUDA_CHECK(cudaMalloc(&g_logits, (size_t)vocab * sizeof(float)));

        cuda_configure_kernel_dynamic_smem(megakernel_speculative_kernel,
                                           smem_bytes);

        megakernel_speculative_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
            d_draft_model, d_target_model, d_draft_kv, d_target_kv,
            prompt, prompt_len, params.max_new_tokens, params.spec_k,
            g_logits, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(g_logits);
    }

    cudaFree(d_draft_model);
    cudaFree(d_target_model);
    cudaFree(d_draft_kv);
    cudaFree(d_target_kv);
}
