#include "kernels.h"
#include "utils.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

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

    int* h_draft_tokens  = new int[k];
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
    // Allocate g_logits for the larger vocab (the target, typically)
    int vocab = (target_model.cfg.vocab_size > draft_model.cfg.vocab_size)
                ? target_model.cfg.vocab_size
                : draft_model.cfg.vocab_size;
    float* g_logits;
    CUDA_CHECK(cudaMalloc(&g_logits, (size_t)vocab * sizeof(float)));

    // Megakernel uses the TARGET model's smem budget (it's always >= draft's)
    size_t smem_bytes = compute_smem_bytes(target_model.cfg);
    cuda_configure_kernel_dynamic_smem(megakernel_speculative_kernel,
                                       smem_bytes);

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

    megakernel_speculative_kernel<<<1, BLOCK_THREADS, smem_bytes>>>(
        d_draft_model, d_target_model, d_draft_kv, d_target_kv,
        prompt, prompt_len, params.max_new_tokens, params.spec_k,
        g_logits, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_draft_model);
    cudaFree(d_target_model);
    cudaFree(d_draft_kv);
    cudaFree(d_target_kv);
    cudaFree(g_logits);
}
