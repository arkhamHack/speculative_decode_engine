#include "model.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <curand_kernel.h>

// ============================================================================
// Host: allocate weight tensors on the GPU (16-byte aligned)
// ============================================================================

static half* alloc_half(size_t n) {
    half* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, align_up(n * sizeof(half), 16)));
    return ptr;
}

void model_alloc(ModelWeights& model, const ModelConfig& cfg) {
    model.cfg = cfg;
    const int d   = cfg.d_model;
    const int dff = cfg.d_ff;
    const int V   = cfg.vocab_size;

    model.token_embedding  = alloc_half((size_t)V * d);
    model.rms_final_weight = alloc_half(d);
    model.output_proj      = alloc_half((size_t)d * V);

    for (int l = 0; l < cfg.n_layers; l++) {
        LayerWeights& lw = model.layers[l];
        lw.rms_attn_weight = alloc_half(d);
        // Q/K/V/O all [d_model, d_model] regardless of n_heads.
        // Multi-head slicing happens at compute time.
        lw.Wq              = alloc_half((size_t)d * d);
        lw.Wk              = alloc_half((size_t)d * d);
        lw.Wv              = alloc_half((size_t)d * d);
        lw.Wo              = alloc_half((size_t)d * d);
        lw.rms_mlp_weight  = alloc_half(d);
        lw.W_gate          = alloc_half((size_t)d * dff);
        lw.W_up            = alloc_half((size_t)d * dff);
        lw.W_down          = alloc_half((size_t)dff * d);
    }
}

void model_free(ModelWeights& model) {
    auto safe_free = [](half*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free(model.token_embedding);
    safe_free(model.rms_final_weight);
    safe_free(model.output_proj);
    for (int l = 0; l < model.cfg.n_layers; l++) {
        LayerWeights& lw = model.layers[l];
        safe_free(lw.rms_attn_weight);
        safe_free(lw.Wq); safe_free(lw.Wk);
        safe_free(lw.Wv); safe_free(lw.Wo);
        safe_free(lw.rms_mlp_weight);
        safe_free(lw.W_gate); safe_free(lw.W_up); safe_free(lw.W_down);
    }
}

// ============================================================================
// Host: random-weight initialisation (benchmark / dummy-tokenizer path)
// ============================================================================

__global__ void init_weights_kernel(half* data, int n, unsigned seed, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState state;
    curand_init(seed, idx + offset, 0, &state);
    data[idx] = __float2half(curand_normal(&state) * 0.02f);
}

__global__ void init_ones_kernel(half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = __float2half(1.0f);
}

static void fill_random(half* ptr, size_t n, unsigned seed, int& offset) {
    int blocks = ((int)n + 255) / 256;
    init_weights_kernel<<<blocks, 256>>>(ptr, (int)n, seed, offset);
    offset += (int)n;
}

static void fill_ones(half* ptr, size_t n) {
    int blocks = ((int)n + 255) / 256;
    init_ones_kernel<<<blocks, 256>>>(ptr, (int)n);
}

void model_init_random(ModelWeights& model, unsigned seed) {
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int V   = model.cfg.vocab_size;
    int offset = 0;

    fill_random(model.token_embedding, (size_t)V * d, seed, offset);
    fill_random(model.output_proj,     (size_t)d * V, seed, offset);
    // RMS norm scales initialised to 1 (standard practice)
    fill_ones(model.rms_final_weight, d);

    for (int l = 0; l < model.cfg.n_layers; l++) {
        LayerWeights& lw = model.layers[l];
        fill_ones(lw.rms_attn_weight, d);
        fill_random(lw.Wq, (size_t)d * d, seed, offset);
        fill_random(lw.Wk, (size_t)d * d, seed, offset);
        fill_random(lw.Wv, (size_t)d * d, seed, offset);
        fill_random(lw.Wo, (size_t)d * d, seed, offset);
        fill_ones(lw.rms_mlp_weight, d);
        fill_random(lw.W_gate, (size_t)d * dff, seed, offset);
        fill_random(lw.W_up,   (size_t)d * dff, seed, offset);
        fill_random(lw.W_down, (size_t)dff * d, seed, offset);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Host: load weights from an SDEC binary file
//
// File format (little-endian):
//   [4 bytes]  magic  "SDEC"
//   [4 bytes]  version  (must be 1)
//   [4 bytes]  n_layers
//   [4 bytes]  d_model
//   [4 bytes]  n_heads
//   [4 bytes]  d_ff
//   [4 bytes]  vocab_size
//
// Tensors in this fixed order (each preceded by uint32 element count):
//   token_embedding  [vocab_size * d_model]
//   rms_final_weight [d_model]
//   output_proj      [d_model * vocab_size]
//   For each layer l = 0..n_layers-1:
//     rms_attn_weight [d_model]
//     Wq              [d_model * d_model]
//     Wk              [d_model * d_model]
//     Wv              [d_model * d_model]
//     Wo              [d_model * d_model]
//     rms_mlp_weight  [d_model]
//     W_gate          [d_model * d_ff]
//     W_up            [d_model * d_ff]
//     W_down          [d_ff    * d_model]
//
// All data values are float16 (2 bytes each).
// The Python export tool tools/export_model.py produces this format.
// ============================================================================

bool model_load_weights(ModelWeights& model, const char* path,
                        ModelConfig* cfg_out) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[model_load] Cannot open '%s'\n", path);
        return false;
    }

    // --- Validate magic ---
    char magic[5] = {0};
    if (fread(magic, 1, 4, f) != 4 || strncmp(magic, "SDEC", 4) != 0) {
        fprintf(stderr, "[model_load] Bad magic in '%s'\n", path);
        fclose(f); return false;
    }

    // --- Read header ---
    uint32_t version, n_layers, d_model, n_heads, d_ff, vocab_size;
    if (fread(&version,    4, 1, f) != 1 || version != 1u ||
        fread(&n_layers,   4, 1, f) != 1 ||
        fread(&d_model,    4, 1, f) != 1 ||
        fread(&n_heads,    4, 1, f) != 1 ||
        fread(&d_ff,       4, 1, f) != 1 ||
        fread(&vocab_size, 4, 1, f) != 1) {
        fprintf(stderr, "[model_load] Truncated header in '%s'\n", path);
        fclose(f); return false;
    }

    if ((int)n_layers > MAX_LAYERS) {
        fprintf(stderr, "[model_load] n_layers=%u exceeds MAX_LAYERS=%d\n",
                n_layers, MAX_LAYERS);
        fclose(f); return false;
    }
    if ((int)d_model % (int)n_heads != 0) {
        fprintf(stderr, "[model_load] d_model=%u not divisible by n_heads=%u\n",
                d_model, n_heads);
        fclose(f); return false;
    }

    ModelConfig cfg;
    cfg.n_layers   = (int)n_layers;
    cfg.d_model    = (int)d_model;
    cfg.d_head     = (int)d_model;  // total KV dim = d_model
    cfg.n_heads    = (int)n_heads;
    cfg.d_ff       = (int)d_ff;
    cfg.vocab_size = (int)vocab_size;
    if (cfg_out) *cfg_out = cfg;

    model_alloc(model, cfg);

    // Helper: read one tensor from file, validate element count, upload to GPU
    auto read_tensor = [&](half* dst, size_t expected_n,
                           const char* name) -> bool {
        uint32_t n;
        if (fread(&n, 4, 1, f) != 1) {
            fprintf(stderr, "[model_load] EOF reading count for '%s'\n", name);
            return false;
        }
        if ((size_t)n != expected_n) {
            fprintf(stderr, "[model_load] '%s': expected %zu elements, got %u\n",
                    name, expected_n, n);
            return false;
        }
        // Read into host buffer then copy to GPU in one shot
        std::vector<half> buf(n);
        if (fread(buf.data(), sizeof(half), n, f) != (size_t)n) {
            fprintf(stderr, "[model_load] Short read for '%s'\n", name);
            return false;
        }
        CUDA_CHECK(cudaMemcpy(dst, buf.data(), n * sizeof(half),
                              cudaMemcpyHostToDevice));
        return true;
    };

    const int d   = cfg.d_model;
    const int dff = cfg.d_ff;
    const int V   = cfg.vocab_size;

    bool ok =
        read_tensor(model.token_embedding,  (size_t)V * d, "token_embedding")  &&
        read_tensor(model.rms_final_weight, (size_t)d,     "rms_final_weight") &&
        read_tensor(model.output_proj,      (size_t)d * V, "output_proj");

    for (int l = 0; l < cfg.n_layers && ok; l++) {
        LayerWeights& lw = model.layers[l];
        char name[64];
        auto n = [&](const char* s) { snprintf(name, sizeof(name), "L%d.%s", l, s); return name; };
        ok =
            read_tensor(lw.rms_attn_weight, (size_t)d,      n("rms_attn")) &&
            read_tensor(lw.Wq,              (size_t)d * d,  n("Wq"))       &&
            read_tensor(lw.Wk,              (size_t)d * d,  n("Wk"))       &&
            read_tensor(lw.Wv,              (size_t)d * d,  n("Wv"))       &&
            read_tensor(lw.Wo,              (size_t)d * d,  n("Wo"))       &&
            read_tensor(lw.rms_mlp_weight,  (size_t)d,      n("rms_mlp")) &&
            read_tensor(lw.W_gate,          (size_t)d * dff,n("W_gate"))   &&
            read_tensor(lw.W_up,            (size_t)d * dff,n("W_up"))     &&
            read_tensor(lw.W_down,          (size_t)dff * d,n("W_down"));
    }

    fclose(f);
    if (!ok) {
        fprintf(stderr, "[model_load] Failed — freeing partial allocations\n");
        model_free(model);
        return false;
    }

    fprintf(stderr, "[model_load] Loaded '%s'  layers=%d d=%d heads=%d d_ff=%d vocab=%d\n",
            path, cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.vocab_size);
    return true;
}

// ============================================================================
// Device: embedding lookup
// ============================================================================

__device__ void model_embed(const ModelWeights& model, int token_id,
                            float* hidden) {
    int tid = threadIdx.x;
    int d   = model.cfg.d_model;
    // Stride loop handles d > BLOCK_THREADS
    for (int i = tid; i < d; i += blockDim.x)
        hidden[i] = __half2float(model.token_embedding[token_id * d + i]);
    __syncthreads();
}

// ============================================================================
// Device: multi-head attention + SwiGLU MLP transformer layer
//
// Multi-head attention overview:
//   n_heads = cfg.n_heads
//   dph     = d_model / n_heads    (per-head key/value dimension)
//
//   1.  RMSNorm(hidden) -> normed
//   2.  Q[d] = normed @ Wq         (all heads concatenated)
//       K[d] = normed @ Wk
//       V[d] = normed @ Wv
//   3.  Append K, V to paged KV cache (stored interleaved per slot: K[d], V[d])
//   4.  For head h in [0, n_heads):
//         a. scores[p] = (Q[h*dph:(h+1)*dph] · K_p[h*dph:(h+1)*dph]) / sqrt(dph)
//            for each cached position p
//         b. softmax(scores)
//         c. attn_out[h*dph:(h+1)*dph] = Σ_p scores[p] * V_p[h*dph:(h+1)*dph]
//   5.  hidden += attn_out @ Wo          (residual)
//   6.  RMSNorm(hidden) -> normed
//   7.  SwiGLU: gate = normed @ W_gate
//               up   = normed @ W_up
//               mlp_out = silu(gate) * up  @ W_down
//   8.  hidden += mlp_out                (residual)
//
// Shared memory scratch layout (smem = shared + d):
//   normed   [d]              RMSNorm output, also MLP scratch
//   q_all    [d]              Q for all heads
//   kv_temp  [d]              K (then V) temp; also output-proj output
//   scores   [MAX_SEQ_LEN]    per-head attention weights
//   attn_out [d]              accumulated per-head attention output
//   gate_buf [d_ff]           SwiGLU gate
//   up_buf   [d_ff]           SwiGLU up
//   scratch  [WARP_SIZE]      block-reduction scratch
// ============================================================================

__device__ void model_layer_forward(const ModelWeights& model, int layer_idx,
                                    float* hidden, KVCache& kv,
                                    int current_seq_len, float* smem) {
    int tid = threadIdx.x;
    int d   = model.cfg.d_model;
    int dff = model.cfg.d_ff;
    int nh  = model.cfg.n_heads;
    int dph = d / nh;   // per-head dim

    const LayerWeights& lw = model.layers[layer_idx];

    // --- Smem pointer arithmetic (runtime offsets, d and dff are runtime values) ---
    float* normed   = smem;
    float* q_all    = smem + d;
    float* kv_temp  = smem + 2 * d;
    float* scores   = smem + 3 * d;              // [MAX_SEQ_LEN]
    float* attn_out = smem + 3 * d + MAX_SEQ_LEN;
    float* gate_buf = smem + 4 * d + MAX_SEQ_LEN;
    float* up_buf   = smem + 4 * d + MAX_SEQ_LEN + dff;
    float* scratch  = smem + 4 * d + MAX_SEQ_LEN + 2 * dff;

    // =========================================================
    // ---- Attention sub-layer --------------------------------
    // =========================================================

    // 1. RMSNorm
    device_rmsnorm(hidden, lw.rms_attn_weight, normed, d, scratch);

    // 2. Full Q[d], K[d], V[d] via matrix-vector multiply
    device_matvec(normed, lw.Wq, q_all, d, d);

    // K: compute into kv_temp, then copy to attn_out (as temporary K storage)
    device_matvec(normed, lw.Wk, kv_temp, d, d);
    for (int i = tid; i < d; i += blockDim.x)
        attn_out[i] = kv_temp[i];    // save K in attn_out temporarily
    __syncthreads();

    // V: compute into kv_temp
    device_matvec(normed, lw.Wv, kv_temp, d, d);

    // 3. Append K (in attn_out) and V (in kv_temp) to paged KV cache.
    //    kv.d_head = d_model: the cache stores the full K[d] and V[d] per slot.
    kv_cache_append(kv, layer_idx, attn_out, kv_temp, current_seq_len);

    // Clear attn_out for per-head accumulation
    for (int i = tid; i < d; i += blockDim.x) attn_out[i] = 0.0f;
    __syncthreads();

    // 4. Multi-head attention
    int total_len = current_seq_len + 1;  // includes the just-appended token
    float scale   = rsqrtf((float)dph);

    for (int h = 0; h < nh; h++) {
        int head_off = h * dph;  // offset in K/V/Q for this head

        // --- 4a. Compute attention scores (position-parallel) ---
        for (int p = tid; p < total_len; p += blockDim.x) {
            int lb = p / KV_BLOCK_SIZE, sl = p % KV_BLOCK_SIZE;
            int pb = kv.layers[layer_idx].block_table[lb];
            // KV block layout: [K: KV_BLOCK_SIZE * d halfs | V: KV_BLOCK_SIZE * d halfs]
            const half* base  = kv.pool + (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
            const half* k_src = base + sl * kv.d_head + head_off;

            float dot = 0.0f;
            for (int e = 0; e < dph; e++)
                dot += q_all[head_off + e] * __half2float(k_src[e]);
            scores[p] = dot * scale;
        }
        __syncthreads();

        // --- 4b. Softmax over scores[0..total_len-1] ---
        block_softmax_inplace(scores, total_len, scratch);

        // --- 4c. Weighted sum of V (element-parallel within this head) ---
        for (int e = tid; e < dph; e += blockDim.x) {
            int global_e = head_off + e;
            float acc = 0.0f;
            for (int p = 0; p < total_len; p++) {
                int lb = p / KV_BLOCK_SIZE, sl = p % KV_BLOCK_SIZE;
                int pb = kv.layers[layer_idx].block_table[lb];
                const half* base  = kv.pool + (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                // V region starts after K region in each block
                const half* v_src = base + KV_BLOCK_SIZE * kv.d_head + sl * kv.d_head;
                acc += scores[p] * __half2float(v_src[global_e]);
            }
            attn_out[global_e] = acc;
        }
        __syncthreads();
    }

    // 5. Output projection: attn_out[d] @ Wo[d,d] -> kv_temp[d]
    device_matvec(attn_out, lw.Wo, kv_temp, d, d);

    // Residual
    for (int i = tid; i < d; i += blockDim.x) hidden[i] += kv_temp[i];
    __syncthreads();

    // =========================================================
    // ---- MLP sub-layer (SwiGLU) -----------------------------
    // =========================================================

    // 6. RMSNorm
    device_rmsnorm(hidden, lw.rms_mlp_weight, normed, d, scratch);

    // 7a. Gate and up projections (non-overlapping smem regions)
    device_matvec(normed, lw.W_gate, gate_buf, d, dff);
    device_matvec(normed, lw.W_up,   up_buf,   d, dff);

    // 7b. SwiGLU activation:  silu(gate) * up
    for (int i = tid; i < dff; i += blockDim.x) {
        float g      = gate_buf[i];
        float silu_g = g / (1.0f + expf(-g));
        gate_buf[i]  = silu_g * up_buf[i];
    }
    __syncthreads();

    // 7c. Down projection into normed
    device_matvec(gate_buf, lw.W_down, normed, dff, d);

    // 8. Residual
    for (int i = tid; i < d; i += blockDim.x) hidden[i] += normed[i];
    __syncthreads();
}

// ============================================================================
// Device: final RMSNorm + output projection
// g_logits is a global-memory buffer of vocab_size floats.
// ============================================================================

__device__ void model_output(const ModelWeights& model,
                             const float* hidden, float* g_logits, float* smem) {
    int d = model.cfg.d_model;
    int V = model.cfg.vocab_size;

    // Reuse the normed and scratch regions from the smem scratch area
    float* normed  = smem;           // [d]
    float* scratch = smem + d;       // [WARP_SIZE]  (well within remaining smem)

    device_rmsnorm(hidden, model.rms_final_weight, normed, d, scratch);
    // Write logits directly to global memory (vocab_size may be >> shared memory)
    device_matvec(normed, model.output_proj, g_logits, d, V);
}

// ============================================================================
// Device: full single-token forward pass
// Returns the greedy next-token id.
// ============================================================================

__device__ int model_forward(const ModelWeights& model, KVCache& kv,
                             int token_id, int current_seq_len,
                             float* hidden, float* g_logits, float* smem) {
    model_embed(model, token_id, hidden);

    for (int l = 0; l < model.cfg.n_layers; l++)
        model_layer_forward(model, l, hidden, kv, current_seq_len, smem);

    model_output(model, hidden, g_logits, smem);

    // Argmax over global-memory logits.
    // Use smem as scratch (2 * BLOCK_THREADS floats needed; smem >> that).
    return global_argmax(g_logits, model.cfg.vocab_size, smem);
}
