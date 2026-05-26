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
//   [4 bytes]  version  (1 or 2)
//   [4 bytes]  n_layers
//   [4 bytes]  d_model
//   [4 bytes]  n_heads
//   [4 bytes]  d_ff
//   [4 bytes]  vocab_size
//   If version >= 2:
//   [4 bytes]  rope_theta  (float32, Llama rotary base)
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
    if (fread(&version, 4, 1, f) != 1) {
        fprintf(stderr, "[model_load] Truncated header in '%s'\n", path);
        fclose(f); return false;
    }
    if (version != 1u && version != 2u) {
        fprintf(stderr, "[model_load] Unsupported SDEC version %u (expected 1 or 2)\n",
                version);
        fclose(f); return false;
    }
    if (fread(&n_layers,   4, 1, f) != 1 ||
        fread(&d_model,    4, 1, f) != 1 ||
        fread(&n_heads,    4, 1, f) != 1 ||
        fread(&d_ff,       4, 1, f) != 1 ||
        fread(&vocab_size, 4, 1, f) != 1) {
        fprintf(stderr, "[model_load] Truncated header in '%s'\n", path);
        fclose(f); return false;
    }

    float rope_theta = 10000.f;
    if (version >= 2u) {
        if (fread(&rope_theta, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "[model_load] Truncated rope_theta in '%s'\n", path);
            fclose(f); return false;
        }
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
    cfg.rope_theta = rope_theta;
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

    fprintf(stderr,
            "[model_load] Loaded '%s'  layers=%d d=%d heads=%d d_ff=%d vocab=%d rope_theta=%g "
            "(SDEC v%u)\n",
            path, cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.vocab_size,
            cfg.rope_theta, version);
    return true;
}

// ============================================================================
// Device: rotary positional embeddings (Llama / GPT-NeoX pair layout)
//
// inv_freq[j] = theta^(-2*j/d_head_per), angle = pos * inv_freq
// Pair (x_{2j}, x_{2j+1}) <- rotation by (cos, sin).
// Applied to Q and K before attention; KV cache stores rotated K (HF-compatible).
// ============================================================================

__device__ void rope_apply_heads_qk_inplace(float* q, float* k, int nh, int dph,
                                            int pos_m, float rope_theta) {
    int tid        = threadIdx.x;
    int nthreads   = blockDim.x;
    const int pairs = dph / 2;
    if (pairs < 1)
        return;

    float theta       = rope_theta > 1e-6f ? rope_theta : 10000.f;
    const float log_theta = logf(theta);

    for (int h = 0; h < nh; h++) {
        int base = h * dph;
        for (int j = tid; j < pairs; j += nthreads) {
            float inv_freq = expf(-log_theta * (float)(2 * j) / (float)dph);
            float angle    = (float)pos_m * inv_freq;
            float c        = cosf(angle);
            float s        = sinf(angle);

            int i0 = base + 2 * j;
            int i1 = base + 2 * j + 1;

            float q0 = q[i0], q1 = q[i1];
            q[i0]    = q0 * c - q1 * s;
            q[i1]    = q0 * s + q1 * c;

            float k0 = k[i0], k1 = k[i1];
            k[i0]    = k0 * c - k1 * s;
            k[i1]    = k0 * s + k1 * c;
        }
        __syncthreads();
    }
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
// Device: Phase 1 of transformer layer — Q/K/V projections + RoPE + KV write.
//
// After return, smem[d .. 2*d) holds Q with RoPE at seq_pos.
// The KV cache has K,V for layer `layer_idx` appended at position seq_pos.
// hidden is not modified.
// ============================================================================
__device__ void model_layer_kv_phase(const ModelWeights& model, int layer_idx,
                                      const float* hidden, KVCache& kv,
                                      int seq_pos, float* smem) {
    int d   = model.cfg.d_model;
    int nh  = model.cfg.n_heads;
    int dph = d / nh;

    const LayerWeights& lw = model.layers[layer_idx];

    float* normed   = smem;
    float* q_all    = smem + d;
    float* kv_temp  = smem + 2 * d;
    float* attn_out = smem + 3 * d;
    float* scratch  = smem + 4 * d;

    device_rmsnorm(hidden, lw.rms_attn_weight, normed, d, scratch);
    device_matvec(normed, lw.Wq, q_all, d, d);
    device_matvec(normed, lw.Wk, attn_out, d, d);   // K → attn_out
    __syncthreads();

    rope_apply_heads_qk_inplace(q_all, attn_out, nh, dph, seq_pos,
                                model.cfg.rope_theta);

    device_matvec(normed, lw.Wv, kv_temp, d, d);    // V → kv_temp
    __syncthreads();

    // Write K (attn_out) and V (kv_temp) to cache at position seq_pos.
    // After this call, smem[d..2d) = q_all with RoPE — preserved for attn phase.
    kv_cache_append(kv, layer_idx, attn_out, kv_temp, seq_pos);
}

// ============================================================================
// Device: Phase 2 of transformer layer — attention over [0..seq_pos] + FFN.
//
// Requires smem[d .. 2*d) = Q with RoPE written by model_layer_kv_phase, and
// the KV cache to already contain K,V at seq_pos (written by kv_phase).
// hidden is updated with the attention + MLP residuals.
// ============================================================================
__device__ void model_layer_attn_mlp_phase(const ModelWeights& model,
                                            int layer_idx,
                                            float* hidden, KVCache& kv,
                                            int seq_pos, float* smem) {
    int tid = threadIdx.x;
    int d   = model.cfg.d_model;
    int dff = model.cfg.d_ff;
    int nh  = model.cfg.n_heads;
    int dph = d / nh;

    const LayerWeights& lw = model.layers[layer_idx];

    // q_all is in smem[d..2d) — left by kv_phase with RoPE applied.
    float* q_all    = smem + d;
    float* kv_temp  = smem + 2 * d;
    float* attn_out = smem + 3 * d;
    float* scratch  = smem + 4 * d;
    float* blk_logits = scratch;

    __shared__ float s_attn_m;
    __shared__ float s_attn_l;
    __shared__ float s_tile_m_new;
    __shared__ float s_attn_alpha;
    __shared__ float s_attn_inv_den;

    // total_len includes the token just written by kv_phase.
    int total_len = seq_pos + 1;
    float scale   = rsqrtf((float)dph);

    // =========================================================
    // ---- Attention sub-layer --------------------------------
    for (int i = tid; i < d; i += blockDim.x) attn_out[i] = 0.0f;
    __syncthreads();

    int n_logical_blocks = (total_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    for (int h = 0; h < nh; h++) {
        int head_off = h * dph;

        if (tid == 0) {
            s_attn_m = -FLT_MAX;
            s_attn_l = 0.f;
        }
        for (int ei = tid; ei < dph; ei += blockDim.x)
            attn_out[head_off + ei] = 0.f;
        __syncthreads();

        for (int log_blk = 0; log_blk < n_logical_blocks; log_blk++) {
            int p0 = log_blk * KV_BLOCK_SIZE;

            for (int sl = tid; sl < KV_BLOCK_SIZE; sl += blockDim.x) {
                int p       = p0 + sl;
                float logit = -FLT_MAX;
                if (p < total_len) {
                    int bidx = p / KV_BLOCK_SIZE, sl_kv = p % KV_BLOCK_SIZE;
                    int pb   = kv.layers[layer_idx].block_table[bidx];
                    const half* base =
                        kv.pool +
                        (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                    const half* k_src = base + sl_kv * kv.d_head + head_off;
                    float dot = 0.f;
                    for (int e = 0; e < dph; e++)
                        dot += q_all[head_off + e] * __half2float(k_src[e]);
                    logit = dot * scale;
                }
                blk_logits[sl] = logit;
            }
            __syncthreads();

            if (tid == 0) {
                float m_tile = -FLT_MAX;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++) {
                    int p = p0 + sl;
                    if (p < total_len)
                        m_tile = fmaxf(m_tile, blk_logits[sl]);
                }
                float m_prev  = s_attn_m;
                float m_new   = fmaxf(m_prev, m_tile);
                float alpha_u = (log_blk > 0) ? expf(m_prev - m_new) : 1.f;
                float l_tile  = 0.f;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++) {
                    int p = p0 + sl;
                    if (p >= total_len) continue;
                    l_tile += expf(blk_logits[sl] - m_new);
                }
                s_attn_m     = m_new;
                s_attn_l     = alpha_u * s_attn_l + l_tile;
                s_tile_m_new = m_new;
                s_attn_alpha = alpha_u;
            }
            __syncthreads();

            float rescale_prior = s_attn_alpha;
            for (int ei = tid; ei < dph; ei += blockDim.x)
                attn_out[head_off + ei] *= rescale_prior;
            __syncthreads();

            float m_here = s_tile_m_new;
            for (int ei = tid; ei < dph; ei += blockDim.x) {
                int global_e = head_off + ei;
                float acc_c  = 0.f;
                for (int sl = 0; sl < KV_BLOCK_SIZE; sl++) {
                    int p = p0 + sl;
                    if (p >= total_len) continue;
                    float w   = expf(blk_logits[sl] - m_here);
                    int bidx  = p / KV_BLOCK_SIZE, sl_kv = p % KV_BLOCK_SIZE;
                    int pb    = kv.layers[layer_idx].block_table[bidx];
                    const half* base =
                        kv.pool +
                        (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                    const half* v_src =
                        base + KV_BLOCK_SIZE * kv.d_head + sl_kv * kv.d_head;
                    acc_c += w * __half2float(v_src[global_e]);
                }
                attn_out[global_e] += acc_c;
            }
            __syncthreads();
        }

        if (tid == 0)
            s_attn_inv_den =
                (s_attn_l > 1e-20f && isfinite(s_attn_l))
                    ? (1.f / s_attn_l) : 0.f;
        __syncthreads();

        float norm_den = s_attn_inv_den;
        for (int ei = tid; ei < dph; ei += blockDim.x)
            attn_out[head_off + ei] *= norm_den;
        __syncthreads();
    }

    device_matvec(attn_out, lw.Wo, kv_temp, d, d);
    for (int i = tid; i < d; i += blockDim.x) hidden[i] += kv_temp[i];
    __syncthreads();

    // =========================================================
    // ---- MLP (tiled SwiGLU)
    // =========================================================
    // normed = smem (safe to reuse: kv_phase Q at smem+d will become mlp_accum)
    float* normed    = smem;
    float* mlp_accum = q_all;   // q_all slot reused after attention is complete

    int tile_ff = MLP_FF_TILE < d ? MLP_FF_TILE : d;

    device_rmsnorm(hidden, lw.rms_mlp_weight, normed, d, scratch);
    for (int qi = tid; qi < d; qi += blockDim.x) mlp_accum[qi] = 0.f;
    __syncthreads();

    for (int r0 = 0; r0 < dff; r0 += tile_ff) {
        int ncol = tile_ff < (dff - r0) ? tile_ff : (dff - r0);

        device_matvec_cols(normed, lw.W_gate, d, dff, r0, ncol, attn_out);
        device_matvec_cols(normed, lw.W_up,   d, dff, r0, ncol, kv_temp);

        for (int jc = tid; jc < ncol; jc += blockDim.x) {
            float g      = attn_out[jc];
            attn_out[jc] = (g / (1.0f + expf(-g))) * kv_temp[jc];
        }
        __syncthreads();

        for (int oc = tid; oc < d; oc += blockDim.x) {
            float dot_d = 0.f;
            for (int jc = 0; jc < ncol; jc++) {
                int row = r0 + jc;
                dot_d += attn_out[jc] * __half2float(lw.W_down[row * d + oc]);
            }
            mlp_accum[oc] += dot_d;
        }
        __syncthreads();
    }

    for (int i = tid; i < d; i += blockDim.x) hidden[i] += mlp_accum[i];
    __syncthreads();
}

// ============================================================================
// Device: multi-head attention + SwiGLU MLP transformer layer
//
// Thin wrapper — calls kv_phase then attn_mlp_phase sequentially.
// Use the two phase functions directly in cooperative multi-block kernels.
// ============================================================================
__device__ void model_layer_forward(const ModelWeights& model, int layer_idx,
                                    float* hidden, KVCache& kv,
                                    int current_seq_len, float* smem) {
    model_layer_kv_phase(model, layer_idx, hidden, kv, current_seq_len, smem);
    model_layer_attn_mlp_phase(model, layer_idx, hidden, kv, current_seq_len, smem);
}

// ============================================================================
// Device: final RMSNorm + output projection
// g_logits is a global-memory buffer of vocab_size floats.
// ============================================================================

__device__ void model_output(const ModelWeights& model,
                             const float* hidden, float* g_logits, float* smem) {
    int d = model.cfg.d_model;
    int V = model.cfg.vocab_size;

    // Reuse normed; block reductions write into scratch at smem+d (after layers done)
    float* normed  = smem;           // [d]
    float* scratch = smem + d;

    device_rmsnorm(hidden, model.rms_final_weight, normed, d, scratch);
    // Write logits directly to global memory (vocab_size may be >> shared memory)
    device_matvec(normed, model.output_proj, g_logits, d, V);
}

// ============================================================================
// Device: full single-token forward (logits only, for sampling kernels)
// ============================================================================

__device__ void model_forward_logits(const ModelWeights& model, KVCache& kv,
                                     int token_id, int current_seq_len,
                                     float* hidden, float* g_logits,
                                     float* smem) {
    model_embed(model, token_id, hidden);

    for (int l = 0; l < model.cfg.n_layers; l++)
        model_layer_forward(model, l, hidden, kv, current_seq_len, smem);

    model_output(model, hidden, g_logits, smem);
}

// ============================================================================
// Device: full single-token forward pass
// Returns the greedy next-token id.
// ============================================================================

__device__ int model_forward(const ModelWeights& model, KVCache& kv,
                             int token_id, int current_seq_len,
                             float* hidden, float* g_logits, float* smem) {
    model_forward_logits(model, kv, token_id, current_seq_len, hidden,
                         g_logits, smem);

    return global_argmax(g_logits, model.cfg.vocab_size, smem);
}

// ============================================================================
// Device: batched forward — process B tokens, reading each weight matrix once.
//
// g_work layout (all B-major, floats):
//   [0          .. B*d)         g_normed
//   [B*d        .. 2*B*d)       g_q
//   [2*B*d      .. 3*B*d)       g_k
//   [3*B*d      .. 4*B*d)       g_v  (reused for Wo output after attention)
//   [4*B*d      .. 5*B*d)       g_tmp (attn_out staging / gate activation)
//   [5*B*d      .. 6*B*d)       g_mlp_acc
//   [6*B*d      .. 6*B*d+B*T)   g_mlp_up  (T = MLP_FF_TILE)
// ============================================================================

__device__ void model_batch_forward_logits(
    const ModelWeights& model, KVCache& kv,
    const int* token_ids, int seq_base, int B,
    float* g_hidden, float* g_work, float* g_logits_out,
    float* smem)
{
    const int tid = threadIdx.x;
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int nh  = model.cfg.n_heads;
    const int dph = d / nh;
    const int V   = model.cfg.vocab_size;

    float* g_normed  = g_work;
    float* g_q       = g_work + 1 * B * d;
    float* g_k       = g_work + 2 * B * d;
    float* g_v       = g_work + 3 * B * d;
    float* g_tmp     = g_work + 4 * B * d;
    float* g_mlp_acc = g_work + 5 * B * d;
    float* g_mlp_up  = g_work + 6 * B * d;

    float* s_buf0   = smem;           // [d] — normed output / hidden load
    float* s_buf1   = smem + d;       // [d] — Q / q_all
    float* s_buf2   = smem + 2 * d;   // [d] — kv_temp
    float* s_buf3   = smem + 3 * d;   // [d] — attn_out / hidden load
    float* s_red    = smem + 4 * d;   // reduction / blk_logits scratch

    // ---- Embed all B tokens → g_hidden ----
    for (int b = 0; b < B; b++) {
        int tok = token_ids[b];
        for (int i = tid; i < d; i += blockDim.x)
            g_hidden[b * d + i] = __half2float(
                model.token_embedding[tok * d + i]);
    }
    __syncthreads();

    // ---- Layer loop ----
    for (int l = 0; l < model.cfg.n_layers; l++) {
        const LayerWeights& lw = model.layers[l];

        // ===== Batch RMSNorm (attention) → g_normed =====
        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = g_hidden[b * d + i];
            __syncthreads();
            device_rmsnorm(s_buf3, lw.rms_attn_weight, s_buf0, d, s_red);
            for (int i = tid; i < d; i += blockDim.x)
                g_normed[b * d + i] = s_buf0[i];
            __syncthreads();
        }

        // ===== Batched Q/K/V projections (each weight read ONCE) =====
        device_matvec_batched(g_normed, lw.Wq, g_q, d, d, B);
        device_matvec_batched(g_normed, lw.Wk, g_k, d, d, B);
        device_matvec_batched(g_normed, lw.Wv, g_v, d, d, B);

        // ===== Per-token: RoPE + KV cache append =====
        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x) {
                s_buf1[i] = g_q[b * d + i];
                s_buf3[i] = g_k[b * d + i];
            }
            __syncthreads();

            rope_apply_heads_qk_inplace(s_buf1, s_buf3, nh, dph,
                                        seq_base + b, model.cfg.rope_theta);

            for (int i = tid; i < d; i += blockDim.x)
                s_buf2[i] = g_v[b * d + i];
            __syncthreads();

            kv_cache_append(kv, l, s_buf3, s_buf2, seq_base + b);

            for (int i = tid; i < d; i += blockDim.x)
                g_q[b * d + i] = s_buf1[i];
            __syncthreads();
        }

        // ===== Per-token attention → g_tmp =====
        __shared__ float s_attn_m_b;
        __shared__ float s_attn_l_b;
        __shared__ float s_tile_m_b;
        __shared__ float s_alpha_b;
        __shared__ float s_inv_den_b;

        float scale = rsqrtf((float)dph);

        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf1[i] = g_q[b * d + i];
            __syncthreads();

            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = 0.f;
            __syncthreads();

            int total_len = seq_base + b + 1;
            int n_log_blks = (total_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

            for (int h = 0; h < nh; h++) {
                int ho = h * dph;
                if (tid == 0) { s_attn_m_b = -FLT_MAX; s_attn_l_b = 0.f; }
                for (int e = tid; e < dph; e += blockDim.x)
                    s_buf3[ho + e] = 0.f;
                __syncthreads();

                for (int lb = 0; lb < n_log_blks; lb++) {
                    int p0 = lb * KV_BLOCK_SIZE;
                    for (int sl = tid; sl < KV_BLOCK_SIZE; sl += blockDim.x) {
                        int p = p0 + sl;
                        float logit = -FLT_MAX;
                        if (p < total_len) {
                            int bi = p / KV_BLOCK_SIZE, si = p % KV_BLOCK_SIZE;
                            int pb = kv.layers[l].block_table[bi];
                            const half* base = kv.pool +
                                (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                            const half* ks = base + si * kv.d_head + ho;
                            float dot = 0.f;
                            for (int e = 0; e < dph; e++)
                                dot += s_buf1[ho + e] * __half2float(ks[e]);
                            logit = dot * scale;
                        }
                        s_red[sl] = logit;
                    }
                    __syncthreads();

                    if (tid == 0) {
                        float mt = -FLT_MAX;
                        for (int sl = 0; sl < KV_BLOCK_SIZE; sl++)
                            if (p0 + sl < total_len)
                                mt = fmaxf(mt, s_red[sl]);
                        float mp = s_attn_m_b;
                        float mn = fmaxf(mp, mt);
                        float au = (lb > 0) ? expf(mp - mn) : 1.f;
                        float lt = 0.f;
                        for (int sl = 0; sl < KV_BLOCK_SIZE; sl++)
                            if (p0 + sl < total_len)
                                lt += expf(s_red[sl] - mn);
                        s_attn_m_b = mn;
                        s_attn_l_b = au * s_attn_l_b + lt;
                        s_tile_m_b = mn;
                        s_alpha_b  = au;
                    }
                    __syncthreads();

                    float rp = s_alpha_b;
                    for (int e = tid; e < dph; e += blockDim.x)
                        s_buf3[ho + e] *= rp;
                    __syncthreads();

                    float mh = s_tile_m_b;
                    for (int e = tid; e < dph; e += blockDim.x) {
                        int ge = ho + e;
                        float ac = 0.f;
                        for (int sl = 0; sl < KV_BLOCK_SIZE; sl++) {
                            int p = p0 + sl;
                            if (p >= total_len) continue;
                            float w = expf(s_red[sl] - mh);
                            int bi = p / KV_BLOCK_SIZE, si = p % KV_BLOCK_SIZE;
                            int pb = kv.layers[l].block_table[bi];
                            const half* base = kv.pool +
                                (size_t)pb * 2 * KV_BLOCK_SIZE * kv.d_head;
                            const half* vs = base +
                                KV_BLOCK_SIZE * kv.d_head + si * kv.d_head;
                            ac += w * __half2float(vs[ge]);
                        }
                        s_buf3[ge] += ac;
                    }
                    __syncthreads();
                }

                if (tid == 0)
                    s_inv_den_b = (s_attn_l_b > 1e-20f && isfinite(s_attn_l_b))
                        ? (1.f / s_attn_l_b) : 0.f;
                __syncthreads();
                float nd = s_inv_den_b;
                for (int e = tid; e < dph; e += blockDim.x)
                    s_buf3[ho + e] *= nd;
                __syncthreads();
            }

            for (int i = tid; i < d; i += blockDim.x)
                g_tmp[b * d + i] = s_buf3[i];
            __syncthreads();
        }

        // ===== Batched Wo projection (read ONCE) =====
        device_matvec_batched(g_tmp, lw.Wo, g_v, d, d, B);

        for (int b = 0; b < B; b++)
            for (int i = tid; i < d; i += blockDim.x)
                g_hidden[b * d + i] += g_v[b * d + i];
        __syncthreads();

        // ===== Batch MLP RMSNorm → g_normed =====
        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = g_hidden[b * d + i];
            __syncthreads();
            device_rmsnorm(s_buf3, lw.rms_mlp_weight, s_buf0, d, s_red);
            for (int i = tid; i < d; i += blockDim.x)
                g_normed[b * d + i] = s_buf0[i];
            __syncthreads();
        }

        // Zero MLP accumulators
        for (int b = 0; b < B; b++)
            for (int i = tid; i < d; i += blockDim.x)
                g_mlp_acc[b * d + i] = 0.f;
        __syncthreads();

        // ===== Batched tiled MLP =====
        int tile_ff = MLP_FF_TILE < d ? MLP_FF_TILE : d;
        for (int r0 = 0; r0 < dff; r0 += tile_ff) {
            int ncol = tile_ff < (dff - r0) ? tile_ff : (dff - r0);

            device_matvec_cols_batched(g_normed, lw.W_gate,
                                       d, dff, r0, ncol, g_tmp, B);
            device_matvec_cols_batched(g_normed, lw.W_up,
                                       d, dff, r0, ncol, g_mlp_up, B);

            for (int b = 0; b < B; b++) {
                for (int jc = tid; jc < ncol; jc += blockDim.x) {
                    float g = g_tmp[b * ncol + jc];
                    g_tmp[b * ncol + jc] =
                        (g / (1.0f + expf(-g))) * g_mlp_up[b * ncol + jc];
                }
            }
            __syncthreads();

            device_down_proj_accum_batched(g_tmp, lw.W_down,
                                           d, dff, r0, ncol, g_mlp_acc, B);
        }

        for (int b = 0; b < B; b++)
            for (int i = tid; i < d; i += blockDim.x)
                g_hidden[b * d + i] += g_mlp_acc[b * d + i];
        __syncthreads();
    }

    // ---- Batched output: RMSNorm all → g_normed, then output_proj once ----
    for (int b = 0; b < B; b++) {
        for (int i = tid; i < d; i += blockDim.x)
            s_buf3[i] = g_hidden[b * d + i];
        __syncthreads();
        device_rmsnorm(s_buf3, model.rms_final_weight, s_buf0, d, s_red);
        for (int i = tid; i < d; i += blockDim.x)
            g_normed[b * d + i] = s_buf0[i];
        __syncthreads();
    }
    device_matvec_batched(g_normed, model.output_proj, g_logits_out, d, V, B);
}
