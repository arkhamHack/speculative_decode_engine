#include "model.h"
#include "attention.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cublas_v2.h>

// Host: allocate weight tensors on the GPU (16-byte aligned)

static half* alloc_half(size_t n) {
    half* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, align_up(n * sizeof(half), 16)));
    return ptr;
}

void model_alloc(ModelWeights& model, const ModelConfig& cfg) {
    model.cfg = cfg;
    const int d   = cfg.d_model;
    const int kd  = kv_dim(cfg);     // n_kv_heads * head_dim  (= d when n_kv_heads == n_heads)
    const int dff = cfg.d_ff;
    const int V   = cfg.vocab_size;

    model.token_embedding  = alloc_half((size_t)V * d);
    model.rms_final_weight = alloc_half(d);
    model.output_proj      = alloc_half((size_t)d * V);

    for (int l = 0; l < cfg.n_layers; l++) {
        LayerWeights& lw = model.layers[l];
        lw.rms_attn_weight = alloc_half(d);
        lw.Wq              = alloc_half((size_t)d * d);
        lw.Wk              = alloc_half((size_t)d * kd);
        lw.Wv              = alloc_half((size_t)d * kd);
        lw.Wo              = alloc_half((size_t)d * d);
        lw.rms_mlp_weight  = alloc_half(d);
        lw.W_gate          = alloc_half((size_t)d * dff);
        lw.W_up            = alloc_half((size_t)d * dff);
        lw.W_down          = alloc_half((size_t)dff * d);
        lw.Wq_bias = nullptr;
        lw.Wk_bias = nullptr;
        lw.Wv_bias = nullptr;
        if (cfg.has_qkv_bias) {
            lw.Wq_bias = alloc_half(d);
            lw.Wk_bias = alloc_half(kd);
            lw.Wv_bias = alloc_half(kd);
        }
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
        safe_free(lw.Wq_bias); safe_free(lw.Wk_bias); safe_free(lw.Wv_bias);
    }
}


// Host: load weights from an SDEC binary file
//
// Supported versions:
//   v1: n_layers, d_model, n_heads, d_ff, vocab_size
//       Wk/Wv are [d_model, d_model] (GQA expanded at export time)
//   v2: adds rope_theta (float32) after vocab_size
//       Wk/Wv still expanded
//   v3: header has n_kv_heads, rope_theta, rope_scaling_type, rope_scaling_factor
//       Wk/Wv are [d_model, kv_dim] (native GQA dimensions)
//   v4: adds attention_type (int32) — 0=softmax, 1=sliding_window, 2=kda, 3=linear
//   v5: adds has_qkv_bias (int32); when non-zero each layer also has
//       Wq_bias[d], Wk_bias[kv_dim], Wv_bias[kv_dim] after Wv
//
// All data values are float16 (2 bytes each).

bool model_load_weights(ModelWeights& model, const char* path,
                        ModelConfig* cfg_out) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[model_load] Cannot open '%s'\n", path);
        return false;
    }

    char magic[5] = {0};
    if (fread(magic, 1, 4, f) != 4 || strncmp(magic, "SDEC", 4) != 0) {
        fprintf(stderr, "[model_load] Bad magic in '%s'\n", path);
        fclose(f); return false;
    }

    uint32_t version;
    if (fread(&version, 4, 1, f) != 1) {
        fprintf(stderr, "[model_load] Truncated header in '%s'\n", path);
        fclose(f); return false;
    }
    if (version < 1u || version > 5u) {
        fprintf(stderr, "[model_load] Unsupported SDEC version %u (expected 1-5)\n",
                version);
        fclose(f); return false;
    }

    uint32_t n_layers, d_model, n_heads, n_kv_heads_file = 0, d_ff, vocab_size;

    if (version <= 2u) {
        if (fread(&n_layers,   4, 1, f) != 1 ||
            fread(&d_model,    4, 1, f) != 1 ||
            fread(&n_heads,    4, 1, f) != 1 ||
            fread(&d_ff,       4, 1, f) != 1 ||
            fread(&vocab_size, 4, 1, f) != 1) {
            fprintf(stderr, "[model_load] Truncated v%u header in '%s'\n", version, path);
            fclose(f); return false;
        }
        n_kv_heads_file = n_heads;   // v1/v2: GQA was expanded, treat as MHA
    } else {
        if (fread(&n_layers,        4, 1, f) != 1 ||
            fread(&d_model,         4, 1, f) != 1 ||
            fread(&n_heads,         4, 1, f) != 1 ||
            fread(&n_kv_heads_file, 4, 1, f) != 1 ||
            fread(&d_ff,            4, 1, f) != 1 ||
            fread(&vocab_size,      4, 1, f) != 1) {
            fprintf(stderr, "[model_load] Truncated v3 header in '%s'\n", path);
            fclose(f); return false;
        }
    }

    float rope_theta = 10000.f;
    if (version >= 2u) {
        if (fread(&rope_theta, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "[model_load] Truncated rope_theta in '%s'\n", path);
            fclose(f); return false;
        }
    }

    int32_t rope_scaling_type_i = 0;
    float   rope_scaling_factor = 1.0f;
    if (version >= 3u) {
        if (fread(&rope_scaling_type_i, 4, 1, f) != 1 ||
            fread(&rope_scaling_factor, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "[model_load] Truncated rope_scaling in '%s'\n", path);
            fclose(f); return false;
        }
    }

    int32_t attention_type_i = 0;   // ATTN_SOFTMAX
    if (version >= 4u) {
        if (fread(&attention_type_i, 4, 1, f) != 1) {
            fprintf(stderr, "[model_load] Truncated attention_type in '%s'\n", path);
            fclose(f); return false;
        }
    }

    int32_t has_qkv_bias_i = 0;
    if (version >= 5u) {
        if (fread(&has_qkv_bias_i, 4, 1, f) != 1) {
            fprintf(stderr, "[model_load] Truncated has_qkv_bias in '%s'\n", path);
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
    if ((int)n_heads % (int)n_kv_heads_file != 0) {
        fprintf(stderr, "[model_load] n_heads=%u not divisible by n_kv_heads=%u\n",
                n_heads, n_kv_heads_file);
        fclose(f); return false;
    }

    ModelConfig cfg;
    cfg.n_layers   = (int)n_layers;
    cfg.d_model    = (int)d_model;
    cfg.n_heads    = (int)n_heads;
    cfg.n_kv_heads = (int)n_kv_heads_file;
    cfg.d_ff       = (int)d_ff;
    cfg.vocab_size = (int)vocab_size;
    cfg.rope_theta = rope_theta;
    cfg.rope_scaling_type   = (RopeScalingType)rope_scaling_type_i;
    cfg.rope_scaling_factor = rope_scaling_factor;
    cfg.attention_type      = (AttentionType)attention_type_i;
    cfg.has_qkv_bias        = (has_qkv_bias_i != 0);

    // For v1/v2 (expanded GQA), n_kv_heads == n_heads, so kv_dim == d_model.
    // For v3 (native GQA), kv_dim = n_kv_heads * head_dim.
    const int kd = kv_dim(cfg);

    if (cfg_out) *cfg_out = cfg;
    model_alloc(model, cfg);

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
        auto nm = [&](const char* s) { snprintf(name, sizeof(name), "L%d.%s", l, s); return name; };
        ok =
            read_tensor(lw.rms_attn_weight, (size_t)d,       nm("rms_attn")) &&
            read_tensor(lw.Wq,              (size_t)d * d,   nm("Wq"))       &&
            read_tensor(lw.Wk,              (size_t)d * kd,  nm("Wk"))       &&
            read_tensor(lw.Wv,              (size_t)d * kd,  nm("Wv"))       &&
            read_tensor(lw.Wo,              (size_t)d * d,   nm("Wo"))       &&
            read_tensor(lw.rms_mlp_weight,  (size_t)d,       nm("rms_mlp")) &&
            read_tensor(lw.W_gate,          (size_t)d * dff, nm("W_gate"))   &&
            read_tensor(lw.W_up,            (size_t)d * dff, nm("W_up"))     &&
            read_tensor(lw.W_down,          (size_t)dff * d, nm("W_down"));
        if (ok && cfg.has_qkv_bias) {
            ok =
                read_tensor(lw.Wq_bias, (size_t)d,  nm("Wq_bias")) &&
                read_tensor(lw.Wk_bias, (size_t)kd, nm("Wk_bias")) &&
                read_tensor(lw.Wv_bias, (size_t)kd, nm("Wv_bias"));
        }
    }

    fclose(f);
    if (!ok) {
        fprintf(stderr, "[model_load] Failed — freeing partial allocations\n");
        model_free(model);
        return false;
    }

    const char* rope_names[] = {"none","linear","ntk","yarn"};
    const char* attn_names[] = {"softmax","sliding_window","kda","linear"};
    fprintf(stderr,
            "[model_load] Loaded '%s'  layers=%d d=%d heads=%d kv_heads=%d d_ff=%d "
            "vocab=%d rope_theta=%g rope=%s(%.2f) attn=%s qkv_bias=%s (SDEC v%u)\n",
            path, cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
            cfg.d_ff, cfg.vocab_size, cfg.rope_theta,
            rope_names[cfg.rope_scaling_type & 3], cfg.rope_scaling_factor,
            attn_names[cfg.attention_type & 3],
            cfg.has_qkv_bias ? "yes" : "no",
            version);
    return true;
}

// Device: rotary positional embeddings with optional scaling
//
// Supports: none (standard), linear, NTK-aware, YaRN
// Applied separately to Q (n_heads) and K (n_kv_heads) for GQA support.

__device__ void rope_apply_inplace(float* x, int n_heads_x, int dph,
                                   int pos, float rope_theta,
                                   RopeScalingType scaling_type,
                                   float scaling_factor) {
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;
    const int pairs = dph / 2;
    if (pairs < 1) return;

    float theta = rope_theta > 1e-6f ? rope_theta : 10000.f;

    // NTK-aware: scale theta so effective context extends by factor
    if (scaling_type == ROPE_SCALING_NTK && scaling_factor > 1.0f)
        theta *= powf(scaling_factor, (float)dph / ((float)dph - 2.f));

    const float log_theta = logf(theta);

    // Linear and YaRN: scale the position index
    float pos_scaled = (float)pos;
    if ((scaling_type == ROPE_SCALING_LINEAR || scaling_type == ROPE_SCALING_YARN)
        && scaling_factor > 1.0f)
        pos_scaled = (float)pos / scaling_factor;

    for (int h = 0; h < n_heads_x; h++) {
        int base = h * dph;
        for (int j = tid; j < pairs; j += nthreads) {
            float inv_freq = expf(-log_theta * (float)(2 * j) / (float)dph);
            float angle    = pos_scaled * inv_freq;
            float c = cosf(angle);
            float s = sinf(angle);

            // rotate_half (HF/Llama/Qwen): pair dim j with dim j+dph/2,
            // NOT the interleaved (2j, 2j+1) GPT-J layout — matches the
            // unpermuted HF weight layout the exporter writes.
            int i0 = base + j;
            int i1 = base + j + pairs;
            float x0 = x[i0], x1 = x[i1];
            x[i0] = x0 * c - x1 * s;
            x[i1] = x0 * s + x1 * c;
        }
        __syncthreads();
    }
}

// Legacy combined Q+K wrapper (MHA only — both arrays must have the same head count)
__device__ void rope_apply_heads_qk_inplace(float* q, float* k, int nh, int dph,
                                            int pos_m, float rope_theta) {
    rope_apply_inplace(q, nh, dph, pos_m, rope_theta, ROPE_SCALING_NONE, 1.0f);
    rope_apply_inplace(k, nh, dph, pos_m, rope_theta, ROPE_SCALING_NONE, 1.0f);
}

__device__ void model_embed(const ModelWeights& model, int token_id,
                            float* hidden) {
    int tid = threadIdx.x;
    int d   = model.cfg.d_model;
    for (int i = tid; i < d; i += blockDim.x)
        hidden[i] = __half2float(model.token_embedding[token_id * d + i]);
    __syncthreads();
}

// Device: Phase 1 of transformer layer — Q/K/V projections + RoPE + KV write.
//
// After return, smem[d .. 2*d) holds Q with RoPE at seq_pos.
// The KV cache has K,V for layer `layer_idx` appended at position seq_pos.
// hidden is not modified.
__device__ void model_layer_kv_phase(const ModelWeights& model, int layer_idx,
                                      const float* hidden, KVCache& kv,
                                      int seq_pos, float* smem) {
    int d    = model.cfg.d_model;
    int nh   = model.cfg.n_heads;
    int nkv  = model.cfg.n_kv_heads;
    int dph  = d / nh;
    int kd   = nkv * dph;   // kv_dim

    const LayerWeights& lw = model.layers[layer_idx];

    float* normed   = smem;
    float* q_all    = smem + d;
    float* kv_temp  = smem + 2 * d;        // [kd] K, then reused for V
    float* attn_out = smem + 2 * d + kd;   // [kd] temporary for K during RoPE
    float* scratch  = smem + 2 * d + 2 * kd;

    device_rmsnorm(hidden, lw.rms_attn_weight, normed, d, scratch);
    device_matvec(normed, lw.Wq, q_all, d, d);
    device_add_bias(q_all, lw.Wq_bias, d);
    device_matvec(normed, lw.Wk, kv_temp, d, kd);   // K → kv_temp [kd]
    device_add_bias(kv_temp, lw.Wk_bias, kd);

    rope_apply_inplace(q_all, nh, dph, seq_pos, model.cfg.rope_theta,
                       model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);
    rope_apply_inplace(kv_temp, nkv, dph, seq_pos, model.cfg.rope_theta,
                       model.cfg.rope_scaling_type, model.cfg.rope_scaling_factor);

    // Save K (kv_temp) into attn_out temporarily so we can reuse kv_temp for V
    for (int i = threadIdx.x; i < kd; i += blockDim.x)
        attn_out[i] = kv_temp[i];
    __syncthreads();

    device_matvec(normed, lw.Wv, kv_temp, d, kd);   // V → kv_temp [kd]
    device_add_bias(kv_temp, lw.Wv_bias, kd);

    kv_cache_append(kv, layer_idx, attn_out, kv_temp, seq_pos);
    // After return: smem[d..2d) = q_all with RoPE — preserved for attn phase.
}

// Device: Phase 2 of transformer layer — attention over [0..seq_pos] + FFN.
//
// Requires smem[d .. 2*d) = Q with RoPE written by model_layer_kv_phase, and
// the KV cache to already contain K,V at seq_pos (written by kv_phase).
// hidden is updated with the attention + MLP residuals.
__device__ void model_layer_attn_mlp_phase(const ModelWeights& model,
                                            int layer_idx,
                                            float* hidden, KVCache& kv,
                                            int seq_pos, float* smem) {
    int tid  = threadIdx.x;
    int d    = model.cfg.d_model;
    int dff  = model.cfg.d_ff;
    int nh   = model.cfg.n_heads;
    int nkv  = model.cfg.n_kv_heads;
    int dph  = d / nh;
    int kd   = nkv * dph;     // kv_dim stored in cache

    const LayerWeights& lw = model.layers[layer_idx];

    // smem layout must match kv_phase output:
    //   smem[0..d)        = normed (reusable)
    //   smem[d..2d)       = q_all with RoPE
    //   smem[2d..2d+kd)   = (used by kv_phase, free now)
    //   smem[2d+kd..)     = scratch
    float* q_all    = smem + d;
    float* attn_out = smem;           // reuse normed slot for attention output [d]
    float* wo_buf   = smem + d;       // reuse q_all slot after attention for Wo result [d]
    float* scratch  = smem + 2 * d + 2 * kd;
    float* blk_logits = scratch;

    __shared__ float s_attn_m;
    __shared__ float s_attn_l;
    __shared__ float s_tile_m_new;
    __shared__ float s_attn_alpha;
    __shared__ float s_attn_inv_den;

    int total_len = seq_pos + 1;

    for (int i = tid; i < d; i += blockDim.x) attn_out[i] = 0.0f;
    __syncthreads();

    device_attention_dispatch(
        model.cfg.attention_type,
        q_all, attn_out, kv, layer_idx,
        nh, nkv, dph, total_len,
        blk_logits,
        s_attn_m, s_attn_l, s_tile_m_new, s_attn_alpha, s_attn_inv_den);

    // Wo projection: attn_out[d] → wo_buf[d] (q_all slot, safe after attention)
    device_matvec(attn_out, lw.Wo, wo_buf, d, d);
    for (int i = tid; i < d; i += blockDim.x) hidden[i] += wo_buf[i];
    __syncthreads();

    // ---- MLP (tiled SwiGLU) ----
    float* normed    = smem;        // reuse slot 0
    float* mlp_accum = smem + d;    // reuse q_all/wo_buf slot
    float* gate_buf  = smem + 2 * d;
    // up_buf must clear gate_buf's full tile width (MLP_FF_TILE), not kd — under
    // GQA kd can be < MLP_FF_TILE, which would alias gate_buf into up_buf.
    float* up_buf    = smem + 2 * d + MLP_FF_TILE;

    int tile_ff = MLP_FF_TILE < d ? MLP_FF_TILE : d;

    device_rmsnorm(hidden, lw.rms_mlp_weight, normed, d, scratch);
    for (int qi = tid; qi < d; qi += blockDim.x) mlp_accum[qi] = 0.f;
    __syncthreads();

    for (int r0 = 0; r0 < dff; r0 += tile_ff) {
        int ncol = tile_ff < (dff - r0) ? tile_ff : (dff - r0);

        device_matvec_cols(normed, lw.W_gate, d, dff, r0, ncol, gate_buf);
        device_matvec_cols(normed, lw.W_up,   d, dff, r0, ncol, up_buf);

        for (int jc = tid; jc < ncol; jc += blockDim.x) {
            float g       = gate_buf[jc];
            gate_buf[jc]  = (g / (1.0f + expf(-g))) * up_buf[jc];
        }
        __syncthreads();

        for (int oc = tid; oc < d; oc += blockDim.x) {
            float dot_d = 0.f;
            for (int jc = 0; jc < ncol; jc++) {
                int row = r0 + jc;
                dot_d += gate_buf[jc] * __half2float(lw.W_down[row * d + oc]);
            }
            mlp_accum[oc] += dot_d;
        }
        __syncthreads();
    }

    for (int i = tid; i < d; i += blockDim.x) hidden[i] += mlp_accum[i];
    __syncthreads();
}

// Device: multi-head attention + SwiGLU MLP transformer layer
//
// Thin wrapper — calls kv_phase then attn_mlp_phase sequentially.
// Use the two phase functions directly in cooperative multi-block kernels.
__device__ void model_layer_forward(const ModelWeights& model, int layer_idx,
                                    float* hidden, KVCache& kv,
                                    int current_seq_len, float* smem) {
    model_layer_kv_phase(model, layer_idx, hidden, kv, current_seq_len, smem);
    model_layer_attn_mlp_phase(model, layer_idx, hidden, kv, current_seq_len, smem);
}

// Device: final RMSNorm + output projection
// g_logits is a global-memory buffer of vocab_size floats.

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

// Device: full single-token forward (logits only, for sampling kernels)

__device__ void model_forward_logits(const ModelWeights& model, KVCache& kv,
                                     int token_id, int current_seq_len,
                                     float* hidden, float* g_logits,
                                     float* smem) {
    model_embed(model, token_id, hidden);

    for (int l = 0; l < model.cfg.n_layers; l++)
        model_layer_forward(model, l, hidden, kv, current_seq_len, smem);

    model_output(model, hidden, g_logits, smem);
}

// Device: full single-token forward pass
// Returns the greedy next-token id.

__device__ int model_forward(const ModelWeights& model, KVCache& kv,
                             int token_id, int current_seq_len,
                             float* hidden, float* g_logits, float* smem) {
    model_forward_logits(model, kv, token_id, current_seq_len, hidden,
                         g_logits, smem);

    return global_argmax(g_logits, model.cfg.vocab_size, smem);
}

// Self-spec partial-forward helpers

__device__ void device_copy_hidden_to_global(const float* hidden, float* g_out,
                                             int d) {
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        g_out[i] = hidden[i];
    __syncthreads();
}

__device__ void model_forward_draft_prefix_save(
    const ModelWeights& model, KVCache& kv,
    int token_id, int current_seq_len,
    int n_prefix_layers,
    float* hidden, float* g_saved_hidden, float* smem) {
    model_embed(model, token_id, hidden);

    for (int l = 0; l < n_prefix_layers; l++)
        model_layer_forward(model, l, hidden, kv, current_seq_len, smem);

    device_copy_hidden_to_global(hidden, g_saved_hidden, model.cfg.d_model);
}

__device__ void model_forward_draft_logits(const ModelWeights& model, KVCache& kv,
                                           int token_id, int current_seq_len,
                                           int n_prefix_layers,
                                           float* hidden, float* g_logits,
                                           float* g_saved_hidden, float* smem) {
    model_forward_draft_prefix_save(
        model, kv, token_id, current_seq_len, n_prefix_layers,
        hidden, g_saved_hidden, smem);

    for (int l = n_prefix_layers; l < model.cfg.n_layers; l++)
        model_layer_forward(model, l, hidden, kv, current_seq_len, smem);

    model_output(model, hidden, g_logits, smem);
}

__device__ void model_forward_verify_from_hidden_logits(
    const ModelWeights& model, KVCache& kv,
    const float* saved_hidden, int layer_start, int current_seq_len,
    float* hidden, float* g_logits, float* smem) {
    const int d = model.cfg.d_model;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        hidden[i] = saved_hidden[i];
    __syncthreads();

    for (int l = layer_start; l < model.cfg.n_layers; l++)
        model_layer_forward(model, l, hidden, kv, current_seq_len, smem);

    model_output(model, hidden, g_logits, smem);
}

// Device: batched forward — process B tokens, reading each weight matrix once.
//
// g_work layout (all B-major, floats):
//   [0          .. B*d)         g_normed
//   [B*d        .. 2*B*d)       g_q
//   [2*B*d      .. 3*B*d)       g_k  (only B*kd used; stride=kd)
//   [3*B*d      .. 4*B*d)       g_v  (B*kd for V proj; later reused at stride d for Wo)
//   [4*B*d      .. 5*B*d)       g_tmp (attn_out staging / gate activation)
//   [5*B*d      .. 6*B*d)       g_mlp_acc
//   [6*B*d      .. 6*B*d+B*T)   g_mlp_up  (T = MLP_FF_TILE)

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
    const int nkv = model.cfg.n_kv_heads;
    const int dph = d / nh;
    const int kd  = nkv * dph;
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
    float* s_buf2   = smem + 2 * d;   // [kd] — K or V temp
    float* s_buf3   = smem + 3 * d;   // [d] — attn_out / hidden load
    float* s_red    = smem + 4 * d;   // reduction / blk_logits scratch

    // ---- Embed ----
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

        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = g_hidden[b * d + i];
            __syncthreads();
            device_rmsnorm(s_buf3, lw.rms_attn_weight, s_buf0, d, s_red);
            for (int i = tid; i < d; i += blockDim.x)
                g_normed[b * d + i] = s_buf0[i];
            __syncthreads();
        }

        device_matvec_batched(g_normed, lw.Wq, g_q, d, d,  B);
        device_add_bias_batched(g_q, lw.Wq_bias, d, B);
        device_matvec_batched(g_normed, lw.Wk, g_k, d, kd, B);
        device_add_bias_batched(g_k, lw.Wk_bias, kd, B);
        device_matvec_batched(g_normed, lw.Wv, g_v, d, kd, B);
        device_add_bias_batched(g_v, lw.Wv_bias, kd, B);

        // Per-token: RoPE + KV cache append
        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf1[i] = g_q[b * d + i];
            for (int i = tid; i < kd; i += blockDim.x)
                s_buf2[i] = g_k[b * kd + i];
            __syncthreads();

            rope_apply_inplace(s_buf1, nh, dph, seq_base + b,
                               model.cfg.rope_theta,
                               model.cfg.rope_scaling_type,
                               model.cfg.rope_scaling_factor);
            rope_apply_inplace(s_buf2, nkv, dph, seq_base + b,
                               model.cfg.rope_theta,
                               model.cfg.rope_scaling_type,
                               model.cfg.rope_scaling_factor);

            // Save Q back, load V
            for (int i = tid; i < d; i += blockDim.x)
                g_q[b * d + i] = s_buf1[i];

            // s_buf3 will hold V temporarily for cache append
            for (int i = tid; i < kd; i += blockDim.x)
                s_buf3[i] = g_v[b * kd + i];
            __syncthreads();

            kv_cache_append(kv, l, s_buf2, s_buf3, seq_base + b);
            __syncthreads();
        }

        // Per-token GQA attention → g_tmp
        __shared__ float s_attn_m_b;
        __shared__ float s_attn_l_b;
        __shared__ float s_tile_m_b;
        __shared__ float s_alpha_b;
        __shared__ float s_inv_den_b;

        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf1[i] = g_q[b * d + i];
            __syncthreads();

            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = 0.f;
            __syncthreads();

            int total_len = seq_base + b + 1;

            device_attention_dispatch(
                model.cfg.attention_type,
                s_buf1, s_buf3, kv, l,
                nh, nkv, dph, total_len,
                s_red,
                s_attn_m_b, s_attn_l_b, s_tile_m_b, s_alpha_b, s_inv_den_b);

            for (int i = tid; i < d; i += blockDim.x)
                g_tmp[b * d + i] = s_buf3[i];
            __syncthreads();
        }

        // Wo projection (g_v reused at stride d for output)
        device_matvec_batched(g_tmp, lw.Wo, g_v, d, d, B);

        for (int b = 0; b < B; b++)
            for (int i = tid; i < d; i += blockDim.x)
                g_hidden[b * d + i] += g_v[b * d + i];
        __syncthreads();

        // MLP
        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = g_hidden[b * d + i];
            __syncthreads();
            device_rmsnorm(s_buf3, lw.rms_mlp_weight, s_buf0, d, s_red);
            for (int i = tid; i < d; i += blockDim.x)
                g_normed[b * d + i] = s_buf0[i];
            __syncthreads();
        }

        for (int b = 0; b < B; b++)
            for (int i = tid; i < d; i += blockDim.x)
                g_mlp_acc[b * d + i] = 0.f;
        __syncthreads();

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

    // Output
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

__device__ void model_batch_forward_selfspec_verify_logits(
    const ModelWeights& model, KVCache& kv,
    const int* token_ids, int seq_base, int B, int layer_start,
    const float* g_saved_hiddens,
    float* g_hidden, float* g_work, float* g_logits_out,
    float* smem)
{
    const int tid = threadIdx.x;
    const int d   = model.cfg.d_model;
    const int dff = model.cfg.d_ff;
    const int nh  = model.cfg.n_heads;
    const int nkv = model.cfg.n_kv_heads;
    const int dph = d / nh;
    const int kd  = nkv * dph;
    const int V   = model.cfg.vocab_size;

    float* g_normed  = g_work;
    float* g_q       = g_work + 1 * B * d;
    float* g_k       = g_work + 2 * B * d;
    float* g_v       = g_work + 3 * B * d;
    float* g_tmp     = g_work + 4 * B * d;
    float* g_mlp_acc = g_work + 5 * B * d;
    float* g_mlp_up  = g_work + 6 * B * d;

    float* s_buf0   = smem;
    float* s_buf1   = smem + d;
    float* s_buf2   = smem + 2 * d;
    float* s_buf3   = smem + 3 * d;
    float* s_red    = smem + 4 * d;

    for (int b = 0; b < B; b++) {
        const float* src = g_saved_hiddens + (size_t)b * d;
        for (int i = tid; i < d; i += blockDim.x)
            g_hidden[b * d + i] = src[i];
        __syncthreads();
    }

    for (int l = layer_start; l < model.cfg.n_layers; l++) {
        const LayerWeights& lw = model.layers[l];

        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = g_hidden[b * d + i];
            __syncthreads();
            device_rmsnorm(s_buf3, lw.rms_attn_weight, s_buf0, d, s_red);
            for (int i = tid; i < d; i += blockDim.x)
                g_normed[b * d + i] = s_buf0[i];
            __syncthreads();
        }

        device_matvec_batched(g_normed, lw.Wq, g_q, d, d,  B);
        device_add_bias_batched(g_q, lw.Wq_bias, d, B);
        device_matvec_batched(g_normed, lw.Wk, g_k, d, kd, B);
        device_add_bias_batched(g_k, lw.Wk_bias, kd, B);
        device_matvec_batched(g_normed, lw.Wv, g_v, d, kd, B);
        device_add_bias_batched(g_v, lw.Wv_bias, kd, B);

        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf1[i] = g_q[b * d + i];
            for (int i = tid; i < kd; i += blockDim.x)
                s_buf2[i] = g_k[b * kd + i];
            __syncthreads();

            rope_apply_inplace(s_buf1, nh, dph, seq_base + b,
                               model.cfg.rope_theta,
                               model.cfg.rope_scaling_type,
                               model.cfg.rope_scaling_factor);
            rope_apply_inplace(s_buf2, nkv, dph, seq_base + b,
                               model.cfg.rope_theta,
                               model.cfg.rope_scaling_type,
                               model.cfg.rope_scaling_factor);

            for (int i = tid; i < d; i += blockDim.x)
                g_q[b * d + i] = s_buf1[i];
            for (int i = tid; i < kd; i += blockDim.x)
                s_buf3[i] = g_v[b * kd + i];
            __syncthreads();

            kv_cache_append(kv, l, s_buf2, s_buf3, seq_base + b);
            __syncthreads();
        }

        __shared__ float s_attn_m_b;
        __shared__ float s_attn_l_b;
        __shared__ float s_tile_m_b;
        __shared__ float s_alpha_b;
        __shared__ float s_inv_den_b;

        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf1[i] = g_q[b * d + i];
            __syncthreads();
            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = 0.f;
            __syncthreads();

            int total_len = seq_base + b + 1;

            device_attention_dispatch(
                model.cfg.attention_type,
                s_buf1, s_buf3, kv, l,
                nh, nkv, dph, total_len,
                s_red,
                s_attn_m_b, s_attn_l_b, s_tile_m_b, s_alpha_b, s_inv_den_b);

            for (int i = tid; i < d; i += blockDim.x)
                g_tmp[b * d + i] = s_buf3[i];
            __syncthreads();
        }

        device_matvec_batched(g_tmp, lw.Wo, g_v, d, d, B);

        for (int b = 0; b < B; b++)
            for (int i = tid; i < d; i += blockDim.x)
                g_hidden[b * d + i] += g_v[b * d + i];
        __syncthreads();

        for (int b = 0; b < B; b++) {
            for (int i = tid; i < d; i += blockDim.x)
                s_buf3[i] = g_hidden[b * d + i];
            __syncthreads();
            device_rmsnorm(s_buf3, lw.rms_mlp_weight, s_buf0, d, s_red);
            for (int i = tid; i < d; i += blockDim.x)
                g_normed[b * d + i] = s_buf0[i];
            __syncthreads();
        }

        for (int b = 0; b < B; b++)
            for (int i = tid; i < d; i += blockDim.x)
                g_mlp_acc[b * d + i] = 0.f;
        __syncthreads();

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

// InferenceEngine: init / destroy

void inference_engine_init(InferenceEngine& eng, const ModelConfig& cfg) {
    // Query device capabilities
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    eng.sm_major = prop.major;
    eng.sm_minor = prop.minor;

    int coop_attr = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_attr,
                                     cudaDevAttrCooperativeLaunch, 0));
    eng.coop_supported = (coop_attr != 0);

    // Upper bound on grid size passed to launch_cooperative_decode_step.
    // The actual per-launch cap is tighter: it's recomputed from
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor each call so that
    // large-model smem requirements (e.g. 40 KB/block on RTX 3050) are
    // respected.  Use 2 blocks/SM here so this value never over-counts.
    eng.max_coop_blocks = prop.multiProcessorCount * 2;

    // cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&eng.cublas));
    if (eng.sm_major >= 8)
        CUBLAS_CHECK(cublasSetMathMode(eng.cublas, CUBLAS_TENSOR_OP_MATH));

    // Stream pool + events
    for (int i = 0; i < STREAM_POOL_SIZE; i++) {
        CUDA_CHECK(cudaStreamCreate(&eng.streams[i]));
        CUDA_CHECK(cudaEventCreate(&eng.sync_events[i]));
    }

    // Cooperative single-token decode scratch
    // Two independent copies so draft (stream[0]) and target (stream[1]) can
    // run concurrently during prefill without aliasing the residual buffer.
    const int d = cfg.d_model;
    size_t scratch_n = (size_t)(7 * d + 2 * MLP_FF_TILE);
    CUDA_CHECK(cudaMalloc(&eng.d_coop_hidden,   (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_coop_scratch,  scratch_n  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_coop_hidden2,  (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_coop_scratch2, scratch_n  * sizeof(float)));

    // Self-spec post-prefix activations: k draft slots + 1 bonus verify slot (max k = 16).
    constexpr int kMaxSelfSpecK = 17;
    CUDA_CHECK(cudaMalloc(&eng.d_selfspec_hiddens,
                          (size_t)kMaxSelfSpecK * d * sizeof(float)));

    // cuBLAS forward scratch (lazy-growth: allocated on first cublas_forward call)
    eng.d_cublas_buf      = nullptr;
    eng.d_cublas_x16      = nullptr;
    eng.cublas_buf_floats = 0;
    eng.cublas_x16_halves = 0;
}

void inference_engine_destroy(InferenceEngine& eng) {
    cublasDestroy(eng.cublas);
    for (int i = 0; i < STREAM_POOL_SIZE; i++) {
        cudaStreamDestroy(eng.streams[i]);
        cudaEventDestroy(eng.sync_events[i]);
    }
    auto sf = [](void* p) { if (p) cudaFree(p); };
    sf(eng.d_coop_hidden);   sf(eng.d_coop_scratch);
    sf(eng.d_coop_hidden2);  sf(eng.d_coop_scratch2);
    sf(eng.d_selfspec_hiddens);
    sf(eng.d_cublas_buf);    sf(eng.d_cublas_x16);
    memset(&eng, 0, sizeof(eng));
}
