#pragma once
#include "config.h"
#include "kv_cache.h"
#include <cuda_fp16.h>

// ============================================================================
// Weight layout  (all tensors in half precision, 16-byte aligned on device)
//
// Attention projections are all [d_model, d_model] regardless of n_heads.
// The multi-head split is done at compute time using d_head_per = d_model/n_heads.
//
// Per layer:
//   rms_attn_weight  [d_model]
//   Wq               [d_model, d_model]   row-major; col = head_concat output
//   Wk               [d_model, d_model]
//   Wv               [d_model, d_model]
//   Wo               [d_model, d_model]   maps concat'd head outputs back to d_model
//   rms_mlp_weight   [d_model]
//   W_gate           [d_model, d_ff]
//   W_up             [d_model, d_ff]
//   W_down           [d_ff,    d_model]
//
// Global:
//   token_embedding  [vocab_size, d_model]
//   rms_final_weight [d_model]
//   output_proj      [d_model, vocab_size]   (logits = normed @ output_proj)
// ============================================================================

struct LayerWeights {
    half* rms_attn_weight;   // [d_model]
    half* Wq;                // [d_model, d_model]
    half* Wk;                // [d_model, d_model]
    half* Wv;                // [d_model, d_model]
    half* Wo;                // [d_model, d_model]
    half* rms_mlp_weight;    // [d_model]
    half* W_gate;            // [d_model, d_ff]
    half* W_up;              // [d_model, d_ff]
    half* W_down;            // [d_ff,    d_model]
};

struct ModelWeights {
    ModelConfig   cfg;
    half*         token_embedding;   // [vocab_size, d_model]
    half*         rms_final_weight;  // [d_model]
    half*         output_proj;       // [d_model, vocab_size]
    LayerWeights  layers[MAX_LAYERS];
};

// ============================================================================
// Host API
// ============================================================================

// Allocate device memory for all weight tensors according to cfg.
void model_alloc(ModelWeights& model, const ModelConfig& cfg);

// Free all device weight allocations.
void model_free(ModelWeights& model);

// Fill weights with small Gaussian random values (for dummy/benchmark mode).
void model_init_random(ModelWeights& model, unsigned seed);

// Load weights from an SDEC binary file produced by tools/export_model.py.
// Fills cfg_out with the model configuration read from the file header.
// Returns true on success; on failure prints an error and returns false.
bool model_load_weights(ModelWeights& model, const char* path,
                        ModelConfig* cfg_out = nullptr);

// ============================================================================
// Device functions  (called from within a single-block CUDA kernel)
//
// All device functions operate on the calling block's shared scratch after hidden[]
// (streaming attention + tiled MLP keep large temporaries factorized/tiled — no seq×d_ff arrays).
// ============================================================================

// Embedding lookup: hidden[d] = token_embedding[token_id, :]
__device__ void model_embed(const ModelWeights& model, int token_id,
                            float* hidden);

// One transformer layer: RMSNorm -> MHA -> Residual -> RMSNorm -> SwiGLU -> Residual
// hidden is modified in-place.
// current_seq_len: number of tokens already in the KV cache for this sequence
//                  (the current token is appended during this call).
__device__ void model_layer_forward(const ModelWeights& model, int layer_idx,
                                    float* hidden, KVCache& kv,
                                    int current_seq_len, float* smem);

// Final RMSNorm + output projection.
// g_logits: global-memory float buffer of size vocab_size (pre-allocated by caller).
__device__ void model_output(const ModelWeights& model,
                             const float* hidden, float* g_logits, float* smem);

// Full single-token forward pass: embed -> layers -> output -> argmax.
// g_logits: global-memory scratch for logits (vocab_size floats).
// Returns the greedy next token id.
__device__ int model_forward(const ModelWeights& model, KVCache& kv,
                             int token_id, int current_seq_len,
                             float* hidden, float* g_logits, float* smem);
