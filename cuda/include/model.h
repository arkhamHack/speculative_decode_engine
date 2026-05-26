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

// Phase 1: Q/K/V projections + RoPE + KV cache write at position seq_pos.
// After return, smem[d..2*d) holds Q with RoPE — required by attn_mlp_phase.
// hidden is read-only; the KV cache gains one entry at seq_pos.
__device__ void model_layer_kv_phase(const ModelWeights& model, int layer_idx,
                                      const float* hidden, KVCache& kv,
                                      int seq_pos, float* smem);

// Phase 2: flash-attention over [0..seq_pos] (causal) + output proj + SwiGLU FFN.
// Requires smem[d..2*d) = Q written by kv_phase, and K,V at seq_pos in cache.
// hidden is updated with the attention + MLP residuals.
__device__ void model_layer_attn_mlp_phase(const ModelWeights& model,
                                            int layer_idx,
                                            float* hidden, KVCache& kv,
                                            int seq_pos, float* smem);

// Full layer: kv_phase followed by attn_mlp_phase (single-block sequential path).
// current_seq_len: tokens already in the KV cache; this token is appended here.
__device__ void model_layer_forward(const ModelWeights& model, int layer_idx,
                                    float* hidden, KVCache& kv,
                                    int current_seq_len, float* smem);

// Final RMSNorm + output projection.
// g_logits: global-memory float buffer of size vocab_size (pre-allocated by caller).
__device__ void model_output(const ModelWeights& model,
                             const float* hidden, float* g_logits, float* smem);

// Full single-token forward: embed -> layers -> logits in g_logits (no argmax).
__device__ void model_forward_logits(const ModelWeights& model, KVCache& kv,
                                     int token_id, int current_seq_len,
                                     float* hidden, float* g_logits,
                                     float* smem);

// Full single-token forward pass: embed -> layers -> output -> argmax.
// g_logits: global-memory scratch for logits (vocab_size floats).
// Returns the greedy next token id.
__device__ int model_forward(const ModelWeights& model, KVCache& kv,
                             int token_id, int current_seq_len,
                             float* hidden, float* g_logits, float* smem);

// Batched forward: process B tokens through the model in one pass.
// Reads each weight matrix ONCE for all B tokens (vs B times for sequential).
// token_ids: [B] array in device-accessible memory.
// g_hidden:  [B * d_model] global scratch for hidden states.
// g_work:    [(7 * d_model + MLP_FF_TILE) * B] global scratch for intermediates.
// g_logits:  [B * vocab_size] output logits for each position.
// smem:      shared memory (same budget as single-token, sized for this model).
// KV cache entries are appended at positions seq_base .. seq_base+B-1.
__device__ void model_batch_forward_logits(
    const ModelWeights& model, KVCache& kv,
    const int* token_ids, int seq_base, int B,
    float* g_hidden, float* g_work, float* g_logits,
    float* smem);
