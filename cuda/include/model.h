#pragma once
#include "config.h"
#include "kv_cache.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

// Weight layout  (all tensors in half precision, 16-byte aligned on device)
//
// Attention projections use native GQA dimensions:
//   Wq has n_heads columns, Wk/Wv have n_kv_heads columns (kv_dim ≤ d_model).
//   SDEC v1/v2 files store expanded Wk/Wv [d_model, d_model]; v3 stores native.
//
// Per layer:
//   rms_attn_weight  [d_model]
//   Wq               [d_model, d_model]    row-major
//   Wk               [d_model, kv_dim]     kv_dim = n_kv_heads * head_dim
//   Wv               [d_model, kv_dim]
//   Wo               [d_model, d_model]
//   rms_mlp_weight   [d_model]
//   W_gate           [d_model, d_ff]
//   W_up             [d_model, d_ff]
//   W_down           [d_ff,    d_model]
//
// Global:
//   token_embedding  [vocab_size, d_model]
//   rms_final_weight [d_model]
//   output_proj      [d_model, vocab_size]   (logits = normed @ output_proj)

struct LayerWeights {
    half* rms_attn_weight;   // [d_model]
    half* Wq;                // [d_model, d_model]
    half* Wk;                // [d_model, kv_dim]
    half* Wv;                // [d_model, kv_dim]
    half* Wo;                // [d_model, d_model]
    half* rms_mlp_weight;    // [d_model]
    half* W_gate;            // [d_model, d_ff]
    half* W_up;              // [d_model, d_ff]
    half* W_down;            // [d_ff,    d_model]
    // Optional Q/K/V biases (SDEC v5+ / Qwen2). nullptr when absent.
    half* Wq_bias;           // [d_model]  or nullptr
    half* Wk_bias;           // [kv_dim]   or nullptr
    half* Wv_bias;           // [kv_dim]   or nullptr
};

struct ModelWeights {
    ModelConfig   cfg;
    half*         token_embedding;   // [vocab_size, d_model]
    half*         rms_final_weight;  // [d_model]
    half*         output_proj;       // [d_model, vocab_size]
    LayerWeights  layers[MAX_LAYERS];
};

void model_alloc(ModelWeights& model, const ModelConfig& cfg);
void model_free(ModelWeights& model);


// Load weights from an SDEC binary file produced by tools/export_model.py.
// Fills cfg_out with the model configuration read from the file header.
// Returns true on success; on failure prints an error and returns false.
bool model_load_weights(ModelWeights& model, const char* path,
                        ModelConfig* cfg_out = nullptr);

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
__device__ void model_batch_forward_logits(
    const ModelWeights& model, KVCache& kv,
    const int* token_ids, int seq_base, int B,
    float* g_hidden, float* g_work, float* g_logits,
    float* smem);

// Self-spec draft: embed -> prefix [0, n_prefix_layers) -> save hidden ->
// suffix [n_prefix_layers, n_layers) -> output head -> g_logits.
// g_saved_hidden[d_model]: post-prefix activation (verify suffix input).
__device__ void model_forward_draft_logits(
    const ModelWeights& model, KVCache& kv,
    int token_id, int current_seq_len,
    int n_prefix_layers,
    float* hidden, float* g_logits, float* g_saved_hidden,
    float* smem);

// Prefix-only forward for the verify bonus slot: save post-prefix hidden, no logits.
__device__ void model_forward_draft_prefix_save(
    const ModelWeights& model, KVCache& kv,
    int token_id, int current_seq_len,
    int n_prefix_layers,
    float* hidden, float* g_saved_hidden, float* smem);

// Self-spec verify from a saved activation: layers [layer_start, n_layers) -> logits.
__device__ void model_forward_verify_from_hidden_logits(
    const ModelWeights& model, KVCache& kv,
    const float* saved_hidden, int layer_start, int current_seq_len,
    float* hidden, float* g_logits, float* smem);

// Batched self-spec verify.  g_saved_hiddens[B*d_model]: saved[b] is the post-prefix
// activation for verify position b (written during draft).  All positions load saved[b]
// and share the suffix layer loop.
__device__ void model_batch_forward_selfspec_verify_logits(
    const ModelWeights& model, KVCache& kv,
    const int* token_ids, int seq_base, int B, int layer_start,
    const float* g_saved_hiddens,
    float* g_hidden, float* g_work, float* g_logits_out,
    float* smem);

// InferenceEngine — host-side state for cuBLAS handle, CUDA stream pool,
// cooperative-launch capability, and scratch buffers for the cooperative
// single-token decode kernel.
//
// Streams:
//   streams[0]  -- draft prefill / intra-layer stream A
//   streams[1]  -- target prefill / intra-layer stream B
//   streams[2..STREAM_POOL_SIZE-1] -- spare
// sync_events[] -- per-stream cudaEvent_t for cross-stream dependencies
//
// Cooperative single-token decode scratch (one pair per concurrent model):
//   d_coop_hidden  / d_coop_scratch  -- used by target (stream[1]) or baseline
//   d_coop_hidden2 / d_coop_scratch2 -- used by draft  (stream[0]) in parallel prefill
// Both are sized for the LARGEST model passed to inference_engine_init.
struct InferenceEngine {
    cublasHandle_t cublas;
    cudaStream_t   streams[STREAM_POOL_SIZE];
    cudaEvent_t    sync_events[STREAM_POOL_SIZE];

    int  sm_major, sm_minor;
    bool coop_supported;   // device supports cudaLaunchCooperativeKernel
    int  max_coop_blocks;  // conservative cap for cooperative_decode_kernel

    GemmBackendType gemm_backend = GEMM_BACKEND_LEGACY;

    // Cooperative single-token decode residual stream + scratch
    float* d_coop_hidden;   // [d_model]
    float* d_coop_scratch;  // [7*d_model + 2*MLP_FF_TILE]

    // Second set so draft and target streams don't alias during parallel prefill
    float* d_coop_hidden2;
    float* d_coop_scratch2;

    // Self-spec: post-prefix activations [(spec_k + 1) * d_model] (bonus slot at index k)
    float* d_selfspec_hiddens;

    // cuBLAS forward scratch (shared by prefill, decode, verify).
    // Lazy-growth: allocated on first cublas_forward call, grown if needed.
    float* d_cublas_buf;       // float scratch for activations
    half*  d_cublas_x16;       // FP16 conversion buffer
    size_t cublas_buf_floats;  // floats allocated in d_cublas_buf
    size_t cublas_x16_halves;  // halfs allocated in d_cublas_x16
};

// Allocate and initialise the engine for the given (largest) model config.
void inference_engine_init(InferenceEngine& eng, const ModelConfig& cfg);

// Release all resources held by the engine.
void inference_engine_destroy(InferenceEngine& eng);
