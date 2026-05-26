#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Compile-time constants (hard upper bounds for static array sizing)
// ============================================================================

// Maximum sequence length (KV logical length bound; chunked attention avoids O(seq) scratch)
constexpr int MAX_SEQ_LEN      = 1024;
// Paged KV cache block size (tokens per physical block)
constexpr int KV_BLOCK_SIZE    = 16;
// Max blocks per layer (enough for MAX_SEQ_LEN / KV_BLOCK_SIZE)
constexpr int MAX_KV_BLOCKS    = MAX_SEQ_LEN / KV_BLOCK_SIZE;   // 64
// Default speculation depth
constexpr int DEFAULT_SPEC_K   = 4;
// CUDA warp size (always 32 on NVIDIA hardware)
constexpr int WARP_SIZE        = 32;
// Thread block size used by all inference kernels.
// Stride loops handle d_model > BLOCK_THREADS transparently.
constexpr int BLOCK_THREADS    = 256;
// Max transformer layers (supports LLaMA-7B/13B with 32/40 layers)
constexpr int MAX_LAYERS       = 40;
// Column tile for SwiGLU projections (stored in repurposed buffers; never materialize full d_ff)
constexpr int MLP_FF_TILE      = 256;
// Max batch size for batched target verification (spec_k + 1).
// Supports spec_k up to 8; raise and add cases to the dispatch switches in utils.h if needed.
constexpr int MAX_VERIFY_BATCH = 9;

// Default vocabulary for the built-in dummy integer tokenizer
constexpr int DEFAULT_VOCAB_SIZE = 256;
// Sentinel: no EOS for the dummy tokenizer (tokens 0-255 all valid)
// Superseded by GenerationParams::eos_token (runtime value from HF tokenizer).
// Kept only to avoid breaking any downstream references.
constexpr int EOS_TOKEN          = -1;

// Binary weight file magic ("SDEC" in little-endian uint32)
constexpr uint32_t SDEC_MAGIC   = 0x43454453u;

// ============================================================================
// Model configuration  (runtime, stored in each ModelWeights)
// ============================================================================

struct ModelConfig {
    int n_layers   = 2;
    int d_model    = 128;
    // d_head: *total* KV dimensionality stored per token per layer (= d_model).
    // The per-head slice is d_head_per = d_model / n_heads.
    int d_head     = 128;
    int n_heads    = 4;    // must divide d_model
    int d_ff       = 512;
    int vocab_size = DEFAULT_VOCAB_SIZE;
    // Llama-style RoPE base frequency (HF: Llama2 ~10000, Llama3 ~500000).
    // SDEC v2 stores this in the file header; v1 loaders default to 10000.
    float rope_theta = 10000.f;
};

// Per-head Q/K/V dimensionality (convenience helper)
inline int d_head_per(const ModelConfig& cfg) {
    return cfg.d_model / cfg.n_heads;
}

// Built-in model presets for the dummy tokenizer path:
//   Draft:  2 layers, d=128, 4 heads (d_head_per=32), d_ff=512
inline ModelConfig make_draft_config() {
    return {2, 128, 128, 4, 512, DEFAULT_VOCAB_SIZE};
}
//   Target: 4 layers, d=256, 8 heads (d_head_per=32), d_ff=1024
inline ModelConfig make_target_config() {
    return {4, 256, 256, 8, 1024, DEFAULT_VOCAB_SIZE};
}

// ============================================================================
// Dynamic shared-memory budget
//
// Shared memory layout per kernel block:
//   hidden   [d_model]       -- token hidden state, persists across layers
//
// Scratch for model_layer_forward / model_output (after hidden[]) :
//   normed   [d_model]       -- RMSNorm output scratch
//   q_all    [d_model]       -- Q projection (all heads concatenated)
//   kv_temp  [d_model]       -- K/V projection temp; also output-proj output
//   attn_out [d_model]       -- per-head V accumulation + Wo temp staging
//   scratch  tail            -- softmax / RMSNorm warp reductions (+ padding)
//
// Attention: FlashAttention-style *streaming softmax* over KV blocks — at most KV_BLOCK_SIZE
// logits materialized at a time in the scratch tail (no full-seq score vector).
// MLP: column-tiles W_gate/W_up/W_down — only O(min(d_ff,d_model_tile)) intermediates live in smem.
//
// After layers, global_argmax overlays the first min(size, 2*BLOCK_THREADS) floats of
// the scratch region — size must never be smaller than that for tiny d_model.
//
// Bytes = hidden + max( 4*d_model + WARP_SIZE, 2*BLOCK_THREADS ) floats (in scratch).
// Example (Llama-ish, d=2048): (2048 + 8216) floats * 4 < 96 KiB.
// ============================================================================

inline size_t compute_smem_bytes(const ModelConfig& cfg) {
    size_t scratch_core  = (size_t)4 * cfg.d_model + WARP_SIZE;
    size_t argmax_pad    = (size_t)BLOCK_THREADS * 2;
    size_t scratch_floats_max = scratch_core > argmax_pad ? scratch_core : argmax_pad;
    size_t floats = (size_t)cfg.d_model + scratch_floats_max;
    return floats * sizeof(float);
}

// ============================================================================
// Generation parameters and result structure
// ============================================================================

struct GenerationParams {
    int    max_new_tokens;
    int    spec_k;         // draft tokens per speculation round
    bool   use_megakernel; // false = multi-kernel loop, true = persistent megakernel
    // EOS token id from the HF tokenizer (-1 = disabled, generation always runs to max_new_tokens)
    int    eos_token      = -1;
    //
    // Stochastic speculative decoding (distribution-level acceptance with p,q and
    // optional adjusted rejection sampling). Supported on multi-kernel and megakernel.
    bool   stochastic_spec_decode = false;
    float  draft_temperature      = 1.f; // softmax temperature for drafting / q probs
    // Heuristic knobs to nudge draft temperature toward a target acceptance fraction
    // ( EWMA ); cooler q often aligns better with the target chain and raises empirical α.
    // Not theoretically exact versus tempered q unless q matches the verifier's policy.
    bool   adaptive_draft_temperature = false;
    float  min_draft_temperature      = 0.55f;
    float  max_draft_temperature      = 1.2f;
    float  stochastic_adapt_target_accept = 0.50f;
    float  stochastic_adapt_temp_gain    = 0.055f;
    float  stochastic_adapt_ewma_mix      = 0.25f;
    unsigned stochastic_rng_seed       = 12345;
};

struct GenerationResult {
    int   output_tokens[MAX_SEQ_LEN];
    int   n_generated;
    int   draft_proposed;
    int   draft_accepted;
    int   spec_iterations;
    float elapsed_ms;
};
