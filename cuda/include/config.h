#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Compile-time constants (hard upper bounds for static array sizing)
// ============================================================================

// Maximum sequence length (scores[] in smem is sized to this at launch time)
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

// Default vocabulary for the built-in dummy integer tokenizer
constexpr int DEFAULT_VOCAB_SIZE = 256;
// Sentinel: no EOS for the dummy tokenizer (tokens 0-255 all valid)
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
//   normed   [d_model]       -- RMSNorm output scratch
//   q_all    [d_model]       -- Q projection (all heads concatenated)
//   kv_temp  [d_model]       -- K/V projection temp; also output-proj output
//   scores   [MAX_SEQ_LEN]   -- per-head attention scores
//   attn_out [d_model]       -- per-head V accumulation (full concat output)
//   gate_buf [d_ff]          -- SwiGLU gate projection
//   up_buf   [d_ff]          -- SwiGLU up projection
//   scratch  [WARP_SIZE]     -- block-reduction scratch (warp results)
//
// Total = 5*d_model + MAX_SEQ_LEN + 2*d_ff + WARP_SIZE floats.
// Example (GPT-2 small, d=768, d_ff=3072):
//   5*768 + 1024 + 2*3072 + 32 = 11040 floats = 44 KB  (under 48 KB limit)
// Example (target preset, d=256, d_ff=1024):
//   5*256 + 1024 + 2*1024 + 32 = 4352 floats = 17 KB
// ============================================================================

inline size_t compute_smem_bytes(const ModelConfig& cfg) {
    size_t floats = (size_t)cfg.d_model * 5     // hidden+normed+q_all+kv_temp+attn_out
                  + MAX_SEQ_LEN                  // scores (compile-time bound)
                  + (size_t)cfg.d_ff * 2         // gate_buf + up_buf
                  + WARP_SIZE;                   // reduction scratch
    return floats * sizeof(float);
}

// ============================================================================
// Generation parameters and result structure
// ============================================================================

struct GenerationParams {
    int  max_new_tokens;
    int  spec_k;           // draft tokens per speculation round
    bool use_megakernel;   // false = multi-kernel loop, true = persistent megakernel
};

struct GenerationResult {
    int   output_tokens[MAX_SEQ_LEN];
    int   n_generated;
    int   draft_proposed;
    int   draft_accepted;
    int   spec_iterations;
    float elapsed_ms;
};
