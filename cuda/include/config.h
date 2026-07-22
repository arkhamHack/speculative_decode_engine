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

// Output columns per cooperative GEMV block (multi-block single-token decode).
constexpr int GEMV_COL_TILE    = 128;

// Size of the CUDA stream pool used for overlapping independent projections.
constexpr int STREAM_POOL_SIZE = 4;

// Default vocabulary for the built-in dummy integer tokenizer
constexpr int DEFAULT_VOCAB_SIZE = 256;

// Binary weight file magic ("SDEC" in little-endian uint32)
constexpr uint32_t SDEC_MAGIC   = 0x43454453u;

// ============================================================================
// RoPE scaling modes  (stored in SDEC v3+ header; runtime enum)
// ============================================================================

enum RopeScalingType : int {
    ROPE_SCALING_NONE   = 0,   // standard RoPE: angle = pos * inv_freq
    ROPE_SCALING_LINEAR = 1,   // angle = (pos / factor) * inv_freq
    ROPE_SCALING_NTK    = 2,   // theta_eff = theta * factor^(d/(d-2))
    ROPE_SCALING_YARN   = 3,   // YaRN (NTK-by-parts, temperature + attn scaling)
};

// ============================================================================
// Attention backend selector  (per-model; stored in SDEC v4+ header)
// ============================================================================

enum AttentionType : int {
    ATTN_SOFTMAX       = 0,   // standard scaled-dot-product (MHA / GQA / MQA)
    ATTN_SLIDING_WINDOW = 1,  // (reserved) softmax with fixed sliding window
    ATTN_KDA           = 2,   // (reserved) Kimi Delta Attention (linear RNN)
    ATTN_LINEAR        = 3,   // (reserved) generic linear attention
};

// ============================================================================
// GEMM backend selector  (host-launched cuBLAS vs device-internal matvec)
// ============================================================================

enum GemmBackendType : int {
    GEMM_BACKEND_LEGACY = 0,   // device_matvec / megakernel (current)
    GEMM_BACKEND_CUBLAS = 1,   // host-orchestrated cublasGemmEx with tensor ops
};

// ============================================================================
// Model configuration  (runtime, stored in each ModelWeights)
// ============================================================================

struct ModelConfig {
    int n_layers   = 2;
    int d_model    = 128;
    int n_heads    = 4;       // query heads; must divide d_model
    int n_kv_heads = 4;       // KV heads (GQA when < n_heads); must divide n_heads
    int d_ff       = 512;
    int vocab_size = DEFAULT_VOCAB_SIZE;
    float rope_theta = 10000.f;

    RopeScalingType rope_scaling_type   = ROPE_SCALING_NONE;
    float           rope_scaling_factor = 1.0f;

    AttentionType   attention_type      = ATTN_SOFTMAX;

    // True when SDEC v5 weights include per-layer Q/K/V projection biases (Qwen2).
    bool            has_qkv_bias        = false;
};

// Derived helpers (always computed from the config, never stored separately)
inline int head_dim(const ModelConfig& c)     { return c.d_model / c.n_heads; }
inline int d_head_per(const ModelConfig& c)   { return c.d_model / c.n_heads; }
inline int kv_dim(const ModelConfig& c)       { return c.n_kv_heads * head_dim(c); }
inline int gqa_ratio(const ModelConfig& c)    { return c.n_heads / c.n_kv_heads; }

// Dummy presets
inline ModelConfig make_draft_config() {
    ModelConfig c; c.n_layers=2; c.d_model=128; c.n_heads=4; c.n_kv_heads=4;
    c.d_ff=512; c.vocab_size=DEFAULT_VOCAB_SIZE; return c;
}
inline ModelConfig make_target_config() {
    ModelConfig c; c.n_layers=4; c.d_model=256; c.n_heads=8; c.n_kv_heads=8;
    c.d_ff=1024; c.vocab_size=DEFAULT_VOCAB_SIZE; return c;
}

// ============================================================================
// Dynamic shared-memory budget
//
// kv_phase  layout: normed[d] | q[d] | kv_temp[kd] | k_save[kd] | scratch
// attn_phase reuses: attn_out[d] | wo_buf[d] | gate[kd] | up[kd] | scratch
// model_output:      buf[d] | scratch[max(d + WARP_SIZE, 2*BLOCK_THREADS)]
//
// We take the max of all phases.
// ============================================================================

inline size_t compute_smem_bytes(const ModelConfig& cfg) {
    int d  = cfg.d_model;
    int kd = cfg.n_kv_heads * (d / cfg.n_heads);   // kv_dim

    // kv_phase + attn_phase: 2*d + 2*kd + scratch
    size_t phase_floats = (size_t)(2 * d + 2 * kd);

    // Scratch needs room for reductions (d + WARP_SIZE) and argmax (2*BLOCK_THREADS)
    size_t scratch_a = (size_t)d + WARP_SIZE;
    size_t scratch_b = (size_t)BLOCK_THREADS * 2;
    size_t scratch   = scratch_a > scratch_b ? scratch_a : scratch_b;

    // model_output needs d + scratch; kv/attn phase needs phase_floats + scratch
    size_t floats = phase_floats + scratch;
    size_t out_floats = (size_t)d + scratch;
    if (out_floats > floats) floats = out_floats;

    return floats * sizeof(float);
}

// ============================================================================
// Generation parameters and result structure
// ============================================================================

struct GenerationParams {
    int    max_new_tokens;
    int    spec_k;         // draft tokens per speculation round
    bool   use_megakernel; // false = multi-kernel loop, true = persistent megakernel
    // Self-speculative: draft = first n_draft_layers of target; same weight tensors.
    bool   self_speculative = false;
    int    n_draft_layers   = 0;   // 0 = use draft_model.cfg.n_layers
    // EOS token id from the HF tokenizer (-1 = disabled, generation always runs to max_new_tokens)
    int    eos_token      = -1;
    //
    // Stochastic speculative decoding (distribution-level acceptance with p,q and
    // optional adjusted rejection sampling). Supported on multi-kernel and megakernel.
    bool   stochastic_spec_decode = false;
    float  draft_temperature      = 1.f;
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
