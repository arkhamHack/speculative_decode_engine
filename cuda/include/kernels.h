#pragma once
#include "config.h"
#include "model.h"
#include "kv_cache.h"

// ============================================================================
// Multi-kernel path: host drives the generation loop, one kernel launch per
// forward pass step.  Higher launch overhead but simpler to debug.
// ============================================================================

// Baseline autoregressive: target model only, one token at a time.
void multikernel_baseline(const ModelWeights& target_model,
                          KVCache& target_kv,
                          const int* prompt, int prompt_len,
                          GenerationResult* d_result,
                          const GenerationParams& params);

// Speculative decoding: draft proposes, target verifies.
void multikernel_speculative(const ModelWeights& draft_model,
                             const ModelWeights& target_model,
                             KVCache& draft_kv,
                             KVCache& target_kv,
                             const int* prompt, int prompt_len,
                             GenerationResult* d_result,
                             const GenerationParams& params);

// ============================================================================
// Persistent megakernel path: single kernel launch, zero CPU-GPU sync during
// generation.  The loop runs entirely on the GPU.
// ============================================================================

void megakernel_baseline(const ModelWeights& target_model,
                         KVCache& target_kv,
                         const int* prompt, int prompt_len,
                         GenerationResult* d_result,
                         const GenerationParams& params);

void megakernel_speculative(const ModelWeights& draft_model,
                            const ModelWeights& target_model,
                            KVCache& draft_kv,
                            KVCache& target_kv,
                            const int* prompt, int prompt_len,
                            GenerationResult* d_result,
                            const GenerationParams& params);
