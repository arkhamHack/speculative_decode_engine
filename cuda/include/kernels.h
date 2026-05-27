#pragma once
#include "config.h"
#include "model.h"
#include "kv_cache.h"

// ============================================================================
// Multi-kernel path: host drives the generation loop, one kernel launch per
// forward pass step.  Higher launch overhead but simpler to debug.
// ============================================================================

// Baseline autoregressive: target model only, one token at a time.
// eng: optional InferenceEngine for cooperative-launch and stream-overlap paths.
void multikernel_baseline(const ModelWeights& target_model,
                          KVCache& target_kv,
                          const int* prompt, int prompt_len,
                          GenerationResult* d_result,
                          const GenerationParams& params,
                          InferenceEngine* eng = nullptr);

// Speculative decoding: draft proposes, target verifies.
// eng: optional InferenceEngine for cooperative-launch and stream-overlap paths.
void multikernel_speculative(const ModelWeights& draft_model,
                             const ModelWeights& target_model,
                             KVCache& draft_kv,
                             KVCache& target_kv,
                             const int* prompt, int prompt_len,
                             GenerationResult* d_result,
                             const GenerationParams& params,
                             InferenceEngine* eng = nullptr);

// ============================================================================
// Persistent megakernel path: typically a single launch (no CPU sync mid-decode).
// Greedy speculative and stochastic speculative (--stochastic-spec) both route here.
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
