#include "config.h"
#include "utils.h"
#include "model.h"
#include "kv_cache.h"
#include "kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace py = pybind11;

// ============================================================================
// run_benchmark: called from Python CLI.
// Returns a dict with timing, tokens, and statistics.
// ============================================================================

static py::dict run_benchmark(const std::string& mode,
                              int max_tokens,
                              int spec_k,
                              unsigned seed,
                              int prompt_len) {
    bool use_mega = (mode == "mega");

    // Validate
    if (max_tokens <= 0 || max_tokens > MAX_SEQ_LEN)
        throw std::runtime_error("max_tokens must be in [1, " +
                                  std::to_string(MAX_SEQ_LEN) + "]");
    if (spec_k <= 0 || spec_k > DEFAULT_SPEC_K * 2)
        throw std::runtime_error("spec_k out of range");
    if (prompt_len <= 0 || prompt_len > MAX_SEQ_LEN / 2)
        throw std::runtime_error("prompt_len out of range");

    // Allocate models
    ModelConfig draft_cfg  = make_draft_config();
    ModelConfig target_cfg = make_target_config();

    ModelWeights draft_model, target_model;
    model_alloc(draft_model, draft_cfg);
    model_alloc(target_model, target_cfg);
    model_init_random(draft_model, seed);
    model_init_random(target_model, seed + 1000);

    // KV caches
    KVCache draft_kv, target_kv, baseline_kv;
    kv_cache_alloc(draft_kv, draft_cfg.n_layers, draft_cfg.d_head, MAX_KV_BLOCKS);
    kv_cache_alloc(target_kv, target_cfg.n_layers, target_cfg.d_head, MAX_KV_BLOCKS);
    kv_cache_alloc(baseline_kv, target_cfg.n_layers, target_cfg.d_head, MAX_KV_BLOCKS);

    // Prompt
    std::vector<int> h_prompt(prompt_len);
    for (int i = 0; i < prompt_len; i++)
        h_prompt[i] = (i + 1) % DEFAULT_VOCAB_SIZE;

    int* d_prompt;
    CUDA_CHECK(cudaMalloc(&d_prompt, prompt_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt.data(), prompt_len * sizeof(int),
                           cudaMemcpyHostToDevice));

    GenerationResult* d_baseline_res;
    GenerationResult* d_spec_res;
    CUDA_CHECK(cudaMalloc(&d_baseline_res, sizeof(GenerationResult)));
    CUDA_CHECK(cudaMalloc(&d_spec_res, sizeof(GenerationResult)));

    GenerationParams params;
    params.max_new_tokens = max_tokens;
    params.spec_k         = spec_k;
    params.use_megakernel = use_mega;

    // ---- Baseline ----
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    if (use_mega)
        megakernel_baseline(target_model, baseline_kv, d_prompt, prompt_len,
                            d_baseline_res, params);
    else
        multikernel_baseline(target_model, baseline_kv, h_prompt.data(), prompt_len,
                             d_baseline_res, params);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float baseline_ms;
    CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, ev_start, ev_stop));

    GenerationResult h_baseline;
    CUDA_CHECK(cudaMemcpy(&h_baseline, d_baseline_res,
                           sizeof(GenerationResult), cudaMemcpyDeviceToHost));

    // ---- Speculative ----
    CUDA_CHECK(cudaEventRecord(ev_start));
    if (use_mega)
        megakernel_speculative(draft_model, target_model, draft_kv, target_kv,
                               d_prompt, prompt_len, d_spec_res, params);
    else
        multikernel_speculative(draft_model, target_model, draft_kv, target_kv,
                                h_prompt.data(), prompt_len, d_spec_res, params);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float spec_ms;
    CUDA_CHECK(cudaEventElapsedTime(&spec_ms, ev_start, ev_stop));

    GenerationResult h_spec;
    CUDA_CHECK(cudaMemcpy(&h_spec, d_spec_res,
                           sizeof(GenerationResult), cudaMemcpyDeviceToHost));

    // Verify
    bool match = (h_baseline.n_generated == h_spec.n_generated);
    for (int i = 0; i < h_baseline.n_generated && match; i++) {
        if (h_baseline.output_tokens[i] != h_spec.output_tokens[i])
            match = false;
    }

    // Build result dict
    py::dict result;
    result["mode"] = mode;

    std::vector<int> bl_tokens(h_baseline.output_tokens,
                               h_baseline.output_tokens + h_baseline.n_generated);
    std::vector<int> sp_tokens(h_spec.output_tokens,
                               h_spec.output_tokens + h_spec.n_generated);

    result["baseline_tokens"]    = bl_tokens;
    result["baseline_n"]         = h_baseline.n_generated;
    result["baseline_ms"]        = baseline_ms;
    result["baseline_tok_per_s"] = h_baseline.n_generated / (baseline_ms / 1000.0f);

    result["spec_tokens"]        = sp_tokens;
    result["spec_n"]             = h_spec.n_generated;
    result["spec_ms"]            = spec_ms;
    result["spec_tok_per_s"]     = h_spec.n_generated / (spec_ms / 1000.0f);
    result["draft_proposed"]     = h_spec.draft_proposed;
    result["draft_accepted"]     = h_spec.draft_accepted;
    result["acceptance_rate"]    = h_spec.draft_proposed > 0
        ? (float)h_spec.draft_accepted / h_spec.draft_proposed : 0.0f;
    result["spec_iterations"]    = h_spec.spec_iterations;
    result["speedup"]            = baseline_ms / spec_ms;
    result["match"]              = match;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_prompt);
    cudaFree(d_baseline_res);
    cudaFree(d_spec_res);
    kv_cache_free(draft_kv);
    kv_cache_free(target_kv);
    kv_cache_free(baseline_kv);
    model_free(draft_model);
    model_free(target_model);

    return result;
}

// ============================================================================
// pybind11 module definition
// ============================================================================

PYBIND11_MODULE(spec_decode_cuda, m) {
    m.doc() = "CUDA speculative decoding engine";
    m.def("run_benchmark", &run_benchmark,
          py::arg("mode") = "multi",
          py::arg("max_tokens") = 32,
          py::arg("spec_k") = DEFAULT_SPEC_K,
          py::arg("seed") = 42,
          py::arg("prompt_len") = 4,
          "Run baseline + speculative decoding benchmark. "
          "Returns a dict with timing and token results.");
}
