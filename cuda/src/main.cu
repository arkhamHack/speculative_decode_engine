#include "config.h"
#include "utils.h"
#include "model.h"
#include "kv_cache.h"
#include "kernels.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// Argument parsing
// ============================================================================

struct Args {
    bool   use_mega     = false;           // --mode=mega
    int    max_new      = 32;              // --max-tokens=N
    int    spec_k       = DEFAULT_SPEC_K;  // --k=N
    unsigned seed       = 42;             // --seed=N
    int    prompt_len   = 4;              // --prompt-len=N  (dummy tokenizer only)
    bool   stochastic_spec = false;       // --stochastic-spec
    float  draft_temp   = 1.f;            // --draft-temp=T
    bool   adaptive_draft_temp = false;   // --adaptive-draft-temp
    float  adapt_accept_target = 0.50f;   // --adapt-accept=R  ( EWMA centre, default 0.5 )
    float  adapt_gain        = 0.055f;   // --adapt-gain=G
    float  adapt_ewma_mix    = 0.25f;   // --adapt-ewma=M (smoothing λ on per-round rate)
    unsigned spec_rng_seed = 12345;       // --spec-seed=N
    char   draft_path[512]  = "";         // --draft=path/to/draft.bin
    char   target_path[512] = "";         // --target=path/to/target.bin
    char   prompt_tok[512]  = "";         // --prompt-tok=path/to/prompt.tok
    char   output_tok[512]  = "";         // --output-tok=path  (write generated ids)
};

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Dummy-tokenizer mode (default, no model files needed):\n"
        "  --mode=multi|mega     kernel path (default: multi)\n"
        "  --max-tokens=N        tokens to generate (default: 32)\n"
        "  --k=N                 speculative draft depth (default: %d)\n"
        "  --seed=N              random weight seed (default: 42)\n"
        "  --prompt-len=N        dummy prompt length (default: 4)\n"
        "  --stochastic-spec     distribution-matching speculative verify (mega + multi)\n"
        "  --draft-temp=T        draft softmax temperature (default: 1)\n"
        "  --adaptive-draft-temp EWMA nudge draft temp using --adapt-* heuristics\n"
        "  --adapt-accept=R      EWMA acceptance target ∈ (0,1), default 0.5\n"
        "  --adapt-gain=G        Δtemp per EWMA deviation step, default 0.055\n"
        "  --adapt-ewma=M       mix weight for EWMA ∈ (0,1), default 0.25\n"
        "  --spec-seed=N         RNG seed for stochastic spec (default: 12345)\n"
        "\n"
        "Real-model mode (requires SDEC weight files + tokenised prompt):\n"
        "  --draft=path.bin      draft model SDEC binary (from tools/export_model.py)\n"
        "  --target=path.bin     target model SDEC binary\n"
        "  --prompt-tok=path.tok tokenised prompt (from tools/hf_tok.py encode)\n"
        "  --output-tok=path.tok write generated token ids here (decode with Python)\n"
        "\n"
        "Tool scripts (in tools/):\n"
        "  python tools/export_model.py <hf_model> <out.bin>   -- export HF model\n"
        "  python tools/hf_tok.py encode <model> <text> <out.tok>\n"
        "  python tools/hf_tok.py decode <model> <in.tok>\n",
        prog, DEFAULT_SPEC_K);
}

static void parse_args(Args& args, int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            args.use_mega = strcmp(argv[i] + 7, "mega") == 0;
        } else if (strncmp(argv[i], "--max-tokens=", 13) == 0) {
            args.max_new = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--k=", 4) == 0) {
            args.spec_k = atoi(argv[i] + 4);
        } else if (strncmp(argv[i], "--seed=", 7) == 0) {
            args.seed = (unsigned)atoi(argv[i] + 7);
        } else if (strcmp(argv[i], "--stochastic-spec") == 0) {
            args.stochastic_spec = true;
        } else if (strncmp(argv[i], "--draft-temp=", 13) == 0) {
            args.draft_temp = static_cast<float>(atof(argv[i] + 13));
        } else if (strcmp(argv[i], "--adaptive-draft-temp") == 0) {
            args.adaptive_draft_temp = true;
        } else if (strncmp(argv[i], "--adapt-accept=", 15) == 0) {
            args.adapt_accept_target =
                static_cast<float>(atof(argv[i] + 15));
        } else if (strncmp(argv[i], "--adapt-gain=", 13) == 0) {
            args.adapt_gain = static_cast<float>(atof(argv[i] + 13));
        } else if (strncmp(argv[i], "--adapt-ewma=", 14) == 0) {
            args.adapt_ewma_mix = static_cast<float>(atof(argv[i] + 14));
        } else if (strncmp(argv[i], "--spec-seed=", 12) == 0) {
            args.spec_rng_seed = (unsigned)atoi(argv[i] + 12);
        } else if (strncmp(argv[i], "--prompt-len=", 13) == 0) {
            args.prompt_len = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--draft=", 8) == 0) {
            strncpy(args.draft_path, argv[i] + 8, 511);
        } else if (strncmp(argv[i], "--target=", 9) == 0) {
            strncpy(args.target_path, argv[i] + 9, 511);
        } else if (strncmp(argv[i], "--prompt-tok=", 13) == 0) {
            strncpy(args.prompt_tok, argv[i] + 13, 511);
        } else if (strncmp(argv[i], "--output-tok=", 13) == 0) {
            strncpy(args.output_tok, argv[i] + 13, 511);
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]); exit(1);
        }
    }
}

// ============================================================================
// Print helpers
// ============================================================================

static void print_tokens(const char* label, const int* tokens, int n) {
    // Web UI parsing (runner._parse_token_list) requires digits/spaces only after ':' on
    // this line — no "..." truncation, or n>20 yields an empty parsed token list.
    printf("%s [%d tokens]: ", label, n);
    for (int i = 0; i < n; i++) printf("%d ", tokens[i]);
    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    Args args;
    parse_args(args, argc, argv);

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // ---- Decide operating mode: real weights or dummy ----
    bool use_real = (args.draft_path[0] != '\0' &&
                     args.target_path[0] != '\0');

    printf("Mode: %s | kernel=%s | max_tokens=%d | k=%d | stochastic_spec=%d\n\n",
           use_real ? "real-model" : "dummy",
           args.use_mega ? "megakernel" : "multi-kernel",
           args.max_new, args.spec_k, args.stochastic_spec ? 1 : 0);

    // ---- Build / load models ----
    ModelWeights draft_model, target_model;

    if (use_real) {
        printf("Loading draft model: %s\n", args.draft_path);
        if (!model_load_weights(draft_model, args.draft_path, nullptr)) {
            fprintf(stderr, "Failed to load draft model\n"); return 1;
        }
        printf("Loading target model: %s\n", args.target_path);
        if (!model_load_weights(target_model, args.target_path, nullptr)) {
            fprintf(stderr, "Failed to load target model\n"); return 1;
        }
        printf("\n");
    } else {
        ModelConfig draft_cfg  = make_draft_config();
        ModelConfig target_cfg = make_target_config();
        model_alloc(draft_model,  draft_cfg);
        model_alloc(target_model, target_cfg);
        model_init_random(draft_model,  args.seed);
        model_init_random(target_model, args.seed + 1000);
        printf("Dummy models: draft d=%d/%dh, target d=%d/%dh, vocab=%d\n",
               draft_cfg.d_model, draft_cfg.n_heads,
               target_cfg.d_model, target_cfg.n_heads,
               target_cfg.vocab_size);
    }

    // ---- Build prompt ----
    int* h_prompt  = nullptr;
    int  prompt_len = 0;

    if (use_real && args.prompt_tok[0] != '\0') {
        h_prompt = tok_load_alloc(args.prompt_tok, &prompt_len);
        if (!h_prompt) {
            fprintf(stderr, "Failed to load prompt from %s\n", args.prompt_tok);
            return 1;
        }
        printf("Prompt: %d tokens loaded from %s\n\n", prompt_len, args.prompt_tok);
    } else {
        // Dummy prompt: tokens 1, 2, 3, ...
        prompt_len = args.prompt_len;
        h_prompt   = new int[prompt_len];
        for (int i = 0; i < prompt_len; i++)
            h_prompt[i] = (i + 1) % DEFAULT_VOCAB_SIZE;
        printf("Dummy prompt: %d tokens\n\n", prompt_len);
    }

    // Copy prompt to device (needed by megakernel path)
    int* d_prompt;
    CUDA_CHECK(cudaMalloc(&d_prompt, prompt_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt, prompt_len * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ---- Allocate KV caches ----
    ModelConfig& dc = draft_model.cfg;
    ModelConfig& tc = target_model.cfg;

    KVCache draft_kv, target_kv, baseline_kv;
    kv_cache_alloc(draft_kv,    dc.n_layers, dc.d_head, MAX_KV_BLOCKS);
    kv_cache_alloc(target_kv,   tc.n_layers, tc.d_head, MAX_KV_BLOCKS);
    kv_cache_alloc(baseline_kv, tc.n_layers, tc.d_head, MAX_KV_BLOCKS);

    // ---- Result buffers ----
    GenerationResult* d_baseline_result;
    GenerationResult* d_spec_result;
    CUDA_CHECK(cudaMalloc(&d_baseline_result, sizeof(GenerationResult)));
    CUDA_CHECK(cudaMalloc(&d_spec_result,     sizeof(GenerationResult)));

    GenerationParams params;
    params.max_new_tokens = args.max_new;
    params.spec_k         = args.spec_k;
    params.use_megakernel = args.use_mega;
    params.stochastic_spec_decode  = args.stochastic_spec;
    params.draft_temperature       = args.draft_temp;
    params.adaptive_draft_temperature = args.adaptive_draft_temp;
    params.stochastic_rng_seed     = args.spec_rng_seed;
    params.stochastic_adapt_target_accept = args.adapt_accept_target;
    params.stochastic_adapt_temp_gain     = args.adapt_gain;
    params.stochastic_adapt_ewma_mix       = args.adapt_ewma_mix;

    // ---- Warmup ----
    printf("Warming up (first-call JIT and CUDA init)...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==========================================================
    // Baseline: target-only autoregressive
    // ==========================================================
    printf("\n=== Baseline (target-only) ===\n");

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    if (args.use_mega) {
        megakernel_baseline(target_model, baseline_kv,
                            d_prompt, prompt_len, d_baseline_result, params);
    } else {
        multikernel_baseline(target_model, baseline_kv,
                             h_prompt, prompt_len, d_baseline_result, params);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float baseline_ms;
    CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, ev_start, ev_stop));

    GenerationResult h_baseline;
    CUDA_CHECK(cudaMemcpy(&h_baseline, d_baseline_result,
                          sizeof(GenerationResult), cudaMemcpyDeviceToHost));

    print_tokens("Baseline out", h_baseline.output_tokens, h_baseline.n_generated);
    printf("Time: %.2f ms | Tokens: %d | Tok/s: %.1f\n",
           baseline_ms, h_baseline.n_generated,
           h_baseline.n_generated / (baseline_ms / 1000.0f));

    // ==========================================================
    // Speculative decoding
    // ==========================================================
    printf("\n=== Speculative (draft+target, k=%d) ===\n", args.spec_k);

    CUDA_CHECK(cudaEventRecord(ev_start));
    if (args.use_mega) {
        megakernel_speculative(draft_model, target_model,
                               draft_kv, target_kv,
                               d_prompt, prompt_len, d_spec_result, params);
    } else {
        multikernel_speculative(draft_model, target_model,
                                draft_kv, target_kv,
                                h_prompt, prompt_len, d_spec_result, params);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float spec_ms;
    CUDA_CHECK(cudaEventElapsedTime(&spec_ms, ev_start, ev_stop));

    GenerationResult h_spec;
    CUDA_CHECK(cudaMemcpy(&h_spec, d_spec_result,
                          sizeof(GenerationResult), cudaMemcpyDeviceToHost));

    print_tokens("Speculative  ", h_spec.output_tokens, h_spec.n_generated);
    printf("Time: %.2f ms | Tokens: %d | Tok/s: %.1f\n",
           spec_ms, h_spec.n_generated,
           h_spec.n_generated / (spec_ms / 1000.0f));
    printf("Draft proposed: %d | Accepted: %d | Rate: %.1f%%\n",
           h_spec.draft_proposed, h_spec.draft_accepted,
           h_spec.draft_proposed > 0
               ? 100.0f * h_spec.draft_accepted / h_spec.draft_proposed
               : 0.0f);
    printf("Speculation iterations: %d\n", h_spec.spec_iterations);

    // ==========================================================
    // Correctness verification
    // ==========================================================
    printf("\n=== Verification ===\n");
    if (params.stochastic_spec_decode) {
        printf("Byte-for-byte baseline match skipped "
               "(stochastic speculative sampling is non-deterministic).\n");
    } else {
        int min_len = h_baseline.n_generated < h_spec.n_generated
                    ? h_baseline.n_generated : h_spec.n_generated;
        bool match  = (h_baseline.n_generated == h_spec.n_generated);
        for (int i = 0; i < min_len && match; i++) {
            if (h_baseline.output_tokens[i] != h_spec.output_tokens[i])
                match = false;
        }
        printf("Output match: %s\n", match ? "PASS ✓" : "FAIL ✗");
        if (!match) {
            printf("  Lengths: baseline=%d spec=%d\n",
                   h_baseline.n_generated, h_spec.n_generated);
            for (int i = 0; i < min_len; i++) {
                if (h_baseline.output_tokens[i] != h_spec.output_tokens[i]) {
                    printf("  First mismatch @ [%d]: baseline=%d spec=%d\n",
                           i, h_baseline.output_tokens[i],
                           h_spec.output_tokens[i]);
                    break;
                }
            }
        }
    }

    // ==========================================================
    // Summary  (machine-parseable lines for web/backend/runner.py)
    // ==========================================================
    printf("\n=== Summary ===\n");
    printf("Baseline:    %.2f ms  (%.1f tok/s)\n",
           baseline_ms, h_baseline.n_generated / (baseline_ms / 1000.0f));
    printf("Speculative: %.2f ms  (%.1f tok/s)\n",
           spec_ms, h_spec.n_generated / (spec_ms / 1000.0f));
    printf("Speedup:     %.3fx\n", baseline_ms / spec_ms);
    printf("Accept rate: %.3f\n",
           h_spec.draft_proposed > 0
               ? (float)h_spec.draft_accepted / h_spec.draft_proposed
               : 0.0f);

    // ==========================================================
    // Optional: write generated token ids to file (for Python decoding)
    // ==========================================================
    if (use_real && args.output_tok[0] != '\0') {
        if (tok_save(args.output_tok, h_spec.output_tokens, h_spec.n_generated))
            printf("\nGenerated tokens saved to: %s\n", args.output_tok);
        printf("Decode with: python tools/hf_tok.py decode <model> %s\n",
               args.output_tok);
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_prompt);
    cudaFree(d_baseline_result);
    cudaFree(d_spec_result);
    kv_cache_free(draft_kv);
    kv_cache_free(target_kv);
    kv_cache_free(baseline_kv);
    model_free(draft_model);
    model_free(target_model);
    delete[] h_prompt;

    printf("\nDone.\n");
    return 0;
}
