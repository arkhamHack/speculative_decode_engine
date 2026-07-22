#include "config.h"
#include "utils.h"
#include "model.h"
#include "kv_cache.h"
#include "kernels.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ============================================================================
// Argument parsing
// ============================================================================

struct Args {
    bool   use_mega     = false;           // --mode=mega
    int    max_new      = 32;              // --max-tokens=N
    int    spec_k       = DEFAULT_SPEC_K;  // --k=N
    int    prompt_len   = 4;              // --prompt-len=N  (dummy tokenizer only)
    bool   stochastic_spec = false;       // --stochastic-spec
    float  draft_temp   = 1.f;            // --draft-temp=T
    bool   adaptive_draft_temp = false;   // --adaptive-draft-temp
    float  adapt_accept_target = 0.50f;   // --adapt-accept=R
    float  adapt_gain        = 0.055f;   // --adapt-gain=G
    float  adapt_ewma_mix    = 0.25f;   // --adapt-ewma=M
    unsigned spec_rng_seed = 12345;       // --spec-seed=N
    int    eos_token        = -1;         // --eos-token=N
    char   draft_path[512]  = "";         // --draft=path/to/draft.bin
    char   target_path[512] = "";         // --target=path/to/target.bin
    char   prompt_tok[512]  = "";         // --prompt-tok=path/to/prompt.tok
    char   output_tok[512]  = "";         // --output-tok=path
    bool   self_spec        = false;      // --self-spec
    int    draft_layers     = 0;          // --draft-layers=N
    GemmBackendType gemm_backend = GEMM_BACKEND_LEGACY;  // --gemm-backend=
    bool   repl             = false;      // --repl  (keep weights loaded)
    bool   spec_only        = false;      // --spec-only (skip baseline compare)
    bool   bench            = false;      // --bench  (force baseline+spec compare)
};

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Serving / interactive (weights stay resident):\n"
        "  --repl                load models once, then accept many prompts\n"
        "  --spec-only           generate with speculative decode only (no baseline)\n"
        "  --bench               compare baseline vs speculative (default without --repl)\n"
        "\n"
        "Common options:\n"
        "  --mode=multi|mega     kernel path (default: multi)\n"
        "  --max-tokens=N        tokens to generate (default: 32)\n"
        "  --k=N                 speculative draft depth (default: %d)\n"
        "  --prompt-len=N        dummy prompt length (default: 4)\n"
        "  --stochastic-spec     distribution-matching speculative verify\n"
        "  --draft-temp=T        draft softmax temperature (default: 1)\n"
        "  --adaptive-draft-temp EWMA nudge draft temp using --adapt-* heuristics\n"
        "  --adapt-accept=R      EWMA acceptance target ∈ (0,1), default 0.5\n"
        "  --adapt-gain=G        Δtemp per EWMA deviation step, default 0.055\n"
        "  --adapt-ewma=M        mix weight for EWMA ∈ (0,1), default 0.25\n"
        "  --spec-seed=N         RNG seed for stochastic spec (default: 12345)\n"
        "  --eos-token=N         stop generation on this token id\n"
        "  --self-spec           early layers draft, full model target\n"
        "  --draft-layers=N      layers 1..N as draft proposer (default: target/2)\n"
        "  --gemm-backend=X      cublas | legacy (default)\n"
        "\n"
        "Real-model mode:\n"
        "  --draft=path.bin      draft model SDEC binary\n"
        "  --target=path.bin     target model SDEC binary\n"
        "  --prompt-tok=path.tok tokenised prompt (tools/hf_tok.py encode)\n"
        "  --output-tok=path.tok write generated token ids\n"
        "\n"
        "REPL commands (when --repl):\n"
        "  <path.tok>            run generation on that prompt file\n"
        "  max-tokens N          change --max-tokens for subsequent runs\n"
        "  k N                   change speculation depth\n"
        "  output <path.tok>     set default --output-tok (empty clears)\n"
        "  bench on|off          toggle baseline comparison\n"
        "  help                  show REPL help\n"
        "  quit | exit           unload models and exit\n"
        "\n"
        "Example (experimental vLLM-style session):\n"
        "  %s --target=weights/m.bin --self-spec --gemm-backend=cublas --repl\n"
        "  sdec> prompt.tok\n"
        "  sdec> another.tok\n"
        "  sdec> quit\n"
        "\n"
        "Tools:\n"
        "  python tools/export_model.py <hf_model> -o out.bin\n"
        "  python tools/hf_tok.py encode <model> <text> <out.tok>\n"
        "  python tools/hf_tok.py decode <model> <in.tok>\n",
        prog, DEFAULT_SPEC_K, prog);
}

static void parse_args(Args& args, int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            args.use_mega = strcmp(argv[i] + 7, "mega") == 0;
        } else if (strncmp(argv[i], "--max-tokens=", 13) == 0) {
            args.max_new = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--k=", 4) == 0) {
            args.spec_k = atoi(argv[i] + 4);
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
        } else if (strncmp(argv[i], "--eos-token=", 12) == 0) {
            args.eos_token = atoi(argv[i] + 12);
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
        } else if (strcmp(argv[i], "--self-spec") == 0) {
            args.self_spec = true;
        } else if (strncmp(argv[i], "--draft-layers=", 15) == 0) {
            args.draft_layers = atoi(argv[i] + 15);
        } else if (strncmp(argv[i], "--gemm-backend=", 15) == 0) {
            const char* val = argv[i] + 15;
            if (strcmp(val, "cublas") == 0) args.gemm_backend = GEMM_BACKEND_CUBLAS;
            else if (strcmp(val, "legacy") == 0) args.gemm_backend = GEMM_BACKEND_LEGACY;
            else { fprintf(stderr, "Unknown gemm backend: %s\n", val); exit(1); }
        } else if (strcmp(argv[i], "--repl") == 0) {
            args.repl = true;
        } else if (strcmp(argv[i], "--spec-only") == 0) {
            args.spec_only = true;
        } else if (strcmp(argv[i], "--bench") == 0) {
            args.bench = true;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]); exit(1);
        }
    }

    // REPL implies serve mode (spec-only) unless --bench was requested.
    if (args.repl && !args.bench)
        args.spec_only = true;
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

static void trim_inplace(char* s) {
    char* start = s;
    while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n')
        start++;
    if (start != s) memmove(s, start, strlen(start) + 1);
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' ||
                     s[n - 1] == '\r' || s[n - 1] == '\n')) {
        s[--n] = '\0';
    }
}

// ============================================================================
// Persistent session — models / KV / engine loaded once
// ============================================================================

struct Session {
    Args             args;
    bool             use_real       = false;
    bool             draft_is_alias = false;
    bool             kv_shared      = false;
    bool             do_baseline    = true;   // false when --spec-only

    ModelWeights     draft_model{};
    ModelWeights     target_model{};
    KVCache          draft_kv{};
    KVCache          target_kv{};
    KVCache          baseline_kv{};
    InferenceEngine  eng{};

    GenerationResult* d_baseline_result = nullptr;
    GenerationResult* d_spec_result     = nullptr;
    int*              d_prompt          = nullptr;  // capacity MAX_SEQ_LEN

    GenerationParams params{};
    cudaEvent_t      ev_start{}, ev_stop{};
};

static void session_fill_params(Session& s) {
    s.params.max_new_tokens               = s.args.max_new;
    s.params.spec_k                       = s.args.spec_k;
    s.params.use_megakernel               = s.args.use_mega;
    s.params.stochastic_spec_decode       = s.args.stochastic_spec;
    s.params.draft_temperature            = s.args.draft_temp;
    s.params.adaptive_draft_temperature   = s.args.adaptive_draft_temp;
    s.params.stochastic_rng_seed          = s.args.spec_rng_seed;
    s.params.stochastic_adapt_target_accept = s.args.adapt_accept_target;
    s.params.stochastic_adapt_temp_gain   = s.args.adapt_gain;
    s.params.stochastic_adapt_ewma_mix    = s.args.adapt_ewma_mix;
    s.params.eos_token                    = s.args.eos_token;
    s.params.self_speculative             = s.args.self_spec;
    s.params.n_draft_layers               = s.draft_model.cfg.n_layers;
}

// Run one generation. Prompt must already be validated (1..MAX_SEQ_LEN).
// Returns false on CUDA / logic failure (rare); true otherwise.
static bool session_generate(Session& s, const int* h_prompt, int prompt_len,
                             const char* output_tok_path) {
    if (prompt_len <= 0 || prompt_len > MAX_SEQ_LEN) {
        fprintf(stderr, "Invalid prompt length %d (max %d)\n",
                prompt_len, MAX_SEQ_LEN);
        return false;
    }

    session_fill_params(s);

    // Fresh KV for this request (vLLM-style: new sequence, same resident weights).
    if (s.kv_shared) {
        kv_cache_reset(s.target_kv);
    } else {
        kv_cache_reset(s.draft_kv);
        kv_cache_reset(s.target_kv);
    }
    if (s.do_baseline)
        kv_cache_reset(s.baseline_kv);

    CUDA_CHECK(cudaMemcpy(s.d_prompt, h_prompt, prompt_len * sizeof(int),
                          cudaMemcpyHostToDevice));

    float baseline_ms = 0.f;
    GenerationResult h_baseline{};

    if (s.do_baseline) {
        printf("\n=== Baseline (target-only) ===\n");
        CUDA_CHECK(cudaEventRecord(s.ev_start));
        if (s.args.use_mega) {
            megakernel_baseline(s.target_model, s.baseline_kv,
                                s.d_prompt, prompt_len, s.d_baseline_result,
                                s.params);
        } else {
            multikernel_baseline(s.target_model, s.baseline_kv,
                                 h_prompt, prompt_len, s.d_baseline_result,
                                 s.params, &s.eng);
        }
        CUDA_CHECK(cudaEventRecord(s.ev_stop));
        CUDA_CHECK(cudaEventSynchronize(s.ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, s.ev_start, s.ev_stop));
        CUDA_CHECK(cudaMemcpy(&h_baseline, s.d_baseline_result,
                              sizeof(GenerationResult), cudaMemcpyDeviceToHost));
        print_tokens("Baseline out", h_baseline.output_tokens, h_baseline.n_generated);
        printf("Time: %.2f ms | Tokens: %d | Tok/s: %.1f\n",
               baseline_ms, h_baseline.n_generated,
               h_baseline.n_generated / (baseline_ms / 1000.0f));
    }

    printf("\n=== Speculative (%s, k=%d) ===\n",
           s.args.self_spec ? "self-spec" : "draft+target", s.args.spec_k);

    CUDA_CHECK(cudaEventRecord(s.ev_start));
    if (s.args.use_mega) {
        megakernel_speculative(s.draft_model, s.target_model,
                               s.draft_kv, s.target_kv,
                               s.d_prompt, prompt_len, s.d_spec_result, s.params);
    } else {
        multikernel_speculative(s.draft_model, s.target_model,
                                s.draft_kv, s.target_kv,
                                h_prompt, prompt_len, s.d_spec_result,
                                s.params, &s.eng);
    }
    CUDA_CHECK(cudaEventRecord(s.ev_stop));
    CUDA_CHECK(cudaEventSynchronize(s.ev_stop));

    float spec_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&spec_ms, s.ev_start, s.ev_stop));

    GenerationResult h_spec{};
    CUDA_CHECK(cudaMemcpy(&h_spec, s.d_spec_result,
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

    if (s.do_baseline) {
        printf("\n=== Verification ===\n");
        if (s.params.stochastic_spec_decode) {
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
            printf("Output match: %s\n", match ? "PASS" : "FAIL");
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

        printf("\n=== Summary ===\n");
        printf("Baseline:    %.2f ms  (%.1f tok/s)\n",
               baseline_ms, h_baseline.n_generated / (baseline_ms / 1000.0f));
        printf("Speculative: %.2f ms  (%.1f tok/s)\n",
               spec_ms, h_spec.n_generated / (spec_ms / 1000.0f));
        printf("Speedup:     %.3fx\n",
               spec_ms > 0.f ? baseline_ms / spec_ms : 0.f);
        printf("Accept rate: %.3f\n",
               h_spec.draft_proposed > 0
                   ? (float)h_spec.draft_accepted / h_spec.draft_proposed
                   : 0.0f);
    } else {
        printf("\n=== Summary ===\n");
        printf("Speculative: %.2f ms  (%.1f tok/s)\n",
               spec_ms, h_spec.n_generated / (spec_ms / 1000.0f));
        printf("Accept rate: %.3f\n",
               h_spec.draft_proposed > 0
                   ? (float)h_spec.draft_accepted / h_spec.draft_proposed
                   : 0.0f);
    }

    if (output_tok_path && output_tok_path[0] != '\0') {
        if (tok_save(output_tok_path, h_spec.output_tokens, h_spec.n_generated))
            printf("\nGenerated tokens saved to: %s\n", output_tok_path);
        printf("Decode with: python tools/hf_tok.py decode <model> %s\n",
               output_tok_path);
    }

    fflush(stdout);
    return true;
}

static void print_repl_help() {
    printf(
        "Commands:\n"
        "  <path.tok>         generate from a tokenised prompt file\n"
        "  max-tokens N       set generation length\n"
        "  k N                set speculation depth\n"
        "  output <path>|off  set/clear default output .tok path\n"
        "  bench on|off       enable/disable baseline comparison\n"
        "  help               this help\n"
        "  quit | exit        exit (unload weights)\n"
        "\n"
        "Tip: encode prompts with:\n"
        "  python tools/hf_tok.py encode <hf_model> \"your text\" prompt.tok\n");
}

static void run_repl(Session& s) {
    printf("\n=== REPL (weights resident) ===\n");
    printf("Models stay loaded. Type a .tok path to generate, or 'help'.\n");
    print_repl_help();
    printf("sdec> ");
    fflush(stdout);

    char line[1024];
    while (fgets(line, sizeof(line), stdin)) {
        trim_inplace(line);
        if (line[0] == '\0') {
            printf("sdec> ");
            fflush(stdout);
            continue;
        }

        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0 ||
            strcmp(line, "q") == 0) {
            break;
        }
        if (strcmp(line, "help") == 0 || strcmp(line, "?") == 0) {
            print_repl_help();
            printf("sdec> ");
            fflush(stdout);
            continue;
        }

        int mt = 0, kk = 0;
        char out_buf[512] = "";
        char bench_buf[32] = "";
        if (sscanf(line, "max-tokens %d", &mt) == 1) {
            if (mt < 1 || mt > MAX_SEQ_LEN - 1) {
                fprintf(stderr, "max-tokens must be in [1, %d]\n", MAX_SEQ_LEN - 1);
            } else {
                s.args.max_new = mt;
                printf("max-tokens = %d\n", mt);
            }
        } else if (sscanf(line, "k %d", &kk) == 1) {
            if (kk < 1 || kk > 16) {
                fprintf(stderr, "k must be in [1, 16]\n");
            } else {
                s.args.spec_k = kk;
                printf("k = %d\n", kk);
            }
        } else if (sscanf(line, "output %511s", out_buf) == 1) {
            if (strcmp(out_buf, "off") == 0 || strcmp(out_buf, "clear") == 0) {
                s.args.output_tok[0] = '\0';
                printf("output-tok cleared\n");
            } else {
                strncpy(s.args.output_tok, out_buf, 511);
                s.args.output_tok[511] = '\0';
                printf("output-tok = %s\n", s.args.output_tok);
            }
        } else if (sscanf(line, "bench %31s", bench_buf) == 1) {
            if (strcmp(bench_buf, "on") == 0) {
                s.do_baseline = true;
                printf("bench = on (baseline + speculative)\n");
            } else if (strcmp(bench_buf, "off") == 0) {
                s.do_baseline = false;
                printf("bench = off (spec-only)\n");
            } else {
                fprintf(stderr, "usage: bench on|off\n");
            }
        } else {
            // Treat as prompt .tok path
            int prompt_len = 0;
            int* h_prompt = tok_load_alloc(line, &prompt_len);
            if (!h_prompt) {
                fprintf(stderr, "Failed to load prompt '%s'\n", line);
            } else {
                printf("Prompt: %d tokens from %s\n", prompt_len, line);
                session_generate(s, h_prompt, prompt_len, s.args.output_tok);
                free(h_prompt);
            }
        }

        printf("sdec> ");
        fflush(stdout);
    }
    printf("\nLeaving REPL.\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    Session s;
    parse_args(s.args, argc, argv);
    s.do_baseline = !s.args.spec_only;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    s.use_real = (s.args.target_path[0] != '\0' &&
                  (s.args.draft_path[0] != '\0' || s.args.self_spec));
    if (s.args.self_spec && s.args.draft_path[0] == '\0' &&
        s.args.target_path[0] != '\0') {
        strncpy(s.args.draft_path, s.args.target_path, 511);
    }

    if (s.args.repl && !s.use_real) {
        fprintf(stderr, "--repl requires real models (--target=...)\n");
        return 1;
    }

    printf("Mode: %s | kernel=%s | max_tokens=%d | k=%d | stochastic_spec=%d",
           s.use_real ? "real-model" : "dummy",
           s.args.use_mega ? "megakernel" : "multi-kernel",
           s.args.max_new, s.args.spec_k, s.args.stochastic_spec ? 1 : 0);
    if (s.args.self_spec) printf(" | self-spec");
    if (s.args.repl)      printf(" | repl");
    if (s.do_baseline)    printf(" | bench");
    else                  printf(" | spec-only");
    printf("\n\n");

    // ---- Load / allocate models (once) ----
    if (s.use_real) {
        if (s.args.self_spec) {
            printf("Loading target model (self-spec): %s\n", s.args.target_path);
            if (!model_load_weights(s.target_model, s.args.target_path, nullptr)) {
                fprintf(stderr, "Failed to load target model\n");
                return 1;
            }
            s.draft_model = s.target_model;
            s.draft_is_alias = true;
        } else {
            printf("Loading draft model: %s\n", s.args.draft_path);
            if (!model_load_weights(s.draft_model, s.args.draft_path, nullptr)) {
                fprintf(stderr, "Failed to load draft model\n");
                return 1;
            }
            printf("Loading target model: %s\n", s.args.target_path);
            if (!model_load_weights(s.target_model, s.args.target_path, nullptr)) {
                fprintf(stderr, "Failed to load target model\n");
                return 1;
            }
        }
        printf("Weights resident in GPU memory.\n\n");
    } else {
        ModelConfig draft_cfg  = make_draft_config();
        ModelConfig target_cfg = make_target_config();
        model_alloc(s.draft_model, draft_cfg);
        model_alloc(s.target_model, target_cfg);
        printf("Dummy models allocated (no weight file).\n\n");
    }

    if (s.args.self_spec) {
        int n_dl = s.args.draft_layers > 0
                 ? s.args.draft_layers
                 : s.target_model.cfg.n_layers / 2;
        if (n_dl < 1) n_dl = 1;
        if (n_dl >= s.target_model.cfg.n_layers)
            n_dl = s.target_model.cfg.n_layers - 1;
        s.draft_model.cfg.n_layers = n_dl;
        printf("Self-spec draft layers: 1..%d | target layers: 1..%d\n\n",
               n_dl, s.target_model.cfg.n_layers);
    }

    // ---- KV + engine (once) ----
    ModelConfig& dc = s.draft_model.cfg;
    ModelConfig& tc = s.target_model.cfg;
    s.kv_shared = s.args.self_spec;

    if (s.kv_shared) {
        kv_cache_alloc(s.target_kv, tc.n_layers, kv_dim(tc), MAX_KV_BLOCKS);
        s.draft_kv = s.target_kv;
        printf("KV cache: shared (target depth %d layers)\n", tc.n_layers);
    } else {
        kv_cache_alloc(s.draft_kv,  dc.n_layers, kv_dim(dc), MAX_KV_BLOCKS);
        kv_cache_alloc(s.target_kv, tc.n_layers, kv_dim(tc), MAX_KV_BLOCKS);
    }
    kv_cache_alloc(s.baseline_kv, tc.n_layers, kv_dim(tc), MAX_KV_BLOCKS);

    inference_engine_init(s.eng, s.target_model.cfg);
    s.eng.gemm_backend = s.args.gemm_backend;
    printf("InferenceEngine: SM%d.%d | coop_launch=%s | max_coop_blocks=%d | gemm=%s\n",
           s.eng.sm_major, s.eng.sm_minor,
           s.eng.coop_supported ? "yes" : "no",
           s.eng.max_coop_blocks,
           s.eng.gemm_backend == GEMM_BACKEND_CUBLAS ? "cublas" : "legacy");

    CUDA_CHECK(cudaMalloc(&s.d_baseline_result, sizeof(GenerationResult)));
    CUDA_CHECK(cudaMalloc(&s.d_spec_result,     sizeof(GenerationResult)));
    CUDA_CHECK(cudaMalloc(&s.d_prompt, (size_t)MAX_SEQ_LEN * sizeof(int)));
    CUDA_CHECK(cudaEventCreate(&s.ev_start));
    CUDA_CHECK(cudaEventCreate(&s.ev_stop));

    if (s.args.eos_token >= 0)
        printf("EOS token: %d\n", s.args.eos_token);

    printf("Warming up...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- One-shot prompt (optional) then REPL or exit ----
    if (s.args.prompt_tok[0] != '\0' || !s.use_real) {
        int* h_prompt = nullptr;
        int  prompt_len = 0;

        if (s.use_real && s.args.prompt_tok[0] != '\0') {
            h_prompt = tok_load_alloc(s.args.prompt_tok, &prompt_len);
            if (!h_prompt) {
                fprintf(stderr, "Failed to load prompt from %s\n",
                        s.args.prompt_tok);
                return 1;
            }
            printf("Prompt: %d tokens loaded from %s\n",
                   prompt_len, s.args.prompt_tok);
        } else if (!s.use_real) {
            prompt_len = s.args.prompt_len;
            h_prompt   = new int[prompt_len];
            for (int i = 0; i < prompt_len; i++)
                h_prompt[i] = (i + 1) % DEFAULT_VOCAB_SIZE;
            printf("Dummy prompt: %d tokens\n", prompt_len);
        }

        if (h_prompt) {
            session_generate(s, h_prompt, prompt_len, s.args.output_tok);
            if (s.use_real) free(h_prompt);
            else            delete[] h_prompt;
        }
    } else if (!s.args.repl) {
        fprintf(stderr,
                "No --prompt-tok= given. Use --prompt-tok=... or --repl.\n");
        return 1;
    }

    if (s.args.repl)
        run_repl(s);

    // ---- Cleanup (once) ----
    inference_engine_destroy(s.eng);
    CUDA_CHECK(cudaEventDestroy(s.ev_start));
    CUDA_CHECK(cudaEventDestroy(s.ev_stop));
    cudaFree(s.d_prompt);
    cudaFree(s.d_baseline_result);
    cudaFree(s.d_spec_result);
    kv_cache_free(s.baseline_kv);
    if (s.kv_shared) {
        kv_cache_free(s.target_kv);
    } else {
        kv_cache_free(s.draft_kv);
        kv_cache_free(s.target_kv);
    }
    if (!s.draft_is_alias)
        model_free(s.draft_model);
    model_free(s.target_model);

    printf("\nDone.\n");
    return 0;
}
