# CUDA LLM Inference Engine

*A batch-1, low-latency LLM inference runtime built around two bets: **megakernels** and **speculative decoding**.*

This is a hand-written CUDA runtime for LLaMA-style transformers, built to make **one sequence's next token arrive sooner**. It is a latency engine, not a throughput engine — everything is measured at batch 1, one sequence, first-token-to-next-token.

The project deliberately builds *both* of everything — a classic multi-kernel loop **and** a persistent megakernel; a paged KV cache for cheap rollback; **two-model** and **self-speculative** modes; **greedy** and **stochastic** verification — so the wins can be *measured* rather than guessed. A companion writeup, [ARTICLE.md](ARTICLE.md), walks through the mechanisms and the findings in depth.

---

## Why these two bets

A single decode step multiplies each weight matrix by a *single* activation vector — a matrix-vector product (GEMV) with roughly **one FLOP per byte of weights**. Modern GPUs sit hundreds of FLOP/byte into the compute-bound regime, so decode is deep into the **memory-bound** regime. Tensor cores, faster math, CUTLASS — none of it helps. The floor is:

```
decode_step_latency ≈ weight_bytes / achievable_HBM_bandwidth
```

On top of that floor, a naive engine pays two avoidable taxes:

- **Launch tax** — every operator is a separate kernel launch, repeated per op, per layer, per token.
- **Residency tax** — between kernels, every intermediate tensor round-trips to HBM and back.

**Megakernels** attack both: one launch, and the hidden state stays resident in on-chip shared memory across the whole block. **Speculative decoding** attacks the token count: a cheap draft proposes several tokens, the target verifies them all in one batched forward, and each expensive target pass can yield multiple accepted tokens.

There is also a subtler bandwidth trap (Little's Law): a single SM can only keep the HBM pipe *partially* full, so a megakernel launched as `<<<1, …>>>` is bandwidth-starved no matter how clean it is. The fix is to spread weight streaming across **every SM** inside one persistent cooperative kernel.

---

## Features

| Area | Details |
|------|---------|
| **Architecture** | RMSNorm, SwiGLU MLP, multi-head / grouped-query attention (GQA) with RoPE, paged KV cache, streaming (online) softmax attention |
| **Execution paths** | **Multi-kernel** (one launch per op, can call cuBLAS tensor ops) or **megakernel** (single persistent GPU-resident loop, hidden state kept in shared memory) |
| **Multi-SM residency** | Cooperative megakernel splits every projection (Q/K/V, output, gate/up, down, vocab) across all SMs via `grid.sync()`, lifting the single-SM bandwidth ceiling |
| **Speculative decoding** | Draft proposes *k* tokens; target verifies all *k* in a single batched forward pass; longest correct prefix + one bonus token is kept |
| **Speculation strategy** | **Two-model** (separate draft + target checkpoints, separate caches) or **self-speculative** (first *N* layers of the target draft, sharing one KV cache) |
| **Verification mode** | **Greedy** (token-identical to the non-speculative baseline) or **stochastic** (exact Leviathan/Chen algorithm — accept with `min(1, p/q)`, resample the correction, provably preserving the target distribution) |
| **Lenience** | Optional knob that scales `q` before comparison, trading a controlled amount of exactness for higher acceptance α |
| **Adaptive draft temp** | EWMA control loop nudges draft temperature toward a target acceptance rate on the fly |
| **Paged KV cache** | Fixed-size blocks with a per-layer block table; rollback of rejected tokens is an O(1) `seq_len` rewind, not data movement |
| **GEMM backends** | Hand-written `half2` GEMV (`__device__`, inlinable into the megakernel) or host-launched **cuBLAS** tensor ops (multi-kernel path) |
| **Weight modes** | **Dummy bench** — random weights + synthetic tokens, no model files. **Production** — SDEC `.bin` weights exported from HuggingFace |

Supported real models (via `tools/export_model.py`): LLaMA, Llama 2/3, Mistral, TinyLlama, SmolLM/SmolLM2, Qwen2.5, and similar LLaMA-family checkpoints. GPT-2 and other non-LLaMA architectures are not supported by the CUDA runtime.

---

## Repository structure

```
.
├── cuda/                 Native CUDA inference engine
│   ├── include/          Headers: attention, kv_cache, model, gemm_backend, utils, …
│   ├── src/              Kernels, model, KV cache, tokenizer, pybind binding, main
│   ├── build.bat         Windows build → spec_decode.exe
│   ├── Makefile          Make targets (binary + pybind module)
│   └── CMakeLists.txt    CMake build
├── cli/
│   └── benchmark.py      Python CLI over the pybind11 module (baseline vs speculative)
├── tools/
│   ├── export_model.py   HuggingFace checkpoint → SDEC binary
│   ├── hf_tok.py         Encode text → .tok, decode output .tok → text
│   ├── sweep_spec_k.py   Sweep draft length k to find the empirical optimum
│   └── compare_cuda_hf.py  Cross-check engine output against HuggingFace
├── web/                  Interactive benchmark UI (React + Vite frontend, FastAPI backend)
├── benchmarks/           PyTorch/HuggingFace reference benchmark (analytical comparison)
└── ARTICLE.md            In-depth writeup of mechanisms and findings
```

---

## Build

```powershell
cd cuda
.\build.bat          # → cuda\spec_decode.exe
```

Or with Make / CMake:

```bash
cd cuda && make            # native binary
cd cuda && make module     # pybind11 module for the Python CLI
# or:
cmake -B build cuda/ && cmake --build build
```

---

## Quick start

### Native binary (dummy bench — no model files needed)

```powershell
cuda\spec_decode.exe --bench --k=2 --max-tokens=64
cuda\spec_decode.exe --bench --k=2 --max-tokens=64 --mode=mega        # persistent megakernel
cuda\spec_decode.exe --bench --k=2 --max-tokens=64 --mode=mega_coop   # multi-SM cooperative
```

### Python CLI (via the pybind11 module)

```bash
python cli/benchmark.py --mode multi      --max-tokens 128 --k 2
python cli/benchmark.py --mode mega       --max-tokens 128 --k 2
python cli/benchmark.py --mode mega_coop  --max-tokens 128 --k 2   # multi-SM baseline
```
# Python/pybind benchmark (baseline vs speculative, with correctness check)
python cli/benchmark.py --mode multi     --max-tokens 128 --k 2
python cli/benchmark.py --mode mega       --max-tokens 128 --k 2
python cli/benchmark.py --mode mega_coop  --max-tokens 128 --k 2   # multi-SM baseline

# Sweep k to find the empirical optimum
python tools/sweep_spec_k.py ‹args›

# Real models via the native binary
python tools/hf_tok.py encode ‹model› "‹prompt›" prompt.tok
cuda/spec_decode.exe --draft=weights/draft.bin --target=weights/target.bin \
                     --prompt-tok=prompt.tok --output-tok=output.tok --k=2

# Stochastic self-spec — reproduces the 1.38× winner on the RTX 3050
cuda/spec_decode.exe --target=weights/smol2_1p7b.sdec.bin --self-spec \
                     --stochastic-spec --mode=mega --bench --k=4 \
                     --max-tokens=64 --prompt-tok=prompt_smol.tok

### Web UI

An interactive benchmark UI (single run, compare modes, sweep k, autotune) lives in [web/](web/README.md) — a React + Vite frontend over a FastAPI backend that drives the CUDA binary. See [web/README.md](web/README.md) for setup.

---

## Key command-line flags

```
Modes:
  --bench                 compare baseline vs speculative (default without --repl)
  --spec-only             generate with speculative decode only (no baseline)
  --repl                  load models once, then accept many prompts
  --mode=multi|mega|mega_coop   kernel path (default: multi)

Generation:
  --max-tokens=N          tokens to generate (default: 32)
  --k=N                   speculative draft depth
  --eos-token=N           stop generation on this token id
  --self-spec             early layers draft, full model target
  --draft-layers=N        layers 1..N as draft proposer (default: target/2)
  --gemm-backend=cublas|legacy   GEMM backend (default: legacy hand-written GEMV)

Speculation / sampling:
  --stochastic-spec       distribution-matching (exact) speculative verify
  --draft-temp=T          draft softmax temperature
  --adaptive-draft-temp   EWMA nudge of draft temp toward a target acceptance
  --adapt-accept=R        EWMA acceptance target ∈ (0,1)
  --spec-seed=N           RNG seed for stochastic spec

Weights / IO:
  --draft=path.bin        draft model SDEC binary
  --target=path.bin       target model SDEC binary
  --prompt-tok=path.tok   tokenised prompt (tools/hf_tok.py encode)
  --output-tok=path.tok   write generated token ids
  --prompt-len=N          dummy prompt length (dummy tokenizer only)
```

---

## The speculative-decoding math

With per-token acceptance probability **α** and draft length *k*, the expected tokens produced per iteration (Leviathan et al., Eq. 1) is:

```
E[tokens/iter] = (1 − α^(k+1)) / (1 − α)
```

Tokens-per-iteration is not speedup — iterations aren't free. With cost ratio **c = time(draft) / time(target)**, the wall-time improvement factor (Thm 3.8) is:

```
speedup = (1 − α^(k+1)) / ((1 − α)(k·c + 1))
```

The numerator rewards acceptance; the denominator punishes drafting cost. **You win only when the draft is both accurate (high α) and cheap (low c).** Differentiating gives an optimal *k* for any (α, c), which the analysis path computes numerically and compares against measured speedup.

---

## Correctness

Every result rests on an engine verified to match HuggingFace **token-for-token** in greedy decoding (see `tools/compare_cuda_hf.py`). Greedy speculation is byte-identical to that baseline by construction; stochastic speculation provably preserves the target distribution. Output of the multi-SM cooperative path is bit-for-bit identical to the single-SM path — splitting work across SMs changes only *where* weights are streamed, not the math.

---

## Benchmarks

Results are documented in [ARTICLE.md](ARTICLE.md). *(Numbers are being finalized and will be added here.)*

Reproduce with:

```bash
python cli/benchmark.py --mode multi      --max-tokens 128 --k 2
python cli/benchmark.py --mode mega       --max-tokens 128 --k 2
python cli/benchmark.py --mode mega_coop  --max-tokens 128 --k 2
python tools/sweep_spec_k.py <args>
```

**Numerics.** Weights are fp16, accumulation is fp32. Greedy decoding is deterministic; stochastic runs are seeded.

---

## What's next

- **Multi-SM attention.** The full multi-SM forward ships (every projection split across SMs); attention is the remaining single-block step and the next bandwidth win for long sequences.
- **Tensor cores where they help.** A device-side (CUTLASS-style) GEMM inside the megakernel could accelerate compute-bound prefill and batched verify without leaving the single-kernel model.
- **Cheaper drafts.** Distilled or heavily-quantized drafts, early-exit layers, or a trained draft head (EAGLE / DeepSeek-DSpark direction).
- **Beyond batch 1.** Continuous batching and paged attention inside the kernel, to see how far the latency advantage survives as load rises.
- **A same-machine external baseline.** A HuggingFace `transformers` CUDA run on the same card as an honest reference point.

---

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding* (2023).
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling* (2023).
