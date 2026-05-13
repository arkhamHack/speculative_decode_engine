# Speculative Decode Engine

Minimal, high-performance LLM inference engine implementing the speculative
sampling algorithm from Leviathan et al.,
*"Fast Inference from Transformers via Speculative Decoding"*.

## Quick start

```bash
pip install -r requirements.txt
python -m benchmarks.benchmark
```

## Options

```
--draft-model NAME        HuggingFace model id for the draft model (default: gpt2)
--target-model NAME       HuggingFace model id for the target model (default: gpt2-medium)
--prompt TEXT             Input prompt
--max-new-tokens N        Number of tokens to generate (default: 64)
--k N                    Draft length γ per speculation round (default: 4)
--device DEVICE           cuda / cpu / auto
--dtype TYPE              float16 / float32 / bfloat16

Sampling:
--temperature T           0 = greedy (default), > 0 = stochastic
--top-k N                 Top-k filtering (0 = disabled)
--top-p P                 Nucleus filtering (1.0 = disabled)
--lenience L              Section 2.4: < 1.0 trades accuracy for speed (1.0 = exact)
--seed N                  Random seed for reproducibility

Output control:
--skip-correctness        Skip output-match assertion
--skip-analysis           Skip Section 3 analytical comparison
```

## Examples

```bash
# Greedy decoding -- correctness check verifies identical outputs
python -m benchmarks.benchmark

# Stochastic sampling with speculative sampling algorithm
python -m benchmarks.benchmark --temperature 0.8

# Nucleus sampling, reproducible
python -m benchmarks.benchmark --temperature 0.7 --top-p 0.9 --seed 42

# Lenience mode: higher acceptance rate, slight distributional shift
python -m benchmarks.benchmark --temperature 0.8 --lenience 0.8

# Larger models, longer generation
python -m benchmarks.benchmark --draft-model gpt2 --target-model gpt2-large --max-new-tokens 128 --k 6
```

## Project structure

```
engine/
  baseline.py       Autoregressive decoding with KV cache (greedy + sampling)
  speculative.py    Speculative decoding with Algorithm 1 from the paper
  kv_cache.py       KV cache manager with append and rollback
  scheduler.py      Dispatch layer for generation modes
models/
  loader.py         Load draft and target HuggingFace models
benchmarks/
  benchmark.py      Compare baseline vs speculative with full analytics
utils/
  tokenizer.py      Tokenizer loading and prompt encoding
  metrics.py        Generation metrics (TTFT, ITL, tokens/sec, acceptance rate)
  sampling.py       SamplingParams, temperature/top-k/top-p/lenience, get_probs
  analysis.py       Section 3 analytical formulas (Eq 1, Thm 3.8, Cor 3.9, etc.)
  cuda_utils.py     CUDA event timers, stream helpers, per-phase profiling
```

## What's implemented from the paper

### Section 2.3 -- Speculative Sampling (Algorithm 1)
- Draft model samples k tokens from q(x)
- Target model verifies in one forward pass
- Accept with probability min(1, p(x)/q(x))
- Rejection correction from p'(x) = norm(max(0, p(x) - q(x)))
- Provably samples from the target distribution p(x)

### Section 2.4 -- Lenience
- Optional parameter l that scales q(x) before comparison
- l < 1.0 increases acceptance rate α at the cost of distributional accuracy
- No token sampled with probability greater than p(x)/l

### Section 3 -- Analysis
- **Equation 1**: E(tokens per iteration) = (1 - α^(γ+1)) / (1 - α)
- **Theorem 3.8**: Walltime improvement = (1 - α^(γ+1)) / ((1 - α)(γc + 1))
- **Corollary 3.9**: Lower bound (1 + α) / (1 + c) when α > c
- **Theorem 3.11**: Arithmetic operations factor
- **Optimal γ**: Numerically found for given α, c
- **D_LK divergence** and **β = Σ min(p, q)** computed per-position

### Metrics reported

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token (wall-clock and GPU-accurate) |
| Mean ITL | Mean inter-token latency |
| Tokens/sec | Throughput |
| α (acceptance rate) | Fraction of draft tokens accepted |
| Mean β | Distributional acceptance rate from Σmin(p,q) |
| Tokens/iteration | Empirical vs predicted (Eq 1) |
| Speedup | Observed vs predicted (Thm 3.8) |
| c | Cost coefficient time(Mq)/time(Mp) measured empirically |
| ĉ | Parameter ratio params(Mq)/params(Mp) |
| Optimal γ | Best draft length for measured α, c |
