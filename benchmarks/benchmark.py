"""Benchmark: compare baseline vs speculative decoding.

Usage:
    python -m benchmarks.benchmark [OPTIONS]

Runs both baseline (target-only) and speculative (draft+target) generation
on the same prompt, verifies output correctness (greedy mode only), and prints:
    - Generation metrics (TTFT, ITL, tokens/sec, acceptance rate)
    - Per-phase CUDA profiling (when on GPU)
    - Analytical predictions from Section 3 of Leviathan et al. vs observed values
      (expected tokens/iteration, walltime improvement, optimal γ, ops factor)

Supports greedy (temperature=0) and stochastic sampling with the speculative
sampling algorithm.  The lenience parameter (Section 2.4) can trade slight
distributional accuracy for higher acceptance rate.
"""

import argparse
import time

import torch

from models.loader import load_draft_and_target
from utils.tokenizer import load_tokenizer, encode_prompt
from utils.sampling import SamplingParams
from utils.analysis import format_analysis
from engine.scheduler import run_generation


DEFAULT_DRAFT = "gpt2"
DEFAULT_TARGET = "gpt2-medium"

DEFAULT_PROMPT = (
    "The future of artificial intelligence lies in"
)
DEFAULT_MAX_TOKENS = 64
DEFAULT_K = 4


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark baseline vs speculative decoding")
    p.add_argument("--draft-model", type=str, default=DEFAULT_DRAFT)
    p.add_argument("--target-model", type=str, default=DEFAULT_TARGET)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--k", type=int, default=DEFAULT_K,
                    help="Draft length γ per speculation round")
    p.add_argument("--device", type=str, default=None,
                    help="Force device (cuda/cpu/auto)")
    p.add_argument("--dtype", type=str, default="float32",
                    choices=["float16", "float32", "bfloat16"])

    # Sampling parameters
    p.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0 = greedy, default)")
    p.add_argument("--top-k", type=int, default=0,
                    help="Top-k filtering (0 = disabled)")
    p.add_argument("--top-p", type=float, default=1.0,
                    help="Nucleus (top-p) filtering (1.0 = disabled)")
    p.add_argument("--lenience", type=float, default=1.0,
                    help="Section 2.4: < 1.0 increases acceptance rate at the cost "
                         "of slight distributional shift (1.0 = exact)")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible stochastic sampling")

    p.add_argument("--skip-correctness", action="store_true",
                    help="Skip output-match assertion")
    p.add_argument("--skip-analysis", action="store_true",
                    help="Skip Section 3 analytical comparison")
    return p.parse_args()


def select_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_dtype(name: str):
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _estimate_cost_coefficient(
    draft_model, target_model, input_ids, device, n_warmup=3, n_measure=10,
) -> tuple[float, float]:
    """Estimate c = time(Mq) / time(Mp) and ĉ = params(Mq) / params(Mp).

    Runs short single-token forward passes to measure wall-clock ratio.
    """
    token = input_ids[:, :1]

    # c_hat from parameter counts
    draft_params = sum(p.numel() for p in draft_model.parameters())
    target_params = sum(p.numel() for p in target_model.parameters())
    c_hat = draft_params / target_params if target_params > 0 else 0.0

    # c from wall-clock timing
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = draft_model(input_ids=token, use_cache=False)
            _ = target_model(input_ids=token, use_cache=False)
    if device == "cuda":
        torch.cuda.synchronize()

    if device == "cuda":
        start_d = torch.cuda.Event(enable_timing=True)
        end_d = torch.cuda.Event(enable_timing=True)
        start_t = torch.cuda.Event(enable_timing=True)
        end_t = torch.cuda.Event(enable_timing=True)

        start_d.record()
        with torch.no_grad():
            for _ in range(n_measure):
                _ = draft_model(input_ids=token, use_cache=False)
        end_d.record()

        start_t.record()
        with torch.no_grad():
            for _ in range(n_measure):
                _ = target_model(input_ids=token, use_cache=False)
        end_t.record()

        torch.cuda.synchronize()
        draft_ms = start_d.elapsed_time(end_d)
        target_ms = start_t.elapsed_time(end_t)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_measure):
                _ = draft_model(input_ids=token, use_cache=False)
        draft_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_measure):
                _ = target_model(input_ids=token, use_cache=False)
        target_ms = (time.perf_counter() - t0) * 1000

    c = draft_ms / target_ms if target_ms > 0 else 0.0
    return c, c_hat


def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = select_dtype(args.dtype)
    use_cuda = device == "cuda"

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        lenience=args.lenience,
    )

    print(f"Device: {device} | Dtype: {dtype}")
    print(f"Draft model:  {args.draft_model}")
    print(f"Target model: {args.target_model}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Max new tokens: {args.max_new_tokens} | γ (k): {args.k}")
    if sampling_params.is_greedy:
        print("Sampling: greedy (temperature=0)")
    else:
        parts = [f"temperature={sampling_params.temperature}"]
        if sampling_params.top_k > 0:
            parts.append(f"top_k={sampling_params.top_k}")
        if sampling_params.top_p < 1.0:
            parts.append(f"top_p={sampling_params.top_p}")
        if sampling_params.lenience != 1.0:
            parts.append(f"lenience={sampling_params.lenience}")
        print(f"Sampling: {' '.join(parts)}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    if use_cuda:
        print(f"CUDA: {torch.cuda.get_device_name()} | "
              f"CUDA events enabled for GPU-accurate timing")
    print()

    tokenizer = load_tokenizer(args.target_model)
    input_ids = encode_prompt(tokenizer, args.prompt, device=device)
    eos_id = tokenizer.eos_token_id

    draft_model, target_model = load_draft_and_target(
        args.draft_model, args.target_model, device=device, dtype=dtype,
    )

    # ── Estimate cost coefficient c and ĉ ──
    if not args.skip_analysis:
        print("Estimating cost coefficient c = time(Mq)/time(Mp)...")
        c, c_hat = _estimate_cost_coefficient(
            draft_model, target_model, input_ids, device,
        )
        print(f"  c  = {c:.4f}  (wall-clock ratio)")
        print(f"  ĉ  = {c_hat:.4f}  (parameter-count ratio)")
        print()

    # ── Warmup ──
    if use_cuda:
        print("Warming up (CUDA kernel compilation)...")
        with torch.no_grad():
            _ = target_model(input_ids=input_ids[:, :1], use_cache=False)
            _ = draft_model(input_ids=input_ids[:, :1], use_cache=False)
        torch.cuda.synchronize()

    # ── Baseline generation ──
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    print("\n>>> Running baseline (target-only) generation...")
    baseline_ids, baseline_metrics, _ = run_generation(
        mode="baseline",
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        sampling_params=sampling_params,
    )
    if use_cuda:
        torch.cuda.synchronize()

    baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
    print(f"Baseline output:\n{baseline_text}")
    print(baseline_metrics.summary("Baseline"))

    # ── Speculative generation ──
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    print(">>> Running speculative generation...")
    spec_ids, spec_metrics, phase_profile = run_generation(
        mode="speculative",
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        draft_model=draft_model,
        k=args.k,
        sampling_params=sampling_params,
    )
    if use_cuda:
        torch.cuda.synchronize()

    spec_text = tokenizer.decode(spec_ids[0], skip_special_tokens=True)
    print(f"Speculative output:\n{spec_text}")
    print(spec_metrics.summary("Speculative"))

    if phase_profile is not None:
        print(phase_profile.summary())

    # ── Correctness check ──
    if not args.skip_correctness:
        baseline_gen = baseline_ids[0, input_ids.shape[1]:].tolist()
        spec_gen = spec_ids[0, input_ids.shape[1]:].tolist()

        if sampling_params.is_greedy:
            min_len = min(len(baseline_gen), len(spec_gen))
            match = baseline_gen[:min_len] == spec_gen[:min_len]

            if match:
                print("CORRECTNESS CHECK: PASSED  (greedy outputs match)")
            else:
                print("CORRECTNESS CHECK: FAILED")
                for i, (b, s) in enumerate(zip(baseline_gen, spec_gen)):
                    if b != s:
                        print(f"  First mismatch at generated token {i}: "
                              f"baseline={b} ({tokenizer.decode([b])!r}) vs "
                              f"spec={s} ({tokenizer.decode([s])!r})")
                        break
                print("  (Use --skip-correctness to suppress)")
        else:
            print("CORRECTNESS NOTE: Stochastic mode -- outputs differ by design.")
            print("  Speculative sampling provably samples from p(x) (the target")
            print("  distribution), but individual sequences will differ from baseline.")

    # ── Comparison summary ──
    print(f"\n{'='*60}")
    print("  Comparison Summary")
    print(f"{'='*60}")

    if use_cuda:
        b_tps = baseline_metrics.gpu_tokens_per_second
        s_tps = spec_metrics.gpu_tokens_per_second
        b_ttft = baseline_metrics.gpu_ttft
        s_ttft = spec_metrics.gpu_ttft
        timing_label = "(GPU)"
    else:
        b_tps = baseline_metrics.tokens_per_second
        s_tps = spec_metrics.tokens_per_second
        b_ttft = baseline_metrics.ttft
        s_ttft = spec_metrics.ttft
        timing_label = "(wall)"

    speedup = s_tps / b_tps if b_tps > 0 else float("nan")

    print(f"  Baseline tokens/sec {timing_label}:     {b_tps:.2f}")
    print(f"  Speculative tokens/sec {timing_label}:  {s_tps:.2f}")
    print(f"  Speedup:                         {speedup:.2f}x")
    print(f"  Baseline TTFT {timing_label}:           {b_ttft * 1000:.2f} ms")
    print(f"  Speculative TTFT {timing_label}:        {s_ttft * 1000:.2f} ms")
    print(f"  Baseline mean ITL {timing_label}:       {baseline_metrics.mean_itl * 1000:.2f} ms")
    print(f"  Speculative mean ITL {timing_label}:    {spec_metrics.mean_itl * 1000:.2f} ms")
    if spec_metrics.draft_tokens_proposed > 0:
        print(f"  Draft acceptance rate (α):        {spec_metrics.acceptance_rate:.4f}")
    print(f"{'='*60}")

    # ── Section 3: Analytical predictions vs empirical ──
    if not args.skip_analysis and spec_metrics.draft_tokens_proposed > 0:
        alpha = spec_metrics.acceptance_rate
        empirical_tok_iter = spec_metrics.empirical_tokens_per_iteration

        print(format_analysis(
            alpha=alpha,
            gamma=args.k,
            c=c,
            c_hat=c_hat,
            empirical_tokens_per_iter=empirical_tok_iter,
            empirical_speedup=speedup,
        ))

        # If distributional α (mean_beta) is available, show comparison
        if spec_metrics.mean_beta > 0:
            print(f"  α from token counts:       {alpha:.4f}")
            print(f"  α from Σmin(p,q) (Cor 3.6):{spec_metrics.mean_beta:.4f}")
            print()


if __name__ == "__main__":
    main()
