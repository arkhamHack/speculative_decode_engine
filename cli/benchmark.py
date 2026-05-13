#!/usr/bin/env python3
"""
Python CLI for the CUDA speculative decoding engine.

Calls the pybind11-compiled `spec_decode_cuda` module, runs baseline vs
speculative decoding, and prints a comparison report.

Usage:
    python cli/benchmark.py [--mode multi|mega] [--max-tokens N] [--k N]
                            [--seed N] [--prompt-len N]
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="CUDA speculative decoding benchmark")
    parser.add_argument("--mode", choices=["multi", "mega"], default="multi",
                        help="Kernel mode: multi-kernel or persistent megakernel")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Maximum new tokens to generate")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of draft tokens per speculation round")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weight initialization")
    parser.add_argument("--prompt-len", type=int, default=4,
                        help="Length of the dummy prompt")
    args = parser.parse_args()

    try:
        import spec_decode_cuda
    except ImportError:
        print("ERROR: spec_decode_cuda module not found.")
        print("Build it first:")
        print("  cmake -B build cuda/ && cmake --build build")
        print("  # or: cd cuda && make module")
        sys.exit(1)

    print(f"Running benchmark: mode={args.mode}, max_tokens={args.max_tokens}, "
          f"k={args.k}, seed={args.seed}, prompt_len={args.prompt_len}")
    print("=" * 60)

    result = spec_decode_cuda.run_benchmark(
        mode=args.mode,
        max_tokens=args.max_tokens,
        spec_k=args.k,
        seed=args.seed,
        prompt_len=args.prompt_len,
    )

    # Baseline results
    print(f"\n--- Baseline (target-only) ---")
    print(f"  Tokens generated: {result['baseline_n']}")
    print(f"  Time:             {result['baseline_ms']:.2f} ms")
    print(f"  Throughput:       {result['baseline_tok_per_s']:.1f} tok/s")
    bl_tokens = result["baseline_tokens"]
    print(f"  Output:           {bl_tokens[:20]}{'...' if len(bl_tokens) > 20 else ''}")

    # Speculative results
    print(f"\n--- Speculative (draft + target, k={args.k}) ---")
    print(f"  Tokens generated: {result['spec_n']}")
    print(f"  Time:             {result['spec_ms']:.2f} ms")
    print(f"  Throughput:       {result['spec_tok_per_s']:.1f} tok/s")
    print(f"  Draft proposed:   {result['draft_proposed']}")
    print(f"  Draft accepted:   {result['draft_accepted']}")
    print(f"  Acceptance rate:  {result['acceptance_rate']:.1%}")
    print(f"  Iterations:       {result['spec_iterations']}")
    sp_tokens = result["spec_tokens"]
    print(f"  Output:           {sp_tokens[:20]}{'...' if len(sp_tokens) > 20 else ''}")

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"  Speedup:          {result['speedup']:.2f}x")
    print(f"  Output match:     {'PASS' if result['match'] else 'FAIL'}")

    if not result["match"]:
        print("\n  WARNING: Outputs do not match! Greedy decoding should produce")
        print("  identical results. This indicates a bug.")
        for i, (b, s) in enumerate(zip(bl_tokens, sp_tokens)):
            if b != s:
                print(f"  First mismatch at position {i}: baseline={b}, spec={s}")
                break
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
