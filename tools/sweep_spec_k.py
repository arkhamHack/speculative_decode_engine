"""
Sweep speculative draft depth k for a draft+target pair.

Usage:
  python tools/sweep_spec_k.py \\
      --draft weights/smol2_135m.sdec.bin \\
      --target weights/smol2_360m.sdec.bin \\
      --prompt-tok prompt_smol.tok \\
      --k 1,2,4,6,8 \\
      --max-tokens 64 \\
      --gemm-backend cublas
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE_CANDIDATES = [
    ROOT / "cuda" / "spec_decode.exe",
    ROOT / "cuda" / "spec_decode",
    ROOT / "cuda" / "build" / "spec_decode.exe",
]


def find_exe() -> Path:
    for p in EXE_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("spec_decode executable not found under cuda/")


def parse_summary(stdout: str) -> dict:
    def f(pat: str, cast=float):
        m = re.search(pat, stdout)
        if not m:
            raise ValueError(f"missing pattern: {pat}")
        return cast(m.group(1))

    return {
        "baseline_ms": f(r"Baseline:\s+([\d.]+)\s+ms"),
        "baseline_tps": f(r"Baseline:\s+[\d.]+\s+ms\s+\(([\d.]+)\s+tok/s\)"),
        "spec_ms": f(r"Speculative:\s+([\d.]+)\s+ms"),
        "spec_tps": f(r"Speculative:\s+[\d.]+\s+ms\s+\(([\d.]+)\s+tok/s\)"),
        "speedup": f(r"Speedup:\s+([\d.]+)x"),
        "accept": f(r"Accept rate:\s+([\d.]+)"),
        "match": "PASS" in stdout and "Output match" in stdout,
    }


def run_one(exe: Path, args: argparse.Namespace, k: int) -> dict:
    cmd = [
        str(exe),
        f"--draft={args.draft}",
        f"--target={args.target}",
        f"--prompt-tok={args.prompt_tok}",
        f"--mode={args.mode}",
        "--bench",
        f"--max-tokens={args.max_tokens}",
        f"--k={k}",
        f"--gemm-backend={args.gemm_backend}",
    ]
    if args.self_spec:
        cmd.append("--self-spec")
        if args.draft_layers:
            cmd.append(f"--draft-layers={args.draft_layers}")

    print(f"\n>>> k={k}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out[-2000:])
        raise RuntimeError(f"spec_decode failed (exit {proc.returncode}) for k={k}")
    row = parse_summary(out)
    row["k"] = k
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep speculative k for draft+target")
    p.add_argument("--draft", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--prompt-tok", required=True)
    p.add_argument("--k", default="1,2,4,6,8", help="comma-separated k values")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--mode", default="multi", choices=["multi", "mega"])
    p.add_argument("--gemm-backend", default="cublas", choices=["cublas", "legacy"])
    p.add_argument("--self-spec", action="store_true")
    p.add_argument("--draft-layers", type=int, default=0)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    exe = find_exe()
    print(f"exe={exe}")
    print(f"draft={args.draft}")
    print(f"target={args.target}")
    print(f"prompt={args.prompt_tok}")
    print(f"ks={ks} max_tokens={args.max_tokens} backend={args.gemm_backend}")

    rows = []
    for k in ks:
        row = run_one(exe, args, k)
        rows.append(row)
        print(
            f"k={k:2d}  speedup={row['speedup']:.3f}x  "
            f"accept={row['accept']:.3f}  "
            f"base={row['baseline_tps']:.1f} tok/s  "
            f"spec={row['spec_tps']:.1f} tok/s  "
            f"match={'PASS' if row['match'] else 'FAIL'}"
        )

    print("\n=== Sweep summary ===")
    print(f"{'k':>4} {'speedup':>8} {'accept':>8} {'base_tps':>10} {'spec_tps':>10} {'match':>6}")
    best = max(rows, key=lambda r: r["speedup"])
    for r in rows:
        mark = " *" if r is best else ""
        print(
            f"{r['k']:4d} {r['speedup']:8.3f} {r['accept']:8.3f} "
            f"{r['baseline_tps']:10.1f} {r['spec_tps']:10.1f} "
            f"{'PASS' if r['match'] else 'FAIL':>6}{mark}"
        )
    print(f"\nBest: k={best['k']} at {best['speedup']:.3f}x "
          f"(accept={best['accept']:.3f})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
