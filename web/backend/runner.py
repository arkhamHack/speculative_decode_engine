"""
runner.py -- Subprocess runner for spec_decode.exe + output parser.

Finds and executes the CUDA binary, parses its stdout into a structured dict.
Falls back to a mock mode if the binary is not found (for UI development).
"""

import subprocess
import re
import os
import time
import random
from pathlib import Path
from typing import Optional

# Locate the exe relative to this file
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXE_CANDIDATES = [
    _REPO_ROOT / "cuda" / "spec_decode.exe",
    _REPO_ROOT / "cuda" / "spec_decode",          # Linux/WSL
    _REPO_ROOT / "cuda" / "build" / "spec_decode.exe",
]


def find_exe() -> Optional[Path]:
    for p in _EXE_CANDIDATES:
        if p.exists():
            return p
    return None


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_token_list(line: str) -> list[int]:
    """Extract bracketed integer list from a 'Label [N tokens]: ...' line."""
    m = re.search(r":\s*([\d\s]+)$", line)
    if not m:
        return []
    return [int(x) for x in m.group(1).split() if x.strip()]


def parse_output(stdout: str) -> dict:
    lines = stdout.splitlines()
    result = {
        "device": "",
        "mode": "",
        "baseline_tokens": [],
        "baseline_n": 0,
        "baseline_ms": 0.0,
        "baseline_tok_per_s": 0.0,
        "spec_tokens": [],
        "spec_n": 0,
        "spec_ms": 0.0,
        "spec_tok_per_s": 0.0,
        "draft_proposed": 0,
        "draft_accepted": 0,
        "acceptance_rate": 0.0,
        "spec_iterations": 0,
        "speedup": 1.0,
        "match": False,
        "raw": stdout,
        "iterations_detail": [],   # per-iteration acceptance counts (estimated)
    }

    for line in lines:
        # Device
        if line.startswith("Device:"):
            result["device"] = line.split("Device:")[1].strip()

        # Mode line: "Mode: multi-kernel | max_tokens=32 | k=4 | ..."
        elif line.startswith("Mode:"):
            result["mode"] = line.split("|")[0].replace("Mode:", "").strip()

        # Baseline token list
        elif line.startswith("Baseline ["):
            result["baseline_tokens"] = _parse_token_list(line)
            m = re.search(r"\[(\d+) tokens\]", line)
            if m:
                result["baseline_n"] = int(m.group(1))

        # Baseline timing
        elif "Time:" in line and "Tok/s:" in line and result["baseline_ms"] == 0.0:
            m_time = re.search(r"Time:\s*([\d.]+)\s*ms", line)
            m_toks = re.search(r"Tok/s:\s*([\d.]+)", line)
            if m_time:
                result["baseline_ms"] = float(m_time.group(1))
            if m_toks:
                result["baseline_tok_per_s"] = float(m_toks.group(1))

        # Speculative token list
        elif line.startswith("Speculative ["):
            result["spec_tokens"] = _parse_token_list(line)
            m = re.search(r"\[(\d+) tokens\]", line)
            if m:
                result["spec_n"] = int(m.group(1))

        # Speculative timing (second occurrence of Time/Tok/s)
        elif "Time:" in line and "Tok/s:" in line and result["spec_ms"] == 0.0 and result["baseline_ms"] > 0:
            m_time = re.search(r"Time:\s*([\d.]+)\s*ms", line)
            m_toks = re.search(r"Tok/s:\s*([\d.]+)", line)
            if m_time:
                result["spec_ms"] = float(m_time.group(1))
            if m_toks:
                result["spec_tok_per_s"] = float(m_toks.group(1))

        # Draft stats
        elif "Draft proposed:" in line:
            m_prop = re.search(r"Draft proposed:\s*(\d+)", line)
            m_acc  = re.search(r"Accepted:\s*(\d+)", line)
            m_rate = re.search(r"Acceptance rate:\s*([\d.]+)%", line)
            if m_prop:
                result["draft_proposed"] = int(m_prop.group(1))
            if m_acc:
                result["draft_accepted"] = int(m_acc.group(1))
            if m_rate:
                result["acceptance_rate"] = float(m_rate.group(1)) / 100.0

        # Iterations
        elif "Speculation iterations:" in line:
            m = re.search(r"(\d+)", line)
            if m:
                result["spec_iterations"] = int(m.group(1))

        # Speedup
        elif "Speedup:" in line:
            m = re.search(r"([\d.]+)x", line)
            if m:
                result["speedup"] = float(m.group(1))

        # Match
        elif "Output match:" in line:
            result["match"] = "PASS" in line

    # Synthesize per-iteration detail from aggregate stats
    # (exe doesn't print per-iteration breakdown; we estimate)
    if result["spec_iterations"] > 0:
        proposed = result["draft_proposed"]
        accepted = result["draft_accepted"]
        iters    = result["spec_iterations"]
        per_iter_proposed = proposed / iters
        per_iter_accepted = accepted / iters
        rng = random.Random(42)
        for i in range(iters):
            k = round(per_iter_proposed)
            acc = min(k, max(0, round(per_iter_accepted + rng.uniform(-1, 1))))
            result["iterations_detail"].append({"iter": i + 1, "proposed": k, "accepted": acc})

    return result


# ── Mock data (when exe not found) ────────────────────────────────────────────

def _mock_result(mode: str, spec: bool, max_tokens: int, k: int, seed: int) -> dict:
    """Return plausible fake data so the UI works without a compiled binary."""
    rng = random.Random(seed)

    base_tps  = 820 + rng.uniform(-50, 50) + (180 if mode == "mega" else 0)
    base_ms   = (max_tokens / base_tps) * 1000

    if spec:
        accept_rate = rng.uniform(0.55, 0.85)
        speedup     = (1 - accept_rate ** (k + 1)) / ((1 - accept_rate) * (k * 0.05 + 1))
        speedup     = max(1.0, min(speedup, 2.5))
        spec_ms     = base_ms / speedup
        spec_tps    = max_tokens / (spec_ms / 1000)
        iters       = max(1, int(max_tokens / (1 + k * accept_rate)))
        proposed    = iters * k
        accepted    = int(proposed * accept_rate)
    else:
        speedup  = 1.0
        spec_ms  = base_ms
        spec_tps = base_tps
        iters    = max_tokens
        proposed = 0
        accepted = 0

    tokens = [rng.randint(1, 255) for _ in range(max_tokens)]

    iter_detail = []
    for i in range(iters):
        acc = min(k, max(0, int(k * accept_rate + rng.uniform(-0.5, 0.5))))
        iter_detail.append({"iter": i + 1, "proposed": k, "accepted": acc})

    return {
        "device": "NVIDIA GeForce RTX 3050 (mock)",
        "mode": mode,
        "baseline_tokens": tokens,
        "baseline_n": max_tokens,
        "baseline_ms": round(base_ms, 2),
        "baseline_tok_per_s": round(base_tps, 1),
        "spec_tokens": tokens,
        "spec_n": max_tokens,
        "spec_ms": round(spec_ms, 2),
        "spec_tok_per_s": round(spec_tps, 1),
        "draft_proposed": proposed,
        "draft_accepted": accepted,
        "acceptance_rate": round(accepted / proposed, 3) if proposed else 0.0,
        "spec_iterations": iters,
        "speedup": round(speedup, 2),
        "match": True,
        "raw": "(mock data -- binary not found)",
        "iterations_detail": iter_detail,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_benchmark(mode: str, spec: bool, max_tokens: int,
                  k: int, seed: int, prompt_len: int = 4) -> dict:
    exe = find_exe()
    if exe is None:
        time.sleep(0.4)   # simulate latency
        return _mock_result(mode, spec, max_tokens, k, seed)

    cmd = [
        str(exe),
        f"--mode={'mega' if mode == 'mega' else 'multi'}",
        f"--max-tokens={max_tokens}",
        f"--k={k}",
        f"--seed={seed}",
        f"--prompt-len={prompt_len}",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(exe.parent),
        )
        stdout = proc.stdout
        if proc.returncode != 0:
            return {"error": proc.stderr or "Non-zero exit code", "raw": stdout}
        result = parse_output(stdout)
        # If spec=False, override so baseline == spec (no speculation)
        if not spec:
            result["spec_tokens"]     = result["baseline_tokens"]
            result["spec_n"]          = result["baseline_n"]
            result["spec_ms"]         = result["baseline_ms"]
            result["spec_tok_per_s"]  = result["baseline_tok_per_s"]
            result["speedup"]         = 1.0
            result["draft_proposed"]  = 0
            result["draft_accepted"]  = 0
            result["acceptance_rate"] = 0.0
            result["spec_iterations"] = 0
            result["match"]           = True
        return result
    except subprocess.TimeoutExpired:
        return {"error": "Benchmark timed out after 120s"}
    except Exception as e:
        return {"error": str(e)}
