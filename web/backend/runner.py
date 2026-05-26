"""
runner.py -- Subprocess runner for spec_decode.exe + output parser.

Finds and executes the CUDA binary, parses its stdout into a structured dict.
Falls back to a mock mode if the binary is not found (for UI development).
"""

import subprocess
import re
import os
import struct
import tempfile
import time
import random
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer

# Locate the exe relative to this file
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CUDA_DIR = _REPO_ROOT / "cuda"
_BUILD_DIR = _CUDA_DIR / "build"
_EXE_CANDIDATES = [
    _CUDA_DIR / "spec_decode.exe",
    _CUDA_DIR / "spec_decode",  # Linux/WSL
    _BUILD_DIR / "spec_decode.exe",  # Ninja / single-config CMake
    _BUILD_DIR / "Release" / "spec_decode.exe",  # VS / multi-config
    _BUILD_DIR / "Debug" / "spec_decode.exe",
    _BUILD_DIR / "RelWithDebInfo" / "spec_decode.exe",
    _BUILD_DIR / "MinSizeRel" / "spec_decode.exe",
]


def find_exe() -> Optional[Path]:
    raw = os.environ.get("SPEC_DECODE_EXE", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return p.resolve()

    for p in _EXE_CANDIDATES:
        if p.exists():
            return p
    return None


def resolve_user_path(p: str) -> Path:
    """Resolve path; relative paths are treated as relative to repo root."""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _write_tok_file(path: Path, ids: list[int]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("I", len(ids)))
        for i in ids:
            f.write(struct.pack("I", int(i) & 0xFFFFFFFF))


def encode_prompt_tok(tokenizer_model: str, text: str, out_path: Path,
                      *, use_chat_template: bool = True) -> tuple[list[int], int]:
    """Encode prompt to token file.  Returns (token_ids, eos_token_id)."""

    tok = AutoTokenizer.from_pretrained(tokenizer_model)

    ids: list[int]
    has_role_tag = any(t in text for t in ("<|im_start|>", "[INST]", "<s>", "### Human"))
    use_template = (use_chat_template
                    and getattr(tok, "chat_template", None)
                    and not has_role_tag)

    if use_template:
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if not isinstance(ids, list):
                try:
                    ids = list(ids["input_ids"])
                except (TypeError, KeyError):
                    ids = list(ids)
        except Exception:
            ids = tok.encode(text, add_special_tokens=True)
    else:
        ids = tok.encode(text, add_special_tokens=True)

    _write_tok_file(out_path, ids)

    eos_id: int = getattr(tok, "eos_token_id", None) or -1
    # Some tokenizers expose a list; take the first entry
    if isinstance(eos_id, list):
        eos_id = eos_id[0] if eos_id else -1
    return ids, int(eos_id)


def decode_token_ids_to_text(tokenizer_model: str, ids: list[int]) -> str:
    if not ids:
        return ""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_model)
    return tok.decode(ids, skip_special_tokens=True)


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_token_list(line: str) -> list[int]:
    """Extract integers after ':' from 'Label [N tokens]: id id ...' output lines."""
    line = line.rstrip("\r\n")
    # Old spec_decode truncated with '...'; strip so end-of-line grammar still works.
    if "..." in line:
        line = line.split("...", 1)[0].rstrip()
    idx = line.find(":")
    if idx < 0:
        return []
    tail = line[idx + 1 :].strip()
    if not tail:
        return []
    ids: list[int] = []
    for tok in tail.split():
        try:
            ids.append(int(tok))
        except ValueError:
            break
    return ids


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
        "iterations_detail": [],  # per-iteration acceptance counts (estimated)
    }

    for line in lines:
        # Device
        if line.startswith("Device:"):
            result["device"] = line.split("Device:")[1].strip()

        # Mode line: "Mode: real-model | kernel=..." or dummy
        elif line.startswith("Mode:"):
            result["mode"] = line.split("|")[0].replace("Mode:", "").strip()

        # Baseline token list — main.cu uses label "Baseline out"
        elif re.search(r"Baseline out\s*\[\d+ tokens\]", line):
            result["baseline_tokens"] = _parse_token_list(line)
            m = re.search(r"\[(\d+) tokens\]", line)
            if m:
                result["baseline_n"] = int(m.group(1))

        # Baseline timing (first Time / Tok/s block after prefill section)
        elif "Time:" in line and "Tok/s:" in line and result["baseline_ms"] == 0.0:
            m_time = re.search(r"Time:\s*([\d.]+)\s*ms", line)
            m_toks = re.search(r"Tok/s:\s*([\d.]+)", line)
            if m_time:
                result["baseline_ms"] = float(m_time.group(1))
            if m_toks:
                result["baseline_tok_per_s"] = float(m_toks.group(1))

        # Speculative token list — main.cu uses label "Speculative  " (spacing varies)
        elif re.search(r"Speculative\s+\[\d+ tokens\]", line):
            result["spec_tokens"] = _parse_token_list(line)
            m = re.search(r"\[(\d+) tokens\]", line)
            if m:
                result["spec_n"] = int(m.group(1))

        # Speculative timing (second Time/Tok/s line)
        elif (
            "Time:" in line
            and "Tok/s:" in line
            and result["spec_ms"] == 0.0
            and result["baseline_ms"] > 0
        ):
            m_time = re.search(r"Time:\s*([\d.]+)\s*ms", line)
            m_toks = re.search(r"Tok/s:\s*([\d.]+)", line)
            if m_time:
                result["spec_ms"] = float(m_time.group(1))
            if m_toks:
                result["spec_tok_per_s"] = float(m_toks.group(1))

        # Draft stats — main.cu: "Draft proposed: N | Accepted: M | Rate: R%"
        elif "Draft proposed:" in line:
            m_prop = re.search(r"Draft proposed:\s*(\d+)", line)
            m_acc = re.search(r"Accepted:\s*(\d+)", line)
            m_rate = re.search(r"Rate:\s*([\d.]+)\s*%", line)
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
        iters = result["spec_iterations"]
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

    base_tps = 820 + rng.uniform(-50, 50) + (180 if mode == "mega" else 0)
    base_ms = (max_tokens / base_tps) * 1000

    if spec:
        accept_rate = rng.uniform(0.55, 0.85)
        speedup = (1 - accept_rate ** (k + 1)) / ((1 - accept_rate) * (k * 0.05 + 1))
        speedup = max(1.0, min(speedup, 2.5))
        spec_ms = base_ms / speedup
        spec_tps = max_tokens / (spec_ms / 1000)
        iters = max(1, int(max_tokens / (1 + k * accept_rate)))
        proposed = iters * k
        accepted = int(proposed * accept_rate)
    else:
        speedup = 1.0
        spec_ms = base_ms
        spec_tps = base_tps
        iters = max_tokens
        proposed = 0
        accepted = 0

    accept_rate = (accepted / proposed) if proposed else 0.0
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
        "acceptance_rate": round(accept_rate, 3) if proposed else 0.0,
        "spec_iterations": iters,
        "speedup": round(speedup, 2),
        "match": True,
        "raw": "(mock data -- binary not found)",
        "iterations_detail": iter_detail,
        "production": False,
        "baseline_text": "",
        "spec_text": "",
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_benchmark(
    mode: str,
    spec: bool,
    max_tokens: int,
    k: int,
    seed: int,
    prompt_len: int = 4,
    *,
    production: bool = False,
    draft_path: Optional[str] = None,
    target_path: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    prompt_text: Optional[str] = None,
    use_chat_template: bool = True,
    # Stochastic speculative decoding (Leviathan et al. exact algorithm)
    stochastic: bool = False,
    draft_temp: float = 1.0,
    adaptive_draft_temp: bool = False,
    adapt_accept: float = 0.5,
    adapt_gain: float = 0.055,
    adapt_ewma: float = 0.25,
    spec_seed: int = 12345,
) -> dict:
    exe = find_exe()
    if exe is None:
        time.sleep(0.4)  # simulate latency
        return _mock_result(mode, spec, max_tokens, k, seed)

    tmp_rm: list[Path] = []

    try:
        if production:
            if not draft_path or not target_path or not tokenizer_model or prompt_text is None:
                return {
                    "error": "production mode requires draft_path, target_path, tokenizer_model, and prompt_text",
                }
            draft = resolve_user_path(draft_path)
            target = resolve_user_path(target_path)
            if not draft.is_file():
                return {"error": f"Draft weights not found: {draft}"}
            if not target.is_file():
                return {"error": f"Target weights not found: {target}"}

            fd, prompt_tok_path = tempfile.mkstemp(suffix=".tok", prefix="sd_prompt_")
            os.close(fd)
            p_prompt = Path(prompt_tok_path)
            tmp_rm.append(p_prompt)

            fd2, out_tok_path = tempfile.mkstemp(suffix=".tok", prefix="sd_out_")
            os.close(fd2)
            p_out = Path(out_tok_path)
            tmp_rm.append(p_out)

            eos_token_id = -1
            try:
                _, eos_token_id = encode_prompt_tok(
                    tokenizer_model, prompt_text, p_prompt,
                    use_chat_template=use_chat_template,
                )
            except Exception as e:
                return {"error": f"Tokenizer encode failed: {e}"}

            cmd = [
                str(exe),
                f"--draft={draft}",
                f"--target={target}",
                f"--prompt-tok={p_prompt}",
                f"--output-tok={p_out}",
                f"--mode={'mega' if mode == 'mega' else 'multi'}",
                f"--max-tokens={max_tokens}",
                f"--k={k}",
            ]
            if eos_token_id >= 0:
                cmd.append(f"--eos-token={eos_token_id}")
            if stochastic:
                cmd.append("--stochastic-spec")
                cmd.append(f"--draft-temp={draft_temp:.4f}")
                cmd.append(f"--spec-seed={spec_seed}")
                if adaptive_draft_temp:
                    cmd.append("--adaptive-draft-temp")
                    cmd.append(f"--adapt-accept={adapt_accept:.4f}")
                    cmd.append(f"--adapt-gain={adapt_gain:.4f}")
                    cmd.append(f"--adapt-ewma={adapt_ewma:.4f}")
            timeout_sec = 600
        else:
            cmd = [
                str(exe),
                f"--mode={'mega' if mode == 'mega' else 'multi'}",
                f"--max-tokens={max_tokens}",
                f"--k={k}",
                f"--seed={seed}",
                f"--prompt-len={prompt_len}",
            ]
            if stochastic:
                cmd.append("--stochastic-spec")
                cmd.append(f"--draft-temp={draft_temp:.4f}")
                cmd.append(f"--spec-seed={spec_seed}")
                if adaptive_draft_temp:
                    cmd.append("--adaptive-draft-temp")
                    cmd.append(f"--adapt-accept={adapt_accept:.4f}")
                    cmd.append(f"--adapt-gain={adapt_gain:.4f}")
                    cmd.append(f"--adapt-ewma={adapt_ewma:.4f}")
            timeout_sec = 120

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(exe.parent),
        )
        stdout = proc.stdout
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or "Non-zero exit code"
            return {"error": err, "raw": stdout}

        result = parse_output(stdout)
        result["production"] = bool(production)
        result["stochastic"] = bool(stochastic)
        if production and prompt_text is not None:
            result["prompt_preview"] = (
                (prompt_text[:200] + "…")
                if len(prompt_text) > 200
                else prompt_text
            )
        else:
            result["prompt_preview"] = ""

        if production and tokenizer_model:
            try:
                result["baseline_text"] = decode_token_ids_to_text(
                    tokenizer_model, result.get("baseline_tokens") or []
                )
                result["spec_text"] = decode_token_ids_to_text(
                    tokenizer_model, result.get("spec_tokens") or []
                )
            except Exception as e:
                result["decode_error"] = str(e)
                result["baseline_text"] = ""
                result["spec_text"] = ""
        else:
            result.setdefault("baseline_text", "")
            result.setdefault("spec_text", "")

        # If spec=False, override so baseline == spec (no speculation)
        if not spec:
            result["spec_tokens"] = result["baseline_tokens"]
            result["spec_n"] = result["baseline_n"]
            result["spec_ms"] = result["baseline_ms"]
            result["spec_tok_per_s"] = result["baseline_tok_per_s"]
            result["speedup"] = 1.0
            result["draft_proposed"] = 0
            result["draft_accepted"] = 0
            result["acceptance_rate"] = 0.0
            result["spec_iterations"] = 0
            result["match"] = True
            result["iterations_detail"] = []
            result["spec_text"] = result.get("baseline_text", "")
        return result
    except subprocess.TimeoutExpired:
        return {"error": f"Benchmark timed out after {timeout_sec}s"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for p in tmp_rm:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
