"""
FastAPI backend for the Speculative Decode Explorer UI.

Endpoints:
  POST /api/run      -- single benchmark run
  POST /api/compare  -- all 4 mode combinations side-by-side
  POST /api/sweep    -- sweep k from 1..k_max, fixed mode
  POST /api/autotune -- grid search k (+ optional draft temp) for best throughput
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from runner import run_benchmark

app = FastAPI(title="Speculative Decode Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)


# ── Request / Response models ──────────────────────────────────────────────────


class ProductionFields(BaseModel):
    """Optional real SDEC weights + HF tokenizer for decoded text."""

    production: bool = False
    draft_path: Optional[str] = None
    target_path: Optional[str] = None
    tokenizer_model: Optional[str] = None
    prompt_text: Optional[str] = Field(None, max_length=100000)
    use_chat_template: bool = True

    @model_validator(mode="after")
    def production_requires_paths(self):
        if not self.production:
            return self
        spec_mode = getattr(self, "spec_mode", "two_model")
        missing = []
        if spec_mode == "self_spec":
            if not (self.target_path and str(self.target_path).strip()):
                missing.append("target_path")
        else:
            if not (self.draft_path and str(self.draft_path).strip()):
                missing.append("draft_path")
            if not (self.target_path and str(self.target_path).strip()):
                missing.append("target_path")
        if not (self.tokenizer_model and str(self.tokenizer_model).strip()):
            missing.append("tokenizer_model")
        if self.prompt_text is None or not str(self.prompt_text).strip():
            missing.append("prompt_text")
        if missing:
            raise ValueError(
                "production mode requires non-empty: " + ", ".join(missing)
            )
        return self


class SpecModeFields(BaseModel):
    """Two-model vs cascade self-speculation on one checkpoint."""

    spec_mode: str = Field("two_model", pattern="^(two_model|self_spec)$")
    draft_layers: int = Field(2, ge=1, le=39)


class StochasticFields(BaseModel):
    """Stochastic speculative decoding knobs (Leviathan et al. exact algorithm)."""

    stochastic: bool = False
    draft_temp: float = Field(1.0, ge=0.1, le=2.0)
    adaptive_draft_temp: bool = False
    adapt_accept: float = Field(0.5, ge=0.1, le=0.95)
    adapt_gain: float = Field(0.055, ge=0.001, le=0.5)
    adapt_ewma: float = Field(0.25, ge=0.05, le=0.95)
    spec_seed: int = Field(12345, ge=0)


class RunRequest(ProductionFields, SpecModeFields, StochasticFields):
    mode: str = Field("multi", pattern="^(multi|mega)$")
    spec: bool = True
    max_tokens: int = Field(32, ge=8, le=512)
    k: int = Field(4, ge=1, le=8)
    seed: int = Field(42, ge=0)
    prompt_len: int = Field(4, ge=1, le=16)


class SweepRequest(ProductionFields, SpecModeFields, StochasticFields):
    mode: str = Field("multi", pattern="^(multi|mega)$")
    max_tokens: int = Field(32, ge=8, le=512)
    k_max: int = Field(8, ge=2, le=8)
    seed: int = Field(42, ge=0)
    prompt_len: int = Field(4, ge=1, le=16)


class AutotuneRequest(ProductionFields, SpecModeFields, StochasticFields):
    mode: str = Field("multi", pattern="^(multi|mega)$")
    max_tokens: int = Field(32, ge=8, le=512)
    k_max: int = Field(8, ge=2, le=8)
    seed: int = Field(42, ge=0)
    prompt_len: int = Field(4, ge=1, le=16)
    min_acceptance: float = Field(0.70, ge=0.0, le=1.0)
    objective: str = Field("speedup", pattern="^(speedup|tok_per_s)$")
    sweep_temp: bool = False


def _bench_kwargs(req: RunRequest | SweepRequest | AutotuneRequest) -> dict:
    return {
        "mode": req.mode,
        "spec": getattr(req, "spec", True),
        "max_tokens": req.max_tokens,
        "k": getattr(req, "k", 4),
        "seed": req.seed,
        "prompt_len": req.prompt_len,
        "production": req.production,
        "draft_path": req.draft_path,
        "target_path": req.target_path,
        "tokenizer_model": req.tokenizer_model,
        "prompt_text": req.prompt_text,
        "use_chat_template": req.use_chat_template,
        "spec_mode": req.spec_mode,
        "draft_layers": req.draft_layers,
        "stochastic": req.stochastic,
        "draft_temp": req.draft_temp,
        "adaptive_draft_temp": req.adaptive_draft_temp,
        "adapt_accept": req.adapt_accept,
        "adapt_gain": req.adapt_gain,
        "adapt_ewma": req.adapt_ewma,
        "spec_seed": req.spec_seed,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _run_async(**kwargs) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: run_benchmark(**kwargs))


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    from runner import find_exe

    exe = find_exe()
    return {
        "status": "ok",
        "exe_found": exe is not None,
        "exe_path": str(exe) if exe else None,
        "mode": "real" if exe else "mock",
    }


@app.post("/api/run")
async def run(req: RunRequest):
    result = await _run_async(**_bench_kwargs(req))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/compare")
async def compare(req: RunRequest):
    """Run all 4 combinations: {multi,mega} x {no-spec,spec}."""
    combos = [
        ("multi", False),
        ("multi", True),
        ("mega", False),
        ("mega", True),
    ]
    base_k = _bench_kwargs(req)
    tasks = [
        _run_async(
            **{**base_k, "mode": mode, "spec": spec},
        )
        for mode, spec in combos
    ]
    results = await asyncio.gather(*tasks)
    out = []
    for (mode, spec), r in zip(combos, results):
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        out.append(
            {
                "label": f"{'Mega' if mode == 'mega' else 'Multi'} {'+ Spec' if spec else ''}".strip(),
                **r,
            }
        )
    return out


@app.post("/api/sweep")
async def sweep(req: SweepRequest):
    """Sweep k from 1..k_max with speculative enabled, return speedup per k."""
    base = _bench_kwargs(req)
    tasks = [
        _run_async(
            **{**base, "k": k, "spec": True},
        )
        for k in range(1, req.k_max + 1)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
    return [
        {
            "k": k,
            "speedup": r.get("speedup", 1.0),
            "acceptance_rate": r.get("acceptance_rate", 0.0),
            "spec_tok_per_s": r.get("spec_tok_per_s", 0.0),
        }
        for k, r in zip(range(1, req.k_max + 1), results)
    ]


@app.post("/api/autotune")
async def autotune(req: AutotuneRequest):
    """Grid-search k (and optionally draft temperature) to maximize throughput."""
    base = _bench_kwargs(req)

    if req.sweep_temp and req.stochastic:
        temps = [0.6, 0.8, 1.0, 1.2]
    else:
        temps = [req.draft_temp]

    combos: list[tuple[int, float]] = [
        (k, temp) for k in range(1, req.k_max + 1) for temp in temps
    ]
    tasks = [
        _run_async(**{**base, "k": k, "spec": True, "draft_temp": temp})
        for k, temp in combos
    ]
    results = await asyncio.gather(*tasks)

    grid = []
    for (k, temp), r in zip(combos, results):
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        grid.append(
            {
                "k": k,
                "draft_temp": temp,
                "speedup": r.get("speedup", 1.0),
                "acceptance_rate": r.get("acceptance_rate", 0.0),
                "spec_tok_per_s": r.get("spec_tok_per_s", 0.0),
                "spec_ms": r.get("spec_ms", 0.0),
            }
        )

    eligible = [
        g for g in grid if g["acceptance_rate"] >= req.min_acceptance
    ]
    pool = eligible if eligible else grid
    metric = "speedup" if req.objective == "speedup" else "spec_tok_per_s"
    best = max(pool, key=lambda g: g[metric])

    return {
        "best": best,
        "grid": grid,
        "min_acceptance": req.min_acceptance,
        "objective": req.objective,
        "n_eligible": len(eligible),
        "n_total": len(grid),
        "used_fallback": len(eligible) == 0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=True)
