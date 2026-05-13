"""
FastAPI backend for the Speculative Decode Explorer UI.

Endpoints:
  POST /api/run      -- single benchmark run
  POST /api/compare  -- all 4 mode combinations side-by-side
  POST /api/sweep    -- sweep k from 1..k_max, fixed mode
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
from concurrent.futures import ThreadPoolExecutor
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

class RunRequest(BaseModel):
    mode:        str  = Field("multi", pattern="^(multi|mega)$")
    spec:        bool = True
    max_tokens:  int  = Field(32,  ge=8,  le=128)
    k:           int  = Field(4,   ge=1,  le=8)
    seed:        int  = Field(42,  ge=0)
    prompt_len:  int  = Field(4,   ge=1,  le=16)


class SweepRequest(BaseModel):
    mode:        str  = Field("multi", pattern="^(multi|mega)$")
    max_tokens:  int  = Field(32,  ge=8,  le=128)
    k_max:       int  = Field(8,   ge=2,  le=8)
    seed:        int  = Field(42,  ge=0)
    prompt_len:  int  = Field(4,   ge=1,  le=16)


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
    result = await _run_async(
        mode=req.mode,
        spec=req.spec,
        max_tokens=req.max_tokens,
        k=req.k,
        seed=req.seed,
        prompt_len=req.prompt_len,
    )
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/compare")
async def compare(req: RunRequest):
    """Run all 4 combinations: {multi,mega} x {no-spec,spec}."""
    combos = [
        ("multi", False),
        ("multi", True),
        ("mega",  False),
        ("mega",  True),
    ]
    tasks = [
        _run_async(
            mode=mode, spec=spec,
            max_tokens=req.max_tokens, k=req.k,
            seed=req.seed, prompt_len=req.prompt_len,
        )
        for mode, spec in combos
    ]
    results = await asyncio.gather(*tasks)
    return [
        {"label": f"{'Mega' if mode == 'mega' else 'Multi'} {'+ Spec' if spec else ''}".strip(), **r}
        for (mode, spec), r in zip(combos, results)
    ]


@app.post("/api/sweep")
async def sweep(req: SweepRequest):
    """Sweep k from 1..k_max with speculative enabled, return speedup per k."""
    tasks = [
        _run_async(
            mode=req.mode, spec=True,
            max_tokens=req.max_tokens, k=k,
            seed=req.seed, prompt_len=req.prompt_len,
        )
        for k in range(1, req.k_max + 1)
    ]
    results = await asyncio.gather(*tasks)
    return [
        {
            "k": k,
            "speedup": r.get("speedup", 1.0),
            "acceptance_rate": r.get("acceptance_rate", 0.0),
            "spec_tok_per_s": r.get("spec_tok_per_s", 0.0),
        }
        for k, r in zip(range(1, req.k_max + 1), results)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
