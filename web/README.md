# Speculative Decode Explorer — Web UI

A React + Vite frontend with a FastAPI backend for interactively benchmarking the CUDA speculative decoding engine.

## Quick start

### 1. Backend

```powershell
cd web\backend
pip install -r requirements.txt
python server.py
# → running on http://localhost:8000
```

### 2. Frontend

```powershell
cd web\frontend
npm install
npm run dev
# → running on http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## Mock mode

If `cuda/spec_decode.exe` has not been compiled yet, the backend automatically falls back to **mock mode** — it returns realistic fake data so the UI is fully usable. A yellow banner at the top warns you when mock mode is active.

To use real GPU data: compile the CUDA binary first (see `cuda/README.md` or `cuda/build.bat`).

---

## Features

| Tab | What it does |
|---|---|
| **Single run** | Run one benchmark with chosen settings; shows throughput, speedup, acceptance rate, token output, and per-iteration timeline |
| **Compare modes** | Runs all 4 combinations (multi-kernel, megakernel) × (no-spec, spec) side-by-side bar chart |
| **Sweep k** | Sweeps draft tokens k from 1 to k_max; plots speedup and acceptance rate to find the empirical optimum |

### Controls

| Control | Effect |
|---|---|
| Kernel Mode | Switch between multi-kernel (one launch per op) and megakernel (single GPU-resident loop) |
| Speculative Decoding | Toggle draft+target vs. target-only baseline |
| Draft tokens k | Number of tokens the draft model proposes per iteration |
| Max tokens | Total tokens to generate |
| Prompt length | Number of input tokens |
| Seed | RNG seed for deterministic weight initialization |

---

## Architecture

```
Browser (React + Recharts)
       │
       │  /api/*  (Vite proxy)
       ▼
FastAPI server  (web/backend/server.py)
       │
       │  subprocess  (cuda/spec_decode.exe)
       ▼
CUDA binary → RTX 3050
```

The backend exposes three endpoints:

- `POST /api/run`     — single benchmark
- `POST /api/compare` — all 4 mode combinations in parallel
- `POST /api/sweep`   — sweep k values in parallel
- `GET  /api/health`  — returns exe path and mock/real status
