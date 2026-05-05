# XAIDissectViz.jl

**Grok-1 MoE Atmosphere** — Interactive visualization of xai-dissect reports for the 64-block, 8-expert Grok-1 Mixture-of-Experts architecture.

## What this is
- Visualizes **xai-dissect JSON reports** (routing, inventory, stats, SAAQ-readiness, experts).
- **Does NOT run Grok-1**, does **NOT load or redistribute any model weights**.
- Simulated router dynamics (logits, top-2 selection, activity) are clearly labeled "synthetic" when real reports are absent.
- CPU path always works. CUDA.jl accelerates router simulation and future visual kernels when available.

## Current MVP (feat/router-cpu-cuda-backends)
- Typed data model: `XAIReportBundle`, `RouterRecord`, `ExpertRecord`, `SAAQReadinessRecord`, `RouterFrame`, etc.
- `load_report_bundle(path)` — parses the 5 standard xai-dissect JSONs or builds a rich **in-memory synthetic bundle** (no fixture files committed).
- Full interactive GLMakie window with four regions:
  - **A. Global MoE Map**: 64×8 heatmap of expert activity + side risk/SAAQ strip. Click rows to select block.
  - **B. Selected Block Graph**: input → router (6144→8 logits) → 8 experts (top-2 glow/pulse) → merge/output.
  - **C. Inspector**: live logits, probabilities, entropy, confidence, router risk, readiness status, provenance.
  - **D. Timeline**: token 0–300 slider, Play/Pause loop, random seed control.
- `simulate_router_frame(bundle, block, token; backend)` — reproducible synthetic routing using existing `router_logits` / `router_probs` / `topk_experts`.
- Backend: `CPUBackend()` (default) or `CUDABackend()` (auto-fallback with warning if CUDA unavailable).
- One small practical CUDA activity helper pattern included.

## Quick start
```julia
using XAIDissectViz

bundle = load_report_bundle()                    # synthetic 64×8 (or pass real report dir)
backend = has_cuda() ? CUDABackend() : CPUBackend()
launch_atmosphere(bundle; backend = backend)
```

See `examples/grok_atmosphere_demo.jl`.

## Running tests
```bash
julia --project -e 'using Test; include("test/runtests.jl")'
```

## Relationship to xai-dissect
- `xai-dissect` extracts Grok-1 checkpoint structure and emits the JSON reports.
- `XAIDissectViz.jl` turns those reports into an explorable interactive atmosphere map.

## License & Disclaimer
See GPU support note below. This project never vendors weights, CUDA binaries, or proprietary artifacts.

## GPU Support
XAIDissectViz uses CUDA.jl for optional GPU acceleration of router simulation and visual kernels. It does **not** vendor or redistribute the NVIDIA CUDA Toolkit, drivers, or any model weights. Users are responsible for installing compatible NVIDIA drivers and accepting upstream licenses.

---

**Status**: MVP complete on `feat/router-cpu-cuda-backends`. Closes GitHub issue #1 (typed structs + report loading).