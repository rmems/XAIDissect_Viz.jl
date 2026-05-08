# XAIDissectViz.jl

**Grok-1 MoE Atmosphere** — Interactive visualization of xai-dissect reports for the 64-block, 8-expert Grok-1 Mixture-of-Experts architecture.

## What this is
- Visualizes **xai-dissect JSON reports** (routing, inventory, stats, SAAQ-readiness, experts).
- **Does NOT run Grok-1**, does **NOT load or redistribute any model weights**.
- Simulated router dynamics (logits, top-2 selection, activity) are clearly labeled "synthetic" when real reports are absent.
- CPU path always works. CUDA.jl accelerates router simulation and future visual kernels when available.

## Quick start
```julia
using XAIDissectViz

bundle = load_report_bundle()                    # pass real report dir)
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

**Status**: CPU/CUDA backends being added on `feat/router-cpu-cuda-backends` branch and GitHub issue #1 (Load xai-dissect JSON reports into typed Julia structs).
- CPU router logits/probaility/top-k utilities
- JSON report loader scaffold
- Intial 'RouterRecord' data structure

**Roadmap**: Create visual representation Grok 1 open weights, for eductional purposes.
- Load full 'xai-dissect' json reports
- Visualize router, blocks and experts
- Render router risk and readiness heatmap
- Simulate router logits and tok-k experts selection
- Build an interactive GLMakie Grok-1 atmosphere view