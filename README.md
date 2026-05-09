# XAIDissectViz.jl

**Grok-1 MoE Atmosphere** — Interactive visualization of xai-dissect reports for the 64-block, 8-expert Grok-1 Mixture-of-Experts architecture.

## What this is
- Visualizes **xai-dissect JSON reports** (routing, inventory, stats, SAAQ-readiness, experts).
- **Does NOT run Grok-1**, does **NOT load or redistribute any model weights**.
- `load_report_bundle` is a **strict real-JSON loader**; it errors when the 5 expected files are missing. No synthetic or in-memory fallback bundles are produced anywhere.
- Router dynamics are simulated (synthetic `W` and `h`) on top of the real report metadata so the atmosphere view stays animated. Weights are never read from disk.
- CPU path always works. CUDA.jl accelerates router simulation and future visual kernels when available.

## Quick start
```julia
using XAIDissectViz

# Point at either a directory containing the 5 JSONs directly,
# or a "run root" with exports/<ckpt_label>/*.json underneath.
ENV["XAI_DISSECT_REPORTS"] = "/path/to/grok1_run2_after_fixes_*"

bundle = load_report_bundle(ENV["XAI_DISSECT_REPORTS"])
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

**Status**: CPU/CUDA backends on `feat/router-cpu-cuda-backends` branch; addresses GitHub issue #1 (Load xai-dissect JSON reports into typed Julia structs).
- CPU router logits / probability / top-k utilities
- Strict real-JSON loader (`load_report_bundle`) parsing the 5 xai-dissect reports
- Typed structs: `RouterRecord`, `ExpertRecord`, `TensorMetricRecord`, `SAAQReadinessRecord`, `XAIReportBundle`

**Roadmap**: Create visual representations of xai-dissect Grok-1 report metadata for educational purposes.
- Load full 'xai-dissect' json reports
- Visualize router, blocks and experts
- Render router risk and readiness heatmap
- Simulate router logits and tok-k experts selection
- Build an interactive GLMakie Grok-1 atmosphere view