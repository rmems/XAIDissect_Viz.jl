# XAIDissectViz.jl

[![codecov](https://codecov.io/gh/rmems/XAIDissectViz.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rmems/XAIDissectViz.jl)

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

## CUDA Atmosphere Engine

The atmosphere viewer animates a synthetic `n_blocks × n_experts` activity field on top of the real xai-dissect metadata. The CPU path is canonical and always works. CUDA kernels accelerate the **visual activity field** — they do not run model inference and never touch Grok-1 weights.

- The CPU path is the reference implementation and the default. CPU-only CI runs the full test suite.
- CUDA kernels (`activity_decay_kernel!`, `topk_boost_kernel!`, `clamp_kernel!`) update the activity field on `CuArray`s when `CUDABackend()` is selected and `CUDA.functional()` is true. They are loaded lazily on first use so `using XAIDissectViz` stays cheap on headless / no-GPU hosts.
- Router dynamics are still simulated on top of real xai-dissect metadata. The package never loads, vendors, or distributes Grok-1 weights.
- A `RouterFrameCache` precomputes per-token top-k / entropy / confidence at launch via `simulate_router_topk_batch`, so the play loop only ticks `update_activity_field!` and a small inspector frame instead of the full synthetic forward pass.

### New public API

```julia
update_activity_field!         # CPU + CUDA dispatch wrapper
simulate_router_topk_batch     # cheap batched top-k/entropy/confidence
RouterFrameCache               # precomputed timeline cache
build_frame_cache              # fill a cache for n_tokens
get_frame                      # (block, token) -> cached state
topk_matrix_for_token          # n_blocks × top_k Int32 matrix
activity_matrix_for_token      # reconstructed activity at token
```

### Run the CUDA benchmark

```bash
julia --project examples/bench_cuda_atmosphere.jl
```

This script benchmarks `update_activity_field!` on CPU and (when `CUDA.functional()`) on CUDA, with warmup + `CUDA.@sync`, and verifies that one CUDA step matches the CPU reference within `atol=1e-5`. It uses a hand-built minimal `XAIReportBundle` when `XAI_DISSECT_REPORTS` is unset, so it runs on a clean checkout without model artifacts.

### Run the CUDA atmosphere demo

```bash
ENV["XAI_DISSECT_REPORTS"]=/path/to/grok1_run julia --project examples/grok_atmosphere_cuda_demo.jl
```

Loads a real xai-dissect report bundle and launches `launch_atmosphere(bundle; backend = CUDABackend())` when CUDA is functional, with a clean fallback to `CPUBackend()` otherwise. Requires a working display / OpenGL context (use `xvfb-run` on headless servers).

---

**Status**: CUDA atmosphere engine on `feat/cuda-atmosphere-engine` branch on top of the merged CPU/CUDA backends from PR #2.
- CPU + CUDA `update_activity_field!` (decay → boost → clamp)
- `RouterFrameCache` + batched `simulate_router_topk_batch`
- `launch_atmosphere` integration: cached play loop, perf label, FPS counter
- CUDA bench + demo scripts; weights never loaded

**Roadmap**: Create visual representations of xai-dissect Grok-1 report metadata for educational purposes.
- Load full 'xai-dissect' json reports
- Visualize router, blocks and experts
- Render router risk and readiness heatmap
- Simulate router logits and tok-k experts selection
- Build an interactive GLMakie Grok-1 atmosphere view
- CUDA-accelerated activity-field animation