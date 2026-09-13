# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

XAIDissectViz.jl is a pure Julia package — no Node.js, Python, or Docker services. It visualises xai-dissect JSON reports for the Grok-1 MoE architecture. See `README.md` for full context.

### API / CUDA probing

- `cuda_available()` — soft probe (env override, cache, `find_package` check, then `CUDA.functional()`); never imports CUDA.jl when it isn't installed
- `has_cuda()` — backward-compatible alias of `cuda_available()`
- Default path is **CPU** (`CPUBackend`); CUDA (`CUDABackend`) is optional and tests skip when unavailable
- Env: `XAIVIZ_CUDA_AVAILABLE=true|false` forces the probe result deterministically (used in CI)
- Other key exports: `update_activity_field!`, `simulate_router_topk_batch`, `RouterFrameCache` / `build_frame_cache` / `get_frame`, `topk_matrix_for_token`, `activity_matrix_for_token`, `load_report_bundle`, `launch_atmosphere` — see `README.md` and `src/XAIDissectViz.jl` for the full export list

### Julia version

Julia **1.12** is required (`Manifest.toml` pins `julia_version = "1.12.6"`). The runtime is installed at `/opt/julia-install/julia-1.12.6/bin/julia` and symlinked to `/usr/local/bin/julia`.

### Running tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Optional local alternative (skips the `Pkg.test` sandbox):

```bash
julia --project=. -e 'using Test; include("test/runtests.jl")'
```

All tests run headlessly on CPU. CUDA tests are gated and gracefully skip when no GPU is present. The `XAI_DISSECT_REPORTS` env var gates a real-report-load test; it is safe to leave unset.

### Precompilation / GLMakie caveat

GLMakie precompilation needs an X display context. On headless Linux, wrap with `xvfb-run`:

```bash
xvfb-run -a julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Once precompiled, the non-visual API (`using XAIDissectViz`, `simulate_router_frame`, etc.) works without `xvfb-run`. Only `launch_atmosphere()` (the interactive GUI) requires a running display.

### System dependencies

`xvfb`, `libgl1`, and `mesa-utils` are required for GLMakie precompilation and are pre-installed in the VM image.

### Key commands (also in README)

| Task | Command |
|---|---|
| Install deps | `julia --project=. -e 'using Pkg; Pkg.instantiate()'` |
| Precompile | `xvfb-run -a julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'` |
| Run tests | `julia --project=. -e 'using Pkg; Pkg.test()'` |
| Smoke load | `julia --project=. -e 'using XAIDissectViz; println("ok")'` |

### CI/CD Tooling

- **Aqua.jl** — quality checks (unused deps, ambiguities, type piracies, stale exports) run inside `test/runtests.jl` (`Aqua.test_all`), via `Pkg.test()`; no separate CI job
- **JuliaFormatter** — style enforcement via `format/Project.toml` + `.JuliaFormatter.toml`, checked in `.github/workflows/format.yml`
  - Local: `julia --project=format -e 'using Pkg; Pkg.instantiate(); using JuliaFormatter; format(".", verbose=true)'`
- **Coverage + Codecov** — `.github/workflows/ci.yml` runs `Pkg.test(coverage=true)`, processes lcov via `julia-actions/julia-processcoverage`, and uploads with `codecov/codecov-action` (`CODECOV_TOKEN`, soft-fail via `fail_ci_if_error: false`)
- **xvfb / OpenGL** — CI installs `xvfb`, `libgl1`, `mesa-utils` so GLMakie can precompile headlessly (see Precompilation section above)

### Boundaries and Constraints

1. Do not install Node.js, Python, or Docker services — this is a pure Julia project.
2. Do not modify the pinned Julia version in `Manifest.toml`.
3. Only run interactive GUI commands within an X display or via `xvfb-run`.
4. Use the provided paths for the Julia runtime rather than attempting to download new versions.
