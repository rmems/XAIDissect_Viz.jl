# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

XAIDissectViz.jl is a pure Julia package — no Node.js, Python, or Docker services. It visualises xai-dissect JSON reports for the Grok-1 MoE architecture. See `README.md` for full context.

### Julia version

Julia **1.12** is required (`Manifest.toml` pins `julia_version = "1.12.6"`). The runtime is installed at `/opt/julia-install/julia-1.12.6/bin/julia` and symlinked to `/usr/local/bin/julia`.

### Running tests

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
| Run tests | `julia --project=. -e 'using Test; include("test/runtests.jl")'` |
| Smoke load | `julia --project=. -e 'using XAIDissectViz; println("ok")'` |
