# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

XAIDissectViz.jl's only declared dependencies are Julia packages (see `Project.toml`) — there is no Node.js, Python, or Docker tooling in this repo today. It visualizes JSON reports produced by the sibling project `xai-dissect` (typically checked out at `~/rmems/xai-dissect`), for the 64-block, 8-expert Grok-1 MoE architecture. `xai-dissect` extracts checkpoint structure and emits reports; this package turns them into an interactive GLMakie "atmosphere" view. It never loads, runs, or redistributes Grok-1 weights — router dynamics are synthetic (seeded PRNG) simulations laid on top of real report metadata.

## Commands

```bash
# Install deps
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Precompile (GLMakie needs an X display context; wrap headless runs in xvfb-run)
xvfb-run -a julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Run tests (primary command; runs Aqua quality checks + the full test suite)
julia --project=. -e 'using Pkg; Pkg.test()'

# `Pkg.test()` above is the only supported entrypoint — test/Project.toml (Aqua, JSON3,
# Random) is merged in by Pkg.test()'s temp sandbox, so `include("test/runtests.jl")`
# under `--project=.` fails on `using Aqua`.

# Run a single @testset: comment out the ones you don't want in test/runtests.jl,
# or wrap the target testset in isolation — there is no built-in filter flag.

# Formatting (separate `format/` environment to avoid dependency conflicts with the main Project.toml)
julia --project=format -e 'using Pkg; Pkg.instantiate(); using JuliaFormatter; format(".", verbose=true)'

# Smoke load (no display required)
julia --project=. -e 'using XAIDissectViz; println("ok")'
```

Julia **1.12** is required (`Manifest.toml` pins `1.12.6`). Tests run headlessly on CPU; CUDA-specific tests skip gracefully when no GPU is present (`XAIVIZ_CUDA_AVAILABLE=false` forces this deterministically, as CI does). The `XAI_DISSECT_REPORTS` env var gates one real-report-load test and is safe to leave unset.

## Architecture

**Module structure** (`src/XAIDissectViz.jl` `include`s these in order): `types.jl` → `backend.jl` → `router.jl` → `kernels.jl` → `cache.jl` → `reports.jl` → `viz.jl`.

**Lazy heavy-dependency loading is a central design constraint here.** `using XAIDissectViz` is meant to stay cheap on headless/no-GPU hosts, so CUDA.jl and the GLMakie/GraphMakie/Graphs/Observables stack are not imported at module init — preserve this pattern for new code unless a maintainer decides to change it. Instead:
- `_ensure_cuda_kernels!()` in `src/kernels.jl` lazily `@eval`s `using CUDA` and `include`s `src/kernels_cuda.jl` (the actual `@cuda`-launched kernels) on first CUDA-backend use, guarded by a `ReentrantLock`.
- `launch_atmosphere()` in `src/viz.jl` lazily `@eval`s `using GLMakie, GraphMakie, Graphs, Observables` and raises a clear error (suggesting `xvfb-run`) if that package load itself fails. It then dispatches via `Base.invokelatest` to `_launch_atmosphere`, since the module's world-age hasn't seen the newly-loaded packages otherwise. Note the `try`/`catch` only wraps the package load: if the packages are already precompiled but no display/OpenGL context exists, the failure surfaces later from `display(fig)` inside `_launch_atmosphere` (outside that `try`) as a raw backend error, not the friendly `xvfb-run` message.
- Any code that needs to check CUDA symbols before they're loaded uses `Base.invokelatest(getfield, XAIDissectViz, :SomeSymbol)` rather than a direct reference — see `kernels.jl` and `cache.jl` for the pattern.

**CPU/CUDA backend dispatch**: `CPUBackend`/`CUDABackend` (in `backend.jl`) are singleton dispatch types passed through `router_logits`, `update_activity_field!`, etc. The CPU path is the reference implementation and default; CUDA accelerates both the visual activity-field kernels (decay → top-k boost → clamp, in `kernels_cuda.jl`) and the synthetic router-logit matmul in `router_logits(::CUDABackend, ...)` (`router.jl`) — never Grok-1 model inference or real weights. `cuda_available()` (alias `has_cuda()`) is a soft probe: env override (`XAIVIZ_CUDA_AVAILABLE`) → cache → `Base.find_package("CUDA")` existence check → `CUDA.functional()` — in that order, never importing CUDA.jl if it's absent.

**Report loading (`reports.jl`)**: `load_report_bundle` parses the 5 `xai-dissect` report files (`inventory.json`, `routing-report.json`, `stats.json`, `saaq-readiness.json`, `experts.json`) into typed structs, raising `ArgumentError` when any is missing (see Constraints below for why that's by design, not a gap to patch). It resolves either a directory containing those files directly, or a "run root" with `exports/<ckpt_label>/` beneath it, and raises `ArgumentError` if that run root has more than one valid checkpoint directory (the caller must disambiguate). `_minimal_bundle()` in `test/runtests.jl` is a hand-built test fixture, not part of the library.

**Router simulation determinism (`router.jl`)**: All randomness goes through local `Xoshiro` RNGs seeded via `deterministic_xoshiro_seed(seed, tag, ...)` — Julia's global RNG is never touched (a test explicitly asserts this). `simulate_router_frame` computes one full (block, token) frame (logits/probs/entropy/activity) and caches its synthetic `W` matrix per `(seed, block, d_model, n_experts)` key in a size-capped module-level `Dict`. `simulate_router_topk_batch` is a cheaper variant that computes only top-k/entropy/confidence across all blocks for one token, used to populate the timeline cache.

**`RouterFrameCache` (`cache.jl`)**: Precomputes `simulate_router_topk_batch` for every token in the atmosphere viewer's timeline (default 300 tokens) at launch, so the ~12 Hz play loop only steps `update_activity_field!` per tick instead of re-simulating a full forward pass. `activity_matrix_for_token` reconstructs the activity field at a given token by replaying cached top-k matrices from zero — deterministic given `(seed, decay, boost)`. Token indices are 0-based (matching the UI slider); `_token_pos` converts to Julia's 1-based storage.

**Viz layout (`viz.jl`)**: `launch_atmosphere` renders four regions — (A) global 64×8 heatmap + risk strip, (B) selected-block routing graph, (C) inspector panel (logits/probs/entropy/SAAQ/provenance), (D) timeline (token slider, play/pause, seed) — driven by `Observable`s in the `AtmosphereState` struct (`types.jl`).

## Constraints

- Pure Julia only — do not introduce Node.js, Python, or Docker.
- Do not modify the pinned Julia version in `Manifest.toml`.
- Avoid adding a synthetic/fallback path to `load_report_bundle`: it's intentional that missing report files raise `ArgumentError` instead, so raise that in code review if a change proposes otherwise.
- `Aqua.test_all` (in `test/runtests.jl`) intentionally disables two checks — this is expected, not a bug to fix: `piracies = false` because the `@eval using CUDA` pattern is deliberate augmentation, not type piracy; `stale_deps = false` because the lazily-loaded packages (CUDA, GLMakie, GraphMakie, Graphs, Observables, Makie) aren't imported at module init, so Aqua's static analysis would otherwise flag them as unused.
- Formatting is enforced by `.github/workflows/format.yml` using the separate `format/` Julia environment, not by `test/runtests.jl`.
