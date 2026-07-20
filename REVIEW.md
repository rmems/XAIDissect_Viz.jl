# Code Review & PR Analysis Report

## PR #19 — "Add AGENTS.md with Cursor Cloud development environment setup"

### Current Status

| Item | Status |
|---|---|
| Merge state | `dirty` — **branch conflict with `main`** |
| Codacy Static Code Analysis | ❌ `action_required` — PR branch AGENTS.md missing "Boundaries and Constraints" section, stale test command |
| Aikido Security (x2) | ✅ Passed |
| Gitar | ✅ Approved |
| CodeRabbit | ⚠️ Changes requested (2 inline comments — both addressed) |

### Branch Conflict

The PR branch `cursor/dev-env-setup-754f` diverges from `main` because **PR #36** already merged a corrected `AGENTS.md` into `main` (commit `e319d0b`). The PR branch still carries the original (stale) version from commit `78d7ec4` plus our fix commit `42ede3f`. Rebasing produces add/add conflicts in `AGENTS.md` at lines 11, 16, and 40.

**Resolution**: This PR is redundant — `main` already contains the complete, corrected `AGENTS.md`. Close PR #19 without merging.

### Codacy Failure

Codacy flagged the PR branch's `AGENTS.md` for:
- Missing "Boundaries and Constraints" section (present on `main` via PR #36)
- Stale test command: `using Test; include("test/runtests.jl")` instead of `using Pkg; Pkg.test()`

Since `main` already has the complete file, closing PR #19 resolves the Codacy finding.

---

## Validation Commands

> **Important**: The commands below are organized by runtime. Use **Bash** commands
> in a terminal shell. Use **Julia REPL** commands inside `julia>` or VS Code's
> Julia REPL.

### Julia REPL Commands (run inside `julia --project=.`)

```julia
# 1. Install dependencies
using Pkg; Pkg.instantiate()

# 2. Precompile
using Pkg; Pkg.instantiate(); Pkg.precompile()

# 3. Headless smoke load
using XAIDissectViz; println("ok")

# 4. Run full test suite (CPU only, CUDA gracefully skipped)
using Pkg; Pkg.test()

# 5. Run tests with coverage
using Pkg; Pkg.test(coverage=true)
```

### Bash Shell Commands (run in a terminal, not the Julia REPL)

```bash
# 1. Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 2. Precompile (requires xvfb on headless Linux for GLMakie)
xvfb-run -a julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# 3. Headless smoke load (no display required)
julia --project=. -e 'using XAIDissectViz; println("ok")'

# 4. Run full test suite (CPU only, CUDA gracefully skipped)
julia --project=. -e 'using Pkg; Pkg.test()'

# 5. Run tests with coverage
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'

# 6. Check formatting (uses the separate format/ environment)
julia --project=format -e 'using JuliaFormatter; exit(format(".") ? 0 : 1)'
```

### Remote CI Commands (GitHub Actions — `.github/workflows/ci.yml`)

The CI workflow runs on `ubuntu-latest` with Julia 1.12 and `XAIVIZ_CUDA_AVAILABLE=false`:

```bash
# CI step 1: Install system deps
sudo apt-get update && sudo apt-get install -y xvfb libgl1 mesa-utils

# CI step 2: Instantiate + precompile
xvfb-run -a julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# CI step 3: Smoke load
julia --project=. -e 'using XAIDissectViz; println("XAIDissectViz loaded headlessly")'

# CI step 4: Run tests (CPU-only, coverage enabled)
XAIVIZ_CUDA_AVAILABLE=false julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'

# CI step 5: Process + upload coverage (julia-processcoverage action → Codecov)
```

---

## CUDA / Blackwell SM_120 GPU Test Commands

> **Target hardware**: NVIDIA Blackwell architecture (SM 12.0 / `sm_120`), e.g. RTX 5090/5080.

The test suite gates CUDA tests behind `cuda_available()` (see `src/backend.jl`). On a machine with a Blackwell GPU and working NVIDIA driver + CUDA toolkit:

### Verify GPU (Bash)

```bash
# Check NVIDIA driver and GPU model
nvidia-smi
```

### Verify GPU from Julia REPL (run inside `julia --project=.`)

```julia
using CUDA
println("CUDA functional: ", CUDA.functional())
if CUDA.functional()
    dev = CUDA.device()
    println("Device: ", CUDA.name(dev))
    cap = CUDA.capability(dev)
    println("Compute capability: ", cap)
    # Blackwell SM_120 = capability (12, 0)
    @assert cap >= v"12.0" "Expected SM 12.0+ (Blackwell), got $cap"
    println("✓ Blackwell SM_120 confirmed")
end
```

### Run CUDA tests from Julia REPL

```julia
# Full test suite — CUDA tests auto-detected when gpu is present
#   Tests that exercise CUDA:
#     - "CUDA path (if available)" — router_frame on CUDABackend
#     - "update_activity_field! CPU vs CUDA match (gated)" — isapprox check
using Pkg; Pkg.test()
```

### Run CUDA tests from Bash

```bash
# Option A: Let the soft probe auto-detect (default — no env var needed)
julia --project=. -e 'using Pkg; Pkg.test()'

# Option B: Explicitly enable CUDA tests via env var
XAIVIZ_CUDA_AVAILABLE=true julia --project=. -e 'using Pkg; Pkg.test()'
```

### CUDA benchmark and demo (Bash)

```bash
# CUDA atmosphere micro-benchmark (CPU vs CUDA speedup)
julia --project examples/bench_cuda_atmosphere.jl

# CUDA atmosphere demo (requires display or xvfb-run)
xvfb-run -a julia --project=. examples/grok_atmosphere_cuda_demo.jl
```

### Targeted CUDA kernel validation (Julia REPL)

```julia
using XAIDissectViz, CUDA, Random

@assert CUDA.functional() "CUDA not functional"
@assert CUDA.capability(CUDA.device()) >= v"12.0" "Not Blackwell SM_120"

n_blocks, n_experts, top_k = 64, 8, 2
rng = Random.Xoshiro(42)

A_cpu = rand(rng, Float32, n_blocks, n_experts) .* 0.3f0
T = rand(rng, Int32(1):Int32(n_experts), n_blocks, top_k)

A_gpu = CUDA.CuArray(copy(A_cpu))
T_gpu = CUDA.CuArray(T)

for _ in 1:10
    update_activity_field!(CPUBackend(), A_cpu, T)
    update_activity_field!(CUDABackend(), A_gpu, T_gpu)
end
CUDA.synchronize()

max_diff = maximum(abs.(Array(A_gpu) .- A_cpu))
println("CPU vs CUDA max diff: ", max_diff)
@assert max_diff < 1e-5 "CPU/CUDA mismatch: $max_diff"
println("✓ CPU/CUDA match on Blackwell SM_120 (atol < 1e-5)")
```

### CUDA test gating logic

The test suite uses three layers of CUDA gating:

| Gate | Mechanism | Effect |
|---|---|---|
| `XAIVIZ_CUDA_AVAILABLE=false` | Env var in CI | `cuda_available()` returns `false` immediately, CUDA.jl never imported |
| `Base.find_package("CUDA")` | Soft probe | If CUDA.jl not installed, returns `false` without `@eval using CUDA` |
| `CUDA.functional()` | Runtime check | If no driver/GPU, returns `false` after loading CUDA.jl |

On CI (GitHub Actions), `XAIVIZ_CUDA_AVAILABLE=false` ensures zero CUDA overhead. Locally on a Blackwell machine, omit the env var or set it to `true` to exercise the full CUDA path.

---

## Recommendation

**Close PR #19** — `AGENTS.md` is already on `main` (via PR #36) with all fixes applied. The branch conflict and Codacy failure are artifacts of this redundancy.
