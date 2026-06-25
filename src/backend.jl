abstract type ComputeBackend end

struct CPUBackend <: ComputeBackend end
struct CUDABackend <: ComputeBackend end

# Module-level cache so the CUDA probe runs at most once per session.
const _cuda_available_cache = Ref{Union{Bool, Nothing}}(nothing)

"""
    cuda_available()

Soft probe for CUDA availability.  Checks (in order):

1. **Environment variable override** — set `XAIVIZ_CUDA_AVAILABLE=true|false`
   for deterministic CI behaviour or user-level opting out.
2. **Cached result** — returns the cached boolean if the probe already ran.
3. **Package existence** — `Base.find_package("CUDA")` without loading the
   package; returns `false` immediately when CUDA.jl is not installed.
4. **Functional check** — as a last resort loads CUDA.jl and calls
   `CUDA.functional()`.

Returns `false` on CPU-only runners without ever importing CUDA.jl when the
package is absent or the env var says so.
"""
function cuda_available()
    # 1. Env-var override for deterministic CI / user control
    env_override = get(ENV, "XAIVIZ_CUDA_AVAILABLE", "")
    if env_override == "false"
        return false
    elseif env_override == "true"
        try
            @eval using CUDA
            return CUDA.functional()
        catch
            return false
        end
    end

    # 2. Cached result
    if _cuda_available_cache[] !== nothing
        return _cuda_available_cache[]::Bool
    end

    # 3. Soft probe: check CUDA.jl existence without importing
    if Base.find_package("CUDA") === nothing
        _cuda_available_cache[] = false
        return false
    end

    # 4. Functional check (CUDA.jl is present but may lack a GPU driver)
    try
        @eval using CUDA
        result = CUDA.functional()
        _cuda_available_cache[] = result
        return result
    catch
        _cuda_available_cache[] = false
        return false
    end
end

# Backward-compatible alias — existing code (tests, examples) can keep
# calling has_cuda() and automatically benefit from the softer probe.
has_cuda() = cuda_available()
