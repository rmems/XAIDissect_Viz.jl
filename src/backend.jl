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
    # 1. Env-var override for deterministic CI / user control.
    #    Accepts true/false/yes/no/1/0 (case-insensitive).
    env_override = lowercase(strip(get(ENV, "XAIVIZ_CUDA_AVAILABLE", "")))
    if !isempty(env_override)
        forced = env_override in ("true", "yes", "1")
        if forced
            try
                @eval using CUDA
                result = @eval CUDA.functional()
                _cuda_available_cache[] = result
                return result
            catch
                _cuda_available_cache[] = false
                return false
            end
        else
            _cuda_available_cache[] = false
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

    # 4. Functional check (CUDA.jl is present but may lack a GPU driver).
    #    Use @eval to avoid world-age errors on Julia 1.12+ when calling
    #    into a module that was loaded at runtime via `@eval using`.
    try
        @eval using CUDA
        result = @eval CUDA.functional()
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
