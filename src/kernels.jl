# Activity-field kernels for the Grok-1 MoE atmosphere engine.
#
# These kernels evolve a synthetic `n_blocks × n_experts` Float32 activity field
# that drives the heatmap animation. They operate purely on visualization state;
# they do NOT touch real Grok-1 weights and do NOT perform model inference.
#
# CPU paths use plain Julia loops on `Array{Float32,2}` / `Array{Int32,2}`.
# CUDA paths dispatch to lightweight `@cuda` launches on `CuArray`s. CUDA.jl is
# loaded lazily on first CUDABackend call to mirror the headless-safe pattern
# established by the merged router/viz backends — `using XAIDissectViz` must
# stay cheap and never require a working GPU.

# --- CPU reference functions ---

"""
    update_activity_field_cpu!(activity, topk_by_block; decay=0.92f0, boost=0.55f0)

In-place CPU update of the `n_blocks × n_experts` Float32 activity field:
multiplies by `decay`, then adds `boost` to the experts selected per block in
`topk_by_block` (a `n_blocks × top_k` `Int32` matrix), then clamps to `[0, 1]`.
Out-of-range expert indices are skipped (defensive; tests guard the contract).
"""
function update_activity_field_cpu!(activity::AbstractMatrix{Float32},
                                    topk_by_block::AbstractMatrix{<:Integer};
                                    decay::Float32 = 0.92f0,
                                    boost::Float32 = 0.55f0)
    n_blocks, n_experts = size(activity)
    top_k = size(topk_by_block, 2)
    size(topk_by_block, 1) == n_blocks ||
        throw(DimensionMismatch("topk_by_block rows ($(size(topk_by_block,1))) must equal n_blocks ($n_blocks)"))
    batch_decay_activity_cpu!(activity, decay)
    apply_topk_boosts_cpu!(activity, topk_by_block, boost)
    @inbounds for i in eachindex(activity)
        v = activity[i]
        activity[i] = v < 0f0 ? 0f0 : (v > 1f0 ? 1f0 : v)
    end
    return activity
end

"""
    batch_decay_activity_cpu!(activity, decay)

Multiply every entry of `activity` by `decay` in place.
"""
function batch_decay_activity_cpu!(activity::AbstractMatrix{Float32}, decay::Float32)
    @inbounds @simd for i in eachindex(activity)
        activity[i] = activity[i] * decay
    end
    return activity
end

"""
    apply_topk_boosts_cpu!(activity, topk_by_block, boost)

For each block row `b`, add `boost` to `activity[b, topk_by_block[b, k]]` for
every `k`, capped at `1.0f0`. Out-of-range expert indices are ignored.
"""
function apply_topk_boosts_cpu!(activity::AbstractMatrix{Float32},
                                topk_by_block::AbstractMatrix{<:Integer},
                                boost::Float32)
    n_blocks, n_experts = size(activity)
    top_k = size(topk_by_block, 2)
    @inbounds for b in 1:n_blocks
        for k in 1:top_k
            e = Int(topk_by_block[b, k])
            if 1 <= e <= n_experts
                v = activity[b, e] + boost
                activity[b, e] = v > 1f0 ? 1f0 : v
            end
        end
    end
    return activity
end

# --- Public dispatch wrapper ---

"""
    update_activity_field!(backend, activity, topk_by_block;
                           decay=0.92f0, boost=0.55f0)

Public wrapper used by the atmosphere viewer. Dispatches to the CPU loop for
[`CPUBackend`](@ref) and to a `@cuda`-launched kernel for [`CUDABackend`](@ref).

Shapes:
- `activity::AbstractMatrix{Float32}` of size `n_blocks × n_experts`
- `topk_by_block::AbstractMatrix{<:Integer}` of size `n_blocks × top_k`

The CUDA path expects `CuArray{Float32,2}` and `CuArray{Int32,2}`; CUDA.jl is
loaded lazily on first invocation.
"""
function update_activity_field!(::CPUBackend,
                                activity::AbstractMatrix{Float32},
                                topk_by_block::AbstractMatrix{<:Integer};
                                decay::Float32 = 0.92f0,
                                boost::Float32 = 0.55f0)
    return update_activity_field_cpu!(activity, topk_by_block; decay = decay, boost = boost)
end

# --- CUDA dispatch (lazy-loaded) ---

const _CUDA_KERNELS_LOADED = Ref(false)
const _CUDA_KERNELS_LOCK = ReentrantLock()
const _CUDA_KERNELS_FILE = joinpath(@__DIR__, "kernels_cuda.jl")

function _ensure_cuda_kernels!()
    _CUDA_KERNELS_LOADED[] && return nothing
    lock(_CUDA_KERNELS_LOCK) do
        _CUDA_KERNELS_LOADED[] && return nothing
        @eval XAIDissectViz using CUDA
        @eval XAIDissectViz include($_CUDA_KERNELS_FILE)
        _CUDA_KERNELS_LOADED[] = true
    end
    return nothing
end

function update_activity_field!(::CUDABackend,
                                activity::AbstractMatrix{Float32},
                                topk_by_block::AbstractMatrix{<:Integer};
                                decay::Float32 = 0.92f0,
                                boost::Float32 = 0.55f0)
    _ensure_cuda_kernels!()
    CuArray_T = Base.invokelatest(getfield, XAIDissectViz, :CuArray)
    is_gpu = Base.invokelatest(isa, activity, CuArray_T) ||
             (activity isa SubArray && Base.invokelatest(isa, parent(activity), CuArray_T))
    if !is_gpu
        cuda_ok = try; Base.invokelatest(getfield, XAIDissectViz, :CUDA) |>
                       m -> Base.invokelatest(getfield, m, :functional) |>
                       f -> Base.invokelatest(f); catch; false; end
        if cuda_ok
            throw(ArgumentError(
                "CUDABackend requires CuArray inputs; got CPU $(typeof(activity)). " *
                "Upload with CuArray(...) first, or use CPUBackend()."))
        else
            @warn "CUDABackend with CPU arrays and CUDA non-functional; falling back to CPU"
            return update_activity_field_cpu!(activity, topk_by_block; decay=decay, boost=boost)
        end
    end
    f = Base.invokelatest(getfield, XAIDissectViz, :_update_activity_field_cuda!)
    return Base.invokelatest(f, activity, topk_by_block, decay, boost)
end
