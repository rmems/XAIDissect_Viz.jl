# CUDA.jl kernels for the activity field. Loaded lazily by
# `_ensure_cuda_kernels!()` in `kernels.jl` so that `using XAIDissectViz` does
# not pay the CUDA load cost on headless hosts.
#
# Conventions:
# - Kernels accept `CuDeviceArray`s and run on the device.
# - Index arithmetic is done in Int32 to match `threadIdx`/`blockIdx`.
# - All kernels return `nothing` per the @cuda contract.

function activity_decay_kernel!(activity, decay::Float32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= length(activity)
        @inbounds activity[idx] = activity[idx] * decay
    end
    return nothing
end

function topk_boost_kernel!(
    activity,
    topk_by_block,
    boost::Float32,
    n_blocks::Int32,
    n_experts::Int32,
    top_k::Int32,
)
    b = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if b > n_blocks
        return nothing
    end
    @inbounds for k = Int32(1):top_k
        e = Int32(topk_by_block[b, k])
        if e >= Int32(1) && e <= n_experts
            v = activity[b, e] + boost
            activity[b, e] = v > 1.0f0 ? 1.0f0 : v
        end
    end
    return nothing
end

function clamp_kernel!(activity, lo::Float32, hi::Float32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= length(activity)
        @inbounds begin
            v = activity[idx]
            activity[idx] = v < lo ? lo : (v > hi ? hi : v)
        end
    end
    return nothing
end

function _update_activity_field_cuda!(
    activity,
    topk_by_block,
    decay::Float32,
    boost::Float32,
)
    n = length(activity)
    n_blocks_v, n_experts_v = size(activity)
    top_k_v = size(topk_by_block, 2)
    size(topk_by_block, 1) == n_blocks_v || throw(
        DimensionMismatch(
            "topk_by_block rows ($(size(topk_by_block,1))) must equal n_blocks ($n_blocks_v)",
        ),
    )

    threads_e = 256
    blocks_e = cld(n, threads_e)
    CUDA.@cuda threads = threads_e blocks = blocks_e activity_decay_kernel!(activity, decay)

    threads_b = 128
    blocks_b = cld(n_blocks_v, threads_b)
    CUDA.@cuda threads = threads_b blocks = blocks_b topk_boost_kernel!(
        activity,
        topk_by_block,
        boost,
        Int32(n_blocks_v),
        Int32(n_experts_v),
        Int32(top_k_v),
    )

    CUDA.@cuda threads = threads_e blocks = blocks_e clamp_kernel!(activity, 0.0f0, 1.0f0)
    return activity
end
