# Precomputed router-frame cache for the atmosphere viewer.
#
# The play loop in `launch_atmosphere` ticks at ~12 Hz across 300 tokens and
# 64 blocks. Calling `simulate_router_frame` per (block, token) every tick is
# wasteful: it allocates a synthetic `d_model × n_experts` W matrix per block
# and a `d_model` hidden vector per token. We pre-compute the lightweight
# top-k / entropy / confidence triplet per (block, token) once at launch via
# `simulate_router_topk_batch`, store it in `RouterFrameCache`, and replay
# from there.
#
# Memory budget: defaults `n_blocks=64`, `n_tokens=301` (token 0..300), top_k=2
# stay tiny:
#   topk         : 64 × 301 × 2  Int32   ≈ 154 KB
#   entropy      : 64 × 301      Float32 ≈  77 KB
#   confidence   : 64 × 301      Float32 ≈  77 KB
#   activity     : 64 × n_experts Float32 ≈ tiny
# Per-block forward sweeps reconstruct the activity field on demand by stepping
# `update_activity_field!` from zeros.
#
# Heavy fields (logits/probs over the full 8 experts for every block) are NOT
# cached. The selected-block inspector in the viewer calls `simulate_router_frame`
# for the single selected (block, token) on demand.

"""
    RouterFrameCache

Holds the precomputed batched router state for an entire timeline plus the
metadata needed to reconstruct the activity field deterministically.

Fields:
- `n_blocks`, `n_experts`, `top_k`
- `n_tokens`            — number of timeline slots (token indices `0` through `n_tokens-1`);
                           equals `n_tokens_arg + 1` where `n_tokens_arg` is the value passed
                           to [`build_frame_cache`](@ref)
- `seed`                — RNG seed used to populate the cache
- `topk`                — `n_blocks × top_k × n_tokens` `Array{Int32,3}` (flat view)
- `entropy`             — `n_blocks × n_tokens` `Matrix{Float32}`
- `confidence`          — `n_blocks × n_tokens` `Matrix{Float32}`
"""
struct RouterFrameCache
    n_blocks::Int
    n_experts::Int
    n_tokens::Int
    top_k::Int
    seed::Int
    topk::Array{Int32,3}            # n_blocks × top_k × n_tokens
    entropy::Matrix{Float32}        # n_blocks × n_tokens
    confidence::Matrix{Float32}     # n_blocks × n_tokens
end

# Token convention: the cache covers token indices 0:(n_tokens-1) inclusive,
# matching the timeline slider's 0-based range. `_token_pos` maps a token
# index to a 1-based slot in `frames`/`topk`/`entropy`/`confidence`.
@inline _token_pos(cache::RouterFrameCache, token_idx::Integer) = Int(token_idx) + 1

function _validate_token_idx(cache::RouterFrameCache, token_idx::Integer)
    0 <= Int(token_idx) <= cache.n_tokens - 1 ||
        throw(BoundsError("token_idx=$token_idx outside cache range 0:$(cache.n_tokens - 1)"))
    return nothing
end

"""
    build_frame_cache(bundle; backend=CPUBackend(), n_tokens=300, seed=42, top_k=2)

Pre-compute `simulate_router_topk_batch` for token indices `0:n_tokens` (inclusive),
i.e. `n_tokens + 1` slots, matching the atmosphere viewer's 0..n_tokens slider.

Returns a [`RouterFrameCache`](@ref).
"""
function build_frame_cache(bundle::XAIReportBundle;
                           backend::ComputeBackend = CPUBackend(),
                           n_tokens::Integer = 300,
                           seed::Integer = 42,
                           top_k::Integer = 2)
    n_tokens >= 0 || throw(ArgumentError("n_tokens must be >= 0"))
    haskey(bundle.metadata, "n_blocks") ||
        throw(ArgumentError("bundle.metadata missing \"n_blocks\""))
    haskey(bundle.metadata, "n_experts") ||
        throw(ArgumentError("bundle.metadata missing \"n_experts\""))
    n_blocks = bundle.metadata["n_blocks"]::Int
    n_experts = bundle.metadata["n_experts"]::Int
    k = Int(top_k)
    1 <= k <= n_experts ||
        throw(ArgumentError("top_k=$k must satisfy 1 <= top_k <= n_experts ($n_experts)"))

    n_slots = Int(n_tokens) + 1
    topk = Array{Int32,3}(undef, n_blocks, k, n_slots)
    entropy = Matrix{Float32}(undef, n_blocks, n_slots)
    confidence = Matrix{Float32}(undef, n_blocks, n_slots)

    @info "Building router frame cache" n_blocks n_experts n_tokens=n_tokens top_k=k seed=seed
    for t in 0:n_tokens
        s = simulate_router_topk_batch(bundle, t; seed=seed, backend=backend, top_k=k)
        slot = t + 1
        @inbounds for j in 1:k, b in 1:n_blocks
            topk[b, j, slot] = s.topk_by_block[b, j]
        end
        @inbounds for b in 1:n_blocks
            entropy[b, slot] = s.entropy_by_block[b]
            confidence[b, slot] = s.confidence_by_block[b]
        end
    end

    return RouterFrameCache(n_blocks, n_experts, Int(n_tokens) + 1, k, seed % Int,
                            topk, entropy, confidence)
end

"""
    get_frame(cache, block, token_idx) -> NamedTuple

Return the per-(block, token) cached state as
`(block, token_idx, topk, entropy, confidence)`. `topk` is a `Vector{Int32}`
of length `top_k`. Use [`simulate_router_frame`](@ref) for full logits/probs.
"""
function get_frame(cache::RouterFrameCache, block::Integer, token_idx::Integer)
    1 <= Int(block) <= cache.n_blocks ||
        throw(BoundsError("block=$block outside 1:$(cache.n_blocks)"))
    _validate_token_idx(cache, token_idx)
    slot = _token_pos(cache, token_idx)
    return (
        block       = Int(block),
        token_idx   = Int(token_idx),
        topk        = Vector{Int32}(cache.topk[Int(block), :, slot]),
        entropy     = cache.entropy[Int(block), slot],
        confidence  = cache.confidence[Int(block), slot],
    )
end

"""
    topk_matrix_for_token(cache, token_idx) -> Matrix{Int32}

Return the `n_blocks × top_k` `Int32` top-k matrix for `token_idx`. The
returned matrix is a fresh copy and safe to upload to a `CuArray`.
"""
function topk_matrix_for_token(cache::RouterFrameCache, token_idx::Integer)::Matrix{Int32}
    _validate_token_idx(cache, token_idx)
    slot = _token_pos(cache, token_idx)
    return Matrix{Int32}(cache.topk[:, :, slot])
end

"""
    activity_matrix_for_token(cache, token_idx;
                              decay=0.92f0, boost=0.55f0,
                              backend=CPUBackend()) -> Matrix{Float32}

Reconstruct the `n_blocks × n_experts` activity field at `token_idx` by
stepping `update_activity_field!` from zeros across token indices `0..token_idx`
using the cached top-k matrices. Deterministic given `(seed, decay, boost)`.

The forward sweep is performed on the requested `backend` (CPU/CUDA), then the
result is downloaded back to a host `Matrix{Float32}` regardless of backend so
callers always receive a CPU-readable matrix.
"""
function activity_matrix_for_token(cache::RouterFrameCache, token_idx::Integer;
                                   decay::Float32 = 0.92f0,
                                   boost::Float32 = 0.55f0,
                                   backend::ComputeBackend = CPUBackend())::Matrix{Float32}
    _validate_token_idx(cache, token_idx)
    slot_end = Int(token_idx) + 1
    activity = zeros(Float32, cache.n_blocks, cache.n_experts)
    if backend isa CUDABackend
        _ensure_cuda_kernels!()
        upload = Base.invokelatest(getfield, XAIDissectViz, :CuArray)
        download = Base.invokelatest(getfield, XAIDissectViz, :Array)
        act_gpu = upload(activity)
        tk_gpu_all = upload(cache.topk[:, :, 1:slot_end])
        for t in 1:slot_end
            tk_gpu = view(tk_gpu_all, :, :, t)
            update_activity_field!(backend, act_gpu, tk_gpu; decay=decay, boost=boost)
        end
        return Matrix{Float32}(download(act_gpu))
    else
        for t in 1:slot_end
            tk = view(cache.topk, :, :, t)
            update_activity_field!(backend, activity, tk; decay=decay, boost=boost)
        end
        return activity
    end
end
