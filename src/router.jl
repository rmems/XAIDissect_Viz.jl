using Random

const _W_CACHE = Dict{Tuple{Int, Int, Int, Int}, Matrix{Float32}}()
const _W_CACHE_MAX_ENTRIES = 256

@inline function _mix_u64(x::UInt64)::UInt64
    x ⊻= x >> 30
    x *= 0xbf58476d1ce4e5b9
    x ⊻= x >> 27
    x *= 0x94d049bb133111eb
    x ⊻= x >> 31
    return x
end

const _ROUTER_SEED_TAG_W = 0x243f6a8885a308d3 % UInt64
const _ROUTER_SEED_TAG_H = 0x13198a2e03707344 % UInt64
const _ROUTER_SEED_TAG_A = 0xa4093822299f31d0 % UInt64
const _ROUTER_SEED_TAG_BATCH = 0x082efa98ec4e6c89 % UInt64

@inline _u64_bits(x::UInt64) = x
@inline _u64_bits(x::Int64) = reinterpret(UInt64, x)
@inline _u64_bits(x::Integer) = x % UInt64

@inline function deterministic_xoshiro_seed(
    seed::Integer,
    tag::UInt64,
    a::Integer,
    b::Integer = 0,
    c::Integer = 0,
    d::Integer = 0,
)::UInt64
    x = _u64_bits(seed) ⊻ tag
    x ⊻= _u64_bits(a) * 0x9e3779b97f4a7c15
    x ⊻= _u64_bits(b) * 0xbf58476d1ce4e5b9
    x ⊻= _u64_bits(c) * 0x94d049bb133111eb
    x ⊻= _u64_bits(d) * 0xd6e8feb86659fd93
    return _mix_u64(x)
end

function router_logits(::CPUBackend, h::AbstractVector, W::AbstractMatrix)
    h32 = h isa AbstractVector{Float32} ? h : Float32.(h)
    W32 = W isa AbstractMatrix{Float32} ? W : Float32.(W)
    return vec(transpose(h32) * W32)
end

function router_logits(::CUDABackend, h::AbstractVector, W::AbstractMatrix)
    @eval using CUDA
    h_gpu = CUDA.CuArray(Float32.(h))
    W_gpu = CUDA.CuArray(Float32.(W))
    return Array(vec(transpose(h_gpu) * W_gpu))
end

function router_probs(logits::AbstractVector)
    shifted = logits .- maximum(logits)
    exps = exp.(shifted)
    return exps ./ sum(exps)
end

function topk_experts(probs::AbstractVector, k::Integer = 2)
    return partialsortperm(probs, 1:k; rev = true)
end

# --- Router simulation for atmosphere timeline ---

# Local RNGs only — never mutate Julia's global RNG. The UI seed parameter
# fully determines the (synthetic, viz-only) router weights and hidden state.
function simulate_router_frame(
    bundle::XAIReportBundle,
    block::Int,
    token_idx::Int;
    backend::ComputeBackend = CPUBackend(),
    seed::Integer = 42,
)::RouterFrame
    haskey(bundle.metadata, "d_model") ||
        throw(ArgumentError("bundle.metadata missing \"d_model\""))
    haskey(bundle.metadata, "n_experts") ||
        throw(ArgumentError("bundle.metadata missing \"n_experts\""))
    d_model = bundle.metadata["d_model"]::Int
    n_experts = bundle.metadata["n_experts"]::Int

    key = (seed % Int, block, d_model, n_experts)
    if !haskey(_W_CACHE, key) && length(_W_CACHE) >= _W_CACHE_MAX_ENTRIES
        empty!(_W_CACHE)  # hard cap: never retain unbounded per-seed weights
    end
    W = get!(_W_CACHE, key) do
        rng_W = Xoshiro(
            deterministic_xoshiro_seed(seed, _ROUTER_SEED_TAG_W, block, d_model, n_experts),
        )
        randn(rng_W, Float32, d_model, n_experts) .* 0.018f0
    end

    rng_h = Xoshiro(deterministic_xoshiro_seed(seed, _ROUTER_SEED_TAG_H, block, token_idx))
    phase = 2π * (token_idx % 50) / 50
    h =
        randn(rng_h, Float32, d_model) .* 0.08f0 .+ sin(phase) * 0.25f0 .+
        cos(phase * 1.7) * 0.12f0

    logits = router_logits(backend, h, W)
    probs = router_probs(logits)
    top_k_cfg = get(bundle.metadata, "top_k", 2)::Int
    k = clamp(top_k_cfg, 1, length(probs))
    topk = topk_experts(probs, k)

    entropy = -sum(@. probs * log(max(probs, 1.0f-12)))

    rng_a = Xoshiro(deterministic_xoshiro_seed(seed, _ROUTER_SEED_TAG_A, block, token_idx))
    activity = fill(0.08f0, n_experts)
    activity[topk] .+= 0.65f0
    activity .+= 0.03f0 .* rand(rng_a, Float32, n_experts)
    clamp!(activity, 0.0f0, 1.0f0)

    RouterFrame(block, token_idx, logits, probs, topk, entropy, activity)
end

# Small helper for live activity decay/boost during play loop (used by viz)
function update_expert_activity!(
    activity::Vector{Float32},
    topk::Vector{Int};
    decay::Float32 = 0.92f0,
    boost::Float32 = 0.55f0,
)
    activity .*= decay
    for e in topk
        activity[e] = min(1.0f0, activity[e] + boost)
    end
    clamp!(activity, 0.0f0, 1.0f0)
    return activity
end

# Optional tiny CUDA activity kernel (practical for future larger expert sets)
function update_expert_activity!(
    activity::Vector{Float32},
    topk::Vector{Int},
    ::CUDABackend;
    decay::Float32 = 0.92f0,
    boost::Float32 = 0.55f0,
)
    if has_cuda()
        @eval using CUDA
        # For 8 experts the CPU version is fine; this shows the pattern
        # A real kernel would be launched with @cuda for large n_experts
        activity .*= decay
        for e in topk
            activity[e] = min(1.0f0, activity[e] + boost)
        end
        clamp!(activity, 0.0f0, 1.0f0)
    else
        @warn "CUDABackend requested but CUDA unavailable; using CPU activity update"
        update_expert_activity!(activity, topk; decay = decay, boost = boost)
    end
    return activity
end

# --- Lightweight batched router top-k for cache build ---
#
# `simulate_router_frame` returns the full logits/probs/activity payload for a
# single (block, token). The atmosphere viewer needs only top-k + entropy +
# confidence per (block, token) for the heatmap loop. This batched variant
# computes those quantities for every block at a fixed token index in one shot,
# without allocating per-block W matrices or producing the heavy fields.
#
# Determinism: seeded from `(seed, block, token_idx)` via a local Xoshiro RNG;
# the global RNG is never touched. Synthetic logits are intentionally cheap
# (sin/cos basis + per-(block,seed) bias) so cache builds stay fast even at
# n_blocks=64, n_tokens=300.

function simulate_router_topk_batch(
    bundle::XAIReportBundle,
    token_idx::Integer;
    seed::Integer = 42,
    backend::ComputeBackend = CPUBackend(),
    top_k::Integer = 2,
)
    haskey(bundle.metadata, "n_blocks") ||
        throw(ArgumentError("bundle.metadata missing \"n_blocks\""))
    haskey(bundle.metadata, "n_experts") ||
        throw(ArgumentError("bundle.metadata missing \"n_experts\""))
    n_blocks = bundle.metadata["n_blocks"]::Int
    n_experts = bundle.metadata["n_experts"]::Int
    k = Int(top_k)
    1 <= k <= n_experts ||
        throw(ArgumentError("top_k=$k must satisfy 1 <= top_k <= n_experts ($n_experts)"))

    topk = Matrix{Int32}(undef, n_blocks, k)
    entropy = Vector{Float32}(undef, n_blocks)
    confidence = Vector{Float32}(undef, n_blocks)
    logits_buf = Vector{Float32}(undef, n_experts)

    phase = 2π * (Int(token_idx) % 50) / 50
    base_sin = Float32(sin(phase))
    base_cos = Float32(cos(phase * 1.7))

    for b = 1:n_blocks
        rng = Xoshiro(
            deterministic_xoshiro_seed(seed, _ROUTER_SEED_TAG_BATCH, b, Int(token_idx)),
        )
        for e = 1:n_experts
            ang = Float32(2π * ((b * 7 + e * 11) % 32) / 32)
            logits_buf[e] =
                base_sin * Float32(cos(ang)) +
                base_cos * Float32(sin(ang)) +
                0.18f0 * randn(rng, Float32)
        end
        m = logits_buf[1]
        @inbounds for e = 2:n_experts
            v = logits_buf[e]
            v > m && (m = v)
        end
        s = 0.0f0
        @inbounds for e = 1:n_experts
            logits_buf[e] = exp(logits_buf[e] - m)
            s += logits_buf[e]
        end
        @inbounds for e = 1:n_experts
            logits_buf[e] /= s
        end

        idx = partialsortperm(logits_buf, 1:k; rev = true)
        @inbounds for j = 1:k
            topk[b, j] = Int32(idx[j])
        end

        h = 0.0f0
        @inbounds for e = 1:n_experts
            p = logits_buf[e]
            h -= p * log(max(p, 1.0f-12))
        end
        entropy[b] = Float32(h)
        confidence[b] = Float32(logits_buf[idx[1]])
    end

    _ = backend  # backend is accepted for API symmetry; the batched math is CPU.
    return (
        topk_by_block = topk,
        entropy_by_block = entropy,
        confidence_by_block = confidence,
    )
end
