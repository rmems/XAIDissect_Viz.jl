using Random

const _W_CACHE = Dict{Tuple{Int,Int,Int,Int}, Matrix{Float32}}()

function router_logits(::CPUBackend, h::AbstractVector, W::AbstractMatrix)
    h32 = Float32.(h)
    W32 = Float32.(W)
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

function topk_experts(probs::AbstractVector, k::Integer=2)
    return partialsortperm(probs, 1:k; rev=true)
end

# --- Router simulation for atmosphere timeline ---

# Local RNGs only — never mutate Julia's global RNG. The UI seed parameter
# fully determines the (synthetic, viz-only) router weights and hidden state.
function simulate_router_frame(bundle::XAIReportBundle, block::Int, token_idx::Int;
                               backend::ComputeBackend = CPUBackend(),
                               seed::Integer = 42)::RouterFrame
    haskey(bundle.metadata, "d_model")  || throw(ArgumentError("bundle.metadata missing \"d_model\""))
    haskey(bundle.metadata, "n_experts") || throw(ArgumentError("bundle.metadata missing \"n_experts\""))
    d_model = bundle.metadata["d_model"]::Int
    n_experts = bundle.metadata["n_experts"]::Int

    key = (Int(seed), block, d_model, n_experts)
    W = get!(_W_CACHE, key) do
        rng_W = Xoshiro(hash((seed, "W", block, d_model, n_experts)))
        randn(rng_W, Float32, d_model, n_experts) .* 0.018f0
    end

    rng_h = Xoshiro(hash((seed, "h", block, token_idx)))
    phase = 2π * (token_idx % 50) / 50
    h = randn(rng_h, Float32, d_model) .* 0.08f0 .+ sin(phase) * 0.25f0 .+ cos(phase*1.7) * 0.12f0

    logits = router_logits(backend, h, W)
    probs = router_probs(logits)
    topk = topk_experts(probs, 2)

    entropy = -sum(@. probs * log(max(probs, 1f-12)))

    rng_a = Xoshiro(hash((seed, "a", block, token_idx)))
    activity = fill(0.08f0, n_experts)
    activity[topk] .+= 0.65f0
    activity .+= 0.03f0 .* rand(rng_a, Float32, n_experts)
    clamp!(activity, 0f0, 1f0)

    RouterFrame(block, token_idx, logits, probs, topk, entropy, activity)
end

# Small helper for live activity decay/boost during play loop (used by viz)
function update_expert_activity!(activity::Vector{Float32}, topk::Vector{Int};
                                 decay::Float32=0.92f0, boost::Float32=0.55f0)
    activity .*= decay
    for e in topk
        activity[e] = min(1f0, activity[e] + boost)
    end
    clamp!(activity, 0f0, 1f0)
    return activity
end

# Optional tiny CUDA activity kernel (practical for future larger expert sets)
function update_expert_activity!(activity::Vector{Float32}, topk::Vector{Int}, ::CUDABackend;
                                 decay::Float32=0.92f0, boost::Float32=0.55f0)
    if has_cuda()
        @eval using CUDA
        # For 8 experts the CPU version is fine; this shows the pattern
        # A real kernel would be launched with @cuda for large n_experts
        activity .*= decay
        for e in topk
            activity[e] = min(1f0, activity[e] + boost)
        end
        clamp!(activity, 0f0, 1f0)
    else
        @warn "CUDABackend requested but CUDA unavailable; using CPU activity update"
        update_expert_activity!(activity, topk; decay=decay, boost=boost)
    end
    return activity
end