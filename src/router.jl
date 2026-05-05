function router_logits(::CPUBackend, h::AbstractVector, W::AbstractMatrix)
    return vec(transpose(h) * W)
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

function simulate_router_frame(bundle::XAIReportBundle, block::Int, token_idx::Int;
                               backend::ComputeBackend = CPUBackend())::RouterFrame
    d_model = get(bundle.metadata, "d_model", 6144)::Int
    n_experts = get(bundle.metadata, "n_experts", 8)::Int

    # Reproducible per-block router weights (never loads real weights)
    Random.seed!(hash(("W", block, d_model, n_experts)))
    W = randn(Float32, d_model, n_experts) .* 0.018f0

    # Token-dependent hidden state (synthetic, visual only)
    Random.seed!(hash(("h", token_idx, block)))
    phase = 2π * (token_idx % 50) / 50
    h = randn(Float32, d_model) .* 0.08f0 .+ sin(phase) * 0.25f0 .+ cos(phase*1.7) * 0.12f0

    logits = router_logits(backend, h, W)
    probs = router_probs(logits)
    topk = topk_experts(probs, 2)

    entropy = -sum(@. probs * log(max(probs, 1f-12)))

    # Activity vector for heatmap glow (boost top-2, mild noise)
    activity = fill(0.08f0, n_experts)
    activity[topk] .+= 0.65f0
    activity .+= 0.03f0 .* rand(Float32, n_experts)
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