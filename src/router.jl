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
