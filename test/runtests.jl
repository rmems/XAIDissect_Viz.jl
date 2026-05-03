using Test
using XAIDissectViz

h = ones(Float32, 4)
W = ones(Float32, 4, 2)

logits = router_logits(CPUBackend(), h, W)
probs = router_probs(logits)

@test length(logits) == 2
@test isapprox(sum(probs), 1.0f0; atol=1f-5)
@test topk_experts(probs, 1)[1] in 1:2
