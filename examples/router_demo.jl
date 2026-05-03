using XAIDissectViz
using Random

Random.seed!(42)

backend = CPUBackend()

h = randn(Float32, 6144)
W = randn(Float32, 6144, 8) .* 0.02f0

logits = router_logits(backend, h, W)
probs = router_probs(logits)
top2 = topk_experts(probs, 2)

println("Router logits:")
println(logits)

println("\nRouter probabilities:")
println(probs)

println("\nTop-2 experts:")
println(top2)
