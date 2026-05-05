using Test
using XAIDissectViz

@testset "Router core (CPU)" begin
    h = ones(Float32, 4)
    W = ones(Float32, 4, 2)
    logits = router_logits(CPUBackend(), h, W)
    probs = router_probs(logits)
    @test length(logits) == 2
    @test isapprox(sum(probs), 1.0f0; atol=1f-5)
    @test topk_experts(probs, 1)[1] in 1:2
end

@testset "Synthetic bundle & load_report_bundle" begin
    bundle = load_report_bundle()  # empty path → synthetic
    @test bundle.provenance == "synthetic"
    @test length(bundle.routers) == 64
    @test length(bundle.experts) == 64 * 8
    @test haskey(bundle.metadata, "d_model")
    @test bundle.metadata["n_blocks"] == 64

    # Bad path still returns valid synthetic (no crash)
    bad = load_report_bundle("/tmp/does_not_exist_$(rand())")
    @test bad.provenance == "synthetic"
end

@testset "simulate_router_frame" begin
    bundle = load_report_bundle()
    frame = simulate_router_frame(bundle, 7, 42; backend=CPUBackend())
    @test frame.block == 7
    @test frame.token_idx == 42
    @test length(frame.logits) == 8
    @test length(frame.probs) == 8
    @test length(frame.topk) == 2
    @test 0 <= frame.entropy
    @test all(0 .<= frame.expert_activity .<= 1)
end

@testset "CUDA path (if available)" begin
    if has_cuda()
        @test has_cuda() == true
        bundle = load_report_bundle()
        frame = simulate_router_frame(bundle, 3, 9; backend=CUDABackend())
        @test length(frame.topk) == 2
    else
        @test has_cuda() == false
        @info "CUDA not functional on this runner — skipping CUDA-specific tests"
    end
end

@testset "Public API exports" begin
    @test isdefined(XAIDissectViz, :load_report_bundle)
    @test isdefined(XAIDissectViz, :launch_atmosphere)
    @test isdefined(XAIDissectViz, :simulate_router_frame)
    @test isdefined(XAIDissectViz, :XAIReportBundle)
    @test isdefined(XAIDissectViz, :RouterFrame)
end