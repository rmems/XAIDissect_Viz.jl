using Test
using XAIDissectViz

# Minimal real-shaped bundle built by hand from struct constructors.
# No JSON, no random data. Used for tests that need a bundle but should
# not depend on a real xai-dissect report directory being present.
function _minimal_bundle()
    meta = Dict{String,Any}(
        "d_model" => 6144,
        "n_experts" => 8,
        "n_blocks" => 64,
        "top_k" => 2,
    )
    return XAIReportBundle(
        meta,
        RouterRecord[],
        ExpertRecord[],
        TensorMetricRecord[],
        SAAQReadinessRecord[],
        "real",
    )
end

@testset "Router core (CPU)" begin
    h = ones(Float32, 4)
    W = ones(Float32, 4, 2)
    logits = router_logits(CPUBackend(), h, W)
    probs = router_probs(logits)
    @test length(logits) == 2
    @test isapprox(sum(probs), 1.0f0; atol=1f-5)
    @test topk_experts(probs, 1)[1] in 1:2
end

@testset "load_report_bundle: strict errors" begin
    @test_throws ArgumentError load_report_bundle("")
    @test_throws ArgumentError load_report_bundle("/tmp/xai_dissect_does_not_exist_$(rand(UInt64))")
end

@testset "load_report_bundle: real reports (gated)" begin
    reports_dir = get(ENV, "XAI_DISSECT_REPORTS", "")
    if !isempty(reports_dir) && isdir(reports_dir)
        bundle = load_report_bundle(reports_dir)
        @test bundle.provenance == "real"
        @test bundle.metadata["d_model"] == 6144
        @test bundle.metadata["n_experts"] == 8
        @test bundle.metadata["n_blocks"] == 64
        @test length(bundle.routers) >= 1
        @test length(bundle.saaq) >= 1
    else
        @info "XAI_DISSECT_REPORTS not set or invalid; skipping real-load test"
        @test true
    end
end

@testset "simulate_router_frame" begin
    bundle = _minimal_bundle()
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
        bundle = _minimal_bundle()
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

@testset "load_json_report" begin
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "tiny.json")
        write(path, """{"block": 1, "slot": 11, "shape": "6144x8", "d_model": 6144}""")
        obj = XAIDissectViz.load_json_report(path)
        @test obj[:block] == 1
        @test obj[:slot] == 11
        @test obj[:shape] == "6144x8"
        @test obj[:d_model] == 6144
    end
    @test_throws SystemError XAIDissectViz.load_json_report("/tmp/xaiviz_does_not_exist_$(rand(UInt64)).json")
end
