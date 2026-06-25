using Test
using JSON3
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

@testset "Router logits CPU: Float64 input gives Float32 logits" begin
    h = ones(Float64, 4)
    W = ones(Float64, 4, 2)
    logits = router_logits(CPUBackend(), h, W)
    @test eltype(logits) === Float32
    @test length(logits) == 2
end

@testset "load_report_bundle: strict errors" begin
    @test_throws ArgumentError load_report_bundle("")
    @test_throws ArgumentError load_report_bundle("/tmp/xai_dissect_does_not_exist_$(rand(UInt64))")
end

@testset "parse_inventory_metadata: JSON null uses semantic defaults" begin
    j = JSON3.read(raw"""{"inferred":{"d_model":null,"n_experts":null,"n_blocks":null,"vocab_size":null,"d_ff":null}}""")
    m = XAIDissectViz.parse_inventory_metadata(j)
    @test m["d_model"] == 6144
    @test m["n_experts"] == 8
    @test m["n_blocks"] == 64
    @test m["vocab_size"] == 0
    @test m["d_ff"] == 0
end

@testset "_resolve_reports_dir: ambiguous run root" begin
    required = ("routing-report.json", "inventory.json", "stats.json",
                "saaq-readiness.json", "experts.json")
    function _make_valid_ckpt_dir(parent, label)
        d = joinpath(parent, label)
        mkpath(d)
        for f in required
            write(joinpath(d, f), "{}")
        end
        return d
    end

    mktempdir() do run_root
        exports = joinpath(run_root, "exports")
        mkpath(exports)
        ckpt_a = _make_valid_ckpt_dir(exports, "grok-1__ckpt-0")
        ckpt_b = _make_valid_ckpt_dir(exports, "grok-1__ckpt-1")

        @test isdir(ckpt_a) && isdir(ckpt_b)

        @test_throws ArgumentError XAIDissectViz._resolve_reports_dir(run_root)
        @test_throws ArgumentError load_report_bundle(run_root)

        # Pointing at one of the checkpoint dirs directly resolves cleanly.
        @test XAIDissectViz._resolve_reports_dir(ckpt_a) == abspath(ckpt_a)
    end

    mktempdir() do run_root
        exports = joinpath(run_root, "exports")
        mkpath(exports)
        ckpt = _make_valid_ckpt_dir(exports, "grok-1__ckpt-0")
        @test XAIDissectViz._resolve_reports_dir(run_root) == abspath(ckpt)
    end
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

@testset "simulate_router_frame: seed honors UI input" begin
    bundle = _minimal_bundle()
    a = simulate_router_frame(bundle, 3, 5; seed=1)
    b = simulate_router_frame(bundle, 3, 5; seed=1)
    c = simulate_router_frame(bundle, 3, 5; seed=2)
    @test a.logits == b.logits          # same seed -> deterministic
    @test a.logits != c.logits          # different seed -> different frame
end

@testset "simulate_router_frame: does not mutate global RNG" begin
    using Random
    bundle = _minimal_bundle()
    Random.seed!(123)
    before = rand(UInt64)
    Random.seed!(123)
    simulate_router_frame(bundle, 1, 0; seed=999)
    after = rand(UInt64)
    @test before == after               # global RNG state untouched
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
    @test isdefined(XAIDissectViz, :cuda_available)
end

@testset "Headless: non-visual API works without GLMakie loaded" begin
    glmakie_id = Base.PkgId(Base.UUID("e9467ef8-e4e7-5192-8a1a-b1aee30e663a"), "GLMakie")
    cuda_id   = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
    @test !haskey(Base.loaded_modules, glmakie_id)
    @test !haskey(Base.loaded_modules, cuda_id)

    bundle = _minimal_bundle()
    frame = simulate_router_frame(bundle, 2, 3)
    @test length(frame.probs) == 8

    h = ones(Float32, 4); W = ones(Float32, 4, 2)
    @test isapprox(sum(router_probs(router_logits(CPUBackend(), h, W))), 1.0f0; atol=1f-5)
    @test topk_experts(Float32[0.1, 0.5, 0.4], 1) == [2]

    @test_throws ArgumentError load_report_bundle("")

    # GLMakie must NOT have been imported (no DISPLAY / OpenGL on CI)
    @test !haskey(Base.loaded_modules, glmakie_id)
    # CUDA must NOT have been imported during CPU-only operations
    @test !haskey(Base.loaded_modules, cuda_id)
end

@testset "cuda_available: soft probe" begin
    # On CI, XAIVIZ_CUDA_AVAILABLE=false should make cuda_available() return
    # false immediately without touching CUDA.jl.
    if get(ENV, "XAIVIZ_CUDA_AVAILABLE", "") == "false"
        @test cuda_available() == false
        cuda_id = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
        @test !haskey(Base.loaded_modules, cuda_id)
    else
        # Outside CI: just verify it returns a bool without erroring
        result = cuda_available()
        @test result isa Bool
        @info "cuda_available() = $result"
    end
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
    missing_path = "/tmp/xaiviz_does_not_exist_$(rand(UInt64)).json"
    @test_throws ArgumentError XAIDissectViz.load_json_report(missing_path)
end

@testset "load_report_bundle: multiple checkpoint exports root" begin
    mktempdir() do root
        required = [
            "routing-report.json",
            "inventory.json",
            "stats.json",
            "saaq-readiness.json",
            "experts.json"]

        @test_throws ArgumentError XAIDissectViz.load_report_bundle(root)
    end
end

# --- CUDA atmosphere engine -------------------------------------------------
#
# CUDA-guarded tests use a local probe rather than `has_cuda()` because the
# probe is robust against Julia 1.12's stricter world-age semantics. CPU-only
# CI never enters the CUDA branches; the CPU-side assertions still run.
function _cuda_functional()
    try
        @eval import CUDA
        return @eval CUDA.functional()
    catch
        return false
    end
end

@testset "update_activity_field! CPU: values stay in [0,1]" begin
    n_blocks, n_experts, top_k = 16, 8, 2
    activity = rand(Float32, n_blocks, n_experts) .* 0.5f0
    topk = rand(Int32(1):Int32(n_experts), n_blocks, top_k)
    for _ in 1:50
        update_activity_field!(CPUBackend(), activity, topk; decay=0.92f0, boost=0.55f0)
        @test all(0f0 .<= activity .<= 1f0)
    end
end

@testset "update_activity_field! CPU: selected top-k experts receive boost" begin
    n_blocks, n_experts, top_k = 8, 8, 2
    activity = zeros(Float32, n_blocks, n_experts)
    topk = repeat(Int32[3 5], n_blocks)  # every block selects experts 3 and 5
    pre = copy(activity)
    update_activity_field!(CPUBackend(), activity, topk; decay=0.92f0, boost=0.55f0)
    for b in 1:n_blocks
        @test activity[b, 3] > pre[b, 3]
        @test activity[b, 5] > pre[b, 5]
    end
end

@testset "update_activity_field! CPU: inactive experts decay" begin
    n_blocks, n_experts, top_k = 8, 8, 2
    activity = fill(0.7f0, n_blocks, n_experts)
    topk = repeat(Int32[3 5], n_blocks)  # only experts 3 and 5 get boosted
    pre = copy(activity)
    update_activity_field!(CPUBackend(), activity, topk; decay=0.92f0, boost=0.0f0)
    for b in 1:n_blocks, e in 1:n_experts
        if e == 3 || e == 5
            continue  # boosted entries explicitly excluded
        end
        @test activity[b, e] < pre[b, e]
    end
end

@testset "update_activity_field! CPU vs CUDA match (gated)" begin
    if _cuda_functional()
        # `import` (not `using`) avoids the `CUDABackend` name collision with
        # XAIDissectViz; we always qualify XAIDissectViz.CUDABackend below.
        @eval import CUDA
        CuArr = @eval CUDA.CuArray
        synchronize_fn = @eval CUDA.synchronize
        host_array_fn = Array
        n_blocks, n_experts, top_k = 64, 8, 2
        rng = Xoshiro(7)
        A = rand(rng, Float32, n_blocks, n_experts) .* 0.3f0
        T = rand(rng, Int32(1):Int32(n_experts), n_blocks, top_k)
        A_cpu = copy(A); A_gpu = CuArr(copy(A)); T_gpu = CuArr(T)
        for _ in 1:5
            update_activity_field!(XAIDissectViz.CPUBackend(), A_cpu, T)
            update_activity_field!(XAIDissectViz.CUDABackend(), A_gpu, T_gpu)
        end
        synchronize_fn()
        @test isapprox(host_array_fn(A_gpu), A_cpu; atol=1f-5)
    else
        @info "CUDA not functional — skipping CPU/CUDA isapprox test"
        @test true
    end
end

@testset "RouterFrameCache: dimensions and exact contracts" begin
    bundle = _minimal_bundle()
    cache = build_frame_cache(bundle; n_tokens=20, seed=42)
    @test cache isa RouterFrameCache
    @test cache.n_blocks == 64
    @test cache.n_experts == 8
    @test cache.top_k == 2
    @test cache.n_tokens == 21  # tokens 0..n_tokens inclusive
    @test size(cache.topk) == (64, 2, 21)
    @test size(cache.entropy) == (64, 21)
    @test size(cache.confidence) == (64, 21)

    M = topk_matrix_for_token(cache, 5)
    @test size(M) == (64, 2)
    @test eltype(M) === Int32
    @test all(1 .<= M .<= 8)

    A = activity_matrix_for_token(cache, 7)
    @test size(A) == (64, 8)
    @test all(0f0 .<= A .<= 1f0)

    f = get_frame(cache, 3, 5)
    @test f.block == 3
    @test f.token_idx == 5
    @test length(f.topk) == 2

    @test_throws BoundsError get_frame(cache, 3, 999)
    @test_throws BoundsError get_frame(cache, 999, 0)
end

@testset "simulate_router_topk_batch: deterministic same seed; different seed differs" begin
    bundle = _minimal_bundle()
    a = simulate_router_topk_batch(bundle, 3; seed=1)
    b = simulate_router_topk_batch(bundle, 3; seed=1)
    c = simulate_router_topk_batch(bundle, 3; seed=2)

    @test a.topk_by_block == b.topk_by_block
    @test a.entropy_by_block == b.entropy_by_block
    @test a.confidence_by_block == b.confidence_by_block

    # different seed must change at least some routing decisions
    @test any(a.topk_by_block .!= c.topk_by_block)
end

@testset "RouterFrameCache: stable top-k for same seed; differs on new seed" begin
    bundle = _minimal_bundle()
    c1 = build_frame_cache(bundle; n_tokens=10, seed=42)
    c2 = build_frame_cache(bundle; n_tokens=10, seed=42)
    c3 = build_frame_cache(bundle; n_tokens=10, seed=43)
    @test c1.topk == c2.topk
    @test any(c1.topk .!= c3.topk)
end

@testset "simulate_router_frame: negative seed does not crash" begin
    bundle = _minimal_bundle()
    frame = simulate_router_frame(bundle, 1, 0; seed=-42)
    @test length(frame.probs) == 8
    frame2 = simulate_router_frame(bundle, 1, 0; seed=-42)
    @test frame.logits == frame2.logits
end

@testset "simulate_router_topk_batch: negative seed does not crash" begin
    bundle = _minimal_bundle()
    r = simulate_router_topk_batch(bundle, 0; seed=-7)
    @test size(r.topk_by_block) == (64, 2)
end

@testset "build_frame_cache: missing metadata keys throw ArgumentError" begin
    bad_meta = Dict{String,Any}("d_model" => 6144)
    bad_bundle = XAIReportBundle(bad_meta, RouterRecord[], ExpertRecord[],
                                 TensorMetricRecord[], SAAQReadinessRecord[], "real")
    @test_throws ArgumentError build_frame_cache(bad_bundle; n_tokens=1)
end

@testset "simulate_router_topk_batch: does not mutate global RNG" begin
    using Random
    bundle = _minimal_bundle()
    Random.seed!(987)
    before = rand(UInt64)
    Random.seed!(987)
    simulate_router_topk_batch(bundle, 0; seed=1234)
    after = rand(UInt64)
    @test before == after
end

@testset "Public API exports: CUDA atmosphere additions" begin
    for sym in (:update_activity_field!, :simulate_router_topk_batch,
                :RouterFrameCache,
                :build_frame_cache, :get_frame,
                :topk_matrix_for_token, :activity_matrix_for_token)
        @test isdefined(XAIDissectViz, sym)
    end
end

@testset "launch_atmosphere remains defined but is not invoked in CI" begin
    # We never invoke launch_atmosphere here — it requires GLMakie + a
    # display / OpenGL context. CI just confirms the symbol is wired up.
    @test isdefined(XAIDissectViz, :launch_atmosphere)
    @test launch_atmosphere isa Function
end

@testset "Headless invariant after kernels/cache: GLMakie unloaded by default" begin
    glmakie_id = Base.PkgId(Base.UUID("e9467ef8-e4e7-5192-8a1a-b1aee30e663a"), "GLMakie")
    @test !haskey(Base.loaded_modules, glmakie_id)

    # Exercise the new APIs from a freshly loaded package state.
    bundle = _minimal_bundle()
    cache = build_frame_cache(bundle; n_tokens=5, seed=1)
    @test size(cache.topk) == (64, 2, 6)

    A = zeros(Float32, 64, 8)
    update_activity_field!(CPUBackend(), A, topk_matrix_for_token(cache, 0))
    @test all(0f0 .<= A .<= 1f0)

    @test !haskey(Base.loaded_modules, glmakie_id)
end
