# RM-1398 audit helpers — dual-simulator agreement, SAAQ parsing, headless load checks.
#
# Read-only research script; not part of the package test suite.
#
# Run:
#   XAIVIZ_CUDA_AVAILABLE=false julia --project=. examples/rm1398_fidelity_experiment.jl

using XAIDissectViz
using JSON3

function main()
    meta = Dict{String, Any}(
        "d_model" => 6144,
        "n_experts" => 8,
        "n_blocks" => 64,
        "top_k" => 2,
    )
    bundle = XAIReportBundle(
        meta,
        RouterRecord[],
        ExpertRecord[],
        TensorMetricRecord[],
        SAAQReadinessRecord[],
        "real",
    )

    println("=== Dual-simulator top-k agreement at seed=42 ===")
    n_mismatch = 0
    n_total = 0
    entropy_maxdiff = 0.0
    for token = 0:20
        batch = simulate_router_topk_batch(bundle, token; seed = 42, top_k = 2)
        for b = 1:64
            frame = simulate_router_frame(bundle, b, token; seed = 42)
            n_total += 1
            bt = Tuple(Int.(batch.topk_by_block[b, :]))
            ft = Tuple(Int.(frame.topk))
            if bt != ft
                n_mismatch += 1
            end
            entropy_maxdiff = max(
                entropy_maxdiff,
                abs(Float64(batch.entropy_by_block[b]) - Float64(frame.entropy)),
            )
        end
    end
    println(
        "tokens 0:20 x 64 blocks: mismatch=",
        n_mismatch,
        "/",
        n_total,
        " (",
        round(100 * n_mismatch / n_total; digits = 1),
        "%)",
    )
    println("max |entropy_batch - entropy_frame| = ", entropy_maxdiff)

    println("examples (block, token, cache_topk, frame_topk):")
    for token in 0:5, b in (1, 7, 32, 64)
        batch = simulate_router_topk_batch(bundle, token; seed = 42, top_k = 2)
        frame = simulate_router_frame(bundle, b, token; seed = 42)
        println(
            "  b=",
            b,
            " t=",
            token,
            " cache=",
            collect(batch.topk_by_block[b, :]),
            " frame=",
            frame.topk,
        )
    end

    println()
    println("=== Cache seed 42 vs inspector seed 99 at t=0 block=1 ===")
    c = build_frame_cache(bundle; n_tokens = 5, seed = 42)
    f42 = simulate_router_frame(bundle, 1, 0; seed = 42)
    f99 = simulate_router_frame(bundle, 1, 0; seed = 99)
    g = get_frame(c, 1, 0)
    println(
        "cache topk=",
        collect(g.topk),
        " frame seed42=",
        f42.topk,
        " frame seed99=",
        f99.topk,
    )

    println()
    println("=== parse_saaq null block_index ===")
    j = JSON3.read(
        raw"""{"layer_readiness":[{"block_index":null,"max_risk_score":0.9,"mean_readiness_score":0.1,"routing_critical":true,"label":"unassigned"},{"block_index":0,"max_risk_score":0.4,"mean_readiness_score":0.8,"routing_critical":false}]}""",
    )
    rows = XAIDissectViz.parse_saaq_readiness(j)
    for r in rows
        println(
            "  block=",
            r.block,
            " risk=",
            r.risk_score,
            " ready=",
            r.readiness,
            " status=",
            r.status,
        )
    end

    println()
    println("=== parse_inventory missing inferred ===")
    j2 = JSON3.read(raw"""{"model_family":"grok"}""")
    m = XAIDissectViz.parse_inventory_metadata(j2)
    println("  ", m)

    println()
    println("=== Headless load: GLMakie/CUDA loaded? ===")
    gl = Base.PkgId(Base.UUID("e9467ef8-e4e7-5192-8a1a-b1aee30e663a"), "GLMakie")
    cu = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
    println("GLMakie loaded=", haskey(Base.loaded_modules, gl))
    println("CUDA loaded=", haskey(Base.loaded_modules, cu))
    println("cuda_available=", cuda_available())

    println()
    println("=== Reconstruct cost at t=5 (64x8) ===")
    t0 = time_ns()
    A = activity_matrix_for_token(c, 5)
    t1 = time_ns()
    println("activity_matrix_for_token t=5: ", (t1 - t0) / 1e6, " ms  mean=", sum(A) / length(A))

    println("EXPERIMENT_OK")
end

main()
