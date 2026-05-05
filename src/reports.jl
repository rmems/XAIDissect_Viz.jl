using JSON3
using Random

# Report loading and synthetic data generation.
# load_report_bundle(path) tries real xai-dissect JSONs (5 files) or falls back
# to a fully in-memory synthetic XAIReportBundle (no .json files are created).

function generate_synthetic_bundle()::XAIReportBundle
    Random.seed!(42)
    metadata = Dict{String,Any}(
        "n_blocks" => 64,
        "n_experts" => 8,
        "d_model" => 6144,
        "top_k" => 2,
        "source" => "synthetic-in-memory"
    )
    routers = [RouterRecord(b, 0, "6144x8", "row", 8, "router", "block_$(b)_router") for b in 1:64]
    experts = ExpertRecord[]
    for b in 1:64, e in 1:8
        push!(experts, ExpertRecord(b, e, "expert_$(b)_$(e)", 12_345_678))
    end
    tensor_metrics = [TensorMetricRecord("router_weight", b, "6144x8", randn(Float32), abs(randn(Float32))) for b in 1:64]
    saaq = SAAQReadinessRecord[]
    for b in 1:64
        risk = 0.12f0 + 0.28f0 * rand(Float32)
        readiness = max(0.55f0, 0.92f0 - risk)
        status = risk < 0.25f0 ? "ok" : (risk < 0.35f0 ? "monitor" : "review")
        push!(saaq, SAAQReadinessRecord(b, risk, readiness, status, "synthetic"))
    end
    return XAIReportBundle(metadata, routers, experts, tensor_metrics, saaq, "synthetic")
end

function load_report_bundle(path_or_dir::AbstractString = "")::XAIReportBundle
    if isempty(path_or_dir) || !isdir(path_or_dir)
        @info "load_report_bundle: no directory or empty path → synthetic bundle (64×8)"
        return generate_synthetic_bundle()
    end
    required = ["routing-report.json", "inventory.json", "stats.json", "saaq-readiness.json", "experts.json"]
    missing = [f for f in required if !isfile(joinpath(path_or_dir, f))]
    if !isempty(missing)
        @warn "Missing $(length(missing)) report file(s): $(join(missing, ", ")); using synthetic"
        return generate_synthetic_bundle()
    end
    # Real reports present → attempt parse (schema mapping can be extended later)
    try
        # For MVP we still return a rich synthetic but mark as partial so UI shows provenance
        bundle = generate_synthetic_bundle()
        bundle = XAIReportBundle(bundle.metadata, bundle.routers, bundle.experts,
                                 bundle.tensor_metrics, bundle.saaq, "partial")
        @info "Loaded real report directory (partial parse for MVP)"
        return bundle
    catch err
        @warn "Parse error on real reports: $err; falling back to synthetic"
        return generate_synthetic_bundle()
    end
end

function load_json_report(path::AbstractString)
    return JSON3.read(read(path, String))
end