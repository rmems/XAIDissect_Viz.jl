using JSON3

# Real xai-dissect JSON loader.
# Parses the 5 standard report files into typed structs:
#   inventory.json, routing-report.json, stats.json, saaq-readiness.json, experts.json
# No synthetic, in-memory, or random fallback. Missing files raise ArgumentError.

const _REQUIRED_REPORT_FILES = (
    "routing-report.json",
    "inventory.json",
    "stats.json",
    "saaq-readiness.json",
    "experts.json",
)

function load_json_report(path::AbstractString)
    isfile(path) || throw(ArgumentError("JSON report not found: $path"))
    return open(path, "r") do io
        JSON3.read(io)
    end
end

# Resolve a user-supplied path to the directory that actually contains the 5 JSONs.
# Accepts either:
#   - a directory containing the 5 files directly, or
#   - a "run root" with exactly one `exports/<ckpt_label>/` underneath that
#     contains the required files.
# If the run root has multiple valid checkpoint report directories, raise an
# ArgumentError instead of silently picking the first one — the caller must
# pass the exact directory.
function _resolve_reports_dir(path::AbstractString)::String
    if all(isfile(joinpath(path, f)) for f in _REQUIRED_REPORT_FILES)
        return abspath(path)
    end
    exports_root = joinpath(path, "exports")
    if isdir(exports_root)
        candidates = String[]
        for entry in readdir(exports_root; join = true)
            if isdir(entry) && all(isfile(joinpath(entry, f)) for f in _REQUIRED_REPORT_FILES)
                push!(candidates, abspath(entry))
            end
        end
        if length(candidates) > 1
            throw(ArgumentError(
                "Multiple xai-dissect report directories found; " *
                "pass the exact checkpoint report directory"))
        elseif length(candidates) == 1
            return candidates[1]
        end
    end
    return abspath(path)
end

_as_int(x) = x === nothing ? 0 : Int(x)
_as_int_block(x) = x === nothing ? 0 : Int(x) + 1  # JSON is 0-based; Julia is 1-based
_as_f32(x) = x === nothing ? 0f0 : Float32(x)
_as_str(x) = x === nothing ? "" : String(x)
_shape_str(s) = s === nothing ? "" : s isa AbstractString ? String(s) : join(string.(s), "x")

function parse_inventory_metadata(json)::Dict{String,Any}
    inferred = get(json, :inferred, nothing)
    metadata = Dict{String,Any}(
        "model_family"   => _as_str(get(json, :model_family, "")),
        "checkpoint"     => _as_str(get(json, :checkpoint_path, "")),
        "shard_count"    => _as_int(get(json, :shard_count, 0)),
        "schema_version" => _as_int(get(json, :schema_version, 0)),
    )
    if inferred !== nothing
        metadata["d_model"]    = _as_int(get(inferred, :d_model, 6144))
        metadata["n_experts"]  = _as_int(get(inferred, :n_experts, 8))
        metadata["n_blocks"]   = _as_int(get(inferred, :n_blocks, 64))
        metadata["vocab_size"] = _as_int(get(inferred, :vocab_size, 0))
        metadata["d_ff"]       = _as_int(get(inferred, :d_ff, 0))
    else
        metadata["d_model"]   = 6144
        metadata["n_experts"] = 8
        metadata["n_blocks"]  = 64
    end
    metadata["top_k"] = 2
    return metadata
end

function parse_routing_report(json)::Vector{RouterRecord}
    candidates = get(json, :candidate_tensors, [])
    out = RouterRecord[]
    sizehint!(out, length(candidates))
    for c in candidates
        push!(out, RouterRecord(
            _as_int_block(get(c, :block_index, nothing)),
            _as_int(get(c, :block_slot, 0)),
            _shape_str(get(c, :shape, nothing)),
            _as_str(get(c, :orientation, "")),
            _as_int(get(c, :linked_expert_count, 0)),
            _as_str(get(c, :kind_label, "")),
            _as_str(get(c, :structural_name, "")),
        ))
    end
    return out
end

function parse_experts(json)::Vector{ExpertRecord}
    blocks = get(json, :blocks, [])
    out = ExpertRecord[]
    for blk in blocks
        block_idx = _as_int_block(get(blk, :block_index, nothing))
        for ex in get(blk, :experts, [])
            tensors = get(ex, :tensors, [])
            params = 0
            for t in tensors
                params += _as_int(get(t, :total_elements, 0))
            end
            name = _as_str(get(ex, :family_label, ""))
            if isempty(name) && !isempty(tensors)
                name = _as_str(get(tensors[1], :structural_name, ""))
            end
            push!(out, ExpertRecord(
                block_idx,
                _as_int(get(ex, :expert_index, 0)) + 1,
                name,
                params,
            ))
        end
    end
    return out
end

function parse_stats_tensors(json)::Vector{TensorMetricRecord}
    tensors = get(json, :tensors, [])
    out = TensorMetricRecord[]
    sizehint!(out, length(tensors))
    for t in tensors
        push!(out, TensorMetricRecord(
            _as_str(get(t, :structural_name, "")),
            _as_int_block(get(t, :block_index, nothing)),
            _shape_str(get(t, :shape, nothing)),
            _as_f32(get(t, :l2_norm, 0)),
            _as_f32(get(t, :rms, 0)),
        ))
    end
    return out
end

function parse_saaq_readiness(json)::Vector{SAAQReadinessRecord}
    layers = get(json, :layer_readiness, [])
    out = SAAQReadinessRecord[]
    sizehint!(out, length(layers))
    for l in layers
        critical = get(l, :routing_critical, false)
        push!(out, SAAQReadinessRecord(
            _as_int_block(get(l, :block_index, nothing)),
            _as_f32(get(l, :max_risk_score, 0)),
            _as_f32(get(l, :mean_readiness_score, 0)),
            critical === true ? "critical" : "ok",
            _as_str(get(l, :label, "")),
        ))
    end
    return out
end

function load_report_bundle(path::AbstractString)::XAIReportBundle
    isempty(path) && throw(ArgumentError(
        "load_report_bundle: path is required (no synthetic fallback)"))
    isdir(path) || throw(ArgumentError(
        "load_report_bundle: not a directory: $path"))

    dir = _resolve_reports_dir(path)
    missing_files = [f for f in _REQUIRED_REPORT_FILES if !isfile(joinpath(dir, f))]
    isempty(missing_files) || throw(ArgumentError(
        "Missing required xai-dissect JSON file(s) in $dir: $(join(missing_files, ", "))"))

    inv  = load_json_report(joinpath(dir, "inventory.json"))
    rr   = load_json_report(joinpath(dir, "routing-report.json"))
    st   = load_json_report(joinpath(dir, "stats.json"))
    saaq = load_json_report(joinpath(dir, "saaq-readiness.json"))
    exps = load_json_report(joinpath(dir, "experts.json"))

    metadata = parse_inventory_metadata(inv)
    metadata["reports_dir"] = dir

    return XAIReportBundle(
        metadata,
        parse_routing_report(rr),
        parse_experts(exps),
        parse_stats_tensors(st),
        parse_saaq_readiness(saaq),
        "real",
    )
end
