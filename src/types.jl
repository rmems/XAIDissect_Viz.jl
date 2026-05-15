# Data model types for XAIDissectViz
# Typed view of xai-dissect JSON reports (inventory, routing, stats, saaq, experts)
# plus router-simulation state used by the interactive visualization.

struct RouterRecord
    block::Int
    slot::Int
    shape::String
    orientation::String
    experts::Int
    kind::String
    structural_name::String
end

struct ExpertRecord
    block::Int
    expert_id::Int
    name::String
    params::Int
end

struct TensorMetricRecord
    name::String
    block::Int
    shape::String
    norm::Float32
    mass::Float32
end

struct SAAQReadinessRecord
    block::Int
    risk_score::Float32
    readiness::Float32
    status::String
    details::String
end

struct XAIReportBundle{V,D<:AbstractDict{String,V}}
    metadata::D
    routers::Vector{RouterRecord}
    experts::Vector{ExpertRecord}
    tensor_metrics::Vector{TensorMetricRecord}
    saaq::Vector{SAAQReadinessRecord}
    provenance::String  # always "real" once load_report_bundle succeeds
end

struct RouterFrame
    block::Int
    token_idx::Int
    logits::Vector{Float32}
    probs::Vector{Float32}
    topk::Vector{Int}
    entropy::Float32
    expert_activity::Vector{Float32}
end

# Observable-driven state for the interactive atmosphere visualization
mutable struct AtmosphereState{OInt,OMat,OBool,OFrame}
    selected_block::OInt    # Observable{Int}
    token_idx::OInt         # Observable{Int}
    activity::OMat          # Observable{Matrix{Float32}}
    is_playing::OBool       # Observable{Bool}
    seed::OInt              # Observable{Int}
    frame::OFrame           # Observable{RouterFrame}
end
