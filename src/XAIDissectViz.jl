module XAIDissectViz

using LinearAlgebra
using Statistics
using Random

include("types.jl")
include("backend.jl")
include("router.jl")
include("reports.jl")
include("viz.jl")

export load_report_bundle,
       launch_atmosphere,
       simulate_router_frame,
       CPUBackend, CUDABackend, has_cuda,
       router_logits, router_probs, topk_experts,
       XAIReportBundle, RouterRecord, ExpertRecord, SAAQReadinessRecord,
       RouterFrame, AtmosphereState

end
