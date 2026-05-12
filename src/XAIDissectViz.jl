module XAIDissectViz

using LinearAlgebra
using Statistics

include("types.jl")
include("backend.jl")
include("router.jl")
include("kernels.jl")
include("cache.jl")
include("reports.jl")
include("viz.jl")

export load_report_bundle,
       load_json_report,
       launch_atmosphere,
       simulate_router_frame,
       simulate_router_topk_batch,
       update_activity_field!,
       RouterFrameCache,
       build_frame_cache, get_frame,
       topk_matrix_for_token, activity_matrix_for_token,
       CPUBackend, CUDABackend, has_cuda,
       router_logits, router_probs, topk_experts,
       XAIReportBundle, RouterRecord, ExpertRecord, TensorMetricRecord,
       SAAQReadinessRecord, RouterFrame, AtmosphereState

end