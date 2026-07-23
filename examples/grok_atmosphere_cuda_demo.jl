# CUDA-accelerated Grok-1 MoE Atmosphere demo.
#
# Loads a real xai-dissect report bundle and launches the interactive
# atmosphere viewer with the CUDA backend (when functional). The activity
# field heatmap is updated by `update_activity_field!` on the CUDA backend
# while the inspector still reads cached top-k/entropy/confidence and
# computes per-(block, token) logits/probs on demand via
# `simulate_router_frame`.
#
# Requires `XAI_DISSECT_REPORTS` to be set to either:
#   - a directory containing the 5 xai-dissect JSONs directly, or
#   - a "run root" with `exports/<ckpt_label>/*.json` underneath.
#
# Example:
#   ENV["XAI_DISSECT_REPORTS"] = "/path/to/grok1_run2_after_fixes_*"
#   include("examples/grok_atmosphere_cuda_demo.jl")
#
# This script never loads or redistributes Grok-1 model weights — it only
# visualizes the metadata the xai-dissect tool already extracted.

using XAIDissectViz

reports_dir = get(ENV, "XAI_DISSECT_REPORTS", "")
isempty(reports_dir) && error(
    "Set XAI_DISSECT_REPORTS to your xai-dissect run directory before running " *
    "this demo (no synthetic fallback in the demo path).",
)

bundle = load_report_bundle(reports_dir)
println(
    "Loaded bundle: provenance=$(bundle.provenance), " *
    "routers=$(length(bundle.routers)), experts=$(length(bundle.experts))",
)

if has_cuda()
    println("CUDA functional — launching with CUDABackend()")
    backend = CUDABackend()
else
    @warn "CUDA not functional on this host — falling back to CPUBackend()"
    backend = CPUBackend()
end

launch_atmosphere(bundle; backend = backend)
