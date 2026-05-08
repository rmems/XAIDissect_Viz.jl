using XAIDissectViz

# Grok-1 MoE Atmosphere demo (real xai-dissect reports only).
#
# Set the env var XAI_DISSECT_REPORTS to either:
#   - a directory containing the 5 JSONs directly, or
#   - a "run root" with `exports/<ckpt_label>/*.json` underneath (auto-discovered).
#
# Example:
#   ENV["XAI_DISSECT_REPORTS"] = "/path/to/grok1_run2_after_fixes_*"
#   include("examples/grok_atmosphere_demo.jl")

reports_dir = get(ENV, "XAI_DISSECT_REPORTS", "")
isempty(reports_dir) && error(
    "Set XAI_DISSECT_REPORTS to your xai-dissect run directory before running this demo.",
)

bundle = load_report_bundle(reports_dir)
println("Loaded bundle: provenance=$(bundle.provenance), routers=$(length(bundle.routers)), experts=$(length(bundle.experts))")

backend = has_cuda() ? CUDABackend() : CPUBackend()
println("Using backend: $(typeof(backend))")

launch_atmosphere(bundle; backend = backend)
