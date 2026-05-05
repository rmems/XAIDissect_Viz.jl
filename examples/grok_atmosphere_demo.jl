using XAIDissectViz

# Demo script for Grok-1 MoE Atmosphere visualization
# Default: fully synthetic in-memory bundle (no JSON required)
# To use real xai-dissect reports: pass a directory containing the 5 JSON files
#   bundle = load_report_bundle("/path/to/grok1_run2_export")

bundle = load_report_bundle()   # synthetic 64 blocks / 8 experts

backend = has_cuda() ? CUDABackend() : CPUBackend()
println("Using backend: $(typeof(backend))")

launch_atmosphere(bundle; backend = backend)
