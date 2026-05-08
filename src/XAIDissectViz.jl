module XAIDissectViz

using LinearAlgebra
using Statistics

include("types.jl")
include("backend.jl")
include("router.jl")
include("reports.jl")
include("viz.jl")

export load_report_bundle,
       load_json_report,
       launch_atmosphere,
       simulate_router_frame,
       CPUBackend, CUDABackend, has_cuda,
       router_logits, router_probs, topk_experts,
       XAIReportBundle, RouterRecord, ExpertRecord, TensorMetricRecord,
       SAAQReadinessRecord, RouterFrame, AtmosphereState

end

@testset "JSON Report Loading" begin
    mktempdir() do tmpdir
        write(io,  """{"block": 1, "slot": 11, "shape":"(6144, 8)", "d_model": 6144, "experts":8, "blocks":64, "model_family": "Grok-1", "checkpoint": "grok-1-official/checkpoint-1000000", "shard_count": 1, "schema_version": 1}""")
        close(io)

        obj = load_json_report(path)
        @test obj[:block] == 1
        @test obj[:slot] == 11
        @test obj[:shape] == "(6144, 8)"
        @test obj[:d_model] == 6144
        @test obj[:experts] == 8
        @test obj[:blocks] == 64
        @test obj[:model_family] == "Grok-1"
        @test obj[:checkpoint] == "grok-1-official/checkpoint-1000000"
        @test obj[:shard_count] == 1
        @test obj[:schema_version] == 1
    end

    @test_throws ArgumentError load_json_report("non_existent_file.json")
end