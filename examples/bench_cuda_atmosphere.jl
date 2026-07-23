# CUDA atmosphere engine micro-benchmark.
#
# Measures the cost of one `update_activity_field!` step across the
# n_blocks × n_experts activity field. Runs the CPU reference, and — when
# `CUDA.functional()` — the CUDA dispatch with a warmup + `CUDA.@sync`.
#
# This benchmark intentionally does NOT load model weights and does NOT run
# inference. It only exercises the viz-side activity field kernels.
#
# Run with:
#   julia --project examples/bench_cuda_atmosphere.jl
#
# If `XAI_DISSECT_REPORTS` is set to a real xai-dissect run directory, the
# bundle is loaded from there. Otherwise a hand-built minimal bundle is used
# so the benchmark stays runnable on a clean checkout.

using XAIDissectViz
using Random
using Printf

# CUDA is a hard dep of the package; loading it here is benign on
# systems without a working driver — `CUDA.functional()` returns false in
# that case and we'll fall back to the CPU-only report.
#
# We import (not `using`) CUDA to avoid the `CUDABackend` name collision
# between XAIDissectViz and CUDA.jl's KernelAbstractions glue.
import CUDA

const N_BLOCKS = 64
const N_EXPERTS = 8
const TOP_K = 2
const N_ITERS = 1_000
const WARMUP = 50

function _build_bundle()
    reports_dir = get(ENV, "XAI_DISSECT_REPORTS", "")
    if !isempty(reports_dir) && isdir(reports_dir)
        try
            return load_report_bundle(reports_dir)
        catch err
            @warn "Failed to load reports from XAI_DISSECT_REPORTS; using minimal bundle" error =
                err
        end
    end
    meta = Dict{String, Any}(
        "d_model" => 6144,
        "n_experts" => N_EXPERTS,
        "n_blocks" => N_BLOCKS,
        "top_k" => TOP_K,
    )
    return XAIReportBundle(
        meta,
        RouterRecord[],
        ExpertRecord[],
        TensorMetricRecord[],
        SAAQReadinessRecord[],
        "real",
    )
end

function _bench_cpu(activity, topk; iters::Int = N_ITERS, warmup::Int = WARMUP)
    for _ = 1:warmup
        update_activity_field!(CPUBackend(), activity, topk)
    end
    t0 = time_ns()
    for _ = 1:iters
        update_activity_field!(CPUBackend(), activity, topk)
    end
    t1 = time_ns()
    return (t1 - t0) / 1e9
end

function _bench_cuda(activity_gpu, topk_gpu; iters::Int = N_ITERS, warmup::Int = WARMUP)
    for _ = 1:warmup
        update_activity_field!(CUDABackend(), activity_gpu, topk_gpu)
    end
    CUDA.synchronize()
    t0 = time_ns()
    for _ = 1:iters
        update_activity_field!(CUDABackend(), activity_gpu, topk_gpu)
    end
    CUDA.synchronize()
    t1 = time_ns()
    return (t1 - t0) / 1e9
end

function main()
    println("XAIDissectViz CUDA atmosphere micro-benchmark")
    println("  n_blocks=$N_BLOCKS  n_experts=$N_EXPERTS  top_k=$TOP_K")
    println("  iters=$N_ITERS  warmup=$WARMUP")
    println()

    bundle = _build_bundle()
    @assert bundle.metadata["n_blocks"] == N_BLOCKS
    @assert bundle.metadata["n_experts"] == N_EXPERTS

    rng = Xoshiro(2026)
    A = rand(rng, Float32, N_BLOCKS, N_EXPERTS) .* 0.3f0
    T = rand(rng, Int32(1):Int32(N_EXPERTS), N_BLOCKS, TOP_K)

    println("=== CPU update_activity_field! ===")
    cpu_secs = _bench_cpu(copy(A), T)
    cpu_per_iter_us = (cpu_secs / N_ITERS) * 1e6
    @printf("CPU: %.3f s total, %.2f µs/iter\n", cpu_secs, cpu_per_iter_us)

    if CUDA.functional()
        println()
        println("=== CUDA update_activity_field! ===")
        A_gpu = CUDA.CuArray(copy(A))
        T_gpu = CUDA.CuArray(T)
        cuda_secs = _bench_cuda(A_gpu, T_gpu)
        cuda_per_iter_us = (cuda_secs / N_ITERS) * 1e6
        @printf("CUDA: %.3f s total, %.2f µs/iter\n", cuda_secs, cuda_per_iter_us)
        speedup = cpu_secs / cuda_secs
        @printf("Speedup (CPU / CUDA) = %.3fx\n", speedup)

        # Verify CUDA result matches CPU reference within tolerance after a
        # single fresh step from the same starting state.
        A_ref = copy(A)
        update_activity_field!(CPUBackend(), A_ref, T)
        A_test = CUDA.CuArray(copy(A))
        update_activity_field!(CUDABackend(), A_test, CUDA.CuArray(T))
        max_diff = maximum(abs.(Array(A_test) .- A_ref))
        @printf(
            "Single-step CPU vs CUDA max abs diff = %.3e (atol target 1e-5)\n",
            max_diff
        )
    else
        println()
        println("CUDA.functional() == false — skipping CUDA benchmark.")
    end
    println()
    println("done.")
end

main()
