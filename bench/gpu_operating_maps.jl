using CUDA
using Dates
using DynamicsKit
using JLD2
using Printf
using Statistics

function parse_resolutions(value::AbstractString)
    resolutions = parse.(Int, strip.(split(value, ',')))
    isempty(resolutions) && throw(ArgumentError("GPU_BENCH_RESOLUTIONS must contain at least one integer"))
    all(>(0), resolutions) || throw(ArgumentError("GPU_BENCH_RESOLUTIONS must contain positive integers"))
    return resolutions
end

function approx_equal_with_nans(lhs, rhs)
    axes(lhs) == axes(rhs) || return false
    for index in eachindex(lhs, rhs)
        isequal(lhs[index], rhs[index]) && continue
        isapprox(lhs[index], rhs[index]) || return false
    end
    return true
end

periodicity_parity(cpu_result, gpu_result) = cpu_result.periodicity == gpu_result.periodicity

function lyapunov_parity(cpu_result, gpu_result)
    return approx_equal_with_nans(cpu_result.exponents, gpu_result.exponents) &&
        cpu_result.classification_status_codes == gpu_result.classification_status_codes &&
        cpu_result.estimation_status_codes == gpu_result.estimation_status_codes &&
        cpu_result.sample_counts == gpu_result.sample_counts
end

function benchmark_pair(cpu_call, gpu_call, parity_check, repeats::Int)
    cpu_reference = cpu_call()
    gpu_reference = gpu_call()
    CUDA.synchronize()

    parity_check(cpu_reference, gpu_reference) ||
        error("CPU/CUDA result mismatch during benchmark warmup")
    gpu_reference.compute_backend == :cuda ||
        error("CUDA request reported compute_backend=$(gpu_reference.compute_backend)")

    cpu_seconds = Float64[]
    gpu_seconds = Float64[]
    for _ in 1:repeats
        GC.gc()
        push!(cpu_seconds, @elapsed cpu_call())
        GC.gc()
        push!(gpu_seconds, @elapsed begin
            gpu_call()
            CUDA.synchronize()
        end)
    end

    return (
        cpu_minimum=minimum(cpu_seconds),
        cpu_median=median(cpu_seconds),
        gpu_minimum=minimum(gpu_seconds),
        gpu_median=median(gpu_seconds),
    )
end

function benchmark_row(analysis::AbstractString, resolution::Int, timings)
    cells = resolution^2
    return (
        analysis=String(analysis),
        resolution=resolution,
        cells=cells,
        parity_passed=true,
        compute_backend="cuda",
        cpu_minimum_seconds=timings.cpu_minimum,
        cpu_median_seconds=timings.cpu_median,
        gpu_minimum_seconds=timings.gpu_minimum,
        gpu_median_seconds=timings.gpu_median,
        minimum_speedup=timings.cpu_minimum / timings.gpu_minimum,
        median_speedup=timings.cpu_median / timings.gpu_median,
        gpu_median_cells_per_second=cells / timings.gpu_median,
    )
end

function write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "analysis,resolution,cells,parity_passed,compute_backend,cpu_minimum_seconds,cpu_median_seconds,gpu_minimum_seconds,gpu_median_seconds,minimum_speedup,median_speedup,gpu_median_cells_per_second")
        for row in rows
            println(io, join(values(row), ','))
        end
    end
end

function main()
    CUDA.functional() || error("CUDA.jl does not report a functional CUDA device")
    gpu_backend_available(:cuda) || error("DynamicsKit's CUDA extension did not register a functional backend")
    CUDA.allowscalar(false)

    cuda_heap_mb = parse(Int, get(ENV, "GPU_BENCH_CUDA_HEAP_MB", "256"))
    cuda_heap_mb > 0 || throw(ArgumentError("GPU_BENCH_CUDA_HEAP_MB must be positive"))
    cuda_heap_bytes = cuda_heap_mb * 1024^2
    CUDA.limit!(CUDA.LIMIT_MALLOC_HEAP_SIZE, cuda_heap_bytes)

    resolutions = parse_resolutions(get(ENV, "GPU_BENCH_RESOLUTIONS", "64,128,256,512"))
    continuous_resolutions = parse_resolutions(get(ENV, "GPU_BENCH_CONTINUOUS_RESOLUTIONS", "4,8,16"))
    repeats = parse(Int, get(ENV, "GPU_BENCH_REPEATS", "3"))
    continuous_repeats = parse(Int, get(ENV, "GPU_BENCH_CONTINUOUS_REPEATS", "2"))
    iterations = parse(Int, get(ENV, "GPU_BENCH_ITERATIONS", "500"))
    lyapunov_iterations = parse(Int, get(ENV, "GPU_BENCH_LYAPUNOV_ITERATIONS", "500"))
    continuous_iterations = parse(Int, get(ENV, "GPU_BENCH_CONTINUOUS_ITERATIONS", "250"))
    repeats > 0 || throw(ArgumentError("GPU_BENCH_REPEATS must be positive"))
    continuous_repeats > 0 || throw(ArgumentError("GPU_BENCH_CONTINUOUS_REPEATS must be positive"))
    iterations > 0 || throw(ArgumentError("GPU_BENCH_ITERATIONS must be positive"))
    lyapunov_iterations > 0 || throw(ArgumentError("GPU_BENCH_LYAPUNOV_ITERATIONS must be positive"))
    continuous_iterations > 0 || throw(ArgumentError("GPU_BENCH_CONTINUOUS_ITERATIONS must be positive"))

    output_path = abspath(get(ENV, "GPU_BENCH_OUT", joinpath("var", "output", "gpu_operating_maps.jld2")))
    csv_path = replace(output_path, r"\.jld2$" => ".csv")
    csv_path == output_path && (csv_path *= ".csv")

    device_name = string(CUDA.name(CUDA.device()))
    metadata = Dict(
        "generated_at" => string(now()),
        "julia_version" => string(VERSION),
        "dynamicskit_version" => string(pkgversion(DynamicsKit)),
        "cuda_version" => string(pkgversion(CUDA)),
        "device" => device_name,
        "device_capability" => string(CUDA.capability(CUDA.device())),
        "cuda_malloc_heap_bytes" => CUDA.limit(CUDA.LIMIT_MALLOC_HEAP_SIZE),
        "os" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
        "cpu_name" => Sys.CPU_NAME,
        "cpu_threads" => Sys.CPU_THREADS,
        "julia_threads" => Threads.nthreads(),
        "resolutions" => resolutions,
        "continuous_resolutions" => continuous_resolutions,
        "repeats" => repeats,
        "continuous_repeats" => continuous_repeats,
        "iterations" => iterations,
        "lyapunov_iterations" => lyapunov_iterations,
        "continuous_iterations" => continuous_iterations,
        "discrete_fixture" => "Henon parameter map and initial-condition basins",
        "continuous_fixture" => "Memristive diode bridge P1/P3 operating map and coexistence basin",
    )

    println("CUDA operating-map benchmark")
    println("  device: $device_name")
    println("  Julia $(VERSION), CUDA.jl $(pkgversion(CUDA)), $(Threads.nthreads()) Julia threads")
    println("  CUDA device malloc heap=$(cuda_heap_mb) MiB (benchmark process only)")
    println("  discrete resolutions=$(join(resolutions, ',')), repeats=$repeats, iterations=$iterations, lyapunov_iterations=$lyapunov_iterations")
    println("  continuous resolutions=$(join(continuous_resolutions, ',')), repeats=$continuous_repeats, iterations=$continuous_iterations")

    sys = henon_map()
    backend = gpu_backend(:cuda)
    rows = NamedTuple[]

    for resolution in resolutions
        map_config = BifurcationMapConfig(
            a_min=1.0, a_max=1.4, a_steps=resolution - 1,
            b_min=0.2, b_max=0.35, b_steps=resolution - 1,
            a_index=1, b_index=2, base_params=[1.4, 0.3],
            max_period=8, iterations=iterations, precision=1e-3, divergence_cutoff=1e6,
        )
        map_timings = benchmark_pair(
            () -> bifurcation_map(sys, map_config),
            () -> bifurcation_map(sys, map_config; backend=backend),
            periodicity_parity,
            repeats,
        )
        map_row = benchmark_row("bifurcation_map", resolution, map_timings)
        push!(rows, map_row)
        println(@sprintf(
            "  %-20s %4dx%-4d cpu=%8.4fs gpu=%8.4fs speedup=%7.2fx throughput=%10.0f cells/s",
            map_row.analysis, resolution, resolution, map_row.cpu_median_seconds,
            map_row.gpu_median_seconds, map_row.median_speedup, map_row.gpu_median_cells_per_second,
        ))

        basins_config = BasinsConfig(
            bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
            x_min=-0.4, x_max=0.4, x_steps=resolution - 1,
            y_min=-0.4, y_max=0.4, y_steps=resolution - 1,
            x_index=1, y_index=2, max_period=8, iterations=iterations, precision=1e-3,
        )
        basins_timings = benchmark_pair(
            () -> basins_of_attraction(sys, basins_config),
            () -> basins_of_attraction(sys, basins_config; backend=backend),
            periodicity_parity,
            repeats,
        )
        basins_row = benchmark_row("basins_of_attraction", resolution, basins_timings)
        push!(rows, basins_row)
        println(@sprintf(
            "  %-20s %4dx%-4d cpu=%8.4fs gpu=%8.4fs speedup=%7.2fx throughput=%10.0f cells/s",
            basins_row.analysis, resolution, resolution, basins_row.cpu_median_seconds,
            basins_row.gpu_median_seconds, basins_row.median_speedup, basins_row.gpu_median_cells_per_second,
        ))

        lyapunov_config = BifurcationMapConfig(
            a_min=1.0, a_max=1.4, a_steps=resolution - 1,
            b_min=0.2, b_max=0.35, b_steps=resolution - 1,
            a_index=1, b_index=2, base_params=[1.4, 0.3],
            iterations=iterations, lyapunov_iterations=lyapunov_iterations,
            divergence_cutoff=1e6,
        )
        lyapunov_timings = benchmark_pair(
            () -> lyapunov_field(sys, lyapunov_config),
            () -> lyapunov_field(sys, lyapunov_config; backend=backend),
            lyapunov_parity,
            repeats,
        )
        lyapunov_row = benchmark_row("lyapunov_field", resolution, lyapunov_timings)
        push!(rows, lyapunov_row)
        println(@sprintf(
            "  %-20s %4dx%-4d cpu=%8.4fs gpu=%8.4fs speedup=%7.2fx throughput=%10.0f cells/s",
            lyapunov_row.analysis, resolution, resolution, lyapunov_row.cpu_median_seconds,
            lyapunov_row.gpu_median_seconds, lyapunov_row.median_speedup, lyapunov_row.gpu_median_cells_per_second,
        ))
    end

    continuous_sys = memristive_diode_bridge(c=6.02e-6, k=0.05)
    for resolution in continuous_resolutions
        map_config = BifurcationMapConfig(
            a_min=0.014, a_max=0.017, a_steps=resolution - 1,
            b_min=5.5e-6, b_max=6.5e-6, b_steps=resolution - 1,
            a_index=1, b_index=2, base_params=[0.0155, 6.02e-6, 0.05],
            max_period=4, iterations=continuous_iterations, precision=1e-3,
        )
        map_timings = benchmark_pair(
            () -> bifurcation_map(continuous_sys, map_config),
            () -> bifurcation_map(continuous_sys, map_config; backend=backend),
            periodicity_parity,
            continuous_repeats,
        )
        map_row = benchmark_row("mdb_bifurcation_map", resolution, map_timings)
        push!(rows, map_row)
        println(@sprintf(
            "  %-20s %4dx%-4d cpu=%8.4fs gpu=%8.4fs speedup=%7.2fx throughput=%10.0f cells/s",
            map_row.analysis, resolution, resolution, map_row.cpu_median_seconds,
            map_row.gpu_median_seconds, map_row.median_speedup, map_row.gpu_median_cells_per_second,
        ))

        basins_config = BasinsConfig(
            bif_param=0.0155, param_index=1, fixed_params=[0.0155, 6.02e-6, 0.05],
            x_min=-4.2, x_max=-3.0, x_steps=resolution - 1,
            y_min=0.0, y_max=0.6, y_steps=resolution - 1,
            x_index=1, y_index=3, max_period=6,
            iterations=continuous_iterations, precision=1e-3,
        )
        basins_timings = benchmark_pair(
            () -> basins_of_attraction(continuous_sys, basins_config),
            () -> basins_of_attraction(continuous_sys, basins_config; backend=backend),
            periodicity_parity,
            continuous_repeats,
        )
        basins_row = benchmark_row("mdb_basins_of_attraction", resolution, basins_timings)
        push!(rows, basins_row)
        println(@sprintf(
            "  %-20s %4dx%-4d cpu=%8.4fs gpu=%8.4fs speedup=%7.2fx throughput=%10.0f cells/s",
            basins_row.analysis, resolution, resolution, basins_row.cpu_median_seconds,
            basins_row.gpu_median_seconds, basins_row.median_speedup, basins_row.gpu_median_cells_per_second,
        ))
    end

    mkpath(dirname(output_path))
    jldsave(output_path; metadata, rows)
    write_csv(csv_path, rows)
    println("Wrote $output_path")
    println("Wrote $csv_path")
end

main()