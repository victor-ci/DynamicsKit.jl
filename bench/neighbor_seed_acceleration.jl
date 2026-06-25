using DynamicsKit
using Dates
using JSON3
using Printf

function _env_int(name::AbstractString, default::Int)
    value = get(ENV, name, nothing)
    isnothing(value) && return default
    return parse(Int, strip(String(value)))
end

function _env_string(name::AbstractString, default::AbstractString)
    return get(ENV, name, String(default))
end

function _env_bool(name::AbstractString, default::Bool=false)
    value = get(ENV, name, nothing)
    isnothing(value) && return default
    return lowercase(strip(String(value))) in ("1", "true", "yes", "on")
end

function _parse_transient_list(raw::AbstractString)
    stripped = strip(raw)
    isempty(stripped) && return Int[]
    return unique(Int[parse(Int, strip(item)) for item in split(stripped, ',') if !isempty(strip(item))])
end

function _benchmark_case(name::AbstractString)
    key = lowercase(strip(String(name)))
    if key in ("ikeda", "ikeda_map")
        sys = ikeda_map()
        return (
            key="ikeda",
            label="Ikeda",
            sys=sys,
            base_params=[0.82, 0.4, 6.0],
            a_index=1,
            b_index=3,
            a_min=0.72,
            a_max=0.92,
            b_min=5.6,
            b_max=6.4,
            max_period=6,
            precision=1e-4,
            iterations=80,
            solver=nothing,
            reltol=nothing,
            abstol=nothing
        )
    elseif key in ("rossler", "rössler", "rossler_oscillator")
        sys = rossler_oscillator()
        base_params = [0.2, 0.2, 4.0]
        return (
            key="rossler",
            label="Rössler",
            sys=sys,
            base_params=base_params,
            a_index=3,
            b_index=1,
            a_min=2.8,
            a_max=4.4,
            b_min=0.16,
            b_max=0.28,
            max_period=4,
            precision=1e-3,
            iterations=26,
            solver=DynamicsKit._workbench_solver(sys, "auto", base_params),
            reltol=1e-7,
            abstol=1e-7
        )
    elseif key in ("memristive", "memristive_diode_bridge", "mdb")
        sys = memristive_diode_bridge()
        base_params = [0.0155, 6.02e-6, 0.05]
        return (
            key="memristive_diode_bridge",
            label="Memristive Diode Bridge",
            sys=sys,
            base_params=base_params,
            a_index=1,
            b_index=3,
            a_min=0.013,
            a_max=0.03,
            b_min=0.035,
            b_max=0.075,
            max_period=6,
            precision=1e-3,
            iterations=80,
            solver=DynamicsKit._workbench_solver(sys, "auto", base_params),
            reltol=1e-8,
            abstol=1e-8
        )
    else
        error("Unsupported SYSTEM='$(name)'. Expected one of: ikeda, rossler, memristive_diode_bridge.")
    end
end

function _run_map(case, cfg::BifurcationMapConfig)
    started = time_ns()
    result, diagnostics = if case.sys isa ContinuousODE
        DynamicsKit._bifurcation_map(
            case.sys,
            cfg;
            initial_point=copy(case.sys.default_initial_state),
            solver=case.solver,
            reltol=case.reltol,
            abstol=case.abstol
        )
    else
        DynamicsKit._bifurcation_map(case.sys, cfg)
    end
    runtime_s = (time_ns() - started) / 1.0e9
    return result, diagnostics, runtime_s
end

function _mismatch_summary(lhs::AbstractMatrix{<:Integer}, rhs::AbstractMatrix{<:Integer})
    size(lhs) == size(rhs) || error("Cannot compare mismatch counts for matrices of different sizes.")
    mismatch_count = count(value -> value != 0, lhs .- rhs)
    total = length(lhs)
    return mismatch_count, total == 0 ? 0.0 : mismatch_count / total
end

function _period_labels(matrix::AbstractMatrix{<:Integer})
    values = sort(unique(vec(Int.(matrix))))
    return join(values, ';')
end

function _result_row(; mode, seed_mode, neighbor_transient, grid_steps, runtime_s, speedup_vs_fixed,
                       mismatch_count, mismatch_fraction, mismatch_vs_neighbor_full,
                       tile_size_a, tile_size_b,
                       unique_periods, diagnostics)
    return Dict(
        "mode" => mode,
        "seed_mode" => seed_mode,
        "neighbor_transient" => neighbor_transient,
        "tile_size_a" => tile_size_a,
        "tile_size_b" => tile_size_b,
        "grid" => "$(grid_steps + 1)x$(grid_steps + 1)",
        "runtime_s" => runtime_s,
        "speedup_vs_fixed" => speedup_vs_fixed,
        "mismatch_count" => mismatch_count,
        "mismatch_fraction" => mismatch_fraction,
        "mismatch_vs_neighbor_full" => mismatch_vs_neighbor_full,
        "unique_periods" => unique_periods,
        "resets" => get(diagnostics, "resets", 0),
        "invalid_resets" => get(diagnostics, "invalidResets", 0),
        "tile_count" => get(diagnostics, "tileCount", 0),
        "full_transient" => get(diagnostics, "fullTransient", nothing),
        "effective_neighbor_transient" => get(diagnostics, "neighborTransient", nothing),
        "requested_neighbor_transient" => get(diagnostics, "requestedNeighborTransient", nothing)
    )
end

function _print_rows(rows)
    columns = [
        "mode",
        "seed_mode",
        "neighbor_transient",
        "tile_size_a",
        "tile_size_b",
        "grid",
        "runtime_s",
        "speedup_vs_fixed",
        "mismatch_count",
        "mismatch_fraction",
        "mismatch_vs_neighbor_full",
        "unique_periods",
        "resets",
        "invalid_resets",
        "tile_count",
        "full_transient",
        "effective_neighbor_transient",
        "requested_neighbor_transient",
    ]
    println(join(columns, ','))
    for row in rows
        values = [get(row, column, nothing) for column in columns]
        println(join(map(values) do value
            if value === nothing
                return ""
            elseif value isa AbstractFloat
                return @sprintf("%.6f", value)
            else
                return string(value)
            end
        end, ','))
    end
end

function _write_summary(rows, case)
    output_dir = strip(_env_string("OUTPUT_DIR", ""))
    save_results = _env_bool("SAVE_RESULTS", false) || !isempty(output_dir)
    save_results || return nothing

    dir = isempty(output_dir) ? joinpath(@__DIR__, "..", "var", "output", "benchmarks") : output_dir
    mkpath(dir)
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    file_path = joinpath(dir, "neighbor_seed_acceleration_$(case.key)_$(stamp).json")
    open(file_path, "w") do io
        JSON3.pretty(io, Dict(
            "generatedAt" => string(now()),
            "threads" => Threads.nthreads(),
            "system" => case.label,
            "rows" => rows
        ))
    end
    @info "Wrote benchmark summary" file=file_path
    return file_path
end

function main()
    case = _benchmark_case(_env_string("SYSTEM", "memristive_diode_bridge"))
    grid_steps = _env_int("GRID_STEPS", 50)
    neighbor_transients = _parse_transient_list(_env_string("NEIGHBOR_TRANSIENTS", "0,2,5,10,20"))
    tile_size_a = _env_int("NEIGHBOR_TILE_SIZE_A", 0)
    tile_size_b = _env_int("NEIGHBOR_TILE_SIZE_B", 0)

    println("# Neighbor-seed acceleration benchmark")
    println("# system=$(case.label), threads=$(Threads.nthreads()), grid=$(grid_steps + 1)x$(grid_steps + 1), transients=$(isempty(neighbor_transients) ? "(none)" : join(neighbor_transients, ',')), tile=$(tile_size_a)x$(tile_size_b)")

    fixed_cfg = BifurcationMapConfig(
        a_min=case.a_min,
        a_max=case.a_max,
        a_steps=grid_steps,
        b_min=case.b_min,
        b_max=case.b_max,
        b_steps=grid_steps,
        a_index=case.a_index,
        b_index=case.b_index,
        max_period=case.max_period,
        precision=case.precision,
        iterations=case.iterations,
        base_params=copy(case.base_params),
    )
    fixed_result, fixed_diag, fixed_runtime = _run_map(case, fixed_cfg)

    neighbor_full_cfg = BifurcationMapConfig(
        a_min=case.a_min,
        a_max=case.a_max,
        a_steps=grid_steps,
        b_min=case.b_min,
        b_max=case.b_max,
        b_steps=grid_steps,
        a_index=case.a_index,
        b_index=case.b_index,
        max_period=case.max_period,
        precision=case.precision,
        iterations=case.iterations,
        base_params=copy(case.base_params),
        reuse_neighbor_seeds=true,
    )
    neighbor_full_result, neighbor_full_diag, neighbor_full_runtime = _run_map(case, neighbor_full_cfg)

    rows = Dict{String, Any}[]
    push!(rows, _result_row(
        mode="fixed",
        seed_mode=get(fixed_diag, "seedMode", "fixed"),
        neighbor_transient=nothing,
        tile_size_a=nothing,
        tile_size_b=nothing,
        grid_steps=grid_steps,
        runtime_s=fixed_runtime,
        speedup_vs_fixed=1.0,
        mismatch_count=0,
        mismatch_fraction=0.0,
        mismatch_vs_neighbor_full=0,
        unique_periods=_period_labels(fixed_result.periodicity),
        diagnostics=fixed_diag
    ))

    mismatch_count, mismatch_fraction = _mismatch_summary(neighbor_full_result.periodicity, fixed_result.periodicity)
    push!(rows, _result_row(
        mode="neighbor_full",
        seed_mode=get(neighbor_full_diag, "seedMode", "neighbor_full"),
        neighbor_transient=nothing,
        tile_size_a=nothing,
        tile_size_b=nothing,
        grid_steps=grid_steps,
        runtime_s=neighbor_full_runtime,
        speedup_vs_fixed=fixed_runtime / max(neighbor_full_runtime, eps(Float64)),
        mismatch_count=mismatch_count,
        mismatch_fraction=mismatch_fraction,
        mismatch_vs_neighbor_full=0,
        unique_periods=_period_labels(neighbor_full_result.periodicity),
        diagnostics=neighbor_full_diag
    ))

    for transient in neighbor_transients
        cfg = BifurcationMapConfig(
            a_min=case.a_min,
            a_max=case.a_max,
            a_steps=grid_steps,
            b_min=case.b_min,
            b_max=case.b_max,
            b_steps=grid_steps,
            a_index=case.a_index,
            b_index=case.b_index,
            max_period=case.max_period,
            precision=case.precision,
            iterations=case.iterations,
            base_params=copy(case.base_params),
            reuse_neighbor_seeds=true,
            neighbor_transient=transient,
        )
        result, diagnostics, runtime_s = _run_map(case, cfg)
        mismatch_count, mismatch_fraction = _mismatch_summary(result.periodicity, fixed_result.periodicity)
        mismatch_vs_neighbor_full, _ = _mismatch_summary(result.periodicity, neighbor_full_result.periodicity)
        push!(rows, _result_row(
            mode="neighbor_accelerated",
            seed_mode=get(diagnostics, "seedMode", "neighbor_accelerated"),
            neighbor_transient=transient,
            tile_size_a=nothing,
            tile_size_b=nothing,
            grid_steps=grid_steps,
            runtime_s=runtime_s,
            speedup_vs_fixed=fixed_runtime / max(runtime_s, eps(Float64)),
            mismatch_count=mismatch_count,
            mismatch_fraction=mismatch_fraction,
            mismatch_vs_neighbor_full=mismatch_vs_neighbor_full,
            unique_periods=_period_labels(result.periodicity),
            diagnostics=diagnostics
        ))

        if tile_size_a > 0 || tile_size_b > 0
            tiled_cfg = BifurcationMapConfig(
                a_min=case.a_min,
                a_max=case.a_max,
                a_steps=grid_steps,
                b_min=case.b_min,
                b_max=case.b_max,
                b_steps=grid_steps,
                a_index=case.a_index,
                b_index=case.b_index,
                max_period=case.max_period,
                precision=case.precision,
                iterations=case.iterations,
                base_params=copy(case.base_params),
                reuse_neighbor_seeds=true,
                neighbor_transient=transient,
                neighbor_tile_size_a=tile_size_a,
                neighbor_tile_size_b=tile_size_b,
            )
            tiled_result, tiled_diagnostics, tiled_runtime_s = _run_map(case, tiled_cfg)
            tiled_mismatch_count, tiled_mismatch_fraction = _mismatch_summary(tiled_result.periodicity, fixed_result.periodicity)
            tiled_mismatch_vs_neighbor_full, _ = _mismatch_summary(tiled_result.periodicity, neighbor_full_result.periodicity)
            push!(rows, _result_row(
                mode="neighbor_accelerated_tiled",
                seed_mode=get(tiled_diagnostics, "seedMode", "neighbor_accelerated"),
                neighbor_transient=transient,
                tile_size_a=tile_size_a,
                tile_size_b=tile_size_b,
                grid_steps=grid_steps,
                runtime_s=tiled_runtime_s,
                speedup_vs_fixed=fixed_runtime / max(tiled_runtime_s, eps(Float64)),
                mismatch_count=tiled_mismatch_count,
                mismatch_fraction=tiled_mismatch_fraction,
                mismatch_vs_neighbor_full=tiled_mismatch_vs_neighbor_full,
                unique_periods=_period_labels(tiled_result.periodicity),
                diagnostics=tiled_diagnostics
            ))
        end
    end

    _print_rows(rows)
    _write_summary(rows, case)
    return nothing
end

main()



