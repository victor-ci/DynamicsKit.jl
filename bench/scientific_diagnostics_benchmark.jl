include(joinpath(@__DIR__, "..", "examples", "scientific_diagnostics_suite.jl"))

using JSON3
using Dates
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

function _write_scientific_diagnostics_benchmark(rows)
    output_dir = strip(_env_string("OUTPUT_DIR", ""))
    save_results = _env_bool("SAVE_RESULTS", false) || !isempty(output_dir)
    save_results || return nothing

    dir = isempty(output_dir) ? joinpath(@__DIR__, "..", "var", "output", "benchmarks") : output_dir
    mkpath(dir)
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    file_path = joinpath(dir, "scientific_diagnostics_benchmark_$(stamp).json")
    open(file_path, "w") do io
        JSON3.pretty(io, Dict{String, Any}(
            "generatedAt" => string(now()),
            "threads" => Threads.nthreads(),
            "rows" => rows,
        ))
    end
    @info "Wrote scientific diagnostics benchmark summary" file=file_path
    return file_path
end

function main()
    repeats = max(_env_int("DIAGNOSTICS_BENCH_REPEATS", 1), 1)
    rows = Dict{String, Any}[]

    println("# Scientific diagnostics benchmark")
    println("# repeats=$(repeats), threads=$(Threads.nthreads())")
    println("repeat,case,runtime_s")

    for repeat_idx in 1:repeats
        for (label, fn) in SCIENTIFIC_DIAGNOSTIC_CASES
            GC.gc()
            row = _case_timer(label, fn)
            row["repeat"] = repeat_idx
            push!(rows, row)
            @printf("%d,%s,%.6f\n", repeat_idx, replace(row["case"], "," => ";"), row["runtime_s"])
        end
    end

    _write_scientific_diagnostics_benchmark(rows)
    return nothing
end

main()
