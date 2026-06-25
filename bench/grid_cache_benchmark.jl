using DynamicsKit
using Dates

function with_temp_workbench_cache_env(f::Function)
    mktempdir() do temp_root
        sandbox_var = joinpath(temp_root, "var")
        withenv(
            "BIFURCATIONEXPLORER_WORKBENCH_VAR_DIR" => sandbox_var,
            "BIFURCATIONEXPLORER_WORKBENCH_LOG_DIR" => joinpath(sandbox_var, "output", "workbench_logs"),
            "BIFURCATIONEXPLORER_WORKBENCH_RESULT_DIR" => joinpath(sandbox_var, "output", "workbench_results"),
            "BIFURCATIONEXPLORER_WORKBENCH_SKELETON_CACHE_DIR" => joinpath(sandbox_var, "cache"),
            "BIFURCATIONEXPLORER_WORKBENCH_GRID_CACHE_DIR" => joinpath(sandbox_var, "cache", "grid_results")
        ) do
            f()
        end
    end
end

function grid_diag(session)
    return get(get(session.data, "diagnostics", Dict{String, Any}()), "gridCache", Dict{String, Any}())
end

function print_case(io::IO, label::AbstractString, session, equality::Union{Nothing, Bool}=nothing)
    diag = grid_diag(session)
    reused = get(diag, "reusedCells", get(diag, "reusedSamples", 0))
    computed = get(diag, "computedCells", get(diag, "computedSamples", 0))
    requested = get(diag, "requestedCells", get(diag, "requestedSamples", 0))
    disabled_reason = get(diag, "disabledReason", nothing)
    equality_text = isnothing(equality) ? "n/a" : string(equality)
    println(io, "[$(Dates.format(now(), dateformat"HH:MM:SS"))] $(rpad(label, 28)) runtime_ms=$(round(session.analysis_runtime_ms; digits=1)) reused=$(reused) computed=$(computed) requested=$(requested) disabledReason=$(disabled_reason) equalToFresh=$(equality_text)")
end

results_equal(lhs::BruteForceResult, rhs::BruteForceResult) = lhs.params == rhs.params && lhs.points == rhs.points
results_equal(lhs::BifurcationMapResult, rhs::BifurcationMapResult) = lhs.a_grid == rhs.a_grid && lhs.b_grid == rhs.b_grid && lhs.periodicity == rhs.periodicity
results_equal(lhs::BasinsResult, rhs::BasinsResult) = lhs.x_grid == rhs.x_grid && lhs.y_grid == rhs.y_grid && lhs.periodicity == rhs.periodicity

function brute_force_payload(; steps::Int, cache::Bool=true, file_cache::Bool=true, salt::AbstractString)
    return Dict(
        "analysisType" => "brute_force",
        "systemKey" => "henon",
        "systemOptions" => Dict{String, Any}(),
        "runConfig" => Dict{String, Any}(
            "analysisParams" => Dict{String, Any}("a" => 1.2, "b" => 0.3),
            "bifurcationParam" => "a",
            "linkedBifurcationParams" => String[],
            "paramMin" => 1.0,
            "paramMax" => 1.3,
            "bfSteps" => steps,
            "bfIterations" => 40,
            "bfTransient" => 20,
            "bruteForceGridCache" => cache,
            "fileCache" => file_cache,
            "cacheSalt" => salt
        )
    )
end

function map_payload(; steps::Int, cache::Bool=true, file_cache::Bool=true, salt::AbstractString)
    return Dict(
        "analysisType" => "bifurcation_map",
        "systemKey" => "ikeda",
        "systemOptions" => Dict{String, Any}(),
        "runConfig" => Dict{String, Any}(
            "analysisParams" => Dict{String, Any}("a" => 0.4, "b" => 6.0),
            "mapAParam" => "u",
            "mapBParam" => "b",
            "mapAMin" => 0.85,
            "mapAMax" => 0.95,
            "mapASteps" => steps,
            "mapBMin" => 5.8,
            "mapBMax" => 6.2,
            "mapBSteps" => steps,
            "mapMaxPeriod" => 4,
            "mapIterations" => 60,
            "mapPrecision" => 1e-4,
            "mapGridCache" => cache,
            "fileCache" => file_cache,
            "cacheSalt" => salt
        )
    )
end

function basins_payload(; steps::Int, cache::Bool=true, file_cache::Bool=true, salt::AbstractString)
    return Dict(
        "analysisType" => "basins",
        "systemKey" => "henon",
        "systemOptions" => Dict{String, Any}(),
        "runConfig" => Dict{String, Any}(
            "analysisParams" => Dict{String, Any}("a" => 1.2, "b" => 0.3),
            "bifurcationParam" => "a",
            "linkedBifurcationParams" => String[],
            "basinsBifParam" => 1.2,
            "gridXMin" => -0.4,
            "gridXMax" => 0.4,
            "gridYMin" => -0.4,
            "gridYMax" => 0.4,
            "gridXSteps" => steps,
            "gridYSteps" => steps,
            "basinsMaxPeriod" => 4,
            "basinsPrecision" => 1e-4,
            "bfIterations" => 40,
            "basinsGridCache" => cache,
            "fileCache" => file_cache,
            "cacheSalt" => salt
        )
    )
end

function benchmark_analysis(name::AbstractString, coarse_payload::Dict{String, Any}, fine_payload::Dict{String, Any}, fresh_payload::Dict{String, Any}, coarsened_payload::Dict{String, Any}, result_key::AbstractString)
    println("\n=== $(name) ===")
    coarse = DynamicsKit._run_workbench_analysis(deepcopy(coarse_payload))
    fine = DynamicsKit._run_workbench_analysis(deepcopy(fine_payload))
    fresh = DynamicsKit._run_workbench_analysis(deepcopy(fresh_payload))
    coarsened = DynamicsKit._run_workbench_analysis(deepcopy(coarsened_payload))

    equal_fine = results_equal(get(fine.data, result_key, nothing), get(fresh.data, result_key, nothing))
    equal_coarsened = results_equal(get(coarsened.data, result_key, nothing), get(coarse.data, result_key, nothing))

    print_case(stdout, "coarse", coarse)
    print_case(stdout, "fine after coarse cache", fine, equal_fine)
    print_case(stdout, "fresh fine", fresh, equal_fine)
    print_case(stdout, "coarse after fine cache", coarsened, equal_coarsened)
end

with_temp_workbench_cache_env() do
    benchmark_analysis(
        "1D brute-force sample cache",
        brute_force_payload(steps=4, salt="bf-coarse"),
        brute_force_payload(steps=8, salt="bf-fine"),
        brute_force_payload(steps=8, cache=false, salt="bf-fresh"),
        brute_force_payload(steps=4, salt="bf-coarsened"),
        "brute_force"
    )

    benchmark_analysis(
        "2D bifurcation-map grid cache",
        map_payload(steps=4, salt="map-coarse"),
        map_payload(steps=8, salt="map-fine"),
        map_payload(steps=8, cache=false, salt="map-fresh"),
        map_payload(steps=4, salt="map-coarsened"),
        "bifurcation_map"
    )

    benchmark_analysis(
        "Basins grid cache",
        basins_payload(steps=4, salt="basins-coarse"),
        basins_payload(steps=8, salt="basins-fine"),
        basins_payload(steps=8, cache=false, salt="basins-fresh"),
        basins_payload(steps=4, salt="basins-coarsened"),
        "basins"
    )
end


