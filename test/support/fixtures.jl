module TestFixtures

using DynamicsKit

export henon_atlas_payload, radial_oscillator, with_temp_workbench_dirs

function henon_atlas_payload(; run_config=Dict{String, Any}(), payload=Dict{String, Any}())
    base_run_config = Dict{String, Any}(
        "analysisParams" => Dict{String, Any}("a" => 0.3),
        "bifurcationParam" => "a",
        "paramMin" => 0.0,
        "paramMax" => 0.35,
        "periods" => [1],
        "bfSteps" => 12,
        "bfIterations" => 120,
        "bfTransient" => 80,
        "ds" => 0.01,
        "dsmax" => 0.03,
        "dsmin" => 1e-6,
        "maxSteps" => 160,
        "newtonTol" => 1e-10,
        "newtonMaxIter" => 40,
        "atlasReconSteps" => 20,
        "atlasReconPrecision" => 1e-4,
        "atlasWindowMinSupport" => 2,
        "atlasSeedPointsPerWindow" => 4,
        "atlasCoverageThreshold" => 0.2,
        "atlasMaxRefinementDepth" => 1,
        "threaded" => false,
        "cacheSalt" => string(time_ns())
    )
    merge!(base_run_config, run_config)

    base_payload = Dict{String, Any}(
        "analysisType" => "atlas",
        "systemKey" => "henon",
        "systemOptions" => Dict{String, Any}(),
        "runConfig" => base_run_config
    )
    merge!(base_payload, payload)
    return base_payload
end

function radial_oscillator()
    function f!(du, u, p, t)
        μ = p[1]
        r2 = u[1]^2 + u[2]^2
        du[1] = u[2] + u[1] * (μ - r2)
        du[2] = -u[1] + u[2] * (μ - r2)
        nothing
    end

    section = PoincareSection(
        (u, t, integrator) -> u[2];
        direction=:up,
        projection=[1],
        template=[0.0, 0.0]
    )

    ContinuousODE(
        f!, 2, section, [:μ], "Radial Oscillator";
        tspan_hint=8.0,
        default_initial_state=[1.0, 0.1],
        default_params=[0.25]
    )
end

function with_temp_workbench_dirs(f::Function)
    mktempdir() do temp_root
        sandbox_var = joinpath(temp_root, "var")
        sandbox_log_dir = joinpath(sandbox_var, "output", "workbench_logs")
        sandbox_result_dir = joinpath(sandbox_var, "output", "workbench_results")
        sandbox_cache_dir = joinpath(sandbox_var, "cache")
        sandbox_grid_cache_dir = joinpath(sandbox_cache_dir, "grid_results")
        paths = (
            root=temp_root,
            var=sandbox_var,
            logs=sandbox_log_dir,
            results=sandbox_result_dir,
            skeleton_cache=sandbox_cache_dir,
            grid_cache=sandbox_grid_cache_dir,
        )

        withenv(
            "BIFURCATIONEXPLORER_WORKBENCH_VAR_DIR" => sandbox_var,
            "BIFURCATIONEXPLORER_WORKBENCH_LOG_DIR" => sandbox_log_dir,
            "BIFURCATIONEXPLORER_WORKBENCH_RESULT_DIR" => sandbox_result_dir,
            "BIFURCATIONEXPLORER_WORKBENCH_SKELETON_CACHE_DIR" => sandbox_cache_dir,
            "BIFURCATIONEXPLORER_WORKBENCH_GRID_CACHE_DIR" => sandbox_grid_cache_dir,
        ) do
            f(paths)
        end
    end
end

end
