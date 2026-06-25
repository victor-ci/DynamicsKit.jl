using DynamicsKit
using Dates
using JSON3
using StaticArrays

# Compact diagnostic workflows that exercise the main analysis modes on
# well-known systems without treating the runtimes as benchmark claims.

function _case_timer(label::AbstractString, fn::Function)
    started = time_ns()
    summary = fn()
    runtime_s = (time_ns() - started) / 1.0e9
    return merge(Dict{String, Any}(
        "case" => String(label),
        "runtime_s" => runtime_s,
    ), summary)
end

function henon_period_doubling_hidden_window_case()
    sys = henon_map()
    settings = [
        (a=0.30, expected=1, note="period-1 fixed point"),
        (a=0.80, expected=2, note="first doubled branch"),
        (a=1.00, expected=4, note="second doubled branch"),
        (a=1.25, expected=7, note="periodic window inside the high-period/chaotic range"),
    ]

    detections = Dict{String, Any}[]
    for item in settings
        result = DynamicsKit._detect_discrete_map_period(
            sys,
            [item.a, 0.3],
            SVector(0.1, 0.1),
            800,
            12,
            1e-6,
            Inf,
        )
        @assert result.period == item.expected "Henon diagnostic at a=$(item.a) expected period $(item.expected), got $(result.period)."
        push!(detections, Dict{String, Any}(
            "a" => item.a,
            "period" => result.period,
            "status" => String(result.status),
            "closureError" => result.min_closure_error,
            "candidatePeriod" => result.closure_candidate_period,
            "note" => item.note,
        ))
    end

    return Dict{String, Any}(
        "system" => sys.name,
        "detections" => detections,
        "periods" => [item["period"] for item in detections],
    )
end

function ikeda_multistability_lyapunov_case()
    sys = ikeda_map()
    cfg = BifurcationMapConfig(
        a_min=0.65,
        a_max=0.65,
        a_steps=0,
        b_min=7.5,
        b_max=7.5,
        b_steps=0,
        a_index=1,
        b_index=3,
        base_params=[0.65, 0.4, 7.5],
        max_period=12,
        precision=1e-5,
        iterations=320,
        multistability_initial_points=[
            [-0.5, -0.5],
            [0.5, 0.5],
            [2.0, 0.0],
            [-2.0, 0.0],
        ],
        lyapunov_enabled=true,
        lyapunov_iterations=24,
    )
    result, diagnostics = DynamicsKit._bifurcation_map(sys, cfg)
    multistability = diagnostics["multistability"]
    lyapunov = diagnostics["lyapunov"]
    period_set = multistability["periodSets"][1, 1]

    @assert multistability["coexistenceCells"] == 1 "Ikeda diagnostic expected a multistable cell."
    @assert period_set == [0, 2] "Ikeda diagnostic expected period set [0, 2], got $(period_set)."
    @assert haskey(lyapunov["statusCounts"], "chaotic_candidate") "Ikeda diagnostic expected a chaotic Lyapunov candidate."

    return Dict{String, Any}(
        "system" => sys.name,
        "dominantPeriod" => result.periodicity[1, 1],
        "periodSet" => period_set,
        "lyapunovStatusCounts" => lyapunov["statusCounts"],
        "largestLyapunovExponent" => lyapunov["exponents"][1, 1],
    )
end

function rossler_continuation_multiplier_case()
    sys = rossler_oscillator()
    params = [0.2, 0.2, 3.0]
    seed = only(DynamicsKit._collect_poincare_points(
        sys,
        params;
        initial_point=[1.0, 1.0, 1.0],
        crossings=1,
        transient=50,
        projected=true,
    ))
    cfg = ContinuationConfig(
        p_min=2.8,
        p_max=3.2,
        ds=0.02,
        dsmax=0.05,
        max_steps=30,
        param_index=3,
        newton_tol=1e-8,
        detect_bifurcation=1,
        ode_jacobian_method=:variational,
    )
    branch = continuation_branch(sys, cfg; initial_point=seed, params=params, n_initial=6)
    diagnostics = continuation_branch_diagnostics(
        sys,
        branch,
        params;
        max_points=5,
        ode_jacobian_method=:variational,
    )

    @assert diagnostics["status"] == "ok" "Rossler diagnostics did not complete successfully."
    @assert diagnostics["evaluatedPointCount"] > 0 "Rossler diagnostics did not evaluate branch points."
    @assert isfinite(diagnostics["maxMultiplierModulus"]) "Rossler multiplier diagnostics were not finite."

    return Dict{String, Any}(
        "system" => sys.name,
        "branchPoints" => length(branch.branch),
        "evaluatedPointCount" => diagnostics["evaluatedPointCount"],
        "maxResidualNorm" => diagnostics["maxResidualNorm"],
        "maxMultiplierModulus" => diagnostics["maxMultiplierModulus"],
        "odeJacobianMethod" => diagnostics["odeJacobianMethod"],
    )
end

function boost_switching_nonsmooth_case()
    sys = boost_converter()
    params = [2.0, 10.0, 20.0, 0.0]
    diagnostics = switching_event_diagnostics(
        sys,
        [[10.0, 2.0], [10.0, 1.0]],
        params,
    )

    @assert diagnostics["eventCount"] == 2 "Boost diagnostic expected two switching guards."
    @assert diagnostics["nearEventCount"] == 2 "Boost diagnostic expected both rail guards to be detected."

    return Dict{String, Any}(
        "system" => sys.name,
        "eventCount" => diagnostics["eventCount"],
        "nearEventCount" => diagnostics["nearEventCount"],
        "nearestEvent" => diagnostics["nearestEvent"],
        "minNormalizedDistance" => diagnostics["minNormalizedDistance"],
    )
end

function memristive_diode_bridge_multistability_case()
    sys = memristive_diode_bridge()
    base_params = [0.0155, 6.02e-6, 0.05]
    solver = DynamicsKit._workbench_solver(sys, "auto", base_params)
    cfg = BifurcationMapConfig(
        a_min=0.0155,
        a_max=0.0155,
        a_steps=0,
        b_min=0.05,
        b_max=0.05,
        b_steps=0,
        a_index=1,
        b_index=3,
        base_params=base_params,
        max_period=6,
        precision=1e-3,
        iterations=80,
        multistability_initial_points=[
            [-1.0, 0.0, 0.5],
            [1.0, 0.0, -0.5],
        ],
    )
    result, diagnostics = DynamicsKit._bifurcation_map(
        sys,
        cfg;
        solver=solver,
        reltol=1e-8,
        abstol=1e-8,
    )
    multistability = diagnostics["multistability"]
    period_set = multistability["periodSets"][1, 1]

    @assert multistability["coexistenceCells"] == 1 "Memristive bridge diagnostic expected one coexistence cell."
    @assert period_set == [1, 3] "Memristive bridge diagnostic expected period set [1, 3], got $(period_set)."

    return Dict{String, Any}(
        "system" => sys.name,
        "dominantPeriod" => result.periodicity[1, 1],
        "periodSet" => period_set,
        "statusCounts" => diagnostics["status"]["statusCounts"],
        "crossingTerminationCounts" => diagnostics["crossing"]["terminationCounts"],
    )
end

const SCIENTIFIC_DIAGNOSTIC_CASES = [
    "Henon period doubling and hidden window" => henon_period_doubling_hidden_window_case,
    "Ikeda multistability and Lyapunov diagnostics" => ikeda_multistability_lyapunov_case,
    "Rossler continuation multiplier diagnostics" => rossler_continuation_multiplier_case,
    "Boost switching nonsmooth guards" => boost_switching_nonsmooth_case,
    "Memristive diode bridge multistability map" => memristive_diode_bridge_multistability_case,
]

function run_scientific_diagnostics_suite(; verbose::Bool=true)
    rows = Dict{String, Any}[]
    for (label, fn) in SCIENTIFIC_DIAGNOSTIC_CASES
        row = _case_timer(label, fn)
        push!(rows, row)
        if verbose
            println("ok: $(row["case"]) ($(round(row["runtime_s"]; digits=3)) s)")
        end
    end
    return rows
end

function main()
    rows = run_scientific_diagnostics_suite()
    JSON3.pretty(stdout, Dict{String, Any}(
        "generatedAt" => string(now()),
        "caseCount" => length(rows),
        "rows" => rows,
    ))
    println()
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
