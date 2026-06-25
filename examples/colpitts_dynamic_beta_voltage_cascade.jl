#!/usr/bin/env julia

using DynamicsKit
using DifferentialEquations
using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "..", "var", "output", "colpitts_dynamic_beta_voltage_cascade")
mkpath(OUTPUT_DIR)

const V_MIN = 1.25
const V_MAX = 1.97
const BF_STEPS = parse(Int, get(ENV, "COLPITTS_DYN_V_BF_STEPS", "80"))
const BF_ITER = parse(Int, get(ENV, "COLPITTS_DYN_V_BF_ITER", "220"))
const BF_TRANSIENT = parse(Int, get(ENV, "COLPITTS_DYN_V_BF_TRANSIENT", "160"))
const SEARCH_MIN = [2.0, -0.82]
const SEARCH_MAX = [2.8, -0.70]
const LINKED_PARAM_INDICES = [4]

const BRANCH_CASES = [
    (period=1, v0=1.5, pmin=1.25, pmax=1.78),
    (period=2, v0=1.85, pmin=1.74, pmax=1.94),
    (period=4, v0=1.95, pmin=1.92, pmax=1.97),
]

branch_param_range(branch_result) = begin
    params = [pt.param for pt in branch_result.branch.branch]
    (minimum(params), maximum(params), length(params))
end

function run_branch(sys, case)
    params = [40e-9, 40e-9, case.v0, case.v0]
    continuation_branch(
        sys,
        ContinuationConfig(
            p_min=case.pmin,
            p_max=case.pmax,
            ds=0.01,
            dsmax=0.03,
            dsmin=1e-5,
            max_steps=220,
            newton_tol=1e-6,
            newton_max_iter=40,
            detect_bifurcation=1,
            param_index=3,
            linked_param_indices=LINKED_PARAM_INDICES,
        ),
        case.period;
        params=params,
        search_min=SEARCH_MIN,
        search_max=SEARCH_MAX,
        n_initial=8,
        tol=1e-5,
        max_iter=80,
        fd_step=1e-6,
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
    )
end

println("═══ Colpitts (dynamic β) — V1=V2 period-doubling cascade ═══\n")
println("Sweep window: V1 = V2 ∈ [$V_MIN, $V_MAX]")
println("Target branches: periods 1, 2, and 4\n")

sys = colpitts_dynamic_beta_oscillator()
base_params = [40e-9, 40e-9, 1.95, 1.95]

println("1. Generating brute-force diagram...")
bf = brute_force_diagram(
    sys,
    BruteForceConfig(
        param_min=V_MIN,
        param_max=V_MAX,
        param_steps=BF_STEPS,
        iterations=BF_ITER,
        transient=BF_TRANSIENT,
        param_index=3,
        fixed_params=copy(base_params),
        linked_param_indices=LINKED_PARAM_INDICES,
    );
    initial_point=copy(sys.default_initial_state),
    solver=Tsit5(),
    reltol=1e-7,
    abstol=1e-7,
)
println("   Collected $(length(bf.params)) Poincaré points.")

println("2. Continuing the validated reference branches...")
branches = BranchResult[]
for case in BRANCH_CASES
    println("   • Period $(case.period) from V=$(case.v0) over [$(case.pmin), $(case.pmax)]")
    branch = run_branch(sys, case)
    lo, hi, count = branch_param_range(branch)
    println("     ↳ $(count) continuation points across V ∈ [$(round(lo; digits=4)), $(round(hi; digits=4))]")
    push!(branches, branch)
end

println("3. Saving plots...")
bf_plot = plot_brute_force(bf)
overlay_plot = plot_overlay(
    bf,
    branches;
    system=sys,
    params=copy(base_params),
    linked_param_indices=LINKED_PARAM_INDICES,
    solver=Tsit5(),
    reltol=1e-7,
    abstol=1e-7,
    min_crossing_time=1e-6,
)

bf_path = joinpath(OUTPUT_DIR, "colpitts_dynamic_beta_v_bruteforce.png")
overlay_path = joinpath(OUTPUT_DIR, "colpitts_dynamic_beta_v_overlay.png")
savefig(bf_plot, bf_path)
savefig(overlay_plot, overlay_path)

println("   → $(bf_path)")
println("   → $(overlay_path)")
println("\nDone.")
