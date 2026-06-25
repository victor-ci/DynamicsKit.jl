#!/usr/bin/env julia

using DynamicsKit
using DifferentialEquations
using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "..", "var", "output", "colpitts_simple_beta_overlay")
mkpath(OUTPUT_DIR)

const BETA_MIN = parse(Float64, get(ENV, "COLPITTS_BETA_MIN", "100.0"))
const BETA_MAX = parse(Float64, get(ENV, "COLPITTS_BETA_MAX", "135.0"))
const BETA_SEED = parse(Float64, get(ENV, "COLPITTS_BETA_SEED", "120.0"))
const BF_STEPS = parse(Int, get(ENV, "COLPITTS_BF_STEPS", "80"))
const BF_ITER = parse(Int, get(ENV, "COLPITTS_BF_ITER", "180"))
const BF_TRANSIENT = parse(Int, get(ENV, "COLPITTS_BF_TRANSIENT", "120"))
const SEARCH_MIN = [4.6, -0.95]
const SEARCH_MAX = [5.8, -0.4]

function seed_point_from_bruteforce(sys, params)
    seed_points = DynamicsKit._collect_poincare_points(
        sys,
        params;
        initial_point=copy(sys.default_initial_state),
        crossings=4,
        transient=BF_TRANSIENT,
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
        projected=true
    )
    isempty(seed_points) && error("No Poincaré crossings were recorded at β=$(params[3]).")
    return collect(first(seed_points))
end

function branch_param_range(branch_result)
    params = [pt.param for pt in branch_result.branch.branch]
    return (minimum(params), maximum(params))
end

println("═══ Colpitts (simple) — β overlay with validated continuation ═══\n")
println("β window: [$BETA_MIN, $BETA_MAX], continuation seed β=$BETA_SEED")

sys = colpitts_simple_oscillator()
params = [40e-9, 40e-9, BETA_SEED, 5.0, 5.0]

println("1. Generating brute-force diagram...")
bf = brute_force_diagram(
    sys,
    BruteForceConfig(
        param_min=BETA_MIN,
        param_max=BETA_MAX,
        param_steps=BF_STEPS,
        iterations=BF_ITER,
        transient=BF_TRANSIENT,
        param_index=3,
        fixed_params=copy(params)
    );
    initial_point=copy(sys.default_initial_state),
    solver=Tsit5(),
    reltol=1e-7,
    abstol=1e-7
)
println("   Collected $(length(bf.params)) Poincaré points.")

println("2. Sampling a seed point at β=$BETA_SEED...")
seed = seed_point_from_bruteforce(sys, params)
println("   Seed ≈ $(round.(seed; digits=6))")

println("3. Refining the period-1 seed with the skeleton solver...")
skeleton = find_periodic_skeleton(
    sys,
    [1],
    BETA_SEED;
    seed_points=[seed],
    search_min=SEARCH_MIN,
    search_max=SEARCH_MAX,
    n_initial=6,
    params=copy(params),
    param_index=3,
    tol=1e-5,
    max_iter=80,
    fd_step=1e-6,
    solver=Tsit5(),
    reltol=1e-7,
    abstol=1e-7,
    threaded=false
)
isempty(skeleton) && error("The period-1 skeleton did not converge at β=$BETA_SEED.")
println("   Skeleton point ≈ $(round.(skeleton[1].point; digits=6))")

println("4. Continuing the period-1 branch across the β window...")
branch = continuation_branch(
    sys,
    ContinuationConfig(
        p_min=BETA_MIN,
        p_max=BETA_MAX,
        ds=0.5,
        dsmax=1.0,
        dsmin=1e-5,
        max_steps=120,
        newton_tol=1e-4,
        newton_max_iter=50,
        detect_bifurcation=1,
        param_index=3
    ),
    1;
    initial_point=seed,
    params=copy(params),
    search_min=SEARCH_MIN,
    search_max=SEARCH_MAX,
    n_initial=6,
    tol=1e-5,
    max_iter=80,
    fd_step=1e-6,
    solver=Tsit5(),
    reltol=1e-7,
    abstol=1e-7
)
β_lo, β_hi = branch_param_range(branch)
println("   Branch points: $(length(branch.branch.branch))")
println("   Continuation reached β ∈ [$(round(β_lo; digits=3)), $(round(β_hi; digits=3))]")

println("5. Saving plots...")
bf_plot = plot_brute_force(bf)
overlay_plot = plot_overlay(
    bf,
    [branch];
    system=sys,
    params=copy(params),
    solver=Tsit5(),
    reltol=1e-7,
    abstol=1e-7,
    min_crossing_time=1e-6
)

bf_path = joinpath(OUTPUT_DIR, "colpitts_simple_beta_bruteforce.png")
overlay_path = joinpath(OUTPUT_DIR, "colpitts_simple_beta_overlay.png")
savefig(bf_plot, bf_path)
savefig(overlay_plot, overlay_path)
println("   → $(bf_path)")
println("   → $(overlay_path)")

println("\nDone.")

