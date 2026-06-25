#!/usr/bin/env julia
#
# A/B benchmark for the continuation improvements:
#   (1) targeted re-seed recovery  — coverage gained when a branch dies in the interior
#   (2) no-regression overhead     — re-seed must be a no-op when branches complete normally
#   (3) PALC adaptivity tuning     — fewer steps / less time when `a`/`dsmax` are pushed
#
# Two systems are exercised:
#   - Radial (Hopf normal form) oscillator: fast, deterministic, isolates each mechanism.
#   - Colpitts (exponential) oscillator over V1 ∈ [4.0, 5.0]: a real, stiff Poincaré-map system
#     (the V1-window atlas case the project actually runs) — shows the cost/benefit on hardware
#     that matters, where each return-map evaluation is a full stiff ODE integration.
#
# Plain `@elapsed` (min-of-N) is used deliberately — no BenchmarkTools dependency, so the
# package test suite and its Aqua quality pass are untouched.
#
# Run from the project root (threads help the bidirectional + atlas paths):
#   julia --threads auto --project=. bench/reseed_benchmark.jl

using DynamicsKit
using DifferentialEquations: Tsit5, AutoTsit5, Rosenbrock23
using Printf

# Radial (Hopf normal form) oscillator: limit cycle of radius √μ for μ > 0.
function radial_oscillator()
    function f!(du, u, p, t)
        μ = p[1]
        r2 = u[1]^2 + u[2]^2
        du[1] = u[2] + u[1] * (μ - r2)
        du[2] = -u[1] + u[2] * (μ - r2)
        nothing
    end
    section = PoincareSection((u, t, integrator) -> u[2]; direction = :up,
                              projection = [1], template = [0.0, 0.0])
    ContinuousODE(f!, 2, section, [:μ], "Radial Oscillator";
                  tspan_hint = 8.0, default_initial_state = [1.0, 0.1],
                  default_params = [0.25])
end

"""Coverage fraction of [p_min, p_max] spanned by a branch's recorded points."""
function coverage_fraction(branch, p_min, p_max)
    pts = DynamicsKit._branch_points(branch)
    isempty(pts) && return 0.0
    params = Float64[p.param for p in pts]
    span = max(p_max - p_min, eps(Float64))
    return clamp((maximum(params) - minimum(params)) / span, 0.0, 1.0)
end

"""Run one continuation A/B sample and collect metrics (min wall-time over `repeats`)."""
function measure_continuation(sys, cfg; params, period = 1, reseed = ReseedConfig(),
                              repeats = 3, solver = Tsit5(), kwargs...)
    total_reseeds = Ref(0)
    on_reseed = (a, b) -> begin
        for d in (a, b)
            d === nothing && continue
            total_reseeds[] += d.attempt_count
        end
    end

    # Warm up (compile) + capture a representative result.
    result = continuation_branch(sys, cfg, period; params = params, reseed = reseed,
                                 on_reseed = on_reseed, solver = solver, kwargs...)

    best = Inf
    for _ in 1:repeats
        total_reseeds[] = 0
        t = @elapsed continuation_branch(sys, cfg, period; params = params, reseed = reseed,
                                         on_reseed = on_reseed, solver = solver, kwargs...)
        best = min(best, t)
    end

    pts = DynamicsKit._branch_points(result)
    return (time = best,
            points = length(pts),
            coverage = coverage_fraction(result, cfg.p_min, cfg.p_max),
            reseeds = total_reseeds[])
end

function print_table(title, rows)
    println("\n## ", title)
    @printf("| %-26s | %8s | %6s | %8s | %7s |\n", "case", "time(s)", "points", "coverage", "reseeds")
    println("|", "-"^28, "|", "-"^10, "|", "-"^8, "|", "-"^10, "|", "-"^9, "|")
    for (name, m) in rows
        @printf("| %-26s | %8.4f | %6d | %8.3f | %7d |\n",
                name, m.time, m.points, m.coverage, m.reseeds)
    end
end

function radial_benchmarks()
    sys = radial_oscillator()
    println("\n=== Radial oscillator (fast, deterministic) ===")

    # (1) Recovery: tiny max_steps forces an interior truncation the re-seed must repair.
    cfg_trunc = ContinuationConfig(p_min = 0.05, p_max = 1.5, ds = 0.03, dsmax = 0.05, max_steps = 4)
    rec_off = measure_continuation(sys, cfg_trunc; params = [0.6], n_initial = 6)
    rec_on  = measure_continuation(sys, cfg_trunc; params = [0.6], n_initial = 6,
                                   reseed = ReseedConfig(enabled = true, max_attempts = 6))
    print_table("Re-seed recovery (max_steps=4 forces interior death)",
                ["reseed off" => rec_off, "reseed on" => rec_on])

    # (2) No-regression: a normal run completes to the boundary; re-seed must add nothing.
    cfg_norm = ContinuationConfig(p_min = 0.05, p_max = 1.5, ds = 0.02, dsmax = 0.05)
    reg_off = measure_continuation(sys, cfg_norm; params = [0.6], n_initial = 6)
    reg_on  = measure_continuation(sys, cfg_norm; params = [0.6], n_initial = 6,
                                   reseed = ReseedConfig(enabled = true))
    print_table("No-regression overhead (branch reaches boundary)",
                ["reseed off" => reg_off, "reseed on" => reg_on])

    # (3) PALC tuning: wider dsmax + more aggressive `a` should need fewer steps over a long range.
    cfg_cons = ContinuationConfig(p_min = 0.05, p_max = 3.0, ds = 0.02, dsmax = 0.05, a = 0.5)
    cfg_aggr = ContinuationConfig(p_min = 0.05, p_max = 3.0, ds = 0.02, dsmax = 0.30, a = 2.0)
    tune_cons = measure_continuation(sys, cfg_cons; params = [0.6], n_initial = 6)
    tune_aggr = measure_continuation(sys, cfg_aggr; params = [0.6], n_initial = 6)
    print_table("PALC tuning over μ∈[0.05, 3.0] (fewer steps = faster)",
                ["a=0.5 dsmax=0.05" => tune_cons, "a=2.0 dsmax=0.30" => tune_aggr])
end

function colpitts_benchmarks()
    sys = colpitts_exponential_oscillator()
    base = copy(sys.default_params)
    pidx = 4  # V1
    solver = AutoTsit5(Rosenbrock23())  # the stiff-aware "auto" solver the workbench uses
    println("\n=== Colpitts (exponential) — real stiff Poincaré system, param V1 ===")
    println("(repeats=1; each return-map eval is a full stiff ODE integration)")

    # (3') PALC tuning on the real stiff system: aggressive settings should cut steps + time.
    cfg_cons = ContinuationConfig(p_min = 4.0, p_max = 5.0, ds = 0.02, dsmax = 0.05, a = 0.5,
                                  max_steps = 400, newton_tol = 1e-8, newton_max_iter = 30,
                                  detect_bifurcation = 1, param_index = pidx)
    cfg_aggr = ContinuationConfig(p_min = 4.0, p_max = 5.0, ds = 0.02, dsmax = 0.20, a = 2.0,
                                  max_steps = 400, newton_tol = 1e-8, newton_max_iter = 30,
                                  detect_bifurcation = 1, param_index = pidx)
    tune_cons = measure_continuation(sys, cfg_cons; params = base, period = 1, n_initial = 10,
                                     solver = solver, repeats = 1)
    tune_aggr = measure_continuation(sys, cfg_aggr; params = base, period = 1, n_initial = 10,
                                     solver = solver, repeats = 1)
    print_table("Colpitts P1 PALC tuning over V1∈[4.0, 5.0]",
                ["a=0.5 dsmax=0.05" => tune_cons, "a=2.0 dsmax=0.20" => tune_aggr])

    # (1') Recovery on the real stiff system: truncate, then let re-seed rebuild coverage.
    cfg_trunc = ContinuationConfig(p_min = 4.0, p_max = 5.0, ds = 0.03, dsmax = 0.05, max_steps = 4,
                                   newton_tol = 1e-8, newton_max_iter = 30,
                                   detect_bifurcation = 1, param_index = pidx)
    rec_off = measure_continuation(sys, cfg_trunc; params = base, period = 1, n_initial = 10,
                                   solver = solver, repeats = 1)
    rec_on  = measure_continuation(sys, cfg_trunc; params = base, period = 1, n_initial = 10,
                                   solver = solver, repeats = 1,
                                   reseed = ReseedConfig(enabled = true, max_attempts = 8,
                                                         n_skeleton_initial = 6))
    print_table("Colpitts P1 re-seed recovery (max_steps=4 forces interior death)",
                ["reseed off" => rec_off, "reseed on" => rec_on])
end

function main()
    println("Re-seed / PALC-tuning benchmark")
    println("threads = ", Threads.nthreads())
    radial_benchmarks()
    if get(ENV, "BENCH_SKIP_COLPITTS", "0") == "1"
        println("\n(Colpitts cases skipped via BENCH_SKIP_COLPITTS=1)")
    else
        colpitts_benchmarks()
    end

    println("\nNotes:")
    println("- recovery rows: coverage jumps from a partial span to ≈1.0 with re-seed on.")
    println("- no-regression: coverage/points identical, time delta ≈ 0 (re-seed no-ops at boundaries).")
    println("- PALC tuning: aggressive a/dsmax cut points (and time) without losing coverage.")
    println("- set BENCH_SKIP_COLPITTS=1 to run only the fast radial cases.")
end

main()
