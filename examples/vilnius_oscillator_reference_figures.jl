#!/usr/bin/env julia

import Logging
Logging.disable_logging(Logging.Warn)

using DynamicsKit
using DifferentialEquations
using Plots
using Dates

# Reference:
#   A. Ipatovs, C. Iheanacho, D. Pikuļins, S. Tjukovs, A. Litviņenko,
#   "Complete Bifurcation Analysis of the Vilnius Chaotic Oscillator",
#   Electronics 12(13), Article 2861 (2023).
#   doi:10.3390/electronics12132861

const OUTPUT_DIR = joinpath(@__DIR__, "..", "var", "output", "vilnius_oscillator_reference_figures")
mkpath(OUTPUT_DIR)

env_bool(name, default=false) = lowercase(get(ENV, name, default ? "true" : "false")) in ("1", "true", "yes", "on")
env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_float_list(name, default::Vector{Float64}) = begin
    raw = strip(get(ENV, name, join(default, ",")))
    isempty(raw) && return copy(default)
    [parse(Float64, strip(item)) for item in split(raw, ',') if !isempty(strip(item))]
end

const MAX_PERIOD = parse(Int, get(ENV, "VILNIUS_MAX_PERIOD", "4"))
const BF_STEPS = parse(Int, get(ENV, "VILNIUS_BF_STEPS", "180"))
const BF_ITER = parse(Int, get(ENV, "VILNIUS_BF_ITER", "360"))
const BF_TRANSIENT = parse(Int, get(ENV, "VILNIUS_BF_TRANSIENT", "300"))
const N_INITIAL = parse(Int, get(ENV, "VILNIUS_N_INITIAL", "6"))
const FIGURE_FILTER = Set(filter(!isempty, split(get(ENV, "VILNIUS_FIGURES", "11,13"), ',')))
const PERIODS = collect(1:parse(Int, get(ENV, "VILNIUS_PERIOD_MAX", string(MAX_PERIOD))))

function log(io, msg)
    line = "[$(Dates.format(now(), "HH:MM:SS"))] $msg"
    println(line)
    println(io, line)
    flush(io)
end

function branch_ranges(branch_result)
    pars = [pt.param for pt in branch_result.branch.branch]
    vals = [getproperty(pt, :x1) for pt in branch_result.branch.branch]
    (minimum(pars), maximum(pars), minimum(vals), maximum(vals))
end

function contiguous_runs(indices)
    isempty(indices) && return UnitRange{Int}[]
    runs = UnitRange{Int}[]
    run_start = first(indices)
    prev = run_start
    for idx in Iterators.drop(indices, 1)
        if idx != prev + 1
            push!(runs, run_start:prev)
            run_start = idx
        end
        prev = idx
    end
    push!(runs, run_start:prev)
    return runs
end

function plot_trace!(p, trace, stable_color, unstable_color; stable_label="", unstable_label="")
    stable_idx = [i for i in eachindex(trace.values) if trace.stable[i] && isfinite(trace.values[i])]
    unstable_idx = [i for i in eachindex(trace.values) if !trace.stable[i] && isfinite(trace.values[i])]

    for run in contiguous_runs(stable_idx)
        plot!(p, trace.params[run], trace.values[run];
            color=stable_color, linewidth=2.4, alpha=0.95,
            label=stable_label)
        stable_label = ""
    end
    for run in contiguous_runs(unstable_idx)
        plot!(p, trace.params[run], trace.values[run];
            color=unstable_color, linewidth=1.9, linestyle=:dash, alpha=0.9,
            label=unstable_label)
        unstable_label = ""
    end

    if !isempty(stable_idx)
        scatter!(p, trace.params[stable_idx], trace.values[stable_idx];
            markersize=2.5, markerstrokewidth=0, color=stable_color, label="")
    end
    if !isempty(unstable_idx)
        scatter!(p, trace.params[unstable_idx], trace.values[unstable_idx];
            markersize=2.1, markershape=:utriangle, markerstrokewidth=0, color=unstable_color, label="")
    end

    return p
end

choose_solver(fixed_ε::Float64) = fixed_ε <= 0.1 ? AutoTsit5(Rosenbrock23()) : Tsit5()

function make_overlay(sys, bf, branches, title_str, file_path;
                      params=sys.default_params,
                      solver=Tsit5(),
                      xlims=nothing,
                      ylims=nothing)
    p = scatter(bf.params, bf.points[:, 1];
        markersize=0.55,
        markerstrokewidth=0,
        markeralpha=0.18,
        color=:gray45,
        label="Brute force",
        xlabel=string(bf.param_name),
        ylabel="x (Poincaré section)",
        title=title_str,
        size=(1200, 800),
        dpi=160,
        legend=:outerright,
        framestyle=:box,
        grid=true,
        xlims=xlims,
        ylims=ylims)

    stable_palette = [:blue, :green, :purple, :orange, :teal, :brown, :navy, :darkgreen]
    unstable_palette = [:red, :salmon, :deeppink, :lightsalmon, :lightcoral, :indianred, :tomato, :maroon]
    period_counters = Dict{Int, Int}()

    for br in branches
        period_counters[br.period] = get(period_counters, br.period, 0) + 1
        regime = period_counters[br.period]
        color_idx = min(regime, length(stable_palette))
        label_suffix = regime == 1 ? "" : " #$regime"
        traces = DynamicsKit._branch_plot_traces(
            sys,
            br;
            orbital=1,
            params=params,
            solver=solver,
            reltol=1e-8,
            abstol=1e-8,
            min_crossing_time=1e-6
        )

        first_trace = true
        for trace in traces
            plot_trace!(
                p,
                trace,
                stable_palette[color_idx],
                unstable_palette[color_idx];
                stable_label=first_trace ? "P$(br.period) stable$(label_suffix)" : "",
                unstable_label=first_trace ? "P$(br.period) unstable$(label_suffix)" : ""
            )
            first_trace = false
        end
    end

    savefig(p, file_path)
    return p
end

function run_diagram(io; figure_no, fixed_ε, a_min, a_max, description, skeleton_params)
    label = "fig$(figure_no)_eps_$(replace(string(fixed_ε), "." => "p"))"
    sys = vilnius_oscillator(b=30.0, ε=fixed_ε)
    threaded_branches = Threads.nthreads() > 1
    solver = choose_solver(fixed_ε)
    cont_ds = env_float("VILNIUS_CONT_DS", fixed_ε <= 0.1 ? 0.0015 : 0.002)
    cont_dsmax = env_float("VILNIUS_CONT_DSMAX", fixed_ε <= 0.1 ? 0.004 : 0.008)
    cont_dsmin = env_float("VILNIUS_CONT_DSMIN", fixed_ε <= 0.1 ? 1e-8 : 1e-7)
    cont_max_steps = env_int("VILNIUS_CONT_MAX_STEPS", fixed_ε <= 0.1 ? 1400 : 900)
    cont_newton_tol = env_float("VILNIUS_CONT_NEWTON_TOL", fixed_ε <= 0.1 ? 1e-6 : 1e-8)
    cont_newton_max_iter = env_int("VILNIUS_CONT_NEWTON_MAX_ITER", fixed_ε <= 0.1 ? 40 : 30)

    log(io, "")
    log(io, "=== Figure $figure_no | ε=$fixed_ε | $description ===")
    log(io, "Brute force: a ∈ [$a_min, $a_max], b=30, ε=$fixed_ε")
    log(io, "Threaded continuation search: $(threaded_branches) (Threads.nthreads()=$(Threads.nthreads()))")
    log(io, "ODE solver: $(typeof(solver))")

    bf_config = BruteForceConfig(
        param_min=a_min,
        param_max=a_max,
        param_steps=BF_STEPS,
        iterations=BF_ITER,
        transient=BF_TRANSIENT,
        param_index=1,
        fixed_params=[0.2, 30.0, fixed_ε]
    )

    t0 = time()
    bf = brute_force_diagram(sys, bf_config;
        initial_point=[0.0, 0.1, 0.0],
        solver=solver,
        reltol=1e-8,
        abstol=1e-8)
    dt_bf = round(time() - t0, digits=1)
    log(io, "Brute force complete: $(length(bf.params)) points in $(dt_bf)s")

    x_lo = minimum(bf.points[:, 1]) - 5.0
    x_hi = maximum(bf.points[:, 1]) + 5.0
    z_lo = minimum(bf.points[:, 2]) - 2.0
    z_hi = maximum(bf.points[:, 2]) + 2.0
    search_min = [x_lo, z_lo]
    search_max = [x_hi, z_hi]
    y_pad = max(1.0, 0.05 * (maximum(bf.points[:, 1]) - minimum(bf.points[:, 1])))
    plot_ylims = (minimum(bf.points[:, 1]) - y_pad, maximum(bf.points[:, 1]) + y_pad)

    bf_png = joinpath(OUTPUT_DIR, "$(label)_bf.png")
    pbf = scatter(bf.params, bf.points[:, 1];
        markersize=0.8,
        markerstrokewidth=0,
        markeralpha=0.65,
        color=:steelblue4,
        label="",
        xlabel="a",
        ylabel="x (Poincaré section)",
        title="Vilnius oscillator Figure $figure_no — brute force (b=30, ε=$fixed_ε)",
        size=(1100, 700),
        dpi=160,
        xlims=(a_min, a_max),
        ylims=plot_ylims,
        framestyle=:box,
        grid=true)
    savefig(pbf, bf_png)
    log(io, "Saved brute-force plot: $(basename(bf_png))")

    log(io, "Search bounds from brute force: x∈[$(round(x_lo,digits=2)), $(round(x_hi,digits=2))], z∈[$(round(z_lo,digits=2)), $(round(z_hi,digits=2))]")
    log(io, "Continuation search: periods=$(PERIODS), skeletons=$(skeleton_params)")

    branch_limit = env_int("VILNIUS_MAX_BRANCHES_PER_PERIOD", figure_no == 11 ? 4 : figure_no == 13 ? 4 : 3)
    reuse_neighbor_seeds = env_bool("VILNIUS_REUSE_NEIGHBOR_SEEDS", figure_no in (11, 13))
    auto_refine = env_bool("VILNIUS_AUTO_REFINE", figure_no == 11)
    auto_refine_passes = env_int("VILNIUS_AUTO_REFINE_PASSES", 1)
    refine_factor = env_float("VILNIUS_REFINE_FACTOR", 4.0)
    refine_gap_factor = env_float("VILNIUS_REFINE_GAP_FACTOR", 3.0)
    refine_padding_factor = env_float("VILNIUS_REFINE_PADDING_FACTOR", 0.75)
    refine_short_branch_span_factor = env_float("VILNIUS_REFINE_SHORT_BRANCH_SPAN_FACTOR", 8.0)
    refine_short_branch_max_points = env_int("VILNIUS_REFINE_SHORT_BRANCH_MAX_POINTS", 180)

    cont_config = ContinuationConfig(
        p_min=a_min,
        p_max=a_max,
        ds=cont_ds,
        dsmax=cont_dsmax,
        dsmin=cont_dsmin,
        max_steps=cont_max_steps,
        newton_tol=cont_newton_tol,
        newton_max_iter=cont_newton_max_iter,
        detect_bifurcation=1,
        param_index=1
    )

    period_batches = begin
        if figure_no == 11 && maximum(PERIODS) > 2
            high_period_skeletons = env_float_list("VILNIUS_HIGH_PERIOD_SKELETONS", [last(skeleton_params)])
            high_period_branch_limit = env_int("VILNIUS_HIGH_PERIOD_MAX_BRANCHES_PER_PERIOD", min(branch_limit, 2))
            low_periods = [p for p in PERIODS if p <= 2]
            high_periods = [p for p in PERIODS if p >= 3]
            batches = NamedTuple[]
            !isempty(low_periods) && push!(batches, (periods=low_periods, skeletons=collect(Float64, skeleton_params), max_branches=branch_limit))
            !isempty(high_periods) && push!(batches, (periods=high_periods, skeletons=high_period_skeletons, max_branches=high_period_branch_limit))
            batches
        else
            [(periods=collect(Int, PERIODS), skeletons=collect(Float64, skeleton_params), max_branches=branch_limit)]
        end
    end

    t1 = time()
    branches = BranchResult[]
    for batch in period_batches
        log(io, "  Continuation batch: periods=$(batch.periods), skeletons=$(batch.skeletons), max_branches=$(batch.max_branches)")
        append!(branches, continuation_branches(
            sys,
            cont_config,
            batch.periods;
            skeleton_params=batch.skeletons,
            params=[first(batch.skeletons), 30.0, fixed_ε],
            search_min=search_min,
            search_max=search_max,
            n_initial=N_INITIAL,
            tol=1e-6,
            max_iter=30,
            fd_step=1e-6,
            solver=solver,
            reltol=1e-8,
            abstol=1e-8,
            max_branches_per_period=batch.max_branches,
            reuse_neighbor_seeds=reuse_neighbor_seeds,
            signature_param_tol=1e-2,
            signature_state_tol=0.75,
            threaded=threaded_branches
        ))
    end
    dt_cont = round(time() - t1, digits=1)
    log(io, "Continuation complete: $(length(branches)) branches in $(dt_cont)s")

    if auto_refine && !isempty(branches)
        log(io, "Auto-refinement: enabled (passes=$(auto_refine_passes), factor=$(round(refine_factor, digits=2)), gap_factor=$(round(refine_gap_factor, digits=2)))")
        t_ref = time()

        function refine_one_branch(br)
            intervals = DynamicsKit._continuous_branch_refinement_intervals(
                br,
                cont_config;
                gap_factor=refine_gap_factor,
                interval_padding_factor=refine_padding_factor,
                detect_short_branches=true,
                short_branch_span_factor=refine_short_branch_span_factor,
                short_branch_max_points=refine_short_branch_max_points
            )

            refined = isempty(intervals) ? br : auto_refine_branch(
                sys,
                br,
                cont_config;
                params=sys.default_params,
                search_min=search_min,
                search_max=search_max,
                n_initial=N_INITIAL,
                tol=1e-6,
                max_iter=30,
                fd_step=1e-6,
                solver=solver,
                reltol=1e-8,
                abstol=1e-8,
                max_passes=auto_refine_passes,
                refine_factor=refine_factor,
                gap_factor=refine_gap_factor,
                interval_padding_factor=refine_padding_factor,
                detect_short_branches=true,
                short_branch_span_factor=refine_short_branch_span_factor,
                short_branch_max_points=refine_short_branch_max_points
            )

            return (intervals=intervals, branch=refined)
        end

        refine_results = if threaded_branches && length(branches) > 1
            tasks = map(branches) do br
                Threads.@spawn refine_one_branch(br)
            end
            fetch.(tasks)
        else
            map(refine_one_branch, branches)
        end

        total_intervals = sum(length(result.intervals) for result in refine_results)
        branches = [result.branch for result in refine_results]
        for (br, result) in zip(branches, refine_results)
            isempty(result.intervals) && continue
            interval_desc = join(["[$(round(lo, digits=4)), $(round(hi, digits=4))]" for (lo, hi) in result.intervals], ", ")
            log(io, "  Refined P$(br.period) over $(length(result.intervals)) interval(s): $interval_desc")
        end
        dt_ref = round(time() - t_ref, digits=1)
        log(io, "Auto-refinement complete: $(total_intervals) interval(s) in $(dt_ref)s")
    end

    for br in branches
        pmin, pmax, xmin, xmax = branch_ranges(br)
        log(io, "  P$(br.period): len=$(length(br.branch)) a∈[$(round(pmin,digits=4)), $(round(pmax,digits=4))] x∈[$(round(xmin,digits=2)), $(round(xmax,digits=2))]")
    end

    overlay_png = joinpath(OUTPUT_DIR, "$(label)_overlay.png")
    make_overlay(
        sys,
        bf,
        branches,
        "Vilnius oscillator Figure $figure_no — brute force + continuation (b=30, ε=$fixed_ε)",
        overlay_png;
        params=sys.default_params,
        solver=solver,
        xlims=(a_min, a_max),
        ylims=plot_ylims
    )
    log(io, "Saved overlay plot: $(basename(overlay_png))")

    return (bf=bf, branches=branches, label=label, bf_png=bf_png, overlay_png=overlay_png)
end

open(joinpath(OUTPUT_DIR, "run_log.txt"), "w") do io
    log(io, "Vilnius oscillator reference-figure run")
    log(io, "Reference: Ipatovs et al., Electronics 12(13), Article 2861 (2023), doi:10.3390/electronics12132861")
    log(io, "Output directory: $OUTPUT_DIR")
    log(io, "MAX_PERIOD=$MAX_PERIOD, BF_STEPS=$BF_STEPS, BF_ITER=$BF_ITER, BF_TRANSIENT=$BF_TRANSIENT, N_INITIAL=$N_INITIAL")

    if "11" in FIGURE_FILTER
        run_diagram(io;
            figure_no=11,
            fixed_ε=0.07,
            a_min=0.05,
            a_max=0.6,
            description="coexisting P1/P2 groups with a secondary period-doubling cascade",
            skeleton_params=[0.10, 0.22, 0.38, 0.48]
        )
    end

    if "13" in FIGURE_FILTER
        run_diagram(io;
            figure_no=13,
            fixed_ε=0.2,
            a_min=0.05,
            a_max=0.6,
            description="single-group smooth double-sided period-doubling cascade",
            skeleton_params=[0.10, 0.20, 0.32, 0.46]
        )
    end

    log(io, "")
    log(io, "Run complete.")
end

println("Outputs written to: $OUTPUT_DIR")
