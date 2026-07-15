"""
Visualization for bifurcation diagrams and branches.
Uses Plots.jl recipes.
"""

using Plots

const _ORBIT_PHASE_ALIGNMENT_TOL = 1e-12

# Defaults for branch-level state-jump detection after phase alignment.
# Tuned so genuine continuation steps stay connected while jumps between
# coexisting orbits (e.g. period-N continuation hopping basins) break the line.
const _PHASE_JUMP_MULTIPLIER = 8.0
const _PHASE_JUMP_RANGE_FRACTION = 0.05
const _PHASE_JUMP_MIN_STEPS = 4
# Phase-jump breaks are uncapped: the detector reports every legitimate
# alignment-shift break, and rendering cost is bounded downstream by consumers
# coalescing segments into single traces with NaN gaps.
const _PHASE_JUMP_MAX_BREAKS = typemax(Int)

"""Return contiguous index runs so stability changes do not create spurious line jumps."""
function _contiguous_runs(indices::AbstractVector{<:Integer}; break_after::AbstractSet{<:Integer}=Set{Int}())
    isempty(indices) && return UnitRange{Int}[]

    runs = UnitRange{Int}[]
    run_start = Int(first(indices))
    prev = run_start
    for raw_idx in Iterators.drop(indices, 1)
        idx = Int(raw_idx)
        if idx != prev + 1 || prev in break_after
            push!(runs, run_start:prev)
            run_start = idx
        end
        prev = idx
    end
    push!(runs, run_start:prev)
    return runs
end

"""
    _phase_jump_break_indices(values_per_dim; multiplier, range_fraction, min_steps)

Identify indices ``i`` where the step from point ``i`` to point ``i+1`` is a discontinuity
across the supplied per-dimension value vectors. Returns a `Set{Int}` of indices.

The threshold is data-driven so the detector adapts to each branch:

  * `multiplier × q25(step)` — a step is suspect when it exceeds a multiple of the
    branch's robust lower-quartile step size.
  * `range_fraction × state_range` — and the absolute jump is a meaningful fraction
    of the branch's overall state extent.

We use `max(...)`, so noisy-but-smooth branches with large baseline steps and noiseless
branches with rare large outliers are both handled. Steps with non-finite endpoints
are skipped (treated as no-jump) so NaN reconstructions do not create spurious breaks.
"""
function _phase_jump_break_indices(values_per_dim::AbstractVector{<:AbstractVector{<:Real}};
                                   multiplier::Float64=_PHASE_JUMP_MULTIPLIER,
                                   range_fraction::Float64=_PHASE_JUMP_RANGE_FRACTION,
                                   min_steps::Int=_PHASE_JUMP_MIN_STEPS,
                                   max_breaks::Int=_PHASE_JUMP_MAX_BREAKS)
    isempty(values_per_dim) && return Set{Int}()
    n = length(first(values_per_dim))
    n < 2 && return Set{Int}()

    jumps = Vector{Float64}(undef, n - 1)
    for i in 1:(n - 1)
        total = 0.0
        matched = 0
        for dim_data in values_per_dim
            length(dim_data) >= n || continue
            a = Float64(dim_data[i])
            b = Float64(dim_data[i + 1])
            if isfinite(a) && isfinite(b)
                total += (b - a)^2
                matched += 1
            end
        end
        jumps[i] = matched == 0 ? NaN : sqrt(total)
    end

    finite_jumps = Float64[d for d in jumps if isfinite(d)]
    length(finite_jumps) < min_steps && return Set{Int}()

    range_sq = 0.0
    for dim_data in values_per_dim
        finite = Float64[Float64(v) for v in dim_data if isfinite(v)]
        isempty(finite) && continue
        range_sq += (maximum(finite) - minimum(finite))^2
    end
    state_range = sqrt(range_sq)

    # Lower-quartile of the step sizes — needed for the "step is much larger
    # than typical" criterion. Use partialsort so we pay O(N) instead of
    # O(N log N) for a single index read; the rest of the sorted order is
    # never consumed.
    quartile_idx = max(1, length(finite_jumps) ÷ 4)
    quartile = partialsort(finite_jumps, quartile_idx)
    threshold = max(multiplier * quartile, range_fraction * state_range)

    # Collect candidate indices and their jump magnitudes. We keep magnitudes
    # so we can prune to the largest jumps if the detector over-fires.
    candidates = Tuple{Int, Float64}[]
    for (i, d) in enumerate(jumps)
        if isfinite(d) && d > threshold
            push!(candidates, (i, d))
        end
    end

    # Hard cap on the number of breaks. The default is effectively no cap
    # (`typemax(Int)`); the parameter is injectable so tests can force pruning
    # and assert it behaves correctly.
    if length(candidates) > max_breaks
        partialsort!(candidates, max_breaks; by=last, rev=true)
        resize!(candidates, max_breaks)
    end

    breaks = Set{Int}()
    for (i, _) in candidates
        push!(breaks, i)
    end
    return breaks
end

"""Read a `:breaks` field from a trace named tuple, defaulting to an empty set."""
_trace_breaks(trace) = hasproperty(trace, :breaks) ? trace.breaks : Set{Int}()

"""Return whether a branch trace has finite points in the requested stability class."""
function _trace_has_stability(trace, stable::Bool)
    return any(i -> trace.stable[i] == stable && isfinite(trace.values[i]), eachindex(trace.values))
end

"""Resolve plotting parameters so phase-expanded branch traces can reconstruct full orbits."""
function _resolve_plot_params(sys::ContinuousODE, params::Vector{Float64})
    _resolve_continuous_params(sys, params)
end

function _resolve_plot_params(sys::DiscreteMap, params::Vector{Float64})
    if !isempty(params)
        return copy(params)
    elseif length(sys.param_names) == 1
        return zeros(Float64, 1)
    end

    error("Pass `params` when phase-expanding branches for $(sys.name), because the plotter needs the fixed parameter values as well.")
end

"""
    _branch_stability_flags(sys, br, branch_points; params, linked_param_indices, solver, …) -> Vector{Bool}

Recompute each branch point's stability with the map criterion |μ| ≤ 1, where μ are the
multipliers of the period-`p` return map DΠ^p. BifurcationKit assesses stability of the
continuation residual `F = Π^p(x) - x` with its equilibrium convention (Re(λ) < 0). For a
map fixed point that convention is wrong: a period-doubling sends a multiplier μ → -1, so the
residual eigenvalue λ = μ - 1 → -2 and never crosses the imaginary axis — the flip (and the
unstable branch beyond it) is silently missed, while folds (μ → +1 ⟹ λ → 0) are caught. We
therefore recompute stability from the multipliers directly so post-period-doubling segments
render as unstable. Falls back to the stored `pt.stable` flag if a point's return fails.
"""
function _branch_stability_flags(sys::DynamicalSystem, br::BranchResult,
                                 branch_points::AbstractVector;
                                 params::Vector{Float64}=Float64[],
                                 linked_param_indices::Vector{Int}=Int[],
                                 solver=Tsit5(),
                                 reltol::Float64=1e-8,
                                 abstol::Float64=1e-8,
                                 tmax::Union{Nothing, Float64}=nothing,
                                 min_crossing_time::Float64=1e-6)
    base_params = _resolve_plot_params(sys, params)
    param_index = something(findfirst(==(br.param_name), sys.param_names), 1)
    proj_dim = state_dim(sys)
    period = max(br.period, 1)
    flags = Vector{Bool}(undef, length(branch_points))
    for (i, pt) in enumerate(branch_points)
        local_params = _inject_param(base_params, param_index, pt.param, linked_param_indices)
        state = _branch_point_state(pt, proj_dim)
        flags[i] = try
            first(_map_stability(sys, state, local_params, period;
                solver=solver, reltol=reltol, abstol=abstol,
                tmax=tmax, min_crossing_time=min_crossing_time))
        catch
            Bool(pt.stable)
        end
    end
    return flags
end

"""Return the sum of squared coordinate differences and finite-coordinate count for one orbit phase."""
function _phase_state_sqdistance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) || return Inf, 0

    total = 0.0
    matched = 0
    for (av, bv) in zip(a, b)
        ai = Float64(av)
        bi = Float64(bv)
        if isfinite(ai) && isfinite(bi)
            total += (ai - bi)^2
            matched += 1
        end
    end

    return matched == 0 ? (Inf, 0) : (total, matched)
end

"""Choose the cyclic orbit-phase shift with the smallest mean squared mismatch per finite coordinate."""
function _orbit_phase_alignment_shift(previous_orbit::Union{Nothing, AbstractVector{<:AbstractVector{<:Real}}},
                                      orbit::AbstractVector{<:AbstractVector{<:Real}})
    period = length(orbit)
    period <= 1 && return 0
    isnothing(previous_orbit) && return 0
    length(previous_orbit) == period || return 0

    best_shift = 0
    best_score = Inf
    best_matched = -1
    for shift in 0:(period - 1)
        score = 0.0
        matched = 0
        for phase in 1:period
            sqdistance, phase_matched = _phase_state_sqdistance(previous_orbit[phase], orbit[mod1(phase + shift, period)])
            if isfinite(sqdistance)
                score += sqdistance
                matched += phase_matched
            end
        end

        matched == 0 && continue
        mean_score = score / matched
        if mean_score + _ORBIT_PHASE_ALIGNMENT_TOL < best_score ||
           (isapprox(mean_score, best_score; atol=_ORBIT_PHASE_ALIGNMENT_TOL, rtol=0.0) && matched > best_matched)
            best_score = mean_score
            best_matched = matched
            best_shift = shift
        end
    end

    return best_shift
end

"""Align orbit phases to the previous branch point by the best cyclic shift."""
function _align_orbit_phases(previous_orbit::Union{Nothing, AbstractVector{<:AbstractVector{<:Real}}},
                             orbit::AbstractVector{<:AbstractVector{<:Real}})
    best_shift = _orbit_phase_alignment_shift(previous_orbit, orbit)
    best_shift == 0 && return orbit
    return [orbit[mod1(phase + best_shift, length(orbit))] for phase in 1:length(orbit)]
end

"""Return one plotting trace per orbit phase for a discrete-map continuation branch."""
function _branch_plot_traces(sys::DiscreteMap, br::BranchResult;
                             orbital::Int=1,
                             params::Vector{Float64}=Float64[],
                             linked_param_indices::Vector{Int}=Int[],
                             recompute_stability::Bool=true,
                             kwargs...)
    branch_points = collect(br.branch.branch)
    pars = Float64[pt.param for pt in branch_points]
    stab = recompute_stability ?
        _branch_stability_flags(sys, br, branch_points; params=params, linked_param_indices=linked_param_indices) :
        Bool[pt.stable for pt in branch_points]
    period = max(br.period, 1)
    if period <= 1
        values = Float64[getproperty(pt, Symbol(:x, orbital)) for pt in branch_points]
        breaks = _phase_jump_break_indices([values])
        return [(params=pars, values=values, stable=stab, breaks=breaks)]
    end

    base_params = _resolve_plot_params(sys, params)
    param_index = something(findfirst(==(br.param_name), sys.param_names), 1)
    dim = sys.dim
    phase_states = [[Float64[] for _ in 1:dim] for _ in 1:period]
    orbit = [Vector{Float64}(undef, dim) for _ in 1:period]
    previous_orbit = [Vector{Float64}(undef, dim) for _ in 1:period]
    has_previous = false

    for pt in branch_points
        local_params = _inject_param(base_params, param_index, pt.param, linked_param_indices)
        current = _branch_point_state(pt, dim)
        for phase in 1:period
            orbit[phase] .= current
            phase < br.period && (current = Array(sys.f(SVector{dim}(current), local_params)))
        end

        shift = _orbit_phase_alignment_shift(has_previous ? previous_orbit : nothing, orbit)
        for phase in 1:period
            aligned_phase = mod1(phase + shift, period)
            for d in 1:dim
                push!(phase_states[phase][d], orbit[aligned_phase][d])
            end
            previous_orbit[phase] .= orbit[aligned_phase]
        end
        has_previous = true
    end

    [(params=pars,
      values=phase_states[phase][orbital],
      stable=stab,
      breaks=_phase_jump_break_indices(phase_states[phase])) for phase in 1:period]
end

"""Return one plotting trace per orbit phase for a continuous-time continuation branch."""
function _branch_plot_traces(sys::ContinuousODE, br::BranchResult;
                             orbital::Int=1,
                             params::Vector{Float64}=Float64[],
                             linked_param_indices::Vector{Int}=Int[],
                             solver=Tsit5(),
                             reltol::Float64=1e-8,
                             abstol::Float64=1e-8,
                             tmax::Union{Nothing, Float64}=nothing,
                             min_crossing_time::Float64=1e-6,
                             recompute_stability::Bool=true)
    branch_points = collect(br.branch.branch)
    pars = Float64[pt.param for pt in branch_points]
    proj_dim = state_dim(sys)
    period = max(br.period, 1)
    stab = recompute_stability ?
        _branch_stability_flags(sys, br, branch_points; params=params,
            linked_param_indices=linked_param_indices, solver=solver, reltol=reltol,
            abstol=abstol, tmax=tmax, min_crossing_time=min_crossing_time) :
        Bool[pt.stable for pt in branch_points]
    if period <= 1
        values = Float64[getproperty(pt, Symbol(:x, orbital)) for pt in branch_points]
        breaks = _phase_jump_break_indices([values])
        return [(params=pars, values=values, stable=stab, breaks=breaks)]
    end

    base_params = _resolve_plot_params(sys, params)
    param_index = something(findfirst(==(br.param_name), sys.param_names), 1)
    phase_states = [[Float64[] for _ in 1:proj_dim] for _ in 1:period]
    orbit = [Vector{Float64}(undef, proj_dim) for _ in 1:period]
    previous_orbit = [Vector{Float64}(undef, proj_dim) for _ in 1:period]
    has_previous = false

    for pt in branch_points
        local_params = _inject_param(base_params, param_index, pt.param, linked_param_indices)
        current = _branch_point_state(pt, proj_dim)
        valid = true
        for phase in 1:period
            if valid
                orbit[phase] .= current
            else
                fill!(orbit[phase], NaN)
            end
            if phase < br.period && valid
                current, valid = _poincare_projected(
                    sys,
                    current,
                    local_params;
                    period=1,
                    solver=solver,
                    reltol=reltol,
                    abstol=abstol,
                    tmax=tmax,
                    min_crossing_time=min_crossing_time
                )
            end
        end

        shift = _orbit_phase_alignment_shift(has_previous ? previous_orbit : nothing, orbit)
        for phase in 1:period
            aligned_phase = mod1(phase + shift, period)
            for d in 1:proj_dim
                push!(phase_states[phase][d], orbit[aligned_phase][d])
            end
            previous_orbit[phase] .= orbit[aligned_phase]
        end
        has_previous = true
    end

    [(params=pars,
      values=phase_states[phase][orbital],
      stable=stab,
      breaks=_phase_jump_break_indices(phase_states[phase])) for phase in 1:period]
end

"""Fallback plotting trace: use the representative point recorded on the branch."""
function _branch_plot_traces(br::BranchResult; orbital::Int=1, kwargs...)
    branch_points = collect(br.branch.branch)
    values = Float64[getproperty(pt, Symbol(:x, orbital)) for pt in branch_points]
    [(params=Float64[pt.param for pt in branch_points],
      values=values,
      stable=Bool[pt.stable for pt in branch_points],
      breaks=_phase_jump_break_indices([values]))]
end

"""Draw one branch trace with separate stable/unstable runs for readability."""
function _plot_branch_trace!(p, trace;
                             stable_color,
                             unstable_color,
                             stable_label::String="",
                             unstable_label::String="",
                             stable_linewidth::Float64=2.2,
                             unstable_linewidth::Float64=1.8,
                             stable_markersize::Float64=2.4,
                             unstable_markersize::Float64=2.1,
                             unstable_markershape=:utriangle,
                             unstable_linestyle=:dash)
    pars = trace.params
    vals = trace.values
    stab = trace.stable
    breaks = _trace_breaks(trace)

    stable_idx = [i for i in eachindex(vals) if stab[i] && isfinite(vals[i])]
    unstable_idx = [i for i in eachindex(vals) if !stab[i] && isfinite(vals[i])]

    for run in _contiguous_runs(stable_idx; break_after=breaks)
        plot!(p, pars[run], vals[run]; color=stable_color, linewidth=stable_linewidth, alpha=0.95, label=stable_label)
        stable_label = ""
    end
    for run in _contiguous_runs(unstable_idx; break_after=breaks)
        plot!(p, pars[run], vals[run]; color=unstable_color, linewidth=unstable_linewidth, linestyle=unstable_linestyle, alpha=0.9, label=unstable_label)
        unstable_label = ""
    end

    !isempty(stable_idx) && scatter!(p, pars[stable_idx], vals[stable_idx]; markersize=stable_markersize, markerstrokewidth=0, color=stable_color, label="")
    !isempty(unstable_idx) && scatter!(p, pars[unstable_idx], vals[unstable_idx]; markersize=unstable_markersize, markershape=unstable_markershape, markerstrokewidth=0, color=unstable_color, label="")
    return p
end

"""
    plot_brute_force(result::BruteForceResult; orbital=1, figsize=(800,500), kwargs...)

Plot a brute-force bifurcation diagram.
"""
function plot_brute_force(result::BruteForceResult; orbital::Int=1, figsize=(800,500), kwargs...)
    p = scatter(result.params, result.points[:, orbital];
        markersize=0.8, markerstrokewidth=0, markeralpha=0.65,
        color=:steelblue4, label="",
        xlabel=string(result.param_name),
        ylabel="x_$orbital",
        title="$(result.system_name) — Brute Force Bifurcation Diagram",
        size=figsize, dpi=160, grid=true, framestyle=:box,
        kwargs...)
    return p
end

"""
    plot_lyapunov_diagram(result::LyapunovDiagramResult; figsize=(850,500), kwargs...)

Plot a 1D largest-Lyapunov diagram with a zero reference line.
"""
function plot_lyapunov_diagram(result::LyapunovDiagramResult;
                               figsize=(850,500),
                               xlabel::Union{Nothing, String}=nothing,
                               ylabel::Union{Nothing, String}=nothing,
                               title::Union{Nothing, String}=nothing,
                               zero_line::Bool=true,
                               kwargs...)
    p = plot(result.params, result.exponents;
        color=:purple4,
        linewidth=1.8,
        label="Largest LE",
        xlabel=something(xlabel, string(result.param_name)),
        ylabel=something(ylabel, "Largest Lyapunov exponent"),
        title=something(title, "$(result.system_name) — Lyapunov Diagram"),
        size=figsize,
        dpi=160,
        grid=true,
        framestyle=:box,
        kwargs...)
    if zero_line
        hline!(p, [0.0]; color=:black, linestyle=:dash, linewidth=1.0, alpha=0.7, label="0")
    end
    return p
end

"""
    plot_lyapunov_spectrum(result::LyapunovSpectrumResult; figsize=(850,500), kwargs...)

Plot the convergence of the finite-time Lyapunov exponents toward the reported
spectrum. Each line is one exponent's running estimate against the accumulation
horizon (iterations for maps, flow time for ODEs).
"""
function plot_lyapunov_spectrum(result::LyapunovSpectrumResult;
                                figsize=(850,500),
                                xlabel::Union{Nothing, String}=nothing,
                                ylabel::Union{Nothing, String}=nothing,
                                title::Union{Nothing, String}=nothing,
                                zero_line::Bool=true,
                                kwargs...)
    n = size(result.convergence, 1)
    k = size(result.convergence, 2)
    n > 0 || throw(ArgumentError(
        "plot_lyapunov_spectrum received a result with no accumulated intervals " *
        "(estimation_status = :$(result.estimation_status)); nothing to plot."))
    horizon_unit = result.kind == :continuous_flow ? result.total_time / max(n, 1) : 1.0
    horizon = collect(1:n) .* horizon_unit
    default_x = result.kind == :continuous_flow ? "Flow time" : "Iterations"
    p = plot(;
        xlabel=something(xlabel, default_x),
        ylabel=something(ylabel, "Finite-time Lyapunov exponent"),
        title=something(title, "$(result.system_name) — Lyapunov Spectrum"),
        size=figsize,
        dpi=160,
        grid=true,
        framestyle=:box,
        kwargs...)
    for i in 1:k
        plot!(p, horizon, result.convergence[:, i];
            linewidth=1.8,
            label="λ$(i) → $(round(result.exponents[i]; digits=4))")
    end
    if zero_line
        hline!(p, [0.0]; color=:black, linestyle=:dash, linewidth=1.0, alpha=0.7, label="0")
    end
    return p
end

"""
    plot_branches(results::Vector{BranchResult}; orbital=1, figsize=(800,500), kwargs...)

Plot continuation branches. Blue = stable, red = unstable.
"""
function plot_branches(results::Vector{BranchResult};
                       orbital::Int=1,
                       figsize=(800,500),
                       system::Union{Nothing, DynamicalSystem}=nothing,
                       params::Vector{Float64}=Float64[],
                       linked_param_indices::Vector{Int}=Int[],
                       solver=Tsit5(),
                       reltol::Float64=1e-8,
                       abstol::Float64=1e-8,
                       tmax::Union{Nothing, Float64}=nothing,
                       min_crossing_time::Float64=1e-6,
                       recompute_stability::Bool=true,
                       unstable_linestyle=:dash,
                       stable_linewidth::Float64=2.2,
                       unstable_linewidth::Float64=1.8,
                       stable_markersize::Float64=2.4,
                       unstable_markersize::Float64=2.1,
                       kwargs...)
    p = plot(; xlabel="param", ylabel="x_$orbital",
             title="Continuation Branches",
             size=figsize, dpi=160, grid=true, framestyle=:box, kwargs...)

    stable_label_pending = true
    unstable_label_pending = true
    for br in results
        traces = isnothing(system) ?
            _branch_plot_traces(br; orbital=orbital) :
            _branch_plot_traces(system, br;
                orbital=orbital,
                params=params,
                linked_param_indices=linked_param_indices,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time,
                recompute_stability=recompute_stability)

        for trace in traces
            stable_label = stable_label_pending && _trace_has_stability(trace, true) ? "Stable" : ""
            unstable_label = unstable_label_pending && _trace_has_stability(trace, false) ? "Unstable" : ""
            _plot_branch_trace!(p, trace;
                stable_color=:blue3,
                unstable_color=:red3,
                stable_label=stable_label,
                unstable_label=unstable_label,
                stable_linewidth=stable_linewidth,
                unstable_linewidth=unstable_linewidth,
                stable_markersize=stable_markersize,
                unstable_markersize=unstable_markersize,
                unstable_linestyle=unstable_linestyle)
            stable_label_pending &= isempty(stable_label)
            unstable_label_pending &= isempty(unstable_label)
        end
    end
    return p
end

"""
    plot_overlay(bf_result::BruteForceResult, branch_results::Vector{BranchResult};
                 orbital=1, kwargs...)

Overlay continuation branches on top of a brute-force diagram.
"""
function plot_overlay(bf_result::BruteForceResult, branch_results::Vector{BranchResult};
                      orbital::Int=1,
                      figsize=(900,600),
                      system::Union{Nothing, DynamicalSystem}=nothing,
                      params::Vector{Float64}=Float64[],
                      linked_param_indices::Vector{Int}=Int[],
                      solver=Tsit5(),
                      reltol::Float64=1e-8,
                      abstol::Float64=1e-8,
                      tmax::Union{Nothing, Float64}=nothing,
                      min_crossing_time::Float64=1e-6,
                      recompute_stability::Bool=true,
                      cloud_markersize::Real=1.2,
                      cloud_markeralpha::Real=0.4,
                      cloud_color=:gray25,
                      unstable_linestyle=:dash,
                      stable_linewidth::Float64=2.4,
                      unstable_linewidth::Float64=2.0,
                      stable_markersize::Float64=2.6,
                      unstable_markersize::Float64=2.2,
                      kwargs...)
    # Brute-force cloud (visible backdrop behind the continuation branches)
    p = scatter(bf_result.params, bf_result.points[:, orbital];
        markersize=cloud_markersize, markerstrokewidth=0, markeralpha=cloud_markeralpha,
        color=cloud_color, label="Brute Force",
        xlabel=string(bf_result.param_name),
        ylabel="x_$orbital",
        title="$(bf_result.system_name) — Overlay",
        size=figsize, dpi=160, grid=true, framestyle=:box,
        kwargs...)

    # Branches on top
    stable_label_pending = true
    unstable_label_pending = true
    for br in branch_results
        traces = isnothing(system) ?
            _branch_plot_traces(br; orbital=orbital) :
            _branch_plot_traces(system, br;
                orbital=orbital,
                params=params,
                linked_param_indices=linked_param_indices,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time,
                recompute_stability=recompute_stability)

        for trace in traces
            stable_label = stable_label_pending && _trace_has_stability(trace, true) ? "Stable" : ""
            unstable_label = unstable_label_pending && _trace_has_stability(trace, false) ? "Unstable" : ""
            _plot_branch_trace!(p, trace;
                stable_color=:blue3,
                unstable_color=:red3,
                stable_linewidth=stable_linewidth,
                unstable_linewidth=unstable_linewidth,
                stable_markersize=stable_markersize,
                unstable_markersize=unstable_markersize,
                stable_label=stable_label,
                unstable_label=unstable_label,
                unstable_linestyle=unstable_linestyle)
            stable_label_pending &= isempty(stable_label)
            unstable_label_pending &= isempty(unstable_label)
        end
    end
    return p
end

"""Discrete color gradient and ticks for integer-period heatmaps."""
function _period_heatmap_style(max_period::Int; zero_label::String="0 / chaotic")
    periods = 0:max_period
    labels = [period == 0 ? zero_label : string(period) for period in periods]
    return (
        color=cgrad(:tab10, max(max_period + 1, 2), categorical=true),
        clims=(-0.5, max_period + 0.5),
        colorbar_ticks=(collect(periods), labels)
    )
end

"""
    plot_basins(result::BasinsResult; figsize=(700,600), kwargs...)

Plot basins of attraction as a colored grid where color indicates detected periodicity.
"""
function plot_basins(result::BasinsResult;
                     figsize=(700,600),
                     xlabel::Union{Nothing, String}=nothing,
                     ylabel::Union{Nothing, String}=nothing,
                     title::Union{Nothing, String}=nothing,
                     xscale::Float64=1.0,
                     yscale::Float64=1.0,
                     zero_label::String="0 / chaotic",
                     kwargs...)
    period_style = _period_heatmap_style(result.max_period; zero_label=zero_label)
    p = heatmap(result.x_grid .* xscale, result.y_grid .* yscale, result.periodicity';
        color=period_style.color, clims=period_style.clims, colorbar_ticks=period_style.colorbar_ticks,
        xlabel=something(xlabel, "x_$(result.x_index)"),
        ylabel=something(ylabel, "x_$(result.y_index)"),
        title=something(title, "$(result.system_name) — Basins of Attraction (a=$(result.bif_param))"),
        size=figsize, colorbar_title="Period",
        kwargs...)
    return p
end

"""
    plot_bifurcation_map(result::BifurcationMapResult; figsize=(700,600), kwargs...)

Plot a 2D bifurcation map where color indicates detected periodicity
across a two-parameter grid.
"""
function plot_bifurcation_map(result::BifurcationMapResult;
                              figsize=(700,600),
                              xlabel::Union{Nothing, String}=nothing,
                              ylabel::Union{Nothing, String}=nothing,
                              title::Union{Nothing, String}=nothing,
                              xscale::Float64=1.0,
                              yscale::Float64=1.0,
                              zero_label::String="0 / chaotic",
                              kwargs...)
    period_style = _period_heatmap_style(result.max_period; zero_label=zero_label)
    p = heatmap(result.a_grid .* xscale, result.b_grid .* yscale, result.periodicity';
        color=period_style.color, clims=period_style.clims, colorbar_ticks=period_style.colorbar_ticks,
        xlabel=something(xlabel, string(result.param_names[1])),
        ylabel=something(ylabel, string(result.param_names[2])),
        title=something(title, "$(result.system_name) — 2D Bifurcation Map"),
        size=figsize, colorbar_title="Period",
        kwargs...)
    return p
end

"""
    plot_lyapunov_field(result::LyapunovFieldResult; zero_contour=false, kwargs...)
    plot_lyapunov_field(result::BifurcationMapResult; kwargs...)

Plot a largest-Lyapunov field over a 2D parameter grid.
"""
function plot_lyapunov_field(result::LyapunovFieldResult;
                             figsize=(760,620),
                             xlabel::Union{Nothing, String}=nothing,
                             ylabel::Union{Nothing, String}=nothing,
                             title::Union{Nothing, String}=nothing,
                             xscale::Float64=1.0,
                             yscale::Float64=1.0,
                             zero_contour::Bool=false,
                             kwargs...)
    p = heatmap(result.a_grid .* xscale, result.b_grid .* yscale, result.exponents';
        color=:balance,
        xlabel=something(xlabel, string(result.param_names[1])),
        ylabel=something(ylabel, string(result.param_names[2])),
        title=something(title, "$(result.system_name) — Lyapunov Field"),
        size=figsize,
        colorbar_title="Largest LE",
        kwargs...)
    if zero_contour
        contour!(p, result.a_grid .* xscale, result.b_grid .* yscale, result.exponents';
            levels=[0.0],
            linewidth=1.4,
            color=:black,
            label="")
    end
    return p
end

plot_lyapunov_field(result::BifurcationMapResult; kwargs...) = plot_lyapunov_field(lyapunov_field(result); kwargs...)

"""Return contiguous runs of valid codim-2 curve samples."""
function _codim2_valid_runs(result::Codim2CurveResult)
    indices = [idx for idx in eachindex(result.valid_mask) if result.valid_mask[idx]]
    return _contiguous_runs(indices)
end

function _codim2_curve_label(result::Codim2CurveResult)
    kind = uppercase(String(result.bifurcation_kind))
    return "$kind (period $(result.period))"
end

"""
    plot_codim2(result::Codim2CurveResult; base=nothing, kwargs...)
    plot_codim2(results::Vector{Codim2CurveResult}; base=nothing, kwargs...)

Overlay one or more codimension-2 bifurcation curves on an optional heatmap base
(`LyapunovFieldResult`, `BifurcationMapResult`, or an existing `Plots.Plot`).
"""
function plot_codim2(result::Codim2CurveResult; kwargs...)
    return plot_codim2([result]; kwargs...)
end

function plot_codim2(results::Vector{Codim2CurveResult};
                     base::Union{Nothing, LyapunovFieldResult, BifurcationMapResult, Plots.Plot}=nothing,
                     figsize=(760,620),
                     xlabel::Union{Nothing, String}=nothing,
                     ylabel::Union{Nothing, String}=nothing,
                     title::Union{Nothing, String}=nothing,
                     xscale::Float64=1.0,
                     yscale::Float64=1.0,
                     linewidth::Float64=2.4,
                     linecolors=[:black, :black, :darkorange3, :purple4],
                     linestyles=[:solid, :dash, :dot, :dashdot],
                     labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
                     kwargs...)
    isempty(results) && throw(ArgumentError("plot_codim2 requires at least one curve result."))
    isnothing(labels) || length(labels) == length(results) || throw(ArgumentError(
        "plot_codim2 received $(length(labels)) labels for $(length(results)) curves."))
    first_result = first(results)
    p = if isnothing(base)
        plot(;
            xlabel=something(xlabel, string(first_result.param_names[1])),
            ylabel=something(ylabel, string(first_result.param_names[2])),
            title=something(title, "$(first_result.system_name) — Codimension-2 Curve"),
            size=figsize,
            dpi=160,
            grid=true,
            framestyle=:box,
            kwargs...
        )
    elseif base isa LyapunovFieldResult
        plot_lyapunov_field(base;
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            xscale=xscale,
            yscale=yscale,
            kwargs...)
    elseif base isa BifurcationMapResult
        plot_bifurcation_map(base;
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            xscale=xscale,
            yscale=yscale,
            kwargs...)
    else
        deepcopy(base)
    end

    for (idx, result) in enumerate(results)
        color = linecolors[mod1(idx, length(linecolors))]
        linestyle = linestyles[mod1(idx, length(linestyles))]
        label = isnothing(labels) ? _codim2_curve_label(result) : labels[idx]
        for run in _codim2_valid_runs(result)
            plot!(p,
                result.primary_values[run] .* xscale,
                result.secondary_values[run] .* yscale;
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)
            label = ""
        end
    end
    return p
end

"""Ordered samples of a defining-system codim-2 curve form a single run."""
_codim2_valid_runs(result::Codim2ContinuationResult) =
    isempty(result.primary_values) ? UnitRange{Int}[] : [1:length(result.primary_values)]

function _codim2_curve_label(result::Codim2ContinuationResult)
    kind = uppercase(String(result.bifurcation_kind))
    return "$kind (period $(result.period), continued)"
end

function plot_codim2(result::Codim2ContinuationResult; kwargs...)
    return plot_codim2([result]; kwargs...)
end

function plot_codim2(results::Vector{Codim2ContinuationResult};
                     base::Union{Nothing, LyapunovFieldResult, BifurcationMapResult, Plots.Plot}=nothing,
                     figsize=(760,620),
                     xlabel::Union{Nothing, String}=nothing,
                     ylabel::Union{Nothing, String}=nothing,
                     title::Union{Nothing, String}=nothing,
                     xscale::Float64=1.0,
                     yscale::Float64=1.0,
                     linewidth::Float64=2.4,
                     linecolors=[:black, :black, :darkorange3, :purple4],
                     linestyles=[:solid, :dash, :dot, :dashdot],
                     labels::Union{Nothing, AbstractVector{<:AbstractString}}=nothing,
                     kwargs...)
    isempty(results) && throw(ArgumentError("plot_codim2 requires at least one curve result."))
    isnothing(labels) || length(labels) == length(results) || throw(ArgumentError(
        "plot_codim2 received $(length(labels)) labels for $(length(results)) curves."))
    first_result = first(results)
    p = if isnothing(base)
        plot(;
            xlabel=something(xlabel, string(first_result.param_names[1])),
            ylabel=something(ylabel, string(first_result.param_names[2])),
            title=something(title, "$(first_result.system_name) — Codimension-2 Curve"),
            size=figsize,
            dpi=160,
            grid=true,
            framestyle=:box,
            kwargs...
        )
    elseif base isa LyapunovFieldResult
        plot_lyapunov_field(base;
            figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title,
            xscale=xscale, yscale=yscale, kwargs...)
    elseif base isa BifurcationMapResult
        plot_bifurcation_map(base;
            figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title,
            xscale=xscale, yscale=yscale, kwargs...)
    else
        deepcopy(base)
    end

    for (idx, result) in enumerate(results)
        color = linecolors[mod1(idx, length(linecolors))]
        linestyle = linestyles[mod1(idx, length(linestyles))]
        label = isnothing(labels) ? _codim2_curve_label(result) : labels[idx]
        for run in _codim2_valid_runs(result)
            plot!(p,
                result.primary_values[run] .* xscale,
                result.secondary_values[run] .* yscale;
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)
            label = ""
        end
    end
    return p
end

"""
    plot_overlay_heatmap(base, curves; kwargs...)

Generic helper for overlaying codim-2 curves on a heatmap-style base result.
"""
plot_overlay_heatmap(base::Union{LyapunovFieldResult, BifurcationMapResult, Plots.Plot},
                     curves::Union{Codim2CurveResult, Vector{Codim2CurveResult}};
                     kwargs...) =
    plot_codim2(curves isa Codim2CurveResult ? [curves] : curves; base=base, kwargs...)

"""
    plot_panel_grid(panels; layout=(rows, cols), figsize=(w, h), kwargs...)

Compose pre-built panels into a grid layout.
"""
function plot_panel_grid(panels::AbstractVector;
                         layout=(1, length(panels)),
                         figsize=(1200,700),
                         kwargs...)
    isempty(panels) && throw(ArgumentError("plot_panel_grid requires at least one panel."))
    return plot(panels...; layout=layout, size=figsize, dpi=160, kwargs...)
end

"""
    plot_seed_pair_composite(left, right; kwargs...)

Convenience helper for the common side-by-side negative/positive seed pairing.
"""
function plot_seed_pair_composite(left, right; figsize=(1450,620), kwargs...)
    return plot_panel_grid([left, right]; layout=(1, 2), figsize=figsize, kwargs...)
end

"""
    plot_phase_portrait(result::PhasePortraitResult; x_index=1, y_index=2, show_poincare=true, kwargs...)

Plot a continuous-time phase portrait with optional Poincaré section markers.
"""
function plot_phase_portrait(result::PhasePortraitResult;
                             x_index::Int=1,
                             y_index::Int=2,
                             show_poincare::Bool=true,
                             figsize=(700,600),
                             xlabel::Union{Nothing, String}=nothing,
                             ylabel::Union{Nothing, String}=nothing,
                             title::Union{Nothing, String}=nothing,
                             trajectory_label::String="Trajectory",
                             poincare_label::String="Poincaré points",
                             xscale::Float64=1.0,
                             yscale::Float64=1.0,
                             kwargs...)
    traj_dim = size(result.trajectory, 2)
    traj_dim >= max(x_index, y_index) || error("Phase portrait axis index exceeds trajectory dimension: x_index=$(x_index), y_index=$(y_index), trajectory_dim=$(traj_dim).")
    x_label = something(xlabel, string(result.state_names[x_index]))
    y_label = something(ylabel, string(result.state_names[y_index]))
    p = plot(result.trajectory[:, x_index] .* xscale, result.trajectory[:, y_index] .* yscale;
        linewidth=1.8,
        color=:steelblue4,
        label=trajectory_label,
        xlabel=x_label,
        ylabel=y_label,
        title=something(title, "$(result.system_name) — Phase Portrait"),
        size=figsize,
        dpi=160,
        grid=true,
        framestyle=:box,
        kwargs...)

    if show_poincare && size(result.poincare_points, 1) > 0
        scatter!(p, result.poincare_points[:, x_index] .* xscale, result.poincare_points[:, y_index] .* yscale;
            color=:red3,
            markersize=4,
            markerstrokewidth=0,
            label=poincare_label)
    end

    return p
end

"""
    plot_power_spectrum(result::PowerSpectrumResult; layout=:row, kwargs...)

Plot the retained time-domain tail beside its one-sided power spectrum.
"""
function plot_power_spectrum(result::PowerSpectrumResult;
                             figsize=(1000,420),
                             time_xlabel::String="t",
                             signal_ylabel::Union{Nothing, String}=nothing,
                             freq_xlabel::String="Frequency",
                             power_ylabel::String="Power",
                             title::Union{Nothing, String}=nothing,
                             spectrum_yscale::Symbol=:identity,
                             time_kwargs::NamedTuple=NamedTuple(),
                             spectrum_kwargs::NamedTuple=NamedTuple(),
                             plot_kwargs::NamedTuple=NamedTuple(),
                             kwargs...)
    left = plot(result.t, result.signal;
        color=:steelblue4,
        linewidth=1.7,
        label="x_$(result.state_index)",
        xlabel=time_xlabel,
        ylabel=something(signal_ylabel, "x_$(result.state_index)"),
        title="Time tail",
        grid=true,
        framestyle=:box,
        time_kwargs...,
        kwargs...)
    right = plot(result.frequency, result.power;
        color=:darkorange3,
        linewidth=1.7,
        label="Power",
        xlabel=freq_xlabel,
        ylabel=power_ylabel,
        title="One-sided spectrum",
        yscale=spectrum_yscale,
        grid=true,
        framestyle=:box,
        spectrum_kwargs...,
        kwargs...)
    return plot(left, right;
        layout=(1, 2),
        size=figsize,
        dpi=160,
        plot_title=something(title, "$(result.system_name) — Power Spectrum"),
        plot_kwargs...)
end

# --- Public trace-data helpers ---
# The data-building helpers behind the Plots recipes, published so consumers (the workbench, custom
# plotting) can assemble their own trace payloads. Their internal helpers (`_contiguous_runs`,
# `_phase_state_sqdistance`, `_resolve_continuous_params`) stay private — only these call them.
"""    branch_plot_traces(sys, br::BranchResult; orbital=1, params, linked_param_indices) — per-phase plot traces."""
const branch_plot_traces = _branch_plot_traces
"""    resolve_plot_params(sys, params::Vector{Float64}) -> Vector{Float64}"""
const resolve_plot_params = _resolve_plot_params
"""    branch_point_state(point, dim::Int) -> Vector{Float64}"""
const branch_point_state = _branch_point_state
"""    orbit_phase_alignment_shift(previous_orbit, orbit) -> Int — phase shift aligning an orbit to its predecessor."""
const orbit_phase_alignment_shift = _orbit_phase_alignment_shift
"""    phase_jump_break_indices(values_per_dim; ...) -> Set{Int} — indices where a phase trace should break."""
const phase_jump_break_indices = _phase_jump_break_indices
"""    trace_breaks(trace) -> Set{Int} — break indices carried on a trace (empty when absent)."""
const trace_breaks = _trace_breaks
"""    codim2_curve_label(result::Codim2CurveResult) -> String"""
const codim2_curve_label = _codim2_curve_label
"""    codim2_valid_runs(result::Codim2CurveResult) — the valid runs of a codim-2 curve."""
const codim2_valid_runs = _codim2_valid_runs
