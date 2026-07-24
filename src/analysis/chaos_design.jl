_cd_log!(log, msg) = isnothing(log) ? nothing : log(String(msg))

"""
    spectral_flatness(power::AbstractVector{<:Real}) -> Float64

Standard Wiener spectral flatness of the supplied one-sided power-spectrum bins
(geometric mean divided by arithmetic mean). A value near 1 indicates noise-like
broadband content; a value near 0 indicates tonal or sparse energy.

The input must **not** include the DC bin — pass `power[2:end]` from an `rfft` result.

Returns `0.0` for an empty or all-zero spectrum. Non-finite or negative power is
rejected. The calculation normalizes by the largest bin before applying an
`eps(Float64)` floor, preserving scale invariance while avoiding `log(0)`.
"""
function spectral_flatness(power::AbstractVector{<:Real})::Float64
    isempty(power) && return 0.0
    all(p -> isfinite(p) && p >= 0.0, power) || throw(ArgumentError(
        "spectral_flatness requires finite, non-negative power bins."))
    max_power = maximum(power)
    max_power == 0.0 && return 0.0
    normalized = Float64.(power) ./ max_power
    arith = mean(normalized)
    log_geo = sum(log(max(p, eps(Float64))) for p in normalized) / length(normalized)
    return clamp(exp(log_geo) / arith, 0.0, 1.0)
end

function _cd_apply_design(
    base::Vector{Float64},
    variables::Vector{ChaosDesignVariable},
    values::Vector{Float64},
)::Vector{Float64}
    required = isempty(variables) ? length(base) :
               max(length(base), maximum(v.param_index for v in variables))
    p = length(base) >= required ? copy(base) :
        vcat(base, zeros(Float64, required - length(base)))
    for (i, var) in enumerate(variables)
        p[var.param_index] = values[i]
    end
    return p
end

function _cd_build_candidate_config(
    config::ChaosDesignConfig,
    design_values::Vector{Float64},
)::RobustChaosConfig
    op   = config.operating_config
    vars = config.variables

    old_lyapunov = op.lyapunov
    new_lyapunov = Accessors.@set old_lyapunov.fixed_params =
        _cd_apply_design(old_lyapunov.fixed_params, vars, design_values)
    old_atlas = op.atlas
    old_brute_force = old_atlas.brute_force
    new_brute_force = Accessors.@set old_brute_force.fixed_params =
        _cd_apply_design(old_brute_force.fixed_params, vars, design_values)
    new_atlas = Accessors.@set old_atlas.brute_force = new_brute_force
    old_basins = op.basins
    new_bas_fp = _cd_apply_design(old_basins.fixed_params, vars, design_values)
    bif_val  = old_basins.bif_param
    bif_slot = old_basins.param_index
    linked   = op.lyapunov.linked_param_indices
    for idx in vcat([bif_slot], linked)
        while length(new_bas_fp) < idx
            push!(new_bas_fp, 0.0)
        end
        new_bas_fp[idx] = bif_val
    end

    new_basins = Accessors.@set old_basins.fixed_params = new_bas_fp

    return RobustChaosConfig(
        lyapunov=new_lyapunov,
        atlas=new_atlas,
        basins=new_basins,
        min_lyapunov_positive_fraction=op.min_lyapunov_positive_fraction,
        min_lyapunov_resolved_fraction=op.min_lyapunov_resolved_fraction,
        min_chaotic_basin_fraction=op.min_chaotic_basin_fraction,
        min_basin_resolved_fraction=op.min_basin_resolved_fraction,
    )
end

function _cd_signal_discrete(
    sys::DiscreteMap,
    params::Vector{Float64},
    cfg::ChaosDesignSignalConfig,
    initial_point::Union{Nothing, AbstractVector},
)
    total_iters = cfg.discrete_transient + cfg.discrete_samples * cfg.discrete_sample_interval
    orbit = _sample_discrete_orbit(
        sys, params;
        initial_point,
        iterations=total_iters,
        transient=cfg.discrete_transient,
        amplitude_cutoff=cfg.divergence_cutoff,
    )

    if orbit.diverged || isempty(orbit.points)
        return (:diverged, nothing, nothing)
    end

    raw = [pt[cfg.state_index] for pt in orbit.points]
    signal = raw[1:cfg.discrete_sample_interval:end]

    if !all(isfinite, signal) || any(abs(x) >= cfg.divergence_cutoff for x in signal)
        return (:diverged, nothing, nothing)
    end
    length(signal) < 4 && return (:insufficient_samples, nothing, nothing)

    amp = maximum(signal) - minimum(signal)
    !isfinite(amp) && return (:diverged, nothing, nothing)

    win      = _spectrum_window(cfg.discrete_window, length(signal))
    windowed = (signal .- mean(signal)) .* win
    power    = Float64.(abs2.(rfft(windowed)))
    flat     = spectral_flatness(length(power) > 1 ? @view(power[2:end]) : Float64[])

    return (:ok, amp, flat)
end

function _cd_signal_continuous(
    sys::ContinuousODE,
    params::Vector{Float64},
    cfg::ChaosDesignSignalConfig;
    initial_point::Union{Nothing, AbstractVector},
    solver,
    reltol::Float64,
    abstol::Float64,
)
    ps_base = if !isnothing(cfg.continuous)
        cfg.continuous
    else
        PowerSpectrumConfig(
            time_start=0.0,
            time_stop=200.0,
            dt=0.05,
            tail_fraction=0.5,
            window=:hann,
            state_index=cfg.state_index,
        )
    end
    ps = ps_base.state_index == cfg.state_index ? ps_base :
        (Accessors.@set ps_base.state_index = cfg.state_index)

    result = power_spectrum(
        sys,
        ps;
        params,
        initial_point,
        solver,
        reltol,
        abstol,
    )

    sig = result.signal
    if !all(isfinite, sig) || any(abs(x) >= cfg.divergence_cutoff for x in sig)
        return (:diverged, nothing, nothing)
    end
    length(sig) < 4 && return (:insufficient_samples, nothing, nothing)

    amp  = maximum(sig) - minimum(sig)
    !isfinite(amp) && return (:diverged, nothing, nothing)
    flat = spectral_flatness(length(result.power) > 1 ? @view(result.power[2:end]) : Float64[])
    return (:ok, amp, flat)
end

function _cd_amplitude_score(amp::Union{Nothing, Float64}, t::ChaosDesignTarget)::Float64
    isnothing(amp) && return 0.0
    !isfinite(amp) && return 0.0
    amp >= t.min_amplitude && amp <= t.max_amplitude && return 1.0
    amp < t.min_amplitude &&
        return t.min_amplitude == 0.0 ? 0.0 : clamp(amp / t.min_amplitude, 0.0, 1.0)
    # amp > t.max_amplitude
    t.max_amplitude == Inf && return 1.0
    return clamp(t.max_amplitude / amp, 0.0, 1.0)
end

function _cd_flatness_score(flat::Union{Nothing, Float64}, t::ChaosDesignTarget)::Float64
    isnothing(flat) && return 0.0
    !isfinite(flat) && return 0.0
    t.min_spectral_flatness == 0.0 && return 1.0
    return clamp(flat / t.min_spectral_flatness, 0.0, 1.0)
end

function _cd_scores(
    cert::RobustChaosCertificate,
    signal_status::Symbol,
    amp::Union{Nothing, Float64},
    flat::Union{Nothing, Float64},
    t::ChaosDesignTarget,
)
    rob   = cert.robustness_score
    a_scr = _cd_amplitude_score(amp, t)
    f_scr = _cd_flatness_score(flat, t)

    feasible = (
        cert.overall_verdict == :certified &&
        rob >= t.min_robustness_score &&
        signal_status == :ok &&
        !isnothing(amp) && isfinite(amp) &&
        amp >= t.min_amplitude && amp <= t.max_amplitude &&
        !isnothing(flat) && isfinite(flat) &&
        flat >= t.min_spectral_flatness
    )

    return (
        feasible=feasible,
        robust_score=rob,
        amplitude_score=a_scr,
        flatness_score=f_scr,
        objective=rob * a_scr * f_scr,
    )
end

function _cd_evaluate(
    sys::Union{DiscreteMap, ContinuousODE},
    config::ChaosDesignConfig,
    design_values::Vector{Float64};
    solver,
    reltol::Float64,
    abstol::Float64,
    initial_point::Union{Nothing, AbstractVector{<:Real}},
    log,
)::ChaosDesignCandidate
    rc_cfg = _cd_build_candidate_config(config, design_values)
    cert   = robust_chaos_certificate(sys, rc_cfg; solver=solver, reltol=reltol,
                                      abstol=abstol, initial_point=initial_point, log=log)

    rep_params = build_basins_params(rc_cfg.basins)

    sig_status, amp, flat =
        if sys isa DiscreteMap
            _cd_signal_discrete(sys, rep_params, config.signal, initial_point)
        else
            _cd_signal_continuous(sys, rep_params, config.signal;
                                   initial_point,
                                   solver=solver, reltol=reltol, abstol=abstol)
        end

    s = _cd_scores(cert, sig_status, amp, flat, config.target)
    return ChaosDesignCandidate(
        design_values, cert, sig_status, amp, flat,
        s.feasible, s.robust_score, s.amplitude_score, s.flatness_score, s.objective,
    )
end

function _cd_cartesian_product(axes::Vector{AbstractVector{Float64}})
    isempty(axes) && throw(ArgumentError("Chaos design requires at least one search axis."))
    return (Float64[values...] for values in Iterators.product(axes...))
end

function _cd_coarse_grid(config::ChaosDesignConfig)
    axes = AbstractVector{Float64}[
        range(v.lower, v.upper; length=config.samples_per_axis)
        for v in config.variables
    ]
    return _cd_cartesian_product(axes)
end

function _cd_refined_grid(
    config::ChaosDesignConfig,
    center::Vector{Float64},
    hw::Vector{Float64},
)
    axes = AbstractVector{Float64}[]
    for (i, var) in enumerate(config.variables)
        lo = max(var.lower, center[i] - hw[i])
        hi = min(var.upper, center[i] + hw[i])
        # If clamping collapses the interval, use only the center point.
        push!(axes, lo >= hi ? [center[i]] :
              range(lo, hi; length=config.samples_per_axis))
    end
    return _cd_cartesian_product(axes)
end

"""
    _cd_rank_candidates(candidates) -> Vector{ChaosDesignCandidate}

Sort candidates: feasible first, then descending objective, then lexicographic
design values for deterministic tie-breaking.
"""
function _cd_rank_candidates(
    candidates::Vector{ChaosDesignCandidate},
)::Vector{ChaosDesignCandidate}
    isempty(candidates) && return ChaosDesignCandidate[]
    return sort(candidates; by = c -> (
        c.feasible ? 0 : 1,
        -c.objective,
        Tuple(c.design_values),
    ))
end

"""
    design_chaos_source(sys, config; kwargs...) -> ChaosDesignResult

Search design-parameter space for configurations satisfying a target operating-band
robust-chaos specification with prescribed signal amplitude and spectral properties.

Each candidate in the Cartesian design space is evaluated by:
1. Building a `RobustChaosConfig` from `config.operating_config` with candidate design
   values substituted into every nested parameter vector, then calling
   `robust_chaos_certificate`.
2. Sampling a representative signal at `config.operating_config.basins.bif_param` and
   computing peak-to-peak amplitude and Wiener spectral flatness (non-DC bins).

The search uses a deterministic coarse-to-fine Cartesian grid: an initial coarse grid
spans the full bounds of each design variable; the top `survivors_per_level` candidates
are carried into each refinement level where a finer grid is centred on each survivor.
Evaluations are deduplicated on exact parameter tuples and the `max_evaluations` budget
is never exceeded. Ranking puts feasible candidates first, then descending objective,
then lexicographic design values for deterministic tie-breaking.

# Keyword arguments
- `solver`: ODE solver (default `Tsit5()`; ignored for `DiscreteMap` systems).
- `reltol`, `abstol`: ODE solver tolerances.
- `initial_point`: Optional starting state shared by each certificate and representative
  signal evaluation. For maps with a discontinuous modulo, a non-degenerate point avoids
  numerically problematic fixed points.
- `log`: Optional `Function(String)` for progress messages using the library's standard
  logging style.
"""
function design_chaos_source(
    sys::Union{DiscreteMap, ContinuousODE},
    config::ChaosDesignConfig;
    solver=Tsit5(),
    reltol::Float64=1e-8,
    abstol::Float64=1e-8,
    initial_point::Union{Nothing, AbstractVector{<:Real}}=nothing,
    log::Union{Nothing, Function}=nothing,
)::ChaosDesignResult
    config.signal.state_index <= sys.dim || throw(ArgumentError(
        "ChaosDesignSignalConfig.state_index $(config.signal.state_index) must lie " *
        "in 1:$(sys.dim) for system '$(sys.name)'."
    ))
    for variable in config.variables
        variable.param_index <= length(sys.param_names) || throw(ArgumentError(
            "ChaosDesignVariable $(variable.name) uses parameter index " *
            "$(variable.param_index), but $(sys.name) declares only " *
            "$(length(sys.param_names)) parameter slots."))
    end

    ts = now()
    all_cands  = ChaosDesignCandidate[]
    seen_keys  = Set{Tuple{Vararg{Float64}}}()
    n_eval     = 0
    budget_hit = false
    levels_done = 0

    # Evaluate one tuple; skip budget-exceeded or duplicates; return false if budget.
    function try_eval!(vals::Vector{Float64})::Bool
        n_eval >= config.max_evaluations && return false
        key = Tuple(vals)
        key in seen_keys && return true
        push!(seen_keys, key)
        n_eval += 1
        _cd_log!(log, "design_chaos_source: eval $n_eval/$(config.max_evaluations) " *
                 "[$(join(round.(vals; sigdigits=5), ", "))]")
        cand = _cd_evaluate(sys, config, vals;
                             solver=solver, reltol=reltol, abstol=abstol,
                             initial_point=initial_point, log=nothing)
        push!(all_cands, cand)
        return true
    end

    # Level 0: coarse grid over full bounds.
    for tup in _cd_coarse_grid(config)
        if n_eval >= config.max_evaluations
            budget_hit = true
            break
        end
        try_eval!(tup)
    end

    # Per-axis initial spacing for half-width computation in refinement.
    coarse_hw = [(v.upper - v.lower) / (2 * (config.samples_per_axis - 1))
                 for v in config.variables]

    # Refinement levels: zoom in on top survivors.
    for level in 1:config.refinement_levels
        n_eval >= config.max_evaluations && (budget_hit = true; break)

        ranked    = _cd_rank_candidates(all_cands)
        survivors = ranked[1:min(config.survivors_per_level, length(ranked))]
        hw        = coarse_hw ./ (2.0^(level - 1))

        for surv in survivors
            for tup in _cd_refined_grid(config, surv.design_values, hw)
                if n_eval >= config.max_evaluations
                    budget_hit = true
                    break
                end
                try_eval!(tup)
            end
            budget_hit && break
        end
        budget_hit || (levels_done = level)
        _cd_log!(log, "design_chaos_source: level $level done — $n_eval total")
    end
    budget_hit && _cd_log!(log, "design_chaos_source: budget of $(config.max_evaluations) reached")
    !budget_hit && (levels_done = config.refinement_levels)

    ranked_final  = _cd_rank_candidates(all_cands)
    best          = isempty(ranked_final) ? nothing : ranked_final[1]
    n_feasible    = count(c -> c.feasible, all_cands)

    _cd_log!(log, "design_chaos_source: complete — $n_eval evaluated, $n_feasible feasible" *
             (isnothing(best) ? "" : ", best verdict=$(best.certificate.overall_verdict)"))

    return ChaosDesignResult(
        sys.name,
        (config.operating_config.lyapunov.param_min, config.operating_config.lyapunov.param_max),
        config.operating_config.lyapunov.param_index,
        config.variables,
        config.target,
        all_cands,
        ranked_final,
        best,
        n_eval,
        n_feasible,
        budget_hit,
        levels_done,
        ts,
    )
end

"""
    chaos_design_summary(result::ChaosDesignResult) -> Dict{String, Any}

Plain-data summary of a `ChaosDesignResult` suitable for display or logging.
"""
function chaos_design_summary(result::ChaosDesignResult)::Dict{String, Any}
    best = result.best_candidate
    return Dict{String, Any}(
        "systemName"            => result.system_name,
        "operatingBand"         => collect(result.operating_band),
        "nVariables"            => length(result.variables),
        "nEvaluated"            => result.n_evaluated,
        "nFeasible"             => result.n_feasible,
        "budgetReached"         => result.budget_reached,
        "refinementLevels"      => result.refinement_levels_completed,
        "bestFeasible"          => !isnothing(best) && best.feasible,
        "bestVerdict"           => isnothing(best) ? nothing : String(best.certificate.overall_verdict),
        "bestObjective"         => isnothing(best) ? nothing : best.objective,
        "bestDesignValues"      => isnothing(best) ? nothing : copy(best.design_values),
    )
end
