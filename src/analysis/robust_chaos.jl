# Conservative robust-chaos region certificate.
# Orchestrates lyapunov_diagram, continuation_atlas, and basins_of_attraction without
# duplicating their analysis kernels.

_rc_log!(log, msg) = isnothing(log) ? nothing : log(String(msg))

"""
    RobustChaosEvidence

Complete evidence bundle produced by `robust_chaos_evidence`. The certificate
summarizes the verdict while the remaining fields retain the exact analysis
results that support each layer for audit, plotting, and reproducibility.

`basin_classifications` records the final certificate classification for every
basin seed (`:chaotic`, `:periodic`, `:non_chaotic`, or `:unresolved`) and has
the same shape as `basins.periodicity`.
"""
struct RobustChaosEvidence
    certificate::RobustChaosCertificate
    lyapunov::LyapunovDiagramResult
    atlas::AtlasResult
    basins::BasinsResult
    basin_classifications::Matrix{Symbol}
end

"""
    _rc_validate_lyapunov_reuse(lya_result, sys, config)

Throw `ArgumentError` if `lya_result` is not a coherent, compatible pre-computed
Lyapunov-diagram result for the given system and `RobustChaosConfig`. Checks system
name, vector-length coherence, exact parameter interval (within floating-point
representation tolerance), sample count, and neutral tolerance.
"""
function _rc_validate_lyapunov_reuse(
    lya_result::LyapunovDiagramResult,
    sys::Union{DiscreteMap, ContinuousODE},
    config::RobustChaosConfig,
)
    lya_result.system_name == sys.name || throw(ArgumentError(
        "Supplied lyapunov_result is for system '$(lya_result.system_name)'; " *
        "current system is '$(sys.name)'."
    ))
    expected_param_name = sys.param_names[config.lyapunov.param_index]
    lya_result.param_name == expected_param_name || throw(ArgumentError(
        "Supplied lyapunov_result uses parameter $(lya_result.param_name); " *
        "config requires $expected_param_name."
    ))
    n = length(lya_result.exponents)
    lengths = (
        params=length(lya_result.params),
        exponents=n,
        classifications=length(lya_result.classifications),
        estimation_statuses=length(lya_result.estimation_statuses),
        sample_counts=length(lya_result.sample_counts),
    )
    all(==(n), values(lengths)) && n >= 1 || throw(ArgumentError(
        "Supplied lyapunov_result has incoherent vector lengths $(repr(lengths))."
    ))
    expected_n = config.lyapunov.param_steps + 1
    n == expected_n || throw(ArgumentError(
        "Supplied lyapunov_result has $n samples; config.lyapunov.param_steps=$(config.lyapunov.param_steps) " *
        "expects $expected_n samples."
    ))
    expected_params = collect(range(
        config.lyapunov.param_min,
        config.lyapunov.param_max;
        length=expected_n,
    ))
    all(_robust_float_repr_equal(actual, expected)
        for (actual, expected) in zip(lya_result.params, expected_params)) || throw(ArgumentError(
            "Supplied lyapunov_result parameter grid does not match config.lyapunov."
        ))
    _robust_float_repr_equal(lya_result.neutral_tolerance, config.lyapunov.neutral_tolerance) || throw(ArgumentError(
        "Supplied lyapunov_result neutral_tolerance=$(lya_result.neutral_tolerance); " *
        "config.lyapunov.neutral_tolerance=$(config.lyapunov.neutral_tolerance)."
    ))
    return nothing
end

"""
    _rc_validate_atlas_reuse(atlas_result, sys, config)

Throw `ArgumentError` if `atlas_result` is not a coherent, compatible pre-computed
atlas result for the given system and `RobustChaosConfig`. Checks system name, that
the brute-force parameter interval exactly matches the certificate interval (within
floating-point representation tolerance), and that every configured search period is
present in the result's diagnostic metadata.
"""
function _rc_validate_atlas_reuse(
    atlas_result::AtlasResult,
    sys::Union{DiscreteMap, ContinuousODE},
    config::RobustChaosConfig,
)
    atlas_result.system_name == sys.name || throw(ArgumentError(
        "Supplied atlas_result is for system '$(atlas_result.system_name)'; " *
        "current system is '$(sys.name)'."
    ))
    expected_param_name = sys.param_names[config.lyapunov.param_index]
    atlas_result.param_name == expected_param_name || throw(ArgumentError(
        "Supplied atlas_result uses parameter $(atlas_result.param_name); " *
        "config requires $expected_param_name."
    ))
    isempty(atlas_result.recon_samples) && throw(ArgumentError(
        "Supplied atlas_result has no reconnaissance samples; cannot verify interval coverage."
    ))
    recon_params = [sample.param for sample in atlas_result.recon_samples]
    _robust_float_repr_equal(minimum(recon_params), config.lyapunov.param_min) &&
    _robust_float_repr_equal(maximum(recon_params), config.lyapunov.param_max) || throw(ArgumentError(
        "Supplied atlas_result reconnaissance interval is " *
        "[$(minimum(recon_params)), $(maximum(recon_params))]; certificate expects " *
        "[$(config.lyapunov.param_min), $(config.lyapunov.param_max)]."
    ))
    bf_params = atlas_result.brute_force.params
    isempty(bf_params) && throw(ArgumentError(
        "Supplied atlas_result has empty brute_force.params; cannot verify parameter interval."
    ))
    bf_min = minimum(bf_params)
    bf_max = maximum(bf_params)
    _robust_float_repr_equal(bf_min, config.lyapunov.param_min) &&
    _robust_float_repr_equal(bf_max, config.lyapunov.param_max) || throw(ArgumentError(
        "Supplied atlas_result brute-force interval is [$bf_min, $bf_max]; " *
        "certificate expects [$(config.lyapunov.param_min), $(config.lyapunov.param_max)]." *
        " Atlas reuse requires exact interval endpoints (no subset coverage)."
    ))
    periods_searched = Int[_as_int(x) for x in get(atlas_result.diagnostics, "periods", Int[])]
    isempty(periods_searched) && throw(ArgumentError(
        "Supplied atlas_result diagnostics missing 'periods'; cannot verify period coverage."
    ))
    required = Set{Int}(config.atlas.periods)
    missing_ps = sort(collect(setdiff(required, Set{Int}(periods_searched))))
    isempty(missing_ps) || throw(ArgumentError(
        "Supplied atlas_result searched periods $(sort(periods_searched)); " *
        "certificate requires periods $(sort(collect(required))). " *
        "Missing: $missing_ps."
    ))
    return nothing
end

# Build a properly-sized base parameter vector from the atlas brute_force config.
# Required for stability recomputation when branch points lack a stored :stable field.
function _rc_atlas_base_params(atlas_config::AtlasConfig)
    bf = atlas_config.brute_force  # non-nothing guaranteed by RobustChaosConfig assertion
    required = max(bf.param_index, maximum(bf.linked_param_indices; init=0))
    params = isempty(bf.fixed_params) ? zeros(Float64, required) : copy(bf.fixed_params)
    if length(params) < required
        old_length = length(params)
        resize!(params, required)
        fill!(@view(params[(old_length + 1):required]), 0.0)
    end
    return params
end

# Classify one basin grid seed via the largest-Lyapunov estimator.
# Returns :chaotic, :non_chaotic, or :unresolved.
# Uses the Lyapunov config's transient/iterations/perturbation/divergence_cutoff so
# the estimator settings are consistent with the certificate's Lyapunov layer.
function _rc_classify_basin_seed(
    sys::DiscreteMap,
    ic::AbstractVector,
    params::AbstractVector,
    lya_config::LyapunovConfig,
)
    sv_ic = SVector{sys.dim, Float64}(ic)
    result = _estimate_discrete_map_largest_lyapunov(
        sys, params, sv_ic,
        lya_config.transient, lya_config.iterations,
        lya_config.perturbation, lya_config.divergence_cutoff,
    )
    if result.estimation_status == :collapsed
        return :non_chaotic
    elseif result.estimation_status != :ok || !isfinite(result.exponent)
        return :unresolved
    elseif result.exponent > lya_config.neutral_tolerance
        return :chaotic
    else
        return :non_chaotic
    end
end

function _rc_classify_basin_seed(
    sys::ContinuousODE,
    ic::AbstractVector,
    params::AbstractVector,
    lya_config::LyapunovConfig;
    solver,
    reltol::Float64,
    abstol::Float64,
    min_crossing_time::Float64,
)
    result = _estimate_continuous_poincare_largest_lyapunov(
        sys, params, collect(Float64, ic),
        lya_config.transient, lya_config.iterations,
        lya_config.perturbation, lya_config.divergence_cutoff;
        solver=solver, reltol=reltol, abstol=abstol,
        min_crossing_time=min_crossing_time,
    )
    if result.estimation_status == :collapsed
        return :non_chaotic
    elseif result.estimation_status != :ok || !isfinite(result.exponent)
        return :unresolved
    elseif result.exponent > lya_config.neutral_tolerance
        return :chaotic
    else
        return :non_chaotic
    end
end

function _rc_contiguous_stable_spans(samples, max_gap::Float64)
    spans = NamedTuple{(:param_min, :param_max, :count), Tuple{Float64, Float64, Int}}[]
    current = Float64[]
    flush!() = if !isempty(current)
        push!(spans, (param_min=minimum(current), param_max=maximum(current), count=length(current)))
        empty!(current)
    end
    previous_param = nothing
    for (param, stable) in samples
        if stable === true
            if !isnothing(previous_param) && abs(param - previous_param) > max_gap
                flush!()
            end
            push!(current, param)
            previous_param = param
        else
            flush!()
            previous_param = nothing
        end
    end
    flush!()
    return spans
end

# Scan atlas branch records for stable-orbit evidence and coalesce each contiguous
# stable run into its own StableWindowEvidence record.
# Uses the stored :stable field when present on a branch point (requires detect_bifurcation > 0
# in the continuation config, which is the default). When absent, recomputes stability via
# _map_stability with the atlas base params.
function _rc_stable_window_evidence(
    atlas_result::AtlasResult,
    sys::DynamicalSystem,
    base_params::AbstractVector,
    atlas_config::AtlasConfig,
    param_min::Float64,
    param_max::Float64,
)::Tuple{Vector{StableWindowEvidence}, Int}
    evidence = StableWindowEvidence[]
    unresolved_stability_count = 0
    for record in atlas_result.branch_records
        period    = record.branch.period
        branch_id = record.id
        window_id = record.window_id
        stability_samples = Tuple{Float64, Union{Bool, Nothing}}[]
        for pt in _branch_points(record.branch)
            pt_param = pt.param
            if !isfinite(pt_param) || !(param_min <= pt_param <= param_max)
                push!(stability_samples, (Float64(pt_param), false))
                continue
            end
            stable = if hasproperty(pt, :stable)
                try
                    Bool(pt.stable)
                catch
                    unresolved_stability_count += 1
                    nothing
                end
            else
                try
                    inj = inject_param(
                        base_params,
                        atlas_config.brute_force.param_index,
                        pt_param,
                        atlas_config.brute_force.linked_param_indices,
                    )
                    s, _ = _map_stability(sys, _branch_point_state(pt), inj, period)
                    s
                catch
                    unresolved_stability_count += 1
                    nothing
                end
            end
            push!(stability_samples, (Float64(pt_param), stable))
        end
        interval_span = max(abs(param_max - param_min), eps(Float64))
        max_gap = max(5 * abs(atlas_config.continuation.dsmax), sqrt(eps(Float64)) * interval_span)
        for span in _rc_contiguous_stable_spans(stability_samples, max_gap)
            push!(evidence, StableWindowEvidence(
                branch_id, window_id, period,
                span.param_min, span.param_max, span.count,
            ))
        end
    end
    return evidence, unresolved_stability_count
end

"""
    _rc_lyapunov_verdict(n_total, n_resolved, n_positive, min_positive_frac, min_resolved_frac) -> Symbol

Compute the Lyapunov-layer verdict (:pass, :fail, or :inconclusive).

- `n_positive / n_resolved` is the positive fraction (fraction of resolved that are chaotic).
- If resolved coverage `n_resolved / n_total < min_resolved_frac`, the verdict is inconclusive
  unless failure is already provable: if even the best case (all unresolved turn positive) still
  yields a positive fraction below `min_positive_frac`, it is `:fail`.
- Once resolved coverage is sufficient, the verdict is `:fail` if the positive fraction is below
  `min_positive_frac`, otherwise `:pass`.
"""
function _rc_lyapunov_verdict(
    n_total::Int, n_resolved::Int, n_positive::Int,
    min_positive_frac::Float64, min_resolved_frac::Float64,
)::Symbol
    n_total == 0 && return :inconclusive  # no samples: cannot certify or refute
    resolved_frac = n_resolved / n_total
    if resolved_frac < min_resolved_frac
        # Resolved coverage is insufficient. Check whether failure is already provable:
        # best case has all unresolved samples become positive, so maximum achievable
        # positive fraction over all samples is (n_positive + n_unresolved) / n_total.
        best_case = (n_positive + (n_total - n_resolved)) / n_total
        return best_case < min_positive_frac ? :fail : :inconclusive
    end
    n_resolved == 0 && return :inconclusive
    return n_positive / n_resolved < min_positive_frac ? :fail : :pass
end

"""
    _rc_basin_verdict(n_total, n_resolved, n_chaotic, min_chaotic_frac, min_resolved_frac) -> Symbol

Basin-layer verdict with symmetric semantics to `_rc_lyapunov_verdict`.
"""
function _rc_basin_verdict(
    n_total::Int, n_resolved::Int, n_chaotic::Int,
    min_chaotic_frac::Float64, min_resolved_frac::Float64,
)::Symbol
    return _rc_lyapunov_verdict(n_total, n_resolved, n_chaotic, min_chaotic_frac, min_resolved_frac)
end

"""
    robust_chaos_evidence(sys, config; kwargs...) -> RobustChaosEvidence

Compute a conservative robust-chaos certificate by orchestrating three analysis layers:

1. **Lyapunov sweep** (`lyapunov_diagram`): verifies that the swept parameter interval yields
   predominantly positive Lyapunov exponents with sufficient resolved sample coverage.
2. **Continuation-atlas search** (`continuation_atlas`): checks that no stable low-period
   orbits exist within the searched region, that the atlas search was not cut short by a time
   budget, and that all candidate windows were adequately covered.
3. **Basin of attraction** (`basins_of_attraction` + per-seed Lyapunov re-estimation): verifies
   that basin-of-attraction grid seeds are predominantly chaotic. Seeds with detected periodicity
   (`BasinsResult.periodicity > 0`) are classified as periodic without re-estimation; seeds with
   undetected period (`periodicity == 0`) are evaluated by the Lyapunov estimator.

**Conservative semantics**: a `:certified` verdict means all three layers passed their configured
thresholds with sufficient resolved coverage within the configured sampling and search. It does
**not** mathematically prove the absence of stable periodic orbits or chaotic behaviour outside the
configured search periods, grid resolution, or parameter range.

# Arguments
- `sys`: `DiscreteMap` or `ContinuousODE`
- `config`: `RobustChaosConfig` (three nested layer configs plus threshold fractions)
- `initial_point`: Optional starting point for Lyapunov and atlas orbit seeding
- `solver`, `reltol`, `abstol`, `min_crossing_time`: ODE integration parameters (`ContinuousODE` only)
- `log`: Optional callback `f(msg::String)` receiving progress messages
- `cache_key`, `cache_file`, `cache_enabled`: Atlas result cache settings

## Pre-computed layer results (optional, for source-result reuse)

- `lyapunov_result`: Pre-computed `LyapunovDiagramResult`. When supplied and valid, layer 1
  skips `lyapunov_diagram` entirely and uses this result directly as bounded evidence.
  Valid means: same system name; vector-length coherence; exact parameter interval
  (`param_steps + 1` samples, endpoints within floating-point representation tolerance);
  matching `neutral_tolerance`. The result's own coverage drives the layer verdict; the
  certificate makes no assumption about which Lyapunov settings produced it beyond what is
  recorded in the result.
- `atlas_result`: Pre-computed `AtlasResult`. When supplied and valid, layer 2 skips
  `continuation_atlas` and uses this result's branch records, diagnostics, and coverage
  as bounded evidence. Valid means: same system name; brute-force parameter interval exactly
  matches `[config.lyapunov.param_min, config.lyapunov.param_max]` (exact endpoints, no
  subset coverage); diagnostics include all configured search periods. **Bounded
  responsibility**: zoomed subinterval certificates rerun the atlas because subset coverage
  accounting is not yet implemented. The source atlas may have used different effort settings
  (step sizes, time budget, seed counts); these differences are reflected in the reused
  result's own diagnostics and coverage summary, which drive the atlas-layer verdict.
# Returns
`RobustChaosEvidence` containing the summary certificate and the exact Lyapunov,
atlas, basin-periodicity, and per-seed basin-classification evidence.
"""
function robust_chaos_evidence(
    sys::Union{DiscreteMap, ContinuousODE},
    config::RobustChaosConfig;
    initial_point::Union{Nothing, AbstractVector} = nothing,
    solver = Tsit5(),
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
    min_crossing_time::Float64 = 1e-6,
    log::Union{Nothing, Function} = nothing,
    cache_key::Union{Nothing, AbstractString} = nothing,
    cache_file::Union{Nothing, AbstractString} = nothing,
    cache_enabled::Bool = config.atlas.cache_enabled,
    lyapunov_result::Union{Nothing, LyapunovDiagramResult} = nothing,
    atlas_result::Union{Nothing, AtlasResult} = nothing,
)::RobustChaosEvidence
    return _robust_chaos_analysis(
        sys,
        config,
        true;
        initial_point,
        solver,
        reltol,
        abstol,
        min_crossing_time,
        log,
        cache_key,
        cache_file,
        cache_enabled,
        lyapunov_result,
        atlas_result,
    )::RobustChaosEvidence
end

function _robust_chaos_analysis(
    sys::Union{DiscreteMap, ContinuousODE},
    config::RobustChaosConfig,
    store_evidence::Bool;
    initial_point::Union{Nothing, AbstractVector} = nothing,
    solver = Tsit5(),
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
    min_crossing_time::Float64 = 1e-6,
    log::Union{Nothing, Function} = nothing,
    cache_key::Union{Nothing, AbstractString} = nothing,
    cache_file::Union{Nothing, AbstractString} = nothing,
    cache_enabled::Bool = config.atlas.cache_enabled,
    lyapunov_result::Union{Nothing, LyapunovDiagramResult} = nothing,
    atlas_result::Union{Nothing, AtlasResult} = nothing,
)
    items = Dict{String, Any}[]
    ts = now()

    # --- Layer 1: Lyapunov sweep ---
    lya_result = if !isnothing(lyapunov_result)
        _rc_validate_lyapunov_reuse(lyapunov_result, sys, config)
        _rc_log!(log, "robust_chaos_certificate: layer 1 — reusing supplied LyapunovDiagramResult over [$(lyapunov_result.params[1]), $(lyapunov_result.params[end])]")
        lyapunov_result
    else
        _rc_log!(log, "robust_chaos_certificate: layer 1 — Lyapunov sweep over [$(config.lyapunov.param_min), $(config.lyapunov.param_max)]")
        if sys isa ContinuousODE
            lyapunov_diagram(sys, config.lyapunov;
                initial_point=initial_point, solver=solver, reltol=reltol, abstol=abstol)
        else
            lyapunov_diagram(sys, config.lyapunov; initial_point=initial_point)
        end
    end

    resolved_indices = findall(eachindex(lya_result.exponents)) do idx
        lya_result.estimation_statuses[idx] == :ok && isfinite(lya_result.exponents[idx])
    end
    n_lya_total    = length(lya_result.exponents)
    n_lya_resolved = length(resolved_indices)
    n_lya_positive = count(
        idx -> lya_result.classifications[idx] == :chaotic_candidate,
        resolved_indices,
    )
    lya_min_exp    = isempty(resolved_indices) ? NaN :
        minimum(lya_result.exponents[k] for k in resolved_indices)
    lya_resolved_frac = n_lya_total > 0 ? n_lya_resolved / n_lya_total : 0.0
    lya_positive_frac = n_lya_resolved > 0 ? n_lya_positive / n_lya_resolved : 0.0
    lya_verdict = _rc_lyapunov_verdict(
        n_lya_total, n_lya_resolved, n_lya_positive,
        config.min_lyapunov_positive_fraction, config.min_lyapunov_resolved_fraction,
    )
    push!(items, Dict{String, Any}(
        "layer"                 => "lyapunov",
        "verdict"               => String(lya_verdict),
        "n_total"               => n_lya_total,
        "n_resolved"            => n_lya_resolved,
        "n_positive"            => n_lya_positive,
        "resolved_fraction"     => lya_resolved_frac,
        "positive_fraction"     => lya_positive_frac,
        "min_resolved_exponent" => lya_min_exp,
    ))
    _rc_log!(log, "layer 1 verdict: $lya_verdict (resolved=$(round(lya_resolved_frac, digits=3)), positive=$(round(lya_positive_frac, digits=3)), min_exp=$(round(lya_min_exp, digits=4)))")

    # --- Layer 2: Continuation-atlas window search ---
    _atlas_result = if !isnothing(atlas_result)
        _rc_validate_atlas_reuse(atlas_result, sys, config)
        let ps = sort(Int[_as_int(x) for x in get(atlas_result.diagnostics, "periods", Int[])]),
            tbe = _as_bool(get(atlas_result.diagnostics, "timeBudgetExceeded", false))
            _rc_log!(log, "robust_chaos_certificate: layer 2 — reusing supplied AtlasResult (searched_periods=$ps, timeBudgetExceeded=$tbe)")
        end
        atlas_result
    else
        _rc_log!(log, "robust_chaos_certificate: layer 2 — continuation-atlas search")
        if sys isa ContinuousODE
            continuation_atlas(sys, config.atlas;
                initial_point=initial_point, solver=solver, reltol=reltol, abstol=abstol,
                min_crossing_time=min_crossing_time, log=log,
                cache_key=cache_key, cache_file=cache_file, cache_enabled=cache_enabled)
        else
            continuation_atlas(sys, config.atlas;
                initial_point=initial_point, log=log,
                cache_key=cache_key, cache_file=cache_file, cache_enabled=cache_enabled)
        end
    end

    time_budget_exceeded  = _as_bool(get(_atlas_result.diagnostics, "timeBudgetExceeded", false))
    calibration_refused   = _atlas_recon_calibration_refused(_atlas_result.diagnostics)
    periods_searched      = Int[_as_int(x) for x in get(_atlas_result.diagnostics, "periods", Int[])]
    n_atlas_covered       = _as_int(get(_atlas_result.coverage_summary, "covered",    0))
    n_atlas_partial       = _as_int(get(_atlas_result.coverage_summary, "partial",    0))
    n_atlas_unresolved_w  = _as_int(get(_atlas_result.coverage_summary, "unresolved", 0))
    n_atlas_windows       = n_atlas_covered + n_atlas_partial + n_atlas_unresolved_w
    n_atlas_gaps          = length(_atlas_result.gaps)
    atlas_search_complete = !time_budget_exceeded && !calibration_refused

    base_params     = _rc_atlas_base_params(config.atlas)
    stable_evidence, unresolved_stability_count = _rc_stable_window_evidence(
        _atlas_result,
        sys,
        base_params,
        config.atlas,
        config.lyapunov.param_min,
        config.lyapunov.param_max,
    )

    # Coverage/effort: how much of the window search space was covered, penalised by budget exhaustion.
    atlas_coverage_effort = if n_atlas_windows == 0
        atlas_search_complete && n_atlas_gaps == 0 ? 1.0 : 0.0
    else
        min(1.0, (n_atlas_covered / n_atlas_windows) * (time_budget_exceeded ? 0.5 : 1.0))
    end

    atlas_verdict = if !isempty(stable_evidence)
        :fail
    elseif calibration_refused
        :inconclusive
    elseif unresolved_stability_count > 0
        :inconclusive
    elseif time_budget_exceeded
        :inconclusive
    elseif n_atlas_gaps > 0
        :inconclusive
    elseif n_atlas_windows == 0
        atlas_search_complete ? :pass : :inconclusive
    elseif n_atlas_partial > 0 || n_atlas_unresolved_w > 0
        :inconclusive
    else
        :pass
    end

    push!(items, Dict{String, Any}(
        "layer"                => "atlas",
        "verdict"              => String(atlas_verdict),
        "searched_periods"     => periods_searched,
        "search_complete"      => atlas_search_complete,
        "coverage_effort"      => atlas_coverage_effort,
        "n_windows"            => n_atlas_windows,
        "n_covered"            => n_atlas_covered,
        "n_partial"            => n_atlas_partial,
        "n_unresolved"         => n_atlas_unresolved_w,
        "n_gaps"               => n_atlas_gaps,
        "unresolved_stability_count" => unresolved_stability_count,
        "stable_evidence_count"=> length(stable_evidence),
        "recon_calibration"    => get(_atlas_result.diagnostics, "reconCalibration", Dict{String, Any}()),
    ))
    _rc_log!(log, "layer 2 verdict: $atlas_verdict (windows=$n_atlas_windows, covered=$n_atlas_covered, stable=$(length(stable_evidence)), periods=$(periods_searched))")

    # --- Layer 3: Basin of attraction ---
    _rc_log!(log, "robust_chaos_certificate: layer 3 — basin of attraction at param=$(config.basins.bif_param)")
    basins_result = if sys isa ContinuousODE
        basins_of_attraction(sys, config.basins; solver=solver, reltol=reltol, abstol=abstol)
    else
        basins_of_attraction(sys, config.basins)
    end

    basin_params = build_basins_params(config.basins)
    x_grid  = basins_result.x_grid
    y_grid  = basins_result.y_grid
    x_idx   = basins_result.x_index
    y_idx   = basins_result.y_index
    base_ic = copy(basins_result.ic_template)

    basin_class_counts = Dict{Symbol, Int}()
    basin_classifications = store_evidence ?
        fill(:unresolved, length(x_grid), length(y_grid)) : nothing
    n_basin_total    = 0
    n_basin_resolved = 0
    n_basin_chaotic  = 0
    ic = copy(base_ic)

    for (j, y_val) in enumerate(y_grid), (i, x_val) in enumerate(x_grid)
        n_basin_total += 1
        period = basins_result.periodicity[i, j]
        copyto!(ic, base_ic)
        ic[x_idx] = x_val
        ic[y_idx] = y_val
        cls = if period > 0
            :periodic
        elseif sys isa ContinuousODE
            _rc_classify_basin_seed(sys, ic, basin_params, config.lyapunov;
                solver=solver, reltol=reltol, abstol=abstol,
                min_crossing_time=min_crossing_time)
        else
            _rc_classify_basin_seed(sys, ic, basin_params, config.lyapunov)
        end
        !isnothing(basin_classifications) && (basin_classifications[i, j] = cls)
        basin_class_counts[cls] = get(basin_class_counts, cls, 0) + 1
        if cls == :chaotic
            n_basin_resolved += 1
            n_basin_chaotic  += 1
        elseif cls in (:periodic, :non_chaotic)
            n_basin_resolved += 1
        end
        # :unresolved seeds are not counted as resolved
    end

    basin_resolved_frac = n_basin_total > 0 ? n_basin_resolved / n_basin_total : 0.0
    basin_chaotic_frac  = n_basin_resolved > 0 ? n_basin_chaotic / n_basin_resolved : 0.0
    basin_verdict = _rc_basin_verdict(
        n_basin_total, n_basin_resolved, n_basin_chaotic,
        config.min_chaotic_basin_fraction, config.min_basin_resolved_fraction,
    )

    push!(items, Dict{String, Any}(
        "layer"             => "basins",
        "verdict"           => String(basin_verdict),
        "n_total"           => n_basin_total,
        "n_resolved"        => n_basin_resolved,
        "n_chaotic"         => n_basin_chaotic,
        "resolved_fraction" => basin_resolved_frac,
        "chaotic_fraction"  => basin_chaotic_frac,
        "class_counts"      => Dict{String, Int}(String(k) => v for (k, v) in basin_class_counts),
    ))
    _rc_log!(log, "layer 3 verdict: $basin_verdict (resolved=$(round(basin_resolved_frac, digits=3)), chaotic=$(round(basin_chaotic_frac, digits=3)))")

    # --- Overall verdict ---
    verdicts = (lya_verdict, atlas_verdict, basin_verdict)
    overall_verdict = if any(v -> v == :fail, verdicts)
        :fragile
    elseif all(v -> v == :pass, verdicts)
        :certified
    else
        :inconclusive
    end

    # Robustness score: conservative minimum of three layer scores.
    # Stable atlas evidence drives the atlas score to zero regardless of coverage effort.
    lya_score    = lya_positive_frac * lya_resolved_frac
    atlas_score  = isempty(stable_evidence) ? atlas_coverage_effort : 0.0
    basin_score  = basin_chaotic_frac * basin_resolved_frac
    robustness_score = min(lya_score, atlas_score, basin_score)

    push!(items, Dict{String, Any}(
        "layer"            => "overall",
        "verdict"          => String(overall_verdict),
        "robustness_score" => robustness_score,
    ))
    _rc_log!(log, "overall verdict: $overall_verdict (robustness_score=$(round(robustness_score, digits=4)))")

    certificate = RobustChaosCertificate(
        config.lyapunov.param_min,
        config.lyapunov.param_max,
        sys.name,
        config.lyapunov.param_index,
        lya_verdict,
        atlas_verdict,
        basin_verdict,
        overall_verdict,
        lya_positive_frac,
        lya_resolved_frac,
        lya_min_exp,
        n_lya_total,
        n_lya_resolved,
        n_lya_positive,
        periods_searched,
        atlas_search_complete,
        atlas_coverage_effort,
        n_atlas_windows,
        n_atlas_covered,
        n_atlas_partial,
        n_atlas_unresolved_w,
        n_atlas_gaps,
        unresolved_stability_count,
        stable_evidence,
        basins_result.bif_param,
        basin_chaotic_frac,
        basin_resolved_frac,
        n_basin_total,
        n_basin_resolved,
        n_basin_chaotic,
        basin_class_counts,
        robustness_score,
        items,
        ts,
    )
    if store_evidence
        return RobustChaosEvidence(
            certificate,
            lya_result,
            _atlas_result,
            basins_result,
            basin_classifications::Matrix{Symbol},
        )
    end
    return certificate
end

"""
    robust_chaos_certificate(sys, config; kwargs...) -> RobustChaosCertificate

Compute the conservative certificate summary without retaining the supporting
analysis results. Use `robust_chaos_evidence` when the exact evidence layers are
needed for audit, visualization, or persistence.
"""
function robust_chaos_certificate(
    sys::Union{DiscreteMap, ContinuousODE},
    config::RobustChaosConfig;
    kwargs...,
)::RobustChaosCertificate
    return _robust_chaos_analysis(sys, config, false; kwargs...)::RobustChaosCertificate
end

# ─── Two-parameter region certificate ─────────────────────────────────────────

@inline _rc_region_cell_area(cell::AdaptiveMapLeafCell) =
    max(0.0, cell.a1 - cell.a0) * max(0.0, cell.b1 - cell.b0)

@inline _rc_region_cell_center(cell::AdaptiveMapLeafCell) =
    ((cell.a0 + cell.a1) * 0.5, (cell.b0 + cell.b1) * 0.5)

function _rc_region_base_params(config::BifurcationMapConfig, a::Float64, b::Float64)
    required = max(
        length(config.base_params),
        config.a_index,
        config.b_index,
        maximum(config.a_linked_param_indices; init=0),
        maximum(config.b_linked_param_indices; init=0),
    )
    params = isempty(config.base_params) ? zeros(Float64, required) : copy(config.base_params)
    if length(params) < required
        old_length = length(params)
        resize!(params, required)
        fill!(@view(params[(old_length + 1):required]), 0.0)
    end
    params[config.a_index] = a
    params[config.b_index] = b
    for idx in config.a_linked_param_indices
        params[idx] = a
    end
    for idx in config.b_linked_param_indices
        params[idx] = b
    end
    return params
end

function _rc_region_coarse_status_codes(result::AdaptiveMapResult)
    na = length(result.coarse_result.a_grid)
    nb = length(result.coarse_result.b_grid)
    status = fill(_map_status_code(:unknown), na, nb)
    a_lookup = Dict(value => idx for (idx, value) in pairs(result.coarse_result.a_grid))
    b_lookup = Dict(value => idx for (idx, value) in pairs(result.coarse_result.b_grid))
    for sample in result.samples
        sample.depth == 0 || continue
        ia = get(a_lookup, sample.a, nothing)
        ib = get(b_lookup, sample.b, nothing)
        if !isnothing(ia) && !isnothing(ib)
            status[ia, ib] = sample.status_code
        end
    end
    return status
end

function _rc_region_leaf_candidate(
    adaptive::AdaptiveMapResult,
    cell::AdaptiveMapLeafCell,
    candidate_codes::Set{Int},
)
    indices = (cell.si_sw, cell.si_se, cell.si_nw, cell.si_ne)
    all(idx -> adaptive.samples[idx].status_code in candidate_codes, indices) || return false
    return cell.si_center == 0 || adaptive.samples[cell.si_center].status_code in candidate_codes
end

@inline _rc_lattice_intervals_overlap(lo1::Int, hi1::Int, lo2::Int, hi2::Int) =
    min(hi1, hi2) > max(lo1, lo2)

function _rc_region_components(adaptive::AdaptiveMapResult, candidate_leaf_indices::Vector{Int})
    left_edges = Dict{Int, Vector{Int}}()
    right_edges = Dict{Int, Vector{Int}}()
    bottom_edges = Dict{Int, Vector{Int}}()
    top_edges = Dict{Int, Vector{Int}}()
    for idx in candidate_leaf_indices
        cell = adaptive.leaf_cells[idx]
        push!(get!(left_edges, cell.ia0, Int[]), idx)
        push!(get!(right_edges, cell.ia1, Int[]), idx)
        push!(get!(bottom_edges, cell.ib0, Int[]), idx)
        push!(get!(top_edges, cell.ib1, Int[]), idx)
    end
    remaining = Set(candidate_leaf_indices)
    components = Vector{Int}[]
    while !isempty(remaining)
        start = first(remaining)
        delete!(remaining, start)
        stack = [start]
        component = Int[]
        while !isempty(stack)
            idx = pop!(stack)
            push!(component, idx)
            cell = adaptive.leaf_cells[idx]
            neighbours = Int[]
            for other in get(left_edges, cell.ia1, Int[])
                other in remaining || continue
                other_cell = adaptive.leaf_cells[other]
                _rc_lattice_intervals_overlap(cell.ib0, cell.ib1, other_cell.ib0, other_cell.ib1) &&
                    push!(neighbours, other)
            end
            for other in get(right_edges, cell.ia0, Int[])
                other in remaining || continue
                other_cell = adaptive.leaf_cells[other]
                _rc_lattice_intervals_overlap(cell.ib0, cell.ib1, other_cell.ib0, other_cell.ib1) &&
                    push!(neighbours, other)
            end
            for other in get(bottom_edges, cell.ib1, Int[])
                other in remaining || continue
                other_cell = adaptive.leaf_cells[other]
                _rc_lattice_intervals_overlap(cell.ia0, cell.ia1, other_cell.ia0, other_cell.ia1) &&
                    push!(neighbours, other)
            end
            for other in get(top_edges, cell.ib0, Int[])
                other in remaining || continue
                other_cell = adaptive.leaf_cells[other]
                _rc_lattice_intervals_overlap(cell.ia0, cell.ia1, other_cell.ia0, other_cell.ia1) &&
                    push!(neighbours, other)
            end
            for other in neighbours
                delete!(remaining, other)
                push!(stack, other)
            end
        end
        push!(components, sort(component))
    end
    sort!(components; by = comp -> (
        minimum(adaptive.leaf_cells[idx].a0 for idx in comp),
        minimum(adaptive.leaf_cells[idx].b0 for idx in comp),
        length(comp),
    ))
    return components
end

function _rc_region_bbox(cells::Vector{AdaptiveMapLeafCell})
    return (
        a_min = minimum(cell.a0 for cell in cells),
        a_max = maximum(cell.a1 for cell in cells),
        b_min = minimum(cell.b0 for cell in cells),
        b_max = maximum(cell.b1 for cell in cells),
    )
end

function _rc_region_grid_indices(cells::Vector{AdaptiveMapLeafCell}, a_grid, b_grid)
    indices = Set{Tuple{Int, Int}}()
    a_values = collect(Float64, a_grid)
    b_values = collect(Float64, b_grid)
    for cell in cells
        i0 = searchsortedfirst(a_values, cell.a0)
        i1 = searchsortedlast(a_values, cell.a1)
        j0 = searchsortedfirst(b_values, cell.b0)
        j1 = searchsortedlast(b_values, cell.b1)
        if i0 <= i1 && j0 <= j1
            for j in j0:j1, i in i0:i1
                push!(indices, (i, j))
            end
        end
    end
    return sort!(collect(indices))
end

function _rc_region_knot_values(values::Vector{Float64}, max_count::Int)
    isempty(values) && return Float64[]
    vals = sort(unique(values))
    length(vals) <= max_count && return vals
    max_count == 1 && return [vals[cld(length(vals), 2)]]
    idxs = round.(Int, range(1, length(vals); length=max_count))
    return vals[unique(idxs)]
end

function _rc_region_sample_centers(cells::Vector{AdaptiveMapLeafCell}, max_count::Int)
    centers = [_rc_region_cell_center(cell) for cell in cells]
    sort!(centers; by = c -> (c[1] + c[2], c[1], c[2]))
    length(centers) <= max_count && return centers
    max_count == 1 && return [centers[cld(length(centers), 2)]]
    idxs = unique(round.(Int, range(1, length(centers); length=max_count)))
    return centers[idxs]
end

function _rc_region_interval_and_secondary(region, axis::Symbol)
    if axis == :a
        return (primary_min=region.a_min, primary_max=region.a_max,
                secondary_min=region.b_min, secondary_max=region.b_max)
    else
        return (primary_min=region.b_min, primary_max=region.b_max,
                secondary_min=region.a_min, secondary_max=region.a_max)
    end
end

function _rc_region_atlas_config(
    config::RobustChaosRegionConfig,
    region_bounds,
    secondary_value::Float64,
)
    axis = config.slice_axis
    primary_index = axis == :a ? config.map.a_index : config.map.b_index
    primary_links = axis == :a ? config.map.a_linked_param_indices : config.map.b_linked_param_indices
    a = axis == :a ? (region_bounds.primary_min + region_bounds.primary_max) * 0.5 : secondary_value
    b = axis == :a ? secondary_value : (region_bounds.primary_min + region_bounds.primary_max) * 0.5
    params = _rc_region_base_params(config.map, a, b)
    atlas = config.atlas
    bf = config.atlas.brute_force
    ct = config.atlas.continuation
    bf2 = Accessors.@set bf.param_min = region_bounds.primary_min
    bf2 = Accessors.@set bf2.param_max = region_bounds.primary_max
    bf2 = Accessors.@set bf2.param_index = primary_index
    bf2 = Accessors.@set bf2.linked_param_indices = copy(primary_links)
    bf2 = Accessors.@set bf2.fixed_params = params
    ct2 = Accessors.@set ct.p_min = region_bounds.primary_min
    ct2 = Accessors.@set ct2.p_max = region_bounds.primary_max
    ct2 = Accessors.@set ct2.param_index = primary_index
    ct2 = Accessors.@set ct2.linked_param_indices = copy(primary_links)
    atlas2 = Accessors.@set atlas.brute_force = bf2
    atlas2 = Accessors.@set atlas2.continuation = ct2
    return atlas2, params
end

function _rc_region_basin_config(config::RobustChaosRegionConfig, a::Float64, b::Float64)
    primary = config.slice_axis == :a ? a : b
    params = _rc_region_base_params(config.map, a, b)
    template = config.basins
    basins = Accessors.@set template.bif_param = primary
    basins = Accessors.@set basins.fixed_params = params
    return basins, params
end

function _rc_region_classify_basin_seed(
    sys::DiscreteMap,
    ic::AbstractVector,
    params::AbstractVector,
    config::BifurcationMapConfig,
)
    result = _estimate_discrete_map_largest_lyapunov(
        sys,
        params,
        SVector{sys.dim, Float64}(ic),
        _map_lyapunov_transient(config),
        _map_lyapunov_iterations(config),
        config.lyapunov_perturbation,
        config.divergence_cutoff,
    )
    if result.estimation_status == :collapsed
        return :non_chaotic
    elseif result.estimation_status != :ok || !isfinite(result.exponent)
        return :unresolved
    elseif result.exponent > config.lyapunov_neutral_tolerance
        return :chaotic
    else
        return :non_chaotic
    end
end

function _rc_region_classify_basin_seed(
    sys::ContinuousODE,
    ic::AbstractVector,
    params::AbstractVector,
    config::BifurcationMapConfig;
    solver,
    reltol::Float64,
    abstol::Float64,
    min_crossing_time::Float64,
)
    result = _estimate_continuous_poincare_largest_lyapunov(
        sys,
        params,
        collect(Float64, ic),
        _map_lyapunov_transient(config),
        _map_lyapunov_iterations(config),
        config.lyapunov_perturbation,
        config.divergence_cutoff;
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=min_crossing_time,
    )
    if result.estimation_status == :collapsed
        return :non_chaotic
    elseif result.estimation_status != :ok || !isfinite(result.exponent)
        return :unresolved
    elseif result.exponent > config.lyapunov_neutral_tolerance
        return :chaotic
    else
        return :non_chaotic
    end
end

function _rc_region_basin_counts(
    sys::Union{DiscreteMap, ContinuousODE},
    basins_result::BasinsResult,
    params::AbstractVector,
    config::RobustChaosRegionConfig;
    solver,
    reltol::Float64,
    abstol::Float64,
    min_crossing_time::Float64,
)
    x_grid = basins_result.x_grid
    y_grid = basins_result.y_grid
    ic = copy(basins_result.ic_template)
    n_total = 0
    n_resolved = 0
    n_chaotic = 0
    counts = Dict{Symbol, Int}()
    for (j, y_val) in enumerate(y_grid), (i, x_val) in enumerate(x_grid)
        n_total += 1
        period = basins_result.periodicity[i, j]
        copyto!(ic, basins_result.ic_template)
        ic[basins_result.x_index] = x_val
        ic[basins_result.y_index] = y_val
        cls = if period > 0
            :periodic
        elseif sys isa ContinuousODE
            _rc_region_classify_basin_seed(
                sys, ic, params, config.lyapunov_field;
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time,
            )
        else
            _rc_region_classify_basin_seed(sys, ic, params, config.lyapunov_field)
        end
        counts[cls] = get(counts, cls, 0) + 1
        if cls == :chaotic
            n_resolved += 1
            n_chaotic += 1
        elseif cls in (:periodic, :non_chaotic)
            n_resolved += 1
        end
    end
    return (n_total=n_total, n_resolved=n_resolved, n_chaotic=n_chaotic, counts=counts)
end

function _rc_region_layer_verdicts(
    sys::Union{DiscreteMap, ContinuousODE},
    config::RobustChaosRegionConfig,
    cells::Vector{AdaptiveMapLeafCell},
    lyapunov_result::LyapunovFieldResult,
    boundary::RegimeBoundaryResult;
    initial_point::Union{Nothing, AbstractVector},
    solver,
    reltol::Float64,
    abstol::Float64,
    min_crossing_time::Float64,
    log::Union{Nothing, Function},
)
    grid_indices = _rc_region_grid_indices(cells, lyapunov_result.a_grid, lyapunov_result.b_grid)
    positive_code = _map_lyapunov_status_code(:chaotic_candidate)
    resolved_codes = Set([
        _map_lyapunov_status_code(:chaotic_candidate),
        _map_lyapunov_status_code(:periodic),
        _map_lyapunov_status_code(:quasiperiodic_neutral_candidate),
    ])
    n_lya_total = length(grid_indices)
    n_lya_resolved = count(idx -> lyapunov_result.classification_status_codes[idx...] in resolved_codes, grid_indices)
    n_lya_positive = count(idx -> lyapunov_result.classification_status_codes[idx...] == positive_code, grid_indices)
    resolved_exps = Float64[
        lyapunov_result.exponents[idx...] for idx in grid_indices
        if lyapunov_result.classification_status_codes[idx...] in resolved_codes &&
           isfinite(lyapunov_result.exponents[idx...])
    ]
    lya_min = isempty(resolved_exps) ? NaN : minimum(resolved_exps)
    lya_resolved_frac = n_lya_total > 0 ? n_lya_resolved / n_lya_total : 0.0
    lya_positive_frac = n_lya_resolved > 0 ? n_lya_positive / n_lya_resolved : 0.0
    lya_verdict = _rc_lyapunov_verdict(
        n_lya_total,
        n_lya_resolved,
        n_lya_positive,
        config.min_lyapunov_positive_fraction,
        config.min_lyapunov_resolved_fraction,
    )

    bounds = _rc_region_bbox(cells)
    region_bounds = _rc_region_interval_and_secondary(bounds, config.slice_axis)
    secondary_centers = config.slice_axis == :a ?
        [(_rc_region_cell_center(cell))[2] for cell in cells] :
        [(_rc_region_cell_center(cell))[1] for cell in cells]
    slice_values = _rc_region_knot_values(secondary_centers, config.max_atlas_slices_per_region)
    stable_evidence_total = 0
    atlas_passed = 0
    atlas_calibration_refused = false
    atlas_coverage = Float64[]
    atlas_items = Dict{String, Any}[]
    for secondary in slice_values
        atlas_cfg, base_params = _rc_region_atlas_config(config, region_bounds, secondary)
        _rc_log!(log, "robust_chaos_region_certificate: atlas slice $(config.slice_axis) secondary=$secondary over [$(region_bounds.primary_min), $(region_bounds.primary_max)]")
        atlas_result = if sys isa ContinuousODE
            continuation_atlas(
                sys,
                atlas_cfg;
                initial_point=initial_point,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time,
                log=log,
            )
        else
            continuation_atlas(sys, atlas_cfg; initial_point=initial_point, log=log)
        end
        stable_evidence, unresolved_stability = _rc_stable_window_evidence(
            atlas_result,
            sys,
            base_params,
            atlas_cfg,
            region_bounds.primary_min,
            region_bounds.primary_max,
        )
        time_budget_exceeded = _as_bool(get(atlas_result.diagnostics, "timeBudgetExceeded", false))
        calibration_refused = _atlas_recon_calibration_refused(atlas_result.diagnostics)
        atlas_calibration_refused |= calibration_refused
        n_covered = _as_int(get(atlas_result.coverage_summary, "covered", 0))
        n_partial = _as_int(get(atlas_result.coverage_summary, "partial", 0))
        n_unresolved = _as_int(get(atlas_result.coverage_summary, "unresolved", 0))
        n_windows = n_covered + n_partial + n_unresolved
        n_gaps = length(atlas_result.gaps)
        coverage_effort = if n_windows == 0
            (!time_budget_exceeded && n_gaps == 0) ? 1.0 : 0.0
        else
            min(1.0, (n_covered / n_windows) * (time_budget_exceeded ? 0.5 : 1.0))
        end
        slice_pass = isempty(stable_evidence) && unresolved_stability == 0 &&
                     !calibration_refused &&
                     !time_budget_exceeded && n_gaps == 0 &&
                     n_partial == 0 && n_unresolved == 0
        atlas_passed += slice_pass ? 1 : 0
        stable_evidence_total += length(stable_evidence)
        push!(atlas_coverage, coverage_effort)
        push!(atlas_items, Dict{String, Any}(
            "secondary" => secondary,
            "passed" => slice_pass,
            "coverageEffort" => coverage_effort,
            "stableEvidenceCount" => length(stable_evidence),
            "unresolvedStabilityCount" => unresolved_stability,
            "timeBudgetExceeded" => time_budget_exceeded,
            "reconCalibrationRefused" => calibration_refused,
            "reconCalibration" => get(atlas_result.diagnostics, "reconCalibration", Dict{String, Any}()),
            "nGaps" => n_gaps,
            "nWindows" => n_windows,
            "nPartial" => n_partial,
            "nUnresolved" => n_unresolved,
        ))
    end
    atlas_slice_count = length(slice_values)
    atlas_pass_fraction = atlas_slice_count > 0 ? atlas_passed / atlas_slice_count : 0.0
    atlas_coverage_effort = isempty(atlas_coverage) ? 0.0 : minimum(atlas_coverage)
    atlas_verdict = if stable_evidence_total > 0
        :fail
    elseif atlas_calibration_refused
        :inconclusive
    elseif atlas_slice_count == 0
        :inconclusive
    elseif atlas_pass_fraction >= config.min_atlas_slice_fraction
        :pass
    else
        :inconclusive
    end

    basin_centers = _rc_region_sample_centers(cells, config.max_basin_knots_per_region)
    basin_total = 0
    basin_resolved = 0
    basin_chaotic = 0
    basin_items = Dict{String, Any}[]
    for (a, b) in basin_centers
        basin_cfg, params = _rc_region_basin_config(config, a, b)
        _rc_log!(log, "robust_chaos_region_certificate: basin knot a=$a b=$b")
        basin_result = if sys isa ContinuousODE
            basins_of_attraction(sys, basin_cfg; solver=solver, reltol=reltol, abstol=abstol)
        else
            basins_of_attraction(sys, basin_cfg)
        end
        counts = _rc_region_basin_counts(
            sys,
            basin_result,
            params,
            config;
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
        )
        basin_total += counts.n_total
        basin_resolved += counts.n_resolved
        basin_chaotic += counts.n_chaotic
        push!(basin_items, Dict{String, Any}(
            "a" => a,
            "b" => b,
            "nTotal" => counts.n_total,
            "nResolved" => counts.n_resolved,
            "nChaotic" => counts.n_chaotic,
            "classCounts" => Dict{String, Int}(String(k) => v for (k, v) in counts.counts),
        ))
    end
    basin_resolved_frac = basin_total > 0 ? basin_resolved / basin_total : 0.0
    basin_chaotic_frac = basin_resolved > 0 ? basin_chaotic / basin_resolved : 0.0
    basin_verdict = _rc_basin_verdict(
        basin_total,
        basin_resolved,
        basin_chaotic,
        config.min_chaotic_basin_fraction,
        config.min_basin_resolved_fraction,
    )

    boundary_indices = _rc_region_grid_indices(cells, boundary.a_grid, boundary.b_grid)
    margins = Float64[
        boundary.distance[idx...] for idx in boundary_indices
        if boundary.valid[idx...] && isfinite(boundary.distance[idx...])
    ]
    margin = isempty(margins) ? NaN : minimum(margins)
    edge_censored = any(idx -> boundary.edge_censored[idx...], boundary_indices)

    counter = String[]
    lya_verdict == :pass || push!(counter, "lyapunov:$lya_verdict positive=$(round(lya_positive_frac, digits=4)) resolved=$(round(lya_resolved_frac, digits=4))")
    atlas_verdict == :pass || push!(counter, "atlas:$atlas_verdict passed=$atlas_passed/$atlas_slice_count stable=$stable_evidence_total")
    basin_verdict == :pass || push!(counter, "basins:$basin_verdict chaotic=$(round(basin_chaotic_frac, digits=4)) resolved=$(round(basin_resolved_frac, digits=4))")
    edge_censored && push!(counter, "boundary:edge_censored")

    verdicts = (lya_verdict, atlas_verdict, basin_verdict)
    overall = if any(==(:fail), verdicts)
        :fragile
    elseif all(==(:pass), verdicts)
        :certified
    else
        :inconclusive
    end
    lya_score = lya_positive_frac * lya_resolved_frac
    atlas_score = stable_evidence_total == 0 ? atlas_coverage_effort * atlas_pass_fraction : 0.0
    basin_score = basin_chaotic_frac * basin_resolved_frac
    score = min(lya_score, atlas_score, basin_score)

    items = Dict{String, Any}[
        Dict("layer" => "lyapunov", "verdict" => String(lya_verdict),
             "nTotal" => n_lya_total, "nResolved" => n_lya_resolved,
             "nPositive" => n_lya_positive,
             "positiveFraction" => lya_positive_frac,
             "resolvedFraction" => lya_resolved_frac,
             "minResolvedExponent" => lya_min),
        Dict("layer" => "atlas", "verdict" => String(atlas_verdict),
             "sliceCount" => atlas_slice_count,
             "passedSlices" => atlas_passed,
             "coverageEffort" => atlas_coverage_effort,
             "stableEvidenceCount" => stable_evidence_total,
             "slices" => atlas_items),
        Dict("layer" => "basins", "verdict" => String(basin_verdict),
             "knotCount" => length(basin_centers),
             "nTotal" => basin_total,
             "nResolved" => basin_resolved,
             "nChaotic" => basin_chaotic,
             "chaoticFraction" => basin_chaotic_frac,
             "resolvedFraction" => basin_resolved_frac,
             "knots" => basin_items),
        Dict("layer" => "boundary", "margin" => margin,
             "edgeCensored" => edge_censored),
        Dict("layer" => "overall", "verdict" => String(overall),
             "robustnessScore" => score),
    ]

    return (
        verdict=overall,
        score=score,
        lyapunov_verdict=lya_verdict,
        lyapunov_positive_fraction=lya_positive_frac,
        lyapunov_resolved_fraction=lya_resolved_frac,
        lyapunov_min_resolved_exponent=lya_min,
        lyapunov_n_total=n_lya_total,
        lyapunov_n_resolved=n_lya_resolved,
        lyapunov_n_positive=n_lya_positive,
        atlas_verdict=atlas_verdict,
        atlas_slice_count=atlas_slice_count,
        atlas_passed_slices=atlas_passed,
        atlas_coverage_effort=atlas_coverage_effort,
        atlas_stable_evidence_count=stable_evidence_total,
        basin_verdict=basin_verdict,
        basin_knot_count=length(basin_centers),
        basin_chaotic_fraction=basin_chaotic_frac,
        basin_resolved_fraction=basin_resolved_frac,
        basin_n_total=basin_total,
        basin_n_resolved=basin_resolved,
        basin_n_chaotic=basin_chaotic,
        boundary_margin=margin,
        boundary_edge_censored=edge_censored,
        counter_evidence=counter,
        certificate_items=items,
    )
end

"""
    _rc_region_config_with_min_crossing_time(config, min_crossing_time) -> RobustChaosRegionConfig

Return a copy of `config` with `min_crossing_time` propagated into every
sub-configuration that carries the field: `map`, `lyapunov_field`, `basins`,
and `atlas.brute_force`. `ContinuationConfig` has no crossing-time field; the
atlas continuation receives the value through the
`continuation_atlas(...; min_crossing_time)` keyword instead.

Each `@set` rebuilds the whole (immutable) config with one nested field
replaced and returns the new root, which must be reassigned. The sequential
rebuilds cost a few small-struct allocations on a path taken once per
certification and are kept for readability.
"""
function _rc_region_config_with_min_crossing_time(
    config::RobustChaosRegionConfig,
    min_crossing_time::Float64,
)::RobustChaosRegionConfig
    config = Accessors.@set config.map.min_crossing_time = min_crossing_time
    config = Accessors.@set config.lyapunov_field.min_crossing_time = min_crossing_time
    config = Accessors.@set config.basins.min_crossing_time = min_crossing_time
    config = Accessors.@set config.atlas.brute_force.min_crossing_time = min_crossing_time
    return config
end

"""
    robust_chaos_region_certificate(sys, config; kwargs...) -> RobustChaosRegionResult

Certify robust-chaos regions in a two-parameter operating plane. The analysis is
conservative: candidate regions come from adaptive map cells, unresolved
refinement or failed evidence layers block certification, and every verdict is
bounded to the recorded finite grid, slice, basin, and continuation budgets.
"""
function robust_chaos_region_certificate(
    sys::Union{DiscreteMap, ContinuousODE},
    config::RobustChaosRegionConfig;
    initial_point::Union{Nothing, AbstractVector} = nothing,
    solver = Tsit5(),
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
    min_crossing_time::Float64 = config.map.min_crossing_time,
    log::Union{Nothing, Function} = nothing,
    compute_backend::ComputeBackend = cpu_backend(),
)::RobustChaosRegionResult
    if sys isa ContinuousODE
        config = _rc_region_config_with_min_crossing_time(config, min_crossing_time)
    end

    _rc_log!(log, "robust_chaos_region_certificate: adaptive candidate map")
    adaptive = if sys isa ContinuousODE
        adaptive_bifurcation_map(
            sys,
            config.map,
            config.adaptive;
            initial_point=initial_point,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            backend=compute_backend,
        )
    else
        adaptive_bifurcation_map(
            sys,
            config.map,
            config.adaptive;
            initial_point=initial_point,
            backend=compute_backend,
        )
    end

    _rc_log!(log, "robust_chaos_region_certificate: Lyapunov field")
    lyapunov_result = if sys isa ContinuousODE
        lyapunov_field(
            sys,
            config.lyapunov_field;
            initial_point=initial_point,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            backend=compute_backend,
        )
    else
        lyapunov_field(
            sys,
            config.lyapunov_field;
            initial_point=initial_point,
            backend=compute_backend,
        )
    end

    status_codes = _rc_region_coarse_status_codes(adaptive)
    boundary = regime_boundary_distances(
        adaptive.coarse_result;
        status_codes=status_codes,
        config=RegimeBoundaryConfig(edge_policy=config.boundary_edge_policy),
    )

    candidate_codes = Set(_map_status_code(status) for status in config.candidate_statuses)
    candidate_leaf_indices = Int[
        idx for idx in eachindex(adaptive.leaf_cells)
        if _rc_region_leaf_candidate(adaptive, adaptive.leaf_cells[idx], candidate_codes)
    ]
    components = _rc_region_components(adaptive, candidate_leaf_indices)

    regions = RobustChaosRegion[]
    rejected = 0
    for component in components
        cells = [adaptive.leaf_cells[idx] for idx in component]
        area = sum(_rc_region_cell_area, cells)
        if area < config.min_region_area
            rejected += length(component)
            continue
        end
        bounds = _rc_region_bbox(cells)
        layers = _rc_region_layer_verdicts(
            sys,
            config,
            cells,
            lyapunov_result,
            boundary;
            initial_point=initial_point,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
            log=log,
        )
        push!(regions, RobustChaosRegion(
            length(regions) + 1,
            layers.verdict,
            layers.score,
            bounds.a_min,
            bounds.a_max,
            bounds.b_min,
            bounds.b_max,
            area,
            length(cells),
            maximum(cell.depth for cell in cells),
            minimum(cell.depth for cell in cells),
            layers.lyapunov_verdict,
            layers.lyapunov_positive_fraction,
            layers.lyapunov_resolved_fraction,
            layers.lyapunov_min_resolved_exponent,
            layers.lyapunov_n_total,
            layers.lyapunov_n_resolved,
            layers.lyapunov_n_positive,
            layers.atlas_verdict,
            layers.atlas_slice_count,
            layers.atlas_passed_slices,
            layers.atlas_coverage_effort,
            layers.atlas_stable_evidence_count,
            layers.basin_verdict,
            layers.basin_knot_count,
            layers.basin_chaotic_fraction,
            layers.basin_resolved_fraction,
            layers.basin_n_total,
            layers.basin_n_resolved,
            layers.basin_n_chaotic,
            layers.boundary_margin,
            layers.boundary_edge_censored,
            layers.counter_evidence,
            layers.certificate_items,
        ))
    end

    items = Dict{String, Any}[
        Dict("layer" => "adaptive_map",
             "candidate_leaf_count" => length(candidate_leaf_indices),
             "component_count" => length(components),
             "budget_used" => adaptive.budget_used,
             "total_budget" => adaptive.total_budget,
             "budget_exhausted" => adaptive.budget_exhausted,
             "uninspected_cell_count" => adaptive.uninspected_cell_count,
             "max_depth_reached" => adaptive.max_depth_reached,
             "max_depth_allowed" => adaptive.max_depth_allowed),
        Dict("layer" => "lyapunov_field",
             "method" => String(lyapunov_result.lyapunov_method),
             "normalization" => String(lyapunov_result.normalization),
             "grid" => [length(lyapunov_result.a_grid), length(lyapunov_result.b_grid)]),
        Dict("layer" => "boundary_margin",
             "edge_policy" => String(boundary.edge_policy),
             "status_evidence" => boundary.status_evidence),
        Dict("layer" => "overall",
             "region_count" => length(regions),
             "certified_region_count" => count(region -> region.verdict == :certified, regions),
             "fragile_region_count" => count(region -> region.verdict == :fragile, regions),
             "inconclusive_region_count" => count(region -> region.verdict == :inconclusive, regions)),
    ]

    return RobustChaosRegionResult(
        regions,
        sys.name,
        config.map.a_index <= length(sys.param_names) && config.map.b_index <= length(sys.param_names) ?
            (sys.param_names[config.map.a_index], sys.param_names[config.map.b_index]) :
            (:a, :b),
        length(candidate_leaf_indices),
        rejected,
        adaptive.budget_used,
        adaptive.total_budget,
        adaptive.budget_exhausted,
        adaptive.uninspected_cell_count,
        adaptive.max_depth_reached,
        adaptive.max_depth_allowed,
        lyapunov_result.lyapunov_method,
        lyapunov_result.normalization,
        boundary.edge_policy,
        now(),
        items,
    )
end
