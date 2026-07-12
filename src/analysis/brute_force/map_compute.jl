function _process_discrete_map_tile!(periodicity::AbstractMatrix{Int},
                                     status_codes::AbstractMatrix{Int},
                                     closure_errors::AbstractMatrix{Float64},
                                     closure_candidate_periods::AbstractMatrix{Int},
                                      observed_points::AbstractMatrix{Int},
                                      closure_confidence::AbstractMatrix{Float64},
                                      switching_diagnostics,
                                      lyapunov_storage,
                                      sys::DiscreteMap,
                                      config::BifurcationMapConfig,
                                     param_template::Vector{Float64},
                                     a_indices::Vector{Int},
                                     b_indices::Vector{Int},
                                     a_vals::Vector{Float64},
                                     b_vals::Vector{Float64},
                                      x0::SVector,
                                      seed_mode::Symbol,
                                      points_to_drop::Int,
                                      neighbor_transient,
                                      tile)
    seed = x0
    first_cell = true
    reset_count = 0
    invalid_reset_count = 0
    param_buffer = copy(param_template)

    for (local_row, i) in enumerate(tile.a_range)
        js = isodd(local_row) ? tile.b_range : reverse(tile.b_range)
        for j in js
            p = _map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
            transient = seed_mode == :neighbor_accelerated && !first_cell ? neighbor_transient : points_to_drop
            result = _detect_discrete_map_period(sys, p, seed, transient, config.max_period, config.precision, config.divergence_cutoff)
            _record_map_detection!(
                periodicity,
                status_codes,
                closure_errors,
                closure_candidate_periods,
                observed_points,
                closure_confidence,
                i,
                j,
                result,
                nothing,
                switching_diagnostics
            )
            _record_discrete_map_lyapunov!(lyapunov_storage, sys, config, p, i, j, result)
            if result.valid
                seed = result.final_point
                first_cell = false
            else
                seed = x0
                first_cell = true
                reset_count += 1
                invalid_reset_count += 1
            end
        end
    end

    return (resets=reset_count, invalid_resets=invalid_reset_count)
end

function _process_continuous_map_tile!(periodicity::AbstractMatrix{Int},
                                       status_codes::AbstractMatrix{Int},
                                       closure_errors::AbstractMatrix{Float64},
                                       closure_candidate_periods::AbstractMatrix{Int},
                                        observed_points::AbstractMatrix{Int},
                                        closure_confidence::AbstractMatrix{Float64},
                                        crossing_summary,
                                        lyapunov_storage,
                                        sys::ContinuousODE,
                                        config::BifurcationMapConfig,
                                       param_template::Vector{Float64},
                                       a_indices::Vector{Int},
                                       b_indices::Vector{Int},
                                       a_vals::Vector{Float64},
                                       b_vals::Vector{Float64},
                                       u0::Vector{Float64},
                                       seed_mode::Symbol,
                                       points_to_drop::Int,
                                       neighbor_transient,
                                       solver,
                                       reltol::Float64,
                                       abstol::Float64,
                                       tile)
    seed = copy(u0)
    first_cell = true
    reset_count = 0
    invalid_reset_count = 0
    param_buffer = copy(param_template)

    for (local_row, i) in enumerate(tile.a_range)
        js = isodd(local_row) ? tile.b_range : reverse(tile.b_range)
        for j in js
            p = _map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
            transient = seed_mode == :neighbor_accelerated && !first_cell ? neighbor_transient : points_to_drop
            result = _detect_continuous_poincare_period(
                sys,
                p;
                initial_point=seed,
                transient=transient,
                max_period=config.max_period,
                precision=config.precision,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                projected=true,
                divergence_cutoff=config.divergence_cutoff,
                min_crossing_time=config.min_crossing_time,
                return_crossing_diagnostics=false
            )
            _record_map_detection!(
                periodicity,
                status_codes,
                closure_errors,
                closure_candidate_periods,
                observed_points,
                closure_confidence,
                i,
                j,
                result
            )
            _record_map_crossing_summary!(crossing_summary, i, j, result)
            _record_continuous_map_lyapunov!(
                lyapunov_storage,
                sys,
                config,
                p,
                i,
                j,
                result;
                solver=solver,
                reltol=reltol,
                abstol=abstol
            )
            if result.valid
                seed = result.final_point
                first_cell = false
            else
                seed = copy(u0)
                first_cell = true
                reset_count += 1
                invalid_reset_count += 1
            end
        end
    end

    return (resets=reset_count, invalid_resets=invalid_reset_count)
end

_state_exceeds_cutoff(state, cutoff::Float64) = _map_state_status(state, cutoff) != :ok

function _closure_period(base, candidate, period::Int, precision::Float64, base_norm::Float64)
    closure = _closure_measure(base, candidate, precision, base_norm)
    return closure.error < closure.threshold ? period : 0
end

function _detect_discrete_map_period(sys::DiscreteMap,
                                     params::AbstractVector,
                                     initial_point::SVector{D, Float64},
                                     transient::Int,
                                     max_period::Int,
                                     precision::Float64,
                                     divergence_cutoff::Float64) where {D}
    has_switching_events = !isempty(switching_events(sys))
    switching_samples = Vector{Vector{Float64}}()
    switching_diag() = has_switching_events ? switching_event_diagnostics(sys, switching_samples, params) : Dict{String, Any}()
    max_period <= 0 && return merge(
        _period_detection_result(0, :invalid_state, Inf, 0, 0, Inf),
        (final_point=initial_point, switching_diagnostics=switching_diag())
    )

    point = initial_point
    for _ in 1:transient
        point = sys.f(point, params)
        status = _map_state_status(point, divergence_cutoff)
        status != :ok && return merge(
            _period_detection_result(0, status, Inf, 0, 0, Inf),
            (
                final_point=point,
                switching_diagnostics=has_switching_events ?
                    switching_event_diagnostics(sys, [collect(Float64, point)], params) :
                    Dict{String, Any}()
            )
        )
    end

    base = sys.f(point, params)
    has_switching_events && push!(switching_samples, collect(Float64, base))
    status = _map_state_status(base, divergence_cutoff)
    status != :ok && return merge(
        _period_detection_result(0, status, Inf, 0, 0, Inf),
        (final_point=base, switching_diagnostics=switching_diag())
    )
    base_norm = norm(base)
    point = base
    min_error = Inf
    min_period = 0
    min_threshold = Inf
    for period in 1:max_period
        point = sys.f(point, params)
        has_switching_events && push!(switching_samples, collect(Float64, point))
        status = _map_state_status(point, divergence_cutoff)
        status != :ok && return merge(
            _period_detection_result(0, status, min_error, min_period, period, min_threshold),
            (final_point=point, switching_diagnostics=switching_diag())
        )
        closure = _closure_measure(base, point, precision, base_norm)
        if closure.error < min_error
            min_error = closure.error
            min_period = period
            min_threshold = closure.threshold
        end
        if closure.error < closure.threshold
            return merge(
                _period_detection_result(period, :periodic, min_error, min_period, period + 1, min_threshold),
                (final_point=point, switching_diagnostics=switching_diag())
            )
        end
    end
    return merge(
        _period_detection_result(0, :aperiodic_or_high_period, min_error, min_period, max_period + 1, min_threshold),
        (final_point=point, switching_diagnostics=switching_diag())
    )
end

function _divergence_callback(cutoff::Float64)
    isfinite(cutoff) || return nothing
    condition = (u, t, integrator) -> _state_exceeds_cutoff(u, cutoff)
    return DiscreteCallback(condition, terminate!; save_positions=(false, false))
end

function _map_state_termination_callback(cutoff::Float64,
                                         status_ref::Base.RefValue{Symbol},
                                         final_point::Base.RefValue{Vector{Float64}},
                                         final_time::Base.RefValue{Float64})
    condition = (u, t, integrator) -> _map_state_status(u, cutoff) != :ok
    affect! = function(integrator)
        status_ref[] = _map_state_status(integrator.u, cutoff)
        final_point[] = collect(Float64, integrator.u)
        final_time[] = Float64(integrator.t)
        terminate!(integrator)
    end
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

function _detect_continuous_poincare_period(sys::ContinuousODE, params::AbstractVector;
                                            initial_point::Union{Nothing, AbstractVector}=nothing,
                                            transient::Int,
                                            max_period::Int,
                                            precision::Float64,
                                            solver=Tsit5(),
                                            reltol::Float64=1e-8,
                                            abstol::Float64=1e-8,
                                            projected::Bool=true,
                                            tmax::Union{Nothing, Float64}=nothing,
                                            maxiters::Int=10_000_000,
                                            min_crossing_time::Float64=1e-6,
                                            divergence_cutoff::Float64=Inf,
                                            return_crossing_diagnostics::Bool=true)
    initial_state = _resolve_initial_state(sys, initial_point)
    if max_period <= 0
        result = (
            _period_detection_result(0, :invalid_state, Inf, 0, 0, Inf)...,
            final_point=initial_state,
            total_crossings_found=0,
            final_time=0.0,
            termination_reason=:invalid_state,
            solver_retcode=:not_run,
            divergence_callback_activated=false,
            state_callback_activated=false
        )
        return return_crossing_diagnostics ? merge(
            result,
            (
                crossing_diagnostics=_poincare_crossing_diagnostics(
                    crossings_requested=0,
                    transient_crossings=transient,
                    total_crossings_requested=transient,
                    crossings_found=0,
                    total_crossings_found=0,
                    final_time=0.0,
                    termination_reason=:invalid_state,
                    final_status=:invalid_state,
                    divergence_callback_activated=false,
                    state_callback_activated=false,
                    solver_retcode="not_run"
                ),
            )
        ) : result
    end

    initial_dt = _default_poincare_initial_dt(min_crossing_time)
    u0 = _plain_float_vector(_warmup_from_section(
        sys,
        initial_state,
        params;
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        maxiters=maxiters,
        min_crossing_time=min_crossing_time,
        initial_dt=initial_dt
    ))
    total_crossings = transient + max_period + 1
    local_tmax = isnothing(tmax) ? sys.tspan_hint * max(total_crossings, 1) * 2 : tmax

    crossing_count = Ref(0)
    base_point = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    base_norm = Ref(0.0)
    detected_period = Ref(0)
    final_point = Ref(copy(u0))
    enough_crossings = Ref(false)
    observed_points = Ref(0)
    min_error = Ref(Inf)
    min_period = Ref(0)
    min_threshold = Ref(Inf)
    terminal_status = Ref(:running)
    final_time = Ref(NaN)
    solver_retcode = Ref(:not_run)

    function on_crossing!(integrator)
        crossing_count[] += 1
        final_point[] = collect(Float64, integrator.u)
        final_time[] = Float64(integrator.t)
        if crossing_count[] > transient
            local_idx = crossing_count[] - transient
            observed_points[] = local_idx
            point = projected ? _project_section_state(sys.section, integrator.u) : collect(Float64, integrator.u)
            if local_idx == 1
                base_point[] = point
                base_norm[] = norm(point)
            else
                base = base_point[]
                if !isnothing(base)
                    closure = _closure_measure(base, point, precision, base_norm[])
                    if closure.error < min_error[]
                        min_error[] = closure.error
                        min_period[] = local_idx - 1
                        min_threshold[] = closure.threshold
                    end
                    if closure.error < closure.threshold
                        detected_period[] = local_idx - 1
                        enough_crossings[] = true
                        terminal_status[] = :periodic
                        terminate!(integrator)
                        return
                    end
                end
            end
        end
        if crossing_count[] >= total_crossings
            enough_crossings[] = true
            terminal_status[] = :aperiodic_or_high_period
            terminate!(integrator)
        end
    end

    section_cb = _make_poincare_callback(sys.section, on_crossing!; min_crossing_time=min_crossing_time)
    state_cb = _map_state_termination_callback(divergence_cutoff, terminal_status, final_point, final_time)
    cb = CallbackSet(section_cb, state_cb)
    prob = ODEProblem(sys.f, u0, (0.0, local_tmax), collect(Float64, params))

    # save_*=false is intentional: this detector never reads states from `sol`.
    # final_point and final_time are captured in `on_crossing!` (on every Poincaré
    # crossing) and in the state-termination callback (on divergence), so they hold
    # the last *section crossing* — the on-section point neighbor seeding and the
    # Lyapunov estimator need — not the off-section solver endpoint at local_tmax.
    # If no crossing ever fires, final_point stays at the warmed-up start (u0), which
    # is the correct fallback when the orbit never reaches the section.
    sol = try
        solve(
            prob,
            solver;
            callback=cb,
            reltol=reltol,
            abstol=abstol,
            dt=initial_dt,
            save_everystep=false,
            save_start=false,
            save_end=false,
            maxiters=maxiters
        )
    catch err
        result = (
            _period_detection_result(0, :integration_failed, min_error[], min_period[], observed_points[], min_threshold[])...,
            final_point=copy(u0),
            total_crossings_found=crossing_count[],
            final_time=final_time[],
            termination_reason=:integration_failed,
            solver_retcode=:exception,
            divergence_callback_activated=false,
            state_callback_activated=false
        )
        return return_crossing_diagnostics ? merge(
            result,
            (
                crossing_diagnostics=_poincare_crossing_diagnostics(
                    crossings_requested=max_period + 1,
                    transient_crossings=transient,
                    total_crossings_requested=total_crossings,
                    crossings_found=observed_points[],
                    total_crossings_found=crossing_count[],
                    final_time=final_time[],
                    termination_reason=:integration_failed,
                    final_status=:integration_failed,
                    divergence_callback_activated=false,
                    state_callback_activated=false,
                    solver_retcode="exception",
                    error_message=_continuation_error_message(err)
                ),
            )
        ) : result
    end
    solver_retcode[] = Symbol(sol.retcode)
    # final_time is captured per crossing (and by the state-termination callback on
    # divergence), so it is only still NaN when no crossing ever fired and the orbit
    # did not diverge. Prefer the integration horizon only when the solver genuinely
    # reached it; for an early/failed exit (MaxIters, DtLessThanMin, …) leave the time
    # unknown rather than misreporting a full-horizon end time.
    final_time[] = isfinite(final_time[]) ? final_time[] :
        (SciMLBase.successful_retcode(sol.retcode) ? local_tmax : NaN)

    if terminal_status[] == :diverged || terminal_status[] == :invalid_state
        status = terminal_status[]
    elseif detected_period[] > 0
        status = :periodic
    elseif enough_crossings[]
        status = :aperiodic_or_high_period
    else
        status = :insufficient_crossings
    end
    termination_reason = if status == :periodic
        :period_detected
    elseif status == :aperiodic_or_high_period
        :max_crossings_reached
    elseif status == :diverged || status == :invalid_state
        status
    else
        :insufficient_crossings
    end

    result = (
        _period_detection_result(detected_period[], status, min_error[], min_period[], observed_points[], min_threshold[])...,
        final_point=final_point[],
        total_crossings_found=crossing_count[],
        final_time=final_time[],
        termination_reason=termination_reason,
        solver_retcode=solver_retcode[],
        divergence_callback_activated=status == :diverged,
        state_callback_activated=status == :diverged || status == :invalid_state
    )
    return return_crossing_diagnostics ? merge(
        result,
        (
            crossing_diagnostics=_poincare_crossing_diagnostics(
                crossings_requested=max_period + 1,
                transient_crossings=transient,
                total_crossings_requested=total_crossings,
                crossings_found=observed_points[],
                total_crossings_found=crossing_count[],
                final_time=final_time[],
                termination_reason=termination_reason,
                final_status=status,
                divergence_callback_activated=status == :diverged,
                state_callback_activated=status == :diverged || status == :invalid_state,
                solver_retcode=String(solver_retcode[])
            ),
        )
    ) : result
end

function _map_adaptive_budget(config::BifurcationMapConfig, base_cell_count::Int)
    !config.adaptive_refinement_enabled && return 0
    config.adaptive_refinement_max_depth <= 0 && return 0
    base_cell_count <= 0 && return 0
    if config.adaptive_refinement_budget > 0
        return config.adaptive_refinement_budget
    end
    return min(max(4 * base_cell_count, 1), 4096)
end

_map_adaptive_point_key(a::Real, b::Real) = (Float64(a), Float64(b))

function _map_adaptive_sample(a::Real, b::Real, period::Integer, status_code::Integer, confidence::Real)
    return (
        a=Float64(a),
        b=Float64(b),
        period=Int(period),
        status_code=Int(status_code),
        confidence=Float64(confidence)
    )
end

function _map_adaptive_sample_payload(a::Real, b::Real, depth::Int, detection)
    status_code = _map_status_code(detection.status)
    return Dict{String, Any}(
        "a" => Float64(a),
        "b" => Float64(b),
        "period" => Int(detection.period),
        "statusCode" => status_code,
        "status" => _map_status_label(status_code),
        "closureError" => Float64(detection.min_closure_error),
        "closureCandidatePeriod" => Int(detection.closure_candidate_period),
        "observedPoints" => Int(detection.observed_points),
        "closureConfidence" => Float64(detection.closure_confidence),
        "depth" => depth,
        "source" => "adaptive_refinement"
    )
end

function _map_adaptive_sample_from_payload(payload::AbstractDict)
    return _map_adaptive_sample(
        payload["a"],
        payload["b"],
        payload["period"],
        payload["statusCode"],
        payload["closureConfidence"]
    )
end

function _map_adaptive_detection(sys::DiscreteMap,
                                 config::BifurcationMapConfig,
                                 params::AbstractVector,
                                 initial_point,
                                 points_to_drop::Int,
                                 multistability_seeds;
                                 kwargs...)
    if !isempty(multistability_seeds)
        seed_results = [
            _detect_discrete_map_period(sys, params, seed, points_to_drop, config.max_period, config.precision, config.divergence_cutoff)
            for seed in multistability_seeds
        ]
        return _summarize_map_multistability(seed_results).selected
    end
    return _detect_discrete_map_period(sys, params, initial_point, points_to_drop, config.max_period, config.precision, config.divergence_cutoff)
end

function _map_adaptive_detection(sys::ContinuousODE,
                                 config::BifurcationMapConfig,
                                 params::AbstractVector,
                                 initial_point,
                                 points_to_drop::Int,
                                 multistability_seeds;
                                 solver=Tsit5(),
                                 reltol::Float64=1e-8,
                                 abstol::Float64=1e-8)
    if !isempty(multistability_seeds)
        seed_results = [
            _detect_continuous_poincare_period(
                sys,
                params;
                initial_point=seed,
                transient=points_to_drop,
                max_period=config.max_period,
                precision=config.precision,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                projected=true,
                divergence_cutoff=config.divergence_cutoff,
                min_crossing_time=config.min_crossing_time
            )
            for seed in multistability_seeds
        ]
        return _summarize_map_multistability(seed_results).selected
    end
    return _detect_continuous_poincare_period(
        sys,
        params;
        initial_point=initial_point,
        transient=points_to_drop,
        max_period=config.max_period,
        precision=config.precision,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        projected=true,
        divergence_cutoff=config.divergence_cutoff,
        min_crossing_time=config.min_crossing_time
    )
end

function _map_adaptive_cell_trigger(corners, config::BifurcationMapConfig)
    reasons = String[]
    periods = unique(Int[corner.period for corner in corners])
    statuses = unique(Int[corner.status_code for corner in corners])
    confidences = Float64[corner.confidence for corner in corners if isfinite(corner.confidence)]
    !isempty(periods) && length(periods) > 1 && push!(reasons, "period")
    !isempty(statuses) && length(statuses) > 1 && push!(reasons, "status")
    min_confidence = isempty(confidences) ? nothing : minimum(confidences)
    max_confidence = isempty(confidences) ? nothing : maximum(confidences)
    if !isnothing(min_confidence) && config.adaptive_refinement_min_confidence > 0.0 &&
            min_confidence < config.adaptive_refinement_min_confidence
        push!(reasons, "low_confidence")
    end
    if !isnothing(min_confidence) && !isnothing(max_confidence) &&
            config.adaptive_refinement_confidence_delta > 0.0 &&
            max_confidence - min_confidence >= config.adaptive_refinement_confidence_delta
        push!(reasons, "confidence_delta")
    end
    return (
        refine=!isempty(reasons),
        reasons=reasons,
        min_confidence=min_confidence,
        max_confidence=max_confidence
    )
end

function _map_adaptive_cell_payload(cell, trigger)
    return Dict{String, Any}(
        "aMin" => Float64(cell.a0),
        "aMax" => Float64(cell.a1),
        "bMin" => Float64(cell.b0),
        "bMax" => Float64(cell.b1),
        "depth" => Int(cell.depth),
        "baseI" => Int(cell.base_i),
        "baseJ" => Int(cell.base_j),
        "reasons" => copy(trigger.reasons),
        "cornerPeriods" => Int[corner.period for corner in cell.corners],
        "cornerStatusCodes" => Int[corner.status_code for corner in cell.corners],
        "minClosureConfidence" => trigger.min_confidence,
        "maxClosureConfidence" => trigger.max_confidence
    )
end

function _map_adaptive_child_cells(cell, samples)
    a0, a1, b0, b1 = cell.a0, cell.a1, cell.b0, cell.b1
    am = (a0 + a1) / 2
    bm = (b0 + b1) / 2
    c00, c10, c01, c11 = cell.corners
    e_b0, e_b1, e_a0, e_a1, center = samples
    depth = cell.depth + 1
    return [
        (a0=a0, a1=am, b0=b0, b1=bm, corners=(c00, e_b0, e_a0, center), depth=depth, base_i=cell.base_i, base_j=cell.base_j),
        (a0=am, a1=a1, b0=b0, b1=bm, corners=(e_b0, c10, center, e_a1), depth=depth, base_i=cell.base_i, base_j=cell.base_j),
        (a0=a0, a1=am, b0=bm, b1=b1, corners=(e_a0, center, c01, e_b1), depth=depth, base_i=cell.base_i, base_j=cell.base_j),
        (a0=am, a1=a1, b0=bm, b1=b1, corners=(center, e_a1, e_b1, c11), depth=depth, base_i=cell.base_i, base_j=cell.base_j)
    ]
end

function _map_adaptive_refinement_diagnostics(sys::DynamicalSystem,
                                              config::BifurcationMapConfig,
                                              a_vals::AbstractVector{<:Real},
                                              b_vals::AbstractVector{<:Real},
                                              periodicity::AbstractMatrix{<:Integer},
                                              status_codes::AbstractMatrix{<:Integer},
                                              closure_confidence::AbstractMatrix{<:Real},
                                              param_template::Vector{Float64},
                                              a_indices::Vector{Int},
                                              b_indices::Vector{Int},
                                              initial_point,
                                              points_to_drop::Int,
                                              multistability_seeds;
                                              solver=Tsit5(),
                                              reltol::Float64=1e-8,
                                              abstol::Float64=1e-8)
    base_cell_count = max(length(a_vals) - 1, 0) * max(length(b_vals) - 1, 0)
    budget = _map_adaptive_budget(config, base_cell_count)
    diagnostics = Dict{String, Any}(
        "enabled" => config.adaptive_refinement_enabled,
        "maxDepth" => Int(config.adaptive_refinement_max_depth),
        "requestedBudget" => config.adaptive_refinement_budget > 0 ? Int(config.adaptive_refinement_budget) : nothing,
        "budget" => Int(budget),
        "automaticBudget" => config.adaptive_refinement_budget == 0,
        "minConfidence" => Float64(config.adaptive_refinement_min_confidence),
        "confidenceDelta" => Float64(config.adaptive_refinement_confidence_delta),
        "baseCellCount" => Int(base_cell_count),
        "flaggedBaseCells" => 0,
        "refinedCellCount" => 0,
        "sampleCount" => 0,
        "budgetExhausted" => false,
        "points" => Any[],
        "cells" => Any[]
    )
    config.adaptive_refinement_enabled || return diagnostics
    base_cell_count > 0 || return diagnostics
    budget > 0 || return diagnostics

    evaluated = Dict{Tuple{Float64, Float64}, Any}()
    for i in eachindex(a_vals), j in eachindex(b_vals)
        key = _map_adaptive_point_key(a_vals[i], b_vals[j])
        evaluated[key] = _map_adaptive_sample(a_vals[i], b_vals[j], periodicity[i, j], status_codes[i, j], closure_confidence[i, j])
    end

    points = Vector{Dict{String, Any}}()
    cells = Vector{Dict{String, Any}}()
    queue = Any[]
    budget_exhausted = Ref(false)

    function sample_at!(a::Real, b::Real, depth::Int)
        key = _map_adaptive_point_key(a, b)
        existing = get(evaluated, key, nothing)
        !isnothing(existing) && return existing
        if length(points) >= budget
            budget_exhausted[] = true
            return nothing
        end
        params = _map_params_from_template(param_template, a_indices, b_indices, Float64(a), Float64(b))
        detection = _map_adaptive_detection(
            sys,
            config,
            params,
            initial_point,
            points_to_drop,
            multistability_seeds;
            solver=solver,
            reltol=reltol,
            abstol=abstol
        )
        payload = _map_adaptive_sample_payload(a, b, depth, detection)
        sample = _map_adaptive_sample_from_payload(payload)
        evaluated[key] = sample
        push!(points, payload)
        return sample
    end

    for i in 1:(length(a_vals) - 1), j in 1:(length(b_vals) - 1)
        corners = (
            evaluated[_map_adaptive_point_key(a_vals[i], b_vals[j])],
            evaluated[_map_adaptive_point_key(a_vals[i + 1], b_vals[j])],
            evaluated[_map_adaptive_point_key(a_vals[i], b_vals[j + 1])],
            evaluated[_map_adaptive_point_key(a_vals[i + 1], b_vals[j + 1])]
        )
        cell = (
            a0=Float64(a_vals[i]),
            a1=Float64(a_vals[i + 1]),
            b0=Float64(b_vals[j]),
            b1=Float64(b_vals[j + 1]),
            corners=corners,
            depth=0,
            base_i=i,
            base_j=j
        )
        trigger = _map_adaptive_cell_trigger(corners, config)
        if trigger.refine
            push!(queue, (cell=cell, trigger=trigger))
        end
    end
    diagnostics["flaggedBaseCells"] = length(queue)

    head = 1
    while head <= length(queue)
        item = queue[head]
        head += 1
        cell = item.cell
        trigger = item.trigger
        push!(cells, _map_adaptive_cell_payload(cell, trigger))
        if cell.depth >= config.adaptive_refinement_max_depth || budget_exhausted[]
            continue
        end
        a0, a1, b0, b1 = cell.a0, cell.a1, cell.b0, cell.b1
        am = (a0 + a1) / 2
        bm = (b0 + b1) / 2
        next_depth = cell.depth + 1
        samples = (
            sample_at!(am, b0, next_depth),
            sample_at!(am, b1, next_depth),
            sample_at!(a0, bm, next_depth),
            sample_at!(a1, bm, next_depth),
            sample_at!(am, bm, next_depth)
        )
        any(isnothing, samples) && continue
        for child in _map_adaptive_child_cells(cell, samples)
            child_trigger = _map_adaptive_cell_trigger(child.corners, config)
            child_trigger.refine && push!(queue, (cell=child, trigger=child_trigger))
        end
    end

    diagnostics["refinedCellCount"] = length(cells)
    diagnostics["sampleCount"] = length(points)
    diagnostics["budgetExhausted"] = budget_exhausted[]
    diagnostics["points"] = points
    diagnostics["cells"] = cells
    return diagnostics
end

"""
    MapCellGrid(na, nb; lyapunov=false)

In/out per-cell state for a 2-D bifurcation-map sweep (sweep cache hook). Allocate it,
optionally pre-seed cells and mark them in `known`, pass to `bifurcation_map(...; cells=grid)`; the
sweep computes the not-`known` cells in place and you read the grid back (e.g. to store a cache
entry). The `known` mask is only consulted in the `:fixed` seed mode (the cacheable path).
"""
mutable struct MapCellGrid
    periodicity::Matrix{Int}
    status_codes::Matrix{Int}
    closure_errors::Matrix{Float64}
    closure_candidate_periods::Matrix{Int}
    observed_points::Matrix{Int}
    closure_confidence::Matrix{Float64}
    lyapunov::Union{Nothing, NamedTuple}
    known::Matrix{Bool}   # Matrix{Bool}, not BitMatrix: threaded sweeps write distinct cells concurrently
end

function MapCellGrid(na::Int, nb::Int; lyapunov::Bool=false)
    return MapCellGrid(
        zeros(Int, na, nb),
        fill(_map_status_code(:unknown), na, nb),
        fill(Inf, na, nb),
        zeros(Int, na, nb),
        zeros(Int, na, nb),
        zeros(Float64, na, nb),
        lyapunov ? _map_lyapunov_storage(na, nb) : nothing,
        fill(false, na, nb),
    )
end

function _bifurcation_map(sys::DiscreteMap, config::BifurcationMapConfig;
                          initial_point::Union{Nothing, AbstractVector}=nothing,
                          cells::Union{Nothing, MapCellGrid}=nothing)
    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    na, nb = length(a_vals), length(b_vals)

    x0 = isnothing(initial_point) ? zeros(SVector{sys.dim, Float64}) : SVector{sys.dim}(initial_point)
    multistability_enabled = _map_multistability_enabled(config)
    extra_seed_vectors = multistability_enabled ? _map_extra_seed_vectors(config, sys.dim) : Vector{Vector{Float64}}()
    multistability_seeds = multistability_enabled ?
        [x0, (SVector{sys.dim, Float64}(seed) for seed in extra_seed_vectors)...] :
        SVector{sys.dim, Float64}[]
    orbit_len = _map_orbit_window(config)
    config.iterations >= orbit_len || throw(ArgumentError(
        "BifurcationMapConfig.iterations ($(config.iterations)) must be at least max_period + 1 ($(orbit_len)); " *
        "otherwise the function would silently iterate more times than requested in order to fill the orbit window."
    ))
    points_to_drop = _map_transient_budget(config)
    seed_mode = _map_seed_mode(config, points_to_drop)
    neighbor_transient = _map_effective_neighbor_transient(config, points_to_drop)

    if cells === nothing
        periodicity = zeros(Int, na, nb)
        status_codes = fill(_map_status_code(:unknown), na, nb)
        closure_errors = fill(Inf, na, nb)
        closure_candidate_periods = zeros(Int, na, nb)
        observed_points = zeros(Int, na, nb)
        closure_confidence = zeros(Float64, na, nb)
        lyapunov_storage = _map_lyapunov_enabled(config) ? _map_lyapunov_storage(na, nb) : nothing
    else
        size(cells.periodicity) == (na, nb) || throw(ArgumentError(
            "cells grid size $(size(cells.periodicity)) does not match the ($na, $nb) sweep grid."))
        (cells.lyapunov === nothing) == !_map_lyapunov_enabled(config) || throw(ArgumentError(
            "cells.lyapunov presence must match config.lyapunov_enabled."))
        periodicity = cells.periodicity
        status_codes = cells.status_codes
        closure_errors = cells.closure_errors
        closure_candidate_periods = cells.closure_candidate_periods
        observed_points = cells.observed_points
        closure_confidence = cells.closure_confidence
        lyapunov_storage = cells.lyapunov
    end
    has_switching_events = !isempty(switching_events(sys))
    switching_diagnostics = has_switching_events ? [Dict{String, Any}() for _ in 1:na, _ in 1:nb] : nothing
    multistability_storage = multistability_enabled ? _map_multistability_storage(na, nb) : nothing

    param_template = _map_param_template(config)
    a_indices = _map_a_write_indices(config)
    b_indices = _map_b_write_indices(config)

    reset_count = 0
    invalid_reset_count = 0
    tile_diagnostics = Any[]

    if seed_mode == :fixed
        chunks = _balanced_index_chunks(na * nb, Threads.nthreads())
        Threads.@threads for chunk_idx in eachindex(chunks)
            param_buffer = copy(param_template)
            for idx in chunks[chunk_idx]
                i = ((idx - 1) % na) + 1
                j = ((idx - 1) ÷ na) + 1
                (cells !== nothing && cells.known[i, j]) && continue   # cache hook: skip pre-seeded (cached) cells
                p = _map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
                detection = if multistability_enabled
                    seed_results = [
                        _detect_discrete_map_period(sys, p, seed, points_to_drop, config.max_period, config.precision, config.divergence_cutoff)
                        for seed in multistability_seeds
                    ]
                    summary = _summarize_map_multistability(seed_results)
                    _record_map_multistability!(multistability_storage, i, j, summary)
                    summary.selected
                else
                    _detect_discrete_map_period(sys, p, x0, points_to_drop, config.max_period, config.precision, config.divergence_cutoff)
                end
                _record_map_detection!(
                    periodicity,
                    status_codes,
                    closure_errors,
                    closure_candidate_periods,
                    observed_points,
                    closure_confidence,
                    i,
                    j,
                    detection,
                    nothing,
                    switching_diagnostics
                )
                _record_discrete_map_lyapunov!(lyapunov_storage, sys, config, p, i, j, detection)
                cells !== nothing && (cells.known[i, j] = true)
            end
        end
    else
        tile_sizes = _map_effective_tile_sizes(config, na, nb, seed_mode)
        tiles = _map_tiles(na, nb, tile_sizes.tile_size_a, tile_sizes.tile_size_b)
        if length(tiles) == 1
            counts = _process_discrete_map_tile!(
                periodicity,
                status_codes,
                closure_errors,
                closure_candidate_periods,
                observed_points,
                closure_confidence,
                switching_diagnostics,
                lyapunov_storage,
                sys,
                config,
                param_template,
                a_indices,
                b_indices,
                a_vals,
                b_vals,
                x0,
                seed_mode,
                points_to_drop,
                neighbor_transient,
                only(tiles)
            )
            reset_count = counts.resets
            invalid_reset_count = counts.invalid_resets
            tile_diagnostics = [_map_tile_diagnostic(only(tiles), counts)]
        else
            reset_counts = zeros(Int, length(tiles))
            invalid_reset_counts = zeros(Int, length(tiles))
            Threads.@threads for tile_idx in eachindex(tiles)
                counts = _process_discrete_map_tile!(
                    periodicity,
                    status_codes,
                    closure_errors,
                    closure_candidate_periods,
                    observed_points,
                    closure_confidence,
                    switching_diagnostics,
                    lyapunov_storage,
                    sys,
                    config,
                    param_template,
                    a_indices,
                    b_indices,
                    a_vals,
                    b_vals,
                    x0,
                    seed_mode,
                    points_to_drop,
                    neighbor_transient,
                    tiles[tile_idx]
                )
                reset_counts[tile_idx] = counts.resets
                invalid_reset_counts[tile_idx] = counts.invalid_resets
            end
            reset_count = sum(reset_counts)
            invalid_reset_count = sum(invalid_reset_counts)
            tile_diagnostics = [
                _map_tile_diagnostic(tiles[idx], (resets=reset_counts[idx], invalid_resets=invalid_reset_counts[idx]))
                for idx in eachindex(tiles)
            ]
        end
    end

    param_names = (sys.param_names[config.a_index], sys.param_names[config.b_index])
    timestamp = now()
    lyapunov = _map_lyapunov_result(lyapunov_storage, a_vals, b_vals, config, sys.name, param_names, timestamp)
    result = BifurcationMapResult(a_vals, b_vals, periodicity, config.max_period,
                                  sys.name, param_names, timestamp, lyapunov)
    diagnostics = _map_neighbor_seed_diagnostics(
        config;
        full_transient=points_to_drop,
        na=na,
        nb=nb,
        resets=reset_count,
        invalid_resets=invalid_reset_count,
        tile_count=seed_mode == :fixed ? 0 : _map_tile_count(config, na, nb, seed_mode),
        tile_diagnostics=tile_diagnostics
    )
    diagnostics["status"] = _map_classification_diagnostics(
        status_codes,
        closure_errors,
        closure_candidate_periods,
        observed_points,
        closure_confidence
    )
    if config.adaptive_refinement_enabled
        diagnostics["adaptiveRefinement"] = _map_adaptive_refinement_diagnostics(
            sys,
            config,
            a_vals,
            b_vals,
            periodicity,
            status_codes,
            closure_confidence,
            param_template,
            a_indices,
            b_indices,
            x0,
            points_to_drop,
            multistability_seeds
        )
    end
    has_switching_events && (diagnostics["switching"] = switching_event_grid_summary(switching_diagnostics))
    multistability_enabled && (diagnostics["multistability"] = _map_multistability_diagnostics(
        multistability_storage,
        periodicity,
        length(multistability_seeds)
    ))
    _map_lyapunov_enabled(config) && (diagnostics["lyapunov"] = _map_lyapunov_diagnostics(
        lyapunov_storage,
        config,
        :two_trajectory_discrete_map
    ))
    return result, diagnostics
end

function _bifurcation_map(sys::ContinuousODE, config::BifurcationMapConfig;
                          initial_point::Union{Nothing, AbstractVector}=nothing,
                          solver=Tsit5(),
                          reltol::Float64=1e-8,
                          abstol::Float64=1e-8,
                          cells::Union{Nothing, MapCellGrid}=nothing)
    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    na, nb = length(a_vals), length(b_vals)

    u0 = _resolve_initial_state(sys, initial_point)
    multistability_enabled = _map_multistability_enabled(config)
    extra_seed_vectors = multistability_enabled ? _map_extra_seed_vectors(config, sys.dim) : Vector{Vector{Float64}}()
    multistability_seeds = multistability_enabled ? [copy(u0), (copy(seed) for seed in extra_seed_vectors)...] : Vector{Vector{Float64}}()
    crossings_needed = _map_orbit_window(config)
    config.iterations >= crossings_needed || throw(ArgumentError(
        "BifurcationMapConfig.iterations ($(config.iterations)) must be at least max_period + 1 ($(crossings_needed)); " *
        "otherwise the function would silently take more Poincaré crossings than requested in order to fill the orbit window."
    ))
    points_to_drop = _map_transient_budget(config)
    seed_mode = _map_seed_mode(config, points_to_drop)
    neighbor_transient = _map_effective_neighbor_transient(config, points_to_drop)
    if cells === nothing
        periodicity = zeros(Int, na, nb)
        status_codes = fill(_map_status_code(:unknown), na, nb)
        closure_errors = fill(Inf, na, nb)
        closure_candidate_periods = zeros(Int, na, nb)
        observed_points = zeros(Int, na, nb)
        closure_confidence = zeros(Float64, na, nb)
        lyapunov_storage = _map_lyapunov_enabled(config) ? _map_lyapunov_storage(na, nb) : nothing
    else
        size(cells.periodicity) == (na, nb) || throw(ArgumentError(
            "cells grid size $(size(cells.periodicity)) does not match the ($na, $nb) sweep grid."))
        (cells.lyapunov === nothing) == !_map_lyapunov_enabled(config) || throw(ArgumentError(
            "cells.lyapunov presence must match config.lyapunov_enabled."))
        periodicity = cells.periodicity
        status_codes = cells.status_codes
        closure_errors = cells.closure_errors
        closure_candidate_periods = cells.closure_candidate_periods
        observed_points = cells.observed_points
        closure_confidence = cells.closure_confidence
        lyapunov_storage = cells.lyapunov
    end
    crossing_summary = _map_crossing_summary_storage(na, nb, config.max_period + 1)
    multistability_storage = multistability_enabled ? _map_multistability_storage(na, nb) : nothing

    param_template = _map_param_template(config)
    a_indices = _map_a_write_indices(config)
    b_indices = _map_b_write_indices(config)

    reset_count = 0
    invalid_reset_count = 0
    tile_diagnostics = Any[]

    if seed_mode == :fixed
        chunks = _balanced_index_chunks(na * nb, Threads.nthreads())
        Threads.@threads for chunk_idx in eachindex(chunks)
            param_buffer = copy(param_template)
            for idx in chunks[chunk_idx]
                i = ((idx - 1) % na) + 1
                j = ((idx - 1) ÷ na) + 1
                (cells !== nothing && cells.known[i, j]) && continue   # cache hook: skip pre-seeded (cached) cells
                p = _map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
                result = if multistability_enabled
                    seed_results = [
                        _detect_continuous_poincare_period(
                            sys,
                            p;
                            initial_point=seed,
                            transient=points_to_drop,
                            max_period=config.max_period,
                            precision=config.precision,
                            solver=solver,
                            reltol=reltol,
                            abstol=abstol,
                            projected=true,
                            divergence_cutoff=config.divergence_cutoff,
                            min_crossing_time=config.min_crossing_time,
                            return_crossing_diagnostics=false
                        )
                        for seed in multistability_seeds
                    ]
                    summary = _summarize_map_multistability(seed_results)
                    _record_map_multistability!(multistability_storage, i, j, summary)
                    summary.selected
                else
                    _detect_continuous_poincare_period(
                        sys,
                        p;
                        initial_point=u0,
                        transient=points_to_drop,
                        max_period=config.max_period,
                        precision=config.precision,
                        solver=solver,
                        reltol=reltol,
                        abstol=abstol,
                        projected=true,
                        divergence_cutoff=config.divergence_cutoff,
                        min_crossing_time=config.min_crossing_time,
                        return_crossing_diagnostics=false
                    )
                end
                _record_map_detection!(
                    periodicity,
                    status_codes,
                    closure_errors,
                    closure_candidate_periods,
                    observed_points,
                    closure_confidence,
                    i,
                    j,
                    result
                )
                _record_map_crossing_summary!(crossing_summary, i, j, result)
                _record_continuous_map_lyapunov!(
                    lyapunov_storage,
                    sys,
                    config,
                    p,
                    i,
                    j,
                    result;
                    solver=solver,
                    reltol=reltol,
                    abstol=abstol
                )
                cells !== nothing && (cells.known[i, j] = true)
            end
        end
    else
        tile_sizes = _map_effective_tile_sizes(config, na, nb, seed_mode)
        tiles = _map_tiles(na, nb, tile_sizes.tile_size_a, tile_sizes.tile_size_b)
        if length(tiles) == 1
            counts = _process_continuous_map_tile!(
                periodicity,
                status_codes,
                closure_errors,
                closure_candidate_periods,
                observed_points,
                closure_confidence,
                crossing_summary,
                lyapunov_storage,
                sys,
                config,
                param_template,
                a_indices,
                b_indices,
                a_vals,
                b_vals,
                u0,
                seed_mode,
                points_to_drop,
                neighbor_transient,
                solver,
                reltol,
                abstol,
                only(tiles)
            )
            reset_count = counts.resets
            invalid_reset_count = counts.invalid_resets
            tile_diagnostics = [_map_tile_diagnostic(only(tiles), counts)]
        else
            reset_counts = zeros(Int, length(tiles))
            invalid_reset_counts = zeros(Int, length(tiles))
            Threads.@threads for tile_idx in eachindex(tiles)
                counts = _process_continuous_map_tile!(
                    periodicity,
                    status_codes,
                    closure_errors,
                    closure_candidate_periods,
                    observed_points,
                    closure_confidence,
                    crossing_summary,
                    lyapunov_storage,
                    sys,
                    config,
                    param_template,
                    a_indices,
                    b_indices,
                    a_vals,
                    b_vals,
                    u0,
                    seed_mode,
                    points_to_drop,
                    neighbor_transient,
                    solver,
                    reltol,
                    abstol,
                    tiles[tile_idx]
                )
                reset_counts[tile_idx] = counts.resets
                invalid_reset_counts[tile_idx] = counts.invalid_resets
            end
            reset_count = sum(reset_counts)
            invalid_reset_count = sum(invalid_reset_counts)
            tile_diagnostics = [
                _map_tile_diagnostic(tiles[idx], (resets=reset_counts[idx], invalid_resets=invalid_reset_counts[idx]))
                for idx in eachindex(tiles)
            ]
        end
    end

    param_names = (sys.param_names[config.a_index], sys.param_names[config.b_index])
    timestamp = now()
    lyapunov = _map_lyapunov_result(lyapunov_storage, a_vals, b_vals, config, sys.name, param_names, timestamp)
    result = BifurcationMapResult(a_vals, b_vals, periodicity, config.max_period,
                                  sys.name, param_names, timestamp, lyapunov)
    diagnostics = _map_neighbor_seed_diagnostics(
        config;
        full_transient=points_to_drop,
        na=na,
        nb=nb,
        resets=reset_count,
        invalid_resets=invalid_reset_count,
        tile_count=seed_mode == :fixed ? 0 : _map_tile_count(config, na, nb, seed_mode),
        tile_diagnostics=tile_diagnostics
    )
    diagnostics["status"] = _map_classification_diagnostics(
        status_codes,
        closure_errors,
        closure_candidate_periods,
        observed_points,
        closure_confidence
    )
    if config.adaptive_refinement_enabled
        diagnostics["adaptiveRefinement"] = _map_adaptive_refinement_diagnostics(
            sys,
            config,
            a_vals,
            b_vals,
            periodicity,
            status_codes,
            closure_confidence,
            param_template,
            a_indices,
            b_indices,
            u0,
            points_to_drop,
            multistability_seeds;
            solver=solver,
            reltol=reltol,
            abstol=abstol
        )
    end
    diagnostics["crossing"] = _poincare_crossing_diagnostics_summary(crossing_summary)
    multistability_enabled && (diagnostics["multistability"] = _map_multistability_diagnostics(
        multistability_storage,
        periodicity,
        length(multistability_seeds)
    ))
    _map_lyapunov_enabled(config) && (diagnostics["lyapunov"] = _map_lyapunov_diagnostics(
        lyapunov_storage,
        config,
        :two_trajectory_poincare_map
    ))
    return result, diagnostics
end

"""
    bifurcation_map(sys::DiscreteMap, config::BifurcationMapConfig;
                    initial_point=nothing) -> BifurcationMapResult

Generate a 2D bifurcation map by sweeping two parameters simultaneously.
For each (a, b) grid point, the map is iterated from `initial_point` and the
periodicity of the resulting attractor is determined.
"""
function bifurcation_map(sys::DiscreteMap, config::BifurcationMapConfig;
                         initial_point::Union{Nothing, AbstractVector}=nothing)
    result, _ = _bifurcation_map(sys, config; initial_point=initial_point)
    return result
end

"""
    bifurcation_map(sys::ContinuousODE, config::BifurcationMapConfig;
                    initial_point=nothing, solver=Tsit5(), reltol=1e-8, abstol=1e-8)
                    -> BifurcationMapResult

Generate a 2D bifurcation map for a continuous-time system by sweeping two parameters
and detecting the periodicity of the resulting Poincaré map orbit.
"""
function bifurcation_map(sys::ContinuousODE, config::BifurcationMapConfig;
                         initial_point::Union{Nothing, AbstractVector}=nothing,
                         solver=Tsit5(),
                         reltol::Float64=1e-8,
                         abstol::Float64=1e-8)
    result, _ = _bifurcation_map(
        sys,
        config;
        initial_point=initial_point,
        solver=solver,
        reltol=reltol,
        abstol=abstol
    )
    return result
end
