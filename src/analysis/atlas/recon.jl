"""Classify a grouped orbit sample against the requested maximum period."""
function _atlas_classify_sample(points::Vector{Vector{Float64}},
                                max_period::Int,
                                precision::Float64,
                                param_value::Float64)
    effective_max_period = max(max_period, 2)
    geometry = _orbit_geometry_summary(points)
    closure_errors = _orbit_closure_errors(points, effective_max_period)
    scale = isempty(geometry.span) ? 1.0 : max(maximum(abs.(geometry.span)), 1.0)
    threshold = max(precision, precision * scale)
    orbit_tail = length(points) >= effective_max_period ? points[end - effective_max_period + 1:end] : points
    detected = length(orbit_tail) >= 2 ? _detect_period(orbit_tail, length(orbit_tail), threshold) : 0
    best_period = detected > 0 && detected <= max_period ? detected : 0
    confidence = best_period > 0 && isfinite(closure_errors[best_period]) ? clamp(1 - closure_errors[best_period] / max(threshold, eps(Float64)), 0.0, 1.0) : 0.0
    classification = isempty(points) ? :insufficient : best_period > 0 ? :periodic : :nonperiodic
    return AtlasReconSample(
        param_value,
        classification,
        best_period,
        confidence,
        closure_errors,
        copy(points),
        copy(geometry.center),
        copy(geometry.span),
        Dict(
            "threshold" => threshold,
            "pointCount" => length(points)
        )
    )
end

"""Return a reconnaissance sample with merged provenance/diagnostic metadata."""
function _atlas_with_recon_diagnostics(sample::AtlasReconSample, updates::AbstractDict{<:AbstractString, <:Any})
    return AtlasReconSample(
        sample.param,
        sample.classification,
        sample.best_period,
        sample.confidence,
        copy(sample.closure_errors),
        [copy(point) for point in sample.support_points],
        copy(sample.orbit_center),
        copy(sample.orbit_span),
        merge(copy(sample.diagnostics), Dict{String, Any}(String(k) => v for (k, v) in pairs(updates)))
    )
end

"""Classify one reconnaissance parameter value for a discrete map."""
_atlas_recon_max_period(periods::Vector{Int}) = max(maximum(periods), 2)
_atlas_discrete_recon_iterations(bf_config::BruteForceConfig, periods::Vector{Int}) =
    max(bf_config.iterations, bf_config.transient + _atlas_recon_max_period(periods) + 4)
_atlas_continuous_recon_crossings(periods::Vector{Int}) = max(_atlas_recon_max_period(periods) + 4, 8)
_atlas_continuous_recon_transient(bf_config::BruteForceConfig) =
    max(4, min(bf_config.transient, max(4, bf_config.iterations ÷ 2)))

function _atlas_recon_sample(sys::DiscreteMap,
                             param_value::Float64,
                             base_params::Vector{Float64},
                             bf_config::BruteForceConfig,
                             atlas_config::AtlasConfig,
                             periods::Vector{Int};
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             solver=nothing,
                              reltol::Float64=1e-8,
                              abstol::Float64=1e-8,
                              min_crossing_time::Float64=1e-6)
    max_period = _atlas_recon_max_period(periods)
    iterations = _atlas_discrete_recon_iterations(bf_config, periods)
    local_params = inject_param(base_params, bf_config.param_index, param_value, bf_config.linked_param_indices)
    orbit = _sample_discrete_orbit(
        sys,
        local_params;
        initial_point=initial_point,
        iterations=iterations,
        transient=bf_config.transient
    )
    return _atlas_classify_sample(orbit.points, max_period, atlas_config.recon_precision, param_value)
end

"""Classify one reconnaissance parameter value for a continuous Poincaré map."""
function _atlas_recon_sample(sys::ContinuousODE,
                             param_value::Float64,
                             base_params::Vector{Float64},
                             bf_config::BruteForceConfig,
                             atlas_config::AtlasConfig,
                             periods::Vector{Int};
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             solver=Tsit5(),
                              reltol::Float64=1e-8,
                              abstol::Float64=1e-8,
                              min_crossing_time::Float64=1e-6)
    max_period = _atlas_recon_max_period(periods)
    crossings = _atlas_continuous_recon_crossings(periods)
    transient = _atlas_continuous_recon_transient(bf_config)
    local_params = inject_param(base_params, bf_config.param_index, param_value, bf_config.linked_param_indices)
    orbit = _sample_continuous_poincare_orbit(
        sys,
        local_params;
        initial_point=isnothing(initial_point) ? copy(sys.default_initial_state) : initial_point,
        crossings=crossings,
        transient=transient,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=min_crossing_time
    )
    sample = _atlas_classify_sample(orbit.points, max_period, atlas_config.recon_precision, param_value)
    return _atlas_with_recon_diagnostics(sample, Dict(
        "crossingDiagnostics" => orbit.crossing_diagnostics
    ))
end

"""Return candidate reconnaissance sample indices for a window, ordered by confidence and locality."""
function _atlas_candidate_sample_indices(window::AtlasWindow, samples::Vector{AtlasReconSample})
    midpoint = (window.param_min + window.param_max) / 2
    indices = [idx for idx in window.sample_indices if samples[idx].classification == :periodic && samples[idx].best_period == window.period]
    if !isempty(indices)
        return sort(indices; by=idx -> (-samples[idx].confidence, abs(samples[idx].param - midpoint)))
    end

    fallback = copy(window.sample_indices)
    return sort(fallback; by=idx -> begin
        sample = samples[idx]
        threshold = Float64(get(sample.diagnostics, "threshold", 1.0))
        closure = length(sample.closure_errors) >= window.period ? sample.closure_errors[window.period] : Inf
        confidence = isfinite(closure) ? clamp(1 - closure / max(threshold, eps(Float64)), 0.0, 1.0) : 0.0
        (-confidence, abs(sample.param - midpoint))
    end)
end

"""Return the proper divisors of a requested period."""
_atlas_proper_divisors(period::Int) = [d for d in 1:(period - 1) if period % d == 0]

"""Return the reconnaissance closure error for one requested period when available."""
function _atlas_period_closure_error(sample::AtlasReconSample, period::Int)
    period <= 0 && return Inf
    length(sample.closure_errors) >= period || return Inf
    return sample.closure_errors[period]
end

"""Return whether a recon sample hides a genuine higher-period candidate not explained by a subharmonic."""
function _atlas_hidden_period_candidate(sample::AtlasReconSample, period::Int)
    sample.classification == :periodic && sample.best_period == period && return false
    threshold = Float64(get(sample.diagnostics, "threshold", Inf))
    isfinite(threshold) || return false
    closure = _atlas_period_closure_error(sample, period)
    isfinite(closure) && closure <= threshold || return false
    return all(_atlas_period_closure_error(sample, divisor) > threshold for divisor in _atlas_proper_divisors(period))
end

"""Return recon sample indices that strongly suggest a hidden target period via closure-error evidence."""
function _atlas_hidden_period_sample_indices(samples::Vector{AtlasReconSample}, period::Int)
    indices = Int[]
    for (idx, sample) in enumerate(samples)
        _atlas_hidden_period_candidate(sample, period) && push!(indices, idx)
    end
    return indices
end

"""Build probe-only atlas windows for requested periods that were missed by the exact-period segmentation."""
function _atlas_hidden_period_probe_windows(samples::Vector{AtlasReconSample}, atlas_config::AtlasConfig, periods::Vector{Int}, existing_windows::Vector{AtlasWindow})
    probe_windows = AtlasWindow[]

    for period in periods
        hidden_indices = [
            idx for idx in _atlas_hidden_period_sample_indices(samples, period)
            if !_atlas_sample_overlaps_period_window(samples[idx], period, existing_windows)
        ]
        isempty(hidden_indices) && continue

        singleton_segments = [(idx, idx) for idx in hidden_indices]
        merged_segments = _atlas_merge_window_segments(singleton_segments, atlas_config.window_merge_gap)
        for (segment_idx, (seg_lo, seg_hi)) in enumerate(merged_segments)
            candidate_indices = [idx for idx in hidden_indices if seg_lo <= idx <= seg_hi]
            isempty(candidate_indices) && continue
            lo_idx = max(first(candidate_indices) - 1, 1)
            hi_idx = min(last(candidate_indices) + 1, length(samples))
            confidence = _atlas_aggregate_confidence(begin
                sample = samples[idx]
                threshold = Float64(get(sample.diagnostics, "threshold", 1.0))
                closure = _atlas_period_closure_error(sample, period)
                clamp(1 - closure / max(threshold, eps(Float64)), 0.0, 1.0)
            end for idx in candidate_indices)
            width = abs(samples[hi_idx].param - samples[lo_idx].param)
            push!(probe_windows, AtlasWindow(
                "atlas-probe-window-$(period)-$(segment_idx)-$(lo_idx)-$(hi_idx)",
                period,
                samples[lo_idx].param,
                samples[hi_idx].param,
                length(candidate_indices),
                confidence,
                :probe_hidden_period,
                copy(candidate_indices),
                _window_priority_score(period, length(candidate_indices), width, confidence) * 0.9,
                :untried,
                Dict(
                    "probeType" => "hidden-period",
                    "candidateIndices" => copy(candidate_indices),
                    "expandedStartIndex" => lo_idx,
                    "expandedEndIndex" => hi_idx
                )
            ))
        end
    end

    return probe_windows
end

"""Return whether a reconnaissance sample is already covered by an existing same-period window."""
function _atlas_sample_overlaps_period_window(sample::AtlasReconSample,
                                             period::Int,
                                             windows::Vector{AtlasWindow})
    p = sample.param
    for window in windows
        window.period == period || continue
        lo = min(window.param_min, window.param_max)
        hi = max(window.param_min, window.param_max)
        tol = max(100eps(Float64), 100eps(max(abs(lo), abs(hi), abs(p), 1.0)))
        lo - tol <= p <= hi + tol && return true
    end
    return false
end

"""Return normalized closure error for adaptive reconnaissance triggers."""
function _atlas_normalized_closure_error(sample::AtlasReconSample, period::Int)
    closure = _atlas_period_closure_error(sample, period)
    isfinite(closure) || return Inf
    threshold = Float64(get(sample.diagnostics, "threshold", 1.0))
    return closure / max(threshold, eps(Float64))
end

"""Return trigger reasons for refining the interval between two reconnaissance samples."""
function _atlas_adaptive_recon_reasons(left::AtlasReconSample,
                                       right::AtlasReconSample,
                                       periods::Vector{Int},
                                       atlas_config::AtlasConfig)
    reasons = String[]

    if left.classification != right.classification || left.best_period != right.best_period
        push!(reasons, "classification-change")
    end

    if any(sample -> sample.classification == :periodic &&
                     sample.confidence < atlas_config.adaptive_recon_confidence_threshold,
           (left, right))
        push!(reasons, "low-confidence-periodic")
    end

    hidden = any(sample -> any(period -> _atlas_hidden_period_candidate(sample, period), periods), (left, right))
    hidden && push!(reasons, "hidden-period-candidate")

    gradient_triggered = false
    for period in periods
        left_norm = _atlas_normalized_closure_error(left, period)
        right_norm = _atlas_normalized_closure_error(right, period)
        if isfinite(left_norm) && isfinite(right_norm) &&
           abs(left_norm - right_norm) >= atlas_config.adaptive_recon_closure_gradient_factor
            gradient_triggered = true
            break
        end
    end
    gradient_triggered && push!(reasons, "closure-gradient")

    return reasons
end

"""Score adaptive reconnaissance triggers; higher scores are sampled first under budget."""
function _atlas_adaptive_recon_score(reasons::Vector{String}, left::AtlasReconSample, right::AtlasReconSample)
    score = 0.0
    "hidden-period-candidate" in reasons && (score += 4.0)
    "classification-change" in reasons && (score += 3.0)
    "low-confidence-periodic" in reasons && (score += 2.0)
    "closure-gradient" in reasons && (score += 1.5)
    score += abs(right.param - left.param)
    return score
end

"""Candidate midpoint samples for adaptive reconnaissance."""
function _atlas_adaptive_recon_candidates(samples::Vector{AtlasReconSample},
                                          periods::Vector{Int},
                                          atlas_config::AtlasConfig)
    ordered = sort(samples; by=sample -> sample.param)
    candidates = NamedTuple{(:param, :score, :reasons, :left_param, :right_param), Tuple{Float64, Float64, Vector{String}, Float64, Float64}}[]
    length(ordered) < 2 && return candidates

    for idx in 1:(length(ordered) - 1)
        left = ordered[idx]
        right = ordered[idx + 1]
        left.param == right.param && continue
        reasons = _atlas_adaptive_recon_reasons(left, right, periods, atlas_config)
        isempty(reasons) && continue
        midpoint = (left.param + right.param) / 2
        push!(candidates, (
            param=midpoint,
            score=_atlas_adaptive_recon_score(reasons, left, right),
            reasons=reasons,
            left_param=left.param,
            right_param=right.param
        ))
    end

    sort!(candidates; by=candidate -> (-candidate.score, candidate.param))
    return candidates
end

"""Return whether a parameter value is already represented in the reconnaissance grid."""
function _atlas_recon_param_present(samples::Vector{AtlasReconSample}, param_value::Float64)
    for sample in samples
        tol = max(100eps(Float64), 100eps(max(abs(sample.param), abs(param_value), 1.0)))
        abs(sample.param - param_value) <= tol && return true
    end
    return false
end

"""Insert budgeted midpoint reconnaissance samples before window segmentation."""
function _atlas_adaptive_reconnaissance(sys::DynamicalSystem,
                                        samples::Vector{AtlasReconSample},
                                        base_params::Vector{Float64},
                                        bf_config::BruteForceConfig,
                                        atlas_config::AtlasConfig,
                                        periods::Vector{Int};
                                        initial_point::Union{Nothing, AbstractVector}=nothing,
                                        solver=Tsit5(),
                                        reltol::Float64=1e-8,
                                        abstol::Float64=1e-8,
                                        min_crossing_time::Float64=1e-6)
    diagnostics = Dict{String, Any}(
        "requested" => atlas_config.adaptive_recon,
        "applied" => false,
        "status" => atlas_config.adaptive_recon ? "no_candidates" : "disabled",
        "initialSampleCount" => length(samples),
        "adaptiveSampleCount" => 0,
        "finalSampleCount" => length(samples),
        "candidateCount" => 0,
        "passCount" => 0,
        "maxSamples" => atlas_config.adaptive_recon_max_samples,
        "maxDepth" => atlas_config.adaptive_recon_max_depth,
        "reasonCounts" => Dict{String, Int}()
    )

    if !atlas_config.adaptive_recon
        return sort(samples; by=sample -> sample.param), diagnostics
    end
    if atlas_config.adaptive_recon_max_samples <= 0 || atlas_config.adaptive_recon_max_depth <= 0 || length(samples) < 2
        diagnostics["status"] = "budget_exhausted"
        return sort(samples; by=sample -> sample.param), diagnostics
    end

    working = sort(samples; by=sample -> sample.param)
    sampled_count = 0
    reason_counts = Dict{String, Int}()

    for depth in 1:atlas_config.adaptive_recon_max_depth
        candidates = [
            candidate for candidate in _atlas_adaptive_recon_candidates(working, periods, atlas_config)
            if !_atlas_recon_param_present(working, candidate.param)
        ]
        diagnostics["candidateCount"] = Int(diagnostics["candidateCount"]) + length(candidates)
        isempty(candidates) && break

        remaining = atlas_config.adaptive_recon_max_samples - sampled_count
        remaining <= 0 && break
        selected = first(candidates, min(remaining, length(candidates)))
        new_samples = AtlasReconSample[]

        for candidate in selected
            raw_sample = _atlas_recon_sample(
                sys,
                candidate.param,
                base_params,
                bf_config,
                atlas_config,
                periods;
                initial_point=initial_point,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time
            )
            sample = _atlas_with_recon_diagnostics(raw_sample, Dict(
                "reconSource" => "adaptive",
                "adaptiveDepth" => depth,
                "adaptiveReasons" => copy(candidate.reasons),
                "adaptiveReason" => join(candidate.reasons, ","),
                "adaptiveLeftParam" => candidate.left_param,
                "adaptiveRightParam" => candidate.right_param
            ))
            push!(new_samples, sample)
            for reason in candidate.reasons
                reason_counts[reason] = get(reason_counts, reason, 0) + 1
            end
        end

        append!(working, new_samples)
        sort!(working; by=sample -> sample.param)
        sampled_count += length(new_samples)
        diagnostics["passCount"] = depth
        sampled_count >= atlas_config.adaptive_recon_max_samples && break
    end

    diagnostics["adaptiveSampleCount"] = sampled_count
    diagnostics["finalSampleCount"] = length(working)
    diagnostics["reasonCounts"] = reason_counts
    diagnostics["applied"] = sampled_count > 0
    diagnostics["status"] = sampled_count > 0 ? "applied" :
        (Int(diagnostics["candidateCount"]) == 0 ? "no_candidates" : "budget_exhausted")

    return working, diagnostics
end

function _atlas_reconnaissance_samples(sample_fn::Function, param_values::Vector{Float64}, atlas_config::AtlasConfig)
    chunks = _balanced_index_chunks(length(param_values), atlas_config.threaded ? Threads.nthreads() : 1)
    buffers = Vector{Vector{AtlasReconSample}}(undef, length(chunks))

    if length(chunks) <= 1
        for chunk_idx in eachindex(chunks)
            chunk = chunks[chunk_idx]
            local_samples = Vector{AtlasReconSample}(undef, length(chunk))
            for (local_idx, param_idx) in enumerate(chunk)
                local_samples[local_idx] = sample_fn(param_values[param_idx])
            end
            buffers[chunk_idx] = local_samples
        end
    else
        Threads.@threads for chunk_idx in eachindex(chunks)
            chunk = chunks[chunk_idx]
            local_samples = Vector{AtlasReconSample}(undef, length(chunk))
            for (local_idx, param_idx) in enumerate(chunk)
                local_samples[local_idx] = sample_fn(param_values[param_idx])
            end
            buffers[chunk_idx] = local_samples
        end
    end

    samples = AtlasReconSample[]
    sizehint!(samples, length(param_values))
    for local_samples in buffers
        append!(samples, local_samples)
    end
    return samples
end

function _atlas_uniform_recon_samples(samples::Vector{AtlasReconSample})
    return [
        sample for sample in samples
        if get(sample.diagnostics, "reconSource", nothing) == "uniform"
    ]
end

function _atlas_param_values_match(samples::Vector{AtlasReconSample}, expected_values::AbstractVector{<:Real})
    length(samples) == length(expected_values) || return false
    for (sample, expected) in zip(samples, expected_values)
        expected_value = Float64(expected)
        tol = max(100eps(Float64), 100eps(max(abs(sample.param), abs(expected_value), 1.0)))
        abs(sample.param - expected_value) <= tol || return false
    end
    return true
end

"""
Closure ratio of a reconnaissance sample in the units `recon_precision` is compared against.

`_atlas_classify_sample` locks a period when `‖x₁ − x_{1+T}‖` on the orbit tail falls below
`precision · max(‖x₁‖, ‖x_{1+T}‖, 1)`. The scale-free ratio of those two quantities is therefore
the statistic the threshold governs, and the only one a calibration may compare against it.
`sample.closure_errors` is a different quantity — a mean over all index pairs `T` apart across the
whole cloud — which on a chaotic sample reports the attractor diameter, not a closure scale.
"""
function _atlas_recon_closure_ratios(sample::AtlasReconSample, periods::Vector{Int})
    ratios = Tuple{Int, Float64}[]
    points = sample.support_points
    length(points) >= 2 || return ratios
    effective_max_period = _atlas_recon_max_period(periods)
    tail = length(points) >= effective_max_period ? points[end - effective_max_period + 1:end] : points
    length(tail) >= 2 || return ratios
    base = tail[1]
    base_norm = norm(base)
    for period in periods
        period > 0 && period + 1 <= length(tail) || continue
        candidate = tail[period + 1]
        scale = max(base_norm, norm(candidate), 1.0)
        ratio = norm(base .- candidate) / scale
        isfinite(ratio) && push!(ratios, (period, ratio))
    end
    return ratios
end

"""Requested period whose tail closure ratio is smallest, or `nothing`."""
function _atlas_best_requested_ratio(sample::AtlasReconSample, periods::Vector{Int})
    best = nothing
    best_ratio = Inf
    for pair in _atlas_recon_closure_ratios(sample, periods)
        if pair[2] < best_ratio
            best = pair
            best_ratio = pair[2]
        end
    end
    return best
end

function _atlas_recon_sample_search_box(sample::AtlasReconSample, atlas_config::AtlasConfig)
    dim = !isempty(sample.orbit_center) ? length(sample.orbit_center) :
          !isempty(sample.support_points) ? length(first(sample.support_points)) : 0
    dim > 0 || return nothing
    center = length(sample.orbit_center) == dim ? copy(sample.orbit_center) : copy(first(sample.support_points))
    span = length(sample.orbit_span) == dim ? abs.(sample.orbit_span) : zeros(Float64, dim)
    scale = max(maximum(abs, center; init=0.0), maximum(span; init=0.0), 1.0)
    floor_width = max(10 * atlas_config.recon_precision, 10 * atlas_config.recon_calibration_newton_tol, sqrt(eps(Float64)) * scale)
    half_width = [max(0.5 * span[i] * (1 + atlas_config.seed_box_padding), floor_width) for i in 1:dim]
    return center .- half_width, center .+ half_width
end

function _atlas_reclassify_recon_sample(sample::AtlasReconSample,
                                        max_period::Int,
                                        precision::Float64,
                                        updates::AbstractDict{<:AbstractString, <:Any})
    reclassified = _atlas_classify_sample(sample.support_points, max_period, precision, sample.param)
    return AtlasReconSample(
        reclassified.param,
        reclassified.classification,
        reclassified.best_period,
        reclassified.confidence,
        reclassified.closure_errors,
        reclassified.support_points,
        reclassified.orbit_center,
        reclassified.orbit_span,
        merge(copy(sample.diagnostics), reclassified.diagnostics,
              Dict{String, Any}(String(k) => v for (k, v) in pairs(updates)))
    )
end

function _atlas_verify_periodic_anchor(sys::DynamicalSystem,
                                       sample::AtlasReconSample,
                                       period::Int,
                                       base_params::Vector{Float64},
                                       bf_config::BruteForceConfig,
                                       cont_config::ContinuationConfig,
                                       atlas_config::AtlasConfig;
                                       solver=Tsit5(),
                                       reltol::Float64=1e-8,
                                       abstol::Float64=1e-8,
                                       min_crossing_time::Float64=1e-6)
    search = _atlas_recon_sample_search_box(sample, atlas_config)
    isnothing(search) && return false, "empty_orbit_support"
    search_min, search_max = search
    n_initial = max(3, min(8, length(sample.support_points)))
    tol = max(atlas_config.recon_calibration_newton_tol, cont_config.newton_tol)
    try
        common_kwargs = (
            n_initial=n_initial,
            search_min=search_min,
            search_max=search_max,
            seed_points=sample.support_points,
            params=base_params,
            param_index=bf_config.param_index,
            linked_param_indices=bf_config.linked_param_indices,
            tol=tol,
            max_iter=cont_config.newton_max_iter,
            threaded=false,
            cache_enabled=false,
        )
        ode_kwargs = sys isa ContinuousODE ? (
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
        ) : NamedTuple()
        seeds = find_periodic_skeleton(sys, [period], sample.param; common_kwargs..., ode_kwargs...)
        return any(seed -> seed.period == period, seeds), ""
    catch err
        err isa InterruptException && rethrow()
        return false, sprint(showerror, err)
    end
end

function _atlas_recon_calibration_bf_config(bf_config::BruteForceConfig,
                                            atlas_config::AtlasConfig,
                                            periods::Vector{Int})
    post_count = max(_atlas_recon_max_period(periods) + 4, 8)
    transient = bf_config.transient + atlas_config.recon_calibration_transient_multiplier * post_count
    iterations = max(bf_config.iterations, transient + post_count)
    return BruteForceConfig(
        param_min=bf_config.param_min,
        param_max=bf_config.param_max,
        param_steps=bf_config.param_steps,
        iterations=iterations,
        transient=transient,
        param_index=bf_config.param_index,
        fixed_params=copy(bf_config.fixed_params),
        linked_param_indices=copy(bf_config.linked_param_indices),
        min_crossing_time=bf_config.min_crossing_time,
    )
end

function _atlas_recon_calibration_sample(sys::DynamicalSystem,
                                         sample::AtlasReconSample,
                                         base_params::Vector{Float64},
                                         bf_config::BruteForceConfig,
                                         atlas_config::AtlasConfig,
                                         periods::Vector{Int};
                                         initial_point::Union{Nothing, AbstractVector}=nothing,
                                         solver=Tsit5(),
                                         reltol::Float64=1e-8,
                                         abstol::Float64=1e-8,
                                         min_crossing_time::Float64=1e-6)
    calibration_bf = _atlas_recon_calibration_bf_config(bf_config, atlas_config, periods)
    return _atlas_recon_sample(
        sys,
        sample.param,
        base_params,
        calibration_bf,
        atlas_config,
        periods;
        initial_point=initial_point,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=min_crossing_time,
    )
end

function _atlas_calibrate_recon_precision(sys::DynamicalSystem,
                                          samples::Vector{AtlasReconSample},
                                          base_params::Vector{Float64},
                                          bf_config::BruteForceConfig,
                                          cont_config::ContinuationConfig,
                                          atlas_config::AtlasConfig,
                                          periods::Vector{Int};
                                          initial_point::Union{Nothing, AbstractVector}=nothing,
                                          solver=Tsit5(),
                                          reltol::Float64=1e-8,
                                          abstol::Float64=1e-8,
                                          min_crossing_time::Float64=1e-6)
    diagnostics = Dict{String, Any}(
        "requested" => atlas_config.recon_calibration == :auto,
        "mode" => string(atlas_config.recon_calibration),
        "configuredPrecision" => atlas_config.recon_precision,
        "effectivePrecision" => atlas_config.recon_precision,
        "status" => atlas_config.recon_calibration == :auto ? "pending" : "fixed",
        "periodicAnchorCount" => 0,
        "aperiodicAnchorCount" => 0,
        "periodicAnchors" => Dict{String, Any}[],
        "aperiodicAnchors" => Dict{String, Any}[],
        "verificationFailures" => Dict{String, Any}[],
        "minSeparation" => atlas_config.recon_calibration_min_separation,
    )
    atlas_config.recon_calibration == :auto || return samples, diagnostics, atlas_config

    max_period = _atlas_recon_max_period(periods)
    candidates = Tuple{Int, Int, Float64}[]
    for (idx, sample) in enumerate(samples)
        best = _atlas_best_requested_ratio(sample, periods)
        isnothing(best) && continue
        period, ratio = best
        push!(candidates, (idx, period, ratio))
    end
    sort!(candidates; by=item -> item[3])

    # Both scales are closure ratios, so the floor is relative: a verified periodic orbit cannot
    # close tighter than the solver/Newton tolerance that produced it.
    solver_floor = max(100 * max(reltol, abstol, atlas_config.recon_calibration_newton_tol),
                       sqrt(eps(Float64)))
    # A sample is periodic *evidence* only if its own orbit closes to numerical precision. Newton
    # verification alone is not sufficient: a chaotic attractor contains embedded periodic orbits, so
    # the search box around any chaotic sample yields a genuine fixed point whose existence says
    # nothing about the sampled orbit. Admitting those would inflate the noise floor to the
    # attractor's recurrence scale and destroy the separation the calibration exists to measure.
    periodic_admissible = solver_floor * atlas_config.recon_calibration_min_separation
    periodic_ratios = Float64[]
    periodic_keys = Set{Tuple{Int, Int}}()
    for (idx, period, ratio) in candidates
        length(periodic_ratios) >= atlas_config.recon_calibration_max_periodic_anchors && break
        # Candidates are sorted by ascending ratio, so the first inadmissible one ends the search.
        ratio <= periodic_admissible || break
        verified, err = _atlas_verify_periodic_anchor(
            sys,
            samples[idx],
            period,
            base_params,
            bf_config,
            cont_config,
            atlas_config;
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
        )
        if verified
            push!(periodic_ratios, max(ratio, 0.0))
            push!(periodic_keys, (idx, period))
            push!(diagnostics["periodicAnchors"], Dict{String, Any}(
                "param" => samples[idx].param,
                "period" => period,
                "closureRatio" => ratio,
            ))
        elseif !isempty(err)
            push!(diagnostics["verificationFailures"], Dict{String, Any}(
                "param" => samples[idx].param,
                "period" => period,
                "error" => err,
            ))
        end
    end

    noise_floor = isempty(periodic_ratios) ? solver_floor : max(maximum(periodic_ratios), solver_floor)
    noise_source = isempty(periodic_ratios) ? "solver_tolerance_floor" : "newton_verified_periodic"
    diagnostics["periodicAnchorCount"] = length(periodic_ratios)
    diagnostics["noiseFloor"] = noise_floor
    diagnostics["noiseSource"] = noise_source
    diagnostics["scaleUnits"] = "closure_ratio"

    recurrence_scales = Float64[]
    # Ascending order matters: the recurrence scale is the *tightest* false closure chaos achieves,
    # so the anchors that bound it are the smallest-ratio aperiodic samples. Walking from the
    # largest ratio down samples the least recurrent points and overestimates the scale, which
    # would push the chosen threshold above the false closures it exists to exclude.
    for (idx, period, ratio) in candidates
        length(recurrence_scales) >= atlas_config.recon_calibration_max_aperiodic_anchors && break
        (idx, period) in periodic_keys && continue
        ratio > noise_floor * atlas_config.recon_calibration_min_separation || continue
        # Aperiodicity is confirmed by escalating the transient, not by Newton: a nearby verified
        # orbit is compatible with a chaotic sample (see the periodic-anchor note above), so using
        # verification as a disqualifier here would reject every anchor on a chaotic band and leave
        # the calibration with nothing to measure the recurrence scale from.
        escalated = _atlas_recon_calibration_sample(
            sys,
            samples[idx],
            base_params,
            bf_config,
            atlas_config,
            periods;
            initial_point=initial_point,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
        )
        escalated_best = _atlas_best_requested_ratio(escalated, periods)
        isnothing(escalated_best) && continue
        escalated_period, escalated_ratio = escalated_best
        recurrence = min(ratio, escalated_ratio)
        recurrence > noise_floor * atlas_config.recon_calibration_min_separation || continue
        push!(recurrence_scales, recurrence)
        push!(diagnostics["aperiodicAnchors"], Dict{String, Any}(
            "param" => samples[idx].param,
            "initialBestPeriod" => period,
            "initialClosureRatio" => ratio,
            "escalatedBestPeriod" => escalated_period,
            "escalatedClosureRatio" => escalated_ratio,
            "recurrenceScale" => recurrence,
        ))
    end
    diagnostics["aperiodicAnchorCount"] = length(recurrence_scales)

    if isempty(recurrence_scales)
        diagnostics["status"] = "refused_insufficient_aperiodic_anchors"
        tagged = [_atlas_with_recon_diagnostics(sample, Dict(
            "thresholdSource" => "calibration_refused",
            "calibrationStatus" => diagnostics["status"],
        )) for sample in samples]
        return tagged, diagnostics, atlas_config
    end

    recurrence_scale = minimum(recurrence_scales)
    separation = recurrence_scale / max(noise_floor, eps(Float64))
    diagnostics["recurrenceScale"] = recurrence_scale
    diagnostics["separation"] = separation
    diagnostics["separationMarginDecades"] = log10(separation)
    if separation <= atlas_config.recon_calibration_min_separation
        diagnostics["status"] = "refused_nonseparating_scales"
        tagged = [_atlas_with_recon_diagnostics(sample, Dict(
            "thresholdSource" => "calibration_refused",
            "calibrationStatus" => diagnostics["status"],
        )) for sample in samples]
        return tagged, diagnostics, atlas_config
    end

    effective_precision = sqrt(noise_floor * recurrence_scale)
    diagnostics["effectivePrecision"] = effective_precision
    diagnostics["status"] = "applied"
    calibrated = [
        _atlas_reclassify_recon_sample(sample, max_period, effective_precision, Dict(
            "thresholdSource" => "auto_calibrated",
            "calibrationStatus" => "applied",
        ))
        for sample in samples
    ]
    effective_config = Accessors.@set atlas_config.recon_precision = effective_precision
    return calibrated, diagnostics, effective_config
end

function _atlas_recon_calibration_refused(diagnostics::AbstractDict)
    calibration = get(diagnostics, "reconCalibration", nothing)
    calibration isa AbstractDict || return false
    status = String(get(calibration, "status", ""))
    return startswith(status, "refused")
end

function _atlas_recon_point_dim(sys::DynamicalSystem, samples::Vector{AtlasReconSample})
    for sample in samples, point in sample.support_points
        return length(point)
    end
    return sys isa ContinuousODE ? length(sys.section.projection) : sys.dim
end

function _atlas_bruteforce_from_recon(sys::DynamicalSystem,
                                      bf_config::BruteForceConfig,
                                      samples::Vector{AtlasReconSample})
    dim = _atlas_recon_point_dim(sys, samples)
    n_total = sum(length(sample.support_points) for sample in samples)
    flat_params = Vector{Float64}(undef, n_total)
    points_mat = Matrix{Float64}(undef, n_total, dim)
    write_idx = 0
    @inbounds for sample in samples
        for point in sample.support_points
            write_idx += 1
            flat_params[write_idx] = sample.param
            for col in 1:dim
                points_mat[write_idx, col] = Float64(point[col])
            end
        end
    end
    return BruteForceResult(flat_params, points_mat, sys.name, sys.param_names[bf_config.param_index], now())
end

function _atlas_recon_bruteforce_reuse_diagnostics(; reused::Bool,
                                                     reason::AbstractString,
                                                     uniform_count::Int,
                                                     expected_count::Int,
                                                     adaptive_count::Int=0,
                                                     point_count::Int=0)
    return Dict{String, Any}(
        "reused" => reused,
        "reason" => String(reason),
        "uniformSampleCount" => uniform_count,
        "expectedSampleCount" => expected_count,
        "adaptiveSamplesIgnored" => adaptive_count,
        "pointCount" => point_count
    )
end

function _atlas_recon_bruteforce_reuse_candidate(sys::DynamicalSystem,
                                                 bf_config::BruteForceConfig,
                                                 atlas_config::AtlasConfig,
                                                 periods::Vector{Int},
                                                 samples::Vector{AtlasReconSample})
    uniform_samples = _atlas_uniform_recon_samples(samples)
    expected_count = bf_config.param_steps + 1
    adaptive_count = length(samples) - length(uniform_samples)
    base_diag_kwargs = (
        uniform_count=length(uniform_samples),
        expected_count=expected_count,
        adaptive_count=adaptive_count
    )

    atlas_config.recon_steps == expected_count || return nothing, _atlas_recon_bruteforce_reuse_diagnostics(;
        reused=false,
        reason="recon_steps_mismatch",
        base_diag_kwargs...
    )
    expected_values = collect(range(bf_config.param_min, bf_config.param_max, length=expected_count))
    _atlas_param_values_match(uniform_samples, expected_values) || return nothing, _atlas_recon_bruteforce_reuse_diagnostics(;
        reused=false,
        reason="parameter_grid_mismatch",
        base_diag_kwargs...
    )

    n_keep = bf_config.iterations - bf_config.transient
    n_keep >= 0 || return nothing, _atlas_recon_bruteforce_reuse_diagnostics(;
        reused=false,
        reason="negative_post_transient_count",
        base_diag_kwargs...
    )

    if sys isa ContinuousODE
        _atlas_continuous_recon_crossings(periods) == n_keep || return nothing, _atlas_recon_bruteforce_reuse_diagnostics(;
            reused=false,
            reason="continuous_crossing_count_mismatch",
            base_diag_kwargs...
        )
        _atlas_continuous_recon_transient(bf_config) == bf_config.transient || return nothing, _atlas_recon_bruteforce_reuse_diagnostics(;
            reused=false,
            reason="continuous_transient_mismatch",
            base_diag_kwargs...
        )
    else
        _atlas_discrete_recon_iterations(bf_config, periods) == bf_config.iterations || return nothing, _atlas_recon_bruteforce_reuse_diagnostics(;
            reused=false,
            reason="discrete_iteration_count_mismatch",
            base_diag_kwargs...
        )
    end

    all(length(sample.support_points) == n_keep for sample in uniform_samples) || return nothing, _atlas_recon_bruteforce_reuse_diagnostics(;
        reused=false,
        reason="point_count_mismatch",
        base_diag_kwargs...
    )

    result = _atlas_bruteforce_from_recon(sys, bf_config, uniform_samples)
    return result, _atlas_recon_bruteforce_reuse_diagnostics(;
        reused=true,
        reason=adaptive_count > 0 ? "uniform_recon_subset" : "exact_uniform_recon",
        point_count=length(result.params),
        base_diag_kwargs...
    )
end

function _atlas_bruteforce_result(sys::DiscreteMap,
                                  bf_config::BruteForceConfig,
                                  atlas_config::AtlasConfig,
                                  periods::Vector{Int},
                                  recon_samples::Vector{AtlasReconSample};
                                  initial_point::Union{Nothing, AbstractVector}=nothing,
                                  log::Union{Nothing, Function}=nothing,
                                  solver=nothing,
                                  reltol::Float64=1e-8,
                                  abstol::Float64=1e-8)
    reused_result, diagnostics = _atlas_recon_bruteforce_reuse_candidate(sys, bf_config, atlas_config, periods, recon_samples)
    if !isnothing(reused_result)
        _atlas_log!(log, "Atlas brute-force plot cloud reused $(diagnostics["pointCount"]) point(s) from reconnaissance samples.")
        return reused_result, diagnostics
    end
    result = brute_force_diagram(sys, bf_config; initial_point=initial_point)
    return result, diagnostics
end

function _atlas_bruteforce_result(sys::ContinuousODE,
                                  bf_config::BruteForceConfig,
                                  atlas_config::AtlasConfig,
                                  periods::Vector{Int},
                                  recon_samples::Vector{AtlasReconSample};
                                  initial_point::Union{Nothing, AbstractVector}=nothing,
                                  log::Union{Nothing, Function}=nothing,
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8,
                                  abstol::Float64=1e-8)
    reused_result, diagnostics = _atlas_recon_bruteforce_reuse_candidate(sys, bf_config, atlas_config, periods, recon_samples)
    if !isnothing(reused_result)
        _atlas_log!(log, "Atlas brute-force plot cloud reused $(diagnostics["pointCount"]) point(s) from reconnaissance samples.")
        return reused_result, diagnostics
    end
    result = brute_force_diagram(sys, bf_config;
        initial_point=isnothing(initial_point) ? copy(sys.default_initial_state) : initial_point,
        solver=solver,
        reltol=reltol,
        abstol=abstol)
    return result, diagnostics
end

"""Run a coarse reconnaissance sweep to classify likely finite-period windows."""
function _atlas_reconnaissance(sys::DiscreteMap,
                               base_params::Vector{Float64},
                               bf_config::BruteForceConfig,
                               atlas_config::AtlasConfig,
                               periods::Vector{Int};
                               initial_point::Union{Nothing, AbstractVector}=nothing)
    param_values = collect(range(bf_config.param_min, bf_config.param_max, length=max(atlas_config.recon_steps, 2)))
    return _atlas_reconnaissance_samples(param_values, atlas_config) do param_value
        sample = _atlas_recon_sample(
            sys,
            Float64(param_value),
            base_params,
            bf_config,
            atlas_config,
            periods;
            initial_point=initial_point,
            solver=nothing
        )
        _atlas_with_recon_diagnostics(sample, Dict("reconSource" => "uniform", "adaptiveDepth" => 0))
    end
end

"""Run a coarse reconnaissance sweep to classify likely finite-period windows."""
function _atlas_reconnaissance(sys::ContinuousODE,
                               base_params::Vector{Float64},
                               bf_config::BruteForceConfig,
                               atlas_config::AtlasConfig,
                               periods::Vector{Int};
                               initial_point::Union{Nothing, AbstractVector}=nothing,
                               solver=Tsit5(),
                               reltol::Float64=1e-8,
                               abstol::Float64=1e-8,
                               min_crossing_time::Float64=1e-6)
    param_values = collect(range(bf_config.param_min, bf_config.param_max, length=max(atlas_config.recon_steps, 2)))
    return _atlas_reconnaissance_samples(param_values, atlas_config) do param_value
        sample = _atlas_recon_sample(
            sys,
            Float64(param_value),
            base_params,
            bf_config,
            atlas_config,
            periods;
            initial_point=initial_point,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time
        )
        _atlas_with_recon_diagnostics(sample, Dict("reconSource" => "uniform", "adaptiveDepth" => 0))
    end
end

"""Build exact contiguous segments for one requested period."""
function _atlas_raw_window_segments(samples::Vector{AtlasReconSample}, period::Int)
    segments = Vector{Tuple{Int, Int}}()
    idx = 1
    while idx <= length(samples)
        if samples[idx].best_period != period || samples[idx].classification != :periodic
            idx += 1
            continue
        end
        stop_idx = idx
        while stop_idx < length(samples) && samples[stop_idx + 1].best_period == period && samples[stop_idx + 1].classification == :periodic
            stop_idx += 1
        end
        push!(segments, (idx, stop_idx))
        idx = stop_idx + 1
    end
    return segments
end

"""Merge nearby same-period windows across short uncertain gaps."""
function _atlas_merge_window_segments(segments::Vector{Tuple{Int, Int}}, merge_gap::Int)
    isempty(segments) && return segments
    merged = [segments[1]]
    for (lo, hi) in Iterators.drop(segments, 1)
        prev_lo, prev_hi = merged[end]
        if lo - prev_hi - 1 <= merge_gap
            merged[end] = (prev_lo, hi)
        else
            push!(merged, (lo, hi))
        end
    end
    return merged
end

"""Compute a simple priority score for candidate windows."""
_window_priority_score(period::Int, support::Int, width::Float64, confidence::Float64) = support + width + confidence + 0.1 / max(period, 1)

"""
Aggregate per-sample confidences for a candidate window.

Bayesian-style shrinkage toward 0.5 with `k_smooth` pseudo-observations: a single
sample at 0.9 confidence is treated as weaker evidence than ten samples at 0.7,
which matches our preference for windows backed by more reconnaissance support.
Output is clamped to `[0, 1]` as a defensive invariant.

Accepts any iterable (vector, generator, etc.) and traverses it once so the
caller's closures are only invoked one time per element.
"""
function _atlas_aggregate_confidence(confidences; k_smooth::Float64=3.0)
    support = 0
    total = 0.0
    for conf in confidences
        support += 1
        total += conf
    end
    support == 0 && return 0.0
    # support * mean_conf == total, so the shrinkage simplifies to one expression.
    shrunk = (total + k_smooth * 0.5) / (support + k_smooth)
    return clamp(shrunk, 0.0, 1.0)
end

"""Segment reconnaissance samples into candidate windows for the requested periods."""
function _segment_period_windows(samples::Vector{AtlasReconSample}, atlas_config::AtlasConfig, periods::Vector{Int})
    windows = AtlasWindow[]

    for period in periods
        raw_segments = _atlas_raw_window_segments(samples, period)
        merged_segments = _atlas_merge_window_segments(raw_segments, atlas_config.window_merge_gap)
        keepers = Tuple{Int, Int}[]
        if !isempty(merged_segments)
            supports = Int[count(sample.best_period == period for sample in samples[lo:hi]) for (lo, hi) in merged_segments]
            keepers = [segment for (segment, support) in zip(merged_segments, supports) if support >= atlas_config.window_min_support]
            if isempty(keepers)
                push!(keepers, merged_segments[argmax(supports)])
            end
        end

        for (segment_idx, (lo, hi)) in enumerate(keepers)
            period_samples = [sample for sample in samples[lo:hi] if sample.best_period == period]
            support = length(period_samples)
            confidence = _atlas_aggregate_confidence(sample.confidence for sample in period_samples)
            width = abs(samples[hi].param - samples[lo].param)
            push!(windows, AtlasWindow(
                "atlas-window-$(period)-$(segment_idx)-$(lo)-$(hi)",
                period,
                samples[lo].param,
                samples[hi].param,
                support,
                confidence,
                :periodic,
                collect(lo:hi),
                _window_priority_score(period, support, width, confidence),
                :untried,
                Dict("startIndex" => lo, "endIndex" => hi)
            ))
        end
    end

    ordered = sort(windows; by=window -> (-window.priority_score, window.period, window.param_min))
    return first(ordered, min(length(ordered), atlas_config.max_total_windows))
end
