"""
    HardwareAcceptanceMismatch

One measured-route observation that does not directly support a certified
hardware-acceptance claim. `required_shift` is the smallest observed-parameter
shift found from the computed route or certificate bounds; `margin` is the
available local tolerance/boundary margin in the same coordinate when one can
be read from the supplied margin field.
"""
struct HardwareAcceptanceMismatch
    observation_index::Int
    measured_parameter::Float64
    aligned_parameter::Float64
    measured_mode::String
    predicted_mode::Union{Nothing, String}
    observation_status::Symbol
    required_shift::Union{Nothing, Float64}
    margin::Union{Nothing, Float64}
    margin_axis::Union{Nothing, Symbol}
    tolerance_probability::Union{Nothing, Float64}
    margin_status::Symbol
    message::String
end

"""
    HardwareAcceptanceResult

Accept/reject/refuse verdict for a measured mode route checked against a
robust-chaos certificate and optional tolerance or boundary-margin fields.

Verdicts:
- `:accepted`: all contrary observations are either absent or within supplied
  margins and the certificate verdict is accepted by the configuration.
- `:rejected`: at least one observation falsifies the certified claim, or the
  certificate itself is explicitly fragile.
- `:inconclusive`: evidence is insufficient but not falsifying.
- `:refused`: the requested axis calibration or input combination is
  underdetermined.
"""
struct HardwareAcceptanceResult
    alignment::Union{Nothing, ModeSequenceAlignment}
    certificate_kind::Symbol
    certificate_verdict::Symbol
    verdict::Symbol
    mismatches::Vector{HardwareAcceptanceMismatch}
    accepted_certificate_verdicts::Vector{Symbol}
    score::Union{Nothing, Float64}
    certificate_items::Vector{Dict{String, Any}}
    timestamp::DateTime
end

const _HARDWARE_CERTIFICATE = Union{
    RobustChaosCertificate,
    RobustChaosRegionResult,
}

function _acceptance_mode_config(base::ModeAssimilationConfig, scale::Float64, offset::Float64)
    return ModeAssimilationConfig(
        experimental_scale=scale,
        experimental_offset=offset,
        transition_tolerance=base.transition_tolerance,
        mode_aliases=base.mode_aliases,
        unresolved_modes=base.unresolved_modes,
    )
end

function _acceptance_same_transition(lhs::ModeTransition, rhs::ModeTransition,
                                     aliases::Dict{String, String})
    return _aliased_mode_key(lhs.from_mode, aliases) == _aliased_mode_key(rhs.from_mode, aliases) &&
           _aliased_mode_key(lhs.to_mode, aliases) == _aliased_mode_key(rhs.to_mode, aliases)
end

function _acceptance_transition_affine_config(experimental::ModeSequence,
                                              predicted::OperatingMapCrossSection,
                                              base::ModeAssimilationConfig)
    aliases = base.mode_aliases
    observed = mode_sequence_transitions(experimental; mode_aliases=aliases)
    theory = mode_sequence_transitions(predicted.sequence; mode_aliases=aliases)
    pairs = Tuple{Float64, Float64}[]
    used = falses(length(theory))
    for obs in observed
        match_index = findfirst(eachindex(theory)) do idx
            !used[idx] && _acceptance_same_transition(obs, theory[idx], aliases)
        end
        match_index === nothing && continue
        used[match_index] = true
        push!(pairs, (obs.location, theory[match_index].location))
    end
    if length(pairs) < 2
        return nothing, "axis calibration requires at least two matching measured/model transition anchors; found $(length(pairs))."
    end
    x = [p[1] for p in pairs]
    y = [p[2] for p in pairs]
    xbar = mean(x)
    ybar = mean(y)
    denom = sum((xi - xbar)^2 for xi in x)
    if !(isfinite(denom) && denom > 0.0)
        return nothing, "axis calibration is underdetermined because matched measured transition anchors are not distinct."
    end
    scale = sum((xi - xbar) * (yi - ybar) for (xi, yi) in zip(x, y)) / denom
    offset = ybar - scale * xbar
    if !(isfinite(scale) && scale > 0.0 && isfinite(offset))
        return nothing, "axis calibration produced a non-finite or non-positive affine map."
    end
    return _acceptance_mode_config(base, scale, offset), ""
end

function _acceptance_certificate_context(certificate::RobustChaosCertificate,
                                         config::HardwareAcceptanceConfig)
    return (
        kind=:band,
        verdict=certificate.overall_verdict,
        intervals=[(certificate.param_min, certificate.param_max)],
        regions=RobustChaosRegion[],
        param_names=nothing,
        items=certificate.certificate_items,
    )
end

function _acceptance_certificate_context(certificate::RobustChaosRegionResult,
                                         config::HardwareAcceptanceConfig)
    accepted = RobustChaosRegion[
        region for region in certificate.regions
        if region.verdict in config.accepted_certificate_verdicts
    ]
    verdict = if !isempty(accepted)
        any(region -> region.verdict == :fragile, accepted) ? :fragile :
        any(region -> region.verdict == :inconclusive, accepted) ? :inconclusive :
        :certified
    elseif any(region -> region.verdict == :fragile, certificate.regions)
        :fragile
    else
        :inconclusive
    end
    return (
        kind=:region_result,
        verdict=verdict,
        intervals=Tuple{Float64, Float64}[],
        regions=accepted,
        param_names=certificate.param_names,
        items=certificate.certificate_items,
    )
end

@inline _interval_shift(x::Float64, lo::Float64, hi::Float64) =
    x < lo ? lo - x : x > hi ? x - hi : 0.0

function _acceptance_region_shift(a::Float64, b::Float64, region::RobustChaosRegion)
    da = _interval_shift(a, region.a_min, region.a_max)
    db = _interval_shift(b, region.b_min, region.b_max)
    return hypot(da, db)
end

function _acceptance_point(cross_section::OperatingMapCrossSection,
                           param_names::Tuple{Symbol, Symbol},
                           value::Float64)
    varying = cross_section.sequence.parameter_name
    fixed = cross_section.fixed_parameter_name
    fixed_value = cross_section.selected_fixed_value
    if varying == param_names[1] && fixed == param_names[2]
        return (a=value, b=fixed_value, axis=:a, param_names=param_names)
    elseif varying == param_names[2] && fixed == param_names[1]
        return (a=fixed_value, b=value, axis=:b, param_names=param_names)
    end
    return nothing
end

function _acceptance_claim_shift(ctx, alignment::ModeSequenceAlignment, value::Float64)
    if ctx.kind == :band
        isempty(ctx.intervals) && return (inside=false, shift=nothing, point=nothing)
        shift = minimum(_interval_shift(value, lo, hi) for (lo, hi) in ctx.intervals)
        return (inside=shift == 0.0, shift=shift, point=nothing)
    end
    point = _acceptance_point(alignment.predicted, ctx.param_names, value)
    point === nothing && return (inside=false, shift=nothing, point=nothing)
    isempty(ctx.regions) && return (inside=false, shift=nothing, point=point)
    shift = minimum(_acceptance_region_shift(point.a, point.b, region) for region in ctx.regions)
    return (inside=shift == 0.0, shift=shift, point=point)
end

function _acceptance_nearest_mode_shift(alignment::ModeSequenceAlignment, index::Int)
    status = alignment.observation_statuses[index]
    status == :matched && return 0.0
    observed = alignment.experimental.modes[index]
    observed_key = _aliased_mode_key(observed, alignment.config.mode_aliases)
    values = alignment.predicted.sequence.parameter_values
    modes = alignment.predicted.sequence.modes
    candidates = Float64[
        values[k] for k in eachindex(values)
        if _aliased_mode_key(modes[k], alignment.config.mode_aliases) == observed_key
    ]
    isempty(candidates) && return nothing
    x = alignment.aligned_parameter_values[index]
    return minimum(abs(candidate - x) for candidate in candidates)
end

function _acceptance_margin(boundary::Union{Nothing, RegimeBoundaryResult},
                            point,
                            axis::Union{Nothing, Symbol})
    boundary === nothing && return (margin=nothing, axis=axis, status=:margin_unavailable)
    point === nothing && return (margin=nothing, axis=axis, status=:margin_unavailable)
    mapped = _acceptance_point_names(boundary.param_names, point)
    mapped === nothing && return (margin=nothing, axis=axis, status=:margin_unavailable)
    a, b, margin_axis = mapped
    if a < boundary.a_grid[1] || a > boundary.a_grid[end] ||
       b < boundary.b_grid[1] || b > boundary.b_grid[end]
        return (margin=nothing, axis=margin_axis, status=:outside_margin_domain)
    end
    i = _nearest_grid_index(boundary.a_grid, a)
    j = _nearest_grid_index(boundary.b_grid, b)
    boundary.valid[i, j] || return (margin=nothing, axis=margin_axis, status=:unresolved_margin)
    axis_margin = margin_axis == :a ? boundary.distance_a[i, j] : boundary.distance_b[i, j]
    margin = isfinite(axis_margin) ? axis_margin : boundary.distance[i, j]
    return isfinite(margin) ?
        (margin=margin, axis=margin_axis, status=:available) :
        (margin=nothing, axis=margin_axis, status=:unresolved_margin)
end

function _acceptance_point_names(param_names::Tuple{Symbol, Symbol}, point)
    point === nothing && return nothing
    point_names = getproperty(point, :param_names)
    point_names == param_names && return (point.a, point.b, point.axis)
    if point_names[1] == param_names[2] && point_names[2] == param_names[1]
        return (point.b, point.a, point.axis == :a ? :b : :a)
    end
    return nothing
end

function _acceptance_tolerance_probability(tolerance::Union{Nothing, ToleranceMapResult}, point)
    tolerance === nothing && return nothing
    point === nothing && return nothing
    mapped = _acceptance_point_names(tolerance.param_names, point)
    mapped === nothing && return nothing
    a, b, _ = mapped
    if a < tolerance.a_grid[1] || a > tolerance.a_grid[end] ||
       b < tolerance.b_grid[1] || b > tolerance.b_grid[end]
        return nothing
    end
    i = _nearest_grid_index(tolerance.a_grid, a)
    j = _nearest_grid_index(tolerance.b_grid, b)
    return tolerance.nominal_probability[i, j]
end

function _acceptance_margin_status(required_shift, margin_info, slack::Float64)
    required_shift === nothing && return :no_compatible_mode
    required_shift <= slack && return :inside_margin
    margin_info.status == :available || return margin_info.status
    return required_shift <= margin_info.margin + slack ? :inside_margin : :outside_margin
end

function _acceptance_mismatch_message(observation_status::Symbol, claim_inside::Bool,
                                      margin_status::Symbol)
    parts = String[]
    observation_status == :matched || push!(parts, "mode observation $(observation_status)")
    claim_inside || push!(parts, "outside certified claim")
    push!(parts, "margin status $(margin_status)")
    return join(parts, "; ")
end

function _acceptance_mismatches(alignment::ModeSequenceAlignment,
                                ctx,
                                boundary::Union{Nothing, RegimeBoundaryResult},
                                tolerance::Union{Nothing, ToleranceMapResult},
                                config::HardwareAcceptanceConfig)
    mismatches = HardwareAcceptanceMismatch[]
    for idx in eachindex(alignment.observation_statuses)
        value = alignment.aligned_parameter_values[idx]
        claim = _acceptance_claim_shift(ctx, alignment, value)
        observation_status = alignment.observation_statuses[idx]
        observation_status == :matched && claim.inside && continue
        mode_shift = _acceptance_nearest_mode_shift(alignment, idx)
        required_shift = if observation_status != :matched && mode_shift === nothing
            nothing
        elseif mode_shift === nothing
            claim.shift
        elseif claim.shift === nothing
            mode_shift
        else
            max(mode_shift, claim.shift)
        end
        point = claim.point
        if point === nothing && boundary !== nothing
            point = _acceptance_point(alignment.predicted, boundary.param_names, value)
        end
        margin_info = _acceptance_margin(boundary, point, point === nothing ? nothing : point.axis)
        tolerance_probability = _acceptance_tolerance_probability(tolerance, point)
        margin_status = _acceptance_margin_status(required_shift, margin_info, config.margin_slack)
        push!(mismatches, HardwareAcceptanceMismatch(
            idx,
            alignment.experimental.parameter_values[idx],
            value,
            alignment.experimental.modes[idx],
            alignment.predicted_modes[idx],
            observation_status,
            required_shift,
            margin_info.margin,
            margin_info.axis,
            tolerance_probability,
            margin_status,
            _acceptance_mismatch_message(observation_status, claim.inside, margin_status),
        ))
    end
    return mismatches
end

function _acceptance_aggregate_items(alignment::Union{Nothing, ModeSequenceAlignment},
                                     ctx,
                                     mismatches::Vector{HardwareAcceptanceMismatch},
                                     config::HardwareAcceptanceConfig,
                                     verdict::Symbol)
    items = Dict{String, Any}[
        Dict(
            "layer" => "certificate",
            "kind" => String(ctx.kind),
            "verdict" => String(ctx.verdict),
            "acceptedVerdicts" => String.(config.accepted_certificate_verdicts),
        ),
        Dict(
            "layer" => "mismatches",
            "count" => length(mismatches),
            "outsideMarginCount" => count(m -> m.margin_status == :outside_margin, mismatches),
            "insideMarginCount" => count(m -> m.margin_status == :inside_margin, mismatches),
            "unresolvedMarginCount" => count(m -> m.margin_status in (:margin_unavailable, :unresolved_margin), mismatches),
        ),
        Dict("layer" => "overall", "verdict" => String(verdict)),
    ]
    if alignment !== nothing
        pushfirst!(items, Dict(
            "layer" => "mode_assimilation",
            "overallScore" => alignment.overall_score,
            "agreementScore" => alignment.agreement_score,
            "transitionF1" => alignment.transition_f1,
            "coverage" => alignment.coverage,
        ))
    end
    return items
end

function _acceptance_verdict(alignment::ModeSequenceAlignment,
                             ctx,
                             mismatches::Vector{HardwareAcceptanceMismatch},
                             config::HardwareAcceptanceConfig)
    certificate_ok = ctx.verdict in config.accepted_certificate_verdicts
    ctx.verdict == :fragile && return :rejected
    certificate_ok || return :inconclusive
    alignment.overall_score + 8eps(Float64) < config.min_overall_score && return :rejected
    if config.min_transition_f1 !== nothing
        (alignment.transition_f1 === nothing ||
         alignment.transition_f1 + 8eps(Float64) < config.min_transition_f1) && return :rejected
    end
    any(m -> m.margin_status in (:outside_margin, :no_compatible_mode, :outside_margin_domain), mismatches) &&
        return :rejected
    any(m -> m.margin_status in (:margin_unavailable, :unresolved_margin), mismatches) &&
        return :inconclusive
    return :accepted
end

function _acceptance_refused(certificate::_HARDWARE_CERTIFICATE,
                             config::HardwareAcceptanceConfig,
                             reason::AbstractString)
    ctx = _acceptance_certificate_context(certificate, config)
    items = Dict{String, Any}[
        Dict("layer" => "axis_calibration", "verdict" => "refused", "reason" => String(reason)),
        Dict("layer" => "overall", "verdict" => "refused"),
    ]
    return HardwareAcceptanceResult(
        nothing,
        ctx.kind,
        ctx.verdict,
        :refused,
        HardwareAcceptanceMismatch[],
        copy(config.accepted_certificate_verdicts),
        nothing,
        items,
        now(),
    )
end

"""
    hardware_acceptance_test(alignment, certificate; boundary=nothing,
                             tolerance=nothing, config=HardwareAcceptanceConfig())

Compose a mode-sequence alignment with robust-chaos certificate evidence and
optional tolerance/margin fields. Mismatched observations are named and scored
against the local margin at their operating point.
"""
function hardware_acceptance_test(alignment::ModeSequenceAlignment,
                                  certificate::_HARDWARE_CERTIFICATE;
                                  boundary::Union{Nothing, RegimeBoundaryResult}=nothing,
                                  tolerance::Union{Nothing, ToleranceMapResult}=nothing,
                                  config::HardwareAcceptanceConfig=HardwareAcceptanceConfig())
    ctx = _acceptance_certificate_context(certificate, config)
    mismatches = _acceptance_mismatches(alignment, ctx, boundary, tolerance, config)
    verdict = _acceptance_verdict(alignment, ctx, mismatches, config)
    return HardwareAcceptanceResult(
        alignment,
        ctx.kind,
        ctx.verdict,
        verdict,
        mismatches,
        copy(config.accepted_certificate_verdicts),
        alignment.overall_score,
        _acceptance_aggregate_items(alignment, ctx, mismatches, config, verdict),
        now(),
    )
end

"""
    hardware_acceptance_test(experimental, predicted, certificate; ...)

Calibrate (or refuse to calibrate) a measured mode sequence, align it to a
computed operating-map route, then run `hardware_acceptance_test` on the
resulting alignment.
"""
function hardware_acceptance_test(experimental::ModeSequence,
                                  predicted::OperatingMapCrossSection,
                                  certificate::_HARDWARE_CERTIFICATE;
                                  boundary::Union{Nothing, RegimeBoundaryResult}=nothing,
                                  tolerance::Union{Nothing, ToleranceMapResult}=nothing,
                                  config::HardwareAcceptanceConfig=HardwareAcceptanceConfig(),
                                  assimilation_config::Union{Nothing, ModeAssimilationConfig}=nothing)
    base = assimilation_config === nothing ? ModeAssimilationConfig() : assimilation_config
    mode_config = if config.axis_calibration == :provided
        if assimilation_config === nothing
            return _acceptance_refused(certificate, config,
                "axis_calibration=:provided requires an explicit ModeAssimilationConfig.")
        end
        base
    elseif config.axis_calibration == :identity
        _acceptance_mode_config(base, 1.0, 0.0)
    else
        calibrated, reason = _acceptance_transition_affine_config(experimental, predicted, base)
        calibrated === nothing && return _acceptance_refused(certificate, config, reason)
        calibrated
    end
    alignment = assimilate_mode_sequence(experimental, predicted; config=mode_config)
    return hardware_acceptance_test(
        alignment,
        certificate;
        boundary=boundary,
        tolerance=tolerance,
        config=config,
    )
end

function hardware_acceptance_summary(result::HardwareAcceptanceResult)
    return (
        verdict=result.verdict,
        certificate_kind=result.certificate_kind,
        certificate_verdict=result.certificate_verdict,
        mismatch_count=length(result.mismatches),
        outside_margin_count=count(m -> m.margin_status == :outside_margin, result.mismatches),
        inside_margin_count=count(m -> m.margin_status == :inside_margin, result.mismatches),
        unresolved_margin_count=count(m -> m.margin_status in (:margin_unavailable, :unresolved_margin), result.mismatches),
        score=result.score,
    )
end
