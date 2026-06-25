"""
Codimension-2 bifurcation-curve assembly from repeated 1D continuation slices.
"""

_normalize_codim2_kind(kind::Symbol) = kind === :hopf ? :ns : kind

function _codim2_base_params(sys::DiscreteMap, config::Codim2Config, params::Vector{Float64})
    if !isempty(params)
        return copy(params)
    elseif !isempty(config.fixed_params)
        return copy(config.fixed_params)
    end
    required = max(length(sys.param_names),
                   config.continuation.param_index,
                   config.second_param_index,
                   isempty(config.continuation.linked_param_indices) ? 1 : maximum(config.continuation.linked_param_indices),
                   isempty(config.second_linked_param_indices) ? 1 : maximum(config.second_linked_param_indices))
    return zeros(Float64, required)
end

function _codim2_base_params(sys::ContinuousODE, config::Codim2Config, params::Vector{Float64})
    if !isempty(params)
        return _resolve_continuous_params(sys, params)
    elseif !isempty(config.fixed_params)
        return copy(config.fixed_params)
    end
    return _resolve_continuous_params(sys, Float64[])
end

function _codim2_tracking_tolerance(config::Codim2Config)
    if !isnothing(config.tracking_tolerance)
        return Float64(config.tracking_tolerance)
    end
    span = abs(config.continuation.p_max - config.continuation.p_min)
    return max(span * 0.1, 1e-8)
end

function _codim2_anchor_second(config::Codim2Config)
    if !isnothing(config.anchor_second)
        return Float64(config.anchor_second)
    end
    return (config.second_min + config.second_max) / 2
end

function _codim2_slice_seed_values(config::Codim2Config, second_values::AbstractVector{<:Real}, base_params::Vector{Float64})
    if isempty(config.primary_seed_values)
        return fill(base_params[config.continuation.param_index], length(second_values))
    end
    length(config.primary_seed_values) == length(second_values) || throw(ArgumentError(
        "Codim2Config.primary_seed_values length $(length(config.primary_seed_values)) must match the secondary sweep length $(length(second_values))."
    ))
    return Float64[Float64(value) for value in config.primary_seed_values]
end

function _codim2_slice_primary_bounds(config::Codim2Config, second_values::AbstractVector{<:Real})
    count = length(second_values)
    min_values = isempty(config.primary_min_values) ? fill(config.continuation.p_min, count) : Float64[Float64(value) for value in config.primary_min_values]
    max_values = isempty(config.primary_max_values) ? fill(config.continuation.p_max, count) : Float64[Float64(value) for value in config.primary_max_values]
    length(min_values) == count || throw(ArgumentError(
        "Codim2Config.primary_min_values length $(length(min_values)) must match the secondary sweep length $(count)."
    ))
    length(max_values) == count || throw(ArgumentError(
        "Codim2Config.primary_max_values length $(length(max_values)) must match the secondary sweep length $(count)."
    ))
    all(min_values .<= max_values) || throw(ArgumentError(
        "Codim2Config.primary_min_values and primary_max_values must satisfy min <= max on every slice."
    ))
    return min_values, max_values
end

function _codim2_special_point_candidates(branch::BranchResult,
                                          kind::Symbol,
                                          p_min::Float64,
                                          p_max::Float64,
                                          endpoint_margin::Float64)
    candidates = Float64[]
    for point in branch.branch.specialpoint
        hasproperty(point, :type) || continue
        _normalize_codim2_kind(Symbol(getproperty(point, :type))) == kind || continue
        param = Float64(getproperty(point, :param))
        (param > p_min + endpoint_margin && param < p_max - endpoint_margin) || continue
        push!(candidates, param)
    end
    sort!(candidates)
    return unique!(candidates)
end

function _codim2_stability_flip_candidates(diag::AbstractDict,
                                           p_min::Float64,
                                           p_max::Float64,
                                           endpoint_margin::Float64)
    params = get(diag, "paramValues", Float64[])
    flags = get(diag, "stabilityFlags", Bool[])
    length(params) == length(flags) || return Float64[]
    length(params) >= 2 || return Float64[]

    flips = Float64[]
    for i in 2:length(flags)
        flags[i - 1] == flags[i] && continue
        candidate = (Float64(params[i - 1]) + Float64(params[i])) / 2
        (candidate > p_min + endpoint_margin && candidate < p_max - endpoint_margin) || continue
        push!(flips, candidate)
    end
    return flips
end

function _codim2_slice_candidates(sys::DynamicalSystem,
                                  branch::BranchResult,
                                  base_params::Vector{Float64},
                                  config::Codim2Config,
                                  continuation::ContinuationConfig=config.continuation;
                                  kwargs...)
    kind = _normalize_codim2_kind(config.bifurcation_kind)
    candidates = _codim2_special_point_candidates(
        branch,
        kind,
        continuation.p_min,
        continuation.p_max,
        config.endpoint_margin
    )
    !isempty(candidates) && return candidates, :special_point, ""

    if kind == :pd && config.fallback_to_stability_flips
        diag = continuation_branch_diagnostics(
            sys,
            branch,
            base_params;
            linked_param_indices=continuation.linked_param_indices,
            max_points=config.diagnostics_max_points,
            include_residuals=false,
            include_switching_events=false,
            # Stability-flip fallback needs multiplier-based stability flags, so do not
            # disable spectra/multiplier work here.
            include_spectra=true,
        )
        flips = _codim2_stability_flip_candidates(
            diag,
            continuation.p_min,
            continuation.p_max,
            config.endpoint_margin
        )
        !isempty(flips) && return flips, :stability_flip, ""
        return Float64[], :none, "no matching period-doubling candidates were found on this slice"
    end

    return Float64[], :none, "no matching $(kind) candidates were found on this slice"
end

function _track_codim2_curve(raw::Vector{Vector{Float64}},
                             secondary_values::Vector{Float64},
                             anchor_second::Float64,
                             tolerance::Float64,
                             anchor_candidate_index::Int)
    n = length(raw)
    out = fill(NaN, n)
    n == 0 && return out

    order = sortperm(abs.(secondary_values .- anchor_second))
    anchor_idx = nothing
    for idx in order
        length(raw[idx]) >= anchor_candidate_index || continue
        anchor_idx = idx
        break
    end
    isnothing(anchor_idx) && return out

    out[anchor_idx] = sort(raw[anchor_idx])[anchor_candidate_index]
    prev = out[anchor_idx]
    for idx in (anchor_idx - 1):-1:1
        isempty(raw[idx]) && break
        candidates = sort(raw[idx])
        candidate = candidates[argmin(abs.(candidates .- prev))]
        abs(candidate - prev) > tolerance && break
        out[idx] = candidate
        prev = candidate
    end
    prev = out[anchor_idx]
    for idx in (anchor_idx + 1):n
        isempty(raw[idx]) && break
        candidates = sort(raw[idx])
        candidate = candidates[argmin(abs.(candidates .- prev))]
        abs(candidate - prev) > tolerance && break
        out[idx] = candidate
        prev = candidate
    end
    return out
end

function _extremal_codim2_curve(raw::Vector{Vector{Float64}}, mode::Symbol)
    selector = mode == :maximum ? maximum : minimum
    tracked = fill(NaN, length(raw))
    for idx in eachindex(raw)
        isempty(raw[idx]) && continue
        tracked[idx] = selector(raw[idx])
    end
    return tracked
end

"""
    codim2_curve(sys, config::Codim2Config; kwargs...) -> Codim2CurveResult
    codim2_curve(sys, config::Codim2Config, period::Int; kwargs...) -> Codim2CurveResult

Assemble a codimension-2 bifurcation curve by sweeping a secondary parameter and
running a 1D continuation slice along the primary parameter at each secondary
value. The tracked curve is stitched from per-slice candidate bifurcation
locations, while `raw_candidates` preserves every detected candidate on each
slice.
"""
function codim2_curve(sys::DynamicalSystem,
                      config::Codim2Config;
                      initial_point::Union{Nothing, AbstractVector}=nothing,
                      params::Vector{Float64}=Float64[],
                      kwargs...)
    return codim2_curve(sys, config, 1; initial_point=initial_point, params=params, kwargs...)
end

function codim2_curve(sys::DynamicalSystem,
                      config::Codim2Config,
                      period::Int;
                      initial_point::Union{Nothing, AbstractVector}=nothing,
                      params::Vector{Float64}=Float64[],
                      kwargs...)
    period >= 1 || throw(ArgumentError("codim2_curve period must be >= 1, got $(period)."))

    continuation = config.continuation
    second_values = collect(range(config.second_min, config.second_max, length=config.second_steps + 1))
    base_params = _codim2_base_params(sys, config, params)
    slice_seed_values = _codim2_slice_seed_values(config, second_values, base_params)
    slice_min_values, slice_max_values = _codim2_slice_primary_bounds(config, second_values)
    raw_candidates = [Float64[] for _ in eachindex(second_values)]
    candidate_sources = fill(:none, length(second_values))
    slice_statuses = fill(:not_run, length(second_values))
    slice_messages = fill("", length(second_values))
    slice_point_counts = zeros(Int, length(second_values))
    slice_special_point_counts = zeros(Int, length(second_values))

    max_required_index = maximum(vcat(
        [continuation.param_index, config.second_param_index],
        continuation.linked_param_indices,
        config.second_linked_param_indices
    ))
    length(base_params) >= max_required_index || throw(ArgumentError(
        "Codim2Config/base params require at least $max_required_index parameters, got $(length(base_params))."
    ))

    run_slice! = function(idx::Int)
        secondary_value = second_values[idx]
        slice_params = _inject_param(base_params, config.second_param_index, secondary_value, config.second_linked_param_indices)
        slice_params = _inject_param(slice_params, continuation.param_index, slice_seed_values[idx], continuation.linked_param_indices)
        local_continuation = continuation
        local_continuation = Setfield.@set local_continuation.p_min = slice_min_values[idx]
        local_continuation = Setfield.@set local_continuation.p_max = slice_max_values[idx]
        branch = try
            continuation_branch(
                sys,
                local_continuation,
                period;
                initial_point=initial_point,
                params=slice_params,
                kwargs...
            )
        catch err
            slice_statuses[idx] = :continuation_failed
            slice_messages[idx] = sprint(showerror, err)
            return nothing
        end

        slice_point_counts[idx] = length(_branch_points(branch))
        slice_special_point_counts[idx] = length(branch.branch.specialpoint)
        candidates, source, message = _codim2_slice_candidates(sys, branch, slice_params, config, local_continuation; kwargs...)
        raw_candidates[idx] = candidates
        candidate_sources[idx] = source
        slice_messages[idx] = message
        slice_statuses[idx] = isempty(candidates) ? :no_candidates : :ok
        return nothing
    end

    threaded = config.threaded && Threads.nthreads() > 1 && length(second_values) > 1
    if threaded
        Threads.@threads for idx in eachindex(second_values)
            run_slice!(idx)
        end
    else
        for idx in eachindex(second_values)
            run_slice!(idx)
        end
    end

    tolerance = _codim2_tracking_tolerance(config)
    anchor_second = _codim2_anchor_second(config)
    tracked = if config.tracking_mode == :nearest
        _track_codim2_curve(raw_candidates, second_values, anchor_second, tolerance, config.anchor_candidate_index)
    else
        _extremal_codim2_curve(raw_candidates, config.tracking_mode)
    end
    valid_mask = .!isnan.(tracked)
    any(valid_mask) || throw(ArgumentError(
        "No codim-2 $(config.bifurcation_kind) candidates were recovered for $(sys.name) period $(period) across the requested secondary sweep."
    ))

    return Codim2CurveResult(
        tracked,
        second_values,
        valid_mask,
        raw_candidates,
        candidate_sources,
        slice_statuses,
        slice_messages,
        slice_point_counts,
        slice_special_point_counts,
        _normalize_codim2_kind(config.bifurcation_kind),
        period,
        sys.name,
        (sys.param_names[continuation.param_index], sys.param_names[config.second_param_index]),
        :slice_tracking,
        anchor_second,
        tolerance,
        now()
    )
end
