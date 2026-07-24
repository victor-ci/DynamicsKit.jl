"""
    continuation_branch_diagnostics(sys, branch, base_params; kwargs...) -> Dict

Compute scientific diagnostics for a continuation branch: residual norms of the
period map and multiplier/Floquet spectra at recorded branch points. `max_points`
limits expensive evaluations by uniformly sampling the branch; `max_points <= 0`
evaluates every point.
"""
function continuation_branch_diagnostics(sys::DynamicalSystem,
                                         branch::BranchResult,
                                         base_params::AbstractVector;
                                         linked_param_indices::AbstractVector{<:Integer}=Int[],
                                          max_points::Int=0,
                                          include_residuals::Bool=true,
                                          include_multipliers::Bool=true,
                                          include_spectra::Bool=true,
                                          include_switching_events::Bool=true,
                                          residual_warning_threshold::Float64=1e-6,
                                          stability_tol::Float64=_MAP_STABILITY_UNIT_CIRCLE_TOL,
                                          kwargs...)
    points = _branch_points(branch)
    point_count = length(points)
    param_index = findfirst(==(branch.param_name), sys.param_names)
    warnings = String[]
    ode_method = sys isa ContinuousODE ? get(kwargs, :ode_jacobian_method, :finite_difference) : :not_applicable
    if isnothing(param_index)
        push!(warnings, "branch parameter $(branch.param_name) is not defined by system $(sys.name)")
        return Dict{String, Any}(
            "version" => 1,
            "status" => "unavailable",
            "warnings" => warnings,
            "pointCount" => point_count,
            "evaluatedPointCount" => 0,
            "sampled" => false,
            "evaluatedIndices" => Int[],
            "paramValues" => Float64[],
            "residualNorms" => Float64[],
            "maxResidualNorm" => NaN,
            "medianResidualNorm" => NaN,
            "maxMultiplierModuli" => Float64[],
            "maxMultiplierModulus" => NaN,
            "stabilityFlags" => Bool[],
            "stableCount" => 0,
            "unstableCount" => 0,
            "multiplierSpectra" => Any[],
            "residualFailureCount" => 0,
            "multiplierFailureCount" => 0,
            "odeJacobianMethod" => string(ode_method),
            "switchingEvents" => switching_event_diagnostics(sys, Vector{Vector{Float64}}(), base_params)
        )
    end

    indices = _diagnostic_sample_indices(point_count, max_points)
    residual_norms = Float64[]
    max_multiplier_moduli = Float64[]
    stability_flags = Bool[]
    spectra = Any[]
    param_values = Float64[]
    residual_failures = 0
    multiplier_failures = 0
    period = max(branch.period, 1)
    base = collect(Float64, base_params)
    linked = collect(Int, linked_param_indices)
    switching_states = Vector{Vector{Float64}}()
    switching_params = Vector{Vector{Float64}}()

    for idx in indices
        pt = points[idx]
        p_value = Float64(pt.param)
        push!(param_values, p_value)
        local_params = inject_param(base, param_index, p_value, linked)
        state = _branch_point_state(pt)
        if include_switching_events
            push!(switching_states, collect(Float64, state))
            push!(switching_params, collect(Float64, local_params))
        end

        if include_residuals
            residual_norm = try
                residual = _map_residual(sys, state, local_params, period; kwargs...)
                norm(residual)
            catch err
                err isa InterruptException && rethrow()
                residual_failures += 1
                push!(warnings, "residual evaluation failed at point $idx: $(_continuation_error_message(err))")
                NaN
            end
            push!(residual_norms, residual_norm)
        end

        if include_multipliers
            point_multipliers = try
                _map_multipliers(sys, state, local_params, period; kwargs...)
            catch err
                err isa InterruptException && rethrow()
                multiplier_failures += 1
                push!(warnings, "multiplier evaluation failed at point $idx: $(_continuation_error_message(err))")
                nothing
            end

            if isnothing(point_multipliers)
                push!(max_multiplier_moduli, NaN)
                push!(stability_flags, false)
                include_spectra && push!(spectra, Any[])
            else
                max_modulus = maximum(abs.(point_multipliers))
                push!(max_multiplier_moduli, Float64(max_modulus))
                push!(stability_flags, _multipliers_are_stable(point_multipliers; tol=stability_tol))
                include_spectra && push!(spectra, _multiplier_spectrum_payload(point_multipliers))
            end
        end
    end

    max_residual = include_residuals ? _diagnostic_max(residual_norms) : NaN
    if include_residuals && isfinite(max_residual) && max_residual > residual_warning_threshold
        push!(warnings, "large residual norm")
    end
    unique!(warnings)

    stable_count = count(identity, stability_flags)
    switching_diag = include_switching_events ?
        switching_event_diagnostics(sys, switching_states, switching_params) :
        Dict{String, Any}("status" => "disabled", "eventCount" => length(switching_events(sys)))
    return Dict{String, Any}(
        "version" => 1,
        "status" => isempty(warnings) ? "ok" : "warning",
        "warnings" => warnings,
        "pointCount" => point_count,
        "evaluatedPointCount" => length(indices),
        "sampled" => length(indices) < point_count,
        "evaluatedIndices" => indices,
        "paramValues" => param_values,
        "residualNorms" => residual_norms,
        "maxResidualNorm" => max_residual,
        "medianResidualNorm" => include_residuals ? _diagnostic_median(residual_norms) : NaN,
        "residualWarningThreshold" => residual_warning_threshold,
        "maxMultiplierModuli" => max_multiplier_moduli,
        "maxMultiplierModulus" => include_multipliers ? _diagnostic_max(max_multiplier_moduli) : NaN,
        "stabilityFlags" => stability_flags,
        "stableCount" => stable_count,
        "unstableCount" => length(stability_flags) - stable_count,
        "multiplierSpectra" => spectra,
        "residualFailureCount" => residual_failures,
        "multiplierFailureCount" => multiplier_failures,
        "odeJacobianMethod" => string(ode_method),
        "switchingEvents" => switching_diag
    )
end

"""
Compact signature used to compare candidate branches without repeatedly rescanning
branch points. Samples the first state coordinate at quartile positions (25 %, 50 %,
75 %) along the branch rather than the midpoint alone, so two morphologically
distinct branches that happen to cross at the midpoint are not conflated.

`sample_states` is empty for an empty branch.
"""
function _branch_signature(branch::BranchResult)
    points = _branch_points(branch)
    isempty(points) && return (period=branch.period, param_min=NaN, param_max=NaN, sample_states=Float64[])
    pars = Float64[pt.param for pt in points]
    n = length(points)
    sample_indices = if n >= 3
        [clamp(round(Int, q * n + 0.5), 1, n) for q in (0.25, 0.5, 0.75)]
    else
        collect(1:n)
    end
    sample_states = [Float64(getproperty(points[i], :x1)) for i in sample_indices]
    return (
        period=branch.period,
        param_min=minimum(pars),
        param_max=maximum(pars),
        sample_states=sample_states
    )
end

"""Check a candidate branch signature against previously accepted signatures."""
function _is_duplicate_signature(candidate_signature,
                                 existing_signatures,
                                 param_tol::Float64,
                                 state_tol::Float64)
    for signature in existing_signatures
        signature.period == candidate_signature.period || continue
        abs(signature.param_min - candidate_signature.param_min) <= param_tol || continue
        abs(signature.param_max - candidate_signature.param_max) <= param_tol || continue
        length(signature.sample_states) == length(candidate_signature.sample_states) || continue
        if all(abs(a - b) <= state_tol for (a, b) in zip(signature.sample_states, candidate_signature.sample_states))
            return true
        end
    end

    return false
end

"""Detect whether a newly generated branch duplicates one that was already found."""
function _is_duplicate_branch(candidate::BranchResult, existing::Vector{BranchResult},
                              param_tol::Float64, state_tol::Float64)
    candidate_signature = _branch_signature(candidate)
    existing_signatures = _branch_signature.(existing)
    return _is_duplicate_signature(candidate_signature, existing_signatures, param_tol, state_tol)
end

"""Run continuation in a single direction for a continuous-time Poincaré-map problem."""
function _run_continuation_direction(prob, config::ContinuationConfig;
                                     p_min::Float64,
                                     p_max::Float64,
                                     ds::Float64)
    newton_opts = NewtonPar(
        tol = config.newton_tol,
        max_iterations = config.newton_max_iter
    )

    cont_opts = ContinuationPar(
        p_min = p_min,
        p_max = p_max,
        ds = ds,
        dsmax = config.dsmax,
        dsmin = config.dsmin,
        max_steps = config.max_steps,
        newton_options = newton_opts,
        detect_bifurcation = config.detect_bifurcation,
        n_inversion = 6,
        a = config.a,
        detect_fold = config.detect_fold,
        save_sol_every_step = config.save_sol_every_step
    )

    continuation(prob, PALC(), cont_opts; normC=norminf, verbosity=0)
end

_continuation_error_message(err) = err isa AbstractString ? String(err) : sprint(showerror, err)

function _report_continuation_error(on_error::Union{Nothing, Function}, context::AbstractString, err)
    isnothing(on_error) && return nothing
    on_error("$context: $(_continuation_error_message(err))")
    return nothing
end

function _run_continuation_direction_safe(prob_builder::Function, config::ContinuationConfig;
                                          p_min::Float64,
                                          p_max::Float64,
                                          ds::Float64,
                                          on_error::Union{Nothing, Function}=nothing,
                                          context::AbstractString="Continuation direction failed")
    try
        return _run_continuation_direction(prob_builder(), config; p_min=p_min, p_max=p_max, ds=ds)
    catch err
        err isa InterruptException && rethrow()
        _report_continuation_error(on_error, context, err)
        return nothing
    end
end

"""Merge backward and forward continuation runs into a complete branch ordered by parameter."""
function _merge_continuation_branches(backward, forward)
    points = Any[]
    specials = Any[]

    if !isnothing(backward)
        append!(points, reverse(backward.branch))
        append!(specials, backward.specialpoint)
    end

    if !isnothing(forward)
        forward_points = collect(forward.branch)
        if !isempty(points) && !isempty(forward_points)
            same_start = abs(points[end].param - forward_points[1].param) <= 1e-10 &&
                         abs(getproperty(points[end], :x1) - getproperty(forward_points[1], :x1)) <= 1e-8
            forward_points = same_start ? forward_points[2:end] : forward_points
        end
        append!(points, forward_points)
        append!(specials, forward.specialpoint)
    end

    CombinedBranchResult(points, specials)
end

"""Summary of what a re-seeded continuation direction did (surfaced via the `on_reseed` callback)."""
struct ReseedDiagnostics
    attempt_count::Int
    reseed_params::Vector{Float64}
    termination_reason::Symbol
end

"""Recorded points of a continuation segment (raw ContResult or CombinedBranchResult)."""
_segment_points(result) = collect(result.branch)

function _segment_specials(result)
    try
        return collect(result.specialpoint)
    catch
        return Any[]
    end
end

"""Append a continuation segment, dropping a duplicate seam point shared with the accumulator."""
function _append_segment!(points::Vector{Any}, specials::Vector{Any}, result;
                          param_tol::Float64=1e-9, state_tol::Float64=1e-8)
    seg_points = _segment_points(result)
    if !isempty(points) && !isempty(seg_points)
        last_pt = points[end]
        first_pt = seg_points[1]
        if abs(last_pt.param - first_pt.param) <= param_tol &&
           abs(getproperty(last_pt, :x1) - getproperty(first_pt, :x1)) <= state_tol
            seg_points = seg_points[2:end]
        end
    end
    append!(points, seg_points)
    append!(specials, _segment_specials(result))
    return points
end

"""Linear least-squares extrapolation of the trailing (param → coordinate) trend to `p_query`."""
function _extrapolate_state(tail_params::Vector{Float64}, tail_states::Vector{Vector{Float64}},
                            p_query::Float64)
    n = length(tail_params)
    n < 2 && return copy(tail_states[end])
    dim = length(tail_states[end])
    p̄ = sum(tail_params) / n
    denom = sum((p - p̄)^2 for p in tail_params)
    x_query = Vector{Float64}(undef, dim)
    for d in 1:dim
        x̄ = sum(s[d] for s in tail_states) / n
        slope = denom > 0 ?
            sum((tail_params[i] - p̄) * (tail_states[i][d] - x̄) for i in 1:n) / denom : 0.0
        x_query[d] = x̄ + slope * (p_query - p̄)
    end
    return x_query
end

"""Per-coordinate sample standard deviation of the trailing states (search-box width proxy)."""
function _coord_spread(tail_states::Vector{Vector{Float64}})
    n = length(tail_states)
    dim = length(tail_states[end])
    spread = zeros(dim)
    n < 2 && return spread
    for d in 1:dim
        m = sum(s[d] for s in tail_states) / n
        spread[d] = sqrt(sum((s[d] - m)^2 for s in tail_states) / (n - 1))
    end
    return spread
end

"""
Run continuation in one direction, automatically re-seeding when the branch dies in the
interior. `prob_from_seed(x_seed, p_seed)` builds a fresh BifurcationProblem; `reseed_skeleton(
p_query, x_extrap, lo, hi)` returns candidate states (already filtered to the correct period).
Returns `(CombinedBranchResult, ReseedDiagnostics)`.
"""
function _run_continuation_direction_with_reseed(prob_from_seed::Function,
                                                 reseed_skeleton::Function,
                                                 config::ContinuationConfig,
                                                 reseed_cfg::ReseedConfig,
                                                 x0::AbstractVector,
                                                 p0::Float64;
                                                 p_min::Float64,
                                                 p_max::Float64,
                                                 ds::Float64,
                                                 on_error::Union{Nothing, Function}=nothing,
                                                 context::AbstractString="Continuation direction failed")
    direction = ds >= 0 ? 1.0 : -1.0
    step = direction * abs(ds)
    prange = p_max - p_min

    result = _run_continuation_direction_safe(
        () -> prob_from_seed(collect(Float64, x0), p0), config;
        p_min=p_min, p_max=p_max, ds=ds, on_error=on_error, context=context)

    isnothing(result) && return nothing, ReseedDiagnostics(0, Float64[], :initial_failure)

    points = Any[]
    specials = Any[]
    _append_segment!(points, specials, result)

    reseed_params = Float64[]
    reason = :completed
    last_result = result

    while length(reseed_params) < reseed_cfg.max_attempts
        info = diagnose_continuation_termination(last_result, config)

        if info.reason == REACHED_BOUNDARY
            reason = :boundary; break
        elseif info.reason == UNKNOWN
            reason = :unknown; break
        elseif !isnothing(info.last_fold_param)
            reason = :fold; break
        end

        states, params = _termination_trajectory(last_result)
        K = min(reseed_cfg.trailing_k, length(states))
        tail_states = states[end - K + 1:end]
        tail_params = params[end - K + 1:end]
        p_dead = params[end]

        p_query = p_dead + step
        if p_query <= p_min || p_query >= p_max
            reason = :boundary; break
        end
        if !isempty(reseed_params) && abs(p_query - p0) <= reseed_cfg.circulus_vitiosus_frac * prange
            reason = :circulus_vitiosus; break
        end

        x_query = _extrapolate_state(tail_params, tail_states, p_query)
        spread = _coord_spread(tail_states)
        half = [max(reseed_cfg.box_half_width_min, reseed_cfg.box_half_width_scale * spread[d])
                for d in eachindex(x_query)]
        lo = x_query .- half
        hi = x_query .+ half

        candidates = reseed_skeleton(p_query, x_query, lo, hi)
        viable = [c for c in candidates if norm(c .- states[end]) > 1e-6]
        if isempty(viable)
            reason = :no_candidate; break
        end
        new_seed = viable[argmin([norm(c .- x_query) for c in viable])]

        seg = _run_continuation_direction_safe(
            () -> prob_from_seed(new_seed, p_query), config;
            p_min=p_min, p_max=p_max, ds=ds, on_error=on_error,
            context="Re-seeded continuation failed near parameter $(round(p_query, digits=6))")

        if isnothing(seg)
            reason = :resume_failed; break
        end

        _, seg_params = _termination_trajectory(seg)
        advanced = direction > 0 ? (maximum(seg_params) - p_dead) : (p_dead - minimum(seg_params))
        if length(seg_params) < reseed_cfg.min_progress_points || advanced < reseed_cfg.min_progress_dp
            reason = :no_progress; break
        end

        push!(reseed_params, p_query)
        _append_segment!(points, specials, seg)
        last_result = seg
    end

    if reason == :completed && length(reseed_params) >= reseed_cfg.max_attempts
        reason = :max_attempts
    end

    return CombinedBranchResult(points, specials), ReseedDiagnostics(length(reseed_params), reseed_params, reason)
end

"""Trace a complete continuous-time branch by continuing from the seed in both directions."""
function _complete_continuous_branch(prob_builder::Function, config::ContinuationConfig, p0::Float64;
                                     on_error::Union{Nothing, Function}=nothing,
                                     reseed_cfg::ReseedConfig=ReseedConfig(),
                                     prob_from_seed::Union{Nothing, Function}=nothing,
                                     reseed_skeleton::Union{Nothing, Function}=nothing,
                                     x0::Union{Nothing, AbstractVector}=nothing,
                                     on_reseed::Union{Nothing, Function}=nothing)
    use_reseed = reseed_cfg.enabled && !isnothing(prob_from_seed) &&
                 !isnothing(reseed_skeleton) && !isnothing(x0)

    run_backward = p0 > config.p_min + 10eps(Float64)
    run_forward = p0 < config.p_max - 10eps(Float64)

    run_dir = function (p_lo, p_hi, dsdir, ctx)
        if use_reseed
            return _run_continuation_direction_with_reseed(
                prob_from_seed, reseed_skeleton, config, reseed_cfg, x0, p0;
                p_min=p_lo, p_max=p_hi, ds=dsdir, on_error=on_error, context=ctx)
        else
            res = _run_continuation_direction_safe(
                prob_builder, config; p_min=p_lo, p_max=p_hi, ds=dsdir,
                on_error=on_error, context=ctx)
            return res, nothing
        end
    end

    backward = nothing
    forward = nothing
    backward_diag = nothing
    forward_diag = nothing
    back_ctx = "Backward continuation failed near parameter $(round(p0, digits=6))"
    fwd_ctx = "Forward continuation failed near parameter $(round(p0, digits=6))"

    if run_backward && run_forward && Threads.nthreads() > 1
        backward_task = Threads.@spawn run_dir(config.p_min, p0, -abs(config.ds), back_ctx)
        forward_task = Threads.@spawn run_dir(p0, config.p_max, abs(config.ds), fwd_ctx)
        backward, backward_diag = fetch(backward_task)
        forward, forward_diag = fetch(forward_task)
    else
        if run_backward
            backward, backward_diag = run_dir(config.p_min, p0, -abs(config.ds), back_ctx)
        end
        if run_forward
            forward, forward_diag = run_dir(p0, config.p_max, abs(config.ds), fwd_ctx)
        end
    end

    if use_reseed && !isnothing(on_reseed)
        on_reseed(backward_diag, forward_diag)
    end

    if isnothing(backward)
        return forward
    elseif isnothing(forward)
        return backward
    end

    return _merge_continuation_branches(backward, forward)
end

"""Extract the recorded branch points as a concrete vector."""
_branch_points(result::BranchResult) = collect(result.branch.branch)
