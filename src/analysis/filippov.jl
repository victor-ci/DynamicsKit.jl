"""
Flow-side Filippov diagnostics for `ContinuousODE` systems with declared
`SwitchingEvent` guards.

The core grazing conditions are the standard nonsmooth-flow test functions
`h(x,p) = 0` and `∇h(x,p)⋅f(x,p) = 0`; generic grazing additionally requires a
nonzero second normal derivative. Sliding is classified from the signs of the
two one-sided vector fields' normal components on the switching surface.
"""

function _filippov_select_event(sys::ContinuousODE, event_name::Union{Nothing, AbstractString})
    events = switching_events(sys)
    isempty(events) && throw(ArgumentError(
        "Continuous system $(sys.name) has no SwitchingEvent guards."))
    isnothing(event_name) && return first(events)
    for event in events
        event.name == String(event_name) && return event
    end
    throw(ArgumentError(
        "Continuous system $(sys.name) has no SwitchingEvent named $(repr(event_name))."))
end

function _filippov_resolve_params(sys::ContinuousODE, params::Vector{Float64})
    if isempty(params) && isempty(sys.default_params) && isempty(sys.param_names)
        return Float64[]
    end
    return _resolve_continuous_params(sys, params)
end

function _filippov_guard_scalar(event::SwitchingEvent, state::AbstractVector, params::AbstractVector)
    value = event.guard(state, params)
    value isa Number && return value
    throw(ArgumentError(
        "Filippov guard diagnostics require a scalar guard; event $(event.name) returned $(typeof(value))."))
end

function _filippov_rhs(sys::ContinuousODE, state::AbstractVector, params::AbstractVector, t::Real=0.0)
    T = promote_type(eltype(state), eltype(params), typeof(t))
    u = collect(T, state)
    p = collect(T, params)
    du = zeros(T, sys.dim)
    sys.f(du, u, p, t)
    return du
end

function _filippov_normal(event::SwitchingEvent, state::AbstractVector, params::AbstractVector)
    x = collect(Float64, state)
    p = collect(Float64, params)
    return ForwardDiff.gradient(y -> _filippov_guard_scalar(event, y, p), x)
end

function _filippov_normal_velocity(sys::ContinuousODE, event::SwitchingEvent,
                                   state::AbstractVector, params::AbstractVector)
    normal = _filippov_normal(event, state, params)
    field = collect(Float64, _filippov_rhs(sys, state, params, 0.0))
    return dot(normal, field), normal, field
end

function _filippov_normal_acceleration(sys::ContinuousODE, event::SwitchingEvent,
                                       state::AbstractVector, params::AbstractVector,
                                       field::AbstractVector, derivative_step::Float64)
    nf = norm(field)
    nf > 0 || return 0.0
    x_plus = collect(Float64, state) .+ derivative_step .* field
    x_minus = collect(Float64, state) .- derivative_step .* field
    v_plus = first(_filippov_normal_velocity(sys, event, x_plus, params))
    v_minus = first(_filippov_normal_velocity(sys, event, x_minus, params))
    return (v_plus - v_minus) / (2derivative_step)
end

"""
    filippov_guard_diagnostic(sys, event_or_name, state, params; derivative_step=1e-5)

Evaluate the guard value, guard normal, normal velocity, and second normal
derivative used by flow-side grazing/sliding tests. Unavailable derivatives are
reported in the returned `status` and `message`.
"""
function filippov_guard_diagnostic(sys::ContinuousODE, event::SwitchingEvent,
                                   state::AbstractVector, params::AbstractVector;
                                   derivative_step::Float64=1e-5)
    derivative_step > 0 || throw(ArgumentError(
        "derivative_step must be positive; got $derivative_step."))
    x = collect(Float64, state)
    p = _filippov_resolve_params(sys, collect(Float64, params))
    length(x) == sys.dim || throw(ArgumentError(
        "Filippov diagnostic state length $(length(x)) does not match system dimension $(sys.dim)."))
    try
        guard = Float64(_filippov_guard_scalar(event, x, p))
        velocity, normal, field = _filippov_normal_velocity(sys, event, x, p)
        acceleration = _filippov_normal_acceleration(
            sys, event, x, p, field, derivative_step)
        return FilippovGuardDiagnostic(
            event.name, x, p, guard, collect(Float64, normal), collect(Float64, field),
            Float64(velocity), Float64(acceleration), :ok, "")
    catch err
        err isa InterruptException && rethrow()
        return FilippovGuardDiagnostic(
            event.name, x, p, NaN, Float64[], Float64[], NaN, NaN,
            :unavailable, sprint(showerror, err))
    end
end

function filippov_guard_diagnostic(sys::ContinuousODE, event_name::Union{Nothing, AbstractString},
                                   state::AbstractVector, params::AbstractVector;
                                   derivative_step::Float64=1e-5)
    return filippov_guard_diagnostic(
        sys, _filippov_select_event(sys, event_name), state, params;
        derivative_step=derivative_step)
end

function _filippov_solve_trajectory(sys::ContinuousODE, config::FilippovGrazingConfig,
                                    params::Vector{Float64}, initial_point, solver,
                                    reltol::Float64, abstol::Float64)
    u0 = _resolve_initial_state(sys, initial_point)
    length(u0) == sys.dim || throw(ArgumentError(
        "Filippov grazing initial point length $(length(u0)) does not match system dimension $(sys.dim)."))
    prob = ODEProblem(sys.f, collect(Float64, u0), (config.t_start, config.t_stop), params)
    return solve(
        prob,
        solver;
        reltol=reltol,
        abstol=abstol,
        dense=true,
        save_everystep=true,
        save_start=true,
        save_end=true,
        maxiters=config.maxiters,
    )
end

function _filippov_state_at(sol, t::Float64)
    return collect(Float64, sol(t))
end

function _filippov_score(sys::ContinuousODE, event::SwitchingEvent, sol, params::Vector{Float64},
                         t::Float64, config::FilippovGrazingConfig)
    diag = filippov_guard_diagnostic(
        sys, event, _filippov_state_at(sol, t), params;
        derivative_step=config.derivative_step)
    diag.status === :ok || return Inf
    gscale = max(config.guard_tolerance, event.tolerance, eps(Float64))
    vscale = max(config.velocity_tolerance, eps(Float64))
    return hypot(diag.guard_value / gscale, diag.normal_velocity / vscale)
end

function _filippov_golden_minimize(f, a::Float64, b::Float64; iterations::Int)
    lo, hi = minmax(a, b)
    phi = (sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)
    fc = f(c)
    fd = f(d)
    for _ in 1:iterations
        if fc <= fd
            hi = d
            d = c
            fd = fc
            c = hi - phi * (hi - lo)
            fc = f(c)
        else
            lo = c
            c = d
            fc = fd
            d = lo + phi * (hi - lo)
            fd = f(d)
        end
    end
    t = (lo + hi) / 2
    return t, f(t)
end

function _filippov_point_from_time(sys::ContinuousODE, event::SwitchingEvent, sol,
                                   params::Vector{Float64}, t::Float64,
                                   config::FilippovGrazingConfig)
    state = _filippov_state_at(sol, t)
    diag = filippov_guard_diagnostic(
        sys, event, state, params; derivative_step=config.derivative_step)
    if diag.status !== :ok
        return nothing, "guard derivatives unavailable near t=$(t): $(diag.message)"
    end
    near_guard = abs(diag.guard_value) <= max(config.guard_tolerance, event.tolerance)
    near_velocity = abs(diag.normal_velocity) <= config.velocity_tolerance
    generic = abs(diag.normal_acceleration) > config.acceleration_tolerance
    status = near_guard && near_velocity ? (generic ? :grazing : :degenerate) : :candidate
    converged = status in (:grazing, :degenerate)
    point = FilippovGrazingPoint(
        event.name, t, state, params, diag.guard_value, diag.normal_velocity,
        diag.normal_acceleration, status, converged)
    return point, ""
end

"""
    filippov_grazing_points(sys, config; params, initial_point, solver, reltol, abstol)

Integrate a continuous system and locate orbit tangencies with the selected
`SwitchingEvent` guard. The detector samples the combined `(h, hdot)` residual,
refines local minima with dense-output minimization, and reports generic and
degenerate tangencies explicitly.
"""
function filippov_grazing_points(sys::ContinuousODE, config::FilippovGrazingConfig;
                                 params::AbstractVector=Float64[],
                                 initial_point::Union{Nothing, AbstractVector}=nothing,
                                 solver=Tsit5(),
                                 reltol::Float64=1e-9,
                                 abstol::Float64=1e-9)
    event = _filippov_select_event(sys, config.event_name)
    local_params = _filippov_resolve_params(sys, collect(Float64, params))
    sol = _filippov_solve_trajectory(
        sys, config, local_params, initial_point, solver, reltol, abstol)
    sample_times = collect(range(config.t_start, config.t_stop; length=config.sample_count))
    scores = [
        _filippov_score(sys, event, sol, local_params, t, config)
        for t in sample_times
    ]
    points = FilippovGrazingPoint[]
    warnings = String[]
    for i in 2:(length(sample_times) - 1)
        isfinite(scores[i]) || continue
        scores[i] <= scores[i - 1] && scores[i] <= scores[i + 1] || continue
        lo, hi = sample_times[i - 1], sample_times[i + 1]
        t_star, _ = _filippov_golden_minimize(
            t -> _filippov_score(sys, event, sol, local_params, t, config),
            lo, hi; iterations=config.refine_iterations)
        if any(point -> abs(point.time - t_star) <= config.min_event_separation, points)
            continue
        end
        point, warning = _filippov_point_from_time(
            sys, event, sol, local_params, t_star, config)
        isnothing(point) && (push!(warnings, warning); continue)
        point.status in (:grazing, :degenerate) || continue
        push!(points, point)
    end
    sort!(points; by=point -> point.time)
    status = isempty(points) ? (isempty(warnings) ? :not_found : :warning) :
             (any(point -> point.status === :grazing, points) ? :grazing : :degenerate)
    return FilippovGrazingResult(
        points, sys.name, event.name, local_params, (config.t_start, config.t_stop),
        status, unique!(warnings), now())
end

function _filippov_inject2(base::Vector{Float64}, primary_index::Int, primary::Float64,
                           secondary_index::Int, secondary::Float64)
    required = max(primary_index, secondary_index, length(base))
    params = length(base) >= required ? copy(base) : vcat(base, zeros(required - length(base)))
    params[primary_index] = primary
    params[secondary_index] = secondary
    return params
end

function _filippov_guard_margin(sys::ContinuousODE, event::SwitchingEvent,
                                config::FilippovGrazingConfig, params::Vector{Float64},
                                initial_point, solver, reltol::Float64, abstol::Float64,
                                margin::Symbol)
    sol = _filippov_solve_trajectory(
        sys, config, params, initial_point, solver, reltol, abstol)
    sample_times = collect(range(config.t_start, config.t_stop; length=config.sample_count))
    values = Float64[]
    for t in sample_times
        value = try
            Float64(_filippov_guard_scalar(event, _filippov_state_at(sol, t), params))
        catch err
            err isa InterruptException && rethrow()
            NaN
        end
        push!(values, value)
    end
    finite_indices = findall(isfinite, values)
    isempty(finite_indices) && return NaN, NaN
    local_index = margin === :minimum ?
        finite_indices[argmin(values[finite_indices])] :
        finite_indices[argmax(values[finite_indices])]
    lo = sample_times[max(1, local_index - 1)]
    hi = sample_times[min(length(sample_times), local_index + 1)]
    objective = margin === :minimum ?
        (t -> Float64(_filippov_guard_scalar(event, _filippov_state_at(sol, t), params))) :
        (t -> -Float64(_filippov_guard_scalar(event, _filippov_state_at(sol, t), params)))
    t_star, value = _filippov_golden_minimize(
        objective, lo, hi; iterations=config.refine_iterations)
    return margin === :minimum ? value : -value, t_star
end

"""
    filippov_grazing_locus(sys, config; initial_point, solver, reltol, abstol)

Continue a two-parameter grazing locus by solving a signed guard-margin root on
each secondary-parameter slice. This is a compact scalar defining-system path
for simple flow-grazing loci; full hybrid-segment continuation is deliberately
outside this API.
"""
function filippov_grazing_locus(sys::ContinuousODE, config::FilippovGrazingLocusConfig;
                                initial_point::Union{Nothing, AbstractVector}=nothing,
                                solver=Tsit5(),
                                reltol::Float64=1e-9,
                                abstol::Float64=1e-9)
    event = _filippov_select_event(sys, config.grazing.event_name)
    required_params = max(config.primary_index, config.secondary_index)
    length(sys.param_names) >= required_params || throw(ArgumentError(
        "Filippov grazing locus requires sys.param_names to include primary_index=$(config.primary_index) " *
        "and secondary_index=$(config.secondary_index); got $(length(sys.param_names)) names."))
    base = isempty(config.fixed_params) ? copy(sys.default_params) : copy(config.fixed_params)
    isempty(base) && throw(ArgumentError(
        "Filippov grazing locus requires fixed_params or system default_params."))
    secondary_values = collect(range(
        config.secondary_min, config.secondary_max; length=config.secondary_steps + 1))
    primary_values = fill(NaN, length(secondary_values))
    guard_values = fill(NaN, length(secondary_values))
    normal_velocities = fill(NaN, length(secondary_values))
    normal_accelerations = fill(NaN, length(secondary_values))
    statuses = fill(:not_bracketed, length(secondary_values))
    states = fill(NaN, sys.dim, length(secondary_values))

    for (idx, secondary) in enumerate(secondary_values)
        margin_cache = Dict{Float64, Float64}()
        margin_at = primary -> get!(margin_cache, Float64(primary)) do
            first(_filippov_guard_margin(
                sys, event, config.grazing,
                _filippov_inject2(base, config.primary_index, primary,
                                  config.secondary_index, secondary),
                initial_point, solver, reltol, abstol, config.margin))
        end
        lo = config.primary_min
        hi = config.primary_max
        f_lo = margin_at(lo)
        f_hi = margin_at(hi)
        if !isfinite(f_lo) || !isfinite(f_hi)
            statuses[idx] = :unavailable
            continue
        end
        if abs(f_lo) <= config.root_tolerance
            root = lo
        elseif abs(f_hi) <= config.root_tolerance
            root = hi
        elseif sign(f_lo) == sign(f_hi)
            statuses[idx] = :not_bracketed
            continue
        else
            root = (lo + hi) / 2
            left, right = lo, hi
            f_left = f_lo
            for _ in 1:config.max_bisection_iterations
                root = (left + right) / 2
                f_mid = margin_at(root)
                if !isfinite(f_mid)
                    statuses[idx] = :unavailable
                    break
                end
                if abs(f_mid) <= config.root_tolerance || abs(right - left) <= config.root_tolerance
                    break
                end
                if sign(f_mid) == sign(f_left)
                    left = root
                    f_left = f_mid
                else
                    right = root
                end
            end
        end
        statuses[idx] === :unavailable && continue
        params = _filippov_inject2(
            base, config.primary_index, root, config.secondary_index, secondary)
        result = filippov_grazing_points(
            sys, config.grazing; params=params, initial_point=initial_point,
            solver=solver, reltol=reltol, abstol=abstol)
        point = isempty(result.points) ? nothing :
            result.points[argmin([abs(item.guard_value) for item in result.points])]
        if isnothing(point)
            statuses[idx] = :not_found
            primary_values[idx] = root
            continue
        end
        primary_values[idx] = root
        states[:, idx] .= point.state
        guard_values[idx] = point.guard_value
        normal_velocities[idx] = point.normal_velocity
        normal_accelerations[idx] = point.normal_acceleration
        statuses[idx] = point.status
    end

    return FilippovGrazingLocusResult(
        primary_values, secondary_values, states, guard_values, normal_velocities,
        normal_accelerations, statuses, sys.name, event.name,
        (sys.param_names[config.primary_index], sys.param_names[config.secondary_index]),
        now())
end

function _filippov_state_samples(states::AbstractMatrix)
    return [collect(Float64, view(states, :, j)) for j in 1:size(states, 2)]
end

function _filippov_state_samples(states::AbstractVector)
    isempty(states) && return Vector{Vector{Float64}}()
    first(states) isa Real && return [collect(Float64, states)]
    return [collect(Float64, state) for state in states]
end

function _filippov_eval_side_field(f::Function, state::AbstractVector, params::AbstractVector)
    x = collect(Float64, state)
    p = collect(Float64, params)
    du = zeros(Float64, length(x))
    try
        value = f(du, x, p, 0.0)
        value === nothing && return du
        value isa AbstractVector && return collect(Float64, value)
    catch err
        err isa InterruptException && rethrow()
    end
    value = f(x, p)
    value isa AbstractVector || throw(ArgumentError(
        "One-sided vector fields must be in-place f!(du,u,p,t) or out-of-place f(u,p)."))
    return collect(Float64, value)
end

function _filippov_sliding_kind(v_minus::Float64, v_plus::Float64, tol::Float64)
    if abs(v_minus) <= tol || abs(v_plus) <= tol
        return :degenerate
    elseif v_minus > tol && v_plus < -tol
        return :attracting
    elseif v_minus < -tol && v_plus > tol
        return :repelling
    end
    return :crossing
end

function _filippov_make_segment(event::SwitchingEvent, samples, kinds, vminus, vplus,
                                start_idx::Int, end_idx::Int)
    return FilippovSlidingSegment(
        event.name, start_idx, end_idx, copy(samples[start_idx]), copy(samples[end_idx]),
        mean(vminus[start_idx:end_idx]), mean(vplus[start_idx:end_idx]),
        kinds[start_idx], :classified)
end

"""
    filippov_sliding_segments(event, states, params, f_minus, f_plus; velocity_tolerance=1e-9)

Classify sampled points on a switching surface from the normal velocities of
the two one-sided vector fields. `f_minus` is the vector field on `h < 0` and
`f_plus` on `h > 0`.
"""
function filippov_sliding_segments(event::SwitchingEvent, states, params::AbstractVector,
                                   f_minus::Function, f_plus::Function;
                                   velocity_tolerance::Float64=1e-9)
    velocity_tolerance >= 0 || throw(ArgumentError(
        "velocity_tolerance must be non-negative; got $velocity_tolerance."))
    samples = _filippov_state_samples(states)
    p = collect(Float64, params)
    isempty(samples) && return FilippovSlidingResult(
        FilippovSlidingSegment[], event.name, 0, :empty, String[], now())
    kinds = Symbol[]
    vminus = Float64[]
    vplus = Float64[]
    warnings = String[]
    for (idx, state) in enumerate(samples)
        try
            normal = _filippov_normal(event, state, p)
            fm = _filippov_eval_side_field(f_minus, state, p)
            fp = _filippov_eval_side_field(f_plus, state, p)
            length(fm) == length(normal) && length(fp) == length(normal) || throw(ArgumentError(
                "One-sided vector-field dimensions must match the guard state dimension."))
            vm = dot(normal, fm)
            vp = dot(normal, fp)
            push!(vminus, vm)
            push!(vplus, vp)
            push!(kinds, _filippov_sliding_kind(vm, vp, velocity_tolerance))
        catch err
            err isa InterruptException && rethrow()
            push!(warnings, "sample $idx could not be classified: $(sprint(showerror, err))")
            push!(vminus, NaN)
            push!(vplus, NaN)
            push!(kinds, :unavailable)
        end
    end

    segments = FilippovSlidingSegment[]
    start_idx = 1
    for idx in 2:length(kinds)
        if kinds[idx] != kinds[start_idx]
            kinds[start_idx] !== :unavailable && push!(
                segments, _filippov_make_segment(
                    event, samples, kinds, vminus, vplus, start_idx, idx - 1))
            start_idx = idx
        end
    end
    kinds[start_idx] !== :unavailable && push!(
        segments, _filippov_make_segment(
            event, samples, kinds, vminus, vplus, start_idx, length(kinds)))
    status = isempty(warnings) ? :ok : (isempty(segments) ? :unavailable : :warning)
    return FilippovSlidingResult(
        segments, event.name, length(samples), status, unique!(warnings), now())
end

function filippov_sliding_segments(sys::ContinuousODE, states, params::AbstractVector,
                                   f_minus::Function, f_plus::Function;
                                   event_name::Union{Nothing, AbstractString}=nothing,
                                   velocity_tolerance::Float64=1e-9)
    event = _filippov_select_event(sys, event_name)
    return filippov_sliding_segments(
        event, states, params, f_minus, f_plus;
        velocity_tolerance=velocity_tolerance)
end
