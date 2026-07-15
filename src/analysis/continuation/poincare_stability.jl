"""Resolve a continuous-time parameter vector, falling back to the system defaults."""
function _resolve_continuous_params(sys::ContinuousODE, params::Vector{Float64})
    resolved = isempty(params) ? copy(sys.default_params) : copy(params)
    isempty(resolved) && error("No parameter vector provided for continuous system $(sys.name). Pass `params` or define `default_params` in the system constructor.")
    return resolved
end

"""Return the projected Poincaré image of a section point after the requested number of returns."""
function _poincare_projected(sys::ContinuousODE, point, params;
                             period::Int=1,
                             solver=Tsit5(),
                             reltol::Float64=1e-8,
                             abstol::Float64=1e-8,
                             tmax::Union{Nothing, Float64}=nothing,
                             min_crossing_time::Float64=1e-6)
    state, found = _poincare_return(
        sys,
        point,
        params;
        period=period,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    )
    return found ? (_project_section_state(sys.section, state), true) : (zeros(Float64, state_dim(sys)), false)
end

"""Compute the full-state Poincaré return of a continuous-time system."""
function _poincare_return(sys::ContinuousODE, point, params;
                          period::Int=1,
                          solver=Tsit5(),
                          reltol::Float64=1e-8,
                          abstol::Float64=1e-8,
                          tmax::Union{Nothing, Float64}=nothing,
                          min_crossing_time::Float64=1e-6,
                          max_retries::Int=3)
    state = _lift_section_state(sys.section, point, sys.dim)
    params_vec = collect(Float64, params)

    for _ in 1:period
        found = false
        local_tmax = isnothing(tmax) ? sys.tspan_hint : tmax
        for retry in 0:max_retries
            returns = _collect_poincare_points(
                sys,
                params_vec;
                initial_point=state,
                crossings=1,
                transient=0,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                projected=false,
                tmax=local_tmax * (2.0 ^ retry),
                min_crossing_time=min_crossing_time
            )
            if !isempty(returns)
                state = returns[end]
                found = true
                break
            end
        end
        found || return zeros(Float64, sys.dim), false
    end

    return state, true
end

"""State Jacobian of an in-place continuous-time RHS with respect to the state."""
function _ode_state_jacobian(sys::ContinuousODE,
                             state::AbstractVector,
                             params::AbstractVector,
                             t::Real)
    # Must stay generic over element types: the variational RHS is itself
    # differentiated by stiff solvers (Rosenbrock W-methods dualize the state
    # and time), so coercing to Float64 here rejects those Dual numbers.
    T = promote_type(eltype(state), eltype(params), typeof(t))
    u0 = collect(T, state)
    rhs = u -> begin
        du = similar(u, promote_type(eltype(u), T))
        sys.f(du, u, params, t)
        du
    end
    return ForwardDiff.jacobian(rhs, u0)
end

"""Evaluate an in-place continuous-time RHS as a Float64 vector."""
function _ode_rhs_vector(sys::ContinuousODE,
                         state::AbstractVector,
                         params::AbstractVector,
                         t::Real)
    du = zeros(Float64, sys.dim)
    sys.f(du, collect(Float64, state), collect(Float64, params), t)
    return du
end

"""Differentiate a smooth Poincaré section condition with respect to state."""
function _section_condition_gradient(sys::ContinuousODE,
                                     state::AbstractVector,
                                     t::Real)
    u0 = collect(Float64, state)
    condition = u -> sys.section.condition(u, t, nothing)
    gradient = try
        ForwardDiff.gradient(condition, u0)
    catch err
        error("Could not differentiate the Poincaré section condition for $(sys.name): $(_continuation_error_message(err))")
    end
    all(isfinite, gradient) || error("Poincaré section gradient for $(sys.name) contains non-finite values.")
    return collect(Float64, gradient)
end

"""Apply the event-time correction to a flow Jacobian at a Poincaré crossing."""
function _poincare_event_corrected_jacobian(sys::ContinuousODE,
                                            event_state::AbstractVector,
                                            params::AbstractVector,
                                            t::Real,
                                            flow_jacobian::AbstractMatrix)
    f_event = _ode_rhs_vector(sys, event_state, params, t)
    section_gradient = _section_condition_gradient(sys, event_state, t)
    denominator = dot(section_gradient, f_event)
    scale = max(norm(section_gradient) * norm(f_event), 1.0)
    abs(denominator) > 100eps(Float64) * scale ||
        error("Poincaré section is tangent to the flow for $(sys.name); variational derivative is ill-conditioned.")
    return Matrix(flow_jacobian) .- f_event * ((section_gradient' * Matrix(flow_jacobian)) ./ denominator)
end

"""Build an augmented RHS for state plus state-transition matrix integration."""
# Recomputes the state Jacobian via ForwardDiff every step and allocates each call; acceptable at
# current problem sizes.
function _variational_rhs(sys::ContinuousODE)
    n = sys.dim
    return function(dz, z, p, t)
        state = @view z[1:n]
        du = @view dz[1:n]
        sys.f(du, state, p, t)
        A = _ode_state_jacobian(sys, state, p, t)
        Φ = reshape(@view(z[n + 1:end]), n, n)
        dΦ = A * Matrix(Φ)
        dz[n + 1:end] .= vec(dΦ)
        nothing
    end
end

"""Advance a variational state slightly off the section before enabling root finding."""
function _warmup_variational_from_section(sys::ContinuousODE,
                                          state::AbstractVector,
                                          params::AbstractVector;
                                          solver=Tsit5(),
                                          reltol::Float64=1e-8,
                                          abstol::Float64=1e-8,
                                          maxiters::Int=10_000_000,
                                          min_crossing_time::Float64=1e-6,
                                          initial_dt::Float64=_default_poincare_initial_dt(min_crossing_time))
    u0 = collect(Float64, state)
    identity_jacobian = Matrix{Float64}(I, sys.dim, sys.dim)
    section_value = _section_value(sys, u0)
    if !isfinite(section_value) || abs(section_value) > 100sqrt(eps(Float64))
        return u0, identity_jacobian
    end

    warmup_t = max(10initial_dt, 10min_crossing_time, min(0.05, max(sys.tspan_hint * 1e-3, initial_dt)))
    z0 = vcat(u0, vec(identity_jacobian))
    prob = ODEProblem(_variational_rhs(sys), z0, (0.0, warmup_t), collect(Float64, params))
    sol = solve(
        prob,
        solver;
        reltol=reltol,
        abstol=abstol,
        save_everystep=false,
        save_start=false,
        save_end=true,
        maxiters=maxiters,
        dt=initial_dt
    )
    !isempty(sol.u) || error("Variational warmup for $(sys.name) produced no states.")
    z = collect(Float64, sol.u[end])
    return z[1:sys.dim], Matrix{Float64}(reshape(z[sys.dim + 1:end], sys.dim, sys.dim))
end

"""Build a direction-aware Poincaré callback for augmented variational states."""
function _make_variational_poincare_callback(section::PoincareSection,
                                             full_dim::Int,
                                             on_crossing!::Function;
                                             min_crossing_time::Float64=1e-6)
    condition = (z, t, integrator) -> section.condition(@view(z[1:full_dim]), t, nothing)
    wrapped! = function(integrator)
        integrator.t <= min_crossing_time && return
        on_crossing!(integrator)
    end

    positive_affect! = section.direction == -1 ? (integrator -> nothing) : wrapped!
    negative_affect! = section.direction == 1 ? nothing : wrapped!

    return ContinuousCallback(
        condition,
        positive_affect!;
        rootfind=SciMLBase.RightRootFind,
        affect_neg! = negative_affect!,
        save_positions=(false, false)
    )
end

"""Integrate one variational Poincaré return from a full state on the section."""
function _poincare_variational_segment(sys::ContinuousODE,
                                       state::AbstractVector,
                                       params::AbstractVector;
                                       solver=Tsit5(),
                                       reltol::Float64=1e-8,
                                       abstol::Float64=1e-8,
                                       tmax::Union{Nothing, Float64}=nothing,
                                       min_crossing_time::Float64=1e-6,
                                       max_retries::Int=3,
                                       maxiters::Int=10_000_000)
    params_vec = collect(Float64, params)
    warm_state, warm_jacobian = _warmup_variational_from_section(
        sys,
        state,
        params_vec;
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        maxiters=maxiters,
        min_crossing_time=min_crossing_time
    )

    for retry in 0:max_retries
        found_state = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        found_jacobian = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
        on_crossing! = integrator -> begin
            z = collect(Float64, integrator.u)
            event_state = z[1:sys.dim]
            flow_jacobian = Matrix{Float64}(reshape(z[sys.dim + 1:end], sys.dim, sys.dim))
            corrected = _poincare_event_corrected_jacobian(
                sys,
                event_state,
                params_vec,
                integrator.t,
                flow_jacobian
            )
            found_state[] = event_state
            found_jacobian[] = corrected * warm_jacobian
            terminate!(integrator)
        end
        callback = _make_variational_poincare_callback(
            sys.section,
            sys.dim,
            on_crossing!;
            min_crossing_time=min_crossing_time
        )
        z0 = vcat(warm_state, vec(Matrix{Float64}(I, sys.dim, sys.dim)))
        local_tmax = (isnothing(tmax) ? sys.tspan_hint : tmax) * (2.0 ^ retry)
        prob = ODEProblem(_variational_rhs(sys), z0, (0.0, local_tmax), params_vec)
        solve(
            prob,
            solver;
            callback=callback,
            reltol=reltol,
            abstol=abstol,
            save_everystep=false,
            save_start=false,
            save_end=false,
            maxiters=maxiters,
            dt=_default_poincare_initial_dt(min_crossing_time)
        )

        if !isnothing(found_state[]) && !isnothing(found_jacobian[])
            return found_state[]::Vector{Float64}, found_jacobian[]::Matrix{Float64}, true
        end
    end

    return zeros(Float64, sys.dim), zeros(Float64, sys.dim, sys.dim), false
end

"""Full-state Poincaré return and derivative from variational equations."""
function _poincare_return_variational(sys::ContinuousODE,
                                      point,
                                      params;
                                      period::Int=1,
                                      solver=Tsit5(),
                                      reltol::Float64=1e-8,
                                      abstol::Float64=1e-8,
                                      tmax::Union{Nothing, Float64}=nothing,
                                      min_crossing_time::Float64=1e-6,
                                      max_retries::Int=3)
    state = _lift_section_state(sys.section, point, sys.dim)
    jacobian = Matrix{Float64}(I, sys.dim, sys.dim)
    params_vec = collect(Float64, params)

    for _ in 1:period
        next_state, segment_jacobian, found = _poincare_variational_segment(
            sys,
            state,
            params_vec;
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time,
            max_retries=max_retries
        )
        found || return zeros(Float64, sys.dim), zeros(Float64, sys.dim, sys.dim), false
        state = next_state
        jacobian = segment_jacobian * jacobian
    end

    return state, jacobian, true
end

"""Projected Poincaré-map derivative from variational equations."""
function _poincare_projected_jacobian_variational(sys::ContinuousODE,
                                                  point,
                                                  params;
                                                  period::Int=1,
                                                  solver=Tsit5(),
                                                  reltol::Float64=1e-8,
                                                  abstol::Float64=1e-8,
                                                  tmax::Union{Nothing, Float64}=nothing,
                                                  min_crossing_time::Float64=1e-6)
    projected = collect(Float64, point)
    map_dim = state_dim(sys)
    length(projected) == map_dim || throw(ArgumentError(
        "Variational Poincaré derivative expects a projected section point of length $map_dim, got $(length(projected))."
    ))
    _, full_jacobian, found = _poincare_return_variational(
        sys,
        projected,
        params;
        period=period,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    )
    found || return zeros(Float64, map_dim, map_dim), false

    lift_jacobian = zeros(Float64, sys.dim, map_dim)
    for (col, row) in pairs(sys.section.projection)
        lift_jacobian[row, col] = 1.0
    end
    return full_jacobian[sys.section.projection, :] * lift_jacobian, true
end

"""Estimate reasonable projected search bounds from a short Poincaré trajectory sample."""
function _estimate_section_bounds(sys::ContinuousODE, param_value::Float64, params::Vector{Float64}, param_index::Int;
                                  linked_param_indices::AbstractVector{<:Integer}=Int[],
                                  initial_point::Union{Nothing, AbstractVector}=nothing,
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8,
                                  abstol::Float64=1e-8,
                                  tmax::Union{Nothing, Float64}=nothing,
                                  min_crossing_time::Float64=1e-6,
                                  transient::Int=20,
                                  crossings::Int=40,
                                  padding::Float64=0.15,
                                  min_half_width::Float64=0.5)
    local_params = _inject_param(params, param_index, param_value, linked_param_indices)
    sample = _collect_poincare_points(
        sys,
        local_params;
        initial_point=initial_point,
        crossings=crossings,
        transient=transient,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        projected=true,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    )

    isempty(sample) && error("Could not estimate Poincaré search bounds for $(sys.name) at parameter $param_value. Provide `search_min`/`search_max` or a better `initial_point`.")

    map_dim = state_dim(sys)
    sample_mat = reduce(vcat, permutedims.(sample))
    lo = zeros(Float64, map_dim)
    hi = zeros(Float64, map_dim)
    for d in 1:map_dim
        smin = minimum(sample_mat[:, d])
        smax = maximum(sample_mat[:, d])
        half_width = max((smax - smin) / 2, min_half_width)
        center = (smin + smax) / 2
        pad = padding * max(1.0, 2half_width)
        lo[d] = center - half_width - pad
        hi[d] = center + half_width + pad
    end
    return lo, hi
end

"""Resolve or automatically find the section seed used to start continuous-time continuation."""
function _resolve_continuous_seed(sys::ContinuousODE, period::Int, param_value::Float64,
                                  initial_point::Union{Nothing, AbstractVector},
                                  params::Vector{Float64}, param_index::Int;
                                  linked_param_indices::AbstractVector{<:Integer}=Int[],
                                  search_min::Union{Nothing, AbstractVector}=nothing,
                                  search_max::Union{Nothing, AbstractVector}=nothing,
                                  n_initial::Int=12,
                                  sample_seed_points::Bool=true,
                                  sample_seed_crossings::Int=0,
                                  sample_seed_transient::Int=0,
                                  tol::Float64=1e-8,
                                  max_iter::Int=40,
                                  fd_step::Float64=1e-6,
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8,
                                  abstol::Float64=1e-8,
                                  tmax::Union{Nothing, Float64}=nothing,
                                  min_crossing_time::Float64=1e-6)
    if !isnothing(initial_point)
        raw = collect(Float64, initial_point)
        return length(raw) == sys.dim ? _project_section_state(sys.section, raw) : raw
    end

    lo, hi = if isnothing(search_min) || isnothing(search_max)
        _estimate_section_bounds(
            sys,
            param_value,
            params,
            param_index;
            linked_param_indices=linked_param_indices,
            initial_point=sys.default_initial_state,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
    else
        collect(Float64, search_min), collect(Float64, search_max)
    end

    seed_points = sample_seed_points ? _collect_trajectory_seed_points(
        sys,
        param_value,
        params,
        param_index;
        linked_param_indices=linked_param_indices,
        initial_point=sys.default_initial_state,
        crossings=sample_seed_crossings > 0 ? sample_seed_crossings : max(16, 3 * n_initial),
        transient=sample_seed_transient > 0 ? sample_seed_transient : max(8, n_initial),
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    ) : nothing

    seeds = find_periodic_skeleton(
        sys,
        [period],
        param_value;
        n_initial=n_initial,
        search_min=lo,
        search_max=hi,
        seed_points=seed_points,
        params=params,
        param_index=param_index,
        linked_param_indices=linked_param_indices,
        tol=tol,
        max_iter=max_iter,
        fd_step=fd_step,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    )

    isempty(seeds) && error("Could not find a period-$period Poincaré fixed point for $(sys.name) at parameter $param_value")
    stable_seeds = filter(seed -> seed.stable, seeds)
    return (isempty(stable_seeds) ? first(seeds) : first(stable_seeds)).point
end

"""Sample Poincaré crossings from a trajectory to obtain likely continuation seed points."""
function _collect_trajectory_seed_points(sys::ContinuousODE, param_value::Float64,
                                         params::Vector{Float64}, param_index::Int;
                                         linked_param_indices::AbstractVector{<:Integer}=Int[],
                                         initial_point::Union{Nothing, AbstractVector}=nothing,
                                         crossings::Int,
                                         transient::Int,
                                         solver=Tsit5(),
                                         reltol::Float64=1e-8,
                                         abstol::Float64=1e-8,
                                         tmax::Union{Nothing, Float64}=nothing,
                                         min_crossing_time::Float64=1e-6)
    local_params = _inject_param(params, param_index, param_value, linked_param_indices)
    return _collect_poincare_points(
        sys,
        local_params;
        initial_point=initial_point,
        crossings=crossings,
        transient=transient,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        projected=true,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    )
end

"""
    collect_trajectory_seed_points(sys::ContinuousODE, param_value, params, param_index;
        linked_param_indices=Int[], initial_point=nothing, crossings, transient,
        solver=Tsit5(), reltol=1e-8, abstol=1e-8, tmax=nothing, min_crossing_time=1e-6)

Sample Poincaré return-map points of `sys` at a single parameter value — the public entry point for
scripted analysis that seeds continuation from a trajectory (e.g. reproducibility scripts). Injects
`param_value` at `param_index` (plus any `linked_param_indices`) and returns the projected crossings.
"""
const collect_trajectory_seed_points = _collect_trajectory_seed_points

"""
Finite-difference Jacobian for vector-valued residual functions.

Uses a per-column step `h_j = max(delta, delta * abs(x[j]))` so the perturbation
scales with the magnitude of each component. A single fixed `delta` causes
catastrophic cancellation in `(F(x+h) - F(x))` when `|x[j]|` is large (the
relevant scale of `F` is large but the absolute change `delta` is tiny) and
unnecessary loss of precision when `|x[j]|` is near zero (where the absolute
floor of `delta` is the right scale).
"""
function _fd_jacobian(F, x, delta::Float64=1e-6)
    x0 = collect(Float64, x)
    Fx = F(x0)
    J = Matrix{Float64}(undef, length(Fx), length(x0))
    for j in eachindex(x0)
        h = max(delta, delta * abs(x0[j]))
        xpert = copy(x0)
        xpert[j] += h
        J[:, j] = (F(xpert) .- Fx) ./ h
    end
    return J
end

const _MAP_STABILITY_UNIT_CIRCLE_TOL = 1e-7

_multipliers_are_stable(multipliers; tol::Float64=_MAP_STABILITY_UNIT_CIRCLE_TOL) =
    all(isfinite(abs(value)) && abs(value) <= 1.0 + tol for value in multipliers)

function _map_multipliers(sys::DiscreteMap,
                          state::AbstractVector,
                          params::AbstractVector,
                          period::Int;
                          kwargs...)
    dim = sys.dim
    x0 = collect(Float64, state)
    local_params = collect(Float64, params)
    F = x -> begin
        current = SVector{dim}(x)
        for _ in 1:period
            current = sys.f(current, local_params)
        end
        Array(current) .- x
    end
    return eigvals(ForwardDiff.jacobian(F, x0) + Matrix{Float64}(I, dim, dim))
end

function _map_multipliers(sys::ContinuousODE,
                          state::AbstractVector,
                          params::AbstractVector,
                          period::Int;
                          fd_step::Float64=1e-6,
                          solver=Tsit5(),
                          reltol::Float64=1e-8,
                          abstol::Float64=1e-8,
                          tmax::Union{Nothing, Float64}=nothing,
                          min_crossing_time::Float64=1e-6,
                          ode_jacobian_method::Symbol=:finite_difference)
    x0 = collect(Float64, state)
    map_dim = length(x0)
    local_params = collect(Float64, params)
    if ode_jacobian_method == :variational
        map_jacobian, found = _poincare_projected_jacobian_variational(
            sys,
            x0,
            local_params;
            period=period,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
        found || error("Variational Poincaré derivative failed to find a period-$period return for $(sys.name).")
        return eigvals(map_jacobian)
    elseif ode_jacobian_method != :finite_difference
        throw(ArgumentError("Unknown ODE Jacobian method $(repr(ode_jacobian_method)); expected :finite_difference or :variational."))
    end

    F = x -> begin
        next_point, found = _poincare_projected(
            sys,
            x,
            local_params;
            period=period,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
        found || return fill(NaN, map_dim)
        next_point .- x
    end
    return eigvals(_fd_jacobian(F, x0, fd_step) + Matrix{Float64}(I, map_dim, map_dim))
end

function _map_stability(sys::DynamicalSystem,
                        state::AbstractVector,
                        params::AbstractVector,
                        period::Int;
                        tol::Float64=_MAP_STABILITY_UNIT_CIRCLE_TOL,
                        kwargs...)
    multipliers = _map_multipliers(sys, state, params, period; kwargs...)
    return _multipliers_are_stable(multipliers; tol=tol), multipliers
end

"""Return a recorded continuation point with a corrected `stable` field.

Continuation branch records are stored as `NamedTuple`s by BifurcationKit and by
the portable serializer. Keep this helper deliberately narrow so stability
post-processing has one predictable representation instead of silently converting
arbitrary property-bearing objects.
"""
_branch_point_with_stability(point::NamedTuple, stable::Bool) =
    merge(point, (; stable))

function _branch_point_with_stability(point, _stable::Bool)
    throw(ArgumentError("Branch stability post-processing expects NamedTuple branch points, got $(typeof(point))."))
end

function _branch_points_with_recomputed_stability(sys::DynamicalSystem,
                                                  branch::BranchResult,
                                                  points::AbstractVector,
                                                  base_params::AbstractVector,
                                                  linked_param_indices::Vector{Int};
                                                  require_state::Bool=true,
                                                  kwargs...)
    isempty(points) && return similar(points, 0)
    param_index = findfirst(==(branch.param_name), sys.param_names)
    isnothing(param_index) && error("Cannot recompute branch stability: branch parameter '$(branch.param_name)' is not present in system '$(sys.name)' parameters $(collect(sys.param_names)).")

    base = collect(Float64, base_params)
    projected_dim = state_dim(sys)
    period = max(branch.period, 1)
    return map(eachindex(points)) do idx
        point = points[idx]
        if !hasproperty(point, :param)
            return point
        end
        has_state = all(i -> hasproperty(point, Symbol(:x, i)), 1:projected_dim)
        if !has_state
            require_state && error("Cannot recompute branch stability for $(branch.system_name) period-$(branch.period) point $idx: recorded state fields x1..x$projected_dim are incomplete.")
            return point
        end
        local_params = _inject_param(base, param_index, Float64(point.param), linked_param_indices)
        state = _branch_point_state(point, projected_dim)
        try
            stable, _ = _map_stability(sys, state, local_params, period; kwargs...)
            return _branch_point_with_stability(point, stable)
        catch err
            error("Failed to recompute return-map stability for $(branch.system_name) period-$(branch.period) branch point $idx at $(branch.param_name)=$(point.param): $(_continuation_error_message(err))")
        end
    end
end

function _branch_with_recomputed_stability(sys::DynamicalSystem,
                                           branch::BranchResult,
                                           base_params::AbstractVector,
                                           linked_param_indices::Vector{Int};
                                           kwargs...)
    points = _branch_points(branch)
    updated_points = _branch_points_with_recomputed_stability(
        sys,
        branch,
        points,
        base_params,
        linked_param_indices;
        kwargs...
    )
    special_points = try
        collect(branch.branch.specialpoint)
    catch
        Any[]
    end
    updated_specials = _branch_points_with_recomputed_stability(
        sys,
        branch,
        special_points,
        base_params,
        linked_param_indices;
        require_state=false,
        kwargs...
    )
    if length(updated_points) == length(points) &&
       all(updated_points[i] == points[i] for i in eachindex(points)) &&
       length(updated_specials) == length(special_points) &&
       all(updated_specials[i] == special_points[i] for i in eachindex(special_points))
        return branch
    end
    return BranchResult(
        CombinedBranchResult(Vector{Any}(updated_points), Vector{Any}(updated_specials)),
        branch.period,
        branch.system_name,
        branch.param_name,
        branch.timestamp
    )
end

function _map_residual(sys::DiscreteMap,
                       state::AbstractVector,
                       params::AbstractVector,
                       period::Int;
                       kwargs...)
    dim = sys.dim
    length(state) == dim || throw(ArgumentError("Discrete residual state has length $(length(state)) but system dimension is $dim."))
    current = SVector{dim, Float64}(state)
    x0 = current
    local_params = collect(Float64, params)
    for _ in 1:period
        current = sys.f(current, local_params)
        all(isfinite, current) || return fill(NaN, dim)
    end
    return Array(current .- x0)
end

function _map_residual(sys::ContinuousODE,
                       state::AbstractVector,
                       params::AbstractVector,
                       period::Int;
                       solver=Tsit5(),
                       reltol::Float64=1e-8,
                       abstol::Float64=1e-8,
                       tmax::Union{Nothing, Float64}=nothing,
                       min_crossing_time::Float64=1e-6,
                       kwargs...)
    x0 = collect(Float64, state)
    next_point, found = _poincare_projected(
        sys,
        x0,
        collect(Float64, params);
        period=period,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    )
    found || return fill(NaN, length(x0))
    return next_point .- x0
end

function _diagnostic_sample_indices(n::Int, max_points::Int)
    n <= 0 && return Int[]
    max_points <= 0 && return collect(1:n)
    n <= max_points && return collect(1:n)
    return unique(round.(Int, range(1, n, length=max_points)))
end

_finite_diagnostic_values(values::AbstractVector{<:Real}) =
    Float64[Float64(value) for value in values if isfinite(Float64(value))]

function _diagnostic_max(values::AbstractVector{<:Real})
    finite = _finite_diagnostic_values(values)
    return isempty(finite) ? NaN : maximum(finite)
end

function _diagnostic_median(values::AbstractVector{<:Real})
    finite = sort(_finite_diagnostic_values(values))
    n = length(finite)
    n == 0 && return NaN
    mid = (n + 1) ÷ 2
    return isodd(n) ? finite[mid] : (finite[mid] + finite[mid + 1]) / 2
end

function _multiplier_payload(value)
    z = complex(value)
    Dict{String, Any}(
        "re" => real(z),
        "im" => imag(z),
        "abs" => abs(z)
    )
end

function _multiplier_spectrum_payload(values)
    [_multiplier_payload(value) for value in values]
end
