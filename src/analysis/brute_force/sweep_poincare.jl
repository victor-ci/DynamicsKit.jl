"""
Brute-force bifurcation diagram generation via parameter sweeps.
"""

"""
    brute_force_diagram(sys::DiscreteMap, config::BruteForceConfig;
                        initial_point=nothing, amplitude_cutoff=500.0) -> BruteForceResult

Generate a bifurcation diagram for a discrete map by sweeping a parameter and
collecting post-transient iterates.
"""
function brute_force_diagram(sys::DiscreteMap, config::BruteForceConfig;
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             amplitude_cutoff::Float64=500.0)
    param_values = range(config.param_min, config.param_max, length=config.param_steps + 1)
    x0 = isnothing(initial_point) ? zeros(SVector{sys.dim, Float64}) : SVector{sys.dim}(initial_point)

    # Collect per-thread results, then compact in a single pass directly into
    # preallocated outputs.
    all_points = Vector{Vector{SVector{sys.dim, Float64}}}(undef, length(param_values))

    Threads.@threads for i in eachindex(param_values)
        a = param_values[i]
        p = build_sweep_params(config, a)
        point = x0
        local_points = SVector{sys.dim, Float64}[]

        for j in 1:config.iterations
            point = sys.f(point, p)
            if j > config.transient && all(abs.(point) .< amplitude_cutoff)
                push!(local_points, point)
            end
        end
        all_points[i] = local_points
    end

    n_total = sum(length(local_points) for local_points in all_points)
    flat_params = Vector{Float64}(undef, n_total)
    points_mat = Matrix{Float64}(undef, n_total, sys.dim)
    write_idx = 0
    @inbounds for i in eachindex(param_values)
        a = param_values[i]
        for pt in all_points[i]
            write_idx += 1
            flat_params[write_idx] = a
            for k in 1:sys.dim
                points_mat[write_idx, k] = pt[k]
            end
        end
    end

    BruteForceResult(flat_params, points_mat, sys.name,
                     sys.param_names[config.param_index], now())
end

"""
    brute_force_diagram(sys::ContinuousODE, config::BruteForceConfig;
                        initial_point=nothing, solver=Tsit5(),
                        reltol=1e-8, abstol=1e-8) -> BruteForceResult

Generate a bifurcation diagram for a continuous ODE system by sweeping a parameter and
collecting post-transient Poincaré section crossings.
"""
function brute_force_diagram(sys::ContinuousODE, config::BruteForceConfig;
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             solver=Tsit5(),
                             reltol::Float64=1e-8,
                             abstol::Float64=1e-8)
    param_values = range(config.param_min, config.param_max, length=config.param_steps + 1)
    u0 = _resolve_initial_state(sys, initial_point)
    proj = sys.section.projection
    map_dim = length(proj)
    n_keep = max(config.iterations - config.transient, 0)

    # As in the discrete-map sibling above: single-pass fill into preallocated outputs.
    all_points = Vector{Vector{Vector{Float64}}}(undef, length(param_values))

    Threads.@threads for i in eachindex(param_values)
        a = param_values[i]
        p = build_sweep_params(config, a)
        all_points[i] = _collect_poincare_points(
            sys,
            p;
            initial_point=u0,
            crossings=n_keep,
            transient=config.transient,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            projected=true,
            min_crossing_time=config.min_crossing_time
        )
    end

    n_total = sum(length(local_points) for local_points in all_points)
    flat_params = Vector{Float64}(undef, n_total)
    points_mat = Matrix{Float64}(undef, n_total, map_dim)
    write_idx = 0
    @inbounds for i in eachindex(param_values)
        a = param_values[i]
        for pt in all_points[i]
            write_idx += 1
            flat_params[write_idx] = a
            for k in 1:map_dim
                points_mat[write_idx, k] = pt[k]
            end
        end
    end

    BruteForceResult(flat_params, points_mat, sys.name,
                     sys.param_names[config.param_index], now())
end

"""Resolve the initial state for a continuous-time system, falling back to the system default."""
function _resolve_initial_state(sys::ContinuousODE, initial_point::Union{Nothing, AbstractVector})
    if isnothing(initial_point)
        return isempty(sys.default_initial_state) ? zeros(Float64, sys.dim) : copy(sys.default_initial_state)
    end
    return collect(Float64, initial_point)
end

"""Choose a conservative explicit initial step to avoid repeated `initdt` warnings."""
_default_poincare_initial_dt(min_crossing_time::Float64) = max(min_crossing_time, 1e-6)

"""Evaluate the section condition at a state, returning `NaN` if the callback signature is incompatible."""
function _section_value(sys::ContinuousODE, u::AbstractVector)
    try
        return Float64(sys.section.condition(u, 0.0, nothing))
    catch
        return NaN
    end
end

"""Advance a state slightly off the Poincaré section before enabling root finding."""
function _warmup_from_section(sys::ContinuousODE, u0::AbstractVector, params::AbstractVector;
                              solver=Tsit5(),
                              reltol::Float64=1e-8,
                              abstol::Float64=1e-8,
                              maxiters::Int=10_000_000,
                              min_crossing_time::Float64=1e-6,
                              initial_dt::Float64=_default_poincare_initial_dt(min_crossing_time))
    section_value = _section_value(sys, u0)
    if !isfinite(section_value) || abs(section_value) > 100sqrt(eps(Float64))
        return collect(Float64, u0)
    end

    warmup_t = max(10initial_dt, 10min_crossing_time, min(0.05, max(sys.tspan_hint * 1e-3, initial_dt)))
    prob = ODEProblem(sys.f, collect(Float64, u0), (0.0, warmup_t), collect(Float64, params))

    try
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
        return collect(Float64, sol.u[end])
    catch
        return collect(Float64, u0)
    end
end

"""Build a direction-aware Poincaré section callback."""
function _make_poincare_callback(section::PoincareSection, on_crossing!::Function;
                                 min_crossing_time::Float64=1e-6)
    wrapped! = function(integrator)
        integrator.t <= min_crossing_time && return
        on_crossing!(integrator)
    end

    positive_affect! = section.direction == -1 ? (integrator -> nothing) : wrapped!
    negative_affect! = section.direction == 1 ? nothing : wrapped!

    ContinuousCallback(
        section.condition,
        positive_affect!;
        rootfind=SciMLBase.RightRootFind,
        affect_neg! = negative_affect!,
        save_positions=(false, false)
    )
end

function _poincare_crossing_diagnostics(; crossings_requested::Int,
                                        transient_crossings::Int,
                                        total_crossings_requested::Int,
                                        crossings_found::Int,
                                        total_crossings_found::Int,
                                        final_time::Float64,
                                        termination_reason::Symbol,
                                        final_status::Symbol,
                                        divergence_callback_activated::Bool,
                                        state_callback_activated::Bool,
                                        solver_retcode::AbstractString,
                                        error_message::Union{Nothing, AbstractString}=nothing)
    diagnostics = Dict{String, Any}(
        "crossingsRequested" => crossings_requested,
        "transientCrossings" => transient_crossings,
        "totalCrossingsRequested" => total_crossings_requested,
        "crossingsFound" => crossings_found,
        "totalCrossingsFound" => total_crossings_found,
        "finalTime" => final_time,
        "terminationReason" => String(termination_reason),
        "finalStatus" => String(final_status),
        "divergenceCallbackActivated" => divergence_callback_activated,
        "stateCallbackActivated" => state_callback_activated,
        "solverRetcode" => solver_retcode
    )
    isnothing(error_message) || (diagnostics["error"] = String(error_message))
    return diagnostics
end

_plain_float_vector(values) = Float64[Float64(values[i]) for i in 1:length(values)]

"""Collect successive Poincaré section crossings for a continuous-time system."""
function _collect_poincare_points(sys::ContinuousODE, params::AbstractVector;
                                  initial_point::Union{Nothing, AbstractVector}=nothing,
                                  crossings::Int,
                                  transient::Int=0,
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8,
                                  abstol::Float64=1e-8,
                                  projected::Bool=true,
                                  tmax::Union{Nothing, Float64}=nothing,
                                  maxiters::Int=10_000_000,
                                  min_crossing_time::Float64=1e-6,
                                  divergence_cutoff::Float64=Inf,
                                  return_diagnostics::Bool=false)
    if crossings <= 0
        points = Vector{Vector{Float64}}()
        diagnostics = _poincare_crossing_diagnostics(
            crossings_requested=max(crossings, 0),
            transient_crossings=max(transient, 0),
            total_crossings_requested=max(transient, 0) + max(crossings, 0),
            crossings_found=0,
            total_crossings_found=0,
            final_time=0.0,
            termination_reason=:no_crossings_requested,
            final_status=:ok,
            divergence_callback_activated=false,
            state_callback_activated=false,
            solver_retcode="not_run"
        )
        return return_diagnostics ? (points=points, diagnostics=diagnostics) : points
    end

    initial_dt = _default_poincare_initial_dt(min_crossing_time)
    u0 = _plain_float_vector(_warmup_from_section(
        sys,
        _resolve_initial_state(sys, initial_point),
        params;
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        maxiters=maxiters,
        min_crossing_time=min_crossing_time,
        initial_dt=initial_dt
    ))
    total_crossings = transient + crossings
    local_tmax = isnothing(tmax) ? sys.tspan_hint * max(total_crossings, 1) * 2 : tmax

    found_points = Vector{Vector{Float64}}()
    crossing_count = Ref(0)
    completed = Ref(false)
    terminal_status = Ref(:running)
    final_point = Ref(copy(u0))

    function on_crossing!(integrator)
        crossing_count[] += 1
        final_point[] = collect(Float64, integrator.u)
        if crossing_count[] > transient
            point = projected ? _project_section_state(sys.section, integrator.u) : collect(Float64, integrator.u)
            push!(found_points, point)
        end
        if crossing_count[] >= total_crossings
            completed[] = true
            terminate!(integrator)
        end
    end

    cb = _make_poincare_callback(sys.section, on_crossing!; min_crossing_time=min_crossing_time)
    final_time = Ref(NaN)
    state_cb = _map_state_termination_callback(divergence_cutoff, terminal_status, final_point, final_time)
    callbacks = CallbackSet(cb, state_cb)
    prob = ODEProblem(sys.f, u0, (0.0, local_tmax), collect(Float64, params))

    sol = try
        solve(
            prob,
            solver;
            callback=callbacks,
            reltol=reltol,
            abstol=abstol,
            dt=initial_dt,
            save_everystep=false,
            save_start=false,
            save_end=true,
            maxiters=maxiters
        )
    catch err
        err isa InterruptException && rethrow()
        points = Vector{Vector{Float64}}()
        diagnostics = _poincare_crossing_diagnostics(
            crossings_requested=crossings,
            transient_crossings=transient,
            total_crossings_requested=total_crossings,
            crossings_found=0,
            total_crossings_found=crossing_count[],
            final_time=NaN,
            termination_reason=:integration_failed,
            final_status=:integration_failed,
            divergence_callback_activated=false,
            state_callback_activated=false,
            solver_retcode="exception",
            error_message=_continuation_error_message(err)
        )
        return return_diagnostics ? (points=points, diagnostics=diagnostics) : points
    end

    final_time = isempty(sol.t) ? local_tmax : Float64(sol.t[end])
    final_status = terminal_status[] == :running ? :ok : terminal_status[]
    termination_reason = if terminal_status[] == :diverged || terminal_status[] == :invalid_state
        terminal_status[]
    elseif completed[]
        :requested_crossings_found
    else
        :insufficient_crossings
    end
    diagnostics = _poincare_crossing_diagnostics(
        crossings_requested=crossings,
        transient_crossings=transient,
        total_crossings_requested=total_crossings,
        crossings_found=length(found_points),
        total_crossings_found=crossing_count[],
        final_time=final_time,
        termination_reason=termination_reason,
        final_status=final_status,
        divergence_callback_activated=terminal_status[] == :diverged,
        state_callback_activated=terminal_status[] == :diverged || terminal_status[] == :invalid_state,
        solver_retcode=string(sol.retcode)
    )

    return return_diagnostics ? (points=found_points, diagnostics=diagnostics) : found_points
end

"""Collect a grouped post-transient discrete orbit for one parameter value."""
function _sample_discrete_orbit(sys::DiscreteMap, params::AbstractVector;
                                initial_point::Union{Nothing, AbstractVector}=nothing,
                                iterations::Int,
                                transient::Int=0,
                                amplitude_cutoff::Float64=500.0)
    total_iterations = max(iterations, 0)
    point = isnothing(initial_point) ? zeros(SVector{sys.dim, Float64}) : SVector{sys.dim}(initial_point)
    sampled_points = Vector{Vector{Float64}}()
    valid = true

    for step in 1:total_iterations
        point = sys.f(point, params)
        if !all(isfinite, point) || !all(abs.(point) .< amplitude_cutoff)
            valid = false
            break
        end
        step > transient && push!(sampled_points, collect(Float64, point))
    end

    return (
        points=sampled_points,
        valid=valid,
        diverged=!valid,
        final_point=collect(Float64, point)
    )
end

"""Collect grouped post-transient Poincaré crossings for one parameter value."""
function _sample_continuous_poincare_orbit(sys::ContinuousODE, params::AbstractVector;
                                           initial_point::Union{Nothing, AbstractVector}=nothing,
                                           crossings::Int,
                                           transient::Int=0,
                                           solver=Tsit5(),
                                           reltol::Float64=1e-8,
                                           abstol::Float64=1e-8,
                                           projected::Bool=true,
                                           tmax::Union{Nothing, Float64}=nothing,
                                           maxiters::Int=10_000_000,
                                           min_crossing_time::Float64=1e-6)
    sample = _collect_poincare_points(
        sys,
        params;
        initial_point=initial_point,
        crossings=crossings,
        transient=transient,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        projected=projected,
        tmax=tmax,
        maxiters=maxiters,
        min_crossing_time=min_crossing_time,
        return_diagnostics=true
    )
    sampled_points = sample.points
    diagnostics = sample.diagnostics

    return (
        points=sampled_points,
        valid=length(sampled_points) == max(crossings, 0),
        diverged=get(diagnostics, "finalStatus", "") == "diverged",
        final_point=isempty(sampled_points) ? Float64[] : copy(last(sampled_points)),
        crossing_diagnostics=diagnostics
    )
end

"""Mean closure error for candidate periods over a sampled orbit."""
function _orbit_closure_errors(points::AbstractVector, max_period::Int)
    errors = fill(Inf, max(max_period, 0))
    isempty(points) && return errors

    normalized_points = [collect(Float64, point) for point in points]
    for period in 1:length(errors)
        length(normalized_points) > period || continue
        total = 0.0
        count = 0
        for idx in 1:(length(normalized_points) - period)
            total += norm(normalized_points[idx] .- normalized_points[idx + period])
            count += 1
        end
        count > 0 && (errors[period] = total / count)
    end

    return errors
end

"""Basic geometry summary for a sampled orbit cloud."""
function _orbit_geometry_summary(points::AbstractVector)
    isempty(points) && return (center=Float64[], span=Float64[], minima=Float64[], maxima=Float64[])

    matrix = reduce(hcat, [collect(Float64, point) for point in points])'
    minima = vec(minimum(matrix; dims=1))
    maxima = vec(maximum(matrix; dims=1))
    center = vec(sum(matrix; dims=1)) ./ size(matrix, 1)
    return (center=center, span=maxima .- minima, minima=minima, maxima=maxima)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Basins of Attraction
# ═══════════════════════════════════════════════════════════════════════════════
