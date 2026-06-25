"""
Continuous-time phase portrait generation.
"""

"""Default state labels used by static plots and serialized phase portraits."""
_default_state_names(dim::Int) = [Symbol(:x, i) for i in 1:dim]

"""Convert a vector of state vectors into an `n × dim` matrix."""
function _state_matrix(states::AbstractVector, dim::Int)
    mat = Matrix{Float64}(undef, length(states), dim)
    for (idx, state) in enumerate(states)
        mat[idx, :] .= state
    end
    return mat
end

function _phase_save_times(config::PhasePortraitConfig)
    max_saved_points = max(config.max_saved_points, 0)
    max_saved_points == 0 && return nothing
    max_saved_points == 1 && return Float64[config.time_stop]
    return collect(range(config.time_start, config.time_stop; length=max_saved_points))
end

function _kept_trajectory(sol, dim::Int, keep_fraction::Float64)
    total_points = length(sol.t)
    total_points == 0 && return Float64[], Matrix{Float64}(undef, 0, dim)

    keep_start = keep_fraction >= 1.0 ? 1 : max(1, floor(Int, total_points * (1.0 - keep_fraction)) + 1)
    kept_count = total_points - keep_start + 1
    kept_t = Vector{Float64}(undef, kept_count)
    kept_states = Matrix{Float64}(undef, kept_count, dim)

    for (row_idx, sol_idx) in enumerate(keep_start:total_points)
        kept_t[row_idx] = Float64(sol.t[sol_idx])
        kept_states[row_idx, :] .= sol.u[sol_idx]
    end

    return kept_t, kept_states
end

"""
    phase_portrait(sys::ContinuousODE, config::PhasePortraitConfig; kwargs...) -> PhasePortraitResult

Integrate a continuous-time system and return the retained full-state trajectory plus
Poincaré section crossings.
"""
function phase_portrait(sys::ContinuousODE, config::PhasePortraitConfig;
                        params::Vector{Float64}=Float64[],
                        initial_point::Union{Nothing, AbstractVector}=nothing,
                        solver=Tsit5(),
                        reltol::Float64=1e-8,
                        abstol::Float64=1e-8,
                        state_names::Vector{Symbol}=_default_state_names(sys.dim))
    config.time_stop > config.time_start || error("Phase portrait time_stop must be greater than time_start.")
    local_params = _resolve_continuous_params(sys, params)
    u0 = _warmup_from_section(
        sys,
        _resolve_initial_state(sys, initial_point),
        local_params;
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        maxiters=config.maxiters,
        min_crossing_time=config.min_crossing_time
    )

    crossing_states = Vector{Vector{Float64}}()
    crossing_count = Ref(0)
    function on_crossing!(integrator)
        config.poincare_crossings > 0 || return
        crossing_count[] < config.poincare_crossings || return
        crossing_count[] += 1
        push!(crossing_states, collect(Float64, integrator.u))
    end

    cb = _make_poincare_callback(sys.section, on_crossing!; min_crossing_time=config.min_crossing_time)
    prob = ODEProblem(sys.f, u0, (config.time_start, config.time_stop), local_params)
    save_times = _phase_save_times(config)
    sol = if isnothing(save_times)
        solve(
            prob,
            solver;
            callback=cb,
            reltol=reltol,
            abstol=abstol,
            save_everystep=true,
            save_start=true,
            save_end=true,
            maxiters=config.maxiters,
            dt=_default_poincare_initial_dt(config.min_crossing_time)
        )
    else
        solve(
            prob,
            solver;
            callback=cb,
            reltol=reltol,
            abstol=abstol,
            save_everystep=false,
            saveat=save_times,
            save_start=false,
            save_end=false,
            maxiters=config.maxiters,
            dt=_default_poincare_initial_dt(config.min_crossing_time)
        )
    end

    keep_fraction = clamp(config.tail_fraction, 0.0, 1.0)
    kept_t, kept_states = _kept_trajectory(sol, sys.dim, keep_fraction)

    kept_crossings = config.poincare_crossings > 0 ?
        first(crossing_states, min(length(crossing_states), config.poincare_crossings)) :
        Vector{Vector{Float64}}()

    PhasePortraitResult(
        kept_t,
        kept_states,
        _state_matrix(kept_crossings, sys.dim),
        local_params,
        sys.name,
        length(state_names) == sys.dim ? copy(state_names) : _default_state_names(sys.dim),
        now()
    )
end
