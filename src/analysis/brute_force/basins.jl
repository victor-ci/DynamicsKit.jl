"""
    BasinsCellGrid(nx, ny)

In/out per-cell state for a basins sweep (sweep cache hook): a `periodicity` matrix over the
(x, y) IC grid + a `known` mask. Pre-seed cached cells, pass via `basins_of_attraction(...; cells=grid)`;
the sweep fills the not-`known` cells in place and you read the grid back.
"""
mutable struct BasinsCellGrid
    periodicity::Matrix{Int}
    known::Matrix{Bool}   # Matrix{Bool}, not BitMatrix: threaded sweeps write distinct cells concurrently
end
BasinsCellGrid(nx::Int, ny::Int) = BasinsCellGrid(zeros(Int, nx, ny), fill(false, nx, ny))

function _basins_periodicity(cells::Union{Nothing, BasinsCellGrid}, nx::Int, ny::Int)
    cells === nothing && return zeros(Int, nx, ny)
    size(cells.periodicity) == (nx, ny) || throw(ArgumentError(
        "cells grid size $(size(cells.periodicity)) does not match the ($nx, $ny) basins grid."))
    return cells.periodicity
end

"""
    basins_of_attraction(sys::DiscreteMap, config::BasinsConfig) -> BasinsResult

Compute basins of attraction for a discrete map at a fixed parameter value.
For each initial condition on an (x, y) grid, iterates the map and determines
the periodicity of the resulting attractor.

The first two state variables are used as the initial condition grid axes.
"""
function basins_of_attraction(sys::DiscreteMap, config::BasinsConfig;
                              cells::Union{Nothing, BasinsCellGrid}=nothing)
    x_vals = collect(range(config.x_min, config.x_max, length=config.x_steps + 1))
    y_vals = collect(range(config.y_min, config.y_max, length=config.y_steps + 1))
    nx, ny = length(x_vals), length(y_vals)

    periodicity = _basins_periodicity(cells, nx, ny)
    # Collect one extra iterate so `_detect_period` can compare orbit[1] vs orbit[max_period + 1]
    # and detect period exactly equal to max_period.
    orbit_len = config.max_period + 1
    config.iterations >= orbit_len || throw(ArgumentError(
        "BasinsConfig.iterations ($(config.iterations)) must be at least max_period + 1 ($(orbit_len)); " *
        "otherwise the function would silently iterate more times than requested in order to fill the orbit window."
    ))
    points_to_drop = config.iterations - orbit_len

    # Build parameter vector
    p = _build_basins_params(config)
    base_ic = _basins_ic_template(sys, config)

    Threads.@threads for i in 1:nx
        for j in 1:ny
            (cells !== nothing && cells.known[i, j]) && continue   # cache hook: skip pre-seeded cells
            # Initial condition: place x,y on the chosen grid dims over the template.
            x0 = setindex(base_ic, x_vals[i], config.x_index)
            x0 = setindex(x0, y_vals[j], config.y_index)

            point = x0
            # Iterate through transient
            for k in 1:points_to_drop
                point = sys.f(point, p)
            end

            orbit = Vector{SVector{sys.dim, Float64}}(undef, orbit_len)
            for k in 1:orbit_len
                point = sys.f(point, p)
                orbit[k] = point
            end

            # Detect periodicity: find smallest T such that orbit[1] ≈ orbit[T+1]
            periodicity[i, j] = _detect_period(orbit, config.max_period, config.precision)
            cells !== nothing && (cells.known[i, j] = true)
        end
    end

    BasinsResult(
        x_vals,
        y_vals,
        periodicity,
        config.bif_param,
        config.max_period,
        sys.name,
        now(),
        config.x_index,
        config.y_index,
        collect(Float64, base_ic)
    )
end

"""
    basins_of_attraction(sys::ContinuousODE, config::BasinsConfig; solver, reltol, abstol) -> BasinsResult

Compute basins of attraction for a continuous ODE at a fixed parameter value.
For each initial condition on the (x, y) grid (varying state dims `x_index` /
`y_index` over `ic_template`), the trajectory is integrated and the periodicity
of its Poincaré return-map orbit is detected — the continuous analogue of the
discrete map version, sharing the same Poincaré-crossing collector and period
detector as the 2D bifurcation map. Quasiperiodic / chaotic basins classify as
period 0.
"""
function basins_of_attraction(sys::ContinuousODE, config::BasinsConfig;
                              solver=Tsit5(), reltol::Float64=1e-8, abstol::Float64=1e-8,
                              cells::Union{Nothing, BasinsCellGrid}=nothing)
    x_vals = collect(range(config.x_min, config.x_max, length=config.x_steps + 1))
    y_vals = collect(range(config.y_min, config.y_max, length=config.y_steps + 1))
    nx, ny = length(x_vals), length(y_vals)

    # Collect max_period + 1 crossings so period == max_period is detectable.
    crossings_needed = config.max_period + 1
    config.iterations >= crossings_needed || throw(ArgumentError(
        "BasinsConfig.iterations ($(config.iterations)) must be at least max_period + 1 ($(crossings_needed)); " *
        "otherwise the function would silently take more Poincaré crossings than requested to fill the orbit window."
    ))
    points_to_drop = max(config.iterations - crossings_needed, 0)

    p = _build_basins_params(config)
    base_ic = collect(Float64, _basins_ic_template(sys, config))
    periodicity = _basins_periodicity(cells, nx, ny)

    Threads.@threads for i in 1:nx
        for j in 1:ny
            (cells !== nothing && cells.known[i, j]) && continue   # cache hook: skip pre-seeded cells
            u0 = copy(base_ic)
            u0[config.x_index] = x_vals[i]
            u0[config.y_index] = y_vals[j]
            orbit_points = _collect_poincare_points(
                sys,
                p;
                initial_point=u0,
                crossings=crossings_needed,
                transient=points_to_drop,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                projected=true,
                min_crossing_time=config.min_crossing_time
            )
            if length(orbit_points) == crossings_needed
                orbit = [SVector{length(point), Float64}(point) for point in orbit_points]
                periodicity[i, j] = _detect_period(orbit, config.max_period, config.precision)
            else
                periodicity[i, j] = 0
            end
            cells !== nothing && (cells.known[i, j] = true)
        end
    end

    BasinsResult(
        x_vals,
        y_vals,
        periodicity,
        config.bif_param,
        config.max_period,
        sys.name,
        now(),
        config.x_index,
        config.y_index,
        collect(Float64, base_ic)
    )
end

