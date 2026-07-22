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
    basins_of_attraction(sys::DiscreteMap, config::BasinsConfig; cells=nothing, backend=CPUBackend()) -> BasinsResult

Compute basins of attraction for a discrete map at a fixed parameter value.
For each initial condition on an (x, y) grid, iterates the map and determines
the periodicity of the resulting attractor.

The first two state variables are used as the initial condition grid axes.

`backend` optionally runs the (always cell-independent) sweep on a GPU; see [`ComputeBackend`](@ref).
The result's `compute_backend` field records what actually ran (`:cpu` unless a GPU was used).
"""
function basins_of_attraction(sys::DiscreteMap, config::BasinsConfig;
                              cells::Union{Nothing, BasinsCellGrid}=nothing,
                              backend::ComputeBackend=CPUBackend())
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
    p = build_basins_params(config)
    base_ic = basins_ic_template(sys, config)

    ka_backend, compute_backend_used = _resolve_gpu_backend(
        backend, true, "basins_of_attraction", "a DiscreteMap system (always true; this message should be unreachable)"
    )

    if ka_backend !== nothing
        x_vals_dev = _gpu_upload(ka_backend, x_vals)
        y_vals_dev = _gpu_upload(ka_backend, y_vals)
        p_sv = SVector{length(p), Float64}(p)
        _gpu_run_2d_sweep!(
            ka_backend, nx, ny, (periodicity,), _basins_gpu_kernel!, cells,
            x_vals_dev, y_vals_dev,
            sys.f, p_sv, base_ic, config.x_index, config.y_index, points_to_drop, config.max_period, config.precision
        )
    else
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
        collect(Float64, base_ic);
        compute_backend=compute_backend_used
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

`backend` optionally runs the (cell-independent) sweep on a GPU via DiffEqGPU's `EnsembleGPUKernel`,
when the system carries a GPU out-of-place RHS and `precision` is no tighter than the section-crossing
localization floor; see [`ComputeBackend`](@ref) and "Optional GPU acceleration" in `docs/julia-package.md`.
The result's `compute_backend` field records what actually ran. `CPUBackend()`/`AutoBackend()` never
error; an explicit `GPUBackend` on an ineligible system/config raises a clear `ArgumentError`.
"""
function basins_of_attraction(sys::ContinuousODE, config::BasinsConfig;
                              solver=Tsit5(), reltol::Float64=1e-8, abstol::Float64=1e-8,
                              cells::Union{Nothing, BasinsCellGrid}=nothing,
                              backend::ComputeBackend=CPUBackend())
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

    p = build_basins_params(config)
    base_ic = collect(Float64, basins_ic_template(sys, config))
    periodicity = _basins_periodicity(cells, nx, ny)

    ka_backend, compute_backend_used = _resolve_continuous_gpu_backend(
        backend, sys, true, config.precision, "basins_of_attraction (ContinuousODE)",
        "a system with a GPU out-of-place right-hand side"
    )

    if ka_backend !== nothing
        cells_to_do = Tuple{Int, Int}[]
        for j in 1:ny, i in 1:nx
            (cells !== nothing && cells.known[i, j]) && continue
            push!(cells_to_do, (i, j))
        end
        if !isempty(cells_to_do)
            u0_list = Vector{Vector{Float64}}(undef, length(cells_to_do))
            for (k, (i, j)) in enumerate(cells_to_do)
                u0 = copy(base_ic)
                u0[config.x_index] = x_vals[i]
                u0[config.y_index] = y_vals[j]
                u0_list[k] = u0
            end
            p_list = [copy(p) for _ in cells_to_do]
            warmed = _continuous_gpu_warmup_states(sys, u0_list, p_list;
                solver=solver, reltol=reltol, abstol=abstol, min_crossing_time=config.min_crossing_time)
            results = _continuous_poincare_gpu_sweep(sys, warmed, p_list, ka_backend;
                transient=points_to_drop, max_period=config.max_period, precision=config.precision,
                reltol=reltol, abstol=abstol, min_crossing_time=config.min_crossing_time,
                divergence_cutoff=Inf)
            for (k, (i, j)) in enumerate(cells_to_do)
                periodicity[i, j] = results[k].period
                cells !== nothing && (cells.known[i, j] = true)
            end
        end
    else
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
        collect(Float64, base_ic);
        compute_backend=compute_backend_used
    )
end
