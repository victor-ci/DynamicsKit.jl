"""
Largest-Lyapunov-Exponent diagnostics promoted to public library APIs.
"""

function _lyapunov_params(sys::DiscreteMap, config::LyapunovConfig, varied_value::Float64)
    required_len = max(
        length(sys.param_names),
        config.param_index,
        isempty(config.linked_param_indices) ? 0 : maximum(config.linked_param_indices)
    )
    params = isempty(config.fixed_params) ? zeros(Float64, required_len) : copy(config.fixed_params)
    if length(params) < required_len
        old_len = length(params)
        resize!(params, required_len)
        fill!(@view(params[(old_len + 1):required_len]), 0.0)
    end
    params[config.param_index] = varied_value
    for idx in config.linked_param_indices
        params[idx] = varied_value
    end
    return params
end

function _lyapunov_params(sys::ContinuousODE, config::LyapunovConfig, varied_value::Float64)
    params = if !isempty(config.fixed_params)
        copy(config.fixed_params)
    else
        _resolve_continuous_params(sys, Float64[])
    end
    required_len = max(
        length(params),
        config.param_index,
        isempty(config.linked_param_indices) ? 0 : maximum(config.linked_param_indices)
    )
    if length(params) < required_len
        old_len = length(params)
        resize!(params, required_len)
        fill!(@view(params[(old_len + 1):required_len]), 0.0)
    end
    params[config.param_index] = varied_value
    for idx in config.linked_param_indices
        params[idx] = varied_value
    end
    return params
end

function _lyapunov_point_classification(exponent::Float64, estimation_status::Symbol, neutral_tolerance::Float64)
    estimation_status == :collapsed && return :periodic
    estimation_status == :ok || return :unresolved
    isfinite(exponent) || return :unresolved
    exponent > neutral_tolerance && return :chaotic_candidate
    abs(exponent) <= neutral_tolerance && return :quasiperiodic_neutral_candidate
    return :periodic
end

"""
    lyapunov_diagram(sys, config::LyapunovConfig; kwargs...) -> LyapunovDiagramResult

Sweep one bifurcation parameter and estimate the largest Lyapunov exponent at each
sample using the existing two-trajectory estimators.
"""
function lyapunov_diagram(sys::DiscreteMap, config::LyapunovConfig;
                          initial_point::Union{Nothing, AbstractVector}=nothing)
    param_values = collect(range(config.param_min, config.param_max, length=config.param_steps + 1))
    x0 = isnothing(initial_point) ? zeros(SVector{sys.dim, Float64}) : SVector{sys.dim}(initial_point)
    exponents = fill(NaN, length(param_values))
    classifications = fill(:unresolved, length(param_values))
    estimation_statuses = fill(:uncomputed, length(param_values))
    sample_counts = zeros(Int, length(param_values))

    Threads.@threads for idx in eachindex(param_values)
        param = param_values[idx]
        params = _lyapunov_params(sys, config, param)
        estimate = _estimate_discrete_map_largest_lyapunov(
            sys,
            params,
            x0,
            config.transient,
            config.iterations,
            config.perturbation,
            config.divergence_cutoff
        )
        exponents[idx] = Float64(estimate.exponent)
        estimation_statuses[idx] = estimate.estimation_status
        sample_counts[idx] = estimate.sample_count
        classifications[idx] = _lyapunov_point_classification(exponents[idx], estimate.estimation_status, config.neutral_tolerance)
    end

    return LyapunovDiagramResult(
        param_values,
        exponents,
        classifications,
        estimation_statuses,
        sample_counts,
        config.neutral_tolerance,
        sys.name,
        sys.param_names[config.param_index],
        now()
    )
end

function lyapunov_diagram(sys::ContinuousODE, config::LyapunovConfig;
                          initial_point::Union{Nothing, AbstractVector}=nothing,
                          solver=Tsit5(),
                          reltol::Float64=1e-8,
                          abstol::Float64=1e-8)
    param_values = collect(range(config.param_min, config.param_max, length=config.param_steps + 1))
    u0 = _resolve_initial_state(sys, initial_point)
    exponents = fill(NaN, length(param_values))
    classifications = fill(:unresolved, length(param_values))
    estimation_statuses = fill(:uncomputed, length(param_values))
    sample_counts = zeros(Int, length(param_values))

    Threads.@threads for idx in eachindex(param_values)
        param = param_values[idx]
        params = _lyapunov_params(sys, config, param)
        estimate = _estimate_continuous_poincare_largest_lyapunov(
            sys,
            params,
            u0,
            config.transient,
            config.iterations,
            config.perturbation,
            config.divergence_cutoff;
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=config.min_crossing_time
        )
        exponents[idx] = Float64(estimate.exponent)
        estimation_statuses[idx] = estimate.estimation_status
        sample_counts[idx] = estimate.sample_count
        classifications[idx] = _lyapunov_point_classification(exponents[idx], estimate.estimation_status, config.neutral_tolerance)
    end

    return LyapunovDiagramResult(
        param_values,
        exponents,
        classifications,
        estimation_statuses,
        sample_counts,
        config.neutral_tolerance,
        sys.name,
        sys.param_names[config.param_index],
        now()
    )
end

"""
    lyapunov_field(sys, config::BifurcationMapConfig; kwargs...) -> LyapunovFieldResult
    lyapunov_field(result::LyapunovFieldResult) -> LyapunovFieldResult
    lyapunov_field(result::BifurcationMapResult) -> LyapunovFieldResult

Estimate a direct 2D largest-Lyapunov-exponent field over the parameter plane, or
return the first-class Lyapunov layer carried by a 2D bifurcation-map result.
"""
function _validate_direct_lyapunov_field_config(config::BifurcationMapConfig)
    config.a_index != config.b_index || throw(ArgumentError(
        "lyapunov_field requires two distinct parameter axes; got a_index=$(config.a_index) and b_index=$(config.b_index)."
    ))
    !config.reuse_neighbor_seeds || throw(ArgumentError(
        "lyapunov_field computes an exact fixed-seed field and does not support neighbour-seeded traversal. " *
        "Pass BifurcationMapConfig(reuse_neighbor_seeds=false, ...)."
    ))
    isempty(config.multistability_initial_points) || throw(ArgumentError(
        "lyapunov_field does not support multistability_initial_points; use one fixed initial condition per run."
    ))
    !config.adaptive_refinement_enabled || throw(ArgumentError(
        "lyapunov_field does not support adaptive_refinement_enabled; rerun at higher resolution instead."
    ))
    return nothing
end

function _record_direct_field_lyapunov!(storage,
                                        i::Int,
                                        j::Int,
                                        estimate,
                                        neutral_tolerance::Float64)
    exponent = Float64(estimate.exponent)
    storage.exponents[i, j] = exponent
    classification = _lyapunov_point_classification(exponent, estimate.estimation_status, neutral_tolerance)
    storage.status_codes[i, j] = _map_lyapunov_status_code(classification)
    storage.estimation_status_codes[i, j] = _map_lyapunov_estimation_status_code(estimate.estimation_status)
    storage.sample_counts[i, j] = estimate.sample_count
    return storage
end

"""
    LyapunovCellGrid(na, nb)

In/out per-cell state for a direct `lyapunov_field` sweep (sweep cache hook): exponent / status /
sample arrays + a `known` mask. Pre-seed cached cells, pass via `lyapunov_field(...; cells=grid)`;
the sweep fills the not-`known` cells in place and you read the grid back (e.g. to store a cache entry).
"""
mutable struct LyapunovCellGrid
    exponents::Matrix{Float64}
    status_codes::Matrix{Int}
    estimation_status_codes::Matrix{Int}
    sample_counts::Matrix{Int}
    known::Matrix{Bool}   # Matrix{Bool}, not BitMatrix: threaded sweeps write distinct cells concurrently
end

function LyapunovCellGrid(na::Int, nb::Int)
    s = _map_lyapunov_storage(na, nb)
    return LyapunovCellGrid(s.exponents, s.status_codes, s.estimation_status_codes, s.sample_counts, fill(false, na, nb))
end

function _lyapunov_field_storage(cells::Union{Nothing, LyapunovCellGrid}, na::Int, nb::Int)
    cells === nothing && return _map_lyapunov_storage(na, nb)
    size(cells.exponents) == (na, nb) || throw(ArgumentError(
        "cells grid size $(size(cells.exponents)) does not match the ($na, $nb) sweep grid."))
    return cells
end

function lyapunov_field(sys::DiscreteMap, config::BifurcationMapConfig;
                        initial_point::Union{Nothing, AbstractVector}=nothing,
                        cells::Union{Nothing, LyapunovCellGrid}=nothing)
    _validate_direct_lyapunov_field_config(config)
    length(sys.param_names) >= 2 || throw(ArgumentError("lyapunov_field requires a system with at least two parameters."))

    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    x0 = isnothing(initial_point) ? zeros(SVector{sys.dim, Float64}) : SVector{sys.dim}(initial_point)
    storage = _lyapunov_field_storage(cells, length(a_vals), length(b_vals))
    param_template = _map_param_template(config)
    a_indices = _map_a_write_indices(config)
    b_indices = _map_b_write_indices(config)
    transient = _map_lyapunov_transient(config)
    iterations = _map_lyapunov_iterations(config)

    chunks = _balanced_index_chunks(length(a_vals) * length(b_vals), Threads.nthreads())
    Threads.@threads for chunk_idx in eachindex(chunks)
        param_buffer = copy(param_template)
        for idx in chunks[chunk_idx]
            i = ((idx - 1) % length(a_vals)) + 1
            j = ((idx - 1) ÷ length(a_vals)) + 1
            (cells !== nothing && cells.known[i, j]) && continue   # cache hook: skip pre-seeded cells
            params = _map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
            estimate = _estimate_discrete_map_largest_lyapunov(
                sys,
                params,
                x0,
                transient,
                iterations,
                config.lyapunov_perturbation,
                config.divergence_cutoff
            )
            _record_direct_field_lyapunov!(storage, i, j, estimate, config.lyapunov_neutral_tolerance)
            cells !== nothing && (cells.known[i, j] = true)
        end
    end

    return LyapunovFieldResult(
        a_vals,
        b_vals,
        Float64.(storage.exponents),
        Int.(storage.status_codes),
        Int.(storage.estimation_status_codes),
        Int.(storage.sample_counts),
        config.lyapunov_neutral_tolerance,
        sys.name,
        (sys.param_names[config.a_index], sys.param_names[config.b_index]),
        now()
    )
end

function lyapunov_field(sys::ContinuousODE, config::BifurcationMapConfig;
                        initial_point::Union{Nothing, AbstractVector}=nothing,
                        solver=Tsit5(),
                        reltol::Float64=1e-8,
                        abstol::Float64=1e-8,
                        cells::Union{Nothing, LyapunovCellGrid}=nothing)
    _validate_direct_lyapunov_field_config(config)
    length(sys.param_names) >= 2 || throw(ArgumentError("lyapunov_field requires a system with at least two parameters."))

    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    u0 = _resolve_initial_state(sys, initial_point)
    storage = _lyapunov_field_storage(cells, length(a_vals), length(b_vals))
    param_template = _map_param_template(config)
    a_indices = _map_a_write_indices(config)
    b_indices = _map_b_write_indices(config)
    transient = _map_lyapunov_transient(config)
    iterations = _map_lyapunov_iterations(config)

    chunks = _balanced_index_chunks(length(a_vals) * length(b_vals), Threads.nthreads())
    Threads.@threads for chunk_idx in eachindex(chunks)
        param_buffer = copy(param_template)
        for idx in chunks[chunk_idx]
            i = ((idx - 1) % length(a_vals)) + 1
            j = ((idx - 1) ÷ length(a_vals)) + 1
            (cells !== nothing && cells.known[i, j]) && continue   # cache hook: skip pre-seeded cells
            params = _map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
            estimate = _estimate_continuous_poincare_largest_lyapunov(
                sys,
                params,
                u0,
                transient,
                iterations,
                config.lyapunov_perturbation,
                config.divergence_cutoff;
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=config.min_crossing_time
            )
            _record_direct_field_lyapunov!(storage, i, j, estimate, config.lyapunov_neutral_tolerance)
            cells !== nothing && (cells.known[i, j] = true)
        end
    end

    return LyapunovFieldResult(
        a_vals,
        b_vals,
        Float64.(storage.exponents),
        Int.(storage.status_codes),
        Int.(storage.estimation_status_codes),
        Int.(storage.sample_counts),
        config.lyapunov_neutral_tolerance,
        sys.name,
        (sys.param_names[config.a_index], sys.param_names[config.b_index]),
        now()
    )
end

lyapunov_field(result::LyapunovFieldResult) = result

function lyapunov_field(result::BifurcationMapResult)
    isnothing(result.lyapunov) && throw(ArgumentError(
        "BifurcationMapResult for $(result.system_name) does not carry a Lyapunov field. " *
        "Re-run bifurcation_map with BifurcationMapConfig(lyapunov_enabled=true, ...)."
    ))
    return result.lyapunov
end
