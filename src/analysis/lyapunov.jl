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

"""
Classify a Lyapunov estimate purely in terms of the estimation-status code space
(`_MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL`) and the classification code space
(`_MAP_LYAPUNOV_STATUS_CODE_BY_SYMBOL`, the return value) — no `Symbol` in or out, so this is safe to
call from inside a `@kernel` body (the direct-field GPU kernel in `gpu_kernels.jl`). CPU-only callers
convert their `Symbol` estimation status to a code with `_map_lyapunov_estimation_status_code` first,
and the returned code back to a `Symbol` with `_map_lyapunov_status_symbol` if a `Symbol` is needed.
Built from the same named code constants as `_map_lyapunov_classification` (`map_diagnostics.jl`), the
single source of truth for this code space (see the constants block above
`_MAP_LYAPUNOV_STATUS_CODE_BY_SYMBOL`).
"""
@inline function _lyapunov_point_classification(exponent::Float64, estimation_status_code::Int, neutral_tolerance::Float64)
    estimation_status_code == _LYAPUNOV_ESTIMATION_STATUS_COLLAPSED && return _LYAPUNOV_STATUS_PERIODIC
    estimation_status_code == _LYAPUNOV_ESTIMATION_STATUS_OK || return _LYAPUNOV_STATUS_UNRESOLVED
    isfinite(exponent) || return _LYAPUNOV_STATUS_UNRESOLVED
    exponent > neutral_tolerance && return _LYAPUNOV_STATUS_CHAOTIC_CANDIDATE
    abs(exponent) <= neutral_tolerance && return _LYAPUNOV_STATUS_QUASIPERIODIC_NEUTRAL_CANDIDATE
    return _LYAPUNOV_STATUS_PERIODIC
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
        classification_code = _lyapunov_point_classification(exponents[idx], _map_lyapunov_estimation_status_code(estimate.estimation_status), config.neutral_tolerance)
        classifications[idx] = _map_lyapunov_status_symbol(classification_code)
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
        classification_code = _lyapunov_point_classification(exponents[idx], _map_lyapunov_estimation_status_code(estimate.estimation_status), config.neutral_tolerance)
        classifications[idx] = _map_lyapunov_status_symbol(classification_code)
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

For `sys::DiscreteMap`, `backend` optionally runs the (always cell-independent) sweep on a GPU; see
[`ComputeBackend`](@ref). The result's `compute_backend` field records what actually ran.

For `sys::ContinuousODE`, the default method (`:auto`) is the **variational** estimator: each cell
integrates one augmented trajectory through the first variational equation and applies the return-time
correction at each Poincaré section return, normalizing the accumulated log-stretch by physical flow
time. The result is the largest transverse Lyapunov exponent, directly comparable to
`lyapunov_spectrum`'s leading transverse exponent. Pass `lyapunov_method=:two_trajectory` in
`config` (via [`BifurcationMapConfig`](@ref)) to use the legacy cheap-screen estimator instead.
GPU acceleration of the variational path is available for built-in systems that provide an
out-of-place `SVector` RHS and a constant section normal; `AutoBackend()` falls back to CPU when
these are absent (e.g. imported user systems). The explicit `:two_trajectory` method is always CPU-only.
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
    return nothing
end

function _record_direct_field_lyapunov!(storage,
                                        i::Int,
                                        j::Int,
                                        estimate,
                                        neutral_tolerance::Float64)
    exponent = Float64(estimate.exponent)
    storage.exponents[i, j] = exponent
    storage.status_codes[i, j] = _lyapunov_point_classification(exponent, _map_lyapunov_estimation_status_code(estimate.estimation_status), neutral_tolerance)
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
                        cells::Union{Nothing, LyapunovCellGrid}=nothing,
                        backend::ComputeBackend=CPUBackend())
    _validate_direct_lyapunov_field_config(config)
    config.lyapunov_method in (:auto, :two_trajectory) || throw(ArgumentError(
        "lyapunov_method=:variational is only available for ContinuousODE; use :auto or :two_trajectory for DiscreteMap."))
    length(sys.param_names) >= 2 || throw(ArgumentError("lyapunov_field requires a system with at least two parameters."))

    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    x0 = isnothing(initial_point) ? zeros(SVector{sys.dim, Float64}) : SVector{sys.dim}(initial_point)
    storage = _lyapunov_field_storage(cells, length(a_vals), length(b_vals))
    param_template = map_param_template(config)
    a_indices = map_a_write_indices(config)
    b_indices = map_b_write_indices(config)
    transient = _map_lyapunov_transient(config)
    iterations = _map_lyapunov_iterations(config)

    lyapunov_gpu_eligible = isempty(config.a_linked_param_indices) && isempty(config.b_linked_param_indices)
    ka_backend, compute_backend_used = _resolve_gpu_backend(
        backend, lyapunov_gpu_eligible, "lyapunov_field",
        "no linked parameter indices (a_linked_param_indices / b_linked_param_indices)"
    )

    if ka_backend !== nothing
        na, nb = length(a_vals), length(b_vals)
        a_vals_dev = _gpu_upload(ka_backend, a_vals)
        b_vals_dev = _gpu_upload(ka_backend, b_vals)
        template_sv = SVector{length(param_template), Float64}(param_template)
        a_index = only(a_indices)
        b_index = only(b_indices)
        _gpu_run_2d_sweep!(
            ka_backend, na, nb,
            (storage.exponents, storage.status_codes, storage.estimation_status_codes, storage.sample_counts),
            _lyapunov_field_gpu_kernel!, cells,
            a_vals_dev, b_vals_dev,
            sys.f, template_sv, a_index, b_index, x0, transient, iterations,
            config.lyapunov_perturbation, config.divergence_cutoff, config.lyapunov_neutral_tolerance
        )
    else
        chunks = _balanced_index_chunks(length(a_vals) * length(b_vals), Threads.nthreads())
        Threads.@threads for chunk_idx in eachindex(chunks)
            param_buffer = copy(param_template)
            for idx in chunks[chunk_idx]
                i = ((idx - 1) % length(a_vals)) + 1
                j = ((idx - 1) ÷ length(a_vals)) + 1
                (cells !== nothing && cells.known[i, j]) && continue   # cache hook: skip pre-seeded cells
                params = map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
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
        now();
        compute_backend=compute_backend_used,
        lyapunov_method=:two_trajectory,
        normalization=:per_iteration
    )
end

function lyapunov_field(sys::ContinuousODE, config::BifurcationMapConfig;
                        initial_point::Union{Nothing, AbstractVector}=nothing,
                        solver=Tsit5(),
                        reltol::Float64=1e-8,
                        abstol::Float64=1e-8,
                        cells::Union{Nothing, LyapunovCellGrid}=nothing,
                        backend::ComputeBackend=CPUBackend())
    _validate_direct_lyapunov_field_config(config)
    length(sys.param_names) >= 2 || throw(ArgumentError("lyapunov_field requires a system with at least two parameters."))
    if !isempty(sys.section.constant_normal)
        length(sys.section.constant_normal) == sys.dim || throw(ArgumentError(
            "PoincareSection.constant_normal has length $(length(sys.section.constant_normal)); expected state dimension $(sys.dim)."))
        all(isfinite, sys.section.constant_normal) || throw(ArgumentError(
            "PoincareSection.constant_normal must contain only finite values."))
    end

    # Resolve effective method: :auto → :variational for ContinuousODE.
    method = config.lyapunov_method === :auto ? :variational : config.lyapunov_method
    method in (:variational, :two_trajectory) || throw(ArgumentError(
        "Unknown lyapunov_method $(repr(method)) for ContinuousODE; use :auto, :variational, or :two_trajectory."))

    # GPU is only available for the variational path (independent per-cell augmented trajectories).
    # The two-trajectory method is coupled and always runs on the CPU.
    if method === :two_trajectory
        isa(backend, GPUBackend) && throw(ArgumentError(
            "lyapunov_field with lyapunov_method=:two_trajectory is CPU-only for ContinuousODE. " *
            "Use backend=CPUBackend() or AutoBackend(), or switch to lyapunov_method=:variational for GPU support."))
        return _lyapunov_field_continuous_two_trajectory(sys, config, initial_point, solver, reltol, abstol, cells)
    end

    # Variational path: check GPU eligibility and dispatch.
    ka_backend, vendor = _resolve_variational_lyapunov_gpu_backend(backend, sys, "lyapunov_field (variational)")
    if ka_backend !== nothing
        return _lyapunov_field_continuous_variational_gpu(
            sys, config, initial_point, solver, reltol, abstol, cells, ka_backend, vendor)
    end
    return _lyapunov_field_continuous_variational_cpu(sys, config, initial_point, solver, reltol, abstol, cells)
end

function _lyapunov_field_continuous_two_trajectory(sys::ContinuousODE, config::BifurcationMapConfig,
                                                    initial_point, solver, reltol, abstol, cells)
    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    u0 = _resolve_initial_state(sys, initial_point)
    storage = _lyapunov_field_storage(cells, length(a_vals), length(b_vals))
    param_template = map_param_template(config)
    a_indices = map_a_write_indices(config)
    b_indices = map_b_write_indices(config)
    transient = _map_lyapunov_transient(config)
    iterations = _map_lyapunov_iterations(config)

    chunks = _balanced_index_chunks(length(a_vals) * length(b_vals), Threads.nthreads())
    Threads.@threads for chunk_idx in eachindex(chunks)
        param_buffer = copy(param_template)
        for idx in chunks[chunk_idx]
            i = ((idx - 1) % length(a_vals)) + 1
            j = ((idx - 1) ÷ length(a_vals)) + 1
            (cells !== nothing && cells.known[i, j]) && continue
            params = map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
            estimate = _estimate_continuous_poincare_largest_lyapunov(
                sys, params, u0, transient, iterations,
                config.lyapunov_perturbation, config.divergence_cutoff;
                solver=solver, reltol=reltol, abstol=abstol,
                min_crossing_time=config.min_crossing_time)
            _record_direct_field_lyapunov!(storage, i, j, estimate, config.lyapunov_neutral_tolerance)
            cells !== nothing && (cells.known[i, j] = true)
        end
    end
    return LyapunovFieldResult(
        a_vals, b_vals,
        Float64.(storage.exponents), Int.(storage.status_codes),
        Int.(storage.estimation_status_codes), Int.(storage.sample_counts),
        config.lyapunov_neutral_tolerance, sys.name,
        (sys.param_names[config.a_index], sys.param_names[config.b_index]),
        now(); compute_backend=:cpu, lyapunov_method=:two_trajectory, normalization=:per_return)
end

function _lyapunov_field_continuous_variational_cpu(sys::ContinuousODE, config::BifurcationMapConfig,
                                                     initial_point, solver, reltol, abstol, cells)
    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    u0 = _resolve_initial_state(sys, initial_point)
    storage = _lyapunov_field_storage(cells, length(a_vals), length(b_vals))
    param_template = map_param_template(config)
    a_indices = map_a_write_indices(config)
    b_indices = map_b_write_indices(config)
    transient = _map_lyapunov_transient(config)
    iterations = _map_lyapunov_iterations(config)

    chunks = _balanced_index_chunks(length(a_vals) * length(b_vals), Threads.nthreads())
    Threads.@threads for chunk_idx in eachindex(chunks)
        param_buffer = copy(param_template)
        for idx in chunks[chunk_idx]
            i = ((idx - 1) % length(a_vals)) + 1
            j = ((idx - 1) ÷ length(a_vals)) + 1
            (cells !== nothing && cells.known[i, j]) && continue
            params = map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
            estimate = _estimate_variational_continuous_lyapunov(
                sys, params, u0, transient, iterations, config.divergence_cutoff;
                solver=solver, reltol=reltol, abstol=abstol,
                min_crossing_time=config.min_crossing_time)
            _record_direct_field_lyapunov!(storage, i, j, estimate, config.lyapunov_neutral_tolerance)
            cells !== nothing && (cells.known[i, j] = true)
        end
    end
    return LyapunovFieldResult(
        a_vals, b_vals,
        Float64.(storage.exponents), Int.(storage.status_codes),
        Int.(storage.estimation_status_codes), Int.(storage.sample_counts),
        config.lyapunov_neutral_tolerance, sys.name,
        (sys.param_names[config.a_index], sys.param_names[config.b_index]),
        now(); compute_backend=:cpu, lyapunov_method=:variational, normalization=:flow_time)
end

function _lyapunov_field_continuous_variational_gpu(sys::ContinuousODE, config::BifurcationMapConfig,
                                                     initial_point, solver, reltol, abstol, cells,
                                                     ka_backend, vendor::Symbol)
    a_vals = collect(range(config.a_min, config.a_max, length=config.a_steps + 1))
    b_vals = collect(range(config.b_min, config.b_max, length=config.b_steps + 1))
    u0 = _resolve_initial_state(sys, initial_point)
    storage = _lyapunov_field_storage(cells, length(a_vals), length(b_vals))
    param_template = map_param_template(config)
    a_indices = map_a_write_indices(config)
    b_indices = map_b_write_indices(config)
    transient = _map_lyapunov_transient(config)
    iterations = _map_lyapunov_iterations(config)

    n_raw = collect(Float64, sys.section.constant_normal)
    nn    = dot(n_raw, n_raw)
    nn > 0 || throw(ArgumentError(
        "lyapunov_field (variational GPU): sys.section.constant_normal is a zero vector."))
    dim = sys.dim

    # Compute a canonical initial tangent vector in the section's tangent plane.
    v0 = zeros(Float64, dim)
    for k in 1:dim
        v0[k] = 1.0
        nv = dot(n_raw, v0)
        v0 .-= (nv / nn) .* n_raw
        vn = norm(v0)
        if vn > sqrt(eps(Float64))
            v0 ./= vn
            break
        end
        fill!(v0, 0.0)
    end
    all(iszero, v0) && throw(ArgumentError(
        "lyapunov_field (variational GPU): could not find a tangent vector orthogonal to the section normal."))

    # Collect uncached cells, solve ensemble on GPU, write back results.
    na = length(a_vals); nb = length(b_vals)
    cell_indices = Vector{Tuple{Int,Int}}()
    for j in 1:nb, i in 1:na
        (cells !== nothing && cells.known[i, j]) && continue
        push!(cell_indices, (i, j))
    end

    if !isempty(cell_indices)
        n_cells = length(cell_indices)
        u0_list = Vector{Vector{Float64}}(undef, n_cells)
        p_list  = Vector{Vector{Float64}}(undef, n_cells)
        v0_list = Vector{Vector{Float64}}(undef, n_cells)
        for (k, (i, j)) in enumerate(cell_indices)
            param_buffer = copy(param_template)
            p_list[k]  = map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a_vals[i], b_vals[j])
            u0_list[k] = collect(Float64, u0)
            v0_list[k] = copy(v0)
        end

        gpu_results = _variational_lyapunov_gpu_sweep(sys, u0_list, p_list, n_raw, v0_list, ka_backend;
            transient=transient, steps=iterations,
            reltol=reltol, abstol=abstol,
            min_crossing_time=config.min_crossing_time,
            divergence_cutoff=config.divergence_cutoff)

        for (k, (i, j)) in enumerate(cell_indices)
            r = gpu_results[k]
            est = _lyapunov_estimate_result(Float64(r.exponent), r.estimation_status, r.sample_count)
            _record_direct_field_lyapunov!(storage, i, j, est, config.lyapunov_neutral_tolerance)
            cells !== nothing && (cells.known[i, j] = true)
        end
    end

    return LyapunovFieldResult(
        a_vals, b_vals,
        Float64.(storage.exponents), Int.(storage.status_codes),
        Int.(storage.estimation_status_codes), Int.(storage.sample_counts),
        config.lyapunov_neutral_tolerance, sys.name,
        (sys.param_names[config.a_index], sys.param_names[config.b_index]),
        now(); compute_backend=vendor, lyapunov_method=:variational, normalization=:flow_time)
end

lyapunov_field(result::LyapunovFieldResult) = result

function lyapunov_field(result::BifurcationMapResult)
    isnothing(result.lyapunov) && throw(ArgumentError(
        "BifurcationMapResult for $(result.system_name) does not carry a Lyapunov field. " *
        "Re-run bifurcation_map with BifurcationMapConfig(lyapunov_enabled=true, ...)."
    ))
    return result.lyapunov
end
