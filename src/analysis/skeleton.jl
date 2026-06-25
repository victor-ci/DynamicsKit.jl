"""
Periodic skeleton generation: find all periodic orbits at a fixed parameter value.
"""

function _serialize_skeleton_result(item)
    Dict(
        "period" => item.period,
        "point" => collect(Float64, item.point),
        "multipliers_real" => Float64[real(value) for value in item.multipliers],
        "multipliers_imag" => Float64[imag(value) for value in item.multipliers],
        "stable" => Bool(item.stable)
    )
end

function _deserialize_skeleton_result(data)
    real_part = _as_float_vector(get(data, "multipliers_real", Float64[]), Float64[])
    imag_part = _as_float_vector(get(data, "multipliers_imag", Float64[]), Float64[])
    return (
        period=_as_int(get(data, "period", 1), 1),
        point=_as_float_vector(get(data, "point", Float64[]), Float64[]),
        multipliers=ComplexF64.(real_part, imag_part),
        stable=_as_bool(get(data, "stable", false), false)
    )
end

function _load_periodic_skeleton_cache(file_path::AbstractString)
    raw = JLD2.load(String(file_path), "results")
    return [_deserialize_skeleton_result(item) for item in raw]
end

function _save_periodic_skeleton_cache(file_path::AbstractString, results;
                                       metadata::AbstractDict{<:AbstractString, <:Any}=Dict{String, Any}())
    dir = dirname(String(file_path))
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    payload = [_serialize_skeleton_result(item) for item in results]
    jldsave(String(file_path); results=payload, metadata=Dict{String, Any}(String(k) => v for (k, v) in pairs(metadata)))
    return String(file_path)
end

"""
    find_periodic_skeleton(sys::DiscreteMap, periods::AbstractVector{Int},
                           param_value::Float64;
                           n_initial=15, search_bounds=(-3.0, 3.0),
                           params=[1.0], param_index=1,
                           tol=1e-10, max_iter=50) -> Vector{NamedTuple}

Find all distinct periodic orbits at a single parameter value by Newton iteration
from a grid of initial conditions.

Returns a vector of named tuples `(period, point, multipliers, stable)`.
"""
function find_periodic_skeleton(sys::DiscreteMap, periods::AbstractVector{Int},
                                param_value::Float64;
                                n_initial::Int=15,
                                search_min::Union{Nothing, AbstractVector}=nothing,
                                search_max::Union{Nothing, AbstractVector}=nothing,
                                seed_points::Union{Nothing, AbstractVector}=nothing,
                                params::Vector{Float64}=[1.0],
                                param_index::Int=1,
                                linked_param_indices::Vector{Int}=Int[],
                                tol::Float64=1e-10,
                                max_iter::Int=50,
                                threaded::Bool=Threads.nthreads() > 1,
                                cache_file::Union{Nothing, AbstractString}=nothing,
                                cache_metadata::AbstractDict{<:AbstractString, <:Any}=Dict{String, Any}(),
                                cache_enabled::Bool=true)
    if cache_enabled && !isnothing(cache_file) && isfile(String(cache_file))
        return _load_periodic_skeleton_cache(String(cache_file))
    end

    dim = sys.dim
    lo = isnothing(search_min) ? fill(-3.0, dim) : collect(search_min)
    hi = isnothing(search_max) ? fill(3.0, dim) : collect(search_max)

    pv = _inject_param(params, param_index, param_value, linked_param_indices)

    # Generate initial points from preferred seeds plus a fallback grid.
    initial_points = _discrete_skeleton_initial_points(lo, hi, n_initial; seed_points=seed_points)

    results = NamedTuple{(:period, :point, :multipliers, :stable), Tuple{Int, Vector{Float64}, Vector{ComplexF64}, Bool}}[]
    found_points = Vector{Float64}[]
    found_periods = Int[]
    map_step = x -> Array(sys.f(SVector{dim}(x), pv))

    for period in periods
        # Define F^period(x) - x = 0
        F = (x) -> begin
            _iterate_map(map_step, x, period) .- x
        end

        candidates = _collect_skeleton_candidates(
            initial_points,
            x0 -> _newton_ad(F, x0, tol, max_iter);
            threaded=threaded
        )

        for point in candidates

            # Reject Newton drift outside the user's declared search region.
            if !_within_search_box(point, lo, hi)
                continue
            end

            # Check uniqueness against already found points
            if !_is_different_regime(map_step, point, found_points, found_periods, tol * 100)
                continue
            end

            # Check true periodicity (not a sub-period fixed point)
            if !_is_true_period(map_step, point, period, tol)
                continue
            end

            # Compute multipliers (eigenvalues of Jacobian of F^period at fixed point)
            # F(x) = f^p(x) - x, so dF/dx = df^p/dx - I, meaning df^p/dx = J + I
            J = ForwardDiff.jacobian(F, point)
            map_jacobian = J + Matrix{Float64}(I, length(point), length(point))
            multipliers = eigvals(map_jacobian)
            stable = all(abs.(multipliers) .< 1.0)

            push!(results, (period=period, point=point, multipliers=multipliers, stable=stable))
            push!(found_points, point)
            push!(found_periods, period)
        end
    end

    if cache_enabled && !isnothing(cache_file)
        _save_periodic_skeleton_cache(String(cache_file), results; metadata=cache_metadata)
    end

    return results
end

"""Build discrete-time skeleton start points from seed hints plus a fallback state grid."""
function _discrete_skeleton_initial_points(lo::AbstractVector, hi::AbstractVector, n_initial::Int;
                                           seed_points::Union{Nothing, AbstractVector}=nothing)
    preferred = _prepare_seed_points(seed_points, lo, hi, n_initial)
    grid_n_initial = isempty(preferred) ? n_initial : max(3, cld(n_initial, 2))
    spacing = _seed_point_spacing(lo, hi, grid_n_initial) .* 0.5
    initial_points = copy(preferred)

    grids = [range(lo[d], hi[d], length=grid_n_initial) for d in eachindex(lo)]
    for point in Iterators.product(grids...)
        _push_unique_seed_point!(initial_points, Float64[point...], spacing)
    end

    return initial_points
end

"""
    find_periodic_skeleton(sys::ContinuousODE, periods::AbstractVector{Int}, param_value::Float64; kwargs...)

Find periodic orbits of a continuous-time system by applying Newton iteration to its projected
Poincaré return map.
"""
function find_periodic_skeleton(sys::ContinuousODE, periods::AbstractVector{Int},
                                param_value::Float64;
                                n_initial::Int=12,
                                search_min::Union{Nothing, AbstractVector}=nothing,
                                search_max::Union{Nothing, AbstractVector}=nothing,
                                seed_points::Union{Nothing, AbstractVector}=nothing,
                                params::Vector{Float64}=Float64[],
                                param_index::Int=1,
                                linked_param_indices::Vector{Int}=Int[],
                                tol::Float64=1e-8,
                                max_iter::Int=40,
                                fd_step::Float64=1e-6,
                                solver=Tsit5(),
                                reltol::Float64=1e-8,
                                abstol::Float64=1e-8,
                                tmax::Union{Nothing, Float64}=nothing,
                                min_crossing_time::Float64=1e-6,
                                threaded::Bool=Threads.nthreads() > 1,
                                cache_file::Union{Nothing, AbstractString}=nothing,
                                cache_metadata::AbstractDict{<:AbstractString, <:Any}=Dict{String, Any}(),
                                cache_enabled::Bool=true)
    if cache_enabled && !isnothing(cache_file) && isfile(String(cache_file))
        return _load_periodic_skeleton_cache(String(cache_file))
    end

    base_params = _resolve_continuous_params(sys, params)
    base_params = _inject_param(base_params, param_index, param_value, linked_param_indices)

    lo, hi = if isnothing(search_min) || isnothing(search_max)
        _estimate_section_bounds(
            sys,
            param_value,
            base_params,
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

    initial_points = _continuous_skeleton_initial_points(
        lo,
        hi,
        n_initial;
        seed_points=seed_points
    )

    results = NamedTuple{(:period, :point, :multipliers, :stable), Tuple{Int, Vector{Float64}, Vector{ComplexF64}, Bool}}[]
    found_points = Vector{Float64}[]
    found_periods = Int[]
    map_step = x -> begin
        next_point, found = _poincare_projected(
            sys,
            x,
            base_params;
            period=1,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
        found || error("Poincaré return failed during skeleton search")
        next_point
    end

    for period in periods
        F = x -> begin
            next_point, found = _poincare_projected(
                sys,
                x,
                base_params;
                period=period,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time
            )
            found || return fill(1e6, length(x))
            next_point .- x
        end

        candidates = _collect_skeleton_candidates(
            initial_points,
            x0 -> _newton_fd(F, x0, tol, max_iter, fd_step);
            threaded=threaded
        )

        for point in candidates

            if !_within_search_box(point, lo, hi)
                continue
            end

            if !_is_different_regime(map_step, point, found_points, found_periods, tol * 100)
                continue
            end

            if !_is_true_period(map_step, point, period, tol * 100)
                continue
            end

            J = _fd_jacobian(F, point, fd_step)
            map_jacobian = J + Matrix{Float64}(I, length(point), length(point))
            multipliers = eigvals(map_jacobian)
            stable = all(abs.(multipliers) .< 1.0)

            push!(results, (period=period, point=point, multipliers=multipliers, stable=stable))
            push!(found_points, point)
            push!(found_periods, period)
        end
    end

    if cache_enabled && !isnothing(cache_file)
        _save_periodic_skeleton_cache(String(cache_file), results; metadata=cache_metadata)
    end

    return results
end

"""Build continuous-time skeleton start points from sampled section points plus a fallback grid."""
function _continuous_skeleton_initial_points(lo::AbstractVector, hi::AbstractVector, n_initial::Int;
                                             seed_points::Union{Nothing, AbstractVector}=nothing)
    preferred = _prepare_seed_points(seed_points, lo, hi, n_initial)
    grid_n_initial = isempty(preferred) ? n_initial : max(4, cld(n_initial, 2))
    spacing = _seed_point_spacing(lo, hi, grid_n_initial) .* 0.5
    initial_points = copy(preferred)

    grids = [range(lo[d], hi[d], length=grid_n_initial) for d in eachindex(lo)]
    for point in Iterators.product(grids...)
        _push_unique_seed_point!(initial_points, Float64[point...], spacing)
    end

    return initial_points
end

"""Normalize externally supplied seed points, clamp them into the search box, and drop near-duplicates."""
function _prepare_seed_points(seed_points::Union{Nothing, AbstractVector},
                              lo::AbstractVector,
                              hi::AbstractVector,
                              n_initial::Int;
                              max_points::Int=max(8, 2 * n_initial))
    isnothing(seed_points) && return Vector{Vector{Float64}}()

    spacing = _seed_point_spacing(lo, hi, n_initial)
    prepared = Vector{Vector{Float64}}()
    for point in seed_points
        raw = collect(Float64, point)
        length(raw) == length(lo) || continue
        clamped = clamp.(raw, lo, hi)
        _push_unique_seed_point!(prepared, clamped, spacing) || continue
        length(prepared) >= max_points && break
    end

    return prepared
end

"""Characteristic spacing used to merge nearby seed points."""
function _seed_point_spacing(lo::AbstractVector, hi::AbstractVector, n_initial::Int)
    denom = max(n_initial - 1, 1)
    [max((hi[d] - lo[d]) / denom, sqrt(eps(Float64))) for d in eachindex(lo)]
end

"""Insert a seed point only when it is not already represented within the local spacing scale."""
function _push_unique_seed_point!(points::Vector{Vector{Float64}}, point::Vector{Float64}, spacing::AbstractVector)
    for existing in points
        if all(abs.(existing .- point) .<= spacing)
            return false
        end
    end

    push!(points, point)
    return true
end

"""Run independent Newton solves for skeleton seeds, optionally across Julia threads."""
function _collect_skeleton_candidates(initial_points, solver!::Function;
                                      threaded::Bool=Threads.nthreads() > 1)
    if !threaded || Threads.nthreads() == 1 || length(initial_points) <= 1
        candidates = Vector{Vector{Float64}}()
        for ip in initial_points
            point, converged = solver!(collect(Float64, ip))
            converged && push!(candidates, point)
        end
        return candidates
    end

    buckets = [Vector{Vector{Float64}}() for _ in 1:Threads.maxthreadid()]

    Threads.@threads for idx in eachindex(initial_points)
        point, converged = solver!(collect(Float64, initial_points[idx]))
        converged || continue
        push!(buckets[Threads.threadid()], point)
    end

    candidates = Vector{Vector{Float64}}()
    for bucket in buckets
        append!(candidates, bucket)
    end
    return candidates
end

"""
Scale-aware convergence threshold used by both Newton solvers. Returns
`max(tol, tol * norm(x))` so the absolute floor `tol` is used near the origin
and a relative `tol * norm(x)` is used for attractors far from the origin.
A purely absolute tolerance fails on large-amplitude attractors (genuine
convergence at `norm(F) = 1e-9` is reported as non-convergent when `tol = 1e-10`
but the relevant scale is `norm(x) ≈ 1e6`).
"""
_newton_convergence_threshold(x, tol) = max(tol, tol * norm(x))

"""Newton-Raphson with ForwardDiff Jacobian."""
function _newton_ad(F, x0, tol, max_iter)
    x = copy(x0)
    for _ in 1:max_iter
        Fx = F(x)
        if norm(Fx) < _newton_convergence_threshold(x, tol)
            return x, true
        end
        J = ForwardDiff.jacobian(F, x)
        cnd = cond(J)
        if !isfinite(cnd) || cnd > 1e15
            return x, false
        end
        x = x .- J \ Fx
    end
    return x, norm(F(x)) < _newton_convergence_threshold(x, tol)
end

"""Newton-Raphson with a finite-difference Jacobian."""
function _newton_fd(F, x0, tol, max_iter, fd_step)
    x = copy(x0)
    for _ in 1:max_iter
        Fx = F(x)
        if norm(Fx) < _newton_convergence_threshold(x, tol)
            return x, true
        end
        J = _fd_jacobian(F, x, fd_step)
        cnd = cond(J)
        if !isfinite(cnd) || cnd > 1e15
            return x, false
        end
        x = x .- J \ Fx
    end
    return x, norm(F(x)) < _newton_convergence_threshold(x, tol)
end

"""Check that a point is a true period-N orbit, not period-M where M divides N."""
function _is_true_period(map_step, point, period, tol)
    if period == 1
        return true
    end
    next_point = copy(point)
    for _ in 1:(period - 1)
        next_point = map_step(next_point)
        if norm(next_point .- point) < tol
            return false  # Returns to start before period steps — lower period
        end
    end
    return true
end

"""Iterate a map closure a fixed number of times."""
function _iterate_map(map_step, point, period)
    next_point = copy(point)
    for _ in 1:period
        next_point = map_step(next_point)
    end
    return next_point
end

"""
Return `true` if `point` lies within the user's search box (extended by `margin`
times the box width on each axis). Newton drift can converge a seed onto a
genuine but unrelated fixed point of the Poincaré map that lives far outside
the region the user actually wanted to explore (e.g. an escape orbit at large
positive z for Rössler). Such candidates are filtered out so the skeleton
result stays inside the declared region of interest.
"""
function _within_search_box(point::AbstractVector, lo::AbstractVector, hi::AbstractVector;
                            margin::Float64=0.25)
    length(point) == length(lo) == length(hi) || return false
    for i in eachindex(point)
        span = hi[i] - lo[i]
        slack = max(margin * abs(span), 1e-9)
        if point[i] < lo[i] - slack || point[i] > hi[i] + slack
            return false
        end
    end
    return true
end

"""Check if a point belongs to a regime already represented by stored orbit seeds."""
function _is_different_regime(map_step, point, found_points, found_periods, tol)
    for (stored_point, stored_period) in zip(found_points, found_periods)
        next_point = copy(stored_point)
        for _ in 1:stored_period
            next_point = map_step(next_point)
            if norm(next_point .- point) < tol
                return false
            end
        end
    end
    return true
end

"""Check if a point is distinct from all found points."""
function _is_unique(point, found_points, tol)
    for fp in found_points
        if norm(point .- fp) < tol
            return false
        end
    end
    return true
end

