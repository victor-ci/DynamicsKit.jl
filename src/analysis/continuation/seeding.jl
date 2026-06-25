function _collect_distinct_period_branches(candidate_sets,
                                          max_branches_per_period::Int,
                                          signature_param_tol::Float64,
                                          signature_state_tol::Float64)
    accepted = BranchResult[]
    accepted_signatures = NamedTuple{(:period, :param_min, :param_max, :sample_states), Tuple{Int, Float64, Float64, Vector{Float64}}}[]
    found_for_period = 0

    for candidate_set in candidate_sets
        for branch in candidate_set
            signature = _branch_signature(branch)
            if !_is_duplicate_signature(signature, accepted_signatures, signature_param_tol, signature_state_tol)
                push!(accepted, branch)
                push!(accepted_signatures, signature)
                found_for_period += 1
                found_for_period >= max_branches_per_period && break
            end
        end

        found_for_period >= max_branches_per_period && break
    end

    return accepted
end

"""Collect candidate branches for one period across all skeleton parameters."""
function _continuation_period_candidates(sys::ContinuousODE, config::ContinuationConfig, period::Int,
                                         skeleton_params::AbstractVector{<:Real},
                                         base_params::Vector{Float64};
                                         initial_point::Union{Nothing, AbstractVector}=nothing,
                                         record::Union{Nothing, Function}=nothing,
                                         search_min::Union{Nothing, AbstractVector}=nothing,
                                         search_max::Union{Nothing, AbstractVector}=nothing,
                                         n_initial::Int=12,
                                         trajectory_seed_points::Bool=true,
                                         trajectory_seed_crossings::Int=0,
                                         trajectory_seed_transient::Int=0,
                                         reuse_neighbor_seeds::Bool=false,
                                         tol::Float64=1e-8,
                                         max_iter::Int=40,
                                         fd_step::Float64=1e-6,
                                         solver=Tsit5(),
                                         reltol::Float64=1e-8,
                                         abstol::Float64=1e-8,
                                         tmax::Union{Nothing, Float64}=nothing,
                                         min_crossing_time::Float64=1e-6,
                                         threaded::Bool=Threads.nthreads() > 1,
                                         threaded_skeleton::Bool=!threaded,
                                         threaded_branches::Union{Nothing, Bool}=nothing,
                                         skeleton_cache_resolver::Union{Nothing, Function}=nothing,
                                         on_error::Union{Nothing, Function}=nothing)
    branch_threaded = isnothing(threaded_branches) ? (threaded && (reuse_neighbor_seeds || length(skeleton_params) <= 1)) : threaded_branches

    if reuse_neighbor_seeds || !threaded || Threads.nthreads() == 1 || length(skeleton_params) <= 1
        if reuse_neighbor_seeds
            return _continuation_period_candidates_with_seed_reuse(
                sys,
                config,
                period,
                skeleton_params,
                base_params;
                initial_point=initial_point,
                record=record,
                search_min=search_min,
                search_max=search_max,
                n_initial=n_initial,
                trajectory_seed_points=trajectory_seed_points,
                trajectory_seed_crossings=trajectory_seed_crossings,
                trajectory_seed_transient=trajectory_seed_transient,
                tol=tol,
                max_iter=max_iter,
                fd_step=fd_step,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time,
                threaded_skeleton=threaded_skeleton,
                threaded_branches=branch_threaded,
                skeleton_cache_resolver=skeleton_cache_resolver,
                on_error=on_error
            )
        end

        return [
            _branches_for_skeleton_param(
                sys,
                config,
                period,
                float(skeleton_param),
                base_params;
                initial_point=initial_point,
                record=record,
                search_min=search_min,
                search_max=search_max,
                n_initial=n_initial,
                trajectory_seed_points=trajectory_seed_points,
                trajectory_seed_crossings=trajectory_seed_crossings,
                trajectory_seed_transient=trajectory_seed_transient,
                tol=tol,
                max_iter=max_iter,
                fd_step=fd_step,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time,
                threaded_skeleton=threaded_skeleton,
                threaded_branches=branch_threaded,
                skeleton_cache_resolver=skeleton_cache_resolver,
                on_error=on_error
            )
            for skeleton_param in skeleton_params
        ]
    end

    tasks = map(skeleton_params) do skeleton_param
        Threads.@spawn begin
            try
                _branches_for_skeleton_param(
                    sys,
                    config,
                    period,
                    float(skeleton_param),
                    base_params;
                    initial_point=initial_point,
                    record=record,
                    search_min=search_min,
                    search_max=search_max,
                    n_initial=n_initial,
                    trajectory_seed_points=trajectory_seed_points,
                    trajectory_seed_crossings=trajectory_seed_crossings,
                    trajectory_seed_transient=trajectory_seed_transient,
                    tol=tol,
                    max_iter=max_iter,
                    fd_step=fd_step,
                    solver=solver,
                    reltol=reltol,
                    abstol=abstol,
                    tmax=tmax,
                    min_crossing_time=min_crossing_time,
                    threaded_skeleton=threaded_skeleton,
                    threaded_branches=false,
                    skeleton_cache_resolver=skeleton_cache_resolver,
                    on_error=on_error
                )
            catch err
                _report_continuation_error(on_error, "Skeleton parameter $(float(skeleton_param)) for period $period failed", err)
                BranchResult[]
            end
        end
    end

    return fetch.(tasks)
end

"""Collect candidate branches serially while reusing successful seeds from nearby skeleton parameters."""
function _continuation_period_candidates_with_seed_reuse(sys::ContinuousODE,
                                                         config::ContinuationConfig,
                                                         period::Int,
                                                         skeleton_params::AbstractVector{<:Real},
                                                         base_params::Vector{Float64};
                                                         initial_point::Union{Nothing, AbstractVector}=nothing,
                                                         record::Union{Nothing, Function}=nothing,
                                                         search_min::Union{Nothing, AbstractVector}=nothing,
                                                         search_max::Union{Nothing, AbstractVector}=nothing,
                                                         n_initial::Int=12,
                                                         trajectory_seed_points::Bool=true,
                                                         trajectory_seed_crossings::Int=0,
                                                         trajectory_seed_transient::Int=0,
                                                         tol::Float64=1e-8,
                                                         max_iter::Int=40,
                                                         fd_step::Float64=1e-6,
                                                         solver=Tsit5(),
                                                         reltol::Float64=1e-8,
                                                         abstol::Float64=1e-8,
                                                         tmax::Union{Nothing, Float64}=nothing,
                                                         min_crossing_time::Float64=1e-6,
                                                         threaded_skeleton::Bool=false,
                                                         threaded_branches::Bool=false,
                                                         skeleton_cache_resolver::Union{Nothing, Function}=nothing,
                                                         on_error::Union{Nothing, Function}=nothing)
    seed_cache = NamedTuple{(:param, :point, :stamp), Tuple{Float64, Vector{Float64}, Int}}[]
    candidate_sets = Vector{Vector{BranchResult}}()
    max_cached_points = max(8, 2 * n_initial)

    for skeleton_param in skeleton_params
        skeleton_value = float(skeleton_param)
        search = _branch_search_for_skeleton_param(
            sys,
            config,
            period,
            skeleton_value,
            base_params;
            initial_point=initial_point,
            record=record,
            search_min=search_min,
            search_max=search_max,
            n_initial=n_initial,
            trajectory_seed_points=trajectory_seed_points,
            trajectory_seed_crossings=trajectory_seed_crossings,
            trajectory_seed_transient=trajectory_seed_transient,
            extra_seed_points=_cached_neighbor_seed_points(seed_cache, skeleton_value; max_points=max_cached_points),
            tol=tol,
            max_iter=max_iter,
            fd_step=fd_step,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time,
            threaded_skeleton=threaded_skeleton,
            threaded_branches=threaded_branches,
            skeleton_cache_resolver=skeleton_cache_resolver,
            on_error=on_error
        )
        push!(candidate_sets, search.branches)
        _update_neighbor_seed_cache!(seed_cache, skeleton_value, search.seed_points; max_entries=max_cached_points)
    end

    return candidate_sets
end

"""Return cached seeds near the requested skeleton parameter, prioritizing nearby and recently found points."""
function _cached_neighbor_seed_points(seed_cache::Vector{<:NamedTuple}, skeleton_param::Float64;
                                      max_points::Int)
    isempty(seed_cache) && return nothing
    order = sortperm(eachindex(seed_cache); by=idx -> (abs(seed_cache[idx].param - skeleton_param), -seed_cache[idx].stamp))
    keep = first(order, min(max_points, length(order)))
    [copy(seed_cache[idx].point) for idx in keep]
end

"""Append freshly converged skeleton seeds to the neighbor-seed cache while capping its size."""
function _update_neighbor_seed_cache!(seed_cache::Vector{NamedTuple{(:param, :point, :stamp), Tuple{Float64, Vector{Float64}, Int}}},
                                      skeleton_param::Float64,
                                      seed_points::AbstractVector;
                                      max_entries::Int)
    next_stamp = isempty(seed_cache) ? 1 : seed_cache[end].stamp + 1
    for point in seed_points
        push!(seed_cache, (param=skeleton_param, point=copy(point), stamp=next_stamp))
        next_stamp += 1
    end

    excess = length(seed_cache) - max_entries
    excess > 0 && deleteat!(seed_cache, 1:excess)
    return seed_cache
end

"""Merge explicit neighbor seeds with trajectory samples, keeping warm starts first."""
function _merge_continuation_seed_points(extra_seed_points::Union{Nothing, AbstractVector},
                                         sampled_seed_points::Union{Nothing, AbstractVector})
    if isnothing(extra_seed_points)
        return isnothing(sampled_seed_points) ? nothing : [collect(Float64, point) for point in sampled_seed_points]
    elseif isnothing(sampled_seed_points)
        return [collect(Float64, point) for point in extra_seed_points]
    end

    vcat(
        [collect(Float64, point) for point in extra_seed_points],
        [collect(Float64, point) for point in sampled_seed_points]
    )
end

"""Run skeleton search plus continuation at one parameter value and return both branches and reusable seeds."""
function _branch_search_for_skeleton_param(sys::ContinuousODE, config::ContinuationConfig, period::Int,
                                           skeleton_param::Float64,
                                           base_params::Vector{Float64};
                                           initial_point::Union{Nothing, AbstractVector}=nothing,
                                           record::Union{Nothing, Function}=nothing,
                                           search_min::Union{Nothing, AbstractVector}=nothing,
                                           search_max::Union{Nothing, AbstractVector}=nothing,
                                           n_initial::Int=12,
                                           trajectory_seed_points::Bool=true,
                                           trajectory_seed_crossings::Int=0,
                                           trajectory_seed_transient::Int=0,
                                           extra_seed_points::Union{Nothing, AbstractVector}=nothing,
                                           tol::Float64=1e-8,
                                           max_iter::Int=40,
                                           fd_step::Float64=1e-6,
                                           solver=Tsit5(),
                                           reltol::Float64=1e-8,
                                           abstol::Float64=1e-8,
                                           tmax::Union{Nothing, Float64}=nothing,
                                           min_crossing_time::Float64=1e-6,
                                           threaded_skeleton::Bool=false,
                                           threaded_branches::Bool=false,
                                           skeleton_cache_resolver::Union{Nothing, Function}=nothing,
                                           on_error::Union{Nothing, Function}=nothing,
                                           reseed::ReseedConfig=ReseedConfig(),
                                           on_reseed::Union{Nothing, Function}=nothing)
    local_params = _inject_param(base_params, config.param_index, skeleton_param, config.linked_param_indices)
    sampled_seed_points = trajectory_seed_points ? _collect_trajectory_seed_points(
        sys,
        skeleton_param,
        base_params,
        config.param_index;
        initial_point=initial_point,
        crossings=trajectory_seed_crossings > 0 ? trajectory_seed_crossings : max(16, 3 * n_initial),
        transient=trajectory_seed_transient > 0 ? trajectory_seed_transient : max(8, n_initial),
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    ) : nothing
    seed_points = _merge_continuation_seed_points(extra_seed_points, sampled_seed_points)
    cache_info = isnothing(skeleton_cache_resolver) ? Dict{String, Any}() : something(skeleton_cache_resolver(
        period,
        skeleton_param;
        search_min=search_min,
        search_max=search_max,
        seed_points=seed_points,
        initial_point=initial_point,
        n_initial=n_initial,
        tol=tol,
        max_iter=max_iter,
        fd_step=fd_step,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time,
        threaded=threaded_skeleton
    ), Dict{String, Any}())
    seeds = find_periodic_skeleton(
        sys,
        [period],
        skeleton_param;
        n_initial=n_initial,
        search_min=search_min,
        search_max=search_max,
        seed_points=seed_points,
        params=local_params,
        param_index=config.param_index,
        tol=tol,
        max_iter=max_iter,
        fd_step=fd_step,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time,
        threaded=threaded_skeleton,
        cache_file=get(cache_info, "file", nothing),
        cache_metadata=get(cache_info, "metadata", Dict{String, Any}()),
        cache_enabled=get(cache_info, "enabled", true)
    )

    branches = if threaded_branches && Threads.nthreads() > 1 && length(seeds) > 1
        tasks = map(seeds) do seed
            Threads.@spawn _continue_seed_branch(
                sys,
                config,
                period,
                seed.point,
                local_params;
                record=record,
                search_min=search_min,
                search_max=search_max,
                n_initial=n_initial,
                tol=tol,
                max_iter=max_iter,
                fd_step=fd_step,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time,
                on_error=on_error,
                reseed=reseed,
                on_reseed=on_reseed,
                context="Period $period seed continuation from skeleton parameter $(round(skeleton_param, digits=6))"
            )
        end
        filter(!isnothing, fetch.(tasks))
    else
        filter(!isnothing, [
            _continue_seed_branch(
                sys,
                config,
                period,
                seed.point,
                local_params;
                record=record,
                search_min=search_min,
                search_max=search_max,
                n_initial=n_initial,
                tol=tol,
                max_iter=max_iter,
                fd_step=fd_step,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time,
                on_error=on_error,
                reseed=reseed,
                on_reseed=on_reseed,
                context="Period $period seed continuation from skeleton parameter $(round(skeleton_param, digits=6))"
            )
            for seed in seeds
        ])
    end

    return (branches=branches, seed_points=[copy(seed.point) for seed in seeds])
end

function _continue_seed_branch(sys::ContinuousODE, config::ContinuationConfig, period::Int,
                               seed_point::AbstractVector, local_params::Vector{Float64};
                               record::Union{Nothing, Function}=nothing,
                               search_min::Union{Nothing, AbstractVector}=nothing,
                               search_max::Union{Nothing, AbstractVector}=nothing,
                               n_initial::Int=12,
                               tol::Float64=1e-8,
                               max_iter::Int=40,
                               fd_step::Float64=1e-6,
                               solver=Tsit5(),
                               reltol::Float64=1e-8,
                               abstol::Float64=1e-8,
                               tmax::Union{Nothing, Float64}=nothing,
                               min_crossing_time::Float64=1e-6,
                               on_error::Union{Nothing, Function}=nothing,
                               reseed::ReseedConfig=ReseedConfig(),
                               on_reseed::Union{Nothing, Function}=nothing,
                               context::AbstractString="Continuous seed continuation")
    try
        continuation_branch(
            sys,
            config,
            period;
            initial_point=seed_point,
            params=local_params,
            record=record,
            on_error=on_error,
            search_min=search_min,
            search_max=search_max,
            n_initial=n_initial,
            tol=tol,
            max_iter=max_iter,
            fd_step=fd_step,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time,
            reseed=reseed,
            on_reseed=on_reseed
        )
    catch err
        _report_continuation_error(on_error, context, err)
        return nothing
    end
end

"""Generate continuation branches from the skeleton seeds at one parameter value."""
function _branches_for_skeleton_param(sys::ContinuousODE, config::ContinuationConfig, period::Int,
                                      skeleton_param::Float64,
                                      base_params::Vector{Float64};
                                      initial_point::Union{Nothing, AbstractVector}=nothing,
                                      record::Union{Nothing, Function}=nothing,
                                      search_min::Union{Nothing, AbstractVector}=nothing,
                                      search_max::Union{Nothing, AbstractVector}=nothing,
                                      n_initial::Int=12,
                                      trajectory_seed_points::Bool=true,
                                      trajectory_seed_crossings::Int=0,
                                      trajectory_seed_transient::Int=0,
                                      extra_seed_points::Union{Nothing, AbstractVector}=nothing,
                                      tol::Float64=1e-8,
                                      max_iter::Int=40,
                                      fd_step::Float64=1e-6,
                                      solver=Tsit5(),
                                      reltol::Float64=1e-8,
                                      abstol::Float64=1e-8,
                                      tmax::Union{Nothing, Float64}=nothing,
                                      min_crossing_time::Float64=1e-6,
                                      threaded_skeleton::Bool=false,
                                      threaded_branches::Bool=false,
                                       skeleton_cache_resolver::Union{Nothing, Function}=nothing,
                                      on_error::Union{Nothing, Function}=nothing)
    _branch_search_for_skeleton_param(
        sys,
        config,
        period,
        skeleton_param,
        base_params;
        initial_point=initial_point,
        record=record,
        search_min=search_min,
        search_max=search_max,
        n_initial=n_initial,
        trajectory_seed_points=trajectory_seed_points,
        trajectory_seed_crossings=trajectory_seed_crossings,
        trajectory_seed_transient=trajectory_seed_transient,
        extra_seed_points=extra_seed_points,
        tol=tol,
        max_iter=max_iter,
        fd_step=fd_step,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time,
        threaded_skeleton=threaded_skeleton,
        threaded_branches=threaded_branches,
        skeleton_cache_resolver=skeleton_cache_resolver,
        on_error=on_error
    ).branches
end

# `_inject_param` moved to analysis/parameter_mapping.jl (Contract A); available here via alias.

