"""Recover candidate branches for one discrete candidate window."""
function _recover_window_branches(sys::DiscreteMap,
                                  window::AtlasWindow,
                                  samples::Vector{AtlasReconSample},
                                  base_params::Vector{Float64},
                                  cont_config::ContinuationConfig,
                                  atlas_config::AtlasConfig;
                                  sample_idx::Union{Nothing, Int}=nothing,
                                  padding_scale::Float64=1.0,
                                  seed_cache::Union{Nothing, _AtlasSeedCache}=nothing,
                                  seed_reuse_events::Union{Nothing, Vector{Dict{String, Any}}}=nothing)
    fallback_dim = sys.dim
    seed_data = _extract_window_seed_data(window, samples, atlas_config, fallback_dim; sample_idx=sample_idx, padding_scale=padding_scale)
    cached_points, seed_lookup_diag = _atlas_cached_seed_points(seed_cache, window, seed_data.skeleton_param, cont_config, atlas_config)
    recovery_seed_points = _atlas_merge_recovery_seed_points(
        cached_points,
        seed_data.seed_points,
        seed_data.search_min,
        seed_data.search_max,
        atlas_config
    )
    local_params = inject_param(base_params, cont_config.param_index, seed_data.skeleton_param, cont_config.linked_param_indices)
    seeds = find_periodic_skeleton(
        sys,
        [window.period],
        seed_data.skeleton_param;
        n_initial=max(4, atlas_config.seed_points_per_window, length(recovery_seed_points)),
        search_min=seed_data.search_min,
        search_max=seed_data.search_max,
        seed_points=recovery_seed_points,
        params=local_params,
        param_index=cont_config.param_index,
        linked_param_indices=cont_config.linked_param_indices,
        tol=max(cont_config.newton_tol, 1e-8),
        max_iter=max(cont_config.newton_max_iter, 40),
        threaded=atlas_config.threaded
    )
    seed_store_diag = _atlas_update_seed_cache!(
        seed_cache,
        window.period,
        seed_data.skeleton_param,
        [seed.point for seed in seeds],
        atlas_config
    )
    seed_reuse_diag = _atlas_seed_reuse_event(seed_lookup_diag, seed_store_diag)
    isnothing(seed_reuse_events) || push!(seed_reuse_events, seed_reuse_diag)

    isempty(seeds) && return BranchResult[], :skeleton_failed, seed_data, Dict(
        "seedCount" => 0,
        "sampleIndex" => isnothing(sample_idx) ? nothing : sample_idx,
        "searchMin" => copy(seed_data.search_min),
        "searchMax" => copy(seed_data.search_max),
        "seedReuse" => seed_reuse_diag,
        "continuationAttempts" => Dict{String, Any}[]
    )

    branches = BranchResult[]
    continuation_attempts = Dict{String, Any}[]
    retry_configs = _atlas_continuation_retry_configs(cont_config, window.param_min, window.param_max, atlas_config.continuation_retry_budget)
    reseed_count = Ref(0)
    reseed_params = Float64[]
    reseed_lock = ReentrantLock()
    on_reseed = (bw, fw) -> lock(reseed_lock) do
        for diag in (bw, fw)
            isnothing(diag) && continue
            reseed_count[] += diag.attempt_count
            append!(reseed_params, diag.reseed_params)
        end
    end
    for seed in seeds
        seed_success = false
        for (retry_idx, retry_config) in enumerate(retry_configs)
            try
                push!(branches, continuation_branch(sys, retry_config, window.period;
                    initial_point=seed.point, params=local_params,
                    reseed=atlas_config.reseed, on_reseed=on_reseed))
                push!(continuation_attempts, Dict(
                    "retryIndex" => retry_idx,
                    "status" => "ok",
                    "seedPoint" => copy(seed.point),
                    "pMin" => retry_config.p_min,
                    "pMax" => retry_config.p_max,
                    "ds" => retry_config.ds
                ))
                seed_success = true
                break
            catch err
                err isa InterruptException && rethrow()
                push!(continuation_attempts, Dict(
                    "retryIndex" => retry_idx,
                    "status" => "failed",
                    "seedPoint" => copy(seed.point),
                    "pMin" => retry_config.p_min,
                    "pMax" => retry_config.p_max,
                    "ds" => retry_config.ds,
                    "error" => _continuation_error_message(err)
                ))
            end
        end
        seed_success || nothing
    end

    isempty(branches) && return BranchResult[], :continuation_failed, seed_data, Dict(
        "seedCount" => length(seeds),
        "sampleIndex" => isnothing(sample_idx) ? nothing : sample_idx,
        "searchMin" => copy(seed_data.search_min),
        "searchMax" => copy(seed_data.search_max),
        "continuationAttempts" => continuation_attempts,
        "seedReuse" => seed_reuse_diag,
        "reseedCount" => reseed_count[],
        "reseedParams" => copy(reseed_params)
    )

    branches, trim_diagnostics = _atlas_trim_recovered_branches(
        sys,
        branches,
        base_params,
        cont_config.linked_param_indices
    )
    isempty(branches) && return BranchResult[], :period_trimmed, seed_data, Dict(
        "seedCount" => length(seeds),
        "sampleIndex" => isnothing(sample_idx) ? nothing : sample_idx,
        "searchMin" => copy(seed_data.search_min),
        "searchMax" => copy(seed_data.search_max),
        "continuationAttempts" => continuation_attempts,
        "minimalPeriodTrim" => trim_diagnostics,
        "seedReuse" => seed_reuse_diag,
        "reseedCount" => reseed_count[],
        "reseedParams" => copy(reseed_params)
    )

    distinct = _collect_distinct_period_branches([branches], min(atlas_config.max_total_branches, length(branches)), 1e-2, 0.5)
    return distinct, :ok, seed_data, Dict(
        "seedCount" => length(seeds),
        "sampleIndex" => isnothing(sample_idx) ? nothing : sample_idx,
        "searchMin" => copy(seed_data.search_min),
        "searchMax" => copy(seed_data.search_max),
        "continuationAttempts" => continuation_attempts,
        "distinctBranchCount" => length(distinct),
        "minimalPeriodTrim" => trim_diagnostics,
        "seedReuse" => seed_reuse_diag,
        "reseedCount" => reseed_count[],
        "reseedParams" => copy(reseed_params)
    )
end

"""Recover candidate branches for one continuous candidate window."""
function _recover_window_branches(sys::ContinuousODE,
                                  window::AtlasWindow,
                                  samples::Vector{AtlasReconSample},
                                  base_params::Vector{Float64},
                                  cont_config::ContinuationConfig,
                                  atlas_config::AtlasConfig;
                                  sample_idx::Union{Nothing, Int}=nothing,
                                  padding_scale::Float64=1.0,
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8,
                                  abstol::Float64=1e-8,
                                  min_crossing_time::Float64=1e-6,
                                  seed_cache::Union{Nothing, _AtlasSeedCache}=nothing,
                                  seed_reuse_events::Union{Nothing, Vector{Dict{String, Any}}}=nothing)
    fallback_dim = state_dim(sys)
    seed_data = _extract_window_seed_data(window, samples, atlas_config, fallback_dim; sample_idx=sample_idx, padding_scale=padding_scale)
    cached_points, seed_lookup_diag = _atlas_cached_seed_points(seed_cache, window, seed_data.skeleton_param, cont_config, atlas_config)
    recovery_seed_points = _atlas_merge_recovery_seed_points(
        cached_points,
        seed_data.seed_points,
        seed_data.search_min,
        seed_data.search_max,
        atlas_config
    )
    branches = BranchResult[]
    continuation_attempts = Dict{String, Any}[]
    retry_configs = _atlas_continuation_retry_configs(cont_config, window.param_min, window.param_max, atlas_config.continuation_retry_budget)
    total_seed_count = 0
    last_seed_store_diag = Dict{String, Any}("requested" => atlas_config.reuse_neighbor_seeds, "storedSeedCount" => 0, "cacheSize" => length(get(isnothing(seed_cache) ? _atlas_seed_cache() : seed_cache, window.period, _AtlasSeedEntry[])))
    reseed_count = Ref(0)
    reseed_params = Float64[]
    reseed_lock = ReentrantLock()
    on_reseed = (bw, fw) -> lock(reseed_lock) do
        for diag in (bw, fw)
            isnothing(diag) && continue
            reseed_count[] += diag.attempt_count
            append!(reseed_params, diag.reseed_params)
        end
    end
    for (retry_idx, retry_config) in enumerate(retry_configs)
        retry_errors = String[]
        retry_error_lock = ReentrantLock()
        search = _branch_search_for_skeleton_param(
            sys,
            retry_config,
            window.period,
            seed_data.skeleton_param,
            base_params;
            search_min=seed_data.search_min,
            search_max=seed_data.search_max,
            n_initial=max(4, atlas_config.seed_points_per_window, length(recovery_seed_points)),
            extra_seed_points=recovery_seed_points,
            tol=max(retry_config.newton_tol, 1e-6),
            max_iter=max(retry_config.newton_max_iter, 30),
            fd_step=1e-6,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
            threaded_skeleton=atlas_config.threaded,
            threaded_branches=atlas_config.threaded,
            reseed=atlas_config.reseed,
            on_reseed=on_reseed,
            on_error=message -> lock(retry_error_lock) do
                push!(retry_errors, String(message))
            end
        )
        total_seed_count = max(total_seed_count, length(search.seed_points))
        last_seed_store_diag = _atlas_update_seed_cache!(
            seed_cache,
            window.period,
            seed_data.skeleton_param,
            search.seed_points,
            atlas_config
        )
        append!(branches, search.branches)
        push!(continuation_attempts, Dict(
            "retryIndex" => retry_idx,
            "status" => isempty(search.branches) ? (isempty(search.seed_points) ? "skeleton_failed" : "continuation_failed") : "ok",
            "seedCount" => length(search.seed_points),
            "seedParam" => seed_data.skeleton_param,
            "pMin" => retry_config.p_min,
            "pMax" => retry_config.p_max,
            "ds" => retry_config.ds,
            "searchMin" => copy(seed_data.search_min),
            "searchMax" => copy(seed_data.search_max),
            "errors" => copy(retry_errors)
        ))
        isempty(search.branches) || break
    end
    seed_reuse_diag = _atlas_seed_reuse_event(seed_lookup_diag, last_seed_store_diag)
    isnothing(seed_reuse_events) || push!(seed_reuse_events, seed_reuse_diag)

    isempty(branches) && return BranchResult[], :continuation_failed, seed_data, Dict(
        "seedCount" => total_seed_count,
        "sampleIndex" => isnothing(sample_idx) ? nothing : sample_idx,
        "searchMin" => copy(seed_data.search_min),
        "searchMax" => copy(seed_data.search_max),
        "continuationAttempts" => continuation_attempts,
        "seedReuse" => seed_reuse_diag,
        "reseedCount" => reseed_count[],
        "reseedParams" => copy(reseed_params)
    )

    branches, trim_diagnostics = _atlas_trim_recovered_branches(
        sys,
        branches,
        base_params,
        cont_config.linked_param_indices;
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=min_crossing_time
    )
    isempty(branches) && return BranchResult[], :period_trimmed, seed_data, Dict(
        "seedCount" => total_seed_count,
        "sampleIndex" => isnothing(sample_idx) ? nothing : sample_idx,
        "searchMin" => copy(seed_data.search_min),
        "searchMax" => copy(seed_data.search_max),
        "continuationAttempts" => continuation_attempts,
        "minimalPeriodTrim" => trim_diagnostics,
        "seedReuse" => seed_reuse_diag,
        "reseedCount" => reseed_count[],
        "reseedParams" => copy(reseed_params)
    )

    distinct = _collect_distinct_period_branches([branches], min(atlas_config.max_total_branches, length(branches)), 1e-2, 0.75)
    return distinct, :ok, seed_data, Dict(
        "seedCount" => total_seed_count,
        "sampleIndex" => isnothing(sample_idx) ? nothing : sample_idx,
        "searchMin" => copy(seed_data.search_min),
        "searchMax" => copy(seed_data.search_max),
        "continuationAttempts" => continuation_attempts,
        "distinctBranchCount" => length(distinct),
        "minimalPeriodTrim" => trim_diagnostics,
        "seedReuse" => seed_reuse_diag,
        "reseedCount" => reseed_count[],
        "reseedParams" => copy(reseed_params)
    )
end

"""Return the parameter support interval of a branch."""
function _atlas_branch_param_support(branch::BranchResult)
    points = _branch_points(branch)
    isempty(points) && return (NaN, NaN)
    params = Float64[point.param for point in points]
    return (minimum(params), maximum(params))
end

"""Compute the fraction of a window covered by a branch support interval."""
function _atlas_branch_coverage(branch::BranchResult, window::AtlasWindow)
    lo, hi = _atlas_branch_param_support(branch)
    isfinite(lo) && isfinite(hi) || return 0.0
    overlap_lo = max(lo, window.param_min)
    overlap_hi = min(hi, window.param_max)
    span = max(window.param_max - window.param_min, eps(Float64))
    return max(0.0, overlap_hi - overlap_lo) / span
end

"""Uniform auto-refine diagnostics payload.

Every `_atlas_maybe_auto_refine_branch` return path emits the same key set so downstream consumers
never have to guard for missing fields. Point counts default to the branch's current size and
`autoRefineIntervalsDetected` to `0` for the skipped / early-exit paths.
"""
function _atlas_auto_refine_diag(branch::BranchResult, reason::AbstractString;
                                 applied::Bool=false,
                                 intervals::Int=0,
                                 before::Union{Nothing, Int}=nothing,
                                 after::Union{Nothing, Int}=nothing)
    n = length(_branch_points(branch))
    return Dict{String, Any}(
        "autoRefineApplied" => applied,
        "autoRefineReason" => reason,
        "autoRefineIntervalsDetected" => intervals,
        "autoRefinePointCountBefore" => something(before, n),
        "autoRefinePointCountAfter" => something(after, n)
    )
end

"""Optionally densify under-resolved atlas branches before coverage is scored."""
function _atlas_maybe_auto_refine_branch(sys::DynamicalSystem,
                                         branch::BranchResult,
                                         base_params::Vector{Float64},
                                         cont_config::ContinuationConfig,
                                         config::AtlasConfig;
                                         log::Union{Nothing, Function} = nothing,
                                         kwargs...)
    !(sys isa Union{DiscreteMap, ContinuousODE}) && return branch, _atlas_auto_refine_diag(branch, "unsupported_system")
    !config.auto_refine_sparse_branches && return branch, _atlas_auto_refine_diag(branch, "disabled")
    config.auto_refine_max_passes <= 0 && return branch, _atlas_auto_refine_diag(branch, "zero_pass_budget")

    # Note: period-1 (fixed-point) discrete branches are intentionally NOT skipped here — sparse
    # tails and coarse short branches occur for fixed points too, and refining them keeps coverage
    # scoring consistent. Phase canonicalization (the genuinely period>1 step) is guarded inside
    # `_canonicalize_branch_representatives` / `auto_refine_branch`, so it stays a no-op for period 1.
    before_points = length(_branch_points(branch))
    before_intervals = if sys isa ContinuousODE
        _continuous_branch_refinement_intervals(branch, cont_config)
    else
        _discrete_branch_refinement_intervals(branch, cont_config)
    end
    isempty(before_intervals) && return branch, _atlas_auto_refine_diag(branch, "well_resolved")

    # Auto-refine is a best-effort densification of an already-recovered branch.
    # If it fails (e.g. Newton cannot relocate a fixed point at the start of a
    # refinement sub-interval), keep the un-refined branch and let the atlas
    # continue to other windows rather than aborting the whole run.
    refined = try
        if sys isa ContinuousODE
            auto_refine_branch(
                sys,
                branch,
                cont_config;
                params=base_params,
                linked_param_indices=cont_config.linked_param_indices,
                max_passes=config.auto_refine_max_passes,
                reseed=config.reseed,
                kwargs...
            )
        else
            auto_refine_branch(
                sys,
                branch,
                cont_config;
                params=base_params,
                linked_param_indices=cont_config.linked_param_indices,
                max_passes=config.auto_refine_max_passes,
                reseed=config.reseed
            )
        end
    catch err
        err isa InterruptException && rethrow()
        lo, hi = _atlas_branch_param_support(branch)
        _atlas_log!(log, "Atlas auto-refine skipped for branch over param ∈ [$(round(lo, digits=4)), $(round(hi, digits=4))]: " *
            "$(sprint(showerror, err)). Keeping the unrefined branch.")
        return branch, _atlas_auto_refine_diag(branch, "refine_error"; intervals=length(before_intervals))
    end
    after_points = length(_branch_points(refined))
    return refined, _atlas_auto_refine_diag(
        refined,
        after_points > before_points ? "densified" : "no_change";
        applied=after_points > before_points,
        intervals=length(before_intervals),
        before=before_points,
        after=after_points
    )
end

"""Extract a branch-point state with an explicit coordinate count."""
function _atlas_branch_point_state(point, dim::Int)
    return [Float64(getproperty(point, Symbol(:x, idx))) for idx in 1:dim]
end

"""Resolve the parameter index traced by an atlas branch."""
function _atlas_branch_param_index(sys::DynamicalSystem, branch::BranchResult)
    return something(findfirst(==(branch.param_name), sys.param_names), 1)
end

"""Build the local parameter vector for one branch point."""
function _atlas_branch_local_params(sys::DynamicalSystem,
                                    branch::BranchResult,
                                    point,
                                    base_params::Vector{Float64},
                                    linked_param_indices::Vector{Int})
    param_index = _atlas_branch_param_index(sys, branch)
    return inject_param(base_params, param_index, Float64(point.param), linked_param_indices)
end

"""Return phase-expanded orbit points for a discrete-map continuation point."""
function _atlas_branch_orbit_points(sys::DiscreteMap,
                                    branch::BranchResult,
                                    point,
                                    base_params::Vector{Float64},
                                    linked_param_indices::Vector{Int};
                                    kwargs...)
    dim = sys.dim
    period = max(branch.period, 1)
    local_params = _atlas_branch_local_params(sys, branch, point, base_params, linked_param_indices)
    current = SVector{dim, Float64}(_atlas_branch_point_state(point, dim))
    all(isfinite, current) || return Vector{Vector{Float64}}(), false

    orbit = Vector{Float64}[]
    for phase in 1:period
        push!(orbit, Array(current))
        phase == period && continue
        current = sys.f(current, local_params)
        all(isfinite, current) || return orbit, false
    end
    return orbit, true
end

"""Return phase-expanded Poincaré-section points for a continuous branch point."""
function _atlas_branch_orbit_points(sys::ContinuousODE,
                                    branch::BranchResult,
                                    point,
                                    base_params::Vector{Float64},
                                    linked_param_indices::Vector{Int};
                                    solver=Tsit5(),
                                    reltol::Float64=1e-8,
                                    abstol::Float64=1e-8,
                                    tmax::Union{Nothing, Float64}=nothing,
                                    min_crossing_time::Float64=1e-6,
                                    kwargs...)
    proj_dim = state_dim(sys)
    period = max(branch.period, 1)
    local_params = _atlas_branch_local_params(sys, branch, point, base_params, linked_param_indices)
    current = _atlas_branch_point_state(point, proj_dim)
    all(isfinite, current) || return Vector{Vector{Float64}}(), false

    orbit = Vector{Float64}[]
    for phase in 1:period
        push!(orbit, copy(current))
        phase == period && continue
        current, found = _poincare_projected(
            sys,
            current,
            local_params;
            period=1,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
        found || return orbit, false
    end
    return orbit, true
end

"""Return the branch point nearest to a reconnaissance parameter value."""
function _atlas_nearest_branch_point(branch::BranchResult, param::Float64)
    points = _branch_points(branch)
    isempty(points) && return nothing, 0, Inf

    best_idx = 1
    best_dist = abs(Float64(points[1].param) - param)
    for idx in 2:length(points)
        dist = abs(Float64(points[idx].param) - param)
        if dist < best_dist
            best_idx = idx
            best_dist = dist
        end
    end
    return points[best_idx], best_idx, best_dist
end

"""Return a robust state scale for comparing reconnaissance and branch point clouds."""
function _atlas_geometry_state_scale(sample::AtlasReconSample, branch_points::Vector{Vector{Float64}})
    scale = 1.0
    for cloud in (sample.support_points, branch_points)
        for point in cloud
            for value in point
                isfinite(value) && (scale = max(scale, abs(Float64(value))))
            end
        end
    end
    for value in sample.orbit_center
        isfinite(value) && (scale = max(scale, abs(Float64(value))))
    end
    for value in sample.orbit_span
        isfinite(value) && (scale = max(scale, abs(Float64(value))))
    end
    return max(scale, eps(Float64))
end

"""Return Euclidean distance between same-dimensional finite state points."""
function _atlas_state_distance(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || return Inf
    total = 0.0
    for idx in 1:length(a)
        av = Float64(a[idx])
        bv = Float64(b[idx])
        isfinite(av) && isfinite(bv) || return Inf
        total += (av - bv)^2
    end
    return sqrt(total)
end

"""Return the mean nearest-neighbor distance from one point cloud to another."""
function _atlas_mean_nearest_distance(from_points::Vector{Vector{Float64}}, to_points::Vector{Vector{Float64}})
    isempty(from_points) && return Inf
    isempty(to_points) && return Inf
    total = 0.0
    for point in from_points
        nearest = minimum((_atlas_state_distance(point, target) for target in to_points); init=Inf)
        isfinite(nearest) || return Inf
        total += nearest
    end
    return total / length(from_points)
end

"""Return a symmetric point-cloud mismatch distance."""
function _atlas_symmetric_cloud_distance(a::Vector{Vector{Float64}}, b::Vector{Vector{Float64}})
    return max(_atlas_mean_nearest_distance(a, b), _atlas_mean_nearest_distance(b, a))
end

"""Score one reconnaissance sample against nearby branch orbit phases."""
function _atlas_geometry_sample_diagnostics(sample::AtlasReconSample,
                                           branch_orbit::Vector{Vector{Float64}},
                                           branch_point_index::Int,
                                           param_distance::Float64,
                                           orbit_valid::Bool)
    if isempty(sample.support_points) || isempty(branch_orbit)
        return Dict{String, Any}(
            "score" => 0.0,
            "status" => "missing_cloud",
            "branchPointIndex" => branch_point_index,
            "paramDistance" => param_distance,
            "orbitValid" => orbit_valid
        )
    end

    scale = _atlas_geometry_state_scale(sample, branch_orbit)
    distance = _atlas_symmetric_cloud_distance(sample.support_points, branch_orbit)
    normalized_distance = distance / scale
    raw_threshold = Float64(get(sample.diagnostics, "threshold", 0.0))
    normalized_threshold = isfinite(raw_threshold) ? raw_threshold / scale : 0.0
    tolerance = max(normalized_threshold, 0.05)
    score = isfinite(normalized_distance) ? clamp(1 - normalized_distance / tolerance, 0.0, 1.0) : 0.0
    return Dict{String, Any}(
        "score" => score,
        "status" => orbit_valid ? "ok" : "partial_orbit",
        "distance" => distance,
        "normalizedDistance" => normalized_distance,
        "stateScale" => scale,
        "tolerance" => tolerance,
        "branchPointIndex" => branch_point_index,
        "paramDistance" => param_distance,
        "supportPointCount" => length(sample.support_points),
        "branchOrbitPointCount" => length(branch_orbit),
        "orbitValid" => orbit_valid
    )
end

"""Compare one recovered branch with the reconnaissance orbit cloud for its window."""
function _atlas_branch_geometry_diagnostics(sys::DynamicalSystem,
                                            branch::BranchResult,
                                            window::AtlasWindow,
                                            samples::Vector{AtlasReconSample},
                                            base_params::Vector{Float64},
                                            linked_param_indices::Vector{Int};
                                            kwargs...)
    points = _branch_points(branch)
    isempty(points) && return Dict{String, Any}(
        "geometryCoverageScore" => 0.0,
        "geometryCoverageStatus" => "no_branch_points",
        "geometrySampleCount" => 0,
        "geometryFailureCount" => 0,
        "geometrySamples" => Dict{String, Any}[]
    )

    lo, hi = _atlas_branch_param_support(branch)
    isfinite(lo) && isfinite(hi) || return Dict{String, Any}(
        "geometryCoverageScore" => 0.0,
        "geometryCoverageStatus" => "invalid_branch_support",
        "geometrySampleCount" => 0,
        "geometryFailureCount" => 0,
        "geometrySamples" => Dict{String, Any}[]
    )

    branch_lo, branch_hi = min(lo, hi), max(lo, hi)
    candidate_indices = [
        idx for idx in window.sample_indices
        if 1 <= idx <= length(samples) &&
           branch_lo <= samples[idx].param <= branch_hi &&
           !isempty(samples[idx].support_points)
    ]
    if isempty(candidate_indices)
        return Dict{String, Any}(
            "geometryCoverageScore" => 1.0,
            "geometryCoverageStatus" => "no_overlapping_recon_samples",
            "geometrySampleCount" => 0,
            "geometryFailureCount" => 0,
            "geometrySamples" => Dict{String, Any}[]
        )
    end

    scores = Float64[]
    sample_diagnostics = Dict{String, Any}[]
    failure_count = 0
    for idx in candidate_indices
        sample = samples[idx]
        point, point_idx, param_distance = _atlas_nearest_branch_point(branch, sample.param)
        if isnothing(point)
            failure_count += 1
            push!(scores, 0.0)
            push!(sample_diagnostics, Dict(
                "sampleIndex" => idx,
                "sampleParam" => sample.param,
                "score" => 0.0,
                "status" => "no_branch_points"
            ))
            continue
        end

        orbit = Vector{Vector{Float64}}()
        orbit_valid = false
        status = "ok"
        try
            orbit, orbit_valid = _atlas_branch_orbit_points(
                sys,
                branch,
                point,
                base_params,
                linked_param_indices;
                kwargs...
            )
            orbit_valid || (status = "partial_orbit")
        catch err
            err isa InterruptException && rethrow()
            failure_count += 1
            status = "orbit_evaluation_failed"
            push!(scores, 0.0)
            push!(sample_diagnostics, Dict(
                "sampleIndex" => idx,
                "sampleParam" => sample.param,
                "score" => 0.0,
                "status" => status,
                "error" => sprint(showerror, err),
                "branchPointIndex" => point_idx,
                "paramDistance" => param_distance
            ))
            continue
        end

        diag = _atlas_geometry_sample_diagnostics(sample, orbit, point_idx, param_distance, orbit_valid)
        diag["sampleIndex"] = idx
        diag["sampleParam"] = sample.param
        diag["branchParam"] = Float64(point.param)
        status != "ok" && (diag["status"] = status)
        score = Float64(diag["score"])
        push!(scores, score)
        push!(sample_diagnostics, diag)
        orbit_valid || (failure_count += 1)
    end

    score = isempty(scores) ? 1.0 : clamp(sum(scores) / length(scores), 0.0, 1.0)
    return Dict{String, Any}(
        "geometryCoverageScore" => score,
        "geometryCoverageStatus" => failure_count == length(scores) ? "failed" : "evaluated",
        "geometrySampleCount" => length(scores),
        "geometryFailureCount" => failure_count,
        "geometryMinimumSampleScore" => isempty(scores) ? nothing : minimum(scores),
        "geometrySamples" => sample_diagnostics
    )
end

"""Return the geometric confidence weight used for interval coverage accounting."""
function _atlas_record_geometry_score(record::AtlasBranchRecord)
    raw = get(record.diagnostics, "geometryCoverageScore", 1.0)
    raw isa Real || return 1.0
    score = Float64(raw)
    isfinite(score) || return 1.0
    return clamp(score, 0.0, 1.0)
end

"""Detect a local switched probe that merely retraces an existing branch segment."""
function _atlas_branch_overlap_duplicate(candidate::BranchResult,
                                         existing::Vector{BranchResult},
                                         param_tol::Float64,
                                         state_tol::Float64)
    candidate_points = _branch_points(candidate)
    isempty(candidate_points) && return true
    candidate_params = Float64[point.param for point in candidate_points]
    candidate_span = maximum(candidate_params) - minimum(candidate_params)
    local_param_tol = max(param_tol, candidate_span / max(length(candidate_points), 1), 1e-6)
    sample_indices = length(candidate_points) >= 3 ?
        [1, clamp(cld(length(candidate_points), 2), 1, length(candidate_points)), length(candidate_points)] :
        collect(1:length(candidate_points))

    for existing_branch in existing
        existing_branch.period == candidate.period || continue
        existing_points = _branch_points(existing_branch)
        isempty(existing_points) && continue
        matched = true
        for idx in sample_indices
            point = candidate_points[idx]
            nearest, _, param_distance = _atlas_nearest_branch_point(existing_branch, Float64(point.param))
            if isnothing(nearest) || param_distance > local_param_tol || _branch_point_distance(point, nearest) > state_tol
                matched = false
                break
            end
        end
        matched && return true
    end
    return false
end

"""Return weighted branch support segments that overlap a requested parameter interval."""
function _atlas_interval_coverage_segments(param_min::Float64,
                                           param_max::Float64,
                                           branch_records::Vector{AtlasBranchRecord},
                                           window_id::AbstractString,
                                           period::Int;
                                           min_geometry_score::Float64=0.0)
    segments = Tuple{Float64, Float64, Float64}[]
    for record in branch_records
        record.window_id == window_id || continue
        record.branch.period == period || continue
        geometry_score = _atlas_record_geometry_score(record)
        geometry_score >= min_geometry_score || continue
        lo, hi = _atlas_branch_param_support(record.branch)
        isfinite(lo) && isfinite(hi) || continue
        overlap_lo = max(min(lo, hi), min(param_min, param_max))
        overlap_hi = min(max(lo, hi), max(param_min, param_max))
        overlap_hi > overlap_lo || continue
        push!(segments, (overlap_lo, overlap_hi, geometry_score))
    end
    return segments
end

"""Return the branch support intervals that overlap a requested parameter interval."""
function _atlas_interval_coverage_intervals(param_min::Float64,
                                            param_max::Float64,
                                            branch_records::Vector{AtlasBranchRecord},
                                            window_id::AbstractString,
                                            period::Int;
                                            min_geometry_score::Float64=0.0)
    intervals = Tuple{Float64, Float64}[
        (lo, hi) for (lo, hi, _) in _atlas_interval_coverage_segments(
            param_min,
            param_max,
            branch_records,
            window_id,
            period;
            min_geometry_score=min_geometry_score
        )
    ]
    return _merge_param_intervals(intervals)
end

"""Trim recovered period-N branches before atlas coverage accounting."""
function _atlas_trim_recovered_branches(sys::DynamicalSystem,
                                        branches::Vector{BranchResult},
                                        base_params::Vector{Float64},
                                        linked_param_indices::Vector{Int};
                                        solver=Tsit5(),
                                        reltol::Float64=1e-8,
                                        abstol::Float64=1e-8,
                                        min_crossing_time::Float64=1e-6)
    trimmed = BranchResult[]
    diagnostics = Dict{String, Any}[]
    for (idx, branch) in enumerate(branches)
        diag_ref = Ref{Any}(nothing)
        next_branch = if sys isa ContinuousODE
            _trim_branch_to_period(
                sys,
                branch,
                base_params,
                linked_param_indices;
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time,
                trim_diagnostics=diag_ref
            )
        else
            _trim_branch_to_period(
                sys,
                branch,
                base_params,
                linked_param_indices;
                trim_diagnostics=diag_ref
            )
        end
        diag = diag_ref[] isa AbstractDict ? Dict{String, Any}(String(k) => v for (k, v) in pairs(diag_ref[])) : Dict{String, Any}()
        diag["sourceBranchIndex"] = idx
        push!(diagnostics, diag)
        isnothing(next_branch) || push!(trimmed, next_branch)
    end
    return trimmed, diagnostics
end

"""Compute the union-based coverage fraction for one parameter interval."""
function _atlas_interval_coverage_fraction(param_min::Float64,
                                           param_max::Float64,
                                           branch_records::Vector{AtlasBranchRecord},
                                           window_id::AbstractString,
                                           period::Int)
    span = _atlas_interval_span(param_min, param_max)
    segments = _atlas_interval_coverage_segments(param_min, param_max, branch_records, window_id, period)
    if isempty(segments)
        return 0.0
    end

    lo = min(param_min, param_max)
    hi = max(param_min, param_max)
    endpoints = Float64[lo, hi]
    for (seg_lo, seg_hi, _) in segments
        push!(endpoints, max(seg_lo, lo))
        push!(endpoints, min(seg_hi, hi))
    end
    sort!(unique!(endpoints))

    covered = 0.0
    for idx in 1:(length(endpoints) - 1)
        seg_lo = endpoints[idx]
        seg_hi = endpoints[idx + 1]
        seg_hi > seg_lo || continue
        midpoint = (seg_lo + seg_hi) / 2
        weight = maximum((
            score for (record_lo, record_hi, score) in segments
            if record_lo <= midpoint <= record_hi
        ); init=0.0)
        covered += (seg_hi - seg_lo) * weight
    end
    return clamp(covered / span, 0.0, 1.0)
end

"""Compute uncovered sub-intervals after subtracting covered intervals from one parameter interval."""
function _atlas_uncovered_intervals(param_min::Float64,
                                    param_max::Float64,
                                    covered_intervals::Vector{Tuple{Float64, Float64}})
    lo = min(param_min, param_max)
    hi = max(param_min, param_max)
    hi > lo || return Tuple{Float64, Float64}[]
    merged = _merge_param_intervals([(max(a, lo), min(b, hi)) for (a, b) in covered_intervals if min(b, hi) > max(a, lo)])
    isempty(merged) && return [(lo, hi)]

    gaps = Tuple{Float64, Float64}[]
    cursor = lo
    for (covered_lo, covered_hi) in merged
        if covered_lo > cursor + 100eps(Float64)
            push!(gaps, (cursor, covered_lo))
        end
        cursor = max(cursor, covered_hi)
    end
    if cursor < hi - 100eps(Float64)
        push!(gaps, (cursor, hi))
    end
    return gaps
end

"""Return the union-based coverage fraction for one atlas window."""
function _atlas_window_coverage_fraction(window::AtlasWindow, branch_records::Vector{AtlasBranchRecord})
    return _atlas_interval_coverage_fraction(window.param_min, window.param_max, branch_records, window.id, window.period)
end
