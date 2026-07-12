"""Build a padded local search box from sampled orbit points."""
function _atlas_search_box(points::Vector{Vector{Float64}}, padding::Float64)
    geometry = _orbit_geometry_summary(points)
    isempty(geometry.center) && return Float64[], Float64[]
    base_padding = max.(geometry.span .* padding, fill(1e-2, length(geometry.center)))
    return geometry.minima .- base_padding, geometry.maxima .+ base_padding
end

"""
Derive search-box bounds from the entire reconnaissance sweep when no window-local
cloud is available. Avoids the hardcoded `[-3, 3]` fallback that silently
mis-seeded atlas searches on systems whose attractors live outside that box
(Colpitts oscillators, for instance, have section states well above 3).

Returns the hardcoded `[-3, 3]` only as a true last resort — when no sample
across the whole reconnaissance produced any support points at all.
"""
function _atlas_global_fallback_bounds(samples::Vector{AtlasReconSample},
                                       fallback_dim::Int,
                                       padding::Float64)
    # Incremental append! instead of reduce(vcat, [comprehension]; init=...)
    # avoids allocating an intermediate vector-of-vectors plus the
    # concatenations on long reconnaissance runs.
    all_points = Vector{Vector{Float64}}()
    for sample in samples
        append!(all_points, sample.support_points)
    end
    if !isempty(all_points)
        fb_min, fb_max = _atlas_search_box(all_points, padding)
        !isempty(fb_min) && return fb_min, fb_max
    end
    return fill(-3.0, fallback_dim), fill(3.0, fallback_dim)
end

"""Extract local seed points and a search box for a candidate window."""
function _extract_window_seed_data(window::AtlasWindow,
                                   samples::Vector{AtlasReconSample},
                                   atlas_config::AtlasConfig,
                                   fallback_dim::Int;
                                   sample_idx::Union{Nothing, Int}=nothing,
                                   padding_scale::Float64=1.0)
    if isnothing(sample_idx)
        candidates = _atlas_candidate_sample_indices(window, samples)
        sample_idx = isempty(candidates) ? nothing : first(candidates)
    end
    if isnothing(sample_idx)
        fb_min, fb_max = _atlas_global_fallback_bounds(samples, fallback_dim, atlas_config.seed_box_padding * padding_scale)
        return (skeleton_param=(window.param_min + window.param_max) / 2,
                seed_points=Vector{Vector{Float64}}(),
                search_min=fb_min,
                search_max=fb_max)
    end

    rep = samples[sample_idx]
    cloud = reduce(vcat, [samples[idx].support_points for idx in window.sample_indices if samples[idx].best_period == window.period]; init=Vector{Vector{Float64}}())
    isempty(cloud) && (cloud = reduce(vcat, [samples[idx].support_points for idx in window.sample_indices]; init=Vector{Vector{Float64}}()))
    local_cloud = isempty(cloud) ? rep.support_points : cloud
    search_min, search_max = _atlas_search_box(local_cloud, atlas_config.seed_box_padding * padding_scale)
    if isempty(search_min)
        search_min, search_max = _atlas_global_fallback_bounds(samples, fallback_dim, atlas_config.seed_box_padding * padding_scale)
    end

    seed_candidates = _prepare_seed_points(rep.support_points, search_min, search_max, max(4, atlas_config.seed_points_per_window);
                                           max_points=max(atlas_config.seed_points_per_window, 8))
    return (
        skeleton_param=rep.param,
        seed_points=first(seed_candidates, min(length(seed_candidates), atlas_config.seed_points_per_window)),
        search_min=search_min,
        search_max=search_max
    )
end

"""Create the per-period atlas neighbor-seed cache."""
_atlas_seed_cache() = _AtlasSeedCache()

"""Maximum parameter distance allowed for atlas seed reuse around one target window."""
function _atlas_neighbor_seed_max_distance(window::AtlasWindow,
                                           cont_config::ContinuationConfig,
                                           atlas_config::AtlasConfig)
    range_span = _atlas_interval_span(cont_config.p_min, cont_config.p_max)
    window_span = _atlas_interval_span(window.param_min, window.param_max)
    frac = max(atlas_config.neighbor_seed_max_distance_fraction, 0.0)
    return max(window_span, range_span * frac)
end

"""Return cached same-period seed hints near a recovery parameter."""
function _atlas_cached_seed_points(seed_cache::Union{Nothing, _AtlasSeedCache},
                                  window::AtlasWindow,
                                  skeleton_param::Float64,
                                  cont_config::ContinuationConfig,
                                  atlas_config::AtlasConfig)
    requested = atlas_config.reuse_neighbor_seeds
    requested || return nothing, Dict{String, Any}(
        "requested" => false,
        "hit" => false,
        "reusedSeedCount" => 0,
        "cacheSize" => 0
    )
    isnothing(seed_cache) && return nothing, Dict{String, Any}(
        "requested" => true,
        "hit" => false,
        "status" => "cache_unavailable",
        "reusedSeedCount" => 0,
        "cacheSize" => 0
    )

    entries = get(seed_cache, window.period, _AtlasSeedEntry[])
    max_distance = _atlas_neighbor_seed_max_distance(window, cont_config, atlas_config)
    max_points = max(0, atlas_config.neighbor_seed_max_points)
    candidates = Tuple{Int, Float64}[]
    for (idx, entry) in enumerate(entries)
        dist = abs(entry.param - skeleton_param)
        dist <= max_distance || continue
        push!(candidates, (idx, dist))
    end
    sort!(candidates; by=item -> (item[2], -entries[item[1]].stamp))
    keep = first(candidates, min(max_points, length(candidates)))
    points = [copy(entries[idx].point) for (idx, _) in keep]
    hit = !isempty(points)
    return hit ? points : nothing, Dict{String, Any}(
        "requested" => true,
        "hit" => hit,
        "status" => hit ? "hit" : (isempty(entries) ? "empty_cache" : "miss_distance"),
        "reusedSeedCount" => length(points),
        "cacheSize" => length(entries),
        "candidateCount" => length(candidates),
        "maxDistance" => max_distance,
        "nearestDistance" => isempty(entries) ? nothing : minimum(abs(entry.param - skeleton_param) for entry in entries),
        "skeletonParam" => skeleton_param,
        "period" => window.period
    )
end

"""Merge cached warm starts ahead of reconnaissance-derived seed hints."""
function _atlas_merge_recovery_seed_points(cached_points::Union{Nothing, AbstractVector},
                                           local_points::AbstractVector,
                                           search_min::AbstractVector,
                                           search_max::AbstractVector,
                                           atlas_config::AtlasConfig)
    ordered = Vector{Vector{Float64}}()
    if !isnothing(cached_points)
        append!(ordered, [collect(Float64, point) for point in cached_points])
    end
    append!(ordered, [collect(Float64, point) for point in local_points])
    isempty(ordered) && return Vector{Vector{Float64}}()
    max_points = max(atlas_config.seed_points_per_window + atlas_config.neighbor_seed_max_points, atlas_config.seed_points_per_window)
    return _prepare_seed_points(ordered, search_min, search_max, max(4, max_points); max_points=max(8, max_points))
end

"""Append successful skeleton seeds to the atlas neighbor-seed cache."""
function _atlas_update_seed_cache!(seed_cache::Union{Nothing, _AtlasSeedCache},
                                  period::Int,
                                  skeleton_param::Float64,
                                  seed_points::AbstractVector,
                                  atlas_config::AtlasConfig)
    if !atlas_config.reuse_neighbor_seeds
        return Dict{String, Any}("requested" => false, "storedSeedCount" => 0, "cacheSize" => 0)
    end
    if isnothing(seed_cache) || atlas_config.neighbor_seed_max_entries <= 0 || isempty(seed_points)
        return Dict{String, Any}(
            "requested" => true,
            "storedSeedCount" => 0,
            "cacheSize" => isnothing(seed_cache) ? 0 : length(get(seed_cache, period, _AtlasSeedEntry[]))
        )
    end

    entries = get!(seed_cache, period, _AtlasSeedEntry[])
    before = length(entries)
    _update_neighbor_seed_cache!(
        entries,
        skeleton_param,
        seed_points;
        max_entries=max(0, atlas_config.neighbor_seed_max_entries)
    )
    return Dict{String, Any}(
        "requested" => true,
        "storedSeedCount" => max(length(entries) - before, 0),
        "cacheSize" => length(entries),
        "period" => period,
        "skeletonParam" => skeleton_param
    )
end

"""Combine lookup/store diagnostics for one atlas seed-reuse attempt."""
function _atlas_seed_reuse_event(lookup_diag::AbstractDict{<:AbstractString, <:Any},
                                 store_diag::AbstractDict{<:AbstractString, <:Any})
    return merge(Dict{String, Any}(String(k) => v for (k, v) in pairs(lookup_diag)),
        Dict(
            "storedSeedCount" => get(store_diag, "storedSeedCount", 0),
            "cacheSizeAfterStore" => get(store_diag, "cacheSize", get(lookup_diag, "cacheSize", 0))
        ))
end

"""Build a set of localized continuation retries for one window-recovery attempt."""
function _atlas_continuation_retry_configs(cont_config::ContinuationConfig,
                                           param_min::Float64,
                                           param_max::Float64,
                                           retry_budget::Int)
    budget = max(retry_budget, 1)
    span = _atlas_interval_span(param_min, param_max)
    # Each branch is traced across the full user-requested parameter window so the bifurcation
    # diagram includes the unstable continuations of each orbit (e.g. the period-1 fixed point
    # continued through its period-doubling). The trim step downstream still removes points
    # whose minimal period collapses to a divisor.
    local_p_min = cont_config.p_min
    local_p_max = cont_config.p_max
    configs = ContinuationConfig[]

    # Global span over which continuation traces (branches are not clamped to the
    # window). The window-relative scaling below adapts step size for narrow windows
    # but is *floored* at the user's nominal ds/dsmax so single-sample windows don't
    # collapse the step into PALC's `dsmin` regime and produce thousands of tiny-step
    # segments.
    global_span = max(local_p_max - local_p_min, abs(cont_config.ds))
    for retry_idx in 1:budget
        scale = 2.0 ^ (retry_idx - 1)
        # Suggested ds shrinks on retry to catch missed detail; never tighter than the
        # user's `ds / scale` and never below `dsmin * 10`. Window-aware narrowing is
        # applied only when the window has a meaningful span.
        suggested_ds = abs(cont_config.ds) / scale
        window_targeted = span > 0 ? span / 80 : suggested_ds
        ds_mag = max(min(suggested_ds, window_targeted), cont_config.dsmin * 10)
        dsmax_mag = max(abs(cont_config.dsmax) / scale, ds_mag)
        # Cap `max_steps` to ~30 × steps-to-cross-the-window at this step size, so a
        # 0.056-wide P7 window doesn't get the same 800-step budget that P1 spanning
        # [0, 1.4] needs. Clamped to [50, cont_config.max_steps] so very narrow windows
        # still get a workable budget and the user's explicit cap is honoured.
        steps_to_cross = ceil(Int, 30 * global_span / dsmax_mag)
        local_max_steps = clamp(steps_to_cross, 50, cont_config.max_steps)
        # Keep `dsmin` at the user's setting so PALC's adapter has full freedom to
        # follow unstable continuations through sub-bifurcations; squeezing the
        # adapter instead drops high-period unstable extensions. Over-counting from
        # PALC's tiny-step regime is handled downstream by `_trim_branch_to_period`
        # (drops lower-period stretches).
        local_dsmin = cont_config.dsmin
        push!(configs, ContinuationConfig(
            p_min=local_p_min,
            p_max=local_p_max,
            ds=sign(cont_config.ds) == 0 ? ds_mag : sign(cont_config.ds) * ds_mag,
            dsmax=dsmax_mag,
            dsmin=local_dsmin,
            max_steps=local_max_steps,
            newton_tol=cont_config.newton_tol,
            newton_max_iter=max(cont_config.newton_max_iter, 25 + 5 * (retry_idx - 1)),
            detect_bifurcation=cont_config.detect_bifurcation,
            param_index=cont_config.param_index,
            linked_param_indices=copy(cont_config.linked_param_indices),
            ode_jacobian_method=cont_config.ode_jacobian_method
        ))
    end

    return configs
end

