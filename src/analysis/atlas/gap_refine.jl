"""Return whether a gap is worth refining recursively."""
function _atlas_should_refine_gap(gap::AtlasGap, config::AtlasConfig, started_at::Float64)
    gap.retryable || return false
    gap.depth < config.max_refinement_depth || return false
    gap.confidence >= min(max(config.coverage_threshold / 2, 0.1), 0.6) || return false
    !_atlas_time_budget_exhausted(config, started_at) || return false
    return _atlas_interval_span(gap.param_min, gap.param_max) > 100eps(Float64)
end

"""Return a human-readable label for a BifurcationKit special point."""
function _atlas_specialpoint_label(special)
    for name in (:type, :label, :kind, :name)
        name in propertynames(special) || continue
        value = getproperty(special, name)
        return value isa Symbol ? String(value) : string(value)
    end
    return string(typeof(special))
end

"""Return the branch parameter associated with a special point, when recoverable."""
function _atlas_specialpoint_param(special, branch::BranchResult)
    for name in (:param, :p, :parameter)
        name in propertynames(special) || continue
        value = getproperty(special, name)
        value isa Real && return Float64(value)
    end

    points = _branch_points(branch)
    isempty(points) && return nothing
    for name in (:idx, :index, :ind, :ind_ev)
        name in propertynames(special) || continue
        value = getproperty(special, name)
        value isa Real || continue
        idx = clamp(round(Int, value), 1, length(points))
        return Float64(points[idx].param)
    end
    return nothing
end

"""Return whether a special-point label is useful for follow-up branch-switching probes."""
function _atlas_specialpoint_is_switch_candidate(label::AbstractString)
    lowered = lowercase(label)
    tokens = ("fold", "limit", "lp", "branch", "bp", "period", "doubling", "pd", "neimark", "sacker", "ns", "hopf")
    return any(token -> occursin(token, lowered), tokens)
end

"""Collect eligible special points from one atlas branch record."""
function _atlas_branch_switching_specialpoints(record::AtlasBranchRecord, max_special_points::Int)
    max_special_points <= 0 && return NamedTuple[]
    candidates = NamedTuple[]
    specials = _segment_specials(record.branch.branch)
    for (special_idx, special) in enumerate(specials)
        label = _atlas_specialpoint_label(special)
        _atlas_specialpoint_is_switch_candidate(label) || continue
        param = _atlas_specialpoint_param(special, record.branch)
        isnothing(param) && continue
        point, point_idx, param_distance = _atlas_nearest_branch_point(record.branch, param)
        isnothing(point) && continue
        push!(candidates, (
            special_index=special_idx,
            label=label,
            param=Float64(param),
            branch_point=point,
            branch_point_index=point_idx,
            param_distance=param_distance
        ))
        length(candidates) >= max_special_points && break
    end
    return candidates
end

"""Build a tightly budgeted local continuation config for one branch-switching probe."""
function _atlas_branch_switching_continuation_config(cont_config::ContinuationConfig,
                                                     atlas_config::AtlasConfig,
                                                     param::Float64)
    global_span = _atlas_interval_span(cont_config.p_min, cont_config.p_max)
    frac = clamp(atlas_config.branch_switching_window_fraction, 0.0, 1.0)
    half_width = max(global_span * frac, 4 * abs(cont_config.ds))
    local_min = max(cont_config.p_min, param - half_width)
    local_max = min(cont_config.p_max, param + half_width)
    local_span = local_max - local_min
    local_span > 100eps(Float64) || return nothing

    ds_mag = max(min(abs(cont_config.ds), local_span / 20), cont_config.dsmin * 10)
    dsmax_mag = max(ds_mag, min(abs(cont_config.dsmax), local_span / 4))
    return ContinuationConfig(
        p_min=local_min,
        p_max=local_max,
        ds=sign(cont_config.ds) == 0 ? ds_mag : sign(cont_config.ds) * ds_mag,
        dsmax=dsmax_mag,
        dsmin=cont_config.dsmin,
        max_steps=min(cont_config.max_steps, max(1, atlas_config.branch_switching_max_steps)),
        newton_tol=cont_config.newton_tol,
        newton_max_iter=cont_config.newton_max_iter,
        detect_bifurcation=cont_config.detect_bifurcation,
        param_index=cont_config.param_index,
        linked_param_indices=copy(cont_config.linked_param_indices),
        a=cont_config.a,
        detect_fold=cont_config.detect_fold,
        save_sol_every_step=cont_config.save_sol_every_step,
        ode_jacobian_method=cont_config.ode_jacobian_method
    )
end

"""Return a small state-space search box and perturbed seed hints around a special point."""
function _atlas_branch_switching_seed_box(state::AbstractVector, atlas_config::AtlasConfig)
    x = collect(Float64, state)
    scale = max(maximum(abs.(x); init=0.0), 1.0)
    half_width = max(scale * atlas_config.branch_switching_perturbation_scale, sqrt(eps(Float64)))
    lo = x .- half_width
    hi = x .+ half_width

    seeds = Vector{Float64}[copy(x)]
    max_seeds = max(1, atlas_config.branch_switching_max_seed_candidates)
    for dim in eachindex(x)
        for direction in (-1.0, 1.0)
            length(seeds) >= max_seeds && return lo, hi, seeds
            candidate = copy(x)
            candidate[dim] += direction * half_width
            push!(seeds, candidate)
        end
    end
    return lo, hi, seeds
end

function _atlas_branch_switching_skeleton(sys::DiscreteMap,
                                          period::Int,
                                          param::Float64,
                                          seed_state::AbstractVector,
                                          base_params::Vector{Float64},
                                          cont_config::ContinuationConfig,
                                          atlas_config::AtlasConfig;
                                          kwargs...)
    lo, hi, seed_points = _atlas_branch_switching_seed_box(seed_state, atlas_config)
    local_params = _inject_param(base_params, cont_config.param_index, param, cont_config.linked_param_indices)
    seeds = find_periodic_skeleton(
        sys,
        [period],
        param;
        n_initial=max(3, length(seed_points)),
        search_min=lo,
        search_max=hi,
        seed_points=seed_points,
        params=local_params,
        param_index=cont_config.param_index,
        linked_param_indices=cont_config.linked_param_indices,
        tol=max(cont_config.newton_tol, 1e-8),
        max_iter=max(cont_config.newton_max_iter, 40),
        threaded=false,
        cache_enabled=false
    )
    return [seed for seed in seeds if seed.period == period], lo, hi, seed_points
end

function _atlas_branch_switching_skeleton(sys::ContinuousODE,
                                          period::Int,
                                          param::Float64,
                                          seed_state::AbstractVector,
                                          base_params::Vector{Float64},
                                          cont_config::ContinuationConfig,
                                          atlas_config::AtlasConfig;
                                          solver=Tsit5(),
                                          reltol::Float64=1e-8,
                                          abstol::Float64=1e-8,
                                          min_crossing_time::Float64=1e-6,
                                          kwargs...)
    lo, hi, seed_points = _atlas_branch_switching_seed_box(seed_state, atlas_config)
    local_params = _inject_param(base_params, cont_config.param_index, param, cont_config.linked_param_indices)
    seeds = find_periodic_skeleton(
        sys,
        [period],
        param;
        n_initial=max(3, length(seed_points)),
        search_min=lo,
        search_max=hi,
        seed_points=seed_points,
        params=local_params,
        param_index=cont_config.param_index,
        linked_param_indices=cont_config.linked_param_indices,
        tol=max(cont_config.newton_tol, 1e-8),
        max_iter=max(cont_config.newton_max_iter, 40),
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=min_crossing_time,
        threaded=false,
        cache_enabled=false
    )
    return [seed for seed in seeds if seed.period == period], lo, hi, seed_points
end

function _atlas_continue_switched_seed(sys::DiscreteMap,
                                       period::Int,
                                       seed_point::AbstractVector,
                                       params::Vector{Float64},
                                       local_config::ContinuationConfig;
                                       kwargs...)
    return continuation_branch(
        sys,
        local_config,
        period;
        initial_point=seed_point,
        params=params,
        reseed=ReseedConfig(enabled=false),
        trim_to_minimal_period=true
    )
end

function _atlas_continue_switched_seed(sys::ContinuousODE,
                                       period::Int,
                                       seed_point::AbstractVector,
                                       params::Vector{Float64},
                                       local_config::ContinuationConfig;
                                       solver=Tsit5(),
                                       reltol::Float64=1e-8,
                                       abstol::Float64=1e-8,
                                       min_crossing_time::Float64=1e-6,
                                       kwargs...)
    return continuation_branch(
        sys,
        local_config,
        period;
        initial_point=seed_point,
        params=params,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=min_crossing_time,
        reseed=ReseedConfig(enabled=false),
        trim_to_minimal_period=true
    )
end

"""Probe for local switched branches near one recorded special point."""
function _atlas_probe_branch_switch(sys::DynamicalSystem,
                                    record::AtlasBranchRecord,
                                    special,
                                    base_params::Vector{Float64},
                                    cont_config::ContinuationConfig,
                                    atlas_config::AtlasConfig;
                                    kwargs...)
    local_config = _atlas_branch_switching_continuation_config(cont_config, atlas_config, special.param)
    isnothing(local_config) && return BranchResult[], Dict{String, Any}(
        "status" => "skipped_degenerate_parameter_window",
        "sourceBranchId" => record.id,
        "specialPointIndex" => special.special_index,
        "specialPointLabel" => special.label,
        "specialPointParam" => special.param
    )

    seed_state = _branch_point_state(special.branch_point)
    skeleton_seeds = Any[]
    search_min = Float64[]
    search_max = Float64[]
    seed_points = Vector{Float64}[]
    try
        skeleton_seeds, search_min, search_max, seed_points = _atlas_branch_switching_skeleton(
            sys,
            record.branch.period,
            special.param,
            seed_state,
            base_params,
            cont_config,
            atlas_config;
            kwargs...
        )
    catch err
        return BranchResult[], Dict{String, Any}(
            "status" => "skeleton_failed",
            "sourceBranchId" => record.id,
            "specialPointIndex" => special.special_index,
            "specialPointLabel" => special.label,
            "specialPointParam" => special.param,
            "error" => _continuation_error_message(err)
        )
    end

    local_params = _inject_param(base_params, cont_config.param_index, special.param, cont_config.linked_param_indices)
    branches = BranchResult[]
    continuation_attempts = Dict{String, Any}[]
    for (seed_idx, seed) in enumerate(skeleton_seeds)
        try
            branch = _atlas_continue_switched_seed(
                sys,
                record.branch.period,
                seed.point,
                local_params,
                local_config;
                kwargs...
            )
            push!(branches, branch)
            push!(continuation_attempts, Dict(
                "seedIndex" => seed_idx,
                "status" => "ok",
                "seedPoint" => copy(seed.point)
            ))
        catch err
            push!(continuation_attempts, Dict(
                "seedIndex" => seed_idx,
                "status" => "failed",
                "seedPoint" => copy(seed.point),
                "error" => _continuation_error_message(err)
            ))
        end
    end

    status = isempty(branches) ? (isempty(skeleton_seeds) ? "no_skeleton_seed" : "continuation_failed") : "ok"
    return branches, Dict{String, Any}(
        "status" => status,
        "sourceBranchId" => record.id,
        "specialPointIndex" => special.special_index,
        "specialPointLabel" => special.label,
        "specialPointParam" => special.param,
        "branchPointIndex" => special.branch_point_index,
        "branchPointParamDistance" => special.param_distance,
        "searchMin" => copy(search_min),
        "searchMax" => copy(search_max),
        "seedPointCount" => length(seed_points),
        "skeletonSeedCount" => length(skeleton_seeds),
        "pMin" => local_config.p_min,
        "pMax" => local_config.p_max,
        "ds" => local_config.ds,
        "maxSteps" => local_config.max_steps,
        "continuationAttempts" => continuation_attempts
    )
end

"""Run bounded special-point follow-up probes for one atlas window."""
function _atlas_branch_switching_followups(sys::DynamicalSystem,
                                           window::AtlasWindow,
                                           samples::Vector{AtlasReconSample},
                                           branch_records::Vector{AtlasBranchRecord},
                                           base_params::Vector{Float64},
                                           cont_config::ContinuationConfig,
                                           atlas_config::AtlasConfig,
                                           id_counter::Base.RefValue{Int};
                                           kwargs...)
    if !atlas_config.branch_switching
        return AtlasBranchRecord[], Dict{String, Any}(
            "requested" => false,
            "applied" => false,
            "status" => "disabled",
            "attemptCount" => 0,
            "newBranchCount" => 0,
            "specialPointCount" => 0,
            "attempts" => Dict{String, Any}[]
        )
    end

    max_new = max(0, min(atlas_config.branch_switching_max_branches, atlas_config.max_total_branches - length(branch_records)))
    if max_new <= 0 || atlas_config.branch_switching_max_special_points <= 0
        return AtlasBranchRecord[], Dict{String, Any}(
            "requested" => true,
            "applied" => false,
            "status" => "budget_exhausted",
            "attemptCount" => 0,
            "newBranchCount" => 0,
            "specialPointCount" => 0,
            "attempts" => Dict{String, Any}[]
        )
    end

    new_records = AtlasBranchRecord[]
    attempts = Dict{String, Any}[]
    special_count = 0
    existing = [record.branch for record in branch_records]
    source_records = [
        record for record in branch_records
        if record.window_id == window.id &&
           record.branch.period == window.period &&
           get(record.diagnostics, "source", "") != "atlas-branch-switching"
    ]

    for source_record in source_records
        specials = _atlas_branch_switching_specialpoints(source_record, atlas_config.branch_switching_max_special_points)
        for special in specials
            special_count += 1
            branches, attempt_diag = _atlas_probe_branch_switch(
                sys,
                source_record,
                special,
                base_params,
                cont_config,
                atlas_config;
                kwargs...
            )

            accepted = 0
            duplicate = 0
            for branch in branches
                length(new_records) >= max_new && break
                if _is_duplicate_branch(branch, existing, 5e-3, 0.75) ||
                   _atlas_branch_overlap_duplicate(branch, existing, 5e-3, 0.75)
                    duplicate += 1
                    continue
                end
                branch, auto_refine_diag = _atlas_maybe_auto_refine_branch(
                    sys,
                    branch,
                    base_params,
                    cont_config,
                    atlas_config;
                    kwargs...
                )
                parameter_coverage = _atlas_branch_coverage(branch, window)
                lo, hi = _atlas_branch_param_support(branch)
                geometry_diagnostics = _atlas_branch_geometry_diagnostics(
                    sys,
                    branch,
                    window,
                    samples,
                    base_params,
                    cont_config.linked_param_indices;
                    kwargs...
                )
                geometry_score = Float64(get(geometry_diagnostics, "geometryCoverageScore", 1.0))
                combined_coverage = clamp(parameter_coverage * geometry_score, 0.0, 1.0)
                push!(new_records, AtlasBranchRecord(
                    _atlas_next_id!(id_counter, "atlas-branch"),
                    branch,
                    special.param,
                    window.id,
                    combined_coverage,
                    lo,
                    hi,
                    merge(geometry_diagnostics, auto_refine_diag, Dict(
                        "source" => "atlas-branch-switching",
                        "parentBranchId" => source_record.id,
                        "sourceWindowId" => window.id,
                        "specialPointIndex" => special.special_index,
                        "specialPointLabel" => special.label,
                        "specialPointParam" => special.param,
                        "parameterCoverageScore" => parameter_coverage,
                        "combinedCoverageScore" => combined_coverage,
                        "coverageScoreType" => "parameter_geometry_product",
                        "branchSwitchingAttempt" => attempt_diag
                    ))
                ))
                push!(existing, branch)
                accepted += 1
            end

            attempt_diag["acceptedBranchCount"] = accepted
            attempt_diag["duplicateBranchCount"] = duplicate
            push!(attempts, attempt_diag)
            length(new_records) >= max_new && break
        end
        length(new_records) >= max_new && break
    end

    status = isempty(attempts) ? "no_eligible_special_points" :
             isempty(new_records) ? "attempted_no_new_branches" : "applied"
    return new_records, Dict{String, Any}(
        "requested" => true,
        "applied" => !isempty(attempts),
        "status" => status,
        "attemptCount" => length(attempts),
        "newBranchCount" => length(new_records),
        "specialPointCount" => special_count,
        "attempts" => attempts
    )
end

"""Summarize branch-switching attempts for atlas-level diagnostics."""
_atlas_diag_int(value) = value isa Integer ? Int(value) : value isa Real ? Int(round(value)) : 0

function _atlas_branch_switching_summary(config::AtlasConfig, diagnostics::Vector{Dict{String, Any}})
    if !config.branch_switching
        return Dict{String, Any}(
            "branchSwitchingRequested" => false,
            "branchSwitchingApplied" => false,
            "branchSwitchingStatus" => "disabled",
            "branchSwitchingMessage" => "Branch switching was not requested.",
            "branchSwitchingAttemptCount" => 0,
            "branchSwitchingNewBranchCount" => 0,
            "branchSwitchingSpecialPointCount" => 0,
            "branchSwitchingDiagnostics" => diagnostics
        )
    end

    attempt_count = sum(_atlas_diag_int(get(diag, "attemptCount", 0)) for diag in diagnostics; init=0)
    new_branch_count = sum(_atlas_diag_int(get(diag, "newBranchCount", 0)) for diag in diagnostics; init=0)
    special_count = sum(_atlas_diag_int(get(diag, "specialPointCount", 0)) for diag in diagnostics; init=0)
    status = attempt_count == 0 ? "no_eligible_special_points" :
             new_branch_count == 0 ? "attempted_no_new_branches" : "applied"
    return Dict{String, Any}(
        "branchSwitchingRequested" => true,
        "branchSwitchingApplied" => attempt_count > 0,
        "branchSwitchingStatus" => status,
        "branchSwitchingMessage" => status == "applied" ?
            "Limited branch-switching probes accepted $(new_branch_count) new branch(es)." :
            status == "attempted_no_new_branches" ?
                "Limited branch-switching probes ran but did not discover distinct new branches." :
                "No eligible recorded special points were available for branch-switching probes.",
        "branchSwitchingAttemptCount" => attempt_count,
        "branchSwitchingNewBranchCount" => new_branch_count,
        "branchSwitchingSpecialPointCount" => special_count,
        "branchSwitchingDiagnostics" => diagnostics
    )
end

"""Summarize atlas neighbor-seed reuse across recovery attempts."""
function _atlas_seed_reuse_summary(config::AtlasConfig, diagnostics::Vector{Dict{String, Any}})
    if !config.reuse_neighbor_seeds
        return Dict{String, Any}(
            "neighborSeedReuseRequested" => false,
            "neighborSeedReuseApplied" => false,
            "neighborSeedReuseStatus" => "disabled",
            "neighborSeedReuseMessage" => "Neighbor seed reuse was not requested.",
            "neighborSeedReuseLookupCount" => 0,
            "neighborSeedReuseHitCount" => 0,
            "neighborSeedReuseReusedSeedCount" => 0,
            "neighborSeedReuseStoredSeedCount" => 0,
            "neighborSeedReuseDiagnostics" => diagnostics
        )
    end

    lookup_count = count(diag -> get(diag, "requested", false) == true, diagnostics)
    hit_count = count(diag -> get(diag, "hit", false) == true, diagnostics)
    reused_count = sum(_atlas_diag_int(get(diag, "reusedSeedCount", 0)) for diag in diagnostics; init=0)
    stored_count = sum(_atlas_diag_int(get(diag, "storedSeedCount", 0)) for diag in diagnostics; init=0)
    status = hit_count > 0 ? "applied" :
             lookup_count > 0 ? "active_no_hits" : "no_recovery_attempts"
    return Dict{String, Any}(
        "neighborSeedReuseRequested" => true,
        "neighborSeedReuseApplied" => hit_count > 0,
        "neighborSeedReuseStatus" => status,
        "neighborSeedReuseMessage" => status == "applied" ?
            "Neighbor seed reuse injected $(reused_count) cached seed hint(s)." :
            status == "active_no_hits" ?
                "Neighbor seed reuse was active, but no cached seeds were close enough to inject." :
                "Neighbor seed reuse was active, but no recovery attempts ran.",
        "neighborSeedReuseLookupCount" => lookup_count,
        "neighborSeedReuseHitCount" => hit_count,
        "neighborSeedReuseReusedSeedCount" => reused_count,
        "neighborSeedReuseStoredSeedCount" => stored_count,
        "neighborSeedReuseDiagnostics" => diagnostics
    )
end

"""Build a refinement-oriented brute-force config for one unresolved gap."""
function _atlas_refined_bruteforce_config(bf_config::BruteForceConfig,
                                          gap::AtlasGap,
                                          atlas_config::AtlasConfig)
    base_span = _atlas_interval_span(bf_config.param_min, bf_config.param_max)
    local_span = _atlas_interval_span(gap.param_min, gap.param_max)
    base_density = max((atlas_config.recon_steps - 1) / base_span, 1 / base_span)
    local_steps = clamp(ceil(Int, base_density * local_span * (2.0 ^ (gap.depth + 1))) + 1, 6, max(12, atlas_config.recon_steps * 4))
    return BruteForceConfig(
        param_min=min(gap.param_min, gap.param_max),
        param_max=max(gap.param_min, gap.param_max),
        param_steps=max(local_steps, 6),
        iterations=bf_config.iterations,
        transient=bf_config.transient,
        param_index=bf_config.param_index,
        fixed_params=copy(bf_config.fixed_params),
        linked_param_indices=copy(bf_config.linked_param_indices)
    )
end

"""Build a refinement-oriented continuation config for one unresolved gap."""
function _atlas_refined_continuation_config(cont_config::ContinuationConfig,
                                            gap::AtlasGap)
    return first(_atlas_continuation_retry_configs(cont_config, gap.param_min, gap.param_max, max(1, gap.depth + 2)))
end

"""Run one retry-aware recovery pass for a target window and assign recovered branches to the parent window."""
function _atlas_attempt_window_recovery(sys::DynamicalSystem,
                                        target_window::AtlasWindow,
                                        parent_window::AtlasWindow,
                                        samples::Vector{AtlasReconSample},
                                        branch_records::Vector{AtlasBranchRecord},
                                        base_params::Vector{Float64},
                                        cont_config::ContinuationConfig,
                                        atlas_config::AtlasConfig,
                                        id_counter::Base.RefValue{Int};
                                        depth::Int=0,
                                        provenance::AbstractDict{<:AbstractString, <:Any}=Dict{String, Any}(),
                                        solver=Tsit5(),
                                        reltol::Float64=1e-8,
                                        abstol::Float64=1e-8,
                                        min_crossing_time::Float64=1e-6,
                                        log::Union{Nothing, Function}=nothing,
                                        seed_cache::Union{Nothing, _AtlasSeedCache}=nothing,
                                        seed_reuse_events::Union{Nothing, Vector{Dict{String, Any}}}=nothing)
    sample_candidates = _atlas_candidate_sample_indices(target_window, samples)
    attempt_limit = min(length(sample_candidates), max(1, atlas_config.skeleton_retry_budget))
    attempts = Dict{String, Any}[]
    new_records = AtlasBranchRecord[]
    best_reason = isempty(sample_candidates) ? :no_seed : :continuation_failed

    _atlas_log!(log, "Atlas window recovery depth=$depth period=$(target_window.period) interval=[$(target_window.param_min), $(target_window.param_max)] candidateSamples=$(length(sample_candidates))")

    for attempt_idx in 1:attempt_limit
        sample_idx = sample_candidates[attempt_idx]
        padding_scale = 1.0 + 0.35 * (attempt_idx - 1)
        branches, reason, seed_data, recovery_diag = if sys isa ContinuousODE
            _recover_window_branches(sys, target_window, samples, base_params, cont_config, atlas_config;
                sample_idx=sample_idx,
                padding_scale=padding_scale,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time,
                seed_cache=seed_cache,
                seed_reuse_events=seed_reuse_events)
        else
            _recover_window_branches(sys, target_window, samples, base_params, cont_config, atlas_config;
                sample_idx=sample_idx,
                padding_scale=padding_scale,
                seed_cache=seed_cache,
                seed_reuse_events=seed_reuse_events)
        end

        best_reason = reason == :ok ? :ok : best_reason
        push!(attempts, merge(recovery_diag, Dict(
            "attemptIndex" => attempt_idx,
            "depth" => depth,
            "reason" => string(reason),
            "sampleIndex" => sample_idx,
            "sampleParam" => samples[sample_idx].param,
            "seedParam" => seed_data.skeleton_param,
            "paddingScale" => padding_scale
        )))

        for branch in branches
            existing = [record.branch for record in vcat(branch_records, new_records)]
            _is_duplicate_branch(branch, existing, 5e-3, 0.75) && continue
            branch, auto_refine_diag = _atlas_maybe_auto_refine_branch(
                sys,
                branch,
                base_params,
                cont_config,
                atlas_config;
                log=log,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time
            )
            lo, hi = _atlas_branch_param_support(branch)
            parameter_coverage = _atlas_branch_coverage(branch, parent_window)
            geometry_diagnostics = _atlas_branch_geometry_diagnostics(
                sys,
                branch,
                parent_window,
                samples,
                base_params,
                cont_config.linked_param_indices;
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time
            )
            geometry_score = Float64(get(geometry_diagnostics, "geometryCoverageScore", 1.0))
            combined_coverage = clamp(parameter_coverage * geometry_score, 0.0, 1.0)
            push!(new_records, AtlasBranchRecord(
                _atlas_next_id!(id_counter, "atlas-branch"),
                branch,
                seed_data.skeleton_param,
                parent_window.id,
                combined_coverage,
                lo,
                hi,
                merge(Dict{String, Any}(String(k) => v for (k, v) in pairs(provenance)), geometry_diagnostics, auto_refine_diag, Dict(
                    "sourceWindowId" => target_window.id,
                    "depth" => depth,
                    "attemptIndex" => attempt_idx,
                    "sampleIndex" => sample_idx,
                    "sampleParam" => samples[sample_idx].param,
                    "reason" => string(reason),
                    "parameterCoverageScore" => parameter_coverage,
                    "combinedCoverageScore" => combined_coverage,
                    "coverageScoreType" => "parameter_geometry_product",
                    "searchMin" => copy(seed_data.search_min),
                    "searchMax" => copy(seed_data.search_max),
                    "recoveryDiagnostics" => recovery_diag
                ))
            ))
        end

        local_coverage = _atlas_interval_coverage_fraction(
            target_window.param_min,
            target_window.param_max,
            vcat(branch_records, new_records),
            parent_window.id,
            parent_window.period
        )
        local_coverage >= atlas_config.coverage_threshold && break
    end

    coverage = _atlas_interval_coverage_fraction(
        target_window.param_min,
        target_window.param_max,
        vcat(branch_records, new_records),
        parent_window.id,
        parent_window.period
    )

    return new_records, best_reason, Dict(
        "attempts" => attempts,
        "attemptCount" => length(attempts),
        "depth" => depth,
        "coverage" => coverage,
        "reason" => string(best_reason)
    )
end

"""Recursively refine one unresolved gap and return newly recovered branches plus leaf gaps."""
function _atlas_refine_gap(sys::DynamicalSystem,
                           gap::AtlasGap,
                           parent_window::AtlasWindow,
                           branch_records::Vector{AtlasBranchRecord},
                           base_params::Vector{Float64},
                           bf_config::BruteForceConfig,
                           cont_config::ContinuationConfig,
                           atlas_config::AtlasConfig,
                           id_counter::Base.RefValue{Int},
                           started_at::Float64;
                           initial_point::Union{Nothing, AbstractVector}=nothing,
                            solver=Tsit5(),
                            reltol::Float64=1e-8,
                            abstol::Float64=1e-8,
                            min_crossing_time::Float64=1e-6,
                            log::Union{Nothing, Function}=nothing,
                            seed_cache::Union{Nothing, _AtlasSeedCache}=nothing,
                            seed_reuse_events::Union{Nothing, Vector{Dict{String, Any}}}=nothing)
    _atlas_should_refine_gap(gap, atlas_config, started_at) || return AtlasBranchRecord[], AtlasGap[gap], Dict(
        "gapId" => gap.id,
        "depth" => gap.depth,
        "refined" => false,
        "reason" => string(gap.reason)
    )

    _atlas_log!(log, "Atlas gap refinement depth=$(gap.depth + 1) period=$(gap.period) interval=[$(gap.param_min), $(gap.param_max)]")

    local_bf = _atlas_refined_bruteforce_config(bf_config, gap, atlas_config)
    local_cont = _atlas_refined_continuation_config(cont_config, gap)
    local_config = AtlasConfig(
        max_period=maximum([gap.period]),
        periods=[gap.period],
        brute_force=local_bf,
        continuation=local_cont,
        recon_steps=max(local_bf.param_steps, 6),
        recon_precision=min(atlas_config.recon_precision, atlas_config.recon_precision / max(1.0, 2.0 ^ gap.depth)),
        window_min_support=max(2, min(atlas_config.window_min_support, 2 + gap.depth)),
        window_merge_gap=0,
        seed_points_per_window=max(atlas_config.seed_points_per_window, 4),
        seed_box_padding=atlas_config.seed_box_padding,
        skeleton_retry_budget=atlas_config.skeleton_retry_budget,
        continuation_retry_budget=atlas_config.continuation_retry_budget,
        max_refinement_depth=atlas_config.max_refinement_depth,
        max_total_windows=atlas_config.max_total_windows,
        max_total_branches=atlas_config.max_total_branches,
        coverage_threshold=atlas_config.coverage_threshold,
        branch_switching=atlas_config.branch_switching,
        branch_switching_max_special_points=atlas_config.branch_switching_max_special_points,
        branch_switching_max_branches=atlas_config.branch_switching_max_branches,
        branch_switching_window_fraction=atlas_config.branch_switching_window_fraction,
        branch_switching_perturbation_scale=atlas_config.branch_switching_perturbation_scale,
        branch_switching_max_steps=atlas_config.branch_switching_max_steps,
        branch_switching_max_seed_candidates=atlas_config.branch_switching_max_seed_candidates,
        reuse_neighbor_seeds=atlas_config.reuse_neighbor_seeds,
        neighbor_seed_max_entries=atlas_config.neighbor_seed_max_entries,
        neighbor_seed_max_distance_fraction=atlas_config.neighbor_seed_max_distance_fraction,
        neighbor_seed_max_points=atlas_config.neighbor_seed_max_points,
        threaded=atlas_config.threaded,
        cache_enabled=false,
        time_budget_s=atlas_config.time_budget_s,
        reseed=atlas_config.reseed
    )

    local_samples = if sys isa ContinuousODE
        _atlas_reconnaissance(
            sys,
            base_params,
            local_bf,
            local_config,
            [gap.period];
            initial_point=initial_point,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time
        )
    else
        _atlas_reconnaissance(sys, base_params, local_bf, local_config, [gap.period]; initial_point=initial_point)
    end

    local_windows = [window for window in _segment_period_windows(local_samples, local_config, [gap.period]) if max(window.param_min, gap.param_min) < min(window.param_max, gap.param_max) + 100eps(Float64)]
    isempty(local_windows) && return AtlasBranchRecord[], AtlasGap[AtlasGap(
        gap.id,
        gap.period,
        gap.param_min,
        gap.param_max,
        gap.confidence,
        :low_confidence,
        gap.depth,
        false,
        merge(copy(gap.diagnostics), Dict(
            "refined" => true,
            "localWindowCount" => 0,
            "reconSampleCount" => length(local_samples)
        ))
    )], Dict(
        "gapId" => gap.id,
        "depth" => gap.depth,
        "refined" => true,
        "localWindowCount" => 0,
        "reconSampleCount" => length(local_samples)
    )

    child_records = AtlasBranchRecord[]
    leaf_gaps = AtlasGap[]
    child_diags = Dict{String, Any}[]

    local_window_intervals = Tuple{Float64, Float64}[(max(window.param_min, gap.param_min), min(window.param_max, gap.param_max)) for window in local_windows]
    for (outside_lo, outside_hi) in _atlas_uncovered_intervals(gap.param_min, gap.param_max, local_window_intervals)
        push!(leaf_gaps, AtlasGap(
            _atlas_next_id!(id_counter, "atlas-gap"),
            gap.period,
            outside_lo,
            outside_hi,
            gap.confidence,
            :low_confidence,
            gap.depth + 1,
            false,
            Dict(
                "parentGapId" => gap.id,
                "parentWindowId" => parent_window.id,
                "source" => "atlas-refinement"
            )
        ))
    end

    for window in local_windows
        target_window = AtlasWindow(
            window.id,
            window.period,
            max(window.param_min, gap.param_min),
            min(window.param_max, gap.param_max),
            window.support,
            window.mean_confidence,
            window.classification,
            copy(window.sample_indices),
            window.priority_score,
            window.status,
            merge(copy(window.diagnostics), Dict(
                "parentGapId" => gap.id,
                "parentWindowId" => parent_window.id,
                "depth" => gap.depth + 1
            ))
        )
        new_records, reason, attempt_diag = _atlas_attempt_window_recovery(
            sys,
            target_window,
            parent_window,
            local_samples,
            vcat(branch_records, child_records),
            base_params,
            local_cont,
            atlas_config,
            id_counter;
            depth=gap.depth + 1,
            provenance=Dict(
                "source" => "atlas-gap-retry",
                "parentGapId" => gap.id,
                "parentWindowId" => parent_window.id
            ),
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
            log=log,
            seed_cache=seed_cache,
            seed_reuse_events=seed_reuse_events
        )
        append!(child_records, new_records)

        covered = _atlas_interval_coverage_intervals(
            target_window.param_min,
            target_window.param_max,
            vcat(branch_records, child_records),
            parent_window.id,
            parent_window.period;
            min_geometry_score=atlas_config.coverage_threshold
        )
        uncovered = _atlas_uncovered_intervals(target_window.param_min, target_window.param_max, covered)

        for (child_lo, child_hi) in uncovered
            child_gap = AtlasGap(
                _atlas_next_id!(id_counter, "atlas-gap"),
                gap.period,
                child_lo,
                child_hi,
                target_window.mean_confidence,
                reason == :ok ? :insufficient_coverage : reason,
                gap.depth + 1,
                true,
                Dict(
                    "parentGapId" => gap.id,
                    "parentWindowId" => parent_window.id,
                    "sourceWindowId" => target_window.id,
                    "coverage" => attempt_diag["coverage"],
                    "attemptDiagnostics" => attempt_diag
                )
            )
            if _atlas_should_refine_gap(child_gap, atlas_config, started_at)
                nested_records, nested_leaf_gaps, nested_diag = _atlas_refine_gap(
                    sys,
                    child_gap,
                    parent_window,
                    vcat(branch_records, child_records),
                    base_params,
                    bf_config,
                    cont_config,
                    atlas_config,
                    id_counter,
                    started_at;
                    initial_point=initial_point,
                    solver=solver,
                    reltol=reltol,
                    abstol=abstol,
                    min_crossing_time=min_crossing_time,
                    log=log,
                    seed_cache=seed_cache,
                    seed_reuse_events=seed_reuse_events
                )
                append!(child_records, nested_records)
                append!(leaf_gaps, nested_leaf_gaps)
                push!(child_diags, nested_diag)
            else
                push!(leaf_gaps, AtlasGap(
                    child_gap.id,
                    child_gap.period,
                    child_gap.param_min,
                    child_gap.param_max,
                    child_gap.confidence,
                    child_gap.reason,
                    child_gap.depth,
                    false,
                    merge(copy(child_gap.diagnostics), Dict("retryable" => false))
                ))
            end
        end

        push!(child_diags, merge(attempt_diag, Dict(
            "windowId" => target_window.id,
            "reason" => string(reason)
        )))
    end

    return child_records, leaf_gaps, Dict(
        "gapId" => gap.id,
        "depth" => gap.depth,
        "refined" => true,
        "localWindowCount" => length(local_windows),
        "reconSampleCount" => length(local_samples),
        "children" => child_diags
    )
end

"""Summarize coverage across candidate windows."""
function _atlas_coverage_summary(windows::Vector{AtlasWindow}, branch_records::Vector{AtlasBranchRecord}, threshold::Float64)
    covered = 0
    partial = 0
    unresolved = 0
    by_period = Dict{String, Dict{String, Int}}()
    coverage_by_window = Dict{String, Float64}()

    for window in windows
        coverage = _atlas_window_coverage_fraction(window, branch_records)
        coverage_by_window[window.id] = coverage
        if coverage >= threshold
            covered += 1
            label = "covered"
        elseif coverage > 0
            partial += 1
            label = "partial"
        else
            unresolved += 1
            label = "unresolved"
        end
        key = string(window.period)
        period_summary = get!(by_period, key, Dict("covered" => 0, "partial" => 0, "unresolved" => 0))
        period_summary[label] = get(period_summary, label, 0) + 1
    end

    return Dict(
        "covered" => covered,
        "partial" => partial,
        "unresolved" => unresolved,
        "windowCount" => length(windows),
        "branchCount" => length(branch_records),
        "byPeriod" => by_period,
        "coverageByWindow" => coverage_by_window
    )
end

"""Build unresolved-gap records for windows not yet adequately covered."""
function _atlas_gap_records(windows::AbstractVector{AtlasWindow},
                            branch_records::AbstractVector{AtlasBranchRecord},
                            threshold::Float64,
                            max_refinement_depth::Int)
    gaps = AtlasGap[]
    for window in windows
        coverage_intervals = _atlas_interval_coverage_intervals(
            window.param_min,
            window.param_max,
            branch_records,
            window.id,
            window.period;
            min_geometry_score=threshold
        )
        coverage = _atlas_window_coverage_fraction(window, branch_records)
        if coverage >= threshold
            continue
        end
        reason = coverage > 0 ? :insufficient_coverage : Symbol(get(window.diagnostics, "recoveryReason", "continuation_failed"))
        retryable = window.mean_confidence >= min(max(threshold / 2, 0.1), 0.6) && max_refinement_depth > 0
        for (gap_lo, gap_hi) in _atlas_uncovered_intervals(window.param_min, window.param_max, coverage_intervals)
            push!(gaps, AtlasGap(
                "atlas-gap-$(window.period)-$(length(gaps) + 1)",
                window.period,
                gap_lo,
                gap_hi,
                window.mean_confidence,
                reason,
                0,
                retryable,
                Dict("windowId" => window.id, "coverage" => coverage)
            ))
        end
    end
    return gaps
end

"""
    continuation_atlas(sys, config; kwargs...) -> AtlasResult

Run the phase-2 automatic continuation atlas workflow: reconnaissance sweep, candidate-window
segmentation, brute-force-derived seeding, targeted continuation recovery, and recursive
gap refinement.
"""
function continuation_atlas(sys::DynamicalSystem,
                            config::AtlasConfig;
                            initial_point::Union{Nothing, AbstractVector}=nothing,
                            params::Vector{Float64}=Float64[],
                            solver=Tsit5(),
                            reltol::Float64=1e-8,
                            abstol::Float64=1e-8,
                            min_crossing_time::Float64=1e-6,
                            log::Union{Nothing, Function}=nothing,
                            cache_key::Union{Nothing, AbstractString}=nothing,
                            cache_file::Union{Nothing, AbstractString}=nothing,
                            cache_enabled::Bool=config.cache_enabled)
    started_at = time()
    cache_path = cache_enabled ? _atlas_cache_path(cache_file) : nothing
    if cache_enabled && !isnothing(cache_path) && isfile(cache_path)
        try
            return _load_cached_atlas_result(cache_path; cache_key=cache_key, log=log)
        catch err
            bt = catch_backtrace()
            _atlas_log!(log, "Atlas cache load failed for '$cache_path'; recomputing. $(sprint(io -> showerror(io, err, bt)))")
        end
    elseif cache_enabled && !isnothing(cache_path)
        _atlas_log!(log, "Atlas cache miss: no cached result at '$cache_path'.")
    elseif cache_enabled && isnothing(cache_path) && (!isnothing(cache_key) || !isnothing(cache_file))
        _atlas_log!(log, "Atlas cache is enabled but no usable cache file path was provided; running without library cache reuse.")
    end

    periods = _atlas_requested_periods(config)
    bf_config = _atlas_bruteforce_config(config, periods)
    cont_config = _atlas_continuation_config(config, bf_config)
    base_params = _atlas_base_params(sys, params, bf_config, cont_config)
    # Pre-flight notice only; the authoritative requested/applied status is
    # reported by the branch-switching / seed-reuse summaries built after the run.
    if config.branch_switching
        _atlas_log!(log, "Limited atlas branch-switching follow-up probes will run if recovered branches expose eligible special points.")
    end
    if config.reuse_neighbor_seeds
        _atlas_log!(log, "Atlas neighbor-seed reuse will inject cached nearby skeleton seeds when compatible seeds are available.")
    end

    _atlas_log!(log, "Atlas reconnaissance started for periods=$(periods) over [$(bf_config.param_min), $(bf_config.param_max)]")
    recon_samples = if sys isa ContinuousODE
        _atlas_reconnaissance(
            sys,
            base_params,
            bf_config,
            config,
            periods;
            initial_point=initial_point,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time
        )
    else
        _atlas_reconnaissance(sys, base_params, bf_config, config, periods; initial_point=initial_point)
    end
    recon_samples, adaptive_recon_diag = _atlas_adaptive_reconnaissance(
        sys,
        recon_samples,
        base_params,
        bf_config,
        config,
        periods;
        initial_point=initial_point,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=min_crossing_time
    )
    if get(adaptive_recon_diag, "applied", false)
        _atlas_log!(log, "Atlas adaptive reconnaissance inserted $(adaptive_recon_diag["adaptiveSampleCount"]) sample(s) across $(adaptive_recon_diag["passCount"]) pass(es).")
    end
    recon_periodic = count(sample -> sample.classification == :periodic, recon_samples)
    _atlas_log!(log, "Atlas reconnaissance classified $(length(recon_samples)) sample(s): periodic=$(recon_periodic), nonperiodic=$(count(sample -> sample.classification == :nonperiodic, recon_samples)), insufficient=$(count(sample -> sample.classification == :insufficient, recon_samples)).")

    windows = _segment_period_windows(recon_samples, config, periods)
    probe_windows = _atlas_hidden_period_probe_windows(recon_samples, config, periods, windows)
    windows = sort(vcat(windows, probe_windows); by=window -> (-window.priority_score, window.period, window.param_min))
    length(windows) > config.max_total_windows && resize!(windows, config.max_total_windows)
    _atlas_log!(log, "Atlas reconnaissance found $(length(windows)) candidate window(s) ($(length(probe_windows)) hidden-period probe window(s)).")

    bf_result, brute_force_reuse_diag = _atlas_bruteforce_result(
        sys,
        bf_config,
        config,
        periods,
        recon_samples;
        initial_point=initial_point,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        log=log
    )

    branch_records = AtlasBranchRecord[]
    resolved_windows = AtlasWindow[]
    gaps = AtlasGap[]
    id_counter = Ref(1)
    refinement_attempts = 0
    max_depth_reached = 0
    window_attempt_count = 0
    time_budget_exhausted = false
    branch_switching_diags = Dict{String, Any}[]
    seed_reuse_diags = Dict{String, Any}[]
    seed_cache = config.reuse_neighbor_seeds ? _atlas_seed_cache() : nothing

    for window in windows
        new_records, reason, attempt_diag = _atlas_attempt_window_recovery(
            sys,
            window,
            window,
            recon_samples,
            branch_records,
            base_params,
            cont_config,
            config,
            id_counter;
            depth=0,
            provenance=Dict("source" => "atlas-initial"),
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            min_crossing_time=min_crossing_time,
            log=log,
            seed_cache=seed_cache,
            seed_reuse_events=seed_reuse_diags
        )
        append!(branch_records, new_records)
        window_attempt_count += get(attempt_diag, "attemptCount", 0)
        branch_switching_diag = Dict{String, Any}()

        if config.branch_switching
            switch_records, branch_switching_diag = _atlas_branch_switching_followups(
                sys,
                window,
                recon_samples,
                branch_records,
                base_params,
                cont_config,
                config,
                id_counter;
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                min_crossing_time=min_crossing_time
            )
            append!(branch_records, switch_records)
            push!(branch_switching_diags, branch_switching_diag)
        end

        covered_intervals = _atlas_interval_coverage_intervals(
            window.param_min,
            window.param_max,
            branch_records,
            window.id,
            window.period;
            min_geometry_score=config.coverage_threshold
        )
        uncovered_intervals = _atlas_uncovered_intervals(window.param_min, window.param_max, covered_intervals)
        leaf_gap_count_before = length(gaps)
        refinement_diags = Dict{String, Any}[]

        for (gap_lo, gap_hi) in uncovered_intervals
            gap = AtlasGap(
                _atlas_next_id!(id_counter, "atlas-gap"),
                window.period,
                gap_lo,
                gap_hi,
                window.mean_confidence,
                reason == :ok ? :insufficient_coverage : reason,
                0,
                true,
                Dict(
                    "windowId" => window.id,
                    "coverage" => attempt_diag["coverage"],
                    "attemptDiagnostics" => attempt_diag
                )
            )

            if _atlas_should_refine_gap(gap, config, started_at)
                refinement_attempts += 1
                new_gap_records, leaf_gaps, refine_diag = _atlas_refine_gap(
                    sys,
                    gap,
                    window,
                    branch_records,
                    base_params,
                    bf_config,
                    cont_config,
                    config,
                    id_counter,
                    started_at;
                    initial_point=initial_point,
                    solver=solver,
                    reltol=reltol,
                    abstol=abstol,
                    min_crossing_time=min_crossing_time,
                    log=log,
                    seed_cache=seed_cache,
                    seed_reuse_events=seed_reuse_diags
                )
                append!(branch_records, new_gap_records)
                append!(gaps, leaf_gaps)
                push!(refinement_diags, refine_diag)
                max_depth_reached = max(max_depth_reached, maximum((leaf.depth for leaf in leaf_gaps); init=0), get(refine_diag, "depth", 0))
            else
                push!(gaps, AtlasGap(
                    gap.id,
                    gap.period,
                    gap.param_min,
                    gap.param_max,
                    gap.confidence,
                    gap.reason,
                    gap.depth,
                    false,
                    merge(copy(gap.diagnostics), Dict("retryable" => false))
                ))
            end

            if _atlas_time_budget_exhausted(config, started_at)
                time_budget_exhausted = true
                break
            end
        end

        coverage = _atlas_window_coverage_fraction(window, branch_records)
        status = coverage >= config.coverage_threshold ? :recovered : coverage > 0 ? :partial : :failed
        window_diagnostics = merge(copy(window.diagnostics), Dict(
            "attempts" => attempt_diag["attempts"],
            "attemptCount" => attempt_diag["attemptCount"],
            "coverage" => coverage,
            "recoveryReason" => string(reason),
            "refinementDiagnostics" => refinement_diags,
            "leafGapCount" => length(gaps) - leaf_gap_count_before
        ))
        config.branch_switching && (window_diagnostics["branchSwitching"] = branch_switching_diag)
        push!(resolved_windows, AtlasWindow(
            window.id,
            window.period,
            window.param_min,
            window.param_max,
            window.support,
            window.mean_confidence,
            window.classification,
            copy(window.sample_indices),
            window.priority_score,
            status,
            window_diagnostics
        ))

        if time_budget_exhausted
            _atlas_log!(log, "Atlas time budget exhausted after processing window $(window.id).")
            break
        end
    end

    isempty(gaps) && (gaps = _atlas_gap_records(resolved_windows, branch_records, config.coverage_threshold, config.max_refinement_depth))
    coverage_summary = _atlas_coverage_summary(resolved_windows, branch_records, config.coverage_threshold)
    branch_switching_summary = _atlas_branch_switching_summary(config, branch_switching_diags)
    seed_reuse_summary = _atlas_seed_reuse_summary(config, seed_reuse_diags)

    result = AtlasResult(
        bf_result,
        recon_samples,
        resolved_windows,
        branch_records,
        gaps,
        coverage_summary,
        sys.name,
        bf_result.param_name,
        now(),
        merge(branch_switching_summary, seed_reuse_summary, Dict(
            "periods" => periods,
            "cacheEnabled" => cache_enabled,
            "cacheKey" => isnothing(cache_key) ? nothing : String(cache_key),
            "cacheFile" => cache_path,
            "cacheHit" => false,
            "elapsedSeconds" => _atlas_elapsed_seconds(started_at),
            "windowAttemptCount" => window_attempt_count,
            "refinementAttempts" => refinement_attempts,
            "maxRefinementDepthReached" => max_depth_reached,
            "timeBudgetExceeded" => time_budget_exhausted,
            "adaptiveRecon" => adaptive_recon_diag,
            "bruteForceReusedFromRecon" => get(brute_force_reuse_diag, "reused", false),
            "bruteForceReuse" => brute_force_reuse_diag
        ))
    )

    _atlas_log!(log, "Atlas recovery summary: windows=$(length(resolved_windows)), branches=$(length(branch_records)), gaps=$(length(gaps)), covered=$(get(coverage_summary, "covered", 0)), partial=$(get(coverage_summary, "partial", 0)), unresolved=$(get(coverage_summary, "unresolved", 0)).")

    if cache_enabled && !isnothing(cache_path)
        try
            _store_cached_atlas_result(result, cache_path; cache_key=cache_key, log=log)
        catch err
            bt = catch_backtrace()
            _atlas_log!(log, "WARNING: Atlas cache store failed for '$cache_path'. $(sprint(io -> showerror(io, err, bt)))")
        end
    end

    return result
end
