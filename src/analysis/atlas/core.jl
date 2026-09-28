"""
Automatic continuation atlas: a best-effort workflow that uses reconnaissance sweeps to
detect likely periodic windows, derives local seed points from the observed orbit clouds,
launches targeted continuation attempts, and adaptively refines unresolved high-confidence
gaps.
"""

struct AtlasReconSample
    param::Float64
    classification::Symbol
    best_period::Int
    confidence::Float64
    closure_errors::Vector{Float64}
    support_points::Vector{Vector{Float64}}
    orbit_center::Vector{Float64}
    orbit_span::Vector{Float64}
    diagnostics::Dict{String, Any}
end

struct AtlasWindow
    id::String
    period::Int
    param_min::Float64
    param_max::Float64
    support::Int
    mean_confidence::Float64
    classification::Symbol
    sample_indices::Vector{Int}
    priority_score::Float64
    status::Symbol
    diagnostics::Dict{String, Any}
end

struct AtlasBranchRecord
    id::String
    branch::BranchResult
    seed_param::Float64
    window_id::String
    coverage_score::Float64
    param_min::Float64
    param_max::Float64
    diagnostics::Dict{String, Any}
end

struct AtlasGap
    id::String
    period::Int
    param_min::Float64
    param_max::Float64
    confidence::Float64
    reason::Symbol
    depth::Int
    retryable::Bool
    diagnostics::Dict{String, Any}
end

struct AtlasResult
    brute_force::BruteForceResult
    recon_samples::Vector{AtlasReconSample}
    windows::Vector{AtlasWindow}
    branch_records::Vector{AtlasBranchRecord}
    gaps::Vector{AtlasGap}
    coverage_summary::Dict{String, Any}
    system_name::String
    param_name::Symbol
    timestamp::DateTime
    diagnostics::Dict{String, Any}
end

const _AtlasSeedEntry = NamedTuple{(:param, :point, :stamp), Tuple{Float64, Vector{Float64}, Int}}
const _AtlasSeedCache = Dict{Int, Vector{_AtlasSeedEntry}}

"""Convenience accessor for the recovered continuation branches in an atlas result."""
atlas_branches(result::AtlasResult) = [record.branch for record in result.branch_records]

"""Emit a workbench/library log message when a logger callback is available."""
function _atlas_log!(log::Union{Nothing, Function}, message::AbstractString)
    isnothing(log) || log(String(message))
    return nothing
end

"""Return an atlas result with merged diagnostic metadata updates."""
function _atlas_with_diagnostics(result::AtlasResult, updates::AbstractDict{<:AbstractString, <:Any})
    return AtlasResult(
        result.brute_force,
        result.recon_samples,
        result.windows,
        result.branch_records,
        result.gaps,
        result.coverage_summary,
        result.system_name,
        result.param_name,
        result.timestamp,
        merge(copy(result.diagnostics), Dict{String, Any}(String(k) => _plain(v) for (k, v) in pairs(updates)))
    )
end

"""Return the atlas library cache file path when one was explicitly supplied."""
function _atlas_cache_path(cache_file::Union{Nothing, AbstractString})
    isnothing(cache_file) && return nothing
    path = String(cache_file)
    isempty(path) && return nothing
    return path
end

"""Load a cached atlas result when available and consistent with the requested cache key."""
function _load_cached_atlas_result(cache_path::String;
                                   cache_key::Union{Nothing, AbstractString}=nothing,
                                   log::Union{Nothing, Function}=nothing)
    payload = JLD2.load(cache_path)
    brute_force_data = get(payload, "brute_force", nothing)
    atlas_data = get(payload, "atlas_result", nothing)
    brute_force_data isa AbstractDict || error("Atlas cache file '$cache_path' does not contain a serialized brute-force payload.")
    atlas_data isa AbstractDict || error("Atlas cache file '$cache_path' does not contain a serialized atlas payload.")
    brute_force = _deserialize_bruteforce_result(_jsonish_dict(brute_force_data))
    result = _deserialize_atlas_result(_jsonish_dict(atlas_data); brute_force=brute_force)
    metadata = _jsonish_dict(get(payload, "metadata", Dict{String, Any}()))
    requested_key = isnothing(cache_key) ? nothing : String(cache_key)
    stored_key = get(metadata, "cacheKey", get(result.diagnostics, "cacheKey", nothing))
    if !isnothing(requested_key) && !isnothing(stored_key) && String(stored_key) != requested_key
        error("Atlas cache key mismatch for '$cache_path': expected '$requested_key', found '$(stored_key)'.")
    end
    _atlas_log!(log, "Atlas cache hit: loaded cached result from '$cache_path'.")
    return _atlas_with_diagnostics(result, Dict(
        "cacheEnabled" => true,
        "cacheKey" => requested_key,
        "cacheFile" => cache_path,
        "cacheHit" => true,
        "cacheLoadedAt" => string(now())
    ))
end

"""Persist an atlas result to a library cache file when requested."""
function _store_cached_atlas_result(result::AtlasResult,
                                    cache_path::String;
                                    cache_key::Union{Nothing, AbstractString}=nothing,
                                    log::Union{Nothing, Function}=nothing)
    dir = dirname(cache_path)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    metadata = Dict(
        "cacheKey" => isnothing(cache_key) ? nothing : String(cache_key),
        "cachedAt" => string(now()),
        "systemName" => result.system_name,
        "paramName" => String(result.param_name),
        "windowCount" => length(result.windows),
        "branchCount" => length(result.branch_records),
        "gapCount" => length(result.gaps)
    )
    jldsave(
        cache_path;
        brute_force=_serialize_bruteforce_result(result.brute_force),
        atlas_result=_serialize_atlas_result(result),
        metadata=metadata
    )
    _atlas_log!(log, "Atlas cache store: wrote result to '$cache_path'.")
    return cache_path
end

"""Return elapsed wall-clock seconds since `started_at`."""
_atlas_elapsed_seconds(started_at::Float64) = max(0.0, time() - started_at)

"""Return whether the atlas wall-clock budget has been exhausted."""
function _atlas_time_budget_exhausted(config::AtlasConfig, started_at::Float64)
    isnothing(config.time_budget_s) && return false
    return _atlas_elapsed_seconds(started_at) >= config.time_budget_s
end

"""
Budget state for one atlas run: the wall-clock start, the number of window-recovery searches
made so far, and whether either budget refused a search that would otherwise have run. One
attempt is one window-recovery search, whether for a reconnaissance window or for a local
window inside a gap refinement. Attempts advance in the atlas's serial window and gap order,
so the work budget stops a run at the same point whatever the machine or thread count.
"""
mutable struct _AtlasBudget
    started_at::Float64
    attempts::Int
    work_refused::Bool
    time_refused::Bool
end
_AtlasBudget(started_at::Float64=time()) = _AtlasBudget(started_at, 0, false, false)

_atlas_elapsed_seconds(budget::_AtlasBudget) = _atlas_elapsed_seconds(budget.started_at)
_atlas_time_budget_exhausted(config::AtlasConfig, budget::_AtlasBudget) =
    _atlas_time_budget_exhausted(config, budget.started_at)

"""Return whether the deterministic recovery-attempt budget has been spent."""
function _atlas_work_budget_exhausted(config::AtlasConfig, budget::_AtlasBudget)
    isnothing(config.max_recovery_attempts) && return false
    return budget.attempts >= config.max_recovery_attempts
end

"""
Return which budget refuses the next recovery search (`:time_budget`, `:work_budget`), or
`nothing` when it may run. A refusal is recorded on the budget, so the result flags a budget only
when it actually cut work.
"""
function _atlas_budget_refusal!(config::AtlasConfig, budget::_AtlasBudget)
    if _atlas_time_budget_exhausted(config, budget)
        budget.time_refused = true
        return :time_budget
    elseif _atlas_work_budget_exhausted(config, budget)
        budget.work_refused = true
        return :work_budget
    end
    return nothing
end

"""Take one recovery attempt from the budget; returns the refusal reason, or `nothing` when taken."""
function _atlas_take_attempt!(config::AtlasConfig, budget::_AtlasBudget)
    refusal = _atlas_budget_refusal!(config, budget)
    isnothing(refusal) && (budget.attempts += 1)
    return refusal
end

"""Return a stable unique identifier for atlas branch/gap records."""
function _atlas_next_id!(counter::Base.RefValue{Int}, prefix::AbstractString)
    id = "$(prefix)-$(counter[])"
    counter[] += 1
    return id
end

"""Return the span of a parameter interval, guarded away from zero."""
_atlas_interval_span(param_min::Float64, param_max::Float64) = max(abs(param_max - param_min), eps(Float64))

"""Normalize the requested periods, defaulting to `1:max_period` when none are supplied."""
function _atlas_requested_periods(config::AtlasConfig)
    periods = isempty(config.periods) ? collect(1:max(config.max_period, 1)) : sort(unique(filter(>(0), config.periods)))
    isempty(periods) && error("AtlasConfig must target at least one positive period.")
    return periods
end

"""Resolve the brute-force configuration for an atlas run."""
function _atlas_bruteforce_config(config::AtlasConfig, periods::AbstractVector{Int})
    !isnothing(config.brute_force) && return config.brute_force
    error("AtlasConfig.brute_force must be provided in the current continuation_atlas implementation.")
end

"""Resolve the continuation configuration for an atlas run."""
function _atlas_continuation_config(config::AtlasConfig, bf_config::BruteForceConfig)
    if !isnothing(config.continuation)
        return config.continuation
    end
    return ContinuationConfig(
        p_min=bf_config.param_min,
        p_max=bf_config.param_max,
        param_index=bf_config.param_index,
        linked_param_indices=copy(bf_config.linked_param_indices)
    )
end

"""Resolve the base parameter vector used for atlas sampling and continuation."""
function _atlas_base_params(sys::DynamicalSystem,
                            params::Vector{Float64},
                            bf_config::BruteForceConfig,
                            cont_config::ContinuationConfig)
    required_len = maximum(vcat(
        [bf_config.param_index, cont_config.param_index, 1],
        bf_config.linked_param_indices,
        cont_config.linked_param_indices,
        sys isa ContinuousODE ? [length(sys.default_params)] : Int[],
        [length(params), length(bf_config.fixed_params)]
    ))
    base = if !isempty(params)
        copy(params)
    elseif !isempty(bf_config.fixed_params)
        copy(bf_config.fixed_params)
    elseif sys isa ContinuousODE && !isempty(sys.default_params)
        copy(sys.default_params)
    else
        zeros(Float64, required_len)
    end
    if length(base) < required_len
        old_len = length(base)
        resize!(base, required_len)
        fill!(@view(base[(old_len + 1):required_len]), 0.0)
    end
    return base
end

