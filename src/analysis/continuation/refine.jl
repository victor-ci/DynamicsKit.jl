"""Recover all recorded section/state coordinates from a continuation point."""
function _branch_point_state(point)
    coords = Float64[]
    for name in propertynames(point)
        startswith(String(name), "x") || continue
        suffix = tryparse(Int, String(name)[2:end])
        isnothing(suffix) && continue
        push!(coords, Float64(getproperty(point, name)))
    end
    return coords
end

"""Distance between two recorded continuation points in state space."""
function _branch_point_distance(a, b)
    ax = _branch_point_state(a)
    bx = _branch_point_state(b)
    length(ax) == length(bx) || return Inf
    isempty(ax) && return abs(a.param - b.param)
    return norm(ax .- bx)
end

"""Return the recorded `x1`, `x2`, … field names sorted by coordinate index."""
function _branch_point_coordinate_fields(point)
    fields = Tuple(
        name for name in propertynames(point)
        if startswith(String(name), "x") && !isnothing(tryparse(Int, String(name)[2:end]))
    )
    return sort(collect(fields); by=name -> tryparse(Int, String(name)[2:end]))
end

"""Rebuild one continuation point with its recorded state coordinates replaced."""
function _branch_point_with_state(point, state::AbstractVector{<:Real})
    coord_fields = _branch_point_coordinate_fields(point)
    length(coord_fields) == length(state) || return point
    fields = propertynames(point)
    coord_lookup = Dict(field => idx for (idx, field) in pairs(coord_fields))
    values = ntuple(length(fields)) do idx
        field = fields[idx]
        coord_idx = get(coord_lookup, field, 0)
        coord_idx == 0 ? getproperty(point, field) : Float64(state[coord_idx])
    end
    return NamedTuple{fields}(values)
end

"""Return the period-`N` orbit through a discrete-map continuation point."""
function _branch_point_orbit(sys::DiscreteMap,
                             point,
                             period::Int,
                             params::AbstractVector)
    dim = sys.dim
    state = _branch_point_state(point)
    length(state) == dim || return Vector{Vector{Float64}}()
    current = SVector{dim, Float64}(state)
    orbit = Vector{Float64}[]
    for phase in 1:max(period, 1)
        push!(orbit, collect(Float64, current))
        phase == period && continue
        current = sys.f(current, params)
        all(isfinite, current) || return Vector{Vector{Float64}}()
    end
    return orbit
end

"""Rotate an orbit so its first phase is deterministic across equivalent shifts."""
function _canonical_orbit_shift(orbit::AbstractVector{<:AbstractVector{<:Real}})
    period = length(orbit)
    period <= 1 && return 0

    # Lexicographically compare the rotation starting at `shift_a` against `shift_b`, walking the
    # orbit phases and coordinates in place. This avoids materializing a rotated orbit and a
    # flattened key per candidate shift, keeping the search allocation-free.
    function rotation_precedes(shift_a::Int, shift_b::Int)
        shift_a == shift_b && return false
        for phase in 1:period
            a = orbit[mod1(phase + shift_a, period)]
            b = orbit[mod1(phase + shift_b, period)]
            for k in eachindex(a)
                a[k] < b[k] && return true
                a[k] > b[k] && return false
            end
        end
        return false
    end

    best_shift = 0
    for shift in 1:(period - 1)
        rotation_precedes(shift, best_shift) && (best_shift = shift)
    end
    return best_shift
end

"""Canonicalize high-period discrete-map branch representatives so `x1`,`x2`,… stay phase-consistent."""
function _canonicalize_branch_representatives(sys::DiscreteMap,
                                              branch::BranchResult,
                                              base_params::AbstractVector,
                                              linked_param_indices::Vector{Int})
    branch.period <= 1 && return branch
    points = _branch_points(branch)
    isempty(points) && return branch
    param_index = findfirst(==(branch.param_name), sys.param_names)
    isnothing(param_index) && return branch

    base = collect(Float64, base_params)
    aligned_points = Any[]
    previous_orbit = nothing
    changed = false

    for point in points
        local_params = _inject_param(base, param_index, Float64(point.param), linked_param_indices)
        orbit = _branch_point_orbit(sys, point, branch.period, local_params)
        isempty(orbit) && (push!(aligned_points, point); previous_orbit = nothing; continue)
        shift = isnothing(previous_orbit) ? _canonical_orbit_shift(orbit) : _orbit_phase_alignment_shift(previous_orbit, orbit)
        aligned_orbit = shift == 0 ? orbit : [orbit[mod1(idx + shift, length(orbit))] for idx in 1:length(orbit)]
        aligned_point = _branch_point_with_state(point, aligned_orbit[1])
        changed |= aligned_point != point
        push!(aligned_points, aligned_point)
        previous_orbit = aligned_orbit
    end

    special_points = try
        collect(branch.branch.specialpoint)
    catch
        Any[]
    end
    aligned_specials = Any[]
    for point in special_points
        hasproperty(point, :param) || (push!(aligned_specials, point); continue)
        local_params = _inject_param(base, param_index, Float64(point.param), linked_param_indices)
        orbit = _branch_point_orbit(sys, point, branch.period, local_params)
        isempty(orbit) && (push!(aligned_specials, point); continue)
        shift = _canonical_orbit_shift(orbit)
        aligned_orbit = shift == 0 ? orbit : [orbit[mod1(idx + shift, length(orbit))] for idx in 1:length(orbit)]
        aligned_point = _branch_point_with_state(point, aligned_orbit[1])
        changed |= aligned_point != point
        push!(aligned_specials, aligned_point)
    end

    changed || return branch
    return BranchResult(
        CombinedBranchResult(Vector{Any}(aligned_points), Vector{Any}(aligned_specials)),
        branch.period,
        branch.system_name,
        branch.param_name,
        branch.timestamp
    )
end

_canonicalize_branch_representatives(::ContinuousODE, branch::BranchResult, _base_params::AbstractVector, _linked_param_indices::Vector{Int}) = branch

# Tolerances for minimal-period classification / branch trimming. The orbit
# closure error at a genuine period-N point is ~Newton tol (≤1e-8), while a
# lower-period orbit point sits O(1) away after m<N applications — so these
# thresholds have wide margin. Continuous uses the looser value because each
# return carries the integrator's reltol. They are deliberately not the
# signature-dedup tolerances (those compare *distinct* branches, not an orbit
# against itself).
const _PERIOD_TRIM_PRECISION_DISCRETE = 1e-4
const _PERIOD_TRIM_PRECISION_CONTINUOUS = 1e-3
# Parameter padding when carrying special points into a trimmed branch.
const _PERIOD_TRIM_PARAM_TOL = 1e-12
# Upper bound on per-point Poincaré classifications for a *continuous* branch:
# above this the branch is classified on a uniform subsample and the remaining
# points inherit the nearest classified neighbour, bounding integration cost.
# Discrete maps are cheap and always classify every point exactly.
const _PERIOD_TRIM_MAX_CONTINUOUS_CHECKS = 400

function _set_period_trim_diagnostics!(target::Union{Nothing, Base.RefValue}, diagnostics::Dict{String, Any})
    isnothing(target) || (target[] = diagnostics)
    return nothing
end

function _param_intervals_for_indices(points::AbstractVector, indices::AbstractVector{Int})
    isempty(indices) && return Dict{String, Any}[]
    intervals = Dict{String, Any}[]
    start_idx = first(indices)
    prev_idx = first(indices)
    for idx in Iterators.drop(indices, 1)
        if idx == prev_idx + 1
            prev_idx = idx
            continue
        end
        params = Float64[Float64(points[i].param) for i in start_idx:prev_idx]
        push!(intervals, Dict{String, Any}(
            "indexStart" => start_idx,
            "indexEnd" => prev_idx,
            "paramMin" => minimum(params),
            "paramMax" => maximum(params)
        ))
        start_idx = idx
        prev_idx = idx
    end
    params = Float64[Float64(points[i].param) for i in start_idx:prev_idx]
    push!(intervals, Dict{String, Any}(
        "indexStart" => start_idx,
        "indexEnd" => prev_idx,
        "paramMin" => minimum(params),
        "paramMax" => maximum(params)
    ))
    return intervals
end

function _minimal_period_counts(minimal_periods::AbstractVector{Int}, requested_period::Int)
    counts = Dict{String, Int}()
    for p in minimal_periods
        p == requested_period && continue
        key = string(p)
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function _minimal_period_trim_diagnostics(branch::BranchResult;
                                          reason::AbstractString,
                                          applied::Bool=false,
                                          point_count::Int=length(_branch_points(branch)),
                                          kept_count::Int=point_count,
                                          dropped_count::Int=max(0, point_count - kept_count),
                                          kept_intervals::Vector{Dict{String, Any}}=Dict{String, Any}[],
                                          dropped_intervals::Vector{Dict{String, Any}}=Dict{String, Any}[],
                                          lower_period_counts::Dict{String, Int}=Dict{String, Int}(),
                                          precision::Union{Nothing, Float64}=nothing,
                                          sampled_classification::Bool=false,
                                          evaluated_count::Int=point_count)
    return Dict{String, Any}(
        "requestedPeriod" => branch.period,
        "applied" => applied,
        "reason" => String(reason),
        "pointCount" => point_count,
        "keptCount" => kept_count,
        "droppedCount" => dropped_count,
        "keptIntervals" => kept_intervals,
        "droppedIntervals" => dropped_intervals,
        "lowerPeriods" => lower_period_counts,
        "precision" => precision,
        "sampledClassification" => sampled_classification,
        "evaluatedCount" => evaluated_count
    )
end

function _finalize_minimal_period_trim(sys::DynamicalSystem,
                                       branch::BranchResult,
                                       base_params::AbstractVector,
                                       linked_param_indices::Vector{Int};
                                       trim_to_minimal_period::Bool=false,
                                       on_trim::Union{Nothing, Function}=nothing,
                                       kwargs...)
    trim_to_minimal_period || return _canonicalize_branch_representatives(sys, branch, base_params, linked_param_indices)
    diag_ref = Ref{Any}(nothing)
    trimmed = _trim_branch_to_period(
        sys,
        branch,
        base_params,
        linked_param_indices;
        trim_diagnostics=diag_ref,
        kwargs...
    )
    isnothing(on_trim) || on_trim(diag_ref[])
    if isnothing(trimmed)
        dropped = diag_ref[] isa AbstractDict ? get(diag_ref[], "droppedCount", length(_branch_points(branch))) : length(_branch_points(branch))
        error("Minimal-period trimming removed all $dropped points from the period-$(branch.period) branch of $(branch.system_name); the continued curve is a lower-period alias.")
    end
    return _canonicalize_branch_representatives(sys, trimmed, base_params, linked_param_indices)
end

"""
Minimal period of the orbit through the recorded fixed point `state` under the
discrete map: the smallest `m ∈ 1:period` with `Pᵐ(state) ≈ state` (relative
tolerance). Because a period-`N` orbit is also a fixed point of `Pᴺ` at every
divisor period, a branch continued as `Pᴺ` can silently follow a lower-period
orbit; this lets callers detect that. Returns `period` if no earlier match (the
point is genuinely period-`period`, or too loosely converged to judge).
"""
function _orbit_minimal_period(sys::DiscreteMap, state::AbstractVector, params::AbstractVector,
                               period::Int; precision::Float64=_PERIOD_TRIM_PRECISION_DISCRETE)
    period <= 1 && return 1
    length(state) == sys.dim || return period
    x0 = SVector{sys.dim, Float64}(state)
    sv = x0
    base = max(norm(x0), 1.0)
    for m in 1:period
        sv = sys.f(sv, params)
        all(isfinite, sv) || return period
        norm(sv .- x0) < precision * max(base, norm(sv)) && return m
    end
    return period
end

"""
Continuous analogue: iterate the Poincaré return map up to `period` times from
the recorded section point and return the smallest `m` whose `m`-th return lands
back on the start (so a `Pᴺ` branch that has degenerated onto a lower-period
orbit is detected). One integration of `period` crossings per call.
"""
function _orbit_minimal_period(sys::ContinuousODE, state::AbstractVector, params::AbstractVector,
                               period::Int; precision::Float64=_PERIOD_TRIM_PRECISION_CONTINUOUS, solver=Tsit5(),
                               reltol::Float64=1e-8, abstol::Float64=1e-8,
                               min_crossing_time::Float64=1e-6)
    period <= 1 && return 1
    lifted = _lift_section_state(sys.section, state, sys.dim)
    pts = _collect_poincare_points(sys, collect(Float64, params);
                                   initial_point=lifted, crossings=period, transient=0,
                                   solver=solver, reltol=reltol, abstol=abstol,
                                   projected=true, min_crossing_time=min_crossing_time)
    length(pts) < period && return period
    s0 = collect(Float64, state)
    base = max(norm(s0), 1.0)
    for m in 1:period
        pm = collect(Float64, pts[m])
        length(pm) == length(s0) || continue
        norm(pm .- s0) < precision * max(base, norm(pm)) && return m
    end
    return period
end

"""
Trim a continuation branch labelled period `N` to only those points whose orbit
is *genuinely* minimal-period `N`, dropping any stretch where the `Pᴺ` fixed
point has run onto a lower-period orbit (period `d | N`). Returns the original
branch when nothing needs trimming, `nothing` when no genuine period-`N` point
remains, or a rebuilt `BranchResult` otherwise. `period == 1` branches are
returned unchanged (a period-1 orbit cannot degenerate further).
"""
function _trim_branch_to_period(sys::DynamicalSystem, branch::BranchResult,
                                base_params::AbstractVector, linked_param_indices::Vector{Int};
                                precision::Union{Nothing, Float64}=nothing,
                                solver=Tsit5(), reltol::Float64=1e-8, abstol::Float64=1e-8,
                                min_crossing_time::Float64=1e-6,
                                trim_diagnostics::Union{Nothing, Base.RefValue}=nothing)
    N = branch.period
    if N <= 1
        _set_period_trim_diagnostics!(
            trim_diagnostics,
            _minimal_period_trim_diagnostics(branch; reason="period_one", applied=false)
        )
        return branch
    end
    pts = _branch_points(branch)
    if isempty(pts)
        _set_period_trim_diagnostics!(
            trim_diagnostics,
            _minimal_period_trim_diagnostics(branch; reason="empty_branch", applied=false, point_count=0, kept_count=0)
        )
        return branch
    end
    # Don't trim against a guessed parameter slot: if the branch's parameter name
    # isn't one of the system's, injecting at index 1 could misclassify every
    # point. Leave the branch untouched instead.
    param_index = findfirst(==(branch.param_name), sys.param_names)
    if isnothing(param_index)
        _set_period_trim_diagnostics!(
            trim_diagnostics,
            _minimal_period_trim_diagnostics(branch; reason="missing_parameter_name", applied=false)
        )
        return branch
    end
    base = collect(Float64, base_params)
    tol = isnothing(precision) ?
          (sys isa ContinuousODE ? _PERIOD_TRIM_PRECISION_CONTINUOUS : _PERIOD_TRIM_PRECISION_DISCRETE) :
          precision

    npts = length(pts)
    minimal_periods = fill(N, npts)
    classify(i) = begin
        pt = pts[i]
        p = _inject_param(base, param_index, Float64(pt.param), linked_param_indices)
        state = _branch_point_state(pt)
        sys isa ContinuousODE ?
            _orbit_minimal_period(sys, state, p, N; precision=tol, solver=solver,
                                  reltol=reltol, abstol=abstol, min_crossing_time=min_crossing_time) :
            _orbit_minimal_period(sys, state, p, N; precision=tol)
    end

    keep = trues(npts)
    sampled_classification = false
    evaluated_count = npts
    if sys isa ContinuousODE && npts > _PERIOD_TRIM_MAX_CONTINUOUS_CHECKS
        # Bound integration cost: classify a uniform subsample, then assign every
        # point the classification of its nearest checked neighbour along the
        # continuation ordering. Do *not* use nearest parameter value here:
        # folded branches can contain multiple distinct orbit states at the same
        # parameter, and those states may have different minimal periods.
        check_idx = unique(round.(Int, range(1, npts, length=_PERIOD_TRIM_MAX_CONTINUOUS_CHECKS)))
        checked_periods = Int[classify(i) for i in check_idx]
        checked_is_n = Bool[p == N for p in checked_periods]
        for i in 1:npts
            nearest = argmin(abs.(check_idx .- i))
            keep[i] = checked_is_n[nearest]
            minimal_periods[i] = checked_periods[nearest]
        end
        sampled_classification = true
        evaluated_count = length(check_idx)
    else
        for i in 1:npts
            minimal_periods[i] = classify(i)
            keep[i] = minimal_periods[i] == N
        end
    end

    kept_count = count(keep)
    dropped_count = npts - kept_count
    kept_indices = findall(identity, keep)
    dropped_indices = findall(!, keep)
    diag = _minimal_period_trim_diagnostics(
        branch;
        reason=all(keep) ? "all_kept" : (kept_count == 0 ? "all_dropped" : "trimmed"),
        applied=!all(keep),
        point_count=npts,
        kept_count=kept_count,
        dropped_count=dropped_count,
        kept_intervals=_param_intervals_for_indices(pts, kept_indices),
        dropped_intervals=_param_intervals_for_indices(pts, dropped_indices),
        lower_period_counts=_minimal_period_counts(minimal_periods, N),
        precision=tol,
        sampled_classification=sampled_classification,
        evaluated_count=evaluated_count
    )
    _set_period_trim_diagnostics!(trim_diagnostics, diag)

    all(keep) && return branch
    kept_count == 0 && return nothing

    kept_pts = pts[keep]
    kept_params = Float64[Float64(pt.param) for pt in kept_pts]
    plo, phi = extrema(kept_params)
    special = try
        collect(branch.branch.specialpoint)
    catch
        Any[]
    end
    kept_special = Any[sp for sp in special
                       if hasproperty(sp, :param) && (plo - _PERIOD_TRIM_PARAM_TOL) <= Float64(sp.param) <= (phi + _PERIOD_TRIM_PARAM_TOL)]
    combined = CombinedBranchResult(Vector{Any}(kept_pts), Vector{Any}(kept_special))
    return BranchResult(combined, N, branch.system_name, branch.param_name, branch.timestamp)
end

"""Merge overlapping parameter intervals after optional padding/clamping."""
function _merge_param_intervals(intervals::Vector{Tuple{Float64, Float64}})
    isempty(intervals) && return Tuple{Float64, Float64}[]
    ordered = sort([(min(lo, hi), max(lo, hi)) for (lo, hi) in intervals]; by=first)
    merged = Tuple{Float64, Float64}[ordered[1]]
    for (lo, hi) in Iterators.drop(ordered, 1)
        prev_lo, prev_hi = merged[end]
        if lo <= prev_hi + 100eps(Float64)
            merged[end] = (prev_lo, max(prev_hi, hi))
        else
            push!(merged, (lo, hi))
        end
    end
    return merged
end

"""Detect parameter intervals that are likely under-resolved for a continuous continuation branch."""
function _continuous_branch_refinement_intervals(branch::BranchResult, config::ContinuationConfig;
                                                 gap_factor::Float64=4.0,
                                                 interval_padding_factor::Float64=0.75,
                                                 detect_short_branches::Bool=true,
                                                 short_branch_span_factor::Float64=8.0,
                                                 short_branch_max_points::Int=160,
                                                 short_branch_min_median_gap_factor::Float64=0.3)
    points = _branch_points(branch)
    length(points) < 2 && return Tuple{Float64, Float64}[]

    pars = Float64[pt.param for pt in points]
    gaps = abs.(diff(pars))
    abs_ds = abs(config.ds)
    dsmax_scale = max(abs(config.dsmax), abs_ds, eps(Float64))
    intervals = Tuple{Float64, Float64}[]

    for i in 1:(length(pars) - 1)
        gap = abs(pars[i + 1] - pars[i])
        gap > gap_factor * abs_ds || continue
        pad = interval_padding_factor * gap
        lo = max(config.p_min, min(pars[i], pars[i + 1]) - pad)
        hi = min(config.p_max, max(pars[i], pars[i + 1]) + pad)
        hi > lo && push!(intervals, (lo, hi))
    end

    if detect_short_branches
        span = maximum(pars) - minimum(pars)
        # Mirror the discrete guard: only re-sweep a short branch when it is uniformly coarse, so a
        # densely sampled branch is not discarded and re-swept beyond its real parameter support.
        uniformly_coarse = _diagnostic_median(gaps) >= short_branch_min_median_gap_factor * dsmax_scale
        if span > 0 &&
           span <= short_branch_span_factor * abs_ds &&
           length(points) <= short_branch_max_points &&
           uniformly_coarse
            pad = max(abs_ds, interval_padding_factor * span)
            lo = max(config.p_min, minimum(pars) - pad)
            hi = min(config.p_max, maximum(pars) + pad)
            hi > lo && push!(intervals, (lo, hi))
        end
    end

    return _merge_param_intervals(intervals)
end

"""Detect sparse-gap and short-span intervals that are likely under-resolved for a discrete branch."""
function _discrete_branch_refinement_intervals(branch::BranchResult, config::ContinuationConfig;
                                               gap_factor::Float64=2.5,
                                               gap_dsmax_factor::Float64=0.7,
                                               interval_padding_factor::Float64=0.75,
                                               tail_window::Int=6,
                                               detect_short_branches::Bool=true,
                                               short_branch_span_factor::Float64=4.0,
                                               short_branch_max_points::Int=32,
                                               short_branch_min_median_gap_factor::Float64=0.3)
    points = _branch_points(branch)
    length(points) < 2 && return Tuple{Float64, Float64}[]

    pars = Float64[pt.param for pt in points]
    gaps = abs.(diff(pars))
    isempty(gaps) && return Tuple{Float64, Float64}[]

    dsmax_scale = max(abs(config.dsmax), abs(config.ds), eps(Float64))
    reference_gaps = length(gaps) > tail_window ? gaps[1:max(end - tail_window, 1)] : gaps
    typical_gap = max(_diagnostic_median(reference_gaps), abs(config.ds), eps(Float64))
    # The dsmax floor catches the sparse, dotted tails where PALC has ramped the step up toward
    # dsmax even though the surrounding samples are dense; keep it below dsmax so those gaps land.
    max_allowed_gap = max(gap_factor * typical_gap, gap_dsmax_factor * dsmax_scale)
    intervals = Tuple{Float64, Float64}[]

    for i in eachindex(gaps)
        gaps[i] > max_allowed_gap || continue
        pad = max(interval_padding_factor * gaps[i], abs(config.ds))
        lo = max(config.p_min, min(pars[i], pars[i + 1]) - pad)
        hi = min(config.p_max, max(pars[i], pars[i + 1]) + pad)
        hi > lo && push!(intervals, (lo, hi))
    end

    if detect_short_branches
        span = maximum(pars) - minimum(pars)
        max_gap = maximum(gaps)
        # Only re-sweep a short branch when it is *uniformly* coarse. A large median gap (relative
        # to dsmax) means every sample is sparse; a small median means the branch is actually
        # densely sampled (e.g. a fine onset cluster) and a full re-sweep would discard resolution
        # and can push the branch past its real parameter support.
        uniformly_coarse = _diagnostic_median(gaps) >= short_branch_min_median_gap_factor * dsmax_scale
        if span > 0 &&
           span <= short_branch_span_factor * dsmax_scale &&
           length(points) <= short_branch_max_points &&
           max_gap >= 0.7 * dsmax_scale &&
           uniformly_coarse
            pad = max(dsmax_scale, interval_padding_factor * span)
            lo = max(config.p_min, minimum(pars) - pad)
            hi = min(config.p_max, maximum(pars) + pad)
            hi > lo && push!(intervals, (lo, hi))
        end
    end

    return _merge_param_intervals(intervals)
end

"""
Pick the best existing branch point to seed a local refinement interval.

Coverage-aware: when two or more existing branch points fall inside the refine
window, the seed is chosen near the centre of the largest gap (including the
gaps to the window boundaries). This sends refinement effort where it is most
needed — under-covered regions — rather than blindly to the geometric midpoint
of the requested interval.

Falls back to the closest-to-midpoint behaviour when the refine window contains
at most one existing branch point (no meaningful coverage signal), or when no
branch points lie inside the window at all.
"""
function _refinement_seed_index(branch::BranchResult, from_param::Float64, to_param::Float64)
    points = _branch_points(branch)
    pars = Float64[pt.param for pt in points]
    lo = min(from_param, to_param)
    hi = max(from_param, to_param)
    in_window = findall(p -> lo - 100eps(Float64) <= p <= hi + 100eps(Float64), pars)

    if length(in_window) >= 2
        # Build a sorted view of the existing coverage inside the window, then
        # treat (lo, sorted...) and (sorted..., hi) as the gap structure.
        sorted_window = sort(in_window; by=i -> pars[i])
        sorted_pars = pars[sorted_window]
        boundaries = Float64[lo, sorted_pars..., hi]
        best_gap_size = -Inf
        best_gap_center = (lo + hi) / 2
        for k in 1:(length(boundaries) - 1)
            gap = boundaries[k + 1] - boundaries[k]
            if gap > best_gap_size
                best_gap_size = gap
                best_gap_center = (boundaries[k + 1] + boundaries[k]) / 2
            end
        end
        local_idx = argmin(abs.(sorted_pars .- best_gap_center))
        return sorted_window[local_idx]
    end

    # Zero or one point in the window: no useful gap signal. Use the legacy
    # midpoint heuristic over the in-window points (or the whole branch as a
    # last resort).
    mid = (lo + hi) / 2
    search_set = isempty(in_window) ? collect(eachindex(pars)) : in_window
    local_idx = argmin(abs.(pars[search_set] .- mid))
    return search_set[local_idx]
end

"""Deduplicate adjacent branch points while preserving continuation order."""
function _deduplicate_branch_points(points::AbstractVector;
                                    param_tol::Float64=1e-10,
                                    state_tol::Float64=1e-7)
    isempty(points) && return points
    deduped = Any[points[1]]
    for pt in Iterators.drop(points, 1)
        prev = deduped[end]
        if abs(pt.param - prev.param) <= param_tol && _branch_point_distance(pt, prev) <= state_tol
            continue
        end
        push!(deduped, pt)
    end
    return deduped
end

function _contiguous_index_runs(indices::Vector{Int})
    isempty(indices) && return Vector{UnitRange{Int}}()
    sorted_idx = sort(indices)
    runs = UnitRange{Int}[]
    start_idx = sorted_idx[1]
    prev_idx = sorted_idx[1]
    for idx in Iterators.drop(sorted_idx, 1)
        if idx == prev_idx + 1
            prev_idx = idx
            continue
        end
        push!(runs, start_idx:prev_idx)
        start_idx = idx
        prev_idx = idx
    end
    push!(runs, start_idx:prev_idx)
    return runs
end

function _branch_state_scale(points::AbstractVector)
    states = [_branch_point_state(pt) for pt in points]
    nonempty = [state for state in states if !isempty(state)]
    isempty(nonempty) && return 1.0
    dim = minimum(length.(nonempty))
    dim == 0 && return 1.0
    spans = Float64[]
    for j in 1:dim
        vals = Float64[state[j] for state in nonempty if length(state) >= j]
        isempty(vals) && continue
        push!(spans, maximum(vals) - minimum(vals))
    end
    return max(isempty(spans) ? 0.0 : maximum(spans), 1.0)
end

function _branch_point_metric(a, b, param_scale::Float64, state_scale::Float64)
    p_dist = abs(Float64(a.param) - Float64(b.param)) / max(param_scale, eps(Float64))
    s_dist = _branch_point_distance(a, b) / max(state_scale, eps(Float64))
    return hypot(p_dist, s_dist)
end

function _refined_segment_run_score(points::AbstractVector,
                                    segment_points::AbstractVector,
                                    run::UnitRange{Int},
                                    param_scale::Float64,
                                    state_scale::Float64)
    isempty(segment_points) && return Inf
    samples = unique(round.(Int, range(1, length(segment_points), length=min(5, length(segment_points)))))
    isempty(samples) && return Inf
    total = 0.0
    for sample_idx in samples
        seg_pt = segment_points[sample_idx]
        total += minimum(_branch_point_metric(seg_pt, points[i], param_scale, state_scale) for i in run)
    end
    return total / length(samples)
end

function _orient_refined_segment(points::AbstractVector,
                                 segment_points::AbstractVector,
                                 replace_run::UnitRange{Int},
                                 param_scale::Float64,
                                 state_scale::Float64)
    length(segment_points) <= 1 && return segment_points
    first_original = points[first(replace_run)]
    last_original = points[last(replace_run)]
    forward_score = _branch_point_metric(segment_points[1], first_original, param_scale, state_scale) +
                    _branch_point_metric(segment_points[end], last_original, param_scale, state_scale)
    reverse_score = _branch_point_metric(segment_points[end], first_original, param_scale, state_scale) +
                    _branch_point_metric(segment_points[1], last_original, param_scale, state_scale)
    return reverse_score < forward_score ? reverse(segment_points) : segment_points
end

function _splice_refined_segment_points(points::AbstractVector,
                                        segment_points::AbstractVector;
                                        param_tol::Float64=1e-10,
                                        state_tol::Float64=1e-7)
    isempty(segment_points) && return points
    isempty(points) && return segment_points

    segment_params = Float64[Float64(pt.param) for pt in segment_points]
    lo, hi = extrema(segment_params)
    param_scale = max(maximum(Float64[Float64(pt.param) for pt in vcat(points, segment_points)]) -
                      minimum(Float64[Float64(pt.param) for pt in vcat(points, segment_points)]), abs(hi - lo), 1.0)
    state_scale = _branch_state_scale(vcat(points, segment_points))
    candidate_indices = Int[i for i in eachindex(points) if lo - param_tol <= Float64(points[i].param) <= hi + param_tol]
    runs = _contiguous_index_runs(candidate_indices)

    if !isempty(runs)
        scores = Float64[_refined_segment_run_score(points, segment_points, run, param_scale, state_scale) for run in runs]
        run = runs[argmin(scores)]
        ordered_segment = _orient_refined_segment(points, segment_points, run, param_scale, state_scale)
        merged = Any[
            points[1:(first(run) - 1)]...,
            ordered_segment...,
            points[(last(run) + 1):end]...
        ]
        return _deduplicate_branch_points(merged; param_tol=param_tol, state_tol=state_tol)
    end

    first_idx = argmin(_branch_point_metric(segment_points[1], pt, param_scale, state_scale) for pt in points)
    last_idx = argmin(_branch_point_metric(segment_points[end], pt, param_scale, state_scale) for pt in points)
    insert_after = min(first_idx, last_idx)
    replace_run = min(first_idx, last_idx):max(first_idx, last_idx)
    ordered_segment = _orient_refined_segment(points, segment_points, replace_run, param_scale, state_scale)
    merged = Any[
        points[1:insert_after]...,
        ordered_segment...,
        points[(insert_after + 1):end]...
    ]
    return _deduplicate_branch_points(merged; param_tol=param_tol, state_tol=state_tol)
end

"""Splice refined local branches back into the original branch representation."""
function _splice_refined_branches(original::BranchResult,
                                  refined_segments::Vector{BranchResult};
                                  param_tol::Float64=1e-10,
                                  state_tol::Float64=1e-7)
    isempty(refined_segments) && return original
    special_points = Any[]
    merged_points = _branch_points(original)

    for segment in refined_segments
        pts = _branch_points(segment)
        isempty(pts) && continue
        merged_points = _splice_refined_segment_points(
            merged_points,
            Vector{Any}(pts);
            param_tol=param_tol,
            state_tol=state_tol
        )
        append!(special_points, segment.branch.specialpoint)
    end

    merged_specials = vcat(original.branch.specialpoint, special_points)
    merged_branch = CombinedBranchResult(merged_points, merged_specials)
    return BranchResult(merged_branch, original.period, original.system_name, original.param_name, now())
end

_splice_refined_continuous_branches(original::BranchResult,
                                    refined_segments::Vector{BranchResult};
                                    param_tol::Float64=1e-10,
                                    state_tol::Float64=1e-7) =
    _splice_refined_branches(original, refined_segments; param_tol=param_tol, state_tol=state_tol)

"""
    refine_branch(sys::ContinuousODE, original::BranchResult, config::RefinementConfig; kwargs...) -> BranchResult

Refine a sub-interval of a continuous-time continuation branch by re-running continuation on that
parameter window from the nearest existing branch point.
"""
function refine_branch(sys::ContinuousODE, original::BranchResult, config::RefinementConfig;
                       params::Vector{Float64}=Float64[],
                       linked_param_indices::Vector{Int}=Int[],
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
                       reseed::ReseedConfig=ReseedConfig(),
                       on_reseed::Union{Nothing, Function}=nothing)
    points = _branch_points(original)
    isempty(points) && error("Cannot refine an empty branch.")

    param_index = something(findfirst(==(original.param_name), sys.param_names), 1)
    seed_idx = _refinement_seed_index(original, config.from_param, config.to_param)
    seed_point = _branch_point_state(points[seed_idx])
    seed_param = Float64(points[seed_idx].param)
    base_params = _resolve_continuous_params(sys, params)
    local_params = _inject_param(base_params, param_index, seed_param, linked_param_indices)

    local_config = ContinuationConfig(
        p_min=min(config.from_param, config.to_param),
        p_max=max(config.from_param, config.to_param),
        ds=sign(config.to_param - config.from_param) == 0 ? config.ds : sign(config.to_param - config.from_param) * abs(config.ds),
        dsmax=config.dsmax,
        dsmin=config.dsmin,
        max_steps=config.max_steps,
        newton_tol=config.newton_tol,
        newton_max_iter=config.newton_max_iter,
        detect_bifurcation=config.detect_bifurcation,
        param_index=param_index,
        linked_param_indices=copy(linked_param_indices),
        a=config.a,
        detect_fold=config.detect_fold,
        save_sol_every_step=config.save_sol_every_step,
        ode_jacobian_method=config.ode_jacobian_method
    )

    return continuation_branch(
        sys,
        local_config,
        original.period;
        initial_point=seed_point,
        params=local_params,
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
        reseed=reseed,
        on_reseed=on_reseed
    )
end

"""
    auto_refine_branch(sys::ContinuousODE, original::BranchResult, base_config::ContinuationConfig; kwargs...) -> BranchResult

Automatically detect under-resolved intervals in a continuous continuation branch, regenerate those
windows with smaller continuation steps, and splice the refined points back into the branch.
"""
function auto_refine_branch(sys::ContinuousODE, original::BranchResult, base_config::ContinuationConfig;
                            params::Vector{Float64}=Float64[],
                            linked_param_indices::Vector{Int}=copy(base_config.linked_param_indices),
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
                            max_passes::Int=1,
                            refine_factor::Float64=4.0,
                            gap_factor::Float64=4.0,
                            interval_padding_factor::Float64=0.75,
                            detect_short_branches::Bool=true,
                            short_branch_span_factor::Float64=8.0,
                            short_branch_max_points::Int=160,
                            reseed::ReseedConfig=ReseedConfig(),
                            on_reseed::Union{Nothing, Function}=nothing)
    refined = original
    current_config = base_config

    for _ in 1:max_passes
        intervals = _continuous_branch_refinement_intervals(
            refined,
            current_config;
            gap_factor=gap_factor,
            interval_padding_factor=interval_padding_factor,
            detect_short_branches=detect_short_branches,
            short_branch_span_factor=short_branch_span_factor,
            short_branch_max_points=short_branch_max_points
        )
        isempty(intervals) && break

        refined_segments = BranchResult[]
        for (from_param, to_param) in intervals
            local_refine = RefinementConfig(
                from_param=from_param,
                to_param=to_param,
                ds=abs(current_config.ds) / refine_factor,
                dsmax=max(abs(current_config.ds) / refine_factor, current_config.dsmax / refine_factor),
                dsmin=min(current_config.dsmin, abs(current_config.ds) / refine_factor^3),
                max_steps=max(current_config.max_steps, ceil(Int, current_config.max_steps * refine_factor)),
                newton_tol=current_config.newton_tol,
                newton_max_iter=max(current_config.newton_max_iter, 30),
                detect_bifurcation=current_config.detect_bifurcation,
                a=current_config.a,
                detect_fold=current_config.detect_fold,
                save_sol_every_step=current_config.save_sol_every_step,
                ode_jacobian_method=current_config.ode_jacobian_method
            )
            push!(refined_segments, refine_branch(
                sys,
                refined,
                local_refine;
                params=params,
                linked_param_indices=linked_param_indices,
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
                reseed=reseed,
                on_reseed=on_reseed
            ))
        end

        refined = _splice_refined_branches(refined, refined_segments)
        current_config = ContinuationConfig(
            p_min=current_config.p_min,
            p_max=current_config.p_max,
            ds=abs(current_config.ds) / refine_factor,
            dsmax=max(abs(current_config.ds) / refine_factor, current_config.dsmax / refine_factor),
            dsmin=min(current_config.dsmin, abs(current_config.ds) / refine_factor^3),
            max_steps=max(current_config.max_steps, ceil(Int, current_config.max_steps * refine_factor)),
            newton_tol=current_config.newton_tol,
            newton_max_iter=current_config.newton_max_iter,
            detect_bifurcation=current_config.detect_bifurcation,
            param_index=current_config.param_index,
            linked_param_indices=copy(current_config.linked_param_indices),
            ode_jacobian_method=current_config.ode_jacobian_method
        )
    end

    return refined
end

"""
    auto_refine_branch(sys::DiscreteMap, original::BranchResult, base_config::ContinuationConfig; kwargs...) -> BranchResult

Automatically detect sparse gaps or very short/coarse discrete continuation branches, re-run those
intervals with smaller continuation steps, and splice the refined points back into the branch.
"""
function auto_refine_branch(sys::DiscreteMap, original::BranchResult, base_config::ContinuationConfig;
                            params::Vector{Float64}=Float64[],
                            linked_param_indices::Vector{Int}=copy(base_config.linked_param_indices),
                            max_passes::Int=1,
                            refine_factor::Float64=4.0,
                            gap_factor::Float64=2.5,
                            interval_padding_factor::Float64=0.75,
                            tail_window::Int=6,
                            detect_short_branches::Bool=true,
                            short_branch_span_factor::Float64=4.0,
                            short_branch_max_points::Int=32,
                            reseed::ReseedConfig=ReseedConfig(),
                            on_reseed::Union{Nothing, Function}=nothing)
    refined = original
    current_config = base_config
    base = isempty(params) ? Float64[] : collect(Float64, params)

    for _ in 1:max_passes
        intervals = _discrete_branch_refinement_intervals(
            refined,
            current_config;
            gap_factor=gap_factor,
            interval_padding_factor=interval_padding_factor,
            tail_window=tail_window,
            detect_short_branches=detect_short_branches,
            short_branch_span_factor=short_branch_span_factor,
            short_branch_max_points=short_branch_max_points
        )
        isempty(intervals) && break

        refined_segments = BranchResult[]
        for (from_param, to_param) in intervals
            local_refine = RefinementConfig(
                from_param=from_param,
                to_param=to_param,
                ds=abs(current_config.ds) / refine_factor,
                dsmax=max(abs(current_config.ds) / refine_factor, current_config.dsmax / refine_factor),
                dsmin=min(current_config.dsmin, abs(current_config.ds) / refine_factor^3),
                max_steps=max(current_config.max_steps, ceil(Int, current_config.max_steps * refine_factor)),
                newton_tol=current_config.newton_tol,
                newton_max_iter=current_config.newton_max_iter,
                detect_bifurcation=current_config.detect_bifurcation,
                a=current_config.a,
                detect_fold=current_config.detect_fold,
                save_sol_every_step=current_config.save_sol_every_step,
                ode_jacobian_method=current_config.ode_jacobian_method
            )
            push!(refined_segments, refine_branch(
                sys,
                refined,
                local_refine;
                params=base,
                linked_param_indices=linked_param_indices,
                reseed=reseed,
                on_reseed=on_reseed
            ))
        end

        refined = _splice_refined_branches(refined, refined_segments)
        refined = _canonicalize_branch_representatives(sys, refined, base, linked_param_indices)
        current_config = ContinuationConfig(
            p_min=current_config.p_min,
            p_max=current_config.p_max,
            ds=abs(current_config.ds) / refine_factor,
            dsmax=max(abs(current_config.ds) / refine_factor, current_config.dsmax / refine_factor),
            dsmin=min(current_config.dsmin, abs(current_config.ds) / refine_factor^3),
            max_steps=max(current_config.max_steps, ceil(Int, current_config.max_steps * refine_factor)),
            newton_tol=current_config.newton_tol,
            newton_max_iter=current_config.newton_max_iter,
            detect_bifurcation=current_config.detect_bifurcation,
            param_index=current_config.param_index,
            linked_param_indices=copy(current_config.linked_param_indices),
            a=current_config.a,
            detect_fold=current_config.detect_fold,
            save_sol_every_step=current_config.save_sol_every_step,
            ode_jacobian_method=current_config.ode_jacobian_method
        )
    end

    return refined
end

# ═══════════════════════════════════════════════════════════════════════════════
# Branch Refinement
# ═══════════════════════════════════════════════════════════════════════════════

"""
    refine_branch(sys::DiscreteMap, original::BranchResult, config::RefinementConfig;
                  params=[1.0]) -> BranchResult

Refine a specific parameter interval of an existing branch by re-running continuation
with finer step sizes. The original branch data outside the interval is preserved,
and the refined interval is spliced in.

This is useful when the original continuation used large steps that may have missed
bifurcation details or fine structure in a specific region.
"""
function refine_branch(sys::DiscreteMap, original::BranchResult, config::RefinementConfig;
                       params::Vector{Float64}=[1.0],
                       linked_param_indices::Vector{Int}=Int[],
                       reseed::ReseedConfig=ReseedConfig(),
                       on_reseed::Union{Nothing, Function}=nothing)
    period = original.period
    branch = original.branch
    param_index = something(findfirst(==(original.param_name), sys.param_names), 1)

    # Extract branch data
    branch_points = collect(branch.branch)
    all_pars = [pt.param for pt in branch_points]
    all_x1 = [getproperty(pt, :x1) for pt in branch_points]

    from_p = min(config.from_param, config.to_param)
    to_p = max(config.from_param, config.to_param)
    start_param = clamp(config.from_param, from_p, to_p)

    # Find a good initial guess from the branch data near the continuation seed
    # edge. `from_param` is intentionally allowed to be the upper bound for a
    # left extension; p_min/p_max stay sorted but the seed and ds sign preserve
    # the requested continuation direction.
    before_idx = findlast(p -> p <= start_param, all_pars)
    after_idx = findfirst(p -> p >= start_param, all_pars)

    if isnothing(before_idx) && isnothing(after_idx)
        error("Refinement interval [$from_p, $to_p] has no overlap with branch parameter range [$(minimum(all_pars)), $(maximum(all_pars))]")
    end

    # Use the nearest branch point as a hint, but start continuation at the requested edge.
    hint_idx = isnothing(before_idx) ? after_idx : before_idx

    dim = sys.dim

    # Use Newton to get exact fixed point at start_param
    F_start = (x) -> begin
        pv = _inject_param(params, param_index, start_param, linked_param_indices)
        sv = SVector{dim}(x)
        for _ in 1:period
            sv = sys.f(sv, pv)
        end
        Array(sv) .- x
    end
    # Seed Newton with the FULL recorded fixed-point state of the nearest branch
    # point — not just x1. The old code zeroed every other coordinate (and the
    # fallback scaled them by x1), which for a 2-D map like the boost converter
    # started Newton from (V, 0) / (V, ±0.3·V) — far from the true I* ≈ O(1) — so
    # the start-of-interval Newton routinely failed even well inside the branch.
    hint_state = _branch_point_state(branch_points[hint_idx])
    x1_hint = isempty(hint_state) ? all_x1[hint_idx] : hint_state[1]
    x0_guess = zeros(dim)
    if isempty(hint_state)
        # No recorded state coordinates: fall back to the legacy x1-only seed.
        x0_guess[1] = x1_hint
    else
        @inbounds for d in 1:min(length(hint_state), dim)
            x0_guess[d] = hint_state[d]
        end
    end

    x0, converged = _newton_for_refine(F_start, x0_guess, 1e-10, 50)
    if !converged
        # Fallback: perturb around the recorded state (relative where non-zero,
        # else fall back to an x1-scaled nudge).
        for scale in [0.3, 0.1, -0.3, 0.5, -0.5]
            x0_guess2 = copy(x0_guess)
            for d in 2:dim
                x0_guess2[d] = abs(x0_guess[d]) > 1e-9 ? x0_guess[d] * (1 + scale) : x1_hint * scale
            end
            x0, converged = _newton_for_refine(F_start, x0_guess2, 1e-10, 50)
            converged && break
        end
    end
    if !converged
        error("Could not find fixed point at start of refinement interval (param=$start_param)")
    end

    # Build the F function for fixed-point continuation (same as original)
    if period == 1
        F = (x, p) -> begin
            pv = _inject_param(params, param_index, p.p, linked_param_indices)
            Array(sys.f(SVector{dim}(x), pv)) .- x
        end
    else
        F = (x, p) -> begin
            pv = _inject_param(params, param_index, p.p, linked_param_indices)
            sv = SVector{dim}(x)
            for _ in 1:period
                sv = sys.f(sv, pv)
            end
            Array(sv) .- x
        end
    end

    # Build a ContinuationConfig from the RefinementConfig so we can route through the shared
    # discrete continuation helper (and benefit from re-seed when enabled).
    local_config = ContinuationConfig(
        p_min=from_p, p_max=to_p,
        ds=sign(config.to_param - config.from_param) == 0 ? config.ds : sign(config.to_param - config.from_param) * abs(config.ds),
        dsmax=config.dsmax, dsmin=config.dsmin,
        max_steps=config.max_steps,
        newton_tol=config.newton_tol, newton_max_iter=config.newton_max_iter,
        detect_bifurcation=config.detect_bifurcation,
        param_index=param_index,
        linked_param_indices=copy(linked_param_indices),
        a=config.a, detect_fold=config.detect_fold,
        save_sol_every_step=config.save_sol_every_step
    )
    # Inject the start_param into params so _run_discrete_continuation seeds at start_param.
    local_params = _inject_param(params, param_index, start_param, linked_param_indices)

    return _run_discrete_continuation(sys, local_config, period, x0, F, _default_record,
                                      local_params, reseed, on_reseed)
end

"""Newton solver for finding starting point during branch refinement."""
function _newton_for_refine(F, x0, tol, max_iter)
    x = copy(x0)
    for _ in 1:max_iter
        Fx = F(x)
        if norm(Fx) < tol
            return x, true
        end
        J = ForwardDiff.jacobian(F, x)
        cnd = cond(J)
        if !isfinite(cnd) || cnd > 1e15
            return x, false
        end
        x = x .- J \ Fx
    end
    return x, norm(F(x)) < tol
end
