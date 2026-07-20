# Multistability-aware continuation with basin reachability.
#
# Pairs continuation branches with a per-parameter basin initial-condition census so each coexisting
# branch is reported with the basin fraction that actually reaches it. Every basin seed's terminal
# periodic orbit is assigned to a *stable* branch identity by period-gated, phase-invariant
# state-space geometry — period alone is never treated as branch identity, and a period group is
# never plurality-assigned as a block. Every seed is accounted for in exactly one category.
#
# The census reuses the public discrete-map period-detection kernel (`_detect_discrete_map_period`,
# the same closure comparison `basins_of_attraction` runs) rather than duplicating a basin kernel;
# the branch state at each requested parameter knot is solved (Newton) at that exact knot so a
# distant branch point is never substituted for the requested sample.

# Per-cell assignment status codes (compact, serialization-stable).
const _REACH_STATUS_LABEL_BY_CODE = Dict{Int, String}(
    0 => "unknown",
    1 => "matched",
    2 => "unmatched",
    3 => "aperiodic",
    4 => "diverged",
    5 => "unresolved",
    6 => "stability_mismatch",
    7 => "outside_coverage",
)
const _REACH_STATUS_CODE_BY_LABEL = Dict{String, Int}(
    label => code for (code, label) in _REACH_STATUS_LABEL_BY_CODE
)
const _REACH_MATCHED = 1
const _REACH_UNMATCHED = 2
const _REACH_APERIODIC = 3
const _REACH_DIVERGED = 4
const _REACH_UNRESOLVED = 5
const _REACH_STABILITY_MISMATCH = 6
const _REACH_OUTSIDE_COVERAGE = 7

"""
    BranchReachabilitySample

Reachability census at one parameter knot. All per-branch vectors are aligned to the parent
result's global branch order (`branch_ids`); the per-cell matrices share the initial-condition grid
shape `(nx, ny)`.

# Fields
- `param`: the parameter knot
- `branch_ids`, `branch_periods`: branch identities and their minimal period solved at this knot
- `branch_stable`: whether each covered branch is attracting (all multipliers inside the unit circle)
  at this knot; unstable branch segments are rejected from the attracting (`matched`) fractions
- `branch_covered`: whether each branch's continued range reaches this knot within tolerance
- `branch_confidence`: branch-state solve confidence at this knot (`1.0` when the Newton fixed-point
  solve converged; lower when only the interpolated continuation estimate was available)
- `matched_counts`, `matched_fractions`: seeds reaching each branch, and their fraction of the full
  seed census (`n_seeds` denominator)
- `n_seeds`, `n_matched`, `n_unmatched`, `n_aperiodic`, `n_diverged`, `n_unresolved`,
  `n_stability_mismatch`, `n_outside_coverage`: category counts; the seven categories partition every
  seed (they sum to `n_seeds`)
- `assignment`: per-cell matched branch index (1-based into `branch_ids`) or `0`
- `status`: per-cell status code (`branch_reachability_status_label`)
- `match_distance`: per-cell phase-invariant match distance (`NaN` where no distance applies)
- `terminal_period`: per-cell detected terminal period of the seed orbit (the basin periodicity)
- `diagnostics`: knot-level diagnostic dict
"""
struct BranchReachabilitySample
    param::Float64
    branch_ids::Vector{String}
    branch_periods::Vector{Int}
    branch_stable::Vector{Bool}
    branch_covered::Vector{Bool}
    branch_confidence::Vector{Float64}
    matched_counts::Vector{Int}
    matched_fractions::Vector{Float64}
    n_seeds::Int
    n_matched::Int
    n_unmatched::Int
    n_aperiodic::Int
    n_diverged::Int
    n_unresolved::Int
    n_stability_mismatch::Int
    n_outside_coverage::Int
    assignment::Matrix{Int}
    status::Matrix{Int}
    match_distance::Matrix{Float64}
    terminal_period::Matrix{Int}
    diagnostics::Dict{String, Any}
end

"""
    BranchReachabilityResult

Plain-data result of `branch_reachability`. Records the system/parameter provenance, the shared
initial-condition grid, the global branch identities, and one `BranchReachabilitySample` per
requested parameter knot.

# Fields
- `system_name`, `param_name`, `param_index`, `linked_param_indices`, `base_params`: provenance of
  the varied continuation parameter and its base vector
- `x_grid`, `y_grid`, `x_index`, `y_index`, `ic_template`: the initial-condition census grid
- `max_period`, `precision`, `divergence_cutoff`: seed terminal-orbit detection settings
- `param_tolerance`, `match_tolerance`, `ambiguity_ratio`, `stability_tol`: assignment tolerances
- `branch_ids`, `branch_periods`: global stable branch identities (input order) and declared periods
- `samples`: one census per parameter knot, ordered like `config.param_samples`
- `status_labels`: status-code → label map for `status` matrices
- `timestamp`
"""
struct BranchReachabilityResult
    system_name::String
    param_name::Symbol
    param_index::Int
    linked_param_indices::Vector{Int}
    base_params::Vector{Float64}
    x_grid::Vector{Float64}
    y_grid::Vector{Float64}
    x_index::Int
    y_index::Int
    ic_template::Vector{Float64}
    max_period::Int
    precision::Float64
    divergence_cutoff::Float64
    param_tolerance::Float64
    match_tolerance::Float64
    ambiguity_ratio::Float64
    stability_tol::Float64
    branch_ids::Vector{String}
    branch_periods::Vector{Int}
    samples::Vector{BranchReachabilitySample}
    status_labels::Dict{Int, String}
    timestamp::DateTime
end

"""    branch_reachability_status_label(code) -> String — label for a per-cell status code."""
branch_reachability_status_label(code::Integer) =
    get(_REACH_STATUS_LABEL_BY_CODE, Int(code), "unknown")

# --- phase-invariant cycle geometry ---

# Amplitude scale of a cycle: the largest per-dimension coordinate span, floored at 1 so tiny
# near-origin cycles keep an absolute distance floor (mirrors the basins/branch-family convention).
function _reach_cycle_scale(points::Vector{Vector{Float64}})
    length(points) <= 1 && return 1.0
    dim = length(first(points))
    span = 0.0
    for d in 1:dim
        lo = Inf
        hi = -Inf
        for pt in points
            lo = min(lo, pt[d])
            hi = max(hi, pt[d])
        end
        span = max(span, hi - lo)
    end
    return max(span, 1.0)
end

# Reduce a period-T orbit to its minimal-period cycle: the smallest divisor m | T for which the
# orbit repeats with period m within an amplitude-relative tolerance. Enforces minimal period so a
# period-2 branch stored as a longer repeated window (or a fixed point iterated T times) is compared
# at its true period.
function _reach_minimal_cycle(points::Vector{Vector{Float64}}, precision::Float64)
    n = length(points)
    n <= 1 && return points
    scale = _reach_cycle_scale(points)
    threshold = precision * scale
    for m in 1:(n - 1)
        n % m == 0 || continue
        repeats = true
        for i in 1:n
            j = mod1(i + m, n)
            if norm(points[i] .- points[j]) > threshold
                repeats = false
                break
            end
        end
        repeats && return points[1:m]
    end
    return points
end

# Phase-invariant distance between two equal-length cycles: the minimum over cyclic phase shifts of
# the RMS pointwise distance. Both cycles are generated by iterating the map, so a cyclic shift
# aligns their phases; taking the minimum makes the comparison invariant to which cycle point is
# "first". Returns `Inf` on a length mismatch (callers period-gate first).
function _reach_cycle_distance(a::Vector{Vector{Float64}}, b::Vector{Vector{Float64}})
    m = length(a)
    m == length(b) || return Inf
    m == 0 && return Inf
    best = Inf
    for shift in 0:(m - 1)
        acc = 0.0
        for i in 1:m
            j = mod1(i + shift, m)
            acc += sum(abs2, a[i] .- b[j])
        end
        best = min(best, sqrt(acc / m))
    end
    return best
end

_reach_normalized_distance(raw::Float64, scale_a::Float64, scale_b::Float64) =
    raw / max(scale_a, scale_b, 1.0)

# Iterate the map `period` times from `state` at `params`, returning the ordered cycle points.
function _reach_reconstruct_cycle(sys::DiscreteMap,
                                  state::AbstractVector,
                                  params::AbstractVector,
                                  period::Int)
    dim = sys.dim
    current = SVector{dim, Float64}(state)
    points = Vector{Vector{Float64}}(undef, period)
    for k in 1:period
        points[k] = collect(Float64, current)
        current = SVector{dim, Float64}(sys.f(current, params))
    end
    return points
end

# --- branch state solved at a requested knot ---

# Extract the first `dim` recorded state coordinates (`x1..xdim`) of a continuation point, by index.
_reach_point_state(pt, dim::Int) = Float64[getproperty(pt, Symbol(:x, i)) for i in 1:dim]

# Newton solve of the exact period-`period` fixed point x = f^period(x) at `params`, seeded from an
# interpolated continuation estimate. Returns `(state, converged)`.
function _reach_newton_fixed_point(sys::DiscreteMap,
                                   seed::AbstractVector,
                                   params::AbstractVector,
                                   period::Int,
                                   max_iter::Int,
                                   tol::Float64)
    dim = sys.dim
    local_params = collect(Float64, params)
    x = collect(Float64, seed)
    g = xx -> begin
        current = SVector{dim}(xx)
        for _ in 1:period
            current = sys.f(current, local_params)
        end
        Array(current) .- xx
    end
    gx = g(x)
    all(isfinite, gx) || return (x, false)
    for _ in 1:max_iter
        norm(gx) < _newton_convergence_threshold(x, tol) && return (x, true)
        jac = ForwardDiff.jacobian(g, x)
        all(isfinite, jac) || return (x, false)
        step = try
            jac \ gx
        catch
            return (x, false)
        end
        all(isfinite, step) || return (x, false)
        x = x .- step
        all(isfinite, x) || return (x, false)
        gx = g(x)
        all(isfinite, gx) || return (x, false)
    end
    return (x, norm(gx) < _newton_convergence_threshold(x, tol))
end

struct _ReachBranchAtParam
    index::Int
    id::String
    declared_period::Int
    covered::Bool
    stability_available::Bool
    stable::Bool
    period::Int
    cycle::Vector{Vector{Float64}}
    scale::Float64
    confidence::Float64
    # Non-empty only when the branch was excluded at this knot for a *recoverable* numerical
    # reason (return-map stability could not be assessed): the exact error text is preserved in
    # the sample diagnostics rather than silently downgrading the branch to "unstable".
    note::String
end

# Positional constructor without a note (the common covered/uncovered cases carry none).
_ReachBranchAtParam(index, id, declared_period, covered, stable, period, cycle, scale, confidence) =
    _ReachBranchAtParam(
        index, id, declared_period, covered, covered, stable, period, cycle, scale, confidence, "")

# Select/solve one branch's real state at a knot. A branch is covered only when the knot lies inside
# its recorded parameter range (± `param_tolerance`); the state is linearly interpolated between the
# two bracketing branch points and then Newton-refined to the exact fixed point at the knot, so a
# branch point far from the knot is never used directly.
function _reach_branch_state_at_param(sys::DiscreteMap,
                                      branch::BranchResult,
                                      index::Int,
                                      branch_id::String,
                                      p::Float64,
                                      base_params::Vector{Float64},
                                      config::BranchReachabilityConfig)
    dim = sys.dim
    period = max(branch.period, 1)
    entries = Tuple{Float64, Vector{Float64}}[]
    for pt in _branch_points(branch)
        hasproperty(pt, :param) || continue
        pv = Float64(pt.param)
        isfinite(pv) || continue
        all(i -> hasproperty(pt, Symbol(:x, i)), 1:dim) || continue
        state = _reach_point_state(pt, dim)
        all(isfinite, state) || continue
        push!(entries, (pv, state))
    end
    isempty(entries) && return _ReachBranchAtParam(
        index, branch_id, period, false, false, period, Vector{Float64}[], 1.0, 0.0)

    params_min = minimum(first(entry) for entry in entries)
    params_max = maximum(first(entry) for entry in entries)
    if p < params_min - config.param_tolerance || p > params_max + config.param_tolerance
        return _ReachBranchAtParam(
            index, branch_id, period, false, false, period, Vector{Float64}[], 1.0, 0.0)
    end

    seed_state = _reach_interpolated_state(entries, p)
    params_at_p = inject_param(base_params, config.param_index, p, config.linked_param_indices)
    refined_state, converged = _reach_newton_fixed_point(
        sys, seed_state, params_at_p, period, config.newton_max_iter, config.newton_tol)
    state = converged ? refined_state : seed_state
    confidence = converged ? 1.0 : 0.5

    raw_cycle = _reach_reconstruct_cycle(sys, state, params_at_p, period)
    cycle = _reach_minimal_cycle(raw_cycle, config.precision)
    minimal_period = length(cycle)
    scale = _reach_cycle_scale(cycle)

    # Return-map stability drives the stable-only branch identity. A genuine unstable branch
    # returns stable=false and is still used (to detect stability_mismatch); but if the multiplier
    # computation itself *fails* (e.g. a non-finite Jacobian) we cannot honestly call the branch
    # stable or unstable, so it is excluded from matching at this knot with its reason preserved —
    # never silently downgraded to "unstable", which would misclassify seeds that reach it.
    stable = false
    try
        stable, _ = _map_stability(sys, state, params_at_p, period; tol=config.stability_tol)
    catch err
        return _ReachBranchAtParam(
            index, branch_id, period, true, false, false, minimal_period, cycle, scale, confidence,
            "stability_unavailable: " * _continuation_error_message(err))
    end

    return _ReachBranchAtParam(
        index, branch_id, period, true, stable, minimal_period, cycle, scale, confidence)
end

# Linear interpolation of the branch state at `p` between the two bracketing continuation points
# (clamped to the nearest recorded point outside the recorded range).
function _reach_interpolated_state(entries::Vector{Tuple{Float64, Vector{Float64}}}, p::Float64)
    below_param = -Inf
    below_state = last(first(entries))
    above_param = Inf
    above_state = last(first(entries))
    have_below = false
    have_above = false
    for (pv, state) in entries
        if pv <= p && pv >= below_param
            below_param = pv
            below_state = state
            have_below = true
        end
        if pv >= p && pv <= above_param
            above_param = pv
            above_state = state
            have_above = true
        end
    end
    if have_below && have_above
        if above_param == below_param
            return copy(below_state)
        end
        t = (p - below_param) / (above_param - below_param)
        return below_state .+ t .* (above_state .- below_state)
    elseif have_below
        return copy(below_state)
    elseif have_above
        return copy(above_state)
    end
    # Fall back to the nearest recorded point.
    nearest = firstindex(entries)
    nearest_distance = abs(first(entries[nearest]) - p)
    for idx in (firstindex(entries) + 1):lastindex(entries)
        distance = abs(first(entries[idx]) - p)
        if distance < nearest_distance
            nearest = idx
            nearest_distance = distance
        end
    end
    return copy(last(entries[nearest]))
end

# --- per-cell classification ---

# Assign an already-reconstructed seed cycle to a stable branch identity (or an accounting
# category). Shared by the discrete and continuous seed classifiers so the seven-category decision
# (period-gated same-period matching, stable-only identity, ambiguity, unmatched vs outside-coverage)
# has one source of truth. Returns `(status_code, branch_index, match_distance, seed_period)`.
function _reach_classify_cycle(seed_cycle::Vector{Vector{Float64}},
                               seed_scale::Float64,
                               seed_period::Int,
                               config::BranchReachabilityConfig,
                               branches_at_p::Vector{_ReachBranchAtParam})
    best_stable_index = 0
    best_stable_dist = Inf
    second_stable_dist = Inf
    best_unstable_dist = Inf
    same_period_present = false
    stability_unavailable_present = false

    for branch in branches_at_p
        (branch.covered && branch.period == seed_period) || continue
        same_period_present = true
        if !branch.stability_available
            stability_unavailable_present = true
            continue
        end
        dist = _reach_normalized_distance(
            _reach_cycle_distance(seed_cycle, branch.cycle), seed_scale, branch.scale)
        if branch.stable
            if dist < best_stable_dist
                second_stable_dist = best_stable_dist
                best_stable_dist = dist
                best_stable_index = branch.index
            elseif dist < second_stable_dist
                second_stable_dist = dist
            end
        else
            best_unstable_dist = min(best_unstable_dist, dist)
        end
    end

    if best_stable_dist <= config.match_tolerance
        if second_stable_dist <= config.match_tolerance &&
           best_stable_dist > config.ambiguity_ratio * second_stable_dist
            return (_REACH_UNRESOLVED, 0, best_stable_dist, seed_period)
        end
        return (_REACH_MATCHED, best_stable_index, best_stable_dist, seed_period)
    end
    if best_unstable_dist <= config.match_tolerance
        return (_REACH_STABILITY_MISMATCH, 0, best_unstable_dist, seed_period)
    end
    if stability_unavailable_present
        return (_REACH_UNRESOLVED, 0, NaN, seed_period)
    end
    if same_period_present
        recorded = isfinite(best_stable_dist) ? best_stable_dist : best_unstable_dist
        return (_REACH_UNMATCHED, 0, recorded, seed_period)
    end
    return (_REACH_OUTSIDE_COVERAGE, 0, NaN, seed_period)
end

# Assign one discrete seed's terminal orbit to a branch identity (or an accounting category). Returns
# `(status_code, branch_index, match_distance, terminal_period)`.
function _reach_classify_seed(sys::DiscreteMap,
                              params_at_p::Vector{Float64},
                              x0::SVector,
                              transient::Int,
                              config::BranchReachabilityConfig,
                              branches_at_p::Vector{_ReachBranchAtParam})
    det = _detect_discrete_map_period(
        sys, params_at_p, x0, transient, config.max_period, config.precision, config.divergence_cutoff)
    if det.status == :diverged || det.status == :invalid_state
        return (_REACH_DIVERGED, 0, NaN, det.period)
    end
    if det.period <= 0
        return (_REACH_APERIODIC, 0, NaN, 0)
    end

    raw_cycle = _reach_reconstruct_cycle(sys, collect(Float64, det.final_point), params_at_p, det.period)
    seed_cycle = _reach_minimal_cycle(raw_cycle, config.precision)
    seed_period = length(seed_cycle)
    seed_scale = _reach_cycle_scale(seed_cycle)
    return _reach_classify_cycle(seed_cycle, seed_scale, seed_period, config, branches_at_p)
end

# --- provenance validation ---

function _reach_validate_branches(sys::DynamicalSystem,
                                  branches::AbstractVector{<:BranchResult},
                                  config::BranchReachabilityConfig)
    isempty(branches) && throw(ArgumentError(
        "branch_reachability requires at least one continuation branch."))
    config.param_index <= length(sys.param_names) || throw(ArgumentError(
        "branch_reachability: param_index=$(config.param_index) exceeds system '$(sys.name)' parameter count $(length(sys.param_names))."))
    for li in config.linked_param_indices
        li <= length(sys.param_names) || throw(ArgumentError(
            "branch_reachability: linked_param_indices contains $li, which exceeds system '$(sys.name)' parameter count $(length(sys.param_names))."))
        li != config.param_index || throw(ArgumentError(
            "branch_reachability: linked_param_indices must not contain the varied param_index=$(config.param_index)."))
    end
    expected_param = sys.param_names[config.param_index]
    for (idx, branch) in enumerate(branches)
        branch.system_name == sys.name || throw(ArgumentError(
            "branch_reachability: branch $idx is for system '$(branch.system_name)', but the analysis system is '$(sys.name)'."))
        branch.param_name == expected_param || throw(ArgumentError(
            "branch_reachability: branch $idx varies parameter $(branch.param_name), but param_index=$(config.param_index) selects $expected_param."))
    end
    if !isempty(config.branch_ids)
        length(config.branch_ids) == length(branches) || throw(ArgumentError(
            "branch_reachability: branch_ids has $(length(config.branch_ids)) entries but there are $(length(branches)) branches."))
        length(unique(config.branch_ids)) == length(config.branch_ids) || throw(ArgumentError(
            "branch_reachability: branch_ids must be unique; got $(config.branch_ids)."))
    end
    return nothing
end

# Strict provenance gate for an optionally-supplied precomputed basins census (evidence reuse). A
# supplied `BasinsResult` is accepted only when it proves parameter/config compatibility; the
# recomputed per-cell periodicity is then required to be consistent with it (a mismatch means the
# supplied census used incompatible detection settings and is rejected).
function _reach_validate_basins_reuse(basins::BasinsResult,
                                      sys::DiscreteMap,
                                      config::BranchReachabilityConfig,
                                      knot_index::Int,
                                      p::Float64,
                                      x_vals::Vector{Float64},
                                      y_vals::Vector{Float64},
                                      ic_template::Vector{Float64})
    basins.system_name == sys.name || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index is for system '$(basins.system_name)', not '$(sys.name)'."))
    _reach_float_repr_equal(basins.bif_param, p) || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index is at bif_param=$(basins.bif_param), but param_samples[$knot_index]=$p."))
    basins.max_period == config.max_period || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index has max_period=$(basins.max_period); config expects $(config.max_period)."))
    basins.x_index == config.x_index && basins.y_index == config.y_index || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index grid indices ($(basins.x_index), $(basins.y_index)) do not match config ($(config.x_index), $(config.y_index))."))
    length(basins.x_grid) == length(x_vals) && length(basins.y_grid) == length(y_vals) || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index grid shape $((length(basins.x_grid), length(basins.y_grid))) does not match the census grid $((length(x_vals), length(y_vals)))."))
    all(_reach_float_repr_equal(a, b) for (a, b) in zip(basins.x_grid, x_vals)) || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index x_grid does not match the census x grid."))
    all(_reach_float_repr_equal(a, b) for (a, b) in zip(basins.y_grid, y_vals)) || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index y_grid does not match the census y grid."))
    length(basins.ic_template) == length(ic_template) &&
        all(_reach_float_repr_equal(a, b) for (a, b) in zip(basins.ic_template, ic_template)) || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index ic_template does not match the census template."))
    return nothing
end

_reach_float_repr_equal(a, b) = _robust_float_repr_equal(a, b)

# --- main orchestration ---

"""
    branch_reachability(sys::DiscreteMap, branches, config::BranchReachabilityConfig; kwargs...)
        -> BranchReachabilityResult

Compute multistability-aware reachability for a set of continuation `branches`: at each parameter
knot in `config.param_samples`, run a basin initial-condition census and assign every seed's
terminal periodic orbit to a stable branch identity using period-gated, phase-invariant state-space
geometry.

Every seed is accounted for in exactly one category — `matched` (reaches one stable branch),
`unmatched` (settles to a same-period orbit not represented by any branch), `aperiodic`, `diverged`,
`unresolved` (two same-period stable branches too close to distinguish), `stability_mismatch`
(matches only an *unstable* branch — rejected from attracting fractions), or `outside_coverage` (its
period is not covered by any branch at that knot). The category counts partition the seed census, so
their fractions sum to one; per-branch `matched_fractions` use the same full-census denominator.

# Arguments
- `sys`: `DiscreteMap`, or a `ContinuousODE` (see that method — reachability runs on the Poincaré
  return map: full-state seeds, section-projected branch/seed cycles)
- `branches`: continuation `BranchResult`s over `config.param_index`; each is validated for system
  and parameter provenance
- `config`: `BranchReachabilityConfig`

# Keyword arguments
- `basins_crosscheck`: optional `Vector{BasinsResult}` aligned to `config.param_samples`. The
  reachability census is recomputed and checked against this independent evidence; it is not a
  cache/reuse shortcut. Cross-checking requires `divergence_cutoff == Inf` because `BasinsResult`
  does not apply or record a divergence cutoff.
- `log`: optional `f(msg::String)` progress callback

# Returns
`BranchReachabilityResult` with one `BranchReachabilitySample` per knot.
"""
function branch_reachability(sys::DiscreteMap,
                             branches::AbstractVector{<:BranchResult},
                             config::BranchReachabilityConfig;
                             basins_crosscheck::Union{Nothing, AbstractVector{BasinsResult}}=nothing,
                             log::Union{Nothing, Function}=nothing)
    _reach_validate_branches(sys, branches, config)

    dim = sys.dim
    (1 <= config.x_index <= dim && 1 <= config.y_index <= dim) || throw(ArgumentError(
        "branch_reachability: grid indices ($(config.x_index), $(config.y_index)) must lie in 1:$dim for system '$(sys.name)'."))
    ic_template = if isempty(config.ic_template)
        zeros(Float64, dim)
    else
        length(config.ic_template) == dim || throw(ArgumentError(
            "branch_reachability: ic_template length $(length(config.ic_template)) does not match system state dim $dim."))
        collect(Float64, config.ic_template)
    end

    if !isnothing(basins_crosscheck)
        isfinite(config.divergence_cutoff) && throw(ArgumentError(
            "branch_reachability: basins_crosscheck requires divergence_cutoff = Inf because BasinsResult does not apply or record a divergence cutoff."))
        length(basins_crosscheck) == length(config.param_samples) || throw(ArgumentError(
            "branch_reachability: basins_crosscheck has $(length(basins_crosscheck)) entries but there are $(length(config.param_samples)) parameter samples."))
    end

    required = max(config.param_index, maximum(config.linked_param_indices; init=0))
    base_params = if isempty(config.base_params)
        zeros(Float64, required)
    else
        length(config.base_params) >= required || throw(ArgumentError(
            "branch_reachability: base_params length $(length(config.base_params)) is shorter than the referenced parameter slot $required."))
        collect(Float64, config.base_params)
    end

    branch_ids = isempty(config.branch_ids) ?
        ["branch-$(idx)" for idx in eachindex(branches)] : copy(config.branch_ids)
    branch_declared_periods = [max(branch.period, 1) for branch in branches]

    x_vals = collect(range(config.x_min, config.x_max, length=config.x_steps + 1))
    y_vals = collect(range(config.y_min, config.y_max, length=config.y_steps + 1))
    nx, ny = length(x_vals), length(y_vals)
    base_ic = SVector{dim, Float64}(ic_template)
    transient = config.iterations - (config.max_period + 1)

    samples = Vector{BranchReachabilitySample}(undef, length(config.param_samples))
    for (knot_index, p) in enumerate(config.param_samples)
        isnothing(log) || log("branch_reachability: knot $knot_index/$(length(config.param_samples)) at $(sys.param_names[config.param_index])=$p")
        params_at_p = inject_param(base_params, config.param_index, p, config.linked_param_indices)

        branches_at_p = [
            _reach_branch_state_at_param(sys, branch, idx, branch_ids[idx], p, base_params, config)
            for (idx, branch) in enumerate(branches)
        ]

        assignment = zeros(Int, nx, ny)
        status = zeros(Int, nx, ny)
        match_distance = fill(NaN, nx, ny)
        terminal_period = zeros(Int, nx, ny)

        census! = function (i_range)
            for i in i_range
                for j in 1:ny
                    x0 = setindex(base_ic, x_vals[i], config.x_index)
                    x0 = setindex(x0, y_vals[j], config.y_index)
                    code, branch_index, dist, tperiod = _reach_classify_seed(
                        sys, params_at_p, x0, transient, config, branches_at_p)
                    status[i, j] = code
                    assignment[i, j] = branch_index
                    match_distance[i, j] = dist
                    terminal_period[i, j] = tperiod
                end
            end
        end
        if config.threaded
            Threads.@threads for i in 1:nx
                census!(i:i)
            end
        else
            census!(1:nx)
        end

        if !isnothing(basins_crosscheck)
            _reach_validate_basins_reuse(
                basins_crosscheck[knot_index], sys, config, knot_index, p, x_vals, y_vals, ic_template)
            _reach_cross_check_periodicity(
                basins_crosscheck[knot_index].periodicity, terminal_period, knot_index)
        end

        samples[knot_index] = _reach_assemble_sample(
            p, branch_ids, branches_at_p, branch_declared_periods,
            assignment, status, match_distance, terminal_period, config)
    end

    return BranchReachabilityResult(
        sys.name,
        sys.param_names[config.param_index],
        config.param_index,
        copy(config.linked_param_indices),
        base_params,
        x_vals,
        y_vals,
        config.x_index,
        config.y_index,
        ic_template,
        config.max_period,
        config.precision,
        config.divergence_cutoff,
        config.param_tolerance,
        config.match_tolerance,
        config.ambiguity_ratio,
        config.stability_tol,
        branch_ids,
        branch_declared_periods,
        samples,
        copy(_REACH_STATUS_LABEL_BY_CODE),
        now(),
    )
end

function _reach_cross_check_periodicity(supplied::Matrix{Int}, recomputed::Matrix{Int}, knot_index::Int)
    size(supplied) == size(recomputed) || throw(ArgumentError(
        "branch_reachability: supplied basins result $knot_index periodicity shape $(size(supplied)) does not match the census $(size(recomputed))."))
    for idx in eachindex(supplied)
        supplied[idx] == recomputed[idx] || throw(ArgumentError(
            "branch_reachability: supplied basins result $knot_index periodicity is inconsistent with the recomputed census at linear index $idx " *
            "($(supplied[idx]) vs $(recomputed[idx])); the supplied census used incompatible detection settings."))
    end
    return nothing
end

function _reach_assemble_sample(p::Float64,
                                branch_ids::Vector{String},
                                branches_at_p::Vector{_ReachBranchAtParam},
                                branch_declared_periods::Vector{Int},
                                assignment::Matrix{Int},
                                status::Matrix{Int},
                                match_distance::Matrix{Float64},
                                terminal_period::Matrix{Int},
                                config::BranchReachabilityConfig)
    n_branches = length(branch_ids)
    n_seeds = length(status)
    matched_counts = zeros(Int, n_branches)
    counts = zeros(Int, 7)
    for idx in eachindex(status)
        code = status[idx]
        (1 <= code <= 7) && (counts[code] += 1)
        if code == _REACH_MATCHED
            branch_index = assignment[idx]
            (1 <= branch_index <= n_branches) && (matched_counts[branch_index] += 1)
        end
    end
    denom = n_seeds == 0 ? 1 : n_seeds
    matched_fractions = matched_counts ./ denom

    branch_periods = [branch.covered ? branch.period : branch_declared_periods[branch.index]
                      for branch in branches_at_p]
    branch_stable = [branch.stable for branch in branches_at_p]
    branch_covered = [branch.covered for branch in branches_at_p]
    branch_confidence = [branch.confidence for branch in branches_at_p]
    stability_available = [branch.stability_available for branch in branches_at_p]

    diagnostics = Dict{String, Any}(
        "paramTolerance" => config.param_tolerance,
        "matchTolerance" => config.match_tolerance,
        "ambiguityRatio" => config.ambiguity_ratio,
        "coveredBranchCount" => count(branch_covered),
        "stableBranchCount" => count(
            i -> branch_covered[i] && stability_available[i] && branch_stable[i],
            eachindex(branch_covered)),
        "stabilityUnavailableBranchCount" => count(
            i -> branch_covered[i] && !stability_available[i], eachindex(branch_covered)),
    )
    # Preserve the exact reason for any branch excluded by a recoverable numerical failure
    # (currently: return-map stability could not be assessed), keyed by branch index.
    branch_notes = Dict{String, Any}(
        string(branch.index) => branch.note for branch in branches_at_p if !isempty(branch.note))
    isempty(branch_notes) || (diagnostics["branchNotes"] = branch_notes)

    return BranchReachabilitySample(
        p,
        copy(branch_ids),
        branch_periods,
        branch_stable,
        branch_covered,
        branch_confidence,
        matched_counts,
        matched_fractions,
        n_seeds,
        counts[_REACH_MATCHED],
        counts[_REACH_UNMATCHED],
        counts[_REACH_APERIODIC],
        counts[_REACH_DIVERGED],
        counts[_REACH_UNRESOLVED],
        counts[_REACH_STABILITY_MISMATCH],
        counts[_REACH_OUTSIDE_COVERAGE],
        assignment,
        status,
        match_distance,
        terminal_period,
        diagnostics,
    )
end

# --- continuous-time (Poincaré return-map) reachability ---
#
# For a ContinuousODE the reachability census runs on the Poincaré return map, reusing the existing
# ODE/Poincaré kernels rather than duplicating solve/callback logic:
#   * seeds are full-state initial conditions on the (x, y) grid (as in basins_of_attraction);
#   * a seed's terminal orbit is detected by `_detect_continuous_poincare_period` and its q-crossing
#     cycle is reconstructed with `_collect_poincare_points` (projected section coordinates);
#   * a branch's state at a knot is the section-projected fixed point, interpolated from the
#     bracketing branch points and Newton-corrected via `_map_residual`/`_newton_fd`;
#   * branch stability uses the polymorphic `_map_stability` (finite-difference return-map multipliers).
# The seven-category assignment (`_reach_classify_cycle`), sample assembly, and provenance validation
# are shared with the discrete path, so the semantics match exactly.

# Resolve the config integration horizon: `Inf` ⇒ the internal `tspan_hint`-scaled default (nothing),
# matching `basins_of_attraction`; a finite value caps each Poincaré segment.
_reach_resolve_ode_tmax(ode_tmax::Float64) = isfinite(ode_tmax) ? ode_tmax : nothing

# Reconstruct a period-`period` Poincaré cycle as `period` successive projected section crossings
# starting from a full state on (or near) the section. Returns `nothing` if the integrator did not
# produce the full set of finite crossings.
function _reach_reconstruct_continuous_cycle(sys::ContinuousODE,
                                             full_state::AbstractVector,
                                             params::AbstractVector,
                                             period::Int,
                                             config::BranchReachabilityConfig,
                                             solver,
                                             tmax::Union{Nothing, Float64})
    pts = _collect_poincare_points(
        sys,
        params;
        initial_point=collect(Float64, full_state),
        crossings=period,
        transient=0,
        solver=solver,
        reltol=config.ode_reltol,
        abstol=config.ode_abstol,
        projected=true,
        tmax=tmax,
        min_crossing_time=config.min_crossing_time,
        divergence_cutoff=config.divergence_cutoff,
    )
    length(pts) == period || return nothing
    all(pt -> all(isfinite, pt), pts) || return nothing
    return pts
end

# Select/solve one branch's projected section state at a knot. As in the discrete path a branch is
# covered only when the knot lies inside its recorded parameter range (± `param_tolerance`); the
# projected state is interpolated between the bracketing branch points and then Newton-corrected to
# the exact period-`period` Poincaré fixed point at the knot. If the return-map Newton solve fails to
# converge (or the cycle cannot be reconstructed), the branch is marked uncovered rather than
# matching seeds against an uncorrected distant estimate.
function _reach_branch_state_at_param(sys::ContinuousODE,
                                      branch::BranchResult,
                                      index::Int,
                                      branch_id::String,
                                      p::Float64,
                                      base_params::Vector{Float64},
                                      config::BranchReachabilityConfig,
                                      solver,
                                      tmax::Union{Nothing, Float64})
    proj_dim = state_dim(sys)
    period = max(branch.period, 1)
    uncovered = _ReachBranchAtParam(
        index, branch_id, period, false, false, period, Vector{Float64}[], 1.0, 0.0)

    entries = Tuple{Float64, Vector{Float64}}[]
    for pt in _branch_points(branch)
        hasproperty(pt, :param) || continue
        pv = Float64(pt.param)
        isfinite(pv) || continue
        all(i -> hasproperty(pt, Symbol(:x, i)), 1:proj_dim) || continue
        state = _reach_point_state(pt, proj_dim)
        all(isfinite, state) || continue
        push!(entries, (pv, state))
    end
    isempty(entries) && return uncovered

    params_min = minimum(first(entry) for entry in entries)
    params_max = maximum(first(entry) for entry in entries)
    if p < params_min - config.param_tolerance || p > params_max + config.param_tolerance
        return uncovered
    end

    seed_state = _reach_interpolated_state(entries, p)
    params_at_p = inject_param(base_params, config.param_index, p, config.linked_param_indices)
    residual = x -> _map_residual(
        sys, x, params_at_p, period;
        solver=solver, reltol=config.ode_reltol, abstol=config.ode_abstol,
        tmax=tmax, min_crossing_time=config.min_crossing_time)
    refined_state, converged = _newton_fd(
        residual, seed_state, config.newton_tol, config.newton_max_iter, config.ode_fd_step)
    converged || return uncovered

    cycle_full = _reach_reconstruct_continuous_cycle(
        sys, _lift_section_state(sys.section, refined_state, sys.dim),
        params_at_p, period, config, solver, tmax)
    isnothing(cycle_full) && return uncovered
    cycle = _reach_minimal_cycle(cycle_full, config.precision)
    minimal_period = length(cycle)
    scale = _reach_cycle_scale(cycle)

    # Newton converged and the projected cycle reconstructed, so this branch state is exact.
    confidence = 1.0
    # As in the discrete path: a failed return-map multiplier computation cannot be honestly
    # reported as "unstable". The parameter coverage remains true, while matching is withheld and the
    # reason is preserved so same-period seeds become unresolved rather than outside-coverage.
    stable = false
    try
        stable, _ = _map_stability(
            sys, refined_state, params_at_p, period;
            tol=config.stability_tol, fd_step=config.ode_fd_step, solver=solver,
            reltol=config.ode_reltol, abstol=config.ode_abstol,
            tmax=tmax, min_crossing_time=config.min_crossing_time)
    catch err
        return _ReachBranchAtParam(
            index, branch_id, period, true, false, false, minimal_period, cycle, scale, confidence,
            "stability_unavailable: " * _continuation_error_message(err))
    end

    return _ReachBranchAtParam(
        index, branch_id, period, true, stable, minimal_period, cycle, scale, confidence)
end

# Assign one continuous seed's terminal Poincaré orbit to a branch identity (or an accounting
# category). Returns `(status_code, branch_index, match_distance, terminal_period)`.
function _reach_classify_seed(sys::ContinuousODE,
                              params_at_p::Vector{Float64},
                              x0::AbstractVector,
                              transient::Int,
                              config::BranchReachabilityConfig,
                              branches_at_p::Vector{_ReachBranchAtParam},
                              solver,
                              tmax::Union{Nothing, Float64})
    det = _detect_continuous_poincare_period(
        sys, params_at_p;
        initial_point=collect(Float64, x0),
        transient=transient,
        max_period=config.max_period,
        precision=config.precision,
        solver=solver,
        reltol=config.ode_reltol,
        abstol=config.ode_abstol,
        projected=true,
        tmax=tmax,
        min_crossing_time=config.min_crossing_time,
        divergence_cutoff=config.divergence_cutoff,
        return_crossing_diagnostics=false,
    )
    if det.status == :diverged || det.status == :invalid_state
        return (_REACH_DIVERGED, 0, NaN, 0)
    end
    # A seed the return-map integrator could not resolve to a bounded periodic orbit — integration
    # failed or too few section crossings within the horizon (e.g. an equilibrium that never returns
    # to the section, or a solver failure) — is honestly `unresolved`: neither a confirmed periodic
    # match, nor confirmed aperiodic chaos, nor divergence.
    if det.status == :integration_failed || det.status == :insufficient_crossings
        return (_REACH_UNRESOLVED, 0, NaN, 0)
    end
    if det.period <= 0
        return (_REACH_APERIODIC, 0, NaN, 0)
    end

    cycle_full = _reach_reconstruct_continuous_cycle(
        sys, det.final_point, params_at_p, det.period, config, solver, tmax)
    isnothing(cycle_full) && return (_REACH_UNRESOLVED, 0, NaN, det.period)
    seed_cycle = _reach_minimal_cycle(cycle_full, config.precision)
    seed_period = length(seed_cycle)
    seed_scale = _reach_cycle_scale(seed_cycle)
    return _reach_classify_cycle(seed_cycle, seed_scale, seed_period, config, branches_at_p)
end

"""
    branch_reachability(sys::ContinuousODE, branches, config::BranchReachabilityConfig; kwargs...)
        -> BranchReachabilityResult

Continuous-time reachability on the Poincaré return map. Seeds are full-state initial conditions on
the `(x, y)` grid; each seed's terminal orbit is detected on the section and its q-crossing cycle
reconstructed in projected section coordinates. Branch states are the section-projected fixed points,
interpolated from the bracketing continuation points and Newton-corrected at the exact knot; a branch
whose return-map Newton solve does not converge is reported uncovered/unavailable at that knot rather
than matched against a distant uncorrected estimate. The seven-category partition, stable-only branch
identity, and full-census fractions are identical to the `DiscreteMap` method.

# Keyword arguments
- `basins_crosscheck`: **not accepted** for `ContinuousODE` — the return-map census cannot prove
  identical crossing semantics against an independent `BasinsResult` census (warm-up, solver, and
  horizon may differ), so it is rejected rather than pretending parity. Passing it throws.
- `log`: optional `f(msg::String)` progress callback

The ODE integration is configured by the `ode_solver` (resolved by `select_ode_solver`),
`ode_reltol` / `ode_abstol`, `min_crossing_time`, `ode_fd_step`, and `ode_tmax` config fields.
"""
function branch_reachability(sys::ContinuousODE,
                             branches::AbstractVector{<:BranchResult},
                             config::BranchReachabilityConfig;
                             basins_crosscheck::Union{Nothing, AbstractVector{BasinsResult}}=nothing,
                             log::Union{Nothing, Function}=nothing)
    _reach_validate_branches(sys, branches, config)

    isnothing(basins_crosscheck) || throw(ArgumentError(
        "branch_reachability: basins_crosscheck is not accepted for ContinuousODE systems. The " *
        "Poincaré return-map census cannot prove identical crossing semantics against an independent " *
        "BasinsResult census (warm-up, solver, and integration horizon may differ), so cross-checking " *
        "is rejected rather than pretending parity. Received system '$(sys.name)'."))

    dim = sys.dim
    (1 <= config.x_index <= dim && 1 <= config.y_index <= dim) || throw(ArgumentError(
        "branch_reachability: grid indices ($(config.x_index), $(config.y_index)) must lie in 1:$dim for system '$(sys.name)'."))
    ic_template = if isempty(config.ic_template)
        zeros(Float64, dim)
    else
        length(config.ic_template) == dim || throw(ArgumentError(
            "branch_reachability: ic_template length $(length(config.ic_template)) does not match system state dim $dim."))
        collect(Float64, config.ic_template)
    end

    # Explicit section/projection/template dimension checks: branch states live in projected section
    # coordinates, so an inconsistent section would silently mis-align branch and seed cycles.
    proj_dim = state_dim(sys)
    proj_dim >= 1 || throw(ArgumentError(
        "branch_reachability: system '$(sys.name)' has empty section projection; a projected Poincaré coordinate is required."))
    all(i -> 1 <= i <= dim, sys.section.projection) || throw(ArgumentError(
        "branch_reachability: section projection indices $(sys.section.projection) must lie in 1:$dim for '$(sys.name)'."))
    if !isempty(sys.section.template)
        length(sys.section.template) == dim || throw(ArgumentError(
            "branch_reachability: section template length $(length(sys.section.template)) does not match system state dim $dim for '$(sys.name)'."))
    end

    solver = select_ode_solver(config.ode_solver)
    tmax = _reach_resolve_ode_tmax(config.ode_tmax)

    required = max(config.param_index, maximum(config.linked_param_indices; init=0))
    base_params = if isempty(config.base_params)
        zeros(Float64, required)
    else
        length(config.base_params) >= required || throw(ArgumentError(
            "branch_reachability: base_params length $(length(config.base_params)) is shorter than the referenced parameter slot $required."))
        collect(Float64, config.base_params)
    end

    branch_ids = isempty(config.branch_ids) ?
        ["branch-$(idx)" for idx in eachindex(branches)] : copy(config.branch_ids)
    branch_declared_periods = [max(branch.period, 1) for branch in branches]

    x_vals = collect(range(config.x_min, config.x_max, length=config.x_steps + 1))
    y_vals = collect(range(config.y_min, config.y_max, length=config.y_steps + 1))
    nx, ny = length(x_vals), length(y_vals)
    base_ic = collect(Float64, ic_template)
    transient = config.iterations - (config.max_period + 1)

    samples = Vector{BranchReachabilitySample}(undef, length(config.param_samples))
    for (knot_index, p) in enumerate(config.param_samples)
        isnothing(log) || log("branch_reachability: knot $knot_index/$(length(config.param_samples)) at $(sys.param_names[config.param_index])=$p")
        params_at_p = inject_param(base_params, config.param_index, p, config.linked_param_indices)

        branches_at_p = [
            _reach_branch_state_at_param(sys, branch, idx, branch_ids[idx], p, base_params, config, solver, tmax)
            for (idx, branch) in enumerate(branches)
        ]

        assignment = zeros(Int, nx, ny)
        status = zeros(Int, nx, ny)
        match_distance = fill(NaN, nx, ny)
        terminal_period = zeros(Int, nx, ny)

        census! = function (i_range)
            for i in i_range
                for j in 1:ny
                    x0 = copy(base_ic)
                    x0[config.x_index] = x_vals[i]
                    x0[config.y_index] = y_vals[j]
                    code, branch_index, dist, tperiod = _reach_classify_seed(
                        sys, params_at_p, x0, transient, config, branches_at_p, solver, tmax)
                    status[i, j] = code
                    assignment[i, j] = branch_index
                    match_distance[i, j] = dist
                    terminal_period[i, j] = tperiod
                end
            end
        end
        if config.threaded
            Threads.@threads for i in 1:nx
                census!(i:i)
            end
        else
            census!(1:nx)
        end

        samples[knot_index] = _reach_assemble_sample(
            p, branch_ids, branches_at_p, branch_declared_periods,
            assignment, status, match_distance, terminal_period, config)
    end

    return BranchReachabilityResult(
        sys.name,
        sys.param_names[config.param_index],
        config.param_index,
        copy(config.linked_param_indices),
        base_params,
        x_vals,
        y_vals,
        config.x_index,
        config.y_index,
        ic_template,
        config.max_period,
        config.precision,
        config.divergence_cutoff,
        config.param_tolerance,
        config.match_tolerance,
        config.ambiguity_ratio,
        config.stability_tol,
        branch_ids,
        branch_declared_periods,
        samples,
        copy(_REACH_STATUS_LABEL_BY_CODE),
        now(),
    )
end

# --- accessors ---

"""    reachability_category_counts(sample) -> NamedTuple — the seven-category seed partition + total."""
function reachability_category_counts(sample::BranchReachabilitySample)
    return (
        matched = sample.n_matched,
        unmatched = sample.n_unmatched,
        aperiodic = sample.n_aperiodic,
        diverged = sample.n_diverged,
        unresolved = sample.n_unresolved,
        stability_mismatch = sample.n_stability_mismatch,
        outside_coverage = sample.n_outside_coverage,
        total = sample.n_seeds,
    )
end

"""    reachability_category_fractions(sample) -> NamedTuple — category fractions (sum to one over a non-empty census)."""
function reachability_category_fractions(sample::BranchReachabilitySample)
    denom = sample.n_seeds == 0 ? 1 : sample.n_seeds
    return (
        matched = sample.n_matched / denom,
        unmatched = sample.n_unmatched / denom,
        aperiodic = sample.n_aperiodic / denom,
        diverged = sample.n_diverged / denom,
        unresolved = sample.n_unresolved / denom,
        stability_mismatch = sample.n_stability_mismatch / denom,
        outside_coverage = sample.n_outside_coverage / denom,
    )
end

"""    branch_reachability_fractions(sample) -> Dict{String,Float64} — branch id → matched basin fraction."""
function branch_reachability_fractions(sample::BranchReachabilitySample)
    return Dict{String, Float64}(
        sample.branch_ids[k] => sample.matched_fractions[k] for k in eachindex(sample.branch_ids))
end
