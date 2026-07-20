# Parameter-robustness / tolerance fields over a classified 2D operating map (roadmap T2.4).
#
# Two clearly separated layers share one classified-map surrogate (the (a, b) regime grid derived
# from a `BifurcationMapResult`, optionally sharpened by per-cell status codes):
#
#   A) `regime_boundary_distances` — a deterministic regime-boundary margin field: for every
#      known-regime cell, the physical Euclidean distance to the nearest regime boundary, plus the
#      per-parameter (per-axis) distances. Computed with an O(NM) generalized separable
#      squared-Euclidean distance transform evaluated at the *true* grid coordinates (supports
#      monotone nonuniform rectilinear grids), so there is no index-distance approximation and no
#      new dependency.
#
#   B) `tolerance_regime_map` — a probabilistic regime map: at each nominal cell the two parameters
#      are independently perturbed by stated component tolerances and each perturbed operating point
#      is classified by nearest physical-grid-cell lookup over the same classified surrogate. This is
#      Monte-Carlo propagation through a *finite classified map*, not model reruns and not a
#      closed-form tolerance proof; unknown and out-of-domain mass are retained (never renormalized).
#
# Unknown cells are never silently promoted to a physical regime: an unresolved cell has no margin
# (marked invalid) and forms a boundary for its known neighbours ("margin to unknown evidence").

# Regime label sentinels for the map-derived classification. Genuine periodic regimes carry their
# positive period; the two non-periodic *physical* outcomes get reserved negative labels so they can
# never collide with a period value; unknown/unresolved cells carry 0 and `resolved = false`.
const _REGIME_UNKNOWN = 0
const _REGIME_APERIODIC = -1
const _REGIME_DIVERGED = -2

"""
    RegimeBoundaryResult

Plain-data result of `regime_boundary_distances` (roadmap T2.4 layer A). All per-cell matrices share
the `(na, nb)` operating-map grid shape and the `[i, j] ↔ (a_grid[i], b_grid[j])` orientation of the
source `BifurcationMapResult`.

# Fields
- `system_name`, `param_names`, `a_grid`, `b_grid`: provenance and the physical parameter axes
- `labels`: per-cell regime label (positive period, `-1` aperiodic, `-2` diverged, `0` unknown)
- `resolved`: known-regime mask; `unknown_mask == .!resolved`
- `valid`: cells with a computed margin (equal to `resolved` — an unresolved query cell is invalid)
- `boundary_mask`: cells that lie on a regime boundary (the distance-transform sources, margin `0`)
- `boundary_kind`: `0` interior, `1` adjacent to a different known regime, `2` adjacent to an unknown
  cell, `3` both
- `distance`: primary Euclidean margin — distance from a cell centre to the nearest boundary cell
  centre, with the `edge_policy` applied. `NaN` where invalid. Boundary cells have `0`. This is a
  finite-grid convention: the reported distance is to the nearest boundary *cell centre*, within one
  cell-diagonal of the true interface.
- `distance_a`, `distance_b`: per-axis margins — the distance to the nearest boundary cell *along the
  same grid line* (varying only `a` / only `b`). `Inf` when that grid line contains no boundary cell;
  `NaN` where invalid. These are the raw per-parameter drift margins and are not edge-capped.
- `edge_censored`: under `:censored` policy, `true` where the reported margin is capped by the
  distance to the sampled-domain edge and is therefore a *lower bound* (a regime change may lie just
  outside the sampled window)
- `edge_policy`: `:censored`, `:boundary`, or `:ignore`
- `status_evidence`: `true` when per-cell status codes distinguished aperiodic / diverged /
  unresolved cells; `false` for the reduced periodicity-only classification (period `0` ⇒ unknown)
- `convention`: human-readable statement of the margin convention
- `timestamp`
"""
struct RegimeBoundaryResult
    system_name::String
    param_names::Tuple{Symbol, Symbol}
    a_grid::Vector{Float64}
    b_grid::Vector{Float64}
    labels::Matrix{Int}
    resolved::Matrix{Bool}
    valid::Matrix{Bool}
    boundary_mask::Matrix{Bool}
    boundary_kind::Matrix{Int}
    distance::Matrix{Float64}
    distance_a::Matrix{Float64}
    distance_b::Matrix{Float64}
    edge_censored::Matrix{Bool}
    edge_policy::Symbol
    status_evidence::Bool
    convention::String
    timestamp::DateTime
end

"""
    ToleranceMapResult

Plain-data result of `tolerance_regime_map` (roadmap T2.4 layer B). Per-cell matrices share the
`(na, nb)` operating-map grid shape and orientation.

# Fields
- `system_name`, `param_names`, `a_grid`, `b_grid`: provenance and the physical parameter axes
- `regime_labels`: sorted distinct physical regime labels tracked (the resolved labels of the grid)
- `regime_probability`: `label → probability matrix` — the probability a perturbed sample from each
  nominal cell lands in that regime
- `nominal_regime`, `nominal_resolved`: the nominal (unperturbed) regime label and whether it is a
  known regime
- `nominal_probability`: probability a perturbed sample stays in the nominal cell's category (its
  regime when resolved, else the unknown category)
- `dominant_regime`, `dominant_probability`: the most probable *physical* regime and its probability
  (`dominant_regime = 0`, `dominant_probability = 0` when no physical regime receives any mass)
- `unknown_probability`, `out_of_domain_probability`: mass landing in unresolved cells / outside the
  sampled domain — retained, never renormalized away
- `entropy`: Shannon entropy (bits) of the full categorical distribution (regimes + unknown + OOD)
- `nominal_standard_error`: binomial standard error of `nominal_probability`
- `nominal_ci_lower`, `nominal_ci_upper`: Wilson 95% score interval for `nominal_probability`
- `n_samples`: Monte-Carlo samples per cell requested
- `n_effective`: samples actually drawn per cell — `0` for the exact deterministic collapse (both
  tolerances zero), otherwise `n_samples`
- `tolerance_a`, `tolerance_b`, `seed`: the propagation settings
- `status_evidence`: whether status codes sharpened the classification
- `convention`: human-readable statement of the propagation convention
- `timestamp`
"""
struct ToleranceMapResult
    system_name::String
    param_names::Tuple{Symbol, Symbol}
    a_grid::Vector{Float64}
    b_grid::Vector{Float64}
    regime_labels::Vector{Int}
    regime_probability::Dict{Int, Matrix{Float64}}
    nominal_regime::Matrix{Int}
    nominal_resolved::Matrix{Bool}
    nominal_probability::Matrix{Float64}
    dominant_regime::Matrix{Int}
    dominant_probability::Matrix{Float64}
    unknown_probability::Matrix{Float64}
    out_of_domain_probability::Matrix{Float64}
    entropy::Matrix{Float64}
    nominal_standard_error::Matrix{Float64}
    nominal_ci_lower::Matrix{Float64}
    nominal_ci_upper::Matrix{Float64}
    n_samples::Int
    n_effective::Int
    tolerance_a::AbstractTolerance
    tolerance_b::AbstractTolerance
    seed::UInt64
    status_evidence::Bool
    convention::String
    timestamp::DateTime
end

# --- shared grid + classification helpers ---

_regime_strictly_increasing(v::AbstractVector) =
    all(isfinite, v) && (length(v) <= 1 || all(v[k] < v[k + 1] for k in 1:(length(v) - 1)))

function _regime_validate_grids(a_grid::AbstractVector{Float64},
                                b_grid::AbstractVector{Float64},
                                labels::AbstractMatrix{<:Integer},
                                resolved::AbstractMatrix{Bool})
    na, nb = length(a_grid), length(b_grid)
    (na >= 1 && nb >= 1) || throw(ArgumentError(
        "regime map requires non-empty a_grid and b_grid; got sizes ($na, $nb)."))
    size(labels) == (na, nb) || throw(ArgumentError(
        "regime map labels size $(size(labels)) must match the grid ($na, $nb)."))
    size(resolved) == (na, nb) || throw(ArgumentError(
        "regime map resolved-mask size $(size(resolved)) must match the grid ($na, $nb)."))
    _regime_strictly_increasing(a_grid) || throw(ArgumentError(
        "regime map a_grid must be strictly increasing and finite (monotone nonuniform is allowed)."))
    _regime_strictly_increasing(b_grid) || throw(ArgumentError(
        "regime map b_grid must be strictly increasing and finite (monotone nonuniform is allowed)."))
    return nothing
end

# Classify a `BifurcationMapResult` into (labels, resolved) with strict provenance validation of any
# supplied status evidence. When status codes are absent the classification is periodicity-only:
# period > 0 ⇒ that periodic regime, period ≤ 0 ⇒ unknown (aperiodic, diverged and unresolved cells
# are indistinguishable and are all conservatively treated as unknown, never as a physical regime).
function _regime_map_classification(map_result::BifurcationMapResult;
                                    cells::Union{Nothing, MapCellGrid}=nothing,
                                    status_codes::Union{Nothing, AbstractMatrix{<:Integer}}=nothing,
                                    aperiodic_is_regime::Bool=true,
                                    diverged_is_regime::Bool=true)
    periodicity = map_result.periodicity
    na, nb = size(periodicity)

    (cells === nothing || status_codes === nothing) || throw(ArgumentError(
        "regime map: supply at most one of `cells` (MapCellGrid) or `status_codes`, not both."))

    status = nothing
    if cells !== nothing
        size(cells.status_codes) == (na, nb) || throw(ArgumentError(
            "regime map: cells.status_codes size $(size(cells.status_codes)) does not match the map grid ($na, $nb)."))
        size(cells.periodicity) == (na, nb) || throw(ArgumentError(
            "regime map: cells.periodicity size $(size(cells.periodicity)) does not match the map grid ($na, $nb)."))
        cells.periodicity == periodicity || throw(ArgumentError(
            "regime map: cells.periodicity does not match the BifurcationMapResult periodicity (status evidence is from a different sweep)."))
        status = cells.status_codes
    elseif status_codes !== nothing
        size(status_codes) == (na, nb) || throw(ArgumentError(
            "regime map: status_codes size $(size(status_codes)) does not match the map grid ($na, $nb)."))
        status = status_codes
    end

    labels = Matrix{Int}(undef, na, nb)
    resolved = Matrix{Bool}(undef, na, nb)

    periodic_code = _map_status_code(:periodic)
    aperiodic_code = _map_status_code(:aperiodic_or_high_period)
    diverged_code = _map_status_code(:diverged)

    @inbounds for j in 1:nb, i in 1:na
        if status === nothing
            period = periodicity[i, j]
            if period > 0
                labels[i, j] = period
                resolved[i, j] = true
            else
                labels[i, j] = _REGIME_UNKNOWN
                resolved[i, j] = false
            end
        else
            code = Int(status[i, j])
            if code == periodic_code && periodicity[i, j] > 0
                labels[i, j] = periodicity[i, j]
                resolved[i, j] = true
            elseif code == aperiodic_code
                labels[i, j] = _REGIME_APERIODIC
                resolved[i, j] = aperiodic_is_regime
            elseif code == diverged_code
                labels[i, j] = _REGIME_DIVERGED
                resolved[i, j] = diverged_is_regime
            else
                labels[i, j] = _REGIME_UNKNOWN
                resolved[i, j] = false
            end
        end
    end

    return (labels = labels, resolved = resolved,
            a_grid = collect(Float64, map_result.a_grid),
            b_grid = collect(Float64, map_result.b_grid),
            param_names = map_result.param_names,
            system_name = map_result.system_name,
            status_evidence = status !== nothing)
end

# --- generalized (true-coordinate) distance transform ---

# 1D generalized squared-Euclidean distance transform (Felzenszwalb & Huttenlocher) evaluated at the
# *true* sample positions `coords` (strictly increasing, possibly nonuniform). Computes
#   D[q] = min_p ( (coords[q] - coords[p])^2 + f[p] )
# by tracking the lower envelope of the parabolas rooted at each finite-cost site. O(n) per line;
# `v` (parabola indices) and `z` (envelope breakpoints, length n+1) are reused work buffers.
function _regime_dt_1d!(D::Vector{Float64}, f::Vector{Float64}, coords::Vector{Float64},
                        v::Vector{Int}, z::Vector{Float64})
    n = length(f)
    n == 0 && return D
    start = 0
    @inbounds for q in 1:n
        if isfinite(f[q])
            start = q
            break
        end
    end
    if start == 0
        fill!(D, Inf)
        return D
    end
    k = 1
    @inbounds begin
        v[1] = start
        z[1] = -Inf
        z[2] = Inf
        s = 0.0
        for q in (start + 1):n
            fq = f[q]
            isfinite(fq) || continue
            cq = coords[q]
            while true
                p = v[k]
                s = ((fq + cq * cq) - (f[p] + coords[p] * coords[p])) / (2 * (cq - coords[p]))
                # z[1] = -Inf guarantees the pop loop stops at k == 1.
                if s <= z[k]
                    k -= 1
                else
                    break
                end
            end
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = Inf
        end
        j = 1
        for q in 1:n
            cq = coords[q]
            while z[j + 1] < cq
                j += 1
            end
            p = v[j]
            d = cq - coords[p]
            D[q] = d * d + f[p]
        end
    end
    return D
end

# Transform every line of `M` along dimension `dim` (1 = down columns over `coords = a_grid`,
# 2 = across rows over `coords = b_grid`), returning the squared-distance matrix.
function _regime_dt_axis(M::Matrix{Float64}, coords::Vector{Float64}, dim::Int)
    na, nb = size(M)
    out = Matrix{Float64}(undef, na, nb)
    n = dim == 1 ? na : nb
    fbuf = Vector{Float64}(undef, n)
    dbuf = Vector{Float64}(undef, n)
    v = Vector{Int}(undef, n)
    z = Vector{Float64}(undef, n + 1)
    if dim == 1
        @inbounds for j in 1:nb
            for i in 1:na
                fbuf[i] = M[i, j]
            end
            _regime_dt_1d!(dbuf, fbuf, coords, v, z)
            for i in 1:na
                out[i, j] = dbuf[i]
            end
        end
    else
        @inbounds for i in 1:na
            for j in 1:nb
                fbuf[j] = M[i, j]
            end
            _regime_dt_1d!(dbuf, fbuf, coords, v, z)
            for j in 1:nb
                out[i, j] = dbuf[j]
            end
        end
    end
    return out
end

# A resolved cell is a boundary cell when a 4-connected neighbour is either a *different* known
# regime or an *unknown* cell. Domain edges are handled separately by the edge policy (an off-grid
# neighbour never makes a boundary cell).
function _regime_boundary_masks(labels::Matrix{Int}, resolved::Matrix{Bool})
    na, nb = size(labels)
    bmask = falses(na, nb)
    bkind = zeros(Int, na, nb)
    @inbounds for j in 1:nb, i in 1:na
        resolved[i, j] || continue
        li = labels[i, j]
        touches_regime = false
        touches_unknown = false
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ii = i + di
            jj = j + dj
            (1 <= ii <= na && 1 <= jj <= nb) || continue
            if !resolved[ii, jj]
                touches_unknown = true
            elseif labels[ii, jj] != li
                touches_regime = true
            end
        end
        if touches_regime || touches_unknown
            bmask[i, j] = true
            bkind[i, j] = (touches_regime ? 1 : 0) + (touches_unknown ? 2 : 0)
        end
    end
    return bmask, bkind
end

function _regime_apply_edge_policy!(distance::AbstractMatrix{Float64}, edge_censored::AbstractMatrix{Bool},
                                    raw::AbstractMatrix{Float64}, resolved::AbstractMatrix{Bool},
                                    a_grid::Vector{Float64}, b_grid::Vector{Float64}, policy::Symbol)
    na, nb = size(raw)
    a1, a2 = a_grid[1], a_grid[end]
    b1, b2 = b_grid[1], b_grid[end]
    @inbounds for j in 1:nb, i in 1:na
        if !resolved[i, j]
            distance[i, j] = NaN
            edge_censored[i, j] = false
            continue
        end
        r = raw[i, j]
        edist = min(a_grid[i] - a1, a2 - a_grid[i], b_grid[j] - b1, b2 - b_grid[j])
        if policy === :ignore
            distance[i, j] = r
            edge_censored[i, j] = false
        elseif policy === :boundary
            distance[i, j] = min(r, edist)
            edge_censored[i, j] = false
        else # :censored
            distance[i, j] = min(r, edist)
            edge_censored[i, j] = edist < r
        end
    end
    return nothing
end

const _REGIME_BOUNDARY_CONVENTION = "Margin is the Euclidean distance from a cell centre to the " *
    "nearest boundary-cell centre (a boundary cell is a resolved cell 4-connected to a different " *
    "known regime or to an unknown cell); boundary cells have margin 0. Finite-grid convention with " *
    "<= one cell-diagonal discretization error versus the true interface. Per-axis distances are the " *
    "along-grid-line (single-parameter) drift margins (Inf when the line has no boundary cell)."

# --- layer A: deterministic regime-boundary margins ---

"""
    regime_boundary_distances(a_grid, b_grid, labels, resolved;
                              config=RegimeBoundaryConfig(), system_name="",
                              param_names=(:a, :b), status_evidence=false) -> RegimeBoundaryResult

Lower-level (analytic) entry point: compute the deterministic regime-boundary margin field directly
from physical parameter axes `a_grid` / `b_grid`, an integer `labels` matrix, and a `resolved`
known-regime mask (`(length(a_grid), length(b_grid))` shaped, `[i, j] ↔ (a_grid[i], b_grid[j])`).

Cells with `resolved[i, j] == false` are treated as unknown: they receive no margin (marked invalid)
and form a boundary for their known neighbours. Grids must be strictly increasing (nonuniform is
supported); the distance transform uses the true coordinates, not indices.
"""
function regime_boundary_distances(a_grid::AbstractVector{<:Real},
                                   b_grid::AbstractVector{<:Real},
                                   labels::AbstractMatrix{<:Integer},
                                   resolved::AbstractMatrix{Bool};
                                   config::RegimeBoundaryConfig=RegimeBoundaryConfig(),
                                   system_name::AbstractString="",
                                   param_names::Tuple{Symbol, Symbol}=(:a, :b),
                                   status_evidence::Bool=false)
    a = collect(Float64, a_grid)
    b = collect(Float64, b_grid)
    labs = Matrix{Int}(labels)
    res = Matrix{Bool}(resolved)
    _regime_validate_grids(a, b, labs, res)
    na, nb = size(labs)

    bmask, bkind = _regime_boundary_masks(labs, res)

    f0 = Matrix{Float64}(undef, na, nb)
    @inbounds for idx in eachindex(f0)
        f0[idx] = bmask[idx] ? 0.0 : Inf
    end

    col2 = _regime_dt_axis(f0, a, 1)      # squared distance to nearest boundary within each a-line
    full2 = _regime_dt_axis(col2, b, 2)   # full 2D squared distance
    row2 = _regime_dt_axis(f0, b, 2)      # squared distance to nearest boundary within each b-line

    raw = sqrt.(full2)
    distance_a = sqrt.(col2)
    distance_b = sqrt.(row2)

    distance = Matrix{Float64}(undef, na, nb)
    edge_censored = falses(na, nb)
    _regime_apply_edge_policy!(distance, edge_censored, raw, res, a, b, config.edge_policy)

    @inbounds for idx in eachindex(res)
        if !res[idx]
            distance_a[idx] = NaN
            distance_b[idx] = NaN
        end
    end

    return RegimeBoundaryResult(
        String(system_name),
        param_names,
        a,
        b,
        labs,
        res,
        copy(res),
        bmask,
        bkind,
        distance,
        distance_a,
        distance_b,
        edge_censored,
        config.edge_policy,
        status_evidence,
        _REGIME_BOUNDARY_CONVENTION,
        now(),
    )
end

"""
    regime_boundary_distances(map_result::BifurcationMapResult; cells=nothing, status_codes=nothing,
                              config=RegimeBoundaryConfig()) -> RegimeBoundaryResult

Compute the deterministic regime-boundary margin field over a classified 2D operating map.

The `BifurcationMapResult` supplies the physical parameter axes and the per-cell periodicity. Regime
classification:

- **With status evidence** (`cells::MapCellGrid` or a `status_codes` matrix, exactly one, validated
  for shape/provenance): `:periodic` cells are the periodic regimes; `:aperiodic_or_high_period` and
  `:diverged` cells are distinct physical regimes when `config.aperiodic_is_regime` /
  `config.diverged_is_regime` (default `true`); every other status
  (`:insufficient_crossings`/`:integration_failed`/`:invalid_state`/`:unknown`) is *unknown*.
- **Without status evidence** (periodicity-only, reduced semantics): period `> 0` is a known regime;
  period `0` is unknown — aperiodic, diverged and unresolved cells cannot be distinguished and are
  never silently promoted to a physical regime.

See the lower-level method for the margin definition and the `edge_policy` behaviour.
"""
function regime_boundary_distances(map_result::BifurcationMapResult;
                                   cells::Union{Nothing, MapCellGrid}=nothing,
                                   status_codes::Union{Nothing, AbstractMatrix{<:Integer}}=nothing,
                                   config::RegimeBoundaryConfig=RegimeBoundaryConfig())
    cls = _regime_map_classification(map_result; cells=cells, status_codes=status_codes,
        aperiodic_is_regime=config.aperiodic_is_regime, diverged_is_regime=config.diverged_is_regime)
    return regime_boundary_distances(cls.a_grid, cls.b_grid, cls.labels, cls.resolved;
        config=config, system_name=cls.system_name, param_names=cls.param_names,
        status_evidence=cls.status_evidence)
end

"""
    regime_boundary_summary(result::RegimeBoundaryResult) -> NamedTuple

Aggregate counts and margin statistics over the valid (known-regime) cells: `n_cells`, `n_valid`,
`n_unknown`, `n_boundary`, `n_edge_censored`, and the `min`/`median` finite margin (`NaN` when no
valid cell has a finite margin).
"""
function regime_boundary_summary(result::RegimeBoundaryResult)
    finite_margins = Float64[result.distance[idx] for idx in eachindex(result.distance)
                             if result.valid[idx] && isfinite(result.distance[idx])]
    return (
        n_cells = length(result.distance),
        n_valid = count(result.valid),
        n_unknown = count(!, result.resolved),
        n_boundary = count(result.boundary_mask),
        n_edge_censored = count(result.edge_censored),
        min_margin = isempty(finite_margins) ? NaN : minimum(finite_margins),
        median_margin = isempty(finite_margins) ? NaN : median(finite_margins),
    )
end

# --- layer B: probabilistic component-tolerance propagation ---

_tolerance_scale(t::UniformTolerance) = t.half_width
_tolerance_scale(t::GaussianTolerance) = t.std
_tolerance_is_zero(t::AbstractTolerance) = _tolerance_scale(t) == 0.0
_tolerance_draw(rng, t::UniformTolerance) = (2.0 * rand(rng) - 1.0) * t.half_width
_tolerance_draw(rng, t::GaussianTolerance) = t.std * randn(rng)

# Stable UInt64 finalizer (splitmix64) — mixes an integer stream deterministically.
function _splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# Deterministic per-cell seed from the global seed and the cell indices, independent of thread count
# or scheduling. UInt64 arithmetic wraps (mod 2^64), which is intentional and reproducible.
function _tolerance_cell_seed(global_seed::UInt64, i::Integer, j::Integer)
    z = _splitmix64(global_seed ⊻ (0x9e3779b97f4a7c15 * UInt64(i)))
    z = _splitmix64(z ⊻ (0xd1b54a32d192ed03 * UInt64(j)))
    return z
end

@inline function _nearest_grid_index(coords::Vector{Float64}, x::Float64)
    n = length(coords)
    x <= coords[1] && return 1
    x >= coords[n] && return n
    hi = searchsortedfirst(coords, x)
    lo = hi - 1
    return (x - coords[lo]) <= (coords[hi] - x) ? lo : hi
end

# 95% Wilson score interval for a binomial proportion (clamped to [0, 1]).
function _wilson_interval(p::Float64, n::Int)
    n <= 0 && return (p, p)
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2n)) / denom
    half = (z / denom) * sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    return (clamp(center - half, 0.0, 1.0), clamp(center + half, 0.0, 1.0))
end

const _TOLERANCE_CONVENTION = "Probabilities are Monte-Carlo estimates of component-tolerance " *
    "propagation through a fixed classified-map surrogate: each perturbed operating point is " *
    "classified by nearest physical-grid-cell lookup (integer regime labels are never interpolated). " *
    "This is not model reruns and not a closed-form tolerance proof. Unknown and out-of-domain mass " *
    "are retained (never renormalized). Entropy is in bits."

# Fill every per-cell output for nominal cell (i, j). Categories are laid out as
# [regime_labels..., unknown, out_of_domain]; the RNG stream draws the a-offset before the b-offset
# and skips the zero (Dirac) axis, so a per-cell seed reproduces bitwise regardless of threading.
function _tolerance_fill_cell!(result_arrays, i::Int, j::Int, a_grid::Vector{Float64},
                               b_grid::Vector{Float64}, labels::Matrix{Int}, resolved::Matrix{Bool},
                               regime_labels::Vector{Int}, label_index::Dict{Int, Int},
                               tol_a::AbstractTolerance, tol_b::AbstractTolerance,
                               za::Bool, zb::Bool, both_zero::Bool, n_samples::Int, seed::UInt64,
                               counts::Vector{Int})
    (regime_prob, nominal_regime, nominal_probability, dominant_regime, dominant_probability,
     unknown_probability, ood_probability, entropy, se, ci_lo, ci_hi) = result_arrays
    nregime = length(regime_labels)
    nom_resolved = resolved[i, j]
    nom_label = nom_resolved ? labels[i, j] : _REGIME_UNKNOWN
    nominal_regime[i, j] = nom_resolved ? labels[i, j] : _REGIME_UNKNOWN

    if both_zero
        for l in regime_labels
            regime_prob[l][i, j] = 0.0
        end
        if nom_resolved
            regime_prob[nom_label][i, j] = 1.0
            dominant_regime[i, j] = nom_label
            dominant_probability[i, j] = 1.0
            unknown_probability[i, j] = 0.0
        else
            dominant_regime[i, j] = _REGIME_UNKNOWN
            dominant_probability[i, j] = 0.0
            unknown_probability[i, j] = 1.0
        end
        nominal_probability[i, j] = 1.0
        ood_probability[i, j] = 0.0
        entropy[i, j] = 0.0
        se[i, j] = 0.0
        ci_lo[i, j] = 1.0
        ci_hi[i, j] = 1.0
        return nothing
    end

    fill!(counts, 0)
    rng = Xoshiro(_tolerance_cell_seed(seed, i, j))
    a0 = a_grid[i]
    b0 = b_grid[j]
    a1 = a_grid[1]
    a2 = a_grid[end]
    b1 = b_grid[1]
    b2 = b_grid[end]
    @inbounds for _ in 1:n_samples
        da = za ? 0.0 : _tolerance_draw(rng, tol_a)
        db = zb ? 0.0 : _tolerance_draw(rng, tol_b)
        pa = a0 + da
        pb = b0 + db
        if pa < a1 || pa > a2 || pb < b1 || pb > b2
            counts[nregime + 2] += 1
        else
            ia = _nearest_grid_index(a_grid, pa)
            jb = _nearest_grid_index(b_grid, pb)
            if resolved[ia, jb]
                counts[label_index[labels[ia, jb]]] += 1
            else
                counts[nregime + 1] += 1
            end
        end
    end

    total = n_samples
    inv_total = 1.0 / total
    best_count = 0
    best_label = _REGIME_UNKNOWN
    H = 0.0
    @inbounds for k in 1:nregime
        c = counts[k]
        p = c * inv_total
        regime_prob[regime_labels[k]][i, j] = p
        if c > best_count
            best_count = c
            best_label = regime_labels[k]
        end
        p > 0.0 && (H -= p * log2(p))
    end
    unknown_p = counts[nregime + 1] * inv_total
    ood_p = counts[nregime + 2] * inv_total
    unknown_p > 0.0 && (H -= unknown_p * log2(unknown_p))
    ood_p > 0.0 && (H -= ood_p * log2(ood_p))

    nom_p = nom_resolved ? (counts[label_index[nom_label]] * inv_total) : unknown_p

    unknown_probability[i, j] = unknown_p
    ood_probability[i, j] = ood_p
    nominal_probability[i, j] = nom_p
    dominant_regime[i, j] = best_count > 0 ? best_label : _REGIME_UNKNOWN
    dominant_probability[i, j] = best_count * inv_total
    entropy[i, j] = H
    se[i, j] = sqrt(max(nom_p * (1.0 - nom_p) * inv_total, 0.0))
    lo, hi = _wilson_interval(nom_p, total)
    ci_lo[i, j] = lo
    ci_hi[i, j] = hi
    return nothing
end

"""
    tolerance_regime_map(a_grid, b_grid, labels, resolved, config::ToleranceConfig;
                         system_name="", param_names=(:a, :b), status_evidence=false)
        -> ToleranceMapResult

Lower-level (analytic) entry point: propagate the component tolerances in `config` through the
classified grid `(labels, resolved)` on physical axes `a_grid` / `b_grid`. See the
`BifurcationMapResult` method and `ToleranceConfig` for the classification and sampling semantics.
"""
function tolerance_regime_map(a_grid::AbstractVector{<:Real},
                              b_grid::AbstractVector{<:Real},
                              labels::AbstractMatrix{<:Integer},
                              resolved::AbstractMatrix{Bool},
                              config::ToleranceConfig;
                              system_name::AbstractString="",
                              param_names::Tuple{Symbol, Symbol}=(:a, :b),
                              status_evidence::Bool=false)
    a = collect(Float64, a_grid)
    b = collect(Float64, b_grid)
    labs = Matrix{Int}(labels)
    res = Matrix{Bool}(resolved)
    _regime_validate_grids(a, b, labs, res)
    na, nb = size(labs)

    regime_labels = Int[]
    seen = Set{Int}()
    @inbounds for j in 1:nb, i in 1:na
        if res[i, j] && !(labs[i, j] in seen)
            push!(seen, labs[i, j])
            push!(regime_labels, labs[i, j])
        end
    end
    sort!(regime_labels)
    label_index = Dict{Int, Int}(l => k for (k, l) in enumerate(regime_labels))

    regime_prob = Dict{Int, Matrix{Float64}}(l => zeros(Float64, na, nb) for l in regime_labels)
    nominal_regime = zeros(Int, na, nb)
    nominal_probability = zeros(Float64, na, nb)
    dominant_regime = zeros(Int, na, nb)
    dominant_probability = zeros(Float64, na, nb)
    unknown_probability = zeros(Float64, na, nb)
    ood_probability = zeros(Float64, na, nb)
    entropy = zeros(Float64, na, nb)
    se = zeros(Float64, na, nb)
    ci_lo = zeros(Float64, na, nb)
    ci_hi = zeros(Float64, na, nb)
    arrays = (regime_prob, nominal_regime, nominal_probability, dominant_regime, dominant_probability,
              unknown_probability, ood_probability, entropy, se, ci_lo, ci_hi)

    za = _tolerance_is_zero(config.tolerance_a)
    zb = _tolerance_is_zero(config.tolerance_b)
    both_zero = za && zb
    n_eff = both_zero ? 0 : config.n_samples
    counts_workspaces = [zeros(Int, length(regime_labels) + 2) for _ in 1:Threads.maxthreadid()]

    if config.threaded
        Threads.@threads for idx in 1:(na * nb)
            i = ((idx - 1) % na) + 1
            j = ((idx - 1) ÷ na) + 1
            _tolerance_fill_cell!(arrays, i, j, a, b, labs, res, regime_labels, label_index,
                config.tolerance_a, config.tolerance_b, za, zb, both_zero, config.n_samples,
                config.seed, counts_workspaces[Threads.threadid()])
        end
    else
        for j in 1:nb, i in 1:na
            _tolerance_fill_cell!(arrays, i, j, a, b, labs, res, regime_labels, label_index,
                config.tolerance_a, config.tolerance_b, za, zb, both_zero, config.n_samples,
                config.seed, counts_workspaces[1])
        end
    end

    return ToleranceMapResult(
        String(system_name),
        param_names,
        a,
        b,
        regime_labels,
        regime_prob,
        nominal_regime,
        copy(res),
        nominal_probability,
        dominant_regime,
        dominant_probability,
        unknown_probability,
        ood_probability,
        entropy,
        se,
        ci_lo,
        ci_hi,
        config.n_samples,
        n_eff,
        config.tolerance_a,
        config.tolerance_b,
        config.seed,
        status_evidence,
        _TOLERANCE_CONVENTION,
        now(),
    )
end

"""
    tolerance_regime_map(map_result::BifurcationMapResult, config::ToleranceConfig;
                         cells=nothing, status_codes=nothing) -> ToleranceMapResult

Propagate stated component tolerances through a classified 2D operating map.

At each nominal grid cell the two parameters are independently perturbed by `config.tolerance_a` /
`config.tolerance_b` (`n_samples` Monte-Carlo draws) and each perturbed operating point is classified
by nearest physical-grid-cell lookup over the map's regime grid — integer regime labels are never
interpolated. The result tracks, per cell, the probability of each regime, the nominal-regime
probability with its binomial standard error and Wilson 95% interval, the dominant regime, the
categorical entropy, and the unknown / out-of-domain mass (retained, never renormalized).

Classification (and the optional `cells` / `status_codes` status evidence) is exactly as in
`regime_boundary_distances`. When both tolerances are zero the analysis returns the deterministic
exact classification with no sampling error (`n_effective = 0`).
"""
function tolerance_regime_map(map_result::BifurcationMapResult,
                              config::ToleranceConfig;
                              cells::Union{Nothing, MapCellGrid}=nothing,
                              status_codes::Union{Nothing, AbstractMatrix{<:Integer}}=nothing)
    cls = _regime_map_classification(map_result; cells=cells, status_codes=status_codes,
        aperiodic_is_regime=config.aperiodic_is_regime, diverged_is_regime=config.diverged_is_regime)
    return tolerance_regime_map(cls.a_grid, cls.b_grid, cls.labels, cls.resolved, config;
        system_name=cls.system_name, param_names=cls.param_names,
        status_evidence=cls.status_evidence)
end

"""
    tolerance_regime_summary(result::ToleranceMapResult) -> NamedTuple

Aggregate statistics over the nominal cells: `n_cells`, `n_regimes`, `n_effective`, and the
mean `nominal_probability`, mean `entropy`, mean `out_of_domain_probability`, and mean
`unknown_probability` across the grid.
"""
function tolerance_regime_summary(result::ToleranceMapResult)
    n = length(result.nominal_probability)
    denom = n == 0 ? 1 : n
    return (
        n_cells = n,
        n_regimes = length(result.regime_labels),
        n_effective = result.n_effective,
        mean_nominal_probability = sum(result.nominal_probability) / denom,
        mean_entropy = sum(result.entropy) / denom,
        mean_out_of_domain_probability = sum(result.out_of_domain_probability) / denom,
        mean_unknown_probability = sum(result.unknown_probability) / denom,
    )
end
