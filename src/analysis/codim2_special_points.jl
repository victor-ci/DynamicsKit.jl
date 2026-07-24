"""
Test-function pass over a `Codim2ContinuationResult` produced by `engine=:defining_system`
to detect and locate organising codim-2 bifurcation points along the continued locus.

Supported detection kinds and the loci they apply to:
- `:cusp`            — fold normal-form coefficient b = 1/2<p,B(q,q)> crossing/vanishing (`:fold` locus)
- `:generalized_flip`— flip normal-form coefficient c crossing zero (`:pd` locus)
- `:fold_flip`       — non-tracked second multiplier reaching the complementary ±1 value
                       (`:pd` or `:fold` locus; requires `curve_diagnostics=true`)
- `:resonance_1_1`   — tracked multiplier angle crossing 0 mod 2π (`:ns` locus)
- `:resonance_1_2`   — tracked multiplier angle crossing π mod 2π (`:ns` locus)
- `:bautin`          — NS first Lyapunov coefficient d crossing zero (`:ns` locus)

Full codim-2 normal-form classification is explicitly out of scope.
"""

const _CODIM2_SPECIAL_POINT_KINDS =
    (:cusp, :generalized_flip, :fold_flip, :resonance_1_1, :resonance_1_2, :bautin)

# ---- test-function helpers ------------------------------------------------

# On a PD locus (tracked μ ≈ −1) the complementary target is +1.
# On a fold locus (tracked μ ≈ +1) the complementary target is −1.
# These reuse the same determinant formulas as map_special_points.jl.
_c2sp_fold_det(mult) = real(prod(z -> z - 1, mult))  # zero when a multiplier crosses +1
_c2sp_pd_det(mult)   = real(prod(z -> z + 1, mult))  # zero when a multiplier crosses −1

function _c2sp_fold_flip_phi(locus_kind::Symbol, mult::AbstractVector{<:Number})
    isempty(mult) && return NaN
    locus_kind === :pd   && return _c2sp_fold_det(mult)
    locus_kind === :fold && return _c2sp_pd_det(mult)
    return NaN
end

# ---- data-availability guards --------------------------------------------

function _c2sp_have_multipliers(result::Codim2ContinuationResult)
    n = length(result.primary_values)
    length(result.multipliers) == n || return false
    n > 0 || return false
    multiplier_count = length(first(result.multipliers))
    multiplier_count > 0 || return false
    all(m -> length(m) == multiplier_count &&
             all(v -> isfinite(real(v)) && isfinite(imag(v)), m), result.multipliers)
end

function _c2sp_have_phase_angles(result::Codim2ContinuationResult)
    result.bifurcation_kind === :ns || return false
    n = length(result.primary_values)
    length(result.phase_angles) == n || return false
    any(isfinite, result.phase_angles)
end

# ---- parameter reconstruction --------------------------------------------

function _c2sp_sample_params(result::Codim2ContinuationResult, j::Int,
                              base::Vector{Float64},
                              pidx::Int, linked1::Vector{Int},
                              sidx::Int, linked2::Vector{Int})
    pv = inject_param(base, pidx, result.primary_values[j], linked1)
    return inject_param(pv, sidx, result.secondary_values[j], linked2)
end

# ---- interpolation helper ------------------------------------------------

# Linearly interpolate a Codim2SpecialPoint between locus samples lo and lo+1.
# alpha=0 → sample lo, alpha=1 → sample lo+1.
function _c2sp_interp(kind::Symbol,
                      result::Codim2ContinuationResult,
                      lo::Int,
                      alpha::Float64,
                      test_val::Float64,
                      have_mult::Bool,
                      nf::Union{Nothing, MapNormalForm}=nothing)
    alpha = clamp(alpha, 0.0, 1.0)
    hi = lo + 1
    p1 = (1 - alpha) * result.primary_values[lo]   + alpha * result.primary_values[hi]
    p2 = (1 - alpha) * result.secondary_values[lo] + alpha * result.secondary_values[hi]
    state = (1 - alpha) .* result.states[:, lo] .+ alpha .* result.states[:, hi]
    mult = if have_mult
        ComplexF64[(1 - alpha) * result.multipliers[lo][k] + alpha * result.multipliers[hi][k]
                   for k in eachindex(result.multipliers[lo])]
    else
        ComplexF64[]
    end
    return Codim2SpecialPoint(kind, result.bifurcation_kind, p1, p2,
                              collect(Float64, state), mult, test_val,
                              result.period, false, :interpolated, nf)
end

function _c2sp_sample(kind::Symbol,
                      result::Codim2ContinuationResult,
                      index::Int,
                      test_val::Float64,
                      have_mult::Bool,
                      nf::Union{Nothing, MapNormalForm}=nothing)
    mult = have_mult ? copy(result.multipliers[index]) : ComplexF64[]
    return Codim2SpecialPoint(
        kind,
        result.bifurcation_kind,
        result.primary_values[index],
        result.secondary_values[index],
        collect(Float64, result.states[:, index]),
        mult,
        test_val,
        result.period,
        false,
        :sampled,
        nf,
    )
end

# ---- per-kind detectors --------------------------------------------------

# Cusp: the fold normal-form quadratic coefficient b = 1/2<p,B(q,q)> vanishes.
#
# A cusp is NOT a turning point of the fold locus in a chosen parameter — that is
# coordinate-dependent and routinely occurs with b != 0 (e.g. the fold parabola of
# x -> x+p1+p2 x+x^2 has b=1 everywhere yet its projection turns at p2=0).  A cusp is
# the intrinsic fold degeneracy b -> 0, so it is detected by evaluating
# map_normal_form(:fold) at each locus sample and locating where b crosses zero.
#
# Orientation invariant: b's sign is orientation-dependent (flipping the real critical
# eigenvector q -> -q, with p rescaled to keep <p,q>=1, flips the sign of b), while |b|
# is orientation-invariant.  `_oriented_eigenvectors` fixes a deterministic orientation
# (the dominant eigenvector component made real and positive); that sign can only jump
# where the dominant component switches index, never at a genuine cusp where |b|
# collapses smoothly to zero.  Hence:
#   * the sampled near-zero |b| detector is orientation-robust and carries the sample's
#     actual (degenerate) MapNormalForm; and
#   * an interpolated sign change is accepted only when |b| is locally minimised at the
#     bracket (a strict |b| valley with finite samples on both outer sides).  This
#     conservative continuity guard rejects flat-magnitude orientation flips and skips
#     edge/unavailable brackets that cannot establish coefficient collapse.  Interpolated
#     points carry normal_form=nothing (the nonzero bracketing form is not the b=0 form).
function _c2sp_detect_cusp(sys::DynamicalSystem,
                            result::Codim2ContinuationResult,
                            base::Vector{Float64},
                            pidx::Int, linked1::Vector{Int},
                            sidx::Int, linked2::Vector{Int};
                            test_tolerance::Float64,
                            normal_form_fd_step::Float64,
                            solver, reltol, abstol, tmax, min_crossing_time)
    result.bifurcation_kind === :fold || return Codim2SpecialPoint[]
    n = length(result.primary_values)
    n >= 2 || return Codim2SpecialPoint[]

    have_mult = _c2sp_have_multipliers(result)
    bs  = Vector{Union{Float64, Nothing}}(nothing, n)
    nfs = Vector{Union{Nothing, MapNormalForm}}(nothing, n)
    for j in 1:n
        params_j = _c2sp_sample_params(result, j, base, pidx, linked1, sidx, linked2)
        state_j  = collect(Float64, result.states[:, j])
        all(isfinite, state_j) && all(isfinite, params_j) || continue
        nf = try
            map_normal_form(sys, :fold, state_j, params_j;
                            period=result.period,
                            normal_form_fd_step=normal_form_fd_step,
                            solver=solver, reltol=reltol, abstol=abstol,
                            tmax=tmax, min_crossing_time=min_crossing_time)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        nfs[j] = nf
        # Accept :ok and :degenerate: a sample sitting on the cusp reports b≈0 with
        # status :degenerate, and that finite coefficient must still anchor a point.
        bs[j] = (nf !== nothing && nf.coefficient !== nothing &&
                  nf.status in (:ok, :degenerate)) ? nf.coefficient : nothing
    end

    points = Codim2SpecialPoint[]
    for j in 1:n
        bs[j] !== nothing || continue
        abs(bs[j]::Float64) <= test_tolerance || continue
        push!(points, _c2sp_sample(:cusp, result, j, bs[j]::Float64, have_mult, nfs[j]))
    end
    for j in 1:(n - 1)
        (bs[j] !== nothing && bs[j + 1] !== nothing) || continue
        b_lo, b_hi = bs[j]::Float64, bs[j + 1]::Float64
        (abs(b_lo) > test_tolerance && abs(b_hi) > test_tolerance) || continue
        b_lo * b_hi < 0 || continue
        _c2sp_cusp_valley(bs, j) || continue
        alpha = b_lo / (b_lo - b_hi)
        # Leave normal_form=nothing: the interpolated point sits at the b=0 location,
        # so a bracketing sample's nonzero-coefficient form would be misleading.
        push!(points, _c2sp_interp(:cusp, result, j, alpha, 0.0, have_mult, nothing))
    end
    return points
end

# |b| valley test for the interpolated cusp detector: a sign change between samples j
# and j+1 is accepted only when finite outer neighbours establish a strict decrease
# in |b| into both sides of the bracket. Missing/boundary neighbours cannot distinguish
# a coefficient collapse from an orientation jump, so those brackets are skipped.
function _c2sp_cusp_valley(bs::Vector{Union{Float64, Nothing}}, j::Int)
    (j > 1 && j + 2 <= length(bs)) || return false
    (bs[j - 1] !== nothing && bs[j + 2] !== nothing) || return false
    return abs(bs[j]::Float64) < abs(bs[j - 1]::Float64) &&
           abs(bs[j + 1]::Float64) < abs(bs[j + 2]::Float64)
end

# Fold-flip: sign change of the complementary multiplier determinant.
function _c2sp_detect_fold_flip(result::Codim2ContinuationResult, test_tolerance::Float64)
    result.bifurcation_kind in (:pd, :fold) || return Codim2SpecialPoint[]
    _c2sp_have_multipliers(result) || return Codim2SpecialPoint[]
    n = length(result.primary_values)
    n >= 2 || return Codim2SpecialPoint[]

    # Require at least 2 multipliers (1D maps can have no complementary multiplier).
    all(m -> length(m) >= 2, result.multipliers) || return Codim2SpecialPoint[]

    lk = result.bifurcation_kind
    phis = [_c2sp_fold_flip_phi(lk, result.multipliers[j]) for j in 1:n]
    points = Codim2SpecialPoint[]
    for j in 1:n
        isfinite(phis[j]) || continue
        abs(phis[j]) <= test_tolerance || continue
        push!(points, _c2sp_sample(:fold_flip, result, j, phis[j], true))
    end
    for j in 1:(n - 1)
        (isfinite(phis[j]) && isfinite(phis[j + 1])) || continue
        (abs(phis[j]) > test_tolerance && abs(phis[j + 1]) > test_tolerance) || continue
        phis[j] * phis[j + 1] < 0 || continue
        alpha = phis[j] / (phis[j] - phis[j + 1])
        push!(points, _c2sp_interp(:fold_flip, result, j, alpha, 0.0, true))
    end
    return points
end

# Resonance 1:1 and 1:2 along an NS locus.
#
# Test functions that respect unwrapped angle periodicity:
#   1:1 (θ = 2πk):      φ(θ) = sin(θ/2)  — zero exactly at θ = 2πk, nonzero at π+2πk.
#   1:2 (θ = π+2πk):    φ(θ) = cos(θ/2)  — zero exactly at θ = π+2πk, nonzero at 2πk.
#
# Using raw θ or θ−π as test functions would give the same zeros at 0 and π but would
# also detect spurious crossings at ±2π, ±3π, etc. when the continuation produces
# unwrapped angles outside (−π, π].  The root-of-unity forms are exact for any winding.
function _c2sp_detect_resonances(result::Codim2ContinuationResult,
                                  want_11::Bool, want_12::Bool,
                                  test_tolerance::Float64)
    (want_11 || want_12) || return Codim2SpecialPoint[]
    result.bifurcation_kind === :ns || return Codim2SpecialPoint[]
    _c2sp_have_phase_angles(result) || return Codim2SpecialPoint[]

    n = length(result.primary_values)
    n >= 2 || return Codim2SpecialPoint[]

    have_mult = _c2sp_have_multipliers(result)
    theta = result.phase_angles
    points = Codim2SpecialPoint[]

    # 1:1 resonance: θ = 2πk.  Test function: sin(θ/2).
    if want_11
        phi = sin.(theta ./ 2)
        for j in 1:n
            isfinite(phi[j]) || continue
            abs(phi[j]) <= test_tolerance || continue
            push!(points, _c2sp_sample(:resonance_1_1, result, j, phi[j], have_mult))
        end
        for j in 1:(n - 1)
            (isfinite(theta[j]) && isfinite(theta[j + 1])) || continue
            (abs(phi[j]) > test_tolerance && abs(phi[j + 1]) > test_tolerance) || continue
            phi[j] * phi[j + 1] < 0 || continue
            alpha = phi[j] / (phi[j] - phi[j + 1])
            push!(points, _c2sp_interp(:resonance_1_1, result, j, alpha,
                                       0.0, have_mult))
        end
    end

    # 1:2 resonance: θ = π+2πk.  Test function: cos(θ/2).
    if want_12
        phi = cos.(theta ./ 2)
        for j in 1:n
            isfinite(phi[j]) || continue
            abs(phi[j]) <= test_tolerance || continue
            push!(points, _c2sp_sample(:resonance_1_2, result, j, phi[j], have_mult))
        end
        for j in 1:(n - 1)
            (isfinite(theta[j]) && isfinite(theta[j + 1])) || continue
            (abs(phi[j]) > test_tolerance && abs(phi[j + 1]) > test_tolerance) || continue
            phi[j] * phi[j + 1] < 0 || continue
            alpha = phi[j] / (phi[j] - phi[j + 1])
            push!(points, _c2sp_interp(:resonance_1_2, result, j, alpha,
                                       0.0, have_mult))
        end
    end
    return points
end

# Generalised flip: sign change of the flip normal-form coefficient c along the PD locus.
#
# Normal-form coefficient c is evaluated at each locus sample; only strict sign
# changes between adjacent samples produce a point.  The interpolated point carries
# `normal_form=nothing` because it represents the coefficient-zero location — attaching
# a bracketing sample's nonzero-coefficient form would be scientifically misleading.
function _c2sp_detect_generalized_flip(sys::DynamicalSystem,
                                        result::Codim2ContinuationResult,
                                        base::Vector{Float64},
                                        pidx::Int, linked1::Vector{Int},
                                        sidx::Int, linked2::Vector{Int};
                                        test_tolerance::Float64,
                                        normal_form_fd_step::Float64,
                                        solver, reltol, abstol, tmax, min_crossing_time)
    result.bifurcation_kind === :pd || return Codim2SpecialPoint[]
    n = length(result.primary_values)
    n >= 2 || return Codim2SpecialPoint[]

    have_mult = _c2sp_have_multipliers(result)
    cs  = Vector{Union{Float64, Nothing}}(nothing, n)
    nfs = Vector{Union{Nothing, MapNormalForm}}(nothing, n)
    for j in 1:n
        params_j = _c2sp_sample_params(result, j, base, pidx, linked1, sidx, linked2)
        state_j  = collect(Float64, result.states[:, j])
        all(isfinite, state_j) && all(isfinite, params_j) || continue
        nf = try
            map_normal_form(sys, :pd, state_j, params_j;
                            period=result.period,
                            normal_form_fd_step=normal_form_fd_step,
                            solver=solver, reltol=reltol, abstol=abstol,
                            tmax=tmax, min_crossing_time=min_crossing_time)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        nfs[j] = nf
        cs[j] = (nf !== nothing && nf.status === :ok &&
                  nf.coefficient !== nothing) ? nf.coefficient : nothing
    end

    points = Codim2SpecialPoint[]
    for j in 1:n
        cs[j] !== nothing || continue
        abs(cs[j]::Float64) <= test_tolerance || continue
        push!(points, _c2sp_sample(
            :generalized_flip, result, j, cs[j]::Float64, have_mult, nfs[j]))
    end
    for j in 1:(n - 1)
        (cs[j] !== nothing && cs[j + 1] !== nothing) || continue
        c_lo, c_hi = cs[j]::Float64, cs[j + 1]::Float64
        (abs(c_lo) > test_tolerance && abs(c_hi) > test_tolerance) || continue
        c_lo * c_hi < 0 || continue
        alpha = c_lo / (c_lo - c_hi)
        # Leave normal_form=nothing: the interpolated point is at the coefficient-zero
        # location; the nearest bracketing sample's nonzero form is not appropriate here.
        push!(points, _c2sp_interp(:generalized_flip, result, j, alpha, 0.0,
                                   have_mult, nothing))
    end
    return points
end

# Bautin: sign change of the NS normal-form coefficient d along the NS locus.
#
# Same conservative policy as generalized flip: interpolated points carry
# `normal_form=nothing` to avoid attaching a nonzero-coefficient form to the
# coefficient-zero location.
function _c2sp_detect_bautin(sys::DynamicalSystem,
                              result::Codim2ContinuationResult,
                              base::Vector{Float64},
                              pidx::Int, linked1::Vector{Int},
                              sidx::Int, linked2::Vector{Int};
                              test_tolerance::Float64,
                              normal_form_fd_step::Float64,
                              solver, reltol, abstol, tmax, min_crossing_time)
    result.bifurcation_kind === :ns || return Codim2SpecialPoint[]
    n = length(result.primary_values)
    n >= 2 || return Codim2SpecialPoint[]

    have_mult = _c2sp_have_multipliers(result)
    ds  = Vector{Union{Float64, Nothing}}(nothing, n)
    nfs = Vector{Union{Nothing, MapNormalForm}}(nothing, n)
    for j in 1:n
        params_j = _c2sp_sample_params(result, j, base, pidx, linked1, sidx, linked2)
        state_j  = collect(Float64, result.states[:, j])
        all(isfinite, state_j) && all(isfinite, params_j) || continue
        nf = try
            map_normal_form(sys, :ns, state_j, params_j;
                            period=result.period,
                            normal_form_fd_step=normal_form_fd_step,
                            solver=solver, reltol=reltol, abstol=abstol,
                            tmax=tmax, min_crossing_time=min_crossing_time)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        nfs[j] = nf
        ds[j] = (nf !== nothing && nf.status === :ok &&
                  nf.coefficient !== nothing) ? nf.coefficient : nothing
    end

    points = Codim2SpecialPoint[]
    for j in 1:n
        ds[j] !== nothing || continue
        abs(ds[j]::Float64) <= test_tolerance || continue
        push!(points, _c2sp_sample(:bautin, result, j, ds[j]::Float64, have_mult, nfs[j]))
    end
    for j in 1:(n - 1)
        (ds[j] !== nothing && ds[j + 1] !== nothing) || continue
        d_lo, d_hi = ds[j]::Float64, ds[j + 1]::Float64
        (abs(d_lo) > test_tolerance && abs(d_hi) > test_tolerance) || continue
        d_lo * d_hi < 0 || continue
        alpha = d_lo / (d_lo - d_hi)
        push!(points, _c2sp_interp(:bautin, result, j, alpha, 0.0, have_mult, nothing))
    end
    return points
end

# ---- deduplication -------------------------------------------------------

function _c2sp_deduplicate(points::Vector{Codim2SpecialPoint},
                            dup_p1::Float64, dup_p2::Float64)
    isempty(points) && return points
    sorted = sort(points;
                  by = pt -> (String(pt.kind), pt.secondary_param, pt.primary_param))
    out = Codim2SpecialPoint[sorted[1]]
    for i in 2:length(sorted)
        prev = out[end]
        curr = sorted[i]
        is_duplicate = curr.kind === prev.kind &&
            abs(curr.primary_param   - prev.primary_param)   <= dup_p1 &&
            abs(curr.secondary_param - prev.secondary_param) <= dup_p2
        if !is_duplicate
            push!(out, curr)
            continue
        end
        if curr.status === :sampled &&
           (prev.status !== :sampled || abs(curr.test_value) < abs(prev.test_value))
            out[end] = curr
        end
    end
    return out
end

# ---- public API ----------------------------------------------------------

"""
    codim2_special_points(sys, result::Codim2ContinuationResult; kwargs...)
        -> Vector{Codim2SpecialPoint}

Test-function pass over a codimension-2 locus from `codim2_curve` with
`engine=:defining_system` to detect and locate the organising codim-2 points.

# Keyword arguments
- `detect`: tuple/vector of kinds to look for, or `nothing` for all six.
  Valid: `:cusp`, `:generalized_flip`, `:fold_flip`, `:resonance_1_1`, `:resonance_1_2`, `:bautin`.
  When a specific applicable kind is requested explicitly and `base_params` / parameter
  indices are missing, an `ArgumentError` is thrown (see below).
- `base_params`: full base parameter vector. Required for `:cusp`, `:generalized_flip`, and
  `:bautin` when those kinds are explicitly listed in `detect`.
- `param_index`: primary parameter index in `base_params`. Required when `base_params` is non-empty.
- `second_param_index`: secondary parameter index. Required when `base_params` is non-empty.
- `linked_param_indices`, `second_linked_param_indices`: parameter slots tied to the primary /
  secondary axes (as in `Codim2Config`).
- `duplicate_primary_tol`, `duplicate_secondary_tol`: proximity thresholds for deduplication.
- `test_tolerance`: absolute scalar-test tolerance for accepting a sampled endpoint/root.
- `normal_form_fd_step`: finite-difference step for continuous normal-form evaluation.
- `solver`, `reltol`, `abstol`, `tmax`, `min_crossing_time`: ODE controls for continuous systems.

# Result
Points are sorted by `(kind, secondary_param, primary_param)` and deduplicated. Each
`Codim2SpecialPoint` carries a `status` of `:interpolated` (sign-change bracketing), `:sampled`
(direct sample), or `:unavailable`. Full codim-2 normal-form classification is out of scope.

# Conservative interpolation policy
`:cusp`, `:generalized_flip`, and `:bautin` evaluate a normal-form coefficient at discrete
locus samples and interpolate to locate the zero.  The resulting point carries
`normal_form=nothing`: attaching the nearest bracketing sample's nonzero-coefficient form to
the coefficient-zero location would be scientifically misleading.  All interpolated points
remain `converged=false`.  For `:cusp`, the fold coefficient `b = 1/2<p,B(q,q)>` is
orientation-dependent while `|b|` is not; the sampled near-zero `|b|` detector is therefore
orientation-robust (and carries the sample's actual degenerate form), and interpolated `b`
sign changes are accepted only when `|b|` is locally minimised at the bracket, guarding against
spurious eigenvector-orientation flips. The first and last sample pairs are not interpolated
because they lack two outer neighbours; a cusp there is detected only by a sampled
`|b| <= test_tolerance` value. Extend the continuation range when a cusp is suspected near a
locus endpoint.

# ArgumentError policy
If an applicable coefficient detector is **explicitly** listed in `detect` but `base_params`,
`param_index`, and `second_param_index` are not provided, an `ArgumentError` is raised:
- `:cusp` on a `:fold` locus without `base_params` → `ArgumentError`
- `:generalized_flip` on a `:pd` locus without `base_params` → `ArgumentError`
- `:bautin` on a `:ns` locus without `base_params` → `ArgumentError`
Inapplicable kinds (wrong locus type) return an empty result without error so that the
default all-kinds pass (`detect=nothing`) works on any locus without requiring `base_params`.

# Applicability
| Kind               | Locus      | Needs `curve_diagnostics` | Needs `base_params` |
|--------------------|------------|---------------------------|---------------------|
| `:cusp`            | `:fold`    | no                        | yes (see above)     |
| `:generalized_flip`| `:pd`      | no                        | yes (see above)     |
| `:fold_flip`       | `:pd`/`:fold`| yes (multipliers)       | no                  |
| `:resonance_1_1`   | `:ns`      | no (phase angles)         | no                  |
| `:resonance_1_2`   | `:ns`      | no (phase angles)         | no                  |
| `:bautin`          | `:ns`      | no                        | yes (see above)     |
"""
function codim2_special_points(
        sys::DynamicalSystem,
        result::Codim2ContinuationResult;
        detect                       = nothing,
        base_params::AbstractVector{<:Real}   = Float64[],
        param_index::Int             = 0,
        second_param_index::Int      = 0,
        linked_param_indices::AbstractVector{<:Integer}        = Int[],
        second_linked_param_indices::AbstractVector{<:Integer} = Int[],
        duplicate_primary_tol::Float64   = 1e-7,
        duplicate_secondary_tol::Float64 = 1e-7,
        test_tolerance::Float64          = 1e-5,
        normal_form_fd_step::Float64 = 3e-3,
        solver                       = Tsit5(),
        reltol::Float64              = 1e-9,
        abstol::Float64              = 1e-9,
        tmax::Union{Nothing, Float64} = nothing,
        min_crossing_time::Float64   = 1e-6,
)
    # --- Validation -------------------------------------------------------
    result.engine === :defining_system || throw(ArgumentError(
        "codim2_special_points requires a result from engine=:defining_system; " *
        "got $(repr(result.engine))."))

    n = length(result.primary_values)
    n >= 2 || return Codim2SpecialPoint[]

    length(result.secondary_values) == n || throw(ArgumentError(
        "codim2_special_points: primary_values and secondary_values have different lengths."))
    size(result.states, 2) == n || throw(ArgumentError(
        "codim2_special_points: states matrix must have $(n) columns."))

    # Normalise detect: nothing → all six kinds (default, no explicit request).
    explicit_detect = detect !== nothing
    local detect_syms::Vector{Symbol}
    if !explicit_detect
        detect_syms = collect(Symbol, _CODIM2_SPECIAL_POINT_KINDS)
    else
        isempty(detect) && return Codim2SpecialPoint[]
        detect_syms = Symbol[Symbol(k) for k in detect]
        for kind in detect_syms
            kind in _CODIM2_SPECIAL_POINT_KINDS || throw(ArgumentError(
                "codim2_special_points: unsupported detect kind $(repr(kind)). " *
                "Valid: $(join(_CODIM2_SPECIAL_POINT_KINDS, ", "))."))
        end
    end

    duplicate_primary_tol   >= 0 || throw(ArgumentError("duplicate_primary_tol must be non-negative."))
    duplicate_secondary_tol >= 0 || throw(ArgumentError("duplicate_secondary_tol must be non-negative."))
    test_tolerance > 0 || throw(ArgumentError("test_tolerance must be positive."))
    normal_form_fd_step > 0 || throw(ArgumentError("normal_form_fd_step must be positive."))

    base    = collect(Float64, base_params)
    pidx    = param_index
    sidx    = second_param_index
    linked1 = collect(Int, linked_param_indices)
    linked2 = collect(Int, second_linked_param_indices)

    params_ready = !isempty(base) && pidx > 0 && sidx > 0

    # ArgumentError when an applicable coefficient detector is explicitly requested
    # without the required parameter information.  Default (detect=nothing) skips
    # silently so it works on any locus.
    if explicit_detect && !params_ready
        :cusp in detect_syms && result.bifurcation_kind === :fold &&
            throw(ArgumentError(
                "codim2_special_points: detect=[:cusp] was explicitly requested on a " *
                ":fold locus but base_params, param_index, and second_param_index are " *
                "required for normal-form coefficient evaluation."))
        :generalized_flip in detect_syms && result.bifurcation_kind === :pd &&
            throw(ArgumentError(
                "codim2_special_points: detect=[:generalized_flip] was explicitly requested " *
                "on a :pd locus but base_params, param_index, and second_param_index are " *
                "required for normal-form coefficient evaluation."))
        :bautin in detect_syms && result.bifurcation_kind === :ns &&
            throw(ArgumentError(
                "codim2_special_points: detect=[:bautin] was explicitly requested on a " *
                ":ns locus but base_params, param_index, and second_param_index are " *
                "required for normal-form coefficient evaluation."))
    end

    if params_ready
        primary_indices = [pidx; linked1]
        secondary_indices = [sidx; linked2]
        all(index -> 1 <= index <= length(base), primary_indices) || throw(ArgumentError(
            "codim2_special_points: primary and linked parameter indices must lie within " *
            "1:$(length(base)); got $(primary_indices)."))
        all(index -> 1 <= index <= length(base), secondary_indices) || throw(ArgumentError(
            "codim2_special_points: secondary and linked parameter indices must lie within " *
            "1:$(length(base)); got $(secondary_indices)."))
        length(unique(primary_indices)) == length(primary_indices) || throw(ArgumentError(
            "codim2_special_points: primary parameter indices must be unique; got $(primary_indices)."))
        length(unique(secondary_indices)) == length(secondary_indices) || throw(ArgumentError(
            "codim2_special_points: secondary parameter indices must be unique; got $(secondary_indices)."))
        isempty(intersect(primary_indices, secondary_indices)) || throw(ArgumentError(
            "codim2_special_points: primary and secondary parameter roles must not overlap; " *
            "got primary=$(primary_indices), secondary=$(secondary_indices)."))
    end

    nf_kw = (normal_form_fd_step=normal_form_fd_step, solver=solver,
             reltol=reltol, abstol=abstol, tmax=tmax, min_crossing_time=min_crossing_time)

    # --- Detection passes -------------------------------------------------
    all_points = Codim2SpecialPoint[]

    if :cusp in detect_syms && params_ready
        append!(all_points,
                _c2sp_detect_cusp(sys, result, base, pidx, linked1,
                                   sidx, linked2;
                                   test_tolerance=test_tolerance, nf_kw...))
    end

    :fold_flip in detect_syms &&
        append!(all_points, _c2sp_detect_fold_flip(result, test_tolerance))

    if (:resonance_1_1 in detect_syms) || (:resonance_1_2 in detect_syms)
        append!(all_points,
                _c2sp_detect_resonances(result,
                                         :resonance_1_1 in detect_syms,
                                         :resonance_1_2 in detect_syms,
                                         test_tolerance))
    end

    if :generalized_flip in detect_syms && params_ready
        append!(all_points,
                _c2sp_detect_generalized_flip(sys, result, base, pidx, linked1,
                                               sidx, linked2;
                                               test_tolerance=test_tolerance, nf_kw...))
    end

    if :bautin in detect_syms && params_ready
        append!(all_points,
                _c2sp_detect_bautin(sys, result, base, pidx, linked1,
                                     sidx, linked2;
                                     test_tolerance=test_tolerance, nf_kw...))
    end

    return _c2sp_deduplicate(all_points, duplicate_primary_tol, duplicate_secondary_tol)
end
