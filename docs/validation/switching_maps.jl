"""
Switching-map generator validation.

Checks three independent claims about the piecewise-linear switching-map
generator (`SwitchingCircuitDescription` + `AffineModeSpec` + `switching_map`,
see `docs/analysis-methods.md` and `docs/systems-catalog.md`):

  1. The exact affine-flow primitive it evaluates per mode reproduces Julia's
     own dense matrix exponential (`Base.exp`, independent of DynamicsKit) to
     within floating-point tolerance, for representative defective, nilpotent,
     and mixed rotation/Jordan state matrices at several state dimensions.
  2. The generator reproduces the pre-existing, independently hand-coded
     `buck_converter()`/`boost_converter()` maps exactly (both are shipped in
     this repository; `buck_converter_description`/`boost_converter_description`
     are the generator-based re-implementations of the same circuits).
  3. The generated 4-D current-programmed Cuk and SEPIC converters' first
     period-doubling ("flip") point lands where the literature sources place
     it (SEPIC) or where the *exact* switched map places it once the printed
     source matrices are used directly (Cuk — see the documented discrepancy
     below).

References:
  Cuk converter:   R. Debbat, A. El Aroudi, M. Bouyadjra, "Chaos and
                   Bifurcation in a Current Programmed Cuk Converter",
                   ICM 2012.
  SEPIC converter: R. Debbat, A. El Aroudi, R. Giral, L. Martinez-Salamero,
                   "Discrete Time Modeling of a Current-Mode Controlled SEPIC
                   Converter", ISCAS 2002.

The Cuk converter's *exact* stroboscopic map (built directly from the
printed source matrices, no averaging) places its first flip near
`Iref ≈ 3.0 A` rather than the source text's approximate 4.0 A (which uses an
averaged model). This is a known, informational discrepancy between the exact
switched map computed here and the source's averaged analysis — see
`docs/systems-catalog.md` — and gate 3.2 below asserts that this discrepancy
is present and of the expected size, rather than treating it as a defect.

Run:

    julia --project=. validation/switching_maps.jl
"""

using DynamicsKit
using StaticArrays
using ForwardDiff
using LinearAlgebra

include(joinpath(@__DIR__, "common.jl"))

gates = ValidationGate[]

# ── Section 1: n-dimensional affine-flow analytic fixtures ──────────────────
# Each fixture is evaluated through the fully public API: a one-mode
# `SwitchingCircuitDescription` whose only mode has `duration=nothing` (so it
# consumes the whole clock period) reduces `switching_map` to exactly the
# exact-affine-flow primitive being checked, with no private internals
# involved. The reference is Julia's own `exp` of the augmented Duhamel
# matrix, which does not depend on DynamicsKit at all.
function affine_flow_reference(x0::AbstractVector, A::AbstractMatrix, b::AbstractVector, tau)
    n = length(x0)
    M = [Matrix(A) collect(b); zeros(1, n) 0.0]
    return (exp(M * tau) * [collect(x0); 1.0])[1:n]
end

function one_mode_map(A::SMatrix{N,N}, b::SVector{N}, tau) where {N}
    desc = SwitchingCircuitDescription(
        (AffineModeSpec(A, b),), tau; name="One-mode affine-flow fixture", state_dim=N)
    return switching_map(desc)
end

affine_fixtures = (
    ("3D defective triangular",
     SMatrix{3,3}(-1.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 3.0, -3.0),
     SVector(1.0, -0.5, 0.25), SVector(0.1, 0.2, 0.3), 0.7),
    ("4D nilpotent chain",
     SMatrix{4,4}(0.0, 0.0, 0.0, 0.0,
                  1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0),
     SVector(1.0, 2.0, 3.0, 4.0), SVector(0.0, 0.1, -0.2, 0.3), 0.125),
    ("6D mixed rotation and Jordan",
     SMatrix{6,6}(-0.2, -1.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, -0.2, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, -0.5, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 1.0, -1.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 1.0, -1.0),
     SVector(1.0, 0.0, -1.0, 0.5, -0.5, 0.25), SVector(0.3, -0.1, 0.2, 0.0, 0.4, -0.2), 0.4),
)

for (label, A, b, x0, tau) in affine_fixtures
    sys = one_mode_map(A, b, tau)
    y = sys.f(x0, Float64[])
    ref = affine_flow_reference(x0, A, b, tau)
    err = maximum(abs.(collect(y) .- ref))
    record_gate!(gates, "$label affine flow matches exp() reference", err <= 1e-13,
        "max_err=$(err)")

    n = length(x0)
    J_ad = ForwardDiff.jacobian(x -> collect(sys.f(SVector{n}(x), Float64[])), collect(x0))
    J_ref = Matrix(exp(Matrix(A) * tau))
    jerr = maximum(abs.(J_ad .- J_ref))
    record_gate!(gates, "$label ForwardDiff Jacobian matches exp(A*tau)", jerr <= 1e-12,
        "max_err=$(jerr)")
end

# ── Section 2: generator vs. hand-coded buck/boost regression ──────────────
# `buck_converter()`/`boost_converter()` are independently hand-coded maps
# shipped in this repository; `buck_converter_description()`/
# `boost_converter_description()` are the generator-based re-implementations
# of the identical circuits. Agreement across representative operating points
# (including saturation-rail cases) confirms the generator reproduces a
# circuit family it did not originate.
buck_legacy = buck_converter()
buck_gen = switching_map(buck_converter_description())
buck_cases = ([2.0, 15.0], [0.5, 15.0], [3.5, 20.0])
buck_states = (SVector(2.0, 1.0), SVector(0.1, 0.05), SVector(5.0, 3.0))
buck_err = maximum(
    maximum(abs.(collect(buck_legacy.f(x, p)) .- collect(buck_gen.f(x, p))))
    for (x, p) in zip(buck_states, buck_cases)
)
record_gate!(gates, "buck generator matches hand-coded map", buck_err <= 1e-12,
    "max_err=$(buck_err) over $(length(buck_cases)) cases")

boost_legacy = boost_converter()
boost_gen = switching_map(boost_converter_description())
boost_cases = ([1.5, 10.0, 20.0, 0.0], [2.5, 10.0, 20.0, 0.0], [1.0, 12.0, 25.0, 0.02])
boost_states = (SVector(10.0, 1.0), SVector(9.0, 2.5), SVector(11.0, 0.5))
boost_err = maximum(
    maximum(abs.(collect(boost_legacy.f(x, p)) .- collect(boost_gen.f(x, p))))
    for (x, p) in zip(boost_states, boost_cases)
)
record_gate!(gates, "boost generator matches hand-coded map", boost_err <= 1e-12,
    "max_err=$(boost_err) over $(length(boost_cases)) cases")

# ── Section 3: Cuk/SEPIC exact-map flip location ────────────────────────────
# Uses DynamicsKit's own public continuation + special-point detection
# (`continuation_branch`, `map_special_points`) rather than a hand-rolled
# root-finder: this is both the idiomatic way any user would locate a flip
# with this library, and more robust than plain undamped Newton across the
# switching-time discontinuities in `cuk_converter`/`sepic_converter`.
function settle_to_fixed_point(sys, p, x0; iters=4000)
    x = SVector{sys.dim}(x0)
    for _ in 1:iters
        x = sys.f(x, p)
    end
    return x
end

function locate_flip(sys, iref_seed, iref_min, iref_max, other_params, x0; iters=4000)
    p0 = vcat(iref_seed, other_params)
    seed = settle_to_fixed_point(sys, p0, x0; iters=iters)
    cont = ContinuationConfig(
        p_min=iref_min, p_max=iref_max, param_index=1,
        ds=0.01, dsmax=0.02, dsmin=1e-6, max_steps=80, newton_tol=1e-11, newton_max_iter=30)
    branch = continuation_branch(sys, cont; initial_point=Vector{Float64}(seed), params=p0)
    points = map_special_points(sys, branch, p0; detect=(:pd,))
    isempty(points) && error("locate_flip: no period-doubling point found on [$iref_min, $iref_max]")
    return points[1]
end

cuk = cuk_converter()
cuk_pd = locate_flip(cuk, 2.8, 2.8, 3.2, [15.0, 10.0], SVector(18.0, 1.8, 33.0, 2.2))
record_gate!(gates, "Cuk exact-map first flip crossing",
    cuk_pd.converged && abs(real(cuk_pd.critical_multiplier) + 1) <= 2e-6 &&
        abs(imag(cuk_pd.critical_multiplier)) <= 1e-6,
    "Iref=$(round(cuk_pd.param, digits=6)) A, multiplier=$(cuk_pd.critical_multiplier)")
cuk_delta = cuk_pd.param - 4.0
record_gate!(gates, "Cuk source-threshold discrepancy is the documented ~1 A gap",
    0.9 <= abs(cuk_delta) <= 1.1,
    "exact-map flip at $(round(cuk_pd.param, digits=4)) A vs. source's approximate 4.0 A (delta=$(round(cuk_delta, digits=4)) A)")
cuk_smoke = all(isfinite, cuk.f(SVector(18.0, 1.8, 33.0, 2.2), [4.0, 15.0, 10.0]))
record_gate!(gates, "Cuk map is finite at the published operating point", cuk_smoke,
    "one exact switched-map iterate is finite at Iref=4.0 A")

sepic = sepic_converter()
sepic_pd = locate_flip(sepic, 5.0, 5.0, 5.5, [24.0, 10.0], SVector(24.0, 3.0, 0.0, 3.0))
record_gate!(gates, "SEPIC published first flip matches the source (~5.25 A)",
    sepic_pd.converged && abs(real(sepic_pd.critical_multiplier) + 1) <= 2e-6 &&
        abs(imag(sepic_pd.critical_multiplier)) <= 1e-6 && abs(sepic_pd.param - 5.25) <= 0.05,
    "Iref=$(round(sepic_pd.param, digits=6)) A, multiplier=$(sepic_pd.critical_multiplier)")

conclude(gates, "switching-map generator validation")
