"""
Two-parameter robust-chaos region certification validation.

`robust_chaos_region_certificate` composes adaptive-map candidate discovery,
Lyapunov-field coverage, atlas branch-absence slices, and basin censuses into
certified regions of an operating plane (see `docs/analysis-methods.md`,
"Robust-chaos region certificate"). This script checks two independent claims:

  1. Analytic: on a parameter plane that is Arnold's cat map (a strong,
     uniformly hyperbolic chaotic map with no attracting low-period orbit)
     everywhere except inside a narrow strip of known width (where it is
     instead a strict contraction to a stable fixed point), the certificate
     must certify regions covering the chaotic area up to its declared scale,
     exclude the strip entirely, replay deterministically, and never certify
     *more* area when given a smaller adaptive-refinement budget.
  2. Physical: on the peak-current-mode buck converter's `(Iref, R)` plane
     (see `docs/systems-catalog.md`), a robust design (`C = 2 uF`) certifies a
     region crossing the published `Iref = 2 A` robust band, while a fragile
     design (`C = 0.1 uF`) does not certify the corresponding window cluster.

Run:

    julia --project=. validation/robust_chaos_region_certification.jl
"""

using DynamicsKit
using StaticArrays

include(joinpath(@__DIR__, "common.jl"))

gates = ValidationGate[]

# ── Section 1: analytic striped-cat-map fixture ─────────────────────────────

const STRIP_CENTER = 0.5
const STRIP_HALF_WIDTH = 0.06
const STRIP_LO = STRIP_CENTER - STRIP_HALF_WIDTH
const STRIP_HI = STRIP_CENTER + STRIP_HALF_WIDTH
const TRUE_CHAOTIC_AREA = 1.0 - 2 * STRIP_HALF_WIDTH

function striped_cat_map()
    rule = function (x, p)
        a = p[1]
        if STRIP_LO <= a <= STRIP_HI
            # Inside the strip: a strict contraction to a stable fixed point.
            return SVector(0.25 * (x[1] - 0.25) + 0.25, 0.25 * (x[2] - 0.25) + 0.25)
        end
        # Outside the strip: Arnold's cat map on the torus (uniformly
        # hyperbolic, largest Lyapunov exponent log((3+sqrt(5))/2) > 0).
        return SVector(mod(x[1] + x[2], 1.0), mod(x[1] + 2 * x[2], 1.0))
    end
    return DiscreteMap(rule, 2, [:a, :b], "Striped cat-map robust-region fixture")
end

function analytic_region_config(; total_budget=850, max_depth=3)
    plane = BifurcationMapConfig(
        a_min=0.0, a_max=1.0, a_steps=8,
        b_min=0.0, b_max=1.0, b_steps=4,
        a_index=1, b_index=2,
        base_params=[0.25, 0.5],
        max_period=3,
        precision=1e-6,
        iterations=120,
        reuse_neighbor_seeds=false,
        lyapunov_iterations=140,
        lyapunov_transient=30,
        lyapunov_neutral_tolerance=1e-3,
    )
    atlas = AtlasConfig(
        brute_force=BruteForceConfig(
            param_min=0.0, param_max=1.0, param_index=1,
            fixed_params=[0.25, 0.5], param_steps=6,
            iterations=120, transient=50,
        ),
        continuation=ContinuationConfig(
            p_min=0.0, p_max=1.0, param_index=1,
            ds=0.04, dsmax=0.08, max_steps=40,
        ),
        periods=[1, 2, 3],
        max_period=3,
        recon_steps=5,
        cache_enabled=false,
        threaded=false,
    )
    basins = BasinsConfig(
        bif_param=0.25, param_index=1, fixed_params=[0.25, 0.5],
        x_min=0.13, x_max=0.87, x_steps=2,
        y_min=0.17, y_max=0.83, y_steps=2,
        iterations=40,
        max_period=3,
        precision=1e-6,
    )
    return RobustChaosRegionConfig(
        map=plane,
        adaptive=AdaptiveMapConfig(total_budget=total_budget, max_depth=max_depth),
        lyapunov_field=plane,
        atlas=atlas,
        basins=basins,
        max_atlas_slices_per_region=2,
        max_basin_knots_per_region=2,
        min_lyapunov_positive_fraction=0.80,
        min_lyapunov_resolved_fraction=0.80,
        min_atlas_slice_fraction=1.0,
        min_chaotic_basin_fraction=0.80,
        min_basin_resolved_fraction=0.80,
        boundary_edge_policy=:censored,
    )
end

region_signature(result) = [
    (verdict=region.verdict,
     a_min=round(region.a_min; digits=12), a_max=round(region.a_max; digits=12),
     b_min=round(region.b_min; digits=12), b_max=round(region.b_max; digits=12),
     area=round(region.area; digits=12), score=round(region.robustness_score; digits=12),
     depth=region.finest_depth,
     margin=isfinite(region.boundary_margin) ? round(region.boundary_margin; digits=12) : region.boundary_margin)
    for region in result.regions
]
certified_area(result) = sum(region.area for region in result.regions if region.verdict == :certified)

sys = striped_cat_map()
initial = [sqrt(2) - 1, sqrt(3) - 1]

full = robust_chaos_region_certificate(sys, analytic_region_config(); initial_point=initial)
replay = robust_chaos_region_certificate(sys, analytic_region_config(); initial_point=initial)
low_budget = robust_chaos_region_certificate(
    sys, analytic_region_config(total_budget=(8 + 1) * (4 + 1), max_depth=0); initial_point=initial)

certified = [region for region in full.regions if region.verdict == :certified]
full_area = certified_area(full)
low_area = certified_area(low_budget)
min_scale = 1 / (8 * 2^analytic_region_config().adaptive.max_depth)

record_gate!(gates, "certified regions exist", length(certified) >= 2,
    "certified_regions=$(length(certified)), total_regions=$(length(full.regions))")
record_gate!(gates, "periodic strip excluded",
    all(region.a_max <= STRIP_LO || region.a_min >= STRIP_HI for region in certified),
    join(["[$(round(r.a_min, digits=4)), $(round(r.a_max, digits=4))]" for r in certified], ", "))
record_gate!(gates, "chaotic area covered to declared scale",
    full_area >= TRUE_CHAOTIC_AREA - 4 * min_scale,
    "certified_area=$(round(full_area, digits=6)) true_area=$(round(TRUE_CHAOTIC_AREA, digits=6)) scale=$(round(min_scale, digits=6))")
record_gate!(gates, "budget monotonicity (a smaller budget never over-certifies)",
    low_area <= full_area + 1e-12,
    "low_budget_area=$(round(low_area, digits=6)) full_area=$(round(full_area, digits=6))")
record_gate!(gates, "deterministic replay", region_signature(full) == region_signature(replay),
    "signature_length=$(length(region_signature(full)))")
record_gate!(gates, "coverage fields round-trip",
    let rt = deserialize_robust_chaos_region_result(serialize_robust_chaos_region_result(full))
        rt.candidate_leaf_count == full.candidate_leaf_count &&
        rt.adaptive_budget_used == full.adaptive_budget_used &&
        rt.adaptive_total_budget == full.adaptive_total_budget &&
        rt.adaptive_budget_exhausted == full.adaptive_budget_exhausted &&
        rt.adaptive_uninspected_cell_count == full.adaptive_uninspected_cell_count &&
        rt.adaptive_max_depth_reached == full.adaptive_max_depth_reached &&
        rt.adaptive_max_depth_allowed == full.adaptive_max_depth_allowed &&
        rt.lyapunov_method == full.lyapunov_method &&
        rt.lyapunov_normalization == full.lyapunov_normalization &&
        rt.boundary_edge_policy == full.boundary_edge_policy &&
        region_signature(rt) == region_signature(full)
    end,
    "candidate_leaf_count=$(full.candidate_leaf_count), budget_used=$(full.adaptive_budget_used)")
record_gate!(gates, "edge censoring reported",
    any(region.boundary_edge_censored for region in full.regions),
    "edge_censored_regions=$(count(region -> region.boundary_edge_censored, full.regions))")

# ── Section 2: physical buck-converter (Iref, R) grounding ──────────────────
# Peak-current-mode buck converter analytic map (see `docs/systems-catalog.md`
# for the closed-form used by `buck_converter`). Reused here at two different
# output-capacitor values: a robust design and a fragile one, both operating
# points documented alongside the buck converter's switching guards.
@inline function pcmc_step(Vn, In, Iref, R, b, tn; C, L, T, E)
    a = -1 / (2 * R * C)
    if tn >= T
        c1 = In - E / R
        c2 = (1 / b) * ((E - Vn) / L - a * (In - E / R))
        In_c = exp(a * T) * (c1 * cos(b * T) + c2 * sin(b * T)) + E / R
        c3 = Vn - E
        c4 = (a / b) * (Vn + E) + In / (b * C)
        Vn_c = exp(a * T) * (c3 * cos(b * T) + c4 * sin(b * T)) + E
    else
        c3 = Vn - E
        c4 = (a / b) * (Vn + E) + In / (b * C)
        Vc0 = exp(a * tn) * (c3 * cos(b * tn) + c4 * sin(b * tn)) + E
        k1 = Iref
        k2 = (-1 / (b * L)) * (Vc0 + a * L * Iref)
        dt = T - tn
        In_c = exp(a * dt) * (k1 * cos(b * dt) + k2 * sin(b * dt))
        Vn_c = -L * exp(a * dt) * ((k1 * a + k2 * b) * cos(b * dt) + (k2 * a - k1 * b) * sin(b * dt))
    end
    return Vn_c, In_c
end

function buck_pcmc_map(; C, L, T, E, name)
    rule = function (x, p)
        Vn, In = x[1], x[2]
        Iref, R = p[1], p[2]
        R > 0 || return SVector(1e12, 1e12)
        denom = E - Vn
        tn = denom == 0 ? Inf : L * (Iref - In) / denom
        isfinite(tn) || return SVector(1e12, 1e12)
        disc = 1 / (L * C) - 1 / (4 * R^2 * C^2)
        if disc >= 0
            Vn_c, In_c = pcmc_step(Vn, In, Iref, R, sqrt(disc), tn; C=C, L=L, T=T, E=E)
        else
            Vc, Ic = pcmc_step(Vn, In, Iref, R, sqrt(complex(disc)), tn; C=C, L=L, T=T, E=E)
            Vn_c, In_c = real(Vc), real(Ic)
        end
        (isfinite(In_c) && isfinite(Vn_c)) || return SVector(1e12, 1e12)
        return SVector(Vn_c, In_c)
    end
    return DiscreteMap(rule, 2, [:Iref, :R], name)
end

function physical_region_config(; iref_min, iref_max, r_min, r_max, max_period, iterations)
    plane = BifurcationMapConfig(
        a_min=iref_min, a_max=iref_max, a_steps=2,
        b_min=r_min, b_max=r_max, b_steps=8,
        a_index=1, b_index=2,
        base_params=[0.5 * (iref_min + iref_max), 0.5 * (r_min + r_max)],
        max_period=max_period, precision=1e-5, iterations=iterations,
        reuse_neighbor_seeds=false, divergence_cutoff=1e8,
        lyapunov_iterations=160, lyapunov_transient=40, lyapunov_neutral_tolerance=1e-3,
    )
    atlas = AtlasConfig(
        brute_force=BruteForceConfig(
            param_min=r_min, param_max=r_max, param_index=2,
            fixed_params=[0.5 * (iref_min + iref_max), 0.5 * (r_min + r_max)],
            param_steps=8, iterations=iterations, transient=max(20, iterations ÷ 2),
        ),
        continuation=ContinuationConfig(
            p_min=r_min, p_max=r_max, param_index=2,
            ds=0.03, dsmax=0.08, max_steps=60, newton_tol=1e-8, newton_max_iter=30,
        ),
        periods=collect(1:max_period), max_period=max_period,
        recon_steps=6, cache_enabled=false, threaded=false, time_budget_s=45.0,
    )
    basins = BasinsConfig(
        bif_param=0.5 * (r_min + r_max), param_index=2,
        fixed_params=[0.5 * (iref_min + iref_max), 0.5 * (r_min + r_max)],
        x_min=0.5, x_max=3.5, x_steps=3, y_min=0.0, y_max=3.0, y_steps=3,
        max_period=max_period, iterations=max(80, iterations ÷ 2), precision=1e-5,
    )
    return RobustChaosRegionConfig(
        map=plane, adaptive=AdaptiveMapConfig(total_budget=180, max_depth=2),
        lyapunov_field=plane, atlas=atlas, basins=basins, slice_axis=:b,
        max_atlas_slices_per_region=1, max_basin_knots_per_region=1,
        min_lyapunov_positive_fraction=0.60, min_lyapunov_resolved_fraction=0.60,
        min_atlas_slice_fraction=1.0, min_chaotic_basin_fraction=0.60,
        min_basin_resolved_fraction=0.60,
    )
end

const NOMINAL = (E=3.7, L=5e-6, T=1e-6)

robust_design = buck_pcmc_map(C=2e-6, L=NOMINAL.L, T=NOMINAL.T, E=NOMINAL.E, name="Buck PCMC C=2uF (Iref x R)")
robust_plane = robust_chaos_region_certificate(
    robust_design,
    physical_region_config(iref_min=1.95, iref_max=2.05, r_min=2.5, r_max=9.5, max_period=8, iterations=500);
    initial_point=[2.0, 1.0])
robust_certified = [r for r in robust_plane.regions if r.verdict == :certified]
record_gate!(gates, "robust design (C=2uF) certifies a region crossing the Iref=2A band",
    any(r -> r.a_min <= 2.0 <= r.a_max && r.b_min <= 4.0 && r.b_max >= 8.0, robust_certified),
    join(["Iref=[$(round(r.a_min, digits=3)), $(round(r.a_max, digits=3))], R=[$(round(r.b_min, digits=3)), $(round(r.b_max, digits=3))], verdict=$(r.verdict)" for r in robust_plane.regions], "; "))

fragile_design = buck_pcmc_map(C=0.1e-6, L=NOMINAL.L, T=NOMINAL.T, E=NOMINAL.E, name="Buck PCMC C=0.1uF (Iref x R)")
fragile_plane = robust_chaos_region_certificate(
    fragile_design,
    physical_region_config(iref_min=0.95, iref_max=1.05, r_min=2.5, r_max=6.5, max_period=8, iterations=500);
    initial_point=[2.0, 1.0])
fragile_overlaps = [r for r in fragile_plane.regions
                    if r.verdict == :certified && r.a_min <= 1.0 <= r.a_max && r.b_min <= 4.0 && r.b_max >= 3.0]
record_gate!(gates, "fragile design (C=0.1uF) does not certify the same window cluster",
    isempty(fragile_overlaps),
    isempty(fragile_overlaps) ? "no certified overlap" :
        join(["R=[$(round(r.b_min, digits=3)), $(round(r.b_max, digits=3))]" for r in fragile_overlaps], "; "))

conclude(gates, "robust-chaos region certification")
