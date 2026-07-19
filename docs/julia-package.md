# Julia package usage

The package API is designed around a small set of system types, configuration structs, result structs, and plotting/I/O helpers.

```julia
using DynamicsKit
```

## Core types

| Type              | Purpose                                                                                       |
|-------------------|-----------------------------------------------------------------------------------------------|
| `DiscreteMap`     | Iterated map `x[n+1] = f(x[n], p)`. Map functions should return `SVector`s.                   |
| `ContinuousODE`   | In-place ODE `f!(du, u, p, t)` plus a `PoincareSection`.                                      |
| `PoincareSection` | Event section, crossing direction, projected coordinates, and optional full-state template.   |
| `SwitchingEvent`  | Optional guard metadata for border-collision, saturation, grazing, or other nonsmooth events. |

Built-in constructors are listed in [`systems-catalog.md`](systems-catalog.md).

## Brute-force diagram

A brute-force diagram sweeps one parameter and records post-transient attractor samples.

```julia
using DynamicsKit

sys = henon_map()

bf = brute_force_diagram(sys, BruteForceConfig(
    param_min=0.0,
    param_max=1.4,
    param_steps=400,
    iterations=600,
    transient=400,
    param_index=1,
    fixed_params=[1.0, 0.3],
))

plot_brute_force(bf)
```

`fixed_params` should be the full base parameter vector whenever the system has more than one parameter. The swept parameter overwrites `fixed_params[param_index]`.

## Lyapunov diagram

Largest-Lyapunov sweeps reuse the same parameter-injection pattern as brute-force diagrams, but store one exponent per parameter sample instead of an attractor cloud.

```julia
lyap = lyapunov_diagram(henon_map(), LyapunovConfig(
    param_min=0.9,
    param_max=1.3,
    param_steps=120,
    iterations=120,
    transient=40,
    param_index=1,
    fixed_params=[1.0, 0.3],
))

plot_lyapunov_diagram(lyap)
```

`LyapunovDiagramResult` stores the sampled parameter grid, exponent vector, per-sample classification labels, estimator-status labels, and sample counts.

## Lyapunov spectrum

For the full spectrum at one operating point (not a sweep), `lyapunov_spectrum(sys, LyapunovSpectrumConfig(...))` evolves an orthonormal tangent frame with the Benettin/QR method: discrete maps use the map's AD Jacobian per iteration, continuous flows integrate the first variational equation `dQ/dt = J(u) Q` and reorthonormalize every `renorm_dt` of flow time.

```julia
spectrum = lyapunov_spectrum(rossler_oscillator(), LyapunovSpectrumConfig(
    transient=200,
    steps=2000,
    renorm_dt=0.5,
); params=[0.2, 0.2, 5.7])

spectrum.exponents        # ordered largest → smallest, e.g. (+, 0, −) for chaotic Rössler
plot_lyapunov_spectrum(spectrum)
```

`LyapunovSpectrumResult` carries the ordered `exponents`, a `convergence` matrix of the running finite-time estimates (one row per accumulated interval, one column per exponent), the `estimation_status`, and `total_time`. Set `k` to track only the leading exponents. Two built-in sanity checks: the exponents sum to the mean log volume-change rate — `log|det J|` for maps, and the long-time average of the flow's divergence (the time-averaged trace of `J(u(t))` along the trajectory) for ODEs — and a bounded non-equilibrium flow always shows one numerically zero exponent.

## Lyapunov field

For a direct 2D exponent sweep, call `lyapunov_field(sys, config)` with the same parameter-plane axes you would use for a 2D map. This path skips period classification entirely and computes the Lyapunov field directly from the fixed initial condition at each cell. For continuous systems, `BifurcationMapConfig.min_crossing_time` controls the minimum accepted separation between Poincare crossings and therefore participates in both the run semantics and cache identity.

```julia
field = lyapunov_field(memristive_diode_bridge(), BifurcationMapConfig(
    a_min=0.001,
    a_max=0.08,
    a_steps=119,
    b_min=0.02,
    b_max=0.08,
    b_steps=89,
    a_index=1,
    b_index=3,
    base_params=[0.001, 6.02e-6, 0.02],
    lyapunov_iterations=220,
    lyapunov_transient=120,
    lyapunov_perturbation=1e-8,
    divergence_cutoff=1e6,
); initial_point=[0.0, -0.01, 0.0])

plot_lyapunov_field(field)
```

`lyapunov_field(result::BifurcationMapResult)` still works for combined map+field runs that were computed via `bifurcation_map(..., lyapunov_enabled=true)`. Use the direct `lyapunov_field(sys, config)` method when you want the exponent field itself as the primary artifact and want to avoid paying for a full period map first.

## Robust-chaos certificate

`robust_chaos_certificate` combines three independently limited evidence layers over one
parameter interval: a largest-Lyapunov sweep, continuation-atlas recovery of stable
low-period windows, and a basin census at one representative operating point. A basin
seed with no detected low period counts as chaotic only when a separate finite-time
Lyapunov estimate is resolved and positive.

```julia
sys = henon_map()
lyap = LyapunovConfig(
    param_min=1.1, param_max=1.2, param_steps=80,
    param_index=1, fixed_params=[1.15, 0.3],
    transient=300, iterations=500,
)
atlas = AtlasConfig(
    periods=collect(1:8),
    brute_force=BruteForceConfig(
        param_min=1.1, param_max=1.2, param_steps=120,
        param_index=1, fixed_params=[1.15, 0.3],
        transient=300, iterations=500,
    ),
    continuation=ContinuationConfig(
        p_min=1.1, p_max=1.2, param_index=1,
        ds=0.001, dsmax=0.005,
    ),
)
basins = BasinsConfig(
    bif_param=1.15, param_index=1, fixed_params=[1.15, 0.3],
    x_min=-1.5, x_max=1.5, y_min=-0.5, y_max=0.5,
    x_steps=30, y_steps=30, max_period=8, iterations=600,
)

certificate = robust_chaos_certificate(
    sys,
    RobustChaosConfig(lyapunov=lyap, atlas=atlas, basins=basins);
    initial_point=[0.1, 0.1],
)
certificate.overall_verdict  # :certified, :fragile, or :inconclusive
certificate.certificate_items
```

When an audit or visualization needs the exact supporting results, call
`robust_chaos_evidence` with the same arguments:

```julia
evidence = robust_chaos_evidence(
    sys,
    RobustChaosConfig(lyapunov=lyap, atlas=atlas, basins=basins);
    initial_point=[0.1, 0.1],
)
evidence.certificate.overall_verdict
evidence.lyapunov
evidence.atlas
evidence.basins
evidence.basin_classifications
```

`basin_classifications` is the final per-seed certificate decision and is not
the same as `evidence.basins.periodicity`: undetected-period seeds have been
classified further as chaotic, non-chaotic, or unresolved by finite-time
Lyapunov estimation.

### Source-result reuse (optional)

If you have already computed a Lyapunov diagram or atlas with identical
system/parameter settings, pass it as a keyword to skip that layer's sweep:

```julia
# Pre-compute the atlas independently (e.g., from an earlier interactive session).
existing_atlas = continuation_atlas(sys, atlas; initial_point=[0.1, 0.1])

# The certificate reuses the atlas result; only the Lyapunov and basin layers run fresh.
certificate = robust_chaos_certificate(
    sys,
    RobustChaosConfig(lyapunov=lyap, atlas=atlas, basins=basins);
    atlas_result=existing_atlas,
    initial_point=[0.1, 0.1],
)
```

Available keywords: `lyapunov_result::LyapunovDiagramResult` and
`atlas_result::AtlasResult`. Each is validated before use; an `ArgumentError` is thrown on
system-name mismatch, interval/grid mismatch, or missing period coverage.
Validation uses floating-point representation tolerance only (tight, ~8 ULP), not scientific
tolerance, so scientifically different values are always rejected.

**Bounded responsibility:** atlas reuse requires exact interval endpoint match — zoomed
subinterval certificates rerun the atlas until subset coverage accounting exists. Brute-force
and continuation sources cannot replace the full atlas reconnaissance, so no reuse is offered
for those. Basin results are not accepted for reuse because `BasinsResult` does not yet retain
enough parameter-injection metadata to prove that a supplied grid represents the same physical
slice; the basin layer therefore always runs from `config.basins`.

`:certified` means all three layers passed their configured thresholds. `:fragile`
means at least one layer found contrary evidence; `:inconclusive` means the available
coverage could not support either conclusion. The claim is always bounded to the
sampled interval, atlas period ceiling and search effort, basin grid, finite-time
Lyapunov settings, and selected initial conditions. It is not a mathematical proof
that no stable orbit exists outside those limits. Use
`serialize_robust_chaos_certificate` and
`deserialize_robust_chaos_certificate` for the compact versioned summary, or
`serialize_robust_chaos_evidence` and `deserialize_robust_chaos_evidence` for
the complete versioned evidence bundle.

## Continuation branch

Continuation traces periodic solutions with pseudo-arclength continuation.

```julia
branch = continuation_branch(
    henon_map(),
    ContinuationConfig(
        p_min=0.0,
        p_max=1.4,
        ds=0.01,
        dsmax=0.05,
        max_steps=1000,
        detect_bifurcation=3,
    );
    initial_point=[0.63, 0.19],
    params=[1.0, 0.3],
)

plot_branches([branch])
```

Higher-period branches use the period argument:

```julia
p2 = continuation_branch(
    henon_map(),
    ContinuationConfig(p_min=0.0, p_max=1.4, ds=0.01, dsmax=0.05),
    2;
    initial_point=[0.5, 0.1],
    params=[1.0, 0.3],
    trim_to_minimal_period=true,
)
```

`trim_to_minimal_period=true` removes lower-period aliases from period-N continuation output.

## Collocation periodic-orbit continuation

For continuous-time systems, `continuation_orbit_collocation` continues the whole
time-parameterized orbit and its period as a boundary-value problem (orthogonal
collocation), instead of a fixed point of the Poincaré return map. Return-map shooting
(`continuation_branch`) stays the default; collocation is an alternative for
ill-conditioned return-map problems and for cross-validating the shooting branches.

```julia
res = continuation_orbit_collocation(
    vilnius_oscillator(),
    CollocationConfig(
        continuation = ContinuationConfig(p_min=0.15, p_max=0.45, ds=0.01, dsmax=0.02,
                                          param_index=1),
        ntst = 40, m = 4,
    );
    period = 1,
    params = [0.25, 30.0, 0.2],
    initial_point = [0.0, 0.1, 0.0],
)

mu = orbit_branch_parameters(res)
T  = orbit_branch_periods(res)                 # flow period of each orbit
A  = orbit_branch_amplitude(res; state_index=1)
t, states = orbit_branch_orbit(res, 1)         # decode one orbit (dim × L)
stable, multipliers = orbit_branch_stability(res, vilnius_oscillator(), 1)
```

The collocation continuation ignores the return-map-specific `ContinuationConfig` fields
(`detect_bifurcation`, `ode_jacobian_method`, `save_sol_every_step`, `detect_fold`); its
Jacobian comes from BifurcationKit's collocation discretization. `ode_jacobian_method`
only affects the *stability* accessors — `orbit_branch_multipliers`/`orbit_branch_stability`
take it as a keyword (default `:variational`) for the return-map monodromy.
`OrbitBranchResult` stores the periodic-orbit `ContResult` and the collocation problem.
Stability comes from the return-map monodromy (the nontrivial Floquet multipliers) via
the same variational machinery as the shooting branches, not BifurcationKit's
collocation-Floquet eigenvalues (whose largest-magnitude entries are spurious
discretization modes). On the analytic radial oscillator the recovered period, amplitude,
and multiplier match the closed-form limit cycle and the shooting return-map multiplier;
on the stiff memristive diode bridge the collocation branch — which return-map shooting's
automatic seeder does not converge on — agrees with an independent shooting Newton solve
to ~1e-4 in both the fixed point and the multipliers.

## Branch diagnostics

Use diagnostics to inspect numerical trustworthiness and stability:

```julia
diag = continuation_branch_diagnostics(
    henon_map(),
    branch,
    [1.0, 0.3];
    max_points=100,
    include_residuals=true,
    include_multipliers=true,
    include_switching_events=true,
)

diag["maxResidualNorm"]
diag["maxMultiplierModulus"]
diag["stabilityFlags"]
```

For ODE branches, `ContinuationConfig(ode_jacobian_method=:variational, ...)` requests variational-equation monodromy where available; `:finite_difference` remains the fallback.

## Branch classification (families and basins)

Two conservative, geometry-based classifiers label continuation branches from their sampled Poincaré-orbit geometry. Both return one record per input branch, in the same order as `branches`.

`branch_family_assignments(sys, branches; kwargs...) -> Vector{BranchFamilyAssignment}` groups branches that trace the same attractor family. Branches are only compared when they share the same minimal period (`same_period_only=true`), so a period-doubled child is never silently merged with its parent unless a caller opts in. Tuning knobs: `sample_count`, `min_overlap_fraction`, `distance_tolerance`, the parameter context (`params`, `linked_param_indices`), and the ODE orbit-sampling controls (`solver`, `reltol`, `abstol`, `tmax`, `min_crossing_time`).

```julia
branches = [
    continuation_branch(henon_map(), ContinuationConfig(p_min=0.0, p_max=1.4);
                        initial_point=[0.63, 0.19], params=[1.0, 0.3]),
    continuation_branch(henon_map(), ContinuationConfig(p_min=0.0, p_max=1.4), 2;
                        initial_point=[0.5, 0.1], params=[1.0, 0.3]),
]

fams = branch_family_assignments(henon_map(), branches; params=[1.0, 0.3])
fams[1].family_id        # e.g. "family-1"
fams[1].confidence
```

Each `BranchFamilyAssignment` carries `branch_index`, `family_index`, `family_id`, `family_label`, `confidence`, and a `diagnostics` dict (`period`, `paramMin`, `paramMax`, `sampleCount`, ...).

`branch_basin_assignments(sys, branches, brute_force; kwargs...) -> Vector{BranchBasinAssignment}` classifies each branch as `observed` or `unobserved` against the attractor cloud reached by a supplied `BruteForceResult` — i.e. whether the branch's sampled orbits appear in what that brute-force seed actually visited. It does **not** enumerate every basin of the system.

```julia
bf  = brute_force_diagram(henon_map(),
                          BruteForceConfig(param_min=0.0, param_max=1.4, param_steps=400))
bas = branch_basin_assignments(henon_map(), branches, bf; params=[1.0, 0.3])
bas[1].observed          # true if this branch matches the seed's observed attractor
```

Each `BranchBasinAssignment` carries `branch_index`, `basin_index`, `basin_id` (`"observed"`/`"unobserved"`), `basin_label`, `observed::Bool`, `confidence`, and a `diagnostics` dict (`medianDistance`, `matchedSampleCount`, `paramTolerance`, ...).

## Switching-event diagnostics

`switching_event_diagnostics(sys, states, params) -> Dict{String,Any}` reports how close sampled states come to a system's `SwitchingEvent` guards (e.g. the switching surfaces of the buck / boost converters). `states` is a vector of state samples; `params` is either one shared parameter vector or one vector per sample. The returned dict summarizes proximity across all events: `eventCount`, `sampledPointCount`, `nearEventCount`, `nearestEvent`, `minDistance`, `minNormalizedDistance`, a per-event `events` array (each with `name`, `kind`, `near`, `minDistance`, ...), any guard-evaluation `warnings`, and a `status` (`"ok"`, `"warning"`, or `"unavailable"` when the system declares no switching events). This is the same producer behind `continuation_branch_diagnostics(...; include_switching_events=true)` and the 2-D map switching diagnostics.

## Map normal forms and special points

`map_normal_form(sys, kind, state, params; period=1)` computes the local coefficient for a fold (`kind=:fold`), flip (`:pd`), or Neimark-Sacker point (`:ns`) of the period-`N` map `G=F^N`. The overload `map_normal_form(sys, point::MapSpecialPoint, params)` uses the point's kind, state, and period. Discrete maps use nested ForwardDiff directional derivatives. Continuous ODEs use centered finite differences of the Poincare return map and require three successive step sizes to agree in sign, classification, and scale. `normal_form_fd_step` controls the initial scale (default `3e-3`); the implementation increases it adaptively when integration error dominates.

The implementation uses the standard Kuznetsov/MATCONT map convention. The right eigenvector has Euclidean norm one and the left eigenvector is scaled so the Hermitian product `dot(p,q)=1`. Complex multilinear forms are evaluated by real/imaginary multilinear expansion because ForwardDiff accepts real directions.
The formulas follow Kuznetsov, *Elements of Applied Bifurcation Theory*, map
normal forms (DOI `10.1007/978-1-4757-3978-7`).

- Fold: `b = 1/2 <p,B(q,q)>`. Its sign depends on the real eigenvector orientation, so the result only reports `:nondegenerate` or `:degenerate`.
- Flip: `c = 1/6 <p,C(q,q,q)> + <p,B(q,h20)>`, where `h20=(I-A)^-1 B(q,q)/2`. `c>0` is supercritical/soft and `c<0` is subcritical/hard.
- Neimark-Sacker: `d = Re(conj(lambda)/2 * (<p,C(q,q,qbar)> + <p,B(h20,qbar)> + 2<p,B(h11,q)>))`, where `h11=-(A-I)^-1 B(q,qbar)` and `h20=-(A-lambda^2 I)^-1 B(q,q)`. `d<0` is supercritical and `d>0` is subcritical.

`MapNormalForm` is plain data with `kind`, `coefficient_name` (`:b`, `:c`, or `:d`), an optional `coefficient`, `criticality`, `status`, and the exact `convention`. Degenerate coefficients, strong resonances, near-singular homological equations, multiple simultaneously critical NS pairs, unstable finite-difference steps, and unavailable critical eigenvectors carry explicit statuses; no coefficient is fabricated when evaluation is unreliable.

`map_special_points(sys, branch, base_params)` locates period-doubling (`:pd`), fold (`:fold`), and Neimark-Sacker (`:ns`) points on a continued map / Poincare return-map branch. BifurcationKit's residual-convention detector misses map period-doublings (multiplier `mu = lambda + 1`, so `mu -> -1` never crosses the imaginary axis); this routine detects sign changes of `det(J-I)` (fold), `det(J+I)` (flip), and `abs(mu_c)-1` for each non-real conjugate pair (NS), then refines each by arclength bisection with a fixed-point re-solve. Real multipliers near `+1` or `-1` are not accepted as NS points. Results are sorted and deduplicated by kind, parameter, and (for NS points) critical multiplier, so distinct simultaneous pairs remain distinct.

```julia
branch = continuation_branch(boost_converter(), ContinuationConfig(p_min=1.2, p_max=1.95, ds=0.005, param_index=1), 1;
                             initial_point=[17.4, 1.11], params=[1.5, 10.0, 20.0, 0.0])
sp = map_special_points(boost_converter(), branch, [1.5, 10.0, 20.0, 0.0]; detect=[:pd])
sp[1].kind             # :pd
sp[1].param            # subharmonic period-doubling (μ = -1), missed by branch.branch.specialpoint
sp[1].critical_multiplier
sp[1].normal_form     # MapNormalForm, attached by default
```

Each `MapSpecialPoint` carries `kind`, `param`, `state`, `multipliers`, `critical_multiplier`, `test_value`, `period`, `converged`, and optional `normal_form`. Pass `attach_normal_forms=false` to skip coefficient evaluation. Complex-conjugate pairs contribute a non-negative factor to each fold/flip determinant, so NS crossings are not reported as PD/fold. On the Henon map the located period-1 flip (`a=0.3675`) and fold (`a=-0.1225`) match their closed-form values.

`serialize_map_normal_form` / `deserialize_map_normal_form` and `serialize_map_special_point` / `deserialize_map_special_point` provide strict, versioned JSON-plain dictionaries. Complex multipliers are represented as `[real, imag]` pairs.

## Codimension-2 bifurcation curves

`codim2_curve` assembles a traced curve in a two-parameter plane by sweeping a secondary parameter and running a 1D continuation slice along the primary parameter at each secondary value.

```julia
using StaticArrays

affine = DiscreteMap(
    (x, p) -> SVector(p[1] * x[1] + p[2]),
    1,
    [:a, :b],
    "Affine map",
)

curve = codim2_curve(affine, Codim2Config(
    continuation=ContinuationConfig(
        p_min=-1.4,
        p_max=-0.6,
        ds=0.02,
        dsmax=0.05,
        detect_bifurcation=3,
        param_index=1,
    ),
    second_min=-0.3,
    second_max=0.3,
    second_steps=24,
    second_param_index=2,
    fixed_params=[-1.2, 0.0],
    bifurcation_kind=:pd,
); initial_point=[0.0])

plot_codim2(curve)
```

`Codim2CurveResult` stores the stitched curve (`primary_values` / `secondary_values`), a `valid_mask`, all per-slice `raw_candidates`, per-slice provenance (`candidate_sources`, `slice_statuses`, `slice_messages`), and the engine metadata (`engine`, `tracking_anchor`, `tracking_tolerance`). `Codim2Config.threaded` now defaults to `false`; enable it explicitly only when you are comfortable running multiple continuation slices concurrently on your Julia build / dependency stack.

### Defining-system engine

Setting `engine=:defining_system` on `Codim2Config` continues the bifurcation
condition itself instead of stitching slices: the fixed-point equation of the
period-`N` map is augmented with the eigenvector condition for the defining
multiplier (`:pd` → −1, `:fold` → +1) and the augmented system is continued in
the secondary parameter with pseudo-arclength continuation. One anchor slice
seeds the curve; every candidate is verified against the actual return-map
multiplier before seeding, and per-sample diagnostics (`fixed_point_residuals`,
`multipliers`) let each returned point be checked against the defining
condition. Points are solved to Newton tolerance (slice tracking is limited to
half the branch sampling distance), and the curve can follow folds of the locus
itself (`curve_fold_secondary_values`). Returns `Codim2ContinuationResult`
(arc-ordered `primary_values`/`secondary_values`, per-sample `states` and
`defining_vectors` columns, seed metadata); `plot_codim2` and
`save_result`/`load_result` accept it, and
`serialize_codim2_continuation_result`/`deserialize_codim2_continuation_result`
provide the JSON-plain wire form. `:ns` curves carry the complex defining
vector in `defining_vectors`/`defining_vectors_imag` and the multiplier angle
in `phase_angles`. The optional
`curve_continuation` config controls the secondary-parameter leg (bounds,
step sizes, Newton settings); its `param_index`/`linked_param_indices` are
ignored — the leg always continues the secondary parameter
(`second_param_index`). `nothing` derives conservative settings from the
secondary grid.

## Periodic skeleton search

Skeleton search finds periodic orbits at one parameter value by Newton iteration from a seed grid.

```julia
skeleton = find_periodic_skeleton(
    henon_map(),
    1:4,
    1.2;
    params=[1.2, 0.3],
    search_min=[-2.0, -1.0],
    search_max=[2.0, 1.0],
    n_initial=15,
)
```

Each returned item includes the period, point, multipliers, and a stability flag.

## Automatic continuation atlas

The atlas pipeline combines a reconnaissance sweep, window detection, skeleton recovery, continuation, gap retries, optional branch switching, optional sparse-tail auto-refinement for discrete and continuous branches, and optional seed reuse.

For high-period discrete-map branches, continuation now phase-canonicalizes the recorded representative point along the orbit before results are returned. That keeps `x1`, `x2`, … summaries and saved branch payloads from spuriously hopping between different phases of the same smooth orbit family. Continuous-time branches keep their existing plot-time phase alignment, while atlas auto-refinement now also applies to under-resolved continuous Poincare branches.

Sparse-tail detection targets two distinct under-resolution patterns. Individual large parameter gaps (the dotted tails PALC leaves when its step ramps toward `dsmax`) are flagged once a gap exceeds `0.7 * dsmax`, so densely sampled branches with only a coarse tail are refined locally around that tail. Separately, a uniformly coarse short branch (e.g. a six-point branch-switch probe) is fully re-swept — but only when its *median* gap is itself a large fraction of `dsmax`. That median-gap gate prevents a branch with a fine interior and a single sparse tail from being re-swept end to end and pushed past its real parameter support. Both heuristics are expressed relative to `ds`/`dsmax`, so they apply uniformly to every discrete and continuous system rather than being tuned to any one map.

```julia
sys = henon_map()

atlas = continuation_atlas(sys, AtlasConfig(
    periods=[1, 2, 4],
    brute_force=BruteForceConfig(
        param_min=0.0,
        param_max=1.4,
        param_steps=160,
        iterations=500,
        transient=320,
        fixed_params=[1.0, 0.3],
    ),
    continuation=ContinuationConfig(
        p_min=0.0,
        p_max=1.4,
        ds=0.01,
        dsmax=0.05,
        max_steps=1000,
    ),
    adaptive_recon=true,
    branch_switching=true,
    reuse_neighbor_seeds=true,
    # Sparse-tail auto-refinement (on by default). Each accepted branch may be densified for up to
    # `auto_refine_max_passes` passes before coverage scoring.
    auto_refine_sparse_branches=true,
    auto_refine_max_passes=1,
); params=[1.0, 0.3])

branches = atlas_branches(atlas)
plot_overlay(atlas.brute_force, branches)
```

Set `auto_refine_sparse_branches=false` (or `auto_refine_max_passes=0`) to disable refinement and accept branches exactly as continuation returns them; raise `auto_refine_max_passes` to densify stubborn tails over additional passes. Each branch's per-attempt outcome is reported in its diagnostics under `autoRefineApplied`, `autoRefineReason`, `autoRefineIntervalsDetected`, and `autoRefinePointCountBefore`/`After`.

Important atlas output fields:

| Field              | Meaning                                                                         |
|--------------------|---------------------------------------------------------------------------------|
| `brute_force`      | Reconnaissance-derived brute-force cloud used for plotting                      |
| `recon_samples`    | Period/status/confidence samples from the reconnaissance pass                   |
| `windows`          | Candidate periodic windows segmented from the reconnaissance data               |
| `branch_records`   | Recovered continuation branches and provenance                                  |
| `gaps`             | Uncovered or weakly covered windows after recovery and retries                  |
| `coverage_summary` | Parameter and geometry coverage diagnostics                                     |
| `diagnostics`      | Feature flags, seed reuse, branch switching, adaptive recon, and cache metadata |

## Continuous-time systems

Continuous systems operate through Poincare return maps. A system carries:

- full ODE dimension;
- section condition and crossing direction;
- projected section coordinates used as the map state;
- default initial state and parameter vector when available.

Example:

```julia
vilnius = vilnius_oscillator()

branch = continuation_branch(vilnius, ContinuationConfig(
    p_min=0.05,
    p_max=0.4,
    ds=0.002,
    dsmax=0.008,
    ode_jacobian_method=:variational,
); params=[0.105, 35.0])
```

## Basins of attraction

Basins classify an initial-condition grid at one fixed parameter value.

```julia
basins = basins_of_attraction(henon_map(), BasinsConfig(
    bif_param=1.2,
    max_period=8,
    precision=1e-4,
    iterations=600,
    x_min=-2.0,
    x_max=2.0,
    x_steps=100,
    y_min=-1.0,
    y_max=1.0,
    y_steps=100,
    fixed_params=[1.2, 0.3],
))

plot_basins(basins)
```

For higher-dimensional systems, use `x_index`, `y_index`, and `ic_template` to choose which state dimensions the grid varies.

`BasinsResult` now keeps the chosen `x_index`, `y_index`, and resolved `ic_template` so saved results preserve the slice-plane definition that produced the heatmap.

## Two-parameter bifurcation map

```julia
map = bifurcation_map(henon_map(), BifurcationMapConfig(
    a_min=0.0,
    a_max=1.4,
    a_steps=80,
    b_min=0.0,
    b_max=0.35,
    b_steps=80,
    a_index=1,
    b_index=2,
    max_period=8,
    precision=1e-3,
    iterations=400,
    base_params=[1.0, 0.3],
))

plot_bifurcation_map(map)
```

The public `BifurcationMapResult.periodicity` matrix preserves the historical convention that `0` means "no finite period detected." When `lyapunov_enabled=true`, the same result also carries a first-class `LyapunovFieldResult` layer:

```julia
map = bifurcation_map(ikeda_map(), BifurcationMapConfig(
    a_min=0.75, a_max=0.95, a_steps=80,
    b_min=5.0, b_max=8.0, b_steps=80,
    a_index=1,
    b_index=3,
    base_params=[0.82, 0.4, 6.0],
    iterations=240,
    max_period=8,
    lyapunov_enabled=true,
    lyapunov_iterations=96,
))

field = lyapunov_field(map)
plot_lyapunov_field(field; zero_contour=true)
```

The sweep functions also return a diagnostics dict with richer information — status, closure
confidence, Lyapunov estimates, multistability, adaptive refinement, and neighbor-seed traversal
metadata (e.g. `bifurcation_map` via the lower-level `DynamicsKit._bifurcation_map`, which returns
`(result, diagnostics)`).

## Phase portrait

```julia
portrait = phase_portrait(rossler_oscillator(), PhasePortraitConfig(
    time_start=0.0,
    time_stop=300.0,
    tail_fraction=0.5,
    poincare_crossings=80,
); params=[0.2, 0.2, 5.7])

plot_phase_portrait(portrait)
```

Phase portraits are for ODE systems. They keep a trajectory tail plus Poincare crossings.

## Power spectrum

Power spectra are ODE-only and compute a one-sided FFT from a uniformly sampled trajectory tail.

```julia
spectrum = power_spectrum(rossler_oscillator(), PowerSpectrumConfig(
    time_stop=300.0,
    dt=0.05,
    tail_fraction=0.5,
    window=:hann,
    state_index=1,
); params=[0.2, 0.2, 5.7])

plot_power_spectrum(spectrum)
```

`PowerSpectrumResult` stores the retained time tail, sampled signal, one-sided frequency grid, power vector, analyzed state index, and parameter vector.

## Plot composition helpers

The plotting layer now includes small composition helpers for figure assembly:

```julia
left = plot_lyapunov_diagram(henon_lyap)
right = plot_codim2(curve)

plot_seed_pair_composite(left, right)
plot_panel_grid([left, right, left, right]; layout=(2, 2))
```

Use `plot_overlay_heatmap(base, curves)` when you want to reuse an existing heatmap base (`LyapunovFieldResult`, `BifurcationMapResult`, or a prebuilt `Plots.Plot`) and overlay one or more codim-2 curves.

## Save and load results

```julia
save_result("henon_bf.jld2", bf)
loaded = load_result("henon_bf.jld2")
```

Supported result types include `BruteForceResult`, `LyapunovDiagramResult`, `LyapunovSpectrumResult`, `BasinsResult`, `LyapunovFieldResult`, `BifurcationMapResult`, `PhasePortraitResult`, `PowerSpectrumResult`, `Codim2CurveResult`, `BranchResult`, and aggregate result objects.

For a portable **JSON-plain** form (e.g. to embed a result in an HTTP payload or a non-JLD2 store)
the result types also have dict serializers — `serialize_bruteforce_result` / `serialize_branch_result`
/ `serialize_atlas_result` (and their `deserialize_*` inverses). Branch points are serialized
columnar so no fragile `BifurcationKit` internals are persisted.

## Integration API: kernels, effective settings, and diagnostics

These are for callers that drive the analyses programmatically and need more than the headline result
(the browser workbench is the primary consumer; scripted pipelines may want them too).

- **`bifurcation_map_kernel(sys, config; initial_point=nothing, cells=nothing[, solver, reltol, abstol])`**
  — like `bifurcation_map` but returns `(result, diagnostics::Dict)` and accepts a pre-seeded
  `cells::MapCellGrid` so a cache layer can compute only the unknown cells. The public `bifurcation_map`
  deliberately drops the diagnostics; `lyapunov_field` / `basins_of_attraction` already accept `cells=`
  directly, so only the map sweep needs a separate kernel entry point.

- **`map_effective_settings(config; na, nb, full_transient)`** — resolves a `BifurcationMapConfig` into
  the derived settings a sweep actually runs with, in one call:
  `(; seed_mode, lyapunov_enabled, lyapunov_iterations, lyapunov_transient, multistability_enabled,
  transient_budget, neighbor_transient, tile_sizes, tile_count)`. `na`/`nb` (grid dimensions, default
  the config's own) only affect `tile_sizes`/`tile_count`.

- **Diagnostics producers** — the summary dicts the map kernel and atlas assemble, exposed so a consumer
  can build the same payloads from raw data it already holds:
  `map_lyapunov_diagnostics`, `map_neighbor_seed_diagnostics`, `poincare_crossing_diagnostics_summary`,
  and `orbit_geometry_summary`. Each returns a JSON-plain `Dict`. (These are the library's own
  diagnostics format — the same dicts appear in the kernel's returned `diagnostics`.)

- **Result/branch accessors** — `branch_points(result)`, `trim_branch_to_period`,
  `collect_distinct_period_branches`, `branch_stability`, `branches_for_skeleton_param`,
  `is_duplicate_branch`, `poincare_projected`, `splice_refined_continuous_branches`; system accessors
  `state_dim(sys)` and `switching_events(sys)`; and the trace-data helpers behind the Plots recipes
  (`branch_plot_traces`, `resolve_plot_params`, `branch_point_state`, `orbit_phase_alignment_shift`,
  `phase_jump_break_indices`, `trace_breaks`, `codim2_curve_label`, `codim2_valid_runs`).

## Adding a system in code

Discrete map:

```julia
using StaticArrays
using DynamicsKit

logistic = DiscreteMap(
    (x, p) -> SVector(p[1] * x[1] * (1 - x[1])),
    1,
    [:r],
    "Logistic map",
)
```

Continuous ODE:

```julia
using DynamicsKit

function duffing!(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * u[2] - u[1]^3
    return nothing
end

section = PoincareSection((u, t, integrator) -> u[2];
    direction=:up,
    projection=[1],
    template=[0.0, 0.0],
)

duffing = ContinuousODE(duffing!, 2, section, [:mu], "Duffing";
    tspan_hint=50.0,
    default_initial_state=[0.1, 0.0],
    default_params=[0.2],
)
```
