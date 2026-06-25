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

`Codim2CurveResult` stores the stitched curve (`primary_values` / `secondary_values`), a `valid_mask`, all per-slice `raw_candidates`, per-slice provenance (`candidate_sources`, `slice_statuses`, `slice_messages`), and the engine metadata (`engine`, `tracking_anchor`, `tracking_tolerance`). The current engine is `:slice_tracking`, so the API stays stable even if the internal tracing engine changes later. `Codim2Config.threaded` now defaults to `false`; enable it explicitly only when you are comfortable running multiple continuation slices concurrently on your Julia build / dependency stack.

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

Supported result types include `BruteForceResult`, `LyapunovDiagramResult`, `BasinsResult`, `LyapunovFieldResult`, `BifurcationMapResult`, `PhasePortraitResult`, `PowerSpectrumResult`, `Codim2CurveResult`, `BranchResult`, and aggregate result objects.

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
