# Examples and cookbook workflows

The `examples/` directory contains executable Julia scripts. This guide shows smaller snippets and points to the longer scripts when a workflow deserves a full file.

## Example scripts

| Script | Purpose |
| --- | --- |
| `examples/henon_complete.jl` | End-to-end Henon map analysis |
| `examples/vilnius_oscillator_reference_figures.jl` | Vilnius oscillator reference figures from Ipatovs et al. (2023) |
| `examples/colpitts_oscillator_parameter_studies.jl` | Colpitts oscillator parameter sweeps, maps, phase portraits, and continuation overlays |
| `examples/colpitts_simple_beta_overlay.jl` | Simple Colpitts beta overlay |
| `examples/colpitts_simple_beta_period_scan.jl` | Simple Colpitts beta period scan |
| `examples/colpitts_exponential_voltage_cascade.jl` | Exponential Colpitts voltage cascade |
| `examples/colpitts_dynamic_beta_voltage_cascade.jl` | Dynamic-beta Colpitts voltage cascade |
| `examples/scientific_diagnostics_suite.jl` | Compact scientific diagnostics across discrete maps, ODE continuation, switching guards, and multistability maps |

Run scripts from the repository root:

```sh
julia --project=. examples/henon_complete.jl
```

The table above lists executable scripts that already exist in the repository. The snippets below are compact cookbook starting points; promote a workflow into `examples/` when it needs to be run as a maintained example.

## Scientific diagnostics suite

```sh
julia --project=. examples/scientific_diagnostics_suite.jl
```

This executable example runs compact diagnostic cases for Henon period doubling/window detection, Ikeda multistability plus Lyapunov diagnostics, Rossler continuation multipliers, boost converter switching guards, and the memristive diode bridge multistability map. It prints a JSON summary after asserting each case's expected diagnostic behavior.

## Reference figure and parameter studies

Vilnius oscillator reference figures:

```sh
julia --project=. examples/vilnius_oscillator_reference_figures.jl
```

The Vilnius script recreates reference-style brute-force and continuation overlays for Figures 11 and 13 from:

> A. Ipatovs, C. Iheanacho, D. Pikuļins, S. Tjukovs, A. Litviņenko, "Complete Bifurcation Analysis of the Vilnius Chaotic Oscillator", *Electronics* 12(13), Article 2861 (2023). doi:10.3390/electronics12132861

Use `VILNIUS_FIGURES=11`, `VILNIUS_FIGURES=13`, or `VILNIUS_FIGURES=11,13` to choose which figures to generate. Outputs are written under `var/output/vilnius_oscillator_reference_figures/`.

Colpitts oscillator parameter studies:

```sh
julia --project=. examples/colpitts_oscillator_parameter_studies.jl
```

The Colpitts script generates a coordinated set of parameter sweeps, 2D maps, phase portraits, and continuation overlays for the simple, exponential, and dynamic-beta transistor models. By default it runs a small smoke configuration; use `COLPITTS_STUDY_MODE=final` for denser grids and longer trajectories.

## Henon period doubling and hidden windows

```julia
using DynamicsKit

sys = henon_map()
base = [1.0, 0.3]

atlas = continuation_atlas(sys, AtlasConfig(
    periods=[1, 2, 3, 4, 6],
    brute_force=BruteForceConfig(
        param_min=0.0,
        param_max=1.4,
        param_steps=240,
        iterations=700,
        transient=500,
        fixed_params=base,
    ),
    continuation=ContinuationConfig(
        p_min=0.0,
        p_max=1.4,
        ds=0.006,
        dsmax=0.03,
        detect_bifurcation=3,
    ),
    adaptive_recon=true,
    reuse_neighbor_seeds=true,
); params=base)

plot_overlay(atlas.brute_force, atlas_branches(atlas))
atlas.coverage_summary
atlas.diagnostics
```

Use this to compare the observed brute-force cascade with recovered period branches and gap diagnostics.

## Ikeda multistability and quasiperiodicity

```julia
using DynamicsKit

sys = ikeda_map()

map = bifurcation_map(sys, BifurcationMapConfig(
    a_min=0.75,
    a_max=0.95,
    a_steps=100,
    b_min=5.0,
    b_max=8.0,
    b_steps=100,
    a_index=1,       # u
    b_index=3,       # b
    base_params=[0.82, 0.4, 6.0],
    max_period=8,
    precision=1e-3,
    iterations=400,
    multistability_initial_points=[
        [-0.5, -0.5],
        [0.5, 0.5],
        [1.5, -1.5],
    ],
    lyapunov_enabled=true,
    lyapunov_iterations=96,
))

plot_bifurcation_map(map)
```

For richer diagnostics, enable `multistability_initial_points` and `lyapunov_enabled` on the
`BifurcationMapConfig`; the returned diagnostics dict surfaces multistability and Lyapunov summaries.

## Lyapunov, codim-2, and ODE spectrum snippets

```julia
using DynamicsKit
using StaticArrays

henon_lyap = lyapunov_diagram(henon_map(), LyapunovConfig(
    param_min=0.9,
    param_max=1.3,
    param_steps=120,
    iterations=120,
    transient=40,
    fixed_params=[1.0, 0.3],
))

plot_lyapunov_diagram(henon_lyap)

affine = DiscreteMap(
    (x, p) -> SVector(p[1] * x[1] + p[2]),
    1,
    [:a, :b],
    "Affine map",
)

pd_curve = codim2_curve(affine, Codim2Config(
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

plot_codim2(pd_curve)

rossler_spec = power_spectrum(rossler_oscillator(), PowerSpectrumConfig(
    time_stop=300.0,
    dt=0.05,
    tail_fraction=0.5,
    state_index=1,
); params=[0.2, 0.2, 5.7])

plot_power_spectrum(rossler_spec)
```

Use the Lyapunov sweep for a compact route-to-chaos diagnostic, `codim2_curve` when you need a traced period-doubling/fold/NS boundary in a two-parameter plane, and the ODE spectrum when you need to distinguish narrowband periodic behavior from broadband candidate chaos at one representative parameter value.

## Rossler phase portrait and continuation

```julia
using DynamicsKit

sys = rossler_oscillator()
params = [0.2, 0.2, 5.7]

portrait = phase_portrait(sys, PhasePortraitConfig(
    time_start=0.0,
    time_stop=300.0,
    tail_fraction=0.5,
    poincare_crossings=100,
); params=params)

plot_phase_portrait(portrait)

branch = continuation_branch(sys, ContinuationConfig(
    p_min=2.0,
    p_max=6.0,
    ds=0.01,
    dsmax=0.05,
    param_index=3,
    ode_jacobian_method=:variational,
); params=params)

diag = continuation_branch_diagnostics(sys, branch, params;
    max_points=80,
    ode_jacobian_method=:variational,
)

diag["maxMultiplierModulus"]
```

## Colpitts ODE continuation with multiplier diagnostics

```julia
using DynamicsKit

sys = colpitts_exponential_oscillator()
base = copy(sys.default_params)

branch = continuation_branch(sys, ContinuationConfig(
    p_min=4.0,
    p_max=5.0,
    ds=0.02,
    dsmax=0.20,
    param_index=4,
    newton_tol=1e-8,
    newton_max_iter=30,
    detect_bifurcation=1,
    ode_jacobian_method=:variational,
); params=base, n_initial=10)

diag = continuation_branch_diagnostics(sys, branch, base;
    max_points=50,
    ode_jacobian_method=:variational,
)
```

Use `bench/reseed_benchmark.jl` to compare conservative and aggressive PALC settings on this real stiff Poincare-map system.

## Buck/boost switching behavior

```julia
using DynamicsKit

sys = boost_converter()
params = [2.0, 10.0, 20.0, 0.0]

branch = continuation_branch(sys, ContinuationConfig(
    p_min=0.5,
    p_max=4.5,
    ds=0.01,
    dsmax=0.05,
); initial_point=[0.0, 0.0], params=params)

diag = continuation_branch_diagnostics(sys, branch, params;
    include_switching_events=true,
)

diag["switchingEvents"]
```

Switching diagnostics report proximity to guard surfaces. Treat near-guard points as nonsmooth event candidates even when smooth multiplier labels are inconclusive.

## Memristive diode bridge multistability map

```julia
using DynamicsKit

sys = memristive_diode_bridge()

map = bifurcation_map(sys, BifurcationMapConfig(
    a_min=0.001,
    a_max=0.08,
    a_steps=60,
    b_min=0.02,
    b_max=0.08,
    b_steps=45,
    a_index=1,
    b_index=3,
    base_params=[0.0155, 6.02e-6, 0.05],
    max_period=6,
    precision=1e-3,
    iterations=120,
    multistability_initial_points=[
        [-1.0, 0.0, 0.5],
        [1.0, 0.0, -0.5],
    ],
    lyapunov_enabled=true,
))
```

Compare fixed-seed runs against neighbor-seeded and multiseed runs to probe multistability.

## Literature anchors

Several examples are built around established dynamical-systems models:

| System | Reference |
| --- | --- |
| Henon map | M. Henon, "A two-dimensional mapping with a strange attractor", *Communications in Mathematical Physics* 50, 69-77 (1976). doi:10.1007/BF01608556 |
| Ikeda map | S. M. Hammel, C. K. R. T. Jones, J. V. Moloney, "Global dynamical behavior of the optical field in a ring cavity", *J. Opt. Soc. Am. B* 2, 552-564 (1985). doi:10.1364/JOSAB.2.000552 |
| Rossler oscillator | O. E. Rossler, "An equation for continuous chaos", *Physics Letters A* 57, 397-398 (1976). doi:10.1016/0375-9601(76)90101-8 |
| Vilnius oscillator | A. Ipatovs, C. Iheanacho, D. Pikuļins, S. Tjukovs, A. Litviņenko, "Complete Bifurcation Analysis of the Vilnius Chaotic Oscillator", *Electronics* 12(13), Article 2861 (2023). doi:10.3390/electronics12132861 |
| Boost converter | W. C. Y. Chan, C. K. Tse, "Study of bifurcations in current-programmed DC/DC boost converters", *IEEE Trans. Circuits Syst. I* 44(12), 1129-1142 (1997). doi:10.1109/81.645151 |
| Memristive diode bridge | Q. Xu, Q. Zhang, N. Wang, H. Wu, B. Bao, "An Improved Memristive Diode Bridge-Based Band Pass Filter Chaotic Circuit", *Mathematical Problems in Engineering*, Article ID 2461964 (2017). doi:10.1155/2017/2461964 |

## Defining a system in your own file

You can define a system constructor in a standalone file without editing package source:

```julia
using DynamicsKit, StaticArrays

function logistic_system()
    DiscreteMap(
        (x, p) -> SVector(p[1] * x[1] * (1 - x[1])),
        1,
        [:r],
        "Logistic map",
    )
end

sys = logistic_system()
# ... then run brute_force_diagram / continuation_branch / bifurcation_map on `sys`.
```

(The browser workbench, in the separate `DynamicsKitWorkbench.jl` package, can also import such a
file and run analyses on it interactively.)
