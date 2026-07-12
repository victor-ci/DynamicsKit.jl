# Built-in systems catalog

The package exports built-in constructors for discrete maps and continuous ODE systems. The
`DynamicsKitWorkbench.jl` package exposes the same systems with documented parameters, defaults, and
presets; the analyses listed below are library capabilities available to any caller.

## System table

| Key | Constructor | Kind | Parameters | Analyses |
| --- | --- | --- | --- | --- |
| `henon` | `henon_map()` | Discrete map | `a`, `b` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, skeleton, basins, 2D map |
| `buck` | `buck_converter()` | Discrete switching map | `Iref`, `Ein` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, skeleton, basins, 2D map |
| `buck_voltage_mode` | `buck_voltage_mode(...)` | Discrete switching map | `E`, `Vref`, `R`, `gain` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, skeleton, basins, 2D map |
| `boost_converter` | `boost_converter(...)` | Discrete switching map | `Iref`, `E`, `R`, `Sc` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, skeleton, 2D map |
| `vilnius` | `vilnius_oscillator(...)` | Continuous ODE | `a`, `b`, `epsilon` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, phase portrait, Power spectrum, skeleton, 2D map |
| `colpitts_simple` | `colpitts_simple_oscillator(...)` | Continuous ODE | `C1`, `C2`, `beta`, `V1`, `V2` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, phase portrait, Power spectrum, skeleton, 2D map |
| `colpitts_exponential` | `colpitts_exponential_oscillator(...)` | Continuous ODE | `C1`, `C2`, `beta`, `V1`, `V2` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, phase portrait, Power spectrum, skeleton, 2D map |
| `colpitts_dynamic_beta` | `colpitts_dynamic_beta_oscillator(...)` | Continuous ODE | `C1`, `C2`, `V1`, `V2` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, phase portrait, Power spectrum, skeleton, 2D map |
| `ikeda` | `ikeda_map(...)` | Discrete map | `u`, `a`, `b` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, skeleton, basins, 2D map |
| `rossler` | `rossler_oscillator(...)` | Continuous ODE | `a`, `b`, `c` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, phase portrait, Power spectrum, skeleton, 2D map |
| `memristive_diode_bridge` | `memristive_diode_bridge(...)` | Continuous ODE | `a`, `c`, `k` | atlas, continuation, brute force, Lyapunov diagram, Codim-2 curves, phase portrait, Power spectrum, skeleton, basins, 2D map |

## Henon map

Classic dissipative quadratic map. The primary sweep is usually `a`, with `b` controlling dissipation.

Typical uses:

- period-doubling cascade;
- hidden finite-period windows;
- atlas window detection and branch recovery;
- fixed-seed 2D `(a, b)` period maps;
- codim-2 overlays on the `(a, b)` plane.

## Buck converter

Piecewise-smooth switching map for a DC-DC buck converter. Parameters include reference current and input voltage. Switching-event diagnostics help identify guard proximity.

Typical uses:

- switching-map period classification;
- border-collision diagnostics;
- basins in state space;
- 2D current/voltage maps.

## Voltage-mode buck converter

Regular-sampled voltage-mode PWM buck converter with a closed-form map. The default gain sweep shows a Neimark-Sacker-like torus birth near the gain range where complex multipliers cross the unit circle. Duty clamp proximity is reported through switching-event guards.

Typical uses:

- gain continuation and multiplier diagnostics;
- quasiperiodic/neutral Lyapunov map regions;
- duty-saturation guard interpretation.

## Peak-current boost converter

Peak-current-mode PWM boost converter with a closed-form map. Sweeping `Iref` shows the classic current-mode subharmonic period-doubling cascade; slope compensation `Sc` stabilizes the period-1 region.

Typical uses:

- subharmonic instability studies;
- `(Iref, Sc)` stabilization maps;
- switching-event diagnostics.

Basin analysis is rarely useful for this model; the default workflow focuses on the monostable cascade.

## Vilnius oscillator

Continuous ODE oscillator with a Poincare-section workflow. Supports phase portrait, brute-force, continuation, atlas, skeleton, power-spectrum, codim-2, and 2D map analyses.

Typical uses:

- ODE continuation on a native Poincare map;
- adaptive atlas discovery;
- phase-portrait inspection;
- paired time-tail / FFT spectra at fixed parameter points;
- variational derivative comparisons.

## Colpitts oscillators

Three variants are exposed:

| Variant | Constructor | Notes |
| --- | --- | --- |
| Simple | `colpitts_simple_oscillator` | Piecewise-linear transistor approximation |
| Exponential | `colpitts_exponential_oscillator` | Exponential transistor current law |
| Dynamic beta | `colpitts_dynamic_beta_oscillator` | Exponential model with current-dependent gain |

Common parameter studies sweep beta, voltage, or capacitance values. The example scripts include document-reproduction phase portraits and atlas/continuation windows.

## Ikeda map

Two-dimensional optical ring-cavity map. Parameters `a` and `b` define the phase law; `u` is the feedback/loss magnitude.

Typical uses:

- period-doubling and chaos as `u` increases;
- multistability and quasiperiodic/neutral map diagnostics;
- dense 2D `(u, b)` maps;
- basins around coexisting regimes.

## Rossler oscillator

Canonical continuous chaotic ODE. The standard cascade is usually studied by sweeping `c` while holding `a = b = 0.2`.

Typical uses:

- phase portraits at canonical chaotic parameters;
- ODE continuation with multipliers;
- 2D `(c, a)` maps;
- one- or multi-point power spectra along `c`;
- Lyapunov diagnostics on period `0` cells.

## Memristive diode bridge band-pass-filter circuit

Third-order memristive diode-bridge BPF chaotic circuit. Parameter `a` is the primary bifurcation parameter and `k` is a secondary feedback parameter. The model is known for multistability under different initial conditions.

Typical uses:

- multiseed 2D maps;
- basins of attraction;
- period and chaos windows near the reported `a` ranges;
- ODE phase portraits and Poincare-map analysis.

## Adding a built-in system

Analyzing your own system does not require modifying the package — construct a `DiscreteMap` or
`ContinuousODE` directly and pass it to any analysis; see "Adding a system in code" in
[`julia-package.md`](julia-package.md).

To contribute a new built-in system to the package:

1. Add a constructor under `src/systems/`.
2. Include and export it from `src/DynamicsKit.jl`.
3. Add system tests in `test/test_systems.jl`.

To also surface the system in the browser workbench, add a catalog spec in the separate
`DynamicsKitWorkbench.jl` package — that lives with the workbench, not here.
