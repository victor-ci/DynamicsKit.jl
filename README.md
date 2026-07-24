# DynamicsKit.jl

[![CI](https://github.com/victor-ci/DynamicsKit.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/victor-ci/DynamicsKit.jl/actions/workflows/CI.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21327845-blue.svg)](https://doi.org/10.5281/zenodo.21327845)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Julia library for bifurcation analysis of discrete maps and continuous-time ODEs, combining direct
simulation, numerical continuation, and evidence-based robustness analysis. Continuous-time
workflows use [Poincaré sections and return maps](docs/julia-package.md#continuous-time-systems) for
sweeps and shooting-based continuation, with orthogonal collocation as an alternative.

| Area | Core capabilities |
| --- | --- |
| **Parameter-space exploration** | Threaded 1-D sweeps; basins of attraction; uniform and adaptive 2-D period, status, and Lyapunov maps; Lyapunov spectra, phase portraits, and power spectra. |
| **Continuation and local bifurcations** | [`BifurcationKit.jl`](https://github.com/bifurcationkit/BifurcationKit.jl) pseudo-arclength continuation; periodic-orbit shooting and collocation; branch refinement; codimension-2 curves and map normal forms; [connecting orbits](docs/julia-package.md#connecting-orbit-continuation-homoclinic--heteroclinic--saddle-cycle); and [border-collision diagnostics](docs/julia-package.md#border-collision-classification). |
| **Automatic structure discovery** | Periodic-skeleton searches and an atlas combining reconnaissance sweeps with targeted seeding and recovery to map periodic branches and windows. |
| **Robustness and design** | Multistability-aware branch reachability; regime-boundary and tolerance maps; layered robust-chaos evidence; and [inverse chaos-source design](docs/julia-package.md#chaos-source-inverse-design). |

[GPU acceleration](docs/julia-package.md#optional-gpu-acceleration) is available through CUDA.jl or
AMDGPU.jl for eligible cell-independent discrete-map sweeps and independent-trajectory ODE
bifurcation maps and basin sweeps. Continuous-time Lyapunov analyses and continuation remain CPU-only;
Metal is unavailable because its lack of Float64 support is incompatible with the validated
classification tolerances.

```julia
using DynamicsKit

sys = henon_map()
diagram = brute_force_diagram(sys, BruteForceConfig(param_min=0.0, param_max=1.4, param_steps=400,
                                                    iterations=400, transient=300))
branch  = continuation_branch(sys, ContinuationConfig(p_min=0.0, p_max=1.4))
```

## Install

Registered in the Julia General registry:

```julia
using Pkg; Pkg.add("DynamicsKit")
```

Or track the development version directly from the repository:

```julia
using Pkg; Pkg.add(url="https://github.com/victor-ci/DynamicsKit.jl")
```

Requires Julia 1.11+.

## Documentation

See [`docs/`](docs/README.md):

- [`docs/setup.md`](docs/setup.md) — install, threading, tests
- [`docs/julia-package.md`](docs/julia-package.md) — exported API, configs, result types, examples
- [`docs/analysis-methods.md`](docs/analysis-methods.md) — choosing among the analyses
- [`docs/scientific-interpretation.md`](docs/scientific-interpretation.md) — periods, status codes, multipliers, Lyapunov, multistability
- [`docs/systems-catalog.md`](docs/systems-catalog.md) — built-in systems and parameters
- [`docs/examples.md`](docs/examples.md) — example scripts and cookbook snippets
- [`docs/benchmarks.md`](docs/benchmarks.md) — benchmark commands, metrics, and reporting guidance
- [`docs/validation.md`](docs/validation.md) — regression targets, quality gates, and validation practices

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the suite on every push to `main` and every pull request.

## Related

The interactive browser **workbench** lives in the separate `DynamicsKitWorkbench.jl` package, which
depends on this library. This library never depends on the workbench.

## License

[MIT](LICENSE) © Victor Iheanacho.
