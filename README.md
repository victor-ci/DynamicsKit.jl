# DynamicsKit.jl

[![CI](https://github.com/victor-ci/DynamicsKit.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/victor-ci/DynamicsKit.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Julia library for bifurcation analysis of dynamical systems — discrete maps and continuous-time
ODEs with Poincaré sections. Two complementary workflows:

- **Brute-force parameter sweeps** (threaded) for discrete maps and ODEs: 1-D bifurcation diagrams,
  basins of attraction, and 2-D parameter maps with period/status/Lyapunov diagnostics — plus
  largest-Lyapunov and full Lyapunov-spectrum sweeps, phase portraits, and power spectra.
- **Pseudo-arclength continuation** wrapping [`BifurcationKit.jl`](https://github.com/bifurcationkit/BifurcationKit.jl)
  (`PALC()`): periodic-branch continuation by return-map shooting or orthogonal collocation,
  map-aware period-doubling/fold detection, adaptive branch refinement, and codimension-2 curve
  tracing — including a higher-level **atlas** that combines reconnaissance sweeps with targeted
  skeleton seeding to map periodic branches automatically.

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
