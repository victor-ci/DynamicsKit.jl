# DynamicsKit.jl

A Julia library for bifurcation analysis of dynamical systems — discrete maps and continuous-time
ODEs with Poincaré sections. Two complementary workflows:

- **Brute-force parameter sweeps** (threaded) for discrete maps and ODEs: 1-D diagrams, basins of
  attraction, and 2-D parameter maps with period/status/Lyapunov diagnostics.
- **Pseudo-arclength continuation** wrapping [`BifurcationKit.jl`](https://github.com/bifurcationkit/BifurcationKit.jl)
  (`PALC()`), including a higher-level **atlas** that combines reconnaissance sweeps with targeted
  skeleton seeding to map periodic branches automatically.

```julia
using DynamicsKit

sys = henon_map()
diagram = brute_force_diagram(sys, BruteForceConfig(param_min=0.0, param_max=1.4, param_steps=400,
                                                    iterations=400, transient=300))
branch  = continuation_branch(sys, ContinuationConfig(p_min=0.0, p_max=1.4))
```

## Install

Not registered yet — add by URL/path:

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

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the suite on every push to `main` and every pull request.

## Related

The interactive browser **workbench** lives in the separate `DynamicsKitWorkbench.jl` package, which
depends on this library. This library never depends on the workbench.

## License / authorship

© Victor Iheanacho.
