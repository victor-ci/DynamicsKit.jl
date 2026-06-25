# DynamicsKit documentation

This directory is the public documentation catalog for **DynamicsKit.jl** — the Julia library for
scripted bifurcation analysis: parameter sweeps, continuation, atlas discovery, plotting, saving,
loading, and reproducible experiments.

> The interactive browser workbench is a **separate package**, `DynamicsKitWorkbench.jl`, which
> depends on this library. UI/HTTP/frontend instructions live in that repository, not here.

## Documentation map

| Guide | Use it for |
| --- | --- |
| [`setup.md`](setup.md) | Installing Julia dependencies, threading, Docker, tests, and artifact paths |
| [`julia-package.md`](julia-package.md) | Calling the library directly from Julia scripts and notebooks |
| [`analysis-methods.md`](analysis-methods.md) | Choosing among brute force, continuation, atlas, skeleton, basins, phase portrait, 2D maps, and refinement |
| [`scientific-interpretation.md`](scientific-interpretation.md) | Interpreting periods, status codes, multipliers, residuals, Lyapunov diagnostics, multistability, and switching events |
| [`systems-catalog.md`](systems-catalog.md) | Built-in maps, ODEs, circuit models, and parameters |
| [`examples.md`](examples.md) | Example scripts plus cookbook snippets for Henon, Ikeda, Rossler, Colpitts, converters, and memristive diode bridge studies |
| [`benchmarks.md`](benchmarks.md) | Benchmark commands, metrics, and reporting guidance for cache behavior, threading, reseeding, PALC tuning, and neighbor-seeded maps |
| [`validation.md`](validation.md) | Regression targets, quality gates, and validation practices |

## End-to-end workflow

```mermaid
flowchart LR
    A[Choose or define a system] --> B[Set base parameters]
    B --> C{Question}
    C -->|Observed attractors| D[Brute-force sweep]
    C -->|Periodic branch geometry| E[Continuation]
    C -->|Unknown windows| F[Atlas]
    C -->|Initial-condition dependence| G[Basins]
    C -->|Two-parameter structure| H[2D map]
    C -->|Continuous trajectory| I[Phase portrait]
    D --> J[Compare with continuation]
    E --> J
    F --> J
    G --> K[Multistability interpretation]
    H --> K
    I --> L[Poincare-section interpretation]
    J --> M[Diagnostics and validation]
    K --> M
    L --> M
```

## Glossary

| Term | Meaning |
| --- | --- |
| Brute-force diagram | Parameter sweep that records post-transient attractor samples. It shows what a trajectory reaches from the selected initial condition. |
| Continuation branch | Curve of periodic solutions traced by pseudo-arclength continuation. It can include unstable or non-attracting orbits that a brute-force sweep will not show. |
| Skeleton | Periodic-orbit seeds found by Newton iteration from a grid or seed list. Skeletons feed continuation and atlas recovery. |
| Atlas | Automated pipeline: reconnaissance sweep, window segmentation, skeleton seeding, continuation recovery, gap retry, and optional branch switching. |
| Poincare map | Return map produced by sampling a continuous ODE each time it crosses a section. Continuation and period detection for ODEs operate on this map. |
| Period `0` | Backward-compatible "no finite period detected" value. Diagnostics explain whether this means high period, aperiodic/chaotic candidate, divergence, insufficient crossings, or solver failure. |
