# Benchmarks and performance comparisons

Benchmarks in this project are designed to compare scientific modes and workflow features, not just raw function speed. Use them to answer questions like:

- Does cache reuse return exactly the same result as a fresh run?
- Does threading help this workload?
- How much coverage does targeted continuation reseeding recover?
- How do neighbor-seeded map modes trade speed for path dependence?
- Does PALC tuning reduce steps without losing branch coverage?

This page documents how to run and interpret the benchmark scripts. It intentionally avoids claiming canonical benchmark results; record machine-specific numbers only after dedicated benchmark runs.

## General benchmark hygiene

Run from the repository root with a fixed Julia environment:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

For threaded comparisons, explicitly choose the thread count:

```sh
JULIA_NUM_THREADS=1 julia --project=. bench/grid_cache_benchmark.jl
JULIA_NUM_THREADS=4 julia --project=. bench/grid_cache_benchmark.jl
```

Recommended practice:

1. Run once to compile.
2. Rerun and compare the second output.
3. Record Julia version, thread count, system, grid size, solver tolerances, and cache settings.
4. Prefer relative comparisons on the same machine over absolute timing claims.

## Cache benchmarks

Script:

```sh
julia --project=. bench/grid_cache_benchmark.jl
```

What it compares:

| Case | Meaning |
| --- | --- |
| Coarse run | Populate cache at a smaller grid |
| Fine after coarse cache | Reuse compatible samples/cells where possible |
| Fresh fine | Disable grid cache and compute the fine grid |
| Coarse after fine cache | Reuse fine-grid samples to satisfy a coarser request |

Covered analyses:

- 1D brute-force sample cache;
- 2D bifurcation-map grid cache;
- basins grid cache.

Metrics printed:

| Metric | Meaning |
| --- | --- |
| `runtime_ms` | End-to-end workbench analysis runtime |
| `reused` | Reused samples/cells |
| `computed` | Newly computed samples/cells |
| `requested` | Total requested samples/cells |
| `disabledReason` | Why cache reuse was disabled, if applicable |
| `equalToFresh` | Whether cached result matches the fresh comparison |

Interpretation:

- A valid cache speedup must preserve equality with the fresh result.
- Exact pointwise grid cache is valid for fixed-seed maps.
- Path-dependent neighbor maps and multiseed maps disable grid-cache reuse by design.
- Whole-session result cache is separate from pointwise grid cache.

## Continuation reseed and PALC tuning benchmark

Script:

```sh
julia --threads auto --project=. bench/reseed_benchmark.jl
```

Fast-only mode:

```sh
BENCH_SKIP_COLPITTS=1 julia --project=. bench/reseed_benchmark.jl
```

What it compares:

| Comparison | Expected interpretation |
| --- | --- |
| Reseed off vs on with forced interior death | Reseed should recover much more parameter coverage |
| Reseed off vs on when branch reaches boundary | Reseed should be a no-op with little overhead |
| Conservative vs aggressive PALC settings | Larger `a`/`dsmax` can reduce point count and runtime if coverage is preserved |

Printed columns:

| Column | Meaning |
| --- | --- |
| `time(s)` | Minimum wall time over repeated samples |
| `points` | Recorded branch points |
| `coverage` | Fraction of requested parameter interval spanned |
| `reseeds` | Number of accepted reseed attempts |

Use this benchmark before changing continuation defaults, reseeding logic, or atlas continuation budgets.

## Neighbor-seeded 2D map benchmark

Script:

```sh
julia --threads auto --project=. bench/neighbor_seed_acceleration.jl
```

Environment controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYSTEM` | `memristive_diode_bridge` | One of `ikeda`, `rossler`, `memristive_diode_bridge` |
| `GRID_STEPS` | `50` | Produces a `(GRID_STEPS + 1)^2` grid |
| `NEIGHBOR_TRANSIENTS` | `0,2,5,10,20` | Accelerated transient values to compare |
| `NEIGHBOR_TILE_SIZE_A` | `0` | Optional tile size along first parameter axis |
| `NEIGHBOR_TILE_SIZE_B` | `0` | Optional tile size along second parameter axis |
| `SAVE_RESULTS` | `false` | Write JSON summary if true |
| `OUTPUT_DIR` | empty | Directory for JSON summary; also enables saving |

Example:

```sh
SYSTEM=ikeda GRID_STEPS=80 NEIGHBOR_TRANSIENTS=0,5,20 \
JULIA_NUM_THREADS=8 julia --project=. bench/neighbor_seed_acceleration.jl
```

What it compares:

| Mode | Meaning |
| --- | --- |
| `fixed` | Pointwise fixed-initial-condition classification |
| `neighbor_full` | Path-following traversal with full transient |
| `neighbor_accelerated` | Path-following traversal with reduced transient |
| `neighbor_accelerated_tiled` | Path-following traversal reset at deterministic tile boundaries |

Metrics:

| Column | Meaning |
| --- | --- |
| `runtime_s` | Runtime for the map |
| `speedup_vs_fixed` | Fixed runtime divided by current runtime |
| `mismatch_count` / `mismatch_fraction` | Difference from fixed-seed classification |
| `mismatch_vs_neighbor_full` | Difference from full-transient path-following baseline |
| `unique_periods` | Period labels present in the matrix |
| `resets`, `invalid_resets`, `tile_count` | Traversal diagnostics |

Interpretation:

- Speedups in neighbor-accelerated mode are exploratory wins only if the mismatch rate is acceptable for the question.
- Mismatch with fixed seed can indicate real path dependence/hysteresis, not merely numerical error.
- Tiling improves parallelism and determinism but changes the path at tile boundaries.

## Scientific diagnostics workflow benchmark

Script:

```sh
julia --project=. bench/scientific_diagnostics_benchmark.jl
```

Environment controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DIAGNOSTICS_BENCH_REPEATS` | `1` | Number of repeated diagnostic passes |
| `SAVE_RESULTS` | `false` | Write JSON summary if true |
| `OUTPUT_DIR` | empty | Directory for JSON summary; also enables saving |

What it runs:

| Case | Diagnostic focus |
| --- | --- |
| Henon period doubling/window | Discrete-map period classification and hidden periodic window detection |
| Ikeda multistability/Lyapunov | Multiseed coexistence payloads and largest-Lyapunov status |
| Rossler continuation multipliers | ODE continuation with variational multiplier diagnostics |
| Boost switching guards | Nonsmooth guard-distance diagnostics |
| Memristive diode bridge map | Continuous multistability map and crossing diagnostics |

This benchmark is a reproducibility harness, not a canonical performance table. Use it to confirm the diagnostic workflows still execute and to collect machine-local timing rows when comparing solver settings, cache modes, thread counts, or analysis implementations.

## Threading comparisons

Threading benefits are workload-dependent.

Use a paired command pattern:

```sh
JULIA_NUM_THREADS=1 julia --project=. bench/grid_cache_benchmark.jl
JULIA_NUM_THREADS=4 julia --project=. bench/grid_cache_benchmark.jl
```

Analyses most likely to benefit:

| Analysis | Why |
| --- | --- |
| Brute-force diagram | Independent parameter samples |
| Skeleton | Independent initial seeds |
| Basins | Independent initial-condition cells |
| Fixed-seed 2D map | Independent parameter cells |
| Atlas reconnaissance | Independent reconnaissance samples |
| Tiled neighbor maps | Independent tiles with local traversal |

Analyses less likely to scale:

| Analysis | Why |
| --- | --- |
| Single continuation branch | Pseudo-arclength path is sequential |
| Very small grids | Compilation and overhead dominate |
| Stiff ODE maps | Solver cost and adaptivity may dominate |

## Workbench cache and benchmark knobs

Workbench payloads support cache controls such as:

| Key | Meaning |
| --- | --- |
| `fileCache` | Enables/disables whole-session result cache |
| `bruteForceGridCache` | Enables/disables brute-force sample cache |
| `basinsGridCache` | Enables/disables basins grid cache |
| `mapGridCache` | Enables/disables 2D map grid cache |
| `cacheSalt` | Optional salt to intentionally separate benchmark runs |

Use `cacheSalt` when you want independent benchmark families without clearing all local artifacts.

## Reporting benchmark results

A useful benchmark report should include:

- command and environment variables;
- Julia version and thread count;
- system and parameters;
- grid sizes or continuation budgets;
- cache settings;
- runtime and reuse metrics;
- equality/mismatch checks;
- interpretation of scientific differences, not only speed.
