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

## CUDA operating-map benchmark

Script:

```sh
julia --project=bench -e 'using Pkg; Pkg.instantiate()'
JULIA_NUM_THREADS=4 julia --project=bench bench/gpu_operating_maps.jl
```

This benchmark requires a functional NVIDIA CUDA device. It warms up both backends, requires CPU/CUDA
parity, verifies that every accelerated result reports `compute_backend == :cuda`, and measures five
paths: Hénon `bifurcation_map`, `basins_of_attraction`, and `lyapunov_field`, plus continuous
memristive-diode-bridge P1/P3 parameter-map and coexistence-basin sweeps. Periodicity matrices must
match exactly; Lyapunov exponents, classification/estimation codes, and sample counts must match.

Environment controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_BENCH_RESOLUTIONS` | `64,128,256,512` | Comma-separated discrete square-grid side lengths |
| `GPU_BENCH_CONTINUOUS_RESOLUTIONS` | `4,8,16` | Comma-separated continuous square-grid side lengths |
| `GPU_BENCH_REPEATS` | `3` | Timed repetitions after warmup |
| `GPU_BENCH_CONTINUOUS_REPEATS` | `2` | Timed continuous repetitions after warmup |
| `GPU_BENCH_ITERATIONS` | `500` | Iterations per grid cell |
| `GPU_BENCH_LYAPUNOV_ITERATIONS` | `500` | Lyapunov iterations per discrete cell |
| `GPU_BENCH_CONTINUOUS_ITERATIONS` | `250` | Poincaré crossings requested per continuous cell |
| `GPU_BENCH_CUDA_HEAP_MB` | `256` | CUDA device malloc heap explicitly recorded by the benchmark process |
| `GPU_BENCH_OUT` | `var/output/gpu_operating_maps.jld2` | JLD2 output path; a CSV is written alongside it |

The printed and saved metrics include minimum and median CPU/GPU wall time, speedup, and GPU cell
throughput. Small grids are expected to be dominated by kernel-launch and transfer overhead; use the
multi-resolution trend to identify the crossover rather than treating every grid as an expected
speedup. The benchmark environment carries CUDA as a benchmark-only dependency, so the library's
runtime dependency surface remains unchanged.

The library's CUDA extension automatically ensures a 256 MiB minimum device malloc heap before a
continuous ensemble launch (`DYNAMICSKIT_CUDA_HEAP_MB` overrides it). The benchmark also sets and
records its requested value explicitly so the evidence does not depend on prior context state.
DiffEqGPU's adaptive `EnsembleGPUKernel` constructs mutable per-trajectory integrators in CUDA dynamic
memory; increasing the heap prevents the driver's 8 MiB default from invalidating repeated continuous
measurements, but does not make the continuous solver faster or remove its per-trajectory allocation.
Treat this as an upstream/runtime limitation, not as a reason to weaken tolerances or switch production
science to Float32.

### RTX 4090 reference run

Recorded on 2026-07-24 with Julia 1.11.9, CUDA.jl 6.2.1, four Julia threads, an NVIDIA RTX 4090
(`sm_89`), the default benchmark settings above, and exact parity passing on all 18 rows. The artifact
is `var/output/gpu_operating_maps_rtx4090_complete.{jld2,csv}`.

| Workload | Grid | Median CPU | Median CUDA | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Hénon bifurcation map | `256²` | 0.0264 s | 0.0140 s | 1.88× |
| Hénon bifurcation map | `512²` | 0.1068 s | 0.0436 s | 2.45× |
| Hénon basins | `512²` | 0.1609 s | 0.0385 s | 4.18× |
| Hénon Lyapunov field | `256²` | 0.1204 s | 0.0724 s | 1.66× |
| Hénon Lyapunov field | `512²` | 0.4817 s | 0.0315 s | 15.28× |
| MDB parameter map | `16²` | 4.3212 s | 6.9012 s | 0.63× |
| MDB coexistence basins | `16²` | 1.5259 s | 5.4337 s | 0.28× |

These numbers support a scoped claim: discrete CUDA acceleration becomes useful after launch overhead
is amortized and is substantial for large Lyapunov fields. They do **not** support a blanket
continuous-ODE speedup claim on a consumer NVIDIA GPU. The RTX 4090 reports a 64:1
single-to-double-precision throughput ratio; adaptive independent trajectories, directional root
finding, branch divergence, and the augmented 17-scalar MDB detector state compound that FP64
disadvantage. Continuous CUDA is validated for correctness here, but the four-thread CPU path remains
faster for the measured MDB grids.

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
| `runtime_ms` | End-to-end analysis runtime |
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
