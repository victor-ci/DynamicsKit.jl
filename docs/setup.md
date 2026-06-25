# Setup and local development

## Requirements

- Julia `1.11` or newer.
- Git.
- Node.js and npm only if you edit or validate the browser frontend.
- Docker Desktop or another Docker/Compose runtime only if you prefer the container workflow.

The Julia package depends on BifurcationKit.jl, DifferentialEquations.jl, ForwardDiff.jl, StaticArrays.jl, JLD2.jl, Plots.jl, HTTP.jl, JSON3.jl, and supporting utility packages.

## Install package dependencies

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

If Julia reports that the project dependencies or compat requirements changed since the manifest was resolved, refresh the manifest with Julia rather than editing it by hand:

```sh
julia --project=. -e 'using Pkg; Pkg.resolve()'
```

## Optional frontend tooling

The browser workbench serves `src/ui/assets/app.js`, which is built from `src/ui/frontend/app.ts`.

```sh
npm install
npm run check:ui
npm run check:ui:tests
npm run lint:ui
npm run test:ui
npm run build:ui
```

Use the combined fast gate when changing frontend source:

```sh
npm run verify:ui
```

Use the CI-style freshness check when you need to confirm that the generated bundle was committed after a rebuild:

```sh
npm run verify:ui:ci
```

## Launch the workbench

```sh
julia --threads auto,2 --project=. examples/workbench.jl
```

The default address is `http://127.0.0.1:9384`. The launcher accepts:

| Environment variable | Purpose |
| --- | --- |
| `BIFURCATION_EXPLORER_HOST` | Host passed to `launch_workbench` |
| `BIFURCATION_EXPLORER_PORT` | Port passed to `launch_workbench` |
| `BIFURCATION_EXPLORER_OPEN_BROWSER` | Set to `false` to avoid opening a browser automatically |

## Threading

Julia fixes the thread count at process startup. To let package analyses and workbench runs use multiple threads, start Julia with more than one thread:

```sh
julia --threads auto,2 --project=. examples/workbench.jl
```

or:

```sh
JULIA_NUM_THREADS=8 julia --project=. examples/workbench.jl
```

The workbench "Enable threaded execution" checkbox controls whether eligible analyses use the threads already granted to the Julia process. It cannot create new Julia threads after startup.

Threading is most useful for:

- brute-force parameter sweeps;
- skeleton searches over many seeds;
- basins grids;
- fixed-seed 2D parameter maps;
- tiled neighbor-seeded maps;
- atlas reconnaissance and preview discovery.

Continuation itself remains more sequential because each branch follows a pseudo-arclength path.

## Runtime artifacts

By default, the workbench writes runtime data under `var/`:

| Artifact | Default path | Override |
| --- | --- | --- |
| Session/result cache | `var/output/workbench_results/` | `BIFURCATIONEXPLORER_WORKBENCH_RESULT_DIR` |
| Per-run logs | `var/output/workbench_logs/` | `BIFURCATIONEXPLORER_WORKBENCH_LOG_DIR` |
| Skeleton cache | `var/cache/` | `BIFURCATIONEXPLORER_WORKBENCH_SKELETON_CACHE_DIR` |
| Grid/sample cache | `var/cache/grid_results/` | `BIFURCATIONEXPLORER_WORKBENCH_GRID_CACHE_DIR` |
| Shared root | `var/` | `BIFURCATIONEXPLORER_WORKBENCH_VAR_DIR` |

These artifacts are intended to be local runtime outputs, not source files.

## Tests

Run the full Julia test suite:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Focused targets are useful during iteration:

```sh
julia --project=. -e 'using Pkg; Pkg.test(test_args=["quality"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["systems"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["brute-force"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["continuation"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["skeleton"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["atlas"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["basins-map-refine"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["workbench"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["workbench-catalog"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["workbench-cache"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["workbench-preview"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["workbench-sessions"])'
julia --project=. -e 'using Pkg; Pkg.test(test_args=["workbench-slow"])'
```

The `workbench-slow` target contains heavier preset-integration and scientific-smoke runs and is not part of the default suite.

When touching threaded atlas or preview code, run the focused workbench target with multiple threads:

```sh
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test(test_args=["workbench"])'
```

## Docker workflow

Build the image:

```sh
docker compose build
```

Run tests:

```sh
docker compose run --rm test
```

Open a Julia REPL:

```sh
docker compose run --rm julia
```

Launch the workbench:

```sh
docker compose up workbench
```

The workbench service binds inside the container to `0.0.0.0`, exposes `127.0.0.1:9384` on the host, sets `JULIA_NUM_THREADS=auto`, and disables browser auto-open.
