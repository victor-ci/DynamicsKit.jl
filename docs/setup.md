# Setup and local development

## Requirements

- Julia `1.11` or newer.
- Git.

The library depends on BifurcationKit.jl, DifferentialEquations.jl, ForwardDiff.jl, StaticArrays.jl,
JLD2.jl, FFTW.jl, Plots.jl, and supporting utility packages.

## Install package dependencies

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` is not committed (library policy — consumers resolve against compat bounds). If Julia
reports that the project dependencies or compat requirements changed since the manifest was resolved,
refresh the manifest with Julia rather than editing it by hand:

```sh
julia --project=. -e 'using Pkg; Pkg.resolve()'
```

## Threading

Julia fixes the thread count at process startup. To let analyses use multiple threads, start Julia
with more than one thread:

```sh
julia --threads auto,2 --project=. examples/henon_complete.jl
```

or:

```sh
JULIA_NUM_THREADS=8 julia --project=. -e 'using DynamicsKit; ...'
```

Threading is most useful for:

- brute-force parameter sweeps;
- skeleton searches over many seeds;
- basins grids;
- fixed-seed 2D parameter maps;
- tiled neighbor-seeded maps;
- atlas reconnaissance.

Continuation itself remains more sequential because each branch follows a pseudo-arclength path.

## Tests

Run the full Julia test suite (includes an Aqua.jl package-quality pass):

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
julia --project=. -e 'using Pkg; Pkg.test(test_args=["public-api"])'
```

Other targets: `parameter-mapping`, `accessors-contract`, `kernels-contract`, `cache-hook`,
`lyapunov`, `codim2`, `spectrum`.

When touching threaded sweep or atlas code, run with multiple threads:

```sh
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the suite on every push to `main` and every pull request (matrix over Julia 1.11 and the
latest 1.x); see `.github/workflows/CI.yml`.
