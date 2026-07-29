# Validation scripts

Executable, self-contained Julia scripts that check specific, named claims
about DynamicsKit against a closed-form analytic result, an independent
reference implementation already shipped in this repository (e.g. the
hand-coded `buck_converter()` vs. the generator-based
`buck_converter_description()`), or a cited external literature source. This
is different from `examples/` (cookbook demos showing how to use the library)
and from `bench/` (wall-clock performance measurement): every script here
reports a pass/fail verdict for each claim it checks.

## Running a script

```sh
julia --project=. docs/validation/<script>.jl
```

Exit code `0` means every required gate passed. A nonzero exit code means at
least one required gate failed; the printed summary lists which one(s) and
why.

## Conventions

- Each script starts with a docstring: what it checks, the analytic/reference
  basis for each claim (with a full literature citation where one applies),
  and the run command.
- `include(joinpath(@__DIR__, "common.jl"))` pulls in the shared
  `ValidationGate`/`record_gate!`/`conclude` helpers (see `common.jl`) so every
  script reports results in the same PASS/FAIL/NOTE format and exits
  consistently.
- A gate is **required** by default (`record_gate!(gates, name, passed, detail)`)
  — any required failure makes `conclude` exit nonzero. Pass
  `required=false` for context that cannot be reduced to a strict pass/fail
  (a cost comparison with no fixed target, or a cross-tool check pending a
  reference artifact that is not available yet); informational gates are
  printed but never fail the run.
- Prefer the public API (`DynamicsKit.<exported name>`) over internal
  (underscore-prefixed) functions wherever the claim can be checked that way —
  these scripts double as runnable, dogfooded usage examples of the public
  surface, not just correctness checks.
- Where a script scales down a claim for reasonable runtime (e.g. a coarser
  grid than a full high-resolution operating-map sweep), the docstring says so
  explicitly and states what is/isn't preserved at the smaller scale.

## Scripts

| Script | Checks |
| --- | --- |
| `switching_maps.jl` | n-dimensional affine-flow exactness against `exp()`; the switching-map generator reproduces the independently hand-coded buck/boost maps; the Cuk/SEPIC converters' first period-doubling point against their literature sources |
| `robust_chaos_region_certification.jl` | Two-parameter region certification on an analytic striped-map fixture (coverage, exclusion, budget monotonicity, determinism, serialization) and physical buck-converter grounding (a robust vs. a fragile design) |
| `cycle_to_cycle_connections.jl` | Cycle-to-cycle connecting-orbit continuation against an analytic rotating-front fixture, automatic seed discovery, and refusal of degenerate/attracting-only inputs |
| `transverse_exponent_field.jl` | The variational transverse-exponent field against a closed-form analytic exponent, and its spurious-positive-rate improvement over the two-trajectory estimator on the Vilnius oscillator (Ipatovs et al. 2023) |
