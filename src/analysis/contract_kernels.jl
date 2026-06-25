"""
Contract B — Analysis kernels. Stable public API for the per-cell computation routines the
workbench's grid-cache engine drives directly (resolve initial state, sample orbits, detect period,
estimate Lyapunov, record into storage, run the map kernel / adaptive refinement). See
`docs/internal/contracts/contract-b-analysis-kernels.md`.

Mechanism: alias-direction (same as Contract C) — the definitions stay in their home files
(`brute_force.jl`, `lyapunov.jl`, `atlas.jl`); here we publish `const <public> = <_internal>`
bindings (with docstrings). The matching `export` statements live in `src/DynamicsKit.jl`
(see the "Exports — analysis kernels (Contract B)" block). This file is `include`d after those
home files. The file-decomposition phase relocates the bodies later.

Design note resolved from the code: `bifurcation_map_kernel` is published (not folded into the
existing public `bifurcation_map`) because `_bifurcation_map` returns `(result, diagnostics)` and
the public `bifurcation_map` deliberately discards the diagnostics; the workbench's cache layer
needs both. Changing `bifurcation_map`'s return would break its existing contract, so the kernel is
exposed under its own name instead.
"""

# --- Group 1: initial-state resolution ---
"""    resolve_initial_state(sys::ContinuousODE, initial_point) -> Vector{Float64}  — full ODE state (uses section template when nothing)."""
const resolve_initial_state = _resolve_initial_state

# --- Group 2: orbit sampling ---
"""    sample_discrete_orbit(sys::DiscreteMap, params; iterations, transient=0, amplitude_cutoff=500.0, initial_point=nothing)"""
const sample_discrete_orbit = _sample_discrete_orbit
"""    sample_continuous_poincare_orbit(sys::ContinuousODE, params; crossings, transient=0, solver, reltol, abstol, …)"""
const sample_continuous_poincare_orbit = _sample_continuous_poincare_orbit
"""    collect_poincare_points(sys::ContinuousODE, params; crossings, transient=0, solver, reltol, abstol, …) -> Vector{<:AbstractVector}"""
const collect_poincare_points = _collect_poincare_points

# --- Group 3: period detection (returns the detection NamedTuple; `detect_period` returns just Int) ---
"""    detect_period(orbit, max_period, precision) -> Int"""
const detect_period = _detect_period
"""    detect_discrete_map_period(sys::DiscreteMap, params, initial_point::SVector, transient, max_period, precision, divergence_cutoff) -> detection NT"""
const detect_discrete_map_period = _detect_discrete_map_period
"""    detect_continuous_poincare_period(sys::ContinuousODE, params; transient, max_period, precision, solver, reltol, abstol, …) -> detection NT"""
const detect_continuous_poincare_period = _detect_continuous_poincare_period

# --- Group 4: largest-Lyapunov estimation (returns (exponent, estimation_status, sample_count)) ---
"""    estimate_discrete_map_largest_lyapunov(sys::DiscreteMap, params, initial_point::SVector, transient, steps, perturbation, divergence_cutoff)"""
const estimate_discrete_map_largest_lyapunov = _estimate_discrete_map_largest_lyapunov
"""    estimate_continuous_poincare_largest_lyapunov(sys::ContinuousODE, params, initial_state, transient, steps, perturbation, divergence_cutoff; solver, reltol, abstol, …)"""
const estimate_continuous_poincare_largest_lyapunov = _estimate_continuous_poincare_largest_lyapunov

# Groups 5–8 (per-cell Lyapunov recorders, `bifurcation_map_kernel` /
# `map_adaptive_refinement_diagnostics`, direct-Lyapunov-field validate/record, and
# `atlas_hidden_period_sample_indices`) reverted to underscore-private under Contract D: the
# workbench now drives caching through the public `bifurcation_map` / `lyapunov_field` /
# `basins_of_attraction` sweeps (`cells=` hook), so it no longer reaches these kernels directly.
# See docs/internal/contracts/contract-d-sweep-cache-hook.md.
