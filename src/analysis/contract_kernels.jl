"""
Public analysis kernels: the per-cell computation routines external cache/scripting layers drive
directly (resolve initial state, sample orbits, detect period, estimate Lyapunov, run the map
kernel).

The definitions stay in their home files (`brute_force.jl`, `lyapunov.jl`, `atlas.jl`); this file
publishes them as `const <public> = <_internal>` bindings with docstrings and is included after
those home files. The matching `export` statements live in `src/DynamicsKit.jl`.
"""

# --- initial-state resolution ---
"""    resolve_initial_state(sys::ContinuousODE, initial_point) -> Vector{Float64}  — full ODE state (uses section template when nothing)."""
const resolve_initial_state = _resolve_initial_state

# --- orbit sampling ---
"""    sample_discrete_orbit(sys::DiscreteMap, params; iterations, transient=0, amplitude_cutoff=500.0, initial_point=nothing)"""
const sample_discrete_orbit = _sample_discrete_orbit
"""    sample_continuous_poincare_orbit(sys::ContinuousODE, params; crossings, transient=0, solver, reltol, abstol, …)"""
const sample_continuous_poincare_orbit = _sample_continuous_poincare_orbit
"""    collect_poincare_points(sys::ContinuousODE, params; crossings, transient=0, solver, reltol, abstol, …) -> Vector{<:AbstractVector}"""
const collect_poincare_points = _collect_poincare_points

# --- period detection (returns the detection NamedTuple; `detect_period` returns just Int) ---
"""    detect_period(orbit, max_period, precision) -> Int"""
const detect_period = _detect_period
"""    detect_discrete_map_period(sys::DiscreteMap, params, initial_point::SVector, transient, max_period, precision, divergence_cutoff) -> detection NT"""
const detect_discrete_map_period = _detect_discrete_map_period
"""    detect_continuous_poincare_period(sys::ContinuousODE, params; transient, max_period, precision, solver, reltol, abstol, …) -> detection NT"""
const detect_continuous_poincare_period = _detect_continuous_poincare_period

# --- largest-Lyapunov estimation (returns (exponent, estimation_status, sample_count)) ---
"""    estimate_discrete_map_largest_lyapunov(sys::DiscreteMap, params, initial_point::SVector, transient, steps, perturbation, divergence_cutoff)"""
const estimate_discrete_map_largest_lyapunov = _estimate_discrete_map_largest_lyapunov
"""    estimate_continuous_poincare_largest_lyapunov(sys::ContinuousODE, params, initial_state, transient, steps, perturbation, divergence_cutoff; solver, reltol, abstol, …)"""
const estimate_continuous_poincare_largest_lyapunov = _estimate_continuous_poincare_largest_lyapunov

# The per-cell Lyapunov recorders, `map_adaptive_refinement_diagnostics`, and the
# direct-Lyapunov-field validate/record helpers stay underscore-private: consumers drive the
# `lyapunov_field` / `basins_of_attraction` sweeps through their public `cells=` hook and do not
# reach those kernels directly.

# --- 2D-map kernel ---
# `bifurcation_map_kernel` is published (not folded into the public `bifurcation_map`) because the
# public sweep deliberately discards diagnostics and has no `cells=` hook, while the cache layer
# needs both. `lyapunov_field` / `basins_of_attraction` expose `cells=` directly, so only the map
# kernel needs a separate tuple-returning entry point.
"""
    bifurcation_map_kernel(sys, config::BifurcationMapConfig; initial_point=nothing,
                           cells=nothing[, solver, reltol, abstol]) -> (BifurcationMapResult, Dict)

Per-cell 2D bifurcation-map kernel. Returns the result **and** the diagnostics dict that the public
`bifurcation_map` discards, and accepts a pre-seeded `cells::MapCellGrid` so a cache layer can
compute only the unknown cells in place.
"""
const bifurcation_map_kernel = _bifurcation_map

# --- atlas hidden-period sampling ---
"""    atlas_hidden_period_sample_indices(...) — sample indices for atlas hidden-period recovery."""
const atlas_hidden_period_sample_indices = _atlas_hidden_period_sample_indices
