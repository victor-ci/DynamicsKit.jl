"""
Public result & diagnostics accessors: config-derived ("effective") settings, diagnostics
producers, result extraction, and continuation branch post-processing.

The definitions stay in their home files (`brute_force.jl`, `continuation.jl`) because they are
tightly coupled to sibling internals there; this file only publishes them as
`const <public> = <_internal>` bindings with docstrings. The matching `export` statements live in
`src/DynamicsKit.jl`, and this file is included after the home files so the underscore names exist.
Transitive helpers (`_map_orbit_window`, `_crossing_diag_*`, `_finite_matrix_extrema`,
`_matrix_label_counts`, `_map_seed_semantics_label`, `_map_tile_count`) stay private — only the
published functions call them.
"""

# --- 2D-map effective settings (config-derived) ---
"""
    map_effective_settings(config::BifurcationMapConfig;
                           na=config.a_steps + 1, nb=config.b_steps + 1,
                           full_transient=...) -> NamedTuple

The derived settings a 2D-map sweep actually runs with, resolved from a `BifurcationMapConfig` in one
call. Returns `(; seed_mode, lyapunov_enabled, lyapunov_iterations, lyapunov_transient,
multistability_enabled, transient_budget, neighbor_transient, tile_sizes, tile_count)`. `na`/`nb` (the
grid dimensions) only affect `tile_sizes`/`tile_count`; they default to the config's own grid.
"""
function map_effective_settings(config::BifurcationMapConfig;
                                na::Integer = config.a_steps + 1,
                                nb::Integer = config.b_steps + 1,
                                full_transient::Integer = _map_transient_budget(config))
    seed_mode = _map_seed_mode(config, full_transient)
    return (
        seed_mode = seed_mode,
        lyapunov_enabled = _map_lyapunov_enabled(config),
        lyapunov_iterations = _map_lyapunov_iterations(config),
        lyapunov_transient = _map_lyapunov_transient(config),
        multistability_enabled = _map_multistability_enabled(config),
        transient_budget = full_transient,
        neighbor_transient = _map_effective_neighbor_transient(config, full_transient),
        tile_sizes = _map_effective_tile_sizes(config, na, nb, seed_mode),
        tile_count = _map_tile_count(config, na, nb, seed_mode),
    )
end

# --- 2D-map / Poincaré diagnostics producers ---
"""    map_lyapunov_diagnostics(...) -> Dict — Lyapunov-field summary for the diagnostics payload."""
const map_lyapunov_diagnostics = _map_lyapunov_diagnostics
"""    map_neighbor_seed_diagnostics(...) -> Dict — neighbor-seed acceleration summary."""
const map_neighbor_seed_diagnostics = _map_neighbor_seed_diagnostics
"""    poincare_crossing_diagnostics_summary(...) -> Dict — Poincaré-crossing diagnostics summary."""
const poincare_crossing_diagnostics_summary = _poincare_crossing_diagnostics_summary
"""    orbit_geometry_summary(...) -> Dict — orbit-geometry summary for the diagnostics payload."""
const orbit_geometry_summary = _orbit_geometry_summary

# --- result extraction ---
"""    branch_points(result::BranchResult) -> Vector — the recorded continuation branch points."""
const branch_points = _branch_points

# --- continuation branch post-processing ---
"""    trim_branch_to_period(sys, branch::BranchResult, base_params, linked_param_indices; …) -> BranchResult"""
const trim_branch_to_period = _trim_branch_to_period
"""    collect_distinct_period_branches(candidate_sets, max_branches_per_period, param_tol, state_tol) -> Vector{BranchResult}"""
const collect_distinct_period_branches = _collect_distinct_period_branches
"""    branch_stability(sys, state, params, period::Int; …)  — orbit/branch stability (was `_map_stability`)."""
const branch_stability = _map_stability
"""    branches_for_skeleton_param(sys::ContinuousODE, config::ContinuationConfig, period, skeleton_param, base_params; …) -> Vector{BranchResult}"""
const branches_for_skeleton_param = _branches_for_skeleton_param
"""    is_duplicate_branch(candidate::BranchResult, existing::Vector{BranchResult}, param_tol, state_tol) -> Bool"""
const is_duplicate_branch = _is_duplicate_branch
"""    poincare_projected(sys::ContinuousODE, point, params; period=1, …)  — project a point through the Poincaré map."""
const poincare_projected = _poincare_projected
"""    splice_refined_continuous_branches(original::BranchResult, refined; …) -> BranchResult — graft a refined segment into a continuous branch."""
const splice_refined_continuous_branches = _splice_refined_continuous_branches
