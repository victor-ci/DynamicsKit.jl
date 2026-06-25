"""
Contract C — Result & diagnostics accessors. Stable public API consumed by the workbench for
config-derived ("effective") settings, the diagnostics payload schema, per-cell storage/recorders,
result extraction, small utilities, and continuation branch post-processing. See
`docs/internal/contracts/contract-c-result-accessors.md`.

Mechanism note: unlike Contract A (which relocated a self-contained family), these definitions stay
in their home files (`brute_force.jl`, `continuation.jl`) because they are tightly coupled to
sibling internals there. Here we only *publish* them as `const <public> = <_internal>` bindings (with docstrings); the
matching `export` statements live in `src/DynamicsKit.jl` ("Exports — result & diagnostics
accessors (Contract C)" block). The underscore names remain the in-place definitions and keep
working for internal callers. The file decomposition phase will relocate and canonicalize. Included
AFTER brute_force/continuation so the underscore functions already exist.

Group 1 (DROP) from the inventory — `_map_lyapunov_enabled`, `_map_multistability_enabled`,
`_map_lyapunov_transient` — are intentionally NOT published: they are trivial reads of public
config fields and are deleted when the workbench migrates to reading the fields directly.
"""

# Groups 2–5 (effective-settings, diagnostics producers, per-cell storage allocators/recorders,
# result extraction + small utilities) reverted to underscore-private under Contract D: the
# workbench now drives caching through the public `bifurcation_map` / `lyapunov_field` /
# `basins_of_attraction` sweeps (`cells=` hook), which assemble these diagnostics and storage
# internally, so the workbench no longer reaches the producers directly.
# See docs/internal/contracts/contract-d-sweep-cache-hook.md.

# --- Group 6: continuation branch post-processing ---
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
