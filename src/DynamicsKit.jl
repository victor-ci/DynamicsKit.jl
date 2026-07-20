"""
    DynamicsKit

A Julia library for bifurcation analysis of dynamical systems using both brute-force
parameter sweeps and continuation methods (via BifurcationKit.jl).

Supports discrete maps and continuous-time ODE systems with Poincaré sections.
"""
module DynamicsKit

# Core dependencies
using StaticArrays
using LinearAlgebra
using Statistics
using ForwardDiff
using DifferentialEquations
using BifurcationKit
using FFTW
using JLD2
using Dates
using SHA
using Unicode
using Parameters
using Setfield
using Accessors
using UUIDs
using Random

# Type system
include("systems/types.jl")
include("systems/henon.jl")
include("systems/vilnius.jl")
include("systems/buck.jl")
include("systems/buck_voltage_mode.jl")
include("systems/boost.jl")
include("systems/colpitts.jl")
include("systems/ikeda.jl")
include("systems/rossler.jl")
include("systems/memristive_diode_bridge.jl")
include("systems/switching_map.jl")            # switching map generator (AffineModeSpec, SwitchingCircuitDescription)

# Utilities (before analysis, as analysis uses config)
include("utils/config.jl")
include("utils/io.jl")
include("utils/coercion.jl")               # value coercion + JSON-plain helpers (analysis caches use these)

# Analysis
include("analysis/parameter_mapping.jl")   # shared param-vector mapping (before sweeps)
include("analysis/solvers.jl")             # public ODE solver selection
include("analysis/brute_force.jl")
include("analysis/lyapunov.jl")
include("analysis/lyapunov_spectrum.jl")
include("analysis/spectrum.jl")
include("analysis/phase_portrait.jl")
include("analysis/continuation.jl")
include("analysis/collocation.jl")
include("analysis/codim2.jl")
include("analysis/skeleton.jl")
include("analysis/normal_forms.jl")
include("analysis/map_special_points.jl")      # map-aware fold/flip/NS emission
include("analysis/border_collision.jl")        # continuous-map border-collision classification
include("analysis/atlas.jl")
include("analysis/contract_kernels.jl")     # publish analysis kernels (after defs)
include("analysis/contract_accessors.jl")   # publish result/diagnostics accessors (after defs)
include("analysis/branch_families.jl")      # conservative orbit-geometry family inference
include("analysis/branch_reachability.jl")  # multistability-aware continuation (basin reachability)
include("analysis/tolerance_fields.jl")     # parameter-robustness: regime-boundary margins + tolerance maps
include("analysis/robust_chaos.jl")         # robust-chaos certificate (after atlas + kernels/accessors)
include("utils/result_serialization.jl")    # serialize library result types (atlas cache; after Atlas* types)

# Visualization
include("visualization/plots.jl")

# Exports — types
export DynamicalSystem, DiscreteMap, ContinuousODE, PoincareSection, SwitchingEvent
export BifurcationResult, BranchResult, BruteForceResult, LyapunovDiagramResult, BasinsResult, LyapunovFieldResult, LyapunovSpectrumResult, BifurcationMapResult, PhasePortraitResult, PowerSpectrumResult, Codim2CurveResult, OrbitBranchResult, MapNormalForm, MapSpecialPoint
export StableWindowEvidence, RobustChaosCertificate, RobustChaosEvidence
export BorderCollisionClassification, BorderCollisionPoint
export AffineModeSpec, SwitchingCircuitDescription

# Exports — system accessors
export state_dim, switching_events

# Exports — built-in systems
export henon_map, vilnius_oscillator, buck_converter, buck_voltage_mode, boost_converter
export colpitts_simple_oscillator, colpitts_exponential_oscillator, colpitts_dynamic_beta_oscillator
export ikeda_map, rossler_oscillator, memristive_diode_bridge

# Exports — switching map generator
export switching_map, buck_converter_description, boost_converter_description

# Exports — parameter mapping
export inject_param, build_sweep_params, build_basins_params, basins_ic_template
export map_param_template, map_a_write_indices, map_b_write_indices
export map_params_from_template, map_params_from_buffer!, build_map_params

# Exports — sweep cache hook: in/out per-cell grids
export MapCellGrid, LyapunovCellGrid, BasinsCellGrid

# Exports — analysis kernels. The orbit-sampling, period-detection and Lyapunov-estimation
# kernels are public for scripted analysis. The 2D-map kernel (`bifurcation_map_kernel`) returns
# `(result, diagnostics)` and takes a `cells=` hook, which external cache layers need and the
# public `bifurcation_map` does not expose. The per-cell Lyapunov recorders / direct-field
# validate-and-record stay private (driven via the `cells=` hook).
export resolve_initial_state, sample_discrete_orbit, sample_continuous_poincare_orbit, collect_poincare_points
export detect_period, detect_discrete_map_period, detect_continuous_poincare_period
export estimate_discrete_map_largest_lyapunov, estimate_continuous_poincare_largest_lyapunov
export bifurcation_map_kernel, atlas_hidden_period_sample_indices

# Exports — result & diagnostics accessors: continuation branch post-processing helpers plus the
# 2D-map effective-settings, diagnostics producers and branch-point extraction. (Per-cell
# storage/recorders stay private — assembled inside the public sweeps.)
export trim_branch_to_period, collect_distinct_period_branches, branch_stability,
       branches_for_skeleton_param, is_duplicate_branch, poincare_projected
export branch_points, splice_refined_continuous_branches
export BranchFamilyAssignment, BranchBasinAssignment, branch_family_assignments, branch_basin_assignments
export BranchReachabilityResult, BranchReachabilitySample, branch_reachability
export reachability_category_counts, reachability_category_fractions, branch_reachability_fractions, branch_reachability_status_label
export RegimeBoundaryResult, ToleranceMapResult, regime_boundary_distances, tolerance_regime_map
export regime_boundary_summary, tolerance_regime_summary
export AbstractTolerance, UniformTolerance, GaussianTolerance
export map_effective_settings
export map_lyapunov_diagnostics, map_neighbor_seed_diagnostics, poincare_crossing_diagnostics_summary, orbit_geometry_summary

# Exports — scripted-analysis helpers (consumed by reproducibility scripts).
export select_ode_solver, collect_trajectory_seed_points

# Exports — analysis
export brute_force_diagram, continuation_branch, continuation_branches, continuation_branch_diagnostics, continuation_atlas, atlas_branches, find_periodic_skeleton
export continuation_orbit_collocation
export orbit_branch_parameters, orbit_branch_periods, orbit_branch_orbit, orbit_branch_amplitude, orbit_branch_multipliers, orbit_branch_stability
export map_normal_form, map_special_points
export border_collision_classify, border_collision_at_cycle, border_collision_points
export basins_of_attraction, bifurcation_map, phase_portrait, refine_branch, auto_refine_branch
export lyapunov_diagram, lyapunov_field, lyapunov_spectrum, power_spectrum, codim2_curve
export switching_event_diagnostics
export robust_chaos_certificate, robust_chaos_evidence

# Exports — atlas + combined-branch results
export AtlasResult, AtlasWindow, AtlasGap, AtlasReconSample, AtlasBranchRecord
export CombinedBranchResult
export Codim2ContinuationResult

# Exports — config
export BruteForceConfig, LyapunovConfig, LyapunovSpectrumConfig, ContinuationConfig, CollocationConfig, BasinsConfig, BifurcationMapConfig, PhasePortraitConfig, PowerSpectrumConfig, Codim2Config, RefinementConfig, AtlasConfig, ReseedConfig
export RobustChaosConfig
export BranchReachabilityConfig
export RegimeBoundaryConfig, ToleranceConfig

# Exports — I/O
export save_result, load_result

# Exports — JSON-plain result serialization (wire format for public result types).
# Per-field sub-helpers stay private.
export serialize_bruteforce_result, deserialize_bruteforce_result
export serialize_branch_result, deserialize_branch_result
export serialize_atlas_result, deserialize_atlas_result
export serialize_codim2_continuation_result, deserialize_codim2_continuation_result
export serialize_robust_chaos_certificate, deserialize_robust_chaos_certificate
export serialize_robust_chaos_evidence, deserialize_robust_chaos_evidence
export serialize_branch_reachability_result, deserialize_branch_reachability_result
export serialize_regime_boundary_result, deserialize_regime_boundary_result
export serialize_tolerance_map_result, deserialize_tolerance_map_result
export serialize_map_normal_form, deserialize_map_normal_form
export serialize_map_special_point, deserialize_map_special_point
export serialize_border_collision_classification, deserialize_border_collision_classification
export serialize_border_collision_point, deserialize_border_collision_point

# Exports — visualization
export plot_brute_force, plot_lyapunov_diagram, plot_lyapunov_spectrum, plot_branches, plot_overlay, plot_basins, plot_bifurcation_map, plot_lyapunov_field, plot_codim2, plot_phase_portrait, plot_power_spectrum
export plot_overlay_heatmap, plot_panel_grid, plot_seed_pair_composite

# Exports — trace-data helpers (data behind the Plots recipes; consumed by the workbench UI layer)
export branch_plot_traces, resolve_plot_params, branch_point_state, orbit_phase_alignment_shift
export phase_jump_break_indices, trace_breaks, codim2_curve_label, codim2_valid_runs

end # module
