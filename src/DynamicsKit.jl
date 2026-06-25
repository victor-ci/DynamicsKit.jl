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

# Utilities (before analysis, as analysis uses config)
include("utils/config.jl")
include("utils/io.jl")
include("utils/coercion.jl")               # value coercion + JSON-plain helpers (analysis caches use these)

# Analysis
include("analysis/parameter_mapping.jl")   # Contract A: shared param-vector mapping (before sweeps)
include("analysis/solvers.jl")             # public ODE solver selection (workbench + scripted analysis)
include("analysis/brute_force.jl")
include("analysis/lyapunov.jl")
include("analysis/spectrum.jl")
include("analysis/phase_portrait.jl")
include("analysis/continuation.jl")
include("analysis/codim2.jl")
include("analysis/skeleton.jl")
include("analysis/atlas.jl")
include("analysis/contract_kernels.jl")      # Contract B: publish analysis kernels (after defs)
include("analysis/contract_accessors.jl")   # Contract C: publish result/diagnostics accessors (after defs)
include("utils/result_serialization.jl")    # serialize library result types (atlas cache; after Atlas* types)

# Visualization
include("visualization/plots.jl")

# Exports — types
export DynamicalSystem, DiscreteMap, ContinuousODE, PoincareSection, SwitchingEvent
export BifurcationResult, BranchResult, BruteForceResult, LyapunovDiagramResult, BasinsResult, LyapunovFieldResult, BifurcationMapResult, PhasePortraitResult, PowerSpectrumResult, Codim2CurveResult

# Exports — built-in systems
export henon_map, vilnius_oscillator, buck_converter, buck_voltage_mode, boost_converter
export colpitts_simple_oscillator, colpitts_exponential_oscillator, colpitts_dynamic_beta_oscillator
export ikeda_map, rossler_oscillator, memristive_diode_bridge

# Exports — parameter mapping (Contract A)
export inject_param, build_sweep_params, build_basins_params, basins_ic_template
export map_param_template, map_a_write_indices, map_b_write_indices
export map_params_from_template, map_params_from_buffer!, build_map_params

# Exports — sweep cache hook (Contract D): in/out per-cell grids
export MapCellGrid, LyapunovCellGrid, BasinsCellGrid

# Exports — analysis kernels (Contract B). Per Contract D, the per-cell recorders / kernel /
# adaptive-refinement / direct-field-validate-and-record / atlas-hidden-sample helpers reverted to
# private (the workbench drives the public sweeps via the `cells=` hook). The orbit-sampling,
# period-detection and Lyapunov-estimation kernels stay public for scripted analysis (paper-artifacts).
export resolve_initial_state, sample_discrete_orbit, sample_continuous_poincare_orbit, collect_poincare_points
export detect_period, detect_discrete_map_period, detect_continuous_poincare_period
export estimate_discrete_map_largest_lyapunov, estimate_continuous_poincare_largest_lyapunov

# Exports — result & diagnostics accessors (Contract C). Per Contract D, the effective-settings,
# diagnostics producers, per-cell storage/recorders and result-extraction utilities reverted to
# private (assembled inside the public sweeps). The continuation branch post-processing helpers stay.
export trim_branch_to_period, collect_distinct_period_branches, branch_stability,
       branches_for_skeleton_param, is_duplicate_branch, poincare_projected

# Exports — scripted-analysis helpers (consumed by reproducibility scripts; library-public so
# paper-artifacts depends only on the library, never the workbench).
export select_ode_solver, collect_trajectory_seed_points

# Exports — analysis
export brute_force_diagram, continuation_branch, continuation_branches, continuation_branch_diagnostics, continuation_atlas, atlas_branches, find_periodic_skeleton
export basins_of_attraction, bifurcation_map, phase_portrait, refine_branch, auto_refine_branch
export lyapunov_diagram, lyapunov_field, power_spectrum, codim2_curve
export switching_event_diagnostics

# Exports — atlas results
export AtlasResult

# Exports — config
export BruteForceConfig, LyapunovConfig, ContinuationConfig, BasinsConfig, BifurcationMapConfig, PhasePortraitConfig, PowerSpectrumConfig, Codim2Config, RefinementConfig, AtlasConfig, ReseedConfig

# Exports — I/O
export save_result, load_result

# Exports — visualization
export plot_brute_force, plot_lyapunov_diagram, plot_branches, plot_overlay, plot_basins, plot_bifurcation_map, plot_lyapunov_field, plot_codim2, plot_phase_portrait, plot_power_spectrum
export plot_overlay_heatmap, plot_panel_grid, plot_seed_pair_composite

end # module
