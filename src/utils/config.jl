"""
Configuration types for analysis algorithms.
"""

"""
    BruteForceConfig

Configuration for brute-force bifurcation diagram generation.

# Fields
- `param_min`: Minimum bifurcation parameter value
- `param_max`: Maximum bifurcation parameter value
- `param_steps`: Number of parameter steps
- `iterations`: Total iterations per parameter value
- `transient`: Number of initial iterations to discard
- `param_index`: Which parameter to vary (index into params vector)
- `fixed_params`: Values for non-varied parameters
 - `linked_param_indices`: Additional parameter slots that should be set to the same swept value
 - `min_crossing_time`: Ignore section crossings before this time for continuous-time systems
"""
@with_kw struct BruteForceConfig
    param_min::Float64
    param_max::Float64
    param_steps::Int = 400
    iterations::Int = 500
    transient::Int = 300
    param_index::Int = 1
    fixed_params::Vector{Float64} = Float64[]
    linked_param_indices::Vector{Int} = Int[]
    min_crossing_time::Float64 = 1e-6
    @assert isfinite(min_crossing_time) && min_crossing_time >= 0.0 "BruteForceConfig.min_crossing_time must be finite and >= 0"
end

"""
    LyapunovConfig

Configuration for a 1D largest-Lyapunov-Exponent parameter sweep.

# Fields
- `param_min`, `param_max`, `param_steps`: Bifurcation-parameter sweep range
- `param_index`: Which parameter to vary
- `linked_param_indices`: Additional parameter slots set to the same swept value
- `fixed_params`: Base parameter vector for non-varied parameters
- `transient`: Steps / Poincare returns discarded before estimation
- `iterations`: Renormalized steps / returns used for the finite-time estimate
- `perturbation`: Initial trajectory separation for the two-trajectory estimator
- `neutral_tolerance`: Absolute exponent threshold used to classify near-neutral samples
- `divergence_cutoff`: Optional state-amplitude cutoff; `Inf` disables bailout
- `min_crossing_time`: Ignore section crossings before this time for continuous-time Poincare-return estimation
"""
@with_kw struct LyapunovConfig
    param_min::Float64
    param_max::Float64
    param_steps::Int = 400
    param_index::Int = 1
    linked_param_indices::Vector{Int} = Int[]
    fixed_params::Vector{Float64} = Float64[]
    transient::Int = 150
    iterations::Int = 300
    perturbation::Float64 = 1e-8
    neutral_tolerance::Float64 = 1e-3
    divergence_cutoff::Float64 = Inf
    min_crossing_time::Float64 = 1e-6
    @assert isfinite(param_min) && isfinite(param_max) && param_max >= param_min "LyapunovConfig requires finite param_min/param_max with param_max >= param_min"
    @assert param_steps >= 1 "LyapunovConfig.param_steps must be >= 1"
    @assert param_index >= 1 "LyapunovConfig.param_index must be >= 1"
    @assert all(>=(1), linked_param_indices) "LyapunovConfig.linked_param_indices must be positive indices"
    @assert transient >= 0 "LyapunovConfig.transient must be >= 0"
    @assert iterations >= 1 "LyapunovConfig.iterations must be >= 1"
    @assert isfinite(perturbation) && perturbation > 0.0 "LyapunovConfig.perturbation must be finite and > 0"
    @assert isfinite(neutral_tolerance) && neutral_tolerance >= 0.0 "LyapunovConfig.neutral_tolerance must be finite and >= 0"
    @assert isfinite(divergence_cutoff) || divergence_cutoff == Inf "LyapunovConfig.divergence_cutoff must be finite or Inf"
    @assert isfinite(min_crossing_time) && min_crossing_time >= 0.0 "LyapunovConfig.min_crossing_time must be finite and >= 0"
end

"""
    LyapunovSpectrumConfig

Configuration for a full Lyapunov-spectrum estimate at a single operating point via
the Benettin/QR (tangent-space) method. Distinct from `LyapunovConfig`, which sweeps
one parameter and estimates only the largest exponent with the two-trajectory method.

The same fields drive both the discrete-map and continuous-flow estimators, but a few
carry a method-specific unit:

# Fields
- `k`: Number of exponents to track from the top of the spectrum. `0` selects the full
  state dimension. Must satisfy `1 <= k <= dim`.
- `transient`: Reorthonormalization intervals discarded before accumulation, letting the
  tangent frame align with the leading covariant directions. For discrete maps one
  interval is one map iteration; for flows it is one `renorm_dt` window.
- `steps`: Reorthonormalization intervals accumulated into the estimate.
- `renorm_dt`: Flow-only integration time between successive QR reorthonormalizations.
  Ignored for discrete maps (one iteration per interval).
- `divergence_cutoff`: Optional state-amplitude bailout; `Inf` disables it.
"""
@with_kw struct LyapunovSpectrumConfig
    k::Int = 0
    transient::Int = 200
    steps::Int = 2000
    renorm_dt::Float64 = 0.5
    divergence_cutoff::Float64 = Inf
    @assert k >= 0 "LyapunovSpectrumConfig.k must be >= 0 (0 selects the full state dimension)"
    @assert transient >= 0 "LyapunovSpectrumConfig.transient must be >= 0"
    @assert steps >= 1 "LyapunovSpectrumConfig.steps must be >= 1"
    @assert isfinite(renorm_dt) && renorm_dt > 0.0 "LyapunovSpectrumConfig.renorm_dt must be finite and > 0"
    @assert isfinite(divergence_cutoff) || divergence_cutoff == Inf "LyapunovSpectrumConfig.divergence_cutoff must be finite or Inf"
end

"""
    ContinuationConfig

Configuration for branch continuation via BifurcationKit.

# Fields
- `p_min`: Minimum bifurcation parameter
- `p_max`: Maximum bifurcation parameter
- `ds`: Initial step size (sign determines direction)
- `dsmax`: Maximum step size
- `dsmin`: Minimum step size
- `max_steps`: Maximum continuation steps
- `newton_tol`: Newton solver tolerance
- `newton_max_iter`: Maximum Newton iterations
- `detect_bifurcation`: Bifurcation detection level (0–3)
- `param_index`: Which parameter to vary
 - `linked_param_indices`: Additional parameter slots that should follow the same continuation value
- `a`: PALC step-adaptation aggressiveness factor (BifurcationKit default 0.5; higher allows larger ds changes per step)
- `detect_fold`: Record fold/limit points in `specialpoint` (BifurcationKit default true)
- `save_sol_every_step`: Save the full solution every N steps; must be > 0 so `branch.sol` carries full state vectors (needed by re-seeding)
- `ode_jacobian_method`: Continuous-ODE Poincaré-map derivative method (`:finite_difference` or `:variational`)
"""
@with_kw struct ContinuationConfig
    p_min::Float64
    p_max::Float64
    ds::Float64 = 0.01
    dsmax::Float64 = 0.05
    dsmin::Float64 = 1e-6
    max_steps::Int = 1000
    newton_tol::Float64 = 1e-10
    newton_max_iter::Int = 25
    detect_bifurcation::Int = 3
    param_index::Int = 1
     linked_param_indices::Vector{Int} = Int[]
    a::Float64 = 0.5
    detect_fold::Bool = true
    save_sol_every_step::Int = 1
    ode_jacobian_method::Symbol = :finite_difference
    @assert isfinite(p_min) && isfinite(p_max) && p_max >= p_min "ContinuationConfig requires finite p_min/p_max with p_max >= p_min"
    @assert isfinite(ds) && ds != 0.0 "ContinuationConfig.ds must be finite and non-zero"
    @assert isfinite(dsmax) && dsmax > 0.0 "ContinuationConfig.dsmax must be finite and > 0"
    @assert isfinite(dsmin) && dsmin > 0.0 "ContinuationConfig.dsmin must be finite and > 0"
    @assert dsmax >= dsmin "ContinuationConfig requires dsmax >= dsmin"
    @assert max_steps >= 1 "ContinuationConfig.max_steps must be >= 1"
    @assert isfinite(newton_tol) && newton_tol > 0.0 "ContinuationConfig.newton_tol must be finite and > 0"
    @assert newton_max_iter >= 1 "ContinuationConfig.newton_max_iter must be >= 1"
    @assert 0 <= detect_bifurcation <= 3 "ContinuationConfig.detect_bifurcation must be in 0:3"
    @assert param_index >= 1 "ContinuationConfig.param_index must be >= 1"
    @assert all(>=(1), linked_param_indices) "ContinuationConfig.linked_param_indices must be positive indices"
    @assert isfinite(a) && a > 0.0 "ContinuationConfig.a must be finite and > 0"
    @assert save_sol_every_step > 0 "ContinuationConfig.save_sol_every_step must be > 0"
    @assert ode_jacobian_method in (:finite_difference, :variational) "ContinuationConfig.ode_jacobian_method must be :finite_difference or :variational"
end

"""
    CollocationConfig

Configuration for full-orbit periodic-orbit continuation by orthogonal collocation, as
an alternative to the default Poincaré return-map shooting. The whole time-parameterized
orbit and its period are continued as a boundary-value problem (a mesh of `ntst`
intervals with degree-`m` polynomials plus a phase condition), rather than a fixed point
of the return map.

# Fields
- `continuation`: Primary-axis continuation settings (`param_index`, `p_min`/`p_max`,
  `ds`/`dsmax`/`dsmin`, `max_steps`, `newton_tol`, `a`, `linked_param_indices`). The
  return-map-specific fields (`detect_bifurcation`, `ode_jacobian_method`,
  `save_sol_every_step`, `detect_fold`) are not used by the collocation path.
- `ntst`: Number of mesh intervals for the collocation discretization.
- `m`: Polynomial degree per interval (Gauss collocation points).
- `mesh_adapt`: Enable BifurcationKit mesh adaptation during continuation.
- `newton_max_iter`: Newton budget for the orbit corrector. Collocation's first
  correction is heavier than the return-map solve, so this defaults higher than
  `ContinuationConfig.newton_max_iter`.
- `settle_time`: Flow time integrated from the seed to settle onto the attractor before
  the orbit seed is extracted.
- `seed_span_factor`: The seed orbit is integrated over `seed_span_factor` periods to
  give the collocation initial guess a full cycle with margin.
- `optimal_period`: Let BifurcationKit refine the seed period around the estimate.
- `bothside`: Continue in both parameter directions from the seed (as the shooting
  branches do), so a mid-window seed covers the whole `[p_min, p_max]` range.
"""
@with_kw struct CollocationConfig
    continuation::ContinuationConfig
    ntst::Int = 40
    m::Int = 4
    mesh_adapt::Bool = false
    newton_max_iter::Int = 30
    settle_time::Float64 = 200.0
    seed_span_factor::Float64 = 1.3
    optimal_period::Bool = true
    bothside::Bool = true
    @assert ntst >= 5 "CollocationConfig.ntst must be >= 5"
    @assert 2 <= m <= 7 "CollocationConfig.m must be in 2:7"
    @assert newton_max_iter >= 1 "CollocationConfig.newton_max_iter must be >= 1"
    @assert isfinite(settle_time) && settle_time >= 0.0 "CollocationConfig.settle_time must be finite and >= 0"
    @assert isfinite(seed_span_factor) && seed_span_factor > 1.0 "CollocationConfig.seed_span_factor must be finite and > 1"
end

"""
    ConnectingOrbitConfig

Configuration for continuing a connecting orbit with the projection
boundary-condition method. Covers homoclinic connections to an equilibrium,
heteroclinic connections between two saddles, and homoclinic connections to a
saddle periodic orbit.

The orbit is discretized on a uniform trapezoidal mesh of `n_mesh` intervals in
rescaled time; the saddle equilibria (or the saddle cycle's Floquet data) are
solved alongside the orbit and the endpoints are pinned to the linear
stable/unstable subspaces by projection boundary conditions. A damped
pseudo-inverse fallback corrector (`use_fallback`) rescues predictor points
where the primary Newton corrector stalls, and its use is recorded in the
result provenance.

# Fields
- `continuation`: Secondary-axis continuation settings (`param_index`,
  `p_min`/`p_max`, `ds`/`dsmax`/`dsmin`, `max_steps`, `newton_tol`,
  `newton_max_iter`). `param_index` selects the free secondary parameter.
- `kind`: `:homoclinic`, `:heteroclinic`, or `:saddle_cycle`.
- `n_mesh`: Number of trapezoidal mesh intervals for the truncated orbit.
- `max_return_time`: Cap on the truncation time `T`.
- `detect_events`: Evaluate the standard HomCont eigenvalue test functions and
  record sign-crossing special points along the locus.
- `test_orbit_flip` / `test_inclination_flip`: Also evaluate the orbit-flip and
  adjoint-transport inclination-flip test functions. Inclination-flip test
  functions are only available when the relevant manifold has at least two
  stable (or two unstable) eigenvalues; otherwise they are reported with an
  `:unavailable` status rather than a fabricated value.
- `use_fallback`: Enable the damped pseudo-inverse fallback corrector.
- `fallback_max_iter`: Iteration budget for the fallback corrector.
- `projector_refresh`: Recompute the frozen stable/unstable projectors every
  `projector_refresh` Newton iterations during correction (`1` = every iteration,
  default). Higher values freeze projectors for more iterations; this can speed
  up costly corrections at the risk of slower convergence if the saddle moves
  significantly between refreshes. Projectors are frozen to `Float64` because
  `eigen`/`schur` cannot differentiate through `ForwardDiff.Dual` matrices.
- `orbit_save_stride` / `max_saved_orbits`: Bounded retention of normalized
  trajectories along the locus.
- `bothside`: Continue in both secondary-parameter directions from the seed.
- `source_index`: Which stored orbit of a source `OrbitBranchResult` seeds the
  connection (`0` selects the long-period endpoint).
- `provenance`: Free-form provenance string stored in the result diagnostics.
- `epsilon_start` / `epsilon_end`: Endpoint distance from the source/target
  saddle (equilibrium) or reference phase point (saddle-cycle). The BVP
  boundary condition pins `|u(0) - xs|` to `epsilon_start` and
  `|u(T) - xt|` to `epsilon_end`. `NaN` (the default) means derive the radius
  from the seed orbit's natural endpoint distance; use an explicit positive
  value to override it. Zero and negative values are rejected.
"""
@with_kw struct ConnectingOrbitConfig
    continuation::ContinuationConfig
    kind::Symbol = :homoclinic
    n_mesh::Int = 120
    epsilon_start::Float64 = NaN
    epsilon_end::Float64 = NaN
    max_return_time::Float64 = Inf
    detect_events::Bool = true
    test_orbit_flip::Bool = true
    test_inclination_flip::Bool = true
    use_fallback::Bool = true
    fallback_max_iter::Int = 150
    projector_refresh::Int = 1
    orbit_save_stride::Int = 10
    max_saved_orbits::Int = 25
    bothside::Bool = false
    source_index::Int = 0
    provenance::String = ""
    @assert kind in (:homoclinic, :heteroclinic, :saddle_cycle) "ConnectingOrbitConfig.kind must be :homoclinic, :heteroclinic, or :saddle_cycle"
    @assert n_mesh >= 10 "ConnectingOrbitConfig.n_mesh must be >= 10"
    @assert isnan(epsilon_start) || (isfinite(epsilon_start) && epsilon_start > 0.0) "ConnectingOrbitConfig.epsilon_start must be a positive finite value or NaN (derive from seed)"
    @assert isnan(epsilon_end) || (isfinite(epsilon_end) && epsilon_end > 0.0) "ConnectingOrbitConfig.epsilon_end must be a positive finite value or NaN (derive from seed)"
    @assert (isfinite(max_return_time) || max_return_time == Inf) && max_return_time > 0.0 "ConnectingOrbitConfig.max_return_time must be positive and finite or Inf"
    @assert fallback_max_iter >= 1 "ConnectingOrbitConfig.fallback_max_iter must be >= 1"
    @assert projector_refresh >= 1 "ConnectingOrbitConfig.projector_refresh must be >= 1"
    @assert source_index >= 0 "ConnectingOrbitConfig.source_index must be >= 0 (0 selects the source endpoint)"
    @assert orbit_save_stride >= 1 "ConnectingOrbitConfig.orbit_save_stride must be >= 1"
    @assert max_saved_orbits >= 2 "ConnectingOrbitConfig.max_saved_orbits must be >= 2"
end

"""
    ReseedConfig

Controls automatic re-seeding when a continuation direction terminates prematurely in the
parameter interior (not at a boundary, and not at a fold PALC already traversed). The branch's
trailing trajectory is extrapolated, a targeted periodic-skeleton search is run there, and a
seed of the same period is used to resume continuation. Disabled by default so existing
continuation behavior is unchanged unless explicitly opted in (the atlas turns it on).

# Fields
- `enabled`: Master switch (default false)
- `max_attempts`: Maximum re-seed attempts per continuation direction
- `trailing_k`: Number of trailing branch points used to extrapolate the trajectory
- `box_half_width_scale`: Skeleton search-box half-width = scale × trailing-state spread
- `box_half_width_min`: Floor for the search-box half-width so the box never degenerates
- `min_progress_dp`: A resumed segment must advance the parameter by at least this …
- `min_progress_points`: … and add at least this many points, else the attempt is abandoned
- `circulus_vitiosus_frac`: Abort if a re-seed query lands within this fraction of (p_max−p_min) of the original seed
- `n_skeleton_initial`: Initial conditions handed to the targeted skeleton search
"""
@with_kw struct ReseedConfig
    enabled::Bool = false
    max_attempts::Int = 3
    trailing_k::Int = 5
    box_half_width_scale::Float64 = 0.5
    box_half_width_min::Float64 = 1e-3
    min_progress_dp::Float64 = 1e-4
    min_progress_points::Int = 3
    circulus_vitiosus_frac::Float64 = 0.05
    n_skeleton_initial::Int = 10
    @assert max_attempts >= 0 "ReseedConfig.max_attempts must be >= 0"
    @assert trailing_k >= 2 "ReseedConfig.trailing_k must be >= 2"
    @assert isfinite(box_half_width_scale) && box_half_width_scale >= 0.0 "ReseedConfig.box_half_width_scale must be finite and >= 0"
    @assert isfinite(box_half_width_min) && box_half_width_min > 0.0 "ReseedConfig.box_half_width_min must be finite and > 0"
    @assert isfinite(min_progress_dp) && min_progress_dp >= 0.0 "ReseedConfig.min_progress_dp must be finite and >= 0"
    @assert min_progress_points >= 1 "ReseedConfig.min_progress_points must be >= 1"
    @assert isfinite(circulus_vitiosus_frac) && circulus_vitiosus_frac >= 0.0 "ReseedConfig.circulus_vitiosus_frac must be finite and >= 0"
    @assert n_skeleton_initial >= 1 "ReseedConfig.n_skeleton_initial must be >= 1"
end

"""
    BasinsConfig

Configuration for basins of attraction computation.

# Fields
- `bif_param`: Fixed bifurcation parameter value
- `max_period`: Maximum period to detect
- `precision`: Tolerance for period detection
- `iterations`: Total iterations per initial condition
- `x_min`, `x_max`, `x_steps`: Grid for first state variable (initial condition)
- `y_min`, `y_max`, `y_steps`: Grid for second state variable (initial condition)
- `fixed_params`: Full parameter vector (bif_param overrides the relevant entry)
- `param_index`: Which parameter slot holds the bifurcation parameter
- `min_crossing_time`: Ignore section crossings before this time for continuous-time systems
"""
@with_kw struct BasinsConfig
    bif_param::Float64
    max_period::Int = 10
    precision::Float64 = 1e-4
    iterations::Int = 1000
    x_min::Float64
    x_max::Float64
    x_steps::Int = 100
    y_min::Float64
    y_max::Float64
    y_steps::Int = 100
    fixed_params::Vector{Float64} = Float64[]
    param_index::Int = 1
    min_crossing_time::Float64 = 1e-6
    # Which state dimensions the (x, y) grid axes vary, and the full-state fill
    # for the non-gridded dimensions. Defaults reproduce the original behaviour
    # (grid dims 1 and 2, all other dims start at 0). `ic_template`, when given,
    # must have length `state dim`; its x_index / y_index entries are overwritten
    # by the grid values.
    x_index::Int = 1
    y_index::Int = 2
    ic_template::Vector{Float64} = Float64[]
    @assert isfinite(min_crossing_time) && min_crossing_time >= 0.0 "BasinsConfig.min_crossing_time must be finite and >= 0"
end

"""
    BifurcationMapConfig

Configuration for 2D bifurcation map (two-parameter periodicity sweep).

# Fields
- `a_min`, `a_max`, `a_steps`: Grid for first bifurcation parameter
- `b_min`, `b_max`, `b_steps`: Grid for second bifurcation parameter
- `a_index`, `b_index`: Parameter vector indices for the two swept parameters
- `a_linked_param_indices`: Additional parameter slots set to the first axis value
- `b_linked_param_indices`: Additional parameter slots set to the second axis value
- `max_period`: Maximum period to detect
- `precision`: Tolerance for period detection
- `iterations`: Total iterations per grid point
- `base_params`: Base parameter vector (both swept params override their entries)
- `divergence_cutoff`: Optional state-amplitude cutoff; `Inf` disables bailout
- `reuse_neighbor_seeds`: Reuse each grid point's final state as a neighbouring point's initial state
 - `neighbor_transient`: Optional reduced transient for neighbour-seeded cells after the first cell in a serpentine pass
 - `neighbor_tile_size_a`, `neighbor_tile_size_b`: Optional tile dimensions for deterministic neighbour traversal; `0` keeps the global (non-tiled) sweep
 - `multistability_initial_points`: Additional fixed initial conditions sampled per parameter cell for opt-in coexistence diagnostics
 - `lyapunov_enabled`: Compute optional largest-Lyapunov diagnostics for each map cell
 - `lyapunov_iterations`: Renormalized steps used for Lyapunov estimation; `0` selects an internal default
 - `lyapunov_transient`: Extra post-classification transient before Lyapunov estimation; `nothing` uses `0`
 - `lyapunov_perturbation`: Initial perturbation size for two-trajectory Lyapunov estimation
 - `lyapunov_neutral_tolerance`: Absolute exponent threshold for neutral/quasiperiodic candidates
 - `min_crossing_time`: Ignore section crossings before this time for continuous-time maps and Lyapunov-field runs
 - `adaptive_refinement_enabled`: Refine boundary/low-confidence 2D map cells into a sparse overlay
 - `adaptive_refinement_max_depth`: Maximum recursive subdivision depth for adaptive refinement
 - `adaptive_refinement_budget`: Maximum number of extra sparse samples; `0` selects a conservative automatic budget
 - `adaptive_refinement_min_confidence`: Refine cells with any corner below this confidence; `0` disables this trigger
 - `adaptive_refinement_confidence_delta`: Refine cells whose corner confidence range exceeds this value; `0` disables this trigger
"""
@with_kw struct BifurcationMapConfig
    a_min::Float64
    a_max::Float64
    a_steps::Int = 100
    b_min::Float64
    b_max::Float64
    b_steps::Int = 100
    a_index::Int = 1
    b_index::Int = 2
    a_linked_param_indices::Vector{Int} = Int[]
    b_linked_param_indices::Vector{Int} = Int[]
    max_period::Int = 10
    precision::Float64 = 1e-4
    iterations::Int = 1000
    base_params::Vector{Float64} = Float64[]
    divergence_cutoff::Float64 = Inf
    reuse_neighbor_seeds::Bool = false
    neighbor_transient::Union{Nothing, Int} = nothing
    neighbor_tile_size_a::Int = 0
    neighbor_tile_size_b::Int = 0
    multistability_initial_points::Vector{Vector{Float64}} = Vector{Float64}[]
    lyapunov_enabled::Bool = false
    lyapunov_iterations::Int = 0
    lyapunov_transient::Union{Nothing, Int} = nothing
    lyapunov_perturbation::Float64 = 1e-8
    lyapunov_neutral_tolerance::Float64 = 1e-3
    min_crossing_time::Float64 = 1e-6
    adaptive_refinement_enabled::Bool = false
    adaptive_refinement_max_depth::Int = 1
    adaptive_refinement_budget::Int = 0
    adaptive_refinement_min_confidence::Float64 = 0.0
    adaptive_refinement_confidence_delta::Float64 = 0.0
    @assert isnothing(neighbor_transient) || neighbor_transient >= 0 "BifurcationMapConfig.neighbor_transient must be nothing or >= 0"
    @assert neighbor_tile_size_a >= 0 "BifurcationMapConfig.neighbor_tile_size_a must be >= 0"
    @assert neighbor_tile_size_b >= 0 "BifurcationMapConfig.neighbor_tile_size_b must be >= 0"
    @assert isempty(multistability_initial_points) || !reuse_neighbor_seeds "BifurcationMapConfig.multistability_initial_points requires fixed-seed traversal (reuse_neighbor_seeds=false)"
    @assert all(!isempty, multistability_initial_points) "BifurcationMapConfig.multistability_initial_points cannot contain empty initial-condition vectors"
    @assert lyapunov_iterations >= 0 "BifurcationMapConfig.lyapunov_iterations must be >= 0"
    @assert isnothing(lyapunov_transient) || lyapunov_transient >= 0 "BifurcationMapConfig.lyapunov_transient must be nothing or >= 0"
    @assert isfinite(lyapunov_perturbation) && lyapunov_perturbation > 0.0 "BifurcationMapConfig.lyapunov_perturbation must be finite and > 0"
    @assert isfinite(lyapunov_neutral_tolerance) && lyapunov_neutral_tolerance >= 0.0 "BifurcationMapConfig.lyapunov_neutral_tolerance must be finite and >= 0"
    @assert isfinite(min_crossing_time) && min_crossing_time >= 0.0 "BifurcationMapConfig.min_crossing_time must be finite and >= 0"
    @assert adaptive_refinement_max_depth >= 0 "BifurcationMapConfig.adaptive_refinement_max_depth must be >= 0"
    @assert adaptive_refinement_budget >= 0 "BifurcationMapConfig.adaptive_refinement_budget must be >= 0"
    @assert isfinite(adaptive_refinement_min_confidence) && adaptive_refinement_min_confidence >= 0.0 && adaptive_refinement_min_confidence <= 1.0 "BifurcationMapConfig.adaptive_refinement_min_confidence must be finite and in [0, 1]"
    @assert isfinite(adaptive_refinement_confidence_delta) && adaptive_refinement_confidence_delta >= 0.0 && adaptive_refinement_confidence_delta <= 1.0 "BifurcationMapConfig.adaptive_refinement_confidence_delta must be finite and in [0, 1]"
    @assert !adaptive_refinement_enabled || !reuse_neighbor_seeds "BifurcationMapConfig.adaptive_refinement_enabled requires fixed-seed traversal (reuse_neighbor_seeds=false)"
end

"""
    PhasePortraitConfig

Configuration for continuous-time phase portrait generation.

# Fields
- `time_start`, `time_stop`: Integration interval
- `tail_fraction`: Fraction of the trajectory retained for plotting after transient decay
- `poincare_crossings`: Maximum number of Poincaré section crossings to retain
- `min_crossing_time`: Ignore section crossings before this time
- `max_saved_points`: Maximum number of trajectory samples retained from the solver (`0` keeps every saved step)
- `maxiters`: Maximum ODE solver iterations
"""
@with_kw struct PhasePortraitConfig
    time_start::Float64 = 0.0
    time_stop::Float64 = 0.02
    tail_fraction::Float64 = 0.25
    poincare_crossings::Int = 50
    min_crossing_time::Float64 = 1e-6
    max_saved_points::Int = 0
    maxiters::Int = 10_000_000
end

"""
    PowerSpectrumConfig

Configuration for one-sided FFT spectrum estimation from a uniformly sampled time tail.

# Fields
- `time_start`, `time_stop`: Integration interval
- `dt`: Uniform sample interval
- `tail_fraction`: Fraction of the saved signal retained for spectrum estimation
- `window`: Spectral window (`:hann` or `:none`)
- `state_index`: State coordinate to analyze
- `maxiters`: Maximum ODE solver iterations
"""
@with_kw struct PowerSpectrumConfig
    time_start::Float64 = 0.0
    time_stop::Float64 = 100.0
    dt::Float64 = 0.05
    tail_fraction::Float64 = 0.6
    window::Symbol = :hann
    state_index::Int = 1
    maxiters::Int = 10_000_000
    @assert isfinite(time_start) && isfinite(time_stop) && time_stop > time_start "PowerSpectrumConfig requires finite time_start/time_stop with time_stop > time_start"
    @assert isfinite(dt) && dt > 0.0 "PowerSpectrumConfig.dt must be finite and > 0"
    @assert isfinite(tail_fraction) && 0.0 < tail_fraction <= 1.0 "PowerSpectrumConfig.tail_fraction must be in (0, 1]"
    @assert window in (:hann, :none) "PowerSpectrumConfig.window must be :hann or :none"
    @assert state_index >= 1 "PowerSpectrumConfig.state_index must be >= 1"
    @assert maxiters >= 1 "PowerSpectrumConfig.maxiters must be >= 1"
end

"""
    Codim2Config

Configuration for codimension-2 bifurcation-curve computation. Two engines are
available: `:slice_tracking` assembles the curve from a family of independent
1D continuation slices, while `:defining_system` continues the bifurcation
condition itself in the secondary parameter via a minimally augmented defining
system (period-doubling: `(DF^N + I)v = 0`; fold: `(DF^N - I)v = 0`) and
returns a `Codim2ContinuationResult`.

# Fields
- `continuation`: Primary-axis continuation settings (`param_index` chooses the
  primary swept parameter)
- `second_min`, `second_max`, `second_steps`: Secondary-parameter grid
- `second_param_index`: Which parameter slot holds the secondary parameter
- `second_linked_param_indices`: Additional slots tied to the secondary value
- `fixed_params`: Full base parameter vector when the system has more than two parameters
- `bifurcation_kind`: Target bifurcation kind (`:pd`, `:fold`, `:ns`; `:hopf` is accepted as an alias for `:ns`)
- `endpoint_margin`: Reject detected candidates within this distance of the primary continuation endpoints
- `tracking_tolerance`: Maximum primary-axis jump allowed when stitching neighbouring slice candidates (`nothing` picks a conservative default from the primary range)
- `tracking_mode`: How per-slice candidates are promoted to the returned curve (`:nearest` stitches a continuous family, `:minimum`/`:maximum` choose an extremal candidate independently on each slice)
- `anchor_second`: Secondary-parameter value used to seed the stitched curve (`nothing` uses the midpoint of the secondary range)
- `anchor_candidate_index`: Which sorted candidate to pick on the anchor slice when multiple are present
- `primary_seed_values`: Optional per-slice primary-parameter seed values used before each continuation slice; when omitted every slice reuses `fixed_params[continuation.param_index]`
- `primary_min_values`, `primary_max_values`: Optional per-slice primary continuation bounds; when omitted every slice reuses `continuation.p_min` / `continuation.p_max`
- `diagnostics_max_points`: Maximum branch points sampled when the period-doubling fallback uses branch diagnostics
- `fallback_to_stability_flips`: When `true`, period-doubling curves fall back to stable/unstable flip detection if BifurcationKit does not emit explicit `:pd` special points
- `threaded`: Opt-in multithreading (default `false`). Slice tracking runs independent continuation slices across threads; the defining-system engine threads the finite-difference augmented-Jacobian columns (ODE systems) and the per-sample curve diagnostics — the continuation walk itself is inherently sequential
- `engine`: `:slice_tracking` (default) or `:defining_system`; the defining-system engine supports `bifurcation_kind` `:pd`, `:fold`, and `:ns` (`:hopf` alias included)
- `curve_continuation`: Optional `ContinuationConfig` governing the defining-system curve leg; its `p_min`/`p_max`/`ds`/step/Newton fields apply to the **secondary** parameter (`nothing` derives conservative settings from `second_min`/`second_max`/`second_steps` and the primary Newton settings). Its `param_index`/`linked_param_indices`/`detect_bifurcation`/`ode_jacobian_method` fields are **ignored**: the leg always continues the secondary parameter (`second_param_index` + `second_linked_param_indices`) through a synthetic augmented problem
- `curve_diagnostics`: When `true` (default), the defining-system engine records per-point fixed-point residual norms and return-map multipliers on the returned curve
"""
@with_kw struct Codim2Config
   continuation::ContinuationConfig
   second_min::Float64
   second_max::Float64
    second_steps::Int = 40
    second_param_index::Int = 2
    second_linked_param_indices::Vector{Int} = Int[]
    fixed_params::Vector{Float64} = Float64[]
    bifurcation_kind::Symbol = :pd
    endpoint_margin::Float64 = 0.0
    tracking_tolerance::Union{Nothing, Float64} = nothing
    tracking_mode::Symbol = :nearest
    anchor_second::Union{Nothing, Float64} = nothing
    anchor_candidate_index::Int = 1
    primary_seed_values::Vector{Float64} = Float64[]
    primary_min_values::Vector{Float64} = Float64[]
    primary_max_values::Vector{Float64} = Float64[]
    diagnostics_max_points::Int = 400
    fallback_to_stability_flips::Bool = true
    threaded::Bool = false
    engine::Symbol = :slice_tracking
    curve_continuation::Union{Nothing, ContinuationConfig} = nothing
    curve_diagnostics::Bool = true
    @assert engine in (:slice_tracking, :defining_system) "Codim2Config.engine must be :slice_tracking or :defining_system"
    @assert isfinite(second_min) && isfinite(second_max) && second_max >= second_min "Codim2Config requires finite second_min/second_max with second_max >= second_min"
    @assert second_steps >= 1 "Codim2Config.second_steps must be >= 1"
    @assert second_param_index >= 1 "Codim2Config.second_param_index must be >= 1"
    @assert continuation.param_index != second_param_index "Codim2Config requires different primary and secondary parameter indices"
    @assert all(>=(1), second_linked_param_indices) "Codim2Config.second_linked_param_indices must be positive indices"
    @assert bifurcation_kind in (:pd, :fold, :ns, :hopf) "Codim2Config.bifurcation_kind must be :pd, :fold, :ns, or :hopf"
    @assert isfinite(endpoint_margin) && endpoint_margin >= 0.0 "Codim2Config.endpoint_margin must be finite and >= 0"
    @assert isnothing(tracking_tolerance) || (isfinite(tracking_tolerance) && tracking_tolerance >= 0.0) "Codim2Config.tracking_tolerance must be nothing or a finite value >= 0"
    @assert tracking_mode in (:nearest, :minimum, :maximum) "Codim2Config.tracking_mode must be :nearest, :minimum, or :maximum"
    @assert isnothing(anchor_second) || isfinite(anchor_second) "Codim2Config.anchor_second must be nothing or finite"
    @assert anchor_candidate_index >= 1 "Codim2Config.anchor_candidate_index must be >= 1"
    @assert isempty(primary_seed_values) || all(isfinite, primary_seed_values) "Codim2Config.primary_seed_values must be empty or contain only finite values"
    @assert isempty(primary_min_values) || all(isfinite, primary_min_values) "Codim2Config.primary_min_values must be empty or contain only finite values"
    @assert isempty(primary_max_values) || all(isfinite, primary_max_values) "Codim2Config.primary_max_values must be empty or contain only finite values"
    @assert isempty(primary_seed_values) || length(primary_seed_values) == second_steps + 1 "Codim2Config.primary_seed_values must be empty or have length second_steps + 1 ($(second_steps + 1))"
    @assert isempty(primary_min_values) || length(primary_min_values) == second_steps + 1 "Codim2Config.primary_min_values must be empty or have length second_steps + 1 ($(second_steps + 1))"
    @assert isempty(primary_max_values) || length(primary_max_values) == second_steps + 1 "Codim2Config.primary_max_values must be empty or have length second_steps + 1 ($(second_steps + 1))"
    @assert diagnostics_max_points >= 0 "Codim2Config.diagnostics_max_points must be >= 0"
end

"""
    AtlasConfig

Configuration scaffold for the automatic continuation atlas workflow.

# Fields
- `max_period`: Maximum period to classify when `periods` is not supplied explicitly
- `periods`: Explicit set of periods to target; defaults to `1:max_period`
- `brute_force`: Optional brute-force sweep settings override
- `continuation`: Optional continuation settings override
- `recon_steps`: Number of reconnaissance samples across the parameter window
- `recon_precision`: Period-classification tolerance used during reconnaissance
- `adaptive_recon`: Whether to insert extra reconnaissance samples before continuation
- `adaptive_recon_max_samples`: Maximum number of extra reconnaissance samples
- `adaptive_recon_max_depth`: Maximum midpoint-refinement passes
- `adaptive_recon_confidence_threshold`: Periodic samples below this confidence trigger local refinement
- `adaptive_recon_closure_gradient_factor`: Normalized closure-error jump that triggers refinement
- `window_min_support`: Minimum number of reconnaissance samples required to keep a candidate window
- `window_merge_gap`: Maximum number of uncertain samples allowed between compatible windows before merging
- `seed_points_per_window`: Number of brute-force-derived seed points to keep per candidate window
- `seed_box_padding`: Fractional padding added around local orbit clouds when building search boxes
- `skeleton_retry_budget`: Number of targeted skeleton retries per candidate window
- `continuation_retry_budget`: Number of continuation retries per recovered seed
- `max_refinement_depth`: Maximum recursive gap-refinement depth
- `max_total_windows`: Global cap on tracked candidate windows
- `max_total_branches`: Global cap on recovered branches
- `coverage_threshold`: Fraction of a window that must be covered before it is considered recovered
- `branch_switching`: Request limited bifurcation-aware follow-up probes from recorded special points
- `branch_switching_max_special_points`: Maximum special points to probe per branch
- `branch_switching_max_branches`: Maximum switched branches to accept per atlas window
- `branch_switching_window_fraction`: Local continuation half-window as a fraction of the continuation range
- `branch_switching_perturbation_scale`: State-space search half-width scale around special-point seeds
- `branch_switching_max_steps`: Maximum continuation steps for each local switched probe
- `branch_switching_max_seed_candidates`: Maximum perturbed seed hints around each special point
- `reuse_neighbor_seeds`: Request recycling successful skeleton seeds into nearby recovery attempts
- `neighbor_seed_max_entries`: Maximum cached seeds per period
- `neighbor_seed_max_distance_fraction`: Maximum reuse distance as a fraction of the continuation parameter span
- `neighbor_seed_max_points`: Maximum cached seed hints injected into one recovery attempt
- `threaded`: Whether atlas substeps may use Julia threads
- `cache_enabled`: Whether atlas intermediate/final artifacts may be cached
- `time_budget_s`: Optional wall-clock budget in seconds
- `reseed`: Targeted re-seed settings for continuation directions that die in the interior (enabled by default; a no-op when branches reach a boundary)
"""
@with_kw struct AtlasConfig
    max_period::Int = 4
    periods::Vector{Int} = Int[]
    brute_force::Union{Nothing, BruteForceConfig} = nothing
    continuation::Union{Nothing, ContinuationConfig} = nothing
    recon_steps::Int = 80
    recon_precision::Float64 = 1e-3
    adaptive_recon::Bool = false
    adaptive_recon_max_samples::Int = 24
    adaptive_recon_max_depth::Int = 1
    adaptive_recon_confidence_threshold::Float64 = 0.35
    adaptive_recon_closure_gradient_factor::Float64 = 0.75
    window_min_support::Int = 3
    window_merge_gap::Int = 1
    seed_points_per_window::Int = 4
    seed_box_padding::Float64 = 0.15
    skeleton_retry_budget::Int = 3
    continuation_retry_budget::Int = 3
    max_refinement_depth::Int = 2
    max_total_windows::Int = 64
    max_total_branches::Int = 128
    coverage_threshold::Float64 = 0.75
    branch_switching::Bool = false
    branch_switching_max_special_points::Int = 4
    branch_switching_max_branches::Int = 4
    branch_switching_window_fraction::Float64 = 0.08
    branch_switching_perturbation_scale::Float64 = 1e-3
    branch_switching_max_steps::Int = 120
    branch_switching_max_seed_candidates::Int = 6
    auto_refine_sparse_branches::Bool = true
    auto_refine_max_passes::Int = 1
    reuse_neighbor_seeds::Bool = false
    neighbor_seed_max_entries::Int = 64
    neighbor_seed_max_distance_fraction::Float64 = 0.15
    neighbor_seed_max_points::Int = 8
    threaded::Bool = Threads.nthreads() > 1
    cache_enabled::Bool = true
    time_budget_s::Union{Nothing, Float64} = nothing
    reseed::ReseedConfig = ReseedConfig(enabled=true)
    @assert auto_refine_max_passes >= 0 "AtlasConfig.auto_refine_max_passes must be >= 0"
end

"""
    RefinementConfig

Configuration for refining a specific interval of an existing continuation branch
with finer step sizes to capture missed details.

# Fields
- `from_param`: Start of the parameter interval to refine
- `to_param`: End of the parameter interval to refine
- `ds`: Step size for the refined region (smaller than original)
- `dsmax`: Maximum step size for the refined region
- `dsmin`: Minimum step size
- `max_steps`: Maximum continuation steps for the refined region
- `newton_tol`: Newton solver tolerance
- `newton_max_iter`: Maximum Newton iterations
- `detect_bifurcation`: Bifurcation detection level (0–3)
- `ode_jacobian_method`: Continuous-ODE Poincaré-map derivative method (`:finite_difference` or `:variational`)
"""
@with_kw struct RefinementConfig
    from_param::Float64
    to_param::Float64
    ds::Float64 = 0.001
    dsmax::Float64 = 0.005
    dsmin::Float64 = 1e-8
    max_steps::Int = 2000
    newton_tol::Float64 = 1e-10
    newton_max_iter::Int = 25
    detect_bifurcation::Int = 3
    a::Float64 = 0.5
    detect_fold::Bool = true
    save_sol_every_step::Int = 1
    ode_jacobian_method::Symbol = :finite_difference
    @assert isfinite(from_param) && isfinite(to_param) "RefinementConfig requires finite from_param/to_param"
    @assert from_param != to_param "RefinementConfig requires a non-empty parameter interval"
    @assert isfinite(ds) && ds > 0.0 "RefinementConfig.ds must be finite and > 0"
    @assert isfinite(dsmax) && dsmax > 0.0 "RefinementConfig.dsmax must be finite and > 0"
    @assert isfinite(dsmin) && dsmin > 0.0 "RefinementConfig.dsmin must be finite and > 0"
    @assert dsmax >= dsmin "RefinementConfig requires dsmax >= dsmin"
    @assert max_steps >= 1 "RefinementConfig.max_steps must be >= 1"
    @assert isfinite(newton_tol) && newton_tol > 0.0 "RefinementConfig.newton_tol must be finite and > 0"
    @assert newton_max_iter >= 1 "RefinementConfig.newton_max_iter must be >= 1"
    @assert 0 <= detect_bifurcation <= 3 "RefinementConfig.detect_bifurcation must be in 0:3"
    @assert isfinite(a) && a > 0.0 "RefinementConfig.a must be finite and > 0"
    @assert save_sol_every_step > 0 "RefinementConfig.save_sol_every_step must be > 0"
    @assert ode_jacobian_method in (:finite_difference, :variational) "RefinementConfig.ode_jacobian_method must be :finite_difference or :variational"
end

"""
    RobustChaosConfig

Configuration for `robust_chaos_certificate`. Nests the three analysis-layer configs and adds
per-layer threshold fractions for conservative pass/fail verdicts.

All three nested configs must describe the same physical parameter slice: matching parameter
indices, linked-parameter indices, non-varied base parameters, and Lyapunov/atlas intervals.
`basins.bif_param` must lie within `[lyapunov.param_min, lyapunov.param_max]`. The atlas must
carry non-`nothing` `brute_force` and `continuation` sub-configs.

# Fields
- `lyapunov`: `LyapunovConfig` for the parameter sweep; defines the certified interval
- `atlas`: `AtlasConfig` for the continuation-atlas window search; must have a `brute_force`
- `basins`: `BasinsConfig` for basin evaluation at `basins.bif_param`
- `min_lyapunov_positive_fraction`: Required fraction of *resolved* Lyapunov samples that must
  be `:chaotic_candidate`. Applied after the resolved-coverage threshold. Default `1.0`.
- `min_lyapunov_resolved_fraction`: Minimum fraction of all Lyapunov samples that must yield a
  finite estimate. If below this, the verdict is inconclusive unless failure is already
  provable. Default `1.0`.
- `min_chaotic_basin_fraction`: Required fraction of *resolved* basin seeds classified as
  chaotic. Applied after the basin resolved-coverage threshold. Default `1.0`.
- `min_basin_resolved_fraction`: Minimum fraction of all basin seeds that must be resolved
  (Lyapunov estimation succeeded or periodicity was detected). Default `1.0`.
"""
function _robust_float_repr_equal(a, b)
    a == b && return true
    a isa Real && b isa Real || return false
    af = Float64(a)
    bf = Float64(b)
    return abs(af - bf) <= 8 * eps(Float64) * max(abs(af), abs(bf), 1.0)
end

function _robust_config_base_params_match(
   lhs::AbstractVector,
   rhs::AbstractVector,
   varied_indices::AbstractVector{Int},
)
   # Parameter builders throughout the package define omitted trailing entries as zero;
   # compare against that same canonical zero-padded representation.
   n = max(length(lhs), length(rhs))
   varied = Set(varied_indices)
   for idx in 1:n
       idx in varied && continue
       left = idx <= length(lhs) ? lhs[idx] : 0.0
       right = idx <= length(rhs) ? rhs[idx] : 0.0
       _robust_float_repr_equal(left, right) || return false
   end
   return true
end

@with_kw struct RobustChaosConfig
    lyapunov::LyapunovConfig
    atlas::AtlasConfig
    basins::BasinsConfig
    min_lyapunov_positive_fraction::Float64 = 1.0
    min_lyapunov_resolved_fraction::Float64 = 1.0
    min_chaotic_basin_fraction::Float64      = 1.0
    min_basin_resolved_fraction::Float64     = 1.0
    @assert 0.0 <= min_lyapunov_positive_fraction <= 1.0 "RobustChaosConfig.min_lyapunov_positive_fraction must be in [0, 1]"
    @assert 0.0 <= min_lyapunov_resolved_fraction <= 1.0 "RobustChaosConfig.min_lyapunov_resolved_fraction must be in [0, 1]"
    @assert 0.0 <= min_chaotic_basin_fraction <= 1.0 "RobustChaosConfig.min_chaotic_basin_fraction must be in [0, 1]"
    @assert 0.0 <= min_basin_resolved_fraction <= 1.0 "RobustChaosConfig.min_basin_resolved_fraction must be in [0, 1]"
    @assert !isnothing(atlas.brute_force) "RobustChaosConfig: atlas must carry a non-nothing brute_force (required to locate periodic windows)"
    @assert !isnothing(atlas.continuation) "RobustChaosConfig: atlas must carry a non-nothing continuation (required to verify the full certificate interval)"
    @assert atlas.brute_force.param_index == lyapunov.param_index "RobustChaosConfig: atlas.brute_force.param_index must match lyapunov.param_index"
    @assert atlas.continuation.param_index == lyapunov.param_index "RobustChaosConfig: atlas.continuation.param_index must match lyapunov.param_index"
    @assert basins.param_index == lyapunov.param_index "RobustChaosConfig: basins.param_index must match lyapunov.param_index"
    @assert atlas.brute_force.param_min == lyapunov.param_min && atlas.brute_force.param_max == lyapunov.param_max "RobustChaosConfig: atlas.brute_force must cover exactly the lyapunov parameter interval"
    @assert atlas.continuation.p_min <= lyapunov.param_min && atlas.continuation.p_max >= lyapunov.param_max "RobustChaosConfig: atlas.continuation must cover the full lyapunov parameter interval"
    @assert sort(unique(atlas.brute_force.linked_param_indices)) == sort(unique(lyapunov.linked_param_indices)) "RobustChaosConfig: atlas.brute_force linked parameters must match lyapunov.linked_param_indices"
    @assert sort(unique(atlas.continuation.linked_param_indices)) == sort(unique(lyapunov.linked_param_indices)) "RobustChaosConfig: atlas.continuation linked parameters must match lyapunov.linked_param_indices"
    @assert _robust_config_base_params_match(
        atlas.brute_force.fixed_params,
        lyapunov.fixed_params,
        unique(vcat([lyapunov.param_index], lyapunov.linked_param_indices)),
    ) "RobustChaosConfig: atlas.brute_force and lyapunov must use the same non-varied base parameters"
    @assert all(
        idx <= length(basins.fixed_params) && basins.fixed_params[idx] == basins.bif_param
        for idx in lyapunov.linked_param_indices
    ) "RobustChaosConfig: basins.fixed_params must set every linked parameter to basins.bif_param"
    @assert _robust_config_base_params_match(
        basins.fixed_params,
        lyapunov.fixed_params,
        unique(vcat([lyapunov.param_index], lyapunov.linked_param_indices)),
    ) "RobustChaosConfig: basins and lyapunov must use the same non-varied base parameters"
    @assert lyapunov.param_min <= basins.bif_param <= lyapunov.param_max "RobustChaosConfig: basins.bif_param must lie within [lyapunov.param_min, lyapunov.param_max]"
end

"""
    BranchReachabilityConfig

Configuration for `branch_reachability` (multistability-aware continuation). Pairs a
set of continuation branches with a per-parameter basin initial-condition census so each coexisting
branch is reported with the basin fraction that actually reaches it, not merely as stable/unstable.

The analysis evaluates a basin census on the `(x, y)` initial-condition grid at each parameter knot
in `param_samples`, assigns every seed's terminal periodic orbit to a stable branch identity using
period-gated, phase-invariant state-space geometry, and accounts for every seed in exactly one
category (`matched` / `unmatched` / `aperiodic` / `diverged` / `unresolved` / `stability_mismatch` /
`outside_coverage`).

# Fields
- `param_samples`: Explicit list of parameter knots to evaluate (non-empty). Branch states are
  selected/solved at each knot; a branch is `covered` at a knot only when the knot lies within its
  continued parameter range (± `param_tolerance`).
- `param_index`: Parameter slot holding the varied continuation parameter
- `linked_param_indices`: Additional parameter slots set to the same varied value
- `base_params`: Base parameter vector (varied + linked slots are overwritten per knot). Empty ⇒ a
  zero vector sized to the referenced parameter slots.
- `x_min`, `x_max`, `x_steps`: Grid bounds and number of intervals for the first
  initial-condition axis (`x_steps + 1` seed points)
- `y_min`, `y_max`, `y_steps`: Grid bounds and number of intervals for the second
  initial-condition axis (`y_steps + 1` seed points)
- `x_index`, `y_index`: State dimensions the grid axes vary
- `ic_template`: Full-state fill for the non-gridded dimensions (empty ⇒ zeros)
- `max_period`: Maximum period to detect for a seed's terminal orbit
- `precision`: Amplitude-relative tolerance for period detection (matches `BasinsConfig.precision`)
- `iterations`: Total iterations per seed (must be at least `max_period + 1`)
- `divergence_cutoff`: State-amplitude cutoff flagging a diverged seed; `Inf` disables bailout
- `param_tolerance`: How far outside a branch's recorded parameter range a knot may lie and still be
  treated as covered (guards against using a branch point far from the requested sample)
- `match_tolerance`: Amplitude-relative phase-invariant distance below which a seed cycle matches a
  branch cycle
- `ambiguity_ratio`: A seed with two same-period stable branches within `match_tolerance` is
  `matched` to the nearest only when the best distance is at most `ambiguity_ratio` times the
  second-best; otherwise it is `unresolved`. Must lie in `(0, 1)`.
- `stability_tol`: Unit-circle tolerance for branch multiplier stability at each knot
- `newton_max_iter`, `newton_tol`: Newton settings for solving the exact branch fixed point at a knot
- `branch_ids`: Optional stable IDs aligned to the input branches; empty ⇒ deterministic
  `"branch-<k>"` fallback IDs by input order
- `threaded`: Thread the per-cell census (assignment is deterministic and thread-parity safe)

## Continuous-time (`ContinuousODE`) fields

For a `ContinuousODE`, seeds are full-state initial conditions, terminal orbits are detected on the
Poincaré section, and branch states are the section-*projected* coordinates. These configure the
shared ODE/Poincaré kernels (they are ignored for `DiscreteMap`):

- `ode_solver`: solver key resolved by `select_ode_solver` (`"auto"`, `"tsit5"`, `"rosenbrock23"`);
  `"auto"` is the stiff/non-stiff auto-switcher. Resolution is strict — an unknown key throws.
- `ode_reltol`, `ode_abstol`: integrator tolerances for section-return integration
- `min_crossing_time`: ignore section crossings before this time (drops the launch crossing)
- `ode_fd_step`: finite-difference step for the return-map Newton/Jacobian (projected fixed-point
  correction and multiplier stability)
- `ode_tmax`: maximum integration horizon per Poincaré segment; `Inf` (the default) uses the
  internal `tspan_hint`-scaled horizon, matching `basins_of_attraction`
"""
@with_kw struct BranchReachabilityConfig
    param_samples::Vector{Float64}
    param_index::Int = 1
    linked_param_indices::Vector{Int} = Int[]
    base_params::Vector{Float64} = Float64[]
    x_min::Float64
    x_max::Float64
    x_steps::Int = 50
    y_min::Float64
    y_max::Float64
    y_steps::Int = 50
    x_index::Int = 1
    y_index::Int = 2
    ic_template::Vector{Float64} = Float64[]
    max_period::Int = 10
    precision::Float64 = 1e-4
    iterations::Int = 1000
    divergence_cutoff::Float64 = Inf
    param_tolerance::Float64 = 1e-6
    match_tolerance::Float64 = 1e-3
    ambiguity_ratio::Float64 = 0.5
    stability_tol::Float64 = 1e-7
    newton_max_iter::Int = 25
    newton_tol::Float64 = 1e-10
    branch_ids::Vector{String} = String[]
    threaded::Bool = true
    ode_solver::String = "auto"
    ode_reltol::Float64 = 1e-8
    ode_abstol::Float64 = 1e-8
    min_crossing_time::Float64 = 1e-6
    ode_fd_step::Float64 = 1e-6
    ode_tmax::Float64 = Inf
    @assert !isempty(param_samples) "BranchReachabilityConfig.param_samples must be non-empty"
    @assert all(isfinite, param_samples) "BranchReachabilityConfig.param_samples must be finite"
    @assert param_index >= 1 "BranchReachabilityConfig.param_index must be >= 1"
    @assert all(>=(1), linked_param_indices) "BranchReachabilityConfig.linked_param_indices must be >= 1"
    @assert x_steps >= 1 "BranchReachabilityConfig.x_steps must be >= 1"
    @assert y_steps >= 1 "BranchReachabilityConfig.y_steps must be >= 1"
    @assert isfinite(x_min) && isfinite(x_max) && x_min < x_max "BranchReachabilityConfig requires finite x_min < x_max"
    @assert isfinite(y_min) && isfinite(y_max) && y_min < y_max "BranchReachabilityConfig requires finite y_min < y_max"
    @assert x_index >= 1 && y_index >= 1 && x_index != y_index "BranchReachabilityConfig requires distinct positive grid indices"
    @assert max_period >= 1 "BranchReachabilityConfig.max_period must be >= 1"
    @assert isfinite(precision) && precision > 0.0 "BranchReachabilityConfig.precision must be finite and > 0"
    @assert iterations >= max_period + 1 "BranchReachabilityConfig.iterations must be at least max_period + 1"
    @assert !isnan(divergence_cutoff) && divergence_cutoff > 0.0 "BranchReachabilityConfig.divergence_cutoff must be > 0 (use Inf to disable)"
    @assert isfinite(param_tolerance) && param_tolerance >= 0.0 "BranchReachabilityConfig.param_tolerance must be finite and >= 0"
    @assert isfinite(match_tolerance) && match_tolerance > 0.0 "BranchReachabilityConfig.match_tolerance must be finite and > 0"
    @assert 0.0 < ambiguity_ratio < 1.0 "BranchReachabilityConfig.ambiguity_ratio must lie in (0, 1)"
    @assert isfinite(stability_tol) && stability_tol >= 0.0 "BranchReachabilityConfig.stability_tol must be finite and >= 0"
    @assert newton_max_iter >= 1 "BranchReachabilityConfig.newton_max_iter must be >= 1"
    @assert isfinite(newton_tol) && newton_tol > 0.0 "BranchReachabilityConfig.newton_tol must be finite and > 0"
    @assert !isempty(ode_solver) "BranchReachabilityConfig.ode_solver must be a non-empty solver key (resolved by select_ode_solver)"
    @assert isfinite(ode_reltol) && ode_reltol > 0.0 "BranchReachabilityConfig.ode_reltol must be finite and > 0"
    @assert isfinite(ode_abstol) && ode_abstol > 0.0 "BranchReachabilityConfig.ode_abstol must be finite and > 0"
    @assert isfinite(min_crossing_time) && min_crossing_time >= 0.0 "BranchReachabilityConfig.min_crossing_time must be finite and >= 0"
    @assert isfinite(ode_fd_step) && ode_fd_step > 0.0 "BranchReachabilityConfig.ode_fd_step must be finite and > 0"
    @assert !isnan(ode_tmax) && ode_tmax > 0.0 "BranchReachabilityConfig.ode_tmax must be > 0 (use Inf for the tspan_hint-scaled horizon)"
end

"""
    RegimeBoundaryConfig

Configuration for `regime_boundary_distances` (deterministic regime-boundary margins over a
classified 2D operating map).

The margin field measures, for every *known-regime* cell, the physical Euclidean distance to the
nearest regime boundary; a cell whose regime cannot be resolved is marked invalid rather than being
silently assigned a physical regime.

# Fields
- `edge_policy`: how the sampled-domain edge is treated.
    - `:censored` (default, "open"): the reported margin is capped at the physical distance to the
      sampled edge and `edge_censored` is set, marking the value as a *lower bound* — a regime change
      may lie just outside the sampled window.
    - `:boundary`: the sampled edge is treated as a genuine regime boundary (leaving the domain is a
      mode change); the margin is capped at the edge distance but *not* flagged as censored.
    - `:ignore`: the edge is ignored; the margin is the raw distance to the nearest interior boundary
      cell (`Inf` when the region has no boundary in the sampled window).
- `aperiodic_is_regime`: when status-code evidence is supplied, treat detected-aperiodic
  (`:aperiodic_or_high_period`) cells as a distinct known regime (chaos is a physical mode). Default
  `true`.
- `diverged_is_regime`: when status-code evidence is supplied, treat diverged (`:diverged`) cells as
  a distinct known regime (escape/unbounded is a physical outcome). Default `true`.
"""
@with_kw struct RegimeBoundaryConfig
    edge_policy::Symbol = :censored
    aperiodic_is_regime::Bool = true
    diverged_is_regime::Bool = true
    @assert edge_policy in (:censored, :boundary, :ignore) "RegimeBoundaryConfig.edge_policy must be :censored, :boundary, or :ignore"
end

"""
    AbstractTolerance

Supertype for zero-inclusive component-tolerance distributions used by `tolerance_regime_map`.
A tolerance with scale `0` is an exact Dirac delta (no perturbation, no RNG
draw). See `UniformTolerance` and `GaussianTolerance`.
"""
abstract type AbstractTolerance end

"""
    UniformTolerance(half_width)

Symmetric uniform component tolerance: offsets are drawn from `U(-half_width, +half_width)`.
`half_width` must be finite and `>= 0`; `UniformTolerance(0.0)` is an exact Dirac delta.
"""
struct UniformTolerance <: AbstractTolerance
    half_width::Float64
    function UniformTolerance(half_width::Real)
        (isfinite(half_width) && half_width >= 0) || throw(ArgumentError(
            "UniformTolerance half_width must be finite and >= 0; got $half_width."))
        return new(Float64(half_width))
    end
end

"""
    GaussianTolerance(std)

Gaussian component tolerance: offsets are drawn from `Normal(0, std)`. `std` must be finite and
`>= 0`; `GaussianTolerance(0.0)` is an exact Dirac delta.
"""
struct GaussianTolerance <: AbstractTolerance
    std::Float64
    function GaussianTolerance(std::Real)
        (isfinite(std) && std >= 0) || throw(ArgumentError(
            "GaussianTolerance std must be finite and >= 0; got $std."))
        return new(Float64(std))
    end
end

"""
    ToleranceConfig

Configuration for `tolerance_regime_map` (probabilistic component-tolerance propagation through the
classified 2D operating map).

At each nominal grid cell the two parameters are independently perturbed by `tolerance_a` /
`tolerance_b`, and each perturbed operating point is classified by *nearest physical-grid-cell
lookup* over the supplied classified map (integer regime labels are never interpolated). The result
is a per-cell categorical distribution over regimes plus unknown and out-of-domain mass.

# Fields
- `tolerance_a`, `tolerance_b`: the per-parameter tolerance distributions (`AbstractTolerance`). A
  zero tolerance is an exact Dirac delta and draws no random offset on that axis. When *both* are
  zero the analysis returns the deterministic exact classification (probability `1`, entropy `0`,
  Wilson interval `[1, 1]`, `n_effective = 0`) with no sampling error.
- `n_samples`: Monte-Carlo samples per cell (`>= 1`).
- `seed`: global `UInt64` seed. Each cell mixes `(seed, i, j)` through a stable `UInt64` mixer into
  an independent `Xoshiro` stream, so results are bitwise-identical regardless of thread count or
  scheduling.
- `threaded`: thread the per-cell sweep (bitwise thread-parity safe).
- `aperiodic_is_regime`, `diverged_is_regime`: as in `RegimeBoundaryConfig` — whether
  detected-aperiodic / diverged status cells count as distinct known regimes when status evidence is
  supplied.
"""
@with_kw struct ToleranceConfig
    tolerance_a::AbstractTolerance = UniformTolerance(0.0)
    tolerance_b::AbstractTolerance = UniformTolerance(0.0)
    n_samples::Int = 2000
    seed::UInt64 = 0x00000000004469df
    threaded::Bool = true
    aperiodic_is_regime::Bool = true
    diverged_is_regime::Bool = true
    @assert n_samples >= 1 "ToleranceConfig.n_samples must be >= 1"
end
