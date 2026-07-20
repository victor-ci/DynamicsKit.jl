"""
Core type definitions for dynamical systems and results.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# System types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DynamicalSystem

Abstract supertype for all dynamical systems.
"""
abstract type DynamicalSystem end

"""
    SwitchingEvent

Optional guard metadata for nonsmooth maps or flows. `guard(x, p)` returns a
signed distance-like value; values near zero indicate a border/grazing event.
"""
struct SwitchingEvent
    name::String
    kind::Symbol
    guard::Function
    description::String
    tolerance::Float64
    scale::Float64
end

function SwitchingEvent(name::AbstractString, guard::Function;
                        kind::Symbol=:border_collision,
                        description::AbstractString="",
                        tolerance::Float64=1e-6,
                        scale::Float64=1.0)
    tolerance >= 0 || throw(ArgumentError("SwitchingEvent tolerance must be non-negative, got $tolerance."))
    scale > 0 || throw(ArgumentError("SwitchingEvent scale must be positive, got $scale."))
    return SwitchingEvent(String(name), kind, guard, String(description), tolerance, scale)
end

"""
    PoincareSection

Defines a Poincaré section for continuous-time systems.

# Fields
- `condition`: Function `(u, t, integrator) -> Real` — zero-crossing defines the section
- `direction`: `:up` (+1), `:down` (-1), or `:both` (0) — crossing direction to detect
- `projection`: Indices of state variables kept after section crossing (the Poincaré map coordinates)
- `template`: Optional full-state template used to lift projected section coordinates back to a full state
"""
struct PoincareSection{F}
    condition::F
    direction::Int
    projection::Vector{Int}
    template::Vector{Float64}
end

function PoincareSection(condition::F;
                         direction::Symbol=:up,
                         projection::Vector{Int}=[1, 3],
                         template::AbstractVector=Float64[]) where F
    dir_int = direction == :up ? 1 : direction == :down ? -1 : 0
    PoincareSection{F}(condition, dir_int, projection, collect(Float64, template))
end

"""
    DiscreteMap{F} <: DynamicalSystem

A discrete-time dynamical system (iterated map) `x_{n+1} = f(x_n, p)`.

# Fields
- `f`: Mapping function `(x::SVector, p::AbstractVector) -> SVector`
- `dim`: State space dimension
- `param_names`: Names of bifurcation parameters
- `name`: Human-readable system name
- `switching_events`: Optional guard functions for nonsmooth borders/grazing events
"""
struct DiscreteMap{F} <: DynamicalSystem
    f::F
    dim::Int
    param_names::Vector{Symbol}
    name::String
    switching_events::Vector{SwitchingEvent}
end

function DiscreteMap(f::F, dim::Int, param_names::Vector{Symbol}, name::String;
                     switching_events::AbstractVector{SwitchingEvent}=SwitchingEvent[]) where F
    DiscreteMap{F}(f, dim, param_names, name, collect(SwitchingEvent, switching_events))
end

"""
    ContinuousODE{F,S} <: DynamicalSystem

A continuous-time ODE system `du/dt = f(u, p, t)` with a Poincaré section.

# Fields
- `f`: ODE right-hand side `(du, u, p, t) -> nothing` (in-place)
- `dim`: Full state space dimension
- `section`: Poincaré section definition
- `param_names`: Names of bifurcation parameters
- `name`: Human-readable system name
- `tspan_hint`: Suggested integration time span for one return to section
- `default_initial_state`: Native default state used by brute-force and skeleton searches
- `default_params`: Native default parameter vector used when callers omit `params`
- `switching_events`: Optional guard functions for nonsmooth borders/grazing events
"""
struct ContinuousODE{F,S<:PoincareSection} <: DynamicalSystem
    f::F
    dim::Int
    section::S
    param_names::Vector{Symbol}
    name::String
    tspan_hint::Float64
    default_initial_state::Vector{Float64}
    default_params::Vector{Float64}
    switching_events::Vector{SwitchingEvent}
end

function ContinuousODE(f::F, dim::Int, section::S, param_names::Vector{Symbol}, name::String;
                       tspan_hint::Float64=20.0,
                       default_initial_state::AbstractVector=zeros(Float64, dim),
                       default_params::AbstractVector=Float64[],
                       switching_events::AbstractVector{SwitchingEvent}=SwitchingEvent[]) where {F, S<:PoincareSection}
    ContinuousODE{F,S}(
        f,
        dim,
        section,
        param_names,
        name,
        tspan_hint,
        collect(Float64, default_initial_state),
        collect(Float64, default_params),
        collect(SwitchingEvent, switching_events)
    )
end

state_dim(sys::DiscreteMap) = sys.dim
state_dim(sys::ContinuousODE) = length(sys.section.projection)

switching_events(sys::DynamicalSystem) = SwitchingEvent[]
switching_events(sys::DiscreteMap) = sys.switching_events
switching_events(sys::ContinuousODE) = sys.switching_events

function _switching_state_samples(state::AbstractVector{<:Real})
    return [collect(Float64, state)]
end

function _switching_state_samples(states::AbstractVector)
    isempty(states) && return Vector{Float64}[]
    first(states) isa Real && return [collect(Float64, states)]
    return [collect(Float64, state) for state in states]
end

function _switching_param_samples(params::AbstractVector{<:Real}, n::Int)
    base = collect(Float64, params)
    return [copy(base) for _ in 1:n]
end

function _switching_param_samples(params::AbstractVector, n::Int)
    if length(params) == n && all(item -> item isa AbstractVector, params)
        return [collect(Float64, item) for item in params]
    end
    return [collect(Float64, params) for _ in 1:n]
end

function _switching_guard_value(event::SwitchingEvent, state::AbstractVector, params::AbstractVector)
    value = event.guard(state, params)
    if value isa Number
        return Float64(value)
    elseif value isa AbstractVector
        finite = Float64[Float64(v) for v in value if isfinite(Float64(v))]
        isempty(finite) && return NaN
        return finite[argmin(abs.(finite))]
    end
    throw(ArgumentError("SwitchingEvent $(event.name) guard returned $(typeof(value)); expected a number or vector."))
end

function switching_event_diagnostics(sys::DynamicalSystem, states, params)
    events = switching_events(sys)
    samples = _switching_state_samples(states)
    param_samples = _switching_param_samples(params, length(samples))
    length(param_samples) == length(samples) || throw(ArgumentError(
        "Switching diagnostic parameter sample count $(length(param_samples)) does not match state sample count $(length(samples))."
    ))

    event_payloads = Vector{Dict{String, Any}}()
    nearest_event = nothing
    nearest_index = nothing
    nearest_distance = Inf
    nearest_normalized = Inf
    near_count = 0
    warnings = String[]

    for event in events
        min_distance = Inf
        min_normalized = Inf
        min_index = nothing
        failure_count = 0
        for idx in eachindex(samples)
            value = try
                _switching_guard_value(event, samples[idx], param_samples[idx])
            catch err
                failure_count += 1
                push!(warnings, "switching guard $(event.name) failed at sample $idx: $(sprint(showerror, err))")
                NaN
            end
            isfinite(value) || continue
            distance = abs(value)
            normalized = distance / event.scale
            if normalized < min_normalized
                min_distance = distance
                min_normalized = normalized
                min_index = idx
            end
        end

        near = isfinite(min_distance) && min_distance <= event.tolerance
        near && (near_count += 1)
        if min_normalized < nearest_normalized
            nearest_event = event.name
            nearest_index = min_index
            nearest_distance = min_distance
            nearest_normalized = min_normalized
        end
        push!(event_payloads, Dict{String, Any}(
            "name" => event.name,
            "kind" => String(event.kind),
            "description" => event.description,
            "tolerance" => event.tolerance,
            "scale" => event.scale,
            "near" => near,
            "minDistance" => isfinite(min_distance) ? min_distance : nothing,
            "minNormalizedDistance" => isfinite(min_normalized) ? min_normalized : nothing,
            "nearestSampleIndex" => min_index,
            "failureCount" => failure_count
        ))
    end
    unique!(warnings)

    return Dict{String, Any}(
        "eventCount" => length(events),
        "sampledPointCount" => length(samples),
        "nearEventCount" => near_count,
        "nearestEvent" => nearest_event,
        "nearestSampleIndex" => nearest_index,
        "minDistance" => isfinite(nearest_distance) ? nearest_distance : nothing,
        "minNormalizedDistance" => isfinite(nearest_normalized) ? nearest_normalized : nothing,
        "events" => event_payloads,
        "warnings" => warnings,
        "status" => isempty(events) ? "unavailable" : (isempty(warnings) ? "ok" : "warning")
    )
end

function switching_event_grid_summary(diagnostics::AbstractArray)
    min_distances = fill(NaN, size(diagnostics))
    nearest_events = fill("", size(diagnostics))
    near_event_cells = 0
    populated_cells = 0
    event_counts = Dict{String, Int}()
    warning_count = 0

    for idx in eachindex(diagnostics)
        diag = diagnostics[idx]
        diag isa AbstractDict || continue
        isempty(diag) && continue
        populated_cells += 1
        distance = get(diag, "minNormalizedDistance", nothing)
        if distance isa Real
            min_distances[idx] = Float64(distance)
        end
        event_name = get(diag, "nearestEvent", nothing)
        if event_name isa AbstractString
            nearest_events[idx] = String(event_name)
        end
        near_count = Int(get(diag, "nearEventCount", 0))
        near_count > 0 && (near_event_cells += 1)
        warnings = get(diag, "warnings", Any[])
        warnings isa AbstractVector && (warning_count += length(warnings))
        for event_diag in get(diag, "events", Any[])
            event_diag isa AbstractDict || continue
            get(event_diag, "near", false) == true || continue
            name = String(get(event_diag, "name", "unknown"))
            event_counts[name] = get(event_counts, name, 0) + 1
        end
    end

    finite_distances = Float64[value for value in min_distances if isfinite(value)]
    return Dict{String, Any}(
        "populatedCells" => populated_cells,
        "nearEventCells" => near_event_cells,
        "nearEventCounts" => event_counts,
        "nearestEvents" => nearest_events,
        "minNormalizedDistances" => min_distances,
        "globalMinNormalizedDistance" => isempty(finite_distances) ? nothing : minimum(finite_distances),
        "warningCount" => warning_count
    )
end

"""Project a full ODE state onto the coordinates retained by the Poincaré section."""
_project_section_state(section::PoincareSection, u) = collect(Float64, u[section.projection])

"""Lift projected Poincaré coordinates back to a full state using the section template when available."""
function _lift_section_state(section::PoincareSection, point, full_dim::Int)
    projected = collect(Float64, point)
    if length(projected) == full_dim
        return projected
    end

    length(projected) == length(section.projection) || error(
        "Projected section point has length $(length(projected)) but section projection expects $(length(section.projection)) coordinates"
    )

    full_state = if !isempty(section.template)
        length(section.template) == full_dim || error(
            "Poincaré section template has length $(length(section.template)) but system dimension is $full_dim"
        )
        copy(section.template)
    else
        zeros(Float64, full_dim)
    end

    full_state[section.projection] = projected
    return full_state
end

# ═══════════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BruteForceResult

Result of a brute-force bifurcation diagram computation.
"""
struct BruteForceResult
    params::Vector{Float64}           # Bifurcation parameter values for each point
    points::Matrix{Float64}           # State values (n_points × dim)
    system_name::String
    param_name::Symbol
    timestamp::DateTime
end

"""
    LyapunovDiagramResult

Result of a 1D largest-Lyapunov-Exponent sweep over one bifurcation parameter.
"""
struct LyapunovDiagramResult
    params::Vector{Float64}
    exponents::Vector{Float64}
    classifications::Vector{Symbol}
    estimation_statuses::Vector{Symbol}
    sample_counts::Vector{Int}
    neutral_tolerance::Float64
    system_name::String
    param_name::Symbol
    timestamp::DateTime
end

"""
    BasinsResult

Result of a basins of attraction computation. For each (x, y) grid point,
stores the detected periodicity at a fixed parameter value.
"""
struct BasinsResult
    x_grid::Vector{Float64}           # x-axis grid values (initial condition 1)
    y_grid::Vector{Float64}           # y-axis grid values (initial condition 2)
    periodicity::Matrix{Int}          # Detected period at each (x, y) — size (nx × ny)
    bif_param::Float64                # Fixed bifurcation parameter value
    max_period::Int                   # Maximum period searched for
    system_name::String
    timestamp::DateTime
    x_index::Int
    y_index::Int
    ic_template::Vector{Float64}
end

function BasinsResult(x_grid::Vector{Float64},
                      y_grid::Vector{Float64},
                      periodicity::Matrix{Int},
                      bif_param::Float64,
                      max_period::Int,
                      system_name::String,
                      timestamp::DateTime)
    return BasinsResult(x_grid, y_grid, periodicity, bif_param, max_period, system_name, timestamp, 1, 2, Float64[])
end

"""
    LyapunovSpectrumResult

Full Lyapunov spectrum at a single operating point, from the Benettin/QR
(tangent-space) method. `exponents` are ordered from largest to smallest;
`convergence` stores the running finite-time estimate after each accumulated
reorthonormalization interval (size `sample_count` × `length(exponents)`), so the
approach to the reported values can be inspected. `total_time` is the accumulation
horizon: iteration count for discrete maps, elapsed time for continuous flows.
`kind` is `:discrete_map` or `:continuous_flow`.
"""
struct LyapunovSpectrumResult
    exponents::Vector{Float64}
    convergence::Matrix{Float64}
    estimation_status::Symbol
    sample_count::Int
    total_time::Float64
    kind::Symbol
    params::Vector{Float64}
    system_name::String
    timestamp::DateTime
end

"""
    LyapunovFieldResult

Largest-Lyapunov-Exponent field over a 2D parameter sweep.
"""
struct LyapunovFieldResult
    a_grid::Vector{Float64}
    b_grid::Vector{Float64}
    exponents::Matrix{Float64}
    classification_status_codes::Matrix{Int}
    estimation_status_codes::Matrix{Int}
    sample_counts::Matrix{Int}
    neutral_tolerance::Float64
    system_name::String
    param_names::Tuple{Symbol, Symbol}
    timestamp::DateTime
end

"""
    BifurcationMapResult

Result of a 2D bifurcation map (two-parameter sweep). For each (a, b) grid point,
stores the detected periodicity of the attractor.
"""
struct BifurcationMapResult
    a_grid::Vector{Float64}           # First parameter grid values
    b_grid::Vector{Float64}           # Second parameter grid values
    periodicity::Matrix{Int}          # Detected period at each (a, b) — size (na × nb)
    max_period::Int                   # Maximum period searched for
    system_name::String
    param_names::Tuple{Symbol, Symbol}
    timestamp::DateTime
    lyapunov::Union{Nothing, LyapunovFieldResult}
end

function BifurcationMapResult(a_grid::Vector{Float64},
                              b_grid::Vector{Float64},
                              periodicity::Matrix{Int},
                              max_period::Int,
                              system_name::String,
                              param_names::Tuple{Symbol, Symbol},
                              timestamp::DateTime)
    return BifurcationMapResult(a_grid, b_grid, periodicity, max_period, system_name, param_names, timestamp, nothing)
end

"""
    PhasePortraitResult

Result of a continuous-time phase portrait integration.
"""
struct PhasePortraitResult
    t::Vector{Float64}                # Retained trajectory timestamps
    trajectory::Matrix{Float64}       # Retained full-state trajectory (n_points × dim)
    poincare_points::Matrix{Float64}  # Full-state Poincaré section crossings (n_points × dim)
    params::Vector{Float64}
    system_name::String
    state_names::Vector{Symbol}
    timestamp::DateTime
end

"""
    PowerSpectrumResult

Time-domain tail and one-sided power spectrum for one observed state coordinate.
"""
struct PowerSpectrumResult
    t::Vector{Float64}
    signal::Vector{Float64}
    frequency::Vector{Float64}
    power::Vector{Float64}
    params::Vector{Float64}
    state_index::Int
    system_name::String
    timestamp::DateTime
end

"""
    Codim2CurveResult

Tracked codimension-2 bifurcation curve assembled from repeated 1D continuation
slices across a second parameter.
"""
struct Codim2CurveResult
    primary_values::Vector{Float64}
    secondary_values::Vector{Float64}
    valid_mask::BitVector
    raw_candidates::Vector{Vector{Float64}}
    candidate_sources::Vector{Symbol}
    slice_statuses::Vector{Symbol}
    slice_messages::Vector{String}
    slice_point_counts::Vector{Int}
    slice_special_point_counts::Vector{Int}
    bifurcation_kind::Symbol
    period::Int
    system_name::String
    param_names::Tuple{Symbol, Symbol}
    engine::Symbol
    tracking_anchor::Float64
    tracking_tolerance::Float64
    timestamp::DateTime
end

"""
    Codim2ContinuationResult

Codimension-2 bifurcation curve obtained by continuing a minimally augmented
defining system (fixed point + bifurcation condition) in the secondary
parameter. Samples are ordered along the continuation arc, so the curve may
fold back in either parameter. `states` and `defining_vectors` hold one column
per sample (the fixed point and the defining eigenvector; for `:ns` curves the
defining vector is complex and `defining_vectors`/`defining_vectors_imag`
carry its real/imaginary parts while `phase_angles` holds the multiplier angle
θ with μ = e^{iθ}; for `:pd`/`:fold` curves `defining_vectors_imag` is empty
and `phase_angles` is NaN). Diagnostics vectors are empty/NaN when
`Codim2Config.curve_diagnostics` was `false`.
"""
struct Codim2ContinuationResult
    primary_values::Vector{Float64}
    secondary_values::Vector{Float64}
    states::Matrix{Float64}
    defining_vectors::Matrix{Float64}
    defining_vectors_imag::Matrix{Float64}
    phase_angles::Vector{Float64}
    fixed_point_residuals::Vector{Float64}
    multipliers::Vector{Vector{ComplexF64}}
    curve_fold_secondary_values::Vector{Float64}
    seed_primary::Float64
    seed_secondary::Float64
    bifurcation_kind::Symbol
    period::Int
    system_name::String
    param_names::Tuple{Symbol, Symbol}
    engine::Symbol
    timestamp::DateTime
end

"""
    BranchResult

Result of a continuation branch computation.
"""
struct BranchResult
    branch::Any                       # BifurcationKit ContResult
    period::Int
    system_name::String
    param_name::Symbol
    timestamp::DateTime
end

"""
    OrbitBranchResult

Result of a full-orbit periodic-orbit continuation by collocation. Unlike `BranchResult`
(a continued fixed point of the Poincaré return map), this carries the whole continued
family of time-parameterized orbits: `branch` is the BifurcationKit periodic-orbit
`ContResult` and `coll` the collocation problem needed to decode each stored orbit and
its period. `period` is the requested topological period (number of section returns per
cycle). `base_params`, `param_index`, and `linked_param_indices` record the
parameter-injection mapping so orbit stability can be evaluated at any branch point.
"""
struct OrbitBranchResult
    branch::Any                       # BifurcationKit periodic-orbit ContResult
    coll::Any                         # BifurcationKit Collocation problem
    period::Int
    base_params::Vector{Float64}
    param_index::Int
    linked_param_indices::Vector{Int}
    system_name::String
    param_name::Symbol
    method::Symbol
    timestamp::DateTime
end

"""
    MapNormalForm

Plain-data normal-form classification for a map bifurcation. `coefficient_name` is `:b`
for a fold, `:c` for a flip, and `:d` for a Neimark-Sacker point. `coefficient` is
`nothing` whenever the coefficient cannot be computed reliably; `status` and
`criticality` then state why no classification was made. `convention` records the
normalization and formula convention used for the coefficient.
"""
struct MapNormalForm
    kind::Symbol
    coefficient_name::Symbol
    coefficient::Union{Nothing, Float64}
    criticality::Symbol
    status::Symbol
    convention::String
end

"""
    MapSpecialPoint

A period-doubling (`:pd`, multiplier crossing -1), fold (`:fold`, multiplier crossing
+1), or Neimark-Sacker (`:ns`, a complex-conjugate pair crossing the unit circle)
special point located on a continued map / Poincare return-map branch.
`critical_multiplier` is the representative critical multiplier; `test_value` is the
map test function at the located point; `converged` reports whether refinement reached
the bifurcation tolerance. Simultaneous Neimark-Sacker pairs are represented by separate
points distinguished by `critical_multiplier`. `normal_form` is the optional local
classification.
"""
struct MapSpecialPoint
    kind::Symbol
    param::Float64
    state::Vector{Float64}
    multipliers::Vector{ComplexF64}
    critical_multiplier::ComplexF64
    test_value::Float64
    period::Int
    converged::Bool
    normal_form::Union{Nothing, MapNormalForm}
end

MapSpecialPoint(kind::Symbol, param::Real, state::AbstractVector,
                multipliers::AbstractVector, critical_multiplier::Number,
                test_value::Real, period::Integer, converged::Bool) =
    MapSpecialPoint(kind, Float64(param), collect(Float64, state),
                    collect(ComplexF64, multipliers), ComplexF64(critical_multiplier),
                    Float64(test_value), Int(period), converged, nothing)

"""
    Codim2SpecialPoint

A codimension-2 bifurcation point detected on a `Codim2ContinuationResult` locus via
`codim2_special_points`. Supported kinds and their applicable loci:

- `:cusp` — fold locus turns in the primary parameter (`:fold` locus)
- `:generalized_flip` — flip normal-form coefficient `c` changes sign (`:pd` locus)
- `:fold_flip` — a non-tracked second multiplier reaches ∓1 (`:pd` or `:fold` locus)
- `:resonance_1_1` — tracked unit-circle angle crosses 0 mod 2π (`:ns` locus)
- `:resonance_1_2` — tracked unit-circle angle crosses π mod 2π (`:ns` locus)
- `:bautin` — NS normal-form coefficient `d` changes sign (`:ns` locus)

`locus_kind` records which locus produced this point. `primary_param` and
`secondary_param` are the two-parameter coordinates of the located point. `state`
is the fixed-point state vector; `multipliers` is the full multiplier set (empty when
`Codim2Config.curve_diagnostics` was `false`). `test_value` is the scalar test
function at the located point (zero for interpolated detections). `converged` is
`true` only when Newton correction to the defining locus succeeded. `status` records
the resolution:

- `:interpolated` — point located by linear interpolation between two bracketing locus
  samples; state is not corrected back to the locus.
- `:sampled` — reported directly from the closest locus sample; no interpolation.
- `:unavailable` — detection was requested but required data (multipliers, normal-form
  coefficients) were absent or numerically unstable.

`normal_form` carries the codim-1 normal form from the nearest bracketing sample when
available (generalized-flip/bautin only); full codim-2 normal-form classification is
out of scope.
"""
struct Codim2SpecialPoint
    kind::Symbol
    locus_kind::Symbol
    primary_param::Float64
    secondary_param::Float64
    state::Vector{Float64}
    multipliers::Vector{ComplexF64}
    test_value::Float64
    period::Int
    converged::Bool
    status::Symbol
    normal_form::Union{Nothing, MapNormalForm}
end

"""
    BifurcationResult

Aggregated result containing brute-force and/or branch data.
"""
struct BifurcationResult
    brute_force::Union{BruteForceResult, Nothing}
    branches::Vector{BranchResult}
    system_name::String
    timestamp::DateTime
end

"""
    StableWindowEvidence

Plain-data record of a stable low-period orbit found on an atlas branch within a
`robust_chaos_certificate` search. Any such evidence means the region may not be
robustly chaotic under the configured search.

# Fields
- `branch_id`: Atlas branch record identifier (`AtlasBranchRecord.id`)
- `window_id`: Atlas window identifier (`AtlasWindow.id`)
- `period`: Period of the orbit
- `param_min`: Lower bound of the parameter sub-interval over which stability was confirmed
- `param_max`: Upper bound of that interval
- `stable_sample_count`: Number of branch points on this record classified as stable
"""
struct StableWindowEvidence
    branch_id::String
    window_id::String
    period::Int
    param_min::Float64
    param_max::Float64
    stable_sample_count::Int
end

"""
    RobustChaosCertificate

Conservative certificate for a robust-chaos region, produced by `robust_chaos_certificate`.
Three analysis layers — Lyapunov sweep, continuation-atlas window search, and basin-of-attraction
Lyapunov re-estimation — are orchestrated and issue layered verdicts.

**Bounded semantics**: certification is bounded to the configured sampling, search periods, and
parameter range. It does **not** mathematically prove the absence of stable orbits or chaotic
behaviour outside the configured search.

# Overall verdicts
- `:certified`: all three layers passed with sufficient resolved coverage
- `:fragile`: at least one layer found positive evidence against chaos (stable periodic orbit, or
  chaotic fraction definitively too low)
- `:inconclusive`: no layer failed but coverage was insufficient to issue a certificate

# Robustness score ∈ [0, 1]
Conservative minimum of three layer scores:
- **Lyapunov**: `lyapunov_positive_fraction × lyapunov_resolved_fraction`
- **Atlas**: `0` if any stable evidence was found; otherwise the atlas coverage/effort measure
- **Basin**: `basin_chaotic_fraction × basin_resolved_fraction`

# Fields
- `param_min`, `param_max`: Certified interval (from `lyapunov.param_min/max`)
- `system_name`: System name
- `param_index`: Bifurcation parameter index shared by all three layers
- `lyapunov_verdict`, `atlas_verdict`, `basin_verdict`, `overall_verdict`: Layer verdicts
- `lyapunov_positive_fraction`: Fraction of *resolved* Lyapunov samples that are `:chaotic_candidate`
- `lyapunov_resolved_fraction`: Fraction of all Lyapunov samples with a finite estimate
- `lyapunov_min_resolved_exponent`: Minimum exponent among resolved samples (can be negative)
- `lyapunov_n_total`, `lyapunov_n_resolved`, `lyapunov_n_positive`: Lyapunov sample counts
- `atlas_searched_periods`: Periods searched by the atlas recon
- `atlas_search_complete`: Whether the atlas ran to completion without time-budget exhaustion
- `atlas_coverage_effort`: Bounded [0, 1] measure of atlas window coverage effort
- `atlas_n_windows`, `atlas_n_covered`, `atlas_n_partial`, `atlas_n_unresolved`, `atlas_n_gaps`
- `atlas_unresolved_stability_count`: Branch samples whose stability could not be evaluated
- `stable_evidence`: `StableWindowEvidence` records (empty when atlas layer passes or is inconclusive)
- `basin_param`: Representative parameter value used for basin evaluation
- `basin_chaotic_fraction`: Fraction of *resolved* basin seeds classified as chaotic
- `basin_resolved_fraction`: Fraction of all basin seeds that were resolved
- `basin_n_total`, `basin_n_resolved`, `basin_n_chaotic`: Basin seed counts
- `basin_class_counts`: Classification → count for all basin seeds
- `robustness_score`: Conservative [0, 1] score (see above)
- `certificate_items`: Ordered audit trail (Vector of plain-data Dicts)
- `timestamp`: When the certificate was computed
"""
struct RobustChaosCertificate
    param_min::Float64
    param_max::Float64
    system_name::String
    param_index::Int

    lyapunov_verdict::Symbol
    atlas_verdict::Symbol
    basin_verdict::Symbol
    overall_verdict::Symbol

    lyapunov_positive_fraction::Float64
    lyapunov_resolved_fraction::Float64
    lyapunov_min_resolved_exponent::Float64
    lyapunov_n_total::Int
    lyapunov_n_resolved::Int
    lyapunov_n_positive::Int

    atlas_searched_periods::Vector{Int}
    atlas_search_complete::Bool
    atlas_coverage_effort::Float64
    atlas_n_windows::Int
    atlas_n_covered::Int
    atlas_n_partial::Int
    atlas_n_unresolved::Int
    atlas_n_gaps::Int
    atlas_unresolved_stability_count::Int
    stable_evidence::Vector{StableWindowEvidence}

    basin_param::Float64
    basin_chaotic_fraction::Float64
    basin_resolved_fraction::Float64
    basin_n_total::Int
    basin_n_resolved::Int
    basin_n_chaotic::Int
    basin_class_counts::Dict{Symbol, Int}

    robustness_score::Float64
    certificate_items::Vector{Dict{String, Any}}
    timestamp::DateTime
end

const _BORDER_COLLISION_CONVENTION =
    "Feigin/Simpson/di Bernardo border-collision classification for continuous piecewise-smooth " *
    "maps. Persistence vs nonsmooth fold from sign(det(I-A_L)*det(I-A_R)); companion 2q-cycle " *
    "creation from sign(det(I+A_L)*det(I+A_R)). A_L, A_R are the one-sided q-return Jacobians " *
    "(guard-negative and guard-positive branch at the colliding phase) and differ only at the " *
    "colliding phase. Stability is reported separately; the classification never infers chaos, " *
    "robust chaos, period-adding, torus creation, or any spectral-radius verdict."

"""
    BorderCollisionClassification

Plain-data classification of a border-collision bifurcation (BCB) of a **continuous**
piecewise-smooth map, following the Feigin/Simpson/di Bernardo determinant-sign theory.

`A_L`/`A_R` are the two one-sided ordered *q*-return Jacobians at the colliding phase
(`jacobian_L` = guard-negative branch, `jacobian_R` = guard-positive branch); for a
period-`q` collision they differ only in the colliding phase's one-sided factor.

# Scenario (only when `status === :ok`)
- `:persistence` — `sign(det(I-A_L)·det(I-A_R)) > 0`, no companion cycle
- `:nonsmooth_fold` — `sign(det(I-A_L)·det(I-A_R)) < 0`, no companion cycle
- `:persistence_with_companion_cycle` — persistence plus a companion `2q`-cycle
  (`sign(det(I+A_L)·det(I+A_R)) < 0`)
- `:nonsmooth_fold_with_companion_cycle` — nonsmooth fold plus a companion `2q`-cycle
- `:undetermined` — no scenario issued (see `status`)

# Status
- `:ok` — a generic scenario was issued
- `:noncontinuous` — the map is not continuous at the border (rank-one continuity/normal
  condition violated); classification is refused for discontinuous maps
- `:nontransversal` — the border/eigenvalue crossing is not transverse
- `:degenerate` — a `+1` or `-1` eigenvalue makes the determinant sign(s) ambiguous
- `:multiple_border_phases` — more than one phase/guard component sits on the border
- `:invalid` — non-square, mismatched, or non-finite Jacobians, or an unusable switching normal
- `:unavailable` — the one-sided return Jacobians could not be formed

Determinant invariants (`det_I_minus_*`, `det_I_plus_*`) and their sign products are the
robust classifiers. The `sigma_plus_*`/`sigma_minus_*` counts (real eigenvalues `> 1` and
`< -1`) are tolerance-aware diagnostics; `sigma_reliable` flags whether they are trustworthy.
Stability (`stable_L`/`stable_R`, `spectral_radius_*`) is reported separately and is `nothing`
when marginal. Companion-cycle fields are populated only when `status === :ok`;
`companion_admissible` remains `nothing` because admissibility is not decidable from the return
Jacobians alone.
"""
struct BorderCollisionClassification
    scenario::Symbol
    status::Symbol
    period::Int

    det_I_minus_L::Float64
    det_I_minus_R::Float64
    det_I_plus_L::Float64
    det_I_plus_R::Float64
    persistence_product::Float64
    persistence_sign::Int
    companion_product::Float64
    companion_sign::Int

    sigma_plus_L::Union{Nothing, Int}
    sigma_plus_R::Union{Nothing, Int}
    sigma_minus_L::Union{Nothing, Int}
    sigma_minus_R::Union{Nothing, Int}
    sigma_reliable::Bool

    spectrum_L::Vector{ComplexF64}
    spectrum_R::Vector{ComplexF64}

    stable_L::Union{Nothing, Bool}
    stable_R::Union{Nothing, Bool}
    spectral_radius_L::Float64
    spectral_radius_R::Float64

    companion_exists::Union{Nothing, Bool}
    companion_admissible::Union{Nothing, Bool}
    companion_stable::Union{Nothing, Bool}
    companion_spectral_radius::Union{Nothing, Float64}
    companion_multipliers::Vector{ComplexF64}

    transversal::Union{Nothing, Bool}
    transversality_measure::Union{Nothing, Float64}

    continuous::Union{Nothing, Bool}
    continuity_residual::Union{Nothing, Float64}
    continuity_tolerance::Float64

    generic::Bool

    jacobian_L::Matrix{Float64}
    jacobian_R::Matrix{Float64}

    inference::String
    warnings::Vector{String}
    convention::String
end

function BorderCollisionClassification(;
        scenario::Symbol,
        status::Symbol,
        period::Integer=1,
        det_I_minus_L::Real=NaN,
        det_I_minus_R::Real=NaN,
        det_I_plus_L::Real=NaN,
        det_I_plus_R::Real=NaN,
        persistence_product::Real=NaN,
        persistence_sign::Integer=0,
        companion_product::Real=NaN,
        companion_sign::Integer=0,
        sigma_plus_L::Union{Nothing, Integer}=nothing,
        sigma_plus_R::Union{Nothing, Integer}=nothing,
        sigma_minus_L::Union{Nothing, Integer}=nothing,
        sigma_minus_R::Union{Nothing, Integer}=nothing,
        sigma_reliable::Bool=false,
        spectrum_L::AbstractVector=ComplexF64[],
        spectrum_R::AbstractVector=ComplexF64[],
        stable_L::Union{Nothing, Bool}=nothing,
        stable_R::Union{Nothing, Bool}=nothing,
        spectral_radius_L::Real=NaN,
        spectral_radius_R::Real=NaN,
        companion_exists::Union{Nothing, Bool}=nothing,
        companion_admissible::Union{Nothing, Bool}=nothing,
        companion_stable::Union{Nothing, Bool}=nothing,
        companion_spectral_radius::Union{Nothing, Real}=nothing,
        companion_multipliers::AbstractVector=ComplexF64[],
        transversal::Union{Nothing, Bool}=nothing,
        transversality_measure::Union{Nothing, Real}=nothing,
        continuous::Union{Nothing, Bool}=nothing,
        continuity_residual::Union{Nothing, Real}=nothing,
        continuity_tolerance::Real=NaN,
        generic::Bool=false,
        jacobian_L::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
        jacobian_R::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
        inference::AbstractString="",
        warnings::AbstractVector=String[],
        convention::AbstractString=_BORDER_COLLISION_CONVENTION)
    return BorderCollisionClassification(
        scenario, status, Int(period),
        Float64(det_I_minus_L), Float64(det_I_minus_R),
        Float64(det_I_plus_L), Float64(det_I_plus_R),
        Float64(persistence_product), Int(persistence_sign),
        Float64(companion_product), Int(companion_sign),
        sigma_plus_L === nothing ? nothing : Int(sigma_plus_L),
        sigma_plus_R === nothing ? nothing : Int(sigma_plus_R),
        sigma_minus_L === nothing ? nothing : Int(sigma_minus_L),
        sigma_minus_R === nothing ? nothing : Int(sigma_minus_R),
        sigma_reliable,
        collect(ComplexF64, spectrum_L), collect(ComplexF64, spectrum_R),
        stable_L, stable_R, Float64(spectral_radius_L), Float64(spectral_radius_R),
        companion_exists, companion_admissible, companion_stable,
        companion_spectral_radius === nothing ? nothing : Float64(companion_spectral_radius),
        collect(ComplexF64, companion_multipliers),
        transversal, transversality_measure === nothing ? nothing : Float64(transversality_measure),
        continuous, continuity_residual === nothing ? nothing : Float64(continuity_residual),
        Float64(continuity_tolerance),
        generic,
        Matrix{Float64}(jacobian_L), Matrix{Float64}(jacobian_R),
        String(inference), collect(String, warnings), String(convention))
end

"""
    BorderCollisionPoint

A located border-collision of a period-`q` cycle on a continued map / return-map branch,
together with its `BorderCollisionClassification`.

# Fields
- `param`: Bifurcation parameter value at the collision (`NaN` when classifying a bare cycle
  without a swept parameter)
- `orbit`: The reconstructed `q`-cycle phases at the collision
- `colliding_phase`: 1-based index of the single phase sitting on the border
- `itinerary`: Sign of the colliding guard component at each phase (`0` at the colliding phase)
- `event_name`: Name of the `SwitchingEvent` whose guard collided
- `guard_component`: 1-based index of the guard component that collided (`1` for a scalar guard)
- `guard_values`: Colliding guard-component value at each phase
- `period`: Cycle period `q`
- `classification`: The determinant-sign classification
- `converged`: Whether both the collision refinement and the one-sided Jacobians converged
"""
struct BorderCollisionPoint
    param::Float64
    orbit::Vector{Vector{Float64}}
    colliding_phase::Int
    itinerary::Vector{Int}
    event_name::String
    guard_component::Int
    guard_values::Vector{Float64}
    period::Int
    classification::BorderCollisionClassification
    converged::Bool
end

function BorderCollisionPoint(param::Real, orbit::AbstractVector, colliding_phase::Integer,
                              itinerary::AbstractVector, event_name::AbstractString,
                              guard_component::Integer, guard_values::AbstractVector,
                              period::Integer, classification::BorderCollisionClassification,
                              converged::Bool)
    return BorderCollisionPoint(
        Float64(param),
        [collect(Float64, phase) for phase in orbit],
        Int(colliding_phase),
        collect(Int, itinerary),
        String(event_name),
        Int(guard_component),
        collect(Float64, guard_values),
        Int(period),
        classification,
        converged)
end
