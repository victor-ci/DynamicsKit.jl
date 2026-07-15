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
    BifurcationResult

Aggregated result containing brute-force and/or branch data.
"""
struct BifurcationResult
    brute_force::Union{BruteForceResult, Nothing}
    branches::Vector{BranchResult}
    system_name::String
    timestamp::DateTime
end
