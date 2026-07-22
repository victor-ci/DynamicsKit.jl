# Named map-status codes: the single compile-time source of truth for this code space. Every literal
# comparison against a map-status code (`_gpu_map_status_code`, `_map_status_code_from_state_code`, the
# Dict below) is built from these constants instead of repeating bare integers, so the code space can
# only drift by editing this block — the isbits `Int`/`@inline` hot paths reachable from `@kernel`
# bodies (`gpu_kernels.jl`) are unaffected: these are still plain `Int` constants, not `Symbol`s.
const _MAP_STATUS_UNKNOWN = 0
const _MAP_STATUS_PERIODIC = 1
const _MAP_STATUS_APERIODIC_OR_HIGH_PERIOD = 2
const _MAP_STATUS_DIVERGED = 3
const _MAP_STATUS_INSUFFICIENT_CROSSINGS = 4
const _MAP_STATUS_INTEGRATION_FAILED = 5
const _MAP_STATUS_INVALID_STATE = 6

const _MAP_STATUS_CODE_BY_SYMBOL = Dict{Symbol, Int}(
    :unknown => _MAP_STATUS_UNKNOWN,
    :periodic => _MAP_STATUS_PERIODIC,
    :aperiodic_or_high_period => _MAP_STATUS_APERIODIC_OR_HIGH_PERIOD,
    :diverged => _MAP_STATUS_DIVERGED,
    :insufficient_crossings => _MAP_STATUS_INSUFFICIENT_CROSSINGS,
    :integration_failed => _MAP_STATUS_INTEGRATION_FAILED,
    :invalid_state => _MAP_STATUS_INVALID_STATE
)

const _MAP_STATUS_LABEL_BY_CODE = Dict{Int, String}(
    code => String(status) for (status, code) in _MAP_STATUS_CODE_BY_SYMBOL
)

_map_status_code(status::Symbol) = get(_MAP_STATUS_CODE_BY_SYMBOL, status, _MAP_STATUS_CODE_BY_SYMBOL[:unknown])
_map_status_label(code::Integer) = get(_MAP_STATUS_LABEL_BY_CODE, Int(code), "unknown")
_map_status_symbol(code::Integer) = Symbol(_map_status_label(code))

# Dict-free mirror of `_map_status_code`, kept in lockstep with `_MAP_STATUS_CODE_BY_SYMBOL` (both are
# built from the same named constants above) and asserted equal to it by test for every known status.
# `Symbol` is not `isbits` and must not flow through GPU-kernel-compiled code (see `gpu_kernels.jl`), so
# the numeric cores below never build one — this conversion is only needed at a CPU host boundary that
# still has a `Symbol` status in hand.
@inline function _gpu_map_status_code(status::Symbol)
    status === :periodic && return _MAP_STATUS_PERIODIC
    status === :aperiodic_or_high_period && return _MAP_STATUS_APERIODIC_OR_HIGH_PERIOD
    status === :diverged && return _MAP_STATUS_DIVERGED
    status === :insufficient_crossings && return _MAP_STATUS_INSUFFICIENT_CROSSINGS
    status === :integration_failed && return _MAP_STATUS_INTEGRATION_FAILED
    status === :invalid_state && return _MAP_STATUS_INVALID_STATE
    return _MAP_STATUS_UNKNOWN
end

function _map_seed_semantics_label(seed_mode::Symbol)
    seed_mode == :fixed && return "fixed_initial_condition"
    seed_mode == :neighbor_full && return "path_following_full_transient"
    seed_mode == :neighbor_accelerated && return "path_following_reduced_transient"
    return "unknown"
end

# GPU-safe (isbits) state-status codes: the *only* representation the numeric cores below use
# internally for "is this state still usable" (as opposed to the public per-cell map-status codes in
# `_MAP_STATUS_CODE_BY_SYMBOL`, which have a different, larger code space). `Symbol` is not `isbits`
# (see `isbits(:ok) === false`) and GPUCompiler requires device-reachable values/return types to be
# `isbits`; even though `KernelAbstractions.CPU()` tolerates a boxed `Symbol` happily, a real GPU vendor
# backend is not guaranteed to. Every function reachable from a `@kernel` body (`_detect_discrete_map_period_core`,
# `_estimate_discrete_map_largest_lyapunov_core`, `_map_lyapunov_classification`, `_lyapunov_point_classification`)
# is built on these codes end-to-end; conversion to/from the public `Symbol` statuses happens only at
# the CPU host-wrapper boundary (`_detect_discrete_map_period`, `_estimate_discrete_map_largest_lyapunov`, …).
const _STATE_CODE_OK = 0
const _STATE_CODE_DIVERGED = 1
const _STATE_CODE_INVALID = 2

@inline function _map_state_status_code(state, cutoff::Float64)
    !all(isfinite, state) && return _STATE_CODE_INVALID
    return isfinite(cutoff) && any(abs(value) > cutoff for value in state) ? _STATE_CODE_DIVERGED : _STATE_CODE_OK
end

# Translate the 3-value internal state code into the map-status code space (`_MAP_STATUS_CODE_BY_SYMBOL`),
# used only for a non-`:ok` state.
@inline _map_status_code_from_state_code(state_code::Int) =
    state_code == _STATE_CODE_DIVERGED ? _MAP_STATUS_DIVERGED : _MAP_STATUS_INVALID_STATE

# Translate the 3-value internal state code into the Lyapunov estimation-status code space
# (`_MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL`), used only for a non-`:ok` state.
@inline _lyapunov_estimation_code_from_state_code(state_code::Int) =
    state_code == _STATE_CODE_DIVERGED ? _LYAPUNOV_ESTIMATION_STATUS_DIVERGED : _LYAPUNOV_ESTIMATION_STATUS_INVALID_STATE

# CPU-only Symbol view of `_map_state_status_code`, used by every non-GPU-reachable call site (the
# switching-events branch of `_detect_discrete_map_period`, the continuous-ODE state-termination
# callback, `lyapunov_spectrum.jl`, and the pure-CPU continuous Lyapunov estimator).
function _map_state_status(state, cutoff::Float64)
    code = _map_state_status_code(state, cutoff)
    code == _STATE_CODE_DIVERGED && return :diverged
    code == _STATE_CODE_INVALID && return :invalid_state
    return :ok
end

function _map_detection_confidence(min_error::Float64, threshold::Float64)
    (isfinite(min_error) && isfinite(threshold) && threshold > 0.0) || return 0.0
    return clamp(1.0 - min_error / threshold, 0.0, 1.0)
end

function _period_detection_result(period::Int,
                                  status::Symbol,
                                  min_error::Float64,
                                  candidate_period::Int,
                                  observed_points::Int,
                                  threshold::Float64)
    return (
        period=period,
        status=status,
        min_closure_error=min_error,
        closure_candidate_period=candidate_period,
        observed_points=observed_points,
        closure_threshold=threshold,
        closure_confidence=_map_detection_confidence(min_error, threshold),
        valid=status == :periodic || status == :aperiodic_or_high_period
    )
end

function _closure_measure(base, candidate, precision::Float64, base_norm::Float64)
    scale = max(base_norm, norm(candidate), 1.0)
    threshold = precision * scale
    return (error=norm(base .- candidate), threshold=threshold)
end

"""
Detect the period and closure quality of an orbit from a window of iterates.

`orbit` must contain at least `max_period + 1` entries to be able to detect period exactly equal
to `max_period`. The loop bound is `min(max_period, length(orbit) - 1)` so that callers that pass
a shorter window (e.g. atlas reconnaissance passing `length(orbit_tail)` as `max_period`) still
get an exhaustive in-window comparison without an out-of-bounds access.

`precision` is treated as an amplitude-relative tolerance, scaled *per pair* by
`max(norm(orbit[1]), norm(orbit[T + 1]), 1)`. Using both endpoints of the comparison avoids
mis-classification when an orbit hasn't fully settled (one endpoint small, the other large) and
keeps an absolute floor of `precision` for orbits living near the origin.
"""
function _detect_period_diagnostics(orbit, max_period, precision)
    length(orbit) >= 2 || return _period_detection_result(0, :insufficient_crossings, Inf, 0, length(orbit), Inf)
    upper = min(max_period, length(orbit) - 1)
    upper >= 1 || return _period_detection_result(0, :insufficient_crossings, Inf, 0, length(orbit), Inf)
    base_norm = norm(orbit[1])
    min_error = Inf
    min_period = 0
    min_threshold = Inf
    for T in 1:upper
        closure = _closure_measure(orbit[1], orbit[T + 1], precision, base_norm)
        if closure.error < min_error
            min_error = closure.error
            min_period = T
            min_threshold = closure.threshold
        end
        if closure.error < closure.threshold
            return _period_detection_result(T, :periodic, min_error, min_period, length(orbit), min_threshold)
        end
    end
    return _period_detection_result(0, :aperiodic_or_high_period, min_error, min_period, length(orbit), min_threshold)
end

"""Detect only the period, preserving the historical public classification behavior."""
function _detect_period(orbit, max_period, precision)
    return _detect_period_diagnostics(orbit, max_period, precision).period
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2D Bifurcation Map (Two-Parameter Periodicity Sweep)
# ═══════════════════════════════════════════════════════════════════════════════

_map_orbit_window(config::BifurcationMapConfig) = config.max_period + 1

_map_transient_budget(config::BifurcationMapConfig) = max(config.iterations - _map_orbit_window(config), 0)

function _balanced_index_chunks(n::Int, requested_chunks::Int)
    n <= 0 && return UnitRange{Int}[]
    chunk_count = clamp(requested_chunks, 1, n)
    chunk_size = cld(n, chunk_count)
    return [start:min(start + chunk_size - 1, n) for start in 1:chunk_size:n]
end

function _map_seed_mode(config::BifurcationMapConfig, full_transient::Int=_map_transient_budget(config))
    if !config.reuse_neighbor_seeds
        return :fixed
    end
    if isnothing(config.neighbor_transient) || config.neighbor_transient >= full_transient
        return :neighbor_full
    end
    return :neighbor_accelerated
end

function _map_effective_neighbor_transient(config::BifurcationMapConfig, full_transient::Int=_map_transient_budget(config))
    return _map_seed_mode(config, full_transient) == :neighbor_accelerated ? Int(config.neighbor_transient) : nothing
end

function _map_effective_tile_sizes(config::BifurcationMapConfig,
                                   na::Int,
                                   nb::Int,
                                   seed_mode::Symbol=_map_seed_mode(config))
    seed_mode == :fixed && return (tile_size_a=0, tile_size_b=0)
    tile_size_a = config.neighbor_tile_size_a > 0 ? min(config.neighbor_tile_size_a, na) : na
    tile_size_b = config.neighbor_tile_size_b > 0 ? min(config.neighbor_tile_size_b, nb) : nb
    return (tile_size_a=max(tile_size_a, 1), tile_size_b=max(tile_size_b, 1))
end

function _map_tile_ranges(n::Int, tile_size::Int)
    tile_size >= 1 || throw(ArgumentError("Tile size must be >= 1, got $(tile_size)."))
    return [start:min(start + tile_size - 1, n) for start in 1:tile_size:n]
end

function _map_tiles(na::Int, nb::Int, tile_size_a::Int, tile_size_b::Int)
    tiles = NamedTuple{(:a_range, :b_range), Tuple{UnitRange{Int}, UnitRange{Int}}}[]
    for a_range in _map_tile_ranges(na, tile_size_a)
        for b_range in _map_tile_ranges(nb, tile_size_b)
            push!(tiles, (a_range=a_range, b_range=b_range))
        end
    end
    return tiles
end

function _map_tile_diagnostic(tile, counts)
    return Dict{String, Any}(
        "aStart" => first(tile.a_range),
        "aStop" => last(tile.a_range),
        "bStart" => first(tile.b_range),
        "bStop" => last(tile.b_range),
        "cellCount" => length(tile.a_range) * length(tile.b_range),
        "resets" => Int(counts.resets),
        "invalidResets" => Int(counts.invalid_resets)
    )
end

_map_tile_count(config::BifurcationMapConfig, na::Int, nb::Int, seed_mode::Symbol=_map_seed_mode(config)) = begin
    seed_mode == :fixed && return 0
    tile_sizes = _map_effective_tile_sizes(config, na, nb, seed_mode)
    return cld(na, tile_sizes.tile_size_a) * cld(nb, tile_sizes.tile_size_b)
end

function _map_neighbor_seed_diagnostics(config::BifurcationMapConfig;
                                        full_transient::Int=_map_transient_budget(config),
                                          na::Union{Nothing, Int}=nothing,
                                          nb::Union{Nothing, Int}=nothing,
                                        resets::Int=0,
                                        invalid_resets::Int=0,
                                        period_change_recomputes::Int=0,
                                        tile_count::Int=0,
                                        tile_diagnostics=nothing)
    seed_mode = _map_seed_mode(config, full_transient)
    tile_sizes = if seed_mode == :fixed || isnothing(na) || isnothing(nb)
        (tile_size_a=0, tile_size_b=0)
    else
        _map_effective_tile_sizes(config, na, nb, seed_mode)
    end
    effective_tile_count = tile_count > 0 ? tile_count : (seed_mode == :fixed || isnothing(na) || isnothing(nb) ? 0 : _map_tile_count(config, na, nb, seed_mode))
    return Dict(
        "seedMode" => String(seed_mode),
        "semantics" => _map_seed_semantics_label(seed_mode),
        "traversalDependent" => seed_mode != :fixed,
        "fullTransient" => full_transient,
        "neighborTransient" => _map_effective_neighbor_transient(config, full_transient),
        "requestedNeighborTransient" => isnothing(config.neighbor_transient) ? nothing : Int(config.neighbor_transient),
        "resets" => resets,
        "invalidResets" => invalid_resets,
        "periodChangeRecomputes" => period_change_recomputes,
        "tileCount" => effective_tile_count,
        "tileSizeA" => effective_tile_count == 0 ? nothing : tile_sizes.tile_size_a,
        "tileSizeB" => effective_tile_count == 0 ? nothing : tile_sizes.tile_size_b,
        "serial" => seed_mode != :fixed && effective_tile_count <= 1,
        "threadedTiles" => effective_tile_count > 1,
        "tileDiagnostics" => isnothing(tile_diagnostics) ? Any[] : tile_diagnostics
    )
end

function _map_status_counts(status_codes::AbstractMatrix{<:Integer})
    counts = Dict{String, Int}()
    for code in status_codes
        label = _map_status_label(code)
        counts[label] = get(counts, label, 0) + 1
    end
    return counts
end

function _finite_matrix_extrema(values::AbstractMatrix{<:Real})
    finite_values = Float64[v for v in values if isfinite(v)]
    isempty(finite_values) && return (minimum=nothing, maximum=nothing)
    return (minimum=minimum(finite_values), maximum=maximum(finite_values))
end

function _map_classification_diagnostics(status_codes::AbstractMatrix{<:Integer},
                                         closure_errors::AbstractMatrix{<:Real},
                                         closure_candidate_periods::AbstractMatrix{<:Integer},
                                         observed_points::AbstractMatrix{<:Integer},
                                         closure_confidence::AbstractMatrix{<:Real})
    error_extrema = _finite_matrix_extrema(closure_errors)
    confidence_extrema = _finite_matrix_extrema(closure_confidence)
    return Dict{String, Any}(
        "statusCodes" => Int.(status_codes),
        "statusLabels" => Dict(string(code) => label for (code, label) in _MAP_STATUS_LABEL_BY_CODE),
        "statusCounts" => _map_status_counts(status_codes),
        "closureErrors" => Float64.(closure_errors),
        "closureCandidatePeriods" => Int.(closure_candidate_periods),
        "observedPoints" => Int.(observed_points),
        "closureConfidence" => Float64.(closure_confidence),
        "minClosureError" => error_extrema.minimum,
        "maxClosureError" => error_extrema.maximum,
        "minClosureConfidence" => confidence_extrema.minimum,
        "maxClosureConfidence" => confidence_extrema.maximum
    )
end

const _LYAPUNOV_STATUS_UNCOMPUTED = 0
const _LYAPUNOV_STATUS_PERIODIC = 1
const _LYAPUNOV_STATUS_CHAOTIC_CANDIDATE = 2
const _LYAPUNOV_STATUS_QUASIPERIODIC_NEUTRAL_CANDIDATE = 3
const _LYAPUNOV_STATUS_UNRESOLVED = 4

const _LYAPUNOV_ESTIMATION_STATUS_NOT_REQUESTED = 0
const _LYAPUNOV_ESTIMATION_STATUS_OK = 1
const _LYAPUNOV_ESTIMATION_STATUS_COLLAPSED = 2
const _LYAPUNOV_ESTIMATION_STATUS_DIVERGED = 3
const _LYAPUNOV_ESTIMATION_STATUS_INVALID_STATE = 4
const _LYAPUNOV_ESTIMATION_STATUS_INSUFFICIENT_CROSSINGS = 5
const _LYAPUNOV_ESTIMATION_STATUS_INTEGRATION_FAILED = 6
const _LYAPUNOV_ESTIMATION_STATUS_INSUFFICIENT_SAMPLES = 7

# As with `_MAP_STATUS_*` above, these named constants are the single compile-time source of truth for
# the Lyapunov classification / estimation-status code spaces: the Dicts below, both Dict-free GPU
# mirrors, and every classifier reachable from a `@kernel` body (`_map_lyapunov_classification` here,
# `_lyapunov_point_classification` in `lyapunov.jl`) are built from the same constants, so the numeric
# code space can only drift by editing this block. Still plain `Int` constants — zero overhead on the
# isbits hot paths.
const _MAP_LYAPUNOV_STATUS_CODE_BY_SYMBOL = Dict{Symbol, Int}(
    :uncomputed => _LYAPUNOV_STATUS_UNCOMPUTED,
    :periodic => _LYAPUNOV_STATUS_PERIODIC,
    :chaotic_candidate => _LYAPUNOV_STATUS_CHAOTIC_CANDIDATE,
    :quasiperiodic_neutral_candidate => _LYAPUNOV_STATUS_QUASIPERIODIC_NEUTRAL_CANDIDATE,
    :unresolved => _LYAPUNOV_STATUS_UNRESOLVED
)

const _MAP_LYAPUNOV_STATUS_LABEL_BY_CODE = Dict{Int, String}(
    code => String(status) for (status, code) in _MAP_LYAPUNOV_STATUS_CODE_BY_SYMBOL
)

const _MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL = Dict{Symbol, Int}(
    :not_requested => _LYAPUNOV_ESTIMATION_STATUS_NOT_REQUESTED,
    :ok => _LYAPUNOV_ESTIMATION_STATUS_OK,
    :collapsed => _LYAPUNOV_ESTIMATION_STATUS_COLLAPSED,
    :diverged => _LYAPUNOV_ESTIMATION_STATUS_DIVERGED,
    :invalid_state => _LYAPUNOV_ESTIMATION_STATUS_INVALID_STATE,
    :insufficient_crossings => _LYAPUNOV_ESTIMATION_STATUS_INSUFFICIENT_CROSSINGS,
    :integration_failed => _LYAPUNOV_ESTIMATION_STATUS_INTEGRATION_FAILED,
    :insufficient_samples => _LYAPUNOV_ESTIMATION_STATUS_INSUFFICIENT_SAMPLES
)

const _MAP_LYAPUNOV_ESTIMATION_STATUS_LABEL_BY_CODE = Dict{Int, String}(
    code => String(status) for (status, code) in _MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL
)

_map_lyapunov_enabled(config::BifurcationMapConfig) = config.lyapunov_enabled
_map_lyapunov_iterations(config::BifurcationMapConfig) = config.lyapunov_iterations > 0 ? config.lyapunov_iterations : max(32, 8 * max(config.max_period, 1))
_map_lyapunov_transient(config::BifurcationMapConfig) = isnothing(config.lyapunov_transient) ? 0 : Int(config.lyapunov_transient)
_map_lyapunov_status_code(status::Symbol) = get(_MAP_LYAPUNOV_STATUS_CODE_BY_SYMBOL, status, _MAP_LYAPUNOV_STATUS_CODE_BY_SYMBOL[:unresolved])
_map_lyapunov_estimation_status_code(status::Symbol) = get(_MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL, status, _MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL[:insufficient_samples])
_map_lyapunov_status_label(code::Integer) = get(_MAP_LYAPUNOV_STATUS_LABEL_BY_CODE, Int(code), "unresolved")
_map_lyapunov_estimation_status_label(code::Integer) = get(_MAP_LYAPUNOV_ESTIMATION_STATUS_LABEL_BY_CODE, Int(code), "insufficient_samples")
_map_lyapunov_status_symbol(code::Integer) = Symbol(_map_lyapunov_status_label(code))
_map_lyapunov_estimation_status_symbol(code::Integer) = Symbol(_map_lyapunov_estimation_status_label(code))

# Dict-free mirrors, kept in lockstep with the Dicts above (both built from the same named constants)
# and asserted equal to them by test for every known status. No longer called from inside a `@kernel`
# body (the numeric cores below emit these exact codes directly); retained as a standalone
# Symbol->code utility.
@inline function _gpu_map_lyapunov_status_code(status::Symbol)
    status === :periodic && return _LYAPUNOV_STATUS_PERIODIC
    status === :chaotic_candidate && return _LYAPUNOV_STATUS_CHAOTIC_CANDIDATE
    status === :quasiperiodic_neutral_candidate && return _LYAPUNOV_STATUS_QUASIPERIODIC_NEUTRAL_CANDIDATE
    status === :unresolved && return _LYAPUNOV_STATUS_UNRESOLVED
    return _LYAPUNOV_STATUS_UNCOMPUTED
end

@inline function _gpu_map_lyapunov_estimation_status_code(status::Symbol)
    status === :ok && return _LYAPUNOV_ESTIMATION_STATUS_OK
    status === :collapsed && return _LYAPUNOV_ESTIMATION_STATUS_COLLAPSED
    status === :diverged && return _LYAPUNOV_ESTIMATION_STATUS_DIVERGED
    status === :invalid_state && return _LYAPUNOV_ESTIMATION_STATUS_INVALID_STATE
    status === :insufficient_crossings && return _LYAPUNOV_ESTIMATION_STATUS_INSUFFICIENT_CROSSINGS
    status === :integration_failed && return _LYAPUNOV_ESTIMATION_STATUS_INTEGRATION_FAILED
    status === :insufficient_samples && return _LYAPUNOV_ESTIMATION_STATUS_INSUFFICIENT_SAMPLES
    return _LYAPUNOV_ESTIMATION_STATUS_NOT_REQUESTED
end

function _map_lyapunov_storage(na::Int, nb::Int)
    return (
        exponents=fill(NaN, na, nb),
        status_codes=fill(_map_lyapunov_status_code(:uncomputed), na, nb),
        estimation_status_codes=fill(_map_lyapunov_estimation_status_code(:not_requested), na, nb),
        sample_counts=zeros(Int, na, nb)
    )
end

function _matrix_label_counts(codes::AbstractMatrix{<:Integer}, label_fn::Function)
    counts = Dict{String, Int}()
    for code in codes
        label = label_fn(code)
        counts[label] = get(counts, label, 0) + 1
    end
    return counts
end

"""
Classify a Lyapunov estimate into a map-cell verdict, purely in terms of the map-status code space
(`_MAP_STATUS_CODE_BY_SYMBOL`), the Lyapunov estimation-status code space
(`_MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL`), and the Lyapunov classification code space
(`_MAP_LYAPUNOV_STATUS_CODE_BY_SYMBOL`, the return value) — no `Symbol` in or out, so this is safe to
call from inside a `@kernel` body (`gpu_kernels.jl`). CPU-only callers (`_record_map_lyapunov!`)
convert their `Symbol` statuses to codes with `_map_status_code` / `_map_lyapunov_estimation_status_code`
before calling this, and use the returned code directly (it already is the final `statusCodes` value).
"""
@inline function _map_lyapunov_classification(detection_period::Int, detection_status_code::Int,
                                              exponent::Float64, estimation_status_code::Int,
                                              neutral_tolerance::Float64)
    detection_period > 0 && return _LYAPUNOV_STATUS_PERIODIC
    detection_status_code == _MAP_STATUS_APERIODIC_OR_HIGH_PERIOD || return _LYAPUNOV_STATUS_UNRESOLVED
    (estimation_status_code == _LYAPUNOV_ESTIMATION_STATUS_OK ||
     estimation_status_code == _LYAPUNOV_ESTIMATION_STATUS_COLLAPSED) || return _LYAPUNOV_STATUS_UNRESOLVED
    # A collapsed trajectory pair (perturbation contracted to zero, exponent = -Inf)
    # is the extreme case of a confidently negative largest exponent: a regular,
    # strongly contracting regime that the period detector could not resolve to a
    # finite period. Classify it like any other negative exponent below rather than
    # discarding it as :unresolved on the `isfinite` check.
    estimation_status_code == _LYAPUNOV_ESTIMATION_STATUS_COLLAPSED && return _LYAPUNOV_STATUS_PERIODIC
    isfinite(exponent) || return _LYAPUNOV_STATUS_UNRESOLVED
    exponent > neutral_tolerance && return _LYAPUNOV_STATUS_CHAOTIC_CANDIDATE
    abs(exponent) <= neutral_tolerance && return _LYAPUNOV_STATUS_QUASIPERIODIC_NEUTRAL_CANDIDATE
    # exponent < -neutral_tolerance: confidently contracting ⟹ regular/periodic-like
    # dynamics (period above the detector's max_period), not an unresolved cell.
    return _LYAPUNOV_STATUS_PERIODIC
end

function _lyapunov_estimate_result(exponent::Float64, estimation_status::Symbol, sample_count::Int)
    return (
        exponent=exponent,
        estimation_status=estimation_status,
        sample_count=sample_count
    )
end

# GPU-safe (isbits) counterpart of `_lyapunov_estimate_result`: `estimation_status` is a code in
# `_MAP_LYAPUNOV_ESTIMATION_STATUS_CODE_BY_SYMBOL`'s space rather than a `Symbol`. This is the shape
# `_estimate_discrete_map_largest_lyapunov_core` returns (device-safe); the CPU host wrapper
# `_estimate_discrete_map_largest_lyapunov` converts it back to `_lyapunov_estimate_result`'s
# `Symbol`-based shape at the boundary.
function _lyapunov_estimate_result_code(exponent::Float64, estimation_status_code::Int, sample_count::Int)
    return (
        exponent=exponent,
        estimation_status=estimation_status_code,
        sample_count=sample_count
    )
end

_partial_lyapunov_exponent(log_sum::Float64, sample_count::Int) = sample_count > 0 ? log_sum / sample_count : NaN

function _lyapunov_initial_direction(::Val{D}) where {D}
    D >= 1 || throw(ArgumentError("Lyapunov estimation requires a positive-dimensional state."))
    return SVector{D, Float64}(ntuple(idx -> idx == 1 ? 1.0 : 0.0, D))
end

"""
Non-allocating two-trajectory Lyapunov estimator core, parameterized on the bare map function `f`
rather than `sys` — shared verbatim by the CPU sweep and the GPU kernel. `Dict`/`String`/`Symbol`-free
and fully `isbits`-safe, so it is safe to call from inside a `KernelAbstractions.@kernel`; the CPU host
wrapper `_estimate_discrete_map_largest_lyapunov` converts its integer `estimation_status` code back to
the public `Symbol` at the boundary.
"""
@inline function _estimate_discrete_map_largest_lyapunov_core(f::F,
                                                               params,
                                                               initial_point::SVector{D, Float64},
                                                               transient::Int,
                                                               steps::Int,
                                                               perturbation::Float64,
                                                               divergence_cutoff::Float64) where {F, D}
    steps > 0 || return _lyapunov_estimate_result_code(NaN, 7, 0)   # :insufficient_samples

    point = initial_point
    for _ in 1:transient
        point = f(point, params)
        state_code = _map_state_status_code(point, divergence_cutoff)
        state_code != _STATE_CODE_OK && return _lyapunov_estimate_result_code(NaN, _lyapunov_estimation_code_from_state_code(state_code), 0)
    end

    direction = _lyapunov_initial_direction(Val(D))
    perturbed = point + perturbation * direction
    state_code = _map_state_status_code(perturbed, divergence_cutoff)
    state_code != _STATE_CODE_OK && return _lyapunov_estimate_result_code(NaN, _lyapunov_estimation_code_from_state_code(state_code), 0)

    log_sum = 0.0
    sample_count = 0
    for _ in 1:steps
        next_point = f(point, params)
        next_perturbed = f(perturbed, params)
        state_code = _map_state_status_code(next_point, divergence_cutoff)
        state_code != _STATE_CODE_OK && return _lyapunov_estimate_result_code(_partial_lyapunov_exponent(log_sum, sample_count), _lyapunov_estimation_code_from_state_code(state_code), sample_count)
        perturbed_state_code = _map_state_status_code(next_perturbed, divergence_cutoff)
        perturbed_state_code != _STATE_CODE_OK && return _lyapunov_estimate_result_code(_partial_lyapunov_exponent(log_sum, sample_count), _lyapunov_estimation_code_from_state_code(perturbed_state_code), sample_count)

        delta = next_perturbed - next_point
        distance = norm(delta)
        # Treat a separation that has shrunk to within machine precision of the
        # state magnitude as a collapse, not just an exact zero. Otherwise a
        # tiny-but-positive distance (e.g. 1e-300 for a strongly contracting
        # orbit) flows into log(distance / perturbation) and dominates log_sum
        # with an unphysical large-negative term.
        collapse_floor = eps(Float64) * max(norm(next_point), 1.0)
        if !isfinite(distance)
            return _lyapunov_estimate_result_code(_partial_lyapunov_exponent(log_sum, sample_count), 4, sample_count)   # :invalid_state
        elseif distance <= collapse_floor
            return _lyapunov_estimate_result_code(-Inf, 2, sample_count + 1)   # :collapsed
        end

        sample_count += 1
        log_sum += log(distance / perturbation)
        direction = delta / distance
        point = next_point
        perturbed = point + perturbation * direction
    end

    return _lyapunov_estimate_result_code(log_sum / sample_count, 1, sample_count)   # :ok
end

function _estimate_discrete_map_largest_lyapunov(sys::DiscreteMap,
                                                 params::AbstractVector,
                                                 initial_point::SVector{D, Float64},
                                                 transient::Int,
                                                 steps::Int,
                                                 perturbation::Float64,
                                                 divergence_cutoff::Float64) where {D}
    core = _estimate_discrete_map_largest_lyapunov_core(sys.f, params, initial_point, transient, steps, perturbation, divergence_cutoff)
    return _lyapunov_estimate_result(core.exponent, _map_lyapunov_estimation_status_symbol(core.estimation_status), core.sample_count)
end

function _poincare_diagnostics_status(diagnostics::AbstractDict)
    final_status = Symbol(String(get(diagnostics, "finalStatus", "unknown")))
    if final_status == :ok
        reason = Symbol(String(get(diagnostics, "terminationReason", "unknown")))
        return reason == :requested_crossings_found ? :ok : :insufficient_crossings
    end
    final_status in (:diverged, :invalid_state, :integration_failed) && return final_status
    return :insufficient_crossings
end

function _poincare_next_full_state(sys::ContinuousODE,
                                   params::AbstractVector,
                                   state::AbstractVector;
                                   solver,
                                   reltol::Float64,
                                   abstol::Float64,
                                   divergence_cutoff::Float64,
                                   min_crossing_time::Float64=1e-6)
    sample = _collect_poincare_points(
        sys,
        params;
        initial_point=state,
        crossings=1,
        transient=0,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        projected=false,
        divergence_cutoff=divergence_cutoff,
        min_crossing_time=min_crossing_time,
        return_diagnostics=true
    )
    status = _poincare_diagnostics_status(sample.diagnostics)
    if status == :ok && length(sample.points) == 1
        return (point=sample.points[1], status=:ok)
    end
    return (point=copy(collect(Float64, state)), status=status)
end

"""
Project a full-state perturbation `direction` onto the Poincaré section's tangent
plane at `state`, returning a unit tangent direction. This keeps a perturbed seed
`state + perturbation * direction` on the section (to first order), so the
reference and perturbed trajectories are compared as iterates of the same Poincaré
return map rather than mixing in a section-normal offset. Falls back to the
(normalized) input direction when the section gradient is unavailable or degenerate.
"""
function _project_direction_onto_section(sys::ContinuousODE, state::AbstractVector, direction::AbstractVector)
    raw_norm = norm(direction)
    raw_norm > 0 || return direction
    normal = try
        _section_condition_gradient(sys, state, 0.0)
    catch
        return direction ./ raw_norm
    end
    nn = dot(normal, normal)
    nn > 0 || return direction ./ raw_norm
    projected = direction .- (dot(normal, direction) / nn) .* normal
    projected_norm = norm(projected)
    projected_norm > 0 || return direction ./ raw_norm
    return projected ./ projected_norm
end

function _estimate_continuous_poincare_largest_lyapunov(sys::ContinuousODE,
                                                        params::AbstractVector,
                                                        initial_state::AbstractVector,
                                                        transient::Int,
                                                        steps::Int,
                                                        perturbation::Float64,
                                                        divergence_cutoff::Float64;
                                                        solver,
                                                        reltol::Float64,
                                                        abstol::Float64,
                                                        min_crossing_time::Float64=1e-6)
    steps > 0 || return _lyapunov_estimate_result(NaN, :insufficient_samples, 0)

    state = collect(Float64, initial_state)
    status = _map_state_status(state, divergence_cutoff)
    status != :ok && return _lyapunov_estimate_result(NaN, status, 0)

    if transient > 0
        warmed = _collect_poincare_points(
            sys,
            params;
            initial_point=state,
            crossings=1,
            transient=transient,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            projected=false,
            divergence_cutoff=divergence_cutoff,
            min_crossing_time=min_crossing_time,
            return_diagnostics=true
        )
        warm_status = _poincare_diagnostics_status(warmed.diagnostics)
        warm_status == :ok && length(warmed.points) == 1 || return _lyapunov_estimate_result(NaN, warm_status, 0)
        state = warmed.points[1]
    end

    direction = zeros(Float64, length(state))
    isempty(direction) && return _lyapunov_estimate_result(NaN, :insufficient_samples, 0)
    direction[1] = 1.0
    direction = _project_direction_onto_section(sys, state, direction)
    perturbed = state .+ perturbation .* direction
    status = _map_state_status(perturbed, divergence_cutoff)
    status != :ok && return _lyapunov_estimate_result(NaN, status, 0)

    log_sum = 0.0
    sample_count = 0
    for _ in 1:steps
        next_state = _poincare_next_full_state(
            sys,
            params,
            state;
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            divergence_cutoff=divergence_cutoff,
            min_crossing_time=min_crossing_time
        )
        next_state.status == :ok || return _lyapunov_estimate_result(_partial_lyapunov_exponent(log_sum, sample_count), next_state.status, sample_count)
        next_perturbed = _poincare_next_full_state(
            sys,
            params,
            perturbed;
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            divergence_cutoff=divergence_cutoff,
            min_crossing_time=min_crossing_time
        )
        next_perturbed.status == :ok || return _lyapunov_estimate_result(_partial_lyapunov_exponent(log_sum, sample_count), next_perturbed.status, sample_count)

        delta = next_perturbed.point .- next_state.point
        distance = norm(delta)
        # See the discrete estimator: floor collapse at machine precision relative
        # to the state magnitude so a tiny-but-positive separation does not feed an
        # unphysical large-negative term into log(distance / perturbation).
        collapse_floor = eps(Float64) * max(norm(next_state.point), 1.0)
        if !isfinite(distance)
            return _lyapunov_estimate_result(_partial_lyapunov_exponent(log_sum, sample_count), :invalid_state, sample_count)
        elseif distance <= collapse_floor
            return _lyapunov_estimate_result(-Inf, :collapsed, sample_count + 1)
        end

        sample_count += 1
        log_sum += log(distance / perturbation)
        direction = _project_direction_onto_section(sys, next_state.point, delta ./ distance)
        state = next_state.point
        perturbed = state .+ perturbation .* direction
    end

    return _lyapunov_estimate_result(log_sum / sample_count, :ok, sample_count)
end

function _record_map_lyapunov!(storage,
                               i::Int,
                               j::Int,
                               detection,
                               estimate,
                               neutral_tolerance::Float64)
    isnothing(storage) && return nothing
    exponent = Float64(estimate.exponent)
    storage.exponents[i, j] = exponent
    storage.status_codes[i, j] = _map_lyapunov_classification(
        Int(detection.period), _map_status_code(detection.status), exponent,
        _map_lyapunov_estimation_status_code(estimate.estimation_status), neutral_tolerance
    )
    storage.estimation_status_codes[i, j] = _map_lyapunov_estimation_status_code(estimate.estimation_status)
    storage.sample_counts[i, j] = estimate.sample_count
    return storage
end

function _record_discrete_map_lyapunov!(storage,
                                        sys::DiscreteMap,
                                        config::BifurcationMapConfig,
                                        params::AbstractVector,
                                        i::Int,
                                        j::Int,
                                        detection)
    isnothing(storage) && return nothing
    estimate = _estimate_discrete_map_largest_lyapunov(
        sys,
        params,
        detection.final_point,
        _map_lyapunov_transient(config),
        _map_lyapunov_iterations(config),
        config.lyapunov_perturbation,
        config.divergence_cutoff
    )
    return _record_map_lyapunov!(storage, i, j, detection, estimate, config.lyapunov_neutral_tolerance)
end

function _record_continuous_map_lyapunov!(storage,
                                          sys::ContinuousODE,
                                          config::BifurcationMapConfig,
                                          params::AbstractVector,
                                          i::Int,
                                          j::Int,
                                          detection;
                                          solver,
                                          reltol::Float64,
                                          abstol::Float64)
    isnothing(storage) && return nothing
    estimate = _estimate_continuous_poincare_largest_lyapunov(
        sys,
        params,
        detection.final_point,
        _map_lyapunov_transient(config),
        _map_lyapunov_iterations(config),
        config.lyapunov_perturbation,
        config.divergence_cutoff;
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        min_crossing_time=config.min_crossing_time
    )
    return _record_map_lyapunov!(storage, i, j, detection, estimate, config.lyapunov_neutral_tolerance)
end

function _map_lyapunov_diagnostics(storage, config::BifurcationMapConfig, method::Symbol)
    extrema = _finite_matrix_extrema(storage.exponents)
    return Dict{String, Any}(
        "enabled" => true,
        "method" => String(method),
        "iterations" => _map_lyapunov_iterations(config),
        "requestedIterations" => Int(config.lyapunov_iterations),
        "transient" => _map_lyapunov_transient(config),
        "requestedTransient" => isnothing(config.lyapunov_transient) ? nothing : Int(config.lyapunov_transient),
        "perturbation" => config.lyapunov_perturbation,
        "neutralTolerance" => config.lyapunov_neutral_tolerance,
        "exponents" => Float64.(storage.exponents),
        "statusCodes" => Int.(storage.status_codes),
        "statusLabels" => Dict(string(code) => label for (code, label) in _MAP_LYAPUNOV_STATUS_LABEL_BY_CODE),
        "statusCounts" => _matrix_label_counts(storage.status_codes, _map_lyapunov_status_label),
        "estimationStatusCodes" => Int.(storage.estimation_status_codes),
        "estimationStatusLabels" => Dict(string(code) => label for (code, label) in _MAP_LYAPUNOV_ESTIMATION_STATUS_LABEL_BY_CODE),
        "estimationStatusCounts" => _matrix_label_counts(storage.estimation_status_codes, _map_lyapunov_estimation_status_label),
        "sampleCounts" => Int.(storage.sample_counts),
        "minExponent" => extrema.minimum,
        "maxExponent" => extrema.maximum
    )
end

function _map_lyapunov_result(storage,
                              a_grid::Vector{Float64},
                              b_grid::Vector{Float64},
                              config::BifurcationMapConfig,
                              system_name::String,
                              param_names::Tuple{Symbol, Symbol},
                              timestamp::DateTime;
                              compute_backend::Symbol=:cpu)
    isnothing(storage) && return nothing
    return LyapunovFieldResult(
        a_grid,
        b_grid,
        Float64.(storage.exponents),
        Int.(storage.status_codes),
        Int.(storage.estimation_status_codes),
        Int.(storage.sample_counts),
        config.lyapunov_neutral_tolerance,
        system_name,
        param_names,
        timestamp;
        compute_backend=compute_backend
    )
end

function _record_map_detection!(periodicity::AbstractMatrix{Int},
                                status_codes::AbstractMatrix{Int},
                                closure_errors::AbstractMatrix{Float64},
                                closure_candidate_periods::AbstractMatrix{Int},
                                 observed_points::AbstractMatrix{Int},
                                 closure_confidence::AbstractMatrix{Float64},
                                 i::Int,
                                 j::Int,
                                 result,
                                 crossing_diagnostics=nothing,
                                 switching_diagnostics=nothing)
    periodicity[i, j] = result.period
    status_codes[i, j] = _map_status_code(result.status)
    closure_errors[i, j] = result.min_closure_error
    closure_candidate_periods[i, j] = result.closure_candidate_period
    observed_points[i, j] = result.observed_points
    closure_confidence[i, j] = result.closure_confidence
    if !isnothing(crossing_diagnostics) && :crossing_diagnostics in propertynames(result)
        crossing_diagnostics[i, j] = result.crossing_diagnostics
    end
    if !isnothing(switching_diagnostics) && :switching_diagnostics in propertynames(result)
        switching_diagnostics[i, j] = result.switching_diagnostics
    end
    return result
end

function _map_crossing_summary_storage(na::Int, nb::Int, crossings_requested::Int)
    return (
        crossings_requested=Int(crossings_requested),
        crossings_found=zeros(Int, na, nb),
        total_crossings_found=zeros(Int, na, nb),
        final_times=fill(NaN, na, nb),
        termination_reasons=fill(:unknown, na, nb),
        solver_retcodes=fill(:unknown, na, nb),
        # Matrix{Bool}, not BitMatrix: cells are written concurrently by distinct threaded tasks, and
        # BitMatrix packs 64 bools per word, so unrelated cells sharing a word race on read-modify-write.
        divergence_callback_activated=fill(false, na, nb),
        state_callback_activated=fill(false, na, nb),
        populated=fill(false, na, nb)
    )
end

function _record_map_crossing_summary!(storage, i::Int, j::Int, result)
    storage.crossings_found[i, j] = Int(result.observed_points)
    storage.total_crossings_found[i, j] = Int(result.total_crossings_found)
    storage.final_times[i, j] = Float64(result.final_time)
    storage.termination_reasons[i, j] = result.termination_reason
    storage.solver_retcodes[i, j] = result.solver_retcode
    storage.divergence_callback_activated[i, j] = Bool(result.divergence_callback_activated)
    storage.state_callback_activated[i, j] = Bool(result.state_callback_activated)
    storage.populated[i, j] = true
    return storage
end

_map_multistability_enabled(config::BifurcationMapConfig) = !isempty(config.multistability_initial_points)

function _map_seed_vector(seed::AbstractVector, dim::Int, label::AbstractString)
    length(seed) == dim || throw(ArgumentError(
        "$label has length $(length(seed)) but the map state dimension is $dim."
    ))
    return collect(Float64, seed)
end

function _map_extra_seed_vectors(config::BifurcationMapConfig, dim::Int)
    [_map_seed_vector(seed, dim, "multistability_initial_points[$idx]")
     for (idx, seed) in enumerate(config.multistability_initial_points)]
end

function _map_multistability_storage(na::Int, nb::Int)
    return (
        attractor_counts=zeros(Int, na, nb),
        coexistence_flags=fill(false, na, nb),
        dominant_fractions=zeros(Float64, na, nb),
        basin_entropy=zeros(Float64, na, nb),
        normalized_basin_entropy=zeros(Float64, na, nb),
        period_sets=[Int[] for _ in 1:na, _ in 1:nb],
        status_sets=[String[] for _ in 1:na, _ in 1:nb],
        period_fractions=[Dict{String, Float64}() for _ in 1:na, _ in 1:nb],
        status_fractions=[Dict{String, Float64}() for _ in 1:na, _ in 1:nb]
    )
end

function _fraction_entropy(counts::AbstractDict, total::Int)
    total <= 0 && return 0.0
    entropy = 0.0
    for count in values(counts)
        count <= 0 && continue
        fraction = count / total
        entropy -= fraction * log(fraction)
    end
    return entropy
end

function _map_detection_status_label(result)
    label = get(_MAP_STATUS_LABEL_BY_CODE, _map_status_code(result.status), string(result.status))
    return String(label)
end

function _summarize_map_multistability(results::AbstractVector)
    isempty(results) && throw(ArgumentError("Cannot summarize multistability without any seed results."))
    period_counts = Dict{Int, Int}()
    status_counts = Dict{String, Int}()
    period_order = Int[]
    status_order = String[]

    for result in results
        period = Int(result.period)
        if !haskey(period_counts, period)
            period_counts[period] = 0
            push!(period_order, period)
        end
        period_counts[period] += 1

        status = _map_detection_status_label(result)
        status_key = "$status:$period"
        if !haskey(status_counts, status_key)
            status_counts[status_key] = 0
            push!(status_order, status_key)
        end
        status_counts[status_key] += 1
    end

    dominant_period = first(period_order)
    dominant_count = period_counts[dominant_period]
    for period in period_order
        count = period_counts[period]
        if count > dominant_count
            dominant_period = period
            dominant_count = count
        end
    end
    selected_idx = something(findfirst(result -> Int(result.period) == dominant_period, results), 1)
    total = length(results)
    entropy = _fraction_entropy(period_counts, total)
    period_fractions = Dict(string(period) => count / total for (period, count) in pairs(period_counts))
    status_fractions = Dict(status => count / total for (status, count) in pairs(status_counts))
    period_set = sort!(collect(keys(period_counts)))
    status_set = sort!(collect(keys(status_counts)))

    return (
        selected=results[selected_idx],
        period_set=period_set,
        status_set=status_set,
        period_fractions=period_fractions,
        status_fractions=status_fractions,
        attractor_count=length(period_set),
        coexistence=length(period_set) > 1,
        dominant_fraction=dominant_count / total,
        basin_entropy=entropy,
        normalized_basin_entropy=length(period_set) <= 1 ? 0.0 : entropy / log(length(period_set))
    )
end

function _record_map_multistability!(storage, i::Int, j::Int, summary)
    storage.attractor_counts[i, j] = summary.attractor_count
    storage.coexistence_flags[i, j] = summary.coexistence
    storage.dominant_fractions[i, j] = summary.dominant_fraction
    storage.basin_entropy[i, j] = summary.basin_entropy
    storage.normalized_basin_entropy[i, j] = summary.normalized_basin_entropy
    storage.period_sets[i, j] = summary.period_set
    storage.status_sets[i, j] = summary.status_set
    storage.period_fractions[i, j] = summary.period_fractions
    storage.status_fractions[i, j] = summary.status_fractions
    return storage
end

function _map_multistability_diagnostics(storage, dominant_periods::AbstractMatrix{Int}, seed_count::Int)
    distinct_periods = sort!(unique(vcat((periods for periods in storage.period_sets)...)))
    return Dict{String, Any}(
        "enabled" => true,
        "seedCount" => seed_count,
        "extraSeedCount" => seed_count - 1,
        "coexistenceCells" => count(identity, storage.coexistence_flags),
        "maxAttractorCount" => isempty(storage.attractor_counts) ? 0 : maximum(storage.attractor_counts),
        "distinctPeriods" => distinct_periods,
        "dominantPeriods" => Int.(dominant_periods),
        "attractorCounts" => Int.(storage.attractor_counts),
        "coexistenceFlags" => Bool.(storage.coexistence_flags),
        "dominantFractions" => Float64.(storage.dominant_fractions),
        "basinEntropy" => Float64.(storage.basin_entropy),
        "normalizedBasinEntropy" => Float64.(storage.normalized_basin_entropy),
        "periodSets" => storage.period_sets,
        "statusSets" => storage.status_sets,
        "periodFractions" => storage.period_fractions,
        "statusFractions" => storage.status_fractions
    )
end

_crossing_diag_int(diag::AbstractDict, key::AbstractString) = Int(get(diag, key, 0))
_crossing_diag_float(diag::AbstractDict, key::AbstractString) = Float64(get(diag, key, NaN))
_crossing_diag_string(diag::AbstractDict, key::AbstractString) = String(get(diag, key, "unknown"))
_crossing_diag_bool(diag::AbstractDict, key::AbstractString) = Bool(get(diag, key, false))

function _increment_count!(counts::Dict{String, Int}, label::AbstractString)
    counts[String(label)] = get(counts, String(label), 0) + 1
    return counts
end

function _poincare_crossing_diagnostics_summary(diagnostics::AbstractArray)
    requested = fill(0, size(diagnostics))
    found = fill(0, size(diagnostics))
    total_found = fill(0, size(diagnostics))
    final_times = fill(NaN, size(diagnostics))
    termination_reasons = fill("unknown", size(diagnostics))
    solver_retcodes = fill("unknown", size(diagnostics))
    termination_counts = Dict{String, Int}()
    retcode_counts = Dict{String, Int}()
    divergence_count = 0
    state_callback_count = 0
    populated = 0

    for idx in eachindex(diagnostics)
        diag = diagnostics[idx]
        diag isa AbstractDict || continue
        isempty(diag) && continue
        populated += 1
        requested[idx] = _crossing_diag_int(diag, "crossingsRequested")
        found[idx] = _crossing_diag_int(diag, "crossingsFound")
        total_found[idx] = _crossing_diag_int(diag, "totalCrossingsFound")
        final_times[idx] = _crossing_diag_float(diag, "finalTime")
        termination_reasons[idx] = _crossing_diag_string(diag, "terminationReason")
        solver_retcodes[idx] = _crossing_diag_string(diag, "solverRetcode")
        _increment_count!(termination_counts, termination_reasons[idx])
        _increment_count!(retcode_counts, solver_retcodes[idx])
        divergence_count += _crossing_diag_bool(diag, "divergenceCallbackActivated") ? 1 : 0
        state_callback_count += _crossing_diag_bool(diag, "stateCallbackActivated") ? 1 : 0
    end

    return Dict{String, Any}(
        "sampleCount" => populated,
        "crossingsRequested" => requested,
        "crossingsFound" => found,
        "totalCrossingsFound" => total_found,
        "finalTimes" => final_times,
        "terminationReasons" => termination_reasons,
        "solverRetcodes" => solver_retcodes,
        "terminationCounts" => termination_counts,
        "solverRetcodeCounts" => retcode_counts,
        "divergenceCallbackCount" => divergence_count,
        "stateCallbackCount" => state_callback_count
    )
end

function _poincare_crossing_diagnostics_summary(storage)
    requested = zeros(Int, size(storage.populated))
    found = zeros(Int, size(storage.populated))
    total_found = zeros(Int, size(storage.populated))
    final_times = fill(NaN, size(storage.populated))
    termination_reasons = fill("unknown", size(storage.populated))
    solver_retcodes = fill("unknown", size(storage.populated))
    termination_counts = Dict{String, Int}()
    retcode_counts = Dict{String, Int}()

    for idx in eachindex(storage.populated)
        storage.populated[idx] || continue
        requested[idx] = storage.crossings_requested
        found[idx] = storage.crossings_found[idx]
        total_found[idx] = storage.total_crossings_found[idx]
        final_times[idx] = storage.final_times[idx]
        termination_reason = String(storage.termination_reasons[idx])
        solver_retcode = String(storage.solver_retcodes[idx])
        termination_reasons[idx] = termination_reason
        solver_retcodes[idx] = solver_retcode
        _increment_count!(termination_counts, termination_reason)
        _increment_count!(retcode_counts, solver_retcode)
    end

    return Dict{String, Any}(
        "sampleCount" => count(identity, storage.populated),
        "crossingsRequested" => requested,
        "crossingsFound" => found,
        "totalCrossingsFound" => total_found,
        "finalTimes" => final_times,
        "terminationReasons" => termination_reasons,
        "solverRetcodes" => solver_retcodes,
        "terminationCounts" => termination_counts,
        "solverRetcodeCounts" => retcode_counts,
        "divergenceCallbackCount" => count(identity, storage.divergence_callback_activated),
        "stateCallbackCount" => count(identity, storage.state_callback_activated)
    )
end

