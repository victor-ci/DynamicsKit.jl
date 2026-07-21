"""
Codimension-2 bifurcation curves. Two engines:

- `:slice_tracking` — assemble the curve from repeated 1D continuation slices
  across the secondary parameter (original engine).
- `:defining_system` — continue the bifurcation condition itself: augment the
  fixed-point equation of the (iterated) map or Poincaré return map with a
  bordered eigenvector condition and run pseudo-arclength continuation of the
  augmented system in the secondary parameter. Supports `:pd` (multiplier -1,
  `(DF^N + I)v = 0`), `:fold` (multiplier +1, `(DF^N - I)v = 0`), and `:ns`
  (complex pair on the unit circle, real bordered two-vector system with the
  multiplier angle as an extra unknown).
"""

_normalize_codim2_kind(kind::Symbol) = kind === :hopf ? :ns : kind

function _codim2_base_params(sys::DiscreteMap, config::Codim2Config, params::Vector{Float64})
    if !isempty(params)
        return copy(params)
    elseif !isempty(config.fixed_params)
        return copy(config.fixed_params)
    end
    required = max(length(sys.param_names),
                   config.continuation.param_index,
                   config.second_param_index,
                   isempty(config.continuation.linked_param_indices) ? 1 : maximum(config.continuation.linked_param_indices),
                   isempty(config.second_linked_param_indices) ? 1 : maximum(config.second_linked_param_indices))
    return zeros(Float64, required)
end

function _codim2_base_params(sys::ContinuousODE, config::Codim2Config, params::Vector{Float64})
    if !isempty(params)
        return _resolve_continuous_params(sys, params)
    elseif !isempty(config.fixed_params)
        return copy(config.fixed_params)
    end
    return _resolve_continuous_params(sys, Float64[])
end

function _codim2_tracking_tolerance(config::Codim2Config)
    if !isnothing(config.tracking_tolerance)
        return Float64(config.tracking_tolerance)
    end
    span = abs(config.continuation.p_max - config.continuation.p_min)
    return max(span * 0.1, 1e-8)
end

function _codim2_anchor_second(config::Codim2Config)
    if !isnothing(config.anchor_second)
        return Float64(config.anchor_second)
    end
    return (config.second_min + config.second_max) / 2
end

function _codim2_slice_seed_values(config::Codim2Config, second_values::AbstractVector{<:Real}, base_params::Vector{Float64})
    if isempty(config.primary_seed_values)
        return fill(base_params[config.continuation.param_index], length(second_values))
    end
    length(config.primary_seed_values) == length(second_values) || throw(ArgumentError(
        "Codim2Config.primary_seed_values length $(length(config.primary_seed_values)) must match the secondary sweep length $(length(second_values))."
    ))
    return Float64[Float64(value) for value in config.primary_seed_values]
end

function _codim2_slice_primary_bounds(config::Codim2Config, second_values::AbstractVector{<:Real})
    count = length(second_values)
    min_values = isempty(config.primary_min_values) ? fill(config.continuation.p_min, count) : Float64[Float64(value) for value in config.primary_min_values]
    max_values = isempty(config.primary_max_values) ? fill(config.continuation.p_max, count) : Float64[Float64(value) for value in config.primary_max_values]
    length(min_values) == count || throw(ArgumentError(
        "Codim2Config.primary_min_values length $(length(min_values)) must match the secondary sweep length $(count)."
    ))
    length(max_values) == count || throw(ArgumentError(
        "Codim2Config.primary_max_values length $(length(max_values)) must match the secondary sweep length $(count)."
    ))
    all(min_values .<= max_values) || throw(ArgumentError(
        "Codim2Config.primary_min_values and primary_max_values must satisfy min <= max on every slice."
    ))
    return min_values, max_values
end

function _codim2_special_point_candidates(branch::BranchResult,
                                          kind::Symbol,
                                          p_min::Float64,
                                          p_max::Float64,
                                          endpoint_margin::Float64)
    candidates = Float64[]
    for point in branch.branch.specialpoint
        hasproperty(point, :type) || continue
        _normalize_codim2_kind(Symbol(getproperty(point, :type))) == kind || continue
        param = Float64(getproperty(point, :param))
        (param > p_min + endpoint_margin && param < p_max - endpoint_margin) || continue
        push!(candidates, param)
    end
    sort!(candidates)
    return unique!(candidates)
end

function _codim2_stability_flip_candidates(diag::AbstractDict,
                                           p_min::Float64,
                                           p_max::Float64,
                                           endpoint_margin::Float64)
    params = get(diag, "paramValues", Float64[])
    flags = get(diag, "stabilityFlags", Bool[])
    length(params) == length(flags) || return Float64[]
    length(params) >= 2 || return Float64[]

    flips = Float64[]
    for i in 2:length(flags)
        flags[i - 1] == flags[i] && continue
        candidate = (Float64(params[i - 1]) + Float64(params[i])) / 2
        (candidate > p_min + endpoint_margin && candidate < p_max - endpoint_margin) || continue
        push!(flips, candidate)
    end
    return flips
end

function _codim2_slice_candidates(sys::DynamicalSystem,
                                  branch::BranchResult,
                                  base_params::Vector{Float64},
                                  config::Codim2Config,
                                  continuation::ContinuationConfig=config.continuation;
                                  kwargs...)
    kind = _normalize_codim2_kind(config.bifurcation_kind)
    candidates = _codim2_special_point_candidates(
        branch,
        kind,
        continuation.p_min,
        continuation.p_max,
        config.endpoint_margin
    )
    !isempty(candidates) && return candidates, :special_point, ""

    if kind == :pd && config.fallback_to_stability_flips
        diag = continuation_branch_diagnostics(
            sys,
            branch,
            base_params;
            linked_param_indices=continuation.linked_param_indices,
            max_points=config.diagnostics_max_points,
            include_residuals=false,
            include_switching_events=false,
            # Stability-flip fallback needs multiplier-based stability flags, so do not
            # disable spectra/multiplier work here.
            include_spectra=true,
        )
        flips = _codim2_stability_flip_candidates(
            diag,
            continuation.p_min,
            continuation.p_max,
            config.endpoint_margin
        )
        !isempty(flips) && return flips, :stability_flip, ""
        return Float64[], :none, "no matching period-doubling candidates were found on this slice"
    end

    return Float64[], :none, "no matching $(kind) candidates were found on this slice"
end

function _track_codim2_curve(raw::Vector{Vector{Float64}},
                             secondary_values::Vector{Float64},
                             anchor_second::Float64,
                             tolerance::Float64,
                             anchor_candidate_index::Int)
    n = length(raw)
    out = fill(NaN, n)
    n == 0 && return out

    order = sortperm(abs.(secondary_values .- anchor_second))
    anchor_idx = nothing
    for idx in order
        length(raw[idx]) >= anchor_candidate_index || continue
        anchor_idx = idx
        break
    end
    isnothing(anchor_idx) && return out

    out[anchor_idx] = sort(raw[anchor_idx])[anchor_candidate_index]
    prev = out[anchor_idx]
    for idx in (anchor_idx - 1):-1:1
        isempty(raw[idx]) && break
        candidates = sort(raw[idx])
        candidate = candidates[argmin(abs.(candidates .- prev))]
        abs(candidate - prev) > tolerance && break
        out[idx] = candidate
        prev = candidate
    end
    prev = out[anchor_idx]
    for idx in (anchor_idx + 1):n
        isempty(raw[idx]) && break
        candidates = sort(raw[idx])
        candidate = candidates[argmin(abs.(candidates .- prev))]
        abs(candidate - prev) > tolerance && break
        out[idx] = candidate
        prev = candidate
    end
    return out
end

function _extremal_codim2_curve(raw::Vector{Vector{Float64}}, mode::Symbol)
    selector = mode == :maximum ? maximum : minimum
    tracked = fill(NaN, length(raw))
    for idx in eachindex(raw)
        isempty(raw[idx]) && continue
        tracked[idx] = selector(raw[idx])
    end
    return tracked
end

"""
    codim2_curve(sys, config::Codim2Config; kwargs...)
    codim2_curve(sys, config::Codim2Config, period::Int; kwargs...)

Compute a codimension-2 bifurcation curve with the engine selected by
`config.engine`:

- `:slice_tracking` (default) sweeps the secondary parameter, runs a 1D
  continuation slice along the primary parameter at each value, and stitches
  per-slice candidate bifurcation locations. Returns a `Codim2CurveResult`
  (with `raw_candidates` preserving every detected candidate per slice).
- `:defining_system` locates one point of the locus on an anchor slice, then
  continues the minimally augmented defining system (fixed point +
  eigenvector condition for the `:pd`/`:fold` multiplier) in the secondary
  parameter with pseudo-arclength continuation. Returns a
  `Codim2ContinuationResult` whose samples follow the curve arc, including
  folds of the locus itself.

Keyword arguments are forwarded to the underlying `continuation_branch` calls
(anchor slice and, for `:slice_tracking`, every slice); for continuous systems
this is where the ODE solver and skeleton-seeding settings go.
"""
function codim2_curve(sys::DynamicalSystem,
                      config::Codim2Config;
                      initial_point::Union{Nothing, AbstractVector}=nothing,
                      params::Vector{Float64}=Float64[],
                      kwargs...)
    return codim2_curve(sys, config, 1; initial_point=initial_point, params=params, kwargs...)
end

function codim2_curve(sys::DynamicalSystem,
                      config::Codim2Config,
                      period::Int;
                      initial_point::Union{Nothing, AbstractVector}=nothing,
                      params::Vector{Float64}=Float64[],
                      kwargs...)
    period >= 1 || throw(ArgumentError("codim2_curve period must be >= 1, got $(period)."))

    if config.engine === :defining_system
        return _codim2_defining_curve(sys, config, period;
                                      initial_point=initial_point, params=params, kwargs...)
    end

    continuation = config.continuation
    second_values = collect(range(config.second_min, config.second_max, length=config.second_steps + 1))
    base_params = _codim2_base_params(sys, config, params)
    slice_seed_values = _codim2_slice_seed_values(config, second_values, base_params)
    slice_min_values, slice_max_values = _codim2_slice_primary_bounds(config, second_values)
    raw_candidates = [Float64[] for _ in eachindex(second_values)]
    candidate_sources = fill(:none, length(second_values))
    slice_statuses = fill(:not_run, length(second_values))
    slice_messages = fill("", length(second_values))
    slice_point_counts = zeros(Int, length(second_values))
    slice_special_point_counts = zeros(Int, length(second_values))

    max_required_index = maximum(vcat(
        [continuation.param_index, config.second_param_index],
        continuation.linked_param_indices,
        config.second_linked_param_indices
    ))
    length(base_params) >= max_required_index || throw(ArgumentError(
        "Codim2Config/base params require at least $max_required_index parameters, got $(length(base_params))."
    ))

    run_slice! = function(idx::Int)
        secondary_value = second_values[idx]
        slice_params = inject_param(base_params, config.second_param_index, secondary_value, config.second_linked_param_indices)
        slice_params = inject_param(slice_params, continuation.param_index, slice_seed_values[idx], continuation.linked_param_indices)
        local_continuation = continuation
        local_continuation = Setfield.@set local_continuation.p_min = slice_min_values[idx]
        local_continuation = Setfield.@set local_continuation.p_max = slice_max_values[idx]
        branch = try
            continuation_branch(
                sys,
                local_continuation,
                period;
                initial_point=initial_point,
                params=slice_params,
                kwargs...
            )
        catch err
            slice_statuses[idx] = :continuation_failed
            slice_messages[idx] = sprint(showerror, err)
            return nothing
        end

        slice_point_counts[idx] = length(_branch_points(branch))
        slice_special_point_counts[idx] = length(branch.branch.specialpoint)
        candidates, source, message = _codim2_slice_candidates(sys, branch, slice_params, config, local_continuation; kwargs...)
        raw_candidates[idx] = candidates
        candidate_sources[idx] = source
        slice_messages[idx] = message
        slice_statuses[idx] = isempty(candidates) ? :no_candidates : :ok
        return nothing
    end

    threaded = config.threaded && Threads.nthreads() > 1 && length(second_values) > 1
    if threaded
        Threads.@threads for idx in eachindex(second_values)
            run_slice!(idx)
        end
    else
        for idx in eachindex(second_values)
            run_slice!(idx)
        end
    end

    tolerance = _codim2_tracking_tolerance(config)
    anchor_second = _codim2_anchor_second(config)
    tracked = if config.tracking_mode == :nearest
        _track_codim2_curve(raw_candidates, second_values, anchor_second, tolerance, config.anchor_candidate_index)
    else
        _extremal_codim2_curve(raw_candidates, config.tracking_mode)
    end
    valid_mask = .!isnan.(tracked)
    any(valid_mask) || throw(ArgumentError(
        "No codim-2 $(config.bifurcation_kind) candidates were recovered for $(sys.name) period $(period) across the requested secondary sweep."
    ))

    return Codim2CurveResult(
        tracked,
        second_values,
        valid_mask,
        raw_candidates,
        candidate_sources,
        slice_statuses,
        slice_messages,
        slice_point_counts,
        slice_special_point_counts,
        _normalize_codim2_kind(config.bifurcation_kind),
        period,
        sys.name,
        (sys.param_names[continuation.param_index], sys.param_names[config.second_param_index]),
        :slice_tracking,
        anchor_second,
        tolerance,
        now()
    )
end

# ============================================================================
# Defining-system engine
#
# Continues the codim-1 bifurcation condition of a fixed point of the
# period-`N` map (discrete iterate or Poincaré return map) in the secondary
# parameter. Augmented unknown z = [x; v; p1] with residual
#
#   R(z, p2) = [ F^N(x; p1, p2) - x
#                (D_x F^N(x; p1, p2) + s I) v      s = +1 (:pd), -1 (:fold)
#                c' v - 1 ]
#
# so R(z, p2) = 0 defines the locus (p1(p2), p2) together with the defining
# eigenvector v (multiplier -1 for :pd, +1 for :fold). The bordered row keeps
# v away from zero; c is frozen at the seed eigenvector. The augmented system
# is continued with pseudo-arclength continuation, so the curve may fold back
# in either parameter.
# ============================================================================

_codim2_defining_shift(kind::Symbol) = kind === :pd ? 1.0 : -1.0

"Parameter injection that preserves dual-number types flowing through AD."
function _codim2_inject_any(base::AbstractVector, index::Int, value, linked::Vector{Int})
    T = promote_type(eltype(base), typeof(value))
    out = Vector{T}(undef, length(base))
    @inbounds for i in eachindex(base)
        out[i] = base[i]
    end
    out[index] = value
    for l in linked
        out[l] = value
    end
    return out
end

"""
Map operators for the defining system: `apply(x, pv)` evaluates the period-`N`
map (returns `nothing` when a Poincaré return cannot be found) and
`jacobian(x, pv)` its state derivative. The third value reports whether the
operators are automatic-differentiation capable end to end.
"""
function _codim2_defining_operators(sys::DiscreteMap, config::Codim2Config, period::Int; kwargs...)
    dim = sys.dim
    apply = (x, pv) -> begin
        sv = SVector{dim}(x)
        for _ in 1:period
            sv = sys.f(sv, pv)
        end
        collect(sv)
    end
    jac = (x, pv) -> ForwardDiff.jacobian(xx -> apply(xx, pv), collect(x))
    return apply, jac, true, 1e-6
end

_codim2_ode_settings(; solver=Tsit5(), reltol::Float64=1e-8, abstol::Float64=1e-8,
                     tmax::Union{Nothing, Float64}=nothing, min_crossing_time::Float64=1e-6,
                     fd_step::Float64=1e-6, _ignored...) =
    (solver=solver, reltol=reltol, abstol=abstol, tmax=tmax,
     min_crossing_time=min_crossing_time, fd_step=fd_step)

function _codim2_defining_operators(sys::ContinuousODE, config::Codim2Config, period::Int; kwargs...)
    ode = _codim2_ode_settings(; kwargs...)
    apply = (x, pv) -> begin
        point, found = _poincare_projected(
            sys, collect(Float64, x), pv;
            period=period, solver=ode.solver, reltol=ode.reltol, abstol=ode.abstol,
            tmax=ode.tmax, min_crossing_time=ode.min_crossing_time)
        found ? collect(Float64, point) : nothing
    end
    jac = if config.continuation.ode_jacobian_method == :variational
        (x, pv) -> begin
            map_jacobian, found = _poincare_projected_jacobian_variational(
                sys, collect(Float64, x), pv;
                period=period, solver=ode.solver, reltol=ode.reltol, abstol=ode.abstol,
                tmax=ode.tmax, min_crossing_time=ode.min_crossing_time)
            found ? map_jacobian : nothing
        end
    else
        (x, pv) -> begin
            probe = apply(x, pv)
            isnothing(probe) && return nothing
            _fd_jacobian(
                xx -> begin
                    value = apply(xx, pv)
                    isnothing(value) ? fill(1e6, length(xx)) : value
                end,
                collect(Float64, x), ode.fd_step)
        end
    end
    return apply, jac, false, ode.fd_step
end

"Real, normalized eigenvector of `Jm` for the eigenvalue nearest the defining multiplier."
function _codim2_null_vector(Jm::AbstractMatrix, kind::Symbol)
    target = kind === :pd ? -1.0 : 1.0
    decomposition = eigen(Matrix{ComplexF64}(Jm))
    idx = argmin(abs.(decomposition.values .- target))
    vc = decomposition.vectors[:, idx]
    anchor = argmax(abs.(vc))
    vc = vc ./ vc[anchor]
    v = real.(vc)
    nv = norm(v)
    nv > 0 || throw(ArgumentError("Defining eigenvector for the $(kind) condition degenerated to zero."))
    return v ./ nv, abs(decomposition.values[idx] - target)
end

"Branch-point state closest in parameter to `p_target`."
function _codim2_state_near_param(branch::BranchResult, p_target::Float64, dim::Int)
    points = _branch_points(branch)
    isempty(points) && throw(ArgumentError("Anchor-slice branch carries no points to seed the defining system."))
    best = argmin([abs(Float64(getproperty(pt, :param)) - p_target) for pt in points])
    return _branch_point_state(points[best], dim)
end

"Conservative curve-leg continuation settings derived from the secondary grid."
function _codim2_default_curve_continuation(config::Codim2Config)
    span = config.second_max - config.second_min
    span > 0 || throw(ArgumentError(
        "Codim2Config engine :defining_system requires second_max > second_min."))
    steps = max(config.second_steps, 1)
    ds = span / (4 * steps)
    return ContinuationConfig(
        p_min=config.second_min, p_max=config.second_max,
        ds=ds, dsmax=span / steps, dsmin=min(ds / 1000, 1e-6),
        max_steps=max(8 * steps, 200),
        newton_tol=config.continuation.newton_tol,
        newton_max_iter=config.continuation.newton_max_iter,
        detect_bifurcation=0,
        param_index=config.second_param_index,
        a=config.continuation.a,
        detect_fold=true,
        save_sol_every_step=1)
end

function _codim2_newton_polish(R, Jbuilder, z0::Vector{Float64}, pt, tol::Float64, max_iter::Int)
    z = copy(z0)
    residual = R(z, pt)
    for _ in 1:max_iter
        norm(residual, Inf) < tol && return z, true
        J = Jbuilder(z, pt)
        step = try
            J \ residual
        catch
            return z, false
        end
        all(isfinite, step) || return z, false
        z .-= step
        residual = R(z, pt)
    end
    return z, norm(residual, Inf) < tol
end

function _codim2_curve_samples(res)
    samples = Tuple{Float64, Vector{Float64}}[]
    isnothing(res) && return samples
    for entry in res.sol
        push!(samples, (Float64(entry.p), collect(Float64, entry.x)))
    end
    return samples
end

function _codim2_curve_folds(res)
    folds = Float64[]
    isnothing(res) && return folds
    for sp in res.specialpoint
        hasproperty(sp, :type) || continue
        getproperty(sp, :type) === :fold || continue
        push!(folds, Float64(getproperty(sp, :param)))
    end
    return folds
end

"""
Anchor-slice candidates for the defining-system engine. Casts a wide net —
special points of the requested kind, `:bp`/`:fold` points (BifurcationKit's
residual convention labels a map-multiplier +1 crossing `:bp`), and
stability-flag flips — because every candidate is subsequently verified
against the actual return-map multiplier gap before seeding.
"""
function _codim2_defining_anchor_candidates(sys::DynamicalSystem,
                                            branch::BranchResult,
                                            slice_params::Vector{Float64},
                                            config::Codim2Config,
                                            kind::Symbol)
    primary = config.continuation
    candidates = Float64[]
    for point in branch.branch.specialpoint
        hasproperty(point, :type) || continue
        point_kind = _normalize_codim2_kind(Symbol(getproperty(point, :type)))
        point_kind in (kind, :bp, :fold) || continue
        param = Float64(getproperty(point, :param))
        (param > primary.p_min + config.endpoint_margin &&
         param < primary.p_max - config.endpoint_margin) || continue
        push!(candidates, param)
    end
    if config.fallback_to_stability_flips
        diag = continuation_branch_diagnostics(
            sys, branch, slice_params;
            linked_param_indices=primary.linked_param_indices,
            max_points=config.diagnostics_max_points,
            include_residuals=false,
            include_switching_events=false,
            include_spectra=true,
        )
        append!(candidates, _codim2_stability_flip_candidates(
            diag, primary.p_min, primary.p_max, config.endpoint_margin))
    end
    sort!(candidates)
    deduped = Float64[]
    for candidate in candidates
        if isempty(deduped) || abs(candidate - deduped[end]) > 1e-9 * max(1.0, abs(candidate))
            push!(deduped, candidate)
        end
    end
    return deduped
end

"""
Forward-difference Jacobian with columns evaluated across threads. Uses the
same per-column stencil as `_fd_jacobian`; the win is wall-time when each
column evaluation is expensive (ODE defining systems: every column costs
Poincaré-return integrations). Threading changes the BLAS execution context,
so results can differ from the serial path in the last bit — numerically
equivalent, but at ill-conditioned seeds a continuation decision may flip.
"""
function _codim2_fd_jacobian_threaded(F, x, delta::Float64)
    x0 = collect(Float64, x)
    Fx = F(x0)
    J = Matrix{Float64}(undef, length(Fx), length(x0))
    Threads.@threads for j in eachindex(x0)
        h = max(delta, delta * abs(x0[j]))
        xpert = copy(x0)
        xpert[j] += h
        J[:, j] = (F(xpert) .- Fx) ./ h
    end
    return J
end

"""
Seed data for the Neimark-Sacker defining system: the complex eigenpair of `Jm`
closest to the unit circle (excluding near-real eigenvalues), returned as
rotated/scaled real and imaginary eigenvector parts (`c'w1 = 1`, `c'w2 = 0`),
the multiplier angle, the bordering vector, and the gap `| |mu| - 1 |`.
Returns `nothing` when no complex pair exists.
"""
function _codim2_ns_seed(Jm::AbstractMatrix)
    decomposition = eigen(Matrix{ComplexF64}(Jm))
    best_index = 0
    best_gap = Inf
    for (i, mu) in enumerate(decomposition.values)
        imag(mu) > 1e-8 || continue
        gap = abs(abs(mu) - 1.0)
        if gap < best_gap
            best_gap = gap
            best_index = i
        end
    end
    best_index == 0 && return nothing
    mu = decomposition.values[best_index]
    vc = decomposition.vectors[:, best_index]
    vc = vc ./ norm(vc)
    w1 = real.(vc)
    w2 = imag.(vc)
    norm(w1) > 1e-12 || return nothing
    c_vec = w1 ./ norm(w1)
    # Rotate the complex phase so c'w2 = 0, then scale so c'w1 = 1.
    phase = atan(-sum(c_vec .* w2), sum(c_vec .* w1))
    w1_rot = cos(phase) .* w1 .- sin(phase) .* w2
    w2_rot = sin(phase) .* w1 .+ cos(phase) .* w2
    scale = sum(c_vec .* w1_rot)
    abs(scale) > 1e-12 || return nothing
    w1_rot ./= scale
    w2_rot ./= scale
    theta = atan(imag(mu), real(mu))
    return (gap=best_gap, w1=w1_rot, w2=w2_rot, theta=theta, c=c_vec)
end

function _codim2_defining_curve(sys::DynamicalSystem,
                                config::Codim2Config,
                                period::Int;
                                initial_point::Union{Nothing, AbstractVector}=nothing,
                                params::Vector{Float64}=Float64[],
                                kwargs...)
    kind = _normalize_codim2_kind(config.bifurcation_kind)
    kind in (:pd, :fold, :ns) || throw(ArgumentError(
        "Codim2Config engine :defining_system supports :pd, :fold, and :ns, got $(config.bifurcation_kind)."))

    primary = config.continuation
    dim = state_dim(sys)
    kind === :ns && dim < 2 && throw(ArgumentError(
        "A Neimark-Sacker defining system needs a state dimension of at least 2, got $(dim)."))
    base_params = _codim2_base_params(sys, config, params)
    max_required_index = maximum(vcat(
        [primary.param_index, config.second_param_index],
        primary.linked_param_indices,
        config.second_linked_param_indices
    ))
    length(base_params) >= max_required_index || throw(ArgumentError(
        "Codim2Config/base params require at least $max_required_index parameters, got $(length(base_params))."
    ))

    # --- Anchor slice: locate one point on the locus with the slice machinery.
    p2_seed = _codim2_anchor_second(config)
    slice_params = inject_param(base_params, config.second_param_index, p2_seed, config.second_linked_param_indices)
    anchor_branch = continuation_branch(sys, primary, period;
                                        initial_point=initial_point, params=slice_params, kwargs...)
    candidates = _codim2_defining_anchor_candidates(sys, anchor_branch, slice_params, config, kind)
    isempty(candidates) && throw(ArgumentError(
        "Defining-system seeding found no candidate stability change on the anchor slice at secondary value $(p2_seed)."))

    # Rank anchor candidates by how close a return-map multiplier actually is to
    # the defining condition (-1 for :pd, +1 for :fold, the unit circle for :ns).
    # Stability-flip candidates can come from any multiplier exit, so ranking by
    # the gap keeps a mislabeled flip from seeding the wrong defining system.
    apply, map_jacobian, ad_capable, fd_step = _codim2_defining_operators(sys, config, period; kwargs...)
    scored = Tuple{Float64, Float64, Vector{Float64}, Any}[]
    for candidate in sort(candidates)
        x_candidate = _codim2_state_near_param(anchor_branch, candidate, dim)
        pv_candidate = inject_param(slice_params, primary.param_index, candidate, primary.linked_param_indices)
        Jm_candidate = map_jacobian(x_candidate, pv_candidate)
        isnothing(Jm_candidate) && continue
        if kind === :ns
            seed = _codim2_ns_seed(Jm_candidate)
            isnothing(seed) && continue
            push!(scored, (seed.gap, candidate, x_candidate, seed))
        else
            v_candidate, gap = _codim2_null_vector(Jm_candidate, kind)
            push!(scored, (gap, candidate, x_candidate, v_candidate))
        end
    end
    isempty(scored) && throw(ArgumentError(
        "Defining-system seeding could not evaluate the map derivative (or find a complex pair) at any anchor candidate."))
    sort!(scored; by=first)
    seed_gap, p1_seed, x_seed, seed_payload = scored[min(config.anchor_candidate_index, length(scored))]
    defining_target = kind === :pd ? "-1" : kind === :fold ? "+1" : "the unit circle"
    seed_gap <= 0.5 || throw(ArgumentError(
        "Anchor candidates at secondary value $(p2_seed) carry no return-map multiplier near " *
        "$(defining_target) (closest gap $(seed_gap)); the detected stability change is " *
        "likely a different bifurcation kind. Adjust the anchor slice or bifurcation_kind."))

    # --- Defining-system residual (augmented unknown layout per kind).
    if kind === :ns
        c_vec = seed_payload.c
        nz = 3 * dim + 2
        R = (z, pt) -> begin
            x = z[1:dim]
            w1 = z[dim + 1:2 * dim]
            w2 = z[2 * dim + 1:3 * dim]
            theta = z[3 * dim + 1]
            pv = _codim2_inject_any(base_params, primary.param_index, z[nz], primary.linked_param_indices)
            pv = _codim2_inject_any(pv, config.second_param_index, pt.p, config.second_linked_param_indices)
            fx = apply(x, pv)
            isnothing(fx) && return fill(1e6, nz)
            Jm = map_jacobian(x, pv)
            isnothing(Jm) && return fill(1e6, nz)
            vcat(fx .- x,
                 Jm * w1 .- cos(theta) .* w1 .+ sin(theta) .* w2,
                 Jm * w2 .- sin(theta) .* w1 .- cos(theta) .* w2,
                 sum(c_vec .* w1) - 1,
                 sum(c_vec .* w2))
        end
        z0 = vcat(collect(Float64, x_seed), seed_payload.w1, seed_payload.w2, seed_payload.theta, p1_seed)
    else
        c_vec = copy(seed_payload)
        shift = _codim2_defining_shift(kind)
        nz = 2 * dim + 1
        R = (z, pt) -> begin
            x = z[1:dim]
            v = z[dim + 1:2 * dim]
            pv = _codim2_inject_any(base_params, primary.param_index, z[nz], primary.linked_param_indices)
            pv = _codim2_inject_any(pv, config.second_param_index, pt.p, config.second_linked_param_indices)
            fx = apply(x, pv)
            isnothing(fx) && return fill(1e6, nz)
            Jm = map_jacobian(x, pv)
            isnothing(Jm) && return fill(1e6, nz)
            vcat(fx .- x, Jm * v .+ shift .* v, sum(c_vec .* v) - 1)
        end
        z0 = vcat(collect(Float64, x_seed), seed_payload, p1_seed)
    end
    use_threads = config.threaded && Threads.nthreads() > 1
    Jaug = if ad_capable
        (z, pt) -> ForwardDiff.jacobian(zz -> R(zz, pt), z)
    elseif use_threads
        (z, pt) -> _codim2_fd_jacobian_threaded(zz -> R(zz, pt), z, fd_step)
    else
        (z, pt) -> _fd_jacobian(zz -> R(zz, pt), z, fd_step)
    end

    # --- Polish the seed onto the locus at fixed p2.
    # Only the step-size/bounds/Newton fields of curve_continuation are used;
    # its param_index/linked_param_indices do not apply because the augmented
    # problem is continued through a synthetic lens on the secondary parameter
    # (injection uses second_param_index/second_linked_param_indices).
    curve_cfg = isnothing(config.curve_continuation) ? _codim2_default_curve_continuation(config) : config.curve_continuation
    z0, converged = _codim2_newton_polish(R, Jaug, z0, (p = p2_seed,), curve_cfg.newton_tol, max(curve_cfg.newton_max_iter, 20))
    converged || throw(ArgumentError(
        "Defining-system seed failed to converge at (primary=$(p1_seed), secondary=$(p2_seed)); " *
        "eigenvalue gap to the $(kind) condition was $(seed_gap). Adjust the anchor slice or Newton settings."))

    # --- Pseudo-arclength continuation of the augmented system in p2.
    # A fresh problem per leg keeps the two directions independent.
    lens = (@optic _.p)
    build_prob = (z_start, p_start) -> if ad_capable
        BifurcationProblem(R, copy(z_start), (p = p_start,), lens; record_from_solution=_default_record)
    else
        BifurcationProblem(R, copy(z_start), (p = p_start,), lens;
                           J=Jaug, R01=BifurcationKit.FiniteDifferences(), delta=fd_step,
                           record_from_solution=_default_record)
    end
    newton_opts = NewtonPar(tol=curve_cfg.newton_tol, max_iterations=curve_cfg.newton_max_iter)
    build_par(ds) = ContinuationPar(
        p_min=curve_cfg.p_min, p_max=curve_cfg.p_max,
        ds=ds, dsmax=curve_cfg.dsmax, dsmin=curve_cfg.dsmin,
        max_steps=curve_cfg.max_steps, newton_options=newton_opts,
        detect_bifurcation=0, a=curve_cfg.a,
        detect_fold=curve_cfg.detect_fold, save_sol_every_step=1)
    # PALC's initial tangent orientation on the augmented system does not
    # reliably follow the sign of `ds` in the parameter (the tangent is
    # normalized over the full augmented state), so tracing "backward" and
    # "forward" legs manually can walk the same direction twice. BifurcationKit's
    # `bothside` handles the two-sided trace correctly; retry with reduced steps
    # on seed-local failures.
    run_curve() = begin
        step = abs(curve_cfg.ds)
        for _ in 1:3
            result = try
                continuation(build_prob(z0, p2_seed), PALC(), build_par(step);
                             normC=norminf, verbosity=0, bothside=true)
            catch err
                @warn "codim2 defining-system continuation attempt failed" step err
                nothing
            end
            isnothing(result) || return result
            step /= 4
            step >= curve_cfg.dsmin || break
        end
        error("No defining-system continuation steps converged for the $(kind) locus of $(sys.name) (period $(period)).")
    end
    curve_result = run_curve()
    samples = _codim2_curve_samples(curve_result)
    length(samples) >= 2 || error(
        "Defining-system continuation recorded fewer than two points for the $(kind) locus of $(sys.name).")

    n = length(samples)
    secondary_values = [s[1] for s in samples]
    primary_values = [s[2][nz] for s in samples]
    states = Matrix{Float64}(undef, dim, n)
    defining_vectors = Matrix{Float64}(undef, dim, n)
    defining_vectors_imag = kind === :ns ? Matrix{Float64}(undef, dim, n) : Matrix{Float64}(undef, 0, 0)
    phase_angles = fill(NaN, n)
    for (j, (_, z)) in enumerate(samples)
        states[:, j] = z[1:dim]
        defining_vectors[:, j] = z[dim + 1:2 * dim]
        if kind === :ns
            defining_vectors_imag[:, j] = z[2 * dim + 1:3 * dim]
            phase_angles[j] = z[3 * dim + 1]
        end
    end

    fixed_point_residuals = fill(NaN, n)
    multipliers = Vector{Vector{ComplexF64}}()
    if config.curve_diagnostics
        multipliers = Vector{Vector{ComplexF64}}(undef, n)
        diagnose_sample! = function(j::Int)
            pv = inject_param(base_params, primary.param_index, primary_values[j], primary.linked_param_indices)
            pv = inject_param(pv, config.second_param_index, secondary_values[j], config.second_linked_param_indices)
            x = states[:, j]
            value = apply(x, pv)
            fixed_point_residuals[j] = isnothing(value) ? NaN : norm(value .- x)
            Jm = map_jacobian(x, pv)
            multipliers[j] = isnothing(Jm) ? ComplexF64[] : Vector{ComplexF64}(eigvals(Matrix{ComplexF64}(Jm)))
            return nothing
        end
        if use_threads
            Threads.@threads for j in 1:n
                diagnose_sample!(j)
            end
        else
            for j in 1:n
                diagnose_sample!(j)
            end
        end
    end

    curve_folds = sort!(_codim2_curve_folds(curve_result))

    return Codim2ContinuationResult(
        primary_values,
        secondary_values,
        states,
        defining_vectors,
        defining_vectors_imag,
        phase_angles,
        fixed_point_residuals,
        multipliers,
        curve_folds,
        p1_seed,
        p2_seed,
        kind,
        period,
        sys.name,
        (sys.param_names[primary.param_index], sys.param_names[config.second_param_index]),
        :defining_system,
        now()
    )
end
