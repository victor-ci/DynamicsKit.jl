"""
Border-collision bifurcation (BCB) classification for **continuous** piecewise-smooth maps,
following the Feigin/Simpson/di Bernardo determinant-sign theory.

For the two one-sided ordered `q`-return Jacobians `A_L` (guard-negative branch at the
colliding phase) and `A_R` (guard-positive branch):

- persistence vs nonsmooth fold: `sign(det(I - A_L) · det(I - A_R))` — `> 0` persistence,
  `< 0` nonsmooth fold,
- companion `2q`-cycle creation: `sign(det(I + A_L) · det(I + A_R))` — `< 0` a companion cycle
  is created.

Determinant signs are the robust classifiers. The `σ₊`/`σ₋` counts (real eigenvalues above
`+1` / below `-1`) are exported as tolerance-aware diagnostics only, since
`sign(det(I ∓ A)) = (-1)^{σ}` only away from `±1` eigenvalues. The classification is refused for
discontinuous maps and for degenerate (`±1` eigenvalue), nontransversal, or ambiguous
(multiple on-border phases) configurations, each with an explicit status. Stability is reported
separately; no chaos / robust-chaos / period-adding / torus verdict is ever inferred.

The public entry points are `border_collision_classify` (bare matrix classifier),
`border_collision_at_cycle` (classify a reconstructed `q`-cycle of a `DiscreteMap`), and
`border_collision_points` (scan a `BranchResult` for crossings, refine, and classify).
"""

# ── determinant-sign core ────────────────────────────────────────────────────────────────

_bcb_scenario_name(scenario::Symbol) = scenario === :persistence ? "persistence" :
    scenario === :nonsmooth_fold ? "nonsmooth fold" :
    scenario === :persistence_with_companion_cycle ? "persistence with a companion cycle" :
    scenario === :nonsmooth_fold_with_companion_cycle ? "nonsmooth fold with a companion cycle" :
    "undetermined"

function _bcb_scenario(persistence_sign::Int, companion_sign::Int)
    persists = persistence_sign > 0
    companion = companion_sign < 0
    if persists
        return companion ? :persistence_with_companion_cycle : :persistence
    end
    return companion ? :nonsmooth_fold_with_companion_cycle : :nonsmooth_fold
end

# σ₊ = number of real eigenvalues > 1, σ₋ = number of real eigenvalues < -1. An eigenvalue is
# counted real when its imaginary part is within `imag_tol`. `near_unit` is true when any
# eigenvalue sits within `unit_tol` of +1 or -1 (which makes the σ counts / determinant signs
# ambiguous).
function _bcb_real_eigen_counts(spectrum::AbstractVector{<:Number};
                                imag_tol::Float64, unit_tol::Float64)
    sigma_plus = 0
    sigma_minus = 0
    near_unit = false
    for lambda in spectrum
        (abs(lambda - 1) <= unit_tol || abs(lambda + 1) <= unit_tol) && (near_unit = true)
        abs(imag(lambda)) <= imag_tol || continue
        re = real(lambda)
        re > 1 && (sigma_plus += 1)
        re < -1 && (sigma_minus += 1)
    end
    return sigma_plus, sigma_minus, near_unit
end

# Continuity/rank-one residual for a continuous piecewise-smooth map at the switching manifold.
# `D = A_L - A_R` must be rank one with row space spanned by the switching normal `C` (in local
# coordinates). With a known normal the residual is the part of `D` whose rows are not parallel
# to `C`; without one it is the second singular value (a bare rank-one check). Returns
# `(residual, continuous)`; `residual === nothing` when the supplied normal is unusable.
function _bcb_continuity_residual(D::AbstractMatrix, switching_normal, tol::Float64)
    n = size(D, 1)
    if switching_normal !== nothing
        switching_normal isa AbstractVector || return (nothing, nothing)
        c = collect(Float64, switching_normal)
        length(c) == size(D, 2) || return (nothing, nothing)
        all(isfinite, c) || return (nothing, nothing)
        nc = norm(c)
        nc > 0 || return (nothing, nothing)
        chat = c ./ nc
        fit = (D * chat) * chat'
        residual = norm(D - fit) / max(norm(D), 1.0)
        return (residual, residual <= tol)
    end

    n == 1 && return (0.0, true)                 # a scalar map is always continuous at the border
    svals = svdvals(Matrix{Float64}(D))
    isempty(svals) && return (0.0, true)
    residual = (length(svals) >= 2 ? svals[2] : 0.0) / max(svals[1], 1.0)
    return (residual, residual <= tol)
end

function _bcb_logabsdet_sign(M::AbstractMatrix{<:Real})
    logabs, raw_sign = logabsdet(M)
    sign = raw_sign > 0 ? 1 : raw_sign < 0 ? -1 : 0
    return Float64(logabs), sign
end

function _bcb_signed_magnitude(logabs::Float64, sign::Int)
    sign == 0 && return 0.0
    logabs > log(floatmax(Float64)) && return sign * floatmax(Float64)
    logabs < log(nextfloat(0.0)) && return copysign(0.0, sign)
    return sign * exp(logabs)
end

function _bcb_spectral_radius(spectrum::AbstractVector{<:Number})
    isempty(spectrum) && return NaN
    return maximum(abs, spectrum)
end

function _bcb_stability(radius::Float64, tol::Float64)
    isfinite(radius) || return nothing
    radius < 1 - tol && return true
    radius > 1 + tol && return false
    return nothing                                # marginal within tolerance
end

function _bcb_inference(scenario::Symbol, status::Symbol, classification_period::Int;
                        stable_L, stable_R, companion_exists, companion_stable)
    disclaimer = "Stability is reported separately; no chaos, robust chaos, period-adding, or " *
                 "torus-creation verdict is inferred from spectral radius."
    status === :noncontinuous && return "The map is not continuous at the switching manifold " *
        "(the rank-one continuity condition failed); border-collision classification is defined " *
        "only for continuous piecewise-smooth maps and is therefore not issued. " * disclaimer
    status === :nontransversal && return "The border crossing is not transverse, so a generic " *
        "border-collision scenario is not issued. " * disclaimer
    status === :degenerate && return "A one-sided return Jacobian has an eigenvalue at +1 or -1, " *
        "so the determinant signs are ambiguous and no generic scenario is issued. " * disclaimer
    status === :multiple_border_phases && return "More than one phase (or guard component) lies " *
        "on the border, so the colliding phase is ambiguous; the collision is treated " *
        "conservatively as degenerate. " * disclaimer
    status === :invalid && return "The supplied Jacobians or switching normal are invalid, so no " *
        "classification is issued. " * disclaimer
    status === :unavailable && return "The one-sided return Jacobians could not be formed, so no " *
        "classification is issued. " * disclaimer

    q = classification_period
    period_phrase = q == 1 ? "fixed point" : "period-$q cycle"
    base = scenario === :persistence ?
        "The $period_phrase persists across the border collision (sign(det(I-A_L)·det(I-A_R)) > 0); " *
        "no nonsmooth fold and no companion cycle." :
        scenario === :nonsmooth_fold ?
        "The border collision is a nonsmooth fold (sign(det(I-A_L)·det(I-A_R)) < 0): the " *
        "$period_phrase exists on one side of the border only; no companion cycle." :
        scenario === :persistence_with_companion_cycle ?
        "The $period_phrase persists across the border collision, and a companion 2q-cycle is " *
        "created (sign(det(I+A_L)·det(I+A_R)) < 0)." :
        "The border collision is a nonsmooth fold and a companion 2q-cycle is created " *
        "(sign(det(I-A_L)·det(I-A_R)) < 0 and sign(det(I+A_L)·det(I+A_R)) < 0)."

    stab = if stable_L === nothing || stable_R === nothing
        " Base-side stability is marginal or undetermined on at least one side."
    elseif stable_L && stable_R
        " Both one-sided base states are stable."
    elseif !stable_L && !stable_R
        " Both one-sided base states are unstable."
    else
        " Exactly one one-sided base state is stable."
    end
    comp = companion_exists === true ?
        (companion_stable === nothing ? " Companion-cycle stability is undetermined/marginal." :
         companion_stable ? " The companion cycle is stable (|spectral radius| < 1)." :
         " The companion cycle is unstable.") : ""
    return base * stab * comp * " " * disclaimer
end

"""
    border_collision_classify(A_L, A_R; kwargs...) -> BorderCollisionClassification

Classify a border-collision bifurcation of a continuous piecewise-smooth map from its two
one-sided ordered `q`-return Jacobians `A_L` (guard-negative branch) and `A_R`
(guard-positive branch). The persistence-vs-fold and companion-cycle verdicts are taken from
`sign(det(I-A_L)·det(I-A_R))` and `sign(det(I+A_L)·det(I+A_R))` respectively.

# Keyword arguments
- `switching_normal`: The switching-manifold normal `C` in local coordinates. When supplied,
  continuity is verified by requiring `A_L - A_R` to be rank one with row space `span(C)`;
  a violation yields status `:noncontinuous`; an incorrectly sized, non-finite, or zero normal
  yields status `:invalid`. When omitted, only a bare rank-one check is made.
- `period`: The cycle period `q` recorded on the result (default `1`).
- `transversality`: A signed transversality measure (e.g. `d(guard)/d(param)` along the
  branch). When supplied and below `transversality_tol` in magnitude the status is
  `:nontransversal`.
- `continuity_tol`, `eigen_tol`, `stability_tol`, `transversality_tol`: Tolerances for the
  continuity residual, `±1` eigenvalue genericity, marginal stability, and transversality.
- `extra_warnings`: Warnings to merge into the result (used by the cycle/branch drivers).

Determinant signs drive the scenario; the `σ₊`/`σ₋` counts are diagnostics. A `±1` eigenvalue
(`:degenerate`), a non-continuous map (`:noncontinuous`), a non-transverse crossing
(`:nontransversal`), or invalid Jacobians/switching normals (`:invalid`) never produce a generic scenario.
"""
function border_collision_classify(A_L::AbstractMatrix, A_R::AbstractMatrix;
                                    switching_normal=nothing,
                                    period::Integer=1,
                                    transversality::Union{Nothing, Real}=nothing,
                                    continuity_tol::Float64=1e-8,
                                    eigen_tol::Float64=1e-6,
                                    stability_tol::Float64=1e-8,
                                    transversality_tol::Float64=1e-9,
                                    extra_warnings::AbstractVector=String[])
    period >= 1 || throw(ArgumentError("border_collision_classify period must be >= 1; got $period."))
    warnings = collect(String, extra_warnings)

    valid = size(A_L) == size(A_R) && size(A_L, 1) == size(A_L, 2) &&
            !isempty(A_L) && all(isfinite, A_L) && all(isfinite, A_R)
    if !valid
        return BorderCollisionClassification(;
            scenario=:undetermined, status=:invalid, period=period,
            jacobian_L=all(isfinite, A_L) && size(A_L,1)==size(A_L,2) ? Matrix{Float64}(A_L) : Matrix{Float64}(undef,0,0),
            jacobian_R=all(isfinite, A_R) && size(A_R,1)==size(A_R,2) ? Matrix{Float64}(A_R) : Matrix{Float64}(undef,0,0),
            continuity_tolerance=continuity_tol, warnings=warnings,
            inference=_bcb_inference(:undetermined, :invalid, period;
                stable_L=nothing, stable_R=nothing, companion_exists=nothing, companion_stable=nothing))
    end

    AL = Matrix{Float64}(A_L)
    AR = Matrix{Float64}(A_R)
    n = size(AL, 1)
    Id = Matrix{Float64}(I, n, n)

    logabs_minus_L, sign_minus_L = _bcb_logabsdet_sign(Id - AL)
    logabs_minus_R, sign_minus_R = _bcb_logabsdet_sign(Id - AR)
    logabs_plus_L, sign_plus_L = _bcb_logabsdet_sign(Id + AL)
    logabs_plus_R, sign_plus_R = _bcb_logabsdet_sign(Id + AR)
    det_minus_L = _bcb_signed_magnitude(logabs_minus_L, sign_minus_L)
    det_minus_R = _bcb_signed_magnitude(logabs_minus_R, sign_minus_R)
    det_plus_L = _bcb_signed_magnitude(logabs_plus_L, sign_plus_L)
    det_plus_R = _bcb_signed_magnitude(logabs_plus_R, sign_plus_R)
    persistence_sign = sign_minus_L * sign_minus_R
    companion_sign = sign_plus_L * sign_plus_R
    persistence_product = _bcb_signed_magnitude(
        logabs_minus_L + logabs_minus_R, persistence_sign)
    companion_product = _bcb_signed_magnitude(
        logabs_plus_L + logabs_plus_R, companion_sign)

    spectrum_L = eigvals(AL)
    spectrum_R = eigvals(AR)
    sigma_plus_L, sigma_minus_L, near_unit_L = _bcb_real_eigen_counts(
        spectrum_L; imag_tol=eigen_tol, unit_tol=eigen_tol)
    sigma_plus_R, sigma_minus_R, near_unit_R = _bcb_real_eigen_counts(
        spectrum_R; imag_tol=eigen_tol, unit_tol=eigen_tol)

    radius_L = _bcb_spectral_radius(spectrum_L)
    radius_R = _bcb_spectral_radius(spectrum_R)
    stable_L = _bcb_stability(radius_L, stability_tol)
    stable_R = _bcb_stability(radius_R, stability_tol)
    (stable_L === nothing || stable_R === nothing) &&
        push!(warnings, "Base-side stability is marginal (spectral radius within tolerance of 1) " *
                        "and is reported as undetermined.")

    residual, continuous = _bcb_continuity_residual(AL - AR, switching_normal, continuity_tol)
    normal_usable = switching_normal === nothing || continuous !== nothing
    !normal_usable &&
        push!(warnings, "The supplied switching normal must be a finite, nonzero vector with " *
                        "one entry per state dimension.")
    switching_normal === nothing && n > 1 &&
        push!(warnings, "No switching normal was supplied; continuity was verified only as a " *
                        "rank-one condition on A_L - A_R, not against the switching normal.")

    determinant_signs_defined = all(!=(0), (
        sign_minus_L, sign_minus_R, sign_plus_L, sign_plus_R))
    generic = determinant_signs_defined && !near_unit_L && !near_unit_R
    sigma_parity_matches =
        sign_minus_L == (isodd(sigma_plus_L) ? -1 : 1) &&
        sign_minus_R == (isodd(sigma_plus_R) ? -1 : 1) &&
        sign_plus_L == (isodd(sigma_minus_L) ? -1 : 1) &&
        sign_plus_R == (isodd(sigma_minus_R) ? -1 : 1)
    sigma_reliable = generic && sigma_parity_matches
    sigma_reliable || push!(warnings, "σ₊/σ₋ eigenvalue counts are near a ±1 threshold or " *
                                      "disagree with determinant parity and are reported as unreliable; " *
                                      "the LU-derived determinant signs are authoritative.")

    transversal = transversality === nothing ? nothing : abs(transversality) > transversality_tol

    status = if !normal_usable
        :invalid
    elseif continuous === false
        :noncontinuous
    elseif transversal === false
        :nontransversal
    elseif !generic
        :degenerate
    else
        :ok
    end
    scenario = status === :ok ? _bcb_scenario(persistence_sign, companion_sign) : :undetermined
    companion_exists = status === :ok ? companion_sign < 0 : nothing
    companion_multipliers = status === :ok ? eigvals(AL * AR) : ComplexF64[]
    companion_radius = status === :ok ?
        _bcb_spectral_radius(companion_multipliers) : nothing
    companion_stable = companion_exists === true ?
        _bcb_stability(companion_radius, stability_tol) : nothing

    inference = _bcb_inference(scenario, status, Int(period);
        stable_L=stable_L, stable_R=stable_R, companion_exists=companion_exists,
        companion_stable=companion_stable)

    return BorderCollisionClassification(;
        scenario=scenario, status=status, period=period,
        det_I_minus_L=det_minus_L, det_I_minus_R=det_minus_R,
        det_I_plus_L=det_plus_L, det_I_plus_R=det_plus_R,
        persistence_product=persistence_product, persistence_sign=persistence_sign,
        companion_product=companion_product, companion_sign=companion_sign,
        sigma_plus_L=sigma_plus_L, sigma_plus_R=sigma_plus_R,
        sigma_minus_L=sigma_minus_L, sigma_minus_R=sigma_minus_R,
        sigma_reliable=sigma_reliable,
        spectrum_L=spectrum_L, spectrum_R=spectrum_R,
        stable_L=stable_L, stable_R=stable_R,
        spectral_radius_L=radius_L, spectral_radius_R=radius_R,
        companion_exists=companion_exists, companion_admissible=nothing,
        companion_stable=companion_stable, companion_spectral_radius=companion_radius,
        companion_multipliers=companion_multipliers,
        transversal=transversal, transversality_measure=transversality,
        continuous=continuous, continuity_residual=residual, continuity_tolerance=continuity_tol,
        generic=generic, jacobian_L=AL, jacobian_R=AR,
        inference=inference, warnings=warnings)
end

# ── guard evaluation + one-sided Jacobians ───────────────────────────────────────────────

# All components of a switching guard at a state (a scalar guard becomes a length-1 vector).
function _bcb_guard_components(event::SwitchingEvent, state::AbstractVector, params::AbstractVector)
    raw = event.guard(state, params)
    if raw isa Number
        return [raw]
    elseif raw isa AbstractVector
        return collect(raw)
    end
    throw(ArgumentError("SwitchingEvent $(event.name) guard returned $(typeof(raw)); expected a number or vector."))
end

# Gradient of guard component `comp` at `state`, via AD with a finite-difference fallback.
function _bcb_guard_gradient(event::SwitchingEvent, comp::Int, state::AbstractVector,
                             params::AbstractVector; fd_step::Float64=1e-7)
    scalar = z -> _bcb_guard_components(event, z, params)[comp]
    grad = try
        g = ForwardDiff.gradient(scalar, collect(Float64, state))
        all(isfinite, g) ? g : nothing
    catch err
        err isa InterruptException && rethrow()
        (err isa MethodError || err isa TypeError) || rethrow()
        nothing
    end
    grad === nothing || return grad
    x0 = collect(Float64, state)
    g = similar(x0)
    for j in eachindex(x0)
        h = max(fd_step, fd_step * abs(x0[j]))
        xp = copy(x0); xp[j] += h
        xm = copy(x0); xm[j] -= h
        g[j] = (scalar(xp) - scalar(xm)) / (2h)
    end
    return g
end

# Centered finite-difference Jacobian of `F` at `y` with step `h`.
function _bcb_centered_jacobian(F, y::AbstractVector, h::Float64)
    n = length(y)
    cols = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        yp = copy(y); yp[j] += h
        ym = copy(y); ym[j] -= h
        cols[j] = (F(yp) .- F(ym)) ./ (2h)
    end
    return reduce(hcat, cols)
end

# Forced one-sided Jacobian of the single map step `F` at the on-border state `x_c`, on the
# `side` (`+1` guard-positive, `-1` guard-negative) of the manifold with normal `grad`. The base
# point is pushed a distance `δ` into the target side and the centered stencil uses `h = δ/4`, so
# every stencil point stays strictly on that side (the guard change `≤ h|∇g|` is smaller than the
# base guard offset `δ|∇g|`). A halved-`δ` sequence with Richardson extrapolation removes the
# leading `O(δ)` bias for smooth branches and is exact for piecewise-affine maps. When `guard` is
# given, a `δ` whose base or stencil point falls on the other side of a curved guard is skipped,
# and the extrapolation restarts from the next usable `δ`.
function _bcb_one_sided_jacobian(F, x_c::AbstractVector, grad::AbstractVector, side::Int;
                                 base_delta::Float64, steps::Int, rel_tol::Float64,
                                 guard=nothing)
    gnorm = norm(grad)
    gnorm > 0 || return (nothing, Inf, false)
    dir = (side / gnorm) .* grad
    prev_J = nothing
    prev_rich = nothing
    out = nothing
    out_resid = Inf
    for i in 0:(steps - 1)
        delta = base_delta * (0.5^i)
        y = collect(Float64, x_c) .+ delta .* dir
        if guard !== nothing && !_bcb_stencil_on_side(guard, y, delta / 4, side)
            prev_J = nothing
            prev_rich = nothing
            continue
        end
        J = _bcb_centered_jacobian(F, y, delta / 4)
        all(isfinite, J) || continue
        if prev_J === nothing
            out = J
        else
            rich = 2 .* J .- prev_J           # Richardson extrapolation (δ halves each step)
            out = rich
            if prev_rich !== nothing
                resid = norm(rich .- prev_rich) / max(norm(rich), 1.0)
                out_resid = resid
                resid <= rel_tol && return (rich, resid, true)
            end
            prev_rich = rich
        end
        prev_J = J
    end
    return (out, out_resid, out !== nothing && out_resid <= rel_tol)
end

# Whether the base point `y` and every centered-stencil point at step `h` lie strictly on `side`.
function _bcb_stencil_on_side(guard, y::AbstractVector, h::Float64, side::Int)
    on_side(z) = (value = guard(z); isfinite(value) && side * value > 0)
    on_side(y) || return false
    for j in eachindex(y)
        yp = copy(y); yp[j] += h
        ym = copy(y); ym[j] -= h
        (on_side(yp) && on_side(ym)) || return false
    end
    return true
end

# Move `x` onto the zero set of `guard` along its gradient (Newton on the guard, a few steps).
# The refined collision leaves the colliding phase up to `border_tol` off the manifold, and the
# two one-sided Jacobians are only related by a rank-one jump on the manifold itself, so they are
# formed at the projected point. Returns `(point, offset)` with the guard value left at `point`,
# or the unprojected `x` when a step fails.
function _bcb_project_to_border(guard, gradient, x::AbstractVector; steps::Int=8)
    y = collect(Float64, x)
    value = guard(y)
    isfinite(value) || return (collect(Float64, x), value)
    for _ in 1:steps
        value == 0 && break
        g = gradient(y)
        gg = dot(g, g)
        (all(isfinite, g) && gg > 0) || break
        candidate = y .- (value / gg) .* g
        next_value = guard(candidate)
        (isfinite(next_value) && abs(next_value) < abs(value)) || break
        y, value = candidate, next_value
    end
    return (y, value)
end

# Single-map-step Jacobian at an interior phase (definite side), via AD with an FD fallback.
function _bcb_phase_jacobian(sys::DiscreteMap, state::AbstractVector, params::AbstractVector;
                             fd_step::Float64=1e-7)
    dim = sys.dim
    step = z -> collect(sys.f(SVector{dim}(z), params))
    J = try
        j = ForwardDiff.jacobian(step, collect(Float64, state))
        all(isfinite, j) ? j : nothing
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    J === nothing || return J
    return _fd_jacobian(x -> collect(Float64, sys.f(SVector{dim}(x), params)),
                        collect(Float64, state), fd_step)
end

# ── cycle-level classification ───────────────────────────────────────────────────────────

# Reconstruct the period-`q` orbit through `seed` by iterating the map.
function _bcb_reconstruct_orbit(sys::DiscreteMap, seed::AbstractVector, params::AbstractVector,
                                period::Int)
    dim = sys.dim
    length(seed) == dim || return Vector{Vector{Float64}}()
    current = SVector{dim, Float64}(seed)
    orbit = Vector{Float64}[]
    for phase in 1:max(period, 1)
        push!(orbit, collect(Float64, current))
        phase == period && continue
        current = sys.f(current, params)
        all(isfinite, current) || return Vector{Vector{Float64}}()
    end
    return orbit
end

# Locate the colliding (phase, event, component) among the orbit's phases. Returns a named tuple
# or a status symbol (`:no_border_phase` / `:multiple_border_phases`).
#
# Without `crossing` the on-border entry must be unique. The branch scanner passes the
# `(event_index, component)` whose sign change it bracketed as `crossing`, plus a `side_hint`
# `(entry, primary) -> (s_neg, s_pos)` giving the side of that entry's guard on the
# guard-negative and guard-positive sides of the primary crossing (`0` when unknown). Another
# on-border entry at a different phase is then accepted as linked when both sides are known and no
# phase lies on two borders: with
# `s_neg == s_pos` the orbit touches that border without crossing it, and with opposite signs it
# crosses that border together with the primary one. In a peak-current converter, for instance,
# the current reaching the reference exactly at a clock edge puts the next phase on the other
# border at the same parameter.
function _bcb_locate_colliding(sys::DiscreteMap, orbit::AbstractVector, params::AbstractVector,
                               events::AbstractVector{SwitchingEvent}, border_tol::Float64;
                               crossing::Union{Nothing, Tuple{Int, Int}}=nothing,
                               side_hint=nothing)
    q = length(orbit)
    guard_table = Dict{Tuple{Int, Int}, Vector{Float64}}()   # (event_idx, comp) -> per-phase values
    on_border = Tuple{Int, Int, Int}[]                       # (phase, event_idx, comp)
    for (ei, event) in pairs(events)
        ncomp = length(_bcb_guard_components(event, orbit[1], params))
        for comp in 1:ncomp
            values = Float64[_bcb_guard_components(event, orbit[phase], params)[comp] for phase in 1:q]
            guard_table[(ei, comp)] = values
            for phase in 1:q
                abs(values[phase]) <= border_tol && push!(on_border, (phase, ei, comp))
            end
        end
    end
    isempty(on_border) && return :no_border_phase
    linked = Tuple{Int, Int, Int, Int, Int}[]                # (phase, event_idx, comp, s_neg, s_pos)
    if crossing === nothing
        length(on_border) == 1 || return :multiple_border_phases
        phase, ei, comp = on_border[1]
    else
        primary = filter(entry -> (entry[2], entry[3]) == crossing, on_border)
        isempty(primary) && return :no_border_phase
        length(primary) == 1 || return :multiple_border_phases
        phases = first.(on_border)
        length(unique(phases)) == length(phases) || return :multiple_border_phases
        phase, ei, comp = primary[1]
        for entry in on_border
            entry == primary[1] && continue
            s_neg, s_pos = side_hint === nothing ? (0, 0) : side_hint(entry, primary[1])
            (s_neg != 0 && s_pos != 0) || return :multiple_border_phases
            push!(linked, (entry..., s_neg, s_pos))
        end
    end
    return (phase=phase, event_index=ei, component=comp, guard_values=guard_table[(ei, comp)],
            linked=linked)
end

# A branch-scan point whose colliding phase is ambiguous, reported with its refusal status.
function _bcb_refused_point(orbit::AbstractVector, param::Float64, q::Int, event_name::String,
                            comp::Int, continuity_tol::Float64)
    classification = BorderCollisionClassification(;
        scenario=:undetermined, status=:multiple_border_phases, period=q,
        continuity_tolerance=continuity_tol,
        warnings=["Several phases lie on borders at this collision and the side each occupies " *
                  "could not be established; the colliding phase is ambiguous."],
        inference=_bcb_inference(:undetermined, :multiple_border_phases, q;
            stable_L=nothing, stable_R=nothing, companion_exists=nothing, companion_stable=nothing))
    return BorderCollisionPoint(param, orbit, 0, Int[], event_name, comp, Float64[], q,
                                classification, false)
end

# Re-express a located collision with its `k`-th linked border as the primary one. Linked sides are
# stored relative to the primary guard, so they are swapped when the new primary is positive on
# the old primary's negative side.
function _bcb_relink(orbit::AbstractVector, params::AbstractVector,
                     events::AbstractVector{SwitchingEvent}, located, k::Int)
    phase, ei, comp, a, _ = located.linked[k]
    flip = a > 0
    sidepair(s_neg, s_pos) = flip ? (s_pos, s_neg) : (s_neg, s_pos)
    linked = Tuple{Int, Int, Int, Int, Int}[(located.phase, located.event_index, located.component,
                                            sidepair(-1, 1)...)]
    for (j, l) in pairs(located.linked)
        j == k && continue
        push!(linked, (l[1], l[2], l[3], sidepair(l[4], l[5])...))
    end
    guard_values = Float64[_bcb_guard_components(events[ei], state, params)[comp] for state in orbit]
    return (phase=phase, event_index=ei, component=comp, guard_values=guard_values, linked=linked)
end

# Build the two one-sided q-return Jacobians (guard-negative `A_L`, guard-positive `A_R`) based
# at the colliding phase, differing only in that phase's one-sided factor.
function _bcb_cycle_return_jacobians(sys::DiscreteMap, orbit::AbstractVector, params::AbstractVector,
                                     event::SwitchingEvent, comp::Int, colliding_phase::Int;
                                     jacobian_base_delta::Float64, jacobian_steps::Int,
                                     jacobian_rel_tol::Float64,
                                     events::AbstractVector{SwitchingEvent}=SwitchingEvent[],
                                     linked::AbstractVector=Tuple{Int, Int, Int, Int, Int}[])
    dim = sys.dim
    q = length(orbit)
    guard = z -> _bcb_guard_components(event, z, params)[comp]
    gradient = z -> _bcb_guard_gradient(event, comp, z, params)
    x_border, _ = _bcb_project_to_border(guard, gradient, orbit[colliding_phase])
    grad = gradient(x_border)
    Fstep = z -> collect(Float64, sys.f(SVector{dim}(z), params))
    A_neg, r_neg, ok_neg = _bcb_one_sided_jacobian(Fstep, x_border, grad, -1;
        base_delta=jacobian_base_delta, steps=jacobian_steps, rel_tol=jacobian_rel_tol,
        guard=guard)
    A_pos, r_pos, ok_pos = _bcb_one_sided_jacobian(Fstep, x_border, grad, +1;
        base_delta=jacobian_base_delta, steps=jacobian_steps, rel_tol=jacobian_rel_tol,
        guard=guard)
    (A_neg === nothing || A_pos === nothing) && return (nothing, nothing, grad, false, r_neg, r_pos, 0.0)

    interior_neg = Dict{Int, Matrix{Float64}}()
    interior_pos = Dict{Int, Matrix{Float64}}()
    linked_converged = true
    r_linked = 0.0
    for (phase, ei, lcomp, s_neg, s_pos) in linked
        # A linked phase lies on another border at the collision, where automatic differentiation
        # could take either branch, so each return Jacobian uses the one-sided Jacobian of the side
        # that phase occupies on its own side of the collision.
        lguard = z -> _bcb_guard_components(events[ei], z, params)[lcomp]
        lgradient = z -> _bcb_guard_gradient(events[ei], lcomp, z, params)
        l_border, _ = _bcb_project_to_border(lguard, lgradient, orbit[phase])
        l_grad = lgradient(l_border)
        sided = Dict{Int, Matrix{Float64}}()
        for side in unique((s_neg, s_pos))
            J, r_side, ok = _bcb_one_sided_jacobian(Fstep, l_border, l_grad, side;
                base_delta=jacobian_base_delta, steps=jacobian_steps, rel_tol=jacobian_rel_tol,
                guard=lguard)
            J === nothing && return (nothing, nothing, grad, false, r_neg, r_pos, Inf)
            sided[side] = J
            linked_converged &= ok
            r_linked = max(r_linked, r_side)
        end
        interior_neg[phase] = sided[s_neg]
        interior_pos[phase] = sided[s_pos]
    end
    for phase in 1:q
        (phase == colliding_phase || haskey(interior_neg, phase)) && continue
        J = _bcb_phase_jacobian(sys, orbit[phase], params)
        interior_neg[phase] = J
        interior_pos[phase] = J
    end

    A_L = Matrix{Float64}(I, dim, dim)
    A_R = Matrix{Float64}(I, dim, dim)
    for i in 0:(q - 1)
        phase = mod1(colliding_phase + i, q)
        Jm, Jp = i == 0 ? (A_neg, A_pos) : (interior_neg[phase], interior_pos[phase])
        A_L = Jm * A_L
        A_R = Jp * A_R
    end
    converged = ok_neg && ok_pos && linked_converged
    return (A_L, A_R, grad, converged, r_neg, r_pos, r_linked)
end

# Assemble a BorderCollisionPoint from a reconstructed cycle at a known colliding phase.
function _bcb_classify_located_cycle(sys::DiscreteMap, orbit::AbstractVector, params::AbstractVector,
                                     events::AbstractVector{SwitchingEvent}, located;
                                     param::Float64, transversality, refine_converged::Bool,
                                     jacobian_base_delta::Float64, jacobian_steps::Int,
                                     jacobian_rel_tol::Float64, continuity_tol::Float64,
                                     eigen_tol::Float64, stability_tol::Float64,
                                     transversality_tol::Float64)
    q = length(orbit)
    event = events[located.event_index]
    comp = located.component
    colliding_phase = located.phase
    itinerary = Int[v > 0 ? 1 : v < 0 ? -1 : 0 for v in located.guard_values]
    itinerary[colliding_phase] = 0

    linked = get(located, :linked, Tuple{Int, Int, Int, Int, Int}[])
    A_L, A_R, grad, jac_converged, r_neg, r_pos, r_linked = _bcb_cycle_return_jacobians(
        sys, orbit, params, event, comp, colliding_phase;
        jacobian_base_delta=jacobian_base_delta, jacobian_steps=jacobian_steps,
        jacobian_rel_tol=jacobian_rel_tol, events=events, linked=linked)

    if A_L === nothing
        classification = BorderCollisionClassification(;
            scenario=:undetermined, status=:unavailable, period=q,
            continuity_tolerance=continuity_tol,
            warnings=["The one-sided return Jacobians could not be formed at the colliding phase " *
                      "(guard gradient degenerate or finite differences non-finite)."],
            inference=_bcb_inference(:undetermined, :unavailable, q;
                stable_L=nothing, stable_R=nothing, companion_exists=nothing, companion_stable=nothing))
        return BorderCollisionPoint(param, orbit, colliding_phase, itinerary, event.name, comp,
                                    located.guard_values, q, classification, false)
    end

    extra = String[]
    side_name(s) = s > 0 ? "guard-positive" : "guard-negative"
    for (phase, ei, lcomp, s_neg, s_pos) in linked
        push!(extra, s_neg == s_pos ?
            "Phase $phase also touches border $(events[ei].name) (component $lcomp) without " *
            "crossing it; its Jacobian is taken from the $(side_name(s_neg)) side, which that phase " *
            "occupies before and after the collision." :
            "Phase $phase crosses border $(events[ei].name) (component $lcomp) at the same " *
            "collision; each return Jacobian takes that phase's Jacobian from the side it occupies " *
            "($(side_name(s_neg)) with A_L, $(side_name(s_pos)) with A_R).")
    end
    jac_converged || push!(extra, "One-sided return Jacobians did not fully converge under " *
        "δ-refinement (residuals: negative=$(round(r_neg, sigdigits=3)), positive=$(round(r_pos, sigdigits=3))" *
        (isempty(linked) ? ")." : ", linked phases=$(round(r_linked, sigdigits=3)))."))

    classification = border_collision_classify(A_L, A_R; switching_normal=grad, period=q,
        transversality=transversality, continuity_tol=continuity_tol, eigen_tol=eigen_tol,
        stability_tol=stability_tol, transversality_tol=transversality_tol, extra_warnings=extra)

    return BorderCollisionPoint(param, orbit, colliding_phase, itinerary, event.name, comp,
                                located.guard_values, q, classification,
                                refine_converged && jac_converged && classification.status === :ok)
end

"""
    border_collision_at_cycle(sys, cycle, params; kwargs...) -> BorderCollisionPoint

Classify a border collision of a period-`q` cycle of a continuous piecewise-smooth
`DiscreteMap`. `cycle` is either the vector of `q` phase states (each of length `state_dim(sys)`)
or a single seed state, in which case the period-`period` orbit is reconstructed by iterating the
map. Every switching guard component is evaluated at every phase; a **single** on-border phase and
guard component must be found, and the two one-sided ordered `q`-return Jacobians are built to
differ only at that colliding phase (the other `q-1` itinerary symbols are held fixed).

The colliding phase's one-sided Jacobians are computed with forced one-sided finite differences
(robust when branch selection invalidates naive automatic differentiation at the border), at the
phase state projected onto the switching manifold; interior phases use automatic differentiation. Zero on-border phases yield status `:no_border_phase`
(reported through `:unavailable`), and multiple on-border phases yield `:multiple_border_phases`.

# Keyword arguments
- `period`: Cycle period when `cycle` is a seed state (default `length(cycle)` for a phase list,
  else `1`).
- `param`: Bifurcation parameter value recorded on the point (provenance only; default `NaN`).
- `param_name`: When given, `param` is read from `params` at that parameter's index.
- `events`: Switching events to scan (default `switching_events(sys)`).
- `transversality`: Optional signed transversality measure passed through to the classifier.
- `border_tol`: Magnitude below which a guard component is treated as on-border.
- `jacobian_base_delta`, `jacobian_steps`, `jacobian_rel_tol`: One-sided Jacobian δ-refinement
  controls.
- `continuity_tol`, `eigen_tol`, `stability_tol`, `transversality_tol`: Classifier tolerances.
"""
function border_collision_at_cycle(sys::DiscreteMap, cycle::AbstractVector, params::AbstractVector;
                                   period::Union{Nothing, Integer}=nothing,
                                   param::Real=NaN,
                                   param_name::Union{Nothing, Symbol}=nothing,
                                   events::AbstractVector{SwitchingEvent}=switching_events(sys),
                                   transversality::Union{Nothing, Real}=nothing,
                                   border_tol::Float64=1e-7,
                                   jacobian_base_delta::Float64=1e-3,
                                   jacobian_steps::Int=7,
                                   jacobian_rel_tol::Float64=1e-7,
                                   continuity_tol::Float64=1e-8,
                                   eigen_tol::Float64=1e-6,
                                   stability_tol::Float64=1e-8,
                                   transversality_tol::Float64=1e-9)
    isempty(events) && throw(ArgumentError(
        "System $(sys.name) defines no switching events; border_collision_at_cycle needs at least one."))
    orbit = if !isempty(cycle) && first(cycle) isa AbstractVector
        [collect(Float64, phase) for phase in cycle]
    else
        q = period === nothing ? 1 : Int(period)
        _bcb_reconstruct_orbit(sys, collect(Float64, cycle), collect(Float64, params), q)
    end
    isempty(orbit) && throw(ArgumentError("Could not reconstruct a finite cycle for $(sys.name)."))
    q = period === nothing ? length(orbit) : Int(period)
    length(orbit) == q || throw(ArgumentError(
        "Cycle has $(length(orbit)) phases but period=$q was requested."))
    all(phase -> length(phase) == sys.dim, orbit) || throw(ArgumentError(
        "Every cycle phase must have length $(sys.dim) for $(sys.name)."))

    param_value = param_name === nothing ? Float64(param) : begin
        idx = findfirst(==(param_name), sys.param_names)
        idx === nothing && throw(ArgumentError(
            "Parameter $param_name is not defined by system $(sys.name)."))
        Float64(params[idx])
    end

    local_params = collect(Float64, params)
    located = _bcb_locate_colliding(sys, orbit, local_params, collect(SwitchingEvent, events), border_tol)
    if located isa Symbol
        status = located === :multiple_border_phases ? :multiple_border_phases : :unavailable
        classification = BorderCollisionClassification(;
            scenario=:undetermined, status=status, period=q, continuity_tolerance=continuity_tol,
            warnings=[located === :multiple_border_phases ?
                "Multiple phases/guard components lie on the border; the colliding phase is ambiguous." :
                "No phase lies on the border within border_tol; the cycle is not at a collision."],
            inference=_bcb_inference(:undetermined, status, q;
                stable_L=nothing, stable_R=nothing, companion_exists=nothing, companion_stable=nothing))
        return BorderCollisionPoint(param_value, orbit, 0, Int[], "", 0, Float64[], q,
                                    classification, false)
    end

    return _bcb_classify_located_cycle(sys, orbit, local_params, collect(SwitchingEvent, events),
        located; param=param_value, transversality=transversality, refine_converged=true,
        jacobian_base_delta=jacobian_base_delta, jacobian_steps=jacobian_steps,
        jacobian_rel_tol=jacobian_rel_tol, continuity_tol=continuity_tol, eigen_tol=eigen_tol,
        stability_tol=stability_tol, transversality_tol=transversality_tol)
end

# ── branch scanning + refinement ─────────────────────────────────────────────────────────

# Signed "nearest-border" guard value of a cycle for one (event, component): the guard-component
# value of the phase closest to the border (sign preserved). As the parameter varies this scalar
# crosses zero exactly when the colliding phase reaches the border.
function _bcb_nearest_border_value(orbit::AbstractVector, event::SwitchingEvent, comp::Int,
                                   params::AbstractVector)
    best = NaN
    best_abs = Inf
    for phase in orbit
        value = _bcb_guard_components(event, phase, params)[comp]
        isfinite(value) || continue
        if abs(value) < best_abs
            best_abs = abs(value)
            best = value
        end
    end
    return best
end

# Re-solve the period-`q` cycle at `params` seeded from `seed` and return its orbit (or empty).
function _bcb_resolve_cycle(sys::DiscreteMap, seed::AbstractVector, params::AbstractVector,
                            period::Int; tol::Float64, max_iter::Int, fd_step::Float64)
    residual = x -> _map_residual(sys, x, params, period)
    xstar, ok = _newton_fd(residual, collect(Float64, seed), tol, max_iter, fd_step)
    ok || return Vector{Vector{Float64}}()
    return _bcb_reconstruct_orbit(sys, xstar, params, period)
end

# Bisect a bracketed sign change of the nearest-border value into a refined collision, honestly
# re-solving the periodic orbit at each trial parameter. Each midpoint is seeded by interpolating
# the current bracket and then from either bracket end, because a finite-difference Newton step
# whose stencil straddles the border can stall; a solve counts only when it has the true period,
# starts at the seeded phase, and lies near the seed, so a coexisting or lower-period cycle is not
# followed. The branch endpoints are solutions already, so the one nearer the border is the
# fallback. Returns `(param, orbit, brackets)` or `nothing`, where `brackets` lists the successive
# `(p_lo, orbit_lo, p_hi, orbit_hi)` brackets from the branch endpoints inwards.
function _bcb_refine_crossing(sys::DiscreteMap, event::SwitchingEvent, comp::Int,
                              base::Vector{Float64}, param_index::Int, linked::Vector{Int},
                              period::Int, p_lo::Float64, p_hi::Float64,
                              x_lo::Vector{Float64}, x_hi::Vector{Float64}, phi_lo::Float64;
                              iterations::Int, tol::Float64, max_iter::Int,
                              fd_step::Float64, endpoint_orbits=nothing, phi_hi::Float64=NaN)
    t_lo, t_hi = 0.0, 1.0
    phi_low = phi_lo
    best = nothing
    best_abs = Inf
    brackets = Tuple{Float64, Vector{Vector{Float64}}, Float64, Vector{Vector{Float64}}}[]
    cur_lo = (p_lo, [collect(Float64, x_lo)])
    cur_hi = (p_hi, [collect(Float64, x_hi)])
    if endpoint_orbits !== nothing
        orbit_lo, orbit_hi = endpoint_orbits
        if isfinite(phi_lo) && !isempty(orbit_lo)
            best, best_abs = (p_lo, orbit_lo), abs(phi_lo)
        end
        if isfinite(phi_hi) && !isempty(orbit_hi) && abs(phi_hi) < best_abs
            best, best_abs = (p_hi, orbit_hi), abs(phi_hi)
        end
        if !isempty(orbit_lo) && !isempty(orbit_hi)
            cur_lo, cur_hi = (p_lo, orbit_lo), (p_hi, orbit_hi)
            push!(brackets, (p_lo, orbit_lo, p_hi, orbit_hi))
        end
    end
    dim = sys.dim
    p_prev = NaN
    for _ in 1:iterations
        t_mid = 0.5 * (t_lo + t_hi)
        (t_mid == t_lo || t_mid == t_hi) && break
        p_mid = (1 - t_mid) * p_lo + t_mid * p_hi
        p_mid == p_prev && break
        p_prev = p_mid
        local_params = inject_param(base, param_index, p_mid, linked)
        a, b = cur_lo[2][1], cur_hi[2][1]
        interp = 0.5 .* (a .+ b)
        reach = max(2 * norm(b .- a), 1e-6 * max(1.0, norm(interp)))
        step = z -> collect(Float64, sys.f(SVector{dim}(z), local_params))
        orbit = Vector{Vector{Float64}}()
        for seed in (interp, a, b)
            candidate = _bcb_resolve_cycle(sys, seed, local_params, period;
                                           tol=tol, max_iter=max_iter, fd_step=fd_step)
            isempty(candidate) && continue
            d0 = norm(candidate[1] .- interp)
            d0 <= reach || continue
            all(k -> norm(candidate[k] .- interp) > d0, 2:length(candidate)) || continue
            _is_true_period(step, candidate[1], period,
                            100 * tol * max(1.0, norm(candidate[1]))) || continue
            orbit = candidate
            break
        end
        isempty(orbit) && break
        phi_mid = _bcb_nearest_border_value(orbit, event, comp, local_params)
        isfinite(phi_mid) || break
        if abs(phi_mid) < best_abs
            best_abs = abs(phi_mid)
            best = (p_mid, orbit)
        end
        phi_mid == 0 && break
        if sign(phi_mid) == sign(phi_low)
            t_lo, phi_low = t_mid, phi_mid
            cur_lo = (p_mid, orbit)
        else
            t_hi = t_mid
            cur_hi = (p_mid, orbit)
        end
        length(cur_lo[2]) == period && length(cur_hi[2]) == period &&
            push!(brackets, (cur_lo[1], cur_lo[2], cur_hi[1], cur_hi[2]))
    end
    best === nothing && return nothing
    return (param=best[1], orbit=best[2], brackets=brackets)
end

# Sides of a linked on-border entry relative to the primary crossing, read from the refinement
# brackets. A bracket is usable when the primary guard at its phase has opposite signs at the two
# ends and the linked guard is clear of the Newton noise at both; the sides are taken from the
# tightest usable bracket and must agree with the widest one, so a linked guard with another zero
# inside the branch step, or one that stays within the Newton noise of its border across the whole bracket, is refused
# (`(0, 0)`).
function _bcb_linked_sides(entry, primary, brackets, events::AbstractVector{SwitchingEvent},
                           params_at, orbit_star::AbstractVector, tol::Float64)
    phase, lei, lcomp = entry
    p_phase, pei, pcomp = primary
    guard(ei, comp, orbit, p, ph) = _bcb_guard_components(events[ei], orbit[ph], params_at(p))[comp]
    noise(ei, comp, ph) = 100 * tol * max(norm(_bcb_guard_gradient(events[ei], comp,
        orbit_star[ph], params_at(NaN))), floatmin(Float64)) * max(1.0, norm(orbit_star[ph]))
    noise_p = noise(pei, pcomp, p_phase)
    noise_l = noise(lei, lcomp, phase)
    function sides(bracket)
        p_lo, orbit_lo, p_hi, orbit_hi = bracket
        (length(orbit_lo) >= max(phase, p_phase) && length(orbit_hi) >= max(phase, p_phase)) ||
            return nothing
        gp_lo, gp_hi = guard(pei, pcomp, orbit_lo, p_lo, p_phase), guard(pei, pcomp, orbit_hi, p_hi, p_phase)
        gl_lo, gl_hi = guard(lei, lcomp, orbit_lo, p_lo, phase), guard(lei, lcomp, orbit_hi, p_hi, phase)
        all(isfinite, (gp_lo, gp_hi, gl_lo, gl_hi)) || return nothing
        (abs(gp_lo) > noise_p && abs(gp_hi) > noise_p && sign(gp_lo) != sign(gp_hi)) || return nothing
        (abs(gl_lo) > noise_l && abs(gl_hi) > noise_l) || return nothing
        s_lo, s_hi = Int(sign(gl_lo)), Int(sign(gl_hi))
        return gp_lo < 0 ? (s_lo, s_hi) : (s_hi, s_lo)
    end
    usable = [s for s in (sides(b) for b in brackets) if s !== nothing]
    isempty(usable) && return (0, 0)
    first(usable) == last(usable) || return (0, 0)
    return last(usable)
end

"""
    border_collision_points(sys, branch::BranchResult, base_params; kwargs...) -> Vector{BorderCollisionPoint}

Scan a continued map / return-map `branch` of a continuous piecewise-smooth `DiscreteMap` for
border collisions of its period-`branch.period` cycle, refine each crossing in both the parameter
and the periodic state, and classify it.

For every switching event and guard component a signed nearest-border value is tracked along the
branch; a sign change between adjacent branch points brackets a collision. Each bracket is refined
by bisection in the arclength fraction, re-solving the period-`q` fixed point at each trial
parameter (so both the collision parameter and the periodic state are honest), and then classified
with `border_collision_at_cycle`. A finite-difference `d(guard)/d(param)` across the bracket is
passed through as the transversality measure. Duplicate detections (same event/component, nearby
parameter and colliding-phase state) are removed.

Bisection runs to the resolution of the bracket. A midpoint solve counts only when it has the true
period, starts at the seeded phase and lies near the seed; a failed solve is reseeded from the
current bracket ends, and the branch endpoint nearer the border is the fallback. A phase that lies
on another border at the collision (any of `events` or of `switching_events(sys)`) is accepted when
the refinement brackets agree, from the widest to the tightest, on which side of its own border it
lies before and after the collision. It either touches that border without crossing it or crosses
it together with the primary border, and each return Jacobian takes that phase's one-sided Jacobian
from the side it lies on there; when the sides cannot be established the point is returned with status
`:multiple_border_phases`. When two borders cross together and the scanned one fails the
continuity check, each co-crossed border is tried as the colliding phase, since the return map has
a single switching manifold only in the coordinates of one of them; the reported `event_name` can
then be a border that was not in `events`.

# Keyword arguments
- `linked_param_indices`: Parameter slots tied to `branch.param_name`.
- `events`: Switching events to scan (default `switching_events(sys)`).
- `border_tol`: Magnitude below which a guard component counts as on-border at the refined
  collision, and below which the refinement is reported as converged.
- `refine_iterations`: Maximum bisection steps per bracket.
- `tol`, `max_iter`, `fd_step`: Fixed-point Newton controls used while re-solving cycles.
- `duplicate_param_tol`, `duplicate_state_tol`: Deduplication tolerances.
- `jacobian_base_delta`, `jacobian_steps`, `jacobian_rel_tol`, `continuity_tol`, `eigen_tol`,
  `stability_tol`, `transversality_tol`: One-sided Jacobian and classifier tolerances.
"""
function border_collision_points(sys::DiscreteMap, branch::BranchResult, base_params::AbstractVector;
                                 linked_param_indices::AbstractVector{<:Integer}=Int[],
                                 events::AbstractVector{SwitchingEvent}=switching_events(sys),
                                 border_tol::Float64=1e-7,
                                 refine_iterations::Int=60,
                                 tol::Float64=1e-10, max_iter::Int=60, fd_step::Float64=1e-6,
                                 duplicate_param_tol::Float64=1e-6,
                                 duplicate_state_tol::Float64=1e-6,
                                 jacobian_base_delta::Float64=1e-3,
                                 jacobian_steps::Int=7,
                                 jacobian_rel_tol::Float64=1e-7,
                                 continuity_tol::Float64=1e-8,
                                 eigen_tol::Float64=1e-6,
                                 stability_tol::Float64=1e-8,
                                 transversality_tol::Float64=1e-9)
    isempty(events) && return BorderCollisionPoint[]
    period = max(branch.period, 1)
    param_index = findfirst(==(branch.param_name), sys.param_names)
    param_index === nothing && throw(ArgumentError(
        "branch parameter $(branch.param_name) is not defined by system $(sys.name)."))
    base = collect(Float64, base_params)
    linked = collect(Int, linked_param_indices)
    event_list = collect(SwitchingEvent, events)
    # The system's other borders are checked too, so a phase that lies on one of them at the
    # collision is differentiated from the correct side even when that border is not scanned.
    touch_events = copy(event_list)
    for extra_event in switching_events(sys)
        any(e -> e.name == extra_event.name, touch_events) || push!(touch_events, extra_event)
    end

    points = _branch_points(branch)
    length(points) >= 2 || return BorderCollisionPoint[]

    n = length(points)
    params = Vector{Float64}(undef, n)
    orbits = Vector{Vector{Vector{Float64}}}(undef, n)
    for i in 1:n
        params[i] = Float64(points[i].param)
        local_params = inject_param(base, param_index, params[i], linked)
        orbits[i] = _branch_point_orbit(sys, points[i], period, local_params)
    end

    located_points = BorderCollisionPoint[]
    covered = NamedTuple{(:param, :bracket, :crossings), Tuple{Float64, Int, Set{Tuple{Int, Int}}}}[]
    for (ei, event) in pairs(event_list)
        ncomp = 0
        for i in 1:n
            isempty(orbits[i]) && continue
            ncomp = length(_bcb_guard_components(event, orbits[i][1],
                inject_param(base, param_index, params[i], linked)))
            break
        end
        for comp in 1:ncomp
            phi = fill(NaN, n)
            for i in 1:n
                isempty(orbits[i]) && continue
                local_params = inject_param(base, param_index, params[i], linked)
                phi[i] = _bcb_nearest_border_value(orbits[i], event, comp, local_params)
            end
            for i in 1:(n - 1)
                (isfinite(phi[i]) && isfinite(phi[i + 1])) || continue
                opposite = (phi[i] < 0 && phi[i + 1] > 0) || (phi[i] > 0 && phi[i + 1] < 0)
                opposite || continue
                seed_lo = orbits[i][1]
                seed_hi = orbits[i + 1][1]
                refined = _bcb_refine_crossing(sys, event, comp, base, param_index, linked, period,
                    params[i], params[i + 1], seed_lo, seed_hi, phi[i];
                    iterations=refine_iterations, tol=tol,
                    max_iter=max_iter, fd_step=fd_step,
                    endpoint_orbits=(orbits[i], orbits[i + 1]), phi_hi=phi[i + 1])
                refined === nothing && continue
                p_star, orbit_star = refined.param, refined.orbit
                local_params = inject_param(base, param_index, p_star, linked)
                span = params[i + 1] - params[i]
                transversality = span == 0 ? nothing : (phi[i + 1] - phi[i]) / span
                params_at = p -> inject_param(base, param_index, isnan(p) ? p_star : p, linked)
                side_hint = (entry, primary) -> _bcb_linked_sides(entry, primary, refined.brackets,
                    touch_events, params_at, orbit_star, tol)
                located = _bcb_locate_colliding(sys, orbit_star, local_params, touch_events, border_tol;
                                                crossing=(ei, comp), side_hint=side_hint)
                # A border crossed together with an earlier one is the same collision seen from
                # another phase; it was classified already.
                any(c -> (c.bracket == i || abs(c.param - p_star) <= duplicate_param_tol) &&
                         (ei, comp) in c.crossings, covered) && continue
                if located isa Symbol
                    # Refusals are reported, so a collision the classifier cannot resolve is not
                    # mistaken for the absence of one.
                    located === :multiple_border_phases && push!(located_points,
                        _bcb_refused_point(orbit_star, p_star, period, event.name, comp, continuity_tol))
                    continue
                end
                push!(covered, (param=p_star, bracket=i, crossings=Set([(ei, comp);
                    [(l[2], l[3]) for l in located.linked if l[4] != l[5]]])))
                converged = abs(_bcb_nearest_border_value(orbit_star, event, comp, local_params)) <= border_tol
                classify = (loc, trans) -> _bcb_classify_located_cycle(sys, orbit_star, local_params,
                    touch_events, loc;
                    param=p_star, transversality=trans, refine_converged=converged,
                    jacobian_base_delta=jacobian_base_delta, jacobian_steps=jacobian_steps,
                    jacobian_rel_tol=jacobian_rel_tol, continuity_tol=continuity_tol,
                    eigen_tol=eigen_tol, stability_tol=stability_tol,
                    transversality_tol=transversality_tol)
                point = classify(located, transversality)
                # When borders at different phases cross together, the return map has a single
                # switching manifold only in the coordinates of one of them, so a continuity
                # failure is retried with each co-crossed border as the colliding phase. Other
                # refusals stand.
                if point.classification.status === :noncontinuous
                    lo_params = inject_param(base, param_index, params[i], linked)
                    hi_params = inject_param(base, param_index, params[i + 1], linked)
                    for (k, l) in pairs(located.linked)
                        l[4] == l[5] && continue
                        alt = _bcb_relink(orbit_star, local_params, touch_events, located, k)
                        g_lo = _bcb_guard_components(touch_events[l[2]], orbits[i][l[1]], lo_params)[l[3]]
                        g_hi = _bcb_guard_components(touch_events[l[2]], orbits[i + 1][l[1]], hi_params)[l[3]]
                        alt_point = classify(alt, span == 0 ? nothing : (g_hi - g_lo) / span)
                        if alt_point.classification.status === :ok
                            point = alt_point
                            break
                        end
                    end
                end
                push!(located_points, point)
            end
        end
    end

    sort!(located_points; by = p -> (p.param, p.event_name, p.guard_component, p.colliding_phase))
    unique_points = BorderCollisionPoint[]
    for point in located_points
        duplicate = any(existing -> existing.event_name == point.event_name &&
                        existing.guard_component == point.guard_component &&
                        abs(existing.param - point.param) <= duplicate_param_tol &&
                        length(existing.orbit) == length(point.orbit) &&
                        existing.colliding_phase in eachindex(existing.orbit) &&
                        point.colliding_phase in eachindex(point.orbit) &&
                        norm(existing.orbit[existing.colliding_phase] -
                             point.orbit[point.colliding_phase], Inf) <= duplicate_state_tol,
                        unique_points)
        duplicate || push!(unique_points, point)
    end
    return unique_points
end
