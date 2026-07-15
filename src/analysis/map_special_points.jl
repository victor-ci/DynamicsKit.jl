"""
Map-aware period-doubling / fold special-point emission for continued map and
Poincaré return-map branches.

BifurcationKit assesses stability and special points with the equilibrium convention
`Re(λ) < 0` on the residual `F = Π^p(x) − x`. For a map fixed point the multiplier is
`μ = λ + 1`, so a period-doubling (`μ → −1`, `λ → −2`) never crosses the imaginary axis
and is missed, while folds (`μ → +1`, `λ → 0`) are caught. This module locates both from
map-native test functions built on the return-map multipliers:

- fold:  `φ_fold(p) = ∏ᵢ (μᵢ − 1) = det(J_map − I)`  — changes sign as a real multiplier crosses +1,
- flip:  `φ_flip(p) = ∏ᵢ (μᵢ + 1) = det(J_map + I)`  — changes sign as a real multiplier crosses −1.

A complex-conjugate pair contributes a non-negative factor to each product, so only a real
multiplier crossing the bifurcation value flips the sign — Neimark–Sacker crossings do not
produce false positives (and are not emitted here). Detected sign changes are refined by
bisection in the arclength fraction between the two bracketing branch points, re-solving the
fixed point at each trial (robust at folds, where the parameter is not monotonic).
"""

# Map test functions from the multiplier set. The product over a conjugate-closed set is
# real up to round-off; take the real part. A period-doubling is a flip (multiplier crossing
# −1), so `_map_pd_test` is the flip determinant det(J + I).
_map_fold_test(mult) = real(prod(z -> z - 1, mult))
_map_pd_test(mult) = real(prod(z -> z + 1, mult))
_map_special_test(kind::Symbol, mult) = kind === :fold ? _map_fold_test(mult) : _map_pd_test(mult)
_map_special_target(kind::Symbol) = kind === :fold ? complex(1.0) : complex(-1.0)

# Distance the critical multiplier must reach the bifurcation value for `converged`. At a
# fold the fixed-point Jacobian is singular (μ = +1) and the multiplier scales as √(Δp) in
# the parameter, so the strict gap is unreachable even when the location is pinned; accept a
# √-scaled threshold there. Period-doublings cross transversally.
_map_converge_threshold(kind::Symbol, bifurcation_tol::Float64) =
    kind === :fold ? sqrt(bifurcation_tol) : bifurcation_tol

# Multiplier nearest the bifurcation value, and its distance to it.
function _map_critical_multiplier(kind::Symbol, mult)
    target = _map_special_target(kind)
    idx = argmin(abs.(mult .- target))
    return mult[idx], abs(mult[idx] - target)
end

# Re-solve the fixed point at `params` seeded from `x_seed` and return (state, multipliers,
# converged). Uses a finite-difference Newton on the return-map residual, uniform for
# discrete maps and Poincaré return maps.
function _map_resolve_point(sys::DynamicalSystem, x_seed::AbstractVector, params::AbstractVector,
                            period::Int; tol::Float64, max_iter::Int, fd_step::Float64, mult_kwargs)
    residual = x -> _map_residual(sys, x, params, period; fd_step=fd_step, mult_kwargs...)
    xstar, ok = _newton_fd(residual, collect(Float64, x_seed), tol, max_iter, fd_step)
    ok || return (collect(Float64, x_seed), nothing, false)
    mult = try
        _map_multipliers(sys, xstar, params, period; fd_step=fd_step, mult_kwargs...)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    return (xstar, mult, mult !== nothing)
end

# Refine a bracketed sign change of the `kind` test function into a located special point.
function _refine_map_special(sys::DynamicalSystem, kind::Symbol,
                             p_lo::Float64, p_hi::Float64,
                             x_lo::Vector{Float64}, x_hi::Vector{Float64},
                             phi_lo::Float64, base::Vector{Float64}, param_index::Int,
                             linked::Vector{Int}, period::Int;
                             iterations::Int, bifurcation_tol::Float64,
                             tol::Float64, max_iter::Int, fd_step::Float64, mult_kwargs)
    t_lo, t_hi = 0.0, 1.0
    phi_low = phi_lo
    best = nothing
    best_gap = Inf
    converge_threshold = _map_converge_threshold(kind, bifurcation_tol)
    for _ in 1:iterations
        t_mid = 0.5 * (t_lo + t_hi)
        p_mid = (1 - t_mid) * p_lo + t_mid * p_hi
        x_seed = (1 - t_mid) .* x_lo .+ t_mid .* x_hi
        local_params = _inject_param(base, param_index, p_mid, linked)
        xstar, mult, ok = _map_resolve_point(sys, x_seed, local_params, period;
                                             tol=tol, max_iter=max_iter, fd_step=fd_step,
                                             mult_kwargs=mult_kwargs)
        (ok && mult !== nothing) || break
        phi_mid = _map_special_test(kind, mult)
        _, gap = _map_critical_multiplier(kind, mult)
        if gap < best_gap
            best_gap = gap
            best = (p_mid, xstar, collect(ComplexF64, mult), phi_mid)
        end
        gap <= bifurcation_tol && break
        if sign(phi_mid) == sign(phi_low)
            t_lo, phi_low = t_mid, phi_mid
        else
            t_hi = t_mid
        end
    end
    if best === nothing
        @warn "A bracketed $(kind) sign change between p = $(p_lo) and p = $(p_hi) could not be " *
              "refined (the fixed-point re-solve failed at the first bisection step); the " *
              "special point was dropped." maxlog=3
        return nothing
    end
    p_star, x_star, mult_star, phi_star = best
    critical, gap = _map_critical_multiplier(kind, mult_star)
    return MapSpecialPoint(kind, p_star, x_star, mult_star, critical, phi_star, period,
                           gap <= converge_threshold)
end

"""
    map_special_points(sys, branch::BranchResult, base_params; kwargs...) -> Vector{MapSpecialPoint}

Locate period-doubling (`:pd`) and fold (`:fold`) special points on a continued map or
Poincaré return-map `branch`, using map-native multiplier test functions instead of
BifurcationKit's equilibrium-convention detection (which misses map period-doublings).

# Keyword arguments
- `linked_param_indices`: Parameter slots tied to the swept parameter.
- `detect`: Which kinds to locate (`[:pd, :fold]` by default).
- `refine_iterations`: Maximum bisection steps per bracket.
- `bifurcation_tol`: Target distance of the critical multiplier to ∓1 for `converged`.
- `tol`, `max_iter`, `fd_step`: Fixed-point Newton controls used during refinement.
- `solver`, `reltol`, `abstol`, `tmax`, `min_crossing_time`, `ode_jacobian_method`:
  Poincaré return-map integration/derivative controls (continuous systems only).

Returns the located points sorted by parameter. Each `MapSpecialPoint` carries the kind,
parameter, fixed-point state, multiplier set, the critical multiplier, and a `converged`
flag.
"""
function map_special_points(sys::DynamicalSystem, branch::BranchResult, base_params::AbstractVector;
                            linked_param_indices::AbstractVector{<:Integer}=Int[],
                            detect=(:pd, :fold),
                            refine_iterations::Int=50,
                            bifurcation_tol::Float64=1e-7,
                            tol::Float64=1e-10, max_iter::Int=60, fd_step::Float64=1e-6,
                            solver=Tsit5(), reltol::Float64=1e-9, abstol::Float64=1e-9,
                            tmax::Union{Nothing, Float64}=nothing, min_crossing_time::Float64=1e-6,
                            ode_jacobian_method::Symbol=:variational)
    for kind in detect
        kind in (:pd, :fold) || throw(ArgumentError("map_special_points detect kinds must be :pd or :fold; got $(repr(kind))."))
    end
    points = _branch_points(branch)
    length(points) >= 2 && return _map_special_points_impl(
        sys, branch, points, base_params, linked_param_indices, detect, refine_iterations,
        bifurcation_tol, tol, max_iter, fd_step, solver, reltol, abstol, tmax,
        min_crossing_time, ode_jacobian_method)
    return MapSpecialPoint[]
end

# Build a located point directly from a branch sample that already sits on the bifurcation
# (a test-function value of exactly zero: a multiplier equals ∓1 at that sample).
function _map_special_from_sample(kind::Symbol, param::Float64, state::AbstractVector,
                                  mult::AbstractVector, phi::Float64, period::Int,
                                  bifurcation_tol::Float64)
    critical, gap = _map_critical_multiplier(kind, mult)
    return MapSpecialPoint(kind, param, collect(Float64, state), collect(ComplexF64, mult),
                           critical, phi, period, gap <= _map_converge_threshold(kind, bifurcation_tol))
end

function _map_special_points_impl(sys, branch, points, base_params, linked_param_indices,
                                  detect, refine_iterations, bifurcation_tol, tol, max_iter,
                                  fd_step, solver, reltol, abstol, tmax, min_crossing_time,
                                  ode_jacobian_method)
    period = max(branch.period, 1)
    param_index = findfirst(==(branch.param_name), sys.param_names)
    isnothing(param_index) && throw(ArgumentError(
        "branch parameter $(branch.param_name) is not defined by system $(sys.name)."))
    base = collect(Float64, base_params)
    linked = collect(Int, linked_param_indices)
    mult_kwargs = (solver=solver, reltol=reltol, abstol=abstol, tmax=tmax,
                   min_crossing_time=min_crossing_time, ode_jacobian_method=ode_jacobian_method)

    n = length(points)
    params = Vector{Float64}(undef, n)
    states = Vector{Vector{Float64}}(undef, n)
    mults = Vector{Union{Nothing, Vector{ComplexF64}}}(undef, n)
    for i in 1:n
        params[i] = Float64(points[i].param)
        states[i] = _branch_point_state(points[i])
        local_params = _inject_param(base, param_index, params[i], linked)
        mults[i] = try
            _map_multipliers(sys, states[i], local_params, period; fd_step=fd_step, mult_kwargs...)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
    end

    specials = MapSpecialPoint[]
    for kind in detect
        phi = [m === nothing ? NaN : _map_special_test(kind, m) for m in mults]
        # A sample exactly on the bifurcation (φ == 0) is emitted directly; the strict
        # opposite-sign bracketing below never fires on a zero endpoint, so it is not
        # double-counted. Sign comparisons use `< 0` / `> 0` (not the product, which can
        # underflow to a signed zero for two tiny opposite-sign values).
        for i in 1:n
            (isfinite(phi[i]) && phi[i] == 0.0) || continue
            push!(specials, _map_special_from_sample(kind, params[i], states[i], mults[i],
                                                     phi[i], period, bifurcation_tol))
        end
        for i in 1:(n - 1)
            (isfinite(phi[i]) && isfinite(phi[i + 1])) || continue
            opposite = (phi[i] < 0 && phi[i + 1] > 0) || (phi[i] > 0 && phi[i + 1] < 0)
            opposite || continue
            sp = _refine_map_special(sys, kind, params[i], params[i + 1], states[i], states[i + 1],
                                     phi[i], base, param_index, linked, period;
                                     iterations=refine_iterations, bifurcation_tol=bifurcation_tol,
                                     tol=tol, max_iter=max_iter, fd_step=fd_step, mult_kwargs=mult_kwargs)
            sp === nothing || push!(specials, sp)
        end
    end
    sort!(specials; by = s -> s.param)
    return specials
end
