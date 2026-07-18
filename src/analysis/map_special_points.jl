"""
Map-aware period-doubling / fold / Neimark-Sacker special-point emission for continued map and
Poincaré return-map branches.

BifurcationKit assesses stability and special points with the equilibrium convention
`Re(λ) < 0` on the residual `F = Π^p(x) − x`. For a map fixed point the multiplier is
`μ = λ + 1`, so a period-doubling (`μ → −1`, `λ → −2`) never crosses the imaginary axis
and is missed, while folds (`μ → +1`, `λ → 0`) are caught. This module locates both from
map-native test functions built on the return-map multipliers:

- fold:  `φ_fold(p) = ∏ᵢ (μᵢ − 1) = det(J_map − I)`  — changes sign as a real multiplier crosses +1,
- flip:  `φ_flip(p) = ∏ᵢ (μᵢ + 1) = det(J_map + I)`  — changes sign as a real multiplier crosses −1,
- NS:    `φ_NS(p) = |μ_c| - 1` for the non-real multiplier with positive imaginary part nearest the unit circle.

A complex-conjugate pair contributes a non-negative factor to each product, so only a real
multiplier crossing the bifurcation value flips the sign, so Neimark-Sacker crossings do not
produce false fold/flip positives. Detected sign changes are refined by
bisection in the arclength fraction between the two bracketing branch points, re-solving the
fixed point at each trial (robust at folds, where the parameter is not monotonic).
"""

# Map test functions from the multiplier set. The product over a conjugate-closed set is
# real up to round-off; take the real part. A period-doubling is a flip (multiplier crossing
# −1), so `_map_pd_test` is the flip determinant det(J + I).
_map_fold_test(mult) = real(prod(z -> z - 1, mult))
_map_pd_test(mult) = real(prod(z -> z + 1, mult))
_map_ns_candidates(mult; imag_tol::Float64=1e-7) =
    ComplexF64[z for z in mult if imag(z) > imag_tol]
_map_ns_candidates(::Nothing; imag_tol::Float64=1e-7) = ComplexF64[]

function _match_map_ns_candidates(left, right)
    edges = [(abs(left[i] - right[j]), i, j)
             for i in eachindex(left) for j in eachindex(right)]
    sort!(edges; by=first)
    used_left = Set{Int}()
    used_right = Set{Int}()
    matches = Tuple{ComplexF64, ComplexF64}[]
    for (_, left_index, right_index) in edges
        (left_index in used_left || right_index in used_right) && continue
        push!(matches, (left[left_index], right[right_index]))
        push!(used_left, left_index)
        push!(used_right, right_index)
        length(matches) == min(length(left), length(right)) && break
    end
    sort!(matches; by=match -> findfirst(==(match[1]), left))
    return matches
end

function _map_ns_multiplier(mult; imag_tol::Float64=1e-7, reference=nothing)
    candidates = _map_ns_candidates(mult; imag_tol=imag_tol)
    isempty(candidates) && return nothing
    metric = reference === nothing ? abs.(abs.(candidates) .- 1.0) :
                                     abs.(candidates .- reference)
    return candidates[argmin(metric)]
end

function _map_special_test(kind::Symbol, mult; ns_reference=nothing)
    kind === :fold && return _map_fold_test(mult)
    kind === :pd && return _map_pd_test(mult)
    critical = _map_ns_multiplier(mult; reference=ns_reference)
    return critical === nothing ? NaN : abs(critical) - 1
end

_map_special_target(kind::Symbol) = kind === :fold ? complex(1.0) : complex(-1.0)

# Distance the critical multiplier must reach the bifurcation value for `converged`. At a
# fold the fixed-point Jacobian is singular (μ = +1) and the multiplier scales as √(Δp) in
# the parameter, so the strict gap is unreachable even when the location is pinned; accept a
# √-scaled threshold there. Period-doublings cross transversally.
_map_converge_threshold(kind::Symbol, bifurcation_tol::Float64) =
    kind === :fold ? sqrt(bifurcation_tol) : bifurcation_tol

# Multiplier nearest the bifurcation value, and its distance to it.
function _map_critical_multiplier(kind::Symbol, mult; ns_reference=nothing)
    if kind === :ns
        critical = _map_ns_multiplier(mult; reference=ns_reference)
        critical === nothing && return complex(NaN), Inf
        return critical, abs(abs(critical) - 1)
    end
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
    mult = _map_special_multipliers(
        sys, xstar, params, period; fd_step=fd_step, mult_kwargs...)
    return (xstar, mult, mult !== nothing)
end

function _map_special_multipliers(sys, state, params, period; kwargs...)
    try
        return _map_multipliers(sys, state, params, period; kwargs...)
    catch err
        err isa InterruptException && rethrow()
        err isa Union{
            ErrorException,
            DomainError,
            LinearAlgebra.LAPACKException,
            LinearAlgebra.SingularException,
            LinearAlgebra.PosDefException,
            LinearAlgebra.ZeroPivotException,
        } || rethrow()
        @warn "Map multiplier evaluation was unavailable; the affected special-point sample was skipped." system=sys.name period exception=(err, catch_backtrace()) maxlog=3
        return nothing
    end
end

# Refine a bracketed sign change of the `kind` test function into a located special point.
function _refine_map_special(sys::DynamicalSystem, kind::Symbol,
                             p_lo::Float64, p_hi::Float64,
                             x_lo::Vector{Float64}, x_hi::Vector{Float64},
                             phi_lo::Float64, base::Vector{Float64}, param_index::Int,
                             linked::Vector{Int}, period::Int;
                             iterations::Int, bifurcation_tol::Float64,
                             tol::Float64, max_iter::Int, fd_step::Float64, mult_kwargs,
                             ns_reference_lo=nothing, ns_reference_hi=nothing)
    t_lo, t_hi = 0.0, 1.0
    phi_low = phi_lo
    reference_low = ns_reference_lo
    reference_high = ns_reference_hi
    best = nothing
    best_gap = Inf
    converge_threshold = _map_converge_threshold(kind, bifurcation_tol)
    for _ in 1:iterations
        t_mid = 0.5 * (t_lo + t_hi)
        (t_mid == t_lo || t_mid == t_hi) && break
        p_mid = (1 - t_mid) * p_lo + t_mid * p_hi
        x_seed = (1 - t_mid) .* x_lo .+ t_mid .* x_hi
        local_params = _inject_param(base, param_index, p_mid, linked)
        xstar, mult, ok = _map_resolve_point(sys, x_seed, local_params, period;
                                             tol=tol, max_iter=max_iter, fd_step=fd_step,
                                             mult_kwargs=mult_kwargs)
        (ok && mult !== nothing) || break
        reference = if kind === :ns
            alpha = (t_mid - t_lo) / (t_hi - t_lo)
            (1 - alpha) * reference_low + alpha * reference_high
        else
            nothing
        end
        critical, gap = _map_critical_multiplier(kind, mult; ns_reference=reference)
        phi_mid = kind === :ns ? abs(critical) - 1 : _map_special_test(kind, mult)
        if gap < best_gap
            best_gap = gap
            best = (p_mid, xstar, collect(ComplexF64, mult), phi_mid, critical)
        end
        gap <= bifurcation_tol && break
        if sign(phi_mid) == sign(phi_low)
            t_lo, phi_low = t_mid, phi_mid
            kind === :ns && (reference_low = critical)
        else
            t_hi = t_mid
            kind === :ns && (reference_high = critical)
        end
    end
    if best === nothing
        @warn "A bracketed $(kind) sign change between p = $(p_lo) and p = $(p_hi) could not be " *
              "refined (the fixed-point re-solve failed at the first bisection step); the " *
              "special point was dropped." maxlog=3
        return nothing
    end
    p_star, x_star, mult_star, phi_star, critical = best
    gap = kind === :ns ? abs(abs(critical) - 1) :
          last(_map_critical_multiplier(kind, mult_star))
    return MapSpecialPoint(kind, p_star, x_star, mult_star, critical, phi_star, period,
                           gap <= converge_threshold)
end

"""
    map_special_points(sys, branch::BranchResult, base_params; kwargs...) -> Vector{MapSpecialPoint}

Locate period-doubling (`:pd`), fold (`:fold`), and Neimark-Sacker (`:ns`) special
points on a continued map or
Poincaré return-map `branch`, using map-native multiplier test functions instead of
BifurcationKit's equilibrium-convention detection (which misses map period-doublings).

# Keyword arguments
- `linked_param_indices`: Parameter slots tied to the swept parameter.
- `detect`: Which kinds to locate (`[:pd, :fold, :ns]` by default).
- `attach_normal_forms`: Compute and attach `MapNormalForm` classifications (default `true`).
- `normal_form_fd_step`: Centered finite-difference step for continuous-system normal forms.
- `duplicate_param_tol`, `duplicate_state_tol`, `duplicate_multiplier_tol`: Parameter,
  state, and critical-multiplier tolerances used to remove repeated detections of the
  same special point.
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
                            detect=(:pd, :fold, :ns),
                            attach_normal_forms::Bool=true,
                            normal_form_fd_step::Float64=3e-3,
                            duplicate_param_tol::Float64=1e-7,
                            duplicate_state_tol::Float64=1e-6,
                            duplicate_multiplier_tol::Float64=1e-6,
                            refine_iterations::Int=50,
                            bifurcation_tol::Float64=1e-7,
                            tol::Float64=1e-10, max_iter::Int=60, fd_step::Float64=1e-6,
                            solver=Tsit5(), reltol::Float64=1e-9, abstol::Float64=1e-9,
                            tmax::Union{Nothing, Float64}=nothing, min_crossing_time::Float64=1e-6,
                            ode_jacobian_method::Symbol=:variational)
    for kind in detect
        kind in (:pd, :fold, :ns) || throw(ArgumentError(
            "map_special_points detect kinds must be :pd, :fold, or :ns; got $(repr(kind))."))
    end
    duplicate_param_tol >= 0 || throw(ArgumentError(
        "duplicate_param_tol must be non-negative; got $duplicate_param_tol."))
    duplicate_state_tol >= 0 || throw(ArgumentError(
        "duplicate_state_tol must be non-negative; got $duplicate_state_tol."))
    duplicate_multiplier_tol >= 0 || throw(ArgumentError(
        "duplicate_multiplier_tol must be non-negative; got $duplicate_multiplier_tol."))
    normal_form_fd_step > 0 || throw(ArgumentError(
        "normal_form_fd_step must be positive; got $normal_form_fd_step."))
    points = _branch_points(branch)
    length(points) >= 2 && return _map_special_points_impl(
        sys, branch, points, base_params, linked_param_indices, detect, refine_iterations,
        bifurcation_tol, tol, max_iter, fd_step, solver, reltol, abstol, tmax,
        min_crossing_time, ode_jacobian_method, attach_normal_forms,
        normal_form_fd_step, duplicate_param_tol, duplicate_state_tol,
        duplicate_multiplier_tol)
    return MapSpecialPoint[]
end

# Build a located point directly from a branch sample that already sits on the bifurcation
# (a test-function value of exactly zero: a multiplier equals ∓1 at that sample).
function _map_special_from_sample(kind::Symbol, param::Float64, state::AbstractVector,
                                  mult::AbstractVector, phi::Float64, period::Int,
                                  bifurcation_tol::Float64; ns_reference=nothing)
    critical, gap = _map_critical_multiplier(
        kind, mult; ns_reference=ns_reference)
    return MapSpecialPoint(kind, param, collect(Float64, state), collect(ComplexF64, mult),
                           critical, phi, period, gap <= _map_converge_threshold(kind, bifurcation_tol))
end

function _map_special_points_impl(sys, branch, points, base_params, linked_param_indices,
                                  detect, refine_iterations, bifurcation_tol, tol, max_iter,
                                  fd_step, solver, reltol, abstol, tmax, min_crossing_time,
                                  ode_jacobian_method, attach_normal_forms,
                                  normal_form_fd_step, duplicate_param_tol,
                                  duplicate_state_tol, duplicate_multiplier_tol)
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
        mults[i] = _map_special_multipliers(
            sys, states[i], local_params, period; fd_step=fd_step, mult_kwargs...)
    end

    specials = MapSpecialPoint[]
    for kind in detect
        phi = kind === :ns ? Float64[] :
              [m === nothing ? NaN : _map_special_test(kind, m) for m in mults]
        # A sample exactly on the bifurcation is emitted directly. Sign comparisons
        # use `< 0` / `> 0` (not the product, which can underflow to a signed zero
        # for two tiny opposite-sign values).
        for i in 1:n
            if kind === :ns
                for critical in _map_ns_candidates(mults[i])
                    critical_phi = abs(critical) - 1
                    critical_phi == 0.0 || continue
                    push!(specials, _map_special_from_sample(
                        kind, params[i], states[i], mults[i], critical_phi, period,
                        bifurcation_tol; ns_reference=critical))
                end
            else
                (isfinite(phi[i]) && phi[i] == 0.0) || continue
                push!(specials, _map_special_from_sample(
                    kind, params[i], states[i], mults[i], phi[i], period,
                    bifurcation_tol))
            end
        end
        for i in 1:(n - 1)
            if kind === :ns
                left = _map_ns_candidates(mults[i])
                right = _map_ns_candidates(mults[i + 1])
                endpoint_matches = _match_map_ns_candidates(left, right)
                any(match -> (abs(match[1]) - 1) * (abs(match[2]) - 1) < 0,
                    endpoint_matches) || continue

                p_mid = 0.5 * (params[i] + params[i + 1])
                x_mid_seed = 0.5 .* (states[i] .+ states[i + 1])
                local_params = _inject_param(base, param_index, p_mid, linked)
                _, midpoint_mult, midpoint_ok = _map_resolve_point(
                    sys, x_mid_seed, local_params, period;
                    tol=tol, max_iter=max_iter, fd_step=fd_step,
                    mult_kwargs=mult_kwargs)
                (midpoint_ok && midpoint_mult !== nothing) || continue
                midpoint = _map_ns_candidates(midpoint_mult)
                left_mid = _match_map_ns_candidates(left, midpoint)
                mid_right = _match_map_ns_candidates(midpoint, right)

                for (left_value, midpoint_value) in left_mid
                    right_match = findfirst(match -> match[1] == midpoint_value, mid_right)
                    isnothing(right_match) && continue
                    right_value = mid_right[right_match][2]
                    phi_left = abs(left_value) - 1
                    phi_right = abs(right_value) - 1
                    opposite = (phi_left < 0 && phi_right > 0) ||
                               (phi_left > 0 && phi_right < 0)
                    opposite || continue
                    sp = _refine_map_special(
                        sys, kind, params[i], params[i + 1], states[i], states[i + 1],
                        phi_left, base, param_index, linked, period;
                        iterations=refine_iterations, bifurcation_tol=bifurcation_tol,
                        tol=tol, max_iter=max_iter, fd_step=fd_step,
                        mult_kwargs=mult_kwargs, ns_reference_lo=left_value,
                        ns_reference_hi=right_value)
                    sp === nothing || push!(specials, sp)
                end
            else
                (isfinite(phi[i]) && isfinite(phi[i + 1])) || continue
                opposite = (phi[i] < 0 && phi[i + 1] > 0) ||
                           (phi[i] > 0 && phi[i + 1] < 0)
                opposite || continue
                sp = _refine_map_special(
                    sys, kind, params[i], params[i + 1], states[i], states[i + 1],
                    phi[i], base, param_index, linked, period;
                    iterations=refine_iterations, bifurcation_tol=bifurcation_tol,
                    tol=tol, max_iter=max_iter, fd_step=fd_step,
                    mult_kwargs=mult_kwargs)
                sp === nothing || push!(specials, sp)
            end
        end
    end
    sort!(specials; by = s -> (
        s.param,
        String(s.kind),
        s.kind === :ns ? angle(s.critical_multiplier) : 0.0,
        real(s.critical_multiplier),
        imag(s.critical_multiplier),
    ))
    unique_specials = MapSpecialPoint[]
    for special in specials
        duplicate = any(existing -> existing.kind === special.kind &&
                                    abs(existing.param - special.param) <= duplicate_param_tol &&
                                    norm(existing.state - special.state, Inf) <=
                                    duplicate_state_tol &&
                                    (special.kind !== :ns ||
                                     abs(existing.critical_multiplier -
                                         special.critical_multiplier) <=
                                     duplicate_multiplier_tol),
                        unique_specials)
        duplicate || push!(unique_specials, special)
    end
    attach_normal_forms || return unique_specials

    return map(unique_specials) do special
        local_params = _inject_param(base, param_index, special.param, linked)
        normal_form = map_normal_form(
            sys, special, local_params; normal_form_fd_step=normal_form_fd_step,
            solver=solver, reltol=reltol, abstol=abstol, tmax=tmax,
            min_crossing_time=min_crossing_time)
        MapSpecialPoint(
            special.kind, special.param, special.state, special.multipliers,
            special.critical_multiplier, special.test_value, special.period,
            special.converged, normal_form)
    end
end
