# Pseudo-arclength continuation of connecting orbits.
#
# The first locus point is obtained by holding the primary parameter at the seed
# value and correcting the projection boundary-value problem. Subsequent points
# are traced by pseudo-arclength continuation with both parameters free. At every
# accepted point the saddle spectrum yields the HomCont eigenvalue test functions
# and, for homoclinic connections with a rich enough spectrum, real adjoint /
# variational transport yields the orbit-flip and inclination-flip test
# functions. Sign crossings of these smooth functions become typed special
# points.

# --- unknown-vector slot indices ---------------------------------------------

_slot_T(prob::_ConnectingProblem) = _state_count(prob) + prob.n * (_has_target(prob) ? 2 : 1) + 1
_slot_primary(prob::_ConnectingProblem) = _slot_T(prob) + 1
_slot_secondary(prob::_ConnectingProblem) = _slot_T(prob) + 2
_valid_return_time(T::Real, max_return_time::Real=Inf) =
    isfinite(T) && T > 0 && T <= max_return_time

# --- seed construction --------------------------------------------------------

"""
    _resample_states(t, states, M) -> Matrix

Linearly resample a `dim × L` trajectory onto `M + 1` uniform nodes.
"""
function _resample_states(t::AbstractVector, states::AbstractMatrix, M::Int)
    n = size(states, 1)
    tmin, tmax = first(t), last(t)
    span = tmax - tmin
    out = Matrix{Float64}(undef, n, M + 1)
    if span <= 0
        for j in 1:M + 1
            out[:, j] = states[:, 1]
        end
        return out
    end
    for j in 1:M + 1
        τ = tmin + span * (j - 1) / M
        idx = searchsortedlast(t, τ)
        idx = clamp(idx, 1, length(t) - 1)
        t0, t1 = t[idx], t[idx + 1]
        w = t1 > t0 ? (τ - t0) / (t1 - t0) : 0.0
        out[:, j] = (1 - w) .* states[:, idx] .+ w .* states[:, idx + 1]
    end
    return out
end

# --- per-point diagnostics ----------------------------------------------------

_realness_status(vals) = any(v -> abs(imag(v)) > 1e-8, vals) ? :degenerate : :available

"""
    _transport_variational(prob, U, T, p, v0) -> Vector

Forward transport of a tangent vector by the discretized variational equation
`v̇ = A(t) v` (trapezoidal, implicit), renormalizing each step. Used for the
stable-side inclination-flip test.
"""
function _transport_variational(prob::_ConnectingProblem, U::AbstractMatrix, T::Real,
                                p::AbstractVector, v0::AbstractVector)
    M = prob.M
    h = T / (2 * M)
    v = collect(float.(v0))
    nv = norm(v)
    nv > 0 && (v ./= nv)
    A_i = _field_jacobian(prob.field, Vector{Float64}(U[:, 1]), p)
    for i in 1:M
        A_ip1 = _field_jacobian(prob.field, Vector{Float64}(U[:, i + 1]), p)
        rhs = (I + h .* A_i) * v
        v = _safe_solve(I - h .* A_ip1, rhs)
        nv = norm(v)
        nv > 0 && (v ./= nv)
        A_i = A_ip1
    end
    return v
end

function _transport_variational_backward(prob::_ConnectingProblem, U::AbstractMatrix, T::Real,
                                         p::AbstractVector, v0::AbstractVector)
    M = prob.M
    h = T / (2 * M)
    v = collect(float.(v0))
    nv = norm(v)
    nv > 0 && (v ./= nv)
    A_ip1 = _field_jacobian(prob.field, Vector{Float64}(U[:, M + 1]), p)
    for i in M:-1:1
        A_i = _field_jacobian(prob.field, Vector{Float64}(U[:, i]), p)
        rhs = (I - h .* A_ip1) * v
        v = _safe_solve(I + h .* A_i, rhs)
        nv = norm(v)
        nv > 0 && (v ./= nv)
        A_ip1 = A_i
    end
    return v
end

"""
    _flip_tests(prob, U, xs, T, p; test_orbit_flip, test_inclination_flip)

Orbit-flip and inclination-flip test functions for a homoclinic connection.
Returns `code => (value, status)`. Inclination-flip functions require at least
two eigenvalues on the relevant side (to separate the weak and strong
directions); otherwise they are reported `:unavailable` rather than fabricated.
"""
function _flip_tests(prob::_ConnectingProblem, U::AbstractMatrix, xs::AbstractVector,
                     T::Real, p::AbstractVector; test_orbit_flip::Bool,
                     test_inclination_flip::Bool)
    out = Dict{Symbol, Tuple{Float64, Symbol}}()
    A = _field_jacobian(prob.field, Vector{Float64}(xs), p)
    E = eigen(A)
    vals = E.values
    vecs = E.vectors
    Winv = inv(vecs)
    stable = findall(<(0), real.(vals))
    unstable = findall(>(0), real.(vals))
    sort!(stable; by=i -> abs(real(vals[i])))    # weak (closest to axis) first
    sort!(unstable; by=i -> abs(real(vals[i])))
    u1 = collect(Float64, U[:, 1])
    uend = collect(Float64, U[:, end])

    if test_orbit_flip
        if length(stable) >= 2
            wl = real.(Winv[stable[end], :])
            dvec = uend .- xs
            nrm = norm(dvec)
            out[:ofs] = (nrm > 0 ? dot(wl, dvec) / nrm : 0.0, _realness_status(vals[stable]))
        else
            out[:ofs] = (NaN, :unavailable)
        end
        if length(unstable) >= 2
            wl = real.(Winv[unstable[end], :])
            dvec = u1 .- xs
            nrm = norm(dvec)
            out[:ofu] = (nrm > 0 ? dot(wl, dvec) / nrm : 0.0, _realness_status(vals[unstable]))
        else
            out[:ofu] = (NaN, :unavailable)
        end
    end

    if test_inclination_flip
        if !isempty(unstable) && length(stable) >= 2
            v0 = real.(vecs[:, unstable[1]])
            vend = _transport_variational(prob, U, T, p, v0)
            wl = real.(Winv[stable[1], :])
            nrm = norm(vend)
            status = _realness_status(vcat(vals[stable[1:2]], vals[unstable[1]]))
            out[:ifs] = (nrm > 0 ? dot(wl, vend) / nrm : 0.0, status)
        else
            out[:ifs] = (NaN, :unavailable)
        end
        if !isempty(stable) && length(unstable) >= 2
            v0 = real.(vecs[:, stable[1]])
            vend = _transport_variational_backward(prob, U, T, p, v0)
            wl = real.(Winv[unstable[1], :])
            nrm = norm(vend)
            status = _realness_status(vcat(vals[unstable[1:2]], vals[stable[1]]))
            out[:ifu] = (nrm > 0 ? dot(wl, vend) / nrm : 0.0, status)
        else
            out[:ifu] = (NaN, :unavailable)
        end
    end
    return out
end

struct _LocusPoint
    z::Vector{Float64}
    primary::Float64
    secondary::Float64
    T::Float64
    xs::Vector{Float64}
    xt::Vector{Float64}
    eps_start::Float64
    eps_end::Float64
    residual::Float64
    path::Symbol
    tests::Dict{Symbol, Float64}
    statuses::Dict{Symbol, Symbol}
    U::Matrix{Float64}
end

function _describe_point(prob::_ConnectingProblem, corr::_CorrectorResult, cfg::ConnectingOrbitConfig)
    z = corr.z
    U, xs, xt, T, α, β = _unpack(z, prob)
    Umat = Matrix{Float64}(U)
    xsv = collect(Float64, xs)
    xtv = collect(Float64, xt)
    p = _param_vector(prob, Float64(α), Float64(β), Float64)
    eps_start = norm(Umat[:, 1] .- xsv)
    eps_end = norm(Umat[:, end] .- xtv)

    tests = Dict{Symbol, Float64}()
    statuses = Dict{Symbol, Symbol}()
    if cfg.detect_events
        A_src = _field_jacobian(prob.field, xsv, p)
        for (k, (v, s)) in _eigen_test_functions(A_src)
            tests[k] = v
            statuses[k] = s
        end
        if prob.kind == :homoclinic && (cfg.test_orbit_flip || cfg.test_inclination_flip)
            for (k, (v, s)) in _flip_tests(prob, Umat, xsv, Float64(T), p;
                                           test_orbit_flip=cfg.test_orbit_flip,
                                           test_inclination_flip=cfg.test_inclination_flip)
                tests[k] = v
                statuses[k] = s
            end
        end
    end
    return _LocusPoint(collect(Float64, z), Float64(α), Float64(β), Float64(T),
                       xsv, xtv, eps_start, eps_end, corr.residual, corr.path,
                       tests, statuses, Umat)
end

# --- continuation sweeps ------------------------------------------------------

function _sweep!(points::Vector{_LocusPoint}, prob::_ConnectingProblem,
                 z1::Vector{Float64}, cfg::ConnectingOrbitConfig, ds_sign::Int;
                 max_return_time::Float64=Inf, projector_refresh::Int=1)
    cont = cfg.continuation
    βslot = _slot_secondary(prob)
    tslot = _slot_T(prob)
    tol = cont.newton_tol
    maxiter = cont.newton_max_iter
    β1 = z1[βslot]
    ds0 = ds_sign * abs(cont.ds)

    z_pred = copy(z1)
    z_pred[βslot] += ds0
    target = β1 + ds0
    boot = _gauss_newton(prob, z_pred; extra=z -> [z[βslot] - target], tol=tol,
                         maxiter=maxiter, use_fallback=cfg.use_fallback,
                         fallback_max_iter=cfg.fallback_max_iter,
                         projector_refresh=projector_refresh)
    boot.converged || return
    # A point with T beyond the cap must not be accepted; stop this direction.
    !_valid_return_time(boot.z[tslot], max_return_time) && return
    push!(points, _describe_point(prob, boot, cfg))

    τ = boot.z .- z1
    nτ = norm(τ)
    nτ > 0 || return
    τ ./= nτ
    z_prev = boot.z
    ds = ds0
    failures = 0
    for _ in 1:cont.max_steps
        z_pred = z_prev .+ ds .* τ
        r = _gauss_newton(prob, z_pred; extra=z -> [dot(z .- z_prev, τ) - ds], tol=tol,
                          maxiter=maxiter, use_fallback=cfg.use_fallback,
                          fallback_max_iter=cfg.fallback_max_iter,
                          projector_refresh=projector_refresh)
        if !r.converged
            ds /= 2
            failures += 1
            (abs(ds) < cont.dsmin || failures > 40) && break
            continue
        end
        # Honest termination: T beyond the cap means this direction has run its course.
        !_valid_return_time(r.z[tslot], max_return_time) && break
        failures = 0
        τnew = r.z .- z_prev
        nn = norm(τnew)
        nn > 0 || break
        τnew ./= nn
        dot(τnew, τ) < 0 && (τnew .*= -1)
        τ = τnew
        z_prev = r.z
        push!(points, _describe_point(prob, r, cfg))
        β = z_prev[βslot]
        (β < cont.p_min || β > cont.p_max) && break
        ds = sign(ds) * min(abs(ds) * 1.1, cont.dsmax)
    end
    return
end

# --- special-point detection --------------------------------------------------

function _detect_special_points(ordered::Vector{_LocusPoint})
    specials = HomoclinicSpecialPoint[]
    isempty(ordered) && return specials
    codes = Symbol[]
    for pt in ordered
        for k in keys(pt.tests)
            k in codes || push!(codes, k)
        end
    end
    for code in codes
        for i in 1:length(ordered) - 1
            a = ordered[i]
            b = ordered[i + 1]
            (haskey(a.tests, code) && haskey(b.tests, code)) || continue
            (get(a.statuses, code, :available) == :available &&
             get(b.statuses, code, :available) == :available) || continue
            va = a.tests[code]
            vb = b.tests[code]
            (isfinite(va) && isfinite(vb)) || continue
            if va == 0.0
                push!(specials, HomoclinicSpecialPoint(code, homoclinic_special_point_label(code),
                                                       i, a.primary, a.secondary, va, :available, 1.0))
            elseif va * vb < 0
                w = va / (va - vb)
                pprimary = a.primary + w * (b.primary - a.primary)
                psecondary = a.secondary + w * (b.secondary - a.secondary)
                denom = abs(va) + abs(vb) + eps()
                quality = clamp(1.0 - abs(va + vb) / denom, 0.0, 1.0)
                push!(specials, HomoclinicSpecialPoint(code, homoclinic_special_point_label(code),
                                                       i, pprimary, psecondary, 0.0, :available, quality))
            end
        end
    end
    return specials
end

# --- orbit retention ----------------------------------------------------------

function _retain_orbits(ordered::Vector{_LocusPoint}, cfg::ConnectingOrbitConfig)
    records = HomoclinicOrbitRecord[]
    isempty(ordered) && return records
    stride = max(cfg.orbit_save_stride, 1)
    indices = collect(1:stride:length(ordered))
    length(indices) > cfg.max_saved_orbits &&
        (indices = indices[round.(Int, range(1, length(indices), length=cfg.max_saved_orbits))])
    for (rank, idx) in enumerate(indices)
        pt = ordered[idx]
        M = size(pt.U, 2) - 1
        t = collect(range(0.0, pt.T, length=M + 1))
        push!(records, HomoclinicOrbitRecord(idx, t, copy(pt.U), copy(pt.xs),
                                             pt.primary, pt.secondary, pt.T,
                                             pt.eps_start, pt.eps_end))
    end
    return records
end

# --- top-level continuation ---------------------------------------------------

"""
    _run_connecting_orbit_continuation(sys, prob, seed, cfg; source_period,
                                       source_index, provenance) -> HomoclinicBranchResult

Validate the connection geometry, solve the first locus point with the primary
parameter fixed, trace the curve by pseudo-arclength continuation, then assemble
the normalized result with explicit residual/fallback provenance.

`max_return_time` (from `cfg`) is enforced throughout: the seed T is pre-checked,
the corrected first-point T is post-checked (an over-time first point is a hard
error), and the continuation sweep terminates honestly when a corrected point
crosses the cap. No clamping occurs; `T` is a free BVP unknown and a sample
beyond the cap is never accepted into the locus.
"""
function _run_connecting_orbit_continuation(
        sys::ContinuousODE, prob::_ConnectingProblem,
        seed::_ConnectingSeed, cfg::ConnectingOrbitConfig;
        source_period::Int=0, source_index::Int=0, provenance::String="")
    max_return_time = cfg.max_return_time
    projector_refresh = cfg.projector_refresh

    # Pre-check: reject a seed whose T already exceeds the cap.
    if !_valid_return_time(seed.T, max_return_time)
        throw(ArgumentError(
            "Seed truncation_time must be finite, positive, and no greater than " *
            "max_return_time $(max_return_time); received $(seed.T)."))
    end

    z_seed = _seed_vector(prob, seed)
    bc0 = _refresh_bc(prob, z_seed)
    k = _validate_geometry(prob, bc0)

    cont = cfg.continuation
    bslot = _slot_secondary(prob)
    tslot = _slot_T(prob)
    first_corr = _gauss_newton(prob, z_seed; extra=z -> [z[bslot] - seed.secondary],
                               tol=cont.newton_tol, maxiter=cont.newton_max_iter,
                               use_fallback=cfg.use_fallback,
                               fallback_max_iter=cfg.fallback_max_iter,
                               projector_refresh=projector_refresh)
    if !first_corr.converged
        throw(ErrorException(
            "Connecting-orbit corrector failed to converge on the seed point " *
            "(final residual = $(first_corr.residual)). Improve the seed orbit/saddle " *
            "guess, refine the mesh (n_mesh), or adjust epsilon_start/epsilon_end."))
    end

    # Post-check: reject a corrected seed whose T is non-positive, non-finite, or exceeds the cap.
    T_first = first_corr.z[tslot]
    _valid_return_time(T_first) ||
        throw(ErrorException(
            "Connecting-orbit corrector converged, but the corrected truncation " *
            "time T = $(T_first) is not finite and positive. Check the seed orbit and " *
            "mesh configuration."))
    if isfinite(max_return_time) && T_first > max_return_time
        throw(ErrorException(
            "Connecting-orbit corrector converged, but the corrected truncation " *
            "time T = $(T_first) exceeds max_return_time $(max_return_time). The initial " *
            "seed point cannot be accepted. Reduce truncation_time or increase " *
            "max_return_time."))
    end

    first_point = _describe_point(prob, first_corr, cfg)

    forward = _LocusPoint[]
    _sweep!(forward, prob, first_corr.z, cfg, 1;
            max_return_time=max_return_time, projector_refresh=projector_refresh)
    backward = _LocusPoint[]
    if cfg.bothside
        _sweep!(backward, prob, first_corr.z, cfg, -1;
                max_return_time=max_return_time, projector_refresh=projector_refresh)
    end

    ordered = vcat(reverse(backward), [first_point], forward)

    npts = length(ordered)
    n = prob.n
    primary_values = [pt.primary for pt in ordered]
    secondary_values = [pt.secondary for pt in ordered]
    return_times = [pt.T for pt in ordered]
    epsilon_start_values = [pt.eps_start for pt in ordered]
    epsilon_end_values = [pt.eps_end for pt in ordered]
    residuals = [pt.residual for pt in ordered]
    corrector_paths = [pt.path for pt in ordered]
    saddles = Matrix{Float64}(undef, n, npts)
    target_saddles = Matrix{Float64}(undef, n, npts)
    for (j, pt) in enumerate(ordered)
        saddles[:, j] = pt.xs
        target_saddles[:, j] = pt.xt
    end

    codes = Symbol[]
    for pt in ordered, kcode in keys(pt.tests)
        kcode in codes || push!(codes, kcode)
    end
    test_functions = Dict{Symbol, Vector{Float64}}()
    test_statuses = Dict{Symbol, Vector{Symbol}}()
    for code in codes
        test_functions[code] = [get(pt.tests, code, NaN) for pt in ordered]
        test_statuses[code] = [get(pt.statuses, code, :available) for pt in ordered]
    end

    specials = _detect_special_points(ordered)
    orbits = _retain_orbits(ordered, cfg)

    fallback_points = count(==(:fallback), corrector_paths)
    diagnostics = Dict{String, Any}(
        "kind" => String(prob.kind),
        "mesh_intervals" => prob.M,
        "epsilon_start" => prob.eps0,
        "epsilon_end" => prob.eps1,
        "deficiency" => k,
        "n_points" => npts,
        "fallback_points" => fallback_points,
        "max_residual" => isempty(residuals) ? 0.0 : maximum(residuals),
        "source_stable_dim" => bc0.ns_src,
        "target_unstable_dim" => bc0.nu_tgt,
        "seed_source" => provenance,
    )

    primary_index = prob.primary_index
    secondary_index = prob.secondary_index
    pnames = _param_name_tuple(sys, primary_index, secondary_index)

    return HomoclinicBranchResult(
        primary_values, secondary_values, return_times,
        epsilon_start_values, epsilon_end_values,
        saddles, target_saddles, test_functions, test_statuses, specials, orbits,
        residuals, corrector_paths, prob.kind,
        source_period, source_index, seed.primary,
        copy(prob.base_params), primary_index, secondary_index,
        sys.name, pnames, diagnostics, now())
end

function _param_name_tuple(sys::ContinuousODE, primary_index::Int, secondary_index::Int)
    np = length(sys.param_names)
    pname = 1 <= primary_index <= np ? sys.param_names[primary_index] : Symbol("p", primary_index)
    sname = 1 <= secondary_index <= np ? sys.param_names[secondary_index] : Symbol("p", secondary_index)
    return (pname, sname)
end
