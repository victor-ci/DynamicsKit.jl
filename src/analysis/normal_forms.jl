const _MAP_FOLD_CONVENTION =
    "Kuznetsov/MATCONT map convention: b=1/2<p,B(q,q)>; sign depends on the chosen real eigenvector orientation."
const _MAP_FLIP_CONVENTION =
    "Kuznetsov/MATCONT map convention: c=1/6<p,C(q,q,q)>+<p,B(q,h20)>, h20=(I-A)^-1 B(q,q)/2; c>0 supercritical/soft, c<0 subcritical/hard."
const _MAP_NS_CONVENTION =
    "Kuznetsov/MATCONT map convention: d=Re(conj(lambda)/2*(<p,C(q,q,qbar)>+<p,B(h20,qbar)>+2<p,B(h11,q)>)), h11=-(A-I)^-1 B(q,qbar), h20=-(A-lambda^2 I)^-1 B(q,q); d<0 supercritical, d>0 subcritical."

_normal_form_name(kind::Symbol) =
    kind === :fold ? :b : kind === :pd ? :c : kind === :ns ? :d :
    throw(ArgumentError("Map normal-form kind must be :fold, :pd, or :ns; got $(repr(kind))."))

_normal_form_convention(kind::Symbol) =
    kind === :fold ? _MAP_FOLD_CONVENTION :
    kind === :pd ? _MAP_FLIP_CONVENTION : _MAP_NS_CONVENTION

function _normal_form_result(kind::Symbol, coefficient, criticality::Symbol, status::Symbol)
    value = coefficient === nothing ? nothing : Float64(real(coefficient))
    return MapNormalForm(kind, _normal_form_name(kind), value, criticality, status,
                         _normal_form_convention(kind))
end

function _iterate_map(sys::DiscreteMap, state, params, period::Int)
    current = SVector{sys.dim}(state)
    for _ in 1:period
        current = sys.f(current, params)
    end
    return collect(current)
end

function _normal_form_map(sys::DiscreteMap, params, period::Int; kwargs...)
    local_params = collect(Float64, params)
    return x -> _iterate_map(sys, x, local_params, period)
end

function _normal_form_map(sys::ContinuousODE, params, period::Int;
                          solver=Tsit5(), reltol::Float64=1e-9, abstol::Float64=1e-9,
                          tmax::Union{Nothing, Float64}=nothing,
                          min_crossing_time::Float64=1e-6, kwargs...)
    local_params = collect(Float64, params)
    return x -> begin
        next_point, found = _poincare_projected(
            sys, x, local_params; period=period, solver=solver, reltol=reltol,
            abstol=abstol, tmax=tmax, min_crossing_time=min_crossing_time)
        found || return fill(NaN, length(x))
        collect(Float64, next_point)
    end
end

function _ad_first_directional(G, x, v)
    return [ForwardDiff.derivative(t -> G(x .+ t .* v)[i], zero(eltype(x)))
            for i in eachindex(G(x))]
end

function _ad_second_directional(G, x, u, v)
    return [ForwardDiff.derivative(
                t -> _ad_first_directional(G, x .+ t .* u, v)[i],
                zero(eltype(x)))
            for i in eachindex(G(x))]
end

function _ad_third_directional(G, x, u, v, w)
    return [ForwardDiff.derivative(
                t -> _ad_second_directional(G, x .+ t .* w, u, v)[i],
                zero(eltype(x)))
            for i in eachindex(G(x))]
end

function _discrete_normal_form_derivatives(G, x)
    A = ForwardDiff.jacobian(G, x)
    B = (u, v) -> _ad_second_directional(G, x, u, v)
    C = (u, v, w) -> _ad_third_directional(G, x, u, v, w)
    return A, B, C
end

function _central_multilinear(G, x, directions, h::Float64)
    n = length(directions)
    total = zeros(Float64, length(G(x)))
    for mask in 0:(2^n - 1)
        signs = ntuple(j -> ((mask >> (j - 1)) & 1) == 1 ? 1.0 : -1.0, n)
        offset = zeros(Float64, length(x))
        weight = 1.0
        for j in 1:n
            offset .+= signs[j] .* directions[j]
            weight *= signs[j]
        end
        total .+= weight .* G(x .+ h .* offset)
    end
    return total ./ ((2h)^n)
end

function _continuous_normal_form_derivatives(G, x, h::Float64)
    n = length(x)
    A = Matrix{Float64}(undef, n, n)
    for j in 1:n
        direction = zeros(Float64, n)
        direction[j] = 1.0
        A[:, j] = (G(x .+ h .* direction) .- G(x .- h .* direction)) ./ (2h)
    end
    B = (u, v) -> _central_multilinear(G, x, (u, v), h)
    C = (u, v, w) -> _central_multilinear(G, x, (u, v, w), h)
    return A, B, C
end

function _continuous_fd_steps(h::Float64, x)
    max_step = max(4h, 0.03 * max(1.0, norm(x, Inf)))
    return [step for step in (h / 2) .* (2.0 .^ (0:9)) if step <= max_step]
end

function _stable_fd_window(results, coefficient_tol::Float64)
    length(results) >= 3 || return nothing
    for i in 1:(length(results) - 2)
        window = results[i:(i + 2)]
        all(result -> result.coefficient !== nothing, window) || continue
        criticality = window[1].criticality
        all(result -> result.status === window[1].status &&
                      result.criticality === criticality, window) || continue
        coefficients = getfield.(window, :coefficient)
        scale = maximum(abs, coefficients)
        tolerance = max(10coefficient_tol, 0.2scale)
        maximum(coefficients) - minimum(coefficients) <= tolerance || continue
        return window[2]
    end
    return nothing
end

function _complex_bilinear(B, u, v)
    ur, ui = real.(u), imag.(u)
    vr, vi = real.(v), imag.(v)
    return complex.(B(ur, vr) .- B(ui, vi), B(ur, vi) .+ B(ui, vr))
end

function _complex_trilinear(C, u, v, w)
    parts_u = (real.(u), imag.(u))
    parts_v = (real.(v), imag.(v))
    parts_w = (real.(w), imag.(w))
    result = zeros(ComplexF64, length(C(parts_u[1], parts_v[1], parts_w[1])))
    for iu in 0:1, iv in 0:1, iw in 0:1
        result .+= (im^(iu + iv + iw)) .* C(
            parts_u[iu + 1], parts_v[iv + 1], parts_w[iw + 1])
    end
    return result
end

function _oriented_eigenvectors(A::AbstractMatrix, kind::Symbol;
                                 eigenvector_tol::Float64)
    eig = eigen(complex.(A))
    if kind === :fold || kind === :pd
        target = kind === :fold ? 1.0 : -1.0
        idx = argmin(abs.(eig.values .- target))
        lambda = eig.values[idx]
        abs(imag(lambda)) <= eigenvector_tol || return nothing
    else
        candidates = findall(z -> imag(z) > eigenvector_tol, eig.values)
        isempty(candidates) && return nothing
        idx = candidates[argmin(abs.(abs.(eig.values[candidates]) .- 1.0))]
        lambda = eig.values[idx]
    end
    q = collect(ComplexF64, eig.vectors[:, idx])
    q ./= norm(q)
    pivot = argmax(abs.(q))
    q .*= exp(-im * angle(q[pivot]))
    real(q[pivot]) < 0 && (q .*= -1)

    left = eigen(adjoint(complex.(A)))
    left_idx = argmin(abs.(left.values .- conj(lambda)))
    p = collect(ComplexF64, left.vectors[:, left_idx])
    overlap = dot(p, q)
    abs(overlap) > eigenvector_tol || return nothing
    p ./= conj(overlap)
    return ComplexF64(lambda), q, p
end

function _guarded_solve(M, rhs; singular_tol::Float64)
    values = svdvals(M)
    isempty(values) && return nothing
    minimum(values) > singular_tol * max(maximum(values), 1.0) || return nothing
    return M \ rhs
end

function _critical_eigenvalue_ok(kind::Symbol, lambda;
                                 critical_tol::Float64, eigenvector_tol::Float64)
    kind === :fold && return abs(lambda - 1) <= critical_tol
    kind === :pd && return abs(lambda + 1) <= critical_tol
    return abs(abs(lambda) - 1) <= critical_tol && abs(imag(lambda)) > eigenvector_tol
end

function _critical_ns_pair_count(values::AbstractVector;
                                 critical_tol::Float64, eigenvector_tol::Float64)
    return count(value -> imag(value) > eigenvector_tol &&
                          abs(abs(value) - 1) <= critical_tol, values)
end

_critical_ns_pair_count(A::AbstractMatrix; kwargs...) =
    _critical_ns_pair_count(eigvals(complex.(A)); kwargs...)

function _map_normal_form_at_step(sys, kind, x, G, fd_step;
                                  critical_tol, coefficient_tol, singular_tol,
                                  resonance_tol, eigenvector_tol)
    A, B_real, C_real = sys isa DiscreteMap ?
        _discrete_normal_form_derivatives(G, x) :
        _continuous_normal_form_derivatives(G, x, fd_step)
    all(isfinite, A) || return _normal_form_result(
        kind, nothing, :unclassified, :derivative_failed)
    if kind === :ns && _critical_ns_pair_count(
            A; critical_tol=critical_tol, eigenvector_tol=eigenvector_tol) > 1
        return _normal_form_result(
            kind, nothing, :unclassified, :multiple_critical_pairs)
    end

    vectors = _oriented_eigenvectors(A, kind; eigenvector_tol=eigenvector_tol)
    vectors === nothing && return _normal_form_result(
        kind, nothing, :unclassified, :critical_eigenvector_unavailable)
    lambda, q, p = vectors
    _critical_eigenvalue_ok(kind, lambda; critical_tol=critical_tol,
                            eigenvector_tol=eigenvector_tol) ||
        return _normal_form_result(kind, nothing, :unclassified, :not_critical)

    B = (u, v) -> _complex_bilinear(B_real, u, v)
    C = (u, v, w) -> _complex_trilinear(C_real, u, v, w)
    I_n = Matrix{ComplexF64}(I, length(x), length(x))

    if kind === :fold
        b = real(dot(p, B(q, q))) / 2
        isfinite(b) || return _normal_form_result(kind, nothing, :unclassified, :derivative_failed)
        abs(b) <= coefficient_tol &&
            return _normal_form_result(kind, b, :degenerate, :degenerate)
        return _normal_form_result(kind, b, :nondegenerate, :ok)
    elseif kind === :pd
        h20 = _guarded_solve(I_n - A, B(q, q) / 2; singular_tol=singular_tol)
        h20 === nothing &&
            return _normal_form_result(kind, nothing, :unclassified, :near_singular)
        c = real(dot(p, C(q, q, q))) / 6 + real(dot(p, B(q, h20)))
        isfinite(c) || return _normal_form_result(kind, nothing, :unclassified, :derivative_failed)
        abs(c) <= coefficient_tol &&
            return _normal_form_result(kind, c, :degenerate, :degenerate)
        criticality = c > 0 ? :supercritical : :subcritical
        return _normal_form_result(kind, c, criticality, :ok)
    end

    any(k -> abs(lambda^k - 1) <= resonance_tol, 1:4) &&
        return _normal_form_result(kind, nothing, :unclassified, :strong_resonance)
    pair_gap = minimum(abs.(eigvals(complex.(A)) .- conj(lambda)))
    pair_gap <= critical_tol || return _normal_form_result(
        kind, nothing, :unclassified, :conjugate_pair_unavailable)
    h11 = _guarded_solve(A - I_n, -B(q, conj.(q)); singular_tol=singular_tol)
    h11 === nothing &&
        return _normal_form_result(kind, nothing, :unclassified, :near_singular)
    h20 = _guarded_solve(A - lambda^2 * I_n, -B(q, q); singular_tol=singular_tol)
    h20 === nothing &&
        return _normal_form_result(kind, nothing, :unclassified, :near_singular)
    term = dot(p, C(q, q, conj.(q))) + dot(p, B(h20, conj.(q))) +
           2dot(p, B(h11, q))
    d = real(conj(lambda) * term / 2)
    isfinite(d) || return _normal_form_result(kind, nothing, :unclassified, :derivative_failed)
    abs(d) <= coefficient_tol &&
        return _normal_form_result(kind, d, :degenerate, :degenerate)
    return _normal_form_result(kind, d, d < 0 ? :supercritical : :subcritical, :ok)
end

"""
    map_normal_form(sys, kind, state, params; period=1, kwargs...) -> MapNormalForm

Compute the Kuznetsov/MATCONT map normal-form coefficient of `G = F^period` at
`state`. Right eigenvectors have Euclidean norm one and left eigenvectors are scaled
so `dot(p, q) == 1` (the Hermitian inner product). Complex multilinear forms are
evaluated through their real/imaginary multilinear expansion because ForwardDiff
accepts real directions only.

For folds, `b = 1/2 <p,B(q,q)>`; its sign depends on the real eigenvector orientation,
so only nondegenerate/degenerate is reported. For flips,
`c = 1/6 <p,C(q,q,q)> + <p,B(q,h20)>`, with
`h20 = (I-A)^-1 B(q,q)/2`; `c > 0` is supercritical/soft and `c < 0` is
subcritical/hard. For Neimark-Sacker points the returned `d` uses
`h11 = -(A-I)^-1 B(q,qbar)`, `h20 = -(A-lambda^2 I)^-1 B(q,q)`, and
`d = Re(conj(lambda)/2 * (<p,C(q,q,qbar)> + <p,B(h20,qbar)> +
2<p,B(h11,q)>))`; `d < 0` is supercritical and `d > 0` subcritical.

Discrete maps use nested ForwardDiff directional derivatives. Continuous systems use
centered finite differences of the Poincare return map at an adaptive sequence of steps
starting around `normal_form_fd_step`. A coefficient is returned only when three
successive steps agree in sign, classification, and scale. Resonances, ill-conditioned
solves, noncritical inputs, unstable finite differences, and degenerate coefficients
return an explicit status and never a fabricated coefficient.
"""
function map_normal_form(sys::DynamicalSystem, kind::Symbol, state::AbstractVector,
                         params::AbstractVector; period::Int=1,
                         normal_form_fd_step::Float64=3e-3,
                         critical_tol::Float64=1e-4,
                         coefficient_tol::Float64=1e-8,
                         singular_tol::Float64=1e-9,
                         resonance_tol::Float64=1e-6,
                         eigenvector_tol::Float64=1e-8,
                         kwargs...)
    _normal_form_name(kind)
    period >= 1 || throw(ArgumentError("Map normal-form period must be >= 1; got $period."))
    normal_form_fd_step > 0 || throw(ArgumentError(
        "normal_form_fd_step must be positive; got $normal_form_fd_step."))
    critical_tol > 0 || throw(ArgumentError("critical_tol must be positive; got $critical_tol."))
    coefficient_tol >= 0 || throw(ArgumentError(
        "coefficient_tol must be non-negative; got $coefficient_tol."))
    singular_tol > 0 || throw(ArgumentError("singular_tol must be positive; got $singular_tol."))
    resonance_tol > 0 || throw(ArgumentError("resonance_tol must be positive; got $resonance_tol."))
    eigenvector_tol > 0 || throw(ArgumentError(
        "eigenvector_tol must be positive; got $eigenvector_tol."))
    length(state) == state_dim(sys) || throw(ArgumentError(
        "Normal-form state has length $(length(state)); expected $(state_dim(sys)) for $(sys.name)."))
    all(isfinite, state) || throw(ArgumentError("Normal-form state must contain only finite values."))
    all(isfinite, params) || throw(ArgumentError("Normal-form parameters must contain only finite values."))

    x = collect(Float64, state)
    G = _normal_form_map(sys, params, period; kwargs...)
    result_at_step = step -> _map_normal_form_at_step(
        sys, kind, x, G, step; critical_tol=critical_tol,
        coefficient_tol=coefficient_tol, singular_tol=singular_tol,
        resonance_tol=resonance_tol, eigenvector_tol=eigenvector_tol)
    sys isa DiscreteMap && return result_at_step(normal_form_fd_step)

    results = MapNormalForm[]
    for step in _continuous_fd_steps(normal_form_fd_step, x)
        result = result_at_step(step)
        if result.coefficient === nothing && result.status in (
                :strong_resonance, :multiple_critical_pairs, :near_singular)
            return result
        end
        push!(results, result)
        stable = _stable_fd_window(results, coefficient_tol)
        stable === nothing || return stable
    end
    return _normal_form_result(kind, nothing, :unclassified, :fd_step_unstable)
end

"""
    map_normal_form(sys, point::MapSpecialPoint, params; kwargs...) -> MapNormalForm

Compute the normal form for a located special point using the supplied full parameter
vector at that point.
"""
function map_normal_form(sys::DynamicalSystem, point::MapSpecialPoint,
                         params::AbstractVector; kwargs...)
    result = map_normal_form(
        sys, point.kind, point.state, params; period=point.period, kwargs...)
    point.kind === :ns || return result
    critical_tol = Float64(get(kwargs, :critical_tol, 1e-4))
    eigenvector_tol = Float64(get(kwargs, :eigenvector_tol, 1e-8))
    _critical_ns_pair_count(
        point.multipliers; critical_tol=critical_tol,
        eigenvector_tol=eigenvector_tol) > 1 || return result
    return _normal_form_result(
        :ns, nothing, :unclassified, :multiple_critical_pairs)
end
