const _MAP_FOLD_CONVENTION =
    "Kuznetsov/MATCONT map convention: b=1/2<p,B(q,q)>; sign depends on the chosen real eigenvector orientation."
const _MAP_FLIP_CONVENTION =
    "Kuznetsov/MATCONT map convention: c=1/6<p,C(q,q,q)>+<p,B(q,h20)>, h20=(I-A)^-1 B(q,q)/2; c>0 supercritical/soft, c<0 subcritical/hard."
const _MAP_NS_CONVENTION =
    "Kuznetsov/MATCONT map convention: d=Re(conj(lambda)/2*(<p,C(q,q,qbar)>+<p,B(h20,qbar)>+2<p,B(h11,q)>)), h11=-(A-I)^-1 B(q,qbar), h20=-(A-lambda^2 I)^-1 B(q,q); d<0 supercritical, d>0 subcritical."

const _CODIM2_CUSP_CONVENTION =
    "Map cusp reduction for H=F^N-I: a=1/6<p,C(q,q,q)>+1/2<p,B(q,h2)>, with (A-I)h2=<p,B(q,q)>q-B(q,q), <p,h2>=0; sign follows the deterministic q orientation."
const _CODIM2_GENERALIZED_FLIP_CONVENTION =
    "Generalized flip scalar odd normal form: e is the x^5 coefficient of the reduced critical map at c=0; e>0 and e<0 put the period-doubled fold on opposite sides of the flip locus under the selected primary-parameter orientation."
const _CODIM2_BAUTIN_CONVENTION =
    "Bautin/Chenciner radial reduction: e is the r^4 coefficient in the unit-circle amplitude ratio after the first Lyapunov coefficient vanishes; e<0 and e>0 are the supercritical and subcritical radial cases."
const _CODIM2_FOLD_FLIP_CONVENTION =
    "Fold-flip nondegeneracy record: one simple multiplier is near +1 and one simple multiplier is near -1; gaps are reported for the nearest representatives."
const _CODIM2_RESONANCE_CONVENTION =
    "Strong-resonance nondegeneracy record: the unit-circle pair angle is checked against the requested root of unity and all reported multiplier gaps are retained."

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

function _codim2_normal_form_result(kind::Symbol, names, coefficients,
                                    criticality::Symbol, status::Symbol,
                                    convention::String)
    return Codim2NormalForm(
        kind,
        Symbol[Symbol(name) for name in names],
        Float64[Float64(real(value)) for value in coefficients],
        criticality,
        status,
        convention,
    )
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
    y = G(x)
    return [ForwardDiff.derivative(t -> G(x .+ t .* v)[i], zero(eltype(x)))
            for i in eachindex(y)]
end

function _ad_second_directional(G, x, u, v)
    y = G(x)
    return [ForwardDiff.derivative(
                t -> _ad_first_directional(G, x .+ t .* u, v)[i],
                zero(eltype(x)))
            for i in eachindex(y)]
end

function _ad_third_directional(G, x, u, v, w)
    y = G(x)
    return [ForwardDiff.derivative(
                t -> _ad_second_directional(G, x .+ t .* w, u, v)[i],
                zero(eltype(x)))
            for i in eachindex(y)]
end

function _ad_fourth_directional(G, x, u, v, w, z)
    y = G(x)
    return [ForwardDiff.derivative(
                t -> _ad_third_directional(G, x .+ t .* z, u, v, w)[i],
                zero(eltype(x)))
            for i in eachindex(y)]
end

function _ad_fifth_directional(G, x, u, v, w, z, r)
    y = G(x)
    return [ForwardDiff.derivative(
                t -> _ad_fourth_directional(G, x .+ t .* r, u, v, w, z)[i],
                zero(eltype(x)))
            for i in eachindex(y)]
end

function _discrete_normal_form_derivatives(G, x)
    A = ForwardDiff.jacobian(G, x)
    B = (u, v) -> _ad_second_directional(G, x, u, v)
    C = (u, v, w) -> _ad_third_directional(G, x, u, v, w)
    return A, B, C
end

function _discrete_codim2_derivatives(G, x)
    A = ForwardDiff.jacobian(G, x)
    B = (u, v) -> _ad_second_directional(G, x, u, v)
    C = (u, v, w) -> _ad_third_directional(G, x, u, v, w)
    D = (u, v, w, z) -> _ad_fourth_directional(G, x, u, v, w, z)
    E = (u, v, w, z, r) -> _ad_fifth_directional(G, x, u, v, w, z, r)
    return A, B, C, D, E
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

function _continuous_codim2_derivatives(G, x, h::Float64)
    A, B, C = _continuous_normal_form_derivatives(G, x, h)
    D = (u, v, w, z) -> _central_multilinear(G, x, (u, v, w, z), h)
    E = (u, v, w, z, r) -> _central_multilinear(G, x, (u, v, w, z, r), h)
    return A, B, C, D, E
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

function _complex_multilinear(F, directions::Tuple)
    parts = map(direction -> (real.(direction), imag.(direction)), directions)
    result = zeros(ComplexF64, length(F((part[1] for part in parts)...)))
    for mask in 0:(2^length(directions) - 1)
        args = map(1:length(directions)) do j
            parts[j][((mask >> (j - 1)) & 1) + 1]
        end
        result .+= im^(count(j -> ((mask >> (j - 1)) & 1) == 1, 1:length(directions))) .*
                   F(args...)
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

function _bordered_kernel_solve(L, q, p, rhs; singular_tol::Float64)
    n = length(q)
    M = zeros(ComplexF64, n + 1, n + 1)
    M[1:n, 1:n] .= complex.(L)
    M[1:n, n + 1] .= q
    M[n + 1, 1:n] .= conj.(p)
    b = vcat(complex.(rhs), 0.0 + 0.0im)
    values = svdvals(M)
    minimum(values) > singular_tol * max(maximum(values), 1.0) || return nothing
    sol = M \ b
    return sol[1:n]
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

_codim2_locus_kind(kind::Symbol) =
    kind === :cusp ? :fold :
    kind === :generalized_flip ? :pd :
    kind === :bautin || kind === :resonance_1_1 || kind === :resonance_1_2 ? :ns :
    kind === :fold_flip ? :fold_flip :
    throw(ArgumentError("Unsupported codim-2 normal-form kind $(repr(kind))."))

function _codim2_unavailable(kind::Symbol, status::Symbol, convention::String)
    return _codim2_normal_form_result(kind, Symbol[], Float64[], :unclassified, status, convention)
end

_finite_norm(value) = all(isfinite, value) ? norm(value) : Inf

function _poly_mul(a::Vector{Float64}, b::Vector{Float64}, degree::Int)
    out = zeros(Float64, degree + 1)
    for i in 0:degree, j in 0:(degree - i)
        out[i + j + 1] += a[i + 1] * b[j + 1]
    end
    return out
end

function _poly_pow(a::Vector{Float64}, n::Int, degree::Int)
    out = zeros(Float64, degree + 1)
    out[1] = 1
    for _ in 1:n
        out = _poly_mul(out, a, degree)
    end
    return out
end

function _poly_compose(outer::Vector{Float64}, inner::Vector{Float64}, degree::Int)
    out = zeros(Float64, degree + 1)
    for k in 0:min(degree, length(outer) - 1)
        out .+= outer[k + 1] .* _poly_pow(inner, k, degree)
    end
    return out
end

function _scalar_flip_reduced_fifth(a2::Float64, a3::Float64, a4::Float64, a5::Float64)
    degree = 5
    fcoeff = [0.0, -1.0, a2, a3, a4, a5]
    reduced(A, C) = begin
        h = [0.0, 1.0, A, 0.0, C, 0.0]
        hinv = [
            0.0,
            1.0,
            -A,
            2A^2,
            -5A^3 - C,
            14A^4 + 6A * C,
        ]
        _poly_compose(hinv, _poly_compose(fcoeff, h, degree), degree)
    end
    c20 = reduced(0.0, 0.0)[3]
    c21 = reduced(1.0, 0.0)[3]
    slope2 = c21 - c20
    abs(slope2) > eps(Float64) || return nothing
    A = -c20 / slope2
    c40 = reduced(A, 0.0)[5]
    c41 = reduced(A, 1.0)[5]
    slope4 = c41 - c40
    abs(slope4) > eps(Float64) || return nothing
    C = -c40 / slope4
    coeffs = reduced(A, C)
    return (cubic=coeffs[4], fifth=coeffs[6])
end

function _codim2_derivatives(sys::DynamicalSystem, x, params, period::Int, fd_step::Float64; kwargs...)
    G = _normal_form_map(sys, params, period; kwargs...)
    if sys isa DiscreteMap
        return G, _discrete_codim2_derivatives(G, x)...
    end
    return G, _continuous_codim2_derivatives(G, x, fd_step)...
end

function _codim2_cusp_normal_form(sys::DynamicalSystem, x, params, period::Int;
                                  normal_form_fd_step::Float64,
                                  critical_tol::Float64,
                                  coefficient_tol::Float64,
                                  singular_tol::Float64,
                                  eigenvector_tol::Float64,
                                  kwargs...)
    convention = _CODIM2_CUSP_CONVENTION
    _G, A, B_real, C_real, _D_real, _E_real =
        _codim2_derivatives(sys, x, params, period, normal_form_fd_step; kwargs...)
    all(isfinite, A) || return _codim2_unavailable(:cusp, :derivative_failed, convention)
    vectors = _oriented_eigenvectors(A, :fold; eigenvector_tol=eigenvector_tol)
    vectors === nothing && return _codim2_unavailable(:cusp, :critical_eigenvector_unavailable, convention)
    lambda, q, p = vectors
    abs(lambda - 1) <= critical_tol ||
        return _codim2_unavailable(:cusp, :not_critical, convention)
    B = (u, v) -> _complex_bilinear(B_real, u, v)
    C = (u, v, w) -> _complex_trilinear(C_real, u, v, w)
    b = real(dot(p, B(q, q))) / 2
    isfinite(b) || return _codim2_unavailable(:cusp, :derivative_failed, convention)
    abs(b) <= coefficient_tol ||
        return _codim2_unavailable(:cusp, :not_codim2, convention)
    L = complex.(A) - Matrix{ComplexF64}(I, length(x), length(x))
    Bqq = B(q, q)
    h2 = _bordered_kernel_solve(L, q, p, dot(p, Bqq) .* q .- Bqq; singular_tol=singular_tol)
    h2 === nothing && return _codim2_unavailable(:cusp, :near_singular, convention)
    a = real(dot(p, C(q, q, q)) / 6 + dot(p, B(q, h2)) / 2)
    isfinite(a) || return _codim2_unavailable(:cusp, :derivative_failed, convention)
    status = abs(a) <= coefficient_tol ? :degenerate : :ok
    criticality = status === :degenerate ? :degenerate : (a > 0 ? :positive : :negative)
    return _codim2_normal_form_result(:cusp, (:cusp_cubic,), (a,), criticality, status, convention)
end

function _codim2_generalized_flip_normal_form(sys::DynamicalSystem, x, params, period::Int;
                                              normal_form_fd_step::Float64,
                                              critical_tol::Float64,
                                              coefficient_tol::Float64,
                                              eigenvector_tol::Float64,
                                              singular_tol::Float64,
                                              kwargs...)
    convention = _CODIM2_GENERALIZED_FLIP_CONVENTION
    length(x) == 1 || return _codim2_unavailable(
        :generalized_flip, :reduction_unavailable, convention)
    _G, A, B_real, C_real, D_real, E_real =
        _codim2_derivatives(sys, x, params, period, normal_form_fd_step; kwargs...)
    all(isfinite, A) || return _codim2_unavailable(:generalized_flip, :derivative_failed, convention)
    vectors = _oriented_eigenvectors(A, :pd; eigenvector_tol=eigenvector_tol)
    vectors === nothing && return _codim2_unavailable(:generalized_flip, :critical_eigenvector_unavailable, convention)
    lambda, q, p = vectors
    abs(lambda + 1) <= critical_tol ||
        return _codim2_unavailable(:generalized_flip, :not_critical, convention)
    codim1 = map_normal_form(
        sys, :pd, x, params; period=period, normal_form_fd_step=normal_form_fd_step,
        critical_tol=critical_tol, coefficient_tol=coefficient_tol,
        singular_tol=singular_tol, eigenvector_tol=eigenvector_tol, kwargs...)
    codim1.coefficient !== nothing && abs(codim1.coefficient) <= coefficient_tol ||
        return _codim2_unavailable(:generalized_flip, :not_codim2, convention)
    B = (u, v) -> _complex_bilinear(B_real, u, v)
    C = (u, v, w) -> _complex_trilinear(C_real, u, v, w)
    D = directions -> _complex_multilinear(D_real, directions)
    E = directions -> _complex_multilinear(E_real, directions)
    reduced = _scalar_flip_reduced_fifth(
        real(dot(p, B(q, q))) / 2,
        real(dot(p, C(q, q, q))) / 6,
        real(dot(p, D((q, q, q, q)))) / 24,
        real(dot(p, E((q, q, q, q, q)))) / 120,
    )
    reduced === nothing &&
        return _codim2_unavailable(:generalized_flip, :reduction_unavailable, convention)
    abs(reduced.cubic) <= max(10coefficient_tol, 1e-10) ||
        return _codim2_unavailable(:generalized_flip, :not_codim2, convention)
    e = reduced.fifth
    isfinite(e) || return _codim2_unavailable(:generalized_flip, :derivative_failed, convention)
    status = abs(e) <= coefficient_tol ? :degenerate : :ok
    criticality = status === :degenerate ? :degenerate :
                  (e > 0 ? :positive_second_flip : :negative_second_flip)
    return _codim2_normal_form_result(:generalized_flip, (:second_flip,), (e,),
                                      criticality, status, convention)
end

function _codim2_bautin_quantity(G, x, q, step::Float64)
    qr = real.(q)
    qi = imag.(q)
    norm(qr) > 0 && (qr = qr ./ norm(qr))
    norm(qi) > 0 && (qi = qi ./ norm(qi))
    norm(qr) > 0 && norm(qi) > 0 || return nothing
    directions = (qr, qi, (qr .+ qi) ./ norm(qr .+ qi), (qr .- qi) ./ norm(qr .- qi))
    radii = step .* (1.0, 1.5, 2.0, 2.5)
    gx = G(x)
    all(isfinite, gx) || return nothing
    per_direction = Float64[]
    for direction in directions
        estimates = Float64[]
        for r in radii
            y = G(x .+ r .* direction) .- gx
            all(isfinite, y) || continue
            amplitude_ratio = norm(y) / r
            # Estimate and subtract the residual cubic amplitude term before the r^4 term.
            push!(estimates, (amplitude_ratio - 1) / r^2)
        end
        length(estimates) == length(radii) || return nothing
        X = hcat(ones(length(radii)), collect(radii .^ 2))
        coeff = X \ estimates
        abs(coeff[1]) <= max(10step^2, 1e-8) || return nothing
        push!(per_direction, coeff[2])
    end
    spread = maximum(per_direction) - minimum(per_direction)
    scale = max(maximum(abs, per_direction), 1.0)
    spread <= 0.05scale || return nothing
    return sum(per_direction) / length(per_direction)
end

function _codim2_bautin_normal_form(sys::DynamicalSystem, x, params, period::Int;
                                    normal_form_fd_step::Float64,
                                    critical_tol::Float64,
                                    coefficient_tol::Float64,
                                    resonance_tol::Float64,
                                    eigenvector_tol::Float64,
                                    kwargs...)
    convention = _CODIM2_BAUTIN_CONVENTION
    G, A, _B_real, _C_real, _D_real, _E_real =
        _codim2_derivatives(sys, x, params, period, normal_form_fd_step; kwargs...)
    all(isfinite, A) || return _codim2_unavailable(:bautin, :derivative_failed, convention)
    vectors = _oriented_eigenvectors(A, :ns; eigenvector_tol=eigenvector_tol)
    vectors === nothing && return _codim2_unavailable(:bautin, :critical_eigenvector_unavailable, convention)
    lambda, q, _p = vectors
    abs(abs(lambda) - 1) <= critical_tol ||
        return _codim2_unavailable(:bautin, :not_critical, convention)
    any(k -> abs(lambda^k - 1) <= resonance_tol, 1:4) &&
        return _codim2_unavailable(:bautin, :strong_resonance, convention)
    codim1 = map_normal_form(
        sys, :ns, x, params; period=period, normal_form_fd_step=normal_form_fd_step,
        critical_tol=critical_tol, coefficient_tol=coefficient_tol,
        resonance_tol=resonance_tol, eigenvector_tol=eigenvector_tol, kwargs...)
    codim1.coefficient !== nothing && abs(codim1.coefficient) <= coefficient_tol ||
        return _codim2_unavailable(:bautin, :not_codim2, convention)
    e = _codim2_bautin_quantity(G, x, q, normal_form_fd_step)
    e !== nothing && isfinite(e) ||
        return _codim2_unavailable(:bautin, :reduction_unavailable, convention)
    status = abs(e) <= coefficient_tol ? :degenerate : :ok
    criticality = status === :degenerate ? :degenerate :
                  (e < 0 ? :supercritical : :subcritical)
    return _codim2_normal_form_result(:bautin, (:second_lyapunov,), (e,),
                                      criticality, status, convention)
end

function _codim2_fold_flip_normal_form(multipliers::AbstractVector{<:Number};
                                       critical_tol::Float64,
                                       coefficient_tol::Float64)
    convention = _CODIM2_FOLD_FLIP_CONVENTION
    isempty(multipliers) && return _codim2_unavailable(:fold_flip, :multipliers_unavailable, convention)
    fold_gaps = abs.(multipliers .- 1)
    flip_gaps = abs.(multipliers .+ 1)
    fold_gap = minimum(fold_gaps)
    flip_gap = minimum(flip_gaps)
    all(isfinite, (fold_gap, flip_gap)) ||
        return _codim2_unavailable(:fold_flip, :multipliers_unavailable, convention)
    fold_count = count(<=(critical_tol), fold_gaps)
    flip_count = count(<=(critical_tol), flip_gaps)
    status = (fold_count == 1 && flip_count == 1 && fold_gap <= critical_tol &&
              flip_gap <= critical_tol) ? :ok : :degenerate
    criticality = status === :ok ? :nondegenerate : :degenerate
    return _codim2_normal_form_result(:fold_flip, (:fold_gap, :flip_gap),
                                      (fold_gap, flip_gap), criticality, status, convention)
end

function _codim2_resonance_normal_form(kind::Symbol, multipliers::AbstractVector{<:Number};
                                       critical_tol::Float64)
    convention = _CODIM2_RESONANCE_CONVENTION
    isempty(multipliers) && return _codim2_unavailable(kind, :multipliers_unavailable, convention)
    target_angle = kind === :resonance_1_1 ? 0.0 : pi
    target_root = kind === :resonance_1_1 ? 1.0 + 0.0im : -1.0 + 0.0im
    idx = argmin(abs.(multipliers .- target_root))
    m = multipliers[idx]
    agap = abs(m - target_root)
    rgap = abs(abs(m) - 1)
    status = (agap <= critical_tol && rgap <= critical_tol) ? :ok : :degenerate
    criticality = status === :ok ? :nondegenerate : :degenerate
    return _codim2_normal_form_result(kind, (:angle_gap, :unit_circle_gap, :target_angle),
                                      (agap, rgap, target_angle), criticality, status, convention)
end

"""
    codim2_normal_form(sys, kind, state, params; kwargs...) -> Codim2NormalForm

Compute a plain-data codimension-two map normal-form classification at a located
organising point. Coefficients are returned only when the relevant reduced
quantity can be evaluated; otherwise `status` records the reason.
"""
function codim2_normal_form(sys::DynamicalSystem, kind::Symbol, state::AbstractVector,
                            params::AbstractVector; period::Int=1,
                            multipliers::AbstractVector{<:Number}=ComplexF64[],
                            normal_form_fd_step::Float64=3e-3,
                            critical_tol::Float64=1e-4,
                            coefficient_tol::Float64=1e-8,
                            singular_tol::Float64=1e-9,
                            resonance_tol::Float64=1e-6,
                            eigenvector_tol::Float64=1e-8,
                            kwargs...)
    period >= 1 || throw(ArgumentError("Codim-2 normal-form period must be >= 1; got $period."))
    normal_form_fd_step > 0 || throw(ArgumentError(
        "normal_form_fd_step must be positive; got $normal_form_fd_step."))
    critical_tol > 0 || throw(ArgumentError("critical_tol must be positive; got $critical_tol."))
    coefficient_tol >= 0 || throw(ArgumentError(
        "coefficient_tol must be non-negative; got $coefficient_tol."))
    singular_tol > 0 || throw(ArgumentError("singular_tol must be positive; got $singular_tol."))
    resonance_tol > 0 || throw(ArgumentError("resonance_tol must be positive; got $resonance_tol."))
    eigenvector_tol > 0 || throw(ArgumentError(
        "eigenvector_tol must be positive; got $eigenvector_tol."))
    kind in _CODIM2_SPECIAL_POINT_KINDS || throw(ArgumentError(
        "Codim-2 normal-form kind must be one of $(join(_CODIM2_SPECIAL_POINT_KINDS, ", ")); got $(repr(kind))."))

    if kind === :fold_flip
        return _codim2_fold_flip_normal_form(multipliers;
                                             critical_tol=critical_tol,
                                             coefficient_tol=coefficient_tol)
    elseif kind in (:resonance_1_1, :resonance_1_2)
        # Resonance records are multiplier-gap checks; eigenvector_tol applies to
        # the reduction-based codim-2 normal forms below.
        return _codim2_resonance_normal_form(kind, multipliers;
                                             critical_tol=critical_tol)
    end

    length(state) == state_dim(sys) || throw(ArgumentError(
        "Codim-2 normal-form state has length $(length(state)); expected $(state_dim(sys)) for $(sys.name)."))
    all(isfinite, state) || throw(ArgumentError("Codim-2 normal-form state must contain only finite values."))
    all(isfinite, params) || throw(ArgumentError("Codim-2 normal-form parameters must contain only finite values."))
    x = collect(Float64, state)
    if kind === :cusp
        return _codim2_cusp_normal_form(sys, x, params, period;
            normal_form_fd_step=normal_form_fd_step, critical_tol=critical_tol,
            coefficient_tol=coefficient_tol, singular_tol=singular_tol,
            eigenvector_tol=eigenvector_tol, kwargs...)
    elseif kind === :generalized_flip
        return _codim2_generalized_flip_normal_form(sys, x, params, period;
            normal_form_fd_step=normal_form_fd_step, critical_tol=critical_tol,
            coefficient_tol=coefficient_tol, eigenvector_tol=eigenvector_tol,
            singular_tol=singular_tol, kwargs...)
    end
    return _codim2_bautin_normal_form(sys, x, params, period;
        normal_form_fd_step=normal_form_fd_step, critical_tol=critical_tol,
        coefficient_tol=coefficient_tol, resonance_tol=resonance_tol,
        eigenvector_tol=eigenvector_tol, kwargs...)
end

"""
    codim2_normal_form(sys, point::Codim2SpecialPoint, params; kwargs...)

Compute the codimension-two normal form for a located codim-2 point using the
full parameter vector at that point.
"""
function codim2_normal_form(sys::DynamicalSystem, point::Codim2SpecialPoint,
                            params::AbstractVector; kwargs...)
    return codim2_normal_form(
        sys, point.kind, point.state, params; period=point.period,
        multipliers=point.multipliers, kwargs...)
end
