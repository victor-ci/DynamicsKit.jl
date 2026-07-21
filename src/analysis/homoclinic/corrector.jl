# Projection boundary-value corrector for connecting orbits.
#
# The truncated orbit is discretized on a uniform trapezoidal mesh in rescaled
# time. The unknown vector packs the mesh states, the source (and, for a
# heteroclinic connection, target) saddle, the truncation time, and the two
# continuation parameters:
#
#   z = [ vec(U) ; xs ; (xt) ; T ; α ; β ]
#
# with U ∈ ℝ^{n×(M+1)}. The stable/unstable projectors are frozen to Float64 and
# refreshed from the current iterate between Newton steps, because `eigen`/`schur`
# cannot differentiate through `ForwardDiff.Dual` matrices. The primary corrector
# is a line-searched Gauss-Newton; a Levenberg-Marquardt / pseudo-inverse path is
# the recorded fallback.

struct _ConnectingProblem
    field::Function
    base_params::Vector{Float64}
    primary_index::Int
    secondary_index::Int
    n::Int
    M::Int
    kind::Symbol
    eps0::Float64
    eps1::Float64
end

_has_target(prob::_ConnectingProblem) = prob.kind == :heteroclinic
_state_count(prob::_ConnectingProblem) = prob.n * (prob.M + 1)

function _nz(prob::_ConnectingProblem)
    return _state_count(prob) + prob.n * (_has_target(prob) ? 2 : 1) + 3
end

struct _FrozenBC
    Ws_src::Matrix{Float64}
    Wu_tgt::Matrix{Float64}
    ns_src::Int
    nu_tgt::Int
end

struct _ConnectingSeed
    U::Matrix{Float64}
    xs::Vector{Float64}
    xt::Vector{Float64}
    T::Float64
    primary::Float64
    secondary::Float64
end

# --- unknown-vector accessors -------------------------------------------------

function _unpack(z::AbstractVector, prob::_ConnectingProblem)
    n = prob.n
    M = prob.M
    U = reshape(view(z, 1:n * (M + 1)), n, M + 1)
    off = n * (M + 1)
    xs = view(z, off + 1:off + n)
    off += n
    if _has_target(prob)
        xt = view(z, off + 1:off + n)
        off += n
    else
        xt = xs
    end
    T = z[off + 1]
    α = z[off + 2]
    β = z[off + 3]
    return U, xs, xt, T, α, β
end

function _param_vector(prob::_ConnectingProblem, α, β, ::Type{T}) where {T}
    p = Vector{T}(undef, length(prob.base_params))
    @inbounds for k in eachindex(p)
        p[k] = prob.base_params[k]
    end
    p[prob.primary_index] = α
    p[prob.secondary_index] = β
    return p
end

function _seed_vector(prob::_ConnectingProblem, seed::_ConnectingSeed)
    z = Vector{Float64}(undef, _nz(prob))
    n = prob.n
    M = prob.M
    @inbounds z[1:n * (M + 1)] = vec(seed.U)
    off = n * (M + 1)
    @inbounds z[off + 1:off + n] = seed.xs
    off += n
    if _has_target(prob)
        @inbounds z[off + 1:off + n] = seed.xt
        off += n
    end
    z[off + 1] = seed.T
    z[off + 2] = seed.primary
    z[off + 3] = seed.secondary
    return z
end

# --- frozen projectors --------------------------------------------------------

"""
    _refresh_bc(prob, z) -> _FrozenBC

Recompute the Float64 stable/unstable left-projector bases from the current
Float64 iterate `z`. Refreshing every Newton step keeps the frozen projectors
consistent with the converging saddle.
"""
function _refresh_bc(prob::_ConnectingProblem, z::AbstractVector{<:Real})
    U, xs, xt, _, α, β = _unpack(z, prob)
    p = _param_vector(prob, Float64(α), Float64(β), Float64)
    A_src = _field_jacobian(prob.field, Vector{Float64}(xs), p)
    Ws_src, _, ns_src, _ = _left_invariant_bases(A_src)
    if _has_target(prob)
        A_tgt = _field_jacobian(prob.field, Vector{Float64}(xt), p)
        _, Wu_tgt, _, nu_tgt = _left_invariant_bases(A_tgt)
    else
        _, Wu_tgt, _, nu_tgt = _left_invariant_bases(A_src)
    end
    return _FrozenBC(Ws_src, Wu_tgt, ns_src, nu_tgt)
end

# --- residual -----------------------------------------------------------------

function _model_residual(z::AbstractVector, prob::_ConnectingProblem, bc::_FrozenBC)
    T = eltype(z)
    n = prob.n
    M = prob.M
    U, xs, xt, Tt, α, β = _unpack(z, prob)
    p = _param_vector(prob, α, β, T)
    fn = [prob.field(view(U, :, i), p) for i in 1:M + 1]

    nsaddle = n * (_has_target(prob) ? 2 : 1)
    Nm = n * M + nsaddle + bc.ns_src + bc.nu_tgt + 2
    res = Vector{T}(undef, Nm)
    c = 0
    h = Tt / (2 * M)
    @inbounds for i in 1:M
        for r in 1:n
            res[c + r] = U[r, i + 1] - U[r, i] - h * (fn[i][r] + fn[i + 1][r])
        end
        c += n
    end
    fs = prob.field(xs, p)
    @inbounds for r in 1:n
        res[c + r] = fs[r]
    end
    c += n
    if _has_target(prob)
        ft = prob.field(xt, p)
        @inbounds for r in 1:n
            res[c + r] = ft[r]
        end
        c += n
    end
    d0 = collect(view(U, :, 1)) .- xs
    d1 = collect(view(U, :, M + 1)) .- xt
    bcl = bc.Ws_src' * d0
    @inbounds for r in 1:bc.ns_src
        res[c + r] = bcl[r]
    end
    c += bc.ns_src
    bcr = bc.Wu_tgt' * d1
    @inbounds for r in 1:bc.nu_tgt
        res[c + r] = bcr[r]
    end
    c += bc.nu_tgt
    res[c + 1] = dot(d0, d0) - prob.eps0^2
    res[c + 2] = dot(d1, d1) - prob.eps1^2
    return res
end

function _augmented_residual(z::AbstractVector, prob::_ConnectingProblem, bc::_FrozenBC,
                             extra)
    r = _model_residual(z, prob, bc)
    extra === nothing && return r
    return vcat(r, extra(z))
end

# --- geometry validation ------------------------------------------------------

"""
    _connection_deficiency(prob, bc) -> Int

Free-direction count `k = Nz − Nm` of the model with both parameters free. A
traceable codimension-one connecting curve needs `k == 1`.
"""
function _connection_deficiency(prob::_ConnectingProblem, bc::_FrozenBC)
    nsaddle = prob.n * (_has_target(prob) ? 2 : 1)
    Nm = prob.n * prob.M + nsaddle + bc.ns_src + bc.nu_tgt + 2
    return _nz(prob) - Nm
end

function _validate_geometry(prob::_ConnectingProblem, bc::_FrozenBC)
    k = _connection_deficiency(prob, bc)
    if k < 1
        throw(ArgumentError(
            "Saddle manifold dimensions cannot support a $(prob.kind) curve in two " *
            "parameters (free-direction count k = $(k) < 1; source stable dim = " *
            "$(bc.ns_src), target unstable dim = $(bc.nu_tgt), state dim = $(prob.n)). " *
            "The requested connection is of higher codimension than the two-parameter " *
            "continuation supports."))
    elseif k > 1
        throw(ArgumentError(
            "Saddle manifold dimensions leave $(k) free directions for a $(prob.kind) " *
            "connection (source stable dim = $(bc.ns_src), target unstable dim = " *
            "$(bc.nu_tgt), state dim = $(prob.n)); the connection is non-isolated in two " *
            "parameters and cannot be traced as a single curve."))
    end
    return k
end

# --- linear solve helpers -----------------------------------------------------

function _safe_solve(A::AbstractMatrix, b::AbstractVector)
    try
        return A \ b
    catch
        return pinv(Matrix(A)) * b
    end
end

function _safe_solve(A::AbstractMatrix, B::AbstractMatrix)
    try
        return A \ B
    catch
        return pinv(Matrix(A)) * B
    end
end

# --- Gauss-Newton with Levenberg-Marquardt fallback ---------------------------

struct _CorrectorResult
    z::Vector{Float64}
    converged::Bool
    residual::Float64
    path::Symbol
    iterations::Int
end

"""
    _gauss_newton(prob, z0; extra, tol, maxiter, use_fallback, fallback_max_iter,
                  projector_refresh)

Correct a predictor `z0`. `extra(z)` supplies the closing equation(s) that make
the augmented system square (or `nothing` for a pure least-squares point solve).
The primary path is a line-searched Gauss-Newton; on stagnation it switches to a
damped Levenberg-Marquardt / pseudo-inverse fallback.

`projector_refresh` is the Newton-iteration cadence for recomputing the frozen
stable/unstable projectors: `1` (default) refreshes every iteration, `k` refreshes
on iterations 1, k+1, 2k+1, …. Projectors are frozen to `Float64` because
`eigen`/`schur` cannot differentiate through `ForwardDiff.Dual` matrices; the
cadence controls the trade-off between projector accuracy and compute cost.
Frozen projectors are reused between scheduled refreshes.
"""
function _gauss_newton(prob::_ConnectingProblem, z0::AbstractVector;
                       extra=nothing, tol::Float64=1e-10, maxiter::Int=60,
                       use_fallback::Bool=true, fallback_max_iter::Int=150,
                       projector_refresh::Int=1)
    z = collect(float.(z0))
    path = :newton
    iterations = 0
    bc = _refresh_bc(prob, z)
    rn = norm(_augmented_residual(z, prob, bc, extra))

    for iter_num in 1:maxiter
        # Refresh the frozen projectors on the scheduled cadence.
        # iter_num == 1 always refreshes (initial point).
        if (iter_num - 1) % projector_refresh == 0
            bc = _refresh_bc(prob, z)
        end
        G = zz -> _augmented_residual(zz, prob, bc, extra)
        r = G(z)
        rn = norm(r)
        if rn <= tol
            bc = _refresh_bc(prob, z)
            G = zz -> _augmented_residual(zz, prob, bc, extra)
            r = G(z)
            rn = norm(r)
            rn <= tol && return _CorrectorResult(z, true, rn, path, iterations)
        end
        J = ForwardDiff.jacobian(G, z)
        δ = _safe_solve(J, -r)
        step = 1.0
        znew = z .+ step .* δ
        rnew = norm(G(znew))
        while rnew > rn && step > 1e-10
            step /= 2
            znew = z .+ step .* δ
            rnew = norm(G(znew))
        end
        iterations += 1
        if rnew < rn || rnew <= tol
            z = znew
            rn = rnew
        else
            break
        end
    end
    bc = _refresh_bc(prob, z)
    rn = norm(_augmented_residual(z, prob, bc, extra))
    rn <= tol && return _CorrectorResult(z, true, rn, path, iterations)
    use_fallback || return _CorrectorResult(z, false, rn, path, iterations)

    # Levenberg-Marquardt fallback from the best primary iterate.
    path = :fallback
    λ = 1e-3
    for iter_num in 1:fallback_max_iter
        if (iter_num - 1) % projector_refresh == 0
            bc = _refresh_bc(prob, z)
        end
        G = zz -> _augmented_residual(zz, prob, bc, extra)
        r = G(z)
        rn = norm(r)
        if rn <= tol
            bc = _refresh_bc(prob, z)
            G = zz -> _augmented_residual(zz, prob, bc, extra)
            r = G(z)
            rn = norm(r)
            rn <= tol && return _CorrectorResult(z, true, rn, path, iterations)
        end
        J = ForwardDiff.jacobian(G, z)
        H = J' * J
        g = J' * r
        δ = _safe_solve(H + λ * I, -g)
        znew = z .+ δ
        rnew = norm(G(znew))
        iterations += 1
        if rnew < rn
            z = znew
            rn = rnew
            λ = max(λ / 3, 1e-12)
        else
            λ = min(λ * 5, 1e8)
        end
    end
    bc = _refresh_bc(prob, z)
    rn = norm(_augmented_residual(z, prob, bc, extra))
    return _CorrectorResult(z, rn <= tol, rn, path, iterations)
end
