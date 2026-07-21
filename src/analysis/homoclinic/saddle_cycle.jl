# Homoclinic connections to a saddle periodic orbit.
#
# The target is a hyperbolic ("saddle") cycle rather than an equilibrium. The
# linear stable/unstable manifolds of the cycle are spanned by the Floquet
# eigenvectors of the monodromy matrix, with one trivial multiplier `+1` along
# the flow. The truncated connecting orbit leaves the cycle along the unstable
# Floquet subspace and returns along the stable one; its endpoints are pinned to
# those subspaces at a reference cross-section (phase-aware endpoint projection).
#
# Full two-parameter continuation of cycle-to-cycle homoclinics is not claimed
# here. This module owns the monodromy/Floquet numerics, strict geometry
# validation, the phase-aware projection corrector, and an honest single-locus
# result with the Floquet data recorded in diagnostics.

"""
    _cycle_monodromy(field, cycle_states, Tc, p) -> Matrix

Monodromy matrix of a periodic orbit sampled uniformly on `[0, Tc]`. The
fundamental solution of the variational equation `Φ̇ = A(t) Φ`, `Φ(0) = I`, is
transported with the same implicit-trapezoidal rule used by the orbit corrector,
so the returned multipliers are consistent with the collocation discretization.
"""
function _cycle_monodromy(field, cycle_states::AbstractMatrix, Tc::Real, p::AbstractVector)
    n = size(cycle_states, 1)
    L = size(cycle_states, 2)
    h = Tc / (2 * (L - 1))
    Φ = Matrix{Float64}(I, n, n)
    A_i = _field_jacobian(field, Vector{Float64}(cycle_states[:, 1]), p)
    for i in 1:L - 1
        A_ip1 = _field_jacobian(field, Vector{Float64}(cycle_states[:, i + 1]), p)
        Φ = _safe_solve(I - h .* A_ip1, (I + h .* A_i) * Φ)
        A_i = A_ip1
    end
    return Φ
end

struct _FloquetSplit
    multipliers::Vector{ComplexF64}
    ns::Int          # stable multipliers (|μ| < 1)
    nu::Int          # unstable multipliers (|μ| > 1)
    nc::Int          # unit-circle multipliers (|μ| ≈ 1)
    Ws_left::Matrix{Float64}   # forces start endpoint into the unstable subspace
    Wu_left::Matrix{Float64}   # forces end endpoint into the stable subspace
end

"""
    _floquet_split(M; tol) -> _FloquetSplit

Classify Floquet multipliers and build the real left-projector bases that pin
the connecting-orbit endpoints to the unstable (start) and stable (end)
subspaces of the cycle. Every unit-circle multiplier is excluded from both
manifolds and annihilated by both projectors. Geometry validation separately
requires exactly one such multiplier and verifies that it is the trivial
flow multiplier near `+1`.
"""
function _floquet_split(M::AbstractMatrix; tol::Float64=1e-6)
    Mf = Matrix{Float64}(M)
    E = eigen(Mf)
    vals = E.values
    modulus = abs.(vals)
    stable = findall(m -> m < 1 - tol, modulus)
    unstable = findall(m -> m > 1 + tol, modulus)
    center = findall(m -> abs(m - 1) <= tol, modulus)
    ns = length(stable)
    nu = length(unstable)
    nc = length(center)

    Winv = inv(E.vectors)   # rows are left eigenvectors
    # Start BC annihilates stable ∪ center (→ endpoint lies in unstable subspace).
    ws_idx = vcat(stable, center)
    # End BC annihilates unstable ∪ center (→ endpoint lies in stable subspace).
    wu_idx = vcat(unstable, center)
    Ws_left = _real_left_basis(Winv, ws_idx)
    Wu_left = _real_left_basis(Winv, wu_idx)
    return _FloquetSplit(vals, ns, nu, nc, Ws_left, Wu_left)
end

# Real orthonormal basis spanning the (possibly complex-conjugate) left
# eigenvectors selected by `idx`.
function _real_left_basis(Winv::AbstractMatrix, idx::Vector{Int})
    isempty(idx) && return Matrix{Float64}(undef, size(Winv, 2), 0)
    cols = Vector{Vector{Float64}}()
    for i in idx
        w = Winv[i, :]
        push!(cols, real.(w))
        if any(abs.(imag.(w)) .> 1e-12)
            push!(cols, imag.(w))
        end
    end
    A = reduce(hcat, cols)
    F = qr(A)
    r = rank(A; rtol=1e-10)
    Q = Matrix(F.Q)
    return Q[:, 1:min(r, size(Q, 2))]
end

"""
    _validate_saddle_cycle_geometry(split, n)

Reject impossible cycle-homoclinic geometries: the cycle must be a genuine
saddle (at least one unstable and one stable non-trivial Floquet multiplier) and
must carry exactly one trivial multiplier along the flow.

A trivial multiplier is one that is real and close to +1 (not merely close to
the unit circle). Additional unit-circle multipliers — including −1 (period
doubling) and complex unit-circle roots (Neimark-Sacker) — indicate a
non-generic or non-hyperbolic cycle and are rejected.
"""
function _validate_saddle_cycle_geometry(split::_FloquetSplit, n::Int)
    tol_trivial = 1e-6
    # Count multipliers that are genuinely trivial (real ≈ +1).
    trivial_count = count(
        μ -> abs(real(μ) - 1) <= tol_trivial && abs(imag(μ)) <= tol_trivial,
        split.multipliers)
    # Count unit-circle multipliers that are NOT trivial (+1): e.g. −1 or e^{iθ}.
    nontrivial_unit_count = count(
        μ -> abs(abs(μ) - 1) <= tol_trivial && !(abs(real(μ) - 1) <= tol_trivial && abs(imag(μ)) <= tol_trivial),
        split.multipliers)

    if trivial_count < 1
        throw(ArgumentError(
            "Saddle-cycle monodromy has no trivial (+1) Floquet multiplier along the flow; " *
            "the sampled trajectory is not a clean periodic orbit."))
    end
    if trivial_count > 1 || nontrivial_unit_count > 0
        throw(ArgumentError(
            "Saddle-cycle monodromy must have exactly one trivial (+1) Floquet multiplier " *
            "along the flow (found $(trivial_count) near +1 and $(nontrivial_unit_count) other " *
            "unit-circle multipliers). Additional unit-circle multipliers (e.g. −1 for period " *
            "doubling, complex roots for Neimark-Sacker) indicate a non-hyperbolic or " *
            "non-generic cycle; the projection boundary condition is undefined here."))
    end
    if split.nu < 1
        throw(ArgumentError(
            "Target cycle has no unstable Floquet direction (|μ| > 1); a homoclinic orbit " *
            "cannot leave the cycle. Manifold dimensions cannot support the connection."))
    end
    if split.ns < 1
        throw(ArgumentError(
            "Target cycle has no stable Floquet direction (|μ| < 1); a homoclinic orbit " *
            "cannot return to the cycle. Manifold dimensions cannot support the connection."))
    end
    if split.ns + split.nu + split.nc != n
        throw(ArgumentError(
            "Floquet multipliers do not partition the $(n)-dimensional phase space " *
            "(stable $(split.ns) + unstable $(split.nu) + center $(split.nc) ≠ $(n)); the " *
            "cycle is non-hyperbolic and the projection is undefined."))
    end
    return nothing
end

struct _CycleProblem
    field::Function
    base_params::Vector{Float64}
    p::Vector{Float64}
    n::Int
    M::Int
    x0::Vector{Float64}
    eps0::Float64
    eps1::Float64
    split::_FloquetSplit
end

function _cycle_residual(z::AbstractVector, prob::_CycleProblem)
    T = eltype(z)
    n = prob.n
    M = prob.M
    U = reshape(view(z, 1:n * (M + 1)), n, M + 1)
    Tt = z[n * (M + 1) + 1]
    p = T.(prob.p)
    fn = [prob.field(view(U, :, i), p) for i in 1:M + 1]
    ns_bc = size(prob.split.Ws_left, 2)
    nu_bc = size(prob.split.Wu_left, 2)
    Nm = n * M + ns_bc + nu_bc + 2
    res = Vector{T}(undef, Nm)
    c = 0
    h = Tt / (2 * M)
    @inbounds for i in 1:M
        for r in 1:n
            res[c + r] = U[r, i + 1] - U[r, i] - h * (fn[i][r] + fn[i + 1][r])
        end
        c += n
    end
    d0 = collect(view(U, :, 1)) .- prob.x0
    d1 = collect(view(U, :, M + 1)) .- prob.x0
    bcl = prob.split.Ws_left' * d0
    @inbounds for r in 1:ns_bc
        res[c + r] = bcl[r]
    end
    c += ns_bc
    bcr = prob.split.Wu_left' * d1
    @inbounds for r in 1:nu_bc
        res[c + r] = bcr[r]
    end
    c += nu_bc
    res[c + 1] = dot(d0, d0) - prob.eps0^2
    res[c + 2] = dot(d1, d1) - prob.eps1^2
    return res
end

"""
    _correct_cycle_homoclinic(prob, z0; tol, maxiter, use_fallback, fallback_max_iter)

Least-squares Gauss-Newton (with Levenberg-Marquardt fallback) correction of the
truncated cycle-homoclinic projection problem with the cycle phase pinned to the
reference cross-section `x0`.
"""
function _correct_cycle_homoclinic(prob::_CycleProblem, z0::AbstractVector;
                                   tol::Float64=1e-9, maxiter::Int=80,
                                   use_fallback::Bool=true, fallback_max_iter::Int=150)
    z = collect(float.(z0))
    G = zz -> _cycle_residual(zz, prob)
    rn = norm(G(z))
    path = :newton
    for _ in 1:maxiter
        r = G(z)
        rn = norm(r)
        rn <= tol && return _CorrectorResult(z, true, rn, path, 0)
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
        if rnew < rn || rnew <= tol
            z = znew
            rn = rnew
        else
            break
        end
    end
    rn <= tol && return _CorrectorResult(z, true, rn, path, 0)
    use_fallback || return _CorrectorResult(z, false, rn, path, 0)
    path = :fallback
    λ = 1e-3
    for _ in 1:fallback_max_iter
        r = G(z)
        rn = norm(r)
        rn <= tol && return _CorrectorResult(z, true, rn, path, 0)
        J = ForwardDiff.jacobian(G, z)
        δ = _safe_solve(J' * J + λ * I, -(J' * r))
        znew = z .+ δ
        rnew = norm(G(znew))
        if rnew < rn
            z = znew
            rn = rnew
            λ = max(λ / 3, 1e-12)
        else
            λ = min(λ * 5, 1e8)
        end
    end
    return _CorrectorResult(z, rn <= tol, rn, path, 0)
end
