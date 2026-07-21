# Projection primitives for connecting-orbit continuation.
#
# The endpoints of a truncated connecting orbit are pinned to the linear
# stable/unstable subspaces of the source and target saddles. These helpers
# solve the saddle equilibria, build real orthonormal bases of the invariant
# subspaces (frozen to `Float64` because `eigen`/`schur` do not accept
# `ForwardDiff.Dual` matrices), and evaluate the standard HomCont eigenvalue
# test functions from the saddle spectrum.

"""
    _ode_field(sys) -> (u, p) -> du

Wrap an in-place `ContinuousODE` right-hand side as an out-of-place, autonomous
field usable under ForwardDiff. The wrapped closure promotes the element type so
it can be differentiated with respect to either the state or the parameters.
"""
function _ode_field(sys::ContinuousODE)
    f! = sys.f
    return function (u::AbstractVector, p::AbstractVector)
        T = promote_type(eltype(u), eltype(p))
        du = Vector{T}(undef, length(u))
        f!(du, u, p, zero(T))
        return du
    end
end

"""
    _field_jacobian(field, x, p) -> Matrix

State Jacobian `∂field/∂x` at `(x, p)` via ForwardDiff.
"""
_field_jacobian(field, x::AbstractVector, p::AbstractVector) =
    ForwardDiff.jacobian(u -> field(u, p), x)

"""
    _solve_saddle(field, x0, p; tol, maxiter) -> (x, converged, residual)

Damped Newton solve of `field(x, p) = 0` seeded at `x0`. Returns the refined
equilibrium, a convergence flag, and the final residual norm.
"""
function _solve_saddle(field, x0::AbstractVector, p::AbstractVector;
                       tol::Float64=1e-11, maxiter::Int=100)
    x = collect(float.(x0))
    r = field(x, p)
    rnorm = norm(r)
    converged = rnorm <= tol
    iter = 0
    while !converged && iter < maxiter
        J = _field_jacobian(field, x, p)
        local δ
        try
            δ = J \ (-r)
        catch
            δ = pinv(J) * (-r)
        end
        step = 1.0
        newx = x .+ step .* δ
        newr = field(newx, p)
        # Backtracking keeps the Newton step from overshooting shallow saddles.
        while norm(newr) > rnorm && step > 1e-6
            step /= 2
            newx = x .+ step .* δ
            newr = field(newx, p)
        end
        x, r = newx, newr
        rnorm = norm(r)
        converged = rnorm <= tol
        iter += 1
    end
    return x, converged, rnorm
end

"""
    _hyperbolic_split(A; tol) -> (n_stable, n_unstable, spectral_gap)

Count stable/unstable eigenvalues of `A` and report the smallest `|Re λ|`
(the distance to non-hyperbolicity). Throws when an eigenvalue sits on the
imaginary axis within `tol`.
"""
function _hyperbolic_split(A::AbstractMatrix; tol::Float64=1e-8)
    vals = eigvals(Matrix{Float64}(A))
    reparts = real.(vals)
    gap = minimum(abs.(reparts))
    if gap <= tol
        throw(ArgumentError(
            "Saddle is non-hyperbolic (an eigenvalue lies on the imaginary axis, " *
            "min|Re λ| = $(gap)); a connecting-orbit projection is not defined here."))
    end
    ns = count(<(0), reparts)
    nu = count(>(0), reparts)
    return ns, nu, gap
end

"""
    _left_invariant_bases(A; tol) -> (Ws, Wu, ns, nu)

Real orthonormal bases of the stable (`Ws`) and unstable (`Wu`) *left* invariant
subspaces of `A`, i.e. the invariant subspaces of `Aᵀ`. Because the stable-left
subspace is orthogonal to the unstable-right subspace, the boundary condition
`Wsᵀ (u − x) = 0` forces `u − x` into the unstable subspace of `A`, and
`Wuᵀ (u − x) = 0` forces it into the stable subspace. Real Schur ordering keeps
the bases real even for saddle-focus spectra.
"""
function _left_invariant_bases(A::AbstractMatrix; tol::Float64=1e-8)
    Af = Matrix{Float64}(A)
    ns, nu, _ = _hyperbolic_split(Af; tol=tol)
    F = schur(collect(Af'))
    stable_sel = real.(F.values) .< 0
    Fs = ordschur(F, stable_sel)
    Ws = Fs.Z[:, 1:ns]
    unstable_sel = real.(F.values) .> 0
    Fu = ordschur(F, unstable_sel)
    Wu = Fu.Z[:, 1:nu]
    return Ws, Wu, ns, nu
end

"""
    _leading_eigen(A) -> NamedTuple

Sorted eigen-decomposition of `A` with the leading (closest to the imaginary
axis) stable and unstable eigenvalues/eigenvectors extracted, plus strong
(farthest) eigenvectors used by the flip test functions.
"""
function _leading_eigen(A::AbstractMatrix)
    E = eigen(Matrix{Float64}(A))
    vals = E.values
    vecs = E.vectors
    stable_idx = findall(<(0), real.(vals))
    unstable_idx = findall(>(0), real.(vals))
    # Leading = closest to axis; strong = farthest.
    sort!(stable_idx; by=i -> real(vals[i]), rev=true)
    sort!(unstable_idx; by=i -> real(vals[i]))
    leading_stable = isempty(stable_idx) ? nothing : vals[stable_idx[1]]
    leading_unstable = isempty(unstable_idx) ? nothing : vals[unstable_idx[1]]
    return (
        values=vals,
        vectors=vecs,
        stable_idx=stable_idx,
        unstable_idx=unstable_idx,
        leading_stable=leading_stable,
        leading_unstable=leading_unstable,
    )
end

# Real part helper tolerant of real or complex eigenvalues.
_re(x) = real(x)

"""
    _eigen_test_functions(A) -> Dict{Symbol, Tuple{Float64, Symbol}}

Evaluate the HomCont saddle-quantity test functions from the source Jacobian
spectrum. `μ₁` is the leading stable eigenvalue and `λ₁` the leading unstable
eigenvalue. Sign changes of these smooth functions along the locus mark the
corresponding codimension-two homoclinic events.

Each entry maps a test-function code to `(value, status)` where `status` is
`:available` for well-defined tests, `:degenerate` when the computation is
meaningful but the geometry is complex (focus spectrum for an eigenvalue test
that assumes real parts only), or `:unavailable` when the spectrum cannot
support the test. Double-real tests (`:drs`, `:dru`, `:tls`, `:tlu`) are only
computed for real eigenvalues; complex conjugate pairs in the
relevant half-plane make the test `:unavailable` because equal real parts of a
conjugate pair do not constitute a double-real degeneracy.
"""
function _eigen_test_functions(A::AbstractMatrix)
    info = _leading_eigen(A)
    tests = Dict{Symbol, Tuple{Float64, Symbol}}()
    μ1 = info.leading_stable
    λ1 = info.leading_unstable
    (μ1 === nothing || λ1 === nothing) && return tests

    reμ1 = _re(μ1)
    reλ1 = _re(λ1)
    μ1_focus = abs(imag(μ1)) > 1e-9
    λ1_focus = abs(imag(λ1)) > 1e-9
    if μ1_focus || λ1_focus
        tests[(μ1_focus && λ1_focus) ? :nff : :nsf] = (reμ1 + reλ1, :available)
    else
        tests[:nns] = (reμ1 + reλ1, :available)
    end
    # Shilnikov saddle quantity and Bogdanov-Takens degeneration.
    μ1_focus && (tests[:sh] = (reμ1 + reλ1, :available))
    tests[:bt] = (reμ1 * reλ1, :available)
    tests[:nch] = (reμ1, :available)

    # Double-real tests are only valid for genuinely real eigenvalues.
    # A complex conjugate pair has the same real part but is not a double-real
    # degeneracy — computing the difference would give 0 and trigger a false event.
    _is_real_eig(v) = abs(imag(v)) <= 1e-9
    stable_vals = info.values[info.stable_idx]
    unstable_vals = info.values[info.unstable_idx]
    stable_real_only = sort(real.(filter(_is_real_eig, stable_vals)); rev=true)
    unstable_real_only = sort(real.(filter(_is_real_eig, unstable_vals)))

    if length(stable_vals) >= 2
        if length(stable_real_only) >= 2
            tests[:drs] = (stable_real_only[1] - stable_real_only[2], :available)
        else
            tests[:drs] = (NaN, :unavailable)
        end
    end
    if length(unstable_vals) >= 2
        if length(unstable_real_only) >= 2
            tests[:dru] = (unstable_real_only[1] - unstable_real_only[2], :available)
        else
            tests[:dru] = (NaN, :unavailable)
        end
    end
    if length(stable_real_only) >= 3
        tests[:tls] = (stable_real_only[2] - stable_real_only[3], :available)
    end
    if length(unstable_real_only) >= 3
        tests[:tlu] = (unstable_real_only[2] - unstable_real_only[3], :available)
    end
    return tests
end
