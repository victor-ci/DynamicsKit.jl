# Homoclinic connections to a saddle periodic orbit.
#
# The target is a hyperbolic ("saddle") cycle rather than an equilibrium. The
# linear stable/unstable manifolds of the cycle are spanned by the Floquet
# eigenvectors of the monodromy matrix, with one trivial multiplier `+1` along
# the flow. The truncated connecting orbit leaves the cycle along the unstable
# Floquet subspace and returns along the stable one; its endpoints are pinned to
# those subspaces at a reference cross-section (phase-aware endpoint projection).
#
# This module owns the monodromy/Floquet numerics, strict geometry validation,
# the phase-aware projection correctors, and the cycle-connection BVP where
# source and target cycle phases are solved automatically.

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

function _real_right_basis(V::AbstractMatrix, idx::Vector{Int})
    isempty(idx) && return Matrix{Float64}(undef, size(V, 1), 0)
    cols = Vector{Vector{Float64}}()
    for i in idx
        v = V[:, i]
        push!(cols, real.(v))
        if any(abs.(imag.(v)) .> 1e-12)
            push!(cols, imag.(v))
        end
    end
    A = reduce(hcat, cols)
    F = qr(A)
    r = rank(A; rtol=1e-10)
    Q = Matrix(F.Q)
    return Q[:, 1:min(r, size(Q, 2))]
end

function _floquet_right_unstable_basis(M::AbstractMatrix; tol::Float64=1e-4)
    E = eigen(Matrix{Float64}(M))
    idx = findall(μ -> abs(μ) > 1 + tol, E.values)
    return _real_right_basis(E.vectors, idx)
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
function _validate_saddle_cycle_geometry(split::_FloquetSplit, n::Int; tol_trivial::Float64=1e-6)
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

# --- cycle-to-cycle connections ----------------------------------------------

struct _CycleConnectionProblem
    field::Function
    base_params::Vector{Float64}
    primary_index::Int
    secondary_index::Int
    n::Int
    M::Int
    Ls::Int
    Lt::Int
    connection_reference::Matrix{Float64}
    connection_tangent::Matrix{Float64}
    eps0::Float64
    eps1::Float64
end

struct _CycleConnectionBC
    source_split::_FloquetSplit
    target_split::_FloquetSplit
end

function _phase_indices(count::Int, requested::Int)
    n = max(count - 1, 1)
    m = clamp(requested, 1, n)
    return unique!(round.(Int, range(1, n, length=m)))
end

function _rotate_closed_cycle(cycle::AbstractMatrix, phase_index::Int)
    L = size(cycle, 2) - 1
    1 <= phase_index <= L || throw(ArgumentError("cycle phase index out of range."))
    order = vcat(phase_index:L, 1:phase_index)
    return Matrix{Float64}(cycle[:, order])
end

function _nearest_cycle_sample(state::AbstractVector, cycle::AbstractMatrix,
                               target_indices::AbstractVector{Int})
    best_idx = first(target_indices)
    best_dist = Inf
    for idx in target_indices
        d = norm(state .- view(cycle, :, idx))
        if d < best_dist
            best_dist = d
            best_idx = idx
        end
    end
    return best_idx, best_dist
end

function _candidate_unstable_directions(field, cycle::AbstractMatrix, period::Real,
                                        p::AbstractVector, max_directions::Int)
    M = _cycle_monodromy(field, cycle, period, p)
    basis = _floquet_right_unstable_basis(M)
    size(basis, 2) >= 1 || throw(ArgumentError(
        "Source cycle has no unstable Floquet direction; automatic connection-seed discovery cannot launch."))
    count = min(size(basis, 2), max_directions)
    return [basis[:, j] ./ norm(basis[:, j]) for j in 1:count]
end

function _discover_cycle_connection_seed(
        sys::ContinuousODE, seed_config::CycleConnectionSeedConfig;
        source_cycle_states::AbstractMatrix,
        source_cycle_period::Real,
        target_cycle_states::AbstractMatrix,
        target_cycle_period::Real,
        base_params::AbstractVector,
        solver=Tsit5(),
        reltol::Float64=1e-9,
        abstol::Float64=1e-9)
    n = sys.dim
    source = Matrix{Float64}(source_cycle_states)
    target = Matrix{Float64}(target_cycle_states)
    field = _ode_field(sys)
    p = collect(Float64, base_params)
    source_indices = _phase_indices(size(source, 2), seed_config.source_phase_samples)
    target_indices = _phase_indices(size(target, 2), seed_config.target_phase_samples)
    saveat = collect(range(0.0, seed_config.max_time, length=seed_config.sample_count))
    min_sample = searchsortedfirst(saveat, seed_config.min_time)
    best = nothing
    attempts = 0
    failures = 0
    autonomous! = (du, u, pp, t) -> (sys.f(du, u, pp, 0.0); nothing)

    for src_idx in source_indices
        rotated_source = _rotate_closed_cycle(source, src_idx)
        local directions
        try
            directions = _candidate_unstable_directions(
                field, rotated_source, source_cycle_period, p, seed_config.max_directions)
        catch err
            err isa ArgumentError || rethrow()
            failures += 1
            continue
        end
        xsrc = collect(Float64, view(rotated_source, :, 1))
        for (dir_idx, direction) in enumerate(directions), sign in (-1.0, 1.0)
            attempts += 1
            u0 = xsrc .+ sign * seed_config.perturbation .* direction
            sol = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                solve(ODEProblem(autonomous!, u0, (0.0, seed_config.max_time), p),
                      solver; reltol=reltol, abstol=abstol, saveat=saveat)
            end
            if !SciMLBase.successful_retcode(sol.retcode)
                failures += 1
                continue
            end
            for j in max(min_sample, 2):length(sol.t)
                state = collect(Float64, sol.u[j])
                tgt_idx, dist = _nearest_cycle_sample(state, target, target_indices)
                if best === nothing || dist < best.distance
                    states = reduce(hcat, (collect(Float64, sol.u[k]) for k in 1:j))
                    best = (
                        distance=dist,
                        source_phase_index=src_idx,
                        target_phase_index=tgt_idx,
                        source_direction_index=dir_idx,
                        source_direction_sign=sign,
                        orbit_guess=states,
                        truncation_time=Float64(sol.t[j] - sol.t[1]),
                        source_cycle_states=rotated_source,
                        target_cycle_states=_rotate_closed_cycle(target, tgt_idx),
                    )
                end
            end
        end
    end

    diagnostics = Dict{String, Any}(
        "attempts" => attempts,
        "failed_integrations" => failures,
        "source_phase_samples" => length(source_indices),
        "target_phase_samples" => length(target_indices),
        "perturbation" => seed_config.perturbation,
        "max_time" => seed_config.max_time,
        "distance_tolerance" => seed_config.distance_tolerance,
    )
    attempts == 0 && throw(ArgumentError(
        "Source cycle has no unstable Floquet direction available for automatic " *
        "cycle-connection seed discovery."))
    best === nothing && return CycleConnectionSeedResult(
        zeros(Float64, n, 0), NaN, source, target, 0, 0, 0, 0.0, Inf,
        :not_found, diagnostics, now())
    status = best.distance <= seed_config.distance_tolerance ? :found : :not_found
    diagnostics["best_distance"] = best.distance
    return CycleConnectionSeedResult(
        best.orbit_guess,
        best.truncation_time,
        best.source_cycle_states,
        best.target_cycle_states,
        best.source_phase_index,
        best.target_phase_index,
        best.source_direction_index,
        best.source_direction_sign,
        best.distance,
        status,
        diagnostics,
        now())
end

function _cycle_connection_state_count(prob::_CycleConnectionProblem)
    return prob.n * (prob.M + 1) +
           prob.n * (prob.Ls + 1) +
           prob.n * (prob.Lt + 1)
end

_cycle_connection_slot_T(prob::_CycleConnectionProblem) =
    _cycle_connection_state_count(prob) + 1
_cycle_connection_slot_source_period(prob::_CycleConnectionProblem) =
    _cycle_connection_slot_T(prob) + 1
_cycle_connection_slot_target_period(prob::_CycleConnectionProblem) =
    _cycle_connection_slot_T(prob) + 2
_cycle_connection_slot_primary(prob::_CycleConnectionProblem) =
    _cycle_connection_slot_T(prob) + 3
_cycle_connection_slot_secondary(prob::_CycleConnectionProblem) =
    _cycle_connection_slot_T(prob) + 4
_cycle_connection_nz(prob::_CycleConnectionProblem) =
    _cycle_connection_state_count(prob) + 5

function _unpack_cycle_connection(z::AbstractVector, prob::_CycleConnectionProblem)
    n = prob.n
    M = prob.M
    Ls = prob.Ls
    Lt = prob.Lt
    c = 0
    U = reshape(view(z, c + 1:c + n * (M + 1)), n, M + 1)
    c += n * (M + 1)
    source_cycle = reshape(view(z, c + 1:c + n * (Ls + 1)), n, Ls + 1)
    c += n * (Ls + 1)
    target_cycle = reshape(view(z, c + 1:c + n * (Lt + 1)), n, Lt + 1)
    c += n * (Lt + 1)
    Tconn = z[c + 1]
    Tsource = z[c + 2]
    Ttarget = z[c + 3]
    α = z[c + 4]
    β = z[c + 5]
    return U, source_cycle, target_cycle, Tconn, Tsource, Ttarget, α, β
end

function _cycle_connection_param_vector(prob::_CycleConnectionProblem, α, β, ::Type{T}) where {T}
    p = Vector{T}(undef, length(prob.base_params))
    @inbounds for k in eachindex(p)
        p[k] = prob.base_params[k]
    end
    p[prob.primary_index] = α
    p[prob.secondary_index] = β
    return p
end

function _cycle_connection_seed_vector(prob::_CycleConnectionProblem, U::AbstractMatrix,
                                       source_cycle::AbstractMatrix,
                                       target_cycle::AbstractMatrix,
                                       Tconn::Real, Tsource::Real, Ttarget::Real,
                                       primary::Real, secondary::Real)
    z = Vector{Float64}(undef, _cycle_connection_nz(prob))
    n = prob.n
    M = prob.M
    Ls = prob.Ls
    Lt = prob.Lt
    c = 0
    z[c + 1:c + n * (M + 1)] = vec(U)
    c += n * (M + 1)
    z[c + 1:c + n * (Ls + 1)] = vec(source_cycle)
    c += n * (Ls + 1)
    z[c + 1:c + n * (Lt + 1)] = vec(target_cycle)
    c += n * (Lt + 1)
    z[c + 1] = Float64(Tconn)
    z[c + 2] = Float64(Tsource)
    z[c + 3] = Float64(Ttarget)
    z[c + 4] = Float64(primary)
    z[c + 5] = Float64(secondary)
    return z
end

function _cycle_mesh_residual(field, cycle::AbstractMatrix, period, p)
    T = promote_type(eltype(cycle), typeof(period), eltype(p))
    n = size(cycle, 1)
    L = size(cycle, 2) - 1
    fn = [field(view(cycle, :, i), p) for i in 1:L + 1]
    res = Vector{T}(undef, n * L + n)
    h = period / (2 * L)
    c = 0
    @inbounds for i in 1:L
        for r in 1:n
            res[c + r] = cycle[r, i + 1] - cycle[r, i] - h * (fn[i][r] + fn[i + 1][r])
        end
        c += n
    end
    @inbounds for r in 1:n
        res[c + r] = cycle[r, L + 1] - cycle[r, 1]
    end
    return res
end

function _connection_phase_residual(U::AbstractMatrix, prob::_CycleConnectionProblem)
    T = eltype(U)
    acc = zero(T)
    @inbounds for j in axes(U, 2), r in axes(U, 1)
        acc += (U[r, j] - prob.connection_reference[r, j]) * prob.connection_tangent[r, j]
    end
    return acc / length(U)
end

function _refresh_cycle_connection_bc(prob::_CycleConnectionProblem, z::AbstractVector{<:Real})
    _, source_cycle, target_cycle, _, Tsource, Ttarget, α, β =
        _unpack_cycle_connection(z, prob)
    p = _cycle_connection_param_vector(prob, Float64(α), Float64(β), Float64)
    Msource = _cycle_monodromy(prob.field, Matrix{Float64}(source_cycle), Float64(Tsource), p)
    Mtarget = _cycle_monodromy(prob.field, Matrix{Float64}(target_cycle), Float64(Ttarget), p)
    source_split = _floquet_split(Msource; tol=1e-4)
    target_split = _floquet_split(Mtarget; tol=1e-4)
    _validate_saddle_cycle_geometry(source_split, prob.n; tol_trivial=1e-4)
    _validate_saddle_cycle_geometry(target_split, prob.n; tol_trivial=1e-4)
    return _CycleConnectionBC(source_split, target_split)
end

function _cycle_connection_model_residual(z::AbstractVector, prob::_CycleConnectionProblem,
                                          bc::_CycleConnectionBC)
    T = eltype(z)
    n = prob.n
    M = prob.M
    U, source_cycle, target_cycle, Tconn, Tsource, Ttarget, α, β =
        _unpack_cycle_connection(z, prob)
    p = _cycle_connection_param_vector(prob, α, β, T)
    source_res = _cycle_mesh_residual(prob.field, source_cycle, Tsource, p)
    target_res = _cycle_mesh_residual(prob.field, target_cycle, Ttarget, p)
    ns_bc = size(bc.source_split.Ws_left, 2)
    nt_bc = size(bc.target_split.Wu_left, 2)
    Nm = n * M + length(source_res) + length(target_res) + ns_bc + nt_bc + 3
    res = Vector{T}(undef, Nm)
    c = 0
    h = Tconn / (2 * M)
    f_prev = prob.field(view(U, :, 1), p)
    @inbounds for i in 1:M
        f_next = prob.field(view(U, :, i + 1), p)
        for r in 1:n
            res[c + r] = U[r, i + 1] - U[r, i] - h * (f_prev[r] + f_next[r])
        end
        f_prev = f_next
        c += n
    end
    @inbounds for r in eachindex(source_res)
        res[c + r] = source_res[r]
    end
    c += length(source_res)
    @inbounds for r in eachindex(target_res)
        res[c + r] = target_res[r]
    end
    c += length(target_res)

    d0 = collect(view(U, :, 1)) .- collect(view(source_cycle, :, 1))
    d1 = collect(view(U, :, M + 1)) .- collect(view(target_cycle, :, 1))
    bcl = bc.source_split.Ws_left' * d0
    @inbounds for r in 1:ns_bc
        res[c + r] = bcl[r]
    end
    c += ns_bc
    bcr = bc.target_split.Wu_left' * d1
    @inbounds for r in 1:nt_bc
        res[c + r] = bcr[r]
    end
    c += nt_bc
    res[c + 1] = dot(d0, d0) - prob.eps0^2
    res[c + 2] = dot(d1, d1) - prob.eps1^2
    res[c + 3] = _connection_phase_residual(U, prob)
    return res
end

function _cycle_connection_augmented_residual(z::AbstractVector,
                                             prob::_CycleConnectionProblem,
                                             bc::_CycleConnectionBC,
                                             extra)
    r = _cycle_connection_model_residual(z, prob, bc)
    extra === nothing && return r
    return vcat(r, extra(z))
end

function _cycle_connection_deficiency(prob::_CycleConnectionProblem, bc::_CycleConnectionBC)
    Nm = prob.n * prob.M +
         prob.n * (prob.Ls + 1) +
         prob.n * (prob.Lt + 1) +
         size(bc.source_split.Ws_left, 2) +
         size(bc.target_split.Wu_left, 2) + 3
    return _cycle_connection_nz(prob) - Nm
end

function _validate_cycle_connection_geometry(prob::_CycleConnectionProblem,
                                             bc::_CycleConnectionBC)
    k = _cycle_connection_deficiency(prob, bc)
    if k < 1
        throw(ArgumentError(
            "Saddle-cycle manifold dimensions cannot support a cycle-connection " *
            "curve in two parameters (free-direction count k = $(k) < 1; source " *
            "stable Floquet dim = $(bc.source_split.ns), target unstable Floquet dim = " *
            "$(bc.target_split.nu), state dim = $(prob.n))."))
    elseif k > 1
        throw(ArgumentError(
            "Saddle-cycle manifold dimensions leave $(k) free directions for a " *
            "cycle connection (source stable Floquet dim = $(bc.source_split.ns), " *
            "target unstable Floquet dim = $(bc.target_split.nu), state dim = $(prob.n)); " *
            "the connection is non-isolated in two parameters."))
    end
    return k
end

function _correct_cycle_connection(prob::_CycleConnectionProblem, z0::AbstractVector;
                                   extra=nothing, tol::Float64=1e-9,
                                   maxiter::Int=80, use_fallback::Bool=true,
                                   fallback_max_iter::Int=150,
                                   projector_refresh::Int=1)
    z = collect(float.(z0))
    path = :newton
    iterations = 0
    bc = _refresh_cycle_connection_bc(prob, z)
    rn = norm(_cycle_connection_augmented_residual(z, prob, bc, extra))
    final_with_fresh_bc(zz, current_path, current_iterations) = begin
        fresh_bc = try
            _refresh_cycle_connection_bc(prob, zz)
        catch err
            err isa ArgumentError || rethrow()
            return _CorrectorResult(collect(Float64, zz), false, Inf, current_path, current_iterations)
        end
        fresh_rn = norm(_cycle_connection_augmented_residual(zz, prob, fresh_bc, extra))
        return _CorrectorResult(collect(Float64, zz), fresh_rn <= tol, fresh_rn,
                                current_path, current_iterations)
    end
    for iter_num in 1:maxiter
        if (iter_num - 1) % projector_refresh == 0 && iter_num > 1
            bc = try
                _refresh_cycle_connection_bc(prob, z)
            catch err
                err isa ArgumentError || rethrow()
                bc
            end
        end
        G = zz -> _cycle_connection_augmented_residual(zz, prob, bc, extra)
        r = G(z)
        rn = norm(r)
        rn <= tol && return final_with_fresh_bc(z, path, iterations)
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
    rn = norm(_cycle_connection_augmented_residual(z, prob, bc, extra))
    rn <= tol && return final_with_fresh_bc(z, path, iterations)
    use_fallback || return _CorrectorResult(z, false, rn, path, iterations)

    path = :fallback
    λ = 1e-3
    for iter_num in 1:fallback_max_iter
        if (iter_num - 1) % projector_refresh == 0 && iter_num > 1
            bc = try
                _refresh_cycle_connection_bc(prob, z)
            catch err
                err isa ArgumentError || rethrow()
                bc
            end
        end
        G = zz -> _cycle_connection_augmented_residual(zz, prob, bc, extra)
        r = G(z)
        rn = norm(r)
        rn <= tol && return final_with_fresh_bc(z, path, iterations)
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
    rn = norm(_cycle_connection_augmented_residual(z, prob, bc, extra))
    rn <= tol && return final_with_fresh_bc(z, path, iterations)
    return _CorrectorResult(z, false, rn, path, iterations)
end
