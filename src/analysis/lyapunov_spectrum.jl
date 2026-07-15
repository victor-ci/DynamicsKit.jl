"""
Full Lyapunov spectra via the Benettin/QR (tangent-space) method.

Complements the two-trajectory largest-exponent estimators in `lyapunov.jl`: those
sweep a parameter and report only the leading exponent, while `lyapunov_spectrum`
evolves an orthonormal tangent frame at one operating point and recovers the whole
ordered spectrum. Discrete maps propagate the frame with the analytic (AD) Jacobian
of the map; continuous flows integrate the first variational equation
`dQ/dt = J(u(t)) Q` alongside the state and reorthonormalize on a fixed time grid.
The exponents are the time-averaged logarithms of the QR stretching factors.
"""

# Resolve the requested number of exponents (0 selects the full state dimension).
function _lyapunov_spectrum_k(dim::Int, k::Int)
    kk = k <= 0 ? dim : k
    1 <= kk <= dim || throw(ArgumentError(
        "LyapunovSpectrumConfig.k=$k must satisfy 1 <= k <= state dimension $dim (0 selects the full dimension)."))
    return kk
end

function _lyapunov_spectrum_failure(k::Int, status::Symbol, kind::Symbol,
                                    params::AbstractVector, system_name::String)
    return LyapunovSpectrumResult(
        fill(NaN, k),
        Matrix{Float64}(undef, 0, k),
        status,
        0,
        0.0,
        kind,
        collect(Float64, params),
        system_name,
        now()
    )
end

# One QR reorthonormalization of the evolved tangent block `Y` (dim × k). Returns the
# sign-corrected orthonormal frame and the log-magnitudes of the R diagonal (the local
# per-interval stretching exponents). Column signs are fixed so the R diagonal is
# positive, keeping the frame a consistent Gram-Schmidt basis across intervals.
function _lyapunov_qr_step(Y::AbstractMatrix, k::Int)
    factorization = qr(Y)
    Q = Matrix(factorization.Q)[:, 1:k]
    R = factorization.R
    log_stretch = Vector{Float64}(undef, k)
    ok = true
    for i in 1:k
        rii = R[i, i]
        magnitude = abs(rii)
        if rii < 0
            @views Q[:, i] .*= -1.0
        end
        if magnitude > 0 && isfinite(magnitude)
            log_stretch[i] = log(magnitude)
        else
            log_stretch[i] = -Inf
            ok = false
        end
    end
    return Q, log_stretch, ok
end

"""
    lyapunov_spectrum(sys::DiscreteMap, config::LyapunovSpectrumConfig; kwargs...) -> LyapunovSpectrumResult

Estimate the full Lyapunov spectrum of a discrete map at a single operating point by
propagating an orthonormal tangent frame with the map's automatic-differentiation
Jacobian and reorthonormalizing (QR) at every iteration.

Keyword arguments:
- `params`: Parameter vector; defaults to a zero vector of the system's parameter arity.
- `initial_point`: Seed state; defaults to the origin.
"""
function lyapunov_spectrum(sys::DiscreteMap, config::LyapunovSpectrumConfig;
                           params::AbstractVector=Float64[],
                           initial_point::Union{Nothing, AbstractVector}=nothing)
    dim = sys.dim
    k = _lyapunov_spectrum_k(dim, config.k)
    pv = isempty(params) ? zeros(Float64, length(sys.param_names)) : collect(Float64, params)
    point = isnothing(initial_point) ? zeros(SVector{dim, Float64}) : SVector{dim, Float64}(initial_point)

    map_rule = x -> sys.f(SVector{dim}(x), pv)
    frame = Matrix{Float64}(I, dim, dim)[:, 1:k]
    log_sum = zeros(Float64, k)
    convergence = Matrix{Float64}(undef, config.steps, k)
    accumulated = 0

    total_intervals = config.transient + config.steps
    for interval in 1:total_intervals
        status = _map_state_status(point, config.divergence_cutoff)
        status == :ok || return _lyapunov_spectrum_failure(k, status, :discrete_map, pv, sys.name)

        jacobian = ForwardDiff.jacobian(map_rule, Vector(point))
        evolved = jacobian * frame
        frame, log_stretch, ok = _lyapunov_qr_step(evolved, k)
        ok || return _lyapunov_spectrum_failure(k, :collapsed, :discrete_map, pv, sys.name)

        if interval > config.transient
            accumulated += 1
            @. log_sum += log_stretch
            @views convergence[accumulated, :] .= log_sum ./ accumulated
        end

        point = sys.f(point, pv)
    end

    exponents = log_sum ./ config.steps
    order = sortperm(exponents; rev=true)
    return LyapunovSpectrumResult(
        exponents[order],
        convergence[:, order],
        :ok,
        config.steps,
        Float64(config.steps),
        :discrete_map,
        pv,
        sys.name,
        now()
    )
end

# Augmented right-hand side for the flow plus its first variational equation. The state
# occupies the first `dim` slots; the tangent block Q (dim × k) is stored flattened after
# it. The Jacobian is taken by automatic differentiation of the (out-of-place) vector
# field, so it stays element-type generic and does not assume an analytic Jacobian.
function _flow_variational_rhs(sys::ContinuousODE, dim::Int, k::Int)
    return function (dz, z, p, t)
        u = @view z[1:dim]
        du = @view dz[1:dim]
        sys.f(du, u, p, t)
        jacobian = ForwardDiff.jacobian(uu -> begin
            out = similar(uu)
            sys.f(out, uu, p, t)
            out
        end, Vector(u))
        Q = reshape(@view(z[(dim + 1):(dim + dim * k)]), dim, k)
        dQ = reshape(@view(dz[(dim + 1):(dim + dim * k)]), dim, k)
        mul!(dQ, jacobian, Q)
        return nothing
    end
end

"""
    lyapunov_spectrum(sys::ContinuousODE, config::LyapunovSpectrumConfig; kwargs...) -> LyapunovSpectrumResult

Estimate the full Lyapunov spectrum of a continuous flow at a single operating point.
The state and an orthonormal tangent frame are integrated together through the first
variational equation and the frame is reorthonormalized (QR) every `config.renorm_dt`
of flow time; exponents are the time-averaged logarithms of the QR stretching factors.
For a bounded, non-equilibrium flow one exponent is (numerically) zero — the direction
along the trajectory — which is a built-in consistency check.

Keyword arguments:
- `params`: Parameter vector; falls back to the system's `default_params`.
- `initial_point`: Seed state; falls back to the system's `default_initial_state`.
- `solver`, `reltol`, `abstol`: ODE integration controls (a non-stiff solver is assumed).
"""
function lyapunov_spectrum(sys::ContinuousODE, config::LyapunovSpectrumConfig;
                           params::AbstractVector=Float64[],
                           initial_point::Union{Nothing, AbstractVector}=nothing,
                           solver=Tsit5(),
                           reltol::Float64=1e-9,
                           abstol::Float64=1e-9)
    dim = sys.dim
    k = _lyapunov_spectrum_k(dim, config.k)
    pv = _resolve_continuous_params(sys, collect(Float64, params))
    u0 = _resolve_initial_state(sys, initial_point)
    _map_state_status(u0, config.divergence_cutoff) == :ok || return _lyapunov_spectrum_failure(
        k, :invalid_state, :continuous_flow, pv, sys.name)

    rhs! = _flow_variational_rhs(sys, dim, k)
    frame = Matrix{Float64}(I, dim, dim)[:, 1:k]
    z = vcat(collect(Float64, u0), vec(frame))
    log_sum = zeros(Float64, k)
    convergence = Matrix{Float64}(undef, config.steps, k)
    accumulated = 0
    accumulated_time = 0.0

    total_intervals = config.transient + config.steps
    for interval in 1:total_intervals
        problem = ODEProblem(rhs!, z, (0.0, config.renorm_dt), pv)
        solution = solve(problem, solver; reltol=reltol, abstol=abstol,
                         save_everystep=false, save_start=false)
        SciMLBase.successful_retcode(solution.retcode) || return _lyapunov_spectrum_failure(
            k, :integration_failed, :continuous_flow, pv, sys.name)
        zf = solution.u[end]

        state = @view zf[1:dim]
        status = _map_state_status(state, config.divergence_cutoff)
        status == :ok || return _lyapunov_spectrum_failure(k, status, :continuous_flow, pv, sys.name)

        evolved = reshape(zf[(dim + 1):(dim + dim * k)], dim, k)
        frame, log_stretch, ok = _lyapunov_qr_step(evolved, k)
        ok || return _lyapunov_spectrum_failure(k, :collapsed, :continuous_flow, pv, sys.name)

        if interval > config.transient
            accumulated += 1
            accumulated_time += config.renorm_dt
            @. log_sum += log_stretch
            @views convergence[accumulated, :] .= log_sum ./ accumulated_time
        end

        z = vcat(collect(Float64, state), vec(frame))
    end

    exponents = log_sum ./ accumulated_time
    order = sortperm(exponents; rev=true)
    return LyapunovSpectrumResult(
        exponents[order],
        convergence[:, order],
        :ok,
        config.steps,
        accumulated_time,
        :continuous_flow,
        pv,
        sys.name,
        now()
    )
end
