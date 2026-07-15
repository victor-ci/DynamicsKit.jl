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
# per-interval stretching exponents). The thin orthonormal factor is formed in economy
# size (dim × k) rather than materializing the full dim × dim Q factor. Column signs are
# fixed so the R diagonal is positive, keeping the frame a consistent Gram-Schmidt basis
# across intervals.
function _lyapunov_qr_step(Y::AbstractMatrix, k::Int)
    dim = size(Y, 1)
    factorization = qr(Y)
    frame = Matrix{Float64}(I, dim, k)
    lmul!(factorization.Q, frame)   # frame ← first k columns of Q, without forming full Q
    R = factorization.R
    log_stretch = Vector{Float64}(undef, k)
    ok = true
    for i in 1:k
        rii = R[i, i]
        magnitude = abs(rii)
        if rii < 0
            @views frame[:, i] .*= -1.0
        end
        if magnitude > 0 && isfinite(magnitude)
            log_stretch[i] = log(magnitude)
        else
            log_stretch[i] = -Inf
            ok = false
        end
    end
    return frame, log_stretch, ok
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

    # Reused Jacobian work buffers + AD config (single-threaded loop, so sharing is safe).
    state_buffer = Vector{Float64}(undef, dim)
    jacobian = Matrix{Float64}(undef, dim, dim)
    evolved = Matrix{Float64}(undef, dim, k)
    jac_config = ForwardDiff.JacobianConfig(map_rule, state_buffer)

    total_intervals = config.transient + config.steps
    for interval in 1:total_intervals
        status = _map_state_status(point, config.divergence_cutoff)
        status == :ok || return _lyapunov_spectrum_failure(k, status, :discrete_map, pv, sys.name)

        copyto!(state_buffer, point)
        ForwardDiff.jacobian!(jacobian, map_rule, state_buffer, jac_config)
        mul!(evolved, jacobian, frame)
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
# it. Two execution paths share one closure: plain Float64 evaluations reuse a
# preallocated ForwardDiff JacobianConfig and work buffers (no per-call allocation), and
# an element-type-generic fallback handles the Dual numbers that stiff Rosenbrock
# W-methods (e.g. `select_ode_solver("auto")`) push through the RHS when differentiating
# it — the same requirement `_ode_state_jacobian` satisfies for the return-map path.
function _flow_variational_rhs(sys::ContinuousODE, dim::Int, k::Int, pv::Vector{Float64})
    state_buffer = Vector{Float64}(undef, dim)
    field_buffer = Vector{Float64}(undef, dim)
    jacobian = Matrix{Float64}(undef, dim, dim)
    time_ref = Ref(0.0)
    field! = (out, x) -> (sys.f(out, x, pv, time_ref[]); nothing)
    jac_config = ForwardDiff.JacobianConfig(field!, field_buffer, state_buffer)
    return function (dz, z, p, t)
        T = promote_type(eltype(z), eltype(dz), typeof(t))
        if T === Float64
            time_ref[] = t
            copyto!(state_buffer, 1, z, 1, dim)
            # `jacobian!` also writes the primal f(u) into `field_buffer`, reused as the state derivative.
            ForwardDiff.jacobian!(jacobian, field!, field_buffer, state_buffer, jac_config)
            copyto!(dz, 1, field_buffer, 1, dim)
            Q = reshape(@view(z[(dim + 1):(dim + dim * k)]), dim, k)
            dQ = reshape(@view(dz[(dim + 1):(dim + dim * k)]), dim, k)
            mul!(dQ, jacobian, Q)
        else
            u = collect(T, @view z[1:dim])
            du = Vector{T}(undef, dim)
            sys.f(du, u, pv, t)
            copyto!(dz, 1, du, 1, dim)
            field_ad = x -> begin
                out = similar(x)
                sys.f(out, x, pv, t)
                out
            end
            J = ForwardDiff.jacobian(field_ad, u)
            Q = reshape(@view(z[(dim + 1):(dim + dim * k)]), dim, k)
            dQ = reshape(@view(dz[(dim + 1):(dim + dim * k)]), dim, k)
            mul!(dQ, J, Q)
        end
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
- `solver`, `reltol`, `abstol`: ODE integration controls. Non-stiff solvers run on a
  preallocated fast path; stiff/auto-switching solvers (e.g. `select_ode_solver("auto")`)
  are supported through an element-type-generic fallback. Flow time is continuous across
  reorthonormalization windows, so nonautonomous right-hand sides see the true `t`.
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

    rhs! = _flow_variational_rhs(sys, dim, k, pv)
    frame = Matrix{Float64}(I, dim, dim)[:, 1:k]
    z = Vector{Float64}(undef, dim + dim * k)
    copyto!(z, 1, collect(Float64, u0), 1, dim)
    copyto!(z, dim + 1, vec(frame), 1, dim * k)
    log_sum = zeros(Float64, k)
    convergence = Matrix{Float64}(undef, config.steps, k)
    accumulated = 0
    accumulated_time = 0.0

    # One integrator advanced over each renormalization window via reinit!, avoiding a
    # fresh ODEProblem/solve setup per interval.
    problem = ODEProblem(rhs!, z, (0.0, config.renorm_dt), pv)
    integrator = init(problem, solver; reltol=reltol, abstol=abstol,
                      save_everystep=false, save_start=false, save_end=false)

    total_intervals = config.transient + config.steps
    for interval in 1:total_intervals
        solve!(integrator)
        SciMLBase.successful_retcode(integrator.sol.retcode) || return _lyapunov_spectrum_failure(
            k, :integration_failed, :continuous_flow, pv, sys.name)

        state = @view integrator.u[1:dim]
        status = _map_state_status(state, config.divergence_cutoff)
        status == :ok || return _lyapunov_spectrum_failure(k, status, :continuous_flow, pv, sys.name)

        evolved = reshape(integrator.u[(dim + 1):(dim + dim * k)], dim, k)
        frame, log_stretch, ok = _lyapunov_qr_step(evolved, k)
        ok || return _lyapunov_spectrum_failure(k, :collapsed, :continuous_flow, pv, sys.name)

        if interval > config.transient
            accumulated += 1
            accumulated_time += config.renorm_dt
            @. log_sum += log_stretch
            @views convergence[accumulated, :] .= log_sum ./ accumulated_time
        end

        if interval < total_intervals
            copyto!(z, 1, integrator.u, 1, dim)
            copyto!(z, dim + 1, vec(frame), 1, dim * k)
            # Keep flow time continuous across windows so nonautonomous
            # right-hand sides see the true t rather than a per-window reset.
            window_start = interval * config.renorm_dt
            reinit!(integrator, z; t0=window_start, tf=window_start + config.renorm_dt)
        end
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
