"""
Uniform-sampling time-series capture and one-sided FFT power spectra.
"""

function _spectrum_tail_start(total_points::Int, tail_fraction::Float64)
    total_points > 0 || return 1
    return max(1, floor(Int, total_points * (1.0 - tail_fraction)) + 1)
end

function _spectrum_window(kind::Symbol, n::Int)
    n >= 1 || return Float64[]
    kind == :none && return ones(Float64, n)
    n == 1 && return ones(Float64, 1)
    kind == :hann || throw(ArgumentError("Unsupported spectrum window $(repr(kind)); expected :hann or :none."))
    return [0.5 * (1 - cos(2pi * (idx - 1) / (n - 1))) for idx in 1:n]
end

function _one_sided_frequency_grid(sample_count::Int, dt::Float64)
    return collect(0:floor(Int, sample_count / 2)) ./ (sample_count * dt)
end

"""
    power_spectrum(sys::ContinuousODE, config::PowerSpectrumConfig; kwargs...) -> PowerSpectrumResult

Integrate a continuous-time system on a uniform sampling grid, keep the requested
tail, detrend it, apply the configured window, and return a one-sided FFT power
spectrum for one observed state coordinate.
"""
function power_spectrum(sys::ContinuousODE, config::PowerSpectrumConfig;
                        params::Vector{Float64}=Float64[],
                        initial_point::Union{Nothing, AbstractVector}=nothing,
                        solver=Tsit5(),
                        reltol::Float64=1e-8,
                        abstol::Float64=1e-8)
    1 <= config.state_index <= sys.dim || throw(ArgumentError(
        "PowerSpectrumConfig.state_index $(config.state_index) must lie in 1:$(sys.dim) for $(sys.name)."
    ))

    local_params = _resolve_continuous_params(sys, params)
    u0 = _resolve_initial_state(sys, initial_point)
    sample_times = collect(config.time_start:config.dt:config.time_stop)
    length(sample_times) >= 2 || throw(ArgumentError(
        "Need at least two uniformly sampled times to compute a power spectrum; got $(length(sample_times)). " *
        "Increase time_stop - time_start or decrease dt."
    ))
    prob = ODEProblem(sys.f, u0, (config.time_start, config.time_stop), local_params)
    sol = solve(
        prob,
        solver;
        saveat=sample_times,
        save_everystep=false,
        save_start=false,
        save_end=false,
        reltol=reltol,
        abstol=abstol,
        maxiters=config.maxiters
    )

    total_points = length(sol.t)
    total_points >= 2 || throw(ArgumentError(
        "Need at least two uniformly sampled points to compute a power spectrum; got $total_points."
    ))
    tail_start = _spectrum_tail_start(total_points, config.tail_fraction)
    tail_t = Float64[Float64(sol.t[idx]) for idx in tail_start:total_points]
    tail_signal = Float64[Float64(sol.u[idx][config.state_index]) for idx in tail_start:total_points]
    length(tail_signal) >= 2 || throw(ArgumentError(
        "Tail extraction kept $(length(tail_signal)) point(s); increase time_stop or tail_fraction for spectrum estimation."
    ))

    centered = tail_signal .- mean(tail_signal)
    window = _spectrum_window(config.window, length(centered))
    windowed = centered .* window
    spectrum = rfft(windowed)
    power = Float64.(abs2.(spectrum))
    frequency = _one_sided_frequency_grid(length(windowed), config.dt)

    return PowerSpectrumResult(
        tail_t,
        tail_signal,
        frequency,
        power,
        local_params,
        config.state_index,
        sys.name,
        now()
    )
end
