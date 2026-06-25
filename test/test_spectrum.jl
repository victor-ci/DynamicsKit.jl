@testset "Power spectrum" begin
    function harmonic_oscillator(; ω::Float64=2pi)
        f! = function(du, u, p, t)
            du[1] = u[2]
            du[2] = -(p[1]^2) * u[1]
        end
        section = PoincareSection((u, t, integrator) -> u[1]; direction=:up, projection=[2], template=[0.0, 1.0])
        return ContinuousODE(
            f!,
            2,
            section,
            [:ω],
            "Harmonic oscillator";
            tspan_hint=2pi / ω,
            default_initial_state=[1.0, 0.0],
            default_params=[ω]
        )
    end

    @testset "Simple harmonic oscillator peaks at the driving frequency" begin
        sys = harmonic_oscillator()
        result = power_spectrum(sys, PowerSpectrumConfig(
            time_stop = 20.0,
            dt = 0.01,
            tail_fraction = 0.5,
            window = :hann,
            state_index = 1
        ))

        @test result isa PowerSpectrumResult
        @test result.system_name == "Harmonic oscillator"
        @test length(result.t) == length(result.signal)
        @test length(result.frequency) == length(result.power)
        @test result.state_index == 1

        dominant_idx = argmax(result.power[2:end]) + 1
        @test result.frequency[dominant_idx] ≈ 1.0 atol=0.05

        @test !isnothing(plot_power_spectrum(result; plot_kwargs=(legend=false,), spectrum_kwargs=(xlims=(0.0, 3.0),)))
    end

    @testset "Spectrum sampling stays on the explicit dt grid" begin
        sys = harmonic_oscillator()
        result = power_spectrum(sys, PowerSpectrumConfig(
            time_stop = 20.03,
            dt = 0.01,
            tail_fraction = 0.5,
            window = :none,
            state_index = 1
        ))

        @test length(result.t) >= 2
        @test all(diff(result.t) .≈ 0.01)
        @test result.t[end] <= 20.03
    end
end
