@testset "Lyapunov spectrum (Benettin/QR)" begin
    @testset "Discrete diagonal map recovers analytic log-multipliers, ordered descending" begin
        sys = DiscreteMap((x, p) -> SVector(p[1] * x[1], p[2] * x[2]), 2, [:a, :b], "Diagonal multipliers")
        result = lyapunov_spectrum(
            sys,
            LyapunovSpectrumConfig(transient=10, steps=400);
            params=[0.5, 0.9],
            initial_point=[0.3, 0.4]
        )

        @test result isa LyapunovSpectrumResult
        @test result.kind == :discrete_map
        @test result.estimation_status == :ok
        @test result.sample_count == 400
        @test result.total_time == 400.0
        @test length(result.exponents) == 2
        # Ordered from largest to smallest: log(0.9) > log(0.5).
        @test result.exponents[1] ≈ log(0.9) atol=1e-8
        @test result.exponents[2] ≈ log(0.5) atol=1e-8
        @test issorted(result.exponents; rev=true)
        @test size(result.convergence) == (400, 2)
        # Running estimate at the final interval equals the reported spectrum.
        @test result.convergence[end, 1] ≈ result.exponents[1] atol=1e-12
        @test result.convergence[end, 2] ≈ result.exponents[2] atol=1e-12

        @test !isnothing(plot_lyapunov_spectrum(result))
    end

    @testset "Hénon spectrum matches literature, the volume invariant, and the estimator" begin
        sys = henon_map()
        params = [1.4, 0.3]
        result = lyapunov_spectrum(
            sys,
            LyapunovSpectrumConfig(transient=1000, steps=12000);
            params=params,
            initial_point=[0.1, 0.1]
        )

        @test result.estimation_status == :ok
        @test length(result.exponents) == 2
        @test issorted(result.exponents; rev=true)
        # Published spectrum for (a, b) = (1.4, 0.3): λ₁ ≈ 0.419, λ₂ ≈ -1.623.
        @test result.exponents[1] ≈ 0.41922 atol=1e-2
        @test result.exponents[2] ≈ -1.62319 atol=1e-2
        # Sum equals the constant log volume-contraction rate log|det J| = log|b|, exactly.
        @test sum(result.exponents) ≈ log(0.3) atol=1e-8

        # The leading spectrum exponent agrees with the two-trajectory estimator.
        estimate = estimate_discrete_map_largest_lyapunov(
            sys, params, SVector(0.1, 0.1), 1000, 12000, 1e-8, Inf)
        @test result.exponents[1] ≈ estimate.exponent atol=5e-3
    end

    @testset "Continuous linear ODE recovers eigenvalue real parts" begin
        sys = ContinuousODE(
            (du, u, p, t) -> begin
                du[1] = -0.5 * u[1]
                du[2] = -1.0 * u[2]
                du[3] = -2.0 * u[3]
                nothing
            end,
            3,
            PoincareSection((u, t, integrator) -> u[1]; direction=:both, projection=[1, 2, 3], template=zeros(3)),
            Symbol[],
            "Linear contraction";
            default_initial_state=[1.0, 1.0, 1.0],
            default_params=[0.0]
        )
        result = lyapunov_spectrum(sys, LyapunovSpectrumConfig(transient=20, steps=200, renorm_dt=0.25))

        @test result.kind == :continuous_flow
        @test result.estimation_status == :ok
        @test result.total_time ≈ 200 * 0.25 atol=1e-9
        @test issorted(result.exponents; rev=true)
        @test result.exponents ≈ [-0.5, -1.0, -2.0] atol=1e-4
    end

    @testset "Stiff (Rosenbrock) solver runs the generic RHS path" begin
        # Rosenbrock W-methods differentiate the RHS with Dual numbers; the
        # variational closure must accept them (element-type-generic fallback).
        sys = ContinuousODE(
            (du, u, p, t) -> begin
                du[1] = -0.5 * u[1]
                du[2] = -1.0 * u[2]
                du[3] = -2.0 * u[3]
                nothing
            end,
            3,
            PoincareSection((u, t, integrator) -> u[1]; direction=:both, projection=[1, 2, 3], template=zeros(3)),
            Symbol[],
            "Linear contraction (stiff solver)";
            default_initial_state=[1.0, 1.0, 1.0],
            default_params=[0.0]
        )
        result = lyapunov_spectrum(
            sys,
            LyapunovSpectrumConfig(transient=20, steps=200, renorm_dt=0.25);
            solver=select_ode_solver("rosenbrock23"),
            reltol=1e-9, abstol=1e-9
        )

        @test result.estimation_status == :ok
        @test result.exponents ≈ [-0.5, -1.0, -2.0] atol=1e-3
    end

    @testset "Nonautonomous flow sees continuous time across windows" begin
        # du = (0.25 + cos(t)) u: the time-averaged exponent is 0.25 with a
        # bounded O(1/T) remainder. A per-window time reset would instead bias
        # the estimate by sin(renorm_dt)/renorm_dt ≈ 0.96.
        sys = ContinuousODE(
            (du, u, p, t) -> (du[1] = (0.25 + cos(t)) * u[1]; nothing),
            1,
            PoincareSection((u, t, integrator) -> u[1]; direction=:both, projection=[1], template=zeros(1)),
            Symbol[],
            "Nonautonomous growth";
            default_initial_state=[1.0],
            default_params=[0.0]
        )
        result = lyapunov_spectrum(sys, LyapunovSpectrumConfig(transient=40, steps=1200, renorm_dt=0.5))

        @test result.estimation_status == :ok
        @test result.exponents[1] ≈ 0.25 atol=6e-3
    end

    @testset "Undamped oscillator yields a neutral (zero) spectrum" begin
        sys = ContinuousODE(
            (du, u, p, t) -> begin
                du[1] = u[2]
                du[2] = -u[1]
                nothing
            end,
            2,
            PoincareSection((u, t, integrator) -> u[1]; direction=:both, projection=[1, 2], template=zeros(2)),
            Symbol[],
            "Harmonic oscillator";
            default_initial_state=[1.0, 0.0],
            default_params=[0.0]
        )
        result = lyapunov_spectrum(sys, LyapunovSpectrumConfig(transient=50, steps=1200, renorm_dt=0.3))

        @test result.estimation_status == :ok
        @test all(abs.(result.exponents) .< 1e-4)
        # Area preservation (zero divergence): the spectrum sums to zero.
        @test sum(result.exponents) ≈ 0.0 atol=1e-6
    end

    @testset "Rössler flow spectrum is (+, 0, -) with the textbook leading exponent" begin
        sys = rossler_oscillator()
        result = lyapunov_spectrum(
            sys,
            LyapunovSpectrumConfig(transient=200, steps=1500, renorm_dt=0.5);
            params=[0.2, 0.2, 5.7]
        )

        @test result.kind == :continuous_flow
        @test result.estimation_status == :ok
        @test length(result.exponents) == 3
        @test issorted(result.exponents; rev=true)
        # Chaotic single-scroll: one positive, one zero (flow direction), one strongly negative.
        @test result.exponents[1] > 0.03
        @test result.exponents[1] ≈ 0.0714 atol=1.5e-2
        @test abs(result.exponents[2]) < 1e-2
        @test result.exponents[3] ≈ -5.39 atol=0.3
        # Dissipative flow: negative sum of exponents.
        @test sum(result.exponents) < 0.0
    end

    @testset "Partial spectra track only the leading exponents" begin
        sys = rossler_oscillator()
        result = lyapunov_spectrum(
            sys,
            LyapunovSpectrumConfig(k=2, transient=200, steps=1200, renorm_dt=0.5);
            params=[0.2, 0.2, 5.7]
        )

        @test length(result.exponents) == 2
        @test size(result.convergence, 2) == 2
        @test issorted(result.exponents; rev=true)
        @test result.exponents[1] > 0.03
        @test abs(result.exponents[2]) < 1e-2
    end

    @testset "Divergent trajectories report a failure status" begin
        sys = DiscreteMap((x, p) -> SVector(2.0 * x[1]), 1, [:a], "Blowup")
        result = lyapunov_spectrum(
            sys,
            LyapunovSpectrumConfig(transient=5, steps=100, divergence_cutoff=1e6);
            params=[0.0],
            initial_point=[1.0]
        )

        @test result.estimation_status == :diverged
        @test result.sample_count == 0
        @test all(isnan, result.exponents)
        @test size(result.convergence) == (0, 1)
        # Failure results carry nothing to plot and say so.
        @test_throws ArgumentError plot_lyapunov_spectrum(result)
    end

    @testset "Requesting more exponents than the dimension is rejected" begin
        sys = henon_map()
        @test_throws ArgumentError lyapunov_spectrum(
            sys, LyapunovSpectrumConfig(k=5); params=[1.4, 0.3])
    end
end
