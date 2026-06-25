@testset "Lyapunov diagnostics" begin
    @testset "Discrete Lyapunov diagram classifies stable, neutral, and expanding samples" begin
        sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:a], "Linear multiplier")
        config = LyapunovConfig(
            param_min = 0.5,
            param_max = 1.5,
            param_steps = 2,
            iterations = 40,
            transient = 0,
            neutral_tolerance = 1e-6,
            divergence_cutoff = Inf
        )
        result = lyapunov_diagram(sys, config; initial_point=[1.0])

        @test result isa LyapunovDiagramResult
        @test result.params == [0.5, 1.0, 1.5]
        @test result.param_name == :a
        @test result.classifications == [:periodic, :quasiperiodic_neutral_candidate, :chaotic_candidate]
        @test result.estimation_statuses == fill(:ok, 3)
        @test result.sample_counts == fill(config.iterations, 3)
        @test result.exponents[1] ≈ log(0.5) atol=1e-8
        @test result.exponents[2] ≈ 0.0 atol=1e-8
        @test result.exponents[3] ≈ log(1.5) atol=5e-3

        @test !isnothing(plot_lyapunov_diagram(result))
    end

    @testset "2D bifurcation maps expose a first-class Lyapunov field" begin
        sys = DiscreteMap((x, p) -> SVector(p[1]), 1, [:a, :b], "Driven fixed point")
        map = bifurcation_map(sys, BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 1,
            b_min = 0.3, b_max = 0.4, b_steps = 1,
            max_period = 3,
            precision = 1e-10,
            iterations = 12,
            base_params = [0.1, 0.3],
            lyapunov_enabled = true,
            lyapunov_iterations = 8
        ))

        field = lyapunov_field(map)
        @test field isa LyapunovFieldResult
        @test size(field.exponents) == size(map.periodicity) == (2, 2)
        @test field.param_names == (:a, :b)
        @test field.neutral_tolerance == 1e-3

        @test !isnothing(plot_lyapunov_field(field))
        @test !isnothing(plot_lyapunov_field(map))
    end

    @testset "Direct Lyapunov fields compute the 2D exponent grid without a map pass" begin
        sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:a, :b], "Linear multiplier field")
        field = lyapunov_field(sys, BifurcationMapConfig(
            a_min = 0.5, a_max = 1.5, a_steps = 2,
            b_min = 0.0, b_max = 0.1, b_steps = 1,
            a_index = 1,
            b_index = 2,
            base_params = [0.5, 0.0],
            lyapunov_iterations = 24,
            lyapunov_transient = 0,
            lyapunov_neutral_tolerance = 1e-6
        ); initial_point=[1.0])

        @test field isa LyapunovFieldResult
        @test field.a_grid == [0.5, 1.0, 1.5]
        @test field.b_grid == [0.0, 0.1]
        @test size(field.exponents) == (3, 2)
        @test all(j -> isapprox(field.exponents[1, j], log(0.5); atol=1e-8), axes(field.exponents, 2))
        @test all(j -> isapprox(field.exponents[2, j], 0.0; atol=1e-8), axes(field.exponents, 2))
        @test all(j -> isapprox(field.exponents[3, j], log(1.5); atol=5e-3), axes(field.exponents, 2))
        @test all(field.sample_counts .== 24)
        @test all(field.classification_status_codes[1, :] .== DynamicsKit._map_lyapunov_status_code(:periodic))
        @test all(field.classification_status_codes[2, :] .== DynamicsKit._map_lyapunov_status_code(:quasiperiodic_neutral_candidate))
        @test all(field.classification_status_codes[3, :] .== DynamicsKit._map_lyapunov_status_code(:chaotic_candidate))
    end

    @testset "Lyapunov parameter construction respects full system parameter vectors" begin
        discrete = DiscreteMap((x, p) -> SVector(p[1] * x[1] + p[2]), 1, [:a, :b], "Two-parameter map")
        discrete_params = DynamicsKit._lyapunov_params(
            discrete,
            LyapunovConfig(param_min=0.0, param_max=1.0, param_steps=1, param_index=1),
            0.5
        )
        @test discrete_params == [0.5, 0.0]

        continuous = ContinuousODE(
            (du, u, p, t) -> begin
                du[1] = p[2] * u[1]
            end,
            1,
            PoincareSection((u, t, integrator) -> u[1] - 1.0; direction=:both, projection=[1], template=[1.0]),
            [:a, :b],
            "Two-parameter ode";
            default_initial_state=[1.0],
            default_params=[0.2, 3.0]
        )
        continuous_params = DynamicsKit._lyapunov_params(
            continuous,
            LyapunovConfig(param_min=0.0, param_max=1.0, param_steps=1, param_index=1),
            0.5
        )
        @test continuous_params == [0.5, 3.0]
    end
end
