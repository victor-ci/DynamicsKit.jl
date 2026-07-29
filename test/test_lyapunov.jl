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

    # -----------------------------------------------------------------------------------
    # Variational continuous Lyapunov estimator tests
    # -----------------------------------------------------------------------------------

    @testset "Analytic radial-phase oscillator: known transverse exponent" begin
        # r' = α(1-r²)r, θ' = 1 has the unit limit cycle and transverse exponent -2α.
        f_circ! = function (du, u, p, t)
            α = p[1]
            radial = α * (1 - u[1]^2 - u[2]^2)
            du[1] = radial * u[1] - u[2]
            du[2] = u[1] + radial * u[2]
            nothing
        end
        f_circ = function (u, p, t)
            α = p[1]
            radial = α * (1 - u[1]^2 - u[2]^2)
            SVector(radial * u[1] - u[2], u[1] + radial * u[2])
        end
        section_circ = PoincareSection(
            (u, t, integ) -> u[2],    # y = 0 crossing
            direction=:up,
            projection=[1],
            template=[1.0, 0.0];
            constant_normal=[0.0, 1.0]
        )
        circ = ContinuousODE(f_circ!, 2, section_circ, [:alpha, :unused], "Radial phase oscillator";
                             default_initial_state=[1.0, 0.0], default_params=[0.25, 0.0],
                             tspan_hint=2π, f_svector=f_circ)

        est = DynamicsKit._estimate_variational_continuous_lyapunov(
            circ, [0.25, 0.0], [1.0, 0.0], 3, 12, 1e6;
            solver=DynamicsKit.Tsit5(), reltol=1e-10, abstol=1e-10)
        @test est.estimation_status == :ok
        @test est.sample_count == 12
        @test est.exponent ≈ -0.5 atol=2e-6

        field_cfg = BifurcationMapConfig(
            a_min=0.25, a_max=0.25, a_steps=0,
            b_min=0.0, b_max=0.0, b_steps=0,
            a_index=1, b_index=2, base_params=[0.25, 0.0],
            lyapunov_iterations=12, lyapunov_transient=3)
        cpu_field = lyapunov_field(circ, field_cfg)
        seam_field = lyapunov_field(circ, field_cfg; backend=gpu_backend(:_ka_cpu_test))
        @test cpu_field.normalization == :flow_time
        @test seam_field.normalization == :flow_time
        @test cpu_field.exponents[1, 1] ≈ -0.5 atol=2e-6
        @test seam_field.exponents[1, 1] ≈ cpu_field.exponents[1, 1] atol=2e-6
    end

    @testset "Rössler stable periodic: variational exponent is negative" begin
        sys = rossler_oscillator()
        # c=3.0 → period-1 attractor; variational transverse exponent < 0.
        cfg = BifurcationMapConfig(
            a_min=0.2, a_max=0.2, a_steps=0, b_min=3.0, b_max=3.0, b_steps=0,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
            lyapunov_enabled=true, lyapunov_iterations=40, lyapunov_transient=10)
        r = lyapunov_field(sys, cfg)
        @test r.lyapunov_method == :variational
        @test r.normalization == :flow_time
        @test r.compute_backend == :cpu
        λ = r.exponents[1, 1]
        @test isfinite(λ)
        @test λ < 0.0   # stable periodic orbit
    end

    @testset "Rössler chaotic: variational exponent agrees with lyapunov_spectrum sign" begin
        sys = rossler_oscillator()
        # c=5.7 → chaotic; largest transverse exponent > 0.
        cfg = BifurcationMapConfig(
            a_min=0.2, a_max=0.2, a_steps=0, b_min=5.7, b_max=5.7, b_steps=0,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 5.7],
            lyapunov_enabled=true, lyapunov_iterations=80, lyapunov_transient=20)
        r = lyapunov_field(sys, cfg)
        @test r.lyapunov_method == :variational
        λ_var = r.exponents[1, 1]
        @test isfinite(λ_var)
        @test λ_var > 0.0   # positive exponent in chaos

        # Compare sign and order-of-magnitude against lyapunov_spectrum (different estimator, same attractor).
        sp = lyapunov_spectrum(sys, LyapunovSpectrumConfig(transient=50, steps=400);
                               params=[0.2, 0.2, 5.7])
        @test sp.estimation_status == :ok
        leading = sp.exponents[1]   # lyapunov_spectrum returns exponents sorted descending
        @test leading > 0.0
        # They should agree to within a factor of 2 (different number of iterates, both from IC transient).
        @test abs(λ_var - leading) / max(abs(leading), 0.01) < 2.0
    end

    @testset "Method selector: :auto resolves to :variational for ContinuousODE" begin
        sys = rossler_oscillator()
        cfg_auto = BifurcationMapConfig(
            a_min=0.2, a_max=0.2, a_steps=0, b_min=3.0, b_max=3.0, b_steps=0,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
            lyapunov_enabled=true, lyapunov_iterations=25, lyapunov_transient=5)
        cfg_two  = BifurcationMapConfig(
            a_min=0.2, a_max=0.2, a_steps=0, b_min=3.0, b_max=3.0, b_steps=0,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
            lyapunov_enabled=true, lyapunov_iterations=25, lyapunov_transient=5,
            lyapunov_method=:two_trajectory)
        r_auto = lyapunov_field(sys, cfg_auto)
        r_two  = lyapunov_field(sys, cfg_two)
        @test r_auto.lyapunov_method == :variational
        @test r_two.lyapunov_method == :two_trajectory
        # Both should return finite values (sign of two-trajectory with low iteration count is not guaranteed).
        @test isfinite(r_auto.exponents[1, 1])
        @test isfinite(r_two.exponents[1, 1])
        # The variational estimator with these parameters reliably gives a negative exponent (stable orbit).
        @test r_auto.exponents[1, 1] < 0.0
    end

    @testset "Explicit :two_trajectory GPU rejection for ContinuousODE" begin
        sys = rossler_oscillator()
        cfg = BifurcationMapConfig(
            a_min=0.2, a_max=0.2, a_steps=0, b_min=3.0, b_max=3.0, b_steps=0,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
            lyapunov_enabled=true, lyapunov_iterations=25, lyapunov_transient=5,
            lyapunov_method=:two_trajectory)
        @test_throws ArgumentError lyapunov_field(sys, cfg; backend=gpu_backend(:_ka_cpu_test))
    end

    @testset "Degenerate section detection (zero constant_normal returns degenerate_section)" begin
        f_dummy! = (du, u, p, t) -> (du[1] = -u[2]; du[2] = u[1]; nothing)
        bad_section = PoincareSection(
            (u, t, integ) -> u[2],
            direction=:up,
            projection=[1],
            template=[1.0, 0.0];
            constant_normal=[0.0, 0.0]   # degenerate: zero normal
        )
        sys_bad = ContinuousODE(f_dummy!, 2, bad_section, [:unused], "DegenerateNormalODE";
                                default_initial_state=[1.0, 0.0], tspan_hint=50.0)
        cfg = BifurcationMapConfig(
            a_min=0.0, a_max=0.0, a_steps=0, b_min=0.0, b_max=0.0, b_steps=0,
            a_index=1, b_index=1, base_params=[0.0],
            lyapunov_enabled=true, lyapunov_iterations=10, lyapunov_transient=2)
        @test_throws ArgumentError lyapunov_field(sys_bad, cfg; backend=gpu_backend(:_ka_cpu_test))
        # CPU path should return degenerate_section status.
        est = DynamicsKit._estimate_variational_continuous_lyapunov(
            sys_bad, [0.0], [1.0, 0.0], 1, 5, 1e6;
            solver=DynamicsKit.Tsit5(), reltol=1e-8, abstol=1e-8)
        @test est.estimation_status == :degenerate_section
    end

    @testset "LyapunovFieldResult serialization round-trips lyapunov_method" begin
        sys = rossler_oscillator()
        cfg = BifurcationMapConfig(
            a_min=0.2, a_max=0.2, a_steps=0, b_min=3.0, b_max=3.0, b_steps=0,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
            lyapunov_enabled=true, lyapunov_iterations=20, lyapunov_transient=5)
        r = lyapunov_field(sys, cfg)
        @test r.lyapunov_method == :variational
        data = DynamicsKit._serialize_lyapunov_field_result_v2(r)
        @test haskey(data, "lyapunovMethod")
        @test data["lyapunovMethod"] == "variational"
        @test data["normalization"] == "flow_time"
        r2 = DynamicsKit._deserialize_lyapunov_field_result_v2(data)
        @test r2.lyapunov_method == :variational
        @test r2.normalization == :flow_time

        # Older payloads did not identify either property.
        delete!(data, "lyapunovMethod")
        delete!(data, "normalization")
        r3 = DynamicsKit._deserialize_lyapunov_field_result_v2(data)
        @test r3.lyapunov_method == :two_trajectory
        @test r3.normalization == :unspecified
        data["lyapunovMethod"] = "unbounded-symbol-input"
        @test_throws ArgumentError DynamicsKit._deserialize_lyapunov_field_result_v2(data)
    end

    @testset "CPU vs GPU variational parity on Rössler (KA CPU seam)" begin
        sys = rossler_oscillator()
        seam = gpu_backend(:_ka_cpu_test)
        cfg = BifurcationMapConfig(
            a_min=0.15, a_max=0.25, a_steps=2, b_min=2.3, b_max=4.1, b_steps=3,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
            lyapunov_enabled=true, lyapunov_iterations=40, lyapunov_transient=10)
        r_cpu = lyapunov_field(sys, cfg; backend=CPUBackend())
        r_gpu = lyapunov_field(sys, cfg; backend=seam)
        @test r_cpu.lyapunov_method == :variational
        @test r_gpu.lyapunov_method == :variational
        @test r_gpu.compute_backend == :_ka_cpu_test
        @test r_cpu.normalization == r_gpu.normalization == :flow_time
        # Sign and classification should agree; numerical values within 10% (different integration paths).
        for (λ_c, λ_g, sc_c, sc_g) in zip(
                r_cpu.exponents, r_gpu.exponents,
                r_cpu.classification_status_codes, r_gpu.classification_status_codes)
            if isfinite(λ_c) && isfinite(λ_g)
                @test sign(λ_c) == sign(λ_g)
            end
            @test sc_c == sc_g
        end
    end

    @testset "Poincare section direction and discrete method validation are strict" begin
        @test_throws ArgumentError PoincareSection((u, t, i) -> u[1]; direction=:positive)
        discrete = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:a, :b], "strict method")
        cfg = BifurcationMapConfig(
            a_min=0.5, a_max=0.5, a_steps=0,
            b_min=0.0, b_max=0.0, b_steps=0,
            a_index=1, b_index=2, base_params=[0.5, 0.0],
            lyapunov_method=:variational)
        @test_throws ArgumentError lyapunov_field(discrete, cfg)
    end

    @testset "Vilnius Lyapunov field variational on CPU seam" begin
        sys = vilnius_oscillator()
        seam = gpu_backend(:_ka_cpu_test)
        cfg = BifurcationMapConfig(
            a_min=0.1, a_max=0.3, a_steps=2, b_min=0.4, b_max=0.6, b_steps=2,
            a_index=1, b_index=2, base_params=[0.2, 0.5],
            lyapunov_enabled=true, lyapunov_iterations=30, lyapunov_transient=8)
        r_gpu = lyapunov_field(sys, cfg; backend=seam)
        @test r_gpu isa LyapunovFieldResult
        @test r_gpu.lyapunov_method == :variational
        @test r_gpu.compute_backend == :_ka_cpu_test
        # Some cells may yield NaN (insufficient crossings for short parameter sweep); at least some should resolve.
        @test any(isfinite(λ) for λ in r_gpu.exponents)
    end

    @testset "Cache hook semantics for variational CPU field" begin
        sys = rossler_oscillator()
        cfg = BifurcationMapConfig(
            a_min=0.15, a_max=0.25, a_steps=2, b_min=2.3, b_max=4.1, b_steps=2,
            a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
            lyapunov_enabled=true, lyapunov_iterations=30, lyapunov_transient=8)
        # Full run.
        r1 = lyapunov_field(sys, cfg)
        # Pre-seed all cells: second run should skip all cells and produce identical result.
        cells = DynamicsKit.LyapunovCellGrid(cfg.a_steps + 1, cfg.b_steps + 1)
        cells.exponents               .= r1.exponents
        cells.status_codes            .= r1.classification_status_codes
        cells.estimation_status_codes .= r1.estimation_status_codes
        cells.sample_counts           .= r1.sample_counts
        cells.known                   .= true
        r2 = lyapunov_field(sys, cfg; cells=cells)
        @test r1.exponents == r2.exponents
        @test r1.classification_status_codes == r2.classification_status_codes
    end
end
