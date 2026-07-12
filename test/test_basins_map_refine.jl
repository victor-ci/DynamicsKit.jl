@testset "Basins of attraction" begin
    @testset "Hénon basins at a=1.0" begin
        sys = henon_map()
        config = BasinsConfig(
            bif_param = 1.0,
            max_period = 8,
            precision = 1e-4,
            iterations = 500,
            x_min = -2.0, x_max = 2.0, x_steps = 20,
            y_min = -1.0, y_max = 1.0, y_steps = 20
        )

        result = basins_of_attraction(sys, config)

        @test result isa BasinsResult
        @test result.system_name == "Hénon"
        @test result.bif_param == 1.0
        @test size(result.periodicity) == (21, 21)
        @test length(result.x_grid) == 21
        @test length(result.y_grid) == 21

        # Should detect some periodic orbits (not all zeros)
        @test any(result.periodicity .> 0)

        # Period values should be in valid range
        @test all(result.periodicity .>= 0)
        @test all(result.periodicity .<= config.max_period)
    end

    @testset "Basins rejects iteration count below max_period + 1" begin
        # If iterations is too small to fill the orbit window, the function would
        # have to iterate beyond what the user requested. Reject explicitly.
        sys = henon_map()
        bad_config = BasinsConfig(
            bif_param = 0.3,
            max_period = 8,
            iterations = 8,  # exactly max_period — one short
            x_min = -1.0, x_max = 1.0, x_steps = 1,
            y_min = -1.0, y_max = 1.0, y_steps = 1
        )
        @test_throws ArgumentError basins_of_attraction(sys, bad_config)
    end

    @testset "Basins at low a (period-1 attractor)" begin
        sys = henon_map()
        config = BasinsConfig(
            bif_param = 0.3,
            max_period = 4,
            precision = 1e-3,
            iterations = 200,
            x_min = -1.0, x_max = 1.5, x_steps = 10,
            y_min = -0.5, y_max = 0.5, y_steps = 10
        )

        result = basins_of_attraction(sys, config)
        # At a=0.3, the system has a stable period-1 attractor
        # Many grid points should converge to period 1
        period1_count = count(result.periodicity .== 1)
        @test period1_count > 0
    end

    @testset "Basins retain slice-plane metadata" begin
        sys = henon_map()
        config = BasinsConfig(
            bif_param = 0.3,
            max_period = 4,
            precision = 1e-3,
            iterations = 30,
            x_min = -1.0, x_max = 1.0, x_steps = 2,
            y_min = -0.5, y_max = 0.5, y_steps = 2,
            x_index = 2,
            y_index = 1,
            ic_template = [0.25, -0.25]
        )
        result = basins_of_attraction(sys, config)
        @test result.x_index == 2
        @test result.y_index == 1
        @test result.ic_template == [0.25, -0.25]
    end

    @testset "Continuous basins — memristive diode bridge multistability" begin
        sys = memristive_diode_bridge()
        solver = select_ode_solver("auto")
        config = BasinsConfig(
            bif_param = 0.0155,
            max_period = 6,
            precision = 1e-3,
            iterations = 120,                       # 113 transient + 7 classification crossings
            x_min = -1.5, x_max = 1.5, x_steps = 14,
            y_min = -1.5, y_max = 1.5, y_steps = 14,
            fixed_params = [0.0155, 6.02e-6, 0.05],
            param_index = 1,
            x_index = 1, y_index = 2,               # grid the (v_C1, v_C2) initial plane
            ic_template = [0.0, 0.0, 0.0]           # i_L(0) = 0
        )
        result = basins_of_attraction(sys, config; solver=solver, reltol=1e-8, abstol=1e-8)
        @test result isa BasinsResult
        @test result.system_name == "Memristive Diode Bridge"
        @test size(result.periodicity) == (15, 15)
        @test all(result.periodicity .>= 0)
        @test all(result.periodicity .<= config.max_period)
        # At a = 0.0155 period-1 and period-3 limit cycles coexist (Xu et al.
        # Fig. 8): the IC grid must contain BOTH a period-1 basin and a
        # higher-period (period-3) basin. Asserting a period > 1 explicitly is
        # stronger than unique ≥ 2, which {0, 1} (period-1 + unclassified) could
        # satisfy without a genuine second periodic attractor.
        @test 1 in result.periodicity
        @test any(p -> p > 1, result.periodicity)
    end

    @testset "Basins ic_template / grid-index validation" begin
        sys = henon_map()  # state dim 2
        base = (bif_param=0.3, max_period=4, iterations=20,
                x_min=-1.0, x_max=1.0, x_steps=2, y_min=-1.0, y_max=1.0, y_steps=2)
        # ic_template length must match the state dimension.
        @test_throws ArgumentError basins_of_attraction(sys, BasinsConfig(; base..., ic_template=[0.0]))
        # grid index must be within 1:dim.
        @test_throws ArgumentError basins_of_attraction(sys, BasinsConfig(; base..., x_index=3))
    end
end

@testset "Bifurcation map (2D)" begin
    @testset "Linked map parameter injection" begin
        config = BifurcationMapConfig(
            a_min = 0.0, a_max = 1.0,
            b_min = 0.0, b_max = 1.0,
            a_index = 1, b_index = 3,
            a_linked_param_indices = [2],
            b_linked_param_indices = [4],
            base_params = [10.0, 20.0, 30.0, 40.0]
        )

        p = DynamicsKit._build_map_params(config, 0.25, 0.75)
        @test p == [0.25, 0.25, 0.75, 0.75]

        empty_link_config = BifurcationMapConfig(
            a_min = 0.0, a_max = 1.0,
            b_min = 0.0, b_max = 1.0,
            a_index = 1, b_index = 2,
            base_params = [10.0, 20.0, 30.0]
        )
        @test DynamicsKit._build_map_params(empty_link_config, 1.5, 2.5) == [1.5, 2.5, 30.0]

        padded_config = BifurcationMapConfig(
            a_min = 0.0, a_max = 1.0,
            b_min = 0.0, b_max = 1.0,
            a_index = 1, b_index = 3,
            base_params = [10.0]
        )
        @test DynamicsKit._build_map_params(padded_config, 1.5, 2.5) == [1.5, 0.0, 2.5]
    end

    @testset "Map parameter buffers reset base values between cells" begin
        config = BifurcationMapConfig(
            a_min = 0.0, a_max = 1.0,
            b_min = 0.0, b_max = 1.0,
            a_index = 1, b_index = 2,
            base_params = [10.0, 20.0, 30.0]
        )
        template = DynamicsKit._map_param_template(config)
        buffer = fill(-1.0, length(template))
        a_indices = DynamicsKit._map_a_write_indices(config)
        b_indices = DynamicsKit._map_b_write_indices(config)

        params = DynamicsKit._map_params_from_buffer!(buffer, template, a_indices, b_indices, 0.25, 0.75)
        @test params === buffer
        @test params == [0.25, 0.75, 30.0]

        buffer[3] = 999.0
        params = DynamicsKit._map_params_from_buffer!(buffer, template, a_indices, b_indices, 0.5, 1.5)
        @test params === buffer
        @test params == [0.5, 1.5, 30.0]
    end

    @testset "Bifurcation map rejects iteration count below max_period + 1" begin
        # Same invariant for both discrete and continuous bifurcation maps.
        sys = DiscreteMap(
            (x, p) -> SVector(1 + x[2] - p[1]*x[1]^2, p[2]*x[1]),
            2, [:a, :b], "Tight Hénon"
        )
        bad_config = BifurcationMapConfig(
            a_min = 0.5, a_max = 1.0, a_steps = 1,
            b_min = 0.1, b_max = 0.3, b_steps = 1,
            max_period = 5,
            iterations = 5,  # one short
            base_params = [1.0, 0.3]
        )
        @test_throws ArgumentError bifurcation_map(sys, bad_config)
    end

    @testset "Bifurcation map optional performance controls" begin
        stable_sys = DiscreteMap(
            (x, p) -> SVector(p[1] + p[2]),
            1, [:a, :b], "Constant 2P"
        )
        base_config = BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 2,
            b_min = 0.3, b_max = 0.4, b_steps = 2,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-10,
            iterations = 20,
            base_params = [0.1, 0.3]
        )
        fixed_seed = bifurcation_map(stable_sys, base_config)
        reuse_config = BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 2,
            b_min = 0.3, b_max = 0.4, b_steps = 2,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-10,
            iterations = 20,
            base_params = [0.1, 0.3],
            reuse_neighbor_seeds = true
        )
        reused_seed = bifurcation_map(stable_sys, reuse_config)
        @test fixed_seed.periodicity == reused_seed.periodicity
        @test all(fixed_seed.periodicity .== 1)

        accelerated_config = BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 2,
            b_min = 0.3, b_max = 0.4, b_steps = 2,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-10,
            iterations = 20,
            base_params = [0.1, 0.3],
            reuse_neighbor_seeds = true,
            neighbor_transient = 0
        )
        accelerated_seed = bifurcation_map(stable_sys, accelerated_config)
        @test accelerated_seed.periodicity == fixed_seed.periodicity
        @test DynamicsKit._map_seed_mode(base_config) == :fixed
        @test DynamicsKit._map_seed_mode(reuse_config) == :neighbor_full
        @test DynamicsKit._map_seed_mode(accelerated_config) == :neighbor_accelerated
        @test DynamicsKit._map_effective_neighbor_transient(accelerated_config) == 0

        full_equiv_config = BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 1,
            b_min = 0.3, b_max = 0.4, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-10,
            iterations = 20,
            base_params = [0.1, 0.3],
            reuse_neighbor_seeds = true,
            neighbor_transient = 18
        )
        @test DynamicsKit._map_seed_mode(full_equiv_config) == :neighbor_full
        @test_throws AssertionError BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 1,
            b_min = 0.3, b_max = 0.4, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 4,
            iterations = 20,
            base_params = [0.1, 0.3],
            reuse_neighbor_seeds = true,
            neighbor_transient = -1
        )
        @test_throws AssertionError BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 1,
            b_min = 0.3, b_max = 0.4, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 4,
            iterations = 20,
            base_params = [0.1, 0.3],
            reuse_neighbor_seeds = true,
            neighbor_tile_size_a = -1
        )

        slow_target_map = DiscreteMap(
            (x, p) -> begin
                target = p[2] < 0.5 ? 1.0 : 1.4
                SVector(0.5 * x[1] + 0.5 * target)
            end,
            1, [:a, :b], "Slow target map"
        )
        slow_full = bifurcation_map(slow_target_map, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 1.0, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 1,
            precision = 1e-3,
            iterations = 10,
            base_params = [0.0, 0.0],
            reuse_neighbor_seeds = true
        ))
        slow_accelerated = bifurcation_map(slow_target_map, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 1.0, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 1,
            precision = 1e-3,
            iterations = 10,
            base_params = [0.0, 0.0],
            reuse_neighbor_seeds = true,
            neighbor_transient = 0
        ))
        @test vec(slow_full.periodicity) == [1, 1]
        @test vec(slow_accelerated.periodicity) == [1, 0]

        slow_tiled_result, slow_tiled_diag = DynamicsKit._bifurcation_map(
            slow_target_map,
            BifurcationMapConfig(
                a_min = 0.0, a_max = 0.0, a_steps = 0,
                b_min = 0.0, b_max = 1.0, b_steps = 1,
                a_index = 1, b_index = 2,
                max_period = 1,
                precision = 1e-3,
                iterations = 10,
                base_params = [0.0, 0.0],
                reuse_neighbor_seeds = true,
                neighbor_transient = 0,
                neighbor_tile_size_b = 1
            )
        )
        @test vec(slow_tiled_result.periodicity) == [1, 1]
        @test slow_tiled_diag["tileCount"] == 2
        @test slow_tiled_diag["tileSizeA"] == 1
        @test slow_tiled_diag["tileSizeB"] == 1
        @test slow_tiled_diag["serial"] == false
        @test slow_tiled_diag["semantics"] == "path_following_reduced_transient"
        @test slow_tiled_diag["traversalDependent"] == true
        @test length(slow_tiled_diag["tileDiagnostics"]) == 2
        @test slow_tiled_diag["tileDiagnostics"][1]["bStart"] == 1
        @test slow_tiled_diag["tileDiagnostics"][2]["bStart"] == 2

        reset_map = DiscreteMap(
            (x, p) -> begin
                b = p[2]
                if b < 0.25 || b > 0.75
                    return SVector(0.5 * x[1] + 0.5)
                end
                return SVector(10.0 * x[1] + 10.0)
            end,
            1, [:a, :b], "Reset map"
        )
        reset_result = bifurcation_map(reset_map, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 1.0, b_steps = 2,
            a_index = 1, b_index = 2,
            max_period = 1,
            precision = 1e-3,
            iterations = 10,
            base_params = [0.0, 0.0],
            divergence_cutoff = 25.0,
            reuse_neighbor_seeds = true,
            neighbor_transient = 0
        ))
        @test vec(reset_result.periodicity) == [1, 0, 1]

        exploding_sys = DiscreteMap(
            (x, p) -> SVector(10.0 * x[1] + 10.0 + p[1] + p[2]),
            1, [:a, :b], "Exploding 2P"
        )
        cutoff_result = bifurcation_map(exploding_sys, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.1, a_steps = 1,
            b_min = 0.0, b_max = 0.1, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 4,
            iterations = 20,
            base_params = [0.0, 0.0],
            divergence_cutoff = 50.0
        ))
        @test size(cutoff_result.periodicity) == (2, 2)
        @test all(cutoff_result.periodicity .== 0)
    end

    @testset "Bifurcation map classification diagnostics" begin
        periodic_map = DiscreteMap(
            (x, p) -> SVector(p[1] + p[2]),
            1, [:a, :b], "Diagnostic period-1 map"
        )
        periodic_result, periodic_diag = DynamicsKit._bifurcation_map(periodic_map, BifurcationMapConfig(
            a_min = 0.1, a_max = 0.2, a_steps = 1,
            b_min = 0.3, b_max = 0.4, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-10,
            iterations = 8,
            base_params = [0.1, 0.3]
        ))
        periodic_status = periodic_diag["status"]
        @test all(periodic_result.periodicity .== 1)
        @test get(periodic_status["statusCounts"], "periodic", 0) == 4
        @test all(periodic_status["statusCodes"] .== DynamicsKit._map_status_code(:periodic))
        @test all(periodic_status["closureCandidatePeriods"] .== 1)
        @test all(periodic_status["observedPoints"] .>= 2)
        @test all((0.0 .<= periodic_status["closureConfidence"]) .& (periodic_status["closureConfidence"] .<= 1.0))
        @test periodic_status["minClosureError"] == 0.0

        exploding_map = DiscreteMap(
            (x, p) -> SVector(10.0 * x[1] + 10.0 + p[1] + p[2]),
            1, [:a, :b], "Diagnostic diverging map"
        )
        diverged_result, diverged_diag = DynamicsKit._bifurcation_map(exploding_map, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.1, a_steps = 1,
            b_min = 0.0, b_max = 0.1, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 4,
            iterations = 8,
            base_params = [0.0, 0.0],
            divergence_cutoff = 50.0
        ))
        diverged_status = diverged_diag["status"]
        @test all(diverged_result.periodicity .== 0)
        @test get(diverged_status["statusCounts"], "diverged", 0) == 4
        @test all(diverged_status["statusCodes"] .== DynamicsKit._map_status_code(:diverged))

        circle_map = DiscreteMap(
            (x, p) -> SVector(mod(x[1] + p[1] + p[2], 1.0)),
            1, [:rotation, :offset], "Diagnostic bounded nonclosing map"
        )
        high_period_result, high_period_diag = DynamicsKit._bifurcation_map(circle_map, BifurcationMapConfig(
            a_min = 0.123456, a_max = 0.123456, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-12,
            iterations = 12,
            base_params = [0.123456, 0.0]
        ))
        high_period_status = high_period_diag["status"]
        @test high_period_result.periodicity[1, 1] == 0
        @test get(high_period_status["statusCounts"], "aperiodic_or_high_period", 0) == 1
        @test high_period_status["statusCodes"][1, 1] == DynamicsKit._map_status_code(:aperiodic_or_high_period)
        @test 1 <= high_period_status["closureCandidatePeriods"][1, 1] <= 4
        @test isfinite(high_period_status["closureErrors"][1, 1])
        @test high_period_status["closureConfidence"][1, 1] == 0.0

        invalid_map = DiscreteMap(
            (_x, _p) -> SVector(NaN),
            1, [:a, :b], "Diagnostic invalid-state map"
        )
        invalid_result, invalid_diag = DynamicsKit._bifurcation_map(invalid_map, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 2,
            iterations = 3,
            base_params = [0.0, 0.0]
        ))
        invalid_status = invalid_diag["status"]
        @test invalid_result.periodicity[1, 1] == 0
        @test get(invalid_status["statusCounts"], "invalid_state", 0) == 1
        @test invalid_status["statusCodes"][1, 1] == DynamicsKit._map_status_code(:invalid_state)

        no_crossing = ContinuousODE(
            (du, _u, _p, _t) -> (du[1] = 0.0),
            1,
            PoincareSection((u, _t, _integrator) -> u[1] - 1.0; direction=:up, projection=[1], template=[0.0]),
            [:a, :b],
            "Diagnostic no-crossing flow";
            tspan_hint=1e-3,
            default_initial_state=[0.0],
            default_params=[0.0, 0.0]
        )
        insufficient_result, insufficient_diag = DynamicsKit._bifurcation_map(no_crossing, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 2,
            iterations = 3,
            base_params = [0.0, 0.0]
        ))
        insufficient_status = insufficient_diag["status"]
        @test insufficient_result.periodicity[1, 1] == 0
        @test get(insufficient_status["statusCounts"], "insufficient_crossings", 0) == 1
        @test insufficient_status["statusCodes"][1, 1] == DynamicsKit._map_status_code(:insufficient_crossings)
        @test get(insufficient_diag["crossing"]["terminationCounts"], "insufficient_crossings", 0) == 1

        failing_flow = ContinuousODE(
            (_du, _u, _p, _t) -> error("intentional integration failure"),
            1,
            PoincareSection((u, _t, _integrator) -> u[1] - 1.0; direction=:up, projection=[1], template=[0.0]),
            [:a, :b],
            "Diagnostic failing flow";
            tspan_hint=1e-3,
            default_initial_state=[0.0],
            default_params=[0.0, 0.0]
        )
        failed_result, failed_diag = DynamicsKit._bifurcation_map(failing_flow, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 2,
            iterations = 3,
            base_params = [0.0, 0.0]
        ))
        failed_status = failed_diag["status"]
        @test failed_result.periodicity[1, 1] == 0
        @test get(failed_status["statusCounts"], "integration_failed", 0) == 1
        @test failed_status["statusCodes"][1, 1] == DynamicsKit._map_status_code(:integration_failed)
        @test get(failed_diag["crossing"]["terminationCounts"], "integration_failed", 0) == 1
    end

    @testset "Bifurcation map crossing summary (populated periodic cell)" begin
        # Harmonic oscillator: u = (cos t, -sin t) from [1, 0], so every upward
        # crossing of the y = 0 section lands at x = -1. The projected (x) return map
        # has a fixed point, giving a clean period-1 orbit that terminates early (well
        # before the max-crossing budget), exercising the populated-cell branch of the
        # crossing-summary storage/recording/summarisation introduced for 2D maps.
        oscillator = ContinuousODE(
            (du, u, _p, _t) -> (du[1] = u[2]; du[2] = -u[1]),
            2,
            PoincareSection((u, _t, _integrator) -> u[2]; direction=:up, projection=[1], template=[0.0, 0.0]),
            [:a, :b],
            "Diagnostic harmonic oscillator";
            tspan_hint=10.0,
            default_initial_state=[1.0, 0.0],
            default_params=[0.0, 0.0]
        )
        result, diag = DynamicsKit._bifurcation_map(oscillator, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-3,
            iterations = 8,
            base_params = [0.0, 0.0]
        ))
        period = result.periodicity[1, 1]
        @test period == 1

        crossing = diag["crossing"]
        @test crossing["sampleCount"] == 1
        # A period-p orbit is detected at its (p+1)-th post-transient crossing.
        @test crossing["crossingsFound"][1, 1] == period + 1
        @test crossing["crossingsRequested"][1, 1] == 4 + 1
        @test crossing["totalCrossingsFound"][1, 1] >= crossing["crossingsFound"][1, 1]
        @test isfinite(crossing["finalTimes"][1, 1]) && crossing["finalTimes"][1, 1] > 0
        @test get(crossing["terminationCounts"], "period_detected", 0) == 1
        @test sum(values(crossing["solverRetcodeCounts"])) == 1
        @test get(crossing["solverRetcodeCounts"], "Terminated", 0) == 1
        @test crossing["stateCallbackCount"] == 0
        @test crossing["divergenceCallbackCount"] == 0
    end

    @testset "Bifurcation map adaptive boundary refinement" begin
        boundary_map = DiscreteMap(
            (x, p) -> p[1] < 0.5 ? SVector(0.0) : SVector(-x[1]),
            1,
            [:a, :b],
            "Adaptive boundary map"
        )
        cfg = BifurcationMapConfig(
            a_min = 0.0, a_max = 1.0, a_steps = 1,
            b_min = 0.0, b_max = 1.0, b_steps = 1,
            a_index = 1, b_index = 2,
            max_period = 2,
            precision = 1e-10,
            iterations = 3,
            base_params = [0.0, 0.0],
            adaptive_refinement_enabled = true,
            adaptive_refinement_max_depth = 1,
            adaptive_refinement_budget = 5
        )
        result, diagnostics = DynamicsKit._bifurcation_map(boundary_map, cfg; initial_point=[1.0])
        adaptive = diagnostics["adaptiveRefinement"]

        @test size(result.periodicity) == (2, 2)
        @test result.periodicity[1, 1] == 1
        @test result.periodicity[2, 1] == 2
        @test adaptive["enabled"] == true
        @test adaptive["baseCellCount"] == 1
        @test adaptive["flaggedBaseCells"] == 1
        @test adaptive["sampleCount"] == 5
        @test adaptive["refinedCellCount"] >= 1
        @test !adaptive["budgetExhausted"]
        @test any(point -> point["a"] == 0.5 && point["period"] == 2, adaptive["points"])
        @test any(cell -> "period" in cell["reasons"], adaptive["cells"])

        @test_throws AssertionError BifurcationMapConfig(
            a_min = 0.0, a_max = 1.0,
            b_min = 0.0, b_max = 1.0,
            reuse_neighbor_seeds = true,
            adaptive_refinement_enabled = true
        )
    end

    @testset "Bifurcation map Lyapunov diagnostics" begin
        logistic_map = DiscreteMap(
            (x, p) -> SVector(p[1] * x[1] * (1.0 - x[1])),
            1,
            [:r, :offset],
            "Logistic Lyapunov map"
        )
        result, diagnostics = DynamicsKit._bifurcation_map(logistic_map, BifurcationMapConfig(
            a_min = 3.2, a_max = 4.0, a_steps = 1,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 8,
            precision = 1e-8,
            iterations = 220,
            base_params = [3.2, 0.0],
            lyapunov_enabled = true,
            lyapunov_iterations = 80,
            lyapunov_transient = 10,
            lyapunov_neutral_tolerance = 1e-2
        ); initial_point=[0.2])
        lyapunov = diagnostics["lyapunov"]

        @test lyapunov["enabled"] == true
        @test lyapunov["method"] == "two_trajectory_discrete_map"
        @test size(lyapunov["exponents"]) == size(result.periodicity)
        @test result.periodicity[1, 1] > 0
        @test result.periodicity[2, 1] == 0
        @test lyapunov["statusCodes"][1, 1] == DynamicsKit._map_lyapunov_status_code(:periodic)
        @test lyapunov["statusCodes"][2, 1] == DynamicsKit._map_lyapunov_status_code(:chaotic_candidate)
        @test lyapunov["exponents"][2, 1] > 0.3
        @test get(lyapunov["statusCounts"], "chaotic_candidate", 0) == 1
        @test all(lyapunov["sampleCounts"] .== 80)

        default_result, default_diagnostics = DynamicsKit._bifurcation_map(logistic_map, BifurcationMapConfig(
            a_min = 4.0, a_max = 4.0, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-8,
            iterations = 80,
            base_params = [4.0, 0.0]
        ); initial_point=[0.2])
        @test default_result.periodicity[1, 1] == 0
        @test !haskey(default_diagnostics, "lyapunov")
    end

    @testset "Lyapunov classification of collapsed and contracting cells" begin
        cls = DynamicsKit._map_lyapunov_classification
        aperiodic = (period = 0, status = :aperiodic_or_high_period)

        # A collapsed trajectory pair returns the sentinel exponent -Inf; it is the
        # extreme of a confidently negative exponent (regular/contracting regime) and
        # must not fall through to :unresolved on the isfinite check.
        @test cls(aperiodic, -Inf, :collapsed, 1e-2) == :periodic
        # Confidently negative finite exponent ⟹ regular/periodic-like, not :unresolved.
        @test cls(aperiodic, -0.5, :ok, 1e-2) == :periodic
        # Positive and near-zero exponents classify as before.
        @test cls(aperiodic, 0.4, :ok, 1e-2) == :chaotic_candidate
        @test cls(aperiodic, 0.0, :ok, 1e-2) == :quasiperiodic_neutral_candidate
        # A detector-resolved finite period short-circuits to :periodic.
        @test cls((period = 3, status = :periodic), NaN, :ok, 1e-2) == :periodic
        # A non-aperiodic detection without a usable estimate stays :unresolved.
        @test cls((period = 0, status = :diverged), NaN, :diverged, 1e-2) == :unresolved
    end

    @testset "Bifurcation map switching-event diagnostics" begin
        border_map = DiscreteMap(
            (x, p) -> SVector(p[1] + p[2]),
            1,
            [:a, :b],
            "Border diagnostic map";
            switching_events=[SwitchingEvent("zero-state-border", (x, p) -> x[1]; tolerance=1e-12)]
        )
        _, diagnostics = DynamicsKit._bifurcation_map(border_map, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.1, a_steps = 1,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 2,
            precision = 1e-10,
            iterations = 5,
            base_params = [0.0, 0.0]
        ))
        switching = diagnostics["switching"]

        @test switching["populatedCells"] == 2
        @test switching["nearEventCells"] == 1
        @test switching["nearEventCounts"]["zero-state-border"] == 1
        @test switching["nearestEvents"][1, 1] == "zero-state-border"
        @test switching["minNormalizedDistances"][1, 1] == 0.0
    end

    @testset "Bifurcation map multistability diagnostics" begin
        multistable_map = DiscreteMap(
            (x, p) -> x[1] < 0.0 ? SVector(-1.0) : (x[1] < 1.5 ? SVector(2.0) : SVector(1.0)),
            1,
            [:a, :b],
            "Two-attractor test map"
        )
        result, diagnostics = DynamicsKit._bifurcation_map(multistable_map, BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            a_index = 1, b_index = 2,
            max_period = 3,
            precision = 1e-10,
            iterations = 6,
            base_params = [0.0, 0.0],
            multistability_initial_points = [[-1.0]]
        ))
        multistability = diagnostics["multistability"]

        @test result.periodicity[1, 1] == 2
        @test multistability["enabled"] == true
        @test multistability["seedCount"] == 2
        @test multistability["attractorCounts"][1, 1] == 2
        @test multistability["coexistenceFlags"][1, 1] == true
        @test multistability["periodSets"][1, 1] == [1, 2]
        @test multistability["periodFractions"][1, 1]["1"] == 0.5
        @test multistability["periodFractions"][1, 1]["2"] == 0.5
        @test multistability["normalizedBasinEntropy"][1, 1] ≈ 1.0

        @test_throws AssertionError BifurcationMapConfig(
            a_min = 0.0, a_max = 0.0, a_steps = 0,
            b_min = 0.0, b_max = 0.0, b_steps = 0,
            reuse_neighbor_seeds = true,
            multistability_initial_points = [[-1.0]]
        )
    end

    @testset "Hénon 2-param map" begin
        # Hénon map with both a and b as parameters
        sys_2p = DiscreteMap(
            (x, p) -> SVector(1 + x[2] - p[1]*x[1]^2, p[2]*x[1]),
            2, [:a, :b], "Hénon 2P"
        )

        config = BifurcationMapConfig(
            a_min = 0.5, a_max = 1.4, a_steps = 15,
            b_min = 0.1, b_max = 0.5, b_steps = 10,
            a_index = 1, b_index = 2,
            max_period = 8,
            precision = 1e-4,
            iterations = 500,
            base_params = [1.0, 0.3]
        )

        result = bifurcation_map(sys_2p, config)

        @test result isa BifurcationMapResult
        @test result.system_name == "Hénon 2P"
        @test result.param_names == (:a, :b)
        @test size(result.periodicity) == (16, 11)

        # Should detect various periodicities
        @test any(result.periodicity .> 0)
        @test all(result.periodicity .>= 0)
    end

    @testset "Continuous 2-param map via Poincaré section" begin
        function radial_two_param_oscillator()
            function f!(du, u, p, t)
                μ = p[1]
                ω = p[2]
                r2 = u[1]^2 + u[2]^2
                du[1] = ω * u[2] + u[1] * (μ - r2)
                du[2] = -ω * u[1] + u[2] * (μ - r2)
                nothing
            end

            section = PoincareSection(
                (u, t, integrator) -> u[2];
                direction=:up,
                projection=[1],
                template=[0.0, 0.0]
            )

            ContinuousODE(
                f!, 2, section, [:μ, :ω], "Radial Oscillator 2P";
                tspan_hint=12.0,
                default_initial_state=[1.0, 0.1],
                default_params=[0.3, 1.0]
            )
        end

        sys = radial_two_param_oscillator()
        config = BifurcationMapConfig(
            a_min = 0.2, a_max = 0.4, a_steps = 4,
            b_min = 0.8, b_max = 1.2, b_steps = 3,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-3,
            iterations = 20,
            base_params = [0.3, 1.0]
        )

        result = bifurcation_map(sys, config; reltol=1e-7, abstol=1e-7)
        reuse_config = BifurcationMapConfig(
            a_min = 0.2, a_max = 0.4, a_steps = 4,
            b_min = 0.8, b_max = 1.2, b_steps = 3,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-3,
            iterations = 20,
            base_params = [0.3, 1.0],
            divergence_cutoff = 10.0,
            reuse_neighbor_seeds = true
        )
        reuse_result = bifurcation_map(sys, reuse_config; reltol=1e-7, abstol=1e-7)
        accelerated_result = bifurcation_map(sys, BifurcationMapConfig(
            a_min = 0.2, a_max = 0.4, a_steps = 4,
            b_min = 0.8, b_max = 1.2, b_steps = 3,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-3,
            iterations = 20,
            base_params = [0.3, 1.0],
            divergence_cutoff = 10.0,
            reuse_neighbor_seeds = true,
            neighbor_transient = 2
        ); reltol=1e-7, abstol=1e-7)
        tiled_result, tiled_diag = DynamicsKit._bifurcation_map(sys, BifurcationMapConfig(
            a_min = 0.2, a_max = 0.4, a_steps = 4,
            b_min = 0.8, b_max = 1.2, b_steps = 3,
            a_index = 1, b_index = 2,
            max_period = 4,
            precision = 1e-3,
            iterations = 20,
            base_params = [0.3, 1.0],
            divergence_cutoff = 10.0,
            reuse_neighbor_seeds = true,
            neighbor_transient = 2,
            neighbor_tile_size_a = 2,
            neighbor_tile_size_b = 2
        ); reltol=1e-7, abstol=1e-7)

        @test result isa BifurcationMapResult
        @test result.system_name == "Radial Oscillator 2P"
        @test result.param_names == (:μ, :ω)
        @test size(result.periodicity) == (5, 4)
        @test all(result.periodicity .>= 0)
        @test any(result.periodicity .== 1)
        @test reuse_result isa BifurcationMapResult
        @test size(reuse_result.periodicity) == size(result.periodicity)
        @test all(reuse_result.periodicity .>= 0)
        @test accelerated_result.periodicity == result.periodicity
        @test tiled_result.periodicity == result.periodicity
        @test tiled_diag["seedMode"] == "neighbor_accelerated"
        @test tiled_diag["tileCount"] == 6
        @test tiled_diag["tileSizeA"] == 2
        @test tiled_diag["tileSizeB"] == 2
        @test tiled_diag["serial"] == false
    end
end

@testset "Phase portraits" begin
    @testset "Continuous phase portrait result and plot" begin
        sys = radial_oscillator()
        config = PhasePortraitConfig(
            time_stop = 8.0,
            tail_fraction = 0.5,
            poincare_crossings = 3,
            max_saved_points = 32,
            min_crossing_time = 1e-5
        )

        result = phase_portrait(sys, config; reltol=1e-7, abstol=1e-7)

        @test result isa PhasePortraitResult
        @test result.system_name == "Radial Oscillator"
        @test size(result.trajectory, 2) == 2
        @test length(result.t) == size(result.trajectory, 1)
        @test length(result.t) <= 32
        @test all(isfinite, result.trajectory)
        @test size(result.poincare_points, 1) <= 3
        @test size(result.poincare_points, 2) == 2

        portrait_plot = plot_phase_portrait(result)
        @test !isnothing(portrait_plot)

        invalid_axis_error = try
            plot_phase_portrait(result; x_index=3, y_index=1)
            nothing
        catch err
            err
        end
        @test invalid_axis_error isa ErrorException
        @test occursin("x_index=3", sprint(showerror, invalid_axis_error))
        @test occursin("y_index=1", sprint(showerror, invalid_axis_error))
        @test occursin("trajectory_dim=2", sprint(showerror, invalid_axis_error))

        map_result = bifurcation_map(
            DiscreteMap((x, p) -> SVector(p[1] - x[1]^2 + 0.1x[2], 0.2x[1] + p[2]), 2, [:a, :b], "Tiny map"),
            BifurcationMapConfig(
                a_min = 0.1, a_max = 0.2, a_steps = 1,
                b_min = 0.1, b_max = 0.2, b_steps = 1,
                max_period = 2,
                iterations = 20,
                base_params = [0.1, 0.1]
            )
        )
        @test !isnothing(plot_bifurcation_map(map_result; xlabel="a", ylabel="b", title="Tiny map"))
    end
end

@testset "Branch refinement" begin
    @testset "Refine Hénon period-1 branch" begin
        sys = henon_map()

        # First generate the original branch
        a0 = 0.3
        x_star = (-0.7 + sqrt(0.49 + 4*a0)) / (2*a0)
        y_star = 0.3 * x_star

        orig_config = ContinuationConfig(
            p_min = 0.0, p_max = 1.4,
            ds = 0.05, dsmax = 0.1,  # deliberately coarse
            max_steps = 200,
            newton_tol = 1e-10,
            detect_bifurcation = 2
        )

        original = continuation_branch(sys, orig_config;
                                       initial_point = [x_star, y_star],
                                       params = [a0])

        orig_points = length(original.branch)
        @test orig_points > 5

        # Now refine a specific interval
        ref_config = RefinementConfig(
            from_param = 0.7, to_param = 0.8,
            ds = 0.001, dsmax = 0.005,
            max_steps = 500,
            detect_bifurcation = 3
        )

        refined = refine_branch(sys, original, ref_config; params = [a0])

        @test refined isa BranchResult
        @test refined.period == 1
        # Refined branch should have more detail in the interval
        ref_pars = [pt.param for pt in refined.branch.branch]
        @test minimum(ref_pars) >= 0.69  # approximately from_param
        @test maximum(ref_pars) <= 0.81  # approximately to_param
    end

    @testset "Refinement seed: coverage-aware gap selection" begin
        # Build a synthetic branch with deliberately uneven coverage inside the
        # refine window. Points cluster densely in [0.10, 0.30] and there is a
        # big empty gap from 0.30 to 0.90. A plain "closest to midpoint" rule
        # would pick a seed at ~0.50 (the midpoint of [0.0, 1.0]) — but the only
        # candidates in-window cluster around 0.10–0.30 and at 0.95, so the
        # midpoint rule lands at 0.30. The coverage-aware rule should instead
        # target the centre of the big gap (~0.60) and pick the candidate
        # closest to that, which is 0.95 — the under-covered side.
        dense_cluster = [0.10, 0.15, 0.20, 0.25, 0.30]
        far_point = 0.95
        points = Any[(param=p, x1=Float64(i)) for (i, p) in enumerate(vcat(dense_cluster, [far_point]))]
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(points, Any[]),
            1,
            "synthetic",
            :p,
            DynamicsKit.now()
        )

        seed_idx = DynamicsKit._refinement_seed_index(branch, 0.0, 1.0)
        seed_param = points[seed_idx].param
        # Must NOT pick the midpoint-closest cluster point (0.30); must pick
        # the under-covered side. Anything > 0.5 is in the gap region.
        @test seed_param > 0.5

        # Single-point fallback: when only one branch point lies in the window,
        # there is no gap structure, and the midpoint heuristic applies. Verify
        # it returns that single in-window point.
        sparse_points = Any[(param=0.05, x1=1.0), (param=0.40, x1=2.0), (param=0.99, x1=3.0)]
        sparse_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(sparse_points, Any[]),
            1,
            "synthetic",
            :p,
            DynamicsKit.now()
        )
        @test DynamicsKit._refinement_seed_index(sparse_branch, 0.3, 0.5) == 2

        # Direction-insensitive: swapping from/to must give the same seed.
        seed_a = DynamicsKit._refinement_seed_index(branch, 0.0, 1.0)
        seed_b = DynamicsKit._refinement_seed_index(branch, 1.0, 0.0)
        @test seed_a == seed_b
    end

    @testset "Branch period trimming (Pᴺ fixed-point runaway)" begin
        # A period-N orbit is also a fixed point of Pᴺ at every divisor period, so
        # continuing/extending a period-N branch across a wide interval can follow
        # a lower-period orbit and get mislabelled. _trim_branch_to_period must
        # drop the degenerate stretch.
        sys = boost_converter()
        p12 = [1.2, 10.0, 20.0, 0.0]   # period-1 regime
        p20 = [2.0, 10.0, 20.0, 0.0]   # period-2 regime

        # Settle onto the attractor, stopping once the orbit closes at period N
        # (Pᴺ(x) ≈ x) instead of a fixed, brittle iteration count.
        function settle(p, period; maxiter=8000, tol=1e-9)
            x = SVector(15.0, 0.9)
            for _ in 1:maxiter
                xN = x
                for _ in 1:period
                    xN = sys.f(xN, p)
                end
                norm(xN - x) < tol * max(norm(x), 1.0) && break
                x = sys.f(x, p)
            end
            return Array(x)
        end
        fp1 = settle(p12, 1)           # period-1 fixed point at Iref=1.2
        x2  = settle(p20, 2)           # a point on the period-2 orbit at Iref=2.0

        # Minimal-period detection: a P⁴ check classifies these by their true period.
        @test DynamicsKit._orbit_minimal_period(sys, fp1, p12, 4) == 1
        @test DynamicsKit._orbit_minimal_period(sys, x2, p20, 4) == 2

        # A "period-2"-labelled branch carrying a degenerate period-1 point
        # (Iref=1.2) plus a genuine period-2 point (Iref=2.0) trims to the genuine
        # point only. base_params are overwritten per-point by the point's param.
        pt_bad  = (x1=fp1[1], x2=fp1[2], param=1.2)
        pt_good = (x1=x2[1],  x2=x2[2],  param=2.0)
        br = BranchResult(DynamicsKit.CombinedBranchResult(Any[pt_bad, pt_good], Any[]),
                          2, sys.name, :Iref, DynamicsKit.now())
        trim_diag = Ref{Any}(nothing)
        trimmed = DynamicsKit._trim_branch_to_period(sys, br, p20, Int[];
                                                             trim_diagnostics=trim_diag)
        @test trimmed !== nothing
        kept = DynamicsKit._branch_points(trimmed)
        @test length(kept) == 1
        @test kept[1].param == 2.0
        @test trimmed.period == 2
        @test trim_diag[]["applied"] == true
        @test trim_diag[]["droppedCount"] == 1
        @test trim_diag[]["keptCount"] == 1
        @test trim_diag[]["lowerPeriods"]["1"] == 1

        # All-degenerate branch (no genuine period-N point) trims to nothing.
        only_bad = BranchResult(DynamicsKit.CombinedBranchResult(Any[pt_bad], Any[]),
                                2, sys.name, :Iref, DynamicsKit.now())
        all_bad_diag = Ref{Any}(nothing)
        @test DynamicsKit._trim_branch_to_period(sys, only_bad, p12, Int[];
                                                         trim_diagnostics=all_bad_diag) === nothing
        @test all_bad_diag[]["reason"] == "all_dropped"
        @test all_bad_diag[]["droppedCount"] == 1

        # Period-1 branches are returned unchanged (cannot degenerate further).
        br1 = BranchResult(DynamicsKit.CombinedBranchResult(Any[pt_bad], Any[]),
                           1, sys.name, :Iref, DynamicsKit.now())
        period_one_diag = Ref{Any}(nothing)
        @test DynamicsKit._trim_branch_to_period(sys, br1, p12, Int[];
                                                         trim_diagnostics=period_one_diag) === br1
        @test period_one_diag[]["reason"] == "period_one"
        @test period_one_diag[]["applied"] == false
    end

    @testset "Continuous branch period trimming (Poincaré return map)" begin
        # Same Pᴺ-fixed-point degeneracy, but for a continuous system the minimal
        # period is measured on the Poincaré return map. Rössler: c=2.5 is a
        # period-1 limit cycle, c=3.5 is period-2 (one upward y=0 crossing per
        # cycle, so the return-map period matches).
        sys = rossler_oscillator()
        p25 = [0.2, 0.2, 2.5]
        p35 = [0.2, 0.2, 3.5]

        # Settled section points (projected (x, z)) on each attractor.
        s1pts = DynamicsKit._collect_poincare_points(sys, p25;
                    initial_point=[1.0, 1.0, 1.0], crossings=1, transient=50, projected=true)
        s2pts = DynamicsKit._collect_poincare_points(sys, p35;
                    initial_point=[1.0, 1.0, 1.0], crossings=2, transient=50, projected=true)
        @test !isempty(s1pts)
        @test length(s2pts) == 2
        s1 = collect(Float64, s1pts[end])
        s2 = collect(Float64, s2pts[end])

        # (a) minimal-period classification on the return map.
        @test DynamicsKit._orbit_minimal_period(sys, s1, p25, 4) == 1
        @test DynamicsKit._orbit_minimal_period(sys, s2, p35, 4) == 2

        # (b) a "period-2"-labelled branch with a degenerate period-1 point (c=2.5)
        #     and a genuine period-2 point (c=3.5) trims to the genuine point —
        #     without returning nothing.
        pt_bad  = (x1=s1[1], x2=s1[2], param=2.5)
        pt_good = (x1=s2[1], x2=s2[2], param=3.5)
        br = BranchResult(DynamicsKit.CombinedBranchResult(Any[pt_bad, pt_good], Any[]),
                          2, sys.name, :c, DynamicsKit.now())
        trim_diag = Ref{Any}(nothing)
        trimmed = DynamicsKit._trim_branch_to_period(sys, br, p35, Int[];
                                                             trim_diagnostics=trim_diag)
        @test trimmed !== nothing
        kept = DynamicsKit._branch_points(trimmed)
        @test length(kept) == 1
        @test kept[1].param == 3.5
        @test trim_diag[]["applied"] == true
        @test trim_diag[]["lowerPeriods"]["1"] == 1

        # Missing parameter name ⇒ leave the branch untouched (never trim against a
        # guessed parameter slot).
        wrong = BranchResult(DynamicsKit.CombinedBranchResult(Any[pt_bad, pt_good], Any[]),
                             2, sys.name, :not_a_param, DynamicsKit.now())
        wrong_diag = Ref{Any}(nothing)
        @test DynamicsKit._trim_branch_to_period(sys, wrong, p35, Int[];
                                                         trim_diagnostics=wrong_diag) === wrong
        @test wrong_diag[]["reason"] == "missing_parameter_name"
    end
end
