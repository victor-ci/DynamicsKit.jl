@testset "Brute-force diagrams" begin
    @testset "Atlas config scaffold and orbit helpers" begin
        atlas = AtlasConfig()
        @test atlas.max_period == 4
        @test isempty(atlas.periods)
        @test atlas.threaded isa Bool
        @test atlas.cache_enabled

        sys = henon_map()
        sampled = DynamicsKit._sample_discrete_orbit(
            sys,
            [0.3];
            initial_point=[0.1, 0.1],
            iterations=20,
            transient=10
        )
        @test sampled.valid
        @test !sampled.diverged
        @test length(sampled.points) == 10
        @test all(length(point) == 2 for point in sampled.points)

        closure = DynamicsKit._orbit_closure_errors(sampled.points, 3)
        geom = DynamicsKit._orbit_geometry_summary(sampled.points)
        @test length(closure) == 3
        @test all(isfinite, closure)
        @test length(geom.center) == 2
        @test length(geom.span) == 2
        @test all(geom.span .>= 0.0)

        osc = radial_oscillator()
        section_orbit = DynamicsKit._sample_continuous_poincare_orbit(
            osc,
            [0.36];
            initial_point=copy(osc.default_initial_state),
            crossings=8,
            transient=4,
            reltol=1e-8,
            abstol=1e-8
        )
        @test section_orbit.valid
        @test !section_orbit.diverged
        @test length(section_orbit.points) == 8
        @test length(section_orbit.final_point) == 1
        @test DynamicsKit._orbit_closure_errors(section_orbit.points, 2)[1] < 1e-2
    end

    @testset "Hénon map brute force" begin
        sys = henon_map()
        config = BruteForceConfig(
            param_min = 0.0,
            param_max = 1.4,
            param_steps = 50,
            iterations = 200,
            transient = 150
        )

        result = brute_force_diagram(sys, config)

        @test result isa BruteForceResult
        @test result.system_name == "Hénon"
        @test result.param_name == :a
        @test length(result.params) == size(result.points, 1)
        @test size(result.points, 2) == 2
        @test length(result.params) > 0

        # Check parameter range is correct
        @test minimum(result.params) >= 0.0
        @test maximum(result.params) <= 1.4

        # At low a values, the attractor should have bounded x values
        low_a_idx = result.params .< 0.3
        if any(low_a_idx)
            @test all(abs.(result.points[low_a_idx, 1]) .< 5.0)
        end
    end

    @testset "Brute force with custom initial point" begin
        sys = henon_map()
        config = BruteForceConfig(param_min=0.5, param_max=1.0, param_steps=10,
                                  iterations=50, transient=30)
        result = brute_force_diagram(sys, config; initial_point=[0.1, 0.1])
        @test length(result.params) > 0
    end

    @testset "Continuous-time brute force via Poincaré section" begin
        sys = radial_oscillator()
        config = BruteForceConfig(
            param_min = 0.2,
            param_max = 0.5,
            param_steps = 6,
            iterations = 60,
            transient = 30
        )

        result = brute_force_diagram(sys, config)

        @test result isa BruteForceResult
        @test result.system_name == "Radial Oscillator"
        @test size(result.points, 2) == 1
        @test length(result.params) == size(result.points, 1)
        @test length(result.params) > 0
        @test all(result.points[:, 1] .< 0.0)
        @test all(isfinite.(result.points))
    end

    @testset "Linked brute-force parameters co-vary with the primary sweep" begin
        sys = colpitts_dynamic_beta_oscillator()
        config = BruteForceConfig(
            param_min = 35e-9,
            param_max = 45e-9,
            param_steps = 4,
            iterations = 12,
            transient = 6,
            param_index = 1,
            fixed_params = copy(sys.default_params),
            linked_param_indices = [2]
        )

        params = [build_sweep_params(config, value) for value in (35e-9, 40e-9, 45e-9)]
        @test all(isapprox(p[1], p[2]; atol=0, rtol=0) for p in params)

        result = brute_force_diagram(sys, config; initial_point=copy(sys.default_initial_state), reltol=1e-7, abstol=1e-7)
        @test result isa BruteForceResult
        @test result.param_name == :C1
        @test length(result.params) > 0
        @test size(result.points, 2) == 2
    end

    @testset "On-section initial states are warmed off the section" begin
        sys = radial_oscillator()
        warmed = DynamicsKit._warmup_from_section(
            sys,
            [1.0, 0.0],
            [0.3];
            min_crossing_time=1e-6
        )

        @test length(warmed) == 2
        @test abs(sys.section.condition(warmed, 0.0, nothing)) > 1e-10
    end

    @testset "_detect_period boundary and amplitude scaling" begin
        # Period exactly equal to max_period must be detectable when the caller supplies
        # at least max_period + 1 iterates.
        period4_orbit = [SVector(0.0), SVector(0.25), SVector(0.5), SVector(0.75), SVector(0.0)]
        @test DynamicsKit._detect_period(period4_orbit, 4, 1e-8) == 4

        # A shorter orbit (only max_period entries) still detects sub-max periods, but
        # cannot detect period == max_period — the loop bound clamps to length(orbit) - 1.
        period4_short = period4_orbit[1:4]
        @test DynamicsKit._detect_period(period4_short, 4, 1e-8) == 0

        # Atlas-style call (max_period == length(orbit)) is exhaustive within the window
        # and must not index out of bounds. Period 2 within a 4-element window.
        period2_window = [SVector(1.0), SVector(-1.0), SVector(1.0), SVector(-1.0)]
        @test DynamicsKit._detect_period(period2_window, 4, 1e-8) == 2

        # Amplitude-scaled tolerance: a large-amplitude orbit with the same relative
        # closure error is still detected (an absolute-tolerance comparison would
        # reject it: the absolute closure error here is ~1e-4 * 1e6 ≈ 100).
        big_amp = 1e6
        period3_big = [
            SVector(big_amp),
            SVector(2 * big_amp),
            SVector(3 * big_amp),
            SVector(big_amp + 1e-4 * big_amp),  # closes within 1e-4 relative tolerance
        ]
        @test DynamicsKit._detect_period(period3_big, 3, 1e-3) == 3

        # Small-amplitude orbits must NOT false-positive at the loose end: a clearly
        # non-closing small orbit still reports chaotic.
        small_chaotic = [SVector(0.1), SVector(0.2), SVector(-0.3), SVector(0.5)]
        @test DynamicsKit._detect_period(small_chaotic, 3, 1e-6) == 0

        # Per-pair scaling sanity: an orbit that starts near the origin and visits
        # large intermediate amplitudes still detects its true period as long as
        # the closure error is small relative to the comparison pair. Each
        # T-comparison uses max(norm(orbit[1]), norm(orbit[T+1]), 1).
        small_to_large = [
            SVector(1e-3),
            SVector(2.0),
            SVector(-1.5),
            SVector(1e-3 + 1e-4),  # closes to orbit[1] within relative precision
        ]
        @test DynamicsKit._detect_period(small_to_large, 3, 1e-3) == 3

        # Per-pair scaling must NOT false-positive when the two comparison points
        # are at very different magnitudes (e.g. orbit[1] small, orbit[T+1] large)
        # but obviously not close. A naive "scale by max magnitude in the window"
        # rule would inflate the threshold here and mis-classify; per-pair scaling
        # keeps the comparison local to the two endpoints.
        not_closing = [SVector(1e-3), SVector(1.0), SVector(-1.0), SVector(50.0)]
        @test DynamicsKit._detect_period(not_closing, 3, 1e-3) == 0
    end

    @testset "Basins detect period exactly equal to max_period" begin
        # Period-4 rotation: f(x) = SVector(mod(x[1] + 0.25, 1.0), 0.0). Every IC
        # cycles through 4 distinct values in x[1]. With max_period = 4 this must
        # be detected — the orbit window has to cover a full period plus closure.
        rot4 = DiscreteMap(
            (x, p) -> SVector(mod(x[1] + 0.25, 1.0), 0.0),
            2,
            [:p],
            "Period-4 Rotation"
        )
        config = BasinsConfig(
            bif_param = 0.0,
            max_period = 4,
            precision = 1e-8,
            iterations = 50,
            x_min = 0.1, x_max = 0.9, x_steps = 4,
            y_min = 0.0, y_max = 0.0, y_steps = 0
        )
        result = basins_of_attraction(rot4, config)
        @test all(result.periodicity .== 4)
    end
end
