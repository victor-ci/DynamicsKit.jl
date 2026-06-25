@testset "Periodic skeleton" begin
    @testset "Hénon skeleton at a=1.2" begin
        sys = henon_map()

        results = find_periodic_skeleton(sys, [1, 2], 1.2;
                                         n_initial=10,
                                         search_min=[-2.0, -1.0],
                                         search_max=[2.0, 1.0],
                                         params=[1.2],
                                         tol=1e-10)

        @test length(results) > 0

        # Should find at least a period-1 fixed point
        period1 = filter(r -> r.period == 1, results)
        @test length(period1) >= 1

        # Verify found points are actual fixed points
        for r in results
            sv = SVector{2}(r.point)
            for _ in 1:r.period
                sv = sys.f(sv, [1.2])
            end
            @test isapprox(norm(Array(sv) .- r.point), 0.0; atol=1e-8)
        end
    end

    @testset "Hénon skeleton finds multiple periods" begin
        sys = henon_map()

        results = find_periodic_skeleton(sys, 1:4, 1.2;
                                         n_initial=12,
                                         search_min=[-2.0, -1.0],
                                         search_max=[2.0, 1.0],
                                         params=[1.2],
                                         tol=1e-10)

        found_periods = unique([r.period for r in results])
        @test length(found_periods) >= 2  # should find at least period 1 and 2
    end

    @testset "Discrete skeleton accepts seed points" begin
        sys = henon_map()
        a = 0.3
        x_star = (-0.7 + sqrt(0.49 + 4 * a)) / (2 * a)
        y_star = 0.3 * x_star

        seeded_points = DynamicsKit._discrete_skeleton_initial_points(
            [-2.0, -1.0],
            [2.0, 1.0],
            6;
            seed_points=[[x_star, y_star], [x_star + 1e-3, y_star + 1e-3], [5.0, -5.0]]
        )
        @test any(p -> isapprox(p[1], x_star; atol=1e-12) && isapprox(p[2], y_star; atol=1e-12), seeded_points)
        @test any(p -> isapprox(p[1], 2.0; atol=1e-12) && isapprox(p[2], -1.0; atol=1e-12), seeded_points)
        @test count(p -> norm(p .- [x_star, y_star]) < 5e-3, seeded_points) == 1
        @test length(seeded_points) < 36

        results = find_periodic_skeleton(sys, [1], a;
                                         n_initial=2,
                                         search_min=[0.8, 0.1],
                                         search_max=[1.2, 0.5],
                                         seed_points=[[x_star, y_star]],
                                         params=[a],
                                         tol=1e-10)

        @test any(r -> r.period == 1 && norm(r.point .- [x_star, y_star]) < 1e-8, results)
    end

    @testset "Continuous-time skeleton via Poincaré map" begin
        sys = radial_oscillator()

        results = find_periodic_skeleton(sys, [1], 0.36;
                                         n_initial=6,
                                         tol=1e-8,
                                         max_iter=30)

        @test length(results) >= 1
        stable_seeds = filter(seed -> seed.stable, results)
        @test !isempty(stable_seeds)
        seed = first(stable_seeds)
        @test seed.period == 1
        @test isapprox(seed.point[1], -sqrt(0.36); atol=0.15)
        @test seed.stable
    end

    @testset "Internal helpers" begin
        # Test _is_unique
        @test DynamicsKit._is_unique([1.0, 2.0], Vector{Float64}[], 0.1)
        @test DynamicsKit._is_unique([1.0, 2.0], [[3.0, 4.0]], 0.1)
        @test !DynamicsKit._is_unique([1.0, 2.0], [[1.0, 2.0001]], 0.01)

        seeded_points = DynamicsKit._continuous_skeleton_initial_points(
            [-2.0, -1.0],
            [2.0, 1.0],
            6;
            seed_points=[[0.0, 0.0], [0.05, 0.04], [4.0, -4.0]]
        )
        @test any(p -> isapprox(p[1], 0.0; atol=1e-12) && isapprox(p[2], 0.0; atol=1e-12), seeded_points)
        @test count(p -> abs(p[1]) <= 0.1 && abs(p[2]) <= 0.1, seeded_points) == 1
        @test any(p -> isapprox(p[1], 2.0; atol=1e-12) && isapprox(p[2], -1.0; atol=1e-12), seeded_points)
        @test length(seeded_points) < 36

        # Test _is_true_period
        sys = henon_map()
        # A period-1 fixed point should not be "true period 2"
        a = 0.3
        x_star = (-0.7 + sqrt(0.49 + 4 * a)) / (2 * a)
        y_star = 0.3 * x_star
        fp = [x_star, y_star]
        step = x -> Array(sys.f(SVector{2}(x), [a]))
        @test DynamicsKit._is_true_period(step, fp, 1, 1e-8)
        @test !DynamicsKit._is_true_period(step, fp, 2, 1e-6)
    end

    @testset "Phase 2: Newton convergence is amplitude-aware" begin
        threshold = DynamicsKit._newton_convergence_threshold

        # Near the origin: absolute floor (max chooses tol).
        @test threshold([0.0, 0.0], 1e-10) == 1e-10
        @test threshold([0.5, 0.3], 1e-10) == 1e-10

        # Large amplitude: relative scaling kicks in (tol * norm(x) > tol).
        @test threshold([1e6, 0.0], 1e-10) ≈ 1e-10 * 1e6
        @test threshold([1e6, 0.0], 1e-10) > threshold([0.5, 0.3], 1e-10)

        # _newton_ad converges on a large-amplitude root that the absolute-only
        # rule would reject. F(x) = x .- target with target = [1e6, 1e6]; from a
        # nearby seed Newton lands exactly on the root in 1 iteration.
        target = [1e6, 1e6]
        F = x -> x .- target
        x_seed = target .+ [1e-2, 1e-2]
        x, converged = DynamicsKit._newton_ad(F, x_seed, 1e-10, 5)
        @test converged
        @test isapprox(x, target; rtol=1e-12)
    end
end

