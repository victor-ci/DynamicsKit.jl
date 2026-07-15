@testset "Map-aware special points (PD / fold)" begin
    @testset "Hénon: analytic PD and fold, recovering the flip BifurcationKit misses" begin
        sys = henon_map()
        b = 0.3
        config = ContinuationConfig(p_min=-0.25, p_max=0.6, ds=0.01, dsmax=0.03,
                                    max_steps=800, detect_bifurcation=3, param_index=1)
        a0 = 0.3
        x_star = (-(1 - b) + sqrt((1 - b)^2 + 4a0)) / (2a0)
        branch = continuation_branch(sys, config, 1; initial_point=[x_star, b * x_star], params=[a0, b])

        specials = map_special_points(sys, branch, [a0, b])
        pd = filter(s -> s.kind == :pd, specials)
        fold = filter(s -> s.kind == :fold, specials)

        @test length(pd) == 1
        # Analytic period-1 flip of the Hénon map at b = 0.3: a = 3(1-b)²/4 = 0.3675.
        @test pd[1].param ≈ 0.3675 atol=1e-4
        @test real(pd[1].critical_multiplier) ≈ -1.0 atol=1e-3
        @test abs(imag(pd[1].critical_multiplier)) < 1e-6
        @test pd[1].converged
        @test pd[1].period == 1

        @test length(fold) == 1
        # Analytic period-1 fold: a = -(1-b)²/4 = -0.1225.
        @test fold[1].param ≈ -0.1225 atol=1e-4
        @test real(fold[1].critical_multiplier) ≈ 1.0 atol=1e-3

        # The value of F4: BifurcationKit's own special points carry no period-doubling
        # near a = 0.3675 (the equilibrium-convention detector misses map flips).
        bk_specials = collect(branch.branch.specialpoint)
        @test !any(sp -> abs(Float64(sp.param) - 0.3675) < 1e-2, bk_specials)
    end

    @testset "Boost converter: recovers the subharmonic period-doubling BifurcationKit misses" begin
        sys = boost_converter()
        base = [1.5, 10.0, 20.0, 0.0]
        # Settle onto the stable period-1 orbit below the subharmonic threshold.
        attractor = foldl((s, _) -> sys.f(s, base), 1:2000; init=SVector(8.0, 1.0))
        config = ContinuationConfig(p_min=1.2, p_max=1.95, ds=0.005, dsmax=0.01,
                                    max_steps=600, detect_bifurcation=3, param_index=1)
        branch = continuation_branch(sys, config, 1; initial_point=collect(attractor), params=base)

        specials = map_special_points(sys, branch, base; detect=[:pd])
        @test length(specials) == 1
        @test specials[1].kind == :pd
        # Peak-current-mode subharmonic instability at duty ratio 1/2 (μ = -1), near Iref ≈ 1.7 A.
        @test 1.6 < specials[1].param < 1.85
        @test real(specials[1].critical_multiplier) ≈ -1.0 atol=1e-2
        @test specials[1].converged

        bk_specials = collect(branch.branch.specialpoint)
        @test !any(sp -> String(sp.type) == "pd", bk_specials)
    end

    @testset "detect filter and guards" begin
        sys = henon_map()
        config = ContinuationConfig(p_min=0.2, p_max=0.5, ds=0.01, dsmax=0.03,
                                    max_steps=400, param_index=1)
        a0 = 0.3
        x_star = (-(1 - 0.3) + sqrt(0.7^2 + 4a0)) / (2a0)
        branch = continuation_branch(sys, config, 1; initial_point=[x_star, 0.3 * x_star], params=[a0, 0.3])

        only_pd = map_special_points(sys, branch, [a0, 0.3]; detect=[:pd])
        @test all(s -> s.kind == :pd, only_pd)
        @test length(only_pd) == 1                       # fold is outside [0.2, 0.5]

        @test_throws ArgumentError map_special_points(sys, branch, [a0, 0.3]; detect=[:hopf])
    end

    @testset "A branch sample exactly on the flip is emitted without double counting" begin
        # x → a·x has fixed point 0 with multiplier a; the flip (μ = −1) sits exactly at
        # a = −1, and a symmetric sweep places a branch sample there (φ = μ + 1 = 0).
        sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:a], "Linear multiplier")
        config = ContinuationConfig(p_min=-1.6, p_max=-0.4, ds=0.1, dsmax=0.1,
                                    max_steps=100, param_index=1)
        branch = continuation_branch(sys, config, 1; initial_point=[0.0], params=[-1.0])

        specials = map_special_points(sys, branch, [-1.0]; detect=[:pd])
        @test length(specials) == 1                      # emitted once, not once per adjacent bracket
        @test specials[1].param ≈ -1.0 atol=1e-9
        @test real(specials[1].critical_multiplier) ≈ -1.0 atol=1e-9
        @test specials[1].converged
    end
end
