# Contract B — analysis kernels. Freezes the public per-cell computation surface the workbench's
# grid-cache engine drives. See docs/internal/contracts/contract-b-analysis-kernels.md.

@testset "Contract B — analysis kernels" begin
    sys = henon_map()                       # DiscreteMap, dim 2
    p_chaos = [1.4, 0.3]                     # classic Hénon chaotic parameters
    x0 = SVector(0.0, 0.0)

    @testset "period detection return shape" begin       # invariant 2
        det = detect_discrete_map_period(sys, p_chaos, x0, 1000, 12, 1e-4, 1e6)
        @test issubset((:period, :status, :closure_confidence, :valid), propertynames(det))
        @test 0.0 <= det.closure_confidence <= 1.0
        @test det.valid == (det.status in (:periodic, :aperiodic_or_high_period))
    end

    @testset "lyapunov return shape + sign + guard" begin # invariant 3
        est = estimate_discrete_map_largest_lyapunov(sys, p_chaos, x0, 1000, 2000, 1e-8, 1e6)
        @test (est.exponent, est.estimation_status, est.sample_count) isa Tuple
        @test est.estimation_status == :ok
        @test isfinite(est.exponent) && est.exponent > 0.1   # Hénon largest LE ≈ 0.42; tolerance avoids near-zero noise
        bad = estimate_discrete_map_largest_lyapunov(sys, p_chaos, x0, 0, 0, 1e-8, 1e6)
        @test bad.estimation_status == :insufficient_samples && bad.sample_count == 0 && isnan(bad.exponent)
    end

    @testset "determinism" begin                          # invariant 1
        a = detect_discrete_map_period(sys, p_chaos, x0, 500, 12, 1e-4, 1e6)
        b = detect_discrete_map_period(sys, p_chaos, x0, 500, 12, 1e-4, 1e6)
        # exact == is intentional: same pure function, identical inputs, no RNG/threading ⇒
        # bitwise-repeatable in-process. This assertion *is* the determinism guarantee; isapprox would weaken it.
        @test a.period == b.period && a.closure_confidence == b.closure_confidence
    end

    @testset "internal map kernel exposes the diagnostics tuple" begin
        # `_bifurcation_map` returns (result, diagnostics); the public `bifurcation_map` discards the
        # diagnostics. Under Contract D the workbench reaches this tuple via the `cells=` hook rather
        # than a public `bifurcation_map_kernel` alias (reverted to private).
        cfg = BifurcationMapConfig(; a_min=1.0, a_max=1.4, a_steps=3, b_min=0.3, b_max=0.3,
                                   b_steps=1, a_index=1, b_index=2, base_params=[1.4, 0.3])
        rk = DynamicsKit._bifurcation_map(sys, cfg)
        @test rk isa Tuple && length(rk) == 2             # (result, diagnostics) — what the cache layer needs
        @test rk[1] isa BifurcationMapResult
        @test bifurcation_map(sys, cfg) isa BifurcationMapResult   # public wrapper still returns just the result
    end

    @testset "public aliases resolve to in-place internals" begin
        @test detect_period === DynamicsKit._detect_period
        @test resolve_initial_state === DynamicsKit._resolve_initial_state
        @test estimate_discrete_map_largest_lyapunov === DynamicsKit._estimate_discrete_map_largest_lyapunov
    end
end
