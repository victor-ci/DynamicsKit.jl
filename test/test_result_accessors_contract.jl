# Contract C — result & diagnostics accessors.
# Superseded by Contract D (docs/internal/contracts/contract-d-sweep-cache-hook.md): the effective
# settings, diagnostics producers, per-cell storage and small utilities reverted to underscore-private
# once the workbench began driving caching through the public sweeps (`cells=` hook). The behavioural
# invariants below still matter for those internals, so they are tested against the private names; only
# the continuation branch post-processing helpers (Group 6) remain part of the public surface.

@testset "Contract C — result & diagnostics accessors (internals + surviving public helpers)" begin
    @testset "classification diagnostics schema lock" begin   # frozen wire keys (internal producer)
        sc = fill(1, 2, 2); ce = fill(0.0, 2, 2); cp = fill(1, 2, 2)
        op = fill(3, 2, 2); cc = fill(1.0, 2, 2)
        d = DynamicsKit._map_classification_diagnostics(sc, ce, cp, op, cc)
        @test Set(keys(d)) == Set([
            "statusCodes", "statusLabels", "statusCounts", "closureErrors",
            "closureCandidatePeriods", "observedPoints", "closureConfidence",
            "minClosureError", "maxClosureError", "minClosureConfidence", "maxClosureConfidence"])
    end

    @testset "effective-settings" begin
        ab = (a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0)
        c_default = BifurcationMapConfig(; max_period=10, ab...)               # lyapunov_iterations=0 ⇒ fallback
        @test DynamicsKit._map_lyapunov_iterations(c_default) == max(32, 8 * 10)            # = 80
        c_explicit = BifurcationMapConfig(; lyapunov_iterations=50, ab...)
        @test DynamicsKit._map_lyapunov_iterations(c_explicit) == 50
        @test DynamicsKit._map_transient_budget(c_default) isa Int && DynamicsKit._map_transient_budget(c_default) >= 0
        @test DynamicsKit._map_seed_mode(BifurcationMapConfig(; ab...)) == :fixed           # reuse_neighbor_seeds=false default
    end

    @testset "storage shape" begin
        s = DynamicsKit._map_lyapunov_storage(2, 3)
        @test size(s.exponents) == (2, 3) && all(isnan, s.exponents)
    end

    @testset "small utilities" begin
        ch = DynamicsKit._balanced_index_chunks(10, 3)
        @test eltype(ch) == UnitRange{Int}
        @test reduce(vcat, collect.(ch)) == collect(1:10)                      # exact partition of 1:n
        @test DynamicsKit._balanced_index_chunks(0, 4) == UnitRange{Int}[]
        g = DynamicsKit._orbit_geometry_summary([[0.0, 0.0], [2.0, 4.0]])
        @test g.span == [2.0, 4.0] && g.center == [1.0, 2.0]
        e = DynamicsKit._orbit_geometry_summary([]); @test isempty(e.center) && isempty(e.span)
        @test DynamicsKit._map_status_code(:definitely_not_a_status) == DynamicsKit._map_status_code(:unknown)   # total mapping
    end

    @testset "config fields read directly (no accessor)" begin
        ab = (a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0)
        c = BifurcationMapConfig(; lyapunov_enabled=true, ab...)
        @test c.lyapunov_enabled == true                                      # workbench reads this directly
        @test something(c.lyapunov_transient, 0) == 0
        @test isempty(c.multistability_initial_points)
    end

    @testset "surviving public aliases resolve to in-place internals (Group 6)" begin
        @test branch_stability === DynamicsKit._map_stability
        @test trim_branch_to_period === DynamicsKit._trim_branch_to_period
    end
end
