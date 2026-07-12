# Freezes the public param-vector mapping API.

@testset "parameter mapping API" begin
    @testset "inject_param non-mutating + linked" begin
        base = [1.0, 2.0, 3.0, 4.0]
        out = inject_param(base, 2, 9.0, [4])
        @test out == [1.0, 9.0, 3.0, 9.0]
        @test base == [1.0, 2.0, 3.0, 4.0]            # invariant 1: not mutated
        @test length(out) == length(base)
        @test inject_param(base, 1, 5.0) == [5.0, 2.0, 3.0, 4.0]   # default empty linked
        # widened public contract: accepts any AbstractVector{<:Real}, returns mutable Vector{Float64}
        sv = inject_param(SVector(1.0, 2.0, 3.0), 2, 9.0)
        @test sv == [1.0, 9.0, 3.0] && sv isa Vector{Float64}
        @test inject_param(1:3, 1, 9.0) == [9.0, 2.0, 3.0]         # integer range input
    end

    @testset "build_sweep_params length rule" begin
        c_empty = BruteForceConfig(param_min=0.0, param_max=1.0, param_index=3, linked_param_indices=[5])
        p = build_sweep_params(c_empty, 7.0)
        @test length(p) == 5 && p[3] == 7.0 && p[5] == 7.0        # invariant 2 (zero-filled)
        @test p[1] == 0.0 && p[2] == 0.0 && p[4] == 0.0
        c_fixed = BruteForceConfig(param_min=0.0, param_max=1.0, param_index=2, fixed_params=[1.0, 1.0, 1.0])
        @test build_sweep_params(c_fixed, 5.0) == [1.0, 5.0, 1.0]
    end

    @testset "build_basins_params" begin
        grid = (x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0)     # required BasinsConfig fields
        @test build_basins_params(BasinsConfig(; bif_param=0.3, grid...)) == [0.3]   # invariant 3
        c = BasinsConfig(; bif_param=0.3, param_index=2, fixed_params=[1.0, 1.0, 1.0], grid...)
        @test build_basins_params(c) == [1.0, 0.3, 1.0]
    end

    @testset "basins_ic_template validation" begin                # invariant 4
        sys = henon_map()                                          # DiscreteMap, dim 2
        grid = (x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, bif_param=0.0)
        t = basins_ic_template(sys, BasinsConfig(; x_index=1, y_index=2, grid...))
        @test t isa SVector && length(t) == 2 && all(==(0.0), t)
        @test_throws ArgumentError basins_ic_template(sys, BasinsConfig(; x_index=1, y_index=1, grid...))
        @test_throws ArgumentError basins_ic_template(sys, BasinsConfig(; x_index=1, y_index=2, ic_template=[0.0], grid...))
    end

    @testset "map template padding + write indices" begin         # invariants 5, 6
        ab = (a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0)         # required BifurcationMapConfig fields
        c = BifurcationMapConfig(; a_index=2, b_index=5, base_params=[1.0, 1.0], ab...)
        t = map_param_template(c)
        @test length(t) == 5 && t[1:2] == [1.0, 1.0] && all(==(0.0), t[3:5])
        @test map_a_write_indices(c) == [2]
        @test map_b_write_indices(c) == [5]
    end

    @testset "buffer form is allocation-free + equivalent" begin  # invariants 7, 8
        ab = (a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0)
        c = BifurcationMapConfig(; a_index=1, b_index=2, base_params=[0.0, 0.0, 9.0],
                                 a_linked_param_indices=[3], ab...)
        t  = map_param_template(c)
        ai = map_a_write_indices(c)
        bi = map_b_write_indices(c)
        @test ai == [1, 3]
        buf = similar(t)
        map_params_from_buffer!(buf, t, ai, bi, 4.0, 7.0)            # warmup in-place form
        map_params_from_template(t, ai, bi, 4.0, 7.0)               # warmup allocating form
        # Version-stable expression of the allocation-free invariant: the in-place buffer form must
        # allocate strictly less than the allocating template form (which copies a vector each call).
        @test (@allocated map_params_from_buffer!(buf, t, ai, bi, 4.0, 7.0)) <
              (@allocated map_params_from_template(t, ai, bi, 4.0, 7.0))
        expected = build_map_params(c, 4.0, 7.0)
        @test buf == expected
        @test map_params_from_template(t, ai, bi, 4.0, 7.0) == expected
        @test t == [0.0, 0.0, 9.0]                                # template not mutated by template form
    end

    @testset "underscore aliases still resolve" begin
        @test DynamicsKit._inject_param === inject_param
        @test DynamicsKit._map_params_from_buffer! === map_params_from_buffer!
        @test DynamicsKit._build_map_params === build_map_params
    end
end
