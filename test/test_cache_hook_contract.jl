# Freezes the in/out cell-grid behaviour for the public sweeps:
# fresh cells == no-cells result; partial pre-seed (round-trip) == full recompute.

@testset "sweep cache hook" begin
    sys = henon_map()
    BE = DynamicsKit

    @testset "bifurcation_map MapCellGrid round-trip" begin
        cfg = BifurcationMapConfig(a_min=1.0, a_max=1.4, a_steps=20, b_min=0.2, b_max=0.35, b_steps=16,
                                   a_index=1, b_index=2, base_params=[1.4, 0.3],
                                   max_period=8, iterations=300, precision=1e-3, divergence_cutoff=1e6)
        na, nb = cfg.a_steps + 1, cfg.b_steps + 1
        rf, _ = BE._bifurcation_map(sys, cfg)
        g1 = MapCellGrid(na, nb)
        r1, _ = BE._bifurcation_map(sys, cfg; cells=g1)
        @test r1.periodicity == rf.periodicity          # fresh cells == no-cells
        @test all(g1.known)
        g2 = MapCellGrid(na, nb)
        for i in 1:na, j in 1:nb
            if (i + j) % 2 == 0
                g2.periodicity[i, j]               = g1.periodicity[i, j]
                g2.status_codes[i, j]              = g1.status_codes[i, j]
                g2.closure_errors[i, j]            = g1.closure_errors[i, j]
                g2.closure_candidate_periods[i, j] = g1.closure_candidate_periods[i, j]
                g2.observed_points[i, j]           = g1.observed_points[i, j]
                g2.closure_confidence[i, j]        = g1.closure_confidence[i, j]
                g2.known[i, j] = true
            end
        end
        r2, _ = BE._bifurcation_map(sys, cfg; cells=g2)
        @test r2.periodicity == rf.periodicity          # partial pre-seed == full
        @test g2.status_codes == g1.status_codes        # raw-array round-trip
        @test all(g2.known)
    end

    @testset "lyapunov_field LyapunovCellGrid round-trip" begin
        cfg = BifurcationMapConfig(a_min=1.0, a_max=1.4, a_steps=10, b_min=0.2, b_max=0.3, b_steps=8,
                                   a_index=1, b_index=2, base_params=[1.4, 0.3],
                                   max_period=6, iterations=300, lyapunov_iterations=150,
                                   precision=1e-3, divergence_cutoff=1e6)
        na, nb = cfg.a_steps + 1, cfg.b_steps + 1
        rf = lyapunov_field(sys, cfg)
        g1 = LyapunovCellGrid(na, nb)
        r1 = lyapunov_field(sys, cfg; cells=g1)
        @test r1.exponents == rf.exponents
        @test all(g1.known)
        g2 = LyapunovCellGrid(na, nb)
        for i in 1:na, j in 1:nb
            if (i + j) % 2 == 0
                g2.exponents[i, j]               = g1.exponents[i, j]
                g2.status_codes[i, j]            = g1.status_codes[i, j]
                g2.estimation_status_codes[i, j] = g1.estimation_status_codes[i, j]
                g2.sample_counts[i, j]           = g1.sample_counts[i, j]
                g2.known[i, j] = true
            end
        end
        r2 = lyapunov_field(sys, cfg; cells=g2)
        @test r2.exponents == rf.exponents
        @test all(g2.known)
    end

    @testset "basins BasinsCellGrid round-trip" begin
        cfg = BasinsConfig(bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
                           x_min=-0.4, x_max=0.4, x_steps=12, y_min=-0.4, y_max=0.4, y_steps=12,
                           x_index=1, y_index=2, max_period=8, iterations=300, precision=1e-3)
        nx, ny = cfg.x_steps + 1, cfg.y_steps + 1
        rf = basins_of_attraction(sys, cfg)
        g1 = BasinsCellGrid(nx, ny)
        r1 = basins_of_attraction(sys, cfg; cells=g1)
        @test r1.periodicity == rf.periodicity
        @test all(g1.known)
        g2 = BasinsCellGrid(nx, ny)
        for i in 1:nx, j in 1:ny
            if (i + j) % 2 == 0
                g2.periodicity[i, j] = g1.periodicity[i, j]
                g2.known[i, j] = true
            end
        end
        r2 = basins_of_attraction(sys, cfg; cells=g2)
        @test r2.periodicity == rf.periodicity
        @test all(g2.known)
    end
end
