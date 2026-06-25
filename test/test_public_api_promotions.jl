# Public-API promotions for the repository split (Stage 4 prep). Freezes the two scripted-analysis
# helpers paper-artifacts depends on, so they resolve through the LIBRARY's public surface (never the
# workbench). See docs/internal/repository-split-plan-v2.md.

@testset "Public-API promotions (scripted-analysis helpers)" begin
    @testset "select_ode_solver — key → solver mapping" begin
        @test select_ode_solver("tsit5") isa DynamicsKit.Tsit5
        @test select_ode_solver("rosenbrock23") isa DynamicsKit.Rosenbrock23
        # "auto" is the stiff/non-stiff auto-switching composite; just assert it constructs.
        @test select_ode_solver("auto") === DynamicsKit.AutoTsit5(DynamicsKit.Rosenbrock23())
        # Strict: unknown keys throw rather than silently falling back.
        @test_throws ArgumentError select_ode_solver("rk4")
        @test_throws ArgumentError select_ode_solver("")
        # Exported from the library (not reached through the workbench).
        @test :select_ode_solver in names(DynamicsKit)
    end

    @testset "collect_trajectory_seed_points — public alias + behaviour" begin
        @test collect_trajectory_seed_points === DynamicsKit._collect_trajectory_seed_points
        @test :collect_trajectory_seed_points in names(DynamicsKit)

        sys = radial_oscillator()                      # ContinuousODE, 1 param (μ), limit cycle at μ=0.25
        pts = collect_trajectory_seed_points(
            sys, 0.25, [0.25], 1;
            initial_point=[1.0, 0.1],
            crossings=5,
            transient=20
        )
        @test pts isa AbstractVector
        @test !isempty(pts)                            # a limit cycle yields recurrent section crossings
        @test all(p -> length(p) == 1, pts)            # projected onto the section's projection=[1]
    end
end
