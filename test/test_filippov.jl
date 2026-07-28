using Dates

@testset "Filippov grazing and sliding" begin
    function harmonic_grazing_system()
        function f!(du, u, p, t)
            omega = p[2]
            du[1] = u[2]
            du[2] = -(omega^2) * u[1]
            nothing
        end
        section = PoincareSection(
            (u, t, integrator) -> u[2];
            direction=:up,
            projection=[1],
            template=[1.0, 0.0],
            constant_normal=[0.0, 1.0],
        )
        event = SwitchingEvent(
            "impact-stop",
            (x, p) -> x[1] - p[1];
            kind=:grazing,
            description="linear oscillator stop",
            tolerance=1e-8,
            scale=1.0,
        )
        return ContinuousODE(
            f!, 2, section, [:stop, :omega], "Harmonic grazing fixture";
            default_initial_state=[1.0, 0.0],
            default_params=[-1.0, 1.0],
            switching_events=[event],
        )
    end

    sys = harmonic_grazing_system()

    @testset "guard diagnostics at a closed-form grazing point" begin
        diag = filippov_guard_diagnostic(sys, "impact-stop", [-1.0, 0.0], [-1.0, 1.0])
        @test diag.status === :ok
        @test isapprox(diag.guard_value, 0.0; atol=1e-12)
        @test isapprox(diag.normal_velocity, 0.0; atol=1e-12)
        @test isapprox(diag.normal_acceleration, 1.0; atol=1e-6)
        fast = filippov_guard_diagnostic(sys, "impact-stop", [-1.0, 0.0], [-1.0, 2.0])
        @test isapprox(fast.normal_acceleration, 4.0; atol=1e-5)
    end

    @testset "orbit grazing detection" begin
        result = filippov_grazing_points(
            sys,
            FilippovGrazingConfig(
                t_stop=2pi,
                sample_count=401,
                guard_tolerance=1e-7,
                velocity_tolerance=1e-7,
                acceleration_tolerance=1e-6,
            );
            params=[-1, 1],
            initial_point=[1, 0],
        )
        @test result.status === :grazing
        @test length(result.points) == 1
        point = only(result.points)
        @test point.status === :grazing
        @test isapprox(point.time, pi; atol=1e-5)
        @test isapprox(point.state[1], -1.0; atol=1e-7)
        @test abs(point.state[2]) <= 1e-6
        @test point.normal_acceleration > 0.9
    end

    @testset "slice-based two-parameter grazing locus" begin
        locus = filippov_grazing_locus(
            sys,
            FilippovGrazingLocusConfig(
                grazing=FilippovGrazingConfig(
                    t_stop=5.0,
                    sample_count=501,
                    guard_tolerance=1e-6,
                    velocity_tolerance=1e-6,
                    acceleration_tolerance=1e-6,
                ),
                primary_index=1,
                secondary_index=2,
                primary_min=-1.2,
                primary_max=-0.8,
                secondary_min=0.8,
                secondary_max=1.2,
                secondary_steps=4,
                fixed_params=[-1.0, 1.0],
                margin=:minimum,
                root_tolerance=1e-6,
            );
            initial_point=[1.0, 0.0],
        )
        @test all(status -> status === :grazing, locus.statuses)
        @test all(isapprox(value, -1.0; atol=2e-5) for value in locus.primary_values)
        @test maximum(abs.(locus.normal_velocities)) <= 1e-5
    end

    @testset "sliding segment classification" begin
        event = SwitchingEvent("surface", (x, p) -> x[1]; kind=:grazing)
        states = [[0.0, -0.5], [0.0, 0.0], [0.0, 0.5], [0.0, 1.2]]
        f_minus = (x, p) -> [1.0 - x[2]^2, 0.0]
        f_plus = (x, p) -> [-1.0 + x[2]^2, 0.0]
        result = filippov_sliding_segments(event, states, Float64[], f_minus, f_plus)
        @test result.status === :ok
        @test length(result.segments) == 2
        @test result.segments[1].kind === :attracting
        @test result.segments[1].start_index == 1
        @test result.segments[1].end_index == 3
        @test result.segments[2].kind === :repelling
        @test result.segments[2].start_index == 4

        matrix_result = filippov_sliding_segments(
            event,
            [0.0 0.0 0.0 0.0; -0.5 0.0 0.5 1.2],
            Float64[],
            f_minus,
            f_plus,
        )
        @test matrix_result.status === :ok
        @test length(matrix_result.segments) == 2
        @test matrix_result.segments[1].kind === :attracting
        @test matrix_result.segments[1].start_index == 1
        @test matrix_result.segments[1].end_index == 3
        @test matrix_result.segments[2].kind === :repelling
        @test matrix_result.segments[2].start_index == 4
    end

    @testset "versioned serialization" begin
        point = FilippovGrazingPoint(
            "impact-stop", pi, [-1.0, 0.0], [-1.0, 1.0], 0.0, 0.0, 1.0,
            :grazing, true)
        grazing = FilippovGrazingResult(
            [point], "Harmonic grazing fixture", "impact-stop", [-1.0, 1.0],
            (0.0, 2pi), :grazing, String[], Dates.now())
        rt_grazing = deserialize_filippov_grazing_result(serialize_filippov_grazing_result(grazing))
        @test rt_grazing.status === grazing.status
        @test only(rt_grazing.points).status === :grazing
        @test isapprox(only(rt_grazing.points).time, pi; atol=0)

        locus = FilippovGrazingLocusResult(
            [-1.0, -1.0], [0.9, 1.1], [-1.0 -1.0; 0.0 0.0],
            [0.0, 0.0], [0.0, 0.0], [0.9, 1.1], [:grazing, :grazing],
            "Harmonic grazing fixture", "impact-stop", (:stop, :omega), Dates.now())
        rt_locus = deserialize_filippov_grazing_locus_result(serialize_filippov_grazing_locus_result(locus))
        @test rt_locus.statuses == locus.statuses
        @test rt_locus.param_names == locus.param_names
        @test rt_locus.states == locus.states

        sliding = FilippovSlidingResult(
            [FilippovSlidingSegment("surface", 1, 3, [0.0, -0.5], [0.0, 0.5], 1.0, -1.0, :attracting, :classified)],
            "surface", 3, :ok, String[], Dates.now())
        rt_sliding = deserialize_filippov_sliding_result(serialize_filippov_sliding_result(sliding))
        @test rt_sliding.status === :ok
        @test only(rt_sliding.segments).kind === :attracting
    end

    @testset "public API exports" begin
        @test :filippov_guard_diagnostic in names(DynamicsKit)
        @test :filippov_grazing_points in names(DynamicsKit)
        @test :filippov_grazing_locus in names(DynamicsKit)
        @test :filippov_sliding_segments in names(DynamicsKit)
        @test :FilippovGrazingConfig in names(DynamicsKit)
        @test :FilippovGrazingLocusConfig in names(DynamicsKit)
        @test :serialize_filippov_grazing_result in names(DynamicsKit)
        @test :serialize_filippov_grazing_locus_result in names(DynamicsKit)
        @test :serialize_filippov_sliding_result in names(DynamicsKit)
    end
end
