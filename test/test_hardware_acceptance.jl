using Dates: DateTime

@testset "Hardware acceptance" begin
    a_grid = collect(0.0:0.1:1.0)
    b_grid = [10.0, 20.0, 30.0]
    periodicity = fill(1, length(a_grid), length(b_grid))
    periodicity[6:end, :] .= 2
    map_result = BifurcationMapResult(
        a_grid,
        b_grid,
        periodicity,
        8,
        "Analytic acceptance route",
        (:gain, :load),
        DateTime(2026, 7, 28),
    )
    route = operating_map_cross_section(map_result; varying_parameter=:gain, fixed_value=20.0)
    boundary = regime_boundary_distances(
        a_grid,
        b_grid,
        periodicity,
        trues(length(a_grid), length(b_grid));
        config=RegimeBoundaryConfig(edge_policy=:ignore),
        system_name="Analytic acceptance route",
        param_names=(:gain, :load),
        status_evidence=true,
    )
    region = RobustChaosRegion(
        1, :certified, 0.9,
        0.0, 1.0, 10.0, 30.0,
        20.0, length(a_grid) * length(b_grid), 0, 0,
        :pass, 1.0, 1.0, 0.1, 9, 9, 9,
        :pass, 1, 1, 1.0, 0,
        :pass, 1, 1.0, 1.0, 9, 9, 9,
        0.1, false,
        String[],
        Dict{String, Any}[Dict("layer" => "overall", "verdict" => "certified")],
    )
    certificate = RobustChaosRegionResult(
        [region],
        "Analytic acceptance route",
        (:gain, :load),
        1, 0, 16, 16, false, 0, 0, 0,
        :two_trajectory,
        :per_iteration,
        :ignore,
        DateTime(2026, 7, 28),
        Dict{String, Any}[Dict("layer" => "overall", "region_count" => 1)],
    )

    @testset "accepts matching route inside certified region" begin
        measured = ModeSequence(:gain, [0.1, 0.3, 0.7, 0.9], ["P1", "P1", "P2", "P2"])
        result = hardware_acceptance_test(
            measured,
            route,
            certificate;
            boundary,
            config=HardwareAcceptanceConfig(axis_calibration=:identity),
        )
        @test result.verdict == :accepted
        @test isempty(result.mismatches)
        @test hardware_acceptance_summary(result).mismatch_count == 0
    end

    @testset "rejects mode shift beyond local margin" begin
        measured = ModeSequence(:gain, [0.1, 0.8], ["P1", "P1"])
        result = hardware_acceptance_test(
            measured,
            route,
            certificate;
            boundary,
            config=HardwareAcceptanceConfig(axis_calibration=:identity),
        )
        @test result.verdict == :rejected
        @test length(result.mismatches) == 1
        mismatch = only(result.mismatches)
        @test mismatch.observation_index == 2
        @test mismatch.measured_mode == "P1"
        @test mismatch.predicted_mode == "P2"
        @test mismatch.required_shift > mismatch.margin
        @test mismatch.margin_status == :outside_margin
    end

    @testset "inconclusive when mismatch margin evidence is unavailable" begin
        measured = ModeSequence(:gain, [0.1, 0.8], ["P1", "P1"])
        no_boundary = hardware_acceptance_test(
            measured,
            route,
            certificate;
            config=HardwareAcceptanceConfig(axis_calibration=:identity),
        )
        @test no_boundary.verdict == :inconclusive
        @test only(no_boundary.mismatches).margin_status == :margin_unavailable

        resolved = trues(size(periodicity))
        resolved[9, 2] = false
        unresolved_boundary = regime_boundary_distances(
            a_grid,
            b_grid,
            periodicity,
            resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore),
            system_name="Analytic acceptance route",
            param_names=(:gain, :load),
            status_evidence=true,
        )
        unresolved = hardware_acceptance_test(
            measured,
            route,
            certificate;
            boundary=unresolved_boundary,
            config=HardwareAcceptanceConfig(axis_calibration=:identity),
        )
        @test unresolved.verdict == :inconclusive
        @test only(unresolved.mismatches).margin_status == :unresolved_margin
    end

    @testset "records tolerance-map nominal probability" begin
        tolerance = tolerance_regime_map(
            a_grid,
            b_grid,
            periodicity,
            trues(size(periodicity)),
            ToleranceConfig(threaded=false);
            system_name="Analytic acceptance route",
            param_names=(:gain, :load),
            status_evidence=true,
        )
        measured = ModeSequence(:gain, [0.1, 0.8], ["P1", "P1"])
        result = hardware_acceptance_test(
            measured,
            route,
            certificate;
            boundary,
            tolerance,
            config=HardwareAcceptanceConfig(axis_calibration=:identity),
        )
        @test result.verdict == :rejected
        @test only(result.mismatches).tolerance_probability == 1.0
    end

    @testset "refuses underdetermined transition calibration" begin
        measured = ModeSequence(:knob, [1.0, 2.0, 3.0], ["P1", "P2", "P2"])
        result = hardware_acceptance_test(
            measured,
            route,
            certificate;
            config=HardwareAcceptanceConfig(axis_calibration=:transition_affine),
        )
        @test result.verdict == :refused
        @test result.alignment === nothing
        @test occursin("at least two", result.certificate_items[1]["reason"])
    end

    @testset "serializes acceptance result" begin
        measured = ModeSequence(:gain, [0.1, 0.8], ["P1", "P1"])
        result = hardware_acceptance_test(
            measured,
            route,
            certificate;
            boundary,
            config=HardwareAcceptanceConfig(axis_calibration=:identity),
        )
        wire = serialize_hardware_acceptance_result(result)
        @test wire["format"] == "hardware-acceptance-v1"
        restored = deserialize_hardware_acceptance_result(wire)
        @test restored.verdict == result.verdict
        @test restored.certificate_kind == :region_result
        @test length(restored.mismatches) == 1
        @test restored.mismatches[1].margin_status == :outside_margin

        invalid_status = deepcopy(wire)
        invalid_status["mismatches"][1]["marginStatus"] = "attacker-controlled-status"
        @test_throws ArgumentError deserialize_hardware_acceptance_result(invalid_status)

        invalid_verdict = deepcopy(wire)
        invalid_verdict["certificateVerdict"] = "attacker-controlled-verdict"
        @test_throws ArgumentError deserialize_hardware_acceptance_result(invalid_verdict)

        missing_timestamp = deepcopy(wire)
        delete!(missing_timestamp, "timestamp")
        @test_throws ErrorException deserialize_hardware_acceptance_result(missing_timestamp)
    end
end
