using Dates

@testset "Experimental mode-sequence assimilation" begin
    @testset "Validated CSV ingestion" begin
        csv = IOBuffer("""
        parameter,mode,weight
        0.05,P1,2
        0.20,period 1,1
        0.35,chaos,0.5
        """)
        sequence = load_mode_sequence_csv(csv; parameter_name=:a, source="Hardware")
        @test sequence.parameter_name == :a
        @test sequence.parameter_values == [0.05, 0.2, 0.35]
        @test sequence.modes == ["P1", "period 1", "chaos"]
        @test sequence.weights == [2.0, 1.0, 0.5]
        @test sequence.source == "Hardware"

        @test_throws ArgumentError load_mode_sequence_csv(IOBuffer("parameter,mode\n0.2,P1\n0.1,P2\n"))
        @test_throws ArgumentError load_mode_sequence_csv(IOBuffer("parameter,mode\n0.1,P1\n0.1,P2\n"))
        @test_throws ArgumentError load_mode_sequence_csv(IOBuffer("parameter,label\n0.1,P1\n0.2,P2\n"))
        @test_throws ArgumentError load_mode_sequence_csv(IOBuffer("parameter,mode,weight\n0.1,P1,0\n0.2,P2,1\n"))
        @test_throws ArgumentError load_mode_sequence_csv(IOBuffer("parameter,mode\nbad,P1\n0.2,P2\n"))
    end

    a_grid = collect(0.0:0.1:1.0)
    b_grid = [10.0, 20.0, 30.0]
    periodicity = fill(1, length(a_grid), length(b_grid))
    periodicity[4:7, :] .= 2
    periodicity[8:9, :] .= 0
    map_result = BifurcationMapResult(
        a_grid,
        b_grid,
        periodicity,
        8,
        "Analytic route",
        (:gain, :load),
        now(),
    )

    @testset "Operating-map cross-section" begin
        route = operating_map_cross_section(
            map_result;
            varying_parameter=:gain,
            fixed_value=19.0,
        )
        @test route.sequence.parameter_name == :gain
        @test route.sequence.modes[1:3] == fill("P1", 3)
        @test route.sequence.modes[4:7] == fill("P2", 4)
        @test route.sequence.modes[8:9] == fill("undetected", 2)
        @test route.selected_fixed_value == 20.0
        @test route.fixed_grid_index == 2
        @test route.fixed_parameter_name == :load
        @test !route.status_evidence

        status = fill(map_status_code(:periodic), size(periodicity))
        status[8, :] .= map_status_code(:aperiodic_or_high_period)
        status[9, :] .= map_status_code(:diverged)
        status[10:11, :] .= map_status_code(:unknown)
        sharpened = operating_map_cross_section(
            map_result;
            varying_parameter=:a,
            fixed_value=20.0,
            status_codes=status,
        )
        @test sharpened.sequence.modes[8] == "aperiodic"
        @test sharpened.sequence.modes[9] == "diverged"
        @test sharpened.sequence.modes[10:11] == fill("undetected", 2)
        @test sharpened.status_evidence

        vertical = operating_map_cross_section(
            map_result;
            varying_parameter=:b,
            fixed_value=0.4,
        )
        @test vertical.sequence.parameter_name == :load
        @test vertical.sequence.modes == fill("P2", 3)
        alias_collision = BifurcationMapResult(
            [0.0, 1.0],
            [0.0, 1.0],
            [1 1; 2 2],
            2,
            "Alias collision",
            (:gain, :a),
            now(),
        )
        exact_second_axis = operating_map_cross_section(
            alias_collision;
            varying_parameter=:a,
            fixed_value=0.0,
        )
        @test exact_second_axis.sequence.parameter_name == :a
        @test_throws ArgumentError operating_map_cross_section(
            map_result;
            varying_parameter=:gain,
            fixed_value=31.0,
        )
    end

    @testset "Observation and transition scores" begin
        scoring_periodicity = copy(periodicity)
        scoring_periodicity[8:end, :] .= 1
        scoring_map = BifurcationMapResult(
            a_grid,
            b_grid,
            scoring_periodicity,
            8,
            "Analytic route",
            (:gain, :load),
            now(),
        )
        theory = operating_map_cross_section(
            scoring_map;
            varying_parameter=:gain,
            fixed_value=20.0,
        )
        experiment = ModeSequence(
            :component,
            [5.0, 20.0, 40.0, 64.0, 95.0, 105.0],
            ["period 1", "P1", "P2", "P2", "P1", "P1"];
            source="Hardware",
        )
        config = ModeAssimilationConfig(
            experimental_scale=0.01,
            transition_tolerance=0.02,
        )
        result = assimilate_mode_sequence(experiment, theory; config=config)
        @test result.aligned_parameter_values ≈ [0.05, 0.2, 0.4, 0.64, 0.95, 1.05]
        @test result.observation_statuses == [
            :matched, :matched, :matched, :matched, :matched, :outside_coverage,
        ]
        @test result.agreement_score == 1.0
        @test result.coverage == 5 / 6
        @test result.overall_score == 5 / 6
        @test length(result.transition_comparisons) == 2
        @test all(comparison -> comparison.status == :matched, result.transition_comparisons)
        @test isempty(result.unexpected_transitions)
        @test result.transition_precision == 1.0
        @test result.transition_recall == 1.0
        @test result.transition_f1 == 1.0

        summary = mode_assimilation_summary(result)
        @test summary.matched_observations == 5
        @test summary.outside_observations == 1
        @test summary.matched_transition_count == 2

        wire = serialize_mode_sequence_alignment(result)
        @test wire["format"] == "mode-sequence-alignment-v1"
        restored = deserialize_mode_sequence_alignment(wire)
        @test restored.experimental.parameter_values == result.experimental.parameter_values
        @test restored.predicted.sequence.modes == result.predicted.sequence.modes
        @test restored.observation_statuses == result.observation_statuses
        @test restored.transition_f1 == result.transition_f1
        @test restored.config.experimental_scale == 0.01
        @test !isnothing(plot_mode_sequence_alignment(restored))

        without_weights = deepcopy(wire)
        delete!(without_weights["experimental"], "weights")
        delete!(without_weights["predicted"]["sequence"], "weights")
        restored_without_weights = deserialize_mode_sequence_alignment(without_weights)
        @test restored_without_weights.experimental.weights ==
            ones(length(restored_without_weights.experimental.parameter_values))
        @test restored_without_weights.predicted.sequence.weights ==
            ones(length(restored_without_weights.predicted.sequence.parameter_values))

        invalid_weights = deepcopy(wire)
        invalid_weights["experimental"]["weights"] = Any[]
        @test_throws ArgumentError deserialize_mode_sequence_alignment(invalid_weights)
        @test_throws ArgumentError deserialize_mode_sequence_alignment(nothing)
        invalid_format = deepcopy(wire)
        invalid_format["format"] = "unknown"
        @test_throws ArgumentError deserialize_mode_sequence_alignment(invalid_format)
        symbol_keys = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in wire)
        @test deserialize_mode_sequence_alignment(symbol_keys).overall_score ==
            result.overall_score

        missing_transition = ModeSequence(
            :gain,
            [0.05, 0.25, 0.95],
            ["P1", "P1", "P1"],
        )
        mismatch = assimilate_mode_sequence(missing_transition, theory)
        @test isempty(mismatch.transition_comparisons)
        @test length(mismatch.unexpected_transitions) == 2
        @test mismatch.transition_precision == 0.0
        @test mismatch.transition_recall === nothing
        @test mismatch.transition_f1 === nothing

        observed_for_assignment = [
            ModeTransition(0.0, 12.0, 6.0, "P1", "P2"),
            ModeTransition(13.0, 20.0, 16.5, "P1", "P2"),
        ]
        predicted_for_assignment = [
            ModeTransition(0.5, 1.5, 1.0, "P1", "P2"),
            ModeTransition(8.5, 9.5, 9.0, "P1", "P2"),
        ]
        assignment = DynamicsKit._transition_assignment(
            observed_for_assignment,
            predicted_for_assignment,
            Dict{String, String}(),
            4.0,
        )
        @test assignment == Dict(1 => 1, 2 => 2)

        aliased = ModeSequence(:gain, [0.75, 0.85], ["UPI-1", "UPI-2"])
        status = fill(map_status_code(:periodic), size(periodicity))
        status[8:9, :] .= map_status_code(:aperiodic_or_high_period)
        status[10:11, :] .= map_status_code(:unknown)
        aperiodic_theory = operating_map_cross_section(
            map_result;
            varying_parameter=:gain,
            fixed_value=20.0,
            status_codes=status,
        )
        alias_result = assimilate_mode_sequence(aliased, aperiodic_theory)
        @test alias_result.observation_statuses == [:matched, :matched]
        @test alias_result.agreement_score == 1.0
    end

    @testset "Configuration validation" begin
        @test_throws ArgumentError ModeAssimilationConfig(experimental_scale=0.0)
        @test_throws ArgumentError ModeAssimilationConfig(transition_tolerance=-1.0)
        sequence = ModeSequence(:gain, [0.1, 0.2], ["P1", "P2"])
        @test_throws ArgumentError mode_sequence_transitions(
            sequence;
            parameter_values=[0.2, 0.1],
        )
        cyclic = ModeAssimilationConfig(mode_aliases=Dict("foo" => "bar", "bar" => "foo"))
        experiment = ModeSequence(:gain, [0.1, 0.2], ["foo", "bar"])
        theory = operating_map_cross_section(
            map_result;
            varying_parameter=:gain,
            fixed_value=20.0,
        )
        @test_throws ArgumentError assimilate_mode_sequence(experiment, theory; config=cyclic)

        valid = assimilate_mode_sequence(
            ModeSequence(:gain, [0.1, 0.2], ["P1", "P1"]),
            theory,
        )
        malformed = serialize_mode_sequence_alignment(valid)
        malformed["coverage"] = 2.0
        @test_throws ArgumentError deserialize_mode_sequence_alignment(malformed)
    end
end
