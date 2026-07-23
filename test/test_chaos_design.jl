using Dates: DateTime

@testset "chaos_design" begin

    # ── spectral_flatness unit tests ────────────────────────────────────────

    @testset "spectral_flatness" begin
        @testset "empty → 0" begin
            @test spectral_flatness(Float64[]) == 0.0
        end

        @testset "all-zero → 0" begin
            @test spectral_flatness([0.0, 0.0, 0.0]) == 0.0
        end

        @testset "invalid power is rejected" begin
            @test_throws ArgumentError spectral_flatness([1.0, NaN, 1.0])
            @test_throws ArgumentError spectral_flatness([1.0, Inf, 1.0])
            @test_throws ArgumentError spectral_flatness([1.0, -1.0, 1.0])
        end

        @testset "single tonal → near 0" begin
            # Impulse: one large bin and the rest near zero.
            p = vcat([0.0], fill(0.0, 30), [1000.0])
            @test spectral_flatness(p) < 0.05
        end

        @testset "flat spectrum → near 1" begin
            # Exactly uniform: geometric mean == arithmetic mean.
            p = fill(2.0, 64)
            @test spectral_flatness(p) ≈ 1.0 atol=1e-10
        end

        @testset "partial zero bins numerically stable" begin
            # A few zero bins should not collapse flatness to 0 when other energy exists.
            p = [0.0, 1.0, 0.0, 1.0, 0.0]
            f = spectral_flatness(p)
            @test 0.0 < f < 1.0
        end

        @testset "single non-zero bin → near 0" begin
            p = [0.0, 0.0, 5.0, 0.0]
            @test spectral_flatness(p) < 0.1
        end

        @testset "return in [0, 1]" begin
            for p in [[0.5, 1.0, 0.5], [100.0, 1.0, 1.0], fill(1.0, 128)]
                @test 0.0 <= spectral_flatness(p) <= 1.0
            end
        end

        @testset "scale invariant" begin
            p = [0.0, 0.25, 1.0, 0.5]
            @test spectral_flatness(p) ≈ spectral_flatness(p .* 1e-300) atol=1e-14
            @test spectral_flatness(p) ≈ spectral_flatness(p .* 1e300) atol=1e-14
        end
    end

    # ── ChaosDesignTarget validation ────────────────────────────────────────

    @testset "ChaosDesignTarget validation" begin
        @test ChaosDesignTarget(min_amplitude=0.5, max_amplitude=2.0) isa ChaosDesignTarget
        @test_throws ArgumentError ChaosDesignTarget(min_amplitude=-1.0)
        @test_throws ArgumentError ChaosDesignTarget(min_amplitude=2.0, max_amplitude=1.0)
        @test_throws ArgumentError ChaosDesignTarget(min_spectral_flatness=1.5)
        @test_throws ArgumentError ChaosDesignTarget(min_robustness_score=-0.1)
    end

    # ── ChaosDesignSignalConfig validation ─────────────────────────────────

    @testset "ChaosDesignSignalConfig validation" begin
        @test ChaosDesignSignalConfig() isa ChaosDesignSignalConfig
        @test_throws AssertionError ChaosDesignSignalConfig(state_index=0)
        @test_throws AssertionError ChaosDesignSignalConfig(discrete_samples=1)
        @test_throws AssertionError ChaosDesignSignalConfig(discrete_sample_interval=0)
        @test_throws AssertionError ChaosDesignSignalConfig(discrete_window=:boxcar)
        @test_throws AssertionError ChaosDesignSignalConfig(divergence_cutoff=-1.0)
        @test_throws AssertionError ChaosDesignSignalConfig(divergence_cutoff=Inf)
    end

    cat_map = DiscreteMap(
        (x, p) -> SVector(
            mod(x[1] + x[2], 1.0),
            mod(x[1] + 2 * x[2], 1.0),
            x[3] >= 0.0 ? -p[2] : p[2],
        ),
        3, [:p, :amplitude], "cat_design_test"
    )

    lya_cfg = LyapunovConfig(
        param_min=0.5, param_max=1.0, param_steps=2,
        param_index=1, fixed_params=[0.7, 0.3],
        transient=30, iterations=80,
    )
    bf_cfg = BruteForceConfig(
        param_min=0.5, param_max=1.0, param_index=1,
        fixed_params=[0.7, 0.3], param_steps=3,
        iterations=40, transient=20,
    )
    cont_cfg = ContinuationConfig(
        p_min=0.5, p_max=1.0, param_index=1,
        ds=0.05, dsmax=0.1, max_steps=30,
    )
    atlas_cfg = AtlasConfig(
        brute_force=bf_cfg, continuation=cont_cfg,
        periods=[1], max_period=1, recon_steps=4,
        cache_enabled=false, threaded=false,
    )
    basins_cfg = BasinsConfig(
        bif_param=0.7, param_index=1, fixed_params=[0.7, 0.3],
        x_min=0.1, x_max=0.9, x_steps=2,
        y_min=0.1, y_max=0.9, y_steps=2,
        x_index=1, y_index=2, ic_template=[0.0, 0.0, 1.0],
        iterations=4, max_period=2,
    )
    rc_cfg = RobustChaosConfig(lyapunov=lya_cfg, atlas=atlas_cfg, basins=basins_cfg)

    # ── ChaosDesignVariable and ChaosDesignConfig validation ───────────────

    @testset "ChaosDesignVariable and ChaosDesignConfig validation" begin
        var_b = ChaosDesignVariable(:amplitude, 2, 0.1, 0.5)
        @test var_b isa ChaosDesignVariable

        signal_cfg = ChaosDesignSignalConfig(
            state_index=1, discrete_transient=50, discrete_samples=64,
            discrete_sample_interval=1, discrete_window=:none,
        )
        target = ChaosDesignTarget(min_amplitude=0.0, max_amplitude=2.0)

        # Valid config
        @test ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b],
            target=target,
            signal=signal_cfg,
            samples_per_axis=2,
            refinement_levels=0,
            survivors_per_level=2,
            max_evaluations=4,
        ) isa ChaosDesignConfig

        # Too many variables
        @test_throws AssertionError ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b, var_b, var_b, var_b],
            target=target,
            signal=signal_cfg,
            samples_per_axis=2,
        )

        # Duplicate param index
        @test_throws AssertionError ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b, ChaosDesignVariable(:amplitude2, 2, 0.3, 0.6)],
            target=target,
            signal=signal_cfg,
            samples_per_axis=2,
        )

        # Overlaps swept slot (param_index=1)
        @test_throws AssertionError ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[ChaosDesignVariable(:p, 1, 0.4, 0.9)],
            target=target,
            signal=signal_cfg,
            samples_per_axis=2,
        )

        # Non-finite bounds
        @test_throws ArgumentError ChaosDesignVariable(:amplitude, 2, 0.3, Inf)

        # samples_per_axis < 2
        @test_throws AssertionError ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b],
            target=target,
            signal=signal_cfg,
            samples_per_axis=1,
        )

        # max_evaluations < 1
        @test_throws AssertionError ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b],
            target=target,
            signal=signal_cfg,
            samples_per_axis=2,
            max_evaluations=0,
        )
    end

    # ── _cd_build_candidate_config coherence ───────────────────────────────

    @testset "_cd_build_candidate_config coherence" begin
        var_b = ChaosDesignVariable(:amplitude, 2, 0.1, 0.5)
        signal_cfg = ChaosDesignSignalConfig(state_index=1, discrete_transient=50,
                                              discrete_samples=64, discrete_window=:none)
        target = ChaosDesignTarget()
        design_cfg = ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b],
            target=target,
            signal=signal_cfg,
            samples_per_axis=2,
            refinement_levels=0,
            max_evaluations=4,
        )

        # Build candidate config at b=0.25
        cand_cfg = DynamicsKit._cd_build_candidate_config(design_cfg, [0.25])
        @test cand_cfg isa RobustChaosConfig

        # Operating param settings must be preserved
        @test cand_cfg.lyapunov.param_min ≈ 0.5
        @test cand_cfg.lyapunov.param_max ≈ 1.0
        @test cand_cfg.lyapunov.param_index == 1
        @test cand_cfg.basins.bif_param ≈ 0.7
        @test cand_cfg.basins.param_index == 1

        # Design value should be written
        @test cand_cfg.lyapunov.fixed_params[2] ≈ 0.25
        @test cand_cfg.atlas.brute_force.fixed_params[2] ≈ 0.25
        @test cand_cfg.basins.fixed_params[2] ≈ 0.25

        # Swept slot must keep its operating value in basins
        @test cand_cfg.basins.fixed_params[1] ≈ 0.7
    end

    # ── Ranking and parameter recovery (analytic fixture) ──────────────────

    @testset "Ranking order — feasible first, then objective, then lex" begin
        # Construct synthetic certificates for ranking tests.
        make_cert(verdict::Symbol, rob::Float64) = RobustChaosCertificate(
            0.5, 1.0, "test", 1,
            :pass, :pass, :pass, verdict,
            rob, 1.0, 0.5, 10, 10, 10,
            [1], true, 1.0, 0, 0, 0, 0, 0, 0, StableWindowEvidence[],
            0.7, rob, 1.0, 9, 9, Int(round(rob * 9)), Dict{Symbol,Int}(),
            rob, Dict{String,Any}[], DateTime(2024, 1, 1),
        )

        cert_good  = make_cert(:certified, 0.9)
        cert_med   = make_cert(:certified, 0.6)
        cert_frag  = make_cert(:fragile,   0.0)

        target = ChaosDesignTarget(
            min_amplitude=0.5, max_amplitude=2.0, min_spectral_flatness=0.0,
            min_robustness_score=0.5,
        )

        best    = ChaosDesignCandidate([0.3], cert_good, :ok, 1.0, 0.5, true,  0.9, 1.0, 1.0, 0.9)
        second  = ChaosDesignCandidate([0.2], cert_med,  :ok, 1.0, 0.5, true,  0.6, 1.0, 1.0, 0.6)
        bad_amp = ChaosDesignCandidate([0.4], cert_good, :ok, 0.01, 0.5, false, 0.9, 0.0, 1.0, 0.0)
        infeas  = ChaosDesignCandidate([0.5], cert_frag, :diverged, nothing, nothing, false, 0.0, 0.0, 0.0, 0.0)
        # Tie on objective between two feasible candidates: lex tiebreak
        tie_lo  = ChaosDesignCandidate([0.1], cert_good, :ok, 1.0, 0.5, true,  0.9, 1.0, 1.0, 0.9)
        tie_hi  = ChaosDesignCandidate([0.35], cert_good, :ok, 1.0, 0.5, true, 0.9, 1.0, 1.0, 0.9)

        ranked = DynamicsKit._cd_rank_candidates([infeas, bad_amp, second, best, tie_hi, tie_lo])

        # Feasible first
        @test ranked[1].feasible == true
        @test ranked[2].feasible == true
        @test ranked[3].feasible == true
        @test ranked[4].feasible == true

        # Among feasible: descending objective
        @test ranked[1].objective >= ranked[2].objective

        # Lex tiebreak: [0.1] < [0.3] < [0.35]
        tie_feasible = filter(c -> c.objective ≈ 0.9, ranked)
        @test tie_feasible[1].design_values == [0.1]
        @test tie_feasible[2].design_values == [0.3]
        @test tie_feasible[3].design_values == [0.35]

        # Infeasible last
        @test !ranked[end].feasible
    end

    # ── Budget and deduplication ────────────────────────────────────────────

    @testset "Budget enforcement and deduplication" begin
        var_b = ChaosDesignVariable(:amplitude, 2, 0.1, 0.5)
        signal_cfg = ChaosDesignSignalConfig(state_index=1, discrete_transient=20,
                                              discrete_samples=64, discrete_window=:none)
        target = ChaosDesignTarget()

        # Budget = 3: must never exceed it even with 2x2=4 coarse grid.
        design_cfg = ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b],
            target=target,
            signal=signal_cfg,
            samples_per_axis=4,
            refinement_levels=0,
            survivors_per_level=2,
            max_evaluations=3,
        )
        result = design_chaos_source(cat_map, design_cfg;
                                      initial_point=[sqrt(2)-1, sqrt(3)-1, 1.0])
        @test result.n_evaluated <= 3
        @test result.budget_reached
        @test length(result.candidates) == result.n_evaluated
        @test !(DynamicsKit._cd_coarse_grid(design_cfg) isa AbstractArray)
        seen = Set{Tuple{Vararg{Float64}}}()
        for c in result.candidates
            key = Tuple(c.design_values)
            @test key ∉ seen
            push!(seen, key)
        end

        refined = collect(DynamicsKit._cd_refined_grid(design_cfg, [0.1], [0.1]))
        refined_keys = Tuple.(refined)
        @test length(refined_keys) == length(unique(refined_keys))
        @test first(refined_keys) == (0.1,)
    end

    # ── Full integration test ───────────────────────────────────────────────

    @testset "Integration — design_chaos_source (cat map)" begin
        var_b = ChaosDesignVariable(:amplitude, 2, 0.5, 1.5)
        signal_cfg = ChaosDesignSignalConfig(
            state_index=3,
            discrete_transient=100,
            discrete_samples=128,
            discrete_sample_interval=1,
            discrete_window=:hann,
        )
        target = ChaosDesignTarget(
            min_amplitude=1.9,
            max_amplitude=2.1,
            min_spectral_flatness=0.0,
            min_robustness_score=0.0,
        )
        design_cfg = ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b],
            target=target,
            signal=signal_cfg,
            samples_per_axis=5,
            refinement_levels=1,
            survivors_per_level=2,
            max_evaluations=9,
        )

        log_lines = String[]
        result = design_chaos_source(cat_map, design_cfg;
                                      initial_point=[sqrt(2) - 1, sqrt(3) - 1, 1.0],
                                      log=msg -> push!(log_lines, msg))

        @test result isa ChaosDesignResult
        @test result.system_name == "cat_design_test"
        @test result.operating_band == (0.5, 1.0)
        @test result.operating_param_index == 1
        @test result.n_evaluated <= design_cfg.max_evaluations
        @test result.n_evaluated == length(result.candidates)
        @test !isnothing(result.best_candidate)
        @test result.ranked_candidates[1] === result.best_candidate
        @test result.best_candidate.feasible
        @test result.best_candidate.design_values[1] ≈ 1.0 atol=1e-12
        @test result.best_candidate.amplitude ≈ 2.0 atol=1e-12
        @test result.n_feasible <= result.n_evaluated
        @test length(result.ranked_candidates) == result.n_evaluated

        # Ranked list: feasible before infeasible
        first_infeasible = findfirst(c -> !c.feasible, result.ranked_candidates)
        if !isnothing(first_infeasible)
            @test all(result.ranked_candidates[i].feasible for i in 1:(first_infeasible - 1))
        end

        # All candidates have a certificate and a valid signal status
        for c in result.candidates
            @test c.certificate isa RobustChaosCertificate
            @test c.signal_status in (:ok, :diverged, :insufficient_samples)
            @test 0.0 <= c.robust_score
            @test 0.0 <= c.amplitude_score <= 1.0
            @test 0.0 <= c.flatness_score <= 1.0
            @test 0.0 <= c.objective <= 1.0
        end

        @test !isempty(log_lines)  # progress callback fired

        # chaos_design_summary
        summ = chaos_design_summary(result)
        @test summ["systemName"] == "cat_design_test"
        @test summ["nEvaluated"] == result.n_evaluated
        @test summ["nFeasible"] == result.n_feasible
    end

    # ── state_index out of range ────────────────────────────────────────────

    @testset "state_index out of range throws ArgumentError" begin
        var_b = ChaosDesignVariable(:amplitude, 2, 0.1, 0.5)
        bad_signal = ChaosDesignSignalConfig(state_index=99, discrete_transient=10,
                                              discrete_samples=64, discrete_window=:none)
        design_cfg = ChaosDesignConfig(
            operating_config=rc_cfg,
            variables=[var_b],
            target=ChaosDesignTarget(),
            signal=bad_signal,
            samples_per_axis=2,
            max_evaluations=4,
        )
        @test_throws ArgumentError design_chaos_source(cat_map, design_cfg)
    end

    # ── Metric semantics ────────────────────────────────────────────────────

    @testset "Amplitude and flatness score semantics" begin
        t = ChaosDesignTarget(min_amplitude=0.5, max_amplitude=2.0,
                               min_spectral_flatness=0.4, min_robustness_score=0.0)
        # In-range → 1.0
        @test DynamicsKit._cd_amplitude_score(1.0, t) == 1.0
        @test DynamicsKit._cd_flatness_score(0.6, t) == 1.0
        # Exactly at bounds → 1.0
        @test DynamicsKit._cd_amplitude_score(0.5, t) == 1.0
        @test DynamicsKit._cd_amplitude_score(2.0, t) == 1.0
        @test DynamicsKit._cd_flatness_score(0.4, t) == 1.0
        # Below min_amplitude → linear
        @test DynamicsKit._cd_amplitude_score(0.25, t) ≈ 0.5 atol=1e-12
        # Above max_amplitude → linear falloff
        @test DynamicsKit._cd_amplitude_score(4.0, t) ≈ 0.5 atol=1e-12
        # Below min_flatness → linear
        @test DynamicsKit._cd_flatness_score(0.2, t) ≈ 0.5 atol=1e-12
        # nothing → 0
        @test DynamicsKit._cd_amplitude_score(nothing, t) == 0.0
        @test DynamicsKit._cd_flatness_score(nothing, t) == 0.0
    end

    # ── Wire roundtrip ──────────────────────────────────────────────────────

    @testset "Serialization roundtrip" begin
        make_cert(verdict::Symbol, rob::Float64) = RobustChaosCertificate(
            0.5, 1.0, "rt_sys", 1,
            :pass, :pass, :pass, verdict,
            rob, 1.0, 0.5, 4, 4, 4,
            [1], true, 1.0, 0, 0, 0, 0, 0, 0, StableWindowEvidence[],
            0.7, rob, 1.0, 4, 4, 4, Dict{Symbol,Int}(:chaotic => 4),
            rob, Dict{String,Any}[Dict("layer"=>"overall","verdict"=>String(verdict))],
            DateTime(2024, 6, 15),
        )

        var_b = ChaosDesignVariable(:amplitude, 2, 0.1, 0.5)
        target = ChaosDesignTarget(min_amplitude=0.3, max_amplitude=1.5,
                                    min_spectral_flatness=0.2, min_robustness_score=0.5)
        cert = make_cert(:certified, 0.85)
        cand = ChaosDesignCandidate([0.3], cert, :ok, 1.2, 0.7, true,
                                     0.85, 1.0, 1.0, 0.85)
        result = ChaosDesignResult(
            "rt_sys",
            (0.5, 1.0),
            1,
            [var_b],
            target,
            [cand],
            [cand],
            cand,
            1, 1, false, 1,
            DateTime(2024, 6, 15),
        )

        plain = serialize_chaos_design_result(result)
        @test plain["format"] == "chaos-design-result-v1"
        @test plain["systemName"] == "rt_sys"
        @test plain["nEvaluated"] == 1
        @test plain["nFeasible"] == 1
        @test plain["budgetReached"] == false
        @test length(plain["candidates"]) == 1
        @test plain["candidates"][1]["feasible"] == true

        rt = deserialize_chaos_design_result(plain)
        @test rt isa ChaosDesignResult
        @test rt.system_name == "rt_sys"
        @test rt.operating_band == (0.5, 1.0)
        @test rt.operating_param_index == 1
        @test length(rt.variables) == 1
        @test rt.variables[1].name == :amplitude
        @test rt.variables[1].param_index == 2
        @test rt.variables[1].lower ≈ 0.1
        @test rt.variables[1].upper ≈ 0.5
        @test rt.target.min_amplitude ≈ 0.3
        @test rt.target.max_amplitude ≈ 1.5
        @test rt.target.min_spectral_flatness ≈ 0.2
        @test rt.n_evaluated == 1
        @test rt.n_feasible == 1
        @test !rt.budget_reached
        @test length(rt.candidates) == 1
        @test rt.candidates[1].feasible == true
        @test rt.candidates[1].design_values ≈ [0.3]
        @test rt.candidates[1].signal_status == :ok
        @test !isnothing(rt.candidates[1].amplitude)
        @test rt.candidates[1].amplitude ≈ 1.2
        @test !isnothing(rt.candidates[1].spectral_flatness_value)
        @test rt.candidates[1].spectral_flatness_value ≈ 0.7
        @test rt.candidates[1].objective ≈ 0.85
        @test !isnothing(rt.best_candidate)
        @test rt.best_candidate.design_values ≈ [0.3]
        @test rt.timestamp == DateTime(2024, 6, 15)

        representational_drift = deepcopy(plain)
        representational_drift["candidates"][1]["certificate"]["paramMax"] =
            nextfloat(1.0)
        representational_drift["candidates"][1]["robustScore"] =
            nextfloat(0.85)
        representational_drift["candidates"][1]["objective"] =
            nextfloat(0.85)
        @test deserialize_chaos_design_result(representational_drift).n_evaluated == 1

        inconsistent_score = deepcopy(plain)
        inconsistent_score["candidates"][1]["objective"] = 0.80
        @test_throws ErrorException deserialize_chaos_design_result(inconsistent_score)

        # Unsupported format throws
        @test_throws ErrorException deserialize_chaos_design_result(
            Dict{String,Any}("format" => "chaos-design-result-v99"))

        # max_amplitude Inf roundtrip
        t2 = ChaosDesignTarget(min_amplitude=0.0, max_amplitude=Inf)
        tgt_plain = DynamicsKit._serialize_chaos_design_target(t2)
        @test tgt_plain["maxAmplitude"] == "Inf"
        t2_rt = DynamicsKit._deserialize_chaos_design_target(tgt_plain)
        @test t2_rt.max_amplitude == Inf
    end

end  # @testset "chaos_design"
