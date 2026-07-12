@testset "Automatic continuation atlas" begin
    @testset "Coverage fraction is zero when no branch intervals overlap a window" begin
        coverage = DynamicsKit._atlas_interval_coverage_fraction(
            1.5,
            1.7,
            DynamicsKit.AtlasBranchRecord[],
            "window-empty",
            1
        )

        @test coverage == 0.0
    end

    @testset "Branch switching diagnostics summarize eligible special-point probes" begin
        disabled = DynamicsKit._atlas_branch_switching_summary(AtlasConfig(branch_switching=false), Dict{String, Any}[])
        @test disabled["branchSwitchingRequested"] == false
        @test disabled["branchSwitchingApplied"] == false
        @test disabled["branchSwitchingStatus"] == "disabled"

        attempted = DynamicsKit._atlas_branch_switching_summary(
            AtlasConfig(branch_switching=true),
            [Dict{String, Any}("attemptCount" => 1, "newBranchCount" => 0, "specialPointCount" => 1)]
        )
        @test attempted["branchSwitchingRequested"] == true
        @test attempted["branchSwitchingApplied"] == true
        @test attempted["branchSwitchingStatus"] == "attempted_no_new_branches"
        @test attempted["branchSwitchingAttemptCount"] == 1
        @test attempted["branchSwitchingNewBranchCount"] == 0

        applied = DynamicsKit._atlas_branch_switching_summary(
            AtlasConfig(branch_switching=true),
            [Dict{String, Any}("attemptCount" => 2, "newBranchCount" => 1, "specialPointCount" => 2)]
        )
        @test applied["branchSwitchingStatus"] == "applied"
        @test applied["branchSwitchingNewBranchCount"] == 1
    end

    @testset "Neighbor seed reuse cache is bounded and distance-gated" begin
        window = DynamicsKit.AtlasWindow(
            "window-seed-cache",
            1,
            0.4,
            0.6,
            1,
            1.0,
            :periodic,
            [1],
            1.0,
            :partial,
            Dict{String, Any}()
        )
        far_window = DynamicsKit.AtlasWindow(
            "window-seed-cache-far",
            1,
            0.9,
            1.0,
            1,
            1.0,
            :periodic,
            [1],
            1.0,
            :partial,
            Dict{String, Any}()
        )
        cont_config = ContinuationConfig(p_min=0.0, p_max=1.0)
        atlas_config = AtlasConfig(
            reuse_neighbor_seeds=true,
            neighbor_seed_max_entries=2,
            neighbor_seed_max_distance_fraction=0.2,
            neighbor_seed_max_points=1
        )
        cache = DynamicsKit._atlas_seed_cache()

        empty_points, empty_diag = DynamicsKit._atlas_cached_seed_points(cache, window, 0.5, cont_config, atlas_config)
        @test isnothing(empty_points)
        @test empty_diag["status"] == "empty_cache"

        store_diag = DynamicsKit._atlas_update_seed_cache!(cache, 1, 0.45, [[0.1], [0.2], [0.3]], atlas_config)
        hit_points, hit_diag = DynamicsKit._atlas_cached_seed_points(cache, window, 0.5, cont_config, atlas_config)
        far_points, far_diag = DynamicsKit._atlas_cached_seed_points(cache, far_window, 0.95, cont_config, atlas_config)
        summary = DynamicsKit._atlas_seed_reuse_summary(
            atlas_config,
            [DynamicsKit._atlas_seed_reuse_event(hit_diag, store_diag)]
        )

        @test length(cache[1]) == 2
        @test !isnothing(hit_points)
        @test length(hit_points) == 1
        @test hit_diag["hit"] == true
        @test isnothing(far_points)
        @test far_diag["status"] == "miss_distance"
        @test summary["neighborSeedReuseStatus"] == "applied"
        @test summary["neighborSeedReuseApplied"] == true
        @test summary["neighborSeedReuseReusedSeedCount"] == 1
    end

    @testset "Adaptive reconnaissance inserts provenance-tagged midpoint samples" begin
        sys = DiscreteMap((x, p) -> SVector(p[1] < 0.5 ? 0.0 : 1.0), 1, [:a], "Step Map")
        bf_config = BruteForceConfig(
            param_min=0.0,
            param_max=1.0,
            param_steps=2,
            iterations=12,
            transient=8,
            fixed_params=[0.0]
        )
        atlas_config = AtlasConfig(
            periods=[1],
            adaptive_recon=true,
            adaptive_recon_max_samples=1,
            adaptive_recon_max_depth=1,
            adaptive_recon_closure_gradient_factor=0.5
        )
        samples = DynamicsKit.AtlasReconSample[
            DynamicsKit.AtlasReconSample(
                0.0, :periodic, 1, 0.95, [0.0], [[0.0]], [0.0], [0.0],
                Dict{String, Any}("threshold" => 1.0, "reconSource" => "uniform")
            ),
            DynamicsKit.AtlasReconSample(
                1.0, :nonperiodic, 0, 0.0, [2.0], [[1.0]], [1.0], [0.0],
                Dict{String, Any}("threshold" => 1.0, "reconSource" => "uniform")
            )
        ]

        refined, diagnostics = DynamicsKit._atlas_adaptive_reconnaissance(
            sys,
            samples,
            [0.0],
            bf_config,
            atlas_config,
            [1]
        )
        adaptive_samples = [sample for sample in refined if get(sample.diagnostics, "reconSource", "") == "adaptive"]

        @test length(refined) == 3
        @test diagnostics["status"] == "applied"
        @test diagnostics["adaptiveSampleCount"] == 1
        @test diagnostics["reasonCounts"]["classification-change"] == 1
        @test length(adaptive_samples) == 1
        @test adaptive_samples[1].param == 0.5
        @test "classification-change" in adaptive_samples[1].diagnostics["adaptiveReasons"]
    end

    @testset "Atlas reconnaissance preserves serial/threaded sample ordering" begin
        sys = DiscreteMap((x, p) -> SVector(p[1]), 1, [:a], "Recon ordering map")
        bf_config = BruteForceConfig(
            param_min=0.0,
            param_max=1.0,
            param_steps=6,
            iterations=8,
            transient=4,
            fixed_params=[0.0]
        )
        serial_config = AtlasConfig(
            periods=[1],
            recon_steps=7,
            recon_precision=1e-12,
            threaded=false
        )
        threaded_config = AtlasConfig(
            periods=[1],
            recon_steps=7,
            recon_precision=1e-12,
            threaded=true
        )

        serial_samples = DynamicsKit._atlas_reconnaissance(
            sys,
            [0.0],
            bf_config,
            serial_config,
            [1]
        )
        threaded_samples = DynamicsKit._atlas_reconnaissance(
            sys,
            [0.0],
            bf_config,
            threaded_config,
            [1]
        )
        sample_signature = samples -> [
            (
                sample.param,
                sample.classification,
                sample.best_period,
                sample.confidence,
                sample.closure_errors,
                sample.orbit_center,
                sample.orbit_span,
                get(sample.diagnostics, "reconSource", nothing),
                get(sample.diagnostics, "adaptiveDepth", nothing)
            )
            for sample in samples
        ]

        @test length(serial_samples) == 7
        @test sample_signature(threaded_samples) == sample_signature(serial_samples)
        @test [sample.param for sample in threaded_samples] == sort([sample.param for sample in threaded_samples])
    end

    @testset "Atlas reuses exact uniform reconnaissance as brute-force cloud" begin
        sys = henon_map()
        bf_config = BruteForceConfig(
            param_min=0.1,
            param_max=0.2,
            param_steps=4,
            iterations=18,
            transient=10,
            fixed_params=[0.3]
        )
        atlas_config = AtlasConfig(
            periods=[1],
            recon_steps=5,
            recon_precision=1e-4,
            threaded=false
        )
        samples = DynamicsKit._atlas_reconnaissance(
            sys,
            [0.3],
            bf_config,
            atlas_config,
            [1]
        )

        reused, diagnostics = DynamicsKit._atlas_bruteforce_result(
            sys,
            bf_config,
            atlas_config,
            [1],
            samples
        )
        fresh = brute_force_diagram(sys, bf_config)

        @test diagnostics["reused"] == true
        @test diagnostics["reason"] == "exact_uniform_recon"
        @test diagnostics["uniformSampleCount"] == 5
        @test diagnostics["pointCount"] == length(fresh.params)
        @test reused.params == fresh.params
        @test reused.points ≈ fresh.points

        mismatch_config = AtlasConfig(
            periods=[1],
            recon_steps=4,
            recon_precision=1e-4,
            threaded=false
        )
        missed, missed_diagnostics = DynamicsKit._atlas_recon_bruteforce_reuse_candidate(
            sys,
            bf_config,
            mismatch_config,
            [1],
            samples
        )
        @test isnothing(missed)
        @test missed_diagnostics["reused"] == false
        @test missed_diagnostics["reason"] == "recon_steps_mismatch"
    end

    @testset "Branch switching special-point candidates are budgeted and labeled" begin
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[
                (param=0.0, x1=0.0),
                (param=0.5, x1=0.5),
                (param=1.0, x1=1.0)
            ], Any[
                (param=0.5, type=:fold),
                (param=0.75, type=:period_doubling)
            ]),
            1,
            "Dummy",
            :p,
            DynamicsKit.now()
        )
        record = DynamicsKit.AtlasBranchRecord(
            "branch-specials",
            branch,
            0.5,
            "window-specials",
            1.0,
            0.0,
            1.0,
            Dict{String, Any}()
        )

        one = DynamicsKit._atlas_branch_switching_specialpoints(record, 1)
        all_candidates = DynamicsKit._atlas_branch_switching_specialpoints(record, 4)

        @test length(one) == 1
        @test one[1].label == "fold"
        @test one[1].param == 0.5
        @test one[1].branch_point_index == 2
        @test length(all_candidates) == 2
        @test all_candidates[2].label == "period_doubling"
    end

    @testset "Branch switching probes are budgeted and duplicate-safe" begin
        sys = DiscreteMap((x, p) -> SVector(p[1]), 1, [:p], "Parameter fixed-point map")
        points = Any[(param=p, x1=p) for p in range(0.0, 1.0; length=51)]
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(points, Any[(param=0.5, type=:fold)]),
            1,
            "Parameter fixed-point map",
            :p,
            DynamicsKit.now()
        )
        window = DynamicsKit.AtlasWindow(
            "window-switch",
            1,
            0.0,
            1.0,
            1,
            1.0,
            :periodic,
            [1],
            1.0,
            :recovered,
            Dict{String, Any}()
        )
        sample = DynamicsKit.AtlasReconSample(
            0.5,
            :periodic,
            1,
            1.0,
            [0.0],
            [[0.5]],
            [0.5],
            [0.0],
            Dict("threshold" => 1e-6)
        )
        record = DynamicsKit.AtlasBranchRecord(
            "branch-switch-source",
            branch,
            0.5,
            window.id,
            1.0,
            0.0,
            1.0,
            Dict{String, Any}()
        )
        cont_config = ContinuationConfig(
            p_min=0.0,
            p_max=1.0,
            ds=0.02,
            dsmax=0.04,
            dsmin=1e-6,
            max_steps=40,
            newton_tol=1e-10,
            newton_max_iter=25,
            detect_bifurcation=1
        )
        atlas_config = AtlasConfig(
            branch_switching=true,
            branch_switching_max_special_points=1,
            branch_switching_max_branches=1,
            branch_switching_max_steps=20,
            branch_switching_max_seed_candidates=3,
            cache_enabled=false,
            threaded=false
        )

        new_records, diagnostics = DynamicsKit._atlas_branch_switching_followups(
            sys,
            window,
            [sample],
            [record],
            [0.5],
            cont_config,
            atlas_config,
            Ref(1)
        )

        @test isempty(new_records)
        @test diagnostics["requested"] == true
        @test diagnostics["applied"] == true
        @test diagnostics["attemptCount"] == 1
        @test diagnostics["specialPointCount"] == 1
        @test diagnostics["status"] == "attempted_no_new_branches"
        @test diagnostics["attempts"][1]["duplicateBranchCount"] >= 1
    end

    @testset "Union coverage scoring combines multiple branch segments" begin
        window = DynamicsKit.AtlasWindow(
            "window-a",
            1,
            0.0,
            1.0,
            4,
            0.9,
            :periodic,
            [1, 2],
            1.0,
            :partial,
            Dict{String, Any}()
        )

        branch_a = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[
                (param=0.0, x1=0.1),
                (param=0.49, x1=0.12)
            ], Any[]),
            1,
            "Dummy",
            :p,
            DynamicsKit.now()
        )
        branch_b = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[
                (param=0.51, x1=0.18),
                (param=1.0, x1=0.21)
            ], Any[]),
            1,
            "Dummy",
            :p,
            DynamicsKit.now()
        )

        records = [
            DynamicsKit.AtlasBranchRecord("branch-a", branch_a, 0.25, window.id, 0.49, 0.0, 0.49, Dict{String, Any}()),
            DynamicsKit.AtlasBranchRecord("branch-b", branch_b, 0.75, window.id, 0.49, 0.51, 1.0, Dict{String, Any}())
        ]

        coverage = DynamicsKit._atlas_window_coverage_fraction(window, records)
        summary = DynamicsKit._atlas_coverage_summary([window], records, 0.95)
        gaps = DynamicsKit._atlas_gap_records([window], records, 0.95, 1)

        @test coverage > 0.95
        @test summary["covered"] == 1
        @test summary["partial"] == 0
        @test isempty(gaps)
    end

    @testset "Geometry-aware coverage reduces mismatched branch support" begin
        window = DynamicsKit.AtlasWindow(
            "window-geometry",
            1,
            0.0,
            1.0,
            2,
            0.9,
            :periodic,
            [1, 2],
            1.0,
            :partial,
            Dict{String, Any}()
        )
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[
                (param=0.0, x1=0.1),
                (param=1.0, x1=0.2)
            ], Any[]),
            1,
            "Dummy",
            :p,
            DynamicsKit.now()
        )
        record = DynamicsKit.AtlasBranchRecord(
            "branch-low-geometry",
            branch,
            0.5,
            window.id,
            0.25,
            0.0,
            1.0,
            Dict{String, Any}("geometryCoverageScore" => 0.25)
        )

        coverage = DynamicsKit._atlas_window_coverage_fraction(window, [record])
        thresholded = DynamicsKit._atlas_interval_coverage_intervals(
            window.param_min,
            window.param_max,
            [record],
            window.id,
            window.period;
            min_geometry_score=0.9
        )
        gaps = DynamicsKit._atlas_gap_records([window], [record], 0.9, 1)

        @test coverage ≈ 0.25
        @test isempty(thresholded)
        @test length(gaps) == 1
        @test gaps[1].param_min == window.param_min
        @test gaps[1].param_max == window.param_max
        @test gaps[1].reason == :insufficient_coverage
    end

    @testset "Atlas branch geometry diagnostics compare recon clouds to branch orbits" begin
        sys = DiscreteMap((x, p) -> SVector(p[1]), 1, [:p], "Constant test map")
        window = DynamicsKit.AtlasWindow(
            "window-cloud",
            1,
            0.0,
            1.0,
            1,
            1.0,
            :periodic,
            [1],
            1.0,
            :partial,
            Dict{String, Any}()
        )
        sample = DynamicsKit.AtlasReconSample(
            0.5,
            :periodic,
            1,
            1.0,
            [0.0],
            [[0.5]],
            [0.5],
            [0.0],
            Dict("threshold" => 1e-6)
        )
        matching_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[(param=0.5, x1=0.5)], Any[]),
            1,
            "Constant test map",
            :p,
            DynamicsKit.now()
        )
        mismatched_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[(param=0.5, x1=2.0)], Any[]),
            1,
            "Constant test map",
            :p,
            DynamicsKit.now()
        )

        good = DynamicsKit._atlas_branch_geometry_diagnostics(sys, matching_branch, window, [sample], [0.0], Int[])
        bad = DynamicsKit._atlas_branch_geometry_diagnostics(sys, mismatched_branch, window, [sample], [0.0], Int[])

        @test good["geometryCoverageStatus"] == "evaluated"
        @test good["geometryCoverageScore"] > 0.99
        @test bad["geometryCoverageStatus"] == "evaluated"
        @test bad["geometryCoverageScore"] == 0.0
        @test bad["geometrySamples"][1]["normalizedDistance"] > 0.0
    end

    @testset "Hidden higher-period closure signatures become atlas probe windows" begin
        harmonic = DynamicsKit.AtlasReconSample(
            0.10,
            :periodic,
            2,
            1.0,
            [1.0, 1e-12, 1.0, 1e-12, 1.0, 1e-12, 1.0, 1e-12],
            [[0.0, 0.0]],
            [0.0, 0.0],
            [1.0, 1.0],
            Dict("threshold" => 1e-3)
        )
        hidden = DynamicsKit.AtlasReconSample(
            0.24,
            :nonperiodic,
            0,
            0.0,
            [6.8, 2.5, 7.3, 0.6, 6.7, 2.6, 7.4, 1e-12],
            [[-50.0, 22.7]],
            [-50.0, 22.7],
            [12.0, 0.1],
            Dict("threshold" => 1e-3)
        )
        samples = [harmonic, hidden]
        indices = DynamicsKit._atlas_hidden_period_sample_indices(samples, 8)
        probe_windows = DynamicsKit._atlas_hidden_period_probe_windows(
            samples,
            AtlasConfig(periods=[8], window_merge_gap=1),
            [8],
            DynamicsKit.AtlasWindow[]
        )

        @test indices == [2]
        @test length(probe_windows) == 1
        @test probe_windows[1].period == 8
        @test probe_windows[1].classification == :probe_hidden_period
        @test probe_windows[1].support == 1
        @test haskey(probe_windows[1].diagnostics, "probeType")
    end

    @testset "Hidden-period probes are suppressed only by overlapping same-period windows" begin
        exact_a = DynamicsKit.AtlasReconSample(
            0.10,
            :periodic,
            8,
            1.0,
            [5.0, 4.0, 3.0, 2.0, 5.0, 4.0, 3.0, 1e-12],
            [[0.0, 0.0]],
            [0.0, 0.0],
            [1.0, 1.0],
            Dict("threshold" => 1e-3)
        )
        exact_b = DynamicsKit.AtlasReconSample(
            0.20,
            :periodic,
            8,
            1.0,
            [5.0, 4.0, 3.0, 2.0, 5.0, 4.0, 3.0, 1e-12],
            [[0.1, 0.0]],
            [0.1, 0.0],
            [1.0, 1.0],
            Dict("threshold" => 1e-3)
        )
        hidden = DynamicsKit.AtlasReconSample(
            0.50,
            :nonperiodic,
            0,
            0.0,
            [6.8, 2.5, 7.3, 0.6, 6.7, 2.6, 7.4, 1e-12],
            [[-50.0, 22.7]],
            [-50.0, 22.7],
            [12.0, 0.1],
            Dict("threshold" => 1e-3)
        )
        samples = [exact_a, exact_b, hidden]
        existing_window = DynamicsKit.AtlasWindow(
            "existing-period-8",
            8,
            0.08,
            0.22,
            2,
            0.9,
            :periodic,
            [1, 2],
            1.0,
            :recovered,
            Dict{String, Any}()
        )

        probe_windows = DynamicsKit._atlas_hidden_period_probe_windows(
            samples,
            AtlasConfig(periods=[8], window_merge_gap=1),
            [8],
            [existing_window]
        )

        @test length(probe_windows) == 1
        @test probe_windows[1].period == 8
        @test probe_windows[1].diagnostics["candidateIndices"] == [3]

        overlapping_window = DynamicsKit.AtlasWindow(
            "overlapping-period-8",
            8,
            0.45,
            0.55,
            1,
            0.9,
            :periodic,
            [3],
            1.0,
            :recovered,
            Dict{String, Any}()
        )
        suppressed = DynamicsKit._atlas_hidden_period_probe_windows(
            samples,
            AtlasConfig(periods=[8], window_merge_gap=1),
            [8],
            [existing_window, overlapping_window]
        )

        @test isempty(suppressed)
    end

    @testset "Hénon atlas recovers a period-1 branch window" begin
        sys = henon_map()
        config = AtlasConfig(
            periods=[1],
            brute_force=BruteForceConfig(
                param_min=0.0,
                param_max=0.35,
                param_steps=30,
                iterations=120,
                transient=80,
                fixed_params=[0.3]
            ),
            continuation=ContinuationConfig(
                p_min=0.0,
                p_max=0.35,
                ds=0.01,
                dsmax=0.03,
                dsmin=1e-6,
                max_steps=160,
                newton_tol=1e-10,
                newton_max_iter=40,
                detect_bifurcation=1
            ),
            recon_steps=20,
            recon_precision=1e-4,
            window_min_support=2,
            seed_points_per_window=4,
            coverage_threshold=0.2,
            threaded=false
        )

        result = continuation_atlas(sys, config; params=[0.3])

        @test result isa AtlasResult
        @test result.system_name == "Hénon"
        @test result.param_name == :a
        @test length(result.recon_samples) == 20
        @test !isempty(result.windows)
        @test any(window.period == 1 for window in result.windows)
        @test !isempty(result.branch_records)
        @test any(record.branch.period == 1 for record in result.branch_records)
        @test result.coverage_summary["windowCount"] == length(result.windows)
        @test result.coverage_summary["branchCount"] == length(result.branch_records)
        @test !isempty(atlas_branches(result))
        @test all(length(record.branch.branch) > 1 for record in result.branch_records)
        @test result.diagnostics["branchSwitchingRequested"] == false
        @test result.diagnostics["branchSwitchingApplied"] == false
        @test result.diagnostics["branchSwitchingStatus"] == "disabled"
        @test result.diagnostics["neighborSeedReuseRequested"] == false
        @test result.diagnostics["neighborSeedReuseApplied"] == false
        @test result.diagnostics["neighborSeedReuseStatus"] == "disabled"
        @test result.diagnostics["adaptiveRecon"]["status"] == "disabled"
        @test haskey(result.diagnostics, "windowAttemptCount")
        @test haskey(result.diagnostics, "refinementAttempts")
        @test all(haskey(window.diagnostics, "attempts") for window in result.windows)
        @test all(haskey(window.diagnostics, "coverage") for window in result.windows)
    end

    @testset "Atlas library cache stores and reloads results when a cache file is provided" begin
        sys = henon_map()
        config = AtlasConfig(
            periods=[1],
            brute_force=BruteForceConfig(
                param_min=0.0,
                param_max=0.35,
                param_steps=24,
                iterations=120,
                transient=80,
                fixed_params=[0.3]
            ),
            continuation=ContinuationConfig(
                p_min=0.0,
                p_max=0.35,
                ds=0.01,
                dsmax=0.03,
                dsmin=1e-6,
                max_steps=160,
                newton_tol=1e-10,
                newton_max_iter=40,
                detect_bifurcation=1
            ),
            recon_steps=18,
            recon_precision=1e-4,
            window_min_support=2,
            seed_points_per_window=4,
            coverage_threshold=0.2,
            threaded=false,
            cache_enabled=true
        )

        mktempdir() do temp_dir
            cache_file = joinpath(temp_dir, "atlas-cache.jld2")
            first_log = String[]
            second_log = String[]

            first_result = continuation_atlas(
                sys,
                config;
                params=[0.3],
                cache_key="henon-cache-test",
                cache_file=cache_file,
                log=message -> push!(first_log, message)
            )
            @test isfile(cache_file)
            @test first_result.diagnostics["cacheEnabled"] == true
            @test first_result.diagnostics["cacheHit"] == false
            @test first_result.diagnostics["cacheFile"] == cache_file
            @test any(occursin("Atlas cache store", message) for message in first_log)

            second_result = continuation_atlas(
                sys,
                config;
                params=[0.3],
                cache_key="henon-cache-test",
                cache_file=cache_file,
                log=message -> push!(second_log, message)
            )
            @test second_result.diagnostics["cacheHit"] == true
            @test second_result.diagnostics["cacheFile"] == cache_file
            @test length(second_result.windows) == length(first_result.windows)
            @test length(second_result.branch_records) == length(first_result.branch_records)
            @test any(occursin("Atlas cache hit", message) for message in second_log)
        end
    end

    @testset "Continuous atlas recovers a radial-oscillator branch window" begin
        sys = radial_oscillator()
        config = AtlasConfig(
            periods=[1],
            brute_force=BruteForceConfig(
                param_min=0.2,
                param_max=0.5,
                param_steps=12,
                iterations=50,
                transient=24,
                fixed_params=[0.36]
            ),
            continuation=ContinuationConfig(
                p_min=0.2,
                p_max=0.5,
                ds=0.02,
                dsmax=0.04,
                dsmin=1e-6,
                max_steps=120,
                newton_tol=1e-8,
                newton_max_iter=30,
                detect_bifurcation=1
            ),
            recon_steps=12,
            recon_precision=1e-3,
            window_min_support=2,
            seed_points_per_window=3,
            coverage_threshold=0.15,
            threaded=false
        )

        result = continuation_atlas(sys, config; params=[0.36], solver=DynamicsKit.Tsit5(), reltol=1e-8, abstol=1e-8)

        @test result isa AtlasResult
        @test result.system_name == "Radial Oscillator"
        @test result.param_name == :μ
        @test !isempty(result.windows)
        @test any(window.period == 1 for window in result.windows)
        @test !isempty(result.branch_records)
        @test any(record.branch.period == 1 for record in result.branch_records)
        @test all(record.coverage_score >= 0.0 for record in result.branch_records)
        @test result.coverage_summary["windowCount"] == length(result.windows)
    end

    @testset "Atlas continuous auto-refine densifies coarse radial branches" begin
        sys = radial_oscillator()
        cont_config = ContinuationConfig(
            p_min=0.30,
            p_max=0.42,
            ds=0.03,
            dsmax=0.04,
            max_steps=4,
            newton_tol=1e-8,
            detect_bifurcation=0
        )
        coarse_branch = continuation_branch(
            sys,
            cont_config;
            initial_point=[-sqrt(0.36)],
            params=[0.36],
            solver=DynamicsKit.Tsit5(),
            reltol=1e-8,
            abstol=1e-8
        )
        atlas_cfg = AtlasConfig(
            periods=[1],
            continuation=cont_config,
            auto_refine_sparse_branches=true,
            auto_refine_max_passes=1,
            threaded=false
        )

        refined_branch, diag = DynamicsKit._atlas_maybe_auto_refine_branch(
            sys,
            coarse_branch,
            [0.36],
            cont_config,
            atlas_cfg;
            solver=DynamicsKit.Tsit5(),
            reltol=1e-8,
            abstol=1e-8
        )

        @test diag["autoRefineIntervalsDetected"] >= 1
        @test diag["autoRefineApplied"] == true
        @test diag["autoRefineReason"] == "densified"
        @test length(refined_branch.branch) > length(coarse_branch.branch)
    end

    @testset "Atlas records refinement diagnostics for unresolved windows" begin
        sys = henon_map()
        config = AtlasConfig(
            periods=[1],
            brute_force=BruteForceConfig(
                param_min=0.0,
                param_max=0.35,
                param_steps=20,
                iterations=100,
                transient=70,
                fixed_params=[0.3]
            ),
            continuation=ContinuationConfig(
                p_min=0.0,
                p_max=0.35,
                ds=0.005,
                dsmax=0.005,
                dsmin=1e-6,
                max_steps=1,
                newton_tol=1e-10,
                newton_max_iter=25,
                detect_bifurcation=1
            ),
            recon_steps=18,
            recon_precision=1e-4,
            window_min_support=2,
            seed_points_per_window=3,
            coverage_threshold=0.9,
            max_refinement_depth=1,
            threaded=false
        )

        result = continuation_atlas(sys, config; params=[0.3])

        @test result isa AtlasResult
        @test haskey(result.diagnostics, "refinementAttempts")
        @test result.diagnostics["refinementAttempts"] >= 1
        @test !isempty(result.gaps)
        @test any(haskey(window.diagnostics, "refinementDiagnostics") for window in result.windows)
        @test any(haskey(gap.diagnostics, "windowId") || haskey(gap.diagnostics, "parentWindowId") for gap in result.gaps)
    end

    @testset "Phase 2: atlas global fallback bounds derive from reconnaissance cloud" begin
        # Synthetic samples whose support points live in [10, 20] in state[1].
        # A hardcoded fallback box (e.g. [-3, 3]) would be useless for a system
        # whose attractor lives well outside that range, so the global fallback
        # must derive bounds that contain the reconnaissance cloud.
        samples = DynamicsKit.AtlasReconSample[]
        # Deterministic cloud spanning state[1] ∈ [10, 15] so the fallback
        # bounds assertion is RNG-independent.
        cloud_x = [10.0, 11.25, 12.5, 13.75, 15.0]
        for p in 1.0:0.1:1.5
            cloud = [[cloud_x[k], 1.0] for k in 1:5]
            push!(samples, DynamicsKit.AtlasReconSample(
                p, :periodic, 1, 0.5,
                Float64[1.0],     # closure_errors (single dummy entry)
                cloud,
                [12.5, 1.0],      # center
                [10.0, 0.0],      # span
                Dict("threshold" => 1.0)
            ))
        end

        fb_min, fb_max = DynamicsKit._atlas_global_fallback_bounds(samples, 2, 0.15)
        @test length(fb_min) == 2
        @test length(fb_max) == 2
        # Should NOT default to [-3, 3]; should bracket the cloud's state[1] range.
        @test fb_min[1] < 10.0
        @test fb_max[1] > 15.0
        @test fb_min[1] > -3.0   # i.e. derived from the cloud, not the static fallback

        # True last resort: no samples at all → the static [-3, 3] box is the
        # documented last-resort behaviour.
        last_resort_min, last_resort_max = DynamicsKit._atlas_global_fallback_bounds(
            DynamicsKit.AtlasReconSample[], 2, 0.15
        )
        @test last_resort_min == [-3.0, -3.0]
        @test last_resort_max == [3.0, 3.0]
    end

    @testset "Window confidence aggregation: shrinkage + clamping" begin
        agg = DynamicsKit._atlas_aggregate_confidence

        # Empty input yields zero, no division by zero
        @test agg(Float64[]) == 0.0

        # Output is always in [0, 1] even with adversarial inputs (the shrinkage
        # math keeps the result bounded, but the clamp guards against future bugs).
        @test 0.0 <= agg([0.0]) <= 1.0
        @test 0.0 <= agg([1.0]) <= 1.0
        @test 0.0 <= agg(fill(0.9, 100)) <= 1.0

        # Small-support correction: a single sample at 0.9 confidence ranks BELOW
        # a window with many samples averaging 0.7 — a raw mean would let the
        # singleton window dominate the stronger multi-sample signal.
        single_strong = agg([0.9])
        many_medium = agg(fill(0.7, 20))
        @test single_strong < many_medium

        # Large support converges toward the mean (the prior fades out).
        @test isapprox(agg(fill(0.8, 1000)), 0.8; atol=1e-2)

        # Pure 0.5 input is a fixed point of the shrinkage (prior is 0.5).
        @test isapprox(agg(fill(0.5, 5)), 0.5; atol=1e-12)
    end
end
