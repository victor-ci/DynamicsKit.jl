using Dates: DateTime

@testset "robust_chaos_certificate" begin

    # --- Validation errors ---
    @testset "RobustChaosConfig validation" begin
        lya = LyapunovConfig(param_min=0.0, param_max=1.0, param_index=1, fixed_params=[0.5])
        bf  = BruteForceConfig(param_min=0.0, param_max=1.0, param_index=1, fixed_params=[0.5])
        ct  = ContinuationConfig(p_min=0.0, p_max=1.0, param_index=1)
        atl = AtlasConfig(brute_force=bf, continuation=ct, cache_enabled=false)
        bas = BasinsConfig(bif_param=0.5, param_index=1, fixed_params=[0.5],
                           x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                           iterations=4, max_period=2)

        # threshold out of range
        @test_throws AssertionError RobustChaosConfig(lyapunov=lya, atlas=atl, basins=bas,
                                                     min_lyapunov_positive_fraction=1.5)
        @test_throws AssertionError RobustChaosConfig(lyapunov=lya, atlas=atl, basins=bas,
                                                     min_lyapunov_resolved_fraction=-0.1)
        @test_throws AssertionError RobustChaosConfig(lyapunov=lya, atlas=atl, basins=bas,
                                                     min_chaotic_basin_fraction=2.0)
        @test_throws AssertionError RobustChaosConfig(lyapunov=lya, atlas=atl, basins=bas,
                                                     min_basin_resolved_fraction=-0.5)

        # atlas without brute_force
        @test_throws AssertionError RobustChaosConfig(
            lyapunov=lya,
            atlas=AtlasConfig(brute_force=nothing, cache_enabled=false),
            basins=bas
        )
        @test_throws AssertionError RobustChaosConfig(
            lyapunov=lya,
            atlas=AtlasConfig(brute_force=bf, continuation=nothing, cache_enabled=false),
            basins=bas
        )

        # atlas must search the same full interval
        @test_throws AssertionError RobustChaosConfig(
            lyapunov=lya,
            atlas=AtlasConfig(
                brute_force=BruteForceConfig(param_min=0.2, param_max=1.0),
                continuation=ct,
                cache_enabled=false,
            ),
            basins=bas
        )
        @test_throws AssertionError RobustChaosConfig(
            lyapunov=lya,
            atlas=AtlasConfig(
                brute_force=bf,
                continuation=ContinuationConfig(p_min=0.2, p_max=1.0),
                cache_enabled=false,
            ),
            basins=bas
        )

        # linked parameters must stay aligned across all layers
        linked_lya = LyapunovConfig(
            param_min=0.0, param_max=1.0, param_index=1,
            linked_param_indices=[2], fixed_params=[0.5, 0.5],
        )
        @test_throws AssertionError RobustChaosConfig(
            lyapunov=linked_lya, atlas=atl, basins=bas
        )

        # param_index mismatch (basins vs lyapunov)
        @test_throws AssertionError RobustChaosConfig(
            lyapunov=lya, atlas=atl,
            basins=BasinsConfig(bif_param=0.5, param_index=2, fixed_params=[0.5, 0.5],
                                x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                                iterations=4, max_period=2)
        )

        # bif_param outside lyapunov range
        @test_throws AssertionError RobustChaosConfig(
            lyapunov=lya, atlas=atl,
            basins=BasinsConfig(bif_param=1.5, param_index=1, fixed_params=[1.5],
                                x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                                iterations=4, max_period=2)
        )

        # valid config constructs without error
        @test RobustChaosConfig(lyapunov=lya, atlas=atl, basins=bas) isa RobustChaosConfig
    end

    # --- Verdict helper unit tests ---
    @testset "_rc_lyapunov_verdict" begin
        f = DynamicsKit._rc_lyapunov_verdict

        # All resolved, all positive → pass
        @test f(10, 10, 10, 1.0, 1.0) == :pass
        # All resolved, some non-positive → fail
        @test f(10, 10, 8, 1.0, 1.0) == :fail
        # Under-coverage, but not provably failing (unresolved could all be positive)
        @test f(10, 8, 8, 1.0, 1.0) == :inconclusive
        # Under-coverage AND already provably failing (one resolved non-positive → best-case < 1.0)
        @test f(10, 8, 7, 1.0, 1.0) == :fail
        # Fractional threshold: resolved=10, positive=7, min=0.7 → pass
        @test f(10, 10, 7, 0.7, 1.0) == :pass
        # Fractional threshold: resolved=10, positive=6, min=0.7 → fail
        @test f(10, 10, 6, 0.7, 1.0) == :fail
        # Zero total → inconclusive (can't prove failure)
        @test f(0, 0, 0, 1.0, 1.0) == :inconclusive
        # A zero resolved-coverage floor cannot turn absent evidence into a pass.
        @test f(10, 0, 0, 0.9, 0.0) == :inconclusive
    end

    @testset "_rc_basin_verdict" begin
        f = DynamicsKit._rc_basin_verdict
        @test f(9, 9, 9, 1.0, 1.0) == :pass
        @test f(9, 9, 8, 1.0, 1.0) == :fail
        @test f(9, 7, 7, 1.0, 1.0) == :inconclusive
        @test f(9, 7, 6, 1.0, 1.0) == :fail
        @test f(9, 0, 0, 0.8, 0.0) == :inconclusive
    end

    @testset "Base parameter matching uses canonical trailing zeros" begin
        f = DynamicsKit._robust_config_base_params_match
        @test f(Float64[], [0.0], Int[])
        @test f([0.0], Float64[], Int[])
        @test f(Float64[], [0.0, 0.0], Int[])
        @test !f(Float64[], [0.0, 2.0], Int[])
        @test f([0.5], [0.5, 0.0], [1])
        @test !f([0.5], [0.5, 2.0], [1])
        @test f([0.7], [0.7000000000000001], Int[])
        @test !f([0.7], [0.7000001], Int[])
    end

    @testset "Stable evidence keeps disjoint runs separate" begin
        samples = Tuple{Float64, Union{Bool, Nothing}}[
            (0.1, true),
            (0.2, true),
            (0.3, false),
            (0.7, true),
            (0.72, true),
            (1.0, true),
        ]
        spans = DynamicsKit._rc_contiguous_stable_spans(samples, 0.15)
        @test spans == [
            (param_min=0.1, param_max=0.2, count=2),
            (param_min=0.7, param_max=0.72, count=2),
            (param_min=1.0, param_max=1.0, count=1),
        ]
    end

    # --- Serialization round-trips ---
    @testset "Serialization — empty stable evidence" begin
        cert = RobustChaosCertificate(
            0.5, 1.0, "test_sys", 1,
            :pass, :pass, :pass, :certified,
            1.0, 1.0, 0.95, 5, 5, 5,
            [1, 2], true, 1.0, 0, 0, 0, 0, 0, 0, StableWindowEvidence[],
            0.7, 1.0, 1.0, 9, 9, 9, Dict(:chaotic => 9),
            1.0,
            [Dict{String, Any}("layer" => "overall", "verdict" => "certified")],
            DateTime(2024, 1, 1, 12, 0, 0)
        )
        plain = serialize_robust_chaos_certificate(cert)
        @test plain["format"] == "robust-chaos-certificate-v1"
        @test plain["overallVerdict"] == "certified"
        @test plain["stableEvidence"] == []
        @test plain["atlasSearchedPeriods"] == [1, 2]

        rt = deserialize_robust_chaos_certificate(plain)
        @test rt.overall_verdict == :certified
        @test rt.lyapunov_verdict == :pass
        @test rt.atlas_verdict == :pass
        @test rt.basin_verdict == :pass
        @test rt.lyapunov_n_total == 5
        @test rt.lyapunov_n_positive == 5
        @test isapprox(rt.lyapunov_min_resolved_exponent, 0.95)
        @test rt.atlas_searched_periods == [1, 2]
        @test rt.atlas_search_complete == true
        @test isapprox(rt.atlas_coverage_effort, 1.0)
        @test rt.atlas_unresolved_stability_count == 0
        @test isempty(rt.stable_evidence)
        @test isapprox(rt.basin_chaotic_fraction, 1.0)
        @test rt.basin_n_total == 9
        @test rt.basin_class_counts == Dict(:chaotic => 9)
        @test isapprox(rt.robustness_score, 1.0)
        @test length(rt.certificate_items) == 1
        @test rt.certificate_items[1]["layer"] == "overall"
        @test rt.timestamp == DateTime(2024, 1, 1, 12, 0, 0)
    end

    @testset "Serialization — nonempty stable evidence" begin
        evidence = [
            StableWindowEvidence("branch-1", "window-1", 1, 0.3, 0.7, 5),
            StableWindowEvidence("branch-2", "window-2", 2, 0.8, 0.9, 3),
        ]
        cert = RobustChaosCertificate(
            0.3, 0.9, "fragile_sys", 1,
            :fail, :fail, :fail, :fragile,
            0.0, 1.0, -1.2, 4, 4, 0,
            [1, 2], true, 0.0, 2, 0, 0, 2, 0, 0, evidence,
            0.5, 0.0, 1.0, 4, 4, 0, Dict(:periodic => 4),
            0.0, Dict{String, Any}[], DateTime(2024, 6, 1)
        )
        plain = serialize_robust_chaos_certificate(cert)
        @test plain["overallVerdict"] == "fragile"
        @test length(plain["stableEvidence"]) == 2
        ev0 = plain["stableEvidence"][1]
        @test ev0["branchId"] == "branch-1"
        @test ev0["windowId"] == "window-1"
        @test ev0["period"] == 1
        @test isapprox(ev0["paramMin"], 0.3)
        @test ev0["stableSampleCount"] == 5

        rt = deserialize_robust_chaos_certificate(plain)
        @test rt.overall_verdict == :fragile
        @test length(rt.stable_evidence) == 2
        @test rt.stable_evidence[1].branch_id == "branch-1"
        @test rt.stable_evidence[2].period == 2
        @test isapprox(rt.stable_evidence[2].param_max, 0.9)
        @test rt.basin_class_counts == Dict(:periodic => 4)

        missing_evidence_field = deepcopy(plain)
        delete!(missing_evidence_field["stableEvidence"][1], "paramMin")
        @test_throws ErrorException deserialize_robust_chaos_certificate(missing_evidence_field)

        invalid_evidence_bounds = deepcopy(plain)
        invalid_evidence_bounds["stableEvidence"][1]["paramMin"] = 1.0
        invalid_evidence_bounds["stableEvidence"][1]["paramMax"] = 0.0
        @test_throws ErrorException deserialize_robust_chaos_certificate(invalid_evidence_bounds)
    end

    @testset "Serialization — format error" begin
        @test_throws ErrorException deserialize_robust_chaos_certificate(
            Dict("format" => "wrong-format-v99"))
        @test_throws ErrorException deserialize_robust_chaos_certificate(
            Dict("format" => "robust-chaos-certificate-v1"))
    end

    # --- Integration: certification test (Arnold's cat map) ---
    # The cat map f(x,p)=(mod(x1+x2,1), mod(x1+2*x2,1)) has Lyapunov exponent ≈0.962 for all
    # non-periodic initial conditions. (0,0) is an unstable fixed point — passing a non-zero
    # initial_point ensures the brute-force orbit explores the chaotic component.
    @testset "Integration — certified (cat map)" begin
        cat_map = DiscreteMap(
            (x, p) -> SVector(mod(x[1] + x[2], 1.0), mod(x[1] + 2 * x[2], 1.0)),
            2, [:p], "cat_map_rc_test"
        )
        lya_cfg = LyapunovConfig(
            param_min=0.5, param_max=1.0, param_steps=2,
            param_index=1, fixed_params=[0.7],
            transient=50, iterations=150,
        )
        bf_cfg = BruteForceConfig(
            param_min=0.5, param_max=1.0, param_index=1,
            fixed_params=[0.7], param_steps=4,
            iterations=80, transient=40,
        )
        cont_cfg = ContinuationConfig(
            p_min=0.5, p_max=1.0, param_index=1,
            ds=0.05, dsmax=0.1, max_steps=50,
        )
        atlas_cfg = AtlasConfig(
            brute_force=bf_cfg, continuation=cont_cfg,
            periods=[1, 2], max_period=2, recon_steps=5,
            cache_enabled=false, threaded=false,
        )
        basins_cfg = BasinsConfig(
            bif_param=0.7, param_index=1, fixed_params=[0.7],
            x_min=0.1, x_max=0.9, x_steps=2,
            y_min=0.1, y_max=0.9, y_steps=2,
            iterations=4, max_period=2,
        )
        rc_cfg = RobustChaosConfig(lyapunov=lya_cfg, atlas=atlas_cfg, basins=basins_cfg)

        # Pass a non-zero initial_point to avoid the unstable fixed point at (0,0)
        cert = robust_chaos_certificate(
            cat_map, rc_cfg;
            initial_point=[sqrt(2) - 1, sqrt(3) - 1],
        )

        @test cert.overall_verdict == :certified
        @test cert.lyapunov_verdict == :pass
        @test cert.atlas_verdict == :pass
        @test cert.basin_verdict == :pass
        @test isempty(cert.stable_evidence)
        @test cert.lyapunov_positive_fraction > 0.9
        @test cert.lyapunov_min_resolved_exponent > 0.0
        @test cert.robustness_score > 0.0
        @test cert.atlas_search_complete
        @test cert.param_min ≈ 0.5
        @test cert.param_max ≈ 1.0
        @test cert.system_name == "cat_map_rc_test"
        @test length(cert.certificate_items) == 4  # lyapunov + atlas + basins + overall

        # Serialization round-trip on the live certificate
        plain = serialize_robust_chaos_certificate(cert)
        @test plain["format"] == "robust-chaos-certificate-v1"
        rt = deserialize_robust_chaos_certificate(plain)
        @test rt.overall_verdict == cert.overall_verdict
        @test rt.lyapunov_n_total == cert.lyapunov_n_total
        @test isapprox(rt.robustness_score, cert.robustness_score)
    end

    # --- Integration: fragile test (contracting fixed-point map) ---
    # f(x,p) = (0.3*(x1-p)+p, 0.3*x2) has a globally stable fixed point at (p,0) with
    # eigenvalues 0.3, 0.3. The atlas finds the stable period-1 branch, failing the certificate.
    @testset "Integration — fragile (stable fixed-point map)" begin
        fragile_map = DiscreteMap(
            (x, p) -> SVector(0.3 * (x[1] - p[1]) + p[1], 0.3 * x[2]),
            2, [:p], "fragile_rc_test"
        )
        lya_cfg2 = LyapunovConfig(
            param_min=0.3, param_max=0.8, param_steps=2,
            param_index=1, fixed_params=[0.5],
            transient=50, iterations=100,
        )
        bf_cfg2 = BruteForceConfig(
            param_min=0.3, param_max=0.8, param_index=1,
            fixed_params=[0.5], param_steps=4,
            iterations=80, transient=40,
        )
        cont_cfg2 = ContinuationConfig(
            p_min=0.3, p_max=0.8, param_index=1,
            ds=0.05, dsmax=0.1, max_steps=30,
        )
        atlas_cfg2 = AtlasConfig(
            brute_force=bf_cfg2, continuation=cont_cfg2,
            periods=[1], max_period=1, recon_steps=5,
            cache_enabled=false, threaded=false,
        )
        basins_cfg2 = BasinsConfig(
            bif_param=0.5, param_index=1, fixed_params=[0.5],
            x_min=0.0, x_max=1.0, x_steps=2,
            y_min=0.0, y_max=1.0, y_steps=2,
            iterations=4, max_period=2,
        )
        rc_cfg2 = RobustChaosConfig(lyapunov=lya_cfg2, atlas=atlas_cfg2, basins=basins_cfg2)

        cert2 = robust_chaos_certificate(fragile_map, rc_cfg2)

        @test cert2.overall_verdict == :fragile
        # Lyapunov layer: exponent ≈ log(0.3) < 0 for all samples → no chaotic candidates
        @test cert2.lyapunov_verdict == :fail
        @test cert2.lyapunov_positive_fraction < 0.1
        @test cert2.lyapunov_min_resolved_exponent < 0.0
        # Atlas layer: stable period-1 branch found
        @test cert2.atlas_verdict == :fail
        @test !isempty(cert2.stable_evidence)
        @test cert2.stable_evidence[1].period == 1
        @test cert2.basin_verdict == :fail
        @test cert2.basin_chaotic_fraction < 0.1
        @test cert2.robustness_score == 0.0
    end

    # --- Source-result reuse ---
    # The cat-map fixture is reused; each sub-testset pre-computes only the layers it
    # needs to supply, then verifies that the log contains the reuse message and that
    # the certificate verdict is unaffected by skipping the underlying sweep.

    @testset "Source-result reuse — setup (cat map configs)" begin
        # This testset exists only to define shared cat-map config values used by the
        # reuse tests below; it contains no assertions itself.
    end

    # Shared cat-map reuse configs (defined at module scope for the sub-testsets below).
    _rc_reuse_cat_map = DiscreteMap(
        (x, p) -> SVector(mod(x[1] + x[2], 1.0), mod(x[1] + 2 * x[2], 1.0)),
        2, [:p], "cat_map_reuse_test"
    )
    _rc_reuse_ip = [sqrt(2) - 1, sqrt(3) - 1]
    _rc_reuse_lya_cfg = LyapunovConfig(
        param_min=0.5, param_max=1.0, param_steps=3,
        param_index=1, fixed_params=[0.7],
        transient=40, iterations=100,
    )
    _rc_reuse_bf_cfg = BruteForceConfig(
        param_min=0.5, param_max=1.0, param_index=1,
        fixed_params=[0.7], param_steps=4,
        iterations=60, transient=30,
    )
    _rc_reuse_cont_cfg = ContinuationConfig(
        p_min=0.5, p_max=1.0, param_index=1,
        ds=0.05, dsmax=0.1, max_steps=40,
    )
    _rc_reuse_atlas_cfg = AtlasConfig(
        brute_force=_rc_reuse_bf_cfg, continuation=_rc_reuse_cont_cfg,
        periods=[1, 2], max_period=2, recon_steps=5,
        cache_enabled=false, threaded=false,
    )
    _rc_reuse_basins_cfg = BasinsConfig(
        bif_param=0.7, param_index=1, fixed_params=[0.7],
        x_min=0.1, x_max=0.9, x_steps=2,
        y_min=0.1, y_max=0.9, y_steps=2,
        iterations=4, max_period=2,
    )
    _rc_reuse_cfg = RobustChaosConfig(
        lyapunov=_rc_reuse_lya_cfg,
        atlas=_rc_reuse_atlas_cfg,
        basins=_rc_reuse_basins_cfg,
    )

    @testset "Source-result reuse — Lyapunov layer" begin
        precomputed_lya = lyapunov_diagram(_rc_reuse_cat_map, _rc_reuse_lya_cfg;
            initial_point=_rc_reuse_ip)

        log_msgs = String[]
        cert = robust_chaos_certificate(
            _rc_reuse_cat_map, _rc_reuse_cfg;
            initial_point=_rc_reuse_ip,
            lyapunov_result=precomputed_lya,
            log=msg -> push!(log_msgs, msg),
        )

        @test any(contains(m, "reusing supplied LyapunovDiagramResult") for m in log_msgs)
        # Layer 1 was reused; layers 2 and 3 ran fresh
        @test any(contains(m, "layer 2 — continuation-atlas search") for m in log_msgs)
        @test any(contains(m, "layer 3 — basin of attraction") for m in log_msgs)
        @test !any(contains(m, "reusing supplied AtlasResult") for m in log_msgs)
        @test cert.overall_verdict == :certified
        @test cert.lyapunov_n_total == precomputed_lya.params |> length
        resolved_indices = findall(eachindex(precomputed_lya.exponents)) do idx
            precomputed_lya.estimation_statuses[idx] == :ok &&
                isfinite(precomputed_lya.exponents[idx])
        end
        expected_positive = count(
            idx -> precomputed_lya.classifications[idx] == :chaotic_candidate,
            resolved_indices,
        )
        @test cert.lyapunov_n_resolved == length(resolved_indices)
        @test cert.lyapunov_n_positive == expected_positive
        @test cert.lyapunov_positive_fraction ≈ expected_positive / length(resolved_indices)
    end

    @testset "Source-result reuse — Atlas layer" begin
        precomputed_atlas = continuation_atlas(_rc_reuse_cat_map, _rc_reuse_atlas_cfg;
            initial_point=_rc_reuse_ip)
        precomputed_atlas.diagnostics["timeBudgetExceeded"] = "0"

        log_msgs = String[]
        cert = robust_chaos_certificate(
            _rc_reuse_cat_map, _rc_reuse_cfg;
            initial_point=_rc_reuse_ip,
            atlas_result=precomputed_atlas,
            log=msg -> push!(log_msgs, msg),
        )

        @test any(contains(m, "reusing supplied AtlasResult") for m in log_msgs)
        @test !any(contains(m, "reusing supplied LyapunovDiagramResult") for m in log_msgs)
        @test cert.overall_verdict == :certified
        @test cert.atlas_search_complete
        @test Set(cert.atlas_searched_periods) == Set(Int[Int(x) for x in get(precomputed_atlas.diagnostics, "periods", Int[])])
    end

    @testset "Source-result reuse — malformed Lyapunov vectors throw" begin
        malformed = LyapunovDiagramResult(
            [0.5, 2 / 3, 5 / 6, 1.0],
            [0.9, 0.9, 0.9, 0.9],
            [:chaotic_candidate],
            fill(:ok, 4),
            fill(100, 4),
            _rc_reuse_lya_cfg.neutral_tolerance,
            _rc_reuse_cat_map.name,
            :p,
            DateTime(2024, 1, 1),
        )
        @test_throws ArgumentError DynamicsKit._rc_validate_lyapunov_reuse(
            malformed,
            _rc_reuse_cat_map,
            _rc_reuse_cfg,
        )
    end

    @testset "Source-result reuse — Lyapunov and atlas layers" begin
        precomputed_lya = lyapunov_diagram(_rc_reuse_cat_map, _rc_reuse_lya_cfg;
            initial_point=_rc_reuse_ip)
        precomputed_atlas = continuation_atlas(_rc_reuse_cat_map, _rc_reuse_atlas_cfg;
            initial_point=_rc_reuse_ip)

        log_msgs = String[]
        cert = robust_chaos_certificate(
            _rc_reuse_cat_map, _rc_reuse_cfg;
            initial_point=_rc_reuse_ip,
            lyapunov_result=precomputed_lya,
            atlas_result=precomputed_atlas,
            log=msg -> push!(log_msgs, msg),
        )

        @test any(contains(m, "reusing supplied LyapunovDiagramResult") for m in log_msgs)
        @test any(contains(m, "reusing supplied AtlasResult") for m in log_msgs)
        @test cert.overall_verdict == :certified
    end

    @testset "Source-result reuse — default compute path unchanged" begin
        # Calling without any precomputed results must still work exactly as before.
        cert = robust_chaos_certificate(
            _rc_reuse_cat_map, _rc_reuse_cfg;
            initial_point=_rc_reuse_ip,
        )
        @test cert.overall_verdict == :certified
        @test cert.lyapunov_verdict == :pass
        @test cert.atlas_verdict == :pass
        @test cert.basin_verdict == :pass
    end

    @testset "Source-result reuse — mismatched system name throws" begin
        wrong_sys = DiscreteMap(
            (x, p) -> SVector(mod(x[1] + x[2], 1.0), mod(x[1] + 2 * x[2], 1.0)),
            2, [:p], "different_system_name"
        )
        good_lya = lyapunov_diagram(_rc_reuse_cat_map, _rc_reuse_lya_cfg)
        @test_throws ArgumentError robust_chaos_certificate(
            wrong_sys, _rc_reuse_cfg;
            lyapunov_result=good_lya,
        )
    end

    @testset "Source-result reuse — mismatched interval throws" begin
        good_lya = lyapunov_diagram(_rc_reuse_cat_map, _rc_reuse_lya_cfg)

        wrong_lya_cfg = LyapunovConfig(
            param_min=0.4, param_max=0.9, param_steps=3,
            param_index=1, fixed_params=[0.7],
        )
        wrong_bf = BruteForceConfig(param_min=0.4, param_max=0.9, param_index=1, fixed_params=[0.7], param_steps=4)
        wrong_cont = ContinuationConfig(p_min=0.4, p_max=0.9, param_index=1)
        wrong_atlas_cfg = AtlasConfig(brute_force=wrong_bf, continuation=wrong_cont, periods=[1, 2], max_period=2, cache_enabled=false, threaded=false)
        wrong_cfg = RobustChaosConfig(
            lyapunov=wrong_lya_cfg,
            atlas=AtlasConfig(brute_force=wrong_bf, continuation=wrong_cont, periods=[1, 2], max_period=2, recon_steps=5, cache_enabled=false, threaded=false),
            basins=BasinsConfig(bif_param=0.65, param_index=1, fixed_params=[0.7], x_min=0.1, x_max=0.9, x_steps=2, y_min=0.1, y_max=0.9, y_steps=2, iterations=4, max_period=2),
        )
        # good_lya covers [0.5, 1.0]; wrong_cfg expects [0.4, 0.9]
        @test_throws ArgumentError robust_chaos_certificate(
            _rc_reuse_cat_map, wrong_cfg;
            lyapunov_result=good_lya,
        )
    end

    @testset "Source-result reuse — mismatched atlas periods throws" begin
        # Atlas searched only period 1; config requests periods [1, 2]
        bf_p1 = BruteForceConfig(param_min=0.5, param_max=1.0, param_index=1, fixed_params=[0.7], param_steps=4, iterations=60, transient=30)
        atlas_p1_cfg = AtlasConfig(brute_force=bf_p1, continuation=_rc_reuse_cont_cfg, periods=[1], max_period=1, recon_steps=5, cache_enabled=false, threaded=false)
        precomputed_p1_atlas = continuation_atlas(_rc_reuse_cat_map, atlas_p1_cfg; initial_point=_rc_reuse_ip)

        # _rc_reuse_cfg requests periods [1, 2]; the pre-computed result searched only [1]
        @test_throws ArgumentError robust_chaos_certificate(
            _rc_reuse_cat_map, _rc_reuse_cfg;
            atlas_result=precomputed_p1_atlas,
        )
    end

end
