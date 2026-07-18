@testset "Continuation branches" begin
    @testset "Hénon period-1 continuation" begin
        sys = henon_map()

        # Start near the known period-1 fixed point at a=0.3
        # Fixed point: x = (1-sqrt(1-4a*0.3))/(2a) for period-1
        # At a=0.3: approximate fixed point around (0.63, 0.19)
        config = ContinuationConfig(
            p_min = 0.0,
            p_max = 1.4,
            ds = 0.01,
            dsmax = 0.05,
            max_steps = 500,
            newton_tol = 1e-10,
            detect_bifurcation = 3
        )

        # Find the fixed point first at a=0.3
        a0 = 0.3
        # Analytical: x* satisfies x = 1 + 0.3x - a*x^2, i.e. a*x^2 + 0.7x - 1 = 0
        x_star = (-0.7 + sqrt(0.49 + 4 * a0)) / (2 * a0)
        y_star = 0.3 * x_star

        trim_diag = Ref{Any}(nothing)
        result = continuation_branch(sys, config;
                                     initial_point = [x_star, y_star],
                                     params = [a0],
                                     trim_to_minimal_period = true,
                                     on_trim = diag -> (trim_diag[] = diag))

        @test result isa BranchResult
        @test result.period == 1
        @test result.system_name == "Hénon"
        @test trim_diag[]["reason"] == "period_one"
        @test trim_diag[]["applied"] == false

        # Branch should have points
        branch = result.branch
        @test length(branch) > 10

        # Parameter range should span a significant portion
        pars = [pt.param for pt in branch.branch]
        @test minimum(pars) < 0.5
        @test maximum(pars) > 1.0

        base_params = [a0]
        param_index = something(findfirst(==(result.param_name), sys.param_names), config.param_index)
        stored_stability = Bool[pt.stable for pt in branch.branch]
        recomputed_stability = Bool[
            first(branch_stability(
                sys,
                DynamicsKit._branch_point_state(pt, state_dim(sys)),
                inject_param(base_params, param_index, Float64(pt.param), config.linked_param_indices),
                result.period
            ))
            for pt in branch.branch
        ]
        @test stored_stability == recomputed_stability
        @test any(!, stored_stability)

        restored = DynamicsKit._deserialize_branch_result(DynamicsKit._serialize_branch_result(result))
        restored_stability = Bool[pt.stable for pt in restored.branch.branch]
        @test restored_stability == recomputed_stability

        # Check that bifurcation points were detected
        @test length(branch.specialpoint) > 0
    end

    @testset "Hénon period-2 continuation" begin
        sys = henon_map()

        # Period-2 orbit near a=1.0: approximately x ≈ [-0.47, 0.32]
        config = ContinuationConfig(
            p_min = 0.0, p_max = 1.4,
            ds = 0.005, dsmax = 0.03,
            max_steps = 500, newton_tol = 1e-10,
            detect_bifurcation = 2
        )

        # Find a period-2 point via Newton
        a0 = 1.0
        sys_h = henon_map()
        F2 = x -> begin
            sv = SVector{2}(x)
            sv = sys_h.f(sv, [a0])
            sv = sys_h.f(sv, [a0])
            Array(sv) .- x
        end

        # Start from approximate period-2 point
        x0 = [-0.5, 0.3]
        point, converged = DynamicsKit._newton_ad(F2, x0, 1e-12, 50)
        @test converged

        result = continuation_branch(sys, config, 2;
                                     initial_point = point,
                                     params = [a0])
        @test result.period == 2
        @test length(result.branch) > 5

        traces = DynamicsKit._branch_plot_traces(sys, result; orbital=1, params=[a0])
        @test length(traces) == 2
        @test all(length(trace.values) == length(result.branch) for trace in traces)
        @test any(abs(traces[1].values[i] - traces[2].values[i]) > 1e-4 for i in eachindex(traces[1].values))
    end

    @testset "Discrete continuation canonicalizes phase-swapped representatives" begin
        sys = DiscreteMap((x, _p) -> SVector(x[2], x[1]), 2, [:a], "Phase swap")
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[
                (param=0.10, x1=0.0, x2=10.0, stable=true),
                (param=0.11, x1=10.1, x2=0.1, stable=true),
                (param=0.12, x1=0.2, x2=10.2, stable=true),
                (param=0.13, x1=10.3, x2=0.3, stable=true)
            ], Any[]),
            2,
            sys.name,
            :a,
            DynamicsKit.Dates.now()
        )

        aligned = DynamicsKit._canonicalize_branch_representatives(sys, branch, [0.0], Int[])
        xs = [pt.x1 for pt in aligned.branch.branch]
        ys = [pt.x2 for pt in aligned.branch.branch]
        @test xs ≈ [0.0, 0.1, 0.2, 0.3]
        @test ys ≈ [10.0, 10.1, 10.2, 10.3]
    end

    @testset "Continuation overlay labels delayed unstable traces" begin
        stable_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[
                (param=0.0, x1=0.0, stable=true),
                (param=0.1, x1=0.1, stable=true),
            ], Any[]),
            1,
            "Synthetic",
            :a,
            DynamicsKit.Dates.now()
        )
        unstable_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[
                (param=0.0, x1=1.0, stable=false),
                (param=0.1, x1=1.1, stable=false),
            ], Any[]),
            1,
            "Synthetic",
            :a,
            DynamicsKit.Dates.now()
        )
        brute_force = BruteForceResult(
            [0.0, 0.1],
            reshape([0.0, 0.1], 2, 1),
            "Synthetic",
            :a,
            DynamicsKit.Dates.now()
        )

        overlay = plot_overlay(
            brute_force,
            [stable_branch, unstable_branch];
            stable_linewidth=0.85,
            unstable_linewidth=0.75,
        )
        labels = [series[:label] for series in overlay.series_list]
        line_widths = [series[:linewidth] for series in overlay.series_list if series[:seriestype] == :path]

        @test count(==("Stable"), labels) == 1
        @test count(==("Unstable"), labels) == 1
        @test 0.85 in line_widths
        @test 0.75 in line_widths
    end

    @testset "Discrete auto-refine detects sparse short branches" begin
        sys = DiscreteMap((_, p) -> SVector(p[1]), 1, [:a], "Param fixed")
        config = ContinuationConfig(
            p_min=0.0,
            p_max=0.12,
            ds=0.03,
            dsmax=0.04,
            max_steps=24,
            newton_tol=1e-10,
            detect_bifurcation=0
        )

        coarse = continuation_branch(sys, config; initial_point=[0.06], params=[0.06])
        intervals = DynamicsKit._discrete_branch_refinement_intervals(coarse, config)
        @test !isempty(intervals)

        refined = auto_refine_branch(sys, coarse, config; params=[0.06], max_passes=1)
        @test length(refined.branch) > length(coarse.branch)
        @test minimum(pt.param for pt in refined.branch.branch) <= minimum(pt.param for pt in coarse.branch.branch) + 1e-10
        @test maximum(pt.param for pt in refined.branch.branch) >= maximum(pt.param for pt in coarse.branch.branch) - 1e-10
    end

    @testset "Discrete refinement catches dotted tails below the dsmax floor" begin
        # System-agnostic: the detector only reads parameter spacing, so this holds for any map.
        sys = DiscreteMap((x, _p) -> SVector(x[1]), 1, [:a], "Generic 1D")
        config = ContinuationConfig(
            p_min=-0.5, p_max=1.0, ds=0.001, dsmax=0.05,
            max_steps=200, newton_tol=1e-10, detect_bifurcation=0
        )
        # Dense interior (gap 0.001) ending in a single sparse tail step of 0.038 = 0.76*dsmax —
        # the ramped tail PALC leaves behind. A fixed 0.9*dsmax gap floor would miss it.
        pars = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.044]
        pts = Any[(param=p, x1=p, stable=true) for p in pars]
        branch = BranchResult(DynamicsKit.CombinedBranchResult(pts, Any[]), 1, sys.name, :a, DynamicsKit.Dates.now())

        @test isempty(DynamicsKit._discrete_branch_refinement_intervals(branch, config; gap_dsmax_factor=0.9))
        intervals = DynamicsKit._discrete_branch_refinement_intervals(branch, config)
        @test !isempty(intervals)
        @test any(lo <= 0.044 <= hi for (lo, hi) in intervals)
    end

    @testset "Short-branch re-sweep skips densely sampled branches" begin
        sys = DiscreteMap((x, _p) -> SVector(x[1]), 1, [:a], "Generic 1D")
        config = ContinuationConfig(
            p_min=-0.5, p_max=1.0, ds=0.001, dsmax=0.05,
            max_steps=200, newton_tol=1e-10, detect_bifurcation=0
        )
        # Dense onset cluster (gap 0.002) then a sparse tail: small median gap, large max gap.
        # The median-gap gate confines refinement to the sparse tail instead of re-sweeping
        # the whole branch past its support.
        pars = vcat(collect(0.0:0.002:0.05), [0.09])
        pts = Any[(param=p, x1=p, stable=true) for p in pars]
        branch = BranchResult(DynamicsKit.CombinedBranchResult(pts, Any[]), 1, sys.name, :a, DynamicsKit.Dates.now())

        gated = DynamicsKit._discrete_branch_refinement_intervals(branch, config)
        ungated = DynamicsKit._discrete_branch_refinement_intervals(branch, config; short_branch_min_median_gap_factor=0.0)
        @test !isempty(gated)
        @test !isempty(ungated)
        # The ungated re-sweep reaches further below the branch's parameter support.
        @test minimum(lo for (lo, _) in ungated) < minimum(lo for (lo, _) in gated)
        @test minimum(lo for (lo, _) in gated) > 0.0
    end

    @testset "Continuous short-branch re-sweep skips densely sampled branches" begin
        # Same median-gap gate on the continuous path (non-Hénon, any ODE branch).
        config = ContinuationConfig(
            p_min=0.0, p_max=1.0, ds=0.01, dsmax=0.05,
            max_steps=200, newton_tol=1e-8, detect_bifurcation=0
        )
        pars = vcat(collect(0.30:0.005:0.33), [0.375])
        pts = Any[(param=p, x1=p, stable=true) for p in pars]
        branch = BranchResult(DynamicsKit.CombinedBranchResult(pts, Any[]), 1, "Generic ODE", :a, DynamicsKit.Dates.now())

        gated = DynamicsKit._continuous_branch_refinement_intervals(branch, config)
        ungated = DynamicsKit._continuous_branch_refinement_intervals(branch, config; short_branch_min_median_gap_factor=0.0)
        @test !isempty(gated)
        @test !isempty(ungated)
        @test minimum(lo for (lo, _) in ungated) < minimum(lo for (lo, _) in gated)
    end

    @testset "Direct continuation minimal-period trimming rejects aliases" begin
        sys = DiscreteMap((_x, _p) -> SVector(0.0), 1, [:a], "Zero map")
        config = ContinuationConfig(
            p_min = -0.1,
            p_max = 0.1,
            ds = 0.01,
            dsmax = 0.02,
            max_steps = 10,
            newton_tol = 1e-10,
            detect_bifurcation = 0
        )

        raw = continuation_branch(sys, config, 2;
                                  initial_point=[0.0],
                                  params=[0.0],
                                  trim_to_minimal_period=false)
        @test raw.period == 2
        @test length(raw.branch) > 0

        trim_diag = Ref{Any}(nothing)
        @test_throws ErrorException continuation_branch(sys, config, 2;
                                                        initial_point=[0.0],
                                                        params=[0.0],
                                                        trim_to_minimal_period=true,
                                                        on_trim=diag -> (trim_diag[] = diag))
        @test trim_diag[]["reason"] == "all_dropped"
        @test trim_diag[]["droppedCount"] == length(raw.branch)
        @test trim_diag[]["lowerPeriods"]["1"] == length(raw.branch)
    end

    @testset "Trim/splice helpers accept concretely-typed branch-point vectors" begin
        # A branch wrapping a raw single-direction BifurcationKit ContResult yields a
        # concretely-typed Vector{NamedTuple} from `_branch_points` (not a Vector{Any}).
        # The trim and splice helpers must accept it — a `::Vector{Any}` signature would
        # MethodError on single-direction period≥2 branches.
        concrete_points = [(param=0.1, x1=0.5), (param=0.2, x1=0.6), (param=0.4, x1=0.7)]
        @test !(concrete_points isa Vector{Any})

        intervals = DynamicsKit._param_intervals_for_indices(concrete_points, [1, 2])
        @test length(intervals) == 1
        @test intervals[1]["paramMin"] == 0.1
        @test intervals[1]["paramMax"] == 0.2

        # First arg concretely typed also exercises the _branch_state_scale /
        # run-score / orient helpers.
        spliced = DynamicsKit._splice_refined_segment_points(
            concrete_points,
            [(param=0.25, x1=0.65)]
        )
        @test !isempty(spliced)
    end

    @testset "Branch diagnostics expose residuals and multipliers" begin
        sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:a], "Linear map")
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(
                Any[(param=0.5, x1=0.0), (param=1.2, x1=0.0)],
                Any[]
            ),
            1,
            sys.name,
            :a,
            DynamicsKit.Dates.now()
        )

        diagnostics = continuation_branch_diagnostics(sys, branch, [0.0])

        @test diagnostics["status"] == "ok"
        @test diagnostics["pointCount"] == 2
        @test diagnostics["evaluatedIndices"] == [1, 2]
        @test diagnostics["residualNorms"] ≈ [0.0, 0.0]
        @test diagnostics["maxMultiplierModuli"] ≈ [0.5, 1.2]
        @test diagnostics["maxMultiplierModulus"] ≈ 1.2
        @test diagnostics["stabilityFlags"] == [true, false]
        @test diagnostics["stableCount"] == 1
        @test diagnostics["unstableCount"] == 1
        @test diagnostics["multiplierSpectra"][2][1]["abs"] ≈ 1.2
    end

    @testset "Branch diagnostics expose switching-event proximity" begin
        sys = DiscreteMap(
            (x, p) -> SVector(p[1]),
            1,
            [:a],
            "Border map";
            switching_events=[SwitchingEvent("border", (x, p) -> x[1]; tolerance=1e-8)]
        )
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(
                Any[(param=-0.1, x1=-0.1), (param=0.0, x1=0.0), (param=0.1, x1=0.1)],
                Any[]
            ),
            1,
            sys.name,
            :a,
            DynamicsKit.Dates.now()
        )

        diagnostics = continuation_branch_diagnostics(
            sys,
            branch,
            [0.0];
            include_residuals=false,
            include_multipliers=false
        )
        switching = diagnostics["switchingEvents"]

        @test switching["eventCount"] == 1
        @test switching["sampledPointCount"] == 3
        @test switching["nearEventCount"] == 1
        @test switching["nearestEvent"] == "border"
        @test switching["minDistance"] == 0.0
    end

    @testset "Period-N plotting keeps orbit phases aligned" begin
        swap_map = DiscreteMap(
            (x, p) -> SVector(x[2], x[1]),
            2,
            [:a],
            "Phase swap"
        )
        points = Any[
            (param=0.1, x1=0.0, x2=1.0, stable=true),
            (param=0.2, x1=1.1, x2=0.1, stable=true),
            (param=0.3, x1=0.2, x2=1.2, stable=true)
        ]
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(points, Any[]),
            2,
            "Phase swap",
            :a,
            DynamicsKit.Dates.now()
        )

        traces = DynamicsKit._branch_plot_traces(swap_map, branch; orbital=1, params=[0.0])
        @test length(traces) == 2
        @test traces[1].values ≈ [0.0, 0.1, 0.2]
        @test traces[2].values ≈ [1.0, 1.1, 1.2]

        sqdistance, matched = DynamicsKit._phase_state_sqdistance([1.0, NaN, 3.0], [4.0, 5.0, Inf])
        @test sqdistance ≈ 9.0
        @test matched == 1

        previous_orbit = [
            [0.0, 100.0],
            [10.0, NaN],
            [20.0, 300.0]
        ]
        current_orbit = [
            [10.2, Inf],
            [20.1, 299.8],
            [0.1, 100.3]
        ]
        shift = DynamicsKit._orbit_phase_alignment_shift(previous_orbit, current_orbit)
        aligned_orbit = DynamicsKit._align_orbit_phases(previous_orbit, current_orbit)
        @test shift == 2
        @test aligned_orbit[1] === current_orbit[3]
        @test aligned_orbit[2] === current_orbit[1]
        @test aligned_orbit[3] === current_orbit[2]

        cycle3_map = DiscreteMap(
            (x, p) -> SVector(x[2], x[3], x[1]),
            3,
            [:a],
            "Phase cycle"
        )
        cycle3_points = Any[
            (param=0.1, x1=0.0, x2=10.0, x3=20.0, stable=true),
            (param=0.2, x1=20.1, x2=0.1, x3=10.1, stable=true),
            (param=0.3, x1=0.2, x2=10.2, x3=20.2, stable=true)
        ]
        cycle3_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(cycle3_points, Any[]),
            3,
            "Phase cycle",
            :a,
            DynamicsKit.Dates.now()
        )

        cycle3_traces = DynamicsKit._branch_plot_traces(cycle3_map, cycle3_branch; orbital=1, params=[0.0])
        @test length(cycle3_traces) == 3
        @test cycle3_traces[1].values ≈ [0.0, 0.1, 0.2]
        @test cycle3_traces[2].values ≈ [10.0, 10.1, 10.2]
        @test cycle3_traces[3].values ≈ [20.0, 20.1, 20.2]

        zero_period_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(points, Any[]),
            0,
            "Phase swap",
            :a,
            DynamicsKit.Dates.now()
        )
        zero_period_traces = DynamicsKit._branch_plot_traces(swap_map, zero_period_branch; orbital=1, params=[0.0])
        @test length(zero_period_traces) == 1
        @test zero_period_traces[1].values ≈ [0.0, 1.1, 0.2]
    end

    @testset "Branch plot traces break on phase discontinuities" begin
        identity_map = DiscreteMap(
            (x, p) -> SVector(x[1], x[2]),
            2,
            [:a],
            "Identity"
        )

        # A single large state jump between an otherwise-smooth run. This is what
        # happens when continuation hops between coexisting orbits at one step but
        # is locally smooth elsewhere — the colpitts P5 and P4 branches in the
        # refined session show this pattern.
        jump_points = Any[
            (param=0.10, x1=0.00, x2=0.10, stable=true),
            (param=0.11, x1=0.01, x2=0.11, stable=true),
            (param=0.12, x1=0.02, x2=0.12, stable=true),
            (param=0.13, x1=0.03, x2=0.13, stable=true),
            (param=0.14, x1=5.00, x2=5.10, stable=true),
            (param=0.15, x1=5.01, x2=5.11, stable=true),
            (param=0.16, x1=5.02, x2=5.12, stable=true),
            (param=0.17, x1=5.03, x2=5.13, stable=true)
        ]
        jump_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(jump_points, Any[]),
            1,
            "Single jump",
            :a,
            DynamicsKit.Dates.now()
        )
        jump_traces = DynamicsKit._branch_plot_traces(identity_map, jump_branch; orbital=1, params=[0.0])
        @test length(jump_traces) == 1
        @test jump_traces[1].breaks == Set([4])

        # A genuinely smooth branch must NOT pick up spurious breaks.
        smooth_points = Any[
            (param=0.10, x1=0.00, x2=1.00, stable=true),
            (param=0.11, x1=0.01, x2=1.01, stable=true),
            (param=0.12, x1=0.02, x2=1.02, stable=true),
            (param=0.13, x1=0.03, x2=1.03, stable=true),
            (param=0.14, x1=0.04, x2=1.04, stable=true),
            (param=0.15, x1=0.05, x2=1.05, stable=true)
        ]
        smooth_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(smooth_points, Any[]),
            1,
            "Smooth",
            :a,
            DynamicsKit.Dates.now()
        )
        smooth_traces = DynamicsKit._branch_plot_traces(identity_map, smooth_branch; orbital=1, params=[0.0])
        @test all(isempty(trace.breaks) for trace in smooth_traces)

        # Period-N branches with a coexisting-orbit hop in the middle — analog of
        # the colpitts P2 zigzag where a smooth segment runs into a region where
        # continuation alternates between two distinct orbits. Phase alignment
        # cannot resolve this; the break detector must mark each hop.
        swap_map = DiscreteMap(
            (x, p) -> SVector(x[2], x[1]),
            2,
            [:a],
            "Phase swap"
        )
        mixed_points = Any[
            # smooth period-2 orbit (phases A=[0,1] and B=[1,0])
            (param=0.10, x1=0.000, x2=1.000, stable=true),
            (param=0.11, x1=0.010, x2=1.010, stable=true),
            (param=0.12, x1=0.020, x2=1.020, stable=true),
            (param=0.13, x1=0.030, x2=1.030, stable=true),
            # hop to a distant coexisting orbit
            (param=0.14, x1=8.000, x2=9.000, stable=true),
            (param=0.15, x1=8.010, x2=9.010, stable=true),
            (param=0.16, x1=8.020, x2=9.020, stable=true),
            (param=0.17, x1=8.030, x2=9.030, stable=true)
        ]
        mixed_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(mixed_points, Any[]),
            2,
            "Coexisting orbit hop",
            :a,
            DynamicsKit.Dates.now()
        )
        mixed_traces = DynamicsKit._branch_plot_traces(swap_map, mixed_branch; orbital=1, params=[0.0])
        @test length(mixed_traces) == 2
        @test 4 in mixed_traces[1].breaks
        @test 4 in mixed_traces[2].breaks
        # Smooth stretches must not pick up adjacent-step breaks.
        @test !(1 in mixed_traces[1].breaks)
        @test !(6 in mixed_traces[1].breaks)

        # _contiguous_runs respects break_after by splitting consecutive indices.
        runs = DynamicsKit._contiguous_runs(collect(1:5); break_after=Set([2]))
        @test runs == UnitRange{Int}[1:2, 3:5]

        # Near-constant branches still surface a single sharp jump: q25 of step sizes
        # collapses to zero, so the threshold falls back entirely to the
        # `range_fraction × state_range` floor. Guards against future regressions in
        # the thresholding math (e.g. an accidental early-return when q25 == 0).
        @test DynamicsKit._phase_jump_break_indices([
            [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]
        ]) == Set([4])

        # Fully constant branches must produce zero breaks: state_range == 0 means
        # every threshold candidate is zero, but there is also nothing to break.
        @test isempty(DynamicsKit._phase_jump_break_indices([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ]))

        # Bounded breaks: with `max_breaks` forced low, a signal that
        # produces many candidate jumps gets pruned to exactly the cap.
        # The signal has small ramp steps (~0.01) most of the time with
        # periodic large jumps (~10) every 5 indices, so q25 stays small,
        # 8 × q25 stays well below the jump magnitude, and the detector
        # reports many candidates. The production default is `typemax(Int)`
        # (no pruning); the injectable parameter lets the test verify the
        # pruning code path without being trivially satisfied.
        spiked = Float64[0.01 * k + (k % 5 == 0 ? 10.0 : 0.0) for k in 1:1000]
        unpruned = DynamicsKit._phase_jump_break_indices([spiked]; max_breaks=typemax(Int))
        @test length(unpruned) > 100   # many candidates exceed the threshold
        pruned = DynamicsKit._phase_jump_break_indices([spiked]; max_breaks=7)
        @test length(pruned) == 7
        @test pruned ⊆ unpruned   # the kept breaks are a subset of all candidates
    end

    @testset "Continuous-time period-1 continuation" begin
        sys = radial_oscillator()

        config = ContinuationConfig(
            p_min = 0.15,
            p_max = 0.8,
            ds = 0.02,
            dsmax = 0.05,
            max_steps = 200,
            newton_tol = 1e-8,
            detect_bifurcation = 1
        )

        result = continuation_branch(sys, config; params=[0.36], n_initial=6)

        @test result isa BranchResult
        @test result.period == 1
        @test result.system_name == "Radial Oscillator"
        @test length(result.branch) > 10

        pars = [pt.param for pt in result.branch.branch]
        vals = [getproperty(pt, :x1) for pt in result.branch.branch]
        @test minimum(pars) < 0.3
        @test maximum(pars) > 0.6
        @test all(isfinite.(vals))
        @test all(vals .< 0.0)
    end

    @testset "Continuous-time variational Poincaré derivatives" begin
        sys = radial_oscillator()
        μ = 0.36
        section_point = [-sqrt(μ)]
        branch = BranchResult(
            DynamicsKit.CombinedBranchResult(Any[(param=μ, x1=section_point[1])], Any[]),
            1,
            sys.name,
            :μ,
            DynamicsKit.Dates.now()
        )

        finite = continuation_branch_diagnostics(
            sys,
            branch,
            [μ];
            ode_jacobian_method=:finite_difference,
            solver=DynamicsKit.Tsit5(),
            reltol=1e-9,
            abstol=1e-9
        )
        variational = continuation_branch_diagnostics(
            sys,
            branch,
            [μ];
            ode_jacobian_method=:variational,
            solver=DynamicsKit.Tsit5(),
            reltol=1e-9,
            abstol=1e-9
        )

        expected_multiplier = exp(-4π * μ)
        @test finite["odeJacobianMethod"] == "finite_difference"
        @test variational["odeJacobianMethod"] == "variational"
        @test variational["residualFailureCount"] == 0
        @test variational["multiplierFailureCount"] == 0
        @test variational["maxMultiplierModuli"][1] ≈ expected_multiplier rtol=2e-2 atol=1e-4
        @test finite["maxMultiplierModuli"][1] ≈ variational["maxMultiplierModuli"][1] rtol=5e-2 atol=1e-4

        config = ContinuationConfig(
            p_min=0.30,
            p_max=0.42,
            ds=0.03,
            dsmax=0.04,
            max_steps=40,
            newton_tol=1e-8,
            detect_bifurcation=0,
            ode_jacobian_method=:variational
        )
        result = continuation_branch(sys, config; initial_point=section_point, params=[μ])
        @test result isa BranchResult
        @test result.period == 1
        @test length(result.branch) > 3
    end

    @testset "Continuous-time Poincaré crossing diagnostics" begin
        sys = radial_oscillator()
        params = [0.36]
        sample = DynamicsKit._collect_poincare_points(
            sys,
            params;
            initial_point=[1.0, 0.1],
            crossings=2,
            transient=1,
            projected=true,
            return_diagnostics=true
        )
        diag = sample.diagnostics
        @test length(sample.points) == 2
        @test diag["crossingsRequested"] == 2
        @test diag["transientCrossings"] == 1
        @test diag["totalCrossingsRequested"] == 3
        @test diag["crossingsFound"] == 2
        @test diag["totalCrossingsFound"] == 3
        @test diag["terminationReason"] == "requested_crossings_found"
        @test diag["solverRetcode"] != "not_run"
        @test !diag["divergenceCallbackActivated"]

        short = DynamicsKit._collect_poincare_points(
            sys,
            params;
            initial_point=[1.0, 0.1],
            crossings=2,
            transient=0,
            projected=true,
            tmax=1e-4,
            return_diagnostics=true
        )
        @test isempty(short.points)
        @test short.diagnostics["terminationReason"] == "insufficient_crossings"
        @test short.diagnostics["crossingsFound"] == 0
        @test short.diagnostics["finalTime"] <= 1e-4 * 1.1

        detected = DynamicsKit._detect_continuous_poincare_period(
            sys,
            params;
            initial_point=[1.0, 0.1],
            transient=4,
            max_period=1,
            precision=1e-3,
            projected=true
        )
        @test :crossing_diagnostics in propertynames(detected)
        @test detected.crossing_diagnostics["crossingsRequested"] == 2
        @test detected.crossing_diagnostics["transientCrossings"] == 4
        @test detected.crossing_diagnostics["totalCrossingsFound"] >= 2
        @test detected.crossing_diagnostics["terminationReason"] in ("period_detected", "max_crossings_reached")

        map_cfg = BifurcationMapConfig(
            a_min=0.36,
            a_max=0.36,
            a_steps=0,
            b_min=0.36,
            b_max=0.36,
            b_steps=0,
            a_index=1,
            b_index=1,
            max_period=1,
            precision=1e-3,
            iterations=8,
            base_params=params
        )
        _, map_diag = DynamicsKit._bifurcation_map(
            sys,
            map_cfg;
            initial_point=[1.0, 0.1],
            solver=DynamicsKit.Tsit5()
        )
        @test haskey(map_diag, "crossing")
        @test map_diag["crossing"]["sampleCount"] == 1
        @test map_diag["crossing"]["crossingsRequested"][1, 1] == 2
        @test map_diag["crossing"]["totalCrossingsFound"][1, 1] >= 2
    end

    @testset "Continuous-time multi-skeleton branch helper" begin
        sys = radial_oscillator()

        config = ContinuationConfig(
            p_min = 0.15,
            p_max = 0.8,
            ds = 0.02,
            dsmax = 0.05,
            max_steps = 100,
            newton_tol = 1e-8,
            detect_bifurcation = 1
        )

        results = continuation_branches(
            sys,
            config,
            [1];
            skeleton_params=[0.25, 0.36, 0.49],
            n_initial=4,
            max_branches_per_period=3,
            signature_state_tol=0.2
        )

        @test length(results) >= 2
        @test all(r.period == 1 for r in results)
        @test all(length(r.branch) > 5 for r in results)

        branch_signatures = [[getproperty(pt, :x1) for pt in r.branch.branch] for r in results]
        @test any(all(abs.(sig) .< 1e-6) for sig in branch_signatures)
        @test any(all(sig .< -0.2) for sig in branch_signatures)

        reuse_results = continuation_branches(
            sys,
            config,
            [1];
            skeleton_params=[0.25, 0.36, 0.49],
            n_initial=4,
            trajectory_seed_points=false,
            reuse_neighbor_seeds=true,
            threaded=false,
            max_branches_per_period=3,
            signature_state_tol=0.2
        )

        @test length(reuse_results) >= 2
        @test all(r.period == 1 for r in reuse_results)

        branch_signature(result) = (
            result.period,
            round(minimum(pt.param for pt in result.branch.branch); digits=3),
            round(maximum(pt.param for pt in result.branch.branch); digits=3),
            round(sum(getproperty(pt, :x1) for pt in result.branch.branch) / length(result.branch.branch); digits=3)
        )

        serial_multi_period = continuation_branches(
            sys,
            config,
            [1, 2];
            skeleton_params=[0.25, 0.36, 0.49],
            n_initial=4,
            threaded=false,
            max_branches_per_period=3,
            signature_state_tol=0.2
        )

        threaded_multi_period = continuation_branches(
            sys,
            config,
            [1, 2];
            skeleton_params=[0.25, 0.36, 0.49],
            n_initial=4,
            threaded=true,
            max_branches_per_period=3,
            signature_state_tol=0.2
        )

        @test sort(branch_signature.(threaded_multi_period)) == sort(branch_signature.(serial_multi_period))
    end

    @testset "Neighbor seed cache helpers" begin
        cache = NamedTuple{(:param, :point, :stamp), Tuple{Float64, Vector{Float64}, Int}}[]
        DynamicsKit._update_neighbor_seed_cache!(cache, 0.25, [[-0.5], [0.0]]; max_entries=4)
        DynamicsKit._update_neighbor_seed_cache!(cache, 0.4, [[-0.7], [0.2]]; max_entries=4)

        reused = DynamicsKit._cached_neighbor_seed_points(cache, 0.38; max_points=3)
        @test length(reused) == 3
        @test abs(reused[1][1]) >= 0.2
        @test any(p -> isapprox(p[1], -0.7; atol=1e-12), reused)
        @test any(p -> isapprox(p[1], 0.2; atol=1e-12), reused)

        DynamicsKit._update_neighbor_seed_cache!(cache, 0.55, [[0.8], [0.9]]; max_entries=4)
        @test length(cache) == 4
        @test all(entry.param >= 0.25 for entry in cache)
    end

    @testset "Branch signatures support duplicate detection" begin
        points_a = Any[(param=0.1, x1=-1.0), (param=0.2, x1=-0.8), (param=0.3, x1=-0.6)]
        points_b = Any[(param=0.1005, x1=-1.0), (param=0.2, x1=-0.79), (param=0.3005, x1=-0.6)]
        points_c = Any[(param=0.1, x1=0.3), (param=0.2, x1=0.45), (param=0.3, x1=0.6)]

        branch_a = BranchResult(DynamicsKit.CombinedBranchResult(points_a, Any[]), 2, "Mock", :a, DynamicsKit.Dates.now())
        branch_b = BranchResult(DynamicsKit.CombinedBranchResult(points_b, Any[]), 2, "Mock", :a, DynamicsKit.Dates.now())
        branch_c = BranchResult(DynamicsKit.CombinedBranchResult(points_c, Any[]), 2, "Mock", :a, DynamicsKit.Dates.now())

        sig_a = DynamicsKit._branch_signature(branch_a)
        sig_b = DynamicsKit._branch_signature(branch_b)
        sig_c = DynamicsKit._branch_signature(branch_c)

        @test DynamicsKit._is_duplicate_signature(sig_b, [sig_a], 1e-2, 0.2)
        @test !DynamicsKit._is_duplicate_signature(sig_c, [sig_a], 1e-2, 0.2)
    end

    @testset "Continuation direction failures can be skipped" begin
        cfg = ContinuationConfig(
            p_min=0.05,
            p_max=0.6,
            ds=0.002,
            dsmax=0.008,
            dsmin=1e-6,
            max_steps=100,
            newton_tol=1e-8,
            detect_bifurcation=1
        )
        messages = String[]
        result = DynamicsKit._run_continuation_direction_safe(
            () -> error("Stopping continuation."),
            cfg;
            p_min=cfg.p_min,
            p_max=cfg.p_max,
            ds=cfg.ds,
            on_error=msg -> push!(messages, msg),
            context="Mock continuation"
        )

        @test isnothing(result)
        @test length(messages) == 1
        @test occursin("Mock continuation", only(messages))
        @test occursin("Stopping continuation.", only(messages))
    end

    @testset "Continuous branch auto-refinement helpers" begin
        coarse_points = Any[
            (param=0.10, x1=-1.0, stable=true),
            (param=0.12, x1=-1.05, stable=true),
            (param=0.18, x1=-1.2, stable=true),
            (param=0.20, x1=-1.25, stable=true)
        ]
        coarse_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(coarse_points, Any[]),
            1,
            "Mock",
            :a,
            DynamicsKit.Dates.now()
        )
        cfg = ContinuationConfig(
            p_min=0.05,
            p_max=0.25,
            ds=0.01,
            dsmax=0.04,
            max_steps=200,
            newton_tol=1e-8,
            detect_bifurcation=1
        )

        intervals = DynamicsKit._continuous_branch_refinement_intervals(
            coarse_branch,
            cfg;
            gap_factor=4.0,
            interval_padding_factor=0.0,
            detect_short_branches=false
        )
        @test intervals == [(0.12, 0.18)]

        refined_segment = BranchResult(
            DynamicsKit.CombinedBranchResult(
                Any[
                    (param=0.12, x1=-1.05, stable=true),
                    (param=0.14, x1=-1.10, stable=true),
                    (param=0.16, x1=-1.15, stable=true),
                    (param=0.18, x1=-1.20, stable=true)
                ],
                Any[]
            ),
            1,
            "Mock",
            :a,
            DynamicsKit.Dates.now()
        )

        merged = DynamicsKit._splice_refined_continuous_branches(
            coarse_branch,
            [refined_segment];
            param_tol=1e-12,
            state_tol=1e-12
        )
        merged_pars = [pt.param for pt in merged.branch.branch]
        @test length(merged_pars) == 6
        @test any(isapprox(p, 0.14; atol=1e-12) for p in merged_pars)
        @test any(isapprox(p, 0.16; atol=1e-12) for p in merged_pars)

        sys = radial_oscillator()
        real_cfg = ContinuationConfig(
            p_min=0.30,
            p_max=0.42,
            ds=0.03,
            dsmax=0.04,
            max_steps=4,
            newton_tol=1e-8,
            detect_bifurcation=0
        )
        coarse_real = continuation_branch(
            sys,
            real_cfg;
            initial_point=[-sqrt(0.36)],
            params=[0.36],
            solver=DynamicsKit.Tsit5(),
            reltol=1e-8,
            abstol=1e-8
        )
        real_intervals = DynamicsKit._continuous_branch_refinement_intervals(coarse_real, real_cfg)
        @test !isempty(real_intervals)

        refined_real = auto_refine_branch(
            sys,
            coarse_real,
            real_cfg;
            params=[0.36],
            max_passes=1,
            solver=DynamicsKit.Tsit5(),
            reltol=1e-8,
            abstol=1e-8
        )
        @test length(refined_real.branch) > length(coarse_real.branch)
    end

    @testset "Refined splicing preserves folded branch order" begin
        folded_points = Any[
            (param=0.00, x1=0.00, stable=true),
            (param=0.20, x1=0.20, stable=true),
            (param=0.40, x1=0.40, stable=true),
            (param=0.60, x1=0.60, stable=true),
            (param=0.40, x1=1.40, stable=true),
            (param=0.20, x1=1.60, stable=true),
            (param=0.00, x1=1.80, stable=true)
        ]
        folded_branch = BranchResult(
            DynamicsKit.CombinedBranchResult(folded_points, Any[]),
            1,
            "FoldedMock",
            :a,
            DynamicsKit.Dates.now()
        )
        reversed_refined_segment = BranchResult(
            DynamicsKit.CombinedBranchResult(
                Any[
                    (param=0.40, x1=0.39, stable=true),
                    (param=0.30, x1=0.30, stable=true),
                    (param=0.20, x1=0.21, stable=true)
                ],
                Any[]
            ),
            1,
            "FoldedMock",
            :a,
            DynamicsKit.Dates.now()
        )

        merged = DynamicsKit._splice_refined_continuous_branches(
            folded_branch,
            [reversed_refined_segment];
            param_tol=1e-12,
            state_tol=1e-12
        )

        xs = [pt.x1 for pt in merged.branch.branch]
        pars = [pt.param for pt in merged.branch.branch]
        @test xs == [0.00, 0.21, 0.30, 0.39, 0.60, 1.40, 1.60, 1.80]
        @test pars == [0.00, 0.20, 0.30, 0.40, 0.60, 0.40, 0.20, 0.00]
    end

    @testset "Phase 2: _fd_jacobian per-column step scaling" begin
        # F(x) = [x[1]^2, x[2]^2]; J = diag(2 * x). Fixed-step finite difference
        # with a small absolute delta on a large x[j] suffers catastrophic
        # cancellation. Per-column scaling preserves accuracy across magnitudes.
        F = x -> [x[1]^2, x[2]^2]
        # x[1] is small (use absolute floor), x[2] is large (use relative step).
        x = [1e-3, 1e5]
        J = DynamicsKit._fd_jacobian(F, x, 1e-6)
        expected = [2 * x[1] 0.0; 0.0 2 * x[2]]
        # Relative accuracy: forward-difference has O(h) truncation error, so
        # ~1e-3 relative is the right ceiling. The step must scale with x: a
        # fixed delta = 1e-6 is a meaningless perturbation on x = 1e5 —
        # catastrophic cancellation in (F(x+h) - F(x)).
        @test isapprox(J[1, 1], expected[1, 1]; rtol=1e-3)
        @test isapprox(J[2, 2], expected[2, 2]; rtol=1e-3)
        @test isapprox(J[1, 2], 0.0; atol=1e-3)
        @test isapprox(J[2, 1], 0.0; atol=1e-3)
    end

    @testset "Phase 2: _branch_signature uses multiple sample states" begin
        # Two synthetic branches with identical period, identical parameter
        # ranges, and identical midpoint x1, but distinct quartile x1 values.
        # The midpoint-only signature would treat these as duplicates.
        branch_a = BranchResult(
            DynamicsKit.CombinedBranchResult(
                Any[(param=p, x1=Float64(p)) for p in 0.0:0.1:1.0],
                Any[]
            ),
            1, "A", :p, DynamicsKit.Dates.now()
        )
        # Same midpoint (~0.5) but the curve dips and rises differently.
        branch_b = BranchResult(
            DynamicsKit.CombinedBranchResult(
                Any[(param=p, x1=0.5 + 0.05 * sin(20p)) for p in 0.0:0.1:1.0],
                Any[]
            ),
            1, "B", :p, DynamicsKit.Dates.now()
        )

        sig_a = DynamicsKit._branch_signature(branch_a)
        sig_b = DynamicsKit._branch_signature(branch_b)
        @test length(sig_a.sample_states) >= 3
        @test length(sig_b.sample_states) >= 3
        # Should NOT be classified as duplicates with reasonable state tolerance.
        @test !DynamicsKit._is_duplicate_signature(sig_a, [sig_b], 1e-6, 0.1)
    end

    @testset "Phase 2: period-N continuation rejects period < 1" begin
        sys = henon_map()
        config = ContinuationConfig(p_min=0.0, p_max=1.4, ds=0.01)
        @test_throws ArgumentError continuation_branch(sys, config, 0; params=[0.3])
        @test_throws ArgumentError continuation_branch(sys, config, -2; params=[0.3])
    end

    @testset "Continuation/reseed/refinement config validation" begin
        @test_throws AssertionError ContinuationConfig(p_min=0.0, p_max=1.0, ds=0.0)
        @test_throws AssertionError ContinuationConfig(p_min=0.0, p_max=1.0, dsmin=0.1, dsmax=0.01)
        @test_throws AssertionError ContinuationConfig(p_min=0.0, p_max=1.0, save_sol_every_step=0)
        @test ContinuationConfig(p_min=0.0, p_max=1.0, ode_jacobian_method=:variational).ode_jacobian_method == :variational
        @test_throws AssertionError ContinuationConfig(p_min=0.0, p_max=1.0, ode_jacobian_method=:unknown)
        @test_throws AssertionError ReseedConfig(max_attempts=-1)
        @test_throws AssertionError ReseedConfig(trailing_k=1)
        @test_throws AssertionError RefinementConfig(from_param=0.2, to_param=0.2)
        @test_throws AssertionError RefinementConfig(from_param=0.2, to_param=0.4, save_sol_every_step=0)
        @test RefinementConfig(from_param=0.2, to_param=0.4, ode_jacobian_method=:variational).ode_jacobian_method == :variational
        @test_throws AssertionError RefinementConfig(from_param=0.2, to_param=0.4, ode_jacobian_method=:unknown)
    end

    @testset "Termination diagnosis classification" begin
        sys = henon_map()

        # Branch that reaches the parameter boundary.
        cfg_b = ContinuationConfig(p_min=0.2, p_max=0.6, ds=0.01, dsmax=0.03, max_steps=1000)
        rb = continuation_branch(sys, cfg_b; initial_point=[0.5, 0.1], params=[0.3])
        info_b = DynamicsKit.diagnose_continuation_termination(rb.branch, cfg_b)
        @test info_b.reason == DynamicsKit.REACHED_BOUNDARY
        @test length(info_b.last_state) == sys.dim
        @test info_b.n_steps >= 2
        @test abs(info_b.last_param - cfg_b.p_max) <= 1e-3 ||
              abs(info_b.last_param - cfg_b.p_min) <= 1e-3
        # local_direction is [Δparam, Δstate...]
        @test length(info_b.local_direction) == sys.dim + 1

        # Branch artificially truncated in the interior by a tiny max_steps budget.
        cfg_m = ContinuationConfig(p_min=0.2, p_max=1.3, ds=0.01, dsmax=0.02, max_steps=5)
        rm = continuation_branch(sys, cfg_m; initial_point=[0.5, 0.1], params=[0.3])
        info_m = DynamicsKit.diagnose_continuation_termination(rm.branch, cfg_m)
        @test info_m.reason != DynamicsKit.REACHED_BOUNDARY
        @test info_m.reason != DynamicsKit.UNKNOWN
        @test cfg_m.p_min + 1e-3 < info_m.last_param < cfg_m.p_max - 1e-3
    end

    @testset "Targeted re-seed recovers a truncated continuous branch" begin
        sys = radial_oscillator()
        # Tiny max_steps forces an interior truncation; the limit cycle exists across the range.
        cfg = ContinuationConfig(p_min=0.1, p_max=1.2, ds=0.03, dsmax=0.05, max_steps=4)

        base = continuation_branch(sys, cfg; params=[0.6], n_initial=6)
        base_pars = [pt.param for pt in base.branch.branch]
        base_span = maximum(base_pars) - minimum(base_pars)

        diags = Tuple{Any, Any}[]
        recovered = continuation_branch(sys, cfg; params=[0.6], n_initial=6,
            reseed=ReseedConfig(enabled=true, max_attempts=6),
            on_reseed=(bw, fw) -> push!(diags, (bw, fw)))
        rec_pars = [pt.param for pt in recovered.branch.branch]
        rec_span = maximum(rec_pars) - minimum(rec_pars)

        # Re-seeding extends coverage well beyond the truncated baseline, reaching both ends.
        @test rec_span > base_span + 0.2
        @test maximum(rec_pars) > 1.0
        @test minimum(rec_pars) < 0.2
        # At least one re-seed attempt fired and was reported via the callback.
        total = sum(d.attempt_count for (bw, fw) in diags for d in (bw, fw) if d !== nothing)
        @test total >= 1
        # Recovered points stay on the limit cycle (finite projected coordinate).
        @test all(isfinite(getproperty(pt, :x1)) for pt in recovered.branch.branch)

        # No-regression: with re-seed on, a branch that completes normally is unchanged.
        cfg_full = ContinuationConfig(p_min=0.15, p_max=0.8, ds=0.02, dsmax=0.05, max_steps=200)
        off = continuation_branch(sys, cfg_full; params=[0.36], n_initial=6)
        on = continuation_branch(sys, cfg_full; params=[0.36], n_initial=6,
                                 reseed=ReseedConfig(enabled=true))
        @test length(off.branch.branch) == length(on.branch.branch)
    end

    @testset "Targeted re-seed recovers a truncated discrete branch" begin
        sys = henon_map()
        # Tiny max_steps truncates the period-1 branch; re-seed must extend it.
        cfg = ContinuationConfig(p_min=0.2, p_max=1.2, ds=0.02, dsmax=0.04, max_steps=4)

        base = continuation_branch(sys, cfg; initial_point=[0.6, 0.1], params=[0.4])
        base_pars = [pt.param for pt in base.branch.branch]
        base_span = maximum(base_pars) - minimum(base_pars)

        diags = Tuple{Any, Any}[]
        recovered = continuation_branch(sys, cfg; initial_point=[0.6, 0.1], params=[0.4],
            reseed=ReseedConfig(enabled=true, max_attempts=8),
            on_reseed=(bw, fw) -> push!(diags, (bw, fw)))
        rec_pars = [pt.param for pt in recovered.branch.branch]
        rec_span = maximum(rec_pars) - minimum(rec_pars)

        @test rec_span > base_span + 0.3
        @test maximum(rec_pars) > 1.0
        total = sum(d.attempt_count for (bw, fw) in diags for d in (bw, fw) if d !== nothing)
        @test total >= 1
        # Discrete continuation is now bidirectional (mirrors the continuous driver), so
        # the diagnostics callback should receive at least one ReseedDiagnostics value.
        @test any(!isnothing(d) for (bw, fw) in diags for d in (bw, fw))
    end

    @testset "Re-seed extends a truncated branch refinement" begin
        sys = henon_map()
        full_cfg = ContinuationConfig(p_min=0.2, p_max=1.2, ds=0.02, dsmax=0.05, max_steps=300)
        base = continuation_branch(sys, full_cfg; initial_point=[0.6, 0.1], params=[0.4])

        # Refine a sub-interval with a tiny max_steps so refinement truncates in the interior.
        rcfg = RefinementConfig(from_param=0.4, to_param=1.0, ds=0.01, dsmax=0.03, max_steps=4)
        ref_off = refine_branch(sys, base, rcfg; params=[0.4])
        off_pars = [pt.param for pt in ref_off.branch.branch]
        off_span = maximum(off_pars) - minimum(off_pars)

        diags = Tuple{Any, Any}[]
        ref_on = refine_branch(sys, base, rcfg; params=[0.4],
            reseed=ReseedConfig(enabled=true, max_attempts=8),
            on_reseed=(bw, fw) -> push!(diags, (bw, fw)))
        on_pars = [pt.param for pt in ref_on.branch.branch]
        on_span = maximum(on_pars) - minimum(on_pars)

        @test on_span > off_span + 0.3
        @test maximum(on_pars) > 0.9
        @test sum(d.attempt_count for (bw, fw) in diags for d in (bw, fw) if d !== nothing) >= 1
    end
end
