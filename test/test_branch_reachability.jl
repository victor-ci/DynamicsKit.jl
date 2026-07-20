@testset "Branch reachability (multistability-aware continuation)" begin
    using Dates: DateTime
    using LinearAlgebra: norm

    # --- fixtures -----------------------------------------------------------------------------
    # Analytic bistable map: x' = 1.5x - 0.5 x^3, y' = 0.5 y. On the sign-preserving domain
    # |x| < sqrt(3) it has stable fixed points (+1, 0) and (-1, 0), an unstable origin, and an
    # exact half-plane basin boundary x = 0. The parameter p is a dummy so synthetic continuation
    # branches can carry it. This is the multiplier-0 cubic (NOT the flawed multiplier -1 fixture).
    cubic = DiscreteMap((x, p) -> SVector(1.5 * x[1] - 0.5 * x[1]^3, 0.5 * x[2]), 2, [:p], "CubicBistable")
    # Logistic map embedded in 2D (y contracts to 0) for genuine multi-phase orbits.
    logistic = DiscreteMap((x, p) -> SVector(p[1] * x[1] * (1 - x[1]), 0.5 * x[2]), 2, [:r], "Logistic2D")

    function const_branch(sys, x1, x2; period=1, pname=:p, pmin=0.0, pmax=1.0, n=5)
        pts = [(param=p, x1=x1, x2=x2, stable=true) for p in range(pmin, pmax, length=n)]
        BranchResult(CombinedBranchResult(pts, Any[]), period, sys.name, pname, DateTime(2026, 1, 1))
    end

    even_grid_cfg(; kwargs...) = BranchReachabilityConfig(;
        param_samples=[0.5], base_params=[0.0],
        x_min=-1.5, x_max=1.5, x_steps=3, y_min=-0.4, y_max=0.4, y_steps=2,
        max_period=4, iterations=400, threaded=false, kwargs...)

    @testset "discrete Newton uses state-scaled convergence" begin
        large_state = DiscreteMap(
            (x, p) -> SVector(50.0 + 0.5 * (x[1] - 50.0)),
            1, [:p], "LargeState")
        seed = [50.0 + 4e-9]
        _, converged_initial = DynamicsKit._reach_newton_fixed_point(
            large_state, seed, [0.0], 1, 1, 1e-10)
        _, converged_final = DynamicsKit._reach_newton_fixed_point(
            large_state, seed, [0.0], 1, 0, 1e-10)
        @test converged_initial
        @test converged_final
    end

    @testset "stability-unavailable branch remains covered" begin
        float_only = DiscreteMap(
            (x, p) -> begin
                eltype(x) === Float64 || error("dual-number Jacobian unavailable")
                SVector(0.5 * x[1], 0.5 * x[2])
            end,
            2, [:p], "FloatOnly")
        branch = const_branch(float_only, 0.0, 0.0)
        result = branch_reachability(float_only, [branch],
            BranchReachabilityConfig(
                param_samples=[0.5], base_params=[0.0],
                x_min=-0.1, x_max=0.1, x_steps=1,
                y_min=-0.1, y_max=0.1, y_steps=1,
                max_period=2, iterations=100, newton_max_iter=1,
                branch_ids=["origin"], threaded=false))
        sample = result.samples[1]

        @test sample.branch_covered == [true]
        @test sample.n_unresolved == sample.n_seeds
        @test sample.n_outside_coverage == 0
        @test sample.diagnostics["stabilityUnavailableBranchCount"] == 1
        @test startswith(sample.diagnostics["branchNotes"]["1"], "stability_unavailable:")
    end

    # --- 1. analytic 50/50: same-period identity matching -------------------------------------
    @testset "analytic 50/50 same-period identity" begin
        branch_plus = const_branch(cubic, 1.0, 0.0)
        branch_minus = const_branch(cubic, -1.0, 0.0)
        result = branch_reachability(cubic, [branch_plus, branch_minus],
            even_grid_cfg(branch_ids=["plus", "minus"]))
        sample = result.samples[1]

        @test result.system_name == "CubicBistable"
        @test result.param_name == :p
        @test sample.branch_ids == ["plus", "minus"]
        @test sample.branch_periods == [1, 1]
        @test sample.branch_stable == [true, true]
        @test sample.branch_covered == [true, true]
        # Two same-period (P1) branches separated by geometry alone -> exact 50/50.
        @test sample.matched_counts == [6, 6]
        @test sample.matched_fractions == [0.5, 0.5]
        @test sample.n_matched == sample.n_seeds == 12
        fractions = branch_reachability_fractions(sample)
        @test fractions["plus"] == 0.5
        @test fractions["minus"] == 0.5
        # x < 0 -> minus (branch 2), x > 0 -> plus (branch 1); exact half-plane boundary.
        @test all(==(2), sample.assignment[1:2, :])
        @test all(==(1), sample.assignment[3:4, :])
        @test all(==(1), sample.status)  # every cell matched
    end

    # --- 2. unstable branch zero reachability -------------------------------------------------
    @testset "unstable branch zero reachability" begin
        branch_plus = const_branch(cubic, 1.0, 0.0)
        branch_minus = const_branch(cubic, -1.0, 0.0)
        branch_origin = const_branch(cubic, 0.0, 0.0)  # origin: x-multiplier 1.5 -> unstable
        result = branch_reachability(cubic, [branch_plus, branch_minus, branch_origin],
            even_grid_cfg(branch_ids=["plus", "minus", "origin"]))
        sample = result.samples[1]
        @test sample.branch_stable == [true, true, false]
        # The unstable origin branch attracts no seeds; its fraction is exactly zero.
        @test sample.matched_counts == [6, 6, 0]
        @test sample.matched_fractions[3] == 0.0
        @test branch_reachability_fractions(sample)["origin"] == 0.0
    end

    # --- 3. unmatched tolerance ---------------------------------------------------------------
    @testset "unmatched tolerance" begin
        # Only the +1 branch supplied; x < 0 seeds settle on (-1, 0), a P1 orbit no branch matches.
        result = branch_reachability(cubic, [const_branch(cubic, 1.0, 0.0)],
            even_grid_cfg(branch_ids=["plus"]))
        sample = result.samples[1]
        @test sample.n_matched == 6
        @test sample.n_unmatched == 6
        @test sample.n_outside_coverage == 0  # a same-period (P1) branch IS present at p
        @test sum(values(reachability_category_fractions(sample))) ≈ 1.0
    end

    # --- 4. aperiodic / diverged accounting ---------------------------------------------------
    @testset "aperiodic and diverged accounting" begin
        # Chaotic logistic (r = 3.9): every seed is aperiodic (period 0).
        chaos = branch_reachability(logistic,
            [const_branch(logistic, 0.6, 0.0; pname=:r, pmin=3.5, pmax=4.0)],
            BranchReachabilityConfig(param_samples=[3.9], base_params=[0.0],
                x_min=0.2, x_max=0.8, x_steps=3, y_min=-0.2, y_max=0.2, y_steps=2,
                max_period=6, iterations=2000, branch_ids=String[], threaded=false)).samples[1]
        @test chaos.n_aperiodic == chaos.n_seeds == 12
        @test all(==(0), chaos.terminal_period)

        # Seeds far outside the sign-preserving domain diverge.
        div = branch_reachability(cubic, [const_branch(cubic, 1.0, 0.0)],
            BranchReachabilityConfig(param_samples=[0.5], base_params=[0.0],
                x_min=3.0, x_max=6.0, x_steps=3, y_min=-0.2, y_max=0.2, y_steps=2,
                max_period=4, iterations=400, divergence_cutoff=1e6,
                branch_ids=["plus"], threaded=false)).samples[1]
        @test div.n_diverged == div.n_seeds == 12
    end

    # --- 5. fractions sum to one --------------------------------------------------------------
    @testset "category fractions sum to one" begin
        result = branch_reachability(cubic,
            [const_branch(cubic, 1.0, 0.0), const_branch(cubic, -1.0, 0.0)],
            even_grid_cfg(branch_ids=["plus", "minus"]))
        for sample in result.samples
            counts = reachability_category_counts(sample)
            @test counts.matched + counts.unmatched + counts.aperiodic + counts.diverged +
                  counts.unresolved + counts.stability_mismatch + counts.outside_coverage == sample.n_seeds
            @test sum(values(reachability_category_fractions(sample))) ≈ 1.0
        end
    end

    # --- 6. phase-invariant matching for a multi-phase orbit ----------------------------------
    @testset "phase-invariant multi-phase orbit" begin
        # Helper: cyclic-shifted copies of the same cycle are distance 0.
        cycle_a = [[0.8, 0.0], [0.5, 0.0]]
        cycle_b = [[0.5, 0.0], [0.8, 0.0]]  # same cycle, opposite phase
        @test DynamicsKit._reach_cycle_distance(cycle_a, cycle_b) == 0.0
        # Minimal period reduces a repeated fixed point to a single point.
        @test length(DynamicsKit._reach_minimal_cycle([[1.0, 0.0], [1.0, 0.0]], 1e-4)) == 1

        # End-to-end: logistic r = 3.2 has a stable 2-cycle (~0.7995, ~0.5130). A branch recorded at
        # ONE phase still matches seeds whose terminal orbit is detected at either phase.
        two_cycle_branch = const_branch(logistic, 0.79945, 0.0; period=2, pname=:r, pmin=3.0, pmax=3.4)
        sample = branch_reachability(logistic, [two_cycle_branch],
            BranchReachabilityConfig(param_samples=[3.2], base_params=[0.0],
                x_min=0.3, x_max=0.7, x_steps=4, y_min=-0.1, y_max=0.1, y_steps=2,
                max_period=6, iterations=3000, branch_ids=["twocycle"], threaded=false)).samples[1]
        @test sample.branch_periods == [2]
        @test sample.branch_stable == [true]
        @test sample.n_matched == sample.n_seeds == 15
        @test all(==(2), sample.terminal_period)
    end

    # --- 7. explicit sample coverage ----------------------------------------------------------
    @testset "explicit sample coverage" begin
        # plus branch continued only over p in [0, 0.4]; evaluate at a covered and an uncovered knot.
        plus_partial = const_branch(cubic, 1.0, 0.0; pmin=0.0, pmax=0.4)
        minus_full = const_branch(cubic, -1.0, 0.0; pmin=0.0, pmax=1.0)
        result = branch_reachability(cubic, [plus_partial, minus_full],
            BranchReachabilityConfig(param_samples=[0.2, 0.9], base_params=[0.0],
                x_min=-1.5, x_max=1.5, x_steps=3, y_min=-0.4, y_max=0.4, y_steps=2,
                max_period=4, iterations=400, branch_ids=["plus", "minus"], threaded=false))
        @test length(result.samples) == 2
        covered = result.samples[1]
        @test covered.param == 0.2
        @test covered.branch_covered == [true, true]
        @test covered.n_matched == 12
        uncovered = result.samples[2]
        @test uncovered.param == 0.9
        @test uncovered.branch_covered == [false, true]  # plus not continued to p = 0.9
        # x > 0 seeds reach (+1, 0) but the plus branch is not covered here -> unmatched.
        @test uncovered.n_matched == 6
        @test uncovered.n_unmatched == 6
    end

    # --- 8. outside_coverage: period not represented ------------------------------------------
    @testset "outside coverage" begin
        # Only a P1 branch supplied; logistic r = 3.2 seeds are P2 -> no same-period branch present.
        p1_only = const_branch(logistic, 0.6875, 0.0; period=1, pname=:r, pmin=3.0, pmax=3.4)
        sample = branch_reachability(logistic, [p1_only],
            BranchReachabilityConfig(param_samples=[3.2], base_params=[0.0],
                x_min=0.3, x_max=0.7, x_steps=4, y_min=-0.1, y_max=0.1, y_steps=2,
                max_period=6, iterations=3000, branch_ids=["fp1"], threaded=false)).samples[1]
        @test sample.n_outside_coverage == sample.n_seeds == 15
    end

    # --- 9. stability mismatch ----------------------------------------------------------------
    @testset "stability mismatch" begin
        # A seed placed exactly on the unstable origin stays there; the only same-period branch
        # within tolerance is the unstable origin branch -> stability_mismatch (rejected from
        # attracting fractions), never counted as matched.
        origin_branch = const_branch(cubic, 0.0, 0.0)
        sample = branch_reachability(cubic, [origin_branch],
            BranchReachabilityConfig(param_samples=[0.5], base_params=[0.0],
                x_min=-1.0, x_max=1.0, x_steps=2, y_min=0.0, y_max=1e-7, y_steps=1,
                max_period=4, iterations=400, branch_ids=["origin"], threaded=false)).samples[1]
        @test sample.branch_stable == [false]
        @test sample.n_matched == 0
        @test sample.n_stability_mismatch == 2  # the two seeds on the x = 0 axis
        @test sum(values(reachability_category_fractions(sample))) ≈ 1.0
    end

    # --- 10. unresolved: ambiguous between two equidistant branches ---------------------------
    @testset "unresolved ambiguity" begin
        # A contracting map with three coexisting stable fixed points (-1, 0), (0, 0), (1, 0).
        # Seeds near the origin are equidistant from the +1 and -1 branches; with a deliberately
        # wide match tolerance neither is decisively closer -> unresolved (not plurality-assigned).
        attractors = (SVector(-1.0, 0.0), SVector(0.0, 0.0), SVector(1.0, 0.0))
        function nearest_att(x)
            best = attractors[1]
            best_d = Inf
            for a in attractors
                d = sum(abs2, x .- a)
                if d < best_d
                    best_d = d
                    best = a
                end
            end
            best
        end
        tri = DiscreteMap((x, p) -> 0.2 .* SVector(x[1], x[2]) .+ 0.8 .* nearest_att(x),
            2, [:p], "TriAttractor")
        sample = branch_reachability(tri,
            [const_branch(tri, 1.0, 0.0), const_branch(tri, -1.0, 0.0)],
            BranchReachabilityConfig(param_samples=[0.5], base_params=[0.0],
                x_min=-0.2, x_max=0.2, x_steps=2, y_min=-0.1, y_max=0.1, y_steps=2,
                max_period=4, iterations=400, match_tolerance=1.5,
                branch_ids=["plus", "minus"], threaded=false)).samples[1]
        @test sample.branch_stable == [true, true]
        @test sample.n_unresolved == sample.n_seeds
        @test sample.n_matched == 0  # a period group is never plurality-assigned as a block
    end

    # --- 11. deterministic / thread parity ----------------------------------------------------
    @testset "thread parity" begin
        branches = [const_branch(cubic, 1.0, 0.0), const_branch(cubic, -1.0, 0.0)]
        dense(threaded) = BranchReachabilityConfig(param_samples=[0.5], base_params=[0.0],
            x_min=-1.5, x_max=1.5, x_steps=20, y_min=-0.4, y_max=0.4, y_steps=20,
            max_period=4, iterations=400, branch_ids=["plus", "minus"], threaded=threaded)
        threaded_sample = branch_reachability(cubic, branches, dense(true)).samples[1]
        serial_sample = branch_reachability(cubic, branches, dense(false)).samples[1]
        @test threaded_sample.status == serial_sample.status
        @test threaded_sample.assignment == serial_sample.assignment
        @test threaded_sample.match_distance == serial_sample.match_distance ||
              all(((a, b),) -> (isnan(a) && isnan(b)) || a == b,
                  zip(vec(threaded_sample.match_distance), vec(serial_sample.match_distance)))
        @test threaded_sample.matched_counts == serial_sample.matched_counts
    end

    # --- 12. evidence cross-check: provenance-gated --------------------------------------------
    @testset "evidence cross-check provenance" begin
        branches = [const_branch(cubic, 1.0, 0.0), const_branch(cubic, -1.0, 0.0)]
        cfg = BranchReachabilityConfig(param_samples=[0.5], base_params=[0.0],
            x_min=-1.5, x_max=1.5, x_steps=3, y_min=-0.4, y_max=0.4, y_steps=2,
            max_period=4, iterations=400, branch_ids=["plus", "minus"], threaded=false)
        basins = basins_of_attraction(cubic, BasinsConfig(bif_param=0.5,
            x_min=-1.5, x_max=1.5, x_steps=3, y_min=-0.4, y_max=0.4, y_steps=2,
            max_period=4, iterations=400))

        # Compatible independent evidence is accepted and cross-checked.
        checked = branch_reachability(cubic, branches, cfg; basins_crosscheck=[basins])
        @test checked.samples[1].n_matched == 12

        # Wrong system provenance is rejected.
        wrong_system = BasinsResult(basins.x_grid, basins.y_grid, basins.periodicity, 0.5, 4,
            "WrongSystem", basins.timestamp, 1, 2, basins.ic_template)
        @test_throws ArgumentError branch_reachability(cubic, branches, cfg;
            basins_crosscheck=[wrong_system])

        # Wrong parameter knot is rejected.
        wrong_param = BasinsResult(basins.x_grid, basins.y_grid, basins.periodicity, 0.9, 4,
            cubic.name, basins.timestamp, 1, 2, basins.ic_template)
        @test_throws ArgumentError branch_reachability(cubic, branches, cfg;
            basins_crosscheck=[wrong_param])

        # Length mismatch (wrong number of supplied censuses) is rejected.
        @test_throws ArgumentError branch_reachability(cubic, branches, cfg;
            basins_crosscheck=[basins, basins])

        # Inconsistent periodicity (right provenance, wrong detected periods) is rejected.
        drifted = BasinsResult(basins.x_grid, basins.y_grid, basins.periodicity .+ 1, 0.5, 4,
            cubic.name, basins.timestamp, 1, 2, basins.ic_template)
        @test_throws ArgumentError branch_reachability(cubic, branches, cfg;
            basins_crosscheck=[drifted])

        finite_cutoff_cfg = BranchReachabilityConfig(param_samples=[0.5], base_params=[0.0],
            x_min=-1.5, x_max=1.5, x_steps=3, y_min=-0.4, y_max=0.4, y_steps=2,
            max_period=4, iterations=400, divergence_cutoff=10.0,
            branch_ids=["plus", "minus"], threaded=false)
        @test_throws ArgumentError branch_reachability(
            cubic, branches, finite_cutoff_cfg; basins_crosscheck=[basins])
    end

    # --- 13. branch provenance validation -----------------------------------------------------
    @testset "branch provenance validation" begin
        cfg = even_grid_cfg(branch_ids=["plus"])
        # Branch for a different system.
        foreign = BranchResult(CombinedBranchResult([(param=0.5, x1=1.0, x2=0.0, stable=true)], Any[]),
            1, "OtherSystem", :p, DateTime(2026, 1, 1))
        @test_throws ArgumentError branch_reachability(cubic, [foreign], cfg)
        # Branch varying a different parameter.
        wrong_param = const_branch(cubic, 1.0, 0.0; pname=:q)
        @test_throws ArgumentError branch_reachability(cubic, [wrong_param], cfg)
        # No branches at all.
        @test_throws ArgumentError branch_reachability(cubic, BranchResult[], cfg)
    end

    # --- 14. config validation ----------------------------------------------------------------
    @testset "config validation" begin
        @test_throws AssertionError BranchReachabilityConfig(param_samples=Float64[],
            x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0)
        @test_throws AssertionError BranchReachabilityConfig(param_samples=[0.5],
            x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, max_period=10, iterations=5)
        @test_throws AssertionError BranchReachabilityConfig(param_samples=[0.5],
            x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, ambiguity_ratio=1.5)
        @test_throws AssertionError BranchReachabilityConfig(param_samples=[0.5],
            x_min=1.0, x_max=-1.0, y_min=-1.0, y_max=1.0)
        @test_throws AssertionError BranchReachabilityConfig(param_samples=[0.5],
            x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, x_index=1, y_index=1)
    end

    # --- 15. serialization roundtrip ----------------------------------------------------------
    @testset "serialization roundtrip" begin
        result = branch_reachability(cubic,
            [const_branch(cubic, 1.0, 0.0), const_branch(cubic, -1.0, 0.0)],
            BranchReachabilityConfig(param_samples=[0.3, 0.7], base_params=[0.0],
                x_min=-1.5, x_max=1.5, x_steps=3, y_min=-0.4, y_max=0.4, y_steps=2,
                max_period=4, iterations=400, divergence_cutoff=1e6,
                branch_ids=["plus", "minus"], threaded=false))
        data = serialize_branch_reachability_result(result)
        @test data["format"] == "branch-reachability-v1"
        restored = deserialize_branch_reachability_result(data)

        @test restored.system_name == result.system_name
        @test restored.param_name == result.param_name
        @test restored.param_index == result.param_index
        @test restored.branch_ids == result.branch_ids
        @test restored.branch_periods == result.branch_periods
        @test restored.x_grid == result.x_grid
        @test restored.y_grid == result.y_grid
        @test restored.divergence_cutoff == result.divergence_cutoff
        @test length(restored.samples) == length(result.samples)
        for (a, b) in zip(restored.samples, result.samples)
            @test a.param == b.param
            @test a.matched_counts == b.matched_counts
            @test a.matched_fractions == b.matched_fractions
            @test a.assignment == b.assignment
            @test a.status == b.status
            @test a.terminal_period == b.terminal_period
            @test a.branch_stable == b.branch_stable
            @test a.branch_covered == b.branch_covered
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x == y,
                zip(vec(a.match_distance), vec(b.match_distance)))
        end

        # An unknown format version is rejected.
        @test_throws ErrorException deserialize_branch_reachability_result(
            Dict{String, Any}("format" => "branch-reachability-v0"))
    end

    # --- 16. status label accessor ------------------------------------------------------------
    @testset "status labels" begin
        @test branch_reachability_status_label(1) == "matched"
        @test branch_reachability_status_label(2) == "unmatched"
        @test branch_reachability_status_label(3) == "aperiodic"
        @test branch_reachability_status_label(4) == "diverged"
        @test branch_reachability_status_label(5) == "unresolved"
        @test branch_reachability_status_label(6) == "stability_mismatch"
        @test branch_reachability_status_label(7) == "outside_coverage"
        @test branch_reachability_status_label(99) == "unknown"
    end

    # --- ContinuousODE / Poincaré fixtures ----------------------------------------------------
    # Analytic planar bistable radial flow: angular speed 1, dr/dt = -(r-1)(r-2)(r-3). In Cartesian
    # (away from r=0): dx = (dr/r) x - y, dy = (dr/r) y + x. It has stable period-1 limit cycles at
    # r=1 and r=3 and an unstable separating cycle at r=2. On the upward y=0 Poincaré section the
    # crossing is at x=+r, so the projected return coordinate is exactly the radius. Radial
    # convergence is fast (f'(1) = f'(3) = -2 -> multiplier e^{-4π} ≈ 3.5e-6 per revolution), so a
    # handful of iterations resolves the terminal cycle deterministically.
    function bistable_radial()
        function f!(du, u, p, t)
            x, y = u[1], u[2]
            r = sqrt(x^2 + y^2)
            dr = -(r - 1.0) * (r - 2.0) * (r - 3.0)
            drr = r > 1e-9 ? dr / r : 0.0
            du[1] = drr * x - y
            du[2] = drr * y + x
            return nothing
        end
        section = PoincareSection(
            (u, t, integrator) -> u[2];
            direction=:up, projection=[1], template=[0.0, 0.0])
        ContinuousODE(f!, 2, section, [:p], "BistableRadial";
            tspan_hint=8.0, default_initial_state=[1.0, 0.0], default_params=[0.0])
    end

    # A synthetic continuation branch pinned at a projected section radius `xproj` across a dummy
    # parameter interval. branch_reachability Newton-corrects the section fixed point and recomputes
    # stability from the return map, so the recorded `stable` flag and exact radius are only seeds.
    radial_branch(sys, xproj; period=1, pname=:p, pmin=0.0, pmax=1.0, n=5) =
        BranchResult(CombinedBranchResult(
            [(param=p, x1=xproj, x2=0.0, stable=true) for p in range(pmin, pmax, length=n)],
            Any[]), period, sys.name, pname, DateTime(2026, 1, 1))

    # Tiny grid over x∈[0.75,3.25] (4 values) × y∈{-0.1,+0.1} (2 values) = 8 seeds, deliberately
    # straddling the r=2 separator (4 seeds inside r<2 → r=1, 4 seeds r>2 → r=3) and avoiding both
    # r=0 (flow singularity) and r=2 (unstable manifold). ode_tmax=Inf uses the tspan_hint-scaled
    # horizon (well above one 2π revolution).
    radial_cfg(; kwargs...) = BranchReachabilityConfig(;
        param_samples=[0.5], base_params=[0.0], param_index=1,
        x_min=0.75, x_max=3.25, x_steps=3, y_min=-0.1, y_max=0.1, y_steps=1,
        x_index=1, y_index=2,
        max_period=2, iterations=8, precision=1e-5,
        match_tolerance=0.1, stability_tol=1e-6, newton_tol=1e-9, newton_max_iter=25,
        ode_solver="tsit5", ode_reltol=1e-9, ode_abstol=1e-9, ode_tmax=Inf,
        threaded=false, kwargs...)

    # --- 17. ContinuousODE Poincaré: two same-period P1 branches, exact 50/50 ------------------
    @testset "continuous Poincaré 50/50 same-period identity" begin
        sys = bistable_radial()
        b1 = radial_branch(sys, 1.0)
        b3 = radial_branch(sys, 3.0)
        result = branch_reachability(sys, [b1, b3], radial_cfg(branch_ids=["r1", "r3"]))
        sample = result.samples[1]

        @test result.system_name == "BistableRadial"
        @test result.param_name == :p
        @test sample.branch_ids == ["r1", "r3"]
        @test sample.branch_periods == [1, 1]
        # branch_reachability recomputes stability from the return map: both r=1 and r=3 are stable.
        @test sample.branch_stable == [true, true]
        @test sample.branch_covered == [true, true]
        @test sample.n_seeds == 8
        # Two same-period (P1) cycles separated by radius alone -> exact 4/4 split.
        @test sample.matched_counts == [4, 4]
        @test sample.matched_fractions == [0.5, 0.5]
        @test sample.n_matched == 8
        fractions = branch_reachability_fractions(sample)
        @test fractions["r1"] == 0.5
        @test fractions["r3"] == 0.5

        # The seven accounting categories exactly partition the census.
        @test sample.n_matched + sample.n_unmatched + sample.n_aperiodic + sample.n_diverged +
              sample.n_unresolved + sample.n_stability_mismatch + sample.n_outside_coverage ==
              sample.n_seeds
        @test sample.n_unmatched == sample.n_aperiodic == sample.n_diverged == 0
        @test sample.n_unresolved == sample.n_stability_mismatch == sample.n_outside_coverage == 0

        # r < 2 -> r=1 (branch 1); r > 2 -> r=3 (branch 2).
        @test sample.assignment == [1 1; 1 1; 2 2; 2 2]
        @test all(sample.terminal_period .== 1)
        # Phase invariance: the two section-crossing phases (y=-0.1 and y=+0.1 launch states) reach
        # the same branch identity at each radius.
        @test sample.assignment[:, 1] == sample.assignment[:, 2]
    end

    # --- 18. ContinuousODE Poincaré: unstable separator receives zero -------------------------
    @testset "continuous Poincaré unstable branch gets zero" begin
        sys = bistable_radial()
        b1 = radial_branch(sys, 1.0)
        b2 = radial_branch(sys, 2.0)
        b3 = radial_branch(sys, 3.0)
        result = branch_reachability(sys, [b1, b2, b3], radial_cfg(branch_ids=["r1", "r2", "r3"]))
        sample = result.samples[1]

        @test sample.branch_covered == [true, true, true]
        # The r=2 fixed point is a genuine return-map fixed point (covered) but unstable: f'(2) = +1
        # -> multiplier e^{2π} ≈ 535. Stable-only identity matching therefore assigns it nothing.
        @test sample.branch_stable == [true, false, true]
        @test sample.matched_counts == [4, 0, 4]
        @test sample.matched_fractions == [0.5, 0.0, 0.5]
        @test sample.n_matched == 8
        @test sample.n_stability_mismatch == 0
        # Category partition still holds with the unstable branch present.
        @test sample.n_matched + sample.n_unmatched + sample.n_aperiodic + sample.n_diverged +
              sample.n_unresolved + sample.n_stability_mismatch + sample.n_outside_coverage ==
              sample.n_seeds == 8
    end

    # --- 19. ContinuousODE Poincaré: thread parity --------------------------------------------
    @testset "continuous Poincaré thread parity" begin
        sys = bistable_radial()
        b1 = radial_branch(sys, 1.0)
        b3 = radial_branch(sys, 3.0)
        serial = branch_reachability(sys, [b1, b3], radial_cfg(threaded=false)).samples[1]
        parallel = branch_reachability(sys, [b1, b3], radial_cfg(threaded=true)).samples[1]
        @test parallel.assignment == serial.assignment
        @test parallel.status == serial.status
        @test parallel.matched_counts == serial.matched_counts
        @test parallel.terminal_period == serial.terminal_period
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x == y,
            zip(vec(parallel.match_distance), vec(serial.match_distance)))
    end

    # --- 20. ContinuousODE Poincaré: solver-horizon failure is unresolved, not a throw --------
    @testset "continuous Poincaré tmax failure -> unresolved/uncovered" begin
        sys = bistable_radial()
        b1 = radial_branch(sys, 1.0)
        b3 = radial_branch(sys, 3.0)
        # ode_tmax below one 2π revolution: no seed completes enough section crossings and no branch
        # return-map Newton solve converges. The census must degrade honestly (unresolved seeds,
        # uncovered branches) rather than throw or fabricate matches.
        result = branch_reachability(sys, [b1, b3],
            radial_cfg(branch_ids=["r1", "r3"], ode_tmax=0.5))
        sample = result.samples[1]
        @test sample.branch_covered == [false, false]
        @test sample.n_matched == 0
        @test sample.n_unresolved == sample.n_seeds == 8
        @test sample.n_diverged == sample.n_unmatched == sample.n_outside_coverage == 0
        @test all(sample.assignment .== 0)
    end

    # --- 21. ContinuousODE Poincaré: config / section validation ------------------------------
    @testset "continuous Poincaré validation" begin
        sys = bistable_radial()
        branch = radial_branch(sys, 1.0)

        # basins_crosscheck is explicitly rejected for ContinuousODE (return-map census cannot prove
        # identical crossing semantics against an independent basins census).
        @test_throws ArgumentError branch_reachability(sys, [branch], radial_cfg();
            basins_crosscheck=BasinsResult[])

        # Grid indices must lie within the full state dimension.
        @test_throws ArgumentError branch_reachability(sys, [branch],
            radial_cfg(x_index=1, y_index=3))

        # ic_template length must match the full state dimension.
        @test_throws ArgumentError branch_reachability(sys, [branch],
            radial_cfg(ic_template=[0.0, 0.0, 0.0]))

        # param_index must not exceed the system's parameter count.
        @test_throws ArgumentError branch_reachability(sys, [branch], radial_cfg(param_index=2))

        # linked_param_indices must reference real parameters and exclude the varied index.
        @test_throws ArgumentError branch_reachability(sys, [branch],
            radial_cfg(linked_param_indices=[5]))

        # A section whose full-state template dimension disagrees with the system is rejected: branch
        # states live in projected coordinates, so an inconsistent lift would silently mis-align.
        bad_section = PoincareSection((u, t, integrator) -> u[2];
            direction=:up, projection=[1], template=[0.0, 0.0, 0.0])
        bad_sys = ContinuousODE((du, u, p, t) -> (du .= 0; nothing), 2, bad_section, [:p],
            "BistableRadial"; tspan_hint=8.0, default_initial_state=[1.0, 0.0], default_params=[0.0])
        bad_branch = radial_branch(bad_sys, 1.0)
        @test_throws ArgumentError branch_reachability(bad_sys, [bad_branch], radial_cfg())
    end
end
