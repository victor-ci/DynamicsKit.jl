using Test
using StaticArrays
using LinearAlgebra
using ForwardDiff

# One-sided q-return Jacobian of the 2D border-collision normal form A = [τ 1; -δ 0].
_bcnf(τ, δ) = [τ 1.0; -δ 0.0]

@testset "Border-collision classification (continuous piecewise-smooth maps)" begin
    @testset "Simpson 2014 1D authoritative fixtures — all four scenarios" begin
        # f(x) = a_s x + μ, one-sided return Jacobians A_L = [a_L], A_R = [a_R].
        # sign(det(I-A_L)·det(I-A_R)) = persistence vs fold;
        # sign(det(I+A_L)·det(I+A_R)) = companion 2-cycle creation.
        cases = [
            (0.4, -0.4, :persistence, false),
            (2.0, -0.4, :nonsmooth_fold, false),
            (0.4, -1.5, :persistence_with_companion_cycle, true),
            (2.0, -1.5, :nonsmooth_fold_with_companion_cycle, true),
        ]
        for (aL, aR, scenario, companion) in cases
            c = border_collision_classify(reshape([aL], 1, 1), reshape([aR], 1, 1);
                switching_normal=[1.0])
            @test c.status == :ok
            @test c.scenario == scenario
            @test c.period == 1
            @test c.generic
            @test c.continuous === true
            @test c.continuity_residual == 0.0            # a scalar map is always continuous
            # Persistence product carries the fold verdict.
            persist = (1 - aL) * (1 - aR)
            comp = (1 + aL) * (1 + aR)
            @test c.persistence_product ≈ persist atol=1e-12
            @test c.companion_product ≈ comp atol=1e-12
            @test c.persistence_sign == (persist > 0 ? 1 : -1)
            @test c.companion_sign == (comp > 0 ? 1 : -1)
            @test c.companion_exists === companion
            # σ counts are diagnostics: sign(det(I∓A)) = (-1)^σ.
            @test c.sigma_plus_L !== nothing && c.sigma_minus_L !== nothing
            @test (-1)^(c.sigma_plus_L) == sign(1 - aL)
            @test (-1)^(c.sigma_minus_L) == sign(1 + aL)
            @test c.sigma_reliable
            @test occursin("no chaos", c.inference)       # never infers chaos/robust chaos
        end
    end

    @testset "2D BCNF fixtures with rank-one continuity — all four scenarios" begin
        # Each pair differs only in column 1 (trace/det entries), so A_L - A_R is rank one with
        # row space e₁ = [1,0]: the continuity/rank-one condition of a continuous PWS map.
        # det(I - [τ 1; -δ 0]) = 1 - τ + δ ; det(I + [τ 1; -δ 0]) = 1 + τ + δ.
        specs = [
            ((0.5, 0.2), (-0.5, 0.2), :persistence),
            ((2.0, 0.5), (0.3, 0.5), :nonsmooth_fold),
            ((0.5, 0.2), (-1.8, 0.5), :persistence_with_companion_cycle),
            ((2.0, 0.5), (-1.8, 0.5), :nonsmooth_fold_with_companion_cycle),
        ]
        for ((τL, δL), (τR, δR), scenario) in specs
            A_L = _bcnf(τL, δL)
            A_R = _bcnf(τR, δR)
            c = border_collision_classify(A_L, A_R; switching_normal=[1.0, 0.0])
            @test c.status == :ok
            @test c.scenario == scenario
            @test c.continuous === true
            @test c.continuity_residual !== nothing && c.continuity_residual < 1e-12
            @test c.persistence_product ≈ (1 - τL + δL) * (1 - τR + δR) atol=1e-12
            @test c.companion_product ≈ (1 + τL + δL) * (1 + τR + δR) atol=1e-12
            @test length(c.spectrum_L) == 2
            @test length(c.companion_multipliers) == 2   # companion monodromy = A_L·A_R
        end
    end

    @testset "Discontinuous map is rejected (:noncontinuous)" begin
        # A_L - A_R is rank two, so no continuous PWS map has these one-sided Jacobians about a
        # scalar switching manifold with normal [1,0]; classification must be refused.
        A_L = [0.5 1.0; -0.2 0.0]
        A_R = [-0.5 0.7; 0.1 0.1]
        D = A_L - A_R
        @test abs(det(D)) > 1e-3                          # genuinely rank two
        c = border_collision_classify(A_L, A_R; switching_normal=[1.0, 0.0])
        @test c.status == :noncontinuous
        @test c.scenario == :undetermined
        @test c.continuous === false
        @test c.continuity_residual > c.continuity_tolerance
        @test occursin("not continuous", c.inference)
        @test occursin("no chaos", c.inference)
    end

    @testset "Unusable switching normals are invalid" begin
        A_L = reshape([0.4], 1, 1)
        A_R = reshape([-0.4], 1, 1)
        wrong_length = border_collision_classify(A_L, A_R;
            switching_normal=[1.0, 0.0])
        @test wrong_length.status == :invalid
        @test wrong_length.scenario == :undetermined
        @test wrong_length.continuous === nothing
        @test any(contains("switching normal"), wrong_length.warnings)

        zero_normal = border_collision_classify(A_L, A_R;
            switching_normal=[0.0])
        @test zero_normal.status == :invalid
        @test zero_normal.companion_exists === nothing
        @test isempty(zero_normal.companion_multipliers)

        discontinuous_L = [0.5 1.0; -0.2 0.0]
        discontinuous_R = [-0.5 0.7; 0.1 0.1]
        refused = border_collision_classify(discontinuous_L, discontinuous_R;
            switching_normal=[0.0, 0.0])
        @test refused.status == :invalid
        @test refused.scenario == :undetermined
    end

    @testset "Genericity follows eigenvalue distance, not determinant magnitude" begin
        n = 24
        A = Matrix(Diagonal(fill(0.5, n)))
        c = border_collision_classify(A, A;
            switching_normal=vcat(1.0, zeros(n - 1)))
        @test c.status == :ok
        @test c.generic
        @test c.scenario == :persistence
        @test c.sigma_reliable

        expanding_n = 250
        expanding = Matrix(Diagonal(fill(30.0, expanding_n)))
        overflow_safe = border_collision_classify(expanding, expanding;
            switching_normal=vcat(1.0, zeros(expanding_n - 1)))
        @test overflow_safe.status == :ok
        @test overflow_safe.persistence_sign == 1
        @test overflow_safe.companion_sign == 1
        @test isfinite(overflow_safe.persistence_product)
        @test isfinite(overflow_safe.companion_product)
    end

    @testset "±1 eigenvalue degeneracy is refused (:degenerate)" begin
        # Eigenvalue exactly +1 ⇒ det(I - A_L) = 0 ⇒ non-generic.
        c_plus = border_collision_classify(reshape([1.0], 1, 1), reshape([0.5], 1, 1);
            switching_normal=[1.0])
        @test c_plus.status == :degenerate
        @test c_plus.scenario == :undetermined
        @test !c_plus.generic
        @test c_plus.persistence_product == 0.0
        @test occursin("eigenvalue at +1 or -1", c_plus.inference)

        # Eigenvalue exactly -1 ⇒ det(I + A_L) = 0 ⇒ non-generic (companion factor vanishes).
        c_minus = border_collision_classify(reshape([-1.0], 1, 1), reshape([0.5], 1, 1);
            switching_normal=[1.0])
        @test c_minus.status == :degenerate
        @test c_minus.companion_product == 0.0
        @test !c_minus.generic
        @test !c_minus.sigma_reliable                    # counts flagged unreliable near ±1
        @test c_minus.companion_exists === nothing
        @test c_minus.companion_stable === nothing
        @test c_minus.companion_spectral_radius === nothing
        @test isempty(c_minus.companion_multipliers)
    end

    @testset "Nontransversal crossing is refused (:nontransversal)" begin
        # Otherwise-persistent fixture, but the supplied transversality measure is ~0.
        c = border_collision_classify(reshape([0.4], 1, 1), reshape([-0.4], 1, 1);
            switching_normal=[1.0], transversality=0.0)
        @test c.status == :nontransversal
        @test c.scenario == :undetermined
        @test c.transversal === false
        @test occursin("not transverse", c.inference)

        c_ok = border_collision_classify(reshape([0.4], 1, 1), reshape([-0.4], 1, 1);
            switching_normal=[1.0], transversality=0.75)
        @test c_ok.status == :ok
        @test c_ok.transversal === true
        @test c_ok.transversality_measure == 0.75
    end

    @testset "Marginal stability is reported as undetermined, not a chaos verdict" begin
        # A_L is a rotation (eigenvalues ±i, spectral radius exactly 1); A_R differs in column 1
        # only, keeping continuity and genericity intact.
        A_L = [0.0 -1.0; 1.0 0.0]
        A_R = [0.5 -1.0; 0.5 0.0]
        c = border_collision_classify(A_L, A_R; switching_normal=[1.0, 0.0])
        @test c.status == :ok
        @test c.scenario == :persistence
        @test c.stable_L === nothing                     # marginal: radius within tol of 1
        @test c.spectral_radius_L ≈ 1.0 atol=1e-12
        @test any(w -> occursin("marginal", w), c.warnings)
        @test c.stable_R !== nothing                     # A_R is off the unit circle
    end

    @testset "Invalid Jacobians are refused (:invalid)" begin
        @test border_collision_classify([1.0 0.0; 0.0 1.0], reshape([0.5], 1, 1)).status == :invalid
        @test border_collision_classify(reshape([NaN], 1, 1), reshape([0.5], 1, 1)).status == :invalid
        nonsquare = border_collision_classify(reshape([1.0, 2.0], 1, 2), reshape([3.0, 4.0], 1, 2))
        @test nonsquare.status == :invalid
        @test nonsquare.scenario == :undetermined
        @test_throws ArgumentError border_collision_classify(
            reshape([0.4], 1, 1), reshape([-0.4], 1, 1); period=0)
    end

    @testset "True period-2 cycle-phase fixture proves q-return handling" begin
        # Nonlinear continuous 2D map. Affine continuous maps structurally forbid a clean
        # single-phase period-2 border collision (the closure forces a -1 eigenvalue on the
        # non-colliding side), so a genuine quadratic term is required.
        # Continuity at x=0: both branches give (0.5y+0.5, 1.4).
        f = function (x, p)
            if x[1] > 0
                return SVector(0.3x[1] + 0.5x[2] + 0.5 - 1.5x[1]^2, -0.4x[1] + 1.4)
            else
                return SVector(-0.6x[1] + 0.5x[2] + 0.5, -0.3x[1] + 1.4)
            end
        end
        ev = SwitchingEvent("border", (x, p) -> x[1])
        sys = DiscreteMap(f, 2, [:mu], "P2 border collision"; switching_events=[ev])
        # Period-2 orbit: P1 = (0,1) sits on the border, P2 = (1,1.4) interior (guard 1 > 0).
        pt = border_collision_at_cycle(sys, [[0.0, 1.0], [1.0, 1.4]], [0.0])

        @test pt.period == 2
        @test pt.colliding_phase == 1
        @test pt.itinerary == [0, 1]                       # phase 1 on border, phase 2 guard-positive
        @test pt.event_name == "border"
        @test pt.guard_component == 1
        @test pt.classification.status == :ok
        @test pt.classification.scenario == :nonsmooth_fold_with_companion_cycle
        @test pt.converged
        @test pt.classification.continuous === true
        @test pt.classification.continuity_residual < 1e-6
        # Determinant invariants are conjugation-invariant, so assert their signs/values, not the
        # exact matrices. (A_L^(2), A_R^(2) each carry one one-sided factor at the colliding phase.)
        @test pt.classification.persistence_product < 0    # fold
        @test pt.classification.companion_product < 0      # companion 2q-cycle created
        @test pt.classification.persistence_product ≈ -0.54 atol=2e-2
        @test pt.classification.companion_product ≈ -0.391 atol=2e-2
        @test size(pt.classification.jacobian_L) == (2, 2)
        # Forced one-sided finite differences with Richardson extrapolation are exact for this
        # piecewise-polynomial map up to rounding.
        @test pt.classification.jacobian_L ≈ [1.47 -1.35; 0.24 -0.2] atol=1e-4
        @test pt.classification.jacobian_R ≈ [-1.01 -1.35; -0.12 -0.2] atol=1e-4
    end

    @testset "Multiple / absent on-border phases handled conservatively" begin
        f = (x, p) -> SVector(0.5x[1] + p[1], 0.5x[2])
        ev = SwitchingEvent("border", (x, p) -> x[1])
        sys = DiscreteMap(f, 2, [:mu], "Ambiguous border"; switching_events=[ev])

        both = border_collision_at_cycle(sys, [[0.0, 1.0], [0.0, 2.0]], [0.0])
        @test both.classification.status == :multiple_border_phases
        @test both.classification.scenario == :undetermined
        @test !both.converged

        none = border_collision_at_cycle(sys, [[1.0, 1.0], [2.0, 1.0]], [0.0])
        @test none.classification.status == :unavailable
        @test none.classification.scenario == :undetermined
        @test occursin("No phase lies on the border", none.classification.warnings[1])
    end

    @testset "Multi-component guard records component identity" begin
        # Vector guard: component 2 collides (x[2]=0), component 1 stays positive.
        f = (x, p) -> SVector(0.5x[1], 0.4x[2])
        ev = SwitchingEvent("wall", (x, p) -> [x[1] - 1.0, x[2]])
        sys = DiscreteMap(f, 2, [:mu], "Vector guard"; switching_events=[ev])
        pt = border_collision_at_cycle(sys, [[2.0, 0.0]], [0.0]; period=1)
        @test pt.guard_component == 2
        @test pt.colliding_phase == 1
        @test pt.classification.status in (:ok, :degenerate, :nontransversal)
    end

    @testset "Guard-gradient AD preserves dual numbers" begin
        ad_seen = Ref(false)
        guard = (x, p) -> begin
            ad_seen[] |= x[1] isa ForwardDiff.Dual
            x[1]^2 + 3x[2]
        end
        ev = SwitchingEvent("curved-border", guard)
        grad = DynamicsKit._bcb_guard_gradient(ev, 1, [2.0, 1.0], [0.0])
        @test ad_seen[]
        @test grad ≈ [4.0, 3.0] atol=1e-12
    end

    @testset "Branch crossing location with a known answer (μ* = 0)" begin
        # Continuous 1D map f(x,μ) = a_s x + μ (a_L = 0.4 for x ≤ 0, a_R = -0.4 for x > 0).
        # The fixed point x*(μ) crosses the border exactly at μ* = 0; persistence, no companion.
        # Write f eltype-generic (SVector(val), not SVector{1,Float64}) so ForwardDiff works.
        f = function (x, p)
            a = x[1] > 0 ? -0.4 : 0.4
            return SVector(a * x[1] + p[1])
        end
        ev = SwitchingEvent("border", (x, p) -> x[1])
        sys = DiscreteMap(f, 1, [:mu], "1D crossing"; switching_events=[ev])
        config = ContinuationConfig(p_min=-0.5, p_max=0.5, ds=0.01, dsmax=0.02,
                                    max_steps=200, param_index=1)
        branch = continuation_branch(sys, config, 1; initial_point=[-0.5], params=[-0.3])

        pts = border_collision_points(sys, branch, [0.0])
        @test length(pts) == 1
        pt = pts[1]
        @test pt.param ≈ 0.0 atol=1e-6
        @test pt.classification.scenario == :persistence
        @test pt.classification.status == :ok
        @test pt.colliding_phase == 1
        @test pt.event_name == "border"
        @test pt.converged
        @test pt.classification.transversal === true
        @test pt.classification.transversality_measure !== nothing
        @test abs(pt.classification.transversality_measure) > 1e-6
    end

    @testset "Serialization round-trips" begin
        c = border_collision_classify(_bcnf(2.0, 0.5), _bcnf(-1.8, 0.5);
            switching_normal=[1.0, 0.0])
        @test c.scenario == :nonsmooth_fold_with_companion_cycle
        data = serialize_border_collision_classification(c)
        @test data["format"] == "border-collision-classification-v1"
        c2 = deserialize_border_collision_classification(data)
        @test c2.scenario == c.scenario
        @test c2.status == c.status
        @test c2.period == c.period
        @test isequal(c2.persistence_product, c.persistence_product)
        @test isequal(c2.companion_product, c.companion_product)
        @test c2.spectrum_L == c.spectrum_L
        @test c2.spectrum_R == c.spectrum_R
        @test c2.companion_multipliers == c.companion_multipliers
        @test c2.jacobian_L == c.jacobian_L
        @test c2.jacobian_R == c.jacobian_R
        @test c2.warnings == c.warnings
        @test c2.continuous === c.continuous
        @test c2.stable_L === c.stable_L

        # Nullable fields (undetermined status) survive as `nothing`.
        deg = border_collision_classify(reshape([1.0], 1, 1), reshape([0.5], 1, 1);
            switching_normal=[1.0])
        deg2 = deserialize_border_collision_classification(
            serialize_border_collision_classification(deg))
        @test deg2.status == :degenerate
        @test deg2.transversal === nothing
        @test isequal(deg2.persistence_product, 0.0)

        f = function (x, p)
            if x[1] > 0
                return SVector(0.3x[1] + 0.5x[2] + 0.5 - 1.5x[1]^2, -0.4x[1] + 1.4)
            else
                return SVector(-0.6x[1] + 0.5x[2] + 0.5, -0.3x[1] + 1.4)
            end
        end
        ev = SwitchingEvent("border", (x, p) -> x[1])
        sys = DiscreteMap(f, 2, [:mu], "P2 serialize"; switching_events=[ev])
        pt = border_collision_at_cycle(sys, [[0.0, 1.0], [1.0, 1.4]], [0.0])
        pdata = serialize_border_collision_point(pt)
        @test pdata["format"] == "border-collision-point-v1"
        pt2 = deserialize_border_collision_point(pdata)
        @test pt2.colliding_phase == pt.colliding_phase
        @test pt2.itinerary == pt.itinerary
        @test pt2.orbit == pt.orbit
        @test pt2.guard_values == pt.guard_values
        @test pt2.event_name == pt.event_name
        @test pt2.guard_component == pt.guard_component
        @test pt2.period == pt.period
        @test pt2.converged == pt.converged
        @test pt2.classification.scenario == pt.classification.scenario
        @test pt2.classification.jacobian_L == pt.classification.jacobian_L

        # Format guards.
        bad = serialize_border_collision_classification(c)
        bad["format"] = "border-collision-classification-v999"
        @test_throws ErrorException deserialize_border_collision_classification(bad)
        missing_field = serialize_border_collision_classification(c)
        delete!(missing_field, "scenario")
        @test_throws ErrorException deserialize_border_collision_classification(missing_field)
        bad_point = serialize_border_collision_point(pt)
        bad_point["format"] = "border-collision-point-v999"
        @test_throws ErrorException deserialize_border_collision_point(bad_point)
    end

    @testset "Scenario prediction: Farey order and 2D BCNF robust-chaos criteria" begin
        order = border_period_adding_order(2)
        @test [r.word for r in order] == ["L", "LLR", "LR", "LRR", "R"]
        @test [r.period for r in order] == [1, 3, 2, 3, 1]
        @test order[3].left_parent == "L"
        @test order[3].right_parent == "R"

        scalar = border_scenario_predict(reshape([0.4], 1, 1), reshape([-1.5], 1, 1);
            switching_normal=[1.0], max_farey_level=1)
        @test scalar.status == :ok
        @test scalar.model == :scalar_pwl
        @test scalar.predicted_cascade == :period_adding_order
        @test scalar.robust_chaos_verdict == :not_applicable
        @test [r.word for r in scalar.period_adding_rungs] == ["L", "LR", "R"]
        @test any(w -> occursin("slopes alone", w), scalar.warnings)

        fixed_point_candidate = border_scenario_predict(_bcnf(1.8, 0.3), _bcnf(-1.7, 0.2);
            switching_normal=[1.0, 0.0])
        @test fixed_point_candidate.status == :ok
        @test fixed_point_candidate.model == :bcnf_2d
        @test fixed_point_candidate.robust_chaos_verdict == :glendinning_fixed_point_candidate
        @test fixed_point_candidate.robust_chaos_conditions["basicWedge"] === true
        @test fixed_point_candidate.robust_chaos_conditions["trappingInequalityHolds"] === true
        @test fixed_point_candidate.robust_chaos_conditions["homoclinicInequalityHolds"] === true
        @test fixed_point_candidate.predicted_cascade == :robust_chaos_candidate
        @test fixed_point_candidate.bcnf_parameters["tau_L"] ≈ 1.8 atol=1e-12
        @test fixed_point_candidate.bcnf_parameters["delta_R"] ≈ 0.2 atol=1e-12

        trapping_only = border_scenario_predict(_bcnf(1.4, 0.3), _bcnf(-1.4, 0.2);
            switching_normal=[1.0, 0.0])
        @test trapping_only.robust_chaos_verdict == :byg_trapping_candidate
        @test trapping_only.robust_chaos_conditions["trappingInequalityHolds"] === true
        @test trapping_only.robust_chaos_conditions["homoclinicInequalityHolds"] === false

        refused = border_scenario_predict([0.5 1.0; -0.2 0.0], [-0.5 0.7; 0.1 0.1];
            switching_normal=[1.0, 0.0])
        @test refused.status == :refused
        @test refused.classification.status == :noncontinuous

        nontransversal = border_scenario_predict(_bcnf(1.8, 0.3), _bcnf(-1.7, 0.2);
            switching_normal=[1.0, 0.0], transversality=0.0)
        @test nontransversal.status == :refused
        @test nontransversal.classification.status == :nontransversal
    end

    @testset "Scenario verification sweep and serialization" begin
        selector = function (x, p)
            p[1] < 1.0 && return SVector(0.2)
            p[1] < 2.0 && return SVector(1.0 - x[1])
            x[1] < 0.4 && return SVector(0.6)
            x[1] < 0.8 && return SVector(0.9)
            return SVector(0.2)
        end
        sys = DiscreteMap(selector, 1, [:a], "Period selector")
        prediction = border_scenario_predict(reshape([0.4], 1, 1), reshape([-1.5], 1, 1);
            switching_normal=[1.0], max_farey_level=1)
        verification = border_scenario_verify(sys, prediction;
            param_index=1, base_params=[0.0], param_min=0.5, param_max=2.5,
            param_steps=3, initial_point=[0.1], transient=8, max_period=4,
            expected_periods=[1, 2, 3], required_prefix_length=3)
        @test verification.status == :ok
        @test verification.observed_periods == [1, 2, 3]
        @test verification.consistency_passed
        @test verification.matched_prefix_length == 3
        @test length(verification.observed_runs) == 3

        pdata = serialize_border_scenario_prediction(prediction)
        @test pdata["format"] == "border-scenario-prediction-v1"
        p2 = deserialize_border_scenario_prediction(pdata)
        @test p2.status == prediction.status
        @test p2.model == prediction.model
        @test [r.word for r in p2.period_adding_rungs] == [r.word for r in prediction.period_adding_rungs]
        @test p2.classification.scenario == prediction.classification.scenario

        vdata = serialize_border_scenario_verification(verification)
        @test vdata["format"] == "border-scenario-verification-v1"
        v2 = deserialize_border_scenario_verification(vdata)
        @test v2.observed_periods == verification.observed_periods
        @test v2.verification_kind == :period_sequence
        @test v2.consistency_passed == verification.consistency_passed

        logistic = DiscreteMap(
            (x, p) -> SVector(p[1] * x[1] * (1 - x[1])),
            1, [:r], "Logistic robust-screen fixture")
        robust_prediction = border_scenario_predict(_bcnf(1.8, 0.3), _bcnf(-1.7, 0.2);
            switching_normal=[1.0, 0.0], transversality=1.0)
        chaos_check = border_scenario_verify(logistic, robust_prediction;
            param_index=1, base_params=[3.9], param_min=3.9, param_max=4.0,
            param_steps=5, initial_point=[0.2], transient=800, max_period=8,
            lyapunov_transient=800, lyapunov_iterations=600,
            required_chaotic_fraction=0.6)
        @test chaos_check.verification_kind == :finite_time_chaos
        @test chaos_check.consistency_passed
        @test chaos_check.positive_lyapunov_fraction >= 0.6
        @test chaos_check.aperiodic_fraction >= 0.6
        @test length(chaos_check.lyapunov_exponents) == 5
        cdata = serialize_border_scenario_verification(chaos_check)
        c2 = deserialize_border_scenario_verification(cdata)
        @test c2.verification_kind == :finite_time_chaos
        @test c2.lyapunov_statuses == chaos_check.lyapunov_statuses
        @test c2.positive_lyapunov_fraction == chaos_check.positive_lyapunov_fraction

        bad_prediction = copy(pdata)
        bad_prediction["format"] = "border-scenario-prediction-v999"
        @test_throws ErrorException deserialize_border_scenario_prediction(bad_prediction)
        bad_parent = serialize_border_scenario_prediction(prediction)
        bad_parent["periodAddingRungs"][1]["leftParent"] = 1
        @test_throws ErrorException deserialize_border_scenario_prediction(bad_parent)
        bad_warnings = serialize_border_scenario_prediction(prediction)
        bad_warnings["warnings"] = "not an array"
        @test_throws ErrorException deserialize_border_scenario_prediction(bad_warnings)
        bad_verification = serialize_border_scenario_verification(verification)
        bad_verification["warnings"] = "not an array"
        @test_throws ErrorException deserialize_border_scenario_verification(bad_verification)
    end

    @testset "Verification refuses a :none-cascade prediction rather than trivially passing" begin
        # A verdict strictly inside the BCNF robust-chaos wedge boundary predicts
        # neither a period-adding order nor a robust-chaos candidate: there is
        # nothing for border_scenario_verify to falsify, and it must say so
        # instead of reporting "passed" whenever the sweep observes any
        # periodic point at all (regression for a prior silent-pass bug).
        no_scenario = border_scenario_predict(_bcnf(1.8, 0.5), _bcnf(-1.8, 0.5);
            switching_normal=[1.0, 0.0])
        @test no_scenario.status == :ok
        @test no_scenario.predicted_cascade == :none

        unrelated = DiscreteMap(
            (x, p) -> SVector(p[1] * x[1] + p[2] * x[2], p[3] * x[2]),
            2, [:a, :b, :c], "Unrelated contracting fixture")
        result = border_scenario_verify(unrelated, no_scenario;
            param_index=1, base_params=[0.3, 0.1, 0.3], param_min=0.3, param_max=0.3,
            param_steps=1, initial_point=[0.5, 0.2])
        @test result.status == :refused
        @test result.verification_kind == :not_applicable
        @test !result.consistency_passed
        @test isempty(result.observed_periods)
    end
end
