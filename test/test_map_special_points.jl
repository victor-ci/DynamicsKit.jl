using Dates

@testset "Map-aware special points (PD / fold)" begin
    @testset "Hénon: analytic PD and fold, recovering the flip BifurcationKit misses" begin
        sys = henon_map()
        b = 0.3
        config = ContinuationConfig(p_min=-0.25, p_max=0.6, ds=0.01, dsmax=0.03,
                                    max_steps=300, detect_bifurcation=3, param_index=1)
        a0 = 0.3
        x_star = (-(1 - b) + sqrt((1 - b)^2 + 4a0)) / (2a0)
        branch = continuation_branch(sys, config, 1; initial_point=[x_star, b * x_star], params=[a0, b])

        specials = map_special_points(sys, branch, [a0, b])
        pd = filter(s -> s.kind == :pd, specials)
        fold = filter(s -> s.kind == :fold, specials)

        @test length(pd) == 1
        # Analytic period-1 flip of the Hénon map at b = 0.3: a = 3(1-b)²/4 = 0.3675.
        @test pd[1].param ≈ 0.3675 atol=1e-4
        @test real(pd[1].critical_multiplier) ≈ -1.0 atol=1e-3
        @test abs(imag(pd[1].critical_multiplier)) < 1e-6
        @test pd[1].converged
        @test pd[1].period == 1
        expected_c = 0.3675^2 / ((1 - b^2) * (1 + b^2))
        @test pd[1].normal_form !== nothing
        @test pd[1].normal_form.coefficient ≈ expected_c rtol=2e-7
        @test pd[1].normal_form.criticality == :supercritical
        @test pd[1].normal_form.status == :ok

        @test length(fold) == 1
        # Analytic period-1 fold: a = -(1-b)²/4 = -0.1225.
        @test fold[1].param ≈ -0.1225 atol=1e-4
        @test real(fold[1].critical_multiplier) ≈ 1.0 atol=1e-3
        @test fold[1].normal_form.criticality == :nondegenerate

        # The value of F4: BifurcationKit's own special points carry no period-doubling
        # near a = 0.3675 (the equilibrium-convention detector misses map flips).
        bk_specials = collect(branch.branch.specialpoint)
        @test !any(sp -> abs(Float64(sp.param) - 0.3675) < 1e-2, bk_specials)
    end

    @testset "Boost converter: recovers the subharmonic period-doubling BifurcationKit misses" begin
        sys = boost_converter()
        base = [1.5, 10.0, 20.0, 0.0]
        # Settle onto the stable period-1 orbit below the subharmonic threshold.
        attractor = foldl((s, _) -> sys.f(s, base), 1:500; init=SVector(8.0, 1.0))
        config = ContinuationConfig(p_min=1.2, p_max=1.95, ds=0.005, dsmax=0.01,
                                    max_steps=200, detect_bifurcation=3, param_index=1)
        branch = continuation_branch(sys, config, 1; initial_point=collect(attractor), params=base)

        specials = map_special_points(sys, branch, base; detect=[:pd])
        @test length(specials) == 1
        @test specials[1].kind == :pd
        # Peak-current-mode subharmonic instability at duty ratio 1/2 (μ = -1), near Iref ≈ 1.7 A.
        @test 1.6 < specials[1].param < 1.85
        @test real(specials[1].critical_multiplier) ≈ -1.0 atol=1e-2
        @test specials[1].converged
        @test specials[1].normal_form !== nothing

        bk_specials = collect(branch.branch.specialpoint)
        @test !any(sp -> String(sp.type) == "pd", bk_specials)
    end

    @testset "detect filter and guards" begin
        sys = henon_map()
        config = ContinuationConfig(p_min=0.2, p_max=0.5, ds=0.01, dsmax=0.03,
                                    max_steps=400, param_index=1)
        a0 = 0.3
        x_star = (-(1 - 0.3) + sqrt(0.7^2 + 4a0)) / (2a0)
        branch = continuation_branch(sys, config, 1; initial_point=[x_star, 0.3 * x_star], params=[a0, 0.3])

        only_pd = map_special_points(sys, branch, [a0, 0.3]; detect=[:pd])
        @test all(s -> s.kind == :pd, only_pd)
        @test length(only_pd) == 1                       # fold is outside [0.2, 0.5]

        @test_throws ArgumentError map_special_points(sys, branch, [a0, 0.3]; detect=[:hopf])
        @test_throws ArgumentError map_special_points(
            sys, branch, [a0, 0.3]; normal_form_fd_step=0.0)
        @test_throws ArgumentError map_normal_form(
            sys, :pd, [x_star, 0.3 * x_star], [a0, 0.3]; singular_tol=0.0)

        nonfinite = DiscreteMap(
            (x, p) -> SVector(iszero(x[1]) ? p[1] * x[1] : NaN),
            1, [:a], "Non-finite perturbed map")
        nonfinite_branch = BranchResult(
            CombinedBranchResult(
                Any[(param=-1.1, x1=0.0), (param=-0.9, x1=0.0)],
                Any[]),
            1, nonfinite.name, :a, now())
        @test isempty(map_special_points(
            nonfinite, nonfinite_branch, [-1.0]; attach_normal_forms=false))

        domain_failure = DiscreteMap(
            (_x, _p) -> throw(DomainError(-1.0, "synthetic multiplier failure")),
            1, [:a], "Domain-failing map")
        domain_branch = BranchResult(
            CombinedBranchResult(
                Any[(param=-1.1, x1=0.0), (param=-0.9, x1=0.0)],
                Any[]),
            1, domain_failure.name, :a, now())
        @test isempty(map_special_points(
            domain_failure, domain_branch, [-1.0]; attach_normal_forms=false))
    end

    @testset "A branch sample exactly on the flip is emitted without double counting" begin
        # x → a·x has fixed point 0 with multiplier a; the flip (μ = −1) sits exactly at
        # a = −1, and a symmetric sweep places a branch sample there (φ = μ + 1 = 0).
        sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:a], "Linear multiplier")
        config = ContinuationConfig(p_min=-1.6, p_max=-0.4, ds=0.1, dsmax=0.1,
                                    max_steps=100, param_index=1)
        branch = continuation_branch(sys, config, 1; initial_point=[0.0], params=[-1.0])

        specials = map_special_points(sys, branch, [-1.0]; detect=[:pd])
        @test length(specials) == 1                      # emitted once, not once per adjacent bracket
        @test specials[1].param ≈ -1.0 atol=1e-9
        @test real(specials[1].critical_multiplier) ≈ -1.0 atol=1e-9
        @test specials[1].converged
        @test specials[1].normal_form.status == :degenerate
        @test specials[1].normal_form.criticality == :degenerate
    end

    @testset "Kuznetsov/MATCONT analytic coefficients" begin
        fold_map = DiscreteMap(
            (x, p) -> SVector(p[1] - x[1]^2), 1, [:mu], "Quadratic fold")
        fold_nf = map_normal_form(fold_map, :fold, [-0.5], [-0.25])
        @test fold_nf.coefficient_name == :b
        @test fold_nf.coefficient ≈ -1.0 atol=1e-12
        @test fold_nf.criticality == :nondegenerate
        @test occursin("orientation", fold_nf.convention)

        logistic = DiscreteMap(
            (x, p) -> SVector(p[1] * x[1] * (1 - x[1])),
            1, [:r], "Logistic")
        logistic_nf = map_normal_form(logistic, :pd, [2 / 3], [3.0])
        @test logistic_nf.coefficient_name == :c
        @test logistic_nf.coefficient ≈ 9.0 atol=1e-10
        @test logistic_nf.criticality == :supercritical

        cubic_flip = DiscreteMap(
            (x, p) -> SVector(-(1 + p[1]) * x[1] - p[2] * x[1]^3),
            1, [:mu, :k], "Cubic flip")
        cubic_nf = map_normal_form(cubic_flip, :pd, [0.0], [0.0, 2.5])
        @test cubic_nf.coefficient ≈ -2.5 atol=1e-10
        @test cubic_nf.criticality == :subcritical

        linear_flip = DiscreteMap(
            (x, p) -> SVector(p[1] * x[1]), 1, [:a], "Linear flip")
        linear_nf = map_normal_form(linear_flip, :pd, [0.0], [-1.0])
        @test linear_nf.coefficient ≈ 0.0 atol=1e-14
        @test linear_nf.status == :degenerate
        @test linear_nf.criticality == :degenerate

        singular_flip = DiscreteMap(
            (x, p) -> SVector(-x[1], x[2] + x[1]^2),
            2, Symbol[], "Singular flip homological equation")
        singular_nf = map_normal_form(singular_flip, :pd, [0.0, 0.0], Float64[])
        @test singular_nf.coefficient === nothing
        @test singular_nf.status == :near_singular
        @test singular_nf.criticality == :unclassified
    end

    @testset "Hénon normalized period-1 flip coefficient" begin
        b = 0.3
        a = 3(1 - b)^2 / 4
        x = (1 - b) / (2a)
        nf = map_normal_form(henon_map(), :pd, [x, b * x], [a, b])

        # With q=(1,-b)/sqrt(1+b^2), p=sqrt(1+b^2)*(1,-1)/(1+b),
        # direct substitution into B(u,v)=(-2a*u1*v1,0) gives this value.
        expected = a^2 / ((1 - b^2) * (1 + b^2))
        @test nf.coefficient ≈ expected rtol=1e-10
        @test nf.criticality == :supercritical
    end

    @testset "Native Neimark-Sacker detection and coefficient" begin
        theta = 0.7
        gamma = -0.4
        rotation = @SMatrix [cos(theta) -sin(theta); sin(theta) cos(theta)]
        sys = DiscreteMap(
            (x, p) -> begin
                y = p[1] .* (rotation * x) .* (1 + p[2] * sum(abs2, x))
                SVector(y...)
            end,
            2, [:rho, :gamma], "Rotation cubic")
        points = Any[
            (param=0.9, x1=0.0, x2=0.0),
            (param=0.95, x1=0.0, x2=0.0),
            (param=1.05, x1=0.0, x2=0.0),
            (param=1.1, x1=0.0, x2=0.0),
        ]
        branch = BranchResult(
            CombinedBranchResult(points, Any[]), 1, sys.name, :rho, now())
        specials = map_special_points(sys, branch, [1.0, gamma])

        ns = filter(point -> point.kind == :ns, specials)
        @test length(ns) == 1
        @test ns[1].param ≈ 1.0 atol=1e-7
        @test abs(ns[1].critical_multiplier) ≈ 1.0 atol=1e-7
        @test abs(imag(ns[1].critical_multiplier)) > 0.1
        @test ns[1].normal_form.coefficient_name == :d
        @test ns[1].normal_form.coefficient ≈ 2gamma atol=1e-10
        @test ns[1].normal_form.criticality == :supercritical

        strong_theta = pi / 2
        strong_rotation = @SMatrix [
            cos(strong_theta) -sin(strong_theta)
            sin(strong_theta) cos(strong_theta)
        ]
        strong = DiscreteMap(
            (x, p) -> SVector((strong_rotation * x .* (1 + p[1] * sum(abs2, x)))...),
            2, [:gamma], "Strong 1:4 resonance")
        resonant_nf = map_normal_form(strong, :ns, [0.0, 0.0], [-0.2])
        @test resonant_nf.coefficient === nothing
        @test resonant_nf.status == :strong_resonance
        @test resonant_nf.criticality == :unclassified

        theta1, theta2 = 0.4, 0.9
        no_crossing = DiscreteMap(
            (x, p) -> begin
                r1 = 0.9 - 0.1p[1]
                r2 = 1.2 - 0.1p[1]
                SVector(
                    r1 * (cos(theta1) * x[1] - sin(theta1) * x[2]),
                    r1 * (sin(theta1) * x[1] + cos(theta1) * x[2]),
                    r2 * (cos(theta2) * x[3] - sin(theta2) * x[4]),
                    r2 * (sin(theta2) * x[3] + cos(theta2) * x[4]),
                )
            end,
            4, [:mu], "Two noncrossing complex pairs")
        no_crossing_branch = BranchResult(
            CombinedBranchResult(
                Any[
                    (param=0.0, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                    (param=1.0, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                ],
                Any[]),
            1, no_crossing.name, :mu, now())
        @test isempty(map_special_points(
            no_crossing, no_crossing_branch, [0.0];
            detect=[:ns], attach_normal_forms=false))

        angle_swap = DiscreteMap(
            (x, p) -> begin
                angle1 = 0.4 + 0.5p[1]
                angle2 = 0.9 - 0.5p[1]
                SVector(
                    0.9 * (cos(angle1) * x[1] - sin(angle1) * x[2]),
                    0.9 * (sin(angle1) * x[1] + cos(angle1) * x[2]),
                    1.1 * (cos(angle2) * x[3] - sin(angle2) * x[4]),
                    1.1 * (sin(angle2) * x[3] + cos(angle2) * x[4]),
                )
            end,
            4, [:mu], "Crossing-angle noncrossing pairs")
        angle_swap_branch = BranchResult(
            CombinedBranchResult(
                Any[
                    (param=0.0, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                    (param=1.0, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                ],
                Any[]),
            1, angle_swap.name, :mu, now())
        @test isempty(map_special_points(
            angle_swap, angle_swap_branch, [0.0];
            detect=[:ns], attach_normal_forms=false))

        simultaneous = DiscreteMap(
            (x, p) -> SVector(
                p[1] * (cos(theta1) * x[1] - sin(theta1) * x[2]),
                p[1] * (sin(theta1) * x[1] + cos(theta1) * x[2]),
                p[1] * (cos(theta2) * x[3] - sin(theta2) * x[4]),
                p[1] * (sin(theta2) * x[3] + cos(theta2) * x[4]),
            ),
            4, [:rho], "Simultaneous rotation crossings")
        simultaneous_branch = BranchResult(
            CombinedBranchResult(
                Any[
                    (param=0.9, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                    (param=0.95, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                    (param=1.05, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                    (param=1.1, x1=0.0, x2=0.0, x3=0.0, x4=0.0),
                ],
                Any[]),
            1, simultaneous.name, :rho, now())
        simultaneous_ns = map_special_points(
            simultaneous, simultaneous_branch, [1.0]; detect=[:ns])
        @test length(simultaneous_ns) == 2
        simultaneous_angles = angle.(getfield.(simultaneous_ns, :critical_multiplier))
        @test issorted(simultaneous_angles)
        @test simultaneous_angles ≈ [theta1, theta2] atol=1e-6
        @test all(point -> point.normal_form.coefficient === nothing, simultaneous_ns)
        @test all(point -> point.normal_form.status == :multiple_critical_pairs,
                  simultaneous_ns)
    end

    @testset "Continuous Poincare coefficients reject integration-noise cancellation" begin
        theta = 0.7
        cubic_rate = -0.2 / pi
        section = PoincareSection(
            (u, _t, _integrator) -> u[3] - 2pi;
            direction=:up, projection=[1, 2], template=zeros(3))
        suspended_ns = ContinuousODE(
            (du, u, p, _t) -> begin
                radius2 = u[1]^2 + u[2]^2
                du[1] = p[1] * u[1] - theta / (2pi) * u[2] +
                        cubic_rate * radius2 * u[1]
                du[2] = theta / (2pi) * u[1] + p[1] * u[2] +
                        cubic_rate * radius2 * u[2]
                du[3] = 1.0
                nothing
            end,
            3, section, [:mu], "Suspended radial NS";
            tspan_hint=7.0, default_initial_state=zeros(3), default_params=[0.0])

        default_nf = map_normal_form(suspended_ns, :ns, zeros(2), [0.0])
        small_step_nf = map_normal_form(
            suspended_ns, :ns, zeros(2), [0.0]; normal_form_fd_step=1e-4)
        @test default_nf.status == :ok
        @test default_nf.coefficient ≈ -0.8 rtol=0.08
        @test default_nf.criticality == :supercritical
        @test small_step_nf.status in (:ok, :fd_step_unstable)
        if small_step_nf.status == :ok
            @test small_step_nf.coefficient ≈ -0.8 rtol=0.08
            @test small_step_nf.criticality == :supercritical
        else
            @test small_step_nf.coefficient === nothing
            @test small_step_nf.criticality == :unclassified
        end
    end

    @testset "Plain serialization and legacy constructor" begin
        logistic = DiscreteMap(
            (x, p) -> SVector(p[1] * x[1] * (1 - x[1])),
            1, [:r], "Logistic")
        nf = map_normal_form(logistic, :pd, [2 / 3], [3.0])
        restored_nf = deserialize_map_normal_form(serialize_map_normal_form(nf))
        @test restored_nf == nf
        unsupported_nf = serialize_map_normal_form(nf)
        unsupported_nf["format"] = "map-normal-form-v999"
        @test_throws ErrorException deserialize_map_normal_form(unsupported_nf)
        mismatched_nf = serialize_map_normal_form(nf)
        mismatched_nf["coefficientName"] = "d"
        @test_throws ErrorException deserialize_map_normal_form(mismatched_nf)
        invalid_status_nf = serialize_map_normal_form(nf)
        invalid_status_nf["status"] = "near_singular"
        @test_throws ErrorException deserialize_map_normal_form(invalid_status_nf)
        invalid_criticality_nf = serialize_map_normal_form(nf)
        invalid_criticality_nf["criticality"] = "unclassified"
        @test_throws ErrorException deserialize_map_normal_form(invalid_criticality_nf)
        nonreal_nf = serialize_map_normal_form(nf)
        nonreal_nf["coefficient"] = "9.0"
        @test_throws ErrorException deserialize_map_normal_form(nonreal_nf)
        unavailable_nf = MapNormalForm(
            :ns, :d, nothing, :unclassified, :multiple_critical_pairs,
            DynamicsKit._MAP_NS_CONVENTION)
        @test deserialize_map_normal_form(
            serialize_map_normal_form(unavailable_nf)) == unavailable_nf
        unstable_nf = MapNormalForm(
            :ns, :d, nothing, :unclassified, :fd_step_unstable,
            DynamicsKit._MAP_NS_CONVENTION)
        @test deserialize_map_normal_form(
            serialize_map_normal_form(unstable_nf)) == unstable_nf

        legacy = MapSpecialPoint(
            :pd, 3.0, [2 / 3], ComplexF64[-1], -1 + 0im, 0.0, 1, true)
        @test legacy.normal_form === nothing
        point = MapSpecialPoint(
            legacy.kind, legacy.param, legacy.state, legacy.multipliers,
            legacy.critical_multiplier, legacy.test_value, legacy.period,
            legacy.converged, nf)
        restored = deserialize_map_special_point(serialize_map_special_point(point))
        @test restored.kind == point.kind
        @test restored.param == point.param
        @test restored.state == point.state
        @test restored.multipliers == point.multipliers
        @test restored.normal_form == nf

        missing_state = serialize_map_special_point(point)
        delete!(missing_state, "state")
        @test_throws ErrorException deserialize_map_special_point(missing_state)
        nonfinite_multiplier = serialize_map_special_point(point)
        nonfinite_multiplier["multipliers"] = [[NaN, 0.0]]
        @test_throws ErrorException deserialize_map_special_point(nonfinite_multiplier)
    end
end
