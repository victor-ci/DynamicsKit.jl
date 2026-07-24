using Test
using DynamicsKit
using ForwardDiff
using StaticArrays

@testset "Switching map generator" begin

    # ── Helpers ──────────────────────────────────────────────────────────────

    # Finite-difference Jacobian for testing AD parity.
    function fd_jacobian(f, x; h=1e-6)
        n = length(x); y0 = f(x)
        m = length(y0)
        J = zeros(m, n)
        for j in 1:n
            xp = copy(collect(x)); xp[j] += h
            xm = copy(collect(x)); xm[j] -= h
            J[:, j] = (f(SVector{n}(xp)) - f(SVector{n}(xm))) / (2h)
        end
        J
    end
    function fd_jacobian_p(f, p; h=1e-6)
        n = length(p); y0 = f(p)
        m = length(y0)
        J = zeros(m, n)
        for j in 1:n
            pp = copy(p); pp[j] += h
            pm = copy(p); pm[j] -= h
            J[:, j] = (f(pp) - f(pm)) / (2h)
        end
        J
    end

    # ── Analytic affine-flow fixtures ─────────────────────────────────────────

    @testset "_affine_flow_2d — over-damped (diagonal A)" begin
        # A = diag(-1, -2), b = [1, 2]
        # Equilibrium: xeq = -A⁻¹ b = [1, 1]
        # x(τ) = [e^{-τ}(x₀₁-1)+1,  e^{-2τ}(x₀₂-1)+1]
        A   = SMatrix{2,2}(-1.0, 0.0, 0.0, -2.0)
        b   = SVector(1.0, 2.0)
        x0  = SVector(2.0, 3.0)
        tau = 0.5
        x   = DynamicsKit._affine_flow_2d(x0, A, b, tau)
        @test isapprox(x[1], exp(-0.5) + 1;           atol=1e-12)
        @test isapprox(x[2], 2*exp(-1.0) + 1;         atol=1e-12)
        # Zero duration: identity.
        x_zero = DynamicsKit._affine_flow_2d(x0, A, b, 0.0)
        @test isapprox(x_zero[1], x0[1]; atol=1e-12)
        @test isapprox(x_zero[2], x0[2]; atol=1e-12)
    end

    @testset "_affine_flow_2d — under-damped (pure rotation)" begin
        # A = [[0,1],[-1,0]], b = 0  →  exp(At) = [[cos t, sin t],[-sin t, cos t]]
        A   = SMatrix{2,2}(0.0, -1.0, 1.0, 0.0)   # col-major: M[2,1]=-1, M[1,2]=1
        b   = SVector(0.0, 0.0)
        x0  = SVector(1.0, 0.0)
        tau = π/2
        x   = DynamicsKit._affine_flow_2d(x0, A, b, tau)
        @test isapprox(x[1],  0.0; atol=1e-12)
        @test isapprox(x[2], -1.0; atol=1e-12)
        # Full rotation: back to start.
        x_full = DynamicsKit._affine_flow_2d(x0, A, b, 2π)
        @test isapprox(x_full[1], 1.0; atol=1e-10)
        @test isapprox(x_full[2], 0.0; atol=1e-10)
    end

    @testset "_affine_flow_2d — critically damped (Jordan block)" begin
        # A = [[-1,1],[0,-1]] (double eigenvalue -1, Jordan block), b = 0
        # exp(At) = e^{-t}·[[1,t],[0,1]]
        # x(τ) = e^{-τ}·[x₁+τ·x₂, x₂]
        A   = SMatrix{2,2}(-1.0, 0.0, 1.0, -1.0)   # col-major: col1=[-1,0], col2=[1,-1]
        b   = SVector(0.0, 0.0)
        x0  = SVector(1.0, 1.0)
        tau = 1.0
        x   = DynamicsKit._affine_flow_2d(x0, A, b, tau)
        @test isapprox(x[1], 2*exp(-1.0); atol=1e-12)
        @test isapprox(x[2],   exp(-1.0); atol=1e-12)
    end

    @testset "_affine_flow_2d — singular A (boost ON pattern)" begin
        # A = [[-α,0],[0,0]], b = [0,β]  →  x(t) = [x₁·e^{-αt}, x₂+βt]
        alpha = 2.0; beta = 3.0
        A   = SMatrix{2,2}(-alpha, 0.0, 0.0, 0.0)
        b   = SVector(0.0, beta)
        x0  = SVector(1.0, 0.0)
        tau = 0.5
        x   = DynamicsKit._affine_flow_2d(x0, A, b, tau)
        @test isapprox(x[1], exp(-alpha*tau);      atol=1e-12)
        @test isapprox(x[2], beta*tau;             atol=1e-12)
        # Larger tau.
        x2  = DynamicsKit._affine_flow_2d(x0, A, b, 1.0)
        @test isapprox(x2[1], exp(-alpha);         atol=1e-12)
        @test isapprox(x2[2], beta;                atol=1e-12)
    end

    @testset "_affine_flow_2d — nilpotent singular A" begin
        # A² = 0, so exp(Aτ) = I + Aτ and
        # ∫exp(As)b ds = τb + τ²Ab/2.
        A   = SMatrix{2,2}(0.0, 0.0, 1.0, 0.0)
        b   = SVector(0.0, 1.0)
        x0  = SVector(1.0, 0.0)
        tau = 2.0
        x   = DynamicsKit._affine_flow_2d(x0, A, b, tau)
        expected = x0 + tau * (A * x0 + b) + (tau^2 / 2) * (A * b)
        @test isapprox(x, expected; atol=1e-12)
        @test all(isfinite, x)
    end

    # ── Parity: generated map vs hand-coded map ───────────────────────────────

    @testset "Buck parity — generated vs hand-coded" begin
        sys_hand = buck_converter()
        sys_gen  = switching_map(buck_converter_description())
        @test sys_gen.dim         == 2
        @test sys_gen.param_names == [:Iref, :Ein]
        @test sys_gen.name        == "Buck Converter"
        @test length(switching_events(sys_gen)) == 1

        # Grid of (x, p) covering normal operation (tn ∈ (0, T)) and the
        # tn ≥ T rail (comparator never trips → full ON period).
        #
        # NOTE: the original buck_converter uses un-clamped tn even when
        # tn < 0 (I > Iref), which effectively runs the ON-phase in negative
        # time.  The generated switching_map clamps durations to [0, T] per
        # boost convention, so the two models differ for tn < 0; those cases
        # are intentionally excluded from this parity test.  Normal operation
        # and the tn ≥ T rail are algebraically identical.
        L = 2.2e-6; T = 1/0.5e6
        normal_cases = [
            (SVector(5.0, 0.5), [0.8, 10.0]),   # tn ≈ 1.3e-7 << T
            (SVector(3.0, 0.8), [0.8, 10.0]),   # tn = 0 (I = Iref)
            (SVector(8.0, 0.2), [0.8, 10.0]),   # tn ≈ 6.6e-7
            (SVector(4.0, 0.3), [0.7, 10.0]),   # different p
            (SVector(6.0, 0.7), [0.9,  9.0]),   # different p
            (SVector(5.0, 0.5), [1.0, 12.0]),
        ]
        # tn ≥ T rail: switch never trips within the period.
        sat_cases = [
            (SVector(9.9, 0.1), [0.8, 10.0]),   # tn ≈ 1.5e-5 >> T
            (SVector(9.5, 0.1), [0.8, 10.0]),   # tn = L*0.7/0.5 = 3.1e-6 > T
        ]

        max_delta = 0.0
        for (x, p) in [normal_cases; sat_cases]
            x_hand = sys_hand.f(x, p)
            x_gen  = sys_gen.f(x, p)
            @test all(isfinite.(x_gen))
            delta = maximum(abs.(x_gen .- x_hand))
            max_delta = max(max_delta, delta)
            @test delta < 1e-10   # algebraic parity (floating-point rounding only)
        end
        @test max_delta < 1e-10

        # Degenerate case: Ein = Vn → tn = Inf → generated map clamps to T (full ON).
        # (The original buck_converter has an unguarded cos(Inf) path; tested here
        #  for the generated map only.)
        @test all(isfinite.(sys_gen.f(SVector(8.0, 0.2), [0.5, 8.0])))

        # A negative raw ON duration skips the mode and must not apply its
        # comparator boundary override.
        x_above = SVector(5.0, 1.0)
        p_above = [0.5, 10.0]
        generated = sys_gen.f(x_above, p_above)
        desc = buck_converter_description()
        off_mode = desc.modes[2]
        expected = DynamicsKit._affine_flow_2d(
            x_above,
            SMatrix{2,2}(DynamicsKit._sw_A(off_mode.A_fn, p_above)),
            SVector{2}(DynamicsKit._sw_b(off_mode.b_fn, p_above)),
            T,
        )
        @test isapprox(generated, expected; atol=1e-12)
    end

    @testset "Boost parity — generated vs hand-coded (incl. t_on = 0 and T rails)" begin
        sys_hand = boost_converter()
        sys_gen  = switching_map(boost_converter_description())
        @test sys_gen.dim         == 2
        @test sys_gen.param_names == [:Iref, :E, :R, :Sc]
        @test sys_gen.name        == "Boost (peak-current)"
        @test length(switching_events(sys_gen)) == 2

        L = 1e-3; T = 100e-6
        # Normal operation.
        normal_cases = [
            (SVector(15.0, 0.9), [1.2, 10.0, 20.0, 0.0]),
            (SVector(12.0, 1.5), [2.0, 10.0, 20.0, 0.0]),
            (SVector(18.0, 0.5), [1.5, 12.0, 15.0, 0.5]),
        ]
        # t_on = 0 rail: I ≥ Iref → unclamped t_on ≤ 0.
        ton0_cases = [
            (SVector(15.0, 2.0), [1.5, 10.0, 20.0, 0.0]),   # I=2.0 > Iref=1.5
            (SVector(15.0, 1.5), [1.5, 10.0, 20.0, 0.0]),   # I = Iref exactly
        ]
        # t_on = T rail: (Iref-I)/(E/L+Sc) ≥ T → never trips.
        tonT_cases = [
            (SVector(15.0, 0.0), [1.2, 10.0, 20.0, 0.0]),   # t_on_raw = 1.2/10000 = 1.2e-4 > T
        ]

        max_delta = 0.0
        for (x, p) in [normal_cases; ton0_cases; tonT_cases]
            x_hand = sys_hand.f(x, p)
            x_gen  = sys_gen.f(x, p)
            @test all(isfinite.(x_gen))
            delta = maximum(abs.(x_gen .- x_hand))
            max_delta = max(max_delta, delta)
            @test delta < 1e-10
        end
        @test max_delta < 1e-10

        # Shortened parameter vector: E, R, Sc fall back to defaults (same as boost_converter).
        x_s = SVector(15.0, 0.9)
        p1  = [1.2]
        @test isapprox(sys_gen.f(x_s, p1), sys_hand.f(x_s, p1); atol=1e-10)
    end

    # ── ForwardDiff Jacobian parity ───────────────────────────────────────────

    @testset "ForwardDiff Jacobian — buck (state and parameter)" begin
        desc = buck_converter_description()
        sys_gen  = switching_map(desc)
        sys_hand = buck_converter()

        x0 = SVector(5.0, 0.5)
        p0 = [0.8, 10.0]   # tn ≈ 3.3e-8 << T = 2e-6: well inside normal operation

        # State Jacobian: AD vs legacy map AD.
        Jx_gen  = ForwardDiff.jacobian(x -> sys_gen.f(x, p0),  x0)
        Jx_hand = ForwardDiff.jacobian(x -> sys_hand.f(x, p0), x0)
        @test isapprox(Jx_gen, Jx_hand; atol=1e-8)

        # State Jacobian: AD vs finite difference.
        Jx_fd = fd_jacobian(x -> sys_gen.f(x, p0), x0)
        @test isapprox(Jx_gen, Jx_fd; atol=1e-5)

        # Parameter Jacobian: AD vs legacy map AD.
        Jp_gen  = ForwardDiff.jacobian(p -> sys_gen.f(x0, p),  p0)
        Jp_hand = ForwardDiff.jacobian(p -> sys_hand.f(x0, p), p0)
        @test isapprox(Jp_gen, Jp_hand; atol=1e-8)
    end

    @testset "ForwardDiff Jacobian — boost (state and parameter)" begin
        desc = boost_converter_description()
        sys_gen  = switching_map(desc)
        sys_hand = boost_converter()

        x0 = SVector(15.0, 0.9)
        p0 = [1.2, 10.0, 20.0, 0.0]   # normal operation: 0 < t_on < T

        Jx_gen  = ForwardDiff.jacobian(x -> sys_gen.f(x, p0),  x0)
        Jx_hand = ForwardDiff.jacobian(x -> sys_hand.f(x, p0), x0)
        @test isapprox(Jx_gen, Jx_hand; atol=1e-8)

        Jp_gen  = ForwardDiff.jacobian(p -> sys_gen.f(x0, p),  p0)
        Jp_hand = ForwardDiff.jacobian(p -> sys_hand.f(x0, p), p0)
        @test isapprox(Jp_gen, Jp_hand; atol=1e-8)
    end

    # ── Multi-mode custom fixture ─────────────────────────────────────────────

    @testset "Custom two-mode affine circuit — analytic parity" begin
        # Diagonal A in each mode; equilibria known; exact solution computable.
        # Mode 1 (ON): A1 = diag(-1,-2), b1=[1,2], xeq1=[1,1]; runs for tau1=0.3
        # Mode 2 (OFF): A2 = diag(-2,-1), b2=[2,1], xeq2=[1,1]; runs for T-tau1
        A1 = SMatrix{2,2}(-1.0, 0.0, 0.0, -2.0)
        b1 = SVector(1.0, 2.0)
        A2 = SMatrix{2,2}(-2.0, 0.0, 0.0, -1.0)
        b2 = SVector(2.0, 1.0)
        T  = 0.5
        tau1_fixed = 0.3
        tau2 = T - tau1_fixed

        # Expected: apply mode1 then mode2 explicitly.
        x0 = SVector(2.0, 3.0)
        x_mid  = DynamicsKit._affine_flow_2d(x0, A1, b1, tau1_fixed)
        x_exact = DynamicsKit._affine_flow_2d(x_mid, A2, b2, tau2)

        mode1 = AffineModeSpec(A1, b1; duration=(x, p) -> tau1_fixed)
        mode2 = AffineModeSpec(A2, b2)
        desc  = SwitchingCircuitDescription((mode1, mode2), T;
                                            param_names=Symbol[], name="Test")
        sys   = switching_map(desc)
        x_gen = sys.f(x0, [])

        @test isapprox(x_gen[1], x_exact[1]; atol=1e-12)
        @test isapprox(x_gen[2], x_exact[2]; atol=1e-12)
    end

    # ── Damping-regime coverage ───────────────────────────────────────────────

    @testset "Under-damped boost OFF flow" begin
        # Default boost params are underdamped; check OFF flow directly.
        R=20.0; L=1e-3; C=12e-6; E=10.0
        A   = SMatrix{2,2}(-1/(R*C), -1/L, 1/C, 0.0)
        b   = SVector(0.0, E/L)
        x0  = SVector(15.0, 0.5)
        tau = 60e-6
        x   = DynamicsKit._affine_flow_2d(x0, A, b, tau)
        @test all(isfinite.(x))
        # Verify via boost helper (same computation, different code path).
        v2, i2 = DynamicsKit._boost_off_flow(x0[1], x0[2], tau, E, E/R, R, L, C)
        @test isapprox(x[1], v2; atol=1e-10)
        @test isapprox(x[2], i2; atol=1e-10)
    end

    # ── Invalid descriptions ──────────────────────────────────────────────────

    @testset "Invalid period throws ArgumentError" begin
        A = SMatrix{2,2}(-1.0, 0.0, 0.0, -1.0)
        b = SVector(0.0, 0.0)
        x = SVector(1.0, 0.0)

        for bad_T in [-1.0, 0.0, NaN, -Inf]
            sys = switching_map(SwitchingCircuitDescription(
                (AffineModeSpec(A, b),), bad_T))
            @test_throws ArgumentError sys.f(x, [])
        end
    end

    @testset "NaN from duration_fn throws ArgumentError" begin
        A = SMatrix{2,2}(-1.0, 0.0, 0.0, -1.0)
        b = SVector(0.0, 0.0)
        x = SVector(1.0, 0.0)
        nan_dur = (x, p) -> NaN

        sys = switching_map(SwitchingCircuitDescription(
            (AffineModeSpec(A, b; duration=nan_dur), AffineModeSpec(A, b)), 1.0))
        @test_throws ArgumentError sys.f(x, [])
    end

    @testset "Intermediate mode with nothing duration throws at construction" begin
        A = SMatrix{2,2}(-1.0, 0.0, 0.0, -1.0)
        b = SVector(0.0, 0.0)
        @test_throws ArgumentError switching_map(SwitchingCircuitDescription(
            (AffineModeSpec(A, b),  # nothing duration but NOT the final mode
             AffineModeSpec(A, b)), 1.0))
    end

    @testset "Final mode with a duration throws at construction" begin
        A = SMatrix{2,2}(-1.0, 0.0, 0.0, -1.0)
        b = SVector(0.0, 0.0)
        @test_throws ArgumentError switching_map(SwitchingCircuitDescription(
            (AffineModeSpec(A, b; duration=(x, p) -> 0.5),), 1.0))
    end

    @testset "Empty modes throws at construction" begin
        @test_throws ArgumentError switching_map(
            SwitchingCircuitDescription((), 1.0))
    end

    @testset "Description rejects non-mode elements" begin
        A = SMatrix{2,2}(-1.0, 0.0, 0.0, -1.0)
        b = SVector(0.0, 0.0)
        @test_throws ArgumentError SwitchingCircuitDescription(
            (AffineModeSpec(A, b), "not a mode"), 1.0)
    end

    # ── Public API surface ────────────────────────────────────────────────────

    @testset "Exported symbols present" begin
        for sym in [:AffineModeSpec, :SwitchingCircuitDescription,
                    :switching_map,
                    :buck_converter_description, :boost_converter_description]
            @test sym in names(DynamicsKit)
        end
    end

    @testset "buck_converter_description returns SwitchingCircuitDescription" begin
        d = buck_converter_description()
        @test d isa SwitchingCircuitDescription
        @test d.param_names == [:Iref, :Ein]
        @test d.name == "Buck Converter"
        @test length(d.modes) == 2

        d2 = buck_converter_description(L=1e-6, T=1e-6)
        @test d2 isa SwitchingCircuitDescription
        @test_throws ArgumentError buck_converter_description(L=-1.0)
    end

    @testset "boost_converter_description returns SwitchingCircuitDescription" begin
        d = boost_converter_description()
        @test d isa SwitchingCircuitDescription
        @test d.param_names == [:Iref, :E, :R, :Sc]
        @test d.name == "Boost (peak-current)"
        @test length(d.modes) == 2

        @test_throws ArgumentError boost_converter_description(L=-1.0)
        @test_throws ArgumentError boost_converter_description(C=0.0)
    end

    @testset "switching_map forwards switching events" begin
        desc = buck_converter_description()
        sys  = switching_map(desc)
        evts = switching_events(sys)
        @test length(evts) == 1
        @test evts[1].name == "switch-time-period-border"

        desc2 = boost_converter_description()
        sys2  = switching_map(desc2)
        evts2 = switching_events(sys2)
        @test length(evts2) == 2
        @test Set(e.name for e in evts2) ==
              Set(["on-time-lower-border", "on-time-upper-border"])

        buck_singular = only(evts).guard(SVector(10.0, 0.8), [0.8, 10.0])
        @test isinf(buck_singular)
        @test !isnan(buck_singular)

        boost_singular = [event.guard(SVector(15.0, 1.2), [1.2, 10.0, 20.0, -10_000.0])
                          for event in evts2]
        @test all(isinf, boost_singular)
        @test all(value -> !isnan(value), boost_singular)
    end

    @testset "Parameter-dependent period" begin
        # Period callable: T(p) = p[1] (first param is period).
        A = SMatrix{2,2}(-1.0, 0.0, 0.0, -1.0)
        b = SVector(0.0, 0.0)
        desc = SwitchingCircuitDescription(
            (AffineModeSpec(A, b),), p -> p[1];
            param_names=[:T_period], name="Test")
        sys = switching_map(desc)
        x0 = SVector(1.0, 0.0)
        # x(T) = [e^{-T}, 0]; check that different T values give different results.
        xT1 = sys.f(x0, [0.5])
        xT2 = sys.f(x0, [1.0])
        @test isapprox(xT1[1], exp(-0.5); atol=1e-12)
        @test isapprox(xT2[1], exp(-1.0); atol=1e-12)
        # Invalid T throws.
        @test_throws ArgumentError sys.f(x0, [0.0])
    end

end
