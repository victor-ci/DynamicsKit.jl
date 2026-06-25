@testset "System definitions" begin
    @testset "Hénon map" begin
        sys = henon_map()
        @test sys.dim == 2
        @test sys.name == "Hénon"
        @test sys.param_names == [:a, :b]

        # Test iteration at a known point. A single-element parameter vector
        # falls back to the constructor default b = 0.3.
        # At a=0, the map is x->1+y, y->0.3x — fixed point at (10/7, 3/7)
        x = SVector(10.0/7, 3.0/7)
        x_next = sys.f(x, [0.0])
        @test isapprox(x_next[1], 1 + x[2]; atol=1e-12)
        @test isapprox(x_next[2], 0.3 * x[1]; atol=1e-12)

        # b is now a real second parameter: a full [a, b] vector overrides it.
        x_b = sys.f(x, [0.0, 0.5])
        @test isapprox(x_b[2], 0.5 * x[1]; atol=1e-12)

        # At a=1.4 the map is chaotic — just check it runs
        x0 = SVector(0.0, 0.0)
        x1 = sys.f(x0, [1.4, 0.3])
        @test length(x1) == 2
        @test all(isfinite.(x1))
    end

    @testset "Vilnius oscillator" begin
        sys = vilnius_oscillator()
        @test sys.dim == 3
        @test sys.name == "Vilnius"
        @test sys.param_names == [:a, :b, :ε]
        @test sys.section.direction == 1  # upward crossing
        @test sys.section.projection == [1, 3]
        @test sys.section.template == [0.0, 0.0, 0.0]
        @test sys.default_initial_state == [0.0, 0.1, 0.0]
        @test sys.default_params == [0.2, 30.0, 0.2]

        sys_custom = vilnius_oscillator(b=40.0, ε=0.07)
        @test sys_custom.default_params == [0.2, 40.0, 0.07]
    end

    @testset "Buck converter" begin
        sys = buck_converter()
        @test sys.dim == 2
        @test sys.name == "Buck Converter"
        @test sys.param_names == [:Iref, :Ein]

        # Test a single iteration doesn't produce NaN
        x = SVector(5.0, 0.5)
        p = [0.8, 10.0]
        x_next = sys.f(x, p)
        @test all(isfinite.(x_next))

        diag = switching_event_diagnostics(sys, [5.0, 0.8 - (1 / 0.5e6) * (10.0 - 5.0) / 2.2e-6], p)
        @test diag["eventCount"] == 1
        @test diag["nearEventCount"] == 1
        @test diag["nearestEvent"] == "switch-time-period-border"
    end

    @testset "Buck converter (voltage-mode)" begin
        sys = buck_voltage_mode()
        @test sys.dim == 2
        @test sys.name == "Buck (voltage-mode)"
        @test sys.param_names == [:E, :Vref, :R, :gain]

        # One closed-form iteration is finite and deterministic.
        x = SVector(7.0, 0.3)
        p = [24.0, 11.3, 22.0, 1.2]
        x_next = sys.f(x, p)
        @test length(x_next) == 2
        @test all(isfinite.(x_next))
        @test sys.f(x, p) == x_next  # deterministic

        # Public-behaviour invariant: at gain = 1.2 the regulated period-1 orbit is
        # stable, so iterating to a settled state yields a genuine fixed point of the
        # map, i.e. f(x*) ≈ x*. (Avoids reaching into internal flow helpers.)
        xstar = SVector(7.0, 0.3)
        for _ in 1:3000
            xstar = sys.f(xstar, p)
        end
        residual = sys.f(xstar, p) .- xstar
        @test all(isfinite.(xstar))
        @test maximum(abs.(residual)) < 1e-4

        # The fallback gain (shortened p) matches the workbench preset regime (1.2),
        # not a different chaos regime: a length-1 p reproduces the full-p result.
        @test sys.f(x, [24.0]) == sys.f(x, [24.0, 11.3, 22.0, 1.2])

        # Constructor kwargs flow through and parameter shortening uses fallbacks.
        sys2 = buck_voltage_mode(L=10e-3, C=22e-6, T=200e-6)
        x_short = sys2.f(SVector(7.0, 0.3), [24.0])  # Vref, R, gain fall back to defaults
        @test all(isfinite.(x_short))

        lower_border_v = 11.3 - 3.8 / 1.2
        upper_border_v = 11.3 - 8.2 / 1.2
        diag = switching_event_diagnostics(sys, [[lower_border_v, 0.3], [upper_border_v, 0.3]], p)
        @test diag["eventCount"] == 2
        @test diag["nearEventCount"] == 2
        @test Set(event["name"] for event in diag["events"] if event["near"]) ==
              Set(["duty-lower-border", "duty-upper-border"])
    end

    @testset "Boost converter (peak-current)" begin
        sys = boost_converter()
        @test sys.dim == 2
        @test sys.name == "Boost (peak-current)"
        @test sys.param_names == [:Iref, :E, :R, :Sc]

        # One closed-form iteration is finite and deterministic.
        x = SVector(15.0, 0.9)
        p = [1.2, 10.0, 20.0, 0.0]
        x_next = sys.f(x, p)
        @test length(x_next) == 2
        @test all(isfinite.(x_next))
        @test sys.f(x, p) == x_next  # deterministic

        # Public-behaviour invariant: at Iref = 1.2 the period-1 orbit is stable,
        # so iterating to a settled state yields a genuine fixed point, f(x*) ≈ x*.
        xstar = SVector(15.0, 0.9)
        for _ in 1:3000
            xstar = sys.f(xstar, p)
        end
        residual = sys.f(xstar, p) .- xstar
        @test all(isfinite.(xstar))
        @test maximum(abs.(residual)) < 1e-4
        @test xstar[2] > 0.0  # continuous-conduction mode: inductor current stays positive

        # The fallback parameters (shortened p) match the documented operating
        # point (E=10, R=20, Sc=0): a length-1 p reproduces the full-p result.
        @test sys.f(x, [1.2]) == sys.f(x, [1.2, 10.0, 20.0, 0.0])

        # Constructor kwargs flow through and parameter shortening uses fallbacks.
        sys2 = boost_converter(L=2e-3, C=22e-6, T=200e-6)
        x_short = sys2.f(SVector(15.0, 0.9), [1.2])  # E, R, Sc fall back to defaults
        @test all(isfinite.(x_short))

        diag = switching_event_diagnostics(sys, [[15.0, 1.2], [15.0, 0.2]], p)
        @test diag["eventCount"] == 2
        @test diag["nearEventCount"] == 2
        @test Set(event["name"] for event in diag["events"] if event["near"]) ==
              Set(["on-time-lower-border", "on-time-upper-border"])
    end

    @testset "Colpitts oscillator variants" begin
        @testset "Simple model" begin
            sys = colpitts_simple_oscillator()
            @test sys.dim == 3
            @test sys.name == "Colpitts (simple)"
            @test sys.param_names == [:C1, :C2, :beta, :V1, :V2]
            @test sys.section.direction == 1
            @test sys.section.projection == [1, 2]
            @test sys.section.template == [0.0, 0.0, 0.0]
            @test sys.default_initial_state == [0.0, 0.0, 0.0]
            @test sys.default_params == [40e-9, 40e-9, 265.0, 5.0, 5.0]

            du = zeros(3)
            u = [0.1, -1.0, 0.2]
            sys.f(du, u, sys.default_params, 0.0)

            expected_ib = (0.0 - u[2] - 0.75) / 400.0
            expected_ic = 265.0 * expected_ib
            @test isapprox(du[1], (u[3] - expected_ic) / 40e-9; rtol=1e-12)
            @test isapprox(du[2], ((-5.0 - u[2]) / 425.0 + u[3] + expected_ib) / 40e-9; rtol=1e-12)
            @test isapprox(du[3], (5.0 - u[1] - u[2] - 33.0 * u[3]) / 80e-6; rtol=1e-12)
        end

        @testset "Exponential model" begin
            sys = colpitts_exponential_oscillator()
            @test sys.dim == 3
            @test sys.name == "Colpitts (exponential)"
            @test sys.param_names == [:C1, :C2, :beta, :V1, :V2]
            @test sys.default_params == [40e-9, 40e-9, 265.0, 5.0, 5.0]

            du = zeros(3)
            u = [0.2, -0.1, 0.05]
            sys.f(du, u, sys.default_params, 0.0)

            expected_ic = 1e-15 * expm1((0.0 - u[2]) / 26e-3)
            expected_ib = expected_ic / 265.0
            @test isapprox(du[1], (u[3] - expected_ic) / 40e-9; rtol=1e-12)
            @test isapprox(du[2], ((-5.0 - u[2]) / 425.0 + u[3] + expected_ib) / 40e-9; rtol=1e-12)
            @test isapprox(du[3], (5.0 - u[1] - u[2] - 33.0 * u[3]) / 80e-6; rtol=1e-12)
        end

        @testset "Dynamic-beta model" begin
            sys = colpitts_dynamic_beta_oscillator()
            @test sys.dim == 3
            @test sys.name == "Colpitts (dynamic beta)"
            @test sys.param_names == [:C1, :C2, :V1, :V2]
            @test sys.default_params == [40e-9, 40e-9, 5.0, 5.0]

            du = zeros(3)
            u = [0.2, -0.1, 0.05]
            sys.f(du, u, sys.default_params, 0.0)

            expected_ic = 1e-15 * expm1((0.0 - u[2]) / 26e-3)
            expected_beta = max(1e-3, 328.82 * (expected_ic / 0.00025) / (1 + (expected_ic / 0.00025) + (expected_ic / 0.0034)^2))
            expected_ib = expected_ic / expected_beta
            @test isapprox(du[1], (u[3] - expected_ic) / 40e-9; rtol=1e-12)
            @test isapprox(du[2], ((-5.0 - u[2]) / 425.0 + u[3] + expected_ib) / 40e-9; rtol=1e-12)
            @test isapprox(du[3], (5.0 - u[1] - u[2] - 33.0 * u[3]) / 80e-6; rtol=1e-12)
        end
    end

    @testset "Ikeda map" begin
        sys = ikeda_map()
        @test sys.dim == 2
        @test sys.name == "Ikeda"
        @test sys.param_names == [:u, :a, :b]

        # Closed-form check at a known seed.
        x = SVector(0.0, 0.0)
        p = [0.9, 0.4, 6.0]
        x1 = sys.f(x, p)
        t0 = 0.4 - 6.0 / (1.0 + 0.0 + 0.0)
        @test isapprox(x1[1], 1 + 0.9 * (0.0 * cos(t0) - 0.0 * sin(t0)); atol=1e-12)
        @test isapprox(x1[2], 0.9 * (0.0 * sin(t0) + 0.0 * cos(t0)); atol=1e-12)
        @test x1[1] == 1.0
        @test x1[2] == 0.0

        # Constructor kwargs flow into the default parameter slots when the
        # caller omits a/b from the bifurcation vector.
        sys_custom = ikeda_map(a=0.3, b=5.0)
        xc = SVector(0.5, -0.5)
        x_short = sys_custom.f(xc, [0.85])
        denom = 1 + xc[1]^2 + xc[2]^2
        t_expected = 0.3 - 5.0 / denom
        @test isapprox(x_short[1], 1 + 0.85 * (xc[1] * cos(t_expected) - xc[2] * sin(t_expected)); atol=1e-12)
        @test isapprox(x_short[2], 0.85 * (xc[1] * sin(t_expected) + xc[2] * cos(t_expected)); atol=1e-12)
    end

    @testset "Rössler oscillator" begin
        sys = rossler_oscillator()
        @test sys.dim == 3
        @test sys.name == "Rössler"
        @test sys.param_names == [:a, :b, :c]
        @test sys.section.direction == 1  # upward y=0 crossing
        @test sys.section.projection == [1, 3]
        @test sys.section.template == [0.0, 0.0, 0.0]
        @test sys.default_initial_state == [1.0, 1.0, 1.0]
        @test sys.default_params == [0.2, 0.2, 5.7]

        # Verify the canonical ODE at (1, 1, 1).
        du = zeros(3)
        sys.f(du, [1.0, 1.0, 1.0], [0.2, 0.2, 5.7], 0.0)
        @test isapprox(du[1], -1.0 - 1.0; atol=1e-12)            # -y - z
        @test isapprox(du[2], 1.0 + 0.2 * 1.0; atol=1e-12)       # x + a*y
        @test isapprox(du[3], 0.2 + 1.0 * (1.0 - 5.7); atol=1e-12)  # b + z*(x-c)

        # Constructor kwargs flow through and parameter shortening still works.
        sys2 = rossler_oscillator(a=0.1, b=0.15, c=4.0)
        @test sys2.default_params == [0.1, 0.15, 4.0]
        du2 = zeros(3)
        sys2.f(du2, [0.5, -0.5, 1.0], [0.1], 0.0)
        @test isapprox(du2[2], 0.5 + 0.1 * (-0.5); atol=1e-12)
        # b and c fall back to constructor defaults (0.15, 4.0) when params is short
        @test isapprox(du2[3], 0.15 + 1.0 * (0.5 - 4.0); atol=1e-12)
    end

    @testset "Memristive diode bridge (BPF)" begin
        sys = memristive_diode_bridge()
        @test sys.dim == 3
        @test sys.name == "Memristive Diode Bridge"
        @test sys.param_names == [:a, :c, :k]
        @test sys.section.direction == 1           # upward y=0 crossing
        @test sys.section.projection == [1, 3]      # keep x and z
        @test sys.section.template == [0.0, 0.0, 0.0]
        @test sys.default_initial_state == [0.0, 0.01, 0.0]
        @test sys.default_params == [0.005, 6.02e-6, 0.05]

        # ODE evaluation matches the paper's dimensionless equations (eq. 11),
        # including that the overflow-safe ln(cosh) refactor equals ln(c·cosh).
        du = zeros(3)
        u = [0.5, -0.3, 0.4]
        p = [0.028, 6.02e-6, 0.05]
        sys.f(du, u, p, 0.0)
        a, c, k = p
        w = u[2] - u[1]
        @test isapprox(du[1], (c + u[3]) * tanh(w); atol=1e-12)
        @test isapprox(du[2], k * u[2] - (k + 1) * u[1] - (c + u[3]) * tanh(w); atol=1e-12)
        @test isapprox(du[3], a * (log(c * cosh(w)) - log(c + u[3])); rtol=1e-12)

        # c and k fall back to constructor defaults when params is short.
        du2 = zeros(3)
        sys.f(du2, u, [0.028], 0.0)
        @test isapprox(du2[2], 0.05 * u[2] - 1.05 * u[1] - (6.02e-6 + u[3]) * tanh(w); atol=1e-12)

        # Constructor kwargs flow into the default parameter slots.
        sys2 = memristive_diode_bridge(c=1e-5, k=0.08)
        @test sys2.default_params == [0.005, 1e-5, 0.08]
        du3 = zeros(3)
        sys2.f(du3, u, [0.028], 0.0)  # c, k fall back to 1e-5, 0.08
        @test isapprox(du3[2], 0.08 * u[2] - 1.08 * u[1] - (1e-5 + u[3]) * tanh(w); atol=1e-12)
    end

end
