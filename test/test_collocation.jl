@testset "Collocation periodic-orbit continuation" begin
    @testset "Radial oscillator recovers the analytic limit cycle" begin
        # dr/dt = r(μ - r²), dθ/dt = -1: limit cycle r = √μ, period 2π,
        # nontrivial Floquet multiplier exp(-4πμ).
        sys = radial_oscillator()
        cont = ContinuationConfig(p_min=0.15, p_max=0.45, ds=0.02, dsmax=0.05,
                                  param_index=1)
        config = CollocationConfig(continuation=cont, ntst=40, m=4, settle_time=60.0)
        result = continuation_orbit_collocation(sys, config; period=1,
                                                params=[0.3], initial_point=[1.0, 0.1])

        @test result isa OrbitBranchResult
        @test result.method == :collocation
        @test result.param_name == :μ

        mu = orbit_branch_parameters(result)
        T = orbit_branch_periods(result)
        A = orbit_branch_amplitude(result; state_index=1)

        @test length(mu) > 5
        # bothside continuation from the mid-window seed covers the whole range.
        @test minimum(mu) < 0.18
        @test maximum(mu) > 0.42
        # Analytic period 2π and amplitude 2√μ.
        @test all(abs.(T .- 2pi) .< 1e-3)
        @test maximum(abs.(A .- 2 .* sqrt.(mu)) ./ (2 .* sqrt.(mu))) < 5e-3

        # Orbit decode shape.
        i = argmin(abs.(mu .- 0.3))
        times, states = orbit_branch_orbit(result, i)
        @test size(states, 1) == 2
        @test length(times) == size(states, 2)

        # Nontrivial multiplier matches the analytic value AND the shooting return-map
        # multiplier at the same operating point.
        m = mu[i]
        multipliers = orbit_branch_multipliers(result, sys, i;
                                               solver=DynamicsKit.Tsit5(), reltol=1e-9, abstol=1e-9)
        @test maximum(abs.(multipliers)) ≈ exp(-4pi * m) atol=2e-3
        shooting = DynamicsKit._map_multipliers(sys, [-sqrt(m)], [m], 1;
                                                ode_jacobian_method=:variational,
                                                solver=DynamicsKit.Tsit5(), reltol=1e-9, abstol=1e-9)
        @test maximum(abs.(multipliers)) ≈ maximum(abs.(shooting)) atol=2e-3

        # exp(-4π·0.3) ≈ 0.023 < 1 ⟹ the orbit is stable.
        stable, _ = orbit_branch_stability(result, sys, i;
                                           solver=DynamicsKit.Tsit5(), reltol=1e-9, abstol=1e-9)
        @test stable
    end

    @testset "Vilnius period-1 orbit closes under direct integration" begin
        sys = vilnius_oscillator()
        cont = ContinuationConfig(p_min=0.20, p_max=0.35, ds=0.02, dsmax=0.03,
                                  param_index=1)
        result = continuation_orbit_collocation(sys, CollocationConfig(continuation=cont, ntst=40);
                                                period=1, params=[0.25, 30.0, 0.2],
                                                initial_point=[0.0, 0.1, 0.0])

        a = orbit_branch_parameters(result)
        T = orbit_branch_periods(result)
        @test length(a) > 3
        @test minimum(a) < 0.22 && maximum(a) > 0.33
        @test all(5.0 .< T .< 7.0)

        # Integrating the flow from the orbit start for one period returns to it.
        i = argmin(abs.(a .- 0.25))
        times, states = orbit_branch_orbit(result, i)
        x0 = states[:, 1]
        prob = DynamicsKit.ODEProblem((du, u, p, t) -> sys.f(du, u, p, 0.0), x0,
                                      (0.0, T[i]), [a[i], 30.0, 0.2])
        sol = DynamicsKit.solve(prob, DynamicsKit.Tsit5(); reltol=1e-10, abstol=1e-10)
        @test norm(sol.u[end] - x0) < 1e-2
    end

    @testset "Memristive diode bridge (stiff): collocation agrees with shooting" begin
        sys = memristive_diode_bridge()
        solver = select_ode_solver("auto")     # stiff-aware, per the MDB model notes
        base = [0.014, 6.02e-6, 0.05]
        cont = ContinuationConfig(p_min=0.012, p_max=0.016, ds=0.001, dsmax=0.002, dsmin=1e-9,
                                  max_steps=15, param_index=1, newton_tol=1e-8)
        result = continuation_orbit_collocation(sys, CollocationConfig(continuation=cont, ntst=60, settle_time=200.0);
                                                period=1, params=base, initial_point=[0.0, 0.01, 0.0],
                                                solver=solver, reltol=1e-9, abstol=1e-9)

        a = orbit_branch_parameters(result)
        T = orbit_branch_periods(result)
        @test length(a) > 3
        # bothside continuation brackets the seed parameter a = 0.014.
        @test minimum(a) < 0.014 && maximum(a) > 0.014
        # Xu et al. Fig. 6: period-1 return ≈ 26-30 dimensionless.
        @test all(26.0 .< T .< 33.0)

        i = argmin(abs.(a .- 0.014))
        a_i = a[i]

        # Orbit closure under stiff direct integration.
        times, states = orbit_branch_orbit(result, i)
        section_state = DynamicsKit._orbit_section_point(sys, times, states)
        prob = DynamicsKit.ODEProblem((du, u, p, t) -> sys.f(du, u, p, 0.0), states[:, 1],
                                      (0.0, T[i]), [a_i, base[2], base[3]])
        sol = DynamicsKit.solve(prob, solver; reltol=1e-10, abstol=1e-10)
        @test norm(sol.u[end] - states[:, 1]) < 1e-2

        # Collocation multipliers vs an independent shooting fixed point (Newton seeded at
        # the collocation section crossing). The lower branch is a stable complex pair.
        projected = DynamicsKit._project_section_state(sys.section, section_state)
        skeleton = find_periodic_skeleton(sys, [1], a_i;
                                          search_min=projected .- 2.0, search_max=projected .+ 2.0,
                                          seed_points=[projected], n_initial=1,
                                          params=[a_i, base[2], base[3]], param_index=1,
                                          tol=1e-9, max_iter=60, solver=solver, reltol=1e-9, abstol=1e-9,
                                          threaded=false, cache_enabled=false)
        @test !isempty(skeleton)
        x_shoot = collect(Float64, skeleton[1].point)
        @test norm(projected - x_shoot) < 1e-3          # collocation orbit hits the shooting fixed point

        mult_coll = orbit_branch_multipliers(result, sys, i; solver=solver, reltol=1e-9, abstol=1e-9)
        mult_shoot = DynamicsKit._map_multipliers(sys, x_shoot, [a_i, base[2], base[3]], 1;
                                                  ode_jacobian_method=:variational,
                                                  solver=solver, reltol=1e-9, abstol=1e-9)
        @test maximum(abs.(mult_coll)) ≈ maximum(abs.(mult_shoot)) atol=5e-3
        @test maximum(abs.(mult_coll)) < 1.0            # stable period-1 at a = 0.014
    end

    @testset "Invalid period is rejected" begin
        sys = radial_oscillator()
        cont = ContinuationConfig(p_min=0.15, p_max=0.45, param_index=1)
        @test_throws ArgumentError continuation_orbit_collocation(
            sys, CollocationConfig(continuation=cont); period=0, params=[0.3])
    end
end