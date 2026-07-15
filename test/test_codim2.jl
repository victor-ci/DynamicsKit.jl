using Test
using StaticArrays
using Plots
using Random

@testset "Codim-2 curves and plotting" begin
    sys = DiscreteMap((x, p) -> SVector(p[1] * x[1] + p[2]), 1, [:a, :b], "Affine map")
    continuation = ContinuationConfig(
        p_min=-1.4,
        p_max=-0.6,
        ds=0.02,
        dsmax=0.05,
        dsmin=1e-6,
        max_steps=120,
        newton_tol=1e-10,
        newton_max_iter=20,
        detect_bifurcation=3,
        param_index=1
    )
    config = Codim2Config(
        continuation=continuation,
        second_min=-0.3,
        second_max=0.3,
        second_steps=6,
        second_param_index=2,
        fixed_params=[-1.2, 0.0],
        bifurcation_kind=:pd,
        endpoint_margin=0.02,
        diagnostics_max_points=0
    )

    curve = codim2_curve(sys, config; initial_point=[0.0], params=[-1.2, 0.0])
    @test config.threaded == false
    @test curve.engine == :slice_tracking
    @test curve.bifurcation_kind == :pd
    @test curve.param_names == (:a, :b)
    @test all(curve.slice_statuses .== :ok)
    @test all(curve.candidate_sources .== :stability_flip)
    @test all(curve.valid_mask)
    @test length(curve.primary_values) == length(curve.secondary_values) == 7
    @test maximum(abs.(curve.primary_values .+ 1.0)) < 0.06

    field = LyapunovFieldResult(
        collect(range(-1.4, -0.6, length=7)),
        collect(range(-0.3, 0.3, length=7)),
        [a + b for a in collect(range(-1.4, -0.6, length=7)), b in collect(range(-0.3, 0.3, length=7))],
        zeros(Int, 7, 7),
        zeros(Int, 7, 7),
        fill(1, 7, 7),
        1e-3,
        "Affine map",
        (:a, :b),
        DynamicsKit.Dates.now()
    )

    codim_plot = plot_codim2(curve; base=field)
    @test codim_plot isa Plots.Plot

    overlay = plot_overlay_heatmap(field, curve)
    @test overlay isa Plots.Plot

    left = plot(curve.secondary_values, curve.primary_values; label="")
    right = plot(curve.secondary_values, curve.primary_values .+ 0.1; label="")
    pair = plot_seed_pair_composite(left, right)
    grid = plot_panel_grid([left, right, left, right]; layout=(2, 2))
    @test pair isa Plots.Plot
    @test grid isa Plots.Plot

    seeded_config = Codim2Config(
        continuation=continuation,
        second_min=-0.3,
        second_max=0.3,
        second_steps=6,
        second_param_index=2,
        fixed_params=[-1.2, 0.0],
        bifurcation_kind=:pd,
        endpoint_margin=0.02,
        primary_seed_values=fill(-1.2, 7),
        primary_min_values=fill(-1.4, 7),
        primary_max_values=fill(-0.6, 7),
        diagnostics_max_points=0,
        threaded=false,
    )
    seeded_curve = codim2_curve(sys, seeded_config; initial_point=[0.0], params=[-1.2, 0.0])
    @test seeded_curve.primary_values ≈ curve.primary_values
    @test seeded_curve.valid_mask == curve.valid_mask
    @test_throws AssertionError Codim2Config(
        continuation=continuation,
        second_min=-0.3,
        second_max=0.3,
        second_steps=6,
        second_param_index=2,
        fixed_params=[-1.2, 0.0],
        bifurcation_kind=:pd,
        primary_seed_values=fill(-1.2, 6),
        primary_min_values=fill(-1.4, 7),
        primary_max_values=fill(-0.6, 7),
        diagnostics_max_points=0,
        threaded=false,
    )
    @test_throws AssertionError Codim2Config(
        continuation=continuation,
        second_min=-0.3,
        second_max=0.3,
        second_steps=6,
        second_param_index=2,
        fixed_params=[-1.2, 0.0],
        bifurcation_kind=:pd,
        primary_seed_values=fill(-1.2, 7),
        primary_min_values=fill(-1.4, 6),
        primary_max_values=fill(-0.6, 7),
        diagnostics_max_points=0,
        threaded=false,
    )
    if Threads.nthreads() > 1
        threaded_curve = codim2_curve(
            sys,
            Codim2Config(
                continuation=continuation,
                second_min=-0.3,
                second_max=0.3,
                second_steps=6,
                second_param_index=2,
                fixed_params=[-1.2, 0.0],
                bifurcation_kind=:pd,
                endpoint_margin=0.02,
                primary_seed_values=fill(-1.2, 7),
                primary_min_values=fill(-1.4, 7),
                primary_max_values=fill(-0.6, 7),
                diagnostics_max_points=0,
                threaded=true,
            );
            initial_point=[0.0],
            params=[-1.2, 0.0],
        )
        @test threaded_curve.primary_values ≈ seeded_curve.primary_values
        @test threaded_curve.valid_mask == seeded_curve.valid_mask
        @test threaded_curve.candidate_sources == seeded_curve.candidate_sources
    end
end

@testset "Codim-2 continuous stability-flip fallback regression" begin
    sys = memristive_diode_bridge()
    cfg = Codim2Config(
        continuation=ContinuationConfig(
            p_min=0.0112,
            p_max=0.024,
            ds=0.0003,
            dsmax=0.0010,
            dsmin=1e-8,
            max_steps=800,
            newton_tol=1e-8,
            newton_max_iter=30,
            detect_bifurcation=1,
            param_index=1,
        ),
        second_min=0.02,
        second_max=0.08,
        second_steps=2,
        second_param_index=3,
        fixed_params=[0.012, 6.02e-6, 0.05],
        bifurcation_kind=:pd,
        endpoint_margin=0.0007,
        tracking_tolerance=0.01,
        anchor_second=0.05,
        diagnostics_max_points=120,
    )

    curve = codim2_curve(
        sys,
        cfg,
        1;
        search_min=[-8.0, 0.0],
        search_max=[8.0, 2.0],
        n_initial=10,
        tol=1e-7,
        max_iter=60,
        fd_step=1e-6,
        solver=select_ode_solver("auto"),
        reltol=1e-8,
        abstol=1e-8,
    )

    @test any(curve.valid_mask)
    @test any(curve.candidate_sources .== :stability_flip)
end

@testset "Codim-2 defining-system engine (discrete, analytic loci)" begin
    # Affine map x -> a x + b: period-doubling exactly at a = -1 for every b.
    sys = DiscreteMap((x, p) -> SVector(p[1] * x[1] + p[2]), 1, [:a, :b], "Affine map")
    continuation = ContinuationConfig(
        p_min=-1.4, p_max=-0.6, ds=0.02, dsmax=0.05, dsmin=1e-6,
        max_steps=120, newton_tol=1e-10, newton_max_iter=20,
        detect_bifurcation=3, param_index=1
    )
    config = Codim2Config(
        continuation=continuation,
        second_min=-0.3, second_max=0.3, second_steps=6,
        second_param_index=2, fixed_params=[-1.2, 0.0],
        bifurcation_kind=:pd, endpoint_margin=0.02,
        diagnostics_max_points=0, engine=:defining_system
    )
    curve = codim2_curve(sys, config)
    @test curve isa Codim2ContinuationResult
    @test curve.engine == :defining_system
    @test curve.bifurcation_kind == :pd
    @test curve.period == 1
    @test curve.param_names == (:a, :b)
    @test length(curve.primary_values) >= 5
    @test maximum(abs.(curve.primary_values .+ 1.0)) < 1e-10
    @test minimum(curve.secondary_values) <= -0.25
    @test maximum(curve.secondary_values) >= 0.25
    @test maximum(curve.fixed_point_residuals) < 1e-10
    @test all(m -> any(mu -> abs(mu + 1.0) < 1e-8, m), curve.multipliers)
    @test size(curve.states) == (1, length(curve.primary_values))
    @test size(curve.defining_vectors) == (1, length(curve.primary_values))

    # JLD2 round-trip stays plain-data safe.
    mktempdir() do dir
        path = joinpath(dir, "codim2_defining.jld2")
        save_result(path, curve)
        loaded = load_result(path)
        @test loaded isa Codim2ContinuationResult
        @test loaded.primary_values == curve.primary_values
        @test loaded.secondary_values == curve.secondary_values
        @test loaded.engine == :defining_system
    end

    # JSON-plain wire-format round-trip.
    wire = serialize_codim2_continuation_result(curve)
    @test wire["format"] == "codim2-continuation-v1"
    decoded = deserialize_codim2_continuation_result(wire)
    @test decoded.primary_values == curve.primary_values
    @test decoded.secondary_values == curve.secondary_values
    @test decoded.states == curve.states
    @test decoded.defining_vectors == curve.defining_vectors
    @test decoded.multipliers == curve.multipliers
    @test decoded.bifurcation_kind == curve.bifurcation_kind
    @test decoded.param_names == curve.param_names
    # Malformed multiplier pairs are rejected with a descriptive error.
    corrupt = deepcopy(wire)
    corrupt["multipliers"] = [[[1.0]]]
    @test_throws ErrorException deserialize_codim2_continuation_result(corrupt)

    # Quadratic map x -> x^2 + c + d: fold locus c = 1/4 - d, fold state x = 1/2.
    fold_sys = DiscreteMap((x, p) -> SVector(x[1]^2 + p[1] + p[2]), 1, [:c, :d], "Quadratic map")
    fold_continuation = ContinuationConfig(
        p_min=-0.5, p_max=0.45, ds=0.02, dsmax=0.05, dsmin=1e-6,
        max_steps=200, newton_tol=1e-10, newton_max_iter=20,
        detect_bifurcation=3, param_index=1
    )
    fold_config = Codim2Config(
        continuation=fold_continuation,
        second_min=-0.2, second_max=0.2, second_steps=8,
        second_param_index=2, fixed_params=[0.0, 0.0],
        bifurcation_kind=:fold, endpoint_margin=0.0,
        engine=:defining_system
    )
    fold_curve = codim2_curve(fold_sys, fold_config)
    @test fold_curve.bifurcation_kind == :fold
    analytic = 0.25 .- fold_curve.secondary_values
    @test maximum(abs.(fold_curve.primary_values .- analytic)) < 1e-10
    @test maximum(abs.(fold_curve.states[1, :] .- 0.5)) < 1e-8
    @test all(m -> any(mu -> abs(mu - 1.0) < 1e-8, m), fold_curve.multipliers)

    # Hénon fixed-point flip locus: a = 3(1 - b)^2 / 4.
    henon = henon_map()
    henon_continuation = ContinuationConfig(
        p_min=0.15, p_max=0.8, ds=0.01, dsmax=0.04, dsmin=1e-8,
        max_steps=400, newton_tol=1e-12, newton_max_iter=25,
        detect_bifurcation=3, param_index=1
    )
    henon_config = Codim2Config(
        continuation=henon_continuation,
        second_min=0.1, second_max=0.5, second_steps=8,
        second_param_index=2, fixed_params=[0.3675, 0.3],
        bifurcation_kind=:pd, endpoint_margin=0.005,
        diagnostics_max_points=300, engine=:defining_system
    )
    henon_curve = codim2_curve(henon, henon_config; initial_point=[0.7 / 0.735, 0.3 * 0.7 / 0.735])
    henon_analytic = 3.0 .* (1.0 .- henon_curve.secondary_values) .^ 2 ./ 4.0
    @test maximum(abs.(henon_curve.primary_values .- henon_analytic)) < 1e-8
    @test minimum(henon_curve.secondary_values) < 0.15
    @test maximum(henon_curve.secondary_values) > 0.45
    @test all(m -> any(mu -> abs(mu + 1.0) < 1e-6, m), henon_curve.multipliers)

end

@testset "Codim-2 defining-system engine (continuous smoke)" begin
    sys = memristive_diode_bridge()
    continuation = ContinuationConfig(
        p_min=0.0112, p_max=0.024, ds=0.0003, dsmax=0.0010, dsmin=1e-8,
        max_steps=800, newton_tol=1e-8, newton_max_iter=30,
        detect_bifurcation=1, param_index=1,
        ode_jacobian_method=:variational
    )
    # In this low-a window the period-1 branch loses stability through a
    # multiplier crossing +1 (a fold), so the defining-system target is :fold.
    # (The slice-tracking stability-flip fallback labels the same boundary :pd —
    # the defining engine's multiplier-gap check is what distinguishes them.)
    config = Codim2Config(
        continuation=continuation,
        second_min=0.04, second_max=0.06, second_steps=2,
        second_param_index=3, fixed_params=[0.012, 6.02e-6, 0.05],
        bifurcation_kind=:fold, endpoint_margin=0.0007,
        anchor_second=0.05, diagnostics_max_points=120,
        engine=:defining_system,
        curve_continuation=ContinuationConfig(
            p_min=0.04, p_max=0.06, ds=0.004, dsmax=0.01, dsmin=1e-8,
            max_steps=10, newton_tol=1e-6, newton_max_iter=30,
            detect_bifurcation=0, param_index=3
        )
    )
    curve = codim2_curve(
        sys, config;
        params=Float64[0.012, 6.02e-6, 0.05],
        search_min=[-8.0, 0.0], search_max=[8.0, 2.0], n_initial=10,
        tol=1e-7, max_iter=60, fd_step=1e-6,
        solver=select_ode_solver("auto"), reltol=1e-8, abstol=1e-8
    )
    @test curve isa Codim2ContinuationResult
    @test length(curve.primary_values) >= 2
    @test all(p -> 0.0112 <= p <= 0.024, curve.primary_values)
    @test all(s -> 0.04 - 1e-9 <= s <= 0.06 + 1e-9, curve.secondary_values)
    # Every sample must satisfy the fold condition: one return-map multiplier
    # near +1 (tolerance reflects the ODE-integration accuracy).
    @test all(m -> any(mu -> abs(mu - 1.0) < 0.05, m), curve.multipliers)
    @test maximum(curve.fixed_point_residuals) < 1e-4
end

@testset "Codim-2 defining-system engine (Neimark-Sacker, analytic locus)" begin
    # Scaled-rotation map x -> a R(b) x + t: multipliers a e^{±ib}, so the
    # Neimark-Sacker locus is exactly a = 1 for every rotation angle b, and the
    # recovered multiplier angle equals b.
    rot = DiscreteMap(
        (x, p) -> begin
            c = cos(p[2]); s = sin(p[2])
            SVector(p[1] * (c * x[1] - s * x[2]) + 1.0, p[1] * (s * x[1] + c * x[2]))
        end,
        2, [:a, :b], "Scaled rotation map"
    )
    continuation = ContinuationConfig(
        p_min=0.6, p_max=1.4, ds=0.02, dsmax=0.05, dsmin=1e-8,
        max_steps=200, newton_tol=1e-12, newton_max_iter=25,
        detect_bifurcation=3, param_index=1
    )
    config = Codim2Config(
        continuation=continuation,
        second_min=0.4, second_max=1.0, second_steps=6,
        second_param_index=2, fixed_params=[0.9, 0.7],
        bifurcation_kind=:ns, endpoint_margin=0.0,
        diagnostics_max_points=300, engine=:defining_system
    )
    curve = codim2_curve(rot, config)
    @test curve isa Codim2ContinuationResult
    @test curve.bifurcation_kind == :ns
    @test length(curve.primary_values) >= 5
    @test maximum(abs.(curve.primary_values .- 1.0)) < 1e-10
    @test minimum(curve.secondary_values) < 0.5
    @test maximum(curve.secondary_values) > 0.9
    # Multiplier angle recovered along the curve equals the rotation angle.
    @test maximum(abs.(abs.(curve.phase_angles) .- curve.secondary_values)) < 1e-8
    @test size(curve.defining_vectors_imag) == size(curve.defining_vectors)
    @test all(m -> any(mu -> imag(mu) > 0 && abs(abs(mu) - 1.0) < 1e-8, m), curve.multipliers)
    @test maximum(curve.fixed_point_residuals) < 1e-10

    # JSON-plain round-trip of the NS-specific payload (complex defining
    # vector parts and multiplier angles).
    wire = serialize_codim2_continuation_result(curve)
    decoded = deserialize_codim2_continuation_result(wire)
    @test decoded.defining_vectors_imag == curve.defining_vectors_imag
    @test decoded.phase_angles == curve.phase_angles
    @test decoded.multipliers == curve.multipliers
    @test decoded.bifurcation_kind == :ns
end

@testset "Codim-2 defining-system threading parity" begin
    # Quintic radial oscillator r' = r (p1 + p2 r^2 - r^4), theta' = -1: the
    # limit-cycle pair annihilates at the analytic fold locus p1 = -p2^2 / 4
    # (radius r^2 = p2 / 2). Non-stiff and free of coexisting sheets, so it is
    # a deterministic fixture for the ODE defining-system path; both engine
    # modes must land on the same analytic curve.
    quintic = let
        function f!(du, u, p, t)
            s2 = u[1]^2 + u[2]^2
            g = p[1] + p[2] * s2 - s2^2
            du[1] = u[2] + u[1] * g
            du[2] = -u[1] + u[2] * g
            nothing
        end
        section = PoincareSection(
            (u, t, integrator) -> u[2];
            direction=:up,
            projection=[1],
            template=[0.0, 0.0]
        )
        ContinuousODE(f!, 2, section, [:p1, :p2], "Quintic radial oscillator";
                      tspan_hint=8.0,
                      default_initial_state=[0.8, 0.0],
                      default_params=[-0.1, 0.8])
    end
    continuation = ContinuationConfig(
        p_min=-0.30, p_max=-0.04, ds=0.005, dsmax=0.02, dsmin=1e-9,
        max_steps=300, newton_tol=1e-10, newton_max_iter=30,
        detect_bifurcation=3, param_index=1, ode_jacobian_method=:variational
    )
    make_config(threaded) = Codim2Config(
        continuation=continuation,
        second_min=0.6, second_max=1.0, second_steps=4,
        second_param_index=2, fixed_params=[-0.1, 0.8],
        bifurcation_kind=:fold, endpoint_margin=0.0,
        anchor_second=0.8, diagnostics_max_points=200,
        engine=:defining_system, threaded=threaded,
        curve_continuation=ContinuationConfig(
            p_min=0.6, p_max=1.0, ds=0.02, dsmax=0.05, dsmin=1e-9,
            max_steps=100, newton_tol=1e-8, newton_max_iter=30,
            detect_bifurcation=0, param_index=2
        )
    )
    x0 = sqrt((0.8 + sqrt(0.8^2 + 4 * (-0.1))) / 2)
    kwargs = (
        params=Float64[-0.1, 0.8], initial_point=[x0],
        tol=1e-9, max_iter=60, fd_step=1e-6,
        reltol=1e-11, abstol=1e-11
    )
    # BifurcationKit draws from the global RNG; seed for run-to-run
    # comparability of the anchor slice.
    Random.seed!(0)
    serial = codim2_curve(quintic, make_config(false); kwargs...)
    Random.seed!(0)
    threaded = codim2_curve(quintic, make_config(true); kwargs...)

    for curve in (serial, threaded)
        @test curve isa Codim2ContinuationResult
        @test length(curve.primary_values) >= 10
        @test maximum(curve.secondary_values) - minimum(curve.secondary_values) > 0.3
        # Both modes must sit on the analytic fold locus p1 = -p2^2 / 4 ...
        analytic = .-(curve.secondary_values .^ 2) ./ 4
        @test maximum(abs.(curve.primary_values .- analytic)) < 1e-6
        # ... at the analytic fold radius, with the fold multiplier +1.
        @test maximum(abs.(curve.states[1, :] .^ 2 .- curve.secondary_values ./ 2)) < 1e-5
        @test all(m -> any(mu -> abs(mu - 1.0) < 1e-3, m), curve.multipliers)
    end
end
