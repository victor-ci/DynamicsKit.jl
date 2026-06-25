using Test
using StaticArrays
using Plots

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
