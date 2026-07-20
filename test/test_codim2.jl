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
            max_steps=400,
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
        max_steps=300, newton_tol=1e-8, newton_max_iter=30,
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
        second_min=0.4, second_max=1.0, second_steps=3,
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
        max_steps=150, newton_tol=1e-10, newton_max_iter=30,
        detect_bifurcation=3, param_index=1, ode_jacobian_method=:variational
    )
    make_config(threaded) = Codim2Config(
        continuation=continuation,
        second_min=0.6, second_max=1.0, second_steps=2,
        second_param_index=2, fixed_params=[-0.1, 0.8],
        bifurcation_kind=:fold, endpoint_margin=0.0,
        anchor_second=0.8, diagnostics_max_points=200,
        engine=:defining_system, threaded=threaded,
        curve_continuation=ContinuationConfig(
            p_min=0.6, p_max=1.0, ds=0.02, dsmax=0.05, dsmin=1e-9,
            max_steps=50, newton_tol=1e-8, newton_max_iter=30,
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

# ---------------------------------------------------------------------------
# Helper: construct a minimal Codim2ContinuationResult from raw arrays.
# This avoids depending on a full continuation run for detector unit tests.
# ---------------------------------------------------------------------------
function _make_c2result(; kind::Symbol, p1, p2, states, mults=nothing, angles=nothing, period=1)
    n = length(p1)
    state_mat = hcat(states...)
    multipliers = mults === nothing ? [ComplexF64[] for _ in 1:n] : mults
    phase_angles = angles === nothing ? zeros(Float64, n) : angles
    Codim2ContinuationResult(
        collect(Float64, p1),
        collect(Float64, p2),
        state_mat,
        zeros(size(state_mat)),      # defining_vectors (unused by detectors)
        zeros(size(state_mat)),      # defining_vectors_imag
        phase_angles,
        zeros(Float64, n),           # fixed_point_residuals
        multipliers,
        Float64[],                   # curve_fold_secondary_values
        first(p1), first(p2),
        kind, period, "test", (:p1, :p2), :defining_system,
        DynamicsKit.Dates.now()
    )
end

# ---------------------------------------------------------------------------
# codim2_special_points validation
# ---------------------------------------------------------------------------
@testset "codim2_special_points — validation" begin
    # wrong engine
    bad_engine = _make_c2result(kind=:fold, p1=[-1.0, -0.9], p2=[0.0, 0.1],
                                 states=[[0.5], [0.5]])
    bad = Codim2ContinuationResult(
        bad_engine.primary_values, bad_engine.secondary_values,
        bad_engine.states, bad_engine.defining_vectors,
        bad_engine.defining_vectors_imag, bad_engine.phase_angles,
        bad_engine.fixed_point_residuals, bad_engine.multipliers,
        bad_engine.curve_fold_secondary_values, bad_engine.seed_primary,
        bad_engine.seed_secondary, bad_engine.bifurcation_kind,
        bad_engine.period, bad_engine.system_name, bad_engine.param_names,
        :slice_tracking, bad_engine.timestamp
    )
    sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:p1, :p2], "test")
    @test_throws ArgumentError codim2_special_points(sys, bad)

    # unknown detect kind
    good = _make_c2result(kind=:fold, p1=[-1.0, -0.9], p2=[0.0, 0.1],
                          states=[[0.5], [0.5]])
    @test_throws ArgumentError codim2_special_points(sys, good; detect=[:cusp, :bogus])

    # negative duplicate tolerance
    @test_throws ArgumentError codim2_special_points(sys, good; duplicate_primary_tol=-1.0)

    # empty detect → empty result
    pts = codim2_special_points(sys, good; detect=Symbol[])
    @test pts isa Vector{Codim2SpecialPoint}
    @test isempty(pts)

    # too few samples → empty result (n=1)
    one_sample = _make_c2result(kind=:fold, p1=[-1.0], p2=[0.0], states=[[0.5]])
    pts = codim2_special_points(sys, one_sample)
    @test isempty(pts)
end

# ---------------------------------------------------------------------------
# Cusp detection on a fold locus with a known turning point
# ---------------------------------------------------------------------------
@testset "codim2_special_points — cusp on fold locus (analytic)" begin
    # Build a fold locus that has a turning point in the primary parameter.
    # p1(t) = -(t-1)^2 + 1, p2(t) = t, t in [0, 2].
    # Primary turning point at t=1: p1=1, p2=1.
    t = range(0.0, 2.0, length=21)
    p1 = [-(ti - 1)^2 + 1 for ti in t]   # parabola, max at t=1
    p2 = collect(t)
    states = [[0.5] for _ in 1:21]
    result = _make_c2result(kind=:fold, p1=p1, p2=p2, states=states)

    sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:p1, :p2], "test fold")
    pts = codim2_special_points(sys, result; detect=[:cusp])
    @test length(pts) >= 1
    cusp = pts[1]
    @test cusp.kind === :cusp
    @test cusp.locus_kind === :fold
    # 3-point quadratic vertex locates the turning point exactly at the middle sample;
    # tolerances are tighter than simple linear interpolation would allow.
    @test abs(cusp.primary_param - 1.0) < 0.02    # near the turning point p1=1
    @test abs(cusp.secondary_param - 1.0) < 0.02  # near t=1 → p2=1
    @test cusp.status === :interpolated
    @test cusp.converged == false

    # No false detections when primary is monotone.
    p1_mono = collect(range(-1.0, 1.0, length=11))
    p2_mono = collect(range(0.0, 1.0, length=11))
    states_mono = [[0.5] for _ in 1:11]
    result_mono = _make_c2result(kind=:fold, p1=p1_mono, p2=p2_mono, states=states_mono)
    pts_mono = codim2_special_points(sys, result_mono; detect=[:cusp])
    @test isempty(pts_mono)

    # Regression: zero increment without direction reversal must not produce a false cusp.
    # p1 = [0, 0, 1]: dp = [0, 1] — no strict sign change → no detection.
    p1_zero_step = [0.0, 0.0, 1.0]
    result_zs = _make_c2result(kind=:fold, p1=p1_zero_step, p2=[0.0, 0.5, 1.0],
                                states=[[0.5], [0.5], [0.5]])
    @test isempty(codim2_special_points(sys, result_zs; detect=[:cusp]))

    # Regression: zero increment at the end — no reversal.
    # p1 = [1, 0, 0]: dp = [-1, 0] — no strict sign change → no detection.
    p1_trail_zero = [1.0, 0.0, 0.0]
    result_tz = _make_c2result(kind=:fold, p1=p1_trail_zero, p2=[0.0, 0.5, 1.0],
                                states=[[0.5], [0.5], [0.5]])
    @test isempty(codim2_special_points(sys, result_tz; detect=[:cusp]))

    # Regression: symmetric zero sample [1, 0, 1] → vertex exactly at the middle sample.
    p1_sym = [1.0, 0.0, 1.0]
    result_sym = _make_c2result(kind=:fold, p1=p1_sym, p2=[0.0, 0.5, 1.0],
                                 states=[[0.5], [0.5], [0.5]])
    pts_sym = codim2_special_points(sys, result_sym; detect=[:cusp])
    @test length(pts_sym) == 1
    @test abs(pts_sym[1].secondary_param - 0.5) < 1e-10  # exactly at middle sample

    # Cusp inapplicable to pd/ns loci.
    result_pd = _make_c2result(kind=:pd, p1=p1, p2=p2, states=states)
    @test isempty(codim2_special_points(sys, result_pd; detect=[:cusp]))

    # Inconsistent multiplier widths disable multiplier interpolation without
    # preventing coordinate/state detection.
    inconsistent_mults = [
        isodd(i) ? ComplexF64[1.0] : ComplexF64[1.0, 0.5]
        for i in eachindex(p1)
    ]
    inconsistent = _make_c2result(
        kind=:fold,
        p1=p1,
        p2=p2,
        states=states,
        mults=inconsistent_mults,
    )
    inconsistent_pts = codim2_special_points(sys, inconsistent; detect=[:cusp])
    @test !isempty(inconsistent_pts)
    @test all(isempty(point.multipliers) for point in inconsistent_pts)
end

# ---------------------------------------------------------------------------
# Fold-flip on a PD locus with analytically controlled multipliers
# ---------------------------------------------------------------------------
@testset "codim2_special_points — fold_flip on PD locus (unit multiplier crossing)" begin
    # On a PD locus (tracked multiplier ~ -1) we want a second multiplier that
    # crosses +1 (fold test function changes sign).
    # Construct: multipliers[j] = [-1, mu2(j)] where mu2 goes from 0.8 to 1.2.
    n = 11
    p1 = collect(range(-1.0, -0.9, length=n))
    p2 = collect(range(0.0, 1.0, length=n))
    states = [[0.5, 0.5] for _ in 1:n]
    # mu2 crosses +1 at j=6 (0.8 + step*5 = 0.8 + 0.2*5/10 = ... let's use linspace 0.8 to 1.2)
    mu2 = range(0.8, 1.2, length=n)
    mults = [[ComplexF64(-1.0), ComplexF64(mu2[j])] for j in 1:n]
    result = _make_c2result(kind=:pd, p1=p1, p2=p2, states=states, mults=mults)

    sys = DiscreteMap((x, p) -> SVector(p[1] * x[1], p[2] * x[2]), 2, [:p1, :p2], "test ff")
    pts = codim2_special_points(sys, result; detect=[:fold_flip])
    @test length(pts) == 1
    ff = pts[1]
    @test ff.kind === :fold_flip
    @test ff.locus_kind === :pd
    @test ff.status === :sampled
    @test abs(ff.test_value) <= 1e-5
    # Crossing at mu2 = 1 → j = 6 in 0-indexed, between samples 5 and 6 (1-indexed).
    # The interpolated primary param should be near the midpoint.
    @test p1[1] <= ff.primary_param <= p1[n]
    @test abs(ff.secondary_param - 0.5) < 0.1   # near p2=0.5 where mu2=1.0

    # fold_flip inapplicable when only 1 multiplier (1D map)
    mults_1d = [[ComplexF64(-1.0)] for _ in 1:n]
    result_1d = _make_c2result(kind=:pd, p1=p1, p2=p2, states=[[0.5] for _ in 1:n], mults=mults_1d)
    @test isempty(codim2_special_points(sys, result_1d; detect=[:fold_flip]))

    # no false detection when no sign change
    mu2_no = range(0.8, 0.95, length=n)    # stays below 1
    mults_no = [[ComplexF64(-1.0), ComplexF64(mu2_no[j])] for j in 1:n]
    result_no = _make_c2result(kind=:pd, p1=p1, p2=p2, states=states, mults=mults_no)
    @test isempty(codim2_special_points(sys, result_no; detect=[:fold_flip]))

    # fold_flip on fold locus: tracked ~ +1, second crossing -1 (pd test)
    mu2_fold = range(-1.2, -0.8, length=n)  # crosses -1
    mults_fold = [[ComplexF64(1.0), ComplexF64(mu2_fold[j])] for j in 1:n]
    result_fold = _make_c2result(kind=:fold, p1=p1, p2=p2,
                                  states=[[0.5, 0.5] for _ in 1:n], mults=mults_fold)
    pts_fold = codim2_special_points(sys, result_fold; detect=[:fold_flip])
    @test length(pts_fold) == 1
    @test pts_fold[1].locus_kind === :fold

    # A near-zero sampled endpoint that also brackets a sign change must emit
    # one sampled point, not an additional nearby interpolated duplicate.
    near_mu2 = [0.999996, 1.01]
    near_mults = [[ComplexF64(-1.0), ComplexF64(value)] for value in near_mu2]
    near_result = _make_c2result(
        kind=:pd,
        p1=[0.0, 0.01],
        p2=[0.4, 0.5],
        states=[[0.5, 0.5], [0.5, 0.5]],
        mults=near_mults,
    )
    near_pts = codim2_special_points(
        sys,
        near_result;
        detect=[:fold_flip],
        test_tolerance=1e-5,
    )
    @test length(near_pts) == 1
    @test near_pts[1].status === :sampled
    @test near_pts[1].secondary_param == 0.4
end

# ---------------------------------------------------------------------------
# Resonance 1:1 and 1:2 on an NS locus with engineered phase angles
# ---------------------------------------------------------------------------
@testset "codim2_special_points — resonance 1:1 and 1:2 on NS locus" begin
    n = 11
    p1 = fill(1.0, n)        # NS locus lives at a=1 (scaled rotation map)
    p2 = collect(range(0.0, 2.0, length=n))
    states = [[0.5, 0.5] for _ in 1:n]
    # Phase angles cross 0 (resonance 1:1) at the midpoint: use range -π/3 to π/3.
    theta_11 = range(-π/3, π/3, length=n)
    result_11 = _make_c2result(kind=:ns, p1=p1, p2=p2, states=states, angles=collect(theta_11))
    sys = DiscreteMap((x, p) -> SVector(p[1] * (cos(p[2]) * x[1] - sin(p[2]) * x[2]),
                                         p[1] * (sin(p[2]) * x[1] + cos(p[2]) * x[2])),
                      2, [:a, :b], "scaled rotation")
    pts_11 = codim2_special_points(sys, result_11; detect=[:resonance_1_1])
    @test length(pts_11) == 1
    r11 = pts_11[1]
    @test r11.kind === :resonance_1_1
    @test r11.locus_kind === :ns
    @test r11.status === :sampled
    @test abs(r11.test_value) <= 1e-5
    @test abs(r11.secondary_param - 1.0) < 0.2   # midpoint of p2 range

    # Phase angles cross π (resonance 1:2) at midpoint: range π/2 to 3π/2.
    theta_12 = range(π/2, 3π/2, length=n)
    result_12 = _make_c2result(kind=:ns, p1=p1, p2=p2, states=states, angles=collect(theta_12))
    pts_12 = codim2_special_points(sys, result_12; detect=[:resonance_1_2])
    @test length(pts_12) == 1
    r12 = pts_12[1]
    @test r12.kind === :resonance_1_2
    @test r12.locus_kind === :ns
    @test r12.status === :sampled
    @test abs(r12.test_value) <= 1e-5
    @test abs(r12.secondary_param - 1.0) < 0.2

    # Both resonances simultaneously detected.
    theta_both = range(-π/3, 4π/3, length=21)
    p2_both = collect(range(0.0, 2.0, length=21))
    p1_both = fill(1.0, 21)
    states_both = [[0.5, 0.5] for _ in 1:21]
    result_both = _make_c2result(kind=:ns, p1=p1_both, p2=p2_both, states=states_both,
                                  angles=collect(theta_both))
    pts_both = codim2_special_points(sys, result_both; detect=[:resonance_1_1, :resonance_1_2])
    kinds_both = Set([p.kind for p in pts_both])
    @test :resonance_1_1 in kinds_both
    @test :resonance_1_2 in kinds_both

    # Resonance 1:1 around 2π: sin(θ/2) test function handles angles beyond (−π,π].
    # Angles in [2π − 0.3, 2π + 0.3] cross 2π (= 0 mod 2π) → 1:1 detected.
    theta_2pi = range(2π - 0.3, 2π + 0.3, length=n)
    result_2pi = _make_c2result(kind=:ns, p1=p1, p2=p2, states=states, angles=collect(theta_2pi))
    pts_2pi = codim2_special_points(sys, result_2pi; detect=[:resonance_1_1])
    @test length(pts_2pi) >= 1
    @test pts_2pi[1].kind === :resonance_1_1

    # Resonance 1:2 around −π: cos(θ/2) test function handles negative angles.
    # Angles in [−π − 0.3, −π + 0.3] cross −π (= π mod 2π) → 1:2 detected.
    theta_neg_pi = range(-π - 0.3, -π + 0.3, length=n)
    result_neg_pi = _make_c2result(kind=:ns, p1=p1, p2=p2, states=states, angles=collect(theta_neg_pi))
    pts_neg_pi = codim2_special_points(sys, result_neg_pi; detect=[:resonance_1_2])
    @test length(pts_neg_pi) >= 1
    @test pts_neg_pi[1].kind === :resonance_1_2

    # No cross-contamination: 1:1 test function does not trigger at θ = π.
    theta_pi_only = range(π - 0.2, π + 0.2, length=n)
    result_pi = _make_c2result(kind=:ns, p1=p1, p2=p2, states=states, angles=collect(theta_pi_only))
    @test isempty(codim2_special_points(sys, result_pi; detect=[:resonance_1_1]))

    # No cross-contamination: 1:2 test function does not trigger at θ = 0.
    theta_zero_only = range(-0.2, 0.2, length=n)
    result_zero = _make_c2result(kind=:ns, p1=p1, p2=p2, states=states, angles=collect(theta_zero_only))
    @test isempty(codim2_special_points(sys, result_zero; detect=[:resonance_1_2]))

    # No detection when angles do not cross the relevant boundary.
    theta_safe = range(0.3, 0.8, length=n)
    result_safe = _make_c2result(kind=:ns, p1=p1, p2=p2, states=states, angles=collect(theta_safe))
    @test isempty(codim2_special_points(sys, result_safe; detect=[:resonance_1_1, :resonance_1_2]))

    # Resonance inapplicable to fold/pd loci.
    result_fold = _make_c2result(kind=:fold, p1=p1, p2=p2, states=states, angles=collect(theta_11))
    @test isempty(codim2_special_points(sys, result_fold; detect=[:resonance_1_1]))
end

# ---------------------------------------------------------------------------
# Generalized flip on a 1D map whose c coefficient changes sign
# ---------------------------------------------------------------------------
@testset "codim2_special_points — generalized flip (normal-form c sign change)" begin
    # Map F(x, p) = -(1 + p1)*x + p2*x^3.
    # At x=0, p1=0 this is a PD point (multiplier = -1).
    # The flip normal-form coefficient c ~ p2 (positive → supercritical, negative → subcritical).
    # The PD locus (fixed point x=0, multiplier -1) is p1 = 0 for all p2.
    # We synthesise a Codim2ContinuationResult along that locus, varying p2 from -0.5 to 0.5.
    # The actual c sign change is detected by evaluating map_normal_form at each sample.
    sys = DiscreteMap(
        (x, p) -> SVector(-(1 + p[1]) * x[1] + p[2] * x[1]^3),
        1, [:p1, :p2], "Flip c sign change"
    )
    n = 10                                       # even: avoids placing a sample exactly at p2=0
    p1_vals = fill(0.0, n)                      # PD locus lives at p1=0
    p2_vals = collect(range(-0.4, 0.4, length=n))  # c ~ p2, crosses zero between samples 5 and 6
    states  = [[0.0] for _ in 1:n]              # fixed point at x=0
    result  = _make_c2result(kind=:pd, p1=p1_vals, p2=p2_vals, states=states)

    pts = codim2_special_points(sys, result;
                                 detect=[:generalized_flip],
                                 base_params=[0.0, 0.0],
                                 param_index=1, second_param_index=2)
    @test length(pts) >= 1
    gf = pts[1]
    @test gf.kind === :generalized_flip
    @test gf.locus_kind === :pd
    @test gf.status === :interpolated
    @test abs(gf.secondary_param) < 0.2
    # Interpolated coefficient-zero point: attaching a nonzero-coefficient normal form
    # from the nearest bracketing sample would be misleading, so normal_form is nothing.
    @test gf.normal_form === nothing

    # Explicitly requesting :generalized_flip on a :pd locus without base_params → error.
    @test_throws ArgumentError codim2_special_points(sys, result; detect=[:generalized_flip])
    @test_throws ArgumentError codim2_special_points(
        sys, result;
        detect=[:generalized_flip],
        base_params=[0.0, 0.0],
        param_index=1,
        second_param_index=1,
    )
    @test_throws ArgumentError codim2_special_points(
        sys, result;
        detect=[:generalized_flip],
        base_params=[0.0, 0.0],
        param_index=1,
        second_param_index=2,
        linked_param_indices=[0],
    )
    @test_throws ArgumentError codim2_special_points(
        sys, result;
        detect=[:generalized_flip],
        base_params=[0.0, 0.0],
        param_index=1,
        second_param_index=2,
        linked_param_indices=[2],
    )
    @test_throws ArgumentError codim2_special_points(
        sys, result;
        detect=[:generalized_flip],
        base_params=[0.0, 0.0],
        param_index=1,
        second_param_index=2,
        second_linked_param_indices=[2],
    )

    # Default all-kinds pass (detect=nothing) on a :pd locus without base_params → no error,
    # generalized_flip simply skipped.
    pts_default = codim2_special_points(sys, result)
    @test pts_default isa Vector{Codim2SpecialPoint}

    # No false point when c does not change sign (p2 always positive).
    p2_pos = collect(range(0.1, 0.8, length=n))
    result_pos = _make_c2result(kind=:pd, p1=p1_vals, p2=p2_pos, states=states)
    pts_pos = codim2_special_points(sys, result_pos;
                                     detect=[:generalized_flip],
                                     base_params=[0.0, 0.0],
                                     param_index=1, second_param_index=2)
    @test isempty(pts_pos)

    # generalized_flip inapplicable on fold/ns loci — no error even without base_params.
    result_fold = _make_c2result(kind=:fold, p1=p1_vals, p2=p2_vals, states=states)
    @test isempty(codim2_special_points(sys, result_fold; detect=[:generalized_flip]))
    # With base_params on an inapplicable locus also returns empty.
    @test isempty(codim2_special_points(sys, result_fold;
                                         detect=[:generalized_flip],
                                         base_params=[0.0, 0.0],
                                         param_index=1, second_param_index=2))
end

# ---------------------------------------------------------------------------
# Bautin argument validation and inapplicability coverage
# ---------------------------------------------------------------------------
@testset "codim2_special_points — bautin validation" begin
    # Scaled rotation map (no cubic) used purely for validation tests.
    rot = DiscreteMap(
        (x, p) -> begin
            c = cos(p[2]); s = sin(p[2])
            SVector(p[1] * (c * x[1] - s * x[2]), p[1] * (s * x[1] + c * x[2]))
        end,
        2, [:a, :b], "Scaled rotation"
    )
    n = 7
    p1_vals = fill(1.0, n)
    p2_vals = collect(range(0.4, 0.8, length=n))
    states_ns = [[0.0, 0.0] for _ in 1:n]
    result_ns = _make_c2result(kind=:ns, p1=p1_vals, p2=p2_vals, states=states_ns)

    # Explicitly requesting :bautin on a :ns locus without base_params → ArgumentError.
    @test_throws ArgumentError codim2_special_points(rot, result_ns; detect=[:bautin])

    # Default all-kinds pass without base_params → no error, bautin skipped.
    pts_default = codim2_special_points(rot, result_ns)
    @test pts_default isa Vector{Codim2SpecialPoint}

    # bautin inapplicable to pd/fold loci — no error even without base_params.
    result_pd = _make_c2result(kind=:pd, p1=p1_vals, p2=p2_vals, states=states_ns)
    @test isempty(codim2_special_points(rot, result_pd; detect=[:bautin]))
    # With base_params on :pd locus: inapplicable → empty.
    @test isempty(codim2_special_points(rot, result_pd;
                                         detect=[:bautin],
                                         base_params=[1.0, 0.6],
                                         param_index=1, second_param_index=2))
end

# ---------------------------------------------------------------------------
# Deduplication and sorting
# ---------------------------------------------------------------------------
@testset "codim2_special_points — deduplication and sorting" begin
    # Construct a fold locus with two distinct turning points.
    # p1(t) = sin(2π t), t in [0,1]: turning points at t=0.25 and t=0.75.
    t = range(0.0, 1.0, length=41)
    p1 = sin.(2π .* t)
    p2 = collect(t)
    states = [[0.5] for _ in 1:41]
    result = _make_c2result(kind=:fold, p1=collect(p1), p2=collect(p2), states=states)
    sys = DiscreteMap((x, p) -> SVector(p[1] * x[1]), 1, [:p1, :p2], "test")

    pts = codim2_special_points(sys, result; detect=[:cusp])
    # Expect exactly 2 cusps, one near t=0.25 (p2≈0.25) and one near t=0.75 (p2≈0.75).
    @test length(pts) == 2
    @test pts[1].kind === :cusp
    @test pts[2].kind === :cusp
    # Sorted by (kind, secondary_param, primary_param): first has smaller secondary_param.
    @test pts[1].secondary_param < pts[2].secondary_param

    # Deduplication: duplicate cusp with very close parameters is removed.
    # Add an engineered third "cusp" at essentially the same location as the first.
    # We do this by constructing a locus with a very tight double turning point.
    # Use the test for explicit tolerance: two cusps within 1e-4 secondary -> one survives.
    t_dup = [0.0, 0.1, 0.200, 0.201, 0.3, 0.4, 0.5]
    p1_dup = [0.0, 0.5, 1.0, 1.0+1e-6, 0.5, 0.0, -0.5]  # two tiny back-forth near t=0.2
    p2_dup = collect(t_dup)
    states_dup = [[0.5] for _ in 1:7]
    result_dup = _make_c2result(kind=:fold, p1=p1_dup, p2=p2_dup, states=states_dup)
    pts_dup = codim2_special_points(sys, result_dup;
                                     detect=[:cusp],
                                     duplicate_secondary_tol=0.01)
    @test length(pts_dup) <= 1   # duplicate within tol should be removed
end

# ---------------------------------------------------------------------------
# Serializer round-trip
# ---------------------------------------------------------------------------
@testset "codim2_special_points — Codim2SpecialPoint serializer round-trip" begin
    # Obtain a valid MapNormalForm by actually evaluating the flip normal form.
    _flip_sys = DiscreteMap(
        (x, p) -> SVector(-(1 + p[1]) * x[1] + p[2] * x[1]^3),
        1, [:p1, :p2], "Flip for serializer test"
    )
    nf = map_normal_form(_flip_sys, :pd, [0.0], [0.0, 0.3]; period=1)
    @test nf.status === :ok
    pt = Codim2SpecialPoint(
        :generalized_flip, :pd,
        -0.5, 0.2,
        [0.1, 0.2],
        ComplexF64[-1.0 + 0.0im, 0.5 + 0.3im],
        0.0, 2, false, :interpolated, nf
    )

    d = serialize_codim2_special_point(pt)
    @test d isa Dict
    @test d["format"] == "codim2-special-point-v1"
    @test d["kind"] == "generalized_flip"
    @test d["locusKind"] == "pd"

    recovered = deserialize_codim2_special_point(d)
    @test recovered isa Codim2SpecialPoint
    @test recovered.kind === :generalized_flip
    @test recovered.locus_kind === :pd
    @test recovered.primary_param == pt.primary_param
    @test recovered.secondary_param == pt.secondary_param
    @test recovered.state == pt.state
    @test recovered.multipliers == pt.multipliers
    @test recovered.test_value == pt.test_value
    @test recovered.period == pt.period
    @test recovered.converged == pt.converged
    @test recovered.status === :interpolated
    @test recovered.normal_form !== nothing
    @test recovered.normal_form.kind === :pd
    @test abs(recovered.normal_form.coefficient - nf.coefficient) < 1e-10  # round-trip exact

    # Round-trip without normal form.
    pt_no_nf = Codim2SpecialPoint(:cusp, :fold, 0.1, 0.5, [0.3], ComplexF64[],
                                   0.0, 1, false, :interpolated, nothing)
    d2 = serialize_codim2_special_point(pt_no_nf)
    r2 = deserialize_codim2_special_point(d2)
    @test r2.kind === :cusp
    @test r2.normal_form === nothing

    # Round-trip with unavailable status.
    pt_unavail = Codim2SpecialPoint(:bautin, :ns, 0.0, 0.5, [0.0, 0.0], ComplexF64[],
                                     NaN, 1, false, :unavailable, nothing)
    d3 = serialize_codim2_special_point(pt_unavail)
    @test d3["status"] == "unavailable"
    r3 = deserialize_codim2_special_point(d3)
    @test r3.status === :unavailable
    @test isnan(r3.test_value)

    # Invalid format string → error.
    bad = copy(d)
    bad["format"] = "codim2-special-point-v99"
    @test_throws ArgumentError deserialize_codim2_special_point(bad)

    # Invalid kind → error.
    bad2 = copy(d)
    bad2["kind"] = "cusp_bogus"
    @test_throws ArgumentError deserialize_codim2_special_point(bad2)
end

# ---------------------------------------------------------------------------
# End-to-end defining-system resonance detection on an analytic NS locus
# ---------------------------------------------------------------------------
@testset "codim2_special_points — end-to-end NS resonance (scaled-rotation)" begin
    # Scaled-rotation map x -> a R(b) x + (1,0): Jacobian a·R(b), NS locus at a=1.
    # Phase angle = b along the locus (Jacobian eigenvalue angle).
    # Sweep b over a positive range away from b=0 (fold degeneration) and b=π (PD
    # degeneration): the NS defining system Jacobian is singular at both endpoints,
    # so the continuation correctly terminates before reaching them.
    rot = DiscreteMap(
        (x, p) -> begin
            c = cos(p[2]); s = sin(p[2])
            SVector(p[1] * (c * x[1] - s * x[2]) + 1.0,
                    p[1] * (s * x[1] + c * x[2]))
        end,
        2, [:a, :b], "Scaled rotation (resonance e2e)"
    )
    continuation = ContinuationConfig(
        p_min=0.6, p_max=1.4, ds=0.02, dsmax=0.05, dsmin=1e-8,
        max_steps=80, newton_tol=1e-12, newton_max_iter=25,
        detect_bifurcation=3, param_index=1
    )
    config = Codim2Config(
        continuation=continuation,
        second_min=0.1, second_max=0.5, second_steps=3,
        second_param_index=2, fixed_params=[0.9, 0.3],
        bifurcation_kind=:ns, endpoint_margin=0.0,
        diagnostics_max_points=200, engine=:defining_system,
        anchor_second=0.3
    )
    curve = codim2_curve(rot, config)
    @test curve isa Codim2ContinuationResult
    @test curve.engine === :defining_system
    @test curve.bifurcation_kind === :ns
    @test length(curve.primary_values) >= 5
    # NS locus is exactly a=1 for the linear rotation map.
    @test maximum(abs.(curve.primary_values .- 1.0)) < 1e-8
    # Phase angles equal the rotation angle b along the locus.
    @test maximum(abs.(abs.(curve.phase_angles) .- abs.(curve.secondary_values))) < 1e-7

    # Run the full detector pipeline.
    pts = codim2_special_points(rot, curve; detect=[:resonance_1_1, :resonance_1_2])
    @test pts isa Vector{Codim2SpecialPoint}
    # The locus stays in b ∈ (0.1, 0.5): all phase angles are negative (b > 0).
    # No 1:1 resonance (b never crosses 0) and no 1:2 resonance (b never crosses π).
    # This verifies no false positives on a well-defined locus segment.
    r11_pts = filter(p -> p.kind === :resonance_1_1, pts)
    @test isempty(r11_pts)   # correct: locus does not span b=0

    # Pipeline on the affine PD locus: cusp inapplicable to PD, returns empty.
    affine = DiscreteMap((x, p) -> SVector(p[1] * x[1] + p[2]), 1, [:a, :b], "Affine")
    aff_cont = ContinuationConfig(
        p_min=-1.4, p_max=-0.6, ds=0.02, dsmax=0.05, dsmin=1e-6,
        max_steps=80, newton_tol=1e-10, newton_max_iter=20,
        detect_bifurcation=3, param_index=1
    )
    aff_config = Codim2Config(
        continuation=aff_cont,
        second_min=-0.3, second_max=0.3, second_steps=6,
        second_param_index=2, fixed_params=[-1.2, 0.0],
        bifurcation_kind=:pd, endpoint_margin=0.02,
        diagnostics_max_points=0, engine=:defining_system
    )
    aff_curve = codim2_curve(affine, aff_config)
    @test aff_curve.engine === :defining_system
    aff_pts = codim2_special_points(affine, aff_curve; detect=[:cusp])
    @test isempty(aff_pts)   # cusp not applicable on :pd locus
end

# ---------------------------------------------------------------------------
# Codim2SpecialPoint type export and public API surface
# ---------------------------------------------------------------------------
@testset "codim2_special_points — public API surface" begin
    @test :Codim2SpecialPoint in names(DynamicsKit)
    @test :codim2_special_points in names(DynamicsKit)
    @test :serialize_codim2_special_point in names(DynamicsKit)
    @test :deserialize_codim2_special_point in names(DynamicsKit)
    @test codim2_special_points isa Function
    @test serialize_codim2_special_point isa Function
    @test deserialize_codim2_special_point isa Function
end

# ---------------------------------------------------------------------------
# End-to-end cusp on the analytic fold parabola F(x,p)=x+p₁+p₂x+x²
# ---------------------------------------------------------------------------
@testset "codim2_special_points — E2E cusp (fold parabola)" begin
    # Fixed point: p₁ + p₂x + x² = 0.  Multiplier F′(x) = 1+p₂+2x.
    # Fold (F′=1): x⋆ = −p₂/2.  Fold locus: p₁ = p₂²/4.
    # Cusp (dp₁/dp₂ = 0): p₂ = 0, p₁ = 0.
    sys = DiscreteMap(
        (x, p) -> SVector(x[1] + p[1] + p[2]*x[1] + x[1]^2),
        1, [:p1, :p2], "Fold parabola"
    )
    cont = ContinuationConfig(
        p_min=-0.12, p_max=0.12, ds=0.01, dsmax=0.03, dsmin=1e-9,
        max_steps=80, newton_tol=1e-12, newton_max_iter=20,
        detect_bifurcation=3, param_index=1
    )
    # Anchor at p2=0.1 (away from the cusp vertex at p2=0).  Starting at p2=0 would
    # place the bothside=true duplicate sample exactly at the cusp, creating a zero
    # primary-difference that blocks sign-change detection.  At p2=0.1 the stable
    # lower branch is x=(-0.1-sqrt(0.17))/2≈-0.256 and the fold is at p1=0.0025.
    config = Codim2Config(
        continuation=cont,
        second_min=-0.25, second_max=0.25, second_steps=7,
        second_param_index=2, fixed_params=[-0.04, 0.1],
        bifurcation_kind=:fold, endpoint_margin=0.0,
        engine=:defining_system, anchor_second=0.1
    )
    Random.seed!(1)
    curve = codim2_curve(sys, config; initial_point=[-0.256])
    @test curve.bifurcation_kind === :fold
    @test length(curve.primary_values) >= 5
    # Fold locus: p₁ = p₂²/4.
    @test maximum(abs.(curve.primary_values .- curve.secondary_values .^ 2 ./ 4)) < 1e-9
    # Fixed point x⋆ = −p₂/2.
    @test maximum(abs.(curve.states[1,:] .+ curve.secondary_values ./ 2)) < 1e-9

    pts = codim2_special_points(sys, curve; detect=[:cusp])
    @test length(pts) >= 1
    cusp = pts[1]
    @test cusp.kind === :cusp
    @test cusp.locus_kind === :fold
    @test cusp.status === :interpolated
    @test cusp.converged == false
    @test abs(cusp.primary_param) < 0.005     # near p₁=0
    @test abs(cusp.secondary_param) < 0.06    # near p₂=0
end

# ---------------------------------------------------------------------------
# End-to-end fold-flip on a generic decoupled map
# ---------------------------------------------------------------------------
@testset "codim2_special_points — E2E fold-flip (generic map)" begin
    # F₁=−(1+p₁)x gives a PD locus at p₁=0. F₂=y+p₂−y² has fixed
    # branches p₂=y² and multiplier 1−2y. PALC follows the PD locus
    # through the fold at (p₂,y)=(0,0), where the second multiplier
    # reaches +1: a fold-flip point.
    sys = DiscreteMap(
        (x, p) -> SVector(-(1+p[1])*x[1], x[2] + p[2] - x[2]^2),
        2, [:p1, :p2], "Generic fold-flip"
    )
    cont = ContinuationConfig(
        p_min=-0.1, p_max=0.1, ds=0.02, dsmax=0.04, dsmin=1e-9,
        max_steps=40, newton_tol=1e-12, newton_max_iter=20,
        detect_bifurcation=3, param_index=1
    )
    config = Codim2Config(
        continuation=cont,
        second_min=0.0, second_max=0.3, second_steps=6,
        second_param_index=2, fixed_params=[-0.05, 0.1],
        bifurcation_kind=:pd, endpoint_margin=0.0,
        diagnostics_max_points=200, engine=:defining_system,
        anchor_second=0.1
    )
    Random.seed!(1)
    curve = codim2_curve(sys, config; initial_point=[0.0, sqrt(0.1)])
    @test curve.bifurcation_kind === :pd
    @test length(curve.primary_values) >= 6
    @test maximum(abs.(curve.primary_values)) < 1e-10   # PD locus: p₁=0
    @test all(m -> any(mu -> abs(mu + 1.0) < 1e-7, m), curve.multipliers)
    @test minimum(curve.states[2, :]) < 1e-5
    @test minimum(curve.secondary_values) < 1e-4

    pts = codim2_special_points(sys, curve; detect=[:fold_flip])
    @test length(pts) >= 1
    ff = pts[1]
    @test ff.kind === :fold_flip
    @test ff.locus_kind === :pd
    @test ff.status === :sampled
    @test abs(ff.primary_param) < 1e-8
    @test abs(ff.secondary_param) < 0.01
    @test abs(ff.state[2]) < 1e-5
    @test abs(ff.test_value) < 1e-5
end

# ---------------------------------------------------------------------------
# End-to-end generalized flip F=-(1+p₁)x+p₂x³
# ---------------------------------------------------------------------------
@testset "codim2_special_points — E2E generalized flip" begin
    # PD locus: p₁=0, fixed point x=0.  c ∝ p₂ → sign change at p₂=0.
    sys = DiscreteMap(
        (x, p) -> SVector(-(1+p[1])*x[1] + p[2]*x[1]^3),
        1, [:p1, :p2], "Generalized flip"
    )
    cont = ContinuationConfig(
        p_min=-0.05, p_max=0.05, ds=0.01, dsmax=0.03, dsmin=1e-9,
        max_steps=30, newton_tol=1e-12, newton_max_iter=20,
        detect_bifurcation=3, param_index=1
    )
    # Anchor at p₂=0.2 (away from p₂=0 where the bothside=true duplicate sample
    # would place the sign-change location exactly on the duplicate gap).
    config = Codim2Config(
        continuation=cont,
        second_min=-0.35, second_max=0.35, second_steps=7,
        second_param_index=2, fixed_params=[-0.02, 0.2],
        bifurcation_kind=:pd, endpoint_margin=0.0,
        engine=:defining_system, anchor_second=0.2
    )
    Random.seed!(1)
    curve = codim2_curve(sys, config; initial_point=[0.0])
    @test curve.bifurcation_kind === :pd
    @test length(curve.primary_values) >= 5
    @test maximum(abs.(curve.primary_values)) < 1e-10   # PD locus: p₁=0

    # Increase normal_form_fd_step: for F=-(1+p₁)x+p₂x³ the cubic coefficient is
    # p₂, which is O(0.05) for small p₂.  The default fd_step=3e-3 gives a cubic
    # contribution of p₂·h³=p₂·2.7e-8, which is below FD noise for |p₂|<0.1.
    # Using fd_step=0.1 lifts the signal well above noise for all p₂ in the range.
    pts = codim2_special_points(sys, curve;
                                 detect=[:generalized_flip],
                                 base_params=[0.0, 0.0],
                                 param_index=1, second_param_index=2,
                                 normal_form_fd_step=0.1)
    @test length(pts) >= 1
    gf = pts[1]
    @test gf.kind === :generalized_flip
    @test gf.locus_kind === :pd
    @test gf.status === :interpolated
    @test abs(gf.secondary_param) < 0.1    # near p₂=0 where c changes sign
    @test gf.normal_form === nothing         # coefficient-zero interpolated point
end

# ---------------------------------------------------------------------------
# End-to-end resonance 1:1 approach on an NS locus
# ---------------------------------------------------------------------------
@testset "codim2_special_points — E2E resonance 1:1 (non-vacuous)" begin
    # Scaled-rotation map x -> a·R(b)·x + (1,0).  NS locus: a=1 exactly (linear map).
    # Phase angle = b.  The 1:1 resonance is at b=0 (eigenvalue +1 = fold condition),
    # where the NS defining-system Jacobian is singular.  The PALC continuation correctly
    # terminates before reaching b=0, so no 1:1 crossing is possible.
    # This test verifies: (1) the locus is computed correctly on the positive-b side,
    # (2) the resonance detector finds no false 1:1 positive (non-vacuous negative test).
    rot = DiscreteMap(
        (x, p) -> begin
            c = cos(p[2]); s = sin(p[2])
            SVector(p[1] * (c * x[1] - s * x[2]) + 1.0,
                    p[1] * (s * x[1] + c * x[2]))
        end,
        2, [:a, :b], "Scaled rotation 1:1"
    )
    cont = ContinuationConfig(
        p_min=0.6, p_max=1.4, ds=0.02, dsmax=0.05, dsmin=1e-8,
        max_steps=80, newton_tol=1e-12, newton_max_iter=25,
        detect_bifurcation=3, param_index=1
    )
    # Anchor at b=0.2 (positive side, away from b=0 degeneration).
    config = Codim2Config(
        continuation=cont,
        second_min=0.05, second_max=0.4, second_steps=4,
        second_param_index=2, fixed_params=[0.9, 0.2],
        bifurcation_kind=:ns, endpoint_margin=0.0,
        diagnostics_max_points=200, engine=:defining_system,
        anchor_second=0.2
    )
    Random.seed!(1)
    curve = codim2_curve(rot, config)
    @test curve.bifurcation_kind === :ns
    @test length(curve.primary_values) >= 5
    @test maximum(abs.(curve.primary_values .- 1.0)) < 1e-8   # NS locus: a=1 exactly
    # Phase angles track b and are positive here (b ∈ (0.05, 0.4)).
    @test all(>(0.0), curve.phase_angles)

    pts = codim2_special_points(rot, curve; detect=[:resonance_1_1])
    # No 1:1 resonance: the locus does not span b=0.  Verifies no false positive.
    @test isempty(pts)
end

# ---------------------------------------------------------------------------
# End-to-end resonance 1:2 on an NS locus spanning b=π
# ---------------------------------------------------------------------------
@testset "codim2_special_points — E2E resonance 1:2 (non-vacuous)" begin
    # Same linear rotation map; sweep b through π so phase angles span through −π.
    # At b=π eigenvalues are real (−a), so the anchor must be placed at b=π+0.1
    # where the eigenvalues are still complex.  The PALC can continue through b=π
    # (eigenvectors remain bounded there, unlike b=0).
    rot = DiscreteMap(
        (x, p) -> begin
            c = cos(p[2]); s = sin(p[2])
            SVector(p[1] * (c * x[1] - s * x[2]) + 1.0,
                    p[1] * (s * x[1] + c * x[2]))
        end,
        2, [:a, :b], "Scaled rotation 1:2"
    )
    cont = ContinuationConfig(
        p_min=0.6, p_max=1.4, ds=0.02, dsmax=0.05, dsmin=1e-8,
        max_steps=80, newton_tol=1e-12, newton_max_iter=25,
        detect_bifurcation=3, param_index=1
    )
    # Anchor at b=π+0.1 (past the 1:2 degeneration): eigenvalues −a·e^{±i·0.1} are complex.
    config = Codim2Config(
        continuation=cont,
        second_min=π - 0.25, second_max=π + 0.25, second_steps=4,
        second_param_index=2, fixed_params=[0.9, π + 0.1],
        bifurcation_kind=:ns, endpoint_margin=0.0,
        diagnostics_max_points=200, engine=:defining_system,
        anchor_second=π + 0.1
    )
    Random.seed!(1)
    curve = codim2_curve(rot, config)
    @test curve.bifurcation_kind === :ns
    @test length(curve.primary_values) >= 5
    @test maximum(abs.(curve.primary_values .- 1.0)) < 1e-8   # NS locus: a=1 exactly
    # Phase angles track b and are around π here (b ∈ (π−0.25, π+0.25)).
    @test minimum(curve.phase_angles) < π
    @test maximum(curve.phase_angles) > π

    pts = codim2_special_points(rot, curve; detect=[:resonance_1_2])
    @test length(pts) >= 1
    r12 = pts[1]
    @test r12.kind === :resonance_1_2
    @test r12.locus_kind === :ns
    @test r12.status === :interpolated
    @test abs(r12.secondary_param - π) < 0.09   # near b=π
end

# ---------------------------------------------------------------------------
# End-to-end Bautin point on the radial-cubic rotation map
# ---------------------------------------------------------------------------
@testset "codim2_special_points — E2E Bautin (radial-cubic rotation)" begin
    # F(x,p) = a·R(θ)·x + b·‖x‖²·x,  θ=0.7 (away from strong resonance).
    # Fixed point: x=0 for all (a,b) — the origin is always a fixed point.
    # Jacobian at x=0: a·R(θ) (cubic term vanishes).  NS locus: a=1 exactly.
    # Bautin coefficient d ∝ b → sign change at b=0.
    # Using the map without a constant offset keeps x*=0 so the NS locus is a=1
    # independent of b.
    θ_fixed = 0.7
    c0 = cos(θ_fixed); s0 = sin(θ_fixed)
    sys = DiscreteMap(
        (x, p) -> begin
            r2 = x[1]^2 + x[2]^2
            SVector(p[1]*(c0*x[1] - s0*x[2]) + p[2]*r2*x[1],
                    p[1]*(s0*x[1] + c0*x[2]) + p[2]*r2*x[2])
        end,
        2, [:a, :b], "Bautin radial cubic"
    )
    cont = ContinuationConfig(
        p_min=0.6, p_max=1.4, ds=0.02, dsmax=0.05, dsmin=1e-9,
        max_steps=80, newton_tol=1e-12, newton_max_iter=25,
        detect_bifurcation=3, param_index=1
    )
    # Anchor at b=0.1 (away from b=0 where the bothside=true duplicate sample would
    # place the sign-change location on a zero-difference gap).
    config = Codim2Config(
        continuation=cont,
        second_min=-0.2, second_max=0.2, second_steps=5,
        second_param_index=2, fixed_params=[0.9, 0.1],
        bifurcation_kind=:ns, endpoint_margin=0.0,
        diagnostics_max_points=300, engine=:defining_system,
        anchor_second=0.1
    )
    Random.seed!(1)
    curve = codim2_curve(sys, config; initial_point=[0.0, 0.0])
    @test curve.bifurcation_kind === :ns
    @test length(curve.primary_values) >= 5
    # NS locus is exactly a=1 (Jacobian at x=0 is a·R(θ) independent of b).
    @test maximum(abs.(curve.primary_values .- 1.0)) < 1e-8
    @test minimum(curve.secondary_values) < 0.0
    @test maximum(curve.secondary_values) > 0.0

    # Use a larger fd_step to resolve the Bautin coefficient for small b.
    pts = codim2_special_points(sys, curve;
                                 detect=[:bautin],
                                 base_params=[1.0, 0.0],
                                 param_index=1, second_param_index=2,
                                 normal_form_fd_step=0.05)
    @test length(pts) >= 1
    bt = pts[1]
    @test bt.kind === :bautin
    @test bt.locus_kind === :ns
    @test bt.status === :interpolated
    @test abs(bt.secondary_param) < 0.1   # near b=0 where d changes sign
    @test bt.normal_form === nothing         # coefficient-zero interpolated point
end
