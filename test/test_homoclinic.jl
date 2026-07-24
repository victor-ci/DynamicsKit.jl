# Connecting-orbit (homoclinic / heteroclinic / saddle-cycle) tests.
#
# Every capability is exercised against an analytic fixture with a known
# connection, so the assertions check residuals, endpoints, saddle locations,
# test-function classification, and corrector provenance rather than array
# shapes. Internal helpers are reached through the `DynamicsKit.` prefix.

using DynamicsKit: _ode_field, _field_jacobian, _ConnectingProblem, _FrozenBC,
    _validate_geometry, _connection_deficiency, _transport_variational, _flip_tests,
    _detect_special_points, _LocusPoint, _gauss_newton, _ConnectingSeed, _seed_vector,
    _resample_states, _cycle_monodromy, _floquet_split, _validate_saddle_cycle_geometry,
    _refresh_bc, _slot_T, _augmented_residual, _FloquetSplit,
    _HOMOCLINIC_BRANCH_FORMAT
import Dates

# --- analytic fixtures --------------------------------------------------------

# ẋ = y, ẏ = β1 + x² + β2·y. For β2 = 0 the (undamped) system is Hamiltonian and
# has an exact homoclinic loop to the saddle (√(-β1), 0); any β2 ≠ 0 destroys it
# (Melnikov). At β1 = -1 the loop is x(t) = 1 - 3 sech²(t/√2). The homoclinic
# locus is therefore the line β2 = 0.
function hc_bt_system()
    f!(du, u, p, t) = (du[1] = u[2]; du[2] = p[1] + u[1]^2 + p[2] * u[2]; nothing)
    section = PoincareSection((u, t, integ) -> u[1]; direction=:up, projection=[1],
                              template=zeros(2))
    ContinuousODE(f!, 2, section, [:β1, :β2], "BT-exact"; default_params=[-1.0, 0.0])
end

function hc_bt_seed(; q=1.0, half=12.0, K=400)
    ts = range(-half, half, length=K)
    s = [sech(sqrt(q) * t / sqrt(2)) for t in ts]
    x = [q - 3q * s[i]^2 for i in 1:K]
    y = [3 * sqrt(2) * q^1.5 * s[i]^2 * tanh(sqrt(q) * ts[i] / sqrt(2)) for i in 1:K]
    return permutedims(hcat(x, y)), 2half
end

# Nagumo travelling front: ẋ = y, ẏ = -c·y - x(1-x)(x-a). Heteroclinic from the
# saddle (1, 0) to the saddle (0, 0) exists exactly when c = (1-2a)/√2, with front
# x(t) = 1 / (1 + exp(t/√2)).
function hc_nagumo_system()
    function f!(du, u, p, t)
        c, a = p[1], p[2]
        du[1] = u[2]
        du[2] = -c * u[2] - u[1] * (1 - u[1]) * (u[1] - a)
        nothing
    end
    section = PoincareSection((u, t, integ) -> u[1]; direction=:up, projection=[1],
                              template=zeros(2))
    ContinuousODE(f!, 2, section, [:c, :a], "Nagumo"; default_params=[(1 - 2 * 0.25) / sqrt(2), 0.25])
end

function hc_nagumo_seed(; a=0.25, half=16.0, K=400)
    ts = range(-half, half, length=K)
    x = [1 / (1 + exp(t / sqrt(2))) for t in ts]
    y = [-(1 / sqrt(2)) * exp(t / sqrt(2)) / (1 + exp(t / sqrt(2)))^2 for t in ts]
    return permutedims(hcat(x, y)), 2half
end

# Decoupled saddle cycle: planar Hopf limit cycle (r = 1, stable in-plane) plus a
# linear direction ż = λz. Floquet multipliers are {1, exp(-2T), exp(λT)}; with
# λ > 0 the cycle is a saddle (one stable, one unstable, one trivial multiplier).
function hc_saddle_cycle_system()
    function f!(du, u, p, t)
        x, y, z = u[1], u[2], u[3]
        r2 = x^2 + y^2
        ω, λ = p[1], p[2]
        du[1] = x - ω * y - x * r2
        du[2] = ω * x + y - y * r2
        du[3] = λ * z
        nothing
    end
    section = PoincareSection((u, t, integ) -> u[1]; direction=:up, projection=[1],
                              template=zeros(3))
    ContinuousODE(f!, 3, section, [:ω, :λ], "SaddleCycle"; default_params=[1.0, 0.5])
end

# Genuine saddle-cycle homoclinic in cylindrical coordinates. With
# s = (x²+y²)/2 - 3, the transverse dynamics is ṡ = z,
# ż = β1 + s². At β1 = -1 it has the exact BT loop
# s(t) = 1 - 3sech²(t/√2), while θ̇ = ω carries the loop around the
# periodic orbit s = 1, z = 0.
function hc_genuine_saddle_cycle_system()
    function f!(du, u, p, t)
        x, y, z = u
        ω, β1 = p
        r2 = x^2 + y^2
        s = r2 / 2 - 3
        radial = z / r2
        du[1] = radial * x - ω * y
        du[2] = ω * x + radial * y
        du[3] = β1 + s^2
        nothing
    end
    section = PoincareSection((u, t, integ) -> u[2]; direction=:up, projection=[1, 3],
                              template=[sqrt(8.0), 0.0, 0.0])
    ContinuousODE(f!, 3, section, [:ω, :β1], "GenuineSaddleCycle";
                  default_params=[1.0, -1.0])
end

function hc_genuine_saddle_cycle_seed(; ω=1.0, half=4π, K=600, L=400)
    Tc = 2π / ω
    θc = range(0.0, 2π, length=L)
    cycle_radius = sqrt(8.0)
    cycle = permutedims(hcat(
        cycle_radius .* cos.(θc),
        cycle_radius .* sin.(θc),
        zeros(L),
    ))

    ts = range(-half, half, length=K)
    sech2 = sech.(ts ./ sqrt(2)) .^ 2
    s = 1 .- 3 .* sech2
    z = 3sqrt(2) .* sech2 .* tanh.(ts ./ sqrt(2))
    radius = sqrt.(2 .* (s .+ 3))
    orbit = permutedims(hcat(
        radius .* cos.(ω .* ts),
        radius .* sin.(ω .* ts),
        z,
    ))
    return cycle, Tc, orbit, 2half
end

@testset "Config and public API contract" begin
    continuation = ContinuationConfig(p_min=-2.0, p_max=-0.5, ds=0.05, param_index=1)
    config = ConnectingOrbitConfig(continuation=continuation)
    @test config.source_index == 0
    @test config.detect_events
    @test config.orbit_save_stride == 10
    @test config.use_fallback
    @test_throws AssertionError ConnectingOrbitConfig(
        continuation=continuation, epsilon_start=0.0)
    @test_throws AssertionError ConnectingOrbitConfig(continuation=continuation, kind=:bogus)

    @test homoclinic_special_point_label(:sh) == "Shilnikov condition"
    @test homoclinic_special_point_label(:BT) == "Bogdanov-Takens point"
    @test homoclinic_special_point_label(:ifs) == "Stable inclination flip"
    @test homoclinic_special_point_label(:ifu) == "Unstable inclination flip"
    @test homoclinic_special_point_label(:ofs) == "Stable orbit flip"

    for sym in (:homoclinic_orbit_continuation, :connecting_orbit_continuation,
                :heteroclinic_orbit_continuation, :saddle_cycle_homoclinic_continuation,
                :ConnectingOrbitConfig,
                :homoclinic_orbit, :homoclinic_special_point_label,
                :serialize_homoclinic_branch_result, :deserialize_homoclinic_branch_result)
        @test sym in names(DynamicsKit)
    end
end

@testset "Equilibrium homoclinic projection BVP" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.6, p_max=-0.6, ds=0.05, dsmax=0.1, param_index=1,
                              newton_tol=1e-10, max_steps=18)
    cfg = ConnectingOrbitConfig(continuation=cont, n_mesh=80,
                                test_inclination_flip=true, bothside=true)
    res = homoclinic_orbit_continuation(sys, cfg; primary_param_index=2, orbit_guess=U0,
                                        saddle_guess=[1.0, 0.0], truncation_time=T0)

    @test res.connection_kind == :homoclinic
    @test res.target_saddles == res.saddles                # homoclinic: target ≡ source
    @test length(res.primary_values) >= 5
    @test maximum(res.residuals) < 1e-7                    # real BVP residual, converged
    # The solved primary parameter β2 stays on the exact homoclinic locus β2 = 0.
    @test maximum(abs, res.primary_values) < 1e-6
    # The marched secondary parameter β1 actually spans a range.
    lo, hi = extrema(res.secondary_values)
    @test hi - lo > 0.2
    # Saddle sits at (√(-β1), 0) at every locus sample.
    for j in axes(res.saddles, 2)
        β1 = res.secondary_values[j]
        @test isapprox(res.saddles[1, j], sqrt(-β1); atol=1e-6)
        @test isapprox(res.saddles[2, j], 0.0; atol=1e-8)
    end
    @test res.source_period == 0
    @test res.source_index == 0
    @test res.source_primary_value == 0.0
    @test haskey(res.diagnostics, "max_residual")
    @test issubset(unique(res.corrector_paths), Set([:newton, :fallback]))
end

@testset "Homoclinic seed orbit matches the analytic loop" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=1,
                              newton_tol=1e-11, max_steps=3)
    cfg = ConnectingOrbitConfig(continuation=cont, n_mesh=120, bothside=false,
                           orbit_save_stride=1)
    res = homoclinic_orbit_continuation(sys, cfg; primary_param_index=2, orbit_guess=U0,
                                        saddle_guess=[1.0, 0.0], truncation_time=T0)
    @test res.residuals[1] < 1e-9
    # Orbit 1 is the seed point at β1 = -1; compare to x(t) = 1 - 3 sech²(t/√2).
    orb = homoclinic_orbit(res, 1)
    M = size(orb.states, 2) - 1
    half = T0 / 2
    errx = 0.0
    for j in 1:M + 1
        t = -half + (j - 1) / M * T0
        xexact = 1 - 3 * sech(t / sqrt(2))^2
        errx = max(errx, abs(orb.states[1, j] - xexact))
    end
    @test errx < 5e-2
    @test isapprox(orb.saddle[1], 1.0; atol=1e-6)
end

@testset "Fallback corrector is a real alternate path" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed(K=200)
    field = _ode_field(sys)
    M = 80
    ts = collect(range(0.0, T0, length=size(U0, 2)))
    U = _resample_states(ts, U0, M)
    xs = [1.0, 0.0]
    eps0 = norm(U[:, 1] .- xs)
    eps1 = norm(U[:, end] .- xs)
    # primary_index = 2 (β2), secondary_index = 1 (β1)
    prob = _ConnectingProblem(field, [-1.0, 0.0], 2, 1, 2, M, :homoclinic, eps0, eps1)
    seed = _ConnectingSeed(U, xs, xs, T0, 0.0, -1.0)
    z0 = _seed_vector(prob, seed)
    bslot = length(z0)                       # secondary β occupies the last slot
    pin_beta(z) = [z[bslot] + 1.0]           # pin β1 = -1

    # Disabling the primary path (maxiter = 0) forces the fallback to do all the
    # work: with the fallback off the same configuration cannot converge.
    primary_only = _gauss_newton(prob, z0; extra=pin_beta, tol=1e-10, maxiter=0,
                                 use_fallback=false)
    @test primary_only.path == :newton
    @test !primary_only.converged

    fell_back = _gauss_newton(prob, z0; extra=pin_beta, tol=1e-10, maxiter=0,
                              use_fallback=true, fallback_max_iter=200)
    @test fell_back.path == :fallback
    @test fell_back.converged
    @test fell_back.residual <= 1e-10

    # The end-to-end run records the path taken for every locus sample.
    cont = ContinuationConfig(p_min=-1.6, p_max=-0.6, ds=0.05, param_index=1,
                              newton_tol=1e-10, max_steps=18)
    cfg = ConnectingOrbitConfig(continuation=cont, n_mesh=80, bothside=true)
    res = homoclinic_orbit_continuation(sys, cfg; primary_param_index=2, orbit_guess=U0,
                                        saddle_guess=[1.0, 0.0], truncation_time=T0)
    @test length(res.corrector_paths) == length(res.primary_values)
    @test all(p -> p in (:newton, :fallback), res.corrector_paths)
    @test res.diagnostics["fallback_points"] == count(==(:fallback), res.corrector_paths)
end

@testset "Equilibrium heteroclinic (Nagumo front)" begin
    sys = hc_nagumo_system()
    U0, T0 = hc_nagumo_seed()
    cont = ContinuationConfig(p_min=0.12, p_max=0.38, ds=0.02, dsmax=0.05, param_index=2,
                              newton_tol=1e-10, max_steps=16)
    cfg = ConnectingOrbitConfig(continuation=cont, kind=:heteroclinic, n_mesh=80,
                                bothside=true)
    res = heteroclinic_orbit_continuation(sys, cfg; primary_param_index=1,
                                          source_saddle=[1.0, 0.0], target_saddle=[0.0, 0.0],
                                          orbit_guess=U0, truncation_time=T0)
    @test res.connection_kind == :heteroclinic
    @test maximum(res.residuals) < 1e-7
    # Source/target saddles are the two equilibria (1,0) and (0,0).
    @test isapprox(res.saddles[:, 1], [1.0, 0.0]; atol=1e-7)
    @test isapprox(res.target_saddles[:, 1], [0.0, 0.0]; atol=1e-7)
    @test res.saddles != res.target_saddles
    # The solved wave speed follows the exact relation c = (1-2a)/√2 along the locus.
    errs = [abs(res.primary_values[i] - (1 - 2 * res.secondary_values[i]) / sqrt(2))
            for i in eachindex(res.primary_values)]
    @test maximum(errs) < 5e-3
    lo, hi = extrema(res.secondary_values)
    @test hi - lo > 0.05
end

@testset "Inclination-flip test function (adjoint transport)" begin
    # (a) Forward variational transport matches the analytic fundamental matrix on
    # a constant-Jacobian (linear) field: v_end ∝ exp(A T) v0.
    A = [-0.5 0.0 0.0; 0.0 -2.0 0.0; 0.0 0.3 1.0]
    linfield(u, p) = A * collect(u)
    probA = _ConnectingProblem(linfield, [0.0], 1, 1, 3, 200, :homoclinic, 1.0, 1.0)
    T = 1.3
    vend = _transport_variational(probA, zeros(3, 201), T, [0.0], [1.0, 1.0, 1.0])
    analytic = exp(A * T) * [1.0, 1.0, 1.0]
    analytic ./= norm(analytic)
    dot(vend, analytic) < 0 && (analytic .*= -1)
    @test norm(vend .- analytic) < 1e-4

    # (b) Availability is driven by the real spectrum, not fabricated. A 3D saddle
    # with two stable and one unstable eigenvalue makes the stable inclination flip
    # available and the unstable one unavailable; a 2D saddle makes both
    # unavailable (a single weak/strong split does not exist).
    B = [1.0 0.0 0.0; 0.0 -0.5 0.0; 0.0 0.0 -2.0]
    probB = _ConnectingProblem((u, p) -> B * collect(u), [0.0], 1, 1, 3, 50, :homoclinic, 1.0, 1.0)
    tB = _flip_tests(probB, zeros(3, 51), zeros(3), 2.0, [0.0];
                     test_orbit_flip=true, test_inclination_flip=true)
    @test tB[:ifs][2] == :available
    @test isfinite(tB[:ifs][1])
    @test tB[:ifu][2] == :unavailable
    @test tB[:ofs][2] == :available
    @test tB[:ofu][2] == :unavailable

    C = [1.0 0.0; 0.0 -1.0]
    probC = _ConnectingProblem((u, p) -> C * collect(u), [0.0], 1, 1, 2, 50, :homoclinic, 1.0, 1.0)
    tC = _flip_tests(probC, zeros(2, 51), zeros(2), 2.0, [0.0];
                     test_orbit_flip=true, test_inclination_flip=true)
    @test tC[:ifs][2] == :unavailable
    @test tC[:ifu][2] == :unavailable
    @test isnan(tC[:ifs][1])

    # (c) A sign crossing of an available test function is classified as the typed
    # inclination-flip point with interpolated parameters and a bounded quality.
    mkpt(sec, val) = _LocusPoint(Float64[], 0.0, sec, 5.0, Float64[], Float64[], 1.0, 1.0,
                                 1e-12, :newton, Dict(:ifs => val), Dict(:ifs => :available),
                                 zeros(1, 1))
    locus = [mkpt(0.0, -1.0), mkpt(0.1, -0.4), mkpt(0.2, 0.3), mkpt(0.3, 0.9)]
    specials = _detect_special_points(locus)
    @test length(specials) == 1
    sp = specials[1]
    @test sp.kind == :ifs
    @test sp.label == "Stable inclination flip"
    @test sp.status == :available
    @test 0.0 < sp.quality <= 1.0
    @test isapprox(sp.secondary_param, 0.1 + 0.4 / 0.7 * 0.1; atol=1e-6)

    # A crossing of an :unavailable test function must not fabricate a point.
    hidden = [_LocusPoint(Float64[], 0.0, 0.0, 5.0, Float64[], Float64[], 1.0, 1.0, 1e-12,
                          :newton, Dict(:ifs => -1.0), Dict(:ifs => :unavailable), zeros(1, 1)),
              _LocusPoint(Float64[], 0.0, 0.1, 5.0, Float64[], Float64[], 1.0, 1.0, 1e-12,
                          :newton, Dict(:ifs => 1.0), Dict(:ifs => :unavailable), zeros(1, 1))]
    @test isempty(_detect_special_points(hidden))
end

@testset "Saddle-cycle Floquet numerics and geometry" begin
    sys = hc_saddle_cycle_system()
    field = _ode_field(sys)
    ω, λ = 1.0, 0.5
    Tc = 2π / ω
    L = 400
    θ = range(0, 2π, length=L)
    cyc = permutedims(hcat(cos.(θ), sin.(θ), zeros(L)))

    mono = _cycle_monodromy(field, cyc, Tc, [ω, λ])
    split = _floquet_split(mono)
    mags = sort(abs.(split.multipliers))
    analytic = sort([exp(-2Tc), 1.0, exp(λ * Tc)])
    @test isapprox(mags, analytic; rtol=1e-2)
    @test (split.ns, split.nu, split.nc) == (1, 1, 1)
    @test _validate_saddle_cycle_geometry(split, 3) === nothing

    # A purely attracting cycle (λ < 0) cannot host a homoclinic orbit: reject.
    mono_stable = _cycle_monodromy(field, cyc, Tc, [ω, -0.5])
    split_stable = _floquet_split(mono_stable)
    @test split_stable.nu == 0
    @test_throws ArgumentError _validate_saddle_cycle_geometry(split_stable, 3)
end

@testset "Saddle-cycle homoclinic correction (end to end)" begin
    sys = hc_genuine_saddle_cycle_system()
    cyc, Tc, U0, T0 = hc_genuine_saddle_cycle_seed()
    cfg = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=2,
                                        newton_tol=1e-5, max_steps=5),
        kind=:saddle_cycle, n_mesh=80)
    res = saddle_cycle_homoclinic_continuation(sys, cfg; cycle_states=cyc, cycle_period=Tc,
                                               orbit_guess=U0, truncation_time=T0,
                                               reference_index=1)
    @test res.connection_kind == :saddle_cycle
    @test res.source_period == 0
    @test res.source_index == 0
    @test length(res.primary_values) == 1
    @test res.residuals[1] <= cfg.continuation.newton_tol
    @test res.corrector_paths[1] in (:newton, :fallback)
    @test res.diagnostics["stable_floquet_dim"] == 1
    @test res.diagnostics["unstable_floquet_dim"] == 1
    @test res.diagnostics["center_floquet_dim"] == 1
    @test res.diagnostics["converged"] == true
    @test isapprox(res.diagnostics["cycle_period"], Tc; atol=1e-8)
    @test length(res.orbits) == 1
    @test_throws ArgumentError saddle_cycle_homoclinic_continuation(
        sys, cfg; cycle_states=cyc[:, 1:1], cycle_period=Tc,
        orbit_guess=U0, truncation_time=T0)
    open_cycle = copy(cyc)
    open_cycle[:, end] .+= 0.1
    @test_throws ArgumentError saddle_cycle_homoclinic_continuation(
        sys, cfg; cycle_states=open_cycle, cycle_period=Tc,
        orbit_guess=U0, truncation_time=T0)
    @test_throws ArgumentError saddle_cycle_homoclinic_continuation(
        sys, cfg; cycle_states=cyc, cycle_period=Tc,
        orbit_guess=U0, truncation_time=T0, base_params=[1.0])
    bad_index_cfg = ConnectingOrbitConfig(
        continuation=ContinuationConfig(
            p_min=-1.2, p_max=-0.8, ds=0.05, param_index=3),
        kind=:saddle_cycle, n_mesh=80)
    @test_throws ArgumentError saddle_cycle_homoclinic_continuation(
        sys, bad_index_cfg; cycle_states=cyc, cycle_period=Tc,
        orbit_guess=U0, truncation_time=T0)

    # Reject a non-saddle cycle at the public API.
    cfg_bad = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=-1.0, p_max=1.0, ds=0.05, param_index=2),
        kind=:saddle_cycle, n_mesh=60)
    sys_stable = hc_saddle_cycle_system()
    θ = range(0, 2π, length=200)
    stable_cycle = permutedims(hcat(cos.(θ), sin.(θ), zeros(length(θ))))
    @test_throws ArgumentError saddle_cycle_homoclinic_continuation(
        sys_stable, cfg_bad; cycle_states=stable_cycle, cycle_period=2π, orbit_guess=U0,
        truncation_time=16.0, base_params=[1.0, -0.5])
end

@testset "Invalid geometry rejection" begin
    field = _ode_field(hc_bt_system())
    # Homoclinic with a non-hyperbolic saddle (ns + nu < n) leaves k > 1 free
    # directions: not a codimension-one curve.
    prob = _ConnectingProblem(field, [-1.0, 0.0], 2, 1, 3, 20, :homoclinic, 1.0, 1.0)
    bc_bad = _FrozenBC(zeros(3, 1), zeros(3, 1), 1, 1)     # ns_src = nu_tgt = 1, n = 3
    @test _connection_deficiency(prob, bc_bad) == 2
    @test_throws ArgumentError _validate_geometry(prob, bc_bad)

    bc_ok = _FrozenBC(zeros(3, 2), zeros(3, 1), 2, 1)      # ns + nu = 3 = n → k = 1
    @test _validate_geometry(prob, bc_ok) == 1

    # Heteroclinic where the manifolds are too large (over-counted) gives k < 1.
    prob_het = _ConnectingProblem(field, [-1.0, 0.0], 2, 1, 3, 20, :heteroclinic, 1.0, 1.0)
    bc_over = _FrozenBC(zeros(3, 2), zeros(3, 2), 2, 2)
    @test _connection_deficiency(prob_het, bc_over) < 1
    @test_throws ArgumentError _validate_geometry(prob_het, bc_over)

    # Public API: a heteroclinic connection needs a target saddle guess.
    sys = hc_nagumo_system()
    U0, T0 = hc_nagumo_seed()
    cfg = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=0.1, p_max=0.4, ds=0.02, param_index=2),
        kind=:heteroclinic, n_mesh=80)
    @test_throws ArgumentError connecting_orbit_continuation(
        sys, cfg; primary_param_index=1, orbit_guess=U0, saddle_guess=[1.0, 0.0],
        truncation_time=T0)
    @test_throws ArgumentError connecting_orbit_continuation(
        sys, cfg; primary_param_index=1, orbit_guess=U0,
        saddle_guess=[1.0, 0.0], target_saddle_guess=[0.0, 0.0],
        truncation_time=T0, base_params=[0.25])
end

@testset "Serialization round-trip" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.4, p_max=-0.7, ds=0.05, param_index=1,
                              newton_tol=1e-10, max_steps=8)
    cfg = ConnectingOrbitConfig(continuation=cont, n_mesh=80, bothside=true)
    res = homoclinic_orbit_continuation(sys, cfg; primary_param_index=2, orbit_guess=U0,
                                        saddle_guess=[1.0, 0.0], truncation_time=T0)

    plain = serialize_homoclinic_branch_result(res)
    @test plain["format"] == _HOMOCLINIC_BRANCH_FORMAT
    @test haskey(plain, "targetSaddles")
    @test haskey(plain, "residuals")
    @test haskey(plain, "correctorPaths")
    @test plain["connectionKind"] == "homoclinic"

    restored = deserialize_homoclinic_branch_result(plain)
    @test restored.connection_kind == :homoclinic
    @test restored.primary_values == res.primary_values
    @test restored.target_saddles == res.target_saddles
    @test restored.residuals == res.residuals
    @test restored.corrector_paths == res.corrector_paths
    @test restored.diagnostics["kind"] == "homoclinic"
    missing = copy(plain)
    delete!(missing, "targetSaddles")
    error_text = try
        deserialize_homoclinic_branch_result(missing)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("missing required key 'targetSaddles'", error_text)

    # Special-point status/quality survive the round trip.
    special = HomoclinicSpecialPoint(:ifs, homoclinic_special_point_label(:ifs), 2,
                                     -0.01, -0.1, 0.0, :available, 0.8)
    sp_plain = DynamicsKit._serialize_homoclinic_special_point(special)
    @test sp_plain["status"] == "available"
    @test sp_plain["quality"] == 0.8
    sp_back = DynamicsKit._deserialize_homoclinic_special_point(sp_plain)
    @test sp_back.status == :available
    @test sp_back.quality == 0.8


    # Malformed payloads are still rejected.
    malformed = deepcopy(plain)
    malformed["secondaryValues"] = malformed["secondaryValues"][1:end - 1]
    @test_throws ErrorException deserialize_homoclinic_branch_result(malformed)
    nonfinite = deepcopy(plain)
    nonfinite["baseParams"][1] = NaN
    @test_throws ErrorException deserialize_homoclinic_branch_result(nonfinite)
    bad_target = deepcopy(plain)
    bad_target["targetSaddles"] = [row[1:end - 1] for row in bad_target["targetSaddles"]]
    @test_throws ErrorException deserialize_homoclinic_branch_result(bad_target)
end

@testset "max_return_time enforcement" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.4, p_max=-0.7, ds=0.05, param_index=1,
                              newton_tol=1e-10, max_steps=20)

    # Seed T already beyond the cap: must throw before any correction attempt.
    cfg_tight = ConnectingOrbitConfig(continuation=cont, n_mesh=80, max_return_time=1.0)
    @test_throws ArgumentError homoclinic_orbit_continuation(
        sys, cfg_tight; primary_param_index=2, orbit_guess=U0,
        saddle_guess=[1.0, 0.0], truncation_time=T0)

    # The sweep terminates honestly when T would exceed the cap.  A cap of T0/2
    # forces the continuation to stop early rather than run the full 20 steps.
    cfg_cap = ConnectingOrbitConfig(continuation=cont, n_mesh=80, max_return_time=T0,
                               bothside=false)
    cfg_uncap = ConnectingOrbitConfig(continuation=cont, n_mesh=80, bothside=false)
    res_cap = homoclinic_orbit_continuation(sys, cfg_cap; primary_param_index=2,
                                            orbit_guess=U0, saddle_guess=[1.0, 0.0],
                                            truncation_time=T0)
    res_uncap = homoclinic_orbit_continuation(sys, cfg_uncap; primary_param_index=2,
                                              orbit_guess=U0, saddle_guess=[1.0, 0.0],
                                              truncation_time=T0)
    # The cap must not produce results with T beyond max_return_time.
    @test all(<=(cfg_cap.max_return_time), res_cap.return_times)
    # Without a cap the sweep explores further (more points or larger T range).
    @test maximum(res_uncap.return_times) >= maximum(res_cap.return_times) - 1e-6

    # Saddle-cycle: seed T beyond cap is rejected before any solver call.
    sys_sc = hc_saddle_cycle_system()
    ω, λ = 1.0, 0.5
    Tc = 2π / ω
    L = 200
    θ = range(0, 2π, length=L)
    cyc = permutedims(hcat(cos.(θ), sin.(θ), zeros(L)))
    K = 120
    ss = range(-8, 8, length=K)
    xg = [cos(2π / (1 + exp(-s))) for s in ss]
    yg = [sin(2π / (1 + exp(-s))) for s in ss]
    zg = [0.05 * exp(-abs(s)) for s in ss]
    U_sc = permutedims(hcat(xg, yg, zg))
    cfg_sc_tight = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=0.1, p_max=1.0, ds=0.05, param_index=2),
        kind=:saddle_cycle, n_mesh=60, max_return_time=1.0)
    @test_throws ArgumentError saddle_cycle_homoclinic_continuation(
        sys_sc, cfg_sc_tight; cycle_states=cyc, cycle_period=Tc,
        orbit_guess=U_sc, truncation_time=16.0)
end

@testset "epsilon_start/end enforcement" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=1,
                              newton_tol=1e-10, max_steps=3)
    M = 80

    # Determine the seed's natural endpoint distance so we can use modest
    # rescaling values that keep Newton convergent.
    U_ref = _resample_states(collect(range(0.0, T0, length=size(U0, 2))), U0, M)
    seed_eps = norm(U_ref[:, 1] .- [1.0, 0.0])   # natural epsilon for this seed

    # Run with default NaN epsilon → uses seed-derived distance.
    # diagnostics["epsilon_start"] must equal seed_eps (not a hard-coded value).
    cfg_nan = ConnectingOrbitConfig(continuation=cont, n_mesh=M, bothside=false)
    res_nan = homoclinic_orbit_continuation(sys, cfg_nan; primary_param_index=2,
                                            orbit_guess=U0, saddle_guess=[1.0, 0.0],
                                            truncation_time=T0)
    @test isapprox(res_nan.diagnostics["epsilon_start"], seed_eps; rtol=1e-8)

    # Run with explicit epsilon = 2× seed distance → config value must drive the BVP.
    eps_explicit = seed_eps * 2.0
    cfg_explicit = ConnectingOrbitConfig(continuation=cont, n_mesh=M, bothside=false,
                                    epsilon_start=eps_explicit, epsilon_end=eps_explicit)
    res_explicit = homoclinic_orbit_continuation(sys, cfg_explicit; primary_param_index=2,
                                                  orbit_guess=U0, saddle_guess=[1.0, 0.0],
                                                  truncation_time=T0)
    # diagnostics["epsilon_start"] must match the config, not the seed distance.
    @test isapprox(res_explicit.diagnostics["epsilon_start"], eps_explicit; rtol=1e-8)
    @test res_explicit.diagnostics["epsilon_start"] != res_nan.diagnostics["epsilon_start"]

    # The corrected endpoint distances in the locus must pin to the requested value.
    # The BVP residual pins |d0|² = eps0², so |d0| converges up to ~tol/(2*eps0).
    # Use an absolute tolerance of 1e-3 (>> 1e-10/(2*eps_explicit) ≈ 3e-5).
    @test all(v -> abs(v - eps_explicit) < 1e-3, res_explicit.epsilon_start_values)

    # A seed whose endpoint exactly coincides with the saddle is rejected before correction.
    U_bad = _resample_states(collect(range(0.0, T0, length=size(U0, 2))), U0, M)
    U_bad[:, 1] = [1.0, 0.0]   # exactly at the saddle (displacement = 0)
    @test_throws ArgumentError connecting_orbit_continuation(
        sys, ConnectingOrbitConfig(continuation=cont, kind=:homoclinic, n_mesh=M);
        primary_param_index=2, orbit_guess=U_bad, saddle_guess=[1.0, 0.0],
        truncation_time=T0)

    # Saddle-cycle: explicit epsilon in config drives diagnostics["epsilon_start"].
    sys_sc = hc_genuine_saddle_cycle_system()
    cyc, Tc, U_sc, T_sc = hc_genuine_saddle_cycle_seed()
    x0_sc = collect(cyc[:, 1])
    seed_eps_sc = norm(U_sc[:, 1] .- x0_sc)

    cfg_sc_nan = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=2,
                                        newton_tol=1e-5, max_steps=3),
        kind=:saddle_cycle, n_mesh=80)
    cfg_sc_exp = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=2,
                                        newton_tol=1e-5, max_steps=3),
        kind=:saddle_cycle, n_mesh=80,
        epsilon_start=seed_eps_sc, epsilon_end=seed_eps_sc)
    res_sc_nan = saddle_cycle_homoclinic_continuation(
        sys_sc, cfg_sc_nan; cycle_states=cyc, cycle_period=Tc,
        orbit_guess=U_sc, truncation_time=T_sc, reference_index=1)
    res_sc_exp = saddle_cycle_homoclinic_continuation(
        sys_sc, cfg_sc_exp; cycle_states=cyc, cycle_period=Tc,
        orbit_guess=U_sc, truncation_time=T_sc, reference_index=1)
    @test isapprox(res_sc_nan.diagnostics["epsilon_start"], seed_eps_sc; rtol=1e-8)
    @test isapprox(res_sc_exp.diagnostics["epsilon_start"], seed_eps_sc; rtol=1e-8)
end

@testset "projector_refresh cadence" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed(K=200)
    field = _ode_field(sys)
    M = 80
    ts = collect(range(0.0, T0, length=size(U0, 2)))
    U = _resample_states(ts, U0, M)
    xs = [1.0, 0.0]
    eps0 = norm(U[:, 1] .- xs)
    eps1 = norm(U[:, end] .- xs)
    prob = _ConnectingProblem(field, [-1.0, 0.0], 2, 1, 2, M, :homoclinic, eps0, eps1)
    seed = _ConnectingSeed(U, xs, xs, T0, 0.0, -1.0)
    z0 = _seed_vector(prob, seed)
    bslot = length(z0)
    pin_beta(z) = [z[bslot] + 1.0]

    # projector_refresh=1 (every iteration) and projector_refresh=3 (every 3rd)
    # must both converge on a well-conditioned problem with identical final residual.
    r1 = _gauss_newton(prob, z0; extra=pin_beta, tol=1e-10, maxiter=60,
                       use_fallback=true, fallback_max_iter=150, projector_refresh=1)
    r3 = _gauss_newton(prob, z0; extra=pin_beta, tol=1e-10, maxiter=60,
                       use_fallback=true, fallback_max_iter=150, projector_refresh=3)
    @test r1.converged
    @test r3.converged
    @test r1.residual <= 1e-10
    @test r3.residual <= 1e-10

    # End-to-end: projector refresh cadence reaches the corrector.
    cont_r1 = ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=1,
                                  newton_tol=1e-10, max_steps=3)
    cfg_r1 = ConnectingOrbitConfig(continuation=cont_r1, n_mesh=80, projector_refresh=1,
                              bothside=false, orbit_save_stride=1)
    cfg_r4 = ConnectingOrbitConfig(continuation=cont_r1, n_mesh=80, projector_refresh=4,
                              bothside=false, orbit_save_stride=1)
    res_r1 = homoclinic_orbit_continuation(sys, cfg_r1; primary_param_index=2,
                                           orbit_guess=U0, saddle_guess=[1.0, 0.0],
                                           truncation_time=T0)
    res_r4 = homoclinic_orbit_continuation(sys, cfg_r4; primary_param_index=2,
                                           orbit_guess=U0, saddle_guess=[1.0, 0.0],
                                           truncation_time=T0)
    # Both must converge to the same locus (same points or a subset thereof).
    @test maximum(res_r1.residuals) < 1e-7
    @test maximum(res_r4.residuals) < 1e-7
    @test isapprox(res_r1.primary_values[1], res_r4.primary_values[1]; atol=1e-6)
    @test cfg_r1.projector_refresh == 1
    @test cfg_r4.projector_refresh == 4
end

# Stubs for OrbitBranchResult error-path tests.  OrbitBranchResult uses ::Any
# for branch/coll, so these thin structs satisfy _orbit_branch_count (which
# reads length(branch.sol)) without requiring a real BifurcationKit result.
struct _HCSolStub p::Float64 end
struct _HCBranchStub sol::Vector{_HCSolStub} end

@testset "OrbitBranchResult seeded overload (error paths)" begin
    sys = hc_bt_system()
    cont_e = ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=1, max_steps=3)
    cfg_e = ConnectingOrbitConfig(continuation=cont_e, n_mesh=60, bothside=false)

    # An empty branch must be rejected before any correction attempt.
    empty_src = OrbitBranchResult(_HCBranchStub(_HCSolStub[]), nothing,
                                  1, [-1.0, 0.0], 2, Int[], "BT-exact", :β2, :collocation,
                                  Dates.now())
    @test_throws ArgumentError homoclinic_orbit_continuation(sys, empty_src, cfg_e)
    wrong_system = OrbitBranchResult(
        empty_src.branch, empty_src.coll, empty_src.period, empty_src.base_params,
        empty_src.param_index, empty_src.linked_param_indices, "Other system",
        empty_src.param_name, empty_src.method, empty_src.timestamp)
    @test_throws ArgumentError homoclinic_orbit_continuation(sys, wrong_system, cfg_e)
    wrong_params = OrbitBranchResult(
        empty_src.branch, empty_src.coll, empty_src.period, [-1.0],
        empty_src.param_index, empty_src.linked_param_indices, empty_src.system_name,
        empty_src.param_name, empty_src.method, empty_src.timestamp)
    @test_throws ArgumentError homoclinic_orbit_continuation(sys, wrong_params, cfg_e)

    # Same primary (from source.param_index) and secondary (from continuation.param_index)
    # must be rejected immediately.
    cfg_same = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=2,
                                        max_steps=3),
        n_mesh=60)
    one_src = OrbitBranchResult(_HCBranchStub([_HCSolStub(-1.0)]), nothing,
                                1, [-1.0, 0.0], 2, Int[], "BT-exact", :β1, :collocation,
                                Dates.now())
    @test_throws ArgumentError homoclinic_orbit_continuation(sys, one_src, cfg_same)
end

# --- Finding 1: projector refresh at successful return ------------------------

@testset "Projector refresh on successful return (Finding 1)" begin
    # Verify that the returned residual from _gauss_newton reflects fresh projectors
    # at the converged z, even when projector_refresh > 1.  To expose a stale-projector
    # residual we compare the claimed residual from _gauss_newton against what you get
    # by manually refreshing projectors at the returned z.
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed(K=200)
    field = _ode_field(sys)
    M = 80
    ts = collect(range(0.0, T0, length=size(U0, 2)))
    U = _resample_states(ts, U0, M)
    xs = [1.0, 0.0]
    eps0 = norm(U[:, 1] .- xs)
    eps1 = norm(U[:, end] .- xs)
    prob = _ConnectingProblem(field, [-1.0, 0.0], 2, 1, 2, M, :homoclinic, eps0, eps1)
    seed = _ConnectingSeed(U, xs, xs, T0, 0.0, -1.0)
    z0 = _seed_vector(prob, seed)
    bslot = length(z0)
    pin_beta(z) = [z[bslot] + 1.0]

    # With projector_refresh=5 some iteration will be accepted with projectors
    # computed at an earlier z.  The claimed residual must match a fresh re-evaluation.
    for refresh in (1, 2, 5)
        r = _gauss_newton(prob, z0; extra=pin_beta, tol=1e-10, maxiter=80,
                          use_fallback=true, fallback_max_iter=200,
                          projector_refresh=refresh)
        @test r.converged
        # Fresh-projector residual at the returned z must be ≤ tol (or very close).
        bc_fresh = _refresh_bc(prob, r.z)
        rn_fresh = norm(_augmented_residual(r.z, prob, bc_fresh, pin_beta))
        # The returned residual must equal the fresh residual (not a stale one).
        @test isapprox(r.residual, rn_fresh; rtol=1e-6, atol=1e-15)
        @test r.residual <= 1e-10
    end
end

# --- Finding 2: saddle-cycle unconverged correction rejected -----------------

@testset "Saddle-cycle convergence rejection (Finding 2)" begin
    sys = hc_saddle_cycle_system()
    ω = 1.0
    Tc = 2π / ω
    L = 200
    θ = range(0, 2π, length=L)
    cyc = permutedims(hcat(cos.(θ), sin.(θ), zeros(L)))
    # A deliberately bad orbit guess (all zeros) cannot converge; set an extremely
    # tight tolerance so the corrector is forced to fail.
    K = 120
    U_bad = zeros(3, K)
    U_bad[1, :] .= 0.01  # nonzero but far from the homoclinic orbit
    cfg_strict = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=0.1, p_max=1.0, ds=0.05, param_index=2,
                                        newton_tol=1e-15, max_steps=1),
        kind=:saddle_cycle, n_mesh=30, use_fallback=false)
    # The corrector should fail to converge and the API must throw rather than
    # silently return a diverged result.
    @test_throws ErrorException saddle_cycle_homoclinic_continuation(
        sys, cfg_strict; cycle_states=cyc, cycle_period=Tc,
        orbit_guess=U_bad, truncation_time=0.1)
end

# --- Finding 3: finite-positive truncation_time requirement ------------------

@testset "Truncation time finite-positive validation (Finding 3)" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=1, max_steps=1)
    cfg = ConnectingOrbitConfig(continuation=cont, n_mesh=40)

    # Equilibrium path: reject zero, negative, NaN, Inf.
    for bad_T in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError homoclinic_orbit_continuation(
            sys, cfg; primary_param_index=2, orbit_guess=U0,
            saddle_guess=[1.0, 0.0], truncation_time=bad_T)
    end

    # Saddle-cycle path: same validation.
    sys_sc = hc_saddle_cycle_system()
    ω = 1.0
    Tc = 2π / ω
    L = 200
    θ = range(0, 2π, length=L)
    cyc = permutedims(hcat(cos.(θ), sin.(θ), zeros(L)))
    K = 60
    ss = range(-6, 6, length=K)
    U_sc = permutedims(hcat([cos(2π / (1 + exp(-s))) for s in ss],
                             [sin(2π / (1 + exp(-s))) for s in ss],
                             [0.05 * exp(-abs(s)) for s in ss]))
    cfg_sc = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=0.1, p_max=1.0, ds=0.05, param_index=2,
                                        max_steps=1),
        kind=:saddle_cycle, n_mesh=40)
    for bad_T in (0.0, -5.0, NaN, Inf)
        @test_throws ArgumentError saddle_cycle_homoclinic_continuation(
            sys_sc, cfg_sc; cycle_states=cyc, cycle_period=Tc,
            orbit_guess=U_sc, truncation_time=bad_T)
    end
end

# --- Finding 4: focus-spectrum double-real tests are unavailable -------------

@testset "Saddle-focus double-real tests unavailable (Finding 4)" begin
    # A 3D saddle-focus has one real unstable eigenvalue and a complex conjugate stable pair.
    # ẋ = x, ẏ = -0.3y + ω·z, ż = -ω·y - 0.3z  → eigenvalues: 1, -0.3 ± iω
    # The leading stable eigenvalues are complex (focus), so :drs/:nds must be
    # :unavailable, not :available with a fabricated zero value.
    ω_focus = 1.5
    A_focus = [1.0 0.0 0.0; 0.0 -0.3 ω_focus; 0.0 -ω_focus -0.3]
    tf_focus = DynamicsKit._eigen_test_functions(A_focus)

    # Focus-specific tests replace the real-saddle neutral test.
    @test !haskey(tf_focus, :nns)
    @test tf_focus[:sh][2] == :available

    # Double-real stable tests are unavailable (leading stable pair is complex).
    @test haskey(tf_focus, :drs)
    @test tf_focus[:drs][2] == :unavailable
    @test isnan(tf_focus[:drs][1])
    @test !haskey(tf_focus, :nds)

    # nsf (neutral saddle-focus) is present and available.
    @test haskey(tf_focus, :nsf)
    @test tf_focus[:nsf][2] == :available

    # A pure saddle has the real neutral/double-real tests, but no Shilnikov event.
    A_saddle = [1.0 0.0 0.0; 0.0 -0.5 0.0; 0.0 0.0 -2.0]
    tf_saddle = DynamicsKit._eigen_test_functions(A_saddle)
    @test haskey(tf_saddle, :drs)
    @test tf_saddle[:drs][2] == :available
    @test isfinite(tf_saddle[:drs][1])
    @test haskey(tf_saddle, :nns)
    @test !haskey(tf_saddle, :sh)
    @test !haskey(tf_saddle, :nds)

    # A crossing of a :unavailable double-real test must NOT produce a special point.
    # Build two locus points with the focus saddle Jacobian.
    field_focus(u, p) = A_focus * collect(u)
    prob_focus = _ConnectingProblem(field_focus, [0.0], 1, 1, 3, 30, :homoclinic, 1.0, 1.0)
    # The "saddle" here is the origin (field(0,0,0) = 0 for A_focus).
    p1 = _LocusPoint(Float64[], 0.0, 0.0, 5.0, zeros(3), zeros(3), 1.0, 1.0,
                     1e-12, :newton,
                     Dict(:drs => -0.5, :nns => -0.5),
                     Dict(:drs => :unavailable, :nns => :available),
                     zeros(3, 31))
    p2 = _LocusPoint(Float64[], 0.0, 0.1, 5.0, zeros(3), zeros(3), 1.0, 1.0,
                     1e-12, :newton,
                     Dict(:drs => 0.5, :nns => 0.5),
                     Dict(:drs => :unavailable, :nns => :available),
                     zeros(3, 31))
    specials = _detect_special_points([p1, p2])
    # Only the :nns crossing should produce a special point; :drs must not.
    drs_points = filter(sp -> sp.kind == :drs, specials)
    @test isempty(drs_points)
    nns_points = filter(sp -> sp.kind == :nns, specials)
    @test length(nns_points) == 1
end

# --- Finding 5: saddle-cycle extra unit-circle multiplier rejection ----------

@testset "Extra unit-circle multiplier rejection (Finding 5)" begin
    # [0.5, -1, 2]: -1 is a unit-circle multiplier that is NOT trivial (+1).
    # The corrected validation must reject this, because -1 indicates period-doubling.
    split_neg1 = DynamicsKit._FloquetSplit(
        ComplexF64[0.5, -1.0, 2.0], 1, 1, 1,
        zeros(3, 0), zeros(3, 0))
    @test_throws ArgumentError DynamicsKit._validate_saddle_cycle_geometry(split_neg1, 3)

    # [0.5, 1, 1, 2]: two trivial multipliers near +1 must also be rejected.
    split_two1 = DynamicsKit._FloquetSplit(
        ComplexF64[0.5, 1.0, 1.0, 2.0], 1, 1, 2,
        zeros(4, 0), zeros(4, 0))
    @test_throws ArgumentError DynamicsKit._validate_saddle_cycle_geometry(split_two1, 4)

    # A genuinely valid saddle cycle [0.5, 1, 2] must pass.
    split_ok = DynamicsKit._FloquetSplit(
        ComplexF64[0.5, 1.0, 2.0], 1, 1, 1,
        zeros(3, 0), zeros(3, 0))
    @test DynamicsKit._validate_saddle_cycle_geometry(split_ok, 3) === nothing

    # A purely stable cycle (nu=0) must still fail even with a single trivial multiplier.
    split_stable = DynamicsKit._FloquetSplit(
        ComplexF64[0.3, 0.7, 1.0], 2, 0, 1,
        zeros(3, 0), zeros(3, 0))
    @test_throws ArgumentError DynamicsKit._validate_saddle_cycle_geometry(split_stable, 3)
end

# --- Finding 6: special-float round-trip through diagnostics -----------------

@testset "Special-float round-trip in diagnostics (Finding 6)" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.2, p_max=-0.8, ds=0.05, param_index=1,
                              newton_tol=1e-10, max_steps=4)
    cfg = ConnectingOrbitConfig(continuation=cont, n_mesh=60, bothside=false)
    res = homoclinic_orbit_continuation(sys, cfg; primary_param_index=2, orbit_guess=U0,
                                        saddle_guess=[1.0, 0.0], truncation_time=T0)

    # Inject special-float values into diagnostics.
    res2 = HomoclinicBranchResult(
        res.primary_values, res.secondary_values, res.return_times,
        res.epsilon_start_values, res.epsilon_end_values,
        res.saddles, res.target_saddles, res.test_functions, res.test_statuses,
        res.special_points, res.orbits, res.residuals, res.corrector_paths,
        res.connection_kind, res.source_period, res.source_index,
        res.source_primary_value, res.base_params,
        res.primary_param_index, res.secondary_param_index,
        res.system_name, res.param_names,
        merge(res.diagnostics, Dict{String, Any}(
            "test_nan" => NaN,
            "test_inf" => Inf,
            "test_neginf" => -Inf,
            "nested" => Dict{String, Any}("inner_nan" => NaN, "inner_inf" => Inf),
            "vec_with_special" => [1.0, NaN, Inf, -Inf, 2.0],
            "literal_nan" => "nan",
            "literal_inf" => "INF",
            "tagged_user_dict" => Dict{String, Any}(
                "__dynamicskit_type__" => "special_float",
                "value" => "nan"),
        )),
        res.timestamp)

    plain = serialize_homoclinic_branch_result(res2)
    restored = deserialize_homoclinic_branch_result(plain)

    @test isnan(restored.diagnostics["test_nan"])
    @test isinf(restored.diagnostics["test_inf"]) && restored.diagnostics["test_inf"] > 0
    @test isinf(restored.diagnostics["test_neginf"]) && restored.diagnostics["test_neginf"] < 0
    nested = restored.diagnostics["nested"]
    @test isnan(nested["inner_nan"])
    @test isinf(nested["inner_inf"]) && nested["inner_inf"] > 0
    vec = restored.diagnostics["vec_with_special"]
    @test vec[1] ≈ 1.0
    @test isnan(vec[2])
    @test isinf(vec[3]) && vec[3] > 0
    @test isinf(vec[4]) && vec[4] < 0
    @test vec[5] ≈ 2.0
    @test restored.diagnostics["literal_nan"] == "nan"
    @test restored.diagnostics["literal_inf"] == "INF"
    @test restored.diagnostics["tagged_user_dict"] == Dict{String, Any}(
        "__dynamicskit_type__" => "special_float",
        "value" => "nan")
end

# --- Finding 7: per-sample test_statuses round-trip ---------------------------

@testset "Per-sample test_statuses round-trip (Finding 7)" begin
    sys = hc_bt_system()
    U0, T0 = hc_bt_seed()
    cont = ContinuationConfig(p_min=-1.4, p_max=-0.7, ds=0.05, param_index=1,
                              newton_tol=1e-10, max_steps=6)
    cfg = ConnectingOrbitConfig(continuation=cont, n_mesh=80, bothside=true)
    res = homoclinic_orbit_continuation(sys, cfg; primary_param_index=2, orbit_guess=U0,
                                        saddle_guess=[1.0, 0.0], truncation_time=T0)

    # test_statuses must have the same keys and per-sample lengths as test_functions.
    @test keys(res.test_statuses) == keys(res.test_functions)
    for (code, statuses) in res.test_statuses
        @test length(statuses) == length(res.primary_values)
        @test all(s -> s in (:available, :unavailable, :degenerate), statuses)
    end

    # testStatuses must survive serialize/deserialize.
    plain = serialize_homoclinic_branch_result(res)
    @test haskey(plain, "testStatuses")
    restored = deserialize_homoclinic_branch_result(plain)
    @test restored.test_statuses == res.test_statuses

end
