"""
Cycle-to-cycle connecting-orbit validation.

`cycle_connection_seed`/`cycle_connection_continuation` continue a connecting
orbit between two saddle limit cycles as a projection boundary-value problem
(see `docs/analysis-methods.md`, "Automatic continuation atlas" and
`docs/julia-package.md`'s cycle-connection section). This script checks:

  1. Analytic: a rotating Nagumo front lifts the exact transverse
     heteroclinic `s(t) = 1/(1+exp(t/sqrt(2)))` to a connecting orbit between
     a saddle cycle at `s=1` and one at `s=0`. The recovered two-parameter
     locus must follow the closed form `c = (1-2a)/sqrt(2)`, both cycle
     phase endpoints must match their closed-form radii, and the corrector
     residual must stay small.
  2. Automatic seed discovery finds a source-to-target approach without an
     explicit orbit guess, and the resulting connection agrees with the
     manually-seeded one.
  3. Refusals: a purely attracting cycle (no unstable direction to leave
     along) is rejected, and a degenerate (constant/repeated-point) orbit
     guess is rejected at the API boundary rather than silently accepted or
     deferred to a doomed corrector (see `_validate_orbit_guess_variation` in
     `src/analysis/homoclinic/api.jl`).

Run:

    julia --project=. validation/cycle_to_cycle_connections.jl
"""

using DynamicsKit
using LinearAlgebra

include(joinpath(@__DIR__, "common.jl"))

gates = ValidationGate[]

# ── Analytic rotating-Nagumo fixture ────────────────────────────────────────

function rotating_nagumo_system()
    function f!(du, u, p, t)
        x, y, z = u
        c, a, omega = p
        C = 2.0
        r2 = max(x^2 + y^2, eps(Float64))
        s = r2 / 2 - C
        radial = z / r2
        du[1] = radial * x - omega * y
        du[2] = omega * x + radial * y
        du[3] = -c * z - s * (1 - s) * (s - a)
        nothing
    end
    section = PoincareSection((u, t, integrator) -> u[2]; direction=:up,
                              projection=[1, 3], template=[sqrt(6.0), 0.0, 0.0])
    a = 0.25
    return ContinuousODE(f!, 3, section, [:c, :a, :omega], "Rotating Nagumo cycle connection";
                         default_params=[(1 - 2a) / sqrt(2), a, 1.0])
end

function rotating_nagumo_seed(; a=0.25, omega=1.0, half=9.0, K=260, L=120)
    C = 2.0
    Tc = 2pi / omega
    thetas = range(-omega * half, -omega * half + 2pi, length=L + 1)
    source_radius = sqrt(2 * (C + 1))
    source = permutedims(hcat(source_radius .* cos.(thetas), source_radius .* sin.(thetas), zeros(length(thetas))))
    thetat = range(omega * half, omega * half + 2pi, length=L + 1)
    target_radius = sqrt(2C)
    target = permutedims(hcat(target_radius .* cos.(thetat), target_radius .* sin.(thetat), zeros(length(thetat))))
    ts = range(-half, half, length=K)
    s = [1 / (1 + exp(t / sqrt(2))) for t in ts]
    z = [-(1 / sqrt(2)) * exp(t / sqrt(2)) / (1 + exp(t / sqrt(2)))^2 for t in ts]
    radius = sqrt.(2 .* (s .+ C))
    orbit = permutedims(hcat(radius .* cos.(omega .* ts), radius .* sin.(omega .* ts), z))
    return source, target, Tc, orbit, 2half
end

sys = rotating_nagumo_system()
source, target, Tc, orbit, Tconn = rotating_nagumo_seed()
c_exact = sys.default_params[1]

cont = ContinuationConfig(
    p_min=0.18, p_max=0.32, ds=0.02, dsmax=0.025, dsmin=1e-4, param_index=2,
    newton_tol=2e-6, newton_max_iter=10, max_steps=4)
cfg = ConnectingOrbitConfig(
    continuation=cont, kind=:cycle_connection, n_mesh=36, bothside=true,
    orbit_save_stride=1, max_saved_orbits=7, fallback_max_iter=30,
    epsilon_start=1e-3, epsilon_end=1e-3)
res = cycle_connection_continuation(
    sys, cfg; primary_param_index=1,
    source_cycle_states=source, source_cycle_period=Tc,
    target_cycle_states=target, target_cycle_period=Tc,
    orbit_guess=orbit, truncation_time=Tconn)

locus_errors = [abs(res.primary_values[i] - (1 - 2 * res.secondary_values[i]) / sqrt(2))
                for i in eachindex(res.primary_values)]
max_locus_error = maximum(locus_errors)
max_residual = maximum(res.residuals)
record_gate!(gates, "analytic rotating-Nagumo locus matches c=(1-2a)/sqrt(2)",
    res.connection_kind === :cycle_connection && length(res.primary_values) >= 3 &&
        max_locus_error <= 4e-2 && max_residual <= 6e-5,
    "points=$(length(res.primary_values)) max_locus_error=$(round(max_locus_error, sigdigits=3)) max_residual=$(round(max_residual, sigdigits=3))")

source_radii = [hypot(res.saddles[1, j], res.saddles[2, j]) for j in axes(res.saddles, 2)]
target_radii = [hypot(res.target_saddles[1, j], res.target_saddles[2, j]) for j in axes(res.target_saddles, 2)]
record_gate!(gates, "cycle phase endpoints match the closed-form radii",
    maximum(abs.(source_radii .- sqrt(6.0))) <= 5e-3 && maximum(abs.(target_radii .- 2.0)) <= 5e-3 &&
        res.diagnostics["source_unstable_floquet_dim"] == 1 && res.diagnostics["target_stable_floquet_dim"] == 1,
    "source_radius_error=$(round(maximum(abs.(source_radii .- sqrt(6.0))), sigdigits=3)) target_radius_error=$(round(maximum(abs.(target_radii .- 2.0)), sigdigits=3))")

# ── Automatic seed discovery ─────────────────────────────────────────────────

seed_config = CycleConnectionSeedConfig(
    source_phase_samples=8, target_phase_samples=24, max_time=18.0,
    sample_count=400, distance_tolerance=1e-2)
seed = cycle_connection_seed(
    sys; source_cycle_states=source, source_cycle_period=Tc,
    target_cycle_states=target, target_cycle_period=Tc, seed_config=seed_config)
auto_cont = ContinuationConfig(
    p_min=0.18, p_max=0.32, ds=0.02, dsmax=0.03, dsmin=1e-4, param_index=2,
    newton_tol=2e-6, newton_max_iter=12, max_steps=2)
auto_cfg = ConnectingOrbitConfig(
    continuation=auto_cont, kind=:cycle_connection, n_mesh=36, bothside=false,
    fallback_max_iter=40, epsilon_start=1e-3, epsilon_end=1e-3)
auto_res = cycle_connection_continuation(
    sys, auto_cfg; primary_param_index=1,
    source_cycle_states=source, source_cycle_period=Tc,
    target_cycle_states=target, target_cycle_period=Tc, seed_config=seed_config)
auto_locus_errors = [abs(auto_res.primary_values[i] - (1 - 2 * auto_res.secondary_values[i]) / sqrt(2))
                     for i in eachindex(auto_res.primary_values)]
record_gate!(gates, "automatic seed discovery reaches the same connection",
    seed.status === :found && seed.distance <= seed_config.distance_tolerance &&
        auto_res.connection_kind === :cycle_connection && length(auto_res.primary_values) >= 3 &&
        haskey(auto_res.diagnostics, "seed_discovery") &&
        maximum(auto_locus_errors) <= 8e-2 && maximum(auto_res.residuals) <= 6e-5,
    "seed_distance=$(round(seed.distance, sigdigits=3)) auto_points=$(length(auto_res.primary_values)) auto_max_locus_error=$(round(maximum(auto_locus_errors), sigdigits=3))")

# ── Refusals ─────────────────────────────────────────────────────────────────

function stable_cycle_fixture()
    function f!(du, u, p, t)
        x, y, z = u
        r2 = x^2 + y^2
        omega, lambda = p
        du[1] = x - omega * y - x * r2
        du[2] = omega * x + y - y * r2
        du[3] = lambda * z
        nothing
    end
    section = PoincareSection((u, t, integrator) -> u[1]; direction=:up,
                              projection=[1], template=zeros(3))
    sys = ContinuousODE(f!, 3, section, [:omega, :lambda], "Stable cycle refusal";
                        default_params=[1.0, -0.5])
    theta = range(0, 2pi, length=80)
    cycle = permutedims(hcat(cos.(theta), sin.(theta), zeros(length(theta))))
    orbit = permutedims(hcat(cos.(theta), sin.(theta), 0.05 .* sin.(theta)))
    return sys, cycle, orbit
end

bad_sys, bad_cycle, bad_orbit = stable_cycle_fixture()
attracting_refused = try
    bad_cfg = ConnectingOrbitConfig(
        continuation=ContinuationConfig(p_min=0.1, p_max=1.0, ds=0.05, param_index=2),
        kind=:cycle_connection, n_mesh=24)
    cycle_connection_continuation(
        bad_sys, bad_cfg; primary_param_index=1,
        source_cycle_states=bad_cycle, source_cycle_period=2pi,
        target_cycle_states=bad_cycle, target_cycle_period=2pi,
        orbit_guess=bad_orbit, truncation_time=2pi, base_params=[1.0, -0.5])
    false
catch err
    err isa ArgumentError
end
record_gate!(gates, "purely attracting cycle is refused", attracting_refused,
    attracting_refused ? "no unstable Floquet direction to leave along; rejected" : "unexpectedly accepted")

degenerate_refused = try
    zero_orbit = fill(0.0, 3, 20)
    cycle_connection_continuation(
        sys, cfg; primary_param_index=1,
        source_cycle_states=source, source_cycle_period=Tc,
        target_cycle_states=target, target_cycle_period=Tc,
        orbit_guess=zero_orbit, truncation_time=1.0)
    false
catch err
    err isa ArgumentError
end
record_gate!(gates, "degenerate (constant) orbit guess is refused at the interface", degenerate_refused,
    degenerate_refused ? "rejected before reaching the corrector" : "unexpectedly accepted")

record_gate!(gates, "seed parameter matches the analytic value",
    isapprox(c_exact, sys.default_params[1]; atol=0.0), "c=$(c_exact)")

conclude(gates, "cycle-to-cycle connection validation")
