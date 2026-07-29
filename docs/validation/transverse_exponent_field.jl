"""
Transverse-exponent field validation.

Continuous `lyapunov_field` defaults to a flow-time-normalized variational
transverse estimator (tangent-space/QR), replacing the two-trajectory
estimator as the field of record (see `docs/analysis-methods.md`, "Lyapunov
diagram", and `docs/julia-package.md`). This script checks two claims:

  1. Analytic: `r' = alpha*(1-r^2)*r, theta' = 1` has the unit limit cycle
     with known closed-form transverse exponent `-2*alpha`. The variational
     field must recover it to tight tolerance.
  2. Vilnius oscillator (Ipatovs et al. 2023): the two-trajectory estimator
     is known to report spurious positive exponents on some periodic cells
     (the orbit has not fully entered its asymptotic period, so a per-return
     finite-difference estimate can read positive before it settles). The
     variational field, being a direct tangent-space computation, must have
     an equal-or-lower spurious-positive rate than the two-trajectory field
     on the same periodic cells, while the two fields must still agree on
     most chaotic cells.

This uses a small, self-computed operating-map grid (no precomputed/external
artifacts) so it runs in a few minutes; it demonstrates the same qualitative
behavior as the full high-resolution operating-map campaign described in
`docs/validation.md`, at a scale suitable for routine, citable, in-repo
reproduction rather than a multi-hour run.

Run:

    julia --project=. validation/transverse_exponent_field.jl
"""

using DynamicsKit
using Accessors

include(joinpath(@__DIR__, "common.jl"))

gates = ValidationGate[]

# ── Section 1: analytic radial-phase oscillator ─────────────────────────────

function radial_phase_oscillator(; alpha=0.25)
    f! = function (du, u, p, t)
        a = p[1]
        radial = a * (1 - u[1]^2 - u[2]^2)
        du[1] = radial * u[1] - u[2]
        du[2] = u[1] + radial * u[2]
        nothing
    end
    section = PoincareSection(
        (u, t, integ) -> u[2], direction=:up, projection=[1], template=[1.0, 0.0];
        constant_normal=[0.0, 1.0])
    return ContinuousODE(f!, 2, section, [:alpha, :unused], "Radial phase oscillator";
                        default_initial_state=[1.0, 0.0], default_params=[alpha, 0.0], tspan_hint=2pi)
end

alpha = 0.25
circ = radial_phase_oscillator(; alpha=alpha)
analytic_cfg = BifurcationMapConfig(
    a_min=alpha, a_max=alpha, a_steps=0, b_min=0.0, b_max=0.0, b_steps=0,
    a_index=1, b_index=2, base_params=[alpha, 0.0],
    lyapunov_iterations=12, lyapunov_transient=3)
analytic_field = lyapunov_field(circ, analytic_cfg)
lambda_hat = analytic_field.exponents[1, 1]
record_gate!(gates, "analytic radial-phase oscillator transverse exponent = -2*alpha",
    isapprox(lambda_hat, -2alpha; atol=2e-6),
    "estimated=$(lambda_hat) expected=$(-2alpha)")

# ── Section 2: Vilnius oscillator spurious-positive-rate comparison ────────

sys = vilnius_oscillator()
grid_cfg = BifurcationMapConfig(
    a_min=0.05, a_max=0.6, a_steps=16,
    b_min=0.05, b_max=0.3, b_steps=10,
    a_index=1, b_index=3,
    base_params=[0.2, 30.0, 0.2],
    max_period=4, precision=1e-4, iterations=1500,
    divergence_cutoff=1e8,
)
periodicity_map = bifurcation_map(sys, grid_cfg)

variational_cfg = Accessors.@set grid_cfg.lyapunov_method = :variational
variational_cfg = Accessors.@set variational_cfg.lyapunov_iterations = 60
variational_cfg = Accessors.@set variational_cfg.lyapunov_transient = 60
variational_field = lyapunov_field(sys, variational_cfg)

two_trajectory_cfg = Accessors.@set grid_cfg.lyapunov_method = :two_trajectory
two_trajectory_cfg = Accessors.@set two_trajectory_cfg.lyapunov_iterations = 60
two_trajectory_cfg = Accessors.@set two_trajectory_cfg.lyapunov_transient = 60
two_trajectory_field = lyapunov_field(sys, two_trajectory_cfg)

na, nb = size(periodicity_map.periodicity)
neutral_tol = 1e-3
interior = [(i, j) for i in 2:(na - 1), j in 2:(nb - 1)]
periodic_cells = [(i, j) for (i, j) in interior if periodicity_map.periodicity[i, j] > 0]
nonperiodic_cells = [(i, j) for (i, j) in interior if periodicity_map.periodicity[i, j] == 0]

record_gate!(gates, "grid has enough interior samples of both kinds",
    length(periodic_cells) >= 15 && length(nonperiodic_cells) >= 10,
    "interior_periodic=$(length(periodic_cells)) interior_nonperiodic=$(length(nonperiodic_cells))")

variational_spurious = count(c -> variational_field.exponents[c...] > neutral_tol, periodic_cells)
two_trajectory_spurious = count(c -> two_trajectory_field.exponents[c...] > neutral_tol, periodic_cells)
variational_rate = length(periodic_cells) == 0 ? 0.0 : variational_spurious / length(periodic_cells)
two_trajectory_rate = length(periodic_cells) == 0 ? 0.0 : two_trajectory_spurious / length(periodic_cells)
record_gate!(gates, "variational field's spurious-positive rate is no worse than two-trajectory's",
    variational_rate <= two_trajectory_rate + 1e-9,
    "variational=$(round(100variational_rate, digits=1))% two_trajectory=$(round(100two_trajectory_rate, digits=1))% ($(length(periodic_cells)) interior periodic cells)")
record_gate!(gates, "variational field's spurious-positive rate on periodic cells is low in absolute terms",
    variational_rate <= 0.1,
    "variational=$(round(100variational_rate, digits=1))% (threshold 10%)")

variational_chaotic = count(c -> variational_field.exponents[c...] > neutral_tol, nonperiodic_cells)
two_trajectory_chaotic = count(c -> two_trajectory_field.exponents[c...] > neutral_tol, nonperiodic_cells)
variational_chaotic_rate = length(nonperiodic_cells) == 0 ? 1.0 : variational_chaotic / length(nonperiodic_cells)
two_trajectory_chaotic_rate = length(nonperiodic_cells) == 0 ? 1.0 : two_trajectory_chaotic / length(nonperiodic_cells)
record_gate!(gates, "both fields agree on most chaotic (non-periodic) cells",
    variational_chaotic_rate >= 0.7 && two_trajectory_chaotic_rate >= 0.7,
    "variational=$(round(100variational_chaotic_rate, digits=1))% two_trajectory=$(round(100two_trajectory_chaotic_rate, digits=1))% ($(length(nonperiodic_cells)) interior non-periodic cells)")

record_gate!(gates, "method/normalization provenance recorded",
    variational_field.lyapunov_method == :variational && variational_field.normalization == :flow_time &&
        two_trajectory_field.lyapunov_method == :two_trajectory,
    "variational: method=$(variational_field.lyapunov_method) normalization=$(variational_field.normalization); two_trajectory: method=$(two_trajectory_field.lyapunov_method)";
    required=false)

conclude(gates, "transverse-exponent field validation")
