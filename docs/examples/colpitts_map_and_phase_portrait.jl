#!/usr/bin/env julia

"""
Colpitts (exponential model): a 2D bifurcation map over two circuit
parameters (`C1` vs `C2`, both swept, `C1` linked to `C2`) and a phase
portrait at a representative operating point. Demonstrates
`bifurcation_map`/`plot_bifurcation_map` and
`phase_portrait`/`plot_phase_portrait`, which the other Colpitts examples
(`colpitts_simple_continuation.jl`) do not
cover.

Environment variables:

    COLPITTS_MAP_STEPS        grid steps per axis (default 30; the default
                               runs in well under a minute, raise for a finer
                               map at the cost of runtime)
    COLPITTS_MAP_ITERATIONS   map iterations per cell (default 150)
    COLPITTS_MAP_PERIOD       maximum period searched for (default 8)

Run:

    julia --project=. examples/colpitts_map_and_phase_portrait.jl
"""

using DynamicsKit
using DifferentialEquations
using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "var", "output", "colpitts_map_and_phase_portrait")
mkpath(OUTPUT_DIR)

const MAP_STEPS = parse(Int, get(ENV, "COLPITTS_MAP_STEPS", "30"))
const MAP_ITERATIONS = parse(Int, get(ENV, "COLPITTS_MAP_ITERATIONS", "150"))
const MAP_PERIOD = parse(Int, get(ENV, "COLPITTS_MAP_PERIOD", "8"))

println("═══ Colpitts (exponential) — 2D map and phase portrait ═══\n")

sys = colpitts_exponential_oscillator()
base_params = [40e-9, 40e-9, 265.0, 1.95, 1.95]   # [C1, C2, beta, V1, V2]

println("1. Computing the C1 vs C2 bifurcation map...")
map_cfg = BifurcationMapConfig(
    a_min=30e-9, a_max=60e-9, a_steps=MAP_STEPS,
    b_min=30e-9, b_max=60e-9, b_steps=MAP_STEPS,
    a_index=1, b_index=2,
    max_period=MAP_PERIOD,
    precision=1e-3,
    iterations=MAP_ITERATIONS,
    base_params=copy(base_params),
)
map_result = bifurcation_map(
    sys, map_cfg;
    initial_point=copy(sys.default_initial_state),
    solver=Tsit5(), reltol=1e-7, abstol=1e-7,
)
println("   Computed a $(MAP_STEPS+1) x $(MAP_STEPS+1) grid.")

map_plot = plot_bifurcation_map(
    map_result;
    xscale=1e9, yscale=1e9,
    xlabel="C1 (nF)", ylabel="C2 (nF)",
    title="Colpitts (exponential) C1 vs C2 map",
)
map_path = joinpath(OUTPUT_DIR, "colpitts_exponential_c1_c2_map.png")
savefig(map_plot, map_path)
println("   → $(map_path)")

println("2. Computing a phase portrait at the default operating point...")
phase_result = phase_portrait(
    sys,
    PhasePortraitConfig(time_stop=0.02, tail_fraction=0.6, poincare_crossings=80, min_crossing_time=1e-6);
    params=copy(base_params),
    initial_point=copy(sys.default_initial_state),
    solver=Tsit5(), reltol=1e-7, abstol=1e-7,
    state_names=[:V_C1, :V_C2, :I_L],
)
phase_plot = plot_phase_portrait(
    phase_result;
    x_index=1, y_index=2, xlabel="V_C1", ylabel="V_C2",
    title="Colpitts (exponential) phase portrait",
)
phase_path = joinpath(OUTPUT_DIR, "colpitts_exponential_phase_portrait.png")
savefig(phase_plot, phase_path)
println("   → $(phase_path)")

println("\nDone.")
