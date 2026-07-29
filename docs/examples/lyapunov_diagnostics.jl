#!/usr/bin/env julia

"""
Lyapunov diagnostics: a 1D Lyapunov diagram sweeping the Hénon map's `a`
parameter (largest exponent vs. parameter, classifying each point as
periodic/neutral/chaotic), and the full Lyapunov spectrum of the Rössler
oscillator at a chaotic operating point (recovering the classic `(+, 0, -)`
signature of a chaotic flow).

Run:

    julia --project=. examples/lyapunov_diagnostics.jl
"""

using DynamicsKit
using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "var", "output", "lyapunov_diagnostics")
mkpath(OUTPUT_DIR)

println("═══ Lyapunov diagnostics ═══\n")

# ── 1. Discrete map: 1D Lyapunov diagram (Hénon) ────────────────────────────
println("1. Computing a Lyapunov diagram for the Hénon map (a ∈ [0.0, 1.4])...")
sys_henon = henon_map()
diagram = lyapunov_diagram(
    sys_henon,
    LyapunovConfig(
        param_min=0.0, param_max=1.4, param_steps=300,
        fixed_params=[1.4, 0.3], transient=200, iterations=400,
    );
    initial_point=[0.1, 0.1],
)
n_chaotic = count(==(:chaotic_candidate), diagram.classifications)
n_other = length(diagram.classifications) - n_chaotic
println("   $(n_chaotic) chaotic-candidate points, $(n_other) periodic/neutral/unresolved points out of $(length(diagram.exponents)).")

diagram_plot = plot_lyapunov_diagram(diagram)
diagram_path = joinpath(OUTPUT_DIR, "henon_lyapunov_diagram.png")
savefig(diagram_plot, diagram_path)
println("   → $(diagram_path)")

# ── 2. Continuous flow: full Lyapunov spectrum (Rössler) ────────────────────
println("\n2. Computing the full Lyapunov spectrum for the Rössler oscillator (chaotic regime)...")
sys_rossler = rossler_oscillator()
spectrum = lyapunov_spectrum(
    sys_rossler,
    LyapunovSpectrumConfig(transient=200, steps=2000, renorm_dt=0.5);
    params=[0.2, 0.2, 5.7],
    initial_point=[1.0, 1.0, 1.0],
)
println("   Spectrum: $(round.(spectrum.exponents; digits=4)) (expected sign pattern (+, 0, -) for chaos)")

spectrum_plot = plot_lyapunov_spectrum(spectrum)
spectrum_path = joinpath(OUTPUT_DIR, "rossler_lyapunov_spectrum.png")
savefig(spectrum_plot, spectrum_path)
println("   → $(spectrum_path)")

println("\nDone.")
