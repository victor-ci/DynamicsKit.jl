#!/usr/bin/env julia

"""
Automatic continuation atlas: `continuation_atlas` combines a reconnaissance
brute-force sweep with automatic seed discovery and continuation to recover
every periodic branch across a parameter window without hand-picking seeds
for each one — demonstrated here on the peak-current-mode boost converter's
current-mode subharmonic cascade (see `docs/systems-catalog.md`).

Run:

    julia --project=. examples/atlas_continuation.jl
"""

using DynamicsKit
using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "var", "output", "atlas_continuation")
mkpath(OUTPUT_DIR)

const IREF_MIN = 1.0
const IREF_MAX = 3.0
const BASE_PARAMS = [1.5, 10.0, 20.0, 0.0]   # [Iref, E, R, Sc]

println("═══ Automatic continuation atlas — boost converter ═══\n")
println("Sweeping Iref ∈ [$IREF_MIN, $IREF_MAX] A (E=10 V, R=20 Ω)")

sys = boost_converter()

atlas_cfg = AtlasConfig(
    brute_force=BruteForceConfig(
        param_min=IREF_MIN, param_max=IREF_MAX, param_index=1,
        fixed_params=copy(BASE_PARAMS),
        param_steps=60, iterations=200, transient=100,
    ),
    continuation=ContinuationConfig(
        p_min=IREF_MIN, p_max=IREF_MAX, param_index=1,
        ds=0.02, dsmax=0.05, max_steps=200,
        newton_tol=1e-8, newton_max_iter=30,
    ),
    periods=[1, 2, 3, 4],
    max_period=4,
    recon_steps=40,
)

println("Running the atlas (reconnaissance sweep + automatic branch discovery)...")
@time atlas = continuation_atlas(sys, atlas_cfg; params=copy(BASE_PARAMS))

println("\nDiscovered $(length(atlas.branch_records)) branch(es):")
for record in atlas.branch_records
    branch = record.branch
    println("  period $(branch.period): $(length(branch.branch)) points, ",
            "seeded near Iref=$(round(record.seed_param, digits=4)) A (window $(record.window_id))")
end

println("\nSaving an overlay of the atlas branches on the reconnaissance sweep...")
recon_bf = brute_force_diagram(
    sys,
    atlas_cfg.brute_force;
    initial_point=[10.0, 1.0],
)
overlay_plot = plot_overlay(recon_bf, atlas_branches(atlas); system=sys, params=copy(BASE_PARAMS))
overlay_path = joinpath(OUTPUT_DIR, "boost_atlas_overlay.png")
savefig(overlay_plot, overlay_path)
println("   → $(overlay_path)")

println("\nDone.")
