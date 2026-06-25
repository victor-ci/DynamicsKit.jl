# brute_force.jl — decomposed into focused files (behavior-preserving: the original top-to-bottom
# definition order is preserved by the include sequence below). See
# docs/internal/repository-split-plan-v2.md.
include("brute_force/sweep_poincare.jl")   # brute_force_diagram + Poincaré section / orbit sampling
include("brute_force/basins.jl")           # basins_of_attraction (discrete + continuous)
include("brute_force/map_diagnostics.jl")  # status codes, seed/tile helpers, Lyapunov, recorders, diagnostics
include("brute_force/map_compute.jl")      # tile processing, period detection, adaptive refinement, bifurcation_map
