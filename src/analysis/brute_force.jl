# Include order matters: later files use definitions from earlier ones.
include("brute_force/sweep_poincare.jl")   # brute_force_diagram + Poincaré section / orbit sampling
include("brute_force/basins.jl")           # basins_of_attraction (discrete + continuous)
include("brute_force/map_diagnostics.jl")  # status codes, seed/tile helpers, Lyapunov, recorders, diagnostics
include("brute_force/map_compute.jl")      # tile processing, period detection, adaptive refinement, bifurcation_map
include("brute_force/gpu_kernels.jl")      # optional GPU acceleration for the fixed-seed map/basins/Lyapunov-field sweeps
include("brute_force/gpu_continuous.jl")   # optional GPU acceleration for continuous-time Poincaré-map sweeps (EnsembleGPUKernel)
