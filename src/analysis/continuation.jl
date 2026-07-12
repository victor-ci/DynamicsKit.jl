# Include order matters: later files use definitions from earlier ones.
include("continuation/branch.jl")              # core types, termination, continuation_branch(es) public API
include("continuation/seeding.jl")             # distinct-period branches, candidates, seed reuse, skeleton-param search
include("continuation/poincare_stability.jl")  # Poincaré projection/return/Jacobian/variational, multipliers/stability
include("continuation/diagnostics_reseed.jl")  # branch diagnostics, signatures, run-direction, reseed, complete branch
include("continuation/refine.jl")              # branch-point helpers, period trim, refinement intervals, refine/auto_refine
