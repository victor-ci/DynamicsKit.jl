# Include order matters: later files use definitions from earlier ones.
include("atlas/core.jl")        # result/window/gap/recon structs, seed-cache types, logging, config + base params
include("atlas/recon.jl")       # reconnaissance sampling, classification, adaptive recon, window segmentation
include("atlas/seeding.jl")     # search boxes, fallback bounds, neighbor-seed cache, continuation retry configs
include("atlas/recovery.jl")    # window branch recovery, geometry diagnostics, interval/window coverage
include("atlas/gap_refine.jl")  # gap refinement, branch-switching probes, summaries, continuation_atlas entry
