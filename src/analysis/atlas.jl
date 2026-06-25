# atlas.jl — decomposed into focused files (behavior-preserving: the original top-to-bottom
# definition order is preserved by the include sequence below). See
# docs/internal/repository-split-plan-v2.md.
include("atlas/core.jl")        # result/window/gap/recon structs, seed-cache types, logging, config + base params
include("atlas/recon.jl")       # reconnaissance sampling, classification, adaptive recon, window segmentation
include("atlas/seeding.jl")     # search boxes, fallback bounds, neighbor-seed cache, continuation retry configs
include("atlas/recovery.jl")    # window branch recovery, geometry diagnostics, interval/window coverage
include("atlas/gap_refine.jl")  # gap refinement, branch-switching probes, summaries, continuation_atlas entry
