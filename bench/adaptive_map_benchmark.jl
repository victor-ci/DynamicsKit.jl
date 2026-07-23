#
# Benchmark for the adaptive bifurcation-map serializer.
#
# Measures serialize + JLD2 write, file size, JLD2 load + deserialize, and
# verifies exact result equality.  Scale is controlled by the HENONSIZE
# environment variable (default 51 for fast runs; use 501 for large-scale
# measurement).
#
# Run:
#   julia --project=bench bench/adaptive_map_benchmark.jl
#   HENONSIZE=501 julia --project=bench bench/adaptive_map_benchmark.jl
#
# Requires the bench/ environment (bench/Project.toml links this DynamicsKit
# checkout and adds BenchmarkTools + JLD2).

using BenchmarkTools
using DynamicsKit
using JLD2
using Dates

const GRID_N = parse(Int, get(ENV, "HENONSIZE", "51"))
println("adaptive_map_benchmark: HENONSIZE=$(GRID_N), Julia $(VERSION)")

const SYS = henon_map()
const COARSE_CFG = BifurcationMapConfig(
    a_min = -1.5, a_max = 1.5, b_min = -1.5, b_max = 1.5,
    a_steps = GRID_N - 1, b_steps = GRID_N - 1,
    a_index = 1, b_index = 2,
    max_period = 16, iterations = 200,
    base_params = [0.3, 0.3],
)
const ADAPT_CFG = AdaptiveMapConfig(
    total_budget = GRID_N * GRID_N * 4,
    max_depth    = 3,
)

println("Running coarse + adaptive map ($(GRID_N)x$(GRID_N) coarse grid)...")
const RESULT = adaptive_bifurcation_map(SYS, COARSE_CFG, ADAPT_CFG; initial_point = [0.0, 0.0])
println("  samples=$(length(RESULT.samples))  leaf_cells=$(length(RESULT.leaf_cells))  segments=$(length(RESULT.boundary_segments))")
println("  budget_used=$(RESULT.budget_used)/$(RESULT.total_budget)  flagged=$(RESULT.flagged_cells)  splits=$(RESULT.split_cells)")

# ── Serialize benchmark ──────────────────────────────────────────────────────────
t_ser = @belapsed serialize_adaptive_map_result($RESULT)
data  = serialize_adaptive_map_result(RESULT)
println("Serialize: $(round(t_ser * 1e3; digits=2)) ms")

# ── JLD2 write benchmark ─────────────────────────────────────────────────────────
tmpfile = joinpath(dirname(@__FILE__), "_bench_adaptive_tmp.jld2")
t_write = @belapsed jldsave($tmpfile; adaptive_result=$data)
fsize_kb = round(filesize(tmpfile) / 1024; digits=1)
println("JLD2 write: $(round(t_write * 1e3; digits=2)) ms  file=$(fsize_kb) KB")

# ── JLD2 load + deserialize benchmark ────────────────────────────────────────────
t_load_deser = @belapsed begin
    raw  = load($tmpfile, "adaptive_result")
    deserialize_adaptive_map_result(raw)
end
println("JLD2 load+deserialize: $(round(t_load_deser * 1e3; digits=2)) ms")

# ── Exact equality check ──────────────────────────────────────────────────────────
raw2    = load(tmpfile, "adaptive_result")
result2 = deserialize_adaptive_map_result(raw2)
@assert length(result2.samples)           == length(RESULT.samples)          "sample count mismatch"
@assert length(result2.leaf_cells)        == length(RESULT.leaf_cells)       "leaf cell count mismatch"
@assert length(result2.boundary_segments) == length(RESULT.boundary_segments) "segment count mismatch"
@assert result2.flagged_cells    == RESULT.flagged_cells    "flagged_cells mismatch"
@assert result2.split_cells      == RESULT.split_cells      "split_cells mismatch"
@assert result2.budget_used      == RESULT.budget_used      "budget_used mismatch"
@assert result2.timestamp        == RESULT.timestamp        "timestamp mismatch"
@assert result2.coarse_result.periodicity == RESULT.coarse_result.periodicity "coarse periodicity mismatch"
println("Exact equality: PASS")

# ── Cleanup ────────────────────────────────────────────────────────────────────────
isfile(tmpfile) && rm(tmpfile)
println("Done.")
