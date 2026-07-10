#
# PkgBenchmark-compatible benchmark suite for DynamicsKit.
#
# This lives in the isolated `bench/` environment (bench/Project.toml), which
# path-links this working copy of DynamicsKit and adds BenchmarkTools +
# PkgBenchmark. Those tools are deliberately kept OUT of the package's own
# [deps] so the committed Manifest and the Aqua quality pass stay clean — the
# existing hand-rolled `@elapsed` benchmarks made the same choice.
#
# Run the raw suite directly (fastest inner loop):
#   julia --project=bench bench/benchmarks.jl
#
# Run via PkgBenchmark (pass script= because the suite lives in bench/, not the
# PkgBenchmark-default benchmark/):
#   julia --project=bench -e 'using PkgBenchmark; r = benchmarkpkg("DynamicsKit"; script="bench/benchmarks.jl"); export_markdown("bench/result.md", r)'
#
# Compare the working copy against a git ref (regression tracking across a
# DynamicsKit release bump — the reason this exists):
#   julia --project=bench -e 'using PkgBenchmark; judge("DynamicsKit", "v0.1.3"; script="bench/benchmarks.jl") |> export_markdown'
#
# All cases use the Hénon map: discrete, cheap, deterministic — fast enough for
# repeated sampling while still exercising the brute-force, continuation, and
# skeleton kernels.

using BenchmarkTools
using DynamicsKit

const SUITE = BenchmarkGroup()

const SYS = henon_map()

# Period-1 fixed point of the Hénon map at a = 0.3 (x* = -0.7 x + 1 - a x², y = 0.3 x).
const A0 = 0.3
const X_STAR = (-0.7 + sqrt(0.49 + 4 * A0)) / (2 * A0)
const P1_POINT = [X_STAR, 0.3 * X_STAR]

# ── Brute-force bifurcation diagram ──
SUITE["brute_force"] = BenchmarkGroup()
SUITE["brute_force"]["henon"] = @benchmarkable brute_force_diagram($SYS, cfg) setup = (
    cfg = BruteForceConfig(
        param_min = 0.0, param_max = 1.4,
        param_steps = 120, iterations = 200, transient = 100,
    )
)

# ── Continuation (period-1 branch) ──
SUITE["continuation"] = BenchmarkGroup()
SUITE["continuation"]["henon_p1"] = @benchmarkable(
    continuation_branch($SYS, cfg; initial_point = $P1_POINT, params = [$A0]),
    setup = (
        cfg = ContinuationConfig(
            p_min = 0.0, p_max = 1.4,
            ds = 0.01, dsmax = 0.05,
            max_steps = 1000, newton_tol = 1e-10, detect_bifurcation = 3,
        )
    )
)

# ── Periodic skeleton search (periods 1–5) ──
SUITE["skeleton"] = BenchmarkGroup()
SUITE["skeleton"]["henon_1to5"] = @benchmarkable find_periodic_skeleton(
    $SYS, 1:5, 1.2;
    n_initial = 8, search_min = [-2.0, -1.0], search_max = [2.0, 1.0], params = [1.2],
)

# When run as a script (not via PkgBenchmark), tune + run and print a summary.
if abspath(PROGRAM_FILE) == @__FILE__
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    println("\n══ median times ══")
    for (group, bg) in results
        for (case, trial) in bg
            println("  $(group)/$(case): ", BenchmarkTools.prettytime(median(trial).time))
        end
    end
end
