"""
Complete Hénon map analysis: brute-force diagram + continuation branches (periods 1–10).
"""

using DynamicsKit
using StaticArrays
using Plots

println("═══ Hénon Map — Complete Bifurcation Analysis ═══\n")

sys = henon_map()

# ── 1. Brute-force bifurcation diagram ──
println("1. Generating brute-force bifurcation diagram...")
bf_config = BruteForceConfig(
    param_min = 0.0,
    param_max = 1.4,
    param_steps = 400,
    iterations = 500,
    transient = 300
)

@time bf_result = brute_force_diagram(sys, bf_config)
println("   $(length(bf_result.params)) points collected.\n")

# ── 2. Period-1 continuation branch ──
println("2. Computing period-1 continuation branch...")
a0 = 0.3
x_star = (-0.7 + sqrt(0.49 + 4*a0)) / (2*a0)
y_star = 0.3 * x_star

cont_config = ContinuationConfig(
    p_min = 0.0, p_max = 1.4,
    ds = 0.01, dsmax = 0.05,
    max_steps = 1000,
    newton_tol = 1e-10,
    detect_bifurcation = 3
)

@time br1 = continuation_branch(sys, cont_config;
                                initial_point = [x_star, y_star],
                                params = [a0])
println("   $(length(br1.branch)) points on branch.")
println("   Special points: ", length(br1.branch.specialpoint))
for sp in br1.branch.specialpoint
    println("     $(sp.type) at a = $(round(sp.param, digits=5))")
end
println()

# ── 3. Period-2 continuation branch ──
println("3. Computing period-2 continuation branch...")
F2 = x -> begin
    sv = SVector{2}(x)
    sv = sys.f(sv, [1.0])
    sv = sys.f(sv, [1.0])
    Array(sv) .- x
end
x0_p2, _ = DynamicsKit._newton_ad(F2, [-0.5, 0.3], 1e-12, 50)

cont_config_p2 = ContinuationConfig(
    p_min = 0.0, p_max = 1.4,
    ds = 0.005, dsmax = 0.03,
    max_steps = 1000,
    newton_tol = 1e-10,
    detect_bifurcation = 3
)

@time br2 = continuation_branch(sys, cont_config_p2, 2;
                                initial_point = x0_p2,
                                params = [1.0])
println("   $(length(br2.branch)) points on period-2 branch.\n")

# ── 4. Periodic skeleton → higher period branches ──
println("4. Finding periodic skeleton at a=1.2...")
@time skeleton = find_periodic_skeleton(sys, 1:10, 1.2;
                                        n_initial=15,
                                        search_min=[-2.0, -1.0],
                                        search_max=[2.0, 1.0],
                                        params=[1.2])
println("   Found $(length(skeleton)) orbits:")
for s in skeleton
    stabstr = s.stable ? "stable" : "unstable"
    println("     Period $(s.period): x₁ = $(round(s.point[1], digits=4)) ($stabstr)")
end
println()

# ── 5. Plot results ──
println("5. Generating plots...")

# Brute-force diagram
p1 = plot_brute_force(bf_result)
savefig(p1, "henon_brute_force.png")
println("   → henon_brute_force.png")

# Branches overlay
branches = [br1, br2]
p2 = plot_overlay(bf_result, branches)
savefig(p2, "henon_overlay.png")
println("   → henon_overlay.png")

println("\n═══ Done! ═══")

