@testset "Adaptive bifurcation-map refinement" begin

@testset "Public status labels" begin
    @test map_status_code(:unknown) == 0
    @test map_status_code(:periodic) == 1
    @test map_status_code(:aperiodic_or_high_period) == 2
    @test map_status_label(0) == "unknown"
    @test map_status_label(1) == "periodic"
    @test map_status_label(2) == "aperiodic_or_high_period"
    @test map_status_label(999) == "unknown"
end

# ─── Analytic circular-boundary map ─────────────────────────────────────────────
#
# f(x, p) = x    (period-1 fixed point)  when p[1]^2 + p[2]^2 < R^2
#           -x   (period-2 flip orbit)   otherwise
#
# The true boundary is a circle of radius R = 0.3 inscribed in [-0.5, 0.5]^2.
# Both branches have exact period detection with precision=1e-6 and iter=50.

_CIRCLE_R_SQ = 0.09

_make_circle_map() = DiscreteMap(
    (x, p) -> p[1]^2 + p[2]^2 < _CIRCLE_R_SQ ? SVector(x[1]) : SVector(-x[1]),
    1, [:a, :b], "Circle boundary map"
)

_circle_coarse_config(; a_steps=6, b_steps=6, kw...) = BifurcationMapConfig(
    a_min=-0.5, a_max=0.5, b_min=-0.5, b_max=0.5,
    a_steps=a_steps, b_steps=b_steps,
    a_index=1, b_index=2,
    max_period=4, iterations=50, precision=1e-6,
    base_params=[0.0, 0.0]; kw...
)

# ─── Analytic enclosed-island map ───────────────────────────────────────────────
#
# All four corners of a coarse cell ([-0.5,0.5]^2 with a_steps=b_steps=1) are
# period-2 (outside r=0.3 circle), but the cell center (0,0) is period-1 (inside).
# With center-screening enabled this island must be discovered.

_make_island_coarse_config() = BifurcationMapConfig(
    a_min=-0.5, a_max=0.5, b_min=-0.5, b_max=0.5,
    a_steps=1, b_steps=1,
    a_index=1, b_index=2,
    max_period=4, iterations=50, precision=1e-6,
    base_params=[0.0, 0.0]
)

# ─── AdaptiveMapConfig validation ───────────────────────────────────────────────

@testset "AdaptiveMapConfig validation" begin
    cfg = AdaptiveMapConfig(total_budget=200, max_depth=3)
    @test cfg.total_budget == 200
    @test cfg.max_depth == 3
    @test cfg.refine_on_period_disagreement == true

    @test_throws AssertionError AdaptiveMapConfig(total_budget=0)
    @test_throws AssertionError AdaptiveMapConfig(max_depth=-1)
    @test_throws AssertionError AdaptiveMapConfig(min_confidence=1.1)
    @test_throws AssertionError AdaptiveMapConfig(confidence_delta=-0.1)
    @test_throws AssertionError AdaptiveMapConfig(
        refine_on_period_disagreement=false,
        refine_on_status_disagreement=false,
        min_confidence=0.0,
        confidence_delta=0.0
    )
end

# ─── Confidence-only trigger mode ────────────────────────────────────────────────

@testset "Classification boundaries remain explicit when disagreement triggers are disabled" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=2, b_steps=2)
    coarse_evals = 3 * 3
    adaptive = AdaptiveMapConfig(
        total_budget=100,
        max_depth=2,
        refine_on_period_disagreement=false,
        refine_on_status_disagreement=false,
        min_confidence=0.5,
    )

    result = adaptive_bifurcation_map(sys, coarse, adaptive; initial_point=[0.5])

    @test result.budget_used == coarse_evals
    @test result.refinement_evaluations == 0
    @test result.flagged_cells == 0
    @test result.split_cells == 0
    @test length(result.leaf_cells) == 4
    @test all(cell -> cell.terminal == :boundary, result.leaf_cells)
    @test all(cell -> cell.si_center == 0, result.leaf_cells)
    @test !isempty(result.boundary_segments)
end

# ─── Strict budget invariant ─────────────────────────────────────────────────────

@testset "Strict budget invariant: reject total_budget < coarse grid size" begin
    sys  = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    coarse_evals = (4 + 1) * (4 + 1)  # 25

    # Budget exactly one below coarse requirement → must throw before any sweep.
    @test_throws ArgumentError adaptive_bifurcation_map(
        sys, coarse, AdaptiveMapConfig(total_budget=coarse_evals - 1);
        initial_point=[0.5])

    # Budget of 0 → must throw.
    @test_throws AssertionError AdaptiveMapConfig(total_budget=0)

    # Budget exactly equal to coarse → succeeds; zero refinement.
    result = adaptive_bifurcation_map(sys, coarse,
                                       AdaptiveMapConfig(total_budget=coarse_evals);
                                       initial_point=[0.5])
    @test result.coarse_evaluations == coarse_evals
    @test result.refinement_evaluations == 0
    @test result.budget_used == coarse_evals
end

# ─── Fixed-seed restriction ──────────────────────────────────────────────────────

@testset "Fixed-seed restriction" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(reuse_neighbor_seeds=true)
    adaptive = AdaptiveMapConfig(total_budget=500)
    @test_throws ArgumentError adaptive_bifurcation_map(sys, coarse, adaptive;
                                                         initial_point=[0.5])
end

# ─── Known classifications ───────────────────────────────────────────────────────

@testset "Analytic classification at known points" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=6, b_steps=6)
    adaptive = AdaptiveMapConfig(total_budget=500, max_depth=2)

    result = adaptive_bifurcation_map(sys, coarse, adaptive; initial_point=[0.5])

    @test result.coarse_evaluations == 7 * 7   # (a_steps+1)*(b_steps+1)

    orig_idx = findfirst(s -> s.a == 0.0 && s.b == 0.0, result.samples)
    @test !isnothing(orig_idx)
    @test result.samples[orig_idx].period == 1

    far_idx = findfirst(s -> s.a == 0.5 && s.b == 0.5, result.samples)
    @test !isnothing(far_idx)
    @test result.samples[far_idx].period == 2

    @test length(result.samples) <= adaptive.total_budget
    @test result isa AdaptiveMapResult
    @test result.coarse_result isa BifurcationMapResult
end

# ─── Budget semantics ────────────────────────────────────────────────────────────

@testset "Budget semantics" begin
    sys    = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    coarse_evals = 5 * 5  # 25

    # Tight budget: no refinement, budget_exhausted reflects uninspected uniform cells.
    result_tight = adaptive_bifurcation_map(sys, coarse,
                                             AdaptiveMapConfig(total_budget=coarse_evals);
                                             initial_point=[0.5])
    @test result_tight.budget_used == coarse_evals
    @test result_tight.refinement_evaluations == 0
    @test length(result_tight.samples) == coarse_evals
    # budget_used <= total_budget always.
    @test result_tight.budget_used <= result_tight.total_budget

    # Medium budget: some refinement, strictly within budget.
    medium_budget = coarse_evals + 30
    result_medium = adaptive_bifurcation_map(sys, coarse,
                                              AdaptiveMapConfig(total_budget=medium_budget);
                                              initial_point=[0.5])
    @test length(result_medium.samples) <= medium_budget
    @test result_medium.budget_used <= medium_budget
    @test result_medium.budget_used == length(result_medium.samples)

    # Large budget: refinement terminates cleanly; budget_exhausted is false.
    # max_depth=2 keeps work bounded to ~100 evals; budget=2000 is generous.
    large_budget = 2000
    result_large = adaptive_bifurcation_map(sys, coarse,
                                             AdaptiveMapConfig(total_budget=large_budget, max_depth=2);
                                             initial_point=[0.5])
    @test length(result_large.samples) <= large_budget
    @test result_large.budget_used <= large_budget
    @test !result_large.budget_exhausted

    # Deliberately tiny budget that SHOULD exhaust mid-refinement.
    tiny_budget = coarse_evals + 3
    result_tiny = adaptive_bifurcation_map(sys, coarse,
                                            AdaptiveMapConfig(total_budget=tiny_budget, max_depth=4);
                                            initial_point=[0.5])
    @test result_tiny.budget_used <= tiny_budget
    # Whether exhausted depends on whether budget_limited leaves exist — just check invariant.
    @test result_tiny.budget_used == length(result_tiny.samples)
end

# ─── Deterministic replay ────────────────────────────────────────────────────────

@testset "Deterministic replay" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    adaptive = AdaptiveMapConfig(total_budget=120, max_depth=2)

    result1 = adaptive_bifurcation_map(sys, coarse, adaptive; initial_point=[0.5])
    result2 = adaptive_bifurcation_map(sys, coarse, adaptive; initial_point=[0.5])

    @test length(result1.samples) == length(result2.samples)
    for (s1, s2) in zip(result1.samples, result2.samples)
        @test s1.a == s2.a && s1.b == s2.b
        @test s1.period == s2.period
        @test s1.depth == s2.depth
    end
    @test length(result1.leaf_cells) == length(result2.leaf_cells)
    for (c1, c2) in zip(result1.leaf_cells, result2.leaf_cells)
        @test c1.a0 == c2.a0 && c1.a1 == c2.a1
        @test c1.b0 == c2.b0 && c1.b1 == c2.b1
        @test c1.depth == c2.depth
        @test c1.terminal == c2.terminal
    end
    @test length(result1.boundary_segments) == length(result2.boundary_segments)
end

# ─── Topology invariants ─────────────────────────────────────────────────────────

@testset "Topology invariants and quadtree tiling" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    adaptive = AdaptiveMapConfig(total_budget=300, max_depth=3)
    result = adaptive_bifurcation_map(sys, coarse, adaptive; initial_point=[0.5])

    n_samples = length(result.samples)

    for cell in result.leaf_cells
        @test cell.a1 > cell.a0
        @test cell.b1 > cell.b0
    end

    for cell in result.leaf_cells
        @test 1 <= cell.si_sw <= n_samples
        @test 1 <= cell.si_se <= n_samples
        @test 1 <= cell.si_nw <= n_samples
        @test 1 <= cell.si_ne <= n_samples
        @test cell.si_center == 0 || (1 <= cell.si_center <= n_samples)
    end

    @test all(c -> c.depth <= adaptive.max_depth, result.leaf_cells)

    domain_area   = (coarse.a_max - coarse.a_min) * (coarse.b_max - coarse.b_min)
    total_leaf_area = sum(c -> (c.a1 - c.a0) * (c.b1 - c.b0), result.leaf_cells)
    @test total_leaf_area ≈ domain_area atol=1e-10

    coarse_count = (coarse.a_steps + 1) * (coarse.b_steps + 1)
    @test count(s -> s.depth == 0, result.samples) == coarse_count
end

# ─── Enclosed island detection via center-screening ──────────────────────────────
#
# The single coarse cell (a_steps=b_steps=1) has all four corners at period-2
# (they lie outside the R=0.3 circle), but the cell center (0,0) is period-1
# (inside the circle).  The coarse sweep produces a uniform interior cell.
# Center-screening must discover the disagreement and trigger refinement.

@testset "Enclosed island detected via center-screening" begin
    sys    = _make_circle_map()
    coarse = _make_island_coarse_config()

    # Budget large enough to center-screen and split.
    result = adaptive_bifurcation_map(sys, coarse,
                                       AdaptiveMapConfig(total_budget=200, max_depth=3);
                                       initial_point=[0.5])

    # The center (0,0) must be present in samples as period-1.
    ctr_idx = findfirst(s -> s.a ≈ 0.0 && s.b ≈ 0.0, result.samples)
    @test !isnothing(ctr_idx)
    @test result.samples[ctr_idx].period == 1

    # At least one split must have occurred (the island triggered refinement).
    @test result.split_cells >= 1

    # Some boundary segments must be present.
    @test length(result.boundary_segments) > 0

    # At least one cell near the origin should be period-1 (the island).
    period1_in_leaf = any(result.leaf_cells) do cell
        cell.si_sw > 0 &&
        result.samples[cell.si_sw].period == 1
    end
    @test period1_in_leaf
end

@testset "Enclosed island NOT discovered when budget too tight" begin
    sys    = _make_circle_map()
    coarse = _make_island_coarse_config()
    coarse_evals = 2 * 2  # 4

    # Budget exactly at coarse level: no center-screening possible.
    result = adaptive_bifurcation_map(sys, coarse,
                                       AdaptiveMapConfig(total_budget=coarse_evals, max_depth=3);
                                       initial_point=[0.5])
    # All coarse corners are period-2 → no boundary segments if center not screened.
    @test result.split_cells == 0
    @test result.uninspected_cell_count == 1
end

# ─── uninspected_cell_count ──────────────────────────────────────────────────────

@testset "uninspected_cell_count reflects budget limitations" begin
    sys    = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    coarse_evals = 25

    # Large budget: all uniform cells should be center-screened → uninspected == 0.
    result_large = adaptive_bifurcation_map(sys, coarse,
                                             AdaptiveMapConfig(total_budget=coarse_evals + 500, max_depth=3);
                                             initial_point=[0.5])
    @test result_large.uninspected_cell_count == 0

    # budget_used <= total_budget always.
    @test result_large.budget_used <= result_large.total_budget
end

# ─── Boundary segment geometry ───────────────────────────────────────────────────

@testset "Boundary segment extraction and geometry" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=6, b_steps=6)

    result_coarse = adaptive_bifurcation_map(sys, coarse,
                                              AdaptiveMapConfig(total_budget=100, max_depth=1);
                                              initial_point=[0.5])
    result_fine   = adaptive_bifurcation_map(sys, coarse,
                                              AdaptiveMapConfig(total_budget=600, max_depth=4);
                                              initial_point=[0.5])

    # Finer run should produce at least as many boundary segments.
    @test length(result_fine.boundary_segments) >= length(result_coarse.boundary_segments)

    # Each segment has distinct endpoints.
    for seg in result_fine.boundary_segments
        @test !(seg.a0 == seg.a1 && seg.b0 == seg.b1)
    end

    # All endpoint coordinates lie within the domain.
    for seg in result_fine.boundary_segments
        @test coarse.a_min <= seg.a0 <= coarse.a_max
        @test coarse.a_min <= seg.a1 <= coarse.a_max
        @test coarse.b_min <= seg.b0 <= coarse.b_max
        @test coarse.b_min <= seg.b1 <= coarse.b_max
    end

    # Canonical key ordering.
    for seg in result_fine.boundary_segments
        @test seg.key_a <= seg.key_b
        @test seg.key_a[1] >= 0
        @test seg.key_b[1] >= 0
    end

    # Ambiguity values are valid symbols.
    valid_ambiguity = Set([:resolved, :ambiguous, :multi_region])
    for seg in result_fine.boundary_segments
        @test seg.ambiguity in valid_ambiguity
    end

    # boundary_length > 0 when segments exist.
    summ = adaptive_map_summary(result_fine)
    @test summ.boundary_length >= 0.0
    if length(result_fine.boundary_segments) > 0
        @test summ.boundary_length > 0.0
    end
    @test summ.resolved_segments + summ.ambiguous_segments + summ.multi_region_segments ==
          length(result_fine.boundary_segments)
end

# ─── Serialization round-trip ────────────────────────────────────────────────────

@testset "Serialization round-trip" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    adaptive = AdaptiveMapConfig(total_budget=150, max_depth=2)
    result = adaptive_bifurcation_map(sys, coarse, adaptive; initial_point=[0.5])

    data = serialize_adaptive_map_result(result)
    @test data["format"] == "adaptive-map-v2"
    @test data["totalBudget"] == result.total_budget
    @test haskey(data, "uninspectedCellCount")
    @test haskey(data, "flaggedCells")
    # columnar structure: samples/leafCells/boundarySegments must be dicts of arrays
    @test data["samples"] isa Dict
    @test data["leafCells"] isa Dict
    @test data["boundarySegments"] isa Dict
    @test haskey(data["samples"], "confidence")
    @test haskey(data["leafCells"], "reasonsBitmask")
    @test haskey(data["boundarySegments"], "ambiguity")

    result2 = deserialize_adaptive_map_result(data)

    @test result2.total_budget         == result.total_budget
    @test result2.budget_used          == result.budget_used
    @test result2.coarse_evaluations   == result.coarse_evaluations
    @test result2.refinement_evaluations == result.refinement_evaluations
    @test result2.budget_exhausted     == result.budget_exhausted
    @test result2.uninspected_cell_count == result.uninspected_cell_count
    @test result2.max_depth_reached    == result.max_depth_reached
    @test result2.max_depth_allowed    == result.max_depth_allowed
    @test result2.flagged_cells        == result.flagged_cells
    @test result2.split_cells          == result.split_cells
    @test result2.system_name          == result.system_name
    @test result2.param_names          == result.param_names
    @test result2.compute_backend      == result.compute_backend

    @test length(result2.samples)         == length(result.samples)
    @test length(result2.leaf_cells)      == length(result.leaf_cells)
    @test length(result2.boundary_segments) == length(result.boundary_segments)

    for (s1, s2) in zip(result.samples, result2.samples)
        @test s1.a == s2.a && s1.b == s2.b
        @test s1.period == s2.period
        @test s1.status_code == s2.status_code
        @test s1.confidence ≈ s2.confidence atol=1e-12
        @test s1.depth == s2.depth
    end

    for (c1, c2) in zip(result.leaf_cells, result2.leaf_cells)
        @test c1.a0 == c2.a0 && c1.a1 == c2.a1
        @test c1.b0 == c2.b0 && c1.b1 == c2.b1
        @test c1.ia0 == c2.ia0 && c1.ia1 == c2.ia1
        @test c1.ib0 == c2.ib0 && c1.ib1 == c2.ib1
        @test c1.depth == c2.depth
        @test c1.si_sw == c2.si_sw && c1.si_se == c2.si_se
        @test c1.si_nw == c2.si_nw && c1.si_ne == c2.si_ne
        @test c1.si_center == c2.si_center
        @test c1.terminal == c2.terminal
        @test c1.reasons == c2.reasons
    end

    for (seg1, seg2) in zip(result.boundary_segments, result2.boundary_segments)
        @test seg1.a0 == seg2.a0 && seg1.b0 == seg2.b0
        @test seg1.a1 == seg2.a1 && seg1.b1 == seg2.b1
        @test seg1.key_a == seg2.key_a && seg1.key_b == seg2.key_b
        @test seg1.ambiguity == seg2.ambiguity
    end

    @test result2.coarse_result.a_grid ≈ result.coarse_result.a_grid
    @test result2.coarse_result.b_grid ≈ result.coarse_result.b_grid
    @test result2.coarse_result.periodicity == result.coarse_result.periodicity
end

# ─── Malformed input rejection ───────────────────────────────────────────────────

@testset "Malformed input rejection" begin
    @test_throws ArgumentError deserialize_adaptive_map_result(
        Dict{String,Any}("format" => "adaptive-map-v99"))

    @test_throws Exception deserialize_adaptive_map_result(Dict{String,Any}())

    # Build a valid base dict and corrupt individual fields.
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=2, b_steps=2)
    result = adaptive_bifurcation_map(sys, coarse,
                                       AdaptiveMapConfig(total_budget=50, max_depth=1);
                                       initial_point=[0.5])
    valid_data = serialize_adaptive_map_result(result)

    # Bad ambiguity code in a segment (only if segments exist).
    if !isempty(result.boundary_segments)
        bad_seg = copy(valid_data)
        bad_segs = deepcopy(valid_data["boundarySegments"])
        bad_segs["ambiguity"][1] = 999
        bad_seg["boundarySegments"] = bad_segs
        @test_throws ArgumentError deserialize_adaptive_map_result(bad_seg)
    end

    # Bad terminal code in a leaf cell.
    bad_cell = copy(valid_data)
    bad_cells = deepcopy(valid_data["leafCells"])
    bad_cells["terminal"][1] = 999
    bad_cell["leafCells"] = bad_cells
    @test_throws ArgumentError deserialize_adaptive_map_result(bad_cell)

    # Out-of-range sample index in a leaf cell.
    bad_idx = copy(valid_data)
    bad_idx_cells = deepcopy(valid_data["leafCells"])
    bad_idx_cells["siSw"][1] = length(result.samples) + 999
    bad_idx["leafCells"] = bad_idx_cells
    @test_throws ArgumentError deserialize_adaptive_map_result(bad_idx)

    # Unequal column lengths in samples.
    bad_lens = copy(valid_data)
    bad_samples = deepcopy(valid_data["samples"])
    pop!(bad_samples["a"])
    bad_lens["samples"] = bad_samples
    @test_throws ArgumentError deserialize_adaptive_map_result(bad_lens)

    # Missing required column in leafCells.
    bad_missing = copy(valid_data)
    bad_lc = deepcopy(valid_data["leafCells"])
    delete!(bad_lc, "terminal")
    bad_missing["leafCells"] = bad_lc
    @test_throws ArgumentError deserialize_adaptive_map_result(bad_missing)

    # budget_used > total_budget.
    bad_budget = copy(valid_data)
    bad_budget["budgetUsed"] = valid_data["totalBudget"] + 1
    @test_throws ArgumentError deserialize_adaptive_map_result(bad_budget)
end

# ─── Cache reuse: deduplication ──────────────────────────────────────────────────

@testset "Cache deduplication (unique coordinates)" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    adaptive = AdaptiveMapConfig(total_budget=300, max_depth=3)

    result = adaptive_bifurcation_map(sys, coarse, adaptive; initial_point=[0.5])

    coords = Set{Tuple{Float64,Float64}}()
    for s in result.samples
        push!(coords, (s.a, s.b))
    end
    @test length(coords) == length(result.samples)
    @test result.budget_used == length(result.samples)
    @test result.budget_used <= result.total_budget
end

# ─── Backend provenance and adaptive_map_summary ─────────────────────────────────

@testset "Backend provenance and adaptive_map_summary" begin
    sys = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    adaptive = AdaptiveMapConfig(total_budget=150, max_depth=2)

    result = adaptive_bifurcation_map(sys, coarse, adaptive;
                                       initial_point=[0.5],
                                       backend=CPUBackend())
    @test result.compute_backend == :cpu

    summ = adaptive_map_summary(result)
    @test summ.compute_backend == :cpu
    @test summ.total_budget == adaptive.total_budget
    @test summ.coarse_evaluations == (coarse.a_steps + 1) * (coarse.b_steps + 1)
    @test summ.budget_used == result.budget_used
    @test summ.coarse_a_steps == coarse.a_steps
    @test summ.coarse_b_steps == coarse.b_steps
    @test summ.uninspected_cell_count == result.uninspected_cell_count
    @test summ.boundary_length >= 0.0
    @test summ.resolved_segments + summ.ambiguous_segments + summ.multi_region_segments ==
          summ.boundary_segment_count
end


# ─── flagged_cells exact counting ───────────────────────────────────────────────
#
# Test that flagged_cells counts ALL triggered cells, including:
#  - coarse cells at max_depth == 0 (cannot split, go directly to :boundary leaf)
#  - children that trigger at max_depth (split happened but children hit depth limit)

@testset "flagged_cells counts all triggered cells at max_depth==0" begin
    sys    = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=6, b_steps=6)

    # max_depth=0: no splitting allowed; triggered coarse cells become boundary leaves.
    result0 = adaptive_bifurcation_map(sys, coarse,
                                        AdaptiveMapConfig(total_budget=10000, max_depth=0);
                                        initial_point=[0.5])
    # Every triggered cell is a boundary leaf; each must be counted.
    boundary_count = count(c -> c.terminal == :boundary, result0.leaf_cells)
    @test result0.flagged_cells == boundary_count
    @test result0.flagged_cells > 0     # circle map must trigger some boundary cells
    @test result0.split_cells == 0      # no splits at depth 0
end

@testset "flagged_cells counts children that trigger at max_depth" begin
    sys    = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)

    # max_depth=1: one level of splitting; children that still disagree become boundary.
    result1 = adaptive_bifurcation_map(sys, coarse,
                                        AdaptiveMapConfig(total_budget=10000, max_depth=1);
                                        initial_point=[0.5])
    # flagged_cells = coarse triggered + promoted center-disagreements + boundary children.
    # It must be >= the number of boundary leaf cells (each boundary was triggered once).
    boundary_count = count(c -> c.terminal == :boundary, result1.leaf_cells)
    @test result1.flagged_cells >= boundary_count

    # With depth=2 we get more splits; flagged_cells must grow or stay >= boundary count.
    result2 = adaptive_bifurcation_map(sys, coarse,
                                        AdaptiveMapConfig(total_budget=10000, max_depth=2);
                                        initial_point=[0.5])
    boundary_count2 = count(c -> c.terminal == :boundary, result2.leaf_cells)
    @test result2.flagged_cells >= boundary_count2

    @test result1.split_cells <= result1.flagged_cells
    @test result2.split_cells <= result2.flagged_cells
end

# ─── Timestamp provenance ───────────────────────────────────────────────────────

@testset "Timestamp matches coarse_result.timestamp" begin
    sys    = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    result = adaptive_bifurcation_map(sys, coarse,
                                       AdaptiveMapConfig(total_budget=200, max_depth=2);
                                       initial_point=[0.5])
    @test result.timestamp == result.coarse_result.timestamp
end

# ─── Type concreteness ──────────────────────────────────────────────────────────

@testset "Work item and crossing types are concrete" begin
    @test isconcretetype(DynamicsKit._AdaptiveWorkItem)
    @test isconcretetype(DynamicsKit._AdaptiveCrossing)
    @test fieldtype(DynamicsKit._AdaptiveWorkItem, :ia0)      == Int
    @test fieldtype(DynamicsKit._AdaptiveWorkItem, :a0)       == Float64
    @test fieldtype(DynamicsKit._AdaptiveWorkItem, :si_sw)    == Int
    @test fieldtype(DynamicsKit._AdaptiveWorkItem, :reasons)  == Vector{Symbol}
    @test fieldtype(DynamicsKit._AdaptiveCrossing, :ia)       == Int
    @test fieldtype(DynamicsKit._AdaptiveCrossing, :a)        == Float64
    @test fieldtype(DynamicsKit._AdaptiveCrossing, :key_left) == Tuple{Int,Int}
end

# ─── v2 serialization exact roundtrip ───────────────────────────────────────────

@testset "v2 serialization exact Float64 confidence roundtrip" begin
    sys    = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    result = adaptive_bifurcation_map(sys, coarse,
                                       AdaptiveMapConfig(total_budget=200, max_depth=2);
                                       initial_point=[0.5])
    data    = serialize_adaptive_map_result(result)
    result2 = deserialize_adaptive_map_result(data)

    # Exact Float64 confidence roundtrip (no narrowing).
    for (s1, s2) in zip(result.samples, result2.samples)
        @test s1.confidence === s2.confidence
    end

    # Exact reason roundtrip via bitmask.
    for (c1, c2) in zip(result.leaf_cells, result2.leaf_cells)
        @test sort(String.(c1.reasons)) == sort(String.(c2.reasons))
        @test c1.terminal == c2.terminal
    end

    # Exact segment key roundtrip.
    for (s1, s2) in zip(result.boundary_segments, result2.boundary_segments)
        @test s1.key_a == s2.key_a && s1.key_b == s2.key_b
        @test s1.ambiguity == s2.ambiguity
        @test s1.a0 === s2.a0 && s1.b0 === s2.b0
    end

    # Coarse result inside adaptive result roundtrips correctly.
    @test result.coarse_result.periodicity == result2.coarse_result.periodicity
    @test result.coarse_result.a_grid == result2.coarse_result.a_grid
end

@testset "v2 reasons bitmask all valid combinations" begin
    # Verify all reason symbols round-trip correctly.
    for sym in (:period_disagreement, :status_disagreement,
                :low_confidence, :confidence_delta, :center_disagreement)
        mask = DynamicsKit._encode_reason_bitmask(Symbol[sym])
        back = DynamicsKit._decode_reason_bitmask(mask)
        @test back == [sym]
    end

    # Combined reasons.
    reasons = [:period_disagreement, :low_confidence]
    mask    = DynamicsKit._encode_reason_bitmask(reasons)
    back    = DynamicsKit._decode_reason_bitmask(mask)
    @test Set(back) == Set(reasons)

    # Unknown reason throws.
    @test_throws ArgumentError DynamicsKit._encode_reason_bitmask([:unknown_reason])

    # Invalid bitmask throws.
    @test_throws ArgumentError DynamicsKit._decode_reason_bitmask(1 << 7)
end

@testset "bifurcation-map-v2 roundtrip" begin
    sys    = _make_circle_map()
    coarse = _circle_coarse_config(a_steps=4, b_steps=4)
    result = adaptive_bifurcation_map(sys, coarse,
                                       AdaptiveMapConfig(total_budget=10000, max_depth=0);
                                       initial_point=[0.5])
    cr  = result.coarse_result
    d   = serialize_bifurcation_map_result(cr)
    @test d["format"] == "bifurcation-map-v2"
    @test d["periodicity"] isa AbstractVector
    @test length(d["periodicity"]) == (coarse.a_steps + 1) * (coarse.b_steps + 1)
    cr2 = deserialize_bifurcation_map_result(d)
    @test cr2.periodicity == cr.periodicity
    @test cr2.a_grid == cr.a_grid
    @test cr2.b_grid == cr.b_grid
    @test cr2.max_period == cr.max_period
    @test cr2.system_name == cr.system_name
    @test cr2.compute_backend == cr.compute_backend
end

# ─── Regression: exhausted budget with center already in lookup ──────────────────
#
# When budget_remaining == 0 but the center lattice key is already in the lookup,
# Phase B must reuse it for free: no uninspected increment, si_center populated.
# If the existing center disagrees with corners it should still promote to refinement.

@testset "Phase B: existing center reused at zero budget — agrees with corners" begin
    # Single 1×1 coarse cell; all four corners period=2 (outside the circle).
    # Pre-insert the cell center into the lookup with the SAME key (period=2).
    # Budget = 0 → the center can only be reused, never newly evaluated.
    adaptive   = AdaptiveMapConfig(total_budget=200, max_depth=2)
    max_depth  = adaptive.max_depth
    scale      = 1 << max_depth   # 4

    # Coarse corners live at lattice coordinates (0,0), (4,0), (0,4), (4,4).
    status_periodic = DynamicsKit._map_status_code(:periodic)
    samples = AdaptiveMapSample[
        AdaptiveMapSample(-0.5, -0.5, 2, status_periodic, 0.9, 0),  # SW (0,0)
        AdaptiveMapSample( 0.5, -0.5, 2, status_periodic, 0.9, 0),  # SE (4,0)
        AdaptiveMapSample(-0.5,  0.5, 2, status_periodic, 0.9, 0),  # NW (0,4)
        AdaptiveMapSample( 0.5,  0.5, 2, status_periodic, 0.9, 0),  # NE (4,4)
    ]
    lookup = Dict{Tuple{Int,Int},Int}(
        (0, 0) => 1, (scale, 0) => 2, (0, scale) => 3, (scale, scale) => 4,
    )

    # Pre-insert center (2,2) with the SAME key as corners (period=2, periodic).
    ia_cc = scale ÷ 2; ib_cc = scale ÷ 2
    push!(samples, AdaptiveMapSample(0.0, 0.0, 2, status_periodic, 0.9, 1))
    lookup[(ia_cc, ib_cc)] = 5

    budget_remaining = Ref(0)
    detection_fn = (a, b) -> error("should not evaluate: budget is zero")

    coarse_result = BifurcationMapResult(
        [-0.5, 0.5], [-0.5, 0.5],
        [2 2; 2 2],
        4, "test", (:a, :b), DynamicsKit.Dates.now()
    )

    leaf_cells, _, _, _, _, uninspected, limited =
        DynamicsKit._run_adaptive_refinement(
            coarse_result, adaptive, samples, lookup, budget_remaining, detection_fn)

    # Existing center reused: no uninspected count, si_center == 5.
    @test uninspected == 0
    @test !limited
    interior = filter(c -> c.si_center == 5, leaf_cells)
    @test length(interior) == 1
    @test interior[1].terminal == :interior
end

@testset "Phase B: existing center reused at zero budget — disagrees with corners" begin
    # Same setup as above but center has a DIFFERENT key (period=1, inside circle).
    # With depth=0 < max_depth=2, the disagreement must promote to refinement.
    # The actual split then hits budget=0 and produces a budget_limited leaf.
    adaptive   = AdaptiveMapConfig(total_budget=200, max_depth=2)
    scale      = 1 << adaptive.max_depth   # 4

    status_periodic = DynamicsKit._map_status_code(:periodic)
    samples = AdaptiveMapSample[
        AdaptiveMapSample(-0.5, -0.5, 2, status_periodic, 0.9, 0),
        AdaptiveMapSample( 0.5, -0.5, 2, status_periodic, 0.9, 0),
        AdaptiveMapSample(-0.5,  0.5, 2, status_periodic, 0.9, 0),
        AdaptiveMapSample( 0.5,  0.5, 2, status_periodic, 0.9, 0),
    ]
    lookup = Dict{Tuple{Int,Int},Int}(
        (0, 0) => 1, (scale, 0) => 2, (0, scale) => 3, (scale, scale) => 4,
    )

    # Pre-insert center with a DIFFERENT key (period=1, same status code).
    ia_cc = scale ÷ 2; ib_cc = scale ÷ 2
    push!(samples, AdaptiveMapSample(0.0, 0.0, 1, status_periodic, 0.9, 1))
    lookup[(ia_cc, ib_cc)] = 5

    budget_remaining = Ref(0)
    detection_fn = (a, b) -> error("should not evaluate: budget is zero")

    coarse_result = BifurcationMapResult(
        [-0.5, 0.5], [-0.5, 0.5],
        [2 2; 2 2],
        4, "test", (:a, :b), DynamicsKit.Dates.now()
    )

    leaf_cells, _, _, _, flagged, uninspected, _ =
        DynamicsKit._run_adaptive_refinement(
            coarse_result, adaptive, samples, lookup, budget_remaining, detection_fn)

    # Center disagreement was detected for free (no budget charge).
    @test uninspected == 0
    # The promoted cell hits zero budget in Phase A → budget_limited leaf; flagged >= 1.
    @test flagged >= 1
    budget_limited = filter(c -> c.terminal == :budget_limited, leaf_cells)
    @test length(budget_limited) >= 1
    # The budget_limited leaf carries si_center == 5 (the existing center).
    @test any(c -> c.si_center == 5, budget_limited)
end

# ─── Regression: checkerboard with third center key → multi_region ───────────────

@testset "Checkerboard with third center key produces multi_region segments" begin
    # Build a checkerboard cell: SW/NE share key A, SE/NW share key B.
    # Center has key C (a third regime).
    # Expected: 4 crossing midpoints → cell-center spokes with :multi_region.
    key_A = (1, 1)   # (status_periodic, period=1)
    key_B = (1, 2)   # (status_periodic, period=2)
    key_C = (1, 3)   # (status_periodic, period=3) — third key

    samples = AdaptiveMapSample[
        AdaptiveMapSample(0.0, 0.0, key_A[2], key_A[1], 0.9, 0),  # SW  idx=1
        AdaptiveMapSample(2.0, 0.0, key_B[2], key_B[1], 0.9, 0),  # SE  idx=2
        AdaptiveMapSample(0.0, 2.0, key_B[2], key_B[1], 0.9, 0),  # NW  idx=3
        AdaptiveMapSample(2.0, 2.0, key_A[2], key_A[1], 0.9, 0),  # NE  idx=4
        AdaptiveMapSample(1.0, 1.0, key_C[2], key_C[1], 0.9, 1),  # center idx=5
    ]

    # ia0=0, ia1=2, ib0=0, ib1=2; integer lattice scale = 1 here for simplicity.
    cell = AdaptiveMapLeafCell(
        0.0, 2.0, 0.0, 2.0,   # a0,a1,b0,b1
        0, 2, 0, 2,            # ia0,ia1,ib0,ib1
        1,                     # depth
        1, 2, 3, 4,            # si_sw,si_se,si_nw,si_ne
        5,                     # si_center (key_C)
        :interior, Symbol[]
    )

    segments = AdaptiveMapSegment[]
    seg_set  = Set{NTuple{8,Int}}()
    DynamicsKit._adaptive_cell_segments!(segments, seg_set, samples, cell)

    # All 4 crossings must connect to the cell center with :multi_region.
    @test length(segments) == 4
    @test all(s -> s.ambiguity == :multi_region, segments)
    # Endpoints must include the cell center (1.0, 1.0).
    cell_center_pairs = filter(s -> (s.a1 ≈ 1.0 && s.b1 ≈ 1.0) || (s.a0 ≈ 1.0 && s.b0 ≈ 1.0), segments)
    @test length(cell_center_pairs) == 4
end

@testset "Checkerboard: center matches k_se still produces resolved segments" begin
    # Verify the elseif branch (k_ctr == k_se) still resolves correctly.
    key_A = (1, 1)
    key_B = (1, 2)

    samples = AdaptiveMapSample[
        AdaptiveMapSample(0.0, 0.0, key_A[2], key_A[1], 0.9, 0),  # SW  idx=1 (key_A)
        AdaptiveMapSample(2.0, 0.0, key_B[2], key_B[1], 0.9, 0),  # SE  idx=2 (key_B)
        AdaptiveMapSample(0.0, 2.0, key_B[2], key_B[1], 0.9, 0),  # NW  idx=3 (key_B)
        AdaptiveMapSample(2.0, 2.0, key_A[2], key_A[1], 0.9, 0),  # NE  idx=4 (key_A)
        AdaptiveMapSample(1.0, 1.0, key_B[2], key_B[1], 0.9, 1),  # center idx=5 (key_B=k_se)
    ]

    cell = AdaptiveMapLeafCell(
        0.0, 2.0, 0.0, 2.0,
        0, 2, 0, 2,
        1,
        1, 2, 3, 4,
        5,
        :interior, Symbol[]
    )

    segments = AdaptiveMapSegment[]
    seg_set  = Set{NTuple{8,Int}}()
    DynamicsKit._adaptive_cell_segments!(segments, seg_set, samples, cell)

    @test length(segments) == 2
    @test all(s -> s.ambiguity == :resolved, segments)
end

# ─── Regression: LyapunovFieldResult v2 non-square round-trip with NaN/Inf ───────

@testset "LyapunovFieldResult v2: non-square matrix with NaN/Inf round-trip" begin
    na = 3; nb = 2
    ag = Float64[1.0, 2.0, 3.0]
    bg = Float64[0.1, 0.2]

    exp_vals  = Float64[0.1, NaN, Inf, -Inf, 0.5, -0.3]
    cls_codes = Int[1, 2, 0, 1, 2, 0]
    est_codes = Int[0, 0, 1, 1, 0, 0]
    cnt_vals  = Int[10, 5, 0, 3, 8, 2]

    r = LyapunovFieldResult(
        ag, bg,
        reshape(exp_vals, na, nb),
        reshape(cls_codes, na, nb),
        reshape(est_codes, na, nb),
        reshape(cnt_vals, na, nb),
        1e-3,
        "test_sys",
        (:p, :q),
        DynamicsKit.Dates.DateTime("2024-01-01T00:00:00.0");
        compute_backend=:cpu,
    )

    data = DynamicsKit._serialize_lyapunov_field_result_v2(r)
    r2   = DynamicsKit._deserialize_lyapunov_field_result_v2(data)

    @test size(r2.exponents) == (na, nb)
    @test size(r2.classification_status_codes) == (na, nb)
    @test size(r2.estimation_status_codes) == (na, nb)
    @test size(r2.sample_counts) == (na, nb)

    # isequal treats NaN == NaN, which == does not.
    @test isequal(r2.exponents, r.exponents)
    @test r2.classification_status_codes == r.classification_status_codes
    @test r2.estimation_status_codes     == r.estimation_status_codes
    @test r2.sample_counts               == r.sample_counts
    @test r2.a_grid == r.a_grid
    @test r2.b_grid == r.b_grid
    @test r2.system_name == r.system_name
    @test r2.param_names == r.param_names
    @test r2.neutral_tolerance == r.neutral_tolerance
end

@testset "LyapunovFieldResult v2: serializer rejects mismatched matrix/grid shapes" begin
    # Construct a LyapunovFieldResult where the exponents matrix is (2×2)
    # but a_grid has length 3 — violates the na×nb contract.
    r_bad = LyapunovFieldResult(
        Float64[1.0, 2.0, 3.0],   # a_grid: length 3
        Float64[0.1, 0.2],         # b_grid: length 2
        fill(0.0, 2, 2),           # exponents: 2×2 (mismatch: should be 3×2)
        fill(0, 2, 2),
        fill(0, 2, 2),
        fill(0, 2, 2),
        1e-3, "test", (:a, :b), DynamicsKit.Dates.DateTime("2024-01-01T00:00:00.0")
    )

    @test_throws ArgumentError DynamicsKit._serialize_lyapunov_field_result_v2(r_bad)

    r_bad_status = LyapunovFieldResult(
        Float64[1.0, 2.0],
        Float64[0.1, 0.2],
        fill(0.0, 2, 2),
        fill(0, 1, 4),
        fill(0, 2, 2),
        fill(0, 2, 2),
        1e-3, "test", (:a, :b), DynamicsKit.Dates.DateTime("2024-01-01T00:00:00.0")
    )
    @test_throws ArgumentError DynamicsKit._serialize_lyapunov_field_result_v2(r_bad_status)
end

end  # @testset "Adaptive bifurcation-map refinement"
