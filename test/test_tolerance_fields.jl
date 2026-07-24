@testset "Tolerance fields (regime-boundary margins + tolerance maps)" begin
    using Dates: DateTime
    using Statistics: mean

    # --- helpers ------------------------------------------------------------------------------

    # Abramowitz & Stegun 7.1.26 error-function approximation (|error| < 1.5e-7); good enough to
    # serve as an independent closed-form reference for the Gaussian-boundary probability test.
    function _erf_approx(x::Float64)
        s = sign(x)
        z = abs(x)
        t = 1.0 / (1.0 + 0.3275911 * z)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * exp(-z * z)
        return s * y
    end
    _normal_cdf(z::Float64) = 0.5 * (1.0 + _erf_approx(z / sqrt(2.0)))

    # Brute-force nearest boundary-cell-centre distance at true coordinates — the ground truth the
    # generalized distance transform must reproduce exactly.
    function _brute_boundary_distance(a::Vector{Float64}, b::Vector{Float64},
                                      boundary_mask::AbstractMatrix{Bool})
        na, nb = length(a), length(b)
        D = fill(Inf, na, nb)
        sources = [(a[p], b[q]) for p in 1:na, q in 1:nb if boundary_mask[p, q]]
        for j in 1:nb, i in 1:na
            best = Inf
            for (ap, bq) in sources
                d2 = (a[i] - ap)^2 + (b[j] - bq)^2
                d2 < best && (best = d2)
            end
            D[i, j] = sqrt(best)
        end
        return D
    end

    vertical_labels(a, b; thresh) = Int[a[i] < thresh ? 1 : 2 for i in eachindex(a), j in eachindex(b)]

    # --- 1. vertical linear boundary on a uniform grid ----------------------------------------
    @testset "vertical linear boundary (uniform grid)" begin
        a = collect(0.0:1.0:10.0)   # 11 cols
        b = collect(0.0:1.0:6.0)    # 7 rows
        labels = vertical_labels(a, b; thresh=5.0)  # regime 1 for a<5 (i<=5), regime 2 for a>=5
        resolved = trues(length(a), length(b))
        res = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))

        # Boundary cells: columns adjacent to the interface (a=4 -> i=5, a=5 -> i=6).
        @test all(res.boundary_mask[5, :])
        @test all(res.boundary_mask[6, :])
        @test !any(res.boundary_mask[1:4, :])
        @test !any(res.boundary_mask[7:end, :])
        @test all(res.distance[5, :] .== 0.0) && all(res.distance[6, :] .== 0.0)

        # Distance is the physical distance to the nearest boundary-cell centre along a.
        @test res.distance[1, 1] ≈ 4.0
        @test res.distance[3, 4] ≈ 2.0
        # No boundary varies with b alone except on the two interface rows (which are all-boundary).
        @test all(isinf, res.distance_b[1:4, :])
        @test all(isinf, res.distance_b[7:end, :])
        @test all(res.distance_b[5, :] .== 0.0) && all(res.distance_b[6, :] .== 0.0)
        @test res.distance_a[1, 1] ≈ 4.0

        # The distance transform matches the brute-force nearest boundary-cell centre exactly.
        brute = _brute_boundary_distance(a, b, res.boundary_mask)
        @test all(isapprox.(res.distance, brute; atol=1e-10))
    end

    # --- 2. diagonal linear boundary on a uniform grid ----------------------------------------
    @testset "diagonal linear boundary (uniform grid)" begin
        a = collect(0.0:1.0:12.0)
        b = collect(0.0:1.0:12.0)
        labels = Int[(a[i] + b[j] < 12.0 ? 1 : 2) for i in eachindex(a), j in eachindex(b)]
        resolved = trues(length(a), length(b))
        res = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))
        brute = _brute_boundary_distance(a, b, res.boundary_mask)
        @test all(isapprox.(res.distance, brute; atol=1e-10))
        # Off-diagonal cells have finite margins on both axes (a boundary exists on every line).
        @test all(isfinite, res.distance_a)
        @test all(isfinite, res.distance_b)
    end

    # --- 3. circular boundary: discretization error <= one cell diagonal ----------------------
    @testset "circular boundary error <= one cell diagonal" begin
        h = 0.25
        a = collect(-3.0:h:3.0)
        b = collect(-3.0:h:3.0)
        radius = 1.7
        labels = Int[(a[i]^2 + b[j]^2 < radius^2 ? 1 : 2) for i in eachindex(a), j in eachindex(b)]
        resolved = trues(length(a), length(b))
        res = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))

        # Exactness of the DT versus brute force.
        brute = _brute_boundary_distance(a, b, res.boundary_mask)
        @test all(isapprox.(res.distance, brute; atol=1e-10))

        # The finite-grid margin is within one cell diagonal of the true radial distance to the
        # circle, away from the domain edge where the circle is fully sampled.
        cell_diag = sqrt(2.0) * h
        maxerr = 0.0
        for j in eachindex(b), i in eachindex(a)
            r = hypot(a[i], b[j])
            r > 2.6 && continue                 # ignore the sparsely-relevant far corners
            true_margin = abs(r - radius)
            maxerr = max(maxerr, abs(res.distance[i, j] - true_margin))
        end
        @test maxerr <= cell_diag + 1e-9
    end

    # --- 4. genuinely nonuniform grid vs brute force ------------------------------------------
    @testset "nonuniform grid vs brute-force nearest boundary" begin
        a = [0.0, 0.3, 0.7, 1.5, 2.0, 4.0, 4.1, 7.0]      # strictly increasing, nonuniform
        b = [-2.0, -1.9, -0.5, 0.0, 0.25, 3.0, 8.0]        # strictly increasing, nonuniform
        na, nb = length(a), length(b)
        labels = Int[(a[i] * 1.3 - b[j] < 1.0 ? 1 : 2) for i in 1:na, j in 1:nb]
        resolved = trues(na, nb)
        res = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))
        brute = _brute_boundary_distance(a, b, res.boundary_mask)
        @test all(isapprox.(res.distance, brute; atol=1e-10))

        # Per-axis distances also match brute force along their single grid line.
        for j in 1:nb, i in 1:na
            expected_a = minimum([abs(a[i] - a[p]) for p in 1:na if res.boundary_mask[p, j]]; init=Inf)
            expected_b = minimum([abs(b[j] - b[q]) for q in 1:nb if res.boundary_mask[i, q]]; init=Inf)
            @test isapprox(res.distance_a[i, j], expected_a; atol=1e-10) ||
                  (isinf(expected_a) && isinf(res.distance_a[i, j]))
            @test isapprox(res.distance_b[i, j], expected_b; atol=1e-10) ||
                  (isinf(expected_b) && isinf(res.distance_b[i, j]))
        end
    end

    # --- 5. per-axis Inf where a grid line has no boundary ------------------------------------
    @testset "per-axis Inf on boundary-free lines" begin
        a = collect(0.0:1.0:5.0)
        b = collect(0.0:1.0:5.0)
        # Horizontal split: regime changes only with b -> a-lines carry the boundary, b-lines do not
        labels = Int[(b[j] < 3.0 ? 1 : 2) for i in eachindex(a), j in eachindex(b)]
        resolved = trues(length(a), length(b))
        res = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))
        # a-lines with no boundary (columns off the interface) carry Inf; interface columns carry 0.
        @test all(isinf, res.distance_a[:, 1:2])
        @test all(isinf, res.distance_a[:, 5:6])
        @test all(res.distance_a[:, 3] .== 0.0) && all(res.distance_a[:, 4] .== 0.0)
        @test all(isfinite, res.distance_b)   # every b-line crosses the boundary
    end

    # --- 6. conservative unknown masks --------------------------------------------------------
    @testset "conservative unknown masks" begin
        a = collect(0.0:1.0:5.0)
        b = collect(0.0:1.0:5.0)
        na, nb = length(a), length(b)
        labels = fill(1, na, nb)
        resolved = trues(na, nb)
        resolved[3, 3] = false          # a single unknown cell
        labels[3, 3] = 0
        res = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))

        # The unknown cell is invalid, carries no margin, and is never a physical regime.
        @test res.valid[3, 3] == false
        @test isnan(res.distance[3, 3])
        @test isnan(res.distance_a[3, 3]) && isnan(res.distance_b[3, 3])
        @test res.resolved[3, 3] == false

        # Its known 4-neighbours are unknown-adjacent boundary cells (kind 2) with margin 0.
        for (i, j) in ((2, 3), (4, 3), (3, 2), (3, 4))
            @test res.boundary_mask[i, j]
            @test res.boundary_kind[i, j] == 2
            @test res.distance[i, j] == 0.0
        end
        # A distant same-regime cell has a finite margin to the boundary ring around the unknown.
        brute = _brute_boundary_distance(a, b, res.boundary_mask)
        @test isfinite(res.distance[1, 1])
        @test res.distance[1, 1] ≈ brute[1, 1]
    end

    # --- 7. all domain edge policies / censor flags -------------------------------------------
    @testset "edge policies and censoring" begin
        a = collect(0.0:1.0:10.0)
        b = collect(0.0:1.0:10.0)
        labels = vertical_labels(a, b; thresh=5.0)
        resolved = trues(length(a), length(b))

        ignore = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))
        censored = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:censored))
        boundary = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:boundary))

        # Cell (1,1): raw boundary distance 4, but the domain edge is 0 away (a=0, b=0 corner).
        @test ignore.distance[1, 1] ≈ 4.0
        @test ignore.edge_censored[1, 1] == false
        @test censored.distance[1, 1] ≈ 0.0            # capped at the edge
        @test censored.edge_censored[1, 1] == true     # lower bound
        @test boundary.distance[1, 1] ≈ 0.0            # capped, but the edge is a real boundary
        @test boundary.edge_censored[1, 1] == false
        @test !any(boundary.edge_censored)             # :boundary never flags censoring

        # A central cell farther from the edge than from the interface is uncensored.
        @test censored.distance[3, 6] ≈ ignore.distance[3, 6]
        @test censored.edge_censored[3, 6] == false
    end

    # --- 8. status-code distinction and missing-status semantics ------------------------------
    @testset "status-code distinction & missing-status semantics" begin
        periodicity = [1 0; 2 2]
        bmr = BifurcationMapResult([0.0, 1.0], [0.0, 1.0], periodicity, 8, "Sys",
            (:mu, :nu), DateTime(2026, 1, 1))

        # Periodicity-only (no status evidence): period 0 -> unknown, never a physical regime.
        res_plain = regime_boundary_distances(bmr; config=RegimeBoundaryConfig())
        @test res_plain.status_evidence == false
        @test res_plain.resolved == [true false; true true]
        @test res_plain.labels[1, 2] == 0

        # With status codes distinguishing an aperiodic cell.
        pc = DynamicsKit._map_status_code(:periodic)
        ac = DynamicsKit._map_status_code(:aperiodic_or_high_period)
        dc = DynamicsKit._map_status_code(:diverged)
        cells = DynamicsKit.MapCellGrid(2, 2)
        cells.periodicity .= periodicity
        cells.status_codes .= [pc ac; pc pc]

        res_ap = regime_boundary_distances(bmr; cells=cells,
            config=RegimeBoundaryConfig(aperiodic_is_regime=true))
        @test res_ap.status_evidence == true
        @test res_ap.resolved[1, 2] == true
        @test res_ap.labels[1, 2] == DynamicsKit._REGIME_APERIODIC

        res_ap_off = regime_boundary_distances(bmr; cells=cells,
            config=RegimeBoundaryConfig(aperiodic_is_regime=false))
        @test res_ap_off.resolved[1, 2] == false      # aperiodic demoted to unknown

        # Diverged distinction via a raw status_codes matrix.
        status = [pc pc; dc pc]
        periodicity2 = [1 1; 0 1]
        bmr2 = BifurcationMapResult([0.0, 1.0], [0.0, 1.0], periodicity2, 8, "Sys",
            (:mu, :nu), DateTime(2026, 1, 1))
        res_dv = regime_boundary_distances(bmr2; status_codes=status,
            config=RegimeBoundaryConfig(diverged_is_regime=true))
        @test res_dv.labels[2, 1] == DynamicsKit._REGIME_DIVERGED
        @test res_dv.resolved[2, 1] == true

        # Provenance / shape / exclusivity validation.
        bad_cells = DynamicsKit.MapCellGrid(2, 2)
        bad_cells.periodicity .= [9 9; 9 9]
        @test_throws ArgumentError regime_boundary_distances(bmr; cells=bad_cells)
        @test_throws ArgumentError regime_boundary_distances(bmr; cells=cells, status_codes=status)
        @test_throws ArgumentError regime_boundary_distances(bmr; status_codes=fill(pc, 3, 3))
    end

    # --- 9. exact zero-tolerance collapse -----------------------------------------------------
    @testset "exact zero-tolerance collapse" begin
        a = collect(0.0:1.0:6.0)
        b = collect(0.0:1.0:6.0)
        labels = vertical_labels(a, b; thresh=3.0)
        resolved = trues(length(a), length(b))
        resolved[2, 2] = false          # one unknown cell to exercise the unresolved branch
        labels[2, 2] = 0
        cfg = ToleranceConfig(tolerance_a=UniformTolerance(0.0), tolerance_b=GaussianTolerance(0.0),
            n_samples=500)
        tr = tolerance_regime_map(a, b, labels, resolved, cfg)

        @test tr.n_effective == 0
        @test all(tr.nominal_probability .== 1.0)
        @test all(tr.entropy .== 0.0)
        @test all(tr.nominal_standard_error .== 0.0)
        @test all(tr.nominal_ci_lower .== 1.0) && all(tr.nominal_ci_upper .== 1.0)
        # Resolved cell keeps its regime with probability 1; the unknown cell has unknown-mass 1.
        @test tr.regime_probability[1][1, 1] == 1.0
        @test tr.unknown_probability[2, 2] == 1.0
        @test tr.dominant_probability[2, 2] == 0.0
    end

    # --- 10. one-axis-zero distributions ------------------------------------------------------
    @testset "one-axis-zero distributions" begin
        a = collect(0.0:1.0:10.0)
        b = collect(0.0:1.0:10.0)
        labels = vertical_labels(a, b; thresh=5.0)
        resolved = trues(length(a), length(b))
        # Only a is perturbed; b is an exact Dirac delta so no b-driven transitions occur.
        cfg = ToleranceConfig(tolerance_a=GaussianTolerance(1.2), tolerance_b=UniformTolerance(0.0),
            n_samples=4000, threaded=false, seed=UInt64(7))
        tr = tolerance_regime_map(a, b, labels, resolved, cfg)
        @test tr.n_effective == 4000
        # A cell adjacent to the interface has a non-degenerate mix of the two regimes.
        p1 = tr.regime_probability[1][5, 3]
        @test 0.0 < p1 < 1.0
        # With b fixed the true probability is b-independent; MC estimates agree within noise
        # (per-cell seeds differ by (i, j), so they are close, not bitwise identical).
        @test all(abs.(tr.nominal_probability[:, 1] .- tr.nominal_probability[:, 5]) .< 0.05)
        @test all(tr.unknown_probability .== 0.0)
    end

    # --- 11. Gaussian linear boundary vs closed-form normal CDF -------------------------------
    @testset "Gaussian boundary probability vs normal CDF" begin
        a = collect(0.0:1.0:20.0)
        b = collect(0.0:1.0:5.0)
        # regime 1 for a<=10 (i<=11), regime 2 for a>=11 (i>=12); decision midpoint at a=10.5.
        labels = Int[(a[i] <= 10.0 ? 1 : 2) for i in eachindex(a), j in eachindex(b)]
        resolved = trues(length(a), length(b))
        sigma = 1.5
        cfg = ToleranceConfig(tolerance_a=GaussianTolerance(sigma), tolerance_b=UniformTolerance(0.0),
            n_samples=40000, threaded=false, seed=UInt64(2024))
        tr = tolerance_regime_map(a, b, labels, resolved, cfg)

        i0 = 9          # a = 8, regime 1
        d = 10.5 - a[i0]                # distance to the nearest-cell decision boundary
        expected = _normal_cdf(d / sigma)
        got = tr.nominal_probability[i0, 1]
        se = tr.nominal_standard_error[i0, 1]
        @test se > 0.0
        @test abs(got - expected) <= 4 * se + 1e-3
        @test tr.out_of_domain_probability[i0, 1] < 1e-3
    end

    # --- 12. out-of-domain mass is retained (not renormalized) --------------------------------
    @testset "out-of-domain mass not renormalized" begin
        a = collect(0.0:1.0:5.0)
        b = collect(0.0:1.0:5.0)
        labels = fill(1, length(a), length(b))
        resolved = trues(length(a), length(b))
        # Corner cell + wide tolerance -> a large fraction of samples leave the sampled domain.
        cfg = ToleranceConfig(tolerance_a=UniformTolerance(4.0), tolerance_b=UniformTolerance(4.0),
            n_samples=6000, threaded=false, seed=UInt64(99))
        tr = tolerance_regime_map(a, b, labels, resolved, cfg)
        @test tr.out_of_domain_probability[1, 1] > 0.3
        # The single regime's probability is NOT inflated to 1; OOD mass is kept separate.
        @test tr.regime_probability[1][1, 1] < 1.0
        @test tr.regime_probability[1][1, 1] + tr.out_of_domain_probability[1, 1] ≈ 1.0 atol = 1e-12
    end

    # --- 13. probabilities + unknown + OOD partition 1 ----------------------------------------
    @testset "categorical partition of unity" begin
        a = collect(0.0:0.5:5.0)
        b = collect(0.0:0.5:5.0)
        na, nb = length(a), length(b)
        labels = Int[(a[i] + b[j] < 5.0 ? 1 : (a[i] < 3.0 ? 2 : 3)) for i in 1:na, j in 1:nb]
        resolved = trues(na, nb)
        resolved[end, end] = false
        labels[end, end] = 0
        cfg = ToleranceConfig(tolerance_a=GaussianTolerance(0.7), tolerance_b=GaussianTolerance(0.7),
            n_samples=1500, threaded=true, seed=UInt64(4))
        tr = tolerance_regime_map(a, b, labels, resolved, cfg)
        for j in 1:nb, i in 1:na
            s = tr.unknown_probability[i, j] + tr.out_of_domain_probability[i, j]
            for l in tr.regime_labels
                s += tr.regime_probability[l][i, j]
            end
            @test s ≈ 1.0 atol = 1e-12
        end
        # Entropy is non-negative and bounded by log2(#categories).
        ncat = length(tr.regime_labels) + 2
        @test all(tr.entropy .>= -1e-12)
        @test all(tr.entropy .<= log2(ncat) + 1e-9)
    end

    # --- 14. Wilson intervals -----------------------------------------------------------------
    @testset "Wilson score intervals" begin
        a = collect(0.0:1.0:10.0)
        b = collect(0.0:1.0:2.0)
        labels = vertical_labels(a, b; thresh=5.0)
        resolved = trues(length(a), length(b))
        cfg = ToleranceConfig(tolerance_a=GaussianTolerance(1.5), tolerance_b=UniformTolerance(0.0),
            n_samples=3000, threaded=false, seed=UInt64(11))
        tr = tolerance_regime_map(a, b, labels, resolved, cfg)
        n = tr.n_effective
        z = 1.959963984540054
        for j in axes(tr.nominal_probability, 2), i in axes(tr.nominal_probability, 1)
            p = tr.nominal_probability[i, j]
            lo, hi = tr.nominal_ci_lower[i, j], tr.nominal_ci_upper[i, j]
            @test 0.0 <= lo <= hi <= 1.0
            @test lo <= p + 1e-9 && p <= hi + 1e-9
            # Reproduce the closed-form Wilson centre for one interior cell.
            denom = 1.0 + z^2 / n
            center = (p + z^2 / (2n)) / denom
            @test lo <= center + 1e-9 && center <= hi + 1e-9
        end
    end

    # --- 15. deterministic reruns + bitwise thread parity -------------------------------------
    @testset "determinism & thread parity" begin
        a = collect(0.0:0.5:8.0)
        b = collect(0.0:0.5:8.0)
        na, nb = length(a), length(b)
        labels = Int[(a[i] + 0.5 * b[j] < 6.0 ? 1 : 2) for i in 1:na, j in 1:nb]
        resolved = trues(na, nb)
        base = (tolerance_a=GaussianTolerance(0.6), tolerance_b=UniformTolerance(0.4),
                n_samples=800, seed=UInt64(0xABCDEF))
        seq1 = tolerance_regime_map(a, b, labels, resolved, ToleranceConfig(; base..., threaded=false))
        seq2 = tolerance_regime_map(a, b, labels, resolved, ToleranceConfig(; base..., threaded=false))
        par  = tolerance_regime_map(a, b, labels, resolved, ToleranceConfig(; base..., threaded=true))

        @test seq1.nominal_probability == seq2.nominal_probability   # bitwise deterministic rerun
        @test seq1.nominal_probability == par.nominal_probability    # bitwise thread parity
        @test seq1.entropy == par.entropy
        for l in seq1.regime_labels
            @test seq1.regime_probability[l] == par.regime_probability[l]
        end
        @test seq1.out_of_domain_probability == par.out_of_domain_probability
    end

    # --- 16. config / input validation --------------------------------------------------------
    @testset "config & input validation" begin
        @test_throws AssertionError RegimeBoundaryConfig(edge_policy=:nonsense)
        @test_throws AssertionError ToleranceConfig(n_samples=0)
        @test_throws ArgumentError UniformTolerance(-1.0)
        @test_throws ArgumentError UniformTolerance(Inf)
        @test_throws ArgumentError GaussianTolerance(-0.5)
        @test_throws ArgumentError GaussianTolerance(NaN)

        a = collect(0.0:1.0:3.0)
        b = collect(0.0:1.0:3.0)
        labels = fill(1, 4, 4)
        resolved = trues(4, 4)
        # Non-monotone grid.
        @test_throws ArgumentError regime_boundary_distances([0.0, 1.0, 1.0, 2.0], b, labels, resolved)
        # Shape mismatch.
        @test_throws ArgumentError regime_boundary_distances(a, b, fill(1, 3, 4), resolved)
        @test_throws ArgumentError tolerance_regime_map(a, b, fill(1, 4, 3), resolved, ToleranceConfig())
        # Non-finite coordinate.
        @test_throws ArgumentError regime_boundary_distances([0.0, 1.0, 2.0, Inf], b, labels, resolved)
    end

    # --- 17. serialization roundtrips ---------------------------------------------------------
    @testset "serialization roundtrip" begin
        a = collect(0.0:1.0:6.0)
        b = collect(0.0:1.0:4.0)
        na, nb = length(a), length(b)
        labels = vertical_labels(a, b; thresh=3.0)
        resolved = trues(na, nb)
        resolved[2, 2] = false
        labels[2, 2] = 0
        rb = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:censored), system_name="Sys", param_names=(:mu, :nu))

        d = serialize_regime_boundary_result(rb)
        @test d["format"] == "regime-boundary-v1"
        rb2 = deserialize_regime_boundary_result(d)
        @test rb2.system_name == "Sys"
        @test rb2.param_names == (:mu, :nu)
        @test rb2.edge_policy == :censored
        @test rb2.a_grid == rb.a_grid && rb2.b_grid == rb.b_grid
        @test rb2.labels == rb.labels
        @test rb2.resolved == rb.resolved
        @test rb2.boundary_kind == rb.boundary_kind
        @test isequal(rb2.distance, rb.distance)       # NaN-aware
        @test isequal(rb2.distance_a, rb.distance_a)   # Inf and NaN preserved distinctly
        @test isequal(rb2.distance_b, rb.distance_b)
        @test rb2.edge_censored == rb.edge_censored
        @test_throws ErrorException deserialize_regime_boundary_result(
            Dict{String, Any}("format" => "regime-boundary-v0"))

        cfg = ToleranceConfig(tolerance_a=GaussianTolerance(0.8), tolerance_b=UniformTolerance(0.5),
            n_samples=600, threaded=false, seed=UInt64(321))
        tr = tolerance_regime_map(a, b, labels, resolved, cfg; system_name="Sys", param_names=(:mu, :nu))
        t = serialize_tolerance_map_result(tr)
        @test t["format"] == "tolerance-map-v1"
        tr2 = deserialize_tolerance_map_result(t)
        @test tr2.system_name == "Sys" && tr2.param_names == (:mu, :nu)
        @test tr2.regime_labels == tr.regime_labels
        @test isequal(tr2.regime_probability, tr.regime_probability)  # Dict{Int,Matrix} roundtrip
        @test tr2.nominal_regime == tr.nominal_regime
        @test tr2.nominal_resolved == tr.nominal_resolved
        @test isequal(tr2.nominal_probability, tr.nominal_probability)
        @test isequal(tr2.entropy, tr.entropy)
        @test isequal(tr2.out_of_domain_probability, tr.out_of_domain_probability)
        @test tr2.n_samples == tr.n_samples && tr2.n_effective == tr.n_effective
        @test tr2.seed == tr.seed
        @test tr2.tolerance_a isa GaussianTolerance && tr2.tolerance_a.std ≈ 0.8
        @test tr2.tolerance_b isa UniformTolerance && tr2.tolerance_b.half_width ≈ 0.5
        @test_throws ErrorException deserialize_tolerance_map_result(
            Dict{String, Any}("format" => "tolerance-map-v0"))
    end

    # --- 18. accessors ------------------------------------------------------------------------
    @testset "summary accessors" begin
        a = collect(0.0:1.0:6.0)
        b = collect(0.0:1.0:6.0)
        labels = vertical_labels(a, b; thresh=3.0)
        resolved = trues(length(a), length(b))
        rb = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))
        s = regime_boundary_summary(rb)
        @test s.n_cells == length(a) * length(b)
        @test s.n_valid == s.n_cells
        @test s.n_boundary > 0
        @test s.min_margin == 0.0

        cfg = ToleranceConfig(tolerance_a=GaussianTolerance(0.5), tolerance_b=UniformTolerance(0.0),
            n_samples=500, threaded=false, seed=UInt64(5))
        tr = tolerance_regime_map(a, b, labels, resolved, cfg)
        ts = tolerance_regime_summary(tr)
        @test ts.n_cells == s.n_cells
        @test ts.n_regimes == 2
        @test ts.n_effective == 500
        @test 0.0 <= ts.mean_nominal_probability <= 1.0
    end

    # --- 19. moderate-grid performance (loose bound, correctness at scale) ---------------------
    @testset "moderate-grid performance" begin
        n = 120
        a = collect(range(0.0, 12.0, length=n))
        b = collect(range(0.0, 12.0, length=n))
        labels = Int[(a[i] + b[j] < 12.0 ? 1 : 2) for i in 1:n, j in 1:n]
        resolved = trues(n, n)

        t_rb = @elapsed rb = regime_boundary_distances(a, b, labels, resolved;
            config=RegimeBoundaryConfig(edge_policy=:ignore))
        brute = _brute_boundary_distance(a, b, rb.boundary_mask)
        @test all(isapprox.(rb.distance, brute; atol=1e-9))

        cfg = ToleranceConfig(tolerance_a=GaussianTolerance(0.4), tolerance_b=GaussianTolerance(0.4),
            n_samples=200, threaded=true, seed=UInt64(1))
        t_tr = @elapsed tr = tolerance_regime_map(a, b, labels, resolved, cfg)
        # Partition of unity still holds at scale.
        ok = true
        for j in 1:n, i in 1:n
            s = tr.unknown_probability[i, j] + tr.out_of_domain_probability[i, j]
            for l in tr.regime_labels
                s += tr.regime_probability[l][i, j]
            end
            abs(s - 1.0) > 1e-12 && (ok = false; break)
        end
        @test ok
        @info "tolerance-fields moderate-grid timing" grid = (n, n) regime_seconds = t_rb tolerance_seconds = t_tr
    end
end
