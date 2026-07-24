# Adaptive bifurcation-map refinement via dyadic (quadtree) subdivision.
#
# Integer-lattice design: at max subdivision depth D, each coarse axis step spans
# 2^D lattice units.  Every dyadic rational subdivision lands on an integer lattice
# coordinate, so shared edge midpoints between adjacent cells have identical integer
# keys and deduplication is exact with no floating-point comparisons.
#
# Refinement priority: corner/status/confidence-triggered cells are processed
# first.  Cells not activated by those triggers are subsequently classified and,
# when their corner classifications are uniform, center-screened in deterministic
# source order (column-major for coarse cells, SW/SE/NW/NE for child cells).
# If budget runs out (or max depth is reached) before center-screening completes,
# unscreened cells are recorded as uninspected in the result.

# ─── Integer-lattice helpers ────────────────────────────────────────────────────

@inline function _adaptive_lattice_scale(max_depth::Int)
    (0 <= max_depth <= _ADAPTIVE_MAP_MAX_DEPTH) || throw(ArgumentError(
        "Adaptive map max depth $max_depth is out of range [0, $(_ADAPTIVE_MAP_MAX_DEPTH)] for lattice scaling."))
    return 1 << max_depth   # 2^max_depth
end

@inline function _adaptive_checked_mul(lhs::Int, rhs::Int, context::AbstractString)
    try
        return Base.checked_mul(lhs, rhs)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "Adaptive map integer overflow while computing $context: $lhs * $rhs exceeds Int range. " *
            "Reduce max_depth (<= $(_ADAPTIVE_MAP_MAX_DEPTH)) or coarse grid resolution."))
    end
end

function _adaptive_axis_lattice_coords(step_count::Int, scale::Int, axis_name::AbstractString)
    coords = Vector{Int}(undef, step_count + 1)
    @inbounds for idx in 0:step_count
        coords[idx + 1] = _adaptive_checked_mul(idx, scale, "$axis_name lattice coordinate")
    end
    return coords
end

@inline _adaptive_class_key(status_code::Int, period::Int) = (status_code, period)
@inline _adaptive_sample_key(s::AdaptiveMapSample) = _adaptive_class_key(s.status_code, s.period)

# ─── Refinement trigger ─────────────────────────────────────────────────────────

function _adaptive_cell_trigger(sw::AdaptiveMapSample, se::AdaptiveMapSample,
                                 nw::AdaptiveMapSample, ne::AdaptiveMapSample,
                                 adaptive::AdaptiveMapConfig)
    reasons = Symbol[]

    if adaptive.refine_on_period_disagreement
        p1 = sw.period; p2 = se.period; p3 = nw.period; p4 = ne.period
        if p1 != p2 || p1 != p3 || p1 != p4
            push!(reasons, :period_disagreement)
        end
    end

    if adaptive.refine_on_status_disagreement
        s1 = sw.status_code; s2 = se.status_code; s3 = nw.status_code; s4 = ne.status_code
        if s1 != s2 || s1 != s3 || s1 != s4
            push!(reasons, :status_disagreement)
        end
    end

    if adaptive.min_confidence > 0.0
        min_c = min(sw.confidence, se.confidence, nw.confidence, ne.confidence)
        if min_c < adaptive.min_confidence
            push!(reasons, :low_confidence)
        end
    end

    if adaptive.confidence_delta > 0.0
        min_c = min(sw.confidence, se.confidence, nw.confidence, ne.confidence)
        max_c = max(sw.confidence, se.confidence, nw.confidence, ne.confidence)
        if max_c - min_c >= adaptive.confidence_delta
            push!(reasons, :confidence_delta)
        end
    end

    return (!isempty(reasons), reasons)
end

# ─── Sample evaluation ──────────────────────────────────────────────────────────

function _adaptive_eval_or_lookup!(samples::Vector{AdaptiveMapSample},
                                    lookup::Dict{Tuple{Int,Int},Int},
                                    budget_remaining::Ref{Int},
                                    ia::Int, ib::Int, depth::Int,
                                    a_phys::Float64, b_phys::Float64,
                                    detection_fn)
    key = (ia, ib)
    idx = get(lookup, key, 0)
    idx > 0 && return idx     # already evaluated — free deduplication

    budget_remaining[] <= 0 && return 0

    det = detection_fn(a_phys, b_phys)
    status_code = _map_status_code(det.status)
    s = AdaptiveMapSample(a_phys, b_phys, det.period, status_code,
                          det.closure_confidence, depth)
    push!(samples, s)
    idx = length(samples)
    lookup[key] = idx
    budget_remaining[] -= 1
    return idx
end

# ─── Boundary-segment extraction (categorical marching-squares) ─────────────────
#
# Each leaf cell is processed independently.  Segment geometry:
#
#   Corner layout:  NW=(ia0,ib1)  NE=(ia1,ib1)
#                   SW=(ia0,ib0)  SE=(ia1,ib0)
#
# Edge midpoints (integer lattice):
#   S = (ia_mid, ib0),  N = (ia_mid, ib1)
#   W = (ia0, ib_mid),  E = (ia1, ib_mid)
#   C = (ia_mid, ib_mid)  ← physical cell center
#
# Connectivity rules by crossing count and distinct-key count:
#   n_cross == 0              → no segments (uniform).
#   n_cross == 2, 2 keys      → connect the two crossing midpoints (:resolved).
#   n_cross == 4, 2 keys      → checkerboard: use center key to choose the two
#                               correct pairings (:resolved); if center absent,
#                               connect all four midpoints to C (:ambiguous).
#   otherwise (3+ keys, or
#     1/3 crossings)          → connect each crossing midpoint to C (:multi_region).
#
# Period values are not interpolated; key pairs identify the regime boundary.

struct _AdaptiveCrossing
    ia::Int; ib::Int
    a::Float64; b::Float64
    key_left::Tuple{Int,Int}
    key_right::Tuple{Int,Int}
end

function _emit_segment!(segments::Vector{AdaptiveMapSegment},
                        seg_set::Set{NTuple{8,Int}},
                        ia_p::Int, ib_p::Int, ia_q::Int, ib_q::Int,
                        a_p::Float64, b_p::Float64, a_q::Float64, b_q::Float64,
                        key_c1::Tuple{Int,Int}, key_c2::Tuple{Int,Int},
                        ambiguity::Symbol)
    ka, kb = key_c1 <= key_c2 ? (key_c1, key_c2) : (key_c2, key_c1)
    # Canonical endpoint order: smaller lattice coordinate first.
    if (ia_p, ib_p) > (ia_q, ib_q)
        ia_p, ib_p, ia_q, ib_q = ia_q, ib_q, ia_p, ib_p
        a_p, b_p, a_q, b_q = a_q, b_q, a_p, b_p
    end
    sig = (ia_p, ib_p, ia_q, ib_q, ka[1], ka[2], kb[1], kb[2])
    sig in seg_set && return
    push!(seg_set, sig)
    push!(segments, AdaptiveMapSegment(a_p, b_p, a_q, b_q, ka, kb, ambiguity))
end

function _adaptive_cell_segments!(segments::Vector{AdaptiveMapSegment},
                                   seg_set::Set{NTuple{8,Int}},
                                   samples::Vector{AdaptiveMapSample},
                                   cell::AdaptiveMapLeafCell)
    k_sw = _adaptive_sample_key(samples[cell.si_sw])
    k_se = _adaptive_sample_key(samples[cell.si_se])
    k_nw = _adaptive_sample_key(samples[cell.si_nw])
    k_ne = _adaptive_sample_key(samples[cell.si_ne])

    cross_s = k_sw != k_se
    cross_n = k_nw != k_ne
    cross_w = k_sw != k_nw
    cross_e = k_se != k_ne
    n_cross = Int(cross_s) + Int(cross_n) + Int(cross_w) + Int(cross_e)
    n_cross == 0 && return

    ia0 = cell.ia0; ia1 = cell.ia1; ib0 = cell.ib0; ib1 = cell.ib1
    ia_mid = ia0 + ((ia1 - ia0) ÷ 2); ib_mid = ib0 + ((ib1 - ib0) ÷ 2)
    a0 = cell.a0; a1 = cell.a1; b0 = cell.b0; b1 = cell.b1
    a_mid = (a0 + a1) * 0.5; b_mid = (b0 + b1) * 0.5

    crossings = _AdaptiveCrossing[]
    cross_s && push!(crossings, _AdaptiveCrossing(ia_mid, ib0, a_mid, b0, k_sw, k_se))
    cross_n && push!(crossings, _AdaptiveCrossing(ia_mid, ib1, a_mid, b1, k_nw, k_ne))
    cross_w && push!(crossings, _AdaptiveCrossing(ia0, ib_mid, a0, b_mid, k_sw, k_nw))
    cross_e && push!(crossings, _AdaptiveCrossing(ia1, ib_mid, a1, b_mid, k_se, k_ne))

    # Distinct corner keys
    all_keys = (k_sw, k_se, k_nw, k_ne)
    n_distinct = length(unique(all_keys))

    if n_distinct == 2 && n_cross == 2
        # Standard two-key two-crossing case: connect the two crossing midpoints.
        c1, c2 = crossings[1], crossings[2]
        k1 = c1.key_left <= c1.key_right ? c1.key_left : c1.key_right
        k2 = c1.key_left <= c1.key_right ? c1.key_right : c1.key_left
        _emit_segment!(segments, seg_set,
                       c1.ia, c1.ib, c2.ia, c2.ib,
                       c1.a, c1.b, c2.a, c2.b,
                       k1, k2, :resolved)
        return
    end

    if n_distinct == 2 && n_cross == 4
        # Checkerboard: SW/NE share one key, SE/NW share the other.
        # Use center key to determine which diagonal pair is "dominant".
        k1, k2 = k_sw, k_se   # k_sw == k_ne, k_se == k_nw by checkerboard structure
        if cell.si_center > 0
            k_ctr = _adaptive_sample_key(samples[cell.si_center])
            # Find crossing midpoints by their lattice coordinates.
            c_s  = findfirst(c -> c.ib == ib0, crossings)
            c_n  = findfirst(c -> c.ib == ib1, crossings)
            c_w  = findfirst(c -> c.ia == ia0, crossings)
            c_e  = findfirst(c -> c.ia == ia1, crossings)
            if !isnothing(c_s) && !isnothing(c_n) && !isnothing(c_w) && !isnothing(c_e)
                if k_ctr == k_sw
                    # SW/NE dominant: S connects to W, N connects to E.
                    s = crossings[c_s]; w = crossings[c_w]
                    n = crossings[c_n]; e = crossings[c_e]
                    _emit_segment!(segments, seg_set, s.ia, s.ib, w.ia, w.ib,
                                   s.a, s.b, w.a, w.b,
                                   k1, k2, :resolved)
                    _emit_segment!(segments, seg_set, n.ia, n.ib, e.ia, e.ib,
                                   n.a, n.b, e.a, e.b,
                                   k1, k2, :resolved)
                elseif k_ctr == k_se
                    # SE/NW dominant: S connects to E, N connects to W.
                    s = crossings[c_s]; e = crossings[c_e]
                    n = crossings[c_n]; w = crossings[c_w]
                    _emit_segment!(segments, seg_set, s.ia, s.ib, e.ia, e.ib,
                                   s.a, s.b, e.a, e.b,
                                   k1, k2, :resolved)
                    _emit_segment!(segments, seg_set, n.ia, n.ib, w.ia, w.ib,
                                   n.a, n.b, w.a, w.b,
                                   k1, k2, :resolved)
                else
                    # Third center key: a third regime is present; route through multi_region.
                    for c in crossings
                        _emit_segment!(segments, seg_set,
                                       c.ia, c.ib, ia_mid, ib_mid,
                                       c.a, c.b, a_mid, b_mid,
                                       c.key_left, c.key_right, :multi_region)
                    end
                end
                return
            end
        end
        # No center available: conservative spokes to cell center.
        for c in crossings
            _emit_segment!(segments, seg_set,
                           c.ia, c.ib, ia_mid, ib_mid,
                           c.a, c.b, a_mid, b_mid,
                           c.key_left, c.key_right, :ambiguous)
        end
        return
    end

    # Multi-region or odd crossing count: connect each crossing midpoint to cell center.
    for c in crossings
        _emit_segment!(segments, seg_set,
                       c.ia, c.ib, ia_mid, ib_mid,
                       c.a, c.b, a_mid, b_mid,
                       c.key_left, c.key_right, :multi_region)
    end
end

# ─── Core adaptive refinement ───────────────────────────────────────────────────

struct _AdaptiveWorkItem
    ia0::Int; ia1::Int; ib0::Int; ib1::Int
    a0::Float64; a1::Float64; b0::Float64; b1::Float64
    si_sw::Int; si_se::Int; si_nw::Int; si_ne::Int
    si_center::Int
    depth::Int
    reasons::Vector{Symbol}
end

@inline function _adaptive_work_item(ia0::Int, ia1::Int, ib0::Int, ib1::Int,
                                     a0::Float64, a1::Float64, b0::Float64, b1::Float64,
                                     si_sw::Int, si_se::Int, si_nw::Int, si_ne::Int,
                                     si_center::Int, depth::Int,
                                     reasons::Vector{Symbol})
    return _AdaptiveWorkItem(ia0, ia1, ib0, ib1, a0, a1, b0, b1,
                             si_sw, si_se, si_nw, si_ne, si_center, depth, reasons)
end

"""
Internal driver for adaptive quadtree refinement.  Called after the coarse grid
has already been evaluated and stored in `samples` / `lookup`.

`detection_fn(a, b)` returns a named-tuple with `.status::Symbol`, `.period::Int`,
`.closure_confidence::Float64`.
"""
function _run_adaptive_refinement(coarse_result::BifurcationMapResult,
                                   adaptive::AdaptiveMapConfig,
                                   samples::Vector{AdaptiveMapSample},
                                   lookup::Dict{Tuple{Int,Int},Int},
                                   budget_remaining::Ref{Int},
                                   detection_fn)
    max_depth = adaptive.max_depth
    scale = _adaptive_lattice_scale(max_depth)
    na = length(coarse_result.a_grid)
    nb = length(coarse_result.b_grid)
    a_lattice = _adaptive_axis_lattice_coords(na - 1, scale, "a")
    b_lattice = _adaptive_axis_lattice_coords(nb - 1, scale, "b")

    leaf_cells   = AdaptiveMapLeafCell[]
    max_depth_reached = Ref(0)
    split_cells  = Ref(0)
    flagged_cells = Ref(0)
    uninspected  = Ref(0)
    budget_limited = Ref(false)   # tracks budget_exhausted semantics

    # Build initial queues from coarse cells.
    # Triggered cells → refine_queue (priority).
    # Uniform cells   → center_queue (screened after triggered work drains).
    refine_queue = _AdaptiveWorkItem[]
    center_queue = _AdaptiveWorkItem[]

    for i in 1:(na - 1), j in 1:(nb - 1)
        ia0 = a_lattice[i]; ia1 = a_lattice[i + 1]
        ib0 = b_lattice[j]; ib1 = b_lattice[j + 1]
        si_sw = lookup[(ia0, ib0)]; si_se = lookup[(ia1, ib0)]
        si_nw = lookup[(ia0, ib1)]; si_ne = lookup[(ia1, ib1)]
        sw = samples[si_sw]; se = samples[si_se]
        nw = samples[si_nw]; ne = samples[si_ne]
        should_refine, reasons = _adaptive_cell_trigger(sw, se, nw, ne, adaptive)

        if should_refine && max_depth > 0
            flagged_cells[] += 1
            push!(refine_queue, _adaptive_work_item(
                ia0, ia1, ib0, ib1,
                coarse_result.a_grid[i], coarse_result.a_grid[i + 1],
                coarse_result.b_grid[j], coarse_result.b_grid[j + 1],
                si_sw, si_se, si_nw, si_ne, 0, 0, reasons))
        elseif should_refine
            # max_depth == 0: cannot refine further; boundary leaf at depth 0.
            flagged_cells[] += 1
            push!(leaf_cells, AdaptiveMapLeafCell(
                coarse_result.a_grid[i], coarse_result.a_grid[i + 1],
                coarse_result.b_grid[j], coarse_result.b_grid[j + 1],
                ia0, ia1, ib0, ib1, 0,
                si_sw, si_se, si_nw, si_ne, 0, :boundary, reasons))
        else
            push!(center_queue, _adaptive_work_item(
                ia0, ia1, ib0, ib1,
                coarse_result.a_grid[i], coarse_result.a_grid[i + 1],
                coarse_result.b_grid[j], coarse_result.b_grid[j + 1],
                si_sw, si_se, si_nw, si_ne, 0, 0, reasons))
        end
    end

    # Iterative refinement: drain refine_queue (priority), then center_queue.
    # Center-checked cells that disagree are promoted into the next refine_queue.
    while !isempty(refine_queue) || !isempty(center_queue)

        # ── Phase A: process all triggered (refine) cells ──────────────────────
        head = 1
        while head <= length(refine_queue)
            item = refine_queue[head]; head += 1
            ia0 = item.ia0; ia1 = item.ia1; ib0 = item.ib0; ib1 = item.ib1
            depth = item.depth
            si_sw = item.si_sw; si_se = item.si_se
            si_nw = item.si_nw; si_ne = item.si_ne

            if depth >= max_depth
                max_depth_reached[] = max(max_depth_reached[], depth)
                push!(leaf_cells, AdaptiveMapLeafCell(
                    item.a0, item.a1, item.b0, item.b1,
                    ia0, ia1, ib0, ib1, depth,
                    si_sw, si_se, si_nw, si_ne, item.si_center,
                    :boundary, item.reasons))
                continue
            end

            ia_mid = ia0 + ((ia1 - ia0) ÷ 2); ib_mid = ib0 + ((ib1 - ib0) ÷ 2)
            next_depth = depth + 1
            a0 = item.a0; a1 = item.a1; b0 = item.b0; b1 = item.b1
            a_mid = (a0 + a1) * 0.5; b_mid = (b0 + b1) * 0.5

            new_positions = (
                (ia_mid, ib0,   a_mid, b0   ),  # S midpoint
                (ia_mid, ib1,   a_mid, b1   ),  # N midpoint
                (ia0,    ib_mid, a0,   b_mid),  # W midpoint
                (ia1,    ib_mid, a1,   b_mid),  # E midpoint
                (ia_mid, ib_mid, a_mid, b_mid)  # center
            )

            n_new = sum(pos -> !haskey(lookup, (pos[1], pos[2])), new_positions)

            if budget_remaining[] < n_new
                # Cannot afford all required evaluations; record as budget-limited.
                budget_limited[] = true
                push!(leaf_cells, AdaptiveMapLeafCell(
                    a0, a1, b0, b1,
                    ia0, ia1, ib0, ib1, depth,
                    si_sw, si_se, si_nw, si_ne,
                    get(lookup, (ia_mid, ib_mid), 0),
                    :budget_limited, item.reasons))
                continue
            end

            evaluated = Vector{Int}(undef, 5)
            for (k, pos) in enumerate(new_positions)
                evaluated[k] = _adaptive_eval_or_lookup!(
                    samples, lookup, budget_remaining,
                    pos[1], pos[2], next_depth, Float64(pos[3]), Float64(pos[4]),
                    detection_fn)
            end

            if any(==(0), evaluated)
                # Budget exhausted mid-evaluation; mark budget-limited.
                budget_limited[] = true
                push!(leaf_cells, AdaptiveMapLeafCell(
                    a0, a1, b0, b1,
                    ia0, ia1, ib0, ib1, depth,
                    si_sw, si_se, si_nw, si_ne,
                    get(lookup, (ia_mid, ib_mid), 0),
                    :budget_limited, item.reasons))
                continue
            end

            split_cells[] += 1
            max_depth_reached[] = max(max_depth_reached[], next_depth)

            si_s = evaluated[1]; si_n = evaluated[2]
            si_w = evaluated[3]; si_e = evaluated[4]
            si_c = evaluated[5]

            children = (
                (ia0=ia0,    ia1=ia_mid, ib0=ib0,    ib1=ib_mid,
                 a0=a0,   a1=a_mid, b0=b0,   b1=b_mid,
                 si_sw=si_sw, si_se=si_s, si_nw=si_w, si_ne=si_c),
                (ia0=ia_mid, ia1=ia1,    ib0=ib0,    ib1=ib_mid,
                 a0=a_mid, a1=a1,   b0=b0,   b1=b_mid,
                 si_sw=si_s,  si_se=si_se, si_nw=si_c, si_ne=si_e),
                (ia0=ia0,    ia1=ia_mid, ib0=ib_mid, ib1=ib1,
                 a0=a0,   a1=a_mid, b0=b_mid, b1=b1,
                 si_sw=si_w,  si_se=si_c,  si_nw=si_nw, si_ne=si_n),
                (ia0=ia_mid, ia1=ia1,    ib0=ib_mid, ib1=ib1,
                 a0=a_mid, a1=a1,   b0=b_mid, b1=b1,
                 si_sw=si_c,  si_se=si_e,  si_nw=si_n,  si_ne=si_ne)
            )

            for child in children
                sw_s = samples[child.si_sw]; se_s = samples[child.si_se]
                nw_s = samples[child.si_nw]; ne_s = samples[child.si_ne]
                should_refine, child_reasons = _adaptive_cell_trigger(sw_s, se_s, nw_s, ne_s, adaptive)

                if should_refine && next_depth < max_depth
                    flagged_cells[] += 1
                    push!(refine_queue, _adaptive_work_item(
                        child.ia0, child.ia1, child.ib0, child.ib1,
                        child.a0, child.a1, child.b0, child.b1,
                        child.si_sw, child.si_se, child.si_nw, child.si_ne,
                        0, next_depth, child_reasons))
                elseif should_refine
                    # Boundary leaf at max_depth.
                    flagged_cells[] += 1
                    push!(leaf_cells, AdaptiveMapLeafCell(
                        child.a0, child.a1, child.b0, child.b1,
                        child.ia0, child.ia1, child.ib0, child.ib1, next_depth,
                        child.si_sw, child.si_se, child.si_nw, child.si_ne,
                        0, :boundary, child_reasons))
                else
                    # Uniform child: enqueue for center screening.
                    push!(center_queue, _adaptive_work_item(
                        child.ia0, child.ia1, child.ib0, child.ib1,
                        child.a0, child.a1, child.b0, child.b1,
                        child.si_sw, child.si_se, child.si_nw, child.si_ne,
                        0, next_depth, child_reasons))
                end
            end
        end
        empty!(refine_queue)

        # ── Phase B: center-screen uniform cells ───────────────────────────────
        new_refine = _AdaptiveWorkItem[]

        for item in center_queue
            depth = item.depth
            ia0 = item.ia0; ia1 = item.ia1; ib0 = item.ib0; ib1 = item.ib1
            corner_keys = (
                _adaptive_sample_key(samples[item.si_sw]),
                _adaptive_sample_key(samples[item.si_se]),
                _adaptive_sample_key(samples[item.si_nw]),
                _adaptive_sample_key(samples[item.si_ne]),
            )
            corners_uniform = all(==(corner_keys[1]), corner_keys)

            if !corners_uniform
                # Classification-disagreement triggers may be disabled. Preserve
                # the observed boundary without spending budget on island screening.
                push!(leaf_cells, AdaptiveMapLeafCell(
                    item.a0, item.a1, item.b0, item.b1,
                    ia0, ia1, ib0, ib1, depth,
                    item.si_sw, item.si_se, item.si_nw, item.si_ne,
                    0, :boundary, item.reasons))
                continue
            end

            ia_cc = ia0 + ((ia1 - ia0) ÷ 2); ib_cc = ib0 + ((ib1 - ib0) ÷ 2)
            a_cc  = (item.a0 + item.a1) * 0.5; b_cc = (item.b0 + item.b1) * 0.5
            has_distinct_center = ia0 < ia_cc < ia1 && ib0 < ib_cc < ib1

            if !has_distinct_center
                # No distinct integer-lattice center exists for this cell at the configured max depth.
                uninspected[] += 1
                push!(leaf_cells, AdaptiveMapLeafCell(
                    item.a0, item.a1, item.b0, item.b1,
                    ia0, ia1, ib0, ib1, depth,
                    item.si_sw, item.si_se, item.si_nw, item.si_ne,
                    0, :uninspected, Symbol[]))
                continue
            end

            # Free reuse: if the center key is already in the lookup, classify without
            # spending budget, regardless of budget_remaining or depth.
            si_existing = get(lookup, (ia_cc, ib_cc), 0)
            if si_existing > 0
                corner_key = corner_keys[1]
                center_key = _adaptive_sample_key(samples[si_existing])
                if center_key != corner_key
                    flagged_cells[] += 1
                    if depth < max_depth
                        push!(new_refine, _adaptive_work_item(
                            ia0, ia1, ib0, ib1,
                            item.a0, item.a1, item.b0, item.b1,
                            item.si_sw, item.si_se, item.si_nw, item.si_ne,
                            si_existing, depth, Symbol[:center_disagreement]))
                    else
                        push!(leaf_cells, AdaptiveMapLeafCell(
                            item.a0, item.a1, item.b0, item.b1,
                            ia0, ia1, ib0, ib1, depth,
                            item.si_sw, item.si_se, item.si_nw, item.si_ne,
                            si_existing, :boundary, Symbol[:center_disagreement]))
                    end
                else
                    push!(leaf_cells, AdaptiveMapLeafCell(
                        item.a0, item.a1, item.b0, item.b1,
                        ia0, ia1, ib0, ib1, depth,
                        item.si_sw, item.si_se, item.si_nw, item.si_ne,
                        si_existing, :interior, Symbol[]))
                end
                continue
            end

            if depth >= max_depth || budget_remaining[] <= 0
                # Cannot center-screen.
                uninspected[] += 1
                (depth < max_depth && budget_remaining[] <= 0) && (budget_limited[] = true)
                push!(leaf_cells, AdaptiveMapLeafCell(
                    item.a0, item.a1, item.b0, item.b1,
                    ia0, ia1, ib0, ib1, depth,
                    item.si_sw, item.si_se, item.si_nw, item.si_ne,
                    0, :uninspected, Symbol[]))
                continue
            end

            si_cc = _adaptive_eval_or_lookup!(
                samples, lookup, budget_remaining,
                ia_cc, ib_cc, depth + 1, a_cc, b_cc, detection_fn)

            if si_cc == 0
                # Budget ran out during center evaluation.
                uninspected[] += 1
                budget_limited[] = true
                push!(leaf_cells, AdaptiveMapLeafCell(
                    item.a0, item.a1, item.b0, item.b1,
                    ia0, ia1, ib0, ib1, depth,
                    item.si_sw, item.si_se, item.si_nw, item.si_ne,
                    0, :uninspected, Symbol[]))
                continue
            end

            corner_key = corner_keys[1]
            center_key = _adaptive_sample_key(samples[si_cc])

            if center_key != corner_key && depth < max_depth
                # Center disagrees → promote to refinement.
                flagged_cells[] += 1
                push!(new_refine, _adaptive_work_item(
                    ia0, ia1, ib0, ib1,
                    item.a0, item.a1, item.b0, item.b1,
                    item.si_sw, item.si_se, item.si_nw, item.si_ne,
                    si_cc, depth, Symbol[:center_disagreement]))
            else
                push!(leaf_cells, AdaptiveMapLeafCell(
                    item.a0, item.a1, item.b0, item.b1,
                    ia0, ia1, ib0, ib1, depth,
                    item.si_sw, item.si_se, item.si_nw, item.si_ne,
                    si_cc, :interior, Symbol[]))
            end
        end
        empty!(center_queue)

        # Promoted center-disagreement cells become the next refine_queue.
        append!(refine_queue, new_refine)
    end

    # ── Boundary segment extraction ─────────────────────────────────────────────
    segments = AdaptiveMapSegment[]
    # Deduplication key: (ia_p, ib_p, ia_q, ib_q, encoded_ka, encoded_kb)
    seg_set = Set{NTuple{8,Int}}()
    for cell in leaf_cells
        _adaptive_cell_segments!(segments, seg_set, samples, cell)
    end

    return leaf_cells, segments, max_depth_reached[], split_cells[],
           flagged_cells[], uninspected[], budget_limited[]
end

# ─── Detection helpers ──────────────────────────────────────────────────────────

# Single-parameter-point detection for the adaptive refinement loop.
# Returns a named-tuple with `.status`, `.period`, `.closure_confidence` — the
# same shape as `_detect_discrete_map_period` / `_detect_continuous_poincare_period`.
function _map_adaptive_detection(sys::DiscreteMap,
                                  config::BifurcationMapConfig,
                                  params::AbstractVector,
                                  initial_point,
                                  points_to_drop::Int,
                                  multistability_seeds;
                                  kwargs...)
    if !isempty(multistability_seeds)
        seed_results = [
            _detect_discrete_map_period(sys, params, seed, points_to_drop,
                                        config.max_period, config.precision,
                                        config.divergence_cutoff)
            for seed in multistability_seeds
        ]
        return _summarize_map_multistability(seed_results).selected
    end
    return _detect_discrete_map_period(sys, params, initial_point, points_to_drop,
                                       config.max_period, config.precision,
                                       config.divergence_cutoff)
end

function _map_adaptive_detection(sys::ContinuousODE,
                                  config::BifurcationMapConfig,
                                  params::AbstractVector,
                                  initial_point,
                                  points_to_drop::Int,
                                  multistability_seeds;
                                  solver=Tsit5(),
                                  reltol::Float64=1e-8,
                                  abstol::Float64=1e-8)
    if !isempty(multistability_seeds)
        seed_results = [
            _detect_continuous_poincare_period(
                sys, params;
                initial_point=seed,
                transient=points_to_drop,
                max_period=config.max_period,
                precision=config.precision,
                solver=solver, reltol=reltol, abstol=abstol,
                projected=true,
                divergence_cutoff=config.divergence_cutoff,
                min_crossing_time=config.min_crossing_time
            )
            for seed in multistability_seeds
        ]
        return _summarize_map_multistability(seed_results).selected
    end
    return _detect_continuous_poincare_period(
        sys, params;
        initial_point=initial_point,
        transient=points_to_drop,
        max_period=config.max_period,
        precision=config.precision,
        solver=solver, reltol=reltol, abstol=abstol,
        projected=true,
        divergence_cutoff=config.divergence_cutoff,
        min_crossing_time=config.min_crossing_time
    )
end

# ─── Detection adapters ─────────────────────────────────────────────────────────

function _adaptive_detection_fn(sys::DiscreteMap,
                                 coarse_config::BifurcationMapConfig,
                                 param_template::Vector{Float64},
                                 a_indices::Vector{Int},
                                 b_indices::Vector{Int},
                                 initial_point,
                                 points_to_drop::Int,
                                 multistability_seeds)
    param_buffer = copy(param_template)
    return function(a::Float64, b::Float64)
        params = map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a, b)
        return _map_adaptive_detection(sys, coarse_config, params, initial_point,
                                       points_to_drop, multistability_seeds)
    end
end

function _adaptive_detection_fn(sys::ContinuousODE,
                                 coarse_config::BifurcationMapConfig,
                                 param_template::Vector{Float64},
                                 a_indices::Vector{Int},
                                 b_indices::Vector{Int},
                                 initial_point,
                                 points_to_drop::Int,
                                 multistability_seeds;
                                 solver=Tsit5(),
                                 reltol::Float64=1e-8,
                                 abstol::Float64=1e-8)
    param_buffer = copy(param_template)
    return function(a::Float64, b::Float64)
        params = map_params_from_buffer!(param_buffer, param_template, a_indices, b_indices, a, b)
        return _map_adaptive_detection(sys, coarse_config, params, initial_point,
                                       points_to_drop, multistability_seeds;
                                       solver=solver, reltol=reltol, abstol=abstol)
    end
end

# ─── Public function ────────────────────────────────────────────────────────────

"""
    adaptive_bifurcation_map(sys::DynamicalSystem, coarse_config::BifurcationMapConfig,
                              adaptive::AdaptiveMapConfig;
                              initial_point=nothing,
                              cells=nothing,
                              backend=CPUBackend(),
                              solver=Tsit5(),
                              reltol=1e-8,
                              abstol=1e-8) -> AdaptiveMapResult

Generate an adaptive 2D bifurcation map by first running a uniform coarse sweep
(`bifurcation_map`) then applying dyadic (quadtree) subdivision to cells that
contain classification boundaries or low-confidence samples, subject to a strict
total evaluation budget.

`coarse_config` must have `reuse_neighbor_seeds=false`.  Traversal-dependent
neighbor modes produce thread-scheduling-dependent coarse results and are rejected.

`adaptive.total_budget` must be at least `(a_steps+1)*(b_steps+1)` — the number of
evaluations required for the coarse grid alone.  Any remaining budget is available
for refinement.

The optional `cells::MapCellGrid` hook follows the same contract as
`bifurcation_map(...; cells=)`.

The adaptive refinement samples are always evaluated on the CPU.
"""
function adaptive_bifurcation_map(sys::DiscreteMap,
                                    coarse_config::BifurcationMapConfig,
                                    adaptive::AdaptiveMapConfig;
                                    initial_point::Union{Nothing, AbstractVector}=nothing,
                                    cells::Union{Nothing, MapCellGrid}=nothing,
                                    backend::ComputeBackend=CPUBackend())
    coarse_config.reuse_neighbor_seeds && throw(ArgumentError(
        "adaptive_bifurcation_map requires fixed-seed coarse traversal " *
        "(reuse_neighbor_seeds=false on the BifurcationMapConfig). " *
        "Traversal-dependent neighbor modes make the result depend on thread scheduling."))

    na = coarse_config.a_steps + 1
    nb = coarse_config.b_steps + 1
    adaptive.total_budget >= na * nb || throw(ArgumentError(
        "AdaptiveMapConfig.total_budget ($(adaptive.total_budget)) must be at least " *
        "(a_steps+1)*(b_steps+1) = $(na * nb) to run the coarse grid."))

    coarse_result, coarse_diagnostics = _bifurcation_map(sys, coarse_config;
                                                          initial_point=initial_point,
                                                          cells=cells, backend=backend)
    _run_adaptive_from_coarse(sys, coarse_config, adaptive, coarse_result, coarse_diagnostics;
                               initial_point=initial_point, backend=backend)
end

function adaptive_bifurcation_map(sys::ContinuousODE,
                                    coarse_config::BifurcationMapConfig,
                                    adaptive::AdaptiveMapConfig;
                                    initial_point::Union{Nothing, AbstractVector}=nothing,
                                    cells::Union{Nothing, MapCellGrid}=nothing,
                                    backend::ComputeBackend=CPUBackend(),
                                    solver=Tsit5(),
                                    reltol::Float64=1e-8,
                                    abstol::Float64=1e-8)
    coarse_config.reuse_neighbor_seeds && throw(ArgumentError(
        "adaptive_bifurcation_map requires fixed-seed coarse traversal " *
        "(reuse_neighbor_seeds=false on the BifurcationMapConfig). " *
        "Traversal-dependent neighbor modes make the result depend on thread scheduling."))

    na = coarse_config.a_steps + 1
    nb = coarse_config.b_steps + 1
    adaptive.total_budget >= na * nb || throw(ArgumentError(
        "AdaptiveMapConfig.total_budget ($(adaptive.total_budget)) must be at least " *
        "(a_steps+1)*(b_steps+1) = $(na * nb) to run the coarse grid."))

    coarse_result, coarse_diagnostics = _bifurcation_map(sys, coarse_config;
                                                          initial_point=initial_point,
                                                          cells=cells, backend=backend,
                                                          solver=solver, reltol=reltol,
                                                          abstol=abstol)
    _run_adaptive_from_coarse(sys, coarse_config, adaptive, coarse_result, coarse_diagnostics;
                               initial_point=initial_point, backend=backend,
                               solver=solver, reltol=reltol, abstol=abstol)
end

# ─── Shared post-coarse entry point ─────────────────────────────────────────────

function _run_adaptive_from_coarse(sys::DynamicalSystem,
                                    coarse_config::BifurcationMapConfig,
                                    adaptive::AdaptiveMapConfig,
                                    coarse_result::BifurcationMapResult,
                                    coarse_diagnostics::AbstractDict;
                                    initial_point=nothing,
                                    backend::ComputeBackend=CPUBackend(),
                                    kwargs...)
    a_steps = coarse_config.a_steps; b_steps = coarse_config.b_steps
    na = a_steps + 1; nb = b_steps + 1
    a_min = coarse_config.a_min; a_max = coarse_config.a_max
    b_min = coarse_config.b_min; b_max = coarse_config.b_max

    coarse_evals  = na * nb
    total_budget  = adaptive.total_budget
    ref_budget    = max(total_budget - coarse_evals, 0)
    scale         = _adaptive_lattice_scale(adaptive.max_depth)
    a_lattice     = _adaptive_axis_lattice_coords(na - 1, scale, "a")
    b_lattice     = _adaptive_axis_lattice_coords(nb - 1, scale, "b")

    status_diag = get(coarse_diagnostics, "status", nothing)
    sc_mat = isnothing(status_diag) ? nothing : get(status_diag, "statusCodes", nothing)
    cc_mat = isnothing(status_diag) ? nothing : get(status_diag, "closureConfidence", nothing)

    samples = AdaptiveMapSample[]
    sizehint!(samples, total_budget)
    lookup  = Dict{Tuple{Int,Int},Int}()
    sizehint!(lookup, total_budget)

    per = coarse_result.periodicity
    for i in 1:na, j in 1:nb
        ia = a_lattice[i]; ib = b_lattice[j]
        period      = per[i, j]
        status_code = isnothing(sc_mat) ? _map_status_code(:unknown) : sc_mat[i, j]
        conf        = isnothing(cc_mat) ? 0.0 : cc_mat[i, j]
        s = AdaptiveMapSample(coarse_result.a_grid[i], coarse_result.b_grid[j],
                              period, status_code, conf, 0)
        push!(samples, s)
        lookup[(ia, ib)] = length(samples)
    end

    budget_remaining = Ref(ref_budget)

    param_template  = map_param_template(coarse_config)
    a_indices       = map_a_write_indices(coarse_config)
    b_indices       = map_b_write_indices(coarse_config)
    points_to_drop  = _map_transient_budget(coarse_config)

    x0 = sys isa DiscreteMap ?
        (isnothing(initial_point) ? zeros(SVector{sys.dim, Float64}) :
                                    SVector{sys.dim}(initial_point)) :
        _resolve_initial_state(sys, initial_point)

    multistability_enabled = _map_multistability_enabled(coarse_config)
    dim = sys isa DiscreteMap ? sys.dim : length(sys.section.projection)
    extra_seed_vectors = multistability_enabled ?
        _map_extra_seed_vectors(coarse_config, dim) : Vector{Vector{Float64}}()
    multistability_seeds = if sys isa DiscreteMap
        multistability_enabled ?
            [x0, (SVector{sys.dim, Float64}(seed) for seed in extra_seed_vectors)...] :
            SVector[]
    else
        multistability_enabled ? [copy(x0), (copy(seed) for seed in extra_seed_vectors)...] :
                                 Vector{Vector{Float64}}()
    end

    detection_fn = _adaptive_detection_fn(sys, coarse_config, param_template,
                                           a_indices, b_indices, x0, points_to_drop,
                                           multistability_seeds; kwargs...)

    leaf_cells, segments, depth_reached, split_count, flagged, uninspected, any_limited =
        _run_adaptive_refinement(coarse_result, adaptive,
                                  samples, lookup,
                                  budget_remaining,
                                  detection_fn)

    budget_used  = coarse_evals + (ref_budget - budget_remaining[])
    ref_evals    = budget_used - coarse_evals

    return AdaptiveMapResult(
        samples,
        leaf_cells,
        segments,
        coarse_result,
        total_budget,
        budget_used,
        coarse_evals,
        ref_evals,
        any_limited,   # budget_exhausted: true only when budget prevented eligible work
        uninspected,
        depth_reached,
        adaptive.max_depth,
        flagged,
        split_count,
        coarse_result.compute_backend,
        coarse_result.system_name,
        coarse_result.param_names,
        coarse_result.timestamp
    )
end

# ─── Public accessors ───────────────────────────────────────────────────────────

"""
    adaptive_map_summary(result::AdaptiveMapResult) -> NamedTuple

Return a concise provenance summary for an `AdaptiveMapResult`.

Fields:
- `system_name`, `param_names`
- `coarse_a_steps`, `coarse_b_steps`
- `total_budget`, `budget_used`
- `coarse_evaluations`, `refinement_evaluations`
- `budget_exhausted` — true when budget prevented at least one eligible refinement or
  center-screening step
- `uninspected_cell_count` — uniform-corner leaf cells whose center was not evaluated
  (`terminal == :uninspected`) due to budget or max-depth limits
- `max_depth_reached`, `max_depth_allowed`
- `flagged_cells`, `split_cells`
- `leaf_cell_count`, `boundary_segment_count`
- `boundary_length` — sum of Euclidean lengths of all boundary segments
- `resolved_segments`, `ambiguous_segments`, `multi_region_segments`
- `compute_backend`
"""
function adaptive_map_summary(result::AdaptiveMapResult)
    segs = result.boundary_segments
    bl = sum(s -> hypot(s.a1 - s.a0, s.b1 - s.b0), segs; init=0.0)
    resolved   = count(s -> s.ambiguity == :resolved,    segs)
    ambiguous  = count(s -> s.ambiguity == :ambiguous,   segs)
    multi_reg  = count(s -> s.ambiguity == :multi_region, segs)
    return (
        system_name              = result.system_name,
        param_names              = result.param_names,
        coarse_a_steps           = length(result.coarse_result.a_grid) - 1,
        coarse_b_steps           = length(result.coarse_result.b_grid) - 1,
        total_budget             = result.total_budget,
        budget_used              = result.budget_used,
        coarse_evaluations       = result.coarse_evaluations,
        refinement_evaluations   = result.refinement_evaluations,
        budget_exhausted         = result.budget_exhausted,
        uninspected_cell_count   = result.uninspected_cell_count,
        max_depth_reached        = result.max_depth_reached,
        max_depth_allowed        = result.max_depth_allowed,
        flagged_cells            = result.flagged_cells,
        split_cells              = result.split_cells,
        leaf_cell_count          = length(result.leaf_cells),
        boundary_segment_count   = length(segs),
        boundary_length          = bl,
        resolved_segments        = resolved,
        ambiguous_segments       = ambiguous,
        multi_region_segments    = multi_reg,
        compute_backend          = result.compute_backend,
    )
end

# ─── Serialization: adaptive-map-v3 (exact columnar format) ─────────────────────
#
# Closed sets for integer encoding — adding a new trigger/terminal/ambiguity value
# requires updating these constants and the format version string.
const _ADAPTIVE_REASON_BITS = (
    (:period_disagreement, 1),
    (:status_disagreement, 2),
    (:low_confidence,      4),
    (:confidence_delta,    8),
    (:center_disagreement, 16),
)
const _ADAPTIVE_REASON_VALID_MASK = 31   # 1|2|4|8|16

const _ADAPTIVE_TERMINAL_CODES = (
    (:interior,      0),
    (:boundary,      1),
    (:budget_limited, 2),
    (:uninspected,   3),
)

const _ADAPTIVE_AMBIGUITY_CODES = (
    (:resolved,     0),
    (:ambiguous,    1),
    (:multi_region, 2),
)

function _encode_reason_bitmask(reasons::Vector{Symbol})
    mask = 0
    for r in reasons
        bit = 0
        for (sym, b) in _ADAPTIVE_REASON_BITS
            sym === r && (bit = b; break)
        end
        bit == 0 && throw(ArgumentError(
            "Unknown reason symbol $(repr(r)); not in the closed reason set."))
        mask |= bit
    end
    return mask
end

function _decode_reason_bitmask(mask::Int)
    (mask & ~_ADAPTIVE_REASON_VALID_MASK) != 0 && throw(ArgumentError(
        "Reason bitmask $mask contains bits outside the valid set (0–$(_ADAPTIVE_REASON_VALID_MASK))."))
    reasons = Symbol[]
    for (sym, bit) in _ADAPTIVE_REASON_BITS
        (mask & bit) != 0 && push!(reasons, sym)
    end
    return reasons
end

function _encode_terminal(t::Symbol)
    for (sym, code) in _ADAPTIVE_TERMINAL_CODES
        sym === t && return code
    end
    throw(ArgumentError("Unknown terminal symbol $(repr(t))."))
end

function _decode_terminal(code::Int)
    for (sym, c) in _ADAPTIVE_TERMINAL_CODES
        c == code && return sym
    end
    valid = join([c for (_, c) in _ADAPTIVE_TERMINAL_CODES], ", ")
    throw(ArgumentError("Unknown terminal code $code; expected one of [$valid]."))
end

function _encode_ambiguity(a::Symbol)
    for (sym, code) in _ADAPTIVE_AMBIGUITY_CODES
        sym === a && return code
    end
    throw(ArgumentError("Unknown ambiguity symbol $(repr(a))."))
end

function _decode_ambiguity(code::Int)
    for (sym, c) in _ADAPTIVE_AMBIGUITY_CODES
        c == code && return sym
    end
    valid = join([c for (_, c) in _ADAPTIVE_AMBIGUITY_CODES], ", ")
    throw(ArgumentError("Unknown ambiguity code $code; expected one of [$valid]."))
end

function _require_adaptive_columns(cols::AbstractDict, required, label::AbstractString)
    for k in required
        haskey(cols, k) || throw(ArgumentError(
            "Missing required column '$(k)' in '$(label)'."))
    end
end

"""
    serialize_adaptive_map_result(result::AdaptiveMapResult) -> Dict{String, Any}

Produce a versioned exact columnar (JSON-compatible) representation of an
`AdaptiveMapResult`.  Format string: `"adaptive-map-v3"`.

Roundtrip: `deserialize_adaptive_map_result(serialize_adaptive_map_result(result))`.
"""
function serialize_adaptive_map_result(result::AdaptiveMapResult)
    n_c   = length(result.leaf_cells)
    n_seg = length(result.boundary_segments)

    terminal_codes  = Vector{Int}(undef, n_c)
    reason_bitmasks = Vector{Int}(undef, n_c)
    for (k, cell) in enumerate(result.leaf_cells)
        terminal_codes[k]  = _encode_terminal(cell.terminal)
        reason_bitmasks[k] = _encode_reason_bitmask(cell.reasons)
    end

    ambiguity_codes = Vector{Int}(undef, n_seg)
    for (k, seg) in enumerate(result.boundary_segments)
        ambiguity_codes[k] = _encode_ambiguity(seg.ambiguity)
    end

    return Dict{String, Any}(
        "format"                => "adaptive-map-v3",
        "systemName"            => result.system_name,
        "paramNames"            => String.(collect(result.param_names)),
        "timestamp"             => _serialize_timestamp(result.timestamp),
        "totalBudget"           => result.total_budget,
        "budgetUsed"            => result.budget_used,
        "coarseEvaluations"     => result.coarse_evaluations,
        "refinementEvaluations" => result.refinement_evaluations,
        "budgetExhausted"       => result.budget_exhausted,
        "uninspectedCellCount"  => result.uninspected_cell_count,
        "maxDepthReached"       => result.max_depth_reached,
        "maxDepthAllowed"       => result.max_depth_allowed,
        "flaggedCells"          => result.flagged_cells,
        "splitCells"            => result.split_cells,
        "computeBackend"        => String(result.compute_backend),
        "samples"               => Dict{String, Any}(
            "a"          => [s.a           for s in result.samples],
            "b"          => [s.b           for s in result.samples],
            "period"     => [s.period      for s in result.samples],
            "statusCode" => [s.status_code for s in result.samples],
            "confidence" => [s.confidence  for s in result.samples],
            "depth"      => [s.depth       for s in result.samples],
        ),
        "leafCells"             => Dict{String, Any}(
            "a0"             => [c.a0        for c in result.leaf_cells],
            "a1"             => [c.a1        for c in result.leaf_cells],
            "b0"             => [c.b0        for c in result.leaf_cells],
            "b1"             => [c.b1        for c in result.leaf_cells],
            "ia0"            => [c.ia0       for c in result.leaf_cells],
            "ia1"            => [c.ia1       for c in result.leaf_cells],
            "ib0"            => [c.ib0       for c in result.leaf_cells],
            "ib1"            => [c.ib1       for c in result.leaf_cells],
            "depth"          => [c.depth     for c in result.leaf_cells],
            "siSw"           => [c.si_sw     for c in result.leaf_cells],
            "siSe"           => [c.si_se     for c in result.leaf_cells],
            "siNw"           => [c.si_nw     for c in result.leaf_cells],
            "siNe"           => [c.si_ne     for c in result.leaf_cells],
            "siCenter"       => [c.si_center for c in result.leaf_cells],
            "terminal"       => terminal_codes,
            "reasonsBitmask" => reason_bitmasks,
        ),
        "boundarySegments"      => Dict{String, Any}(
            "a0"         => [s.a0       for s in result.boundary_segments],
            "b0"         => [s.b0       for s in result.boundary_segments],
            "a1"         => [s.a1       for s in result.boundary_segments],
            "b1"         => [s.b1       for s in result.boundary_segments],
            "keyAStatus" => [s.key_a[1] for s in result.boundary_segments],
            "keyAPeriod" => [s.key_a[2] for s in result.boundary_segments],
            "keyBStatus" => [s.key_b[1] for s in result.boundary_segments],
            "keyBPeriod" => [s.key_b[2] for s in result.boundary_segments],
            "ambiguity"  => ambiguity_codes,
        ),
        "coarseResult"          => serialize_bifurcation_map_result(result.coarse_result),
    )
end

"""
    deserialize_adaptive_map_result(data::AbstractDict) -> AdaptiveMapResult

Reconstruct an `AdaptiveMapResult` from a plain-data dict produced by
`serialize_adaptive_map_result`.  Validates format version, column presence,
equal lengths, numeric ranges, enum encodings, and canonical key ordering.
Rejects malformed payloads with an `ArgumentError`.
"""
function deserialize_adaptive_map_result(data::AbstractDict)
    fmt = get(data, "format", "")
    fmt == "adaptive-map-v3" || throw(ArgumentError(
        "Unrecognised adaptive map result format $(repr(fmt)); expected \"adaptive-map-v3\"."))

    haskey(data, "paramNames") || throw(ArgumentError("Missing required field 'paramNames'."))
    pn_raw = data["paramNames"]
    length(pn_raw) == 2 || throw(ArgumentError("'paramNames' must have exactly 2 elements."))
    param_names = (Symbol(pn_raw[1]), Symbol(pn_raw[2]))

    for req in ("coarseResult", "samples", "leafCells", "boundarySegments",
                "totalBudget", "budgetUsed", "coarseEvaluations", "refinementEvaluations",
                "budgetExhausted", "maxDepthReached", "maxDepthAllowed",
                "flaggedCells", "splitCells", "computeBackend", "timestamp", "systemName")
        haskey(data, req) || throw(ArgumentError("Missing required field '$(req)'."))
    end

    coarse = deserialize_bifurcation_map_result(data["coarseResult"])

    # ── Samples ──────────────────────────────────────────────────────────────────
    sc = data["samples"]
    _require_adaptive_columns(sc, ("a","b","period","statusCode","confidence","depth"), "samples")
    sa      = Float64.(sc["a"])
    sb      = Float64.(sc["b"])
    speriod = Int.(sc["period"])
    sstatus = Int.(sc["statusCode"])
    sconf   = Float64.(sc["confidence"])
    sdepth  = Int.(sc["depth"])
    n_s     = length(sa)
    all(==(n_s), (length(sb), length(speriod), length(sstatus), length(sconf), length(sdepth))) ||
        throw(ArgumentError("'samples' columns have unequal lengths."))
    all(isfinite, sa) || throw(ArgumentError("'samples.a' contains non-finite values."))
    all(isfinite, sb) || throw(ArgumentError("'samples.b' contains non-finite values."))
    all(>=(0.0), sconf) && all(<=(1.0), sconf) ||
        throw(ArgumentError("'samples.confidence' values must be in [0, 1]."))
    all(>=(0), speriod) || throw(ArgumentError("'samples.period' values must be >= 0."))
    all(>=(0), sstatus) || throw(ArgumentError("'samples.statusCode' values must be >= 0."))
    all(>=(0), sdepth) || throw(ArgumentError("'samples.depth' values must be >= 0."))

    max_depth_allowed = Int(data["maxDepthAllowed"])
    (0 <= max_depth_allowed <= _ADAPTIVE_MAP_MAX_DEPTH) || throw(ArgumentError(
        "'maxDepthAllowed' must be in [0, $(_ADAPTIVE_MAP_MAX_DEPTH)] to avoid integer-lattice overflow; got $max_depth_allowed."))
    na = length(coarse.a_grid) - 1
    nb = length(coarse.b_grid) - 1
    scale = _adaptive_lattice_scale(max_depth_allowed)
    max_ia = _adaptive_checked_mul(na, scale, "leaf-cell a-lattice range bound")
    max_ib = _adaptive_checked_mul(nb, scale, "leaf-cell b-lattice range bound")

    samples = Vector{AdaptiveMapSample}(undef, n_s)
    for i in 1:n_s
        samples[i] = AdaptiveMapSample(sa[i], sb[i], speriod[i], sstatus[i], sconf[i], sdepth[i])
    end

    # ── Leaf cells ───────────────────────────────────────────────────────────────
    cc = data["leafCells"]
    _require_adaptive_columns(cc, ("a0","a1","b0","b1","ia0","ia1","ib0","ib1","depth",
                                    "siSw","siSe","siNw","siNe","siCenter",
                                    "terminal","reasonsBitmask"), "leafCells")
    ca0  = Float64.(cc["a0"]);  ca1 = Float64.(cc["a1"])
    cb0  = Float64.(cc["b0"]);  cb1 = Float64.(cc["b1"])
    cia0 = Int.(cc["ia0"]);    cia1 = Int.(cc["ia1"])
    cib0 = Int.(cc["ib0"]);    cib1 = Int.(cc["ib1"])
    cdepth   = Int.(cc["depth"])
    csisw    = Int.(cc["siSw"]);  csise  = Int.(cc["siSe"])
    csinw    = Int.(cc["siNw"]);  csine  = Int.(cc["siNe"])
    csictr   = Int.(cc["siCenter"])
    ctermraw = Int.(cc["terminal"])
    creasraw = Int.(cc["reasonsBitmask"])
    n_c = length(ca0)
    all(==(n_c), (length(ca1), length(cb0), length(cb1),
                  length(cia0), length(cia1), length(cib0), length(cib1),
                  length(cdepth), length(csisw), length(csise), length(csinw), length(csine),
                  length(csictr), length(ctermraw), length(creasraw))) ||
        throw(ArgumentError("'leafCells' columns have unequal lengths."))

    all(isfinite, ca0) && all(isfinite, ca1) || throw(ArgumentError("'leafCells.a0/a1' contain non-finite values."))
    all(isfinite, cb0) && all(isfinite, cb1) || throw(ArgumentError("'leafCells.b0/b1' contain non-finite values."))
    all(>=(0), cia0) && all(<=(max_ia), cia0) || throw(ArgumentError("'leafCells.ia0' out of range [0, $max_ia]."))
    all(>=(0), cia1) && all(<=(max_ia), cia1) || throw(ArgumentError("'leafCells.ia1' out of range [0, $max_ia]."))
    all(>=(0), cib0) && all(<=(max_ib), cib0) || throw(ArgumentError("'leafCells.ib0' out of range [0, $max_ib]."))
    all(>=(0), cib1) && all(<=(max_ib), cib1) || throw(ArgumentError("'leafCells.ib1' out of range [0, $max_ib]."))
    all(i -> cia0[i] < cia1[i], 1:n_c) || throw(ArgumentError("'leafCells' has ia0 >= ia1 for some cell."))
    all(i -> cib0[i] < cib1[i], 1:n_c) || throw(ArgumentError("'leafCells' has ib0 >= ib1 for some cell."))
    all(x -> 1 <= x <= n_s, csisw) && all(x -> 1 <= x <= n_s, csise) &&
    all(x -> 1 <= x <= n_s, csinw) && all(x -> 1 <= x <= n_s, csine) ||
        throw(ArgumentError("'leafCells' corner sample indices out of range [1, $n_s]."))
    all(x -> x == 0 || (1 <= x <= n_s), csictr) ||
        throw(ArgumentError("'leafCells.siCenter' values must be 0 or in [1, $n_s]."))
    all(>=(0), cdepth) && all(<=(max_depth_allowed), cdepth) ||
        throw(ArgumentError("'leafCells.depth' out of range [0, $max_depth_allowed]."))

    leaf_cells = Vector{AdaptiveMapLeafCell}(undef, n_c)
    for i in 1:n_c
        leaf_cells[i] = AdaptiveMapLeafCell(
            ca0[i], ca1[i], cb0[i], cb1[i],
            cia0[i], cia1[i], cib0[i], cib1[i],
            cdepth[i],
            csisw[i], csise[i], csinw[i], csine[i], csictr[i],
            _decode_terminal(ctermraw[i]),
            _decode_reason_bitmask(creasraw[i]),
        )
    end

    # ── Boundary segments ─────────────────────────────────────────────────────────
    sg = data["boundarySegments"]
    _require_adaptive_columns(sg, ("a0","b0","a1","b1","keyAStatus","keyAPeriod",
                                    "keyBStatus","keyBPeriod","ambiguity"), "boundarySegments")
    sa0  = Float64.(sg["a0"]);  sb0  = Float64.(sg["b0"])
    sa1  = Float64.(sg["a1"]);  sb1  = Float64.(sg["b1"])
    skas = Int.(sg["keyAStatus"]); skap = Int.(sg["keyAPeriod"])
    skbs = Int.(sg["keyBStatus"]); skbp = Int.(sg["keyBPeriod"])
    sambig = Int.(sg["ambiguity"])
    n_seg = length(sa0)
    all(==(n_seg), (length(sb0), length(sa1), length(sb1),
                    length(skas), length(skap), length(skbs), length(skbp), length(sambig))) ||
        throw(ArgumentError("'boundarySegments' columns have unequal lengths."))
    all(isfinite, sa0) && all(isfinite, sb0) && all(isfinite, sa1) && all(isfinite, sb1) ||
        throw(ArgumentError("'boundarySegments' endpoint coordinates contain non-finite values."))
    all(>=(0), skas) && all(>=(0), skbs) ||
        throw(ArgumentError("'boundarySegments' key status codes must be >= 0."))
    all(i -> (skas[i], skap[i]) <= (skbs[i], skbp[i]), 1:n_seg) ||
        throw(ArgumentError("'boundarySegments' keys not in canonical order (keyA <= keyB)."))

    segments = Vector{AdaptiveMapSegment}(undef, n_seg)
    for i in 1:n_seg
        segments[i] = AdaptiveMapSegment(
            sa0[i], sb0[i], sa1[i], sb1[i],
            (skas[i], skap[i]), (skbs[i], skbp[i]),
            _decode_ambiguity(sambig[i]),
        )
    end

    # ── Budget & provenance ───────────────────────────────────────────────────────
    total_budget = Int(data["totalBudget"])
    budget_used  = Int(data["budgetUsed"])
    (0 <= budget_used <= total_budget) || throw(ArgumentError(
        "budget_used=$budget_used outside [0, total_budget=$total_budget]."))

    return AdaptiveMapResult(
        samples,
        leaf_cells,
        segments,
        coarse,
        total_budget,
        budget_used,
        Int(data["coarseEvaluations"]),
        Int(data["refinementEvaluations"]),
        Bool(data["budgetExhausted"]),
        Int(get(data, "uninspectedCellCount", 0)),
        Int(data["maxDepthReached"]),
        max_depth_allowed,
        Int(data["flaggedCells"]),
        Int(data["splitCells"]),
        _compute_backend_symbol(String(data["computeBackend"])),
        String(data["systemName"]),
        param_names,
        _deserialize_timestamp(data["timestamp"]),
    )
end
