using LinearAlgebra: norm
using Statistics: median

"""
    BranchFamilyAssignment

Conservative geometric family assignment for a continuation branch. `family_id`
is stable within one assignment pass and groups branches whose sampled
Poincaré-orbit geometry overlaps within tolerance.
"""
struct BranchFamilyAssignment
    branch_index::Int
    family_index::Int
    family_id::String
    family_label::String
    confidence::Float64
    diagnostics::Dict{String, Any}
end

struct _BranchFamilySample
    param::Float64
    canonical_orbit::Vector{Float64}
    scale::Float64
end

struct _BranchFamilySignature
    branch_index::Int
    period::Int
    param_min::Float64
    param_max::Float64
    samples::Vector{_BranchFamilySample}
end

function _branch_family_sample_indices(n::Int, sample_count::Int)
    n <= 0 && return Int[]
    count = min(max(sample_count, 1), n)
    count == 1 && return [cld(n, 2)]
    return [floor(Int, 1 + (idx - 1) * (n - 1) / (count - 1)) for idx in 1:count]
end

function _branch_family_lexkey(point::AbstractVector{<:Real})
    return Tuple(Float64(v) for v in point)
end

function _branch_family_canonical_orbit(points::Vector{Vector{Float64}})
    isempty(points) && return Float64[]
    ordered = sort(points; by=_branch_family_lexkey)
    return reduce(vcat, ordered)
end

function _branch_family_orbit_scale(points::Vector{Vector{Float64}})
    length(points) <= 1 && return 1.0
    mat = reduce(hcat, points)'
    span = maximum(vec(maximum(mat; dims=1) .- minimum(mat; dims=1)))
    return max(Float64(span), 1.0)
end

function _branch_family_orbit(sys::DiscreteMap,
                              state::Vector{Float64},
                              params::Vector{Float64},
                              period::Int;
                              kwargs...)
    dim = sys.dim
    current = copy(state)
    points = Vector{Float64}[copy(current)]
    for _ in 2:period
        current = Array(sys.f(SVector{dim}(current), params))
        push!(points, copy(current))
    end
    return points, true
end

function _branch_family_orbit(sys::ContinuousODE,
                              state::Vector{Float64},
                              params::Vector{Float64},
                              period::Int;
                              solver=Tsit5(),
                              reltol::Float64=1e-8,
                              abstol::Float64=1e-8,
                              tmax::Union{Nothing, Float64}=nothing,
                              min_crossing_time::Float64=1e-6)
    current = copy(state)
    points = Vector{Float64}[copy(current)]
    for _ in 2:period
        next_point, found = _poincare_projected(
            sys,
            current,
            params;
            period=1,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
        found || return points, false
        current = collect(Float64, next_point)
        push!(points, copy(current))
    end
    return points, true
end

function _branch_family_signature(sys::DynamicalSystem,
                                  branch::BranchResult,
                                  branch_index::Int;
                                  params::Vector{Float64}=Float64[],
                                  linked_param_indices::Vector{Int}=Int[],
                                  sample_count::Int=9,
                                  kwargs...)
    points = _branch_points(branch)
    isempty(points) && return _BranchFamilySignature(branch_index, max(branch.period, 1), NaN, NaN, _BranchFamilySample[])

    base_params = isempty(params) ? copy(sys.default_params) : copy(params)
    param_index = findfirst(==(branch.param_name), sys.param_names)
    isnothing(param_index) && error("Cannot infer branch family: branch parameter '$(branch.param_name)' is not present in system '$(sys.name)' parameters $(collect(sys.param_names)).")
    projected_dim = state_dim(sys)
    period = max(branch.period, 1)
    samples = _BranchFamilySample[]
    for idx in _branch_family_sample_indices(length(points), sample_count)
        pt = points[idx]
        local_params = inject_param(base_params, param_index, Float64(pt.param), linked_param_indices)
        state = branch_point_state(pt, projected_dim)
        orbit, valid = _branch_family_orbit(sys, state, local_params, period; kwargs...)
        valid || continue
        push!(
            samples,
            _BranchFamilySample(
                Float64(pt.param),
                _branch_family_canonical_orbit(orbit),
                _branch_family_orbit_scale(orbit)
            )
        )
    end

    pars = Float64[Float64(getproperty(pt, :param)) for pt in points]
    return _BranchFamilySignature(branch_index, period, minimum(pars), maximum(pars), samples)
end

function _branch_family_overlap(a::_BranchFamilySignature, b::_BranchFamilySignature)
    lo = max(a.param_min, b.param_min)
    hi = min(a.param_max, b.param_max)
    overlap = hi - lo
    span = min(a.param_max - a.param_min, b.param_max - b.param_min)
    if !isfinite(overlap) || !isfinite(span) || span <= 0
        return 0.0
    end
    return max(0.0, overlap) / span
end

function _branch_family_sample_distance(a::_BranchFamilySample, b::_BranchFamilySample)
    length(a.canonical_orbit) == length(b.canonical_orbit) || return Inf
    raw = norm(a.canonical_orbit .- b.canonical_orbit) / sqrt(length(a.canonical_orbit))
    return raw / max(a.scale, b.scale, 1.0)
end

function _branch_family_distance(a::_BranchFamilySignature, b::_BranchFamilySignature)
    (isempty(a.samples) || isempty(b.samples)) && return Inf
    distances = Float64[]
    for sa in a.samples
        best = Inf
        for sb in b.samples
            best = min(best, _branch_family_sample_distance(sa, sb))
        end
        isfinite(best) && push!(distances, best)
    end
    isempty(distances) && return Inf
    return median(distances)
end

mutable struct _BranchFamilyUnionFind
    parent::Vector{Int}
end

_BranchFamilyUnionFind(n::Int) = _BranchFamilyUnionFind(collect(1:n))

function _branch_family_find!(uf::_BranchFamilyUnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]
        x = uf.parent[x]
    end
    return x
end

function _branch_family_union!(uf::_BranchFamilyUnionFind, a::Int, b::Int)
    ra = _branch_family_find!(uf, a)
    rb = _branch_family_find!(uf, b)
    ra == rb && return
    uf.parent[max(ra, rb)] = min(ra, rb)
end

"""
    branch_family_assignments(sys, branches; kwargs...) -> Vector{BranchFamilyAssignment}

Infer conservative attractor-family IDs for continuation branches from sampled
orbit geometry. Branches are only compared when they share the same minimal
period by default; this avoids scientifically unsafe claims that a period-doubled
child is the same attractor as its parent unless explicit lineage is supplied by
a caller. The returned vector is ordered like `branches`.
"""
function branch_family_assignments(sys::DynamicalSystem,
                                   branches::AbstractVector{<:BranchResult};
                                   params::Vector{Float64}=Float64[],
                                   linked_param_indices::Vector{Int}=Int[],
                                   sample_count::Int=9,
                                   same_period_only::Bool=true,
                                   min_overlap_fraction::Float64=0.15,
                                   distance_tolerance::Float64=0.03,
                                   solver=Tsit5(),
                                   reltol::Float64=1e-8,
                                   abstol::Float64=1e-8,
                                   tmax::Union{Nothing, Float64}=nothing,
                                   min_crossing_time::Float64=1e-6)
    signatures = [
        _branch_family_signature(
            sys,
            branch,
            idx;
            params=params,
            linked_param_indices=linked_param_indices,
            sample_count=sample_count,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
        for (idx, branch) in enumerate(branches)
    ]
    uf = _BranchFamilyUnionFind(length(branches))
    accepted_distances = [Float64[] for _ in eachindex(branches)]
    for i in 1:length(signatures), j in (i + 1):length(signatures)
        a = signatures[i]
        b = signatures[j]
        same_period_only && a.period != b.period && continue
        _branch_family_overlap(a, b) >= min_overlap_fraction || continue
        distance = _branch_family_distance(a, b)
        if distance <= distance_tolerance
            _branch_family_union!(uf, i, j)
            push!(accepted_distances[i], distance)
            push!(accepted_distances[j], distance)
        end
    end

    roots = [_branch_family_find!(uf, idx) for idx in eachindex(branches)]
    ordered_roots = unique(roots[sortperm(eachindex(roots); by=i -> (
        signatures[i].period,
        signatures[i].param_min,
        signatures[i].param_max,
        i
    ))])
    family_index_for_root = Dict(root => idx for (idx, root) in enumerate(ordered_roots))
    assignments = BranchFamilyAssignment[]
    sizehint!(assignments, length(branches))
    for idx in eachindex(branches)
        root = roots[idx]
        family_index = family_index_for_root[root]
        distances = accepted_distances[idx]
        confidence = isempty(distances) ? 1.0 : max(0.0, min(1.0, 1.0 - median(distances) / max(distance_tolerance, eps(Float64))))
        diagnostics = Dict{String, Any}(
            "period" => signatures[idx].period,
            "paramMin" => signatures[idx].param_min,
            "paramMax" => signatures[idx].param_max,
            "sampleCount" => length(signatures[idx].samples),
            "distanceTolerance" => distance_tolerance,
            "samePeriodOnly" => same_period_only
        )
        push!(assignments, BranchFamilyAssignment(
            idx,
            family_index,
            "family-$(family_index)",
            "Family $(family_index)",
            confidence,
            diagnostics
        ))
    end
    return assignments
end
