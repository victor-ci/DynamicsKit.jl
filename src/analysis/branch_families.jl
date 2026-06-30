using LinearAlgebra: norm
using Statistics: median

const _BASIN_PARAM_TOLERANCE_GRID_FRACTION = 0.55

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

"""
    BranchBasinAssignment

Classification of a continuation branch against an observed brute-force
attractor cloud. `basin_id == "observed"` means sampled branch orbits are
geometrically represented in the supplied brute-force result; `unobserved`
means the branch was continued but was not reached by that brute-force seed.
"""
struct BranchBasinAssignment
    branch_index::Int
    basin_index::Int
    basin_id::String
    basin_label::String
    observed::Bool
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

function _basin_param_tolerance(params::AbstractVector{<:Real})
    length(params) <= 1 && return Inf
    unique_params = sort(unique(Float64.(params)))
    length(unique_params) <= 1 && return Inf
    gaps = diff(unique_params)
    finite_gaps = [gap for gap in gaps if isfinite(gap) && gap > 0]
    isempty(finite_gaps) && return Inf
    # Slightly more than half a grid step captures the nearest brute-force slice
    # despite roundoff while avoiding accidental matches to neighbouring slices.
    return _BASIN_PARAM_TOLERANCE_GRID_FRACTION * median(finite_gaps)
end

function _basin_cloud_points(brute_force::BruteForceResult, param::Float64, param_tolerance::Float64)
    isempty(brute_force.params) && return Vector{Vector{Float64}}()
    distances = abs.(brute_force.params .- param)
    threshold = isinf(param_tolerance) ? Inf : param_tolerance
    indices = findall(distance -> distance <= threshold, distances)
    return [vec(Float64.(brute_force.points[idx, :])) for idx in indices]
end

function _squared_distance_to_cloud(canonical_orbit::Vector{Float64},
                                    point_index::Int,
                                    dim::Int,
                                    cloud::Vector{Vector{Float64}})
    offset = (point_index - 1) * dim
    best = Inf
    for cloud_point in cloud
        length(cloud_point) == dim || continue
        total = 0.0
        @inbounds for coordinate in 1:dim
            delta = canonical_orbit[offset + coordinate] - cloud_point[coordinate]
            total += delta * delta
        end
        best = min(best, total)
    end
    return best
end

function _basin_sample_distance(sample::_BranchFamilySample,
                                brute_force::BruteForceResult,
                                param_tolerance::Float64)
    cloud = _basin_cloud_points(brute_force, sample.param, param_tolerance)
    isempty(cloud) && return Inf
    dim = length(first(cloud))
    dim == 0 && return Inf
    length(sample.canonical_orbit) % dim == 0 || return Inf
    orbit_count = length(sample.canonical_orbit) ÷ dim
    orbit_count == 0 && return Inf
    distances = Float64[]
    sizehint!(distances, orbit_count)
    for point_index in 1:orbit_count
        squared = _squared_distance_to_cloud(sample.canonical_orbit, point_index, dim, cloud)
        isfinite(squared) && push!(distances, sqrt(squared))
    end
    isempty(distances) && return Inf
    return median(distances) / max(sample.scale, 1.0)
end

"""
    branch_basin_assignments(sys, branches, brute_force; kwargs...) -> Vector{BranchBasinAssignment}

Classify branches by whether their sampled orbits match the attractor cloud
observed by a supplied brute-force run. This does **not** infer every basin of
the dynamical system; it identifies the basin reached by that brute-force seed
versus continued branches that are not represented in that observed trajectory.
"""
function branch_basin_assignments(sys::DynamicalSystem,
                                  branches::AbstractVector{<:BranchResult},
                                  brute_force::BruteForceResult;
                                  params::Vector{Float64}=Float64[],
                                  linked_param_indices::Vector{Int}=Int[],
                                  sample_count::Int=9,
                                  param_tolerance::Union{Nothing, Float64}=nothing,
                                  distance_tolerance::Float64=0.05,
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
    local_param_tolerance = isnothing(param_tolerance) ? _basin_param_tolerance(brute_force.params) : param_tolerance
    assignments = BranchBasinAssignment[]
    sizehint!(assignments, length(branches))
    for (idx, signature) in enumerate(signatures)
        distances = Float64[]
        for sample in signature.samples
            distance = _basin_sample_distance(sample, brute_force, local_param_tolerance)
            isfinite(distance) && push!(distances, distance)
        end
        median_distance = isempty(distances) ? Inf : median(distances)
        observed = median_distance <= distance_tolerance
        confidence = isfinite(median_distance) ? max(0.0, min(1.0, 1.0 - median_distance / max(distance_tolerance, eps(Float64)))) : 0.0
        diagnostics = Dict{String, Any}(
            "period" => signature.period,
            "paramMin" => signature.param_min,
            "paramMax" => signature.param_max,
            "sampleCount" => length(signature.samples),
            "matchedSampleCount" => length(distances),
            "medianDistance" => median_distance,
            "distanceTolerance" => distance_tolerance,
            "paramTolerance" => local_param_tolerance
        )
        push!(assignments, BranchBasinAssignment(
            idx,
            observed ? 1 : 0,
            observed ? "observed" : "unobserved",
            observed ? "Observed seed basin" : "Not reached by seed",
            observed,
            confidence,
            diagnostics
        ))
    end
    return assignments
end
