"""
Serialization of library result types to/from portable JSON-plain dicts (columnar branch points
avoid serializing fragile BifurcationKit internals). Used by the atlas library cache
(`continuation_atlas(...; cache_file=...)`) and by external persistence layers such as the
workbench's session store.
"""

_serialize_timestamp(dt::DateTime) = Dates.format(dt, dateformat"yyyy-mm-ddTHH:MM:SS.s")
_deserialize_timestamp(value) = DateTime(String(value))

function _serialize_branch_point(point)
    Dict(String(name) => _plain(getproperty(point, name)) for name in propertynames(point))
end

function _serialize_branch_point_columns(points::AbstractVector)
    isempty(points) && return Dict(
        "format" => "columnar-v1",
        "fields" => String[],
        "columns" => Any[]
    )

    fields = [String(name) for name in propertynames(first(points))]
    columns = [map(point -> _plain(getproperty(point, Symbol(field))), points) for field in fields]
    return Dict(
        "format" => "columnar-v1",
        "fields" => fields,
        "columns" => columns
    )
end

function _deserialize_branch_point(data::AbstractDict{<:AbstractString, <:Any})
    keys_syms = Tuple(Symbol(String(k)) for k in keys(data))
    values_tuple = Tuple(_plain(v) for v in values(data))
    NamedTuple{keys_syms}(values_tuple)
end

function _deserialize_branch_point_collection(data)
    if data isa AbstractVector
        return Any[_deserialize_branch_point(item) for item in data]
    elseif data isa AbstractDict{<:AbstractString, <:Any}
        format = _as_string(get(data, "format", ""), "")
        format == "columnar-v1" || error("Unsupported branch point serialization format '$format'.")
        fields = Tuple(Symbol(String(name)) for name in get(data, "fields", Any[]))
        columns = collect(get(data, "columns", Any[]))
        length(fields) == length(columns) || error("Serialized branch point columns do not match the declared fields.")
        point_count = isempty(columns) ? 0 : length(columns[1])
        all(length(column) == point_count for column in columns) || error("Serialized branch point columns must all have the same length.")
        return Any[
            NamedTuple{fields}(Tuple(_plain(column[idx]) for column in columns))
            for idx in 1:point_count
        ]
    end
    error("Unsupported branch point payload type $(typeof(data)).")
end

function _serialize_branch_result(branch::BranchResult)
    Dict(
        "period" => branch.period,
        "systemName" => branch.system_name,
        "paramName" => String(branch.param_name),
        "timestamp" => _serialize_timestamp(branch.timestamp),
        "points" => _serialize_branch_point_columns(_branch_points(branch)),
        "specialPoints" => _serialize_branch_point_columns(branch.branch.specialpoint)
    )
end

function _deserialize_branch_result(data::AbstractDict{<:AbstractString, <:Any})
    points = _deserialize_branch_point_collection(data["points"])
    specials = _deserialize_branch_point_collection(get(data, "specialPoints", Any[]))
    BranchResult(
        CombinedBranchResult(points, specials),
        _as_int(get(data, "period", 1), 1),
        _as_string(get(data, "systemName", ""), ""),
        Symbol(_as_string(get(data, "paramName", "p"), "p")),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now())))
    )
end

function _serialize_bruteforce_result(result::BruteForceResult)
    Dict(
        "params" => copy(result.params),
        "points" => Array(result.points),
        "systemName" => result.system_name,
        "paramName" => String(result.param_name),
        "timestamp" => _serialize_timestamp(result.timestamp)
    )
end

# Reconstruct an (n × dim) point matrix from a serialized payload. The serializer
# stores `Array(result.points)` (a Matrix, possibly 0×dim), so the matrix branch
# preserves the true dimension even for an empty cloud (0×1, 0×2, …; reachable via
# `transient >= iterations`, which records zero points). The vector branch handles
# JSON-style rows; when those are empty the dimension is genuinely unknown, so we
# return a 0×0 matrix rather than fabricating a 2-D shape.
function _deserialize_point_matrix(raw)
    raw isa AbstractMatrix && return Float64.(raw)
    rows = [Float64[_as_float(v) for v in row] for row in raw]
    isempty(rows) && return Matrix{Float64}(undef, 0, 0)
    return permutedims(reduce(hcat, rows))
end

function _deserialize_bruteforce_result(data::AbstractDict{<:AbstractString, <:Any})
    BruteForceResult(
        Float64[_as_float(v) for v in data["params"]],
        _deserialize_point_matrix(data["points"]),
        _as_string(get(data, "systemName", ""), ""),
        Symbol(_as_string(get(data, "paramName", "p"), "p")),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now())))
    )
end

function _serialize_atlas_recon_sample(sample::AtlasReconSample)
    Dict(
        "param" => sample.param,
        "classification" => String(sample.classification),
        "bestPeriod" => sample.best_period,
        "confidence" => sample.confidence,
        "closureErrors" => copy(sample.closure_errors),
        "supportPoints" => [copy(point) for point in sample.support_points],
        "orbitCenter" => copy(sample.orbit_center),
        "orbitSpan" => copy(sample.orbit_span),
        "diagnostics" => _plain(sample.diagnostics)
    )
end

function _deserialize_atlas_recon_sample(data::AbstractDict{<:AbstractString, <:Any})
    AtlasReconSample(
        _as_float(get(data, "param", 0.0), 0.0),
        Symbol(_as_string(get(data, "classification", "insufficient"), "insufficient")),
        _as_int(get(data, "bestPeriod", 0), 0),
        _as_float(get(data, "confidence", 0.0), 0.0),
        _as_float_vector(get(data, "closureErrors", Float64[]), Float64[]),
        [Float64[_as_float(v) for v in point] for point in get(data, "supportPoints", Any[])],
        _as_float_vector(get(data, "orbitCenter", Float64[]), Float64[]),
        _as_float_vector(get(data, "orbitSpan", Float64[]), Float64[]),
        _jsonish_dict(get(data, "diagnostics", Dict{String, Any}()))
    )
end

function _serialize_atlas_window(window::AtlasWindow)
    Dict(
        "id" => window.id,
        "period" => window.period,
        "paramMin" => window.param_min,
        "paramMax" => window.param_max,
        "support" => window.support,
        "meanConfidence" => window.mean_confidence,
        "classification" => String(window.classification),
        "sampleIndices" => copy(window.sample_indices),
        "priorityScore" => window.priority_score,
        "status" => String(window.status),
        "diagnostics" => _plain(window.diagnostics)
    )
end

function _deserialize_atlas_window(data::AbstractDict{<:AbstractString, <:Any})
    AtlasWindow(
        _as_string(get(data, "id", string(uuid4())), string(uuid4())),
        _as_int(get(data, "period", 0), 0),
        _as_float(get(data, "paramMin", 0.0), 0.0),
        _as_float(get(data, "paramMax", 0.0), 0.0),
        _as_int(get(data, "support", 0), 0),
        _as_float(get(data, "meanConfidence", 0.0), 0.0),
        Symbol(_as_string(get(data, "classification", "periodic"), "periodic")),
        _as_int_vector(get(data, "sampleIndices", Int[]), Int[]),
        _as_float(get(data, "priorityScore", 0.0), 0.0),
        Symbol(_as_string(get(data, "status", "untried"), "untried")),
        _jsonish_dict(get(data, "diagnostics", Dict{String, Any}()))
    )
end

function _serialize_atlas_branch_record(record::AtlasBranchRecord)
    Dict(
        "id" => record.id,
        "branch" => _serialize_branch_result(record.branch),
        "seedParam" => record.seed_param,
        "windowId" => record.window_id,
        "coverageScore" => record.coverage_score,
        "paramMin" => record.param_min,
        "paramMax" => record.param_max,
        "diagnostics" => _plain(record.diagnostics)
    )
end

function _deserialize_atlas_branch_record(data::AbstractDict{<:AbstractString, <:Any})
    AtlasBranchRecord(
        _as_string(get(data, "id", string(uuid4())), string(uuid4())),
        _deserialize_branch_result(data["branch"]),
        _as_float(get(data, "seedParam", 0.0), 0.0),
        _as_string(get(data, "windowId", ""), ""),
        _as_float(get(data, "coverageScore", 0.0), 0.0),
        _as_float(get(data, "paramMin", 0.0), 0.0),
        _as_float(get(data, "paramMax", 0.0), 0.0),
        _jsonish_dict(get(data, "diagnostics", Dict{String, Any}()))
    )
end

function _serialize_atlas_gap(gap::AtlasGap)
    Dict(
        "id" => gap.id,
        "period" => gap.period,
        "paramMin" => gap.param_min,
        "paramMax" => gap.param_max,
        "confidence" => gap.confidence,
        "reason" => String(gap.reason),
        "depth" => gap.depth,
        "retryable" => gap.retryable,
        "diagnostics" => _plain(gap.diagnostics)
    )
end

function _deserialize_atlas_gap(data::AbstractDict{<:AbstractString, <:Any})
    AtlasGap(
        _as_string(get(data, "id", string(uuid4())), string(uuid4())),
        _as_int(get(data, "period", 0), 0),
        _as_float(get(data, "paramMin", 0.0), 0.0),
        _as_float(get(data, "paramMax", 0.0), 0.0),
        _as_float(get(data, "confidence", 0.0), 0.0),
        Symbol(_as_string(get(data, "reason", "continuation_failed"), "continuation_failed")),
        _as_int(get(data, "depth", 0), 0),
        _as_bool(get(data, "retryable", false), false),
        _jsonish_dict(get(data, "diagnostics", Dict{String, Any}()))
    )
end

function _serialize_atlas_result(result::AtlasResult)
    Dict(
        "reconSamples" => [_serialize_atlas_recon_sample(sample) for sample in result.recon_samples],
        "windows" => [_serialize_atlas_window(window) for window in result.windows],
        "branchRecords" => [_serialize_atlas_branch_record(record) for record in result.branch_records],
        "gaps" => [_serialize_atlas_gap(gap) for gap in result.gaps],
        "coverageSummary" => _plain(result.coverage_summary),
        "systemName" => result.system_name,
        "paramName" => String(result.param_name),
        "timestamp" => _serialize_timestamp(result.timestamp),
        "diagnostics" => _plain(result.diagnostics)
    )
end

function _deserialize_atlas_result(data::AbstractDict{<:AbstractString, <:Any}; brute_force::BruteForceResult)
    AtlasResult(
        brute_force,
        AtlasReconSample[_deserialize_atlas_recon_sample(item) for item in get(data, "reconSamples", Any[])],
        AtlasWindow[_deserialize_atlas_window(item) for item in get(data, "windows", Any[])],
        AtlasBranchRecord[_deserialize_atlas_branch_record(item) for item in get(data, "branchRecords", Any[])],
        AtlasGap[_deserialize_atlas_gap(item) for item in get(data, "gaps", Any[])],
        _jsonish_dict(get(data, "coverageSummary", Dict{String, Any}())),
        _as_string(get(data, "systemName", brute_force.system_name), brute_force.system_name),
        Symbol(_as_string(get(data, "paramName", String(brute_force.param_name)), String(brute_force.param_name))),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now()))),
        _jsonish_dict(get(data, "diagnostics", Dict{String, Any}()))
    )
end

const _CODIM2_CONTINUATION_FORMAT = "codim2-continuation-v1"

function _serialize_codim2_continuation_result(result::Codim2ContinuationResult)
    return Dict{String, Any}(
        "format" => _CODIM2_CONTINUATION_FORMAT,
        "primaryValues" => copy(result.primary_values),
        "secondaryValues" => copy(result.secondary_values),
        "states" => [collect(Float64, view(result.states, :, j)) for j in 1:size(result.states, 2)],
        "definingVectors" => [collect(Float64, view(result.defining_vectors, :, j)) for j in 1:size(result.defining_vectors, 2)],
        "definingVectorsImag" => [collect(Float64, view(result.defining_vectors_imag, :, j)) for j in 1:size(result.defining_vectors_imag, 2)],
        "phaseAngles" => copy(result.phase_angles),
        "fixedPointResiduals" => copy(result.fixed_point_residuals),
        "multipliers" => [[Float64[real(mu), imag(mu)] for mu in sample] for sample in result.multipliers],
        "curveFoldSecondaryValues" => copy(result.curve_fold_secondary_values),
        "seedPrimary" => result.seed_primary,
        "seedSecondary" => result.seed_secondary,
        "bifurcationKind" => String(result.bifurcation_kind),
        "period" => result.period,
        "systemName" => result.system_name,
        "paramNames" => [String(result.param_names[1]), String(result.param_names[2])],
        "engine" => String(result.engine),
        "timestamp" => _serialize_timestamp(result.timestamp),
    )
end

function _codim2_columns_to_matrix(columns)
    cols = [collect(Float64, col) for col in columns]
    n = length(cols)
    dim = isempty(cols) ? 0 : length(cols[1])
    out = Matrix{Float64}(undef, dim, n)
    for (j, col) in enumerate(cols)
        length(col) == dim || error("Serialized codim-2 state columns must all share one dimension.")
        out[:, j] = col
    end
    return out
end

function _codim2_multiplier_from_pair(pair)
    (pair isa AbstractVector && length(pair) == 2 && all(v -> v isa Real, pair)) || error(
        "Serialized codim-2 multipliers must be [re, im] pairs of reals; got $(repr(pair)).")
    return complex(Float64(pair[1]), Float64(pair[2]))
end

function _deserialize_codim2_continuation_result(data::AbstractDict)
    format = _as_string(get(data, "format", ""), "")
    format == _CODIM2_CONTINUATION_FORMAT || error(
        "Unsupported codim-2 continuation serialization format '$format'.")
    param_names = collect(get(data, "paramNames", Any[]))
    length(param_names) == 2 || error("Serialized codim-2 continuation result needs exactly two paramNames.")
    return Codim2ContinuationResult(
        collect(Float64, get(data, "primaryValues", Float64[])),
        collect(Float64, get(data, "secondaryValues", Float64[])),
        _codim2_columns_to_matrix(get(data, "states", Any[])),
        _codim2_columns_to_matrix(get(data, "definingVectors", Any[])),
        _codim2_columns_to_matrix(get(data, "definingVectorsImag", Any[])),
        collect(Float64, get(data, "phaseAngles", Float64[])),
        collect(Float64, get(data, "fixedPointResiduals", Float64[])),
        [[_codim2_multiplier_from_pair(pair) for pair in sample]
         for sample in collect(get(data, "multipliers", Any[]))],
        collect(Float64, get(data, "curveFoldSecondaryValues", Float64[])),
        Float64(get(data, "seedPrimary", NaN)),
        Float64(get(data, "seedSecondary", NaN)),
        Symbol(_as_string(get(data, "bifurcationKind", "pd"), "pd")),
        Int(get(data, "period", 1)),
        _as_string(get(data, "systemName", ""), ""),
        (Symbol(String(param_names[1])), Symbol(String(param_names[2]))),
        Symbol(_as_string(get(data, "engine", "defining_system"), "defining_system")),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now())))
    )
end

# --- Public serialization API ---
# The JSON-plain wire format for the library's public result types; external persistence layers
# build on these. The per-field sub-helpers above stay private — only these entry points are public.
"""    serialize_bruteforce_result(result::BruteForceResult) -> Dict — JSON-plain form."""
const serialize_bruteforce_result = _serialize_bruteforce_result
"""    deserialize_bruteforce_result(data::AbstractDict) -> BruteForceResult"""
const deserialize_bruteforce_result = _deserialize_bruteforce_result
"""    serialize_branch_result(result::BranchResult) -> Dict — columnar JSON-plain form (no BifurcationKit internals)."""
const serialize_branch_result = _serialize_branch_result
"""    deserialize_branch_result(data::AbstractDict) -> BranchResult"""
const deserialize_branch_result = _deserialize_branch_result
"""    serialize_atlas_result(result::AtlasResult) -> Dict — JSON-plain form."""
const serialize_atlas_result = _serialize_atlas_result
"""    deserialize_atlas_result(data::AbstractDict) -> AtlasResult"""
const deserialize_atlas_result = _deserialize_atlas_result
"""    serialize_codim2_continuation_result(result::Codim2ContinuationResult) -> Dict — JSON-plain form (multipliers stored as [re, im] pairs)."""
const serialize_codim2_continuation_result = _serialize_codim2_continuation_result
"""    deserialize_codim2_continuation_result(data::AbstractDict) -> Codim2ContinuationResult"""
const deserialize_codim2_continuation_result = _deserialize_codim2_continuation_result
