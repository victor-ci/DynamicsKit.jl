"""
Serialization of library result types to/from portable JSON-plain dicts (columnar branch points
avoid serializing fragile BifurcationKit internals). Used by the atlas library cache
(`continuation_atlas(...; cache_file=...)`) and by external persistence layers such as the
workbench's session store.
"""

_serialize_timestamp(dt::DateTime) = Dates.format(dt, dateformat"yyyy-mm-ddTHH:MM:SS.s")
_deserialize_timestamp(value) = DateTime(String(value))

const _MAP_NORMAL_FORM_FORMAT = "map-normal-form-v1"
const _MAP_SPECIAL_POINT_FORMAT = "map-special-point-v1"
const _BORDER_COLLISION_CLASSIFICATION_FORMAT = "border-collision-classification-v1"
const _BORDER_COLLISION_POINT_FORMAT = "border-collision-point-v1"

function _require_serialized_fields(data::AbstractDict, fields, label::AbstractString)
    missing = filter(field -> !haskey(data, field), fields)
    isempty(missing) || error(
        "Serialized $label is missing required fields: $(join(missing, ", ")).")
end

function _validate_map_normal_form(normal_form::MapNormalForm)
    expected_name = _normal_form_name(normal_form.kind)
    normal_form.coefficient_name === expected_name || error(
        "Map normal form kind $(normal_form.kind) requires coefficient_name $expected_name.")
    normal_form.convention == _normal_form_convention(normal_form.kind) || error(
        "Map normal form convention does not match the declared $(normal_form.kind) kind.")
    coefficient = normal_form.coefficient
    coefficient === nothing || isfinite(coefficient) || error(
        "Map normal-form coefficient must be finite when present.")

    if normal_form.status === :ok
        coefficient === nothing && error("A map normal form with status :ok requires a coefficient.")
        allowed = normal_form.kind === :fold ? (:nondegenerate,) :
                  (:supercritical, :subcritical)
        normal_form.criticality in allowed || error(
            "Map normal-form status :ok has invalid criticality $(normal_form.criticality).")
    elseif normal_form.status === :degenerate
        coefficient === nothing && error(
            "A degenerate map normal form requires its evaluated coefficient.")
        normal_form.criticality === :degenerate || error(
            "Map normal-form status :degenerate requires criticality :degenerate.")
    elseif normal_form.status in (
            :near_singular, :strong_resonance, :not_critical,
            :critical_eigenvector_unavailable, :derivative_failed,
            :conjugate_pair_unavailable, :multiple_critical_pairs,
            :fd_step_unstable)
        coefficient === nothing || error(
            "Unavailable map normal forms must not carry a coefficient.")
        normal_form.criticality === :unclassified || error(
            "Unavailable map normal forms require criticality :unclassified.")
    else
        error("Unknown map normal-form status $(repr(normal_form.status)).")
    end
    return normal_form
end

function _serialize_map_normal_form(normal_form::MapNormalForm)
    _validate_map_normal_form(normal_form)
    return Dict{String, Any}(
        "format" => _MAP_NORMAL_FORM_FORMAT,
        "kind" => String(normal_form.kind),
        "coefficientName" => String(normal_form.coefficient_name),
        "coefficient" => normal_form.coefficient,
        "criticality" => String(normal_form.criticality),
        "status" => String(normal_form.status),
        "convention" => normal_form.convention,
    )
end

function _deserialize_map_normal_form(data::AbstractDict)
    _require_serialized_fields(
        data,
        ("format", "kind", "coefficientName", "coefficient", "criticality",
         "status", "convention"),
        "map normal form")
    format = _as_string(get(data, "format", ""), "")
    format == _MAP_NORMAL_FORM_FORMAT || error(
        "Unsupported map normal-form serialization format '$format'.")
    kind = Symbol(_as_string(get(data, "kind", ""), ""))
    coefficient_name = Symbol(_as_string(get(data, "coefficientName", ""), ""))
    coefficient = get(data, "coefficient", nothing)
    expected_name = _normal_form_name(kind)
    coefficient_name === expected_name || error(
        "Serialized map normal form kind $kind requires coefficientName $expected_name; got $coefficient_name.")
    coefficient === nothing || coefficient isa Real || error(
        "Serialized map normal-form coefficient must be a real number or nothing.")
    value = coefficient === nothing ? nothing : Float64(coefficient)
    value === nothing || isfinite(value) || error(
        "Serialized map normal-form coefficient must be finite when present.")
    return _validate_map_normal_form(MapNormalForm(
        kind,
        coefficient_name,
        value,
        Symbol(_as_string(get(data, "criticality", "unclassified"), "unclassified")),
        Symbol(_as_string(get(data, "status", ""), "")),
        _as_string(get(data, "convention", ""), ""),
    ))
end

function _serialize_map_special_point(point::MapSpecialPoint)
    isfinite(point.param) || error("Map special-point param must be finite.")
    all(isfinite, point.state) || error("Map special-point state must be finite.")
    all(value -> isfinite(real(value)) && isfinite(imag(value)), point.multipliers) ||
        error("Map special-point multipliers must be finite.")
    isfinite(real(point.critical_multiplier)) &&
        isfinite(imag(point.critical_multiplier)) || error(
            "Map special-point critical multiplier must be finite.")
    isfinite(point.test_value) || error("Map special-point test value must be finite.")
    point.period >= 1 || error("Map special-point period must be >= 1.")
    point.kind in (:fold, :pd, :ns) || error(
        "Map special-point kind must be fold, pd, or ns.")
    point.normal_form === nothing || point.normal_form.kind === point.kind || error(
        "Map special-point normal-form kind must match the point kind.")
    return Dict{String, Any}(
        "format" => _MAP_SPECIAL_POINT_FORMAT,
        "kind" => String(point.kind),
        "param" => point.param,
        "state" => copy(point.state),
        "multipliers" => [[real(value), imag(value)] for value in point.multipliers],
        "criticalMultiplier" => [
            real(point.critical_multiplier), imag(point.critical_multiplier)],
        "testValue" => point.test_value,
        "period" => point.period,
        "converged" => point.converged,
        "normalForm" => point.normal_form === nothing ? nothing :
                        _serialize_map_normal_form(point.normal_form),
    )
end

function _deserialize_map_special_point(data::AbstractDict)
    _require_serialized_fields(
        data,
        ("format", "kind", "param", "state", "multipliers",
         "criticalMultiplier", "testValue", "period", "converged", "normalForm"),
        "map special point")
    format = _as_string(get(data, "format", ""), "")
    format == _MAP_SPECIAL_POINT_FORMAT || error(
        "Unsupported map special-point serialization format '$format'.")
    pair(value, field) = begin
        value isa AbstractVector && length(value) == 2 &&
            all(item -> item isa Real, value) || error(
                "Serialized map special-point $field must be a [re, im] pair.")
        result = complex(Float64(value[1]), Float64(value[2]))
        isfinite(real(result)) && isfinite(imag(result)) || error(
            "Serialized map special-point $field must be finite.")
        result
    end
    normal_form_data = get(data, "normalForm", nothing)
    normal_form = normal_form_data === nothing ? nothing :
                  _deserialize_map_normal_form(normal_form_data)
    period = _as_int(get(data, "period", 0), 0)
    period >= 1 || error("Serialized map special-point period must be >= 1; got $period.")
    state = collect(Float64, get(data, "state", Float64[]))
    all(isfinite, state) || error("Serialized map special-point state must be finite.")
    multipliers = ComplexF64[
        pair(value, "multipliers entry") for value in get(data, "multipliers", Any[])]
    critical = pair(get(data, "criticalMultiplier", Any[]), "criticalMultiplier")
    param = _as_float(get(data, "param", NaN), NaN)
    test_value = _as_float(get(data, "testValue", NaN), NaN)
    isfinite(param) || error("Serialized map special-point param must be finite.")
    isfinite(test_value) || error("Serialized map special-point testValue must be finite.")
    kind = Symbol(_as_string(get(data, "kind", ""), ""))
    kind in (:fold, :pd, :ns) || error(
        "Serialized map special-point kind must be fold, pd, or ns; got $(repr(kind)).")
    normal_form === nothing || normal_form.kind === kind || error(
        "Serialized map special-point normal form kind does not match point kind $kind.")
    get(data, "converged", nothing) isa Bool || error(
        "Serialized map special-point converged must be a boolean.")
    return MapSpecialPoint(
        kind, param, state, multipliers, critical, test_value, period,
        data["converged"], normal_form)
end

function _serialize_branch_point(point)
    Dict(String(name) => _plain(getproperty(point, name)) for name in propertynames(point))
end

_bcb_complex_pairs(values) = [[real(v), imag(v)] for v in values]

function _bcb_matrix_plain(matrix::AbstractMatrix)
    return [collect(Float64, row) for row in eachrow(matrix)]
end

function _bcb_read_matrix(value, field::AbstractString)
    value === nothing && return Matrix{Float64}(undef, 0, 0)
    value isa AbstractVector || error(
        "Serialized border-collision $field must be an array of rows.")
    isempty(value) && return Matrix{Float64}(undef, 0, 0)
    rows = Vector{Vector{Float64}}()
    ncol = -1
    for row in value
        row isa AbstractVector || error(
            "Serialized border-collision $field rows must be arrays.")
        vals = Float64[_as_float(item, NaN) for item in row]
        if ncol < 0
            ncol = length(vals)
        elseif length(vals) != ncol
            error("Serialized border-collision $field rows must have equal length.")
        end
        push!(rows, vals)
    end
    matrix = Matrix{Float64}(undef, length(rows), ncol)
    for (i, row) in enumerate(rows)
        matrix[i, :] = row
    end
    return matrix
end

function _bcb_read_pairs(value, field::AbstractString)
    value === nothing && return ComplexF64[]
    value isa AbstractVector || error(
        "Serialized border-collision $field must be an array of [re, im] pairs.")
    out = ComplexF64[]
    for entry in value
        entry isa AbstractVector && length(entry) == 2 || error(
            "Serialized border-collision $field entries must be [re, im] pairs.")
        push!(out, complex(_as_float(entry[1], NaN), _as_float(entry[2], NaN)))
    end
    return out
end

_bcb_opt_bool(value) = value === nothing ? nothing : _as_bool(value, false)
_bcb_opt_int(value) = value === nothing ? nothing : _as_int(value, 0)
_bcb_opt_float(value) = value === nothing ? nothing : _as_float(value, NaN)

function _serialize_border_collision_classification(c::BorderCollisionClassification)
    return Dict{String, Any}(
        "format"                  => _BORDER_COLLISION_CLASSIFICATION_FORMAT,
        "scenario"                => String(c.scenario),
        "status"                  => String(c.status),
        "period"                  => c.period,
        "detIMinusL"              => c.det_I_minus_L,
        "detIMinusR"              => c.det_I_minus_R,
        "detIPlusL"               => c.det_I_plus_L,
        "detIPlusR"               => c.det_I_plus_R,
        "persistenceProduct"      => c.persistence_product,
        "persistenceSign"         => c.persistence_sign,
        "companionProduct"        => c.companion_product,
        "companionSign"           => c.companion_sign,
        "sigmaPlusL"              => c.sigma_plus_L,
        "sigmaPlusR"              => c.sigma_plus_R,
        "sigmaMinusL"             => c.sigma_minus_L,
        "sigmaMinusR"             => c.sigma_minus_R,
        "sigmaReliable"           => c.sigma_reliable,
        "spectrumL"               => _bcb_complex_pairs(c.spectrum_L),
        "spectrumR"               => _bcb_complex_pairs(c.spectrum_R),
        "stableL"                 => c.stable_L,
        "stableR"                 => c.stable_R,
        "spectralRadiusL"         => c.spectral_radius_L,
        "spectralRadiusR"         => c.spectral_radius_R,
        "companionExists"         => c.companion_exists,
        "companionAdmissible"     => c.companion_admissible,
        "companionStable"         => c.companion_stable,
        "companionSpectralRadius" => c.companion_spectral_radius,
        "companionMultipliers"    => _bcb_complex_pairs(c.companion_multipliers),
        "transversal"             => c.transversal,
        "transversalityMeasure"   => c.transversality_measure,
        "continuous"              => c.continuous,
        "continuityResidual"      => c.continuity_residual,
        "continuityTolerance"     => c.continuity_tolerance,
        "generic"                 => c.generic,
        "jacobianL"               => _bcb_matrix_plain(c.jacobian_L),
        "jacobianR"               => _bcb_matrix_plain(c.jacobian_R),
        "inference"               => c.inference,
        "warnings"                => copy(c.warnings),
        "convention"              => c.convention,
    )
end

function _deserialize_border_collision_classification(data::AbstractDict)
    _require_serialized_fields(
        data,
        ("format", "scenario", "status", "period",
         "detIMinusL", "detIMinusR", "detIPlusL", "detIPlusR",
         "persistenceProduct", "persistenceSign", "companionProduct", "companionSign",
         "spectrumL", "spectrumR", "jacobianL", "jacobianR",
         "generic", "continuityTolerance", "inference", "warnings", "convention"),
        "border collision classification")
    format = _as_string(get(data, "format", ""), "")
    format == _BORDER_COLLISION_CLASSIFICATION_FORMAT || error(
        "Unsupported border-collision classification serialization format '$format'.")
    period = _as_int(get(data, "period", 0), 0)
    period >= 1 || error(
        "Serialized border-collision classification period must be >= 1; got $period.")
    warnings = String[_as_string(w, "") for w in get(data, "warnings", Any[])]
    return BorderCollisionClassification(
        scenario = Symbol(_as_string(data["scenario"], "")),
        status = Symbol(_as_string(data["status"], "")),
        period = period,
        det_I_minus_L = _as_float(data["detIMinusL"], NaN),
        det_I_minus_R = _as_float(data["detIMinusR"], NaN),
        det_I_plus_L = _as_float(data["detIPlusL"], NaN),
        det_I_plus_R = _as_float(data["detIPlusR"], NaN),
        persistence_product = _as_float(data["persistenceProduct"], NaN),
        persistence_sign = _as_int(data["persistenceSign"], 0),
        companion_product = _as_float(data["companionProduct"], NaN),
        companion_sign = _as_int(data["companionSign"], 0),
        sigma_plus_L = _bcb_opt_int(get(data, "sigmaPlusL", nothing)),
        sigma_plus_R = _bcb_opt_int(get(data, "sigmaPlusR", nothing)),
        sigma_minus_L = _bcb_opt_int(get(data, "sigmaMinusL", nothing)),
        sigma_minus_R = _bcb_opt_int(get(data, "sigmaMinusR", nothing)),
        sigma_reliable = _as_bool(get(data, "sigmaReliable", false), false),
        spectrum_L = _bcb_read_pairs(get(data, "spectrumL", nothing), "spectrumL"),
        spectrum_R = _bcb_read_pairs(get(data, "spectrumR", nothing), "spectrumR"),
        stable_L = _bcb_opt_bool(get(data, "stableL", nothing)),
        stable_R = _bcb_opt_bool(get(data, "stableR", nothing)),
        spectral_radius_L = _as_float(get(data, "spectralRadiusL", NaN), NaN),
        spectral_radius_R = _as_float(get(data, "spectralRadiusR", NaN), NaN),
        companion_exists = _bcb_opt_bool(get(data, "companionExists", nothing)),
        companion_admissible = _bcb_opt_bool(get(data, "companionAdmissible", nothing)),
        companion_stable = _bcb_opt_bool(get(data, "companionStable", nothing)),
        companion_spectral_radius = _bcb_opt_float(get(data, "companionSpectralRadius", nothing)),
        companion_multipliers = _bcb_read_pairs(
            get(data, "companionMultipliers", nothing), "companionMultipliers"),
        transversal = _bcb_opt_bool(get(data, "transversal", nothing)),
        transversality_measure = _bcb_opt_float(get(data, "transversalityMeasure", nothing)),
        continuous = _bcb_opt_bool(get(data, "continuous", nothing)),
        continuity_residual = _bcb_opt_float(get(data, "continuityResidual", nothing)),
        continuity_tolerance = _as_float(data["continuityTolerance"], NaN),
        generic = _as_bool(data["generic"], false),
        jacobian_L = _bcb_read_matrix(data["jacobianL"], "jacobianL"),
        jacobian_R = _bcb_read_matrix(data["jacobianR"], "jacobianR"),
        inference = _as_string(data["inference"], ""),
        warnings = warnings,
        convention = _as_string(data["convention"], ""),
    )
end

function _serialize_border_collision_point(point::BorderCollisionPoint)
    return Dict{String, Any}(
        "format"          => _BORDER_COLLISION_POINT_FORMAT,
        "param"           => point.param,
        "orbit"           => [collect(Float64, phase) for phase in point.orbit],
        "collidingPhase"  => point.colliding_phase,
        "itinerary"       => copy(point.itinerary),
        "eventName"       => point.event_name,
        "guardComponent"  => point.guard_component,
        "guardValues"     => copy(point.guard_values),
        "period"          => point.period,
        "classification"  => _serialize_border_collision_classification(point.classification),
        "converged"       => point.converged,
    )
end

function _deserialize_border_collision_point(data::AbstractDict)
    _require_serialized_fields(
        data,
        ("format", "param", "orbit", "collidingPhase", "itinerary", "eventName",
         "guardComponent", "guardValues", "period", "classification", "converged"),
        "border collision point")
    format = _as_string(get(data, "format", ""), "")
    format == _BORDER_COLLISION_POINT_FORMAT || error(
        "Unsupported border-collision point serialization format '$format'.")
    orbit_data = get(data, "orbit", Any[])
    orbit_data isa AbstractVector || error(
        "Serialized border-collision point orbit must be an array of phases.")
    orbit = [collect(Float64, phase) for phase in orbit_data]
    period = _as_int(get(data, "period", 0), 0)
    period >= 1 || error(
        "Serialized border-collision point period must be >= 1; got $period.")
    get(data, "converged", nothing) isa Bool || error(
        "Serialized border-collision point converged must be a boolean.")
    classification = _deserialize_border_collision_classification(data["classification"])
    return BorderCollisionPoint(
        _as_float(data["param"], NaN),
        orbit,
        _as_int(data["collidingPhase"], 0),
        Int[_as_int(v, 0) for v in get(data, "itinerary", Any[])],
        _as_string(data["eventName"], ""),
        _as_int(data["guardComponent"], 1),
        Float64[_as_float(v, NaN) for v in get(data, "guardValues", Any[])],
        period,
        classification,
        data["converged"],
    )
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

const _ROBUST_CHAOS_FORMAT = "robust-chaos-certificate-v1"
const _ROBUST_CHAOS_EVIDENCE_FORMAT = "robust-chaos-evidence-v1"

function _serialize_stable_window_evidence(e::StableWindowEvidence)::Dict{String, Any}
    return Dict{String, Any}(
        "branchId"          => e.branch_id,
        "windowId"          => e.window_id,
        "period"            => e.period,
        "paramMin"          => e.param_min,
        "paramMax"          => e.param_max,
        "stableSampleCount" => e.stable_sample_count,
    )
end

function _deserialize_stable_window_evidence(data::AbstractDict)::StableWindowEvidence
    required = ("branchId", "windowId", "period", "paramMin", "paramMax", "stableSampleCount")
    missing = filter(key -> !haskey(data, key), required)
    isempty(missing) || error(
        "Serialized stable-window evidence is missing required fields: $(join(missing, ", ")).")
    branch_id = _as_string(data["branchId"], "")
    window_id = _as_string(data["windowId"], "")
    period = _as_int(data["period"], 0)
    param_min = _as_float(data["paramMin"], NaN)
    param_max = _as_float(data["paramMax"], NaN)
    stable_sample_count = _as_int(data["stableSampleCount"], 0)
    isempty(branch_id) && error("Serialized stable-window evidence requires a non-empty branchId.")
    isempty(window_id) && error("Serialized stable-window evidence requires a non-empty windowId.")
    period >= 1 || error("Serialized stable-window evidence period must be >= 1; got $period.")
    isfinite(param_min) && isfinite(param_max) || error(
        "Serialized stable-window evidence parameter bounds must be finite.")
    param_min <= param_max || error(
        "Serialized stable-window evidence requires paramMin <= paramMax; got $param_min > $param_max.")
    stable_sample_count >= 1 || error(
        "Serialized stable-window evidence stableSampleCount must be >= 1; got $stable_sample_count.")
    return StableWindowEvidence(
        branch_id,
        window_id,
        period,
        param_min,
        param_max,
        stable_sample_count,
    )
end

function _serialize_robust_chaos_certificate(cert::RobustChaosCertificate)::Dict{String, Any}
    return Dict{String, Any}(
        "format"                      => _ROBUST_CHAOS_FORMAT,
        "paramMin"                    => cert.param_min,
        "paramMax"                    => cert.param_max,
        "systemName"                  => cert.system_name,
        "paramIndex"                  => cert.param_index,
        "lyapunovVerdict"             => String(cert.lyapunov_verdict),
        "atlasVerdict"                => String(cert.atlas_verdict),
        "basinVerdict"                => String(cert.basin_verdict),
        "overallVerdict"              => String(cert.overall_verdict),
        "lyapunovPositiveFraction"    => cert.lyapunov_positive_fraction,
        "lyapunovResolvedFraction"    => cert.lyapunov_resolved_fraction,
        "lyapunovMinResolvedExponent" => cert.lyapunov_min_resolved_exponent,
        "lyapunovNTotal"              => cert.lyapunov_n_total,
        "lyapunovNResolved"           => cert.lyapunov_n_resolved,
        "lyapunovNPositive"           => cert.lyapunov_n_positive,
        "atlasSearchedPeriods"        => copy(cert.atlas_searched_periods),
        "atlasSearchComplete"         => cert.atlas_search_complete,
        "atlasCoverageEffort"         => cert.atlas_coverage_effort,
        "atlasNWindows"               => cert.atlas_n_windows,
        "atlasNCovered"               => cert.atlas_n_covered,
        "atlasNPartial"               => cert.atlas_n_partial,
        "atlasNUnresolved"            => cert.atlas_n_unresolved,
        "atlasNGaps"                  => cert.atlas_n_gaps,
        "atlasUnresolvedStabilityCount" => cert.atlas_unresolved_stability_count,
        "stableEvidence"              => [_serialize_stable_window_evidence(e) for e in cert.stable_evidence],
        "basinParam"                  => cert.basin_param,
        "basinChaoticFraction"        => cert.basin_chaotic_fraction,
        "basinResolvedFraction"       => cert.basin_resolved_fraction,
        "basinNTotal"                 => cert.basin_n_total,
        "basinNResolved"              => cert.basin_n_resolved,
        "basinNChaotic"               => cert.basin_n_chaotic,
        "basinClassCounts"            => Dict{String, Any}(String(k) => v for (k, v) in cert.basin_class_counts),
        "robustnessScore"             => cert.robustness_score,
        "certificateItems"            => copy(cert.certificate_items),
        "timestamp"                   => _serialize_timestamp(cert.timestamp),
    )
end

function _deserialize_robust_chaos_certificate(data::AbstractDict)::RobustChaosCertificate
    format = _as_string(get(data, "format", ""), "")
    format == _ROBUST_CHAOS_FORMAT || error(
        "Unsupported robust-chaos certificate format '$format'; expected '$(_ROBUST_CHAOS_FORMAT)'.")
    haskey(data, "timestamp") || error(
        "Serialized robust-chaos certificate format '$(_ROBUST_CHAOS_FORMAT)' requires a timestamp.")
    basin_class_counts = Dict{Symbol, Int}(
        Symbol(String(k)) => _as_int(v)
        for (k, v) in get(data, "basinClassCounts", Dict{String, Any}())
    )
    items = [Dict{String, Any}(String(k) => v for (k, v) in item)
             for item in collect(get(data, "certificateItems", Any[]))]
    return RobustChaosCertificate(
        _as_float(get(data, "paramMin", NaN)),
        _as_float(get(data, "paramMax", NaN)),
        _as_string(get(data, "systemName", ""), ""),
        _as_int(get(data, "paramIndex", 1)),
        Symbol(_as_string(get(data, "lyapunovVerdict", "inconclusive"), "inconclusive")),
        Symbol(_as_string(get(data, "atlasVerdict",    "inconclusive"), "inconclusive")),
        Symbol(_as_string(get(data, "basinVerdict",    "inconclusive"), "inconclusive")),
        Symbol(_as_string(get(data, "overallVerdict",  "inconclusive"), "inconclusive")),
        _as_float(get(data, "lyapunovPositiveFraction",    0.0)),
        _as_float(get(data, "lyapunovResolvedFraction",    0.0)),
        _as_float(get(data, "lyapunovMinResolvedExponent", NaN)),
        _as_int(get(data, "lyapunovNTotal",    0)),
        _as_int(get(data, "lyapunovNResolved", 0)),
        _as_int(get(data, "lyapunovNPositive", 0)),
        Int[_as_int(x) for x in collect(get(data, "atlasSearchedPeriods", Any[]))],
        _as_bool(get(data, "atlasSearchComplete",  false)),
        _as_float(get(data, "atlasCoverageEffort", 0.0)),
        _as_int(get(data, "atlasNWindows",    0)),
        _as_int(get(data, "atlasNCovered",    0)),
        _as_int(get(data, "atlasNPartial",    0)),
        _as_int(get(data, "atlasNUnresolved", 0)),
        _as_int(get(data, "atlasNGaps",       0)),
        _as_int(get(data, "atlasUnresolvedStabilityCount", 0)),
        StableWindowEvidence[_deserialize_stable_window_evidence(e)
                              for e in collect(get(data, "stableEvidence", Any[]))],
        _as_float(get(data, "basinParam",            NaN)),
        _as_float(get(data, "basinChaoticFraction",  0.0)),
        _as_float(get(data, "basinResolvedFraction", 0.0)),
        _as_int(get(data, "basinNTotal",    0)),
        _as_int(get(data, "basinNResolved", 0)),
        _as_int(get(data, "basinNChaotic",  0)),
        basin_class_counts,
        _as_float(get(data, "robustnessScore", 0.0)),
        items,
        _deserialize_timestamp(data["timestamp"])
    )
end

function _serialize_robust_chaos_lyapunov(result::LyapunovDiagramResult)
    return Dict{String, Any}(
        "params" => copy(result.params),
        "exponents" => [isfinite(value) ? value : nothing for value in result.exponents],
        "classifications" => String.(result.classifications),
        "estimationStatuses" => String.(result.estimation_statuses),
        "sampleCounts" => copy(result.sample_counts),
        "neutralTolerance" => result.neutral_tolerance,
        "systemName" => result.system_name,
        "paramName" => String(result.param_name),
        "timestamp" => _serialize_timestamp(result.timestamp),
    )
end

function _deserialize_robust_chaos_lyapunov(data::AbstractDict)
    params = Float64[_as_float(value) for value in get(data, "params", Any[])]
    exponents = Float64[
        isnothing(value) ? NaN : _as_float(value, NaN)
        for value in get(data, "exponents", Any[])
    ]
    classifications = Symbol[Symbol(_as_string(value, "unresolved"))
                             for value in get(data, "classifications", Any[])]
    statuses = Symbol[Symbol(_as_string(value, "uncomputed"))
                      for value in get(data, "estimationStatuses", Any[])]
    sample_counts = Int[_as_int(value, 0) for value in get(data, "sampleCounts", Any[])]
    lengths = (length(params), length(exponents), length(classifications),
               length(statuses), length(sample_counts))
    all(==(first(lengths)), lengths) || error(
        "Serialized robust-chaos Lyapunov evidence vectors must have equal lengths; got $(lengths).")
    return LyapunovDiagramResult(
        params,
        exponents,
        classifications,
        statuses,
        sample_counts,
        _as_float(get(data, "neutralTolerance", 1e-3), 1e-3),
        _as_string(get(data, "systemName", ""), ""),
        Symbol(_as_string(get(data, "paramName", "p"), "p")),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now()))),
    )
end

function _serialize_robust_chaos_bruteforce(result::BruteForceResult)
    return Dict{String, Any}(
        "params" => copy(result.params),
        "pointValues" => vec(copy(result.points)),
        "pointShape" => [size(result.points, 1), size(result.points, 2)],
        "systemName" => result.system_name,
        "paramName" => String(result.param_name),
        "timestamp" => _serialize_timestamp(result.timestamp),
    )
end

function _deserialize_robust_chaos_bruteforce(data::AbstractDict)
    shape = Int[_as_int(value, -1) for value in get(data, "pointShape", Any[])]
    length(shape) == 2 && all(>=(0), shape) || error(
        "Serialized robust-chaos brute-force pointShape must contain two non-negative dimensions.")
    values = Float64[_as_float(value) for value in get(data, "pointValues", Any[])]
    length(values) == prod(shape) || error(
        "Serialized robust-chaos brute-force pointValues length $(length(values)) " *
        "does not match pointShape $(Tuple(shape)).")
    return BruteForceResult(
        Float64[_as_float(value) for value in get(data, "params", Any[])],
        reshape(values, Tuple(shape)),
        _as_string(get(data, "systemName", ""), ""),
        Symbol(_as_string(get(data, "paramName", "p"), "p")),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now()))),
    )
end

function _serialize_robust_chaos_basins(result::BasinsResult)
    return Dict{String, Any}(
        "xGrid" => copy(result.x_grid),
        "yGrid" => copy(result.y_grid),
        "periodicity" => [collect(view(result.periodicity, i, :))
                          for i in axes(result.periodicity, 1)],
        "bifParam" => result.bif_param,
        "maxPeriod" => result.max_period,
        "systemName" => result.system_name,
        "timestamp" => _serialize_timestamp(result.timestamp),
        "xIndex" => result.x_index,
        "yIndex" => result.y_index,
        "icTemplate" => copy(result.ic_template),
    )
end

function _deserialize_robust_chaos_matrix(raw, ::Type{T}, convert_value::Function,
                                          label::AbstractString) where T
    raw isa AbstractMatrix && return T.(map(convert_value, raw))
    rows = collect(raw)
    isempty(rows) && return Matrix{T}(undef, 0, 0)
    all(row -> row isa AbstractVector, rows) || error(
        "Serialized robust-chaos $label must be a vector of rows.")
    width = length(first(rows))
    all(row -> length(row) == width, rows) || error(
        "Serialized robust-chaos $label rows must have equal lengths.")
    converted = T[convert_value(value) for row in rows for value in row]
    return permutedims(reshape(converted, width, length(rows)))
end

function _deserialize_robust_chaos_basins(data::AbstractDict)
    periodicity = _deserialize_robust_chaos_matrix(
        get(data, "periodicity", Any[]),
        Int,
        value -> _as_int(value, 0),
        "basin periodicity",
    )
    return BasinsResult(
        Float64[_as_float(value) for value in get(data, "xGrid", Any[])],
        Float64[_as_float(value) for value in get(data, "yGrid", Any[])],
        periodicity,
        _as_float(get(data, "bifParam", NaN), NaN),
        _as_int(get(data, "maxPeriod", 0), 0),
        _as_string(get(data, "systemName", ""), ""),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now()))),
        _as_int(get(data, "xIndex", 1), 1),
        _as_int(get(data, "yIndex", 2), 2),
        Float64[_as_float(value) for value in get(data, "icTemplate", Any[])],
    )
end

function _serialize_robust_chaos_evidence(evidence::RobustChaosEvidence)::Dict{String, Any}
    classifications = [
        String.(collect(view(evidence.basin_classifications, i, :)))
        for i in axes(evidence.basin_classifications, 1)
    ]
    return Dict{String, Any}(
        "format" => _ROBUST_CHAOS_EVIDENCE_FORMAT,
        "certificate" => _serialize_robust_chaos_certificate(evidence.certificate),
        "lyapunov" => _serialize_robust_chaos_lyapunov(evidence.lyapunov),
        "atlasBruteForce" => _serialize_robust_chaos_bruteforce(evidence.atlas.brute_force),
        "atlas" => _serialize_atlas_result(evidence.atlas),
        "basins" => _serialize_robust_chaos_basins(evidence.basins),
        "basinClassifications" => classifications,
    )
end

function _deserialize_robust_chaos_evidence(data::AbstractDict)::RobustChaosEvidence
    format = _as_string(get(data, "format", ""), "")
    format == _ROBUST_CHAOS_EVIDENCE_FORMAT || error(
        "Unsupported robust-chaos evidence format '$format'; expected '$(_ROBUST_CHAOS_EVIDENCE_FORMAT)'.")
    required = ("certificate", "lyapunov", "atlasBruteForce", "atlas", "basins",
                "basinClassifications")
    missing = filter(key -> !haskey(data, key), required)
    isempty(missing) || error(
        "Serialized robust-chaos evidence is missing required fields: $(join(missing, ", ")).")
    brute_force = _deserialize_robust_chaos_bruteforce(
        _jsonish_dict(data["atlasBruteForce"]))
    atlas = _deserialize_atlas_result(_jsonish_dict(data["atlas"]); brute_force=brute_force)
    basins = _deserialize_robust_chaos_basins(_jsonish_dict(data["basins"]))
    classifications = _deserialize_robust_chaos_matrix(
        data["basinClassifications"],
        Symbol,
        value -> Symbol(_as_string(value, "unresolved")),
        "basin classifications",
    )
    size(classifications) == size(basins.periodicity) || error(
        "Serialized robust-chaos basin classifications must match basin periodicity shape; " *
        "got $(size(classifications)) and $(size(basins.periodicity)).")
    return RobustChaosEvidence(
        _deserialize_robust_chaos_certificate(_jsonish_dict(data["certificate"])),
        _deserialize_robust_chaos_lyapunov(_jsonish_dict(data["lyapunov"])),
        atlas,
        basins,
        classifications,
    )
end

const _BRANCH_REACHABILITY_FORMAT = "branch-reachability-v1"

_serialize_reach_int_matrix(m::AbstractMatrix{<:Integer}) =
    [collect(view(m, i, :)) for i in axes(m, 1)]

_serialize_reach_float_matrix(m::AbstractMatrix{<:Real}) =
    [[isfinite(value) ? value : nothing for value in view(m, i, :)] for i in axes(m, 1)]

function _serialize_branch_reachability_sample(sample::BranchReachabilitySample)::Dict{String, Any}
    return Dict{String, Any}(
        "param" => sample.param,
        "branchIds" => copy(sample.branch_ids),
        "branchPeriods" => copy(sample.branch_periods),
        "branchStable" => copy(sample.branch_stable),
        "branchCovered" => copy(sample.branch_covered),
        "branchConfidence" => copy(sample.branch_confidence),
        "matchedCounts" => copy(sample.matched_counts),
        "matchedFractions" => copy(sample.matched_fractions),
        "nSeeds" => sample.n_seeds,
        "nMatched" => sample.n_matched,
        "nUnmatched" => sample.n_unmatched,
        "nAperiodic" => sample.n_aperiodic,
        "nDiverged" => sample.n_diverged,
        "nUnresolved" => sample.n_unresolved,
        "nStabilityMismatch" => sample.n_stability_mismatch,
        "nOutsideCoverage" => sample.n_outside_coverage,
        "assignment" => _serialize_reach_int_matrix(sample.assignment),
        "status" => _serialize_reach_int_matrix(sample.status),
        "matchDistance" => _serialize_reach_float_matrix(sample.match_distance),
        "terminalPeriod" => _serialize_reach_int_matrix(sample.terminal_period),
        "diagnostics" => sample.diagnostics,
    )
end

function _deserialize_branch_reachability_sample(data::AbstractDict)::BranchReachabilitySample
    assignment = _deserialize_robust_chaos_matrix(
        get(data, "assignment", Any[]), Int, value -> _as_int(value, 0), "reachability assignment")
    status = _deserialize_robust_chaos_matrix(
        get(data, "status", Any[]), Int, value -> _as_int(value, 0), "reachability status")
    match_distance = _deserialize_robust_chaos_matrix(
        get(data, "matchDistance", Any[]), Float64,
        value -> isnothing(value) ? NaN : _as_float(value, NaN), "reachability match distance")
    terminal_period = _deserialize_robust_chaos_matrix(
        get(data, "terminalPeriod", Any[]), Int, value -> _as_int(value, 0), "reachability terminal period")
    return BranchReachabilitySample(
        _as_float(get(data, "param", NaN), NaN),
        String[_as_string(value, "") for value in get(data, "branchIds", Any[])],
        Int[_as_int(value, 0) for value in get(data, "branchPeriods", Any[])],
        Bool[_as_bool(value, false) for value in get(data, "branchStable", Any[])],
        Bool[_as_bool(value, false) for value in get(data, "branchCovered", Any[])],
        Float64[_as_float(value, NaN) for value in get(data, "branchConfidence", Any[])],
        Int[_as_int(value, 0) for value in get(data, "matchedCounts", Any[])],
        Float64[_as_float(value, 0.0) for value in get(data, "matchedFractions", Any[])],
        _as_int(get(data, "nSeeds", 0), 0),
        _as_int(get(data, "nMatched", 0), 0),
        _as_int(get(data, "nUnmatched", 0), 0),
        _as_int(get(data, "nAperiodic", 0), 0),
        _as_int(get(data, "nDiverged", 0), 0),
        _as_int(get(data, "nUnresolved", 0), 0),
        _as_int(get(data, "nStabilityMismatch", 0), 0),
        _as_int(get(data, "nOutsideCoverage", 0), 0),
        assignment,
        status,
        match_distance,
        terminal_period,
        _jsonish_dict(get(data, "diagnostics", Dict{String, Any}())),
    )
end

function _serialize_branch_reachability_result(result::BranchReachabilityResult)::Dict{String, Any}
    return Dict{String, Any}(
        "format" => _BRANCH_REACHABILITY_FORMAT,
        "systemName" => result.system_name,
        "paramName" => String(result.param_name),
        "paramIndex" => result.param_index,
        "linkedParamIndices" => copy(result.linked_param_indices),
        "baseParams" => copy(result.base_params),
        "xGrid" => copy(result.x_grid),
        "yGrid" => copy(result.y_grid),
        "xIndex" => result.x_index,
        "yIndex" => result.y_index,
        "icTemplate" => copy(result.ic_template),
        "maxPeriod" => result.max_period,
        "precision" => result.precision,
        "divergenceCutoff" => isfinite(result.divergence_cutoff) ? result.divergence_cutoff : nothing,
        "paramTolerance" => result.param_tolerance,
        "matchTolerance" => result.match_tolerance,
        "ambiguityRatio" => result.ambiguity_ratio,
        "stabilityTol" => result.stability_tol,
        "branchIds" => copy(result.branch_ids),
        "branchPeriods" => copy(result.branch_periods),
        "statusLabels" => Dict{String, Any}(string(code) => label for (code, label) in result.status_labels),
        "samples" => [_serialize_branch_reachability_sample(sample) for sample in result.samples],
        "timestamp" => _serialize_timestamp(result.timestamp),
    )
end

function _deserialize_branch_reachability_result(data::AbstractDict)::BranchReachabilityResult
    format = _as_string(get(data, "format", ""), "")
    format == _BRANCH_REACHABILITY_FORMAT || error(
        "Unsupported branch-reachability format '$format'; expected '$(_BRANCH_REACHABILITY_FORMAT)'.")
    raw_cutoff = get(data, "divergenceCutoff", nothing)
    divergence_cutoff = isnothing(raw_cutoff) ? Inf : _as_float(raw_cutoff, Inf)
    status_labels = Dict{Int, String}()
    for (key, value) in _jsonish_dict(get(data, "statusLabels", Dict{String, Any}()))
        code = tryparse(Int, String(key))
        isnothing(code) || (status_labels[code] = _as_string(value, "unknown"))
    end
    isempty(status_labels) && (status_labels = copy(_REACH_STATUS_LABEL_BY_CODE))
    samples = [_deserialize_branch_reachability_sample(_jsonish_dict(sample))
               for sample in get(data, "samples", Any[])]
    return BranchReachabilityResult(
        _as_string(get(data, "systemName", ""), ""),
        Symbol(_as_string(get(data, "paramName", "p"), "p")),
        _as_int(get(data, "paramIndex", 1), 1),
        Int[_as_int(value, 0) for value in get(data, "linkedParamIndices", Any[])],
        Float64[_as_float(value) for value in get(data, "baseParams", Any[])],
        Float64[_as_float(value) for value in get(data, "xGrid", Any[])],
        Float64[_as_float(value) for value in get(data, "yGrid", Any[])],
        _as_int(get(data, "xIndex", 1), 1),
        _as_int(get(data, "yIndex", 2), 2),
        Float64[_as_float(value) for value in get(data, "icTemplate", Any[])],
        _as_int(get(data, "maxPeriod", 0), 0),
        _as_float(get(data, "precision", 1e-4), 1e-4),
        divergence_cutoff,
        _as_float(get(data, "paramTolerance", 1e-6), 1e-6),
        _as_float(get(data, "matchTolerance", 1e-3), 1e-3),
        _as_float(get(data, "ambiguityRatio", 0.5), 0.5),
        _as_float(get(data, "stabilityTol", 1e-7), 1e-7),
        String[_as_string(value, "") for value in get(data, "branchIds", Any[])],
        Int[_as_int(value, 0) for value in get(data, "branchPeriods", Any[])],
        samples,
        status_labels,
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now()))),
    )
end

const _REGIME_BOUNDARY_FORMAT = "regime-boundary-v1"
const _TOLERANCE_MAP_FORMAT = "tolerance-map-v1"

# Special-float aware matrix (de)serialization: unlike the reachability helpers, margin fields
# distinguish Inf (no boundary on this line) from NaN (invalid cell), so non-finite values are
# encoded as explicit tokens rather than collapsed to `nothing`.
_encode_special_float(x::Real) =
    isfinite(x) ? Float64(x) : (isnan(x) ? "nan" : (x > 0 ? "inf" : "-inf"))

function _decode_special_float(value)
    value isa AbstractString || return _as_float(value, NaN)
    token = lowercase(strip(String(value)))
    token == "nan" && return NaN
    token == "inf" && return Inf
    token in ("-inf", "-infinity") && return -Inf
    token == "infinity" && return Inf
    return _as_float(value, NaN)
end

_serialize_special_float_matrix(m::AbstractMatrix{<:Real}) =
    [[_encode_special_float(value) for value in view(m, i, :)] for i in axes(m, 1)]

_deserialize_special_float_matrix(raw, label::AbstractString) =
    _deserialize_robust_chaos_matrix(raw, Float64, _decode_special_float, label)

_serialize_tolerance(t::UniformTolerance) =
    Dict{String, Any}("kind" => "uniform", "scale" => t.half_width)
_serialize_tolerance(t::GaussianTolerance) =
    Dict{String, Any}("kind" => "gaussian", "scale" => t.std)

function _deserialize_tolerance(data)::AbstractTolerance
    dict = _jsonish_dict(data)
    kind = lowercase(_as_string(get(dict, "kind", "uniform"), "uniform"))
    scale = _as_float(get(dict, "scale", 0.0), 0.0)
    kind == "gaussian" && return GaussianTolerance(scale)
    return UniformTolerance(scale)
end

function _serialize_regime_boundary_result(result::RegimeBoundaryResult)::Dict{String, Any}
    return Dict{String, Any}(
        "format" => _REGIME_BOUNDARY_FORMAT,
        "systemName" => result.system_name,
        "paramNames" => [String(result.param_names[1]), String(result.param_names[2])],
        "aGrid" => copy(result.a_grid),
        "bGrid" => copy(result.b_grid),
        "labels" => _serialize_reach_int_matrix(result.labels),
        "resolved" => _serialize_reach_int_matrix(Int.(result.resolved)),
        "valid" => _serialize_reach_int_matrix(Int.(result.valid)),
        "boundaryMask" => _serialize_reach_int_matrix(Int.(result.boundary_mask)),
        "boundaryKind" => _serialize_reach_int_matrix(result.boundary_kind),
        "distance" => _serialize_special_float_matrix(result.distance),
        "distanceA" => _serialize_special_float_matrix(result.distance_a),
        "distanceB" => _serialize_special_float_matrix(result.distance_b),
        "edgeCensored" => _serialize_reach_int_matrix(Int.(result.edge_censored)),
        "edgePolicy" => String(result.edge_policy),
        "statusEvidence" => result.status_evidence,
        "convention" => result.convention,
        "timestamp" => _serialize_timestamp(result.timestamp),
    )
end

function _deserialize_regime_boundary_result(data::AbstractDict)::RegimeBoundaryResult
    format = _as_string(get(data, "format", ""), "")
    format == _REGIME_BOUNDARY_FORMAT || error(
        "Unsupported regime-boundary format '$format'; expected '$(_REGIME_BOUNDARY_FORMAT)'.")
    param_names_raw = get(data, "paramNames", Any["a", "b"])
    pnames = (Symbol(_as_string(param_names_raw[1], "a")), Symbol(_as_string(param_names_raw[2], "b")))
    resolved = _deserialize_robust_chaos_matrix(
        get(data, "resolved", Any[]), Bool, value -> _as_int(value, 0) != 0, "regime resolved mask")
    valid = _deserialize_robust_chaos_matrix(
        get(data, "valid", Any[]), Bool, value -> _as_int(value, 0) != 0, "regime valid mask")
    boundary_mask = _deserialize_robust_chaos_matrix(
        get(data, "boundaryMask", Any[]), Bool, value -> _as_int(value, 0) != 0, "regime boundary mask")
    edge_censored = _deserialize_robust_chaos_matrix(
        get(data, "edgeCensored", Any[]), Bool, value -> _as_int(value, 0) != 0, "regime edge-censored mask")
    return RegimeBoundaryResult(
        _as_string(get(data, "systemName", ""), ""),
        pnames,
        Float64[_as_float(value) for value in get(data, "aGrid", Any[])],
        Float64[_as_float(value) for value in get(data, "bGrid", Any[])],
        _deserialize_robust_chaos_matrix(
            get(data, "labels", Any[]), Int, value -> _as_int(value, 0), "regime labels"),
        resolved,
        valid,
        boundary_mask,
        _deserialize_robust_chaos_matrix(
            get(data, "boundaryKind", Any[]), Int, value -> _as_int(value, 0), "regime boundary kind"),
        _deserialize_special_float_matrix(get(data, "distance", Any[]), "regime distance"),
        _deserialize_special_float_matrix(get(data, "distanceA", Any[]), "regime distance a"),
        _deserialize_special_float_matrix(get(data, "distanceB", Any[]), "regime distance b"),
        edge_censored,
        Symbol(_as_string(get(data, "edgePolicy", "censored"), "censored")),
        _as_bool(get(data, "statusEvidence", false), false),
        _as_string(get(data, "convention", ""), ""),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now()))),
    )
end

function _serialize_tolerance_map_result(result::ToleranceMapResult)::Dict{String, Any}
    return Dict{String, Any}(
        "format" => _TOLERANCE_MAP_FORMAT,
        "systemName" => result.system_name,
        "paramNames" => [String(result.param_names[1]), String(result.param_names[2])],
        "aGrid" => copy(result.a_grid),
        "bGrid" => copy(result.b_grid),
        "regimeLabels" => copy(result.regime_labels),
        "regimeProbability" => Dict{String, Any}(
            string(label) => _serialize_special_float_matrix(matrix)
            for (label, matrix) in result.regime_probability),
        "nominalRegime" => _serialize_reach_int_matrix(result.nominal_regime),
        "nominalResolved" => _serialize_reach_int_matrix(Int.(result.nominal_resolved)),
        "nominalProbability" => _serialize_special_float_matrix(result.nominal_probability),
        "dominantRegime" => _serialize_reach_int_matrix(result.dominant_regime),
        "dominantProbability" => _serialize_special_float_matrix(result.dominant_probability),
        "unknownProbability" => _serialize_special_float_matrix(result.unknown_probability),
        "outOfDomainProbability" => _serialize_special_float_matrix(result.out_of_domain_probability),
        "entropy" => _serialize_special_float_matrix(result.entropy),
        "nominalStandardError" => _serialize_special_float_matrix(result.nominal_standard_error),
        "nominalCiLower" => _serialize_special_float_matrix(result.nominal_ci_lower),
        "nominalCiUpper" => _serialize_special_float_matrix(result.nominal_ci_upper),
        "nSamples" => result.n_samples,
        "nEffective" => result.n_effective,
        "toleranceA" => _serialize_tolerance(result.tolerance_a),
        "toleranceB" => _serialize_tolerance(result.tolerance_b),
        "seed" => string(result.seed),
        "statusEvidence" => result.status_evidence,
        "convention" => result.convention,
        "timestamp" => _serialize_timestamp(result.timestamp),
    )
end

function _deserialize_tolerance_map_result(data::AbstractDict)::ToleranceMapResult
    format = _as_string(get(data, "format", ""), "")
    format == _TOLERANCE_MAP_FORMAT || error(
        "Unsupported tolerance-map format '$format'; expected '$(_TOLERANCE_MAP_FORMAT)'.")
    param_names_raw = get(data, "paramNames", Any["a", "b"])
    pnames = (Symbol(_as_string(param_names_raw[1], "a")), Symbol(_as_string(param_names_raw[2], "b")))
    regime_labels = Int[_as_int(value, 0) for value in get(data, "regimeLabels", Any[])]
    regime_probability = Dict{Int, Matrix{Float64}}()
    for (key, value) in _jsonish_dict(get(data, "regimeProbability", Dict{String, Any}()))
        label = tryparse(Int, String(key))
        isnothing(label) && continue
        regime_probability[label] = _deserialize_special_float_matrix(value, "tolerance regime probability")
    end
    seed_raw = get(data, "seed", "0")
    seed = seed_raw isa Integer ? UInt64(seed_raw) :
           something(tryparse(UInt64, String(seed_raw)), UInt64(0))
    return ToleranceMapResult(
        _as_string(get(data, "systemName", ""), ""),
        pnames,
        Float64[_as_float(value) for value in get(data, "aGrid", Any[])],
        Float64[_as_float(value) for value in get(data, "bGrid", Any[])],
        regime_labels,
        regime_probability,
        _deserialize_robust_chaos_matrix(
            get(data, "nominalRegime", Any[]), Int, value -> _as_int(value, 0), "tolerance nominal regime"),
        _deserialize_robust_chaos_matrix(
            get(data, "nominalResolved", Any[]), Bool, value -> _as_int(value, 0) != 0, "tolerance nominal resolved"),
        _deserialize_special_float_matrix(get(data, "nominalProbability", Any[]), "tolerance nominal probability"),
        _deserialize_robust_chaos_matrix(
            get(data, "dominantRegime", Any[]), Int, value -> _as_int(value, 0), "tolerance dominant regime"),
        _deserialize_special_float_matrix(get(data, "dominantProbability", Any[]), "tolerance dominant probability"),
        _deserialize_special_float_matrix(get(data, "unknownProbability", Any[]), "tolerance unknown probability"),
        _deserialize_special_float_matrix(get(data, "outOfDomainProbability", Any[]), "tolerance out-of-domain probability"),
        _deserialize_special_float_matrix(get(data, "entropy", Any[]), "tolerance entropy"),
        _deserialize_special_float_matrix(get(data, "nominalStandardError", Any[]), "tolerance nominal standard error"),
        _deserialize_special_float_matrix(get(data, "nominalCiLower", Any[]), "tolerance nominal CI lower"),
        _deserialize_special_float_matrix(get(data, "nominalCiUpper", Any[]), "tolerance nominal CI upper"),
        _as_int(get(data, "nSamples", 0), 0),
        _as_int(get(data, "nEffective", 0), 0),
        _deserialize_tolerance(get(data, "toleranceA", Dict{String, Any}())),
        _deserialize_tolerance(get(data, "toleranceB", Dict{String, Any}())),
        seed,
        _as_bool(get(data, "statusEvidence", false), false),
        _as_string(get(data, "convention", ""), ""),
        _deserialize_timestamp(get(data, "timestamp", _serialize_timestamp(now()))),
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
"""    serialize_robust_chaos_certificate(cert::RobustChaosCertificate) -> Dict — JSON-plain form (format version "robust-chaos-certificate-v1")."""
const serialize_robust_chaos_certificate = _serialize_robust_chaos_certificate
"""    deserialize_robust_chaos_certificate(data::AbstractDict) -> RobustChaosCertificate"""
const deserialize_robust_chaos_certificate = _deserialize_robust_chaos_certificate
"""    serialize_robust_chaos_evidence(evidence::RobustChaosEvidence) -> Dict — versioned JSON-plain exact evidence bundle."""
const serialize_robust_chaos_evidence = _serialize_robust_chaos_evidence
"""    deserialize_robust_chaos_evidence(data::AbstractDict) -> RobustChaosEvidence"""
const deserialize_robust_chaos_evidence = _deserialize_robust_chaos_evidence
"""    serialize_branch_reachability_result(result::BranchReachabilityResult) -> Dict — versioned JSON-plain form (format "branch-reachability-v1")."""
const serialize_branch_reachability_result = _serialize_branch_reachability_result
"""    deserialize_branch_reachability_result(data::AbstractDict) -> BranchReachabilityResult"""
const deserialize_branch_reachability_result = _deserialize_branch_reachability_result
"""    serialize_regime_boundary_result(result::RegimeBoundaryResult) -> Dict — versioned JSON-plain form (format "regime-boundary-v1"; Inf/NaN margins preserved distinctly)."""
const serialize_regime_boundary_result = _serialize_regime_boundary_result
"""    deserialize_regime_boundary_result(data::AbstractDict) -> RegimeBoundaryResult"""
const deserialize_regime_boundary_result = _deserialize_regime_boundary_result
"""    serialize_tolerance_map_result(result::ToleranceMapResult) -> Dict — versioned JSON-plain form (format "tolerance-map-v1"; per-regime probability matrices keyed by label)."""
const serialize_tolerance_map_result = _serialize_tolerance_map_result
"""    deserialize_tolerance_map_result(data::AbstractDict) -> ToleranceMapResult"""
const deserialize_tolerance_map_result = _deserialize_tolerance_map_result
"""    serialize_map_normal_form(normal_form::MapNormalForm) -> Dict — versioned JSON-plain form."""
const serialize_map_normal_form = _serialize_map_normal_form
"""    deserialize_map_normal_form(data::AbstractDict) -> MapNormalForm"""
const deserialize_map_normal_form = _deserialize_map_normal_form
"""    serialize_map_special_point(point::MapSpecialPoint) -> Dict — versioned JSON-plain form."""
const serialize_map_special_point = _serialize_map_special_point
"""    deserialize_map_special_point(data::AbstractDict) -> MapSpecialPoint"""
const deserialize_map_special_point = _deserialize_map_special_point
"""    serialize_border_collision_classification(c::BorderCollisionClassification) -> Dict — versioned JSON-plain form (format "border-collision-classification-v1")."""
const serialize_border_collision_classification = _serialize_border_collision_classification
"""    deserialize_border_collision_classification(data::AbstractDict) -> BorderCollisionClassification"""
const deserialize_border_collision_classification = _deserialize_border_collision_classification
"""    serialize_border_collision_point(point::BorderCollisionPoint) -> Dict — versioned JSON-plain form (format "border-collision-point-v1")."""
const serialize_border_collision_point = _serialize_border_collision_point
"""    deserialize_border_collision_point(data::AbstractDict) -> BorderCollisionPoint"""
const deserialize_border_collision_point = _deserialize_border_collision_point

# ---- Codim2SpecialPoint serialization ------------------------------------

const _CODIM2_SPECIAL_POINT_FORMAT = "codim2-special-point-v1"

function _serialize_codim2_special_point(point::Codim2SpecialPoint)::Dict{String, Any}
    point.kind in _CODIM2_SPECIAL_POINT_KINDS || throw(ArgumentError(
        "Codim2SpecialPoint kind must be one of $(join(_CODIM2_SPECIAL_POINT_KINDS, ", ")); " *
        "got $(repr(point.kind))."))
    point.locus_kind in (:fold, :pd, :ns) || throw(ArgumentError(
        "Codim2SpecialPoint locus_kind must be fold, pd, or ns; got $(repr(point.locus_kind))."))
    isfinite(point.primary_param) || throw(ArgumentError("Codim2SpecialPoint primary_param must be finite."))
    isfinite(point.secondary_param) || throw(ArgumentError("Codim2SpecialPoint secondary_param must be finite."))
    all(isfinite, point.state) || throw(ArgumentError("Codim2SpecialPoint state must be finite."))
    for (k, m) in enumerate(point.multipliers)
        isfinite(real(m)) && isfinite(imag(m)) || throw(ArgumentError(
            "Codim2SpecialPoint multiplier[$k] must be finite."))
    end
    point.period >= 1 || throw(ArgumentError("Codim2SpecialPoint period must be >= 1."))
    point.status in (:interpolated, :sampled, :unavailable) || throw(ArgumentError(
        "Codim2SpecialPoint status must be interpolated, sampled, or unavailable; " *
        "got $(repr(point.status))."))
    return Dict{String, Any}(
        "format"         => _CODIM2_SPECIAL_POINT_FORMAT,
        "kind"           => String(point.kind),
        "locusKind"      => String(point.locus_kind),
        "primaryParam"   => point.primary_param,
        "secondaryParam" => point.secondary_param,
        "state"          => collect(Float64, point.state),
        "multipliers"    => [[real(m), imag(m)] for m in point.multipliers],
        "testValue"      => point.test_value,
        "period"         => point.period,
        "converged"      => point.converged,
        "status"         => String(point.status),
        "normalForm"     => point.normal_form === nothing ? nothing :
                            _serialize_map_normal_form(point.normal_form),
    )
end

function _deserialize_codim2_special_point(data::AbstractDict)::Codim2SpecialPoint
    _require_serialized_fields(
        data,
        ("format", "kind", "locusKind", "primaryParam", "secondaryParam",
         "state", "multipliers", "testValue", "period", "converged", "status"),
        "codim2 special point")
    format = _as_string(get(data, "format", ""), "")
    format == _CODIM2_SPECIAL_POINT_FORMAT || throw(ArgumentError(
        "Unsupported codim2 special-point serialization format '$format'."))
    kind = Symbol(_as_string(get(data, "kind", ""), ""))
    kind in _CODIM2_SPECIAL_POINT_KINDS || throw(ArgumentError(
        "Serialized codim2 special-point kind must be one of " *
        "$(join(_CODIM2_SPECIAL_POINT_KINDS, ", ")); got $(repr(kind))."))
    locus_kind = Symbol(_as_string(get(data, "locusKind", ""), ""))
    locus_kind in (:fold, :pd, :ns) || throw(ArgumentError(
        "Serialized codim2 special-point locusKind must be fold, pd, or ns; " *
        "got $(repr(locus_kind))."))
    primary_param   = _as_float(get(data, "primaryParam",   NaN), NaN)
    secondary_param = _as_float(get(data, "secondaryParam", NaN), NaN)
    isfinite(primary_param)   || throw(ArgumentError("Serialized codim2 special-point primaryParam must be finite."))
    isfinite(secondary_param) || throw(ArgumentError("Serialized codim2 special-point secondaryParam must be finite."))
    state = collect(Float64, get(data, "state", Float64[]))
    all(isfinite, state) || throw(ArgumentError("Serialized codim2 special-point state must be finite."))
    raw_mult = get(data, "multipliers", Any[])
    raw_mult isa AbstractVector || throw(ArgumentError("Serialized codim2 special-point multipliers must be an array."))
    multipliers = ComplexF64[]
    for entry in raw_mult
        entry isa AbstractVector && length(entry) == 2 &&
            all(item -> item isa Real, entry) || throw(ArgumentError(
                "Serialized codim2 special-point multiplier entry must be [re, im]."))
        v = complex(Float64(entry[1]), Float64(entry[2]))
        isfinite(real(v)) && isfinite(imag(v)) || throw(ArgumentError(
            "Serialized codim2 special-point multiplier must be finite."))
        push!(multipliers, v)
    end
    test_value = _as_float(get(data, "testValue", NaN), NaN)
    period = _as_int(get(data, "period", 0), 0)
    period >= 1 || throw(ArgumentError("Serialized codim2 special-point period must be >= 1; got $period."))
    get(data, "converged", nothing) isa Bool || throw(ArgumentError(
        "Serialized codim2 special-point converged must be a boolean."))
    converged = data["converged"]::Bool
    status = Symbol(_as_string(get(data, "status", ""), ""))
    status in (:interpolated, :sampled, :unavailable) || throw(ArgumentError(
        "Serialized codim2 special-point status must be interpolated, sampled, or unavailable; " *
        "got $(repr(status))."))
    nf_data = get(data, "normalForm", nothing)
    normal_form = nf_data === nothing ? nothing : _deserialize_map_normal_form(nf_data)
    return Codim2SpecialPoint(kind, locus_kind, primary_param, secondary_param,
                              state, multipliers, test_value, period, converged,
                              status, normal_form)
end

"""    serialize_codim2_special_point(point::Codim2SpecialPoint) -> Dict — versioned JSON-plain form (format "codim2-special-point-v1")."""
const serialize_codim2_special_point = _serialize_codim2_special_point
"""    deserialize_codim2_special_point(data::AbstractDict) -> Codim2SpecialPoint"""
const deserialize_codim2_special_point = _deserialize_codim2_special_point

# ---- Homoclinic continuation serialization --------------------------------

const _HOMOCLINIC_BRANCH_FORMAT = "homoclinic-branch-v1"
const _HOMOCLINIC_CONNECTION_KINDS = (:homoclinic, :heteroclinic, :saddle_cycle)
const _HOMOCLINIC_POINT_STATUSES = (:available, :unavailable, :degenerate)

function _serialize_homoclinic_special_point(point::HomoclinicSpecialPoint)
    haskey(_HOMOCLINIC_EVENT_LABELS, point.kind) || error(
        "Unsupported homoclinic special-point kind $(repr(point.kind)).")
    point.branch_index >= 1 || error("Homoclinic special-point branch index must be >= 1.")
    all(isfinite, (point.primary_param, point.secondary_param, point.test_value)) || error(
        "Homoclinic special-point values must be finite.")
    point.status in _HOMOCLINIC_POINT_STATUSES || error(
        "Unsupported homoclinic special-point status $(repr(point.status)).")
    (isfinite(point.quality) && 0.0 <= point.quality <= 1.0) || error(
        "Homoclinic special-point quality must lie in [0, 1].")
    return Dict{String, Any}(
        "kind" => String(point.kind),
        "label" => point.label,
        "branchIndex" => point.branch_index,
        "primaryParam" => point.primary_param,
        "secondaryParam" => point.secondary_param,
        "testValue" => point.test_value,
        "status" => String(point.status),
        "quality" => point.quality,
    )
end

function _deserialize_homoclinic_special_point(data)
    dict = _jsonish_dict(data)
    kind = Symbol(lowercase(_as_string(get(dict, "kind", ""), "")))
    haskey(_HOMOCLINIC_EVENT_LABELS, kind) || error(
        "Unsupported serialized homoclinic special-point kind $(repr(kind)).")
    branch_index = _as_int(get(dict, "branchIndex", 0), 0)
    branch_index >= 1 || error("Serialized homoclinic special-point branchIndex must be >= 1.")
    primary = _as_float(get(dict, "primaryParam", NaN), NaN)
    secondary = _as_float(get(dict, "secondaryParam", NaN), NaN)
    test_value = _as_float(get(dict, "testValue", NaN), NaN)
    all(isfinite, (primary, secondary, test_value)) || error(
        "Serialized homoclinic special-point values must be finite.")
    status = Symbol(lowercase(_as_string(get(dict, "status", ""), "")))
    status in _HOMOCLINIC_POINT_STATUSES || error(
        "Unsupported serialized homoclinic special-point status $(repr(status)).")
    quality = _as_float(get(dict, "quality", NaN), NaN)
    (isfinite(quality) && 0.0 <= quality <= 1.0) || error(
        "Serialized homoclinic special-point quality must lie in [0, 1].")
    return HomoclinicSpecialPoint(
        kind,
        _as_string(get(dict, "label", homoclinic_special_point_label(kind)),
                   homoclinic_special_point_label(kind)),
        branch_index,
        primary,
        secondary,
        test_value,
        status,
        quality,
    )
end

function _serialize_homoclinic_orbit(orbit::HomoclinicOrbitRecord)
    orbit.branch_index >= 1 || error("Homoclinic orbit branch index must be >= 1.")
    length(orbit.t) == size(orbit.states, 2) || error(
        "Homoclinic orbit time/state sample counts do not match.")
    length(orbit.saddle) == size(orbit.states, 1) || error(
        "Homoclinic orbit saddle dimension does not match the state dimension.")
    all(isfinite, orbit.t) && all(isfinite, orbit.states) && all(isfinite, orbit.saddle) ||
        error("Homoclinic orbit samples must be finite.")
    all(isfinite, (
        orbit.primary_param, orbit.secondary_param, orbit.return_time,
        orbit.epsilon_start, orbit.epsilon_end,
    )) || error("Homoclinic orbit metadata must be finite.")
    return Dict{String, Any}(
        "branchIndex" => orbit.branch_index,
        "t" => copy(orbit.t),
        "states" => [collect(Float64, row) for row in eachrow(orbit.states)],
        "saddle" => copy(orbit.saddle),
        "primaryParam" => orbit.primary_param,
        "secondaryParam" => orbit.secondary_param,
        "returnTime" => orbit.return_time,
        "epsilonStart" => orbit.epsilon_start,
        "epsilonEnd" => orbit.epsilon_end,
    )
end

function _deserialize_homoclinic_orbit(data)
    dict = _jsonish_dict(data)
    branch_index = _as_int(get(dict, "branchIndex", 0), 0)
    branch_index >= 1 || error("Serialized homoclinic orbit branchIndex must be >= 1.")
    t = Float64[_as_float(value, NaN) for value in get(dict, "t", Any[])]
    states = _deserialize_robust_chaos_matrix(
        get(dict, "states", Any[]), Float64, value -> _as_float(value, NaN),
        "homoclinic orbit states")
    saddle = Float64[_as_float(value, NaN) for value in get(dict, "saddle", Any[])]
    length(t) == size(states, 2) || error(
        "Serialized homoclinic orbit time/state sample counts do not match.")
    length(saddle) == size(states, 1) || error(
        "Serialized homoclinic orbit saddle dimension does not match the state dimension.")
    all(isfinite, t) && all(isfinite, states) && all(isfinite, saddle) || error(
        "Serialized homoclinic orbit samples must be finite.")
    values = (
        _as_float(get(dict, "primaryParam", NaN), NaN),
        _as_float(get(dict, "secondaryParam", NaN), NaN),
        _as_float(get(dict, "returnTime", NaN), NaN),
        _as_float(get(dict, "epsilonStart", NaN), NaN),
        _as_float(get(dict, "epsilonEnd", NaN), NaN),
    )
    all(isfinite, values) || error("Serialized homoclinic orbit metadata must be finite.")
    return HomoclinicOrbitRecord(
        branch_index, t, states, saddle, values...)
end

function _validate_homoclinic_result(result::HomoclinicBranchResult)
    count = length(result.primary_values)
    count > 0 || error("Homoclinic branch result must contain at least one locus sample.")
    for values in (
        result.secondary_values, result.return_times, result.epsilon_start_values,
        result.epsilon_end_values,
    )
        length(values) == count || error("Homoclinic branch column lengths do not match.")
        all(isfinite, values) || error("Homoclinic branch columns must be finite.")
    end
    all(isfinite, result.primary_values) || error(
        "Homoclinic branch primary values must be finite.")
    size(result.saddles, 2) == count || error(
        "Homoclinic saddle sample count does not match the locus.")
    all(isfinite, result.saddles) || error("Homoclinic saddle samples must be finite.")
    size(result.target_saddles, 2) == count || error(
        "Homoclinic target-saddle sample count does not match the locus.")
    size(result.target_saddles, 1) == size(result.saddles, 1) || error(
        "Homoclinic target-saddle dimension does not match the source saddle.")
    all(isfinite, result.target_saddles) || error(
        "Homoclinic target-saddle samples must be finite.")
    length(result.residuals) == count || error(
        "Homoclinic residual count does not match the locus.")
    all(r -> isfinite(r) && r >= 0, result.residuals) || error(
        "Homoclinic residuals must be finite and non-negative.")
    length(result.corrector_paths) == count || error(
        "Homoclinic corrector-path count does not match the locus.")
    result.connection_kind in _HOMOCLINIC_CONNECTION_KINDS || error(
        "Unsupported homoclinic connection kind $(repr(result.connection_kind)).")
    for (kind, values) in result.test_functions
        haskey(_HOMOCLINIC_EVENT_LABELS, kind) || error(
            "Unsupported homoclinic test-function kind $(repr(kind)).")
        length(values) == count || error(
            "Homoclinic test-function column $kind does not match the locus.")
    end
    for (kind, statuses) in result.test_statuses
        haskey(_HOMOCLINIC_EVENT_LABELS, kind) || error(
            "Unsupported homoclinic test-status kind $(repr(kind)).")
        length(statuses) == count || error(
            "Homoclinic test-status column $kind does not match the locus.")
        all(s -> s in _HOMOCLINIC_POINT_STATUSES, statuses) || error(
            "Homoclinic test-status column $kind contains unsupported status values.")
    end
    keys(result.test_statuses) == keys(result.test_functions) || error(
        "Homoclinic test-status keys must match the test-function keys.")
    result.source_period >= 0 || error("Homoclinic source period must be >= 0.")
    result.source_index >= 0 || error("Homoclinic source index must be >= 0.")
    ((result.source_period == 0) == (result.source_index == 0)) || error(
        "Homoclinic source period and index must either both be zero or both be positive.")
    isfinite(result.source_primary_value) || error(
        "Homoclinic source primary value must be finite.")
    all(isfinite, result.base_params) || error("Homoclinic base parameters must be finite.")
    result.primary_param_index >= 1 && result.secondary_param_index >= 1 ||
        error("Homoclinic parameter indices must be >= 1.")
    result.primary_param_index <= length(result.base_params) &&
        result.secondary_param_index <= length(result.base_params) || error(
            "Homoclinic parameter indices must lie inside the base parameter vector.")
    result.primary_param_index != result.secondary_param_index || count == 1 || error(
        "Homoclinic primary and secondary parameter indices must differ.")
    all(point -> point.branch_index <= count, result.special_points) || error(
        "Homoclinic special-point branch indices must lie inside the locus.")
    all(orbit -> orbit.branch_index <= count, result.orbits) || error(
        "Homoclinic stored-orbit branch indices must lie inside the locus.")
    return result
end

function _serialize_homoclinic_branch_result(result::HomoclinicBranchResult)
    _validate_homoclinic_result(result)
    return Dict{String, Any}(
        "format" => _HOMOCLINIC_BRANCH_FORMAT,
        "primaryValues" => copy(result.primary_values),
        "secondaryValues" => copy(result.secondary_values),
        "returnTimes" => copy(result.return_times),
        "epsilonStartValues" => copy(result.epsilon_start_values),
        "epsilonEndValues" => copy(result.epsilon_end_values),
        "saddles" => [collect(Float64, row) for row in eachrow(result.saddles)],
        "targetSaddles" => [collect(Float64, row) for row in eachrow(result.target_saddles)],
        "testFunctions" => Dict(
            String(kind) => [_encode_special_float(value) for value in values]
            for (kind, values) in result.test_functions),
        "testStatuses" => Dict(
            String(kind) => String.(values)
            for (kind, values) in result.test_statuses),
        "specialPoints" => [_serialize_homoclinic_special_point(point)
                            for point in result.special_points],
        "orbits" => [_serialize_homoclinic_orbit(orbit) for orbit in result.orbits],
        "residuals" => [_encode_special_float(value) for value in result.residuals],
        "correctorPaths" => String[String(path) for path in result.corrector_paths],
        "connectionKind" => String(result.connection_kind),
        "sourcePeriod" => result.source_period,
        "sourceIndex" => result.source_index,
        "sourcePrimaryValue" => result.source_primary_value,
        "baseParams" => copy(result.base_params),
        "primaryParamIndex" => result.primary_param_index,
        "secondaryParamIndex" => result.secondary_param_index,
        "systemName" => result.system_name,
        "paramNames" => String[String(result.param_names[1]), String(result.param_names[2])],
        "diagnostics" => _serialize_homoclinic_diagnostics(result.diagnostics),
        "timestamp" => _serialize_timestamp(result.timestamp),
    )
end

# Diagnostics is a free-form provenance bag; keep serialization lossless for the
# JSON-plain scalar/collection values the continuation driver stores, and stringify
# anything exotic so a round-trip never fails on an unexpected entry.
function _serialize_homoclinic_diagnostics(diagnostics::AbstractDict)
    out = Dict{String, Any}()
    for (key, value) in diagnostics
        out[String(key)] = _jsonish_plain(value)
    end
    return out
end

const _JSONISH_TYPE_TAG = "__dynamicskit_type__"

_jsonish_plain(value::Union{Nothing, Bool, Integer, AbstractString}) = value
function _jsonish_plain(value::Real)
    encoded = _encode_special_float(value)
    return encoded isa AbstractString ?
           Dict{String, Any}(_JSONISH_TYPE_TAG => "special_float", "value" => encoded) :
           encoded
end
_jsonish_plain(value::Symbol) = String(value)
_jsonish_plain(value::AbstractVector) = [_jsonish_plain(v) for v in value]
function _jsonish_plain(value::AbstractDict)
    encoded = Dict{String, Any}(String(k) => _jsonish_plain(v) for (k, v) in value)
    return haskey(encoded, _JSONISH_TYPE_TAG) ?
           Dict{String, Any}(_JSONISH_TYPE_TAG => "dict", "value" => encoded) :
           encoded
end
_jsonish_plain(value) = string(value)

# Inverse of `_jsonish_plain`. Tagged dictionaries distinguish encoded
# non-finite numbers from legitimate strings such as "nan" and "inf".
_jsonish_decode(value::AbstractString) = String(value)
_jsonish_decode(value::AbstractVector) = [_jsonish_decode(v) for v in value]
_jsonish_decode_dict_entries(value::AbstractDict) =
    Dict{String, Any}(String(k) => _jsonish_decode(v) for (k, v) in value)
function _jsonish_decode(value::AbstractDict)
    tag = get(value, _JSONISH_TYPE_TAG, nothing)
    if tag == "special_float"
        token = get(value, "value", nothing)
        token isa AbstractString || error(
            "Encoded special-float diagnostic is missing its string value.")
        lowercase(String(token)) in ("nan", "inf", "-inf") || error(
            "Unsupported encoded special-float diagnostic $(repr(token)).")
        return _decode_special_float(token)
    elseif tag == "dict"
        inner = get(value, "value", nothing)
        inner isa AbstractDict || error(
            "Encoded diagnostic dictionary is missing its dictionary value.")
        return _jsonish_decode_dict_entries(inner)
    end
    return _jsonish_decode_dict_entries(value)
end
_jsonish_decode(value) = value   # Bool, Int, Nothing, already-numeric values

function _required_homoclinic_field(data::AbstractDict, key::AbstractString)
    haskey(data, key) || error(
        "Serialized homoclinic branch is missing required key '$key'.")
    return data[key]
end

function _deserialize_homoclinic_branch_result(data::AbstractDict)
    format = _as_string(get(data, "format", ""), "")
    format == _HOMOCLINIC_BRANCH_FORMAT || error(
        "Unsupported homoclinic branch format '$format'; expected '$(_HOMOCLINIC_BRANCH_FORMAT)'.")
    primary = Float64[_as_float(value, NaN) for value in
                      _required_homoclinic_field(data, "primaryValues")]
    tests = Dict{Symbol, Vector{Float64}}()
    for (key, values) in _jsonish_dict(
            _required_homoclinic_field(data, "testFunctions"))
        tests[Symbol(lowercase(String(key)))] =
            Float64[_decode_special_float(value) for value in values]
    end
    param_names = _required_homoclinic_field(data, "paramNames")
    length(param_names) == 2 || error(
        "Serialized homoclinic branch paramNames must contain two entries.")
    saddles = _deserialize_robust_chaos_matrix(
        _required_homoclinic_field(data, "saddles"),
        Float64, value -> _as_float(value, NaN),
        "homoclinic saddles")
    target_saddles = _deserialize_robust_chaos_matrix(
        _required_homoclinic_field(data, "targetSaddles"),
        Float64, value -> _as_float(value, NaN),
        "homoclinic target saddles")
    residuals = Float64[_decode_special_float(value) for value in
                        _required_homoclinic_field(data, "residuals")]
    corrector_paths = Symbol[
        Symbol(_as_string(value, "")) for value in
        _required_homoclinic_field(data, "correctorPaths")]
    connection_kind = Symbol(lowercase(_as_string(
        _required_homoclinic_field(data, "connectionKind"), "")))
    # Recursively decode special-float tokens (nan/inf/-inf) and nested structures
    # inside diagnostics. _jsonish_plain encodes floats as strings on serialization;
    # _jsonish_decode inverts that encoding so the round-trip is lossless.
    diagnostics = Dict{String, Any}(
        String(k) => _jsonish_decode(v)
        for (k, v) in _jsonish_dict(
            _required_homoclinic_field(data, "diagnostics")))
    _valid_status(s) = s in (:available, :unavailable, :degenerate)
    raw_statuses = _jsonish_dict(
        _required_homoclinic_field(data, "testStatuses"))
    test_statuses = Dict{Symbol, Vector{Symbol}}(
        Symbol(lowercase(String(k))) =>
            Symbol[let s = Symbol(_as_string(sv, ""))
                       _valid_status(s) || error(
                           "Unsupported serialized homoclinic test status $(repr(s)).")
                       s
                   end
                   for sv in v]
        for (k, v) in raw_statuses)
    result = HomoclinicBranchResult(
        primary,
        Float64[_as_float(value, NaN) for value in
                _required_homoclinic_field(data, "secondaryValues")],
        Float64[_as_float(value, NaN) for value in
                _required_homoclinic_field(data, "returnTimes")],
        Float64[_as_float(value, NaN) for value in
                _required_homoclinic_field(data, "epsilonStartValues")],
        Float64[_as_float(value, NaN) for value in
                _required_homoclinic_field(data, "epsilonEndValues")],
        saddles,
        target_saddles,
        tests,
        test_statuses,
        [_deserialize_homoclinic_special_point(point)
         for point in _required_homoclinic_field(data, "specialPoints")],
        [_deserialize_homoclinic_orbit(orbit) for orbit in
         _required_homoclinic_field(data, "orbits")],
        residuals,
        corrector_paths,
        connection_kind,
        _as_int(_required_homoclinic_field(data, "sourcePeriod"), -1),
        _as_int(_required_homoclinic_field(data, "sourceIndex"), -1),
        _as_float(_required_homoclinic_field(data, "sourcePrimaryValue"), NaN),
        Float64[_as_float(value, NaN) for value in
                _required_homoclinic_field(data, "baseParams")],
        _as_int(_required_homoclinic_field(data, "primaryParamIndex"), 0),
        _as_int(_required_homoclinic_field(data, "secondaryParamIndex"), 0),
        _as_string(_required_homoclinic_field(data, "systemName"), ""),
        (Symbol(_as_string(param_names[1], "p1")), Symbol(_as_string(param_names[2], "p2"))),
        diagnostics,
        _deserialize_timestamp(_required_homoclinic_field(data, "timestamp")),
    )
    return _validate_homoclinic_result(result)
end

"""    serialize_homoclinic_branch_result(result::HomoclinicBranchResult) -> Dict — versioned JSON-plain form."""
const serialize_homoclinic_branch_result = _serialize_homoclinic_branch_result
"""    deserialize_homoclinic_branch_result(data::AbstractDict) -> HomoclinicBranchResult"""
const deserialize_homoclinic_branch_result = _deserialize_homoclinic_branch_result
