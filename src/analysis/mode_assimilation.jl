"""
    ModeSequence(parameter_name, parameter_values, modes; source="Experiment", weights=ones(...))

Validated observations of operating modes along one strictly increasing parameter
axis. `modes` retain their source labels for plotting; assimilation canonicalizes
common aliases such as `P1`, `period 1`, `chaos`, and `UPI-*` only for comparison.
"""
struct ModeSequence
    parameter_name::Symbol
    parameter_values::Vector{Float64}
    modes::Vector{String}
    source::String
    weights::Vector{Float64}

    function ModeSequence(parameter_name::Symbol,
                          parameter_values::AbstractVector,
                          modes::AbstractVector;
                          source::AbstractString="Experiment",
                          weights::AbstractVector=ones(length(parameter_values)))
        values = collect(Float64, parameter_values)
        labels = strip.(string.(modes))
        point_weights = collect(Float64, weights)
        n = length(values)
        n >= 2 || throw(ArgumentError("mode sequence requires at least two observations; got $n."))
        length(labels) == n || throw(ArgumentError(
            "mode sequence has $n parameter values but $(length(labels)) mode labels."))
        length(point_weights) == n || throw(ArgumentError(
            "mode sequence has $n parameter values but $(length(point_weights)) weights."))
        all(isfinite, values) || throw(ArgumentError(
            "mode sequence parameter values must all be finite."))
        all(values[i] < values[i + 1] for i in 1:(n - 1)) || throw(ArgumentError(
            "mode sequence parameter values must be strictly increasing with no duplicates."))
        all(!isempty, labels) || throw(ArgumentError("mode sequence labels must not be blank."))
        all(weight -> isfinite(weight) && weight > 0.0, point_weights) || throw(ArgumentError(
            "mode sequence weights must be finite and greater than zero."))
        isempty(strip(source)) && throw(ArgumentError("mode sequence source must not be blank."))
        new(parameter_name, values, labels, String(source), point_weights)
    end
end

"""
    OperatingMapCrossSection

One categorical route through a `BifurcationMapResult`, including the requested
and selected fixed-parameter values so nearest-grid selection remains explicit.
"""
struct OperatingMapCrossSection
    sequence::ModeSequence
    fixed_parameter_name::Symbol
    requested_fixed_value::Float64
    selected_fixed_value::Float64
    fixed_grid_index::Int
    system_name::String
    status_evidence::Bool

    function OperatingMapCrossSection(sequence::ModeSequence,
                                      fixed_parameter_name::Symbol,
                                      requested_fixed_value::Float64,
                                      selected_fixed_value::Float64,
                                      fixed_grid_index::Int,
                                      system_name::String,
                                      status_evidence::Bool)
        isfinite(requested_fixed_value) || throw(ArgumentError(
            "cross-section requested_fixed_value must be finite."))
        isfinite(selected_fixed_value) || throw(ArgumentError(
            "cross-section selected_fixed_value must be finite."))
        fixed_grid_index >= 1 || throw(ArgumentError(
            "cross-section fixed_grid_index must be positive."))
        isempty(strip(system_name)) && throw(ArgumentError(
            "cross-section system_name must not be blank."))
        new(sequence, fixed_parameter_name, requested_fixed_value,
            selected_fixed_value, fixed_grid_index, system_name, status_evidence)
    end
end

struct ModeTransition
    left_parameter::Float64
    right_parameter::Float64
    location::Float64
    from_mode::String
    to_mode::String
end

struct ModeTransitionComparison
    observed::ModeTransition
    predicted::Union{Nothing, ModeTransition}
    status::Symbol
    distance::Union{Nothing, Float64}
end

"""
    ModeAssimilationConfig(; experimental_scale=1, experimental_offset=0,
                           transition_tolerance=0, mode_aliases=Dict(),
                           unresolved_modes=[...])

`experimental_scale` and `experimental_offset` map uploaded coordinates onto the
computed parameter axis: `aligned = scale * measured + offset`.
Experimental transitions are interval-censored between adjacent measurements; a
predicted transition matches when it has the same ordered mode pair and lies
inside that interval expanded by `transition_tolerance`.
"""
struct ModeAssimilationConfig
    experimental_scale::Float64
    experimental_offset::Float64
    transition_tolerance::Float64
    mode_aliases::Dict{String, String}
    unresolved_modes::Vector{String}

    function ModeAssimilationConfig(;
            experimental_scale::Real=1.0,
            experimental_offset::Real=0.0,
            transition_tolerance::Real=0.0,
            mode_aliases::AbstractDict=Dict{String, String}(),
            unresolved_modes::AbstractVector=[
                "unknown", "undetected", "insufficient_crossings",
                "integration_failed", "invalid_state",
            ])
        scale = Float64(experimental_scale)
        offset = Float64(experimental_offset)
        tolerance = Float64(transition_tolerance)
        isfinite(scale) && scale > 0.0 || throw(ArgumentError(
            "experimental_scale must be finite and greater than zero."))
        isfinite(offset) || throw(ArgumentError("experimental_offset must be finite."))
        isfinite(tolerance) && tolerance >= 0.0 || throw(ArgumentError(
            "transition_tolerance must be finite and nonnegative."))
        aliases = Dict{String, String}()
        for (key, value) in mode_aliases
            from = _mode_comparison_key(string(key))
            to = _mode_comparison_key(string(value))
            isempty(from) && throw(ArgumentError("mode alias keys must not be blank."))
            isempty(to) && throw(ArgumentError("mode alias values must not be blank."))
            aliases[from] = to
        end
        unresolved = unique(_mode_comparison_key.(string.(unresolved_modes)))
        new(scale, offset, tolerance, aliases, unresolved)
    end
end

"""
    ModeSequenceAlignment

Quantitative comparison between measured observations and a computed
operating-map cross-section.

`agreement_score` uses only comparable observations; `coverage` is the weighted
fraction that was comparable; `overall_score` conservatively uses all uploaded
weight. Transition precision/recall/F1 compare one-to-one ordered mode changes
inside the uploaded parameter span.
"""
struct ModeSequenceAlignment
    experimental::ModeSequence
    predicted::OperatingMapCrossSection
    aligned_parameter_values::Vector{Float64}
    predicted_modes::Vector{Union{Nothing, String}}
    observation_statuses::Vector{Symbol}
    agreement_score::Union{Nothing, Float64}
    coverage::Float64
    overall_score::Float64
    transition_comparisons::Vector{ModeTransitionComparison}
    unexpected_transitions::Vector{ModeTransition}
    transition_precision::Union{Nothing, Float64}
    transition_recall::Union{Nothing, Float64}
    transition_f1::Union{Nothing, Float64}
    config::ModeAssimilationConfig
    timestamp::DateTime

    function ModeSequenceAlignment(
            experimental::ModeSequence,
            predicted::OperatingMapCrossSection,
            aligned_parameter_values::Vector{Float64},
            predicted_modes::Vector{Union{Nothing, String}},
            observation_statuses::Vector{Symbol},
            agreement_score::Union{Nothing, Float64},
            coverage::Float64,
            overall_score::Float64,
            transition_comparisons::Vector{ModeTransitionComparison},
            unexpected_transitions::Vector{ModeTransition},
            transition_precision::Union{Nothing, Float64},
            transition_recall::Union{Nothing, Float64},
            transition_f1::Union{Nothing, Float64},
            config::ModeAssimilationConfig,
            timestamp::DateTime)
        n = length(experimental.parameter_values)
        length(aligned_parameter_values) == n || throw(ArgumentError(
            "alignment has $n observations but $(length(aligned_parameter_values)) aligned parameter values."))
        length(predicted_modes) == n || throw(ArgumentError(
            "alignment has $n observations but $(length(predicted_modes)) predicted modes."))
        length(observation_statuses) == n || throw(ArgumentError(
            "alignment has $n observations but $(length(observation_statuses)) observation statuses."))
        _validate_transition_parameters(aligned_parameter_values, "aligned parameter values")
        allowed_statuses = (:matched, :mismatched, :unresolved_prediction, :outside_coverage)
        all(status -> status in allowed_statuses, observation_statuses) || throw(ArgumentError(
            "alignment contains an unknown observation status."))
        for (label, score) in (
                ("agreement_score", agreement_score),
                ("coverage", coverage),
                ("overall_score", overall_score),
                ("transition_precision", transition_precision),
                ("transition_recall", transition_recall),
                ("transition_f1", transition_f1))
            score === nothing && continue
            isfinite(score) && 0.0 <= score <= 1.0 || throw(ArgumentError(
                "$label must be nothing or a finite value in [0, 1]."))
        end
        overall_score <= coverage + 8eps(Float64) || throw(ArgumentError(
            "alignment overall_score cannot exceed coverage."))
        agreement_score === nothing && coverage > 0.0 && throw(ArgumentError(
            "alignment agreement_score cannot be nothing when coverage is positive."))
        for comparison in transition_comparisons
            _validate_mode_transition(comparison.observed)
            comparison.status in (:matched, :missing) || throw(ArgumentError(
                "alignment contains an unknown transition-comparison status."))
            if comparison.status == :matched
                comparison.predicted === nothing && throw(ArgumentError(
                    "matched transition comparisons require a predicted transition."))
                comparison.distance === nothing && throw(ArgumentError(
                    "matched transition comparisons require a distance."))
                _validate_mode_transition(comparison.predicted)
                isfinite(comparison.distance) && comparison.distance >= 0.0 || throw(ArgumentError(
                    "matched transition distance must be finite and nonnegative."))
            else
                comparison.predicted === nothing || throw(ArgumentError(
                    "missing transition comparisons must not carry a predicted transition."))
                comparison.distance === nothing || throw(ArgumentError(
                    "missing transition comparisons must not carry a distance."))
            end
        end
        foreach(_validate_mode_transition, unexpected_transitions)
        new(experimental, predicted, aligned_parameter_values, predicted_modes,
            observation_statuses, agreement_score, coverage, overall_score,
            transition_comparisons, unexpected_transitions, transition_precision,
            transition_recall, transition_f1, config, timestamp)
    end
end

function _validate_transition_parameters(values::AbstractVector, label::AbstractString)
    all(isfinite, values) || throw(ArgumentError("$label must all be finite."))
    length(values) <= 1 || all(values[index] < values[index + 1]
                               for index in 1:(length(values) - 1)) ||
        throw(ArgumentError("$label must be strictly increasing with no duplicates."))
    return nothing
end

function _validate_mode_transition(transition::ModeTransition)
    values = (transition.left_parameter, transition.right_parameter, transition.location)
    all(isfinite, values) || throw(ArgumentError("mode-transition coordinates must be finite."))
    transition.left_parameter < transition.right_parameter || throw(ArgumentError(
        "mode-transition left_parameter must be less than right_parameter."))
    transition.left_parameter <= transition.location <= transition.right_parameter ||
        throw(ArgumentError("mode-transition location must lie inside its observation interval."))
    isempty(strip(transition.from_mode)) && throw(ArgumentError(
        "mode-transition from_mode must not be blank."))
    isempty(strip(transition.to_mode)) && throw(ArgumentError(
        "mode-transition to_mode must not be blank."))
    return nothing
end

function _mode_comparison_key(mode::AbstractString)
    label = lowercase(strip(mode))
    isempty(label) && return ""
    compact = replace(label, r"[\s_\-]+" => "")
    periodic = match(r"^(?:p|period)(\d+)", compact)
    periodic !== nothing && return "p$(parse(Int, periodic.captures[1]))"
    all(isdigit, compact) && return "p$(parse(Int, compact))"
    if compact in ("ch", "chaos", "chaotic", "aperiodic", "highperiod") ||
       startswith(compact, "upi")
        return "aperiodic"
    end
    return replace(label, r"\s+" => "_")
end

function _aliased_mode_key(mode::AbstractString, aliases::Dict{String, String})
    key = _mode_comparison_key(mode)
    visited = Set{String}()
    while haskey(aliases, key)
        key in visited && throw(ArgumentError("mode_aliases contains a cycle involving '$key'."))
        push!(visited, key)
        key = aliases[key]
    end
    return key
end

function _csv_column(columns, requested::Union{Symbol, AbstractString}, role::AbstractString;
                     required::Bool=true)
    target = lowercase(strip(String(requested)))
    matches = [column for column in columns if lowercase(strip(String(column))) == target]
    if isempty(matches)
        required || return nothing
        available = join(string.(columns), ", ")
        throw(ArgumentError("mode-sequence CSV is missing the '$requested' $role column; available columns: $available."))
    end
    length(matches) == 1 || throw(ArgumentError(
        "mode-sequence CSV has multiple columns matching '$requested'."))
    return Symbol(only(matches))
end

function _csv_cell(row, column::Symbol, row_number::Int, role::AbstractString)
    value = getproperty(row, column)
    (value === missing || isempty(strip(string(value)))) && throw(ArgumentError(
        "mode-sequence CSV row $row_number has a blank $role value."))
    return value
end

"""
    load_mode_sequence_csv(input; parameter_column="parameter", mode_column="mode",
                           weight_column="weight", parameter_name=nothing,
                           source="Experiment") -> ModeSequence

Read and validate a CSV path or `IO`. The required columns are `parameter` and
`mode` by default; an optional `weight` column supplies positive observation
weights. Column matching is case-insensitive.
"""
function load_mode_sequence_csv(input;
                                parameter_column::Union{Symbol, AbstractString}="parameter",
                                mode_column::Union{Symbol, AbstractString}="mode",
                                weight_column::Union{Nothing, Symbol, AbstractString}="weight",
                                parameter_name::Union{Nothing, Symbol, AbstractString}=nothing,
                                source::AbstractString="Experiment")
    rows = try
        collect(CSV.File(input; types=String, strict=true))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("could not parse mode-sequence CSV: $(sprint(showerror, err))"))
    end
    isempty(rows) && throw(ArgumentError("mode-sequence CSV contains no data rows."))
    columns = propertynames(first(rows))
    parameter_key = _csv_column(columns, parameter_column, "parameter")
    mode_key = _csv_column(columns, mode_column, "mode")
    weight_key = weight_column === nothing ? nothing :
        _csv_column(columns, weight_column, "weight"; required=false)

    values = Float64[]
    modes = String[]
    weights = Float64[]
    for (index, row) in enumerate(rows)
        row_number = index + 1
        raw_parameter = _csv_cell(row, parameter_key, row_number, "parameter")
        parameter = tryparse(Float64, strip(string(raw_parameter)))
        parameter === nothing && throw(ArgumentError(
            "mode-sequence CSV row $row_number parameter '$(raw_parameter)' is not a number."))
        push!(values, parameter)
        push!(modes, strip(string(_csv_cell(row, mode_key, row_number, "mode"))))
        if weight_key === nothing
            push!(weights, 1.0)
        else
            raw_weight = _csv_cell(row, weight_key, row_number, "weight")
            weight = tryparse(Float64, strip(string(raw_weight)))
            weight === nothing && throw(ArgumentError(
                "mode-sequence CSV row $row_number weight '$(raw_weight)' is not a number."))
            push!(weights, weight)
        end
    end

    axis_name = parameter_name === nothing ? Symbol(parameter_key) : Symbol(parameter_name)
    return ModeSequence(axis_name, values, modes; source=source, weights=weights)
end

function _cross_section_axis(map_result::BifurcationMapResult, varying_parameter::Symbol)
    names = map_result.param_names
    exact = findall(==(varying_parameter), names)
    length(exact) == 1 && return only(exact)
    length(exact) > 1 && throw(ArgumentError(
        "varying_parameter $varying_parameter is ambiguous because both map axes use that name."))
    varying_parameter == :a && return 1
    varying_parameter == :b && return 2
    throw(ArgumentError(
        "varying_parameter must be :a, :b, $(names[1]), or $(names[2]); got $varying_parameter."))
end

_mode_label(label::Int, resolved::Bool) =
    !resolved ? "undetected" :
    label > 0 ? "P$label" :
    label == _REGIME_APERIODIC ? "aperiodic" :
    label == _REGIME_DIVERGED ? "diverged" : "unknown"

"""
    operating_map_cross_section(map_result; varying_parameter=:a, fixed_value,
                                cells=nothing, status_codes=nothing, source="Model")

Extract the nearest-grid categorical route from a computed operating map.
Supplying map status evidence distinguishes aperiodic and diverged cells from
unresolved period-zero cells.
"""
function operating_map_cross_section(map_result::BifurcationMapResult;
                                     varying_parameter::Symbol=:a,
                                     fixed_value::Real,
                                     cells::Union{Nothing, MapCellGrid}=nothing,
                                     status_codes::Union{Nothing, AbstractMatrix{<:Integer}}=nothing,
                                     source::AbstractString="Model")
    axis = _cross_section_axis(map_result, varying_parameter)
    cls = _regime_map_classification(map_result; cells=cells, status_codes=status_codes)
    fixed_grid = axis == 1 ? cls.b_grid : cls.a_grid
    requested = Float64(fixed_value)
    isfinite(requested) || throw(ArgumentError("fixed_value must be finite."))
    fixed_min, fixed_max = extrema(fixed_grid)
    fixed_min <= requested <= fixed_max || throw(ArgumentError(
        "fixed_value $requested lies outside the computed $(axis == 1 ? cls.param_names[2] : cls.param_names[1]) range " *
        "[$fixed_min, $fixed_max]."))
    fixed_index = argmin(abs.(fixed_grid .- requested))
    parameter_values = axis == 1 ? cls.a_grid : cls.b_grid
    labels = axis == 1 ? cls.labels[:, fixed_index] : cls.labels[fixed_index, :]
    resolved = axis == 1 ? cls.resolved[:, fixed_index] : cls.resolved[fixed_index, :]
    if first(parameter_values) > last(parameter_values)
        parameter_values = reverse(parameter_values)
        labels = reverse(labels)
        resolved = reverse(resolved)
    end
    modes = [_mode_label(labels[index], resolved[index]) for index in eachindex(labels)]
    sequence = ModeSequence(
        cls.param_names[axis],
        parameter_values,
        modes;
        source=source,
    )
    return OperatingMapCrossSection(
        sequence,
        cls.param_names[3 - axis],
        requested,
        fixed_grid[fixed_index],
        fixed_index,
        cls.system_name,
        cls.status_evidence,
    )
end

function mode_sequence_transitions(sequence::ModeSequence;
                                   parameter_values::AbstractVector=sequence.parameter_values,
                                   mode_aliases::AbstractDict=Dict{String, String}())
    values = collect(Float64, parameter_values)
    length(values) == length(sequence.modes) || throw(ArgumentError(
        "transition parameter values must match the mode-sequence length."))
    _validate_transition_parameters(values, "transition parameter values")
    aliases = Dict(
        _mode_comparison_key(string(key)) => _mode_comparison_key(string(value))
        for (key, value) in mode_aliases
    )
    transitions = ModeTransition[]
    for index in 1:(length(values) - 1)
        left_key = _aliased_mode_key(sequence.modes[index], aliases)
        right_key = _aliased_mode_key(sequence.modes[index + 1], aliases)
        left_key == right_key && continue
        left = values[index]
        right = values[index + 1]
        push!(transitions, ModeTransition(
            left, right, (left + right) / 2,
            sequence.modes[index], sequence.modes[index + 1],
        ))
    end
    return transitions
end

function _nearest_mode(sequence::ModeSequence, parameter::Float64)
    values = sequence.parameter_values
    (parameter < first(values) || parameter > last(values)) && return nothing
    upper = searchsortedfirst(values, parameter)
    upper == 1 && return sequence.modes[1]
    upper > length(values) && return sequence.modes[end]
    lower = upper - 1
    index = parameter - values[lower] <= values[upper] - parameter ? lower : upper
    return sequence.modes[index]
end

function _score_or_nothing(numerator::Real, denominator::Real)
    denominator > 0 || return nothing
    return Float64(numerator / denominator)
end

function _transition_metrics(matched::Int, observed::Int, predicted::Int)
    precision = _score_or_nothing(matched, predicted)
    recall = _score_or_nothing(matched, observed)
    f1 = precision === nothing || recall === nothing || precision + recall == 0.0 ?
        nothing : 2 * precision * recall / (precision + recall)
    return precision, recall, f1
end

function _transition_assignment(observed::Vector{ModeTransition},
                                predicted::Vector{ModeTransition},
                                aliases::Dict{String, String},
                                tolerance::Float64)
    n, m = length(observed), length(predicted)
    matches = fill(typemin(Int), n + 1, m + 1)
    cost = fill(Inf, n + 1, m + 1)
    parent_i = fill(-1, n + 1, m + 1)
    parent_j = fill(-1, n + 1, m + 1)
    action = fill(UInt8(0), n + 1, m + 1)
    matches[1, 1] = 0
    cost[1, 1] = 0.0

    function update!(to_i, to_j, candidate_matches, candidate_cost,
                     from_i, from_j, candidate_action)
        current_matches = matches[to_i + 1, to_j + 1]
        current_cost = cost[to_i + 1, to_j + 1]
        current_action = action[to_i + 1, to_j + 1]
        better = candidate_matches > current_matches ||
            (candidate_matches == current_matches &&
             (candidate_cost < current_cost ||
              (candidate_cost == current_cost && candidate_action > current_action)))
        better || return
        matches[to_i + 1, to_j + 1] = candidate_matches
        cost[to_i + 1, to_j + 1] = candidate_cost
        parent_i[to_i + 1, to_j + 1] = from_i
        parent_j[to_i + 1, to_j + 1] = from_j
        action[to_i + 1, to_j + 1] = candidate_action
    end

    for i in 0:n, j in 0:m
        current_matches = matches[i + 1, j + 1]
        current_matches == typemin(Int) && continue
        current_cost = cost[i + 1, j + 1]
        i < n && update!(i + 1, j, current_matches, current_cost, i, j, UInt8(1))
        j < m && update!(i, j + 1, current_matches, current_cost, i, j, UInt8(2))
        if i < n && j < m
            obs = observed[i + 1]
            candidate = predicted[j + 1]
            same_modes =
                _aliased_mode_key(obs.from_mode, aliases) ==
                    _aliased_mode_key(candidate.from_mode, aliases) &&
                _aliased_mode_key(obs.to_mode, aliases) ==
                    _aliased_mode_key(candidate.to_mode, aliases)
            in_interval =
                obs.left_parameter - tolerance <= candidate.location <=
                    obs.right_parameter + tolerance
            if same_modes && in_interval
                distance = abs(candidate.location - obs.location)
                update!(i + 1, j + 1, current_matches + 1,
                        current_cost + distance, i, j, UInt8(3))
            end
        end
    end

    assignment = Dict{Int, Int}()
    i, j = n, m
    while i > 0 || j > 0
        step = action[i + 1, j + 1]
        previous_i = parent_i[i + 1, j + 1]
        previous_j = parent_j[i + 1, j + 1]
        previous_i >= 0 && previous_j >= 0 || error(
            "internal transition-assignment reconstruction failed.")
        step == UInt8(3) && (assignment[i] = j)
        i, j = previous_i, previous_j
    end
    return assignment
end

"""
    assimilate_mode_sequence(experimental, predicted; config=ModeAssimilationConfig())

Align sparse measured modes to a computed operating-map cross-section and score
both observations and ordered mode transitions.
"""
function assimilate_mode_sequence(experimental::ModeSequence,
                                  predicted::OperatingMapCrossSection;
                                  config::ModeAssimilationConfig=ModeAssimilationConfig())
    aligned_values = config.experimental_scale .* experimental.parameter_values .+
        config.experimental_offset
    theory = predicted.sequence
    aliases = config.mode_aliases
    unresolved = Set(config.unresolved_modes)

    predicted_modes = Union{Nothing, String}[]
    statuses = Symbol[]
    matched_weight = 0.0
    comparable_weight = 0.0
    total_weight = sum(experimental.weights)
    for index in eachindex(aligned_values)
        predicted_mode = _nearest_mode(theory, aligned_values[index])
        push!(predicted_modes, predicted_mode)
        if predicted_mode === nothing
            push!(statuses, :outside_coverage)
            continue
        end
        predicted_key = _aliased_mode_key(predicted_mode, aliases)
        if predicted_key in unresolved
            push!(statuses, :unresolved_prediction)
            continue
        end
        comparable_weight += experimental.weights[index]
        if _aliased_mode_key(experimental.modes[index], aliases) == predicted_key
            matched_weight += experimental.weights[index]
            push!(statuses, :matched)
        else
            push!(statuses, :mismatched)
        end
    end

    observed_transitions = mode_sequence_transitions(
        experimental;
        parameter_values=aligned_values,
        mode_aliases=aliases,
    )
    predicted_transitions = mode_sequence_transitions(theory; mode_aliases=aliases)
    observed_transitions = [
        transition for transition in observed_transitions
        if !(_aliased_mode_key(transition.from_mode, aliases) in unresolved) &&
           !(_aliased_mode_key(transition.to_mode, aliases) in unresolved)
    ]
    predicted_transitions = [
        transition for transition in predicted_transitions
        if !(_aliased_mode_key(transition.from_mode, aliases) in unresolved) &&
           !(_aliased_mode_key(transition.to_mode, aliases) in unresolved)
    ]
    span_min, span_max = extrema(aligned_values)
    in_span = [
        transition for transition in predicted_transitions
        if span_min <= transition.location <= span_max
    ]
    assignment = _transition_assignment(
        observed_transitions,
        in_span,
        aliases,
        config.transition_tolerance,
    )
    used = falses(length(in_span))
    comparisons = ModeTransitionComparison[]
    for (observed_index, observed) in enumerate(observed_transitions)
        predicted_index = get(assignment, observed_index, 0)
        if predicted_index == 0
            push!(comparisons, ModeTransitionComparison(observed, nothing, :missing, nothing))
        else
            used[predicted_index] = true
            candidate = in_span[predicted_index]
            push!(comparisons, ModeTransitionComparison(
                observed,
                candidate,
                :matched,
                abs(candidate.location - observed.location),
            ))
        end
    end
    unexpected = [in_span[index] for index in eachindex(in_span) if !used[index]]
    matched_transitions = count(comparison -> comparison.status == :matched, comparisons)
    precision, recall, f1 = _transition_metrics(
        matched_transitions,
        length(observed_transitions),
        length(in_span),
    )

    return ModeSequenceAlignment(
        experimental,
        predicted,
        aligned_values,
        predicted_modes,
        statuses,
        _score_or_nothing(matched_weight, comparable_weight),
        comparable_weight / total_weight,
        matched_weight / total_weight,
        comparisons,
        unexpected,
        precision,
        recall,
        f1,
        config,
        now(),
    )
end

function mode_assimilation_summary(result::ModeSequenceAlignment)
    matched_observations = count(==(:matched), result.observation_statuses)
    mismatched_observations = count(==(:mismatched), result.observation_statuses)
    unresolved_observations = count(==(:unresolved_prediction), result.observation_statuses)
    outside_observations = count(==(:outside_coverage), result.observation_statuses)
    matched_transitions = count(comparison -> comparison.status == :matched,
                                result.transition_comparisons)
    return (
        observation_count=length(result.observation_statuses),
        matched_observations,
        mismatched_observations,
        unresolved_observations,
        outside_observations,
        agreement_score=result.agreement_score,
        coverage=result.coverage,
        overall_score=result.overall_score,
        observed_transition_count=length(result.transition_comparisons),
        predicted_transition_count=matched_transitions + length(result.unexpected_transitions),
        matched_transition_count=matched_transitions,
        unexpected_transition_count=length(result.unexpected_transitions),
        transition_precision=result.transition_precision,
        transition_recall=result.transition_recall,
        transition_f1=result.transition_f1,
    )
end
