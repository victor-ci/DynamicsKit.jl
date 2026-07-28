"""
Border-collision scenario prediction beyond the determinant-sign classification.

The 2-D path evaluates the continuous border-collision normal form (BCNF) trace /
determinant criteria used by Banerjee--Yorke--Grebogi and Glendinning. The
period-adding path is deliberately limited to the scalar symbolic ordering:
discontinuous 1-D period-adding theorems are not promoted to continuous 2-D BCNF
claims without an additional verification sweep.
"""

const _BORDER_SCENARIO_CONVENTION =
    "2-D BCNF uses τ=tr(A), δ=det(A); scalar period adding uses Farey/Stern-Brocot ordering with daughter word = left parent followed by right parent."

function _bsp_farey_rungs(max_level::Integer; left_symbol::AbstractString="L",
                          right_symbol::AbstractString="R")
    max_level >= 0 || throw(ArgumentError("max_level must be >= 0; got $max_level."))
    left = BorderScenarioRung(left_symbol, 0, 1, nothing, nothing)
    right = BorderScenarioRung(right_symbol, 1, 1, nothing, nothing)
    intervals = [(left, right)]
    rungs = BorderScenarioRung[left, right]
    for _ in 1:max_level
        next_intervals = Tuple{BorderScenarioRung, BorderScenarioRung}[]
        daughters = BorderScenarioRung[]
        for (a, b) in intervals
            daughter = BorderScenarioRung(
                a.word * b.word,
                a.rotation_numerator + b.rotation_numerator,
                a.period + b.period,
                a.word,
                b.word,
            )
            push!(daughters, daughter)
            push!(next_intervals, (a, daughter))
            push!(next_intervals, (daughter, b))
        end
        append!(rungs, daughters)
        intervals = next_intervals
    end
    return sort(rungs; by=r -> (Rational(r.rotation_numerator, r.period), r.period, r.word))
end

function _bsp_bcnf_parameters(A_L::AbstractMatrix, A_R::AbstractMatrix)
    τL = tr(A_L)
    τR = tr(A_R)
    δL = det(A_L)
    δR = det(A_R)
    return Dict(
        "tau_L" => Float64(τL),
        "delta_L" => Float64(δL),
        "tau_R" => Float64(τR),
        "delta_R" => Float64(δR),
    )
end

function _bsp_real_roots_trace_det(τ::Float64, δ::Float64)
    disc = τ * τ - 4δ
    disc >= 0 || return nothing
    root = sqrt(max(disc, 0.0))
    λa = (τ - root) / 2
    λb = (τ + root) / 2
    return sort([λa, λb])
end

function _bsp_bcnf_robust_conditions(params::Dict{String, Float64}; det_tol::Float64=1e-12)
    τL = params["tau_L"]; δL = params["delta_L"]
    τR = params["tau_R"]; δR = params["delta_R"]
    roots_L = _bsp_real_roots_trace_det(τL, δL)
    roots_R = _bsp_real_roots_trace_det(τR, δR)
    wedge = 0 < δL <= 1 && 0 < δR <= 1 &&
            τL > 1 + δL && τR < -(1 + δR)
    byg_trace_bound = δL > 0 && τL > 2sqrt(δL)
    left_order = roots_L !== nothing && 0 < roots_L[1] < 1 < roots_L[2]
    right_order = roots_R !== nothing && roots_R[1] < -1 < roots_R[2] < 0

    trapping_value = NaN
    homoclinic_value = NaN
    if roots_L !== nothing
        λ2L, λ1L = roots_L[1], roots_L[2]
        trapping_value = τL * δL - δL^2 - δL * λ2L - δL * δR +
                         τR * δL * λ1L + δR * λ2L - τR * δL
    end
    if roots_R !== nothing && abs(δR) > det_tol
        λ1R, λ2R = roots_R[1], roots_R[2]
        homoclinic_value = (τL * τR - δR) * λ1R +
                           (δL / δR + δL - 1) * λ2R -
                           τR * δL - τL * δR + τR - τL
    end
    trapping = isfinite(trapping_value) && trapping_value > 0
    homoclinic = isfinite(homoclinic_value) && homoclinic_value > 0

    verdict = if !(wedge && left_order && right_order)
        :outside_bcnf_rc
    elseif trapping && homoclinic
        :glendinning_fixed_point_candidate
    elseif trapping
        :byg_trapping_candidate
    else
        :inside_bcnf_rc_wedge
    end

    return Dict{String, Any}(
        "basicWedge" => wedge,
        "bygTraceBound" => byg_trace_bound,
        "leftEigenvalueOrder" => left_order,
        "rightEigenvalueOrder" => right_order,
        "trappingInequalityValue" => trapping_value,
        "trappingInequalityHolds" => trapping,
        "homoclinicDenominatorSafe" => abs(δR) > det_tol,
        "homoclinicDenominatorTolerance" => det_tol,
        "homoclinicInequalityValue" => homoclinic_value,
        "homoclinicInequalityHolds" => homoclinic,
        "verdict" => String(verdict),
    ), verdict
end

function _bsp_prediction_inference(model::Symbol, verdict::Symbol, conditions, rungs)
    if model === :bcnf_2d
        verdict === :glendinning_fixed_point_candidate && return "The one-sided Jacobians reduce to the 2-D BCNF robust-chaos wedge; the BYG trapping inequality and Glendinning transverse-homoclinic inequality are both positive. This is reported as a local BCNF robust-chaos candidate, not as a global proof for the original nonlinear map."
        verdict === :byg_trapping_candidate && return "The one-sided Jacobians reduce to the 2-D BCNF robust-chaos wedge and satisfy the BYG trapping inequality, but Glendinning's transverse-homoclinic condition is not positive. Robust chaos may still occur through other invariant sets, so the prediction is a trapping candidate only."
        verdict === :inside_bcnf_rc_wedge && return "The one-sided Jacobians fall in the basic 2-D BCNF robust-chaos wedge, but the evaluated trapping inequality is not positive. No robust-chaos scenario is predicted from the published BCNF criteria."
        return "The one-sided Jacobians do not satisfy the continuous 2-D BCNF robust-chaos hypotheses, so no robust-chaos scenario is predicted."
    elseif model === :scalar_pwl
        return "The scalar one-sided slopes support only the Farey/Stern-Brocot symbolic period-adding order. Admissibility and parameter-window existence require the piecewise-linear offsets and a verification sweep."
    end
    return "The border-collision data do not meet the supported scenario-prediction hypotheses."
end

"""
    border_period_adding_order(max_level=3; left_symbol="L", right_symbol="R")

Return the Farey/Stern-Brocot symbolic ordering used by scalar 1-D
piecewise-linear period-adding theory. Daughter words concatenate neighbouring
parents and have period equal to the sum of parent periods.
"""
function border_period_adding_order(max_level::Integer=3; left_symbol::AbstractString="L",
                                    right_symbol::AbstractString="R")
    return _bsp_farey_rungs(max_level; left_symbol=left_symbol, right_symbol=right_symbol)
end

"""
    border_scenario_predict(A_L, A_R; kwargs...) -> BorderScenarioPrediction

Predict local scenario structure from one-sided border-collision Jacobians.
For 2-D continuous BCNF-compatible data this evaluates the BYG/Glendinning
robust-chaos inequalities. For scalar data it emits the Farey symbolic
period-adding order as a combinatorial prediction only.
"""
function border_scenario_predict(A_L::AbstractMatrix, A_R::AbstractMatrix;
                                 switching_normal=nothing,
                                 classification::Union{Nothing, BorderCollisionClassification}=nothing,
                                 period::Integer=1,
                                 transversality::Union{Nothing, Real}=nothing,
                                 max_farey_level::Integer=3,
                                 continuity_tol::Float64=1e-8,
                                 eigen_tol::Float64=1e-6,
                                 stability_tol::Float64=1e-8,
                                 transversality_tol::Float64=1e-9)
    c = classification === nothing ? border_collision_classify(A_L, A_R;
        switching_normal=switching_normal, period=period, continuity_tol=continuity_tol,
        transversality=transversality, eigen_tol=eigen_tol, stability_tol=stability_tol,
        transversality_tol=transversality_tol) : classification
    warnings = copy(c.warnings)
    if c.status !== :ok
        push!(warnings, "Scenario prediction is refused because the border-collision classification status is $(c.status).")
        return BorderScenarioPrediction(status=:refused, model=:unsupported, classification=c,
            predicted_cascade=:undetermined, inference=_bsp_prediction_inference(:unsupported, :not_applicable, nothing, nothing),
            warnings=warnings, convention=_BORDER_SCENARIO_CONVENTION)
    end

    n = size(c.jacobian_L, 1)
    if n == 1
        rungs = _bsp_farey_rungs(max_farey_level)
        push!(warnings, "Scalar period-adding order is symbolic; one-sided slopes alone do not prove admissible periodicity regions.")
        return BorderScenarioPrediction(status=:ok, model=:scalar_pwl, classification=c,
            predicted_cascade=:period_adding_order, robust_chaos_verdict=:not_applicable,
            period_adding_rungs=rungs,
            validity_region=Dict("hypotheses" => "scalar discontinuous piecewise-linear normal form with admissible offsets"),
            inference=_bsp_prediction_inference(:scalar_pwl, :not_applicable, nothing, rungs),
            warnings=warnings, convention=_BORDER_SCENARIO_CONVENTION)
    elseif n == 2
        params = _bsp_bcnf_parameters(c.jacobian_L, c.jacobian_R)
        conditions, verdict = _bsp_bcnf_robust_conditions(params)
        conditions["source"] = "Banerjee-Yorke-Grebogi trapping inequality and Glendinning 2017 transverse-homoclinic condition"
        push!(warnings, "Discontinuous 1-D period-adding theorems are not transferred to the continuous 2-D BCNF; use verification to compare observed windows.")
        cascade = verdict in (:glendinning_fixed_point_candidate, :byg_trapping_candidate) ?
                  :robust_chaos_candidate : :none
        return BorderScenarioPrediction(status=:ok, model=:bcnf_2d, classification=c,
            bcnf_parameters=params, predicted_cascade=cascade, robust_chaos_verdict=verdict,
            robust_chaos_conditions=conditions, period_adding_rungs=BorderScenarioRung[],
            validity_region=Dict("mu" => "positive orientation after local BCNF scaling",
                                 "locality" => "piecewise-linear normal-form neighbourhood"),
            inference=_bsp_prediction_inference(:bcnf_2d, verdict, conditions, nothing),
            warnings=warnings, convention=_BORDER_SCENARIO_CONVENTION)
    end

    push!(warnings, "Only scalar period-adding order and two-dimensional BCNF robust-chaos criteria are implemented.")
    return BorderScenarioPrediction(status=:refused, model=:unsupported, classification=c,
        predicted_cascade=:undetermined, inference=_bsp_prediction_inference(:unsupported, :not_applicable, nothing, nothing),
        warnings=warnings, convention=_BORDER_SCENARIO_CONVENTION)
end

border_scenario_predict(point::BorderCollisionPoint; kwargs...) =
    border_scenario_predict(point.classification.jacobian_L, point.classification.jacobian_R;
        classification=point.classification, period=point.period, kwargs...)

function _bsp_iterate_window(sys::DiscreteMap, params::AbstractVector, initial_point::AbstractVector;
                             transient::Int, window::Int, divergence_cutoff::Float64)
    dim = sys.dim
    x = SVector{dim, Float64}(initial_point)
    for _ in 1:transient
        x = sys.f(x, params)
        maximum(abs, x) <= divergence_cutoff || return SVector{dim, Float64}[]
    end
    orbit = SVector{dim, Float64}[]
    for _ in 1:window
        x = sys.f(x, params)
        maximum(abs, x) <= divergence_cutoff || return SVector{dim, Float64}[]
        push!(orbit, x)
    end
    return orbit
end

function _bsp_period_runs(param_values::Vector{Float64}, periods::Vector{Int})
    isempty(periods) && return Dict{String, Any}[]
    runs = Dict{String, Any}[]
    start = 1
    for i in 2:(length(periods) + 1)
        if i > length(periods) || periods[i] != periods[start]
            push!(runs, Dict{String, Any}(
                "period" => periods[start],
                "startIndex" => start,
                "endIndex" => i - 1,
                "paramMin" => param_values[start],
                "paramMax" => param_values[i - 1],
                "sampleCount" => i - start,
            ))
            start = i
        end
    end
    return runs
end

function _bsp_positive_sequence(periods::Vector{Int})
    seq = Int[]
    prev = typemin(Int)
    for p in periods
        p > 0 || continue
        p == prev && continue
        push!(seq, p)
        prev = p
    end
    return seq
end

function _bsp_default_expected_periods(prediction::BorderScenarioPrediction)
    prediction.predicted_cascade === :period_adding_order || return Int[]
    return Int[r.period for r in prediction.period_adding_rungs]
end

function _bsp_is_robust_candidate(prediction::BorderScenarioPrediction)
    prediction.predicted_cascade === :robust_chaos_candidate || return false
    return prediction.robust_chaos_verdict in
        (:glendinning_fixed_point_candidate, :byg_trapping_candidate)
end

"""
    border_scenario_verify(sys, prediction; kwargs...) -> BorderScenarioVerification

Run a targeted one-parameter sweep against a scenario prediction. Scalar
period-adding predictions compare the observed positive-period sequence against
`expected_periods` or the prediction's symbolic order. BCNF robust-chaos
candidates use a finite-time screen: high-period/aperiodic samples and positive
largest Lyapunov estimates across the requested fraction of the sweep.
"""
function border_scenario_verify(sys::DiscreteMap, prediction::BorderScenarioPrediction;
                                param_index::Integer,
                                base_params::AbstractVector,
                                param_min::Real,
                                param_max::Real,
                                param_steps::Integer=101,
                                initial_point::AbstractVector,
                                linked_param_indices::AbstractVector{<:Integer}=Int[],
                                transient::Integer=500,
                                max_period::Integer=16,
                                precision::Float64=1e-6,
                                divergence_cutoff::Float64=1e9,
                                expected_periods::AbstractVector{<:Integer}=Int[],
                                required_prefix_length::Integer=0,
                                lyapunov_transient::Integer=transient,
                                lyapunov_iterations::Integer=max(200, transient),
                                lyapunov_perturbation::Float64=1e-8,
                                lyapunov_min::Float64=1e-4,
                                required_chaotic_fraction::Float64=0.6)
    param_steps >= 1 || throw(ArgumentError("param_steps must be >= 1; got $param_steps."))
    max_period >= 1 || throw(ArgumentError("max_period must be >= 1; got $max_period."))
    transient >= 0 || throw(ArgumentError("transient must be >= 0; got $transient."))
    lyapunov_transient >= 0 || throw(ArgumentError("lyapunov_transient must be >= 0; got $lyapunov_transient."))
    lyapunov_iterations >= 1 || throw(ArgumentError("lyapunov_iterations must be >= 1; got $lyapunov_iterations."))
    0 <= required_chaotic_fraction <= 1 || throw(ArgumentError(
        "required_chaotic_fraction must be in [0, 1]; got $required_chaotic_fraction."))
    pvals = param_steps == 1 ? [Float64(param_min)] :
            collect(range(Float64(param_min), Float64(param_max), length=Int(param_steps)))
    periods = Int[]
    lyaps = Float64[]
    lyap_statuses = Symbol[]
    base = collect(Float64, base_params)
    linked = collect(Int, linked_param_indices)
    window = Int(max_period) + 1
    robust_candidate = _bsp_is_robust_candidate(prediction)
    for p in pvals
        params = inject_param(base, Int(param_index), p, linked)
        orbit = _bsp_iterate_window(sys, params, initial_point;
            transient=Int(transient), window=window, divergence_cutoff=divergence_cutoff)
        detected = length(orbit) == window ? _detect_period(orbit, Int(max_period), precision) : 0
        push!(periods, detected)
        if robust_candidate
            estimate = _estimate_discrete_map_largest_lyapunov(
                sys,
                params,
                SVector{sys.dim, Float64}(initial_point),
                Int(lyapunov_transient),
                Int(lyapunov_iterations),
                lyapunov_perturbation,
                divergence_cutoff,
            )
            push!(lyaps, Float64(estimate.exponent))
            push!(lyap_statuses, estimate.estimation_status)
        end
    end
    runs = _bsp_period_runs(pvals, periods)
    if robust_candidate
        finite_ok = [isfinite(lyaps[i]) && lyap_statuses[i] === :ok for i in eachindex(lyaps)]
        resolved = count(identity, finite_ok)
        positive = count(i -> finite_ok[i] && lyaps[i] > lyapunov_min, eachindex(lyaps))
        high_period = count(==(0), periods)
        denom = max(length(pvals), 1)
        positive_fraction = positive / denom
        aperiodic_fraction = high_period / denom
        passed = positive_fraction >= required_chaotic_fraction &&
                 aperiodic_fraction >= required_chaotic_fraction
        inference = passed ?
            "Finite-time verification found positive Lyapunov estimates and no low-period closure across the required sweep fraction." :
            "Finite-time verification did not meet the required positive-Lyapunov/high-period fraction."
        push!(runs, Dict{String, Any}(
            "kind" => "lyapunov_summary",
            "resolved" => resolved,
            "positive" => positive,
            "positiveFraction" => positive_fraction,
            "aperiodicOrHighPeriod" => high_period,
            "aperiodicFraction" => aperiodic_fraction,
            "lyapunovMinimum" => lyapunov_min,
            "requiredFraction" => required_chaotic_fraction,
        ))
        return BorderScenarioVerification(status=:ok, prediction_status=prediction.status,
            verification_kind=:finite_time_chaos, param_values=pvals, observed_periods=periods,
            observed_runs=runs, lyapunov_exponents=lyaps, lyapunov_statuses=lyap_statuses,
            positive_lyapunov_fraction=positive_fraction,
            aperiodic_fraction=aperiodic_fraction,
            matched_prefix_length=0, required_prefix_length=0,
            consistency_passed=passed, inference=inference)
    end
    observed_sequence = _bsp_positive_sequence(periods)
    expected = isempty(expected_periods) ? _bsp_default_expected_periods(prediction) :
               collect(Int, expected_periods)
    required = Int(required_prefix_length)
    required == 0 && (required = isempty(expected) ? 0 : min(3, length(expected)))
    matched = 0
    for i in 1:min(length(observed_sequence), length(expected))
        observed_sequence[i] == expected[i] || break
        matched += 1
    end
    passed = required == 0 ? !isempty(observed_sequence) : matched >= required
    inference = passed ?
        "Observed period sequence matched $matched predicted prefix entries." :
        "Observed period sequence matched $matched predicted prefix entries; required $required."
    return BorderScenarioVerification(status=:ok, prediction_status=prediction.status,
        verification_kind=:period_sequence, param_values=pvals, observed_periods=periods,
        observed_runs=runs,
        positive_lyapunov_fraction=NaN, aperiodic_fraction=NaN,
        matched_prefix_length=matched, required_prefix_length=required,
        consistency_passed=passed, inference=inference)
end
