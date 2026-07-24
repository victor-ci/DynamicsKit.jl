"""
Branch continuation via BifurcationKit.jl with automatic bifurcation detection.
"""

"""Default branch recorder: store all section/state coordinates as `x1`, `x2`, ..."""
function _default_record(x, p; k...)
    names = ntuple(i -> Symbol(:x, i), length(x))
    NamedTuple{names}(Tuple(x))
end

"""Extract a fixed number of recorded state coordinates from a branch point."""
_branch_point_state(point, dim::Int) = [Float64(getproperty(point, Symbol(:x, i))) for i in 1:dim]

"""Lightweight merged continuation result used for complete two-sided branches."""
struct CombinedBranchResult
    branch::Vector{Any}
    specialpoint::Vector{Any}
end

Base.length(result::CombinedBranchResult) = length(result.branch)

"""Why a single-direction PALC run stopped (used to decide whether to re-seed)."""
@enum TerminationReason REACHED_BOUNDARY HIT_MAX_STEPS CORRECTOR_FAILURE UNKNOWN

"""
Diagnosis of a terminated continuation direction.

`local_direction` is `[Δparam, Δstate...]` averaged over the trailing steps and is used to
extrapolate where the branch was heading. `last_fold_param` is set when the final recorded
special point is a fold (PALC already turned through it, so re-seeding should be skipped).
"""
struct ContinuationTerminationInfo
    reason::TerminationReason
    last_param::Float64
    last_state::Vector{Float64}
    local_direction::Vector{Float64}
    last_fold_param::Union{Float64, Nothing}
    n_steps::Int
end

"""Recover the (state, param) trajectory of a BifurcationKit ContResult.

Prefers the full state vectors in `br.sol` (recorded when `save_sol_every_step > 0`); falls
back to the recorded `x1,x2,…` projections (which `_default_record` populates with the full
state) when `sol` is unavailable.
"""
function _termination_trajectory(cont_result)
    sols = nothing
    try
        sols = cont_result.sol
    catch
        sols = nothing
    end
    if !isnothing(sols) && !isempty(sols)
        states = [collect(Float64, s.x) for s in sols]
        params = [Float64(s.p) for s in sols]
        return states, params
    end
    pts = collect(cont_result.branch)
    states = [_branch_point_state(p) for p in pts]
    params = [Float64(p.param) for p in pts]
    return states, params
end

"""
    diagnose_continuation_termination(cont_result, config; boundary_tol=1e-4, direction_window=3)

Classify why a single-direction PALC run stopped and return the data needed to re-seed:
the last point's parameter + full state, the local trajectory direction, and the parameter of
the last fold (if the run ended at one). `cont_result` is a raw BifurcationKit `ContResult`.
"""
function diagnose_continuation_termination(cont_result, config::ContinuationConfig;
                                           boundary_tol::Float64=1e-4,
                                           direction_window::Int=3)
    n = length(cont_result.branch)
    if n < 2
        return ContinuationTerminationInfo(UNKNOWN, NaN, Float64[], Float64[], nothing, n)
    end

    states, params = _termination_trajectory(cont_result)
    last_state = states[end]
    last_param = params[end]

    w = min(direction_window, length(states) - 1)
    local_direction = if w >= 1
        dp = (params[end] - params[end - w]) / w
        dstate = (states[end] .- states[end - w]) ./ w
        vcat(dp, dstate)
    else
        vcat(0.0, zeros(length(last_state)))
    end

    last_fold_param = nothing
    try
        for sp in Iterators.reverse(cont_result.specialpoint)
            if hasproperty(sp, :type) && sp.type === :fold
                last_fold_param = Float64(sp.param)
                break
            end
        end
    catch
        last_fold_param = nothing
    end

    reason = if abs(last_param - config.p_min) <= boundary_tol ||
                abs(last_param - config.p_max) <= boundary_tol
        REACHED_BOUNDARY
    elseif n >= config.max_steps + 1
        HIT_MAX_STEPS
    else
        CORRECTOR_FAILURE
    end

    return ContinuationTerminationInfo(reason, last_param, last_state, local_direction,
                                       last_fold_param, n)
end

"""
    continuation_branch(sys::DiscreteMap, config::ContinuationConfig;
                        initial_point=nothing, params=[1.0],
                        record=nothing) -> BranchResult

Compute a continuation branch for a discrete map's fixed points (period-1).
Uses pseudo-arclength continuation with automatic differentiation Jacobians.
"""
function continuation_branch(sys::DiscreteMap, config::ContinuationConfig;
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             params::Vector{Float64}=[1.0],
                             record::Union{Nothing, Function}=nothing,
                             reseed::ReseedConfig=ReseedConfig(),
                             on_reseed::Union{Nothing, Function}=nothing,
                             trim_to_minimal_period::Bool=false,
                             on_trim::Union{Nothing, Function}=nothing)
    dim = sys.dim
    x0 = isnothing(initial_point) ? zeros(dim) : collect(Float64, initial_point)

    # F(x, p) = 0 at fixed points: f(x,p) - x = 0
    F = (x, p) -> begin
        pv = inject_param(params, config.param_index, p.p, config.linked_param_indices)
        Array(sys.f(SVector{dim}(x), pv)) .- x
    end

    # Default recorder: first state variable
    # BifurcationKit passes keyword args (iter, state) — must accept them
    rec = isnothing(record) ? _default_record : record

    branch = _run_discrete_continuation(sys, config, 1, x0, F, rec, params, reseed, on_reseed)
    return _finalize_minimal_period_trim(
        sys, branch, params, config.linked_param_indices;
        trim_to_minimal_period=trim_to_minimal_period,
        on_trim=on_trim
    )
end

"""
Run a discrete-map continuation (period-`period`) from `x0`, optionally re-seeding when a PALC
direction dies in the interior. The seed parameter `p0` may sit anywhere in `[p_min, p_max]`;
we run PALC in **both** directions from `p0` (mirroring the continuous bidirectional driver) so
the resulting branch covers the orbit's full extent regardless of where the seed landed. Each
direction is configured with the same `|config.ds|` magnitude.
"""
function _run_discrete_continuation(sys::DiscreteMap, config::ContinuationConfig, period::Int,
                                    x0::Vector{Float64}, F, rec, params::Vector{Float64},
                                    reseed::ReseedConfig, on_reseed::Union{Nothing, Function})
    lens = (@optic _.p)
    p0 = params[config.param_index]
    prob_from_seed = (x_seed, p_seed) -> BifurcationProblem(
        F, collect(Float64, x_seed), (p = p_seed,), lens; record_from_solution = rec)

    run_backward = p0 > config.p_min + 10eps(Float64)
    run_forward = p0 < config.p_max - 10eps(Float64)

    # Seed exactly on a boundary: keep the user's `config.ds` sign so we still trace one
    # direction across the window (this also preserves backward-compat for callers that
    # explicitly seeded at `p_min` with `ds > 0`).
    if !(run_backward || run_forward)
        run_forward = config.ds >= 0
        run_backward = !run_forward
    end

    if reseed.enabled
        reseed_skeleton = (p_query, x_extrap, lo, hi) -> begin
            sk = find_periodic_skeleton(
                sys, [period], p_query;
                search_min=lo, search_max=hi, seed_points=[x_extrap],
                n_initial=reseed.n_skeleton_initial,
                params=params, param_index=config.param_index,
                linked_param_indices=config.linked_param_indices,
                tol=config.newton_tol, max_iter=config.newton_max_iter,
                threaded=false, cache_enabled=false
            )
            return [collect(Float64, item.point) for item in sk if item.period == period]
        end

        backward_branch = nothing
        forward_branch = nothing
        backward_diag = nothing
        forward_diag = nothing
        if run_backward
            backward_branch, backward_diag = _run_continuation_direction_with_reseed(
                prob_from_seed, reseed_skeleton, config, reseed, x0, p0;
                p_min=config.p_min, p_max=p0, ds=-abs(config.ds),
                context="Discrete period-$period backward continuation")
        end
        if run_forward
            forward_branch, forward_diag = _run_continuation_direction_with_reseed(
                prob_from_seed, reseed_skeleton, config, reseed, x0, p0;
                p_min=p0, p_max=config.p_max, ds=abs(config.ds),
                context="Discrete period-$period forward continuation")
        end
        isnothing(on_reseed) || on_reseed(backward_diag, forward_diag)

        if isnothing(backward_branch) && isnothing(forward_branch)
            error("No continuation steps converged for the period-$period branch of $(sys.name).")
        end
        merged = if isnothing(backward_branch)
            forward_branch
        elseif isnothing(forward_branch)
            backward_branch
        else
            _merge_continuation_branches(backward_branch, forward_branch)
        end
        return BranchResult(merged, period, sys.name, sys.param_names[config.param_index], now())
    end

    newton_opts = NewtonPar(tol = config.newton_tol, max_iterations = config.newton_max_iter)
    build_par(ds) = ContinuationPar(
        p_min = config.p_min, p_max = config.p_max,
        ds = ds, dsmax = config.dsmax, dsmin = config.dsmin,
        max_steps = config.max_steps, newton_options = newton_opts,
        detect_bifurcation = config.detect_bifurcation, n_inversion = 6,
        a = config.a, detect_fold = config.detect_fold,
        save_sol_every_step = config.save_sol_every_step
    )
    safe_run(ds) = try
        (
            continuation(prob_from_seed(x0, p0), PALC(), build_par(ds); normC = norminf, verbosity = 0),
            nothing,
        )
    catch err
        err isa InterruptException && rethrow()
        (nothing, sprint(showerror, err))
    end

    backward, backward_error = run_backward ? safe_run(-abs(config.ds)) : (nothing, nothing)
    forward, forward_error = run_forward ? safe_run(abs(config.ds)) : (nothing, nothing)

    if isnothing(backward) && isnothing(forward)
        failures = String[]
        !isnothing(backward_error) && push!(failures, "backward: $backward_error")
        !isnothing(forward_error) && push!(failures, "forward: $forward_error")
        details = isempty(failures) ? "" : " " * join(failures, "; ")
        error("No continuation steps converged for the period-$period branch of $(sys.name).$details")
    end
    merged = if isnothing(backward)
        forward
    elseif isnothing(forward)
        backward
    else
        _merge_continuation_branches(backward, forward)
    end
    return BranchResult(merged, period, sys.name, sys.param_names[config.param_index], now())
end

"""
    continuation_branch(sys::DiscreteMap, config::ContinuationConfig, period::Int;
                        initial_point=nothing, params=[1.0]) -> BranchResult

Compute a continuation branch for period-N orbits of a discrete map.
The map is composed N times: F^N(x, p) - x = 0.
"""
function continuation_branch(sys::DiscreteMap, config::ContinuationConfig, period::Int;
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             params::Vector{Float64}=[1.0],
                             reseed::ReseedConfig=ReseedConfig(),
                             on_reseed::Union{Nothing, Function}=nothing,
                             trim_to_minimal_period::Bool=false,
                             on_trim::Union{Nothing, Function}=nothing)
    period >= 1 || throw(ArgumentError(period == 0 ?
        "continuation_branch period must be >= 1, got 0; F^0(x) - x = 0 is the trivial identity, satisfied everywhere." :
        "continuation_branch period must be >= 1, got $(period)."))
    dim = sys.dim

    if period == 1
        return continuation_branch(sys, config; initial_point=initial_point, params=params,
                                   reseed=reseed, on_reseed=on_reseed,
                                   trim_to_minimal_period=trim_to_minimal_period,
                                   on_trim=on_trim)
    end

    x0 = isnothing(initial_point) ? zeros(dim) : collect(Float64, initial_point)

    # F^period(x, p) - x = 0
    F = (x, p) -> begin
        pv = inject_param(params, config.param_index, p.p, config.linked_param_indices)
        sv = SVector{dim}(x)
        for _ in 1:period
            sv = sys.f(sv, pv)
        end
        Array(sv) .- x
    end

    branch = _run_discrete_continuation(sys, config, period, x0, F, _default_record, params, reseed, on_reseed)
    return _finalize_minimal_period_trim(
        sys, branch, params, config.linked_param_indices;
        trim_to_minimal_period=trim_to_minimal_period,
        on_trim=on_trim
    )
end

"""
    continuation_branch(sys::ContinuousODE, config::ContinuationConfig;
                        initial_point=nothing, params=Float64[], record=nothing, kwargs...) -> BranchResult

Compute a continuation branch for a continuous-time system by continuing fixed points of its
Poincaré return map. If `initial_point` is omitted, a seed is found automatically via the
periodic skeleton search at `params[config.param_index]`.
"""
function continuation_branch(sys::ContinuousODE, config::ContinuationConfig;
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             params::Vector{Float64}=Float64[],
                             record::Union{Nothing, Function}=nothing,
                             kwargs...)
    continuation_branch(sys, config, 1; initial_point=initial_point, params=params, record=record, kwargs...)
end

"""
    continuation_branch(sys::ContinuousODE, config::ContinuationConfig, period::Int; kwargs...) -> BranchResult

Continue period-`period` orbits of a continuous-time system by solving
`Π^period(x, p) - x = 0`, where `Π` is the Poincaré return map.
"""
function continuation_branch(sys::ContinuousODE, config::ContinuationConfig, period::Int;
                             initial_point::Union{Nothing, AbstractVector}=nothing,
                             params::Vector{Float64}=Float64[],
                             record::Union{Nothing, Function}=nothing,
                             on_error::Union{Nothing, Function}=nothing,
                             search_min::Union{Nothing, AbstractVector}=nothing,
                             search_max::Union{Nothing, AbstractVector}=nothing,
                             n_initial::Int=12,
                             sample_seed_points::Bool=true,
                             sample_seed_crossings::Int=0,
                             sample_seed_transient::Int=0,
                             tol::Float64=1e-8,
                             max_iter::Int=40,
                             fd_step::Float64=1e-6,
                             solver=Tsit5(),
                             reltol::Float64=1e-8,
                             abstol::Float64=1e-8,
                             tmax::Union{Nothing, Float64}=nothing,
                              min_crossing_time::Float64=1e-6,
                              reseed::ReseedConfig=ReseedConfig(),
                              on_reseed::Union{Nothing, Function}=nothing,
                              trim_to_minimal_period::Bool=false,
                              on_trim::Union{Nothing, Function}=nothing)
    period >= 1 || throw(ArgumentError(period == 0 ?
        "continuation_branch period must be >= 1, got 0; Π^0(x) - x = 0 is the trivial identity, satisfied everywhere." :
        "continuation_branch period must be >= 1, got $(period)."))
    base_params = _resolve_continuous_params(sys, params)
    p0 = base_params[config.param_index]
    x0 = _resolve_continuous_seed(
        sys,
        period,
        p0,
        initial_point,
        base_params,
        config.param_index;
        search_min=search_min,
        search_max=search_max,
        n_initial=n_initial,
        sample_seed_points=sample_seed_points,
        sample_seed_crossings=sample_seed_crossings,
        sample_seed_transient=sample_seed_transient,
        tol=tol,
        max_iter=max_iter,
        fd_step=fd_step,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        min_crossing_time=min_crossing_time
    )

    F = (x, p) -> begin
        pv = inject_param(base_params, config.param_index, p.p, config.linked_param_indices)
        next_point, found = _poincare_projected(
            sys,
            x,
            pv;
            period=period,
            solver=solver,
            reltol=reltol,
            abstol=abstol,
            tmax=tmax,
            min_crossing_time=min_crossing_time
        )
        found || return fill(1e6, length(x))
        next_point .- x
    end

    J = if config.ode_jacobian_method == :variational
        (x, p) -> begin
            pv = inject_param(base_params, config.param_index, p.p, config.linked_param_indices)
            map_jacobian, found = _poincare_projected_jacobian_variational(
                sys,
                x,
                pv;
                period=period,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time
            )
            found || error("Variational Poincaré derivative failed to find a period-$period return for $(sys.name).")
            return map_jacobian - Matrix{Float64}(I, length(x), length(x))
        end
    else
        (x, p) -> _fd_jacobian(z -> F(z, p), x, fd_step)
    end
    rec = isnothing(record) ? _default_record : record

    lens = (@optic _.p)
    prob_from_seed = (x_seed, p_seed) -> BifurcationProblem(
        F,
        collect(Float64, x_seed),
        (p = p_seed,),
        lens;
        J=J,
        R01=BifurcationKit.FiniteDifferences(),
        delta=fd_step,
        record_from_solution=rec
    )
    prob_builder = () -> prob_from_seed(x0, p0)

    reseed_skeleton = (p_query, x_extrap, lo, hi) -> begin
        sk = find_periodic_skeleton(
            sys, [period], p_query;
            search_min=lo, search_max=hi, seed_points=[x_extrap],
            n_initial=reseed.n_skeleton_initial,
            params=base_params, param_index=config.param_index,
            linked_param_indices=config.linked_param_indices,
            tol=tol, max_iter=max_iter, fd_step=fd_step,
            solver=solver, reltol=reltol, abstol=abstol, tmax=tmax,
            min_crossing_time=min_crossing_time,
            threaded=false, cache_enabled=false
        )
        return [collect(Float64, item.point) for item in sk if item.period == period]
    end

    branch = _complete_continuous_branch(
        prob_builder, config, p0;
        on_error=on_error, reseed_cfg=reseed, prob_from_seed=prob_from_seed,
        reseed_skeleton=reseed_skeleton, x0=x0, on_reseed=on_reseed
    )
    isnothing(branch) && error("No continuation steps converged for the period-$period branch of $(sys.name) near parameter $p0.")
    length(branch) > 0 || error("No continuation points were recorded for the period-$period branch of $(sys.name) near parameter $p0.")

    result = BranchResult(branch, period, sys.name, sys.param_names[config.param_index], now())
    return _finalize_minimal_period_trim(
        sys, result, base_params, config.linked_param_indices;
        trim_to_minimal_period=trim_to_minimal_period,
        on_trim=on_trim,
        fd_step=fd_step,
        solver=solver,
        reltol=reltol,
        abstol=abstol,
        tmax=tmax,
        ode_jacobian_method=config.ode_jacobian_method,
        min_crossing_time=min_crossing_time
    )
end

"""
    continuation_branches(sys::ContinuousODE, config::ContinuationConfig, periods;
                          skeleton_params, kwargs...) -> Vector{BranchResult}

Find seeds at multiple skeleton parameter values and continue all distinct branches found for the
requested periods. This mirrors the MATLAB multi-skeleton workflow natively.
"""
function continuation_branches(sys::ContinuousODE, config::ContinuationConfig, periods::AbstractVector{Int};
                               skeleton_params::AbstractVector{<:Real},
                               params::Vector{Float64}=Float64[],
                               initial_point::Union{Nothing, AbstractVector}=nothing,
                               record::Union{Nothing, Function}=nothing,
                               search_min::Union{Nothing, AbstractVector}=nothing,
                               search_max::Union{Nothing, AbstractVector}=nothing,
                               n_initial::Int=12,
                               trajectory_seed_points::Bool=true,
                               trajectory_seed_crossings::Int=0,
                               trajectory_seed_transient::Int=0,
                               reuse_neighbor_seeds::Union{Nothing, Bool}=nothing,
                               tol::Float64=1e-8,
                               max_iter::Int=40,
                               fd_step::Float64=1e-6,
                               solver=Tsit5(),
                               reltol::Float64=1e-8,
                               abstol::Float64=1e-8,
                               tmax::Union{Nothing, Float64}=nothing,
                               min_crossing_time::Float64=1e-6,
                               max_branches_per_period::Int=typemax(Int),
                               signature_param_tol::Float64=5e-3,
                               signature_state_tol::Float64=0.5,
                               threaded::Bool=Threads.nthreads() > 1,
                               threaded_skeleton::Union{Nothing, Bool}=nothing,
                                skeleton_cache_resolver::Union{Nothing, Function}=nothing,
                               on_error::Union{Nothing, Function}=nothing)
    results = BranchResult[]
    base_params = _resolve_continuous_params(sys, params)
    skeleton_threaded = isnothing(threaded_skeleton) ? !threaded : threaded_skeleton
    reuse_seeds = isnothing(reuse_neighbor_seeds) ? !threaded : reuse_neighbor_seeds
    period_threaded = threaded && Threads.nthreads() > 1 && !reuse_seeds && length(periods) > 1

    if period_threaded
        tasks = map(periods) do period
            Threads.@spawn begin
                try
                    period_candidates = _continuation_period_candidates(
                        sys,
                        config,
                        period,
                        skeleton_params,
                        base_params;
                        initial_point=initial_point,
                        record=record,
                        search_min=search_min,
                        search_max=search_max,
                        n_initial=n_initial,
                        trajectory_seed_points=trajectory_seed_points,
                        trajectory_seed_crossings=trajectory_seed_crossings,
                        trajectory_seed_transient=trajectory_seed_transient,
                        reuse_neighbor_seeds=false,
                        tol=tol,
                        max_iter=max_iter,
                        fd_step=fd_step,
                        solver=solver,
                        reltol=reltol,
                        abstol=abstol,
                        tmax=tmax,
                        min_crossing_time=min_crossing_time,
                        threaded=false,
                        threaded_skeleton=false,
                        threaded_branches=false,
                        skeleton_cache_resolver=skeleton_cache_resolver,
                        on_error=on_error
                    )
                    _collect_distinct_period_branches(
                        period_candidates,
                        max_branches_per_period,
                        signature_param_tol,
                        signature_state_tol
                    )
                catch err
                    err isa InterruptException && rethrow()
                    _report_continuation_error(on_error, "Continuation batch for period $period failed", err)
                    BranchResult[]
                end
            end
        end

        return reduce(vcat, fetch.(tasks); init=BranchResult[])
    end

    for period in periods
        try
            period_candidates = _continuation_period_candidates(
                sys,
                config,
                period,
                skeleton_params,
                base_params;
                record=record,
                search_min=search_min,
                search_max=search_max,
                n_initial=n_initial,
                trajectory_seed_points=trajectory_seed_points,
                trajectory_seed_crossings=trajectory_seed_crossings,
                trajectory_seed_transient=trajectory_seed_transient,
                reuse_neighbor_seeds=reuse_seeds,
                tol=tol,
                max_iter=max_iter,
                fd_step=fd_step,
                solver=solver,
                reltol=reltol,
                abstol=abstol,
                tmax=tmax,
                min_crossing_time=min_crossing_time,
                threaded=threaded,
                threaded_skeleton=skeleton_threaded,
                threaded_branches=threaded && !reuse_seeds && length(skeleton_params) <= 1,
                skeleton_cache_resolver=skeleton_cache_resolver,
                on_error=on_error
            )

            append!(results, _collect_distinct_period_branches(
                period_candidates,
                max_branches_per_period,
                signature_param_tol,
                signature_state_tol
            ))
        catch err
            err isa InterruptException && rethrow()
            _report_continuation_error(on_error, "Continuation batch for period $period failed", err)
        end
    end

    return results
end
