# GPU-accelerated continuous-time Poincaré-map sweeps (`bifurcation_map` / `basins_of_attraction`
# for `sys::ContinuousODE`), built on DiffEqGPU's `EnsembleGPUKernel`. This is the *same* numerical
# method the CPU path uses — adaptive integration (GPU `Tsit5`) with a directional Poincaré
# `ContinuousCallback` (RightRootFind), warmup/min-crossing handling, termination after the required
# crossings, and Float64 closure-based period detection — distributed across an ensemble of
# independent per-cell trajectories instead of Julia threads. It is *not* a fixed-step or Float32
# reimplementation.
#
# How the branchy CPU detector becomes a uniform GPU kernel: `EnsembleGPUKernel` needs an out-of-place
# StaticArray RHS and a fixed-size StaticArray state, and the period detector's variable-length,
# closure-terminated control flow cannot be a growing host `Vector`. So each trajectory carries a
# fixed-capacity *augmented state* whose extra slots are the running detector registers (crossing
# count, base point, min-closure error/period/threshold, detected period, status, last-crossing point).
# The augmented RHS evolves the physical state via the system's `f_svector` and holds every extra slot
# constant (zero derivative); the directional callback runs the exact CPU `on_crossing!` closure logic
# on those slots and calls `terminate!` per trajectory. The physical result read back from the terminal
# augmented state is fed through the *same* `_record_map_detection!` / crossing-summary machinery as the
# CPU sweep, so the two cannot classify differently beyond integration tolerance.
#
# Tolerance note: DiffEqGPU localizes the section crossing with a fixed root-find tolerance
# (`100*eps(Float32)`, from `Base.convert(GPUContinuousCallback, ::ContinuousCallback)`). Callers whose
# closure `precision` is tighter than this floor are rejected up front (`_continuous_gpu_precision_ok`)
# rather than silently given a weaker localization — see `compute_backend.jl`.

# One scalar sentinel standing in for `Inf` inside the augmented state: keeping genuine `Inf` out of the
# integrated vector avoids any `Inf`-arithmetic surprises in the adaptive controller (the augmented
# slots carry zero derivative, but a finite sentinel is portable across every KernelAbstractions
# backend). Decoded back to `Inf` on read-back.
const _CONTINUOUS_GPU_SENTINEL = 1.0e300

# Augmented-state layout for physical dim `D` and projection dim `P`:
#   1 .. D                     physical state u
#   D+1                        crossing count
#   D+2                        observed post-transient crossings
#   D+3                        detected period
#   D+4                        status code (0 running, 1 periodic, 2 aperiodic, 3 diverged, 6 invalid)
#   D+5                        min closure error
#   D+6                        min-closure candidate period
#   D+7                        min-closure threshold
#   D+8                        base-point norm
#   D+9                        time of last section crossing
#   D+10   .. D+9+P            base (first post-transient) projected coordinates
#   D+10+P .. D+9+P+D          last-crossing physical coordinates (final_point)
# Total N = 2D + 9 + P.
@inline _continuous_gpu_state_dim(D::Int, P::Int) = 2D + 9 + P

@inline function _continuous_gpu_indices(::Val{D}, ::Val{P}) where {D, P}
    return (CC = D + 1, OBS = D + 2, DP = D + 3, ST = D + 4, ME = D + 5,
            MP = D + 6, MT = D + 7, BN = D + 8, FT = D + 9, BASE = D + 9, LP = D + 9 + P)
end

# Augmented out-of-place RHS: physical derivatives from the system's `f_svector`, zero elsewhere.
@inline function _continuous_gpu_rhs(f, u::SVector{N, T}, p, t, ::Val{D}) where {N, T, D}
    uphys = SVector{D, T}(ntuple(i -> u[i], Val(D)))
    dphys = f(uphys, p, t)
    return SVector{N, T}(ntuple(i -> i <= D ? dphys[min(i, D)] : zero(T), Val(N)))
end

# Per-accepted-step state check, run as a `DiscreteCallback` alongside the section `ContinuousCallback`
# in a `CallbackSet` — mirrors the CPU `_map_state_termination_callback` exactly (and independently of
# `on_crossing!`/`_make_continuous_gpu_affect`, exactly as the CPU's `state_cb` is independent of
# `section_cb`): checked on *every* accepted integrator step, not only at Poincaré crossings, so a state
# excursion that exceeds `divergence_cutoff` (or goes non-finite) between two crossings — and might later
# have returned to a bounded regime — is still caught and terminates the trajectory immediately, exactly
# like the CPU path. Sets the same ST/FT/last-crossing-coordinate slots `_decode_continuous_gpu_result`
# already reads, using the *current* (divergent) physical state as `final_point`, matching the CPU
# callback's `final_point[] = collect(Float64, integrator.u)`.
@inline function _continuous_gpu_state_status_code(u::SVector{N, T}, ::Val{D}, divergence_cutoff::Float64) where {N, T, D}
    invalid = false
    diverged = false
    @inbounds for k in 1:D
        uk = u[k]
        isfinite(uk) || (invalid = true)
        (isfinite(divergence_cutoff) && abs(uk) > divergence_cutoff) && (diverged = true)
    end
    invalid && return 6.0   # matches the ST status-code convention in the layout comment above
    diverged && return 3.0
    return 0.0
end

function _continuous_gpu_state_termination_callback(::Val{D}, ::Val{P}, divergence_cutoff::Float64) where {D, P}
    idx = _continuous_gpu_indices(Val(D), Val(P))
    ST = idx.ST; FT = idx.FT; LP = idx.LP
    condition = (u, t, integrator) -> _continuous_gpu_state_status_code(u, Val(D), divergence_cutoff) != 0.0
    affect! = function (integrator)
        u = integrator.u
        code = _continuous_gpu_state_status_code(u, Val(D), divergence_cutoff)
        u = setindex(u, code, ST)
        u = setindex(u, integrator.t, FT)
        @inbounds for k in 1:D
            u = setindex(u, u[k], LP + k)
        end
        integrator.u = u
        terminate!(integrator)
    end
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

# The per-crossing detector closure, run inside the ContinuousCallback affect!. Reproduces the CPU
# `on_crossing!` in `_detect_continuous_poincare_period` exactly, operating on the augmented slots.
# Divergence/invalid-state is handled entirely by `_continuous_gpu_state_termination_callback` above
# (not here), exactly as CPU's `on_crossing!` has no `_map_state_status` check of its own.
function _make_continuous_gpu_affect(::Val{D}, ::Val{P}, proj::SVector{P, Int},
                                     transient::Int, max_period::Int, precision::Float64,
                                     min_crossing_time::Float64) where {D, P}
    idx = _continuous_gpu_indices(Val(D), Val(P))
    CC = idx.CC; OBS = idx.OBS; DP = idx.DP; ST = idx.ST; ME = idx.ME
    MP = idx.MP; MT = idx.MT; BN = idx.BN; FT = idx.FT; BASE = idx.BASE; LP = idx.LP
    total_crossings = transient + max_period + 1
    return function (integrator)
        u = integrator.u
        t = integrator.t
        t <= min_crossing_time && return

        cc = u[CC] + 1.0
        u = setindex(u, cc, CC)
        u = setindex(u, t, FT)
        # Record the current (on-section) physical state as the running final_point.
        @inbounds for k in 1:D
            u = setindex(u, u[k], LP + k)
        end

        if cc > transient
            local_idx = cc - transient
            u = setindex(u, local_idx, OBS)
            if local_idx == 1.0
                cand_norm2 = 0.0
                @inbounds for k in 1:P
                    ck = u[proj[k]]
                    u = setindex(u, ck, BASE + k)
                    cand_norm2 += ck * ck
                end
                u = setindex(u, sqrt(cand_norm2), BN)
            else
                bn = u[BN]
                err2 = 0.0
                cand_norm2 = 0.0
                @inbounds for k in 1:P
                    ck = u[proj[k]]
                    bk = u[BASE + k]
                    d = bk - ck
                    err2 += d * d
                    cand_norm2 += ck * ck
                end
                err = sqrt(err2)
                scale = max(bn, sqrt(cand_norm2), 1.0)
                threshold = precision * scale
                if err < u[ME]
                    u = setindex(u, err, ME)
                    u = setindex(u, local_idx - 1.0, MP)
                    u = setindex(u, threshold, MT)
                end
                if err < threshold
                    u = setindex(u, local_idx - 1.0, DP)
                    u = setindex(u, 1.0, ST)
                    integrator.u = u
                    terminate!(integrator)
                    return
                end
            end
        end

        if cc >= total_crossings
            u = setindex(u, 2.0, ST)
            integrator.u = u
            terminate!(integrator)
            return
        end

        integrator.u = u
        return
    end
end

# Build the direction-aware Poincaré ContinuousCallback (converted to a GPUContinuousCallback by
# DiffEqGPU at solve time). Mirrors `_make_poincare_callback`'s directional affect!/affect_neg! wiring.
function _continuous_gpu_callback(section::PoincareSection, detect!)
    noop = (integrator) -> nothing
    positive_affect! = section.direction == -1 ? noop : detect!
    negative_affect! = section.direction == 1 ? nothing : detect!
    return ContinuousCallback(
        section.condition,
        positive_affect!;
        affect_neg! = negative_affect!,
        rootfind = SciMLBase.RightRootFind,
        save_positions = (false, false),
    )
end

# Decode one terminal augmented state into the result NamedTuple the CPU detector produces, so the
# shared recorders (`_record_map_detection!`, `_record_map_crossing_summary!`) consume it unchanged.
function _decode_continuous_gpu_result(uend::AbstractVector, ::Val{D}, ::Val{P},
                                       transient::Int, max_period::Int, local_tmax::Float64,
                                       solver_success::Bool) where {D, P}
    idx = _continuous_gpu_indices(Val(D), Val(P))
    st_code = Int(round(uend[idx.ST]))
    cc = Int(round(uend[idx.CC]))
    obs = Int(round(uend[idx.OBS]))
    det_p = Int(round(uend[idx.DP]))
    me = uend[idx.ME]; me = me >= _CONTINUOUS_GPU_SENTINEL / 2 ? Inf : Float64(me)
    mt = uend[idx.MT]; mt = mt >= _CONTINUOUS_GPU_SENTINEL / 2 ? Inf : Float64(mt)
    mp = Int(round(uend[idx.MP]))
    ft_raw = Float64(uend[idx.FT])

    status = st_code == 1 ? :periodic :
             st_code == 2 ? :aperiodic_or_high_period :
             st_code == 3 ? :diverged :
             st_code == 6 ? :invalid_state : :insufficient_crossings
    period = status == :periodic ? det_p : 0
    termination_reason = status == :periodic ? :period_detected :
                         status == :aperiodic_or_high_period ? :max_crossings_reached :
                         status == :diverged ? :diverged :
                         status == :invalid_state ? :invalid_state : :insufficient_crossings

    # FT is written on every crossing (cc > 0) *and* now on a divergence/invalid-state termination at
    # any point (`_continuous_gpu_state_termination_callback`, which can fire before any crossing) — so
    # only fall back to the integration horizon / NaN when neither ever happened.
    recorded_final_time = cc > 0 || status == :diverged || status == :invalid_state
    final_time = recorded_final_time ? ft_raw : (solver_success ? local_tmax : NaN)
    final_point = Float64[Float64(uend[idx.LP + k]) for k in 1:D]

    return (
        _period_detection_result(period, status, me, mp, obs, mt)...,
        final_point = final_point,
        total_crossings_found = cc,
        final_time = final_time,
        termination_reason = termination_reason,
        solver_retcode = solver_success ? :Terminated : :MaxIters,
        divergence_callback_activated = status == :diverged,
        state_callback_activated = status == :diverged || status == :invalid_state,
    )
end

"""
Per-trajectory `EnsembleProblem` `prob_func` that assigns cell `k`'s augmented initial state and
parameters. Defined as a callable struct with methods for both the SciMLBase ≥ 3 two-argument form
`(prob, ctx)` (selecting `ctx.sim_id`) and the legacy three-argument `(prob, i, repeat)` form, so the
continuous GPU path works across the SciMLBase/DiffEqGPU versions the library's compat bounds allow.
"""
struct _ContinuousEnsembleProbFunc{A, P}
    aug0::A
    p_sv::P
end
(f::_ContinuousEnsembleProbFunc)(prob, ctx) = remake(prob; u0 = f.aug0[ctx.sim_id], p = f.p_sv[ctx.sim_id])
(f::_ContinuousEnsembleProbFunc)(prob, i::Integer, repeat) = remake(prob; u0 = f.aug0[i], p = f.p_sv[i])

"""
Run a GPU ensemble of independent continuous-time Poincaré-map period detections — one per
`(u0_list[k], p_list[k])` trajectory — on `ka_backend` via DiffEqGPU's `EnsembleGPUKernel`, and return
a `Vector` of per-trajectory result NamedTuples in the CPU detector's exact shape.

`u0_list` must be *already warmed* physical initial states (host-side `_warmup_from_section`, so the GPU
path shares the CPU warmup semantics verbatim). Requires `has_continuous_gpu_rhs(sys)`.
"""
function _continuous_poincare_gpu_sweep(sys::ContinuousODE,
                                        u0_list::Vector{<:AbstractVector},
                                        p_list::Vector{<:AbstractVector},
                                        ka_backend;
                                        transient::Int,
                                        max_period::Int,
                                        precision::Float64,
                                        reltol::Float64,
                                        abstol::Float64,
                                        min_crossing_time::Float64,
                                        divergence_cutoff::Float64,
                                        alg = GPUTsit5())
    f_oop = continuous_gpu_rhs(sys)
    f_oop === nothing && throw(ArgumentError(
        "_continuous_poincare_gpu_sweep requires a system with a GPU out-of-place RHS (f_svector)."))
    n = length(u0_list)
    n > 0 || throw(ArgumentError(
        "_continuous_poincare_gpu_sweep requires at least one trajectory: u0_list/p_list must not be empty."))
    length(p_list) == n || throw(ArgumentError("u0_list and p_list must have equal length."))
    D = sys.dim
    P = length(sys.section.projection)
    N = _continuous_gpu_state_dim(D, P)
    PP = length(first(p_list))
    # Every trajectory shares one `SVector{PP}`/`SVector{D}` type, so a short vector anywhere in
    # `p_list`/`u0_list` would otherwise surface as an opaque `BoundsError` deep inside the `ntuple`
    # closures below (or silently read garbage on a real GPU device, which has no bounds checking).
    # Fail fast here with a message naming the offending index and the expected length instead.
    for (k, pv) in enumerate(p_list)
        length(pv) == PP || throw(ArgumentError(
            "p_list[$k] has length $(length(pv)); expected $PP (the length of p_list[1])."))
    end
    for (k, uv) in enumerate(u0_list)
        length(uv) == D || throw(ArgumentError(
            "u0_list[$k] has length $(length(uv)); expected the system dimension $D."))
    end
    proj = SVector{P, Int}(sys.section.projection)
    idx = _continuous_gpu_indices(Val(D), Val(P))

    total_crossings = transient + max_period + 1
    local_tmax = sys.tspan_hint * max(total_crossings, 1) * 2
    initial_dt = _default_poincare_initial_dt(min_crossing_time)

    # Build the augmented initial states (physical warmed state + detector registers seeded so that
    # min_error/min_threshold start at the sentinel and last-crossing coords start at the warmed state).
    aug0 = Vector{SVector{N, Float64}}(undef, n)
    for k in 1:n
        u0 = u0_list[k]
        aug0[k] = SVector{N, Float64}(ntuple(Val(N)) do i
            if i <= D
                Float64(u0[i])
            elseif i == idx.ME || i == idx.MT
                _CONTINUOUS_GPU_SENTINEL
            elseif i > idx.LP && i <= idx.LP + D
                Float64(u0[i - idx.LP])
            else
                0.0
            end
        end)
    end
    p_sv = [SVector{PP, Float64}(ntuple(i -> Float64(p_list[k][i]), Val(PP))) for k in 1:n]

    rhs_valD = Val(D)
    rhs = (u, p, t) -> _continuous_gpu_rhs(f_oop, u, p, t, rhs_valD)
    detect! = _make_continuous_gpu_affect(Val(D), Val(P), proj, transient, max_period, precision,
                                          min_crossing_time)
    section_cb = _continuous_gpu_callback(sys.section, detect!)
    state_cb = _continuous_gpu_state_termination_callback(Val(D), Val(P), divergence_cutoff)
    # Order mirrors the CPU `CallbackSet(section_cb, state_cb)` exactly (section/continuous callback
    # first, per-step state callback second) so a crossing and a divergence landing on the same step
    # resolve identically on both paths.
    cb = CallbackSet(section_cb, state_cb)

    prob = ODEProblem{false}(rhs, aug0[1], (0.0, local_tmax), p_sv[1])
    eprob = EnsembleProblem(prob; prob_func = _ContinuousEnsembleProbFunc(aug0, p_sv), safetycopy = false)

    sol = solve(eprob, alg, EnsembleGPUKernel(ka_backend);
                trajectories = n,
                adaptive = true,
                dt = initial_dt,
                callback = cb,
                merge_callbacks = true,
                save_everystep = false,
                save_start = false,
                save_end = true,
                reltol = reltol,
                abstol = abstol)

    results = Vector{Any}(undef, n)
    for k in 1:n
        traj = sol.u[k]
        uend = traj.u[end]
        solver_success = SciMLBase.successful_retcode(traj.retcode) ||
                         traj.retcode == SciMLBase.ReturnCode.Terminated
        results[k] = _decode_continuous_gpu_result(uend, Val(D), Val(P), transient, max_period,
                                                    local_tmax, solver_success)
    end
    return results
end

# Host-side warmup of each trajectory's initial state (shared verbatim with the CPU path so the GPU
# ensemble starts from identical on/off-section states). Threaded because each warmup is independent.
function _continuous_gpu_warmup_states(sys::ContinuousODE, u0_list::Vector{<:AbstractVector},
                                       p_list::Vector{<:AbstractVector};
                                       solver, reltol::Float64, abstol::Float64,
                                       min_crossing_time::Float64)
    n = length(u0_list)
    warmed = Vector{Vector{Float64}}(undef, n)
    initial_dt = _default_poincare_initial_dt(min_crossing_time)
    Threads.@threads for k in 1:n
        warmed[k] = _plain_float_vector(_warmup_from_section(
            sys, u0_list[k], p_list[k];
            solver = solver, reltol = reltol, abstol = abstol,
            min_crossing_time = min_crossing_time, initial_dt = initial_dt))
    end
    return warmed
end

# Record continuous-GPU solver/tolerance provenance onto a sweep's diagnostics dict, so a
# GPU-computed result honestly reports the (adaptive) GPU integrator and the fixed section-crossing
# localization tolerance it ran with.
function _record_continuous_gpu_provenance!(diagnostics::AbstractDict, config)
    diagnostics["gpuSolver"] = "GPUTsit5"
    diagnostics["gpuAdaptive"] = true
    diagnostics["gpuRootfindAbstol"] = _CONTINUOUS_GPU_ROOTFIND_ABSTOL
    diagnostics["closurePrecision"] = config.precision
    return diagnostics
end
