# GPU-kernel implementations of the embarrassingly-parallel per-cell sweeps: the fixed-seed 2D
# bifurcation map, the direct 2D Lyapunov field, and basins of attraction. Each kernel is written
# against `KernelAbstractions.jl` and calls the exact same non-allocating numeric cores as the CPU
# sweep (`_detect_discrete_map_period_core`, `_estimate_discrete_map_largest_lyapunov_core`) so the two
# code paths cannot drift apart — this is the same math, just distributed across a device's threads
# instead of Julia's.
#
# Every value flowing through these kernel bodies — the numeric cores' return NamedTuples, the
# classification helpers, the codes written to the output arrays — is a plain `Int`/`Float64`/`SVector`,
# never a `Symbol` (`Symbol` is not `isbits`; see the comment above `_map_state_status_code` in
# `map_diagnostics.jl`). The cores already emit the exact integer codes the output arrays store, so no
# Symbol<->code conversion happens inside a kernel; that conversion exists only at CPU host boundaries
# (`_detect_discrete_map_period`, `_estimate_discrete_map_largest_lyapunov`, `lyapunov_diagram`, …).
#
# GPU eligibility (checked by the caller before any of these are reached):
#   - `sys::DiscreteMap` only. Continuous-time (Poincaré-map) sweeps use an adaptive-step ODE
#     integrator with event-based section-crossing detection — a fundamentally branchy, variable-length
#     control flow that does not compile to a uniform GPU kernel without a fixed-step reimplementation
#     that would itself be a different (weaker) numerical method. See docs/julia-package.md.
#   - Fixed-seed traversal only (`reuse_neighbor_seeds=false`): the neighbour-seeded modes are
#     traversal-dependent (each cell's initial condition is the previous cell's final state in
#     serpentine order) and therefore not cell-independent.
#   - No switching events, no multistability seeds, no linked parameter indices: these require
#     per-cell `Dict`/`Vector`-of-`Dict` diagnostics or repeated multi-seed dispatch that are not (yet)
#     written in a GPU-safe non-allocating form.

@kernel function _bifurcation_map_gpu_kernel!(periodicity, status_codes, closure_errors,
                                              closure_candidate_periods, observed_points, closure_confidence,
                                              @Const(a_vals), @Const(b_vals), @Const(known),
                                              f::F, template::SVector{N, Float64}, a_index::Int, b_index::Int,
                                              x0::SVector{D, Float64}, transient::Int, max_period::Int,
                                              precision::Float64, divergence_cutoff::Float64) where {F, N, D}
    i, j = @index(Global, NTuple)
    @inbounds if !known[i, j]
        p = setindex(setindex(template, a_vals[i], a_index), b_vals[j], b_index)
        core = _detect_discrete_map_period_core(f, p, x0, transient, max_period, precision, divergence_cutoff)
        periodicity[i, j] = core.period
        status_codes[i, j] = core.status
        closure_errors[i, j] = core.min_closure_error
        closure_candidate_periods[i, j] = core.closure_candidate_period
        observed_points[i, j] = core.observed_points
        closure_confidence[i, j] = core.closure_confidence
    end
end

@kernel function _bifurcation_map_gpu_kernel_lyapunov!(periodicity, status_codes, closure_errors,
                                                       closure_candidate_periods, observed_points, closure_confidence,
                                                       lyap_exponents, lyap_status_codes, lyap_estimation_status_codes, lyap_sample_counts,
                                                       @Const(a_vals), @Const(b_vals), @Const(known),
                                                       f::F, template::SVector{N, Float64}, a_index::Int, b_index::Int,
                                                       x0::SVector{D, Float64}, transient::Int, max_period::Int,
                                                       precision::Float64, divergence_cutoff::Float64,
                                                       lyap_transient::Int, lyap_iterations::Int,
                                                       lyap_perturbation::Float64, lyap_neutral_tolerance::Float64) where {F, N, D}
    i, j = @index(Global, NTuple)
    @inbounds if !known[i, j]
        p = setindex(setindex(template, a_vals[i], a_index), b_vals[j], b_index)
        core = _detect_discrete_map_period_core(f, p, x0, transient, max_period, precision, divergence_cutoff)
        periodicity[i, j] = core.period
        status_codes[i, j] = core.status
        closure_errors[i, j] = core.min_closure_error
        closure_candidate_periods[i, j] = core.closure_candidate_period
        observed_points[i, j] = core.observed_points
        closure_confidence[i, j] = core.closure_confidence

        estimate = _estimate_discrete_map_largest_lyapunov_core(f, p, core.final_point, lyap_transient, lyap_iterations, lyap_perturbation, divergence_cutoff)
        exponent = Float64(estimate.exponent)
        lyap_exponents[i, j] = exponent
        lyap_status_codes[i, j] = _map_lyapunov_classification(core.period, core.status, exponent, estimate.estimation_status, lyap_neutral_tolerance)
        lyap_estimation_status_codes[i, j] = estimate.estimation_status
        lyap_sample_counts[i, j] = estimate.sample_count
    end
end

@kernel function _lyapunov_field_gpu_kernel!(exponents, status_codes, estimation_status_codes, sample_counts,
                                             @Const(a_vals), @Const(b_vals), @Const(known),
                                             f::F, template::SVector{N, Float64}, a_index::Int, b_index::Int,
                                             x0::SVector{D, Float64}, transient::Int, iterations::Int,
                                             perturbation::Float64, divergence_cutoff::Float64,
                                             neutral_tolerance::Float64) where {F, N, D}
    i, j = @index(Global, NTuple)
    @inbounds if !known[i, j]
        p = setindex(setindex(template, a_vals[i], a_index), b_vals[j], b_index)
        estimate = _estimate_discrete_map_largest_lyapunov_core(f, p, x0, transient, iterations, perturbation, divergence_cutoff)
        exponent = Float64(estimate.exponent)
        exponents[i, j] = exponent
        status_codes[i, j] = _lyapunov_point_classification(exponent, estimate.estimation_status, neutral_tolerance)
        estimation_status_codes[i, j] = estimate.estimation_status
        sample_counts[i, j] = estimate.sample_count
    end
end

@kernel function _basins_gpu_kernel!(periodicity, @Const(x_vals), @Const(y_vals), @Const(known),
                                     f::F, p::SVector{P, Float64}, base_ic::SVector{D, Float64},
                                     x_index::Int, y_index::Int, transient::Int, max_period::Int,
                                     precision::Float64) where {F, P, D}
    i, j = @index(Global, NTuple)
    @inbounds if !known[i, j]
        x0 = setindex(setindex(base_ic, x_vals[i], x_index), y_vals[j], y_index)
        # divergence_cutoff = Inf: basins has no state-amplitude cutoff of its own (unlike
        # BifurcationMapConfig) — matches the CPU basins loop, which iterates unconditionally and
        # lets a diverging orbit poison the closure comparison (never closes ⇒ period 0) rather than
        # short-circuiting on an explicit cutoff.
        core = _detect_discrete_map_period_core(f, p, x0, transient, max_period, precision, Inf)
        periodicity[i, j] = core.period
    end
end


_gpu_known_mask(cells::Nothing, na::Int, nb::Int) = fill(false, na, nb)
_gpu_known_mask(cells, na::Int, nb::Int) = cells.known

# Upload a (possibly `nothing`) cache-hook grid's current contents (so pre-seeded/known cells survive
# the round trip), launch `kernel!` over the full (na, nb) range, then copy the results back — honoring
# the cache hook exactly like the CPU `:fixed` path (skip known cells, mark all cells known afterwards).
function _gpu_run_2d_sweep!(ka_backend, na::Int, nb::Int, host_arrays::Tuple,
                            kernel_fn, cells, extra_args...)
    known_host = _gpu_known_mask(cells, na, nb)
    known_dev = _gpu_upload(ka_backend, known_host)
    dev_arrays = map(arr -> _gpu_upload(ka_backend, arr), host_arrays)

    a_vals_dev, b_vals_dev = extra_args[1], extra_args[2]
    kernel_fn(ka_backend)(dev_arrays..., a_vals_dev, b_vals_dev, known_dev, extra_args[3:end]...; ndrange=(na, nb))
    KernelAbstractions.synchronize(ka_backend)

    # Copy directly into existing host buffers to avoid an intermediate `Array(dev)` allocation.
    for (host, dev) in zip(host_arrays, dev_arrays)
        copyto!(host, dev)
    end
    cells !== nothing && fill!(cells.known, true)
    return nothing
end
