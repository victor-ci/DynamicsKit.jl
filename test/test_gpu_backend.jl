# GPU-backend selection API + eligibility/parity behaviour for bifurcation_map / lyapunov_field /
# basins_of_attraction. No real GPU vendor extension ships in this release (see
# docs/julia-package.md, "Optional GPU acceleration"), so the GPU *kernel* code path is exercised via
# a private, non-public test seam (`GPUBackend(:_ka_cpu_test)`) that resolves to
# `KernelAbstractions.CPU()` — this validates the exact upload/launch/copy-back/cache-hook-merge
# machinery a real vendor backend would run through, without requiring GPU hardware.
#
# `Metal` is loaded here as a real, hardware-gated check: this host has a physical Apple GPU, so the
# tests assert the library's actual (intentional, documented) refusal to use it for double-precision
# science, rather than skipping the vendor entirely.

using Metal

@testset "GPU compute backend" begin
    BE = DynamicsKit

    @testset "backend selector construction and equality" begin
        @test cpu_backend() isa CPUBackend
        @test auto_backend() isa AutoBackend
        @test gpu_backend(:cuda) isa GPUBackend
        @test gpu_vendor(gpu_backend(:cuda)) == :cuda
        @test gpu_backend(:cuda) == gpu_backend(:cuda)          # singleton-typed, so `==` holds
        @test CPUBackend() isa ComputeBackend
        @test AutoBackend() isa ComputeBackend
        @test GPUBackend(:cuda) isa ComputeBackend
    end

    @testset "availability query: no working vendor in this session" begin
        @test available_gpu_backends() == Symbol[]
        @test gpu_backend_available(:cuda) == false
        @test gpu_backend_available(:amdgpu) == false
        @test gpu_backend_available(gpu_backend(:cuda)) == false
    end

    @testset "Metal is a real, hardware-present, permanently-disqualified vendor" begin
        # This assertion is the point of loading Metal here: the host GPU is real and detected, but
        # DynamicsKit must still refuse it for double-precision science.
        @test gpu_backend_available(:metal) == false
        @test :metal ∉ available_gpu_backends()
    end

    sys = henon_map()
    cfg = BifurcationMapConfig(a_min=1.0, a_max=1.4, a_steps=20, b_min=0.2, b_max=0.35, b_steps=16,
                               a_index=1, b_index=2, base_params=[1.4, 0.3],
                               max_period=8, iterations=300, precision=1e-3, divergence_cutoff=1e6,
                               lyapunov_enabled=true, lyapunov_iterations=64)

    @testset "auto backend never errors and matches the CPU result" begin
        r_cpu = bifurcation_map(sys, cfg)
        r_auto = bifurcation_map(sys, cfg; backend=auto_backend())
        @test r_auto.periodicity == r_cpu.periodicity
        @test r_auto.compute_backend == :cpu
        @test r_cpu.compute_backend == :cpu

        b_cfg = BasinsConfig(bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
                             x_min=-0.4, x_max=0.4, x_steps=10, y_min=-0.4, y_max=0.4, y_steps=10,
                             x_index=1, y_index=2, max_period=8, iterations=300, precision=1e-3)
        @test basins_of_attraction(sys, b_cfg; backend=auto_backend()).compute_backend == :cpu

        @test lyapunov_field(sys, cfg; backend=auto_backend()).compute_backend == :cpu
    end

    @testset "explicit unavailable vendor errors clearly, never silently falls back" begin
        @test_throws ArgumentError bifurcation_map(sys, cfg; backend=gpu_backend(:cuda))
        try
            bifurcation_map(sys, cfg; backend=gpu_backend(:cuda))
        catch e
            @test e isa ArgumentError
            @test occursin("cuda", lowercase(e.msg))
            @test occursin("not available", lowercase(e.msg))
        end
    end

    @testset "explicit Metal request errors with the FP64 reason, not a generic message" begin
        @test_throws ArgumentError bifurcation_map(sys, cfg; backend=gpu_backend(:metal))
        try
            bifurcation_map(sys, cfg; backend=gpu_backend(:metal))
        catch e
            @test e isa ArgumentError
            @test occursin("double-precision", lowercase(e.msg)) || occursin("float64", lowercase(e.msg))
        end
    end

    @testset "explicit GPU request on an ineligible bifurcation_map config errors, auto falls back" begin
        neighbor_cfg = BifurcationMapConfig(a_min=1.0, a_max=1.4, a_steps=10, b_min=0.2, b_max=0.35, b_steps=8,
                                            a_index=1, b_index=2, base_params=[1.4, 0.3],
                                            max_period=8, iterations=300, precision=1e-3,
                                            reuse_neighbor_seeds=true)
        @test_throws ArgumentError bifurcation_map(sys, neighbor_cfg; backend=gpu_backend(:_ka_cpu_test))
        @test_throws ArgumentError BE._bifurcation_map(sys, neighbor_cfg; backend=BE.GPUBackend(:_ka_cpu_test))
        # Auto never errors on ineligibility either — it just runs on the CPU.
        r_auto = bifurcation_map(sys, neighbor_cfg; backend=auto_backend())
        @test r_auto.compute_backend == :cpu

        multistability_cfg = BifurcationMapConfig(a_min=1.0, a_max=1.4, a_steps=10, b_min=0.2, b_max=0.35, b_steps=8,
                                                   a_index=1, b_index=2, base_params=[1.4, 0.3],
                                                   max_period=8, iterations=300, precision=1e-3,
                                                   multistability_initial_points=[[0.1, 0.1]])
        @test_throws ArgumentError BE._bifurcation_map(sys, multistability_cfg; backend=BE.GPUBackend(:_ka_cpu_test))

        linked_cfg = BifurcationMapConfig(a_min=1.0, a_max=1.4, a_steps=10, b_min=0.2, b_max=0.35, b_steps=8,
                                          a_index=1, b_index=2, base_params=[1.4, 0.3, 0.3],
                                          a_linked_param_indices=[3],
                                          max_period=8, iterations=300, precision=1e-3)
        @test_throws ArgumentError BE._bifurcation_map(sys, linked_cfg; backend=BE.GPUBackend(:_ka_cpu_test))

        switching_sys = DiscreteMap((x, p) -> x, 1, [:a], "switching-stub";
                                    switching_events=[SwitchingEvent("s", (x, p) -> x[1])])
        switching_cfg = BifurcationMapConfig(a_min=0.0, a_max=1.0, a_steps=4, b_min=0.0, b_max=1.0, b_steps=4,
                                             a_index=1, b_index=1, max_period=4, iterations=100, precision=1e-3)
        @test_throws ArgumentError BE._bifurcation_map(switching_sys, switching_cfg; backend=BE.GPUBackend(:_ka_cpu_test))
    end

    @testset "explicit GPU request on ContinuousODE always errors (architectural, not a missing extension)" begin
        # Harmonic oscillator: u = (cos t, -sin t) from [1, 0] gives a clean, fast-terminating
        # period-1 orbit (same minimal fixture pattern as the existing map_compute crossing-summary
        # test) — this test only needs the call to complete quickly, not to be dynamically interesting.
        oscillator = ContinuousODE(
            (du, u, _p, _t) -> (du[1] = u[2]; du[2] = -u[1]),
            2,
            PoincareSection((u, _t, _integrator) -> u[2]; direction=:up, projection=[1], template=[0.0, 0.0]),
            [:a, :b],
            "GPU-rejection harmonic oscillator";
            tspan_hint=10.0,
            default_initial_state=[1.0, 0.0],
            default_params=[0.0, 0.0]
        )
        map_cfg = BifurcationMapConfig(
            a_min=0.0, a_max=0.0, a_steps=0, b_min=0.0, b_max=0.0, b_steps=0,
            a_index=1, b_index=2, max_period=4, precision=1e-3, iterations=8, base_params=[0.0, 0.0]
        )
        @test_throws ArgumentError bifurcation_map(oscillator, map_cfg; backend=gpu_backend(:_ka_cpu_test))
        @test bifurcation_map(oscillator, map_cfg; backend=auto_backend()) isa BifurcationMapResult

        basins_cfg = BasinsConfig(
            bif_param=0.0, fixed_params=[0.0, 0.0], x_min=0.5, x_max=1.5, x_steps=1,
            y_min=-0.5, y_max=0.5, y_steps=1, max_period=4, precision=1e-3, iterations=8
        )
        @test_throws ArgumentError basins_of_attraction(oscillator, basins_cfg; backend=gpu_backend(:_ka_cpu_test))
        @test basins_of_attraction(oscillator, basins_cfg; backend=auto_backend()) isa BasinsResult
    end

    @testset "GPU-kernel parity vs CPU reference (via internal KernelAbstractions-CPU test seam)" begin
        r_cpu, diag_cpu = BE._bifurcation_map(sys, cfg)
        r_gpu, diag_gpu = BE._bifurcation_map(sys, cfg; backend=BE.GPUBackend(:_ka_cpu_test))

        @test r_gpu.periodicity == r_cpu.periodicity
        @test r_gpu.compute_backend == :_ka_cpu_test
        @test diag_gpu["computeBackend"] == "_ka_cpu_test"
        @test r_gpu.lyapunov.exponents ≈ r_cpu.lyapunov.exponents nans=true
        @test r_gpu.lyapunov.classification_status_codes == r_cpu.lyapunov.classification_status_codes
        @test r_gpu.lyapunov.estimation_status_codes == r_cpu.lyapunov.estimation_status_codes
        @test r_gpu.lyapunov.sample_counts == r_cpu.lyapunov.sample_counts
        @test r_gpu.lyapunov.compute_backend == :_ka_cpu_test

        b_cfg = BasinsConfig(bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
                             x_min=-0.4, x_max=0.4, x_steps=16, y_min=-0.4, y_max=0.4, y_steps=16,
                             x_index=1, y_index=2, max_period=8, iterations=300, precision=1e-3)
        b_cpu = basins_of_attraction(sys, b_cfg)
        b_gpu = basins_of_attraction(sys, b_cfg; backend=gpu_backend(:_ka_cpu_test))
        @test b_gpu.periodicity == b_cpu.periodicity
        @test b_gpu.compute_backend == :_ka_cpu_test

        lf_cpu = lyapunov_field(sys, cfg)
        lf_gpu = lyapunov_field(sys, cfg; backend=gpu_backend(:_ka_cpu_test))
        @test lf_gpu.exponents ≈ lf_cpu.exponents nans=true
        @test lf_gpu.classification_status_codes == lf_cpu.classification_status_codes
        @test lf_gpu.estimation_status_codes == lf_cpu.estimation_status_codes
        @test lf_gpu.sample_counts == lf_cpu.sample_counts
        @test lf_gpu.compute_backend == :_ka_cpu_test
    end

    @testset "a divergent/chaotic fixture still gives GPU/CPU parity" begin
        # Regression guard for the divergence_cutoff=Inf choice in the basins GPU kernel: an orbit
        # that truly diverges must still classify identically (period 0) on both paths.
        b_cfg = BasinsConfig(bif_param=2.2, param_index=1, fixed_params=[2.2, 0.3],
                             x_min=-3.0, x_max=3.0, x_steps=12, y_min=-3.0, y_max=3.0, y_steps=12,
                             x_index=1, y_index=2, max_period=6, iterations=60, precision=1e-3)
        b_cpu = basins_of_attraction(sys, b_cfg)
        b_gpu = basins_of_attraction(sys, b_cfg; backend=gpu_backend(:_ka_cpu_test))
        @test b_gpu.periodicity == b_cpu.periodicity
        @test any(==(0), b_cpu.periodicity)   # fixture actually exercises the diverging/aperiodic path
    end

    @testset "cells= cache hook honored under a GPU backend (skip known, mark computed known)" begin
        na, nb = cfg.a_steps + 1, cfg.b_steps + 1
        rf, _ = BE._bifurcation_map(sys, cfg; backend=BE.GPUBackend(:_ka_cpu_test))
        g1 = MapCellGrid(na, nb; lyapunov=true)
        r1, _ = BE._bifurcation_map(sys, cfg; cells=g1, backend=BE.GPUBackend(:_ka_cpu_test))
        @test r1.periodicity == rf.periodicity
        @test all(g1.known)

        g2 = MapCellGrid(na, nb; lyapunov=true)
        for i in 1:na, j in 1:nb
            if (i + j) % 2 == 0
                g2.periodicity[i, j]               = g1.periodicity[i, j]
                g2.status_codes[i, j]              = g1.status_codes[i, j]
                g2.closure_errors[i, j]             = g1.closure_errors[i, j]
                g2.closure_candidate_periods[i, j]  = g1.closure_candidate_periods[i, j]
                g2.observed_points[i, j]            = g1.observed_points[i, j]
                g2.closure_confidence[i, j]         = g1.closure_confidence[i, j]
                g2.lyapunov.exponents[i, j]         = g1.lyapunov.exponents[i, j]
                g2.lyapunov.status_codes[i, j]      = g1.lyapunov.status_codes[i, j]
                g2.lyapunov.estimation_status_codes[i, j] = g1.lyapunov.estimation_status_codes[i, j]
                g2.lyapunov.sample_counts[i, j]      = g1.lyapunov.sample_counts[i, j]
                g2.known[i, j] = true
            end
        end
        r2, _ = BE._bifurcation_map(sys, cfg; cells=g2, backend=BE.GPUBackend(:_ka_cpu_test))
        @test r2.periodicity == rf.periodicity
        @test all(g2.known)

        nx, ny = 10 + 1, 10 + 1
        b_cfg = BasinsConfig(bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
                             x_min=-0.4, x_max=0.4, x_steps=10, y_min=-0.4, y_max=0.4, y_steps=10,
                             x_index=1, y_index=2, max_period=8, iterations=300, precision=1e-3)
        bf = basins_of_attraction(sys, b_cfg; backend=gpu_backend(:_ka_cpu_test))
        bg1 = BasinsCellGrid(nx, ny)
        br1 = basins_of_attraction(sys, b_cfg; cells=bg1, backend=gpu_backend(:_ka_cpu_test))
        @test br1.periodicity == bf.periodicity
        @test all(bg1.known)
        bg2 = BasinsCellGrid(nx, ny)
        for i in 1:nx, j in 1:ny
            if (i + j) % 2 == 0
                bg2.periodicity[i, j] = bg1.periodicity[i, j]
                bg2.known[i, j] = true
            end
        end
        br2 = basins_of_attraction(sys, b_cfg; cells=bg2, backend=gpu_backend(:_ka_cpu_test))
        @test br2.periodicity == bf.periodicity
        @test all(bg2.known)
    end

    @testset "device-array copy-back avoids an intermediate Array(dev) allocation" begin
        # `_gpu_run_2d_sweep!` downloads each device output array straight into its pre-allocated host
        # array with `copyto!(host, dev)`. Going through `Array(dev)` first allocates and populates a
        # full-size throwaway array before the copy — this regresses if that intermediate returns.
        ka_backend = BE._dynamicskit_gpu_backend(Val(:_ka_cpu_test))
        host = zeros(64, 64)
        dev = BE._gpu_upload(ka_backend, rand(64, 64))
        copyto!(host, dev)   # warm up / compile before measuring
        @test host == Array(dev)
        direct_bytes = @allocated copyto!(host, dev)
        via_intermediate_bytes = @allocated copyto!(host, Array(dev))
        @test direct_bytes == 0
        @test direct_bytes < via_intermediate_bytes
    end

    @testset "status-code GPU/Dict lockstep (every known status maps identically)" begin
        for status in (:unknown, :periodic, :aperiodic_or_high_period, :diverged,
                       :insufficient_crossings, :integration_failed, :invalid_state)
            @test BE._gpu_map_status_code(status) == BE._map_status_code(status)
            @test BE._map_status_symbol(BE._map_status_code(status)) == status
        end
        for status in (:uncomputed, :periodic, :chaotic_candidate, :quasiperiodic_neutral_candidate, :unresolved)
            @test BE._gpu_map_lyapunov_status_code(status) == BE._map_lyapunov_status_code(status)
            @test BE._map_lyapunov_status_symbol(BE._map_lyapunov_status_code(status)) == status
        end
        for status in (:not_requested, :ok, :collapsed, :diverged, :invalid_state,
                       :insufficient_crossings, :integration_failed, :insufficient_samples)
            @test BE._gpu_map_lyapunov_estimation_status_code(status) == BE._map_lyapunov_estimation_status_code(status)
            @test BE._map_lyapunov_estimation_status_symbol(BE._map_lyapunov_estimation_status_code(status)) == status
        end
    end

    @testset "Lyapunov classification codes stay in lockstep with their documented Symbol meaning" begin
        # Independent Symbol-level re-implementation of `_map_lyapunov_classification`'s and
        # `_lyapunov_point_classification`'s decision trees, written directly from their docstrings and
        # *not* sharing the named numeric constants those functions use — this is what actually catches
        # the isbits code space silently drifting away from the Dict-defined Symbol meaning, which a
        # type-only (`isbits`) check cannot.
        reference_map_lyapunov_classification = function (detection_period::Int, detection_status::Symbol,
                                                           exponent::Float64, estimation_status::Symbol,
                                                           neutral_tolerance::Float64)
            detection_period > 0 && return :periodic
            detection_status == :aperiodic_or_high_period || return :unresolved
            estimation_status in (:ok, :collapsed) || return :unresolved
            estimation_status == :collapsed && return :periodic
            isfinite(exponent) || return :unresolved
            exponent > neutral_tolerance && return :chaotic_candidate
            abs(exponent) <= neutral_tolerance && return :quasiperiodic_neutral_candidate
            return :periodic
        end
        reference_lyapunov_point_classification = function (exponent::Float64, estimation_status::Symbol,
                                                             neutral_tolerance::Float64)
            estimation_status == :collapsed && return :periodic
            estimation_status == :ok || return :unresolved
            isfinite(exponent) || return :unresolved
            exponent > neutral_tolerance && return :chaotic_candidate
            abs(exponent) <= neutral_tolerance && return :quasiperiodic_neutral_candidate
            return :periodic
        end

        detection_statuses = (:periodic, :aperiodic_or_high_period, :diverged, :insufficient_crossings,
                              :integration_failed, :invalid_state, :unknown)
        estimation_statuses = (:not_requested, :ok, :collapsed, :diverged, :invalid_state,
                               :insufficient_crossings, :integration_failed, :insufficient_samples)
        exponents = (-0.5, -1e-4, 0.0, 1e-4, 0.5, NaN, -Inf, Inf)
        neutral_tolerance = 1e-3

        for detection_status in detection_statuses, period in (0, 3),
            estimation_status in estimation_statuses, exponent in exponents

            expected = reference_map_lyapunov_classification(period, detection_status, exponent, estimation_status, neutral_tolerance)
            actual_code = BE._map_lyapunov_classification(period, BE._map_status_code(detection_status), exponent,
                                                          BE._map_lyapunov_estimation_status_code(estimation_status), neutral_tolerance)
            @test BE._map_lyapunov_status_symbol(actual_code) == expected

            field_expected = reference_lyapunov_point_classification(exponent, estimation_status, neutral_tolerance)
            field_actual_code = BE._lyapunov_point_classification(exponent, BE._map_lyapunov_estimation_status_code(estimation_status), neutral_tolerance)
            @test BE._map_lyapunov_status_symbol(field_actual_code) == field_expected
        end

        # `_map_status_code_from_state_code` / `_lyapunov_estimation_code_from_state_code`: the 3-value
        # internal state-code translators used only for a non-`:ok` state. Their comment previously
        # claimed lockstep with the public Dicts "asserted by test" with no such test in place; this
        # closes that gap.
        @test BE._map_status_code_from_state_code(BE._STATE_CODE_DIVERGED) == BE._map_status_code(:diverged)
        @test BE._map_status_code_from_state_code(BE._STATE_CODE_INVALID) == BE._map_status_code(:invalid_state)
        @test BE._lyapunov_estimation_code_from_state_code(BE._STATE_CODE_DIVERGED) == BE._map_lyapunov_estimation_status_code(:diverged)
        @test BE._lyapunov_estimation_code_from_state_code(BE._STATE_CODE_INVALID) == BE._map_lyapunov_estimation_status_code(:invalid_state)
    end

    @testset "device-safe status codes: isbits kernel-return types, no Symbol reachable from a kernel" begin
        # `Symbol` is not `isbits` (`isbits(:ok) === false`) and must never flow through GPU-compiled
        # code. This locks the exact functions `gpu_kernels.jl` calls from inside a `@kernel` body to a
        # fully isbits return type, so a real GPU vendor backend (unlike KernelAbstractions.CPU, which
        # tolerates boxed values) cannot silently receive a non-device-safe value here.
        p = [1.35, 0.3]
        x0 = zeros(SVector{2, Float64})
        core = BE._detect_discrete_map_period_core(sys.f, p, x0, 100, 8, 1e-3, 1e6)
        @test isbits(core)
        @test core.status isa Int

        estimate_core = BE._estimate_discrete_map_largest_lyapunov_core(sys.f, p, x0, 50, 64, 1e-8, 1e6)
        @test isbits(estimate_core)
        @test estimate_core.estimation_status isa Int

        classification_code = BE._map_lyapunov_classification(core.period, core.status, 0.1, estimate_core.estimation_status, 1e-3)
        @test classification_code isa Int
        @test isbits(classification_code)

        field_classification_code = BE._lyapunov_point_classification(0.1, estimate_core.estimation_status, 1e-3)
        @test field_classification_code isa Int
        @test isbits(field_classification_code)

        # Internal 3-value state code the cores share (`_map_state_status_code`) is likewise isbits.
        state_code = BE._map_state_status_code(x0, 1e6)
        @test state_code isa Int
        @test isbits(state_code)
    end

    @testset "shared numeric core matches the existing CPU period/Lyapunov kernels exactly" begin
        p = [1.35, 0.3]
        x0 = zeros(SVector{2, Float64})
        core = BE._detect_discrete_map_period_core(sys.f, p, x0, 100, 8, 1e-3, 1e6)
        legacy = BE._detect_discrete_map_period(sys, p, x0, 100, 8, 1e-3, 1e6)
        @test core.period == legacy.period
        @test core.status isa Int
        @test legacy.status isa Symbol
        @test BE._map_status_symbol(core.status) == legacy.status
        @test core.status == BE._map_status_code(legacy.status)
        @test core.min_closure_error == legacy.min_closure_error
        @test core.closure_candidate_period == legacy.closure_candidate_period
        @test core.observed_points == legacy.observed_points
        @test core.closure_confidence == legacy.closure_confidence
        @test core.final_point == legacy.final_point

        estimate_core = BE._estimate_discrete_map_largest_lyapunov_core(sys.f, p, x0, 50, 64, 1e-8, 1e6)
        estimate_legacy = BE._estimate_discrete_map_largest_lyapunov(sys, p, x0, 50, 64, 1e-8, 1e6)
        @test estimate_core.exponent == estimate_legacy.exponent
        @test estimate_core.estimation_status isa Int
        @test estimate_legacy.estimation_status isa Symbol
        @test BE._map_lyapunov_estimation_status_symbol(estimate_core.estimation_status) == estimate_legacy.estimation_status
        @test estimate_core.estimation_status == BE._map_lyapunov_estimation_status_code(estimate_legacy.estimation_status)
        @test estimate_core.sample_count == estimate_legacy.sample_count

        # Status-code parity across every reachable discrete-map status (not just the one fixture
        # above): the CPU host wrapper (Symbol) and the GPU-kernel-callable core (Int) must classify
        # every seed identically once the code is mapped back to its Symbol.
        seeds_and_cutoffs = [
            ([1.35, 0.3], 1e6),        # ordinary periodic/aperiodic sweep — exercises :periodic
            ([1.0, 0.3], 1.5),         # tiny cutoff on the henon map's growing orbit — exercises :diverged
            ([NaN, 0.0], 1e6),         # NaN seed feeds straight into f — exercises :invalid_state
        ]
        for (params, cutoff) in seeds_and_cutoffs
            c = BE._detect_discrete_map_period_core(sys.f, params, x0, 5, 6, 1e-3, cutoff)
            l = BE._detect_discrete_map_period(sys, params, x0, 5, 6, 1e-3, cutoff)
            @test BE._map_status_symbol(c.status) == l.status
        end
        # max_period <= 0 exercises the :invalid_state early return.
        c0 = BE._detect_discrete_map_period_core(sys.f, [1.35, 0.3], x0, 5, 0, 1e-3, 1e6)
        l0 = BE._detect_discrete_map_period(sys, [1.35, 0.3], x0, 5, 0, 1e-3, 1e6)
        @test BE._map_status_symbol(c0.status) == l0.status == :invalid_state
    end

    @testset "result provenance defaults to :cpu and round-trips through JSON serialization" begin
        basins = basins_of_attraction(sys, BasinsConfig(bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
                                                        x_min=-0.4, x_max=0.4, x_steps=6, y_min=-0.4, y_max=0.4, y_steps=6,
                                                        max_period=6, iterations=100, precision=1e-3))
        @test basins.compute_backend == :cpu

        gpu_basins = basins_of_attraction(sys, BasinsConfig(bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
                                                            x_min=-0.4, x_max=0.4, x_steps=6, y_min=-0.4, y_max=0.4, y_steps=6,
                                                            max_period=6, iterations=100, precision=1e-3);
                                          backend=gpu_backend(:_ka_cpu_test))
        plain = BE._serialize_robust_chaos_basins(gpu_basins)
        @test plain["computeBackend"] == "_ka_cpu_test"
        restored = BE._deserialize_robust_chaos_basins(plain)
        @test restored.compute_backend == :_ka_cpu_test

        # Legacy payloads without a computeBackend key deserialize to the :cpu default.
        legacy_plain = BE._serialize_robust_chaos_basins(basins)
        delete!(legacy_plain, "computeBackend")
        legacy_restored = BE._deserialize_robust_chaos_basins(legacy_plain)
        @test legacy_restored.compute_backend == :cpu
    end

    @testset "computeBackend deserialization rejects unknown/malicious values without interning a Symbol" begin
        # Every value the allowlist should accept round-trips: :cpu, every known vendor, and the
        # private test seam (needed because it is a genuine recorded provenance value, see the
        # round-trip testset above).
        for name in ("cpu", "cuda", "amdgpu", "metal", "_ka_cpu_test")
            @test BE._compute_backend_symbol(name) == Symbol(name)
        end

        # Unknown/attacker-influenced strings must be rejected with a precise, catchable error rather
        # than silently interned as a new (permanent, never-garbage-collected) Symbol.
        for malicious in ("cpu; rm -rf /", "nonexistent_vendor", "", " ", "Cpu", "CUDA",
                          "eval(Meta.parse(\"1+1\"))", "a"^10_000)
            @test_throws ArgumentError BE._compute_backend_symbol(malicious)
        end

        legacy_plain = BE._serialize_robust_chaos_basins(
            basins_of_attraction(sys, BasinsConfig(bif_param=1.4, param_index=1, fixed_params=[1.4, 0.3],
                                                    x_min=-0.4, x_max=0.4, x_steps=6, y_min=-0.4, y_max=0.4, y_steps=6,
                                                    max_period=6, iterations=100, precision=1e-3)))
        malicious_plain = copy(legacy_plain)
        malicious_plain["computeBackend"] = "not_a_real_vendor"
        @test_throws ArgumentError BE._deserialize_robust_chaos_basins(malicious_plain)
    end
end
