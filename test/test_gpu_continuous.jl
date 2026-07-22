# Continuous-time GPU Poincaré-map sweeps (`bifurcation_map` / `basins_of_attraction` for
# `sys::ContinuousODE`) via DiffEqGPU's `EnsembleGPUKernel`. As with the discrete GPU tests, no real
# vendor extension ships, so the GPU *kernel* path is exercised through the private CPU test seam
# `gpu_backend(:_ka_cpu_test)` → `EnsembleGPUKernel(ka_cpu)` (the exact DiffEqGPU
# CPU seam, not a mock). Parity is asserted against the existing CPU adaptive Poincaré-return method on
# analytic/representative fixtures with documented scientific tolerances.

using StaticArrays

@testset "Continuous GPU Poincaré-map sweeps" begin
    BE = DynamicsKit
    seam = gpu_backend(:_ka_cpu_test)
    ka_cpu = BE._dynamicskit_gpu_backend(Val(:_ka_cpu_test))

    @testset "GPU out-of-place RHS accessor" begin
        for ctor in (rossler_oscillator, vilnius_oscillator, memristive_diode_bridge,
                     colpitts_simple_oscillator, colpitts_exponential_oscillator,
                     colpitts_dynamic_beta_oscillator)
            @test BE.has_continuous_gpu_rhs(ctor())
        end
        # A ContinuousODE built without `f_svector` (as imported/user systems are) is GPU-ineligible.
        bare = ContinuousODE((du, u, p, t) -> (du[1] = u[2]; du[2] = -u[1]), 2,
                             PoincareSection((u, t, i) -> u[1]; direction=:up, projection=[2], template=[0.0, 0.0]),
                             [:a, :b], "bare-imported"; default_initial_state=[1.0, 0.0], default_params=[0.0, 0.0])
        @test !BE.has_continuous_gpu_rhs(bare)
        @test BE.continuous_gpu_rhs(bare) === nothing
    end

    @testset "out-of-place RHS matches the in-place RHS exactly" begin
        cases = [
            (rossler_oscillator(), [0.2, 0.2, 5.7], [1.0, 1.0, 1.0]),
            (vilnius_oscillator(), [0.25, 30.0, 0.2], [0.3, 0.1, 0.2]),
            (memristive_diode_bridge(), [0.005, 6.02e-6, 0.05], [0.1, 0.01, 0.2]),
            (colpitts_simple_oscillator(), [40e-9, 40e-9, 265.0, 5.0, 5.0], [0.3, -0.4, 0.01]),
            (colpitts_exponential_oscillator(), [40e-9, 40e-9, 265.0, 5.0, 5.0], [0.3, -0.4, 0.01]),
            (colpitts_dynamic_beta_oscillator(), [40e-9, 40e-9, 5.0, 5.0], [0.3, -0.4, 0.01]),
        ]
        for (sys, p, u) in cases
            du = zeros(3); sys.f(du, u, p, 0.0)
            duo = BE.continuous_gpu_rhs(sys)(SVector{3}(u), p, 0.0)
            @test collect(duo) == du
        end
    end

    sys = rossler_oscillator()
    # Robust period-doubling cascade (c in [2.3, 4.1], pre-chaos): classifications are numerically
    # stable, so GPU/CPU parity is exact. (In deep chaos, near-threshold closures are sensitive to
    # integrator-path differences between CPU Tsit5 and GPU Tsit5 — see the wide-grid test below.)
    cascade = BifurcationMapConfig(a_min=0.15, a_max=0.25, a_steps=3, b_min=2.3, b_max=4.1, b_steps=5,
                                   a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
                                   max_period=8, iterations=200, precision=1e-3)

    @testset "bifurcation_map: exact CPU/GPU parity on the cascade + provenance" begin
        r_cpu, diag_cpu = BE._bifurcation_map(sys, cascade)
        r_gpu, diag_gpu = BE._bifurcation_map(sys, cascade; backend=seam)
        @test r_cpu.periodicity == r_gpu.periodicity
        @test r_cpu.compute_backend == :cpu
        @test r_gpu.compute_backend == :_ka_cpu_test
        @test diag_cpu["computeBackend"] == "cpu"
        @test diag_gpu["computeBackend"] == "_ka_cpu_test"
        @test diag_gpu["gpuSolver"] == "GPUTsit5"
        @test diag_gpu["gpuAdaptive"] == true
        @test diag_gpu["gpuRootfindAbstol"] == BE._CONTINUOUS_GPU_ROOTFIND_ABSTOL
        @test !haskey(diag_cpu, "gpuSolver")
        # Public wrapper carries the provenance too.
        @test bifurcation_map(sys, cascade; backend=seam).compute_backend == :_ka_cpu_test
    end

    @testset "threaded fixed-seed CPU sweep is deterministic across repeats (regression: shared param_buffer race)" begin
        # `_bifurcation_map(::ContinuousODE, ...)` fixed-seed mode dispatches per-chunk work across
        # `Threads.@threads` tasks; a previous defect let two sibling branches (this CPU path and the
        # GPU-host upload path) share a lexically-named parameter buffer, which Julia boxed into a
        # single `Core.Box` shared by every worker task, so tasks intermittently integrated cells with
        # another task's (a, b) parameters. Confirmed on this exact grid: the pre-fix code mismatched
        # its own first-run reference on 20/20 repeats at 4 threads; run enough repeats here to make a
        # reintroduced race reliably fail rather than pass by chance.
        if Threads.nthreads() >= 4
            reference, _ = BE._bifurcation_map(sys, cascade)
            n_repeats = 20
            for r in 1:n_repeats
                repeated, _ = BE._bifurcation_map(sys, cascade)
                @test repeated.periodicity == reference.periodicity
            end
        else
            @info "Skipping threaded fixed-seed determinism regression (needs JULIA_NUM_THREADS >= 4, got $(Threads.nthreads()))."
        end
    end

    @testset "no shared Core.Box for the CPU fixed-seed param_buffer (supplementary implementation check)" begin
        # Structural check only, not the behavioral proof above: confirms the fixed-seed per-chunk
        # parameter buffer is a plain function local (in `_process_continuous_map_fixed_chunk!`), not a
        # variable boxed once per call to `_bifurcation_map` and shared by every `Threads.@threads` task.
        stub_ct = code_typed(BE._bifurcation_map, (typeof(sys), typeof(cascade)); optimize=false)
        io = IOBuffer(); show(io, stub_ct[1][1]); stub_src = String(take!(io))
        name_match = match(r"var\"(#_bifurcation_map#\d+)\"", stub_src)
        if name_match === nothing
            @test_skip "could not locate the kwarg-body method name for lowered-code introspection"
        else
            inner_f = getfield(BE, Symbol(name_match.captures[1]))
            argtypes = (Nothing, typeof(BE.Tsit5()), Float64, Float64, Nothing, CPUBackend,
                        typeof(BE._bifurcation_map), typeof(sys), typeof(cascade))
            inner_ct = code_typed(inner_f, argtypes; optimize=false)
            io2 = IOBuffer(); show(io2, inner_ct[1][1]); inner_src = String(take!(io2))
            boxed_param_buffer = any(split(inner_src, "\n")) do line
                occursin(r"(?<![a-zA-Z_])param_buffer(?![a-zA-Z_])", line) && occursin("Box", line)
            end
            @test !boxed_param_buffer
        end
    end

    @testset "auto backend never errors and matches the CPU result" begin
        r_cpu = bifurcation_map(sys, cascade)
        r_auto = bifurcation_map(sys, cascade; backend=auto_backend())
        @test r_auto.compute_backend == :cpu
        @test r_auto.periodicity == r_cpu.periodicity
    end

    @testset "basins: CPU/GPU parity + provenance" begin
        bcfg = BasinsConfig(bif_param=5.0, param_index=3, fixed_params=[0.2, 0.2, 5.0],
                            x_min=-6.0, x_max=6.0, x_steps=6, y_min=0.0, y_max=6.0, y_steps=6,
                            x_index=1, y_index=3, max_period=6, iterations=120, precision=1e-3)
        b_cpu = basins_of_attraction(sys, bcfg)
        b_gpu = basins_of_attraction(sys, bcfg; backend=seam)
        @test b_cpu.periodicity == b_gpu.periodicity
        @test b_cpu.compute_backend == :cpu
        @test b_gpu.compute_backend == :_ka_cpu_test
        @test basins_of_attraction(sys, bcfg; backend=auto_backend()).compute_backend == :cpu
    end

    @testset "low-level ensemble sweep matches the CPU detector (period + final_point)" begin
        # Directly compare `_continuous_poincare_gpu_sweep` with `_detect_continuous_poincare_period`
        # across a set of (u0, p) seeds spanning period-1/2/4 and chaos.
        u0 = [1.0, 1.0, 1.0]
        plist = [[0.2, 0.2, c] for c in (2.5, 3.5, 4.0, 5.7)]
        u0list = [copy(u0) for _ in plist]
        warmed = BE._continuous_gpu_warmup_states(sys, u0list, plist;
            solver=BE.Tsit5(), reltol=1e-9, abstol=1e-9, min_crossing_time=1e-6)
        gpu = BE._continuous_poincare_gpu_sweep(sys, warmed, plist, ka_cpu;
            transient=200, max_period=8, precision=1e-3, reltol=1e-9, abstol=1e-9,
            min_crossing_time=1e-6, divergence_cutoff=Inf)
        for (k, p) in enumerate(plist)
            cpu = BE._detect_continuous_poincare_period(sys, p; initial_point=u0,
                transient=200, max_period=8, precision=1e-3, reltol=1e-9, abstol=1e-9,
                projected=true, min_crossing_time=1e-6, return_crossing_diagnostics=false)
            @test gpu[k].period == cpu.period
            @test gpu[k].status == cpu.status
        end
    end

    @testset "_continuous_poincare_gpu_sweep fails fast on empty/mismatched inputs" begin
        u0 = [1.0, 1.0, 1.0]
        p = [0.2, 0.2, 3.5]
        sweep_kwargs = (transient=20, max_period=8, precision=1e-3, reltol=1e-9, abstol=1e-9,
                       min_crossing_time=1e-6, divergence_cutoff=Inf)

        # Empty u0_list/p_list must be rejected before `first(p_list)` is ever reached.
        @test_throws ArgumentError BE._continuous_poincare_gpu_sweep(sys, Vector{Float64}[], Vector{Float64}[], ka_cpu; sweep_kwargs...)

        # Mismatched u0_list/p_list lengths.
        @test_throws ArgumentError BE._continuous_poincare_gpu_sweep(sys, [copy(u0)], Vector{Float64}[], ka_cpu; sweep_kwargs...)
        @test_throws ArgumentError BE._continuous_poincare_gpu_sweep(sys, Vector{Float64}[], [copy(p)], ka_cpu; sweep_kwargs...)

        # Non-uniform parameter-vector lengths within p_list.
        @test_throws ArgumentError BE._continuous_poincare_gpu_sweep(
            sys, [copy(u0), copy(u0)], [copy(p), [0.2, 0.2]], ka_cpu; sweep_kwargs...)

        # Non-uniform (or wrong-dimension) initial-state vectors within u0_list.
        @test_throws ArgumentError BE._continuous_poincare_gpu_sweep(
            sys, [copy(u0), [1.0, 1.0]], [copy(p), copy(p)], ka_cpu; sweep_kwargs...)

        # A valid multi-trajectory call still runs cleanly (not over-rejected by the new checks). Single
        # -trajectory (`n == 1`) ensembles are a separate, pre-existing DiffEqGPU/EnsembleGPUKernel
        # limitation unrelated to this validation and are not asserted on here.
        result = BE._continuous_poincare_gpu_sweep(sys, [copy(u0), copy(u0)], [copy(p), copy(p)], ka_cpu; sweep_kwargs...)
        @test length(result) == 2
    end

    @testset "finite divergence_cutoff: excursion between crossings (not at one) is still caught" begin
        # Regression fixture for the GPU/CPU cutoff-detection gap: with divergence_cutoff=Inf this
        # Rössler trajectory is a perfectly ordinary bounded chaotic orbit (aperiodic within
        # max_period=8, never flagged as diverged) — its Poincaré-crossing-sampled states never exceed
        # 18.0 either. But *between* crossings (dense sampling shows t≈24.34–24.61) the flow briefly
        # swings up to |state|≈18.5 before returning to the bounded regime. A detector that only samples
        # state at crossings (the pre-fix GPU path) never sees this and reports :aperiodic_or_high_period
        # exactly like the cutoff=Inf case; the CPU path's per-step `_map_state_termination_callback`
        # catches it immediately and reports :diverged. The GPU path must match CPU here.
        divergent_u0 = [1.0, 1.0, 1.0]
        divergent_p = [0.2, 0.2, 5.7]
        divergent_kwargs = (transient=20, max_period=8, precision=1e-3, reltol=1e-9, abstol=1e-9,
                            min_crossing_time=1e-6)

        cpu_unbounded = BE._detect_continuous_poincare_period(sys, divergent_p; initial_point=divergent_u0,
            projected=true, divergence_cutoff=Inf, return_crossing_diagnostics=false, divergent_kwargs...)
        # Control: with no cutoff, the orbit is unremarkable — it is a genuine transient excursion that
        # later returns, not a runaway trajectory that would diverge regardless of the cutoff value.
        @test cpu_unbounded.status == :aperiodic_or_high_period

        cutoff = 18.0
        cpu = BE._detect_continuous_poincare_period(sys, divergent_p; initial_point=divergent_u0,
            projected=true, divergence_cutoff=cutoff, return_crossing_diagnostics=false, divergent_kwargs...)
        @test cpu.status == :diverged
        @test cpu.state_callback_activated
        @test cpu.divergence_callback_activated
        # The excursion is genuinely off-section: every crossing-sampled coordinate the closure
        # detector records stays within the cutoff, only the between-crossing flow exceeds it.
        @test maximum(abs, cpu.final_point) > cutoff

        warmed = BE._continuous_gpu_warmup_states(sys, [divergent_u0, divergent_u0], [divergent_p, [0.2, 0.2, 4.0]];
            solver=BE.Tsit5(), reltol=1e-9, abstol=1e-9, min_crossing_time=1e-6)
        gpu = BE._continuous_poincare_gpu_sweep(sys, warmed, [divergent_p, [0.2, 0.2, 4.0]], ka_cpu;
            divergence_cutoff=cutoff, divergent_kwargs...)
        @test gpu[1].status == :diverged
        @test gpu[1].status == cpu.status

        # End-to-end through the public sweep too (the path an application actually calls). A second
        # b value pads the sweep past the single-trajectory ensemble edge case and is asserted for
        # CPU/GPU consistency alongside the divergent cell rather than dropped.
        cfg = BifurcationMapConfig(a_min=0.2, a_max=0.2, a_steps=0, b_min=5.7, b_max=5.8, b_steps=1,
                                   a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
                                   max_period=8, iterations=29, precision=1e-3, divergence_cutoff=cutoff)
        r_cpu, diag_cpu = BE._bifurcation_map(sys, cfg)
        r_gpu, diag_gpu = BE._bifurcation_map(sys, cfg; backend=seam)
        @test r_cpu.periodicity == r_gpu.periodicity
        @test diag_cpu["status"]["statusCounts"] == diag_gpu["status"]["statusCounts"]
        @test get(diag_cpu["status"]["statusCounts"], "diverged", 0) >= 1
    end

    @testset "callback direction is honored (:down section parity vs CPU)" begin
        # Same Rössler flow, but detect DOWN-crossings of y = 0. The down-crossing return map differs
        # from the up-crossing one, so matching the CPU :down detector proves the GPU affect_neg!
        # wiring (not a coincidental symmetry).
        f_oop = BE.continuous_gpu_rhs(sys)
        down = ContinuousODE(sys.f, 3,
            PoincareSection((u, t, i) -> u[2]; direction=:down, projection=[1, 3], template=[0.0, 0.0, 0.0]),
            [:a, :b, :c], "Rössler-down"; tspan_hint=6.5, default_initial_state=[1.0, 1.0, 1.0],
            default_params=[0.2, 0.2, 5.0], f_svector=f_oop)
        plist = [[0.2, 0.2, c] for c in (2.5, 3.5, 4.0)]
        u0list = [[1.0, 1.0, 1.0] for _ in plist]
        warmed = BE._continuous_gpu_warmup_states(down, u0list, plist;
            solver=BE.Tsit5(), reltol=1e-9, abstol=1e-9, min_crossing_time=1e-6)
        gpu = BE._continuous_poincare_gpu_sweep(down, warmed, plist, ka_cpu;
            transient=200, max_period=8, precision=1e-3, reltol=1e-9, abstol=1e-9,
            min_crossing_time=1e-6, divergence_cutoff=Inf)
        for (k, p) in enumerate(plist)
            cpu = BE._detect_continuous_poincare_period(down, p; initial_point=[1.0, 1.0, 1.0],
                transient=200, max_period=8, precision=1e-3, reltol=1e-9, abstol=1e-9,
                projected=true, min_crossing_time=1e-6, return_crossing_diagnostics=false)
            @test gpu[k].period == cpu.period
        end
    end

    @testset "explicit GPU on ineligible continuous configs errors; auto falls back" begin
        base = (a_min=0.15, a_max=0.25, a_steps=2, b_min=2.3, b_max=4.1, b_steps=2,
                a_index=1, b_index=3, max_period=6, iterations=140, precision=1e-3)

        lyap = BifurcationMapConfig(; base..., base_params=[0.2, 0.2, 3.0],
                                    lyapunov_enabled=true, lyapunov_iterations=40)
        @test_throws ArgumentError bifurcation_map(sys, lyap; backend=seam)
        @test bifurcation_map(sys, lyap; backend=auto_backend()).compute_backend == :cpu

        neigh = BifurcationMapConfig(; base..., base_params=[0.2, 0.2, 3.0], reuse_neighbor_seeds=true)
        @test_throws ArgumentError bifurcation_map(sys, neigh; backend=seam)
        @test bifurcation_map(sys, neigh; backend=auto_backend()).compute_backend == :cpu

        multi = BifurcationMapConfig(; base..., base_params=[0.2, 0.2, 3.0],
                                     multistability_initial_points=[[1.0, 1.0, 1.0]])
        @test_throws ArgumentError bifurcation_map(sys, multi; backend=seam)

        linked = BifurcationMapConfig(; base..., base_params=[0.2, 0.2, 3.0], a_linked_param_indices=[2])
        @test_throws ArgumentError bifurcation_map(sys, linked; backend=seam)
    end

    @testset "precision tighter than the section-localization floor is rejected" begin
        tight = BifurcationMapConfig(a_min=0.15, a_max=0.25, a_steps=2, b_min=2.3, b_max=4.1, b_steps=2,
                                     a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
                                     max_period=6, iterations=140, precision=1e-7)
        @test !BE._continuous_gpu_precision_ok(1e-7)
        @test BE._continuous_gpu_precision_ok(1e-4)
        @test_throws ArgumentError bifurcation_map(sys, tight; backend=seam)
        try
            bifurcation_map(sys, tight; backend=seam)
        catch e
            @test occursin("precision", lowercase(e.msg)) || occursin("localization", lowercase(e.msg))
        end
        # Auto falls back rather than erroring.
        @test bifurcation_map(sys, tight; backend=auto_backend()).compute_backend == :cpu
    end

    @testset "system without a GPU RHS is rejected (explicit) / falls back (auto)" begin
        bare = ContinuousODE((du, u, p, t) -> (du[1] = u[2]; du[2] = -u[1]), 2,
                             PoincareSection((u, t, i) -> u[2]; direction=:up, projection=[1], template=[0.0, 0.0]),
                             [:a, :b], "bare-imported"; tspan_hint=10.0,
                             default_initial_state=[1.0, 0.0], default_params=[0.0, 0.0])
        cfg = BifurcationMapConfig(a_min=0.0, a_max=0.0, a_steps=0, b_min=0.0, b_max=0.0, b_steps=0,
                                   a_index=1, b_index=2, max_period=4, precision=1e-3, iterations=8, base_params=[0.0, 0.0])
        @test_throws ArgumentError bifurcation_map(bare, cfg; backend=seam)
        @test bifurcation_map(bare, cfg; backend=auto_backend()) isa BifurcationMapResult
    end

    @testset "continuous Lyapunov field GPU request is rejected with its coupled-trajectory reason" begin
        lcfg = BifurcationMapConfig(a_min=0.15, a_max=0.25, a_steps=2, b_min=2.3, b_max=4.1, b_steps=2,
                                    a_index=1, b_index=3, base_params=[0.2, 0.2, 3.0],
                                    max_period=6, iterations=140, precision=1e-3,
                                    lyapunov_enabled=true, lyapunov_iterations=40)
        @test_throws ArgumentError lyapunov_field(sys, lcfg; backend=seam)
        try
            lyapunov_field(sys, lcfg; backend=seam)
        catch e
            @test occursin("coupled", lowercase(e.msg))
        end
        # Auto/CPU run on the CPU without error.
        @test lyapunov_field(sys, lcfg; backend=auto_backend()) isa LyapunovFieldResult
    end

    @testset "cells= cache hook honored under the continuous GPU backend" begin
        na, nb = cascade.a_steps + 1, cascade.b_steps + 1
        rf, _ = BE._bifurcation_map(sys, cascade; backend=seam)
        g1 = MapCellGrid(na, nb)
        r1, _ = BE._bifurcation_map(sys, cascade; cells=g1, backend=seam)
        @test r1.periodicity == rf.periodicity
        @test all(g1.known)
        # Pre-seed half the cells as known; the GPU sweep must skip them and reproduce the same grid.
        g2 = MapCellGrid(na, nb)
        for i in 1:na, j in 1:nb
            if (i + j) % 2 == 0
                g2.periodicity[i, j]              = g1.periodicity[i, j]
                g2.status_codes[i, j]             = g1.status_codes[i, j]
                g2.closure_errors[i, j]           = g1.closure_errors[i, j]
                g2.closure_candidate_periods[i, j] = g1.closure_candidate_periods[i, j]
                g2.observed_points[i, j]          = g1.observed_points[i, j]
                g2.closure_confidence[i, j]        = g1.closure_confidence[i, j]
                g2.known[i, j] = true
            end
        end
        r2, _ = BE._bifurcation_map(sys, cascade; cells=g2, backend=seam)
        @test r2.periodicity == rf.periodicity
        @test all(g2.known)

        bcfg = BasinsConfig(bif_param=5.0, param_index=3, fixed_params=[0.2, 0.2, 5.0],
                            x_min=-6.0, x_max=6.0, x_steps=5, y_min=0.0, y_max=6.0, y_steps=5,
                            x_index=1, y_index=3, max_period=6, iterations=120, precision=1e-3)
        bf = basins_of_attraction(sys, bcfg; backend=seam)
        nx, ny = bcfg.x_steps + 1, bcfg.y_steps + 1
        bg1 = BasinsCellGrid(nx, ny)
        br1 = basins_of_attraction(sys, bcfg; cells=bg1, backend=seam)
        @test br1.periodicity == bf.periodicity
        @test all(bg1.known)
    end

    @testset "device-gated real-GPU FP64 parity ($(vendor))" for (vendor, pkg) in ((:cuda, :CUDA), (:amdgpu, :AMDGPU))
        # Load the vendor package only if it is installed; skip cleanly when absent (no hardware here).
        # Only a "package not found" `ArgumentError` is treated as absence — any other failure (e.g. the
        # package is installed but its `DynamicsKit*Ext` extension errors while loading) must not be
        # silently swallowed as a skip; it is a real defect and should fail this test loudly.
        loaded = try
            @eval import $pkg
            true
        catch e
            if e isa ArgumentError && occursin("not found in current path", e.msg)
                false
            else
                rethrow()
            end
        end
        if !loaded
            @info "Skipping device-gated $(vendor) parity test ($(pkg) is not installed in this environment)."
            @test_skip "vendor package $(pkg) not installed"
        elseif !gpu_backend_available(vendor)
            # The package imported (so its extension load did not error) but reports no functional
            # device — a legitimate, distinct "device unavailable" skip, not a package-absence skip.
            @test Base.get_extension(DynamicsKit, Symbol("DynamicsKit$(pkg)Ext")) !== nothing
            @info "Skipping device-gated $(vendor) parity test ($(pkg) loaded but reports no functional device)."
            @test_skip "vendor $(vendor) device not available"
        else
            r_cpu = bifurcation_map(sys, cascade)
            r_gpu = bifurcation_map(sys, cascade; backend=gpu_backend(vendor))
            @test r_gpu.compute_backend == vendor
            @test r_gpu.periodicity == r_cpu.periodicity           # FP64 scientific parity on real hardware
            b_gpu = basins_of_attraction(sys,
                BasinsConfig(bif_param=5.0, param_index=3, fixed_params=[0.2, 0.2, 5.0],
                             x_min=-6.0, x_max=6.0, x_steps=6, y_min=0.0, y_max=6.0, y_steps=6,
                             x_index=1, y_index=3, max_period=6, iterations=120, precision=1e-3);
                backend=gpu_backend(vendor))
            @test b_gpu.compute_backend == vendor
        end
    end
end
