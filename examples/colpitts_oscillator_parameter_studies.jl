#!/usr/bin/env julia

using DynamicsKit
using DifferentialEquations
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots

const MODE = lowercase(get(ENV, "COLPITTS_STUDY_MODE", "smoke"))
const SMOKE = MODE != "final"
const OUTPUT_DIR = joinpath(@__DIR__, "..", "var", "output", "colpitts_oscillator_parameter_studies")

mkpath(OUTPUT_DIR)

const PROFILE = (
    map_steps = SMOKE ? 4 : 100,
    map_iterations = SMOKE ? 18 : 180,
    map_period = SMOKE ? 6 : 10,
    bf_steps = SMOKE ? 14 : 120,
    bf_iterations = SMOKE ? 45 : 180,
    bf_transient = SMOKE ? 25 : 120,
    phase_time = SMOKE ? 0.003 : 0.02,
    phase_crossings = SMOKE ? 8 : 80,
)

const EXPECTED_FILES = String[]

function save_named(plot_obj, filename)
    path = joinpath(OUTPUT_DIR, filename)
    savefig(plot_obj, path)
    push!(EXPECTED_FILES, path)
    println("    saved $(path)")
    return path
end

function param_index(sys, name::Symbol)
    idx = findfirst(==(name), sys.param_names)
    isnothing(idx) && error("Parameter $(name) is not available on $(sys.name).")
    return idx
end

function linked_indices(sys, names)
    Int[param_index(sys, Symbol(name)) for name in names]
end

function params_for(sys; pairs...)
    params = copy(sys.default_params)
    for (name, value) in pairs
        params[param_index(sys, Symbol(name))] = Float64(value)
    end
    return params
end

function save_bruteforce(sys, filename; param::Symbol, pmin, pmax, params, linked=Symbol[], title="", xlabel="")
    println("  brute force $(sys.name): $(param)")
    bf = brute_force_diagram(
        sys,
        BruteForceConfig(
            param_min=Float64(pmin),
            param_max=Float64(pmax),
            param_steps=PROFILE.bf_steps,
            iterations=PROFILE.bf_iterations,
            transient=min(PROFILE.bf_transient, PROFILE.bf_iterations - 1),
            param_index=param_index(sys, param),
            fixed_params=copy(params),
            linked_param_indices=linked_indices(sys, linked),
        );
        initial_point=copy(sys.default_initial_state),
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
    )
    p = plot_brute_force(bf)
    !isempty(xlabel) && xlabel!(p, xlabel)
    ylabel!(p, "V_C1 on Poincare section")
    !isempty(title) && title!(p, title)
    save_named(p, filename)
    return bf
end

function save_map(sys, filename; a::Symbol, b::Symbol, amin, amax, bmin, bmax, params,
                  a_linked=Symbol[], b_linked=Symbol[], title="", xlabel="", ylabel="", xscale=1.0, yscale=1.0)
    println("  map $(sys.name): $(a) vs $(b)")
    cfg = BifurcationMapConfig(
        a_min=Float64(amin),
        a_max=Float64(amax),
        a_steps=PROFILE.map_steps,
        b_min=Float64(bmin),
        b_max=Float64(bmax),
        b_steps=PROFILE.map_steps,
        a_index=param_index(sys, a),
        b_index=param_index(sys, b),
        a_linked_param_indices=linked_indices(sys, a_linked),
        b_linked_param_indices=linked_indices(sys, b_linked),
        max_period=PROFILE.map_period,
        precision=1e-3,
        iterations=PROFILE.map_iterations,
        base_params=copy(params),
    )
    result = bifurcation_map(
        sys,
        cfg;
        initial_point=copy(sys.default_initial_state),
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
    )
    p = plot_bifurcation_map(
        result;
        xscale=xscale,
        yscale=yscale,
        xlabel=isempty(xlabel) ? nothing : xlabel,
        ylabel=isempty(ylabel) ? nothing : ylabel,
        title=isempty(title) ? nothing : title,
    )
    save_named(p, filename)
    return result
end

function save_phase(sys, filename; params, title="")
    println("  phase portrait $(sys.name)")
    result = phase_portrait(
        sys,
        PhasePortraitConfig(
            time_stop=PROFILE.phase_time,
            tail_fraction=0.6,
            poincare_crossings=PROFILE.phase_crossings,
            min_crossing_time=1e-6,
        );
        params=copy(params),
        initial_point=copy(sys.default_initial_state),
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
        state_names=[Symbol("V_C1"), Symbol("V_C2"), Symbol("I_L")],
    )
    p = plot_phase_portrait(
        result;
        x_index=1,
        y_index=2,
        xlabel="V_C1",
        ylabel="V_C2",
        title=isempty(title) ? nothing : title,
    )
    save_named(p, filename)
    return result
end

function safe_branch(thunk, label)
    try
        return thunk()
    catch err
        @warn "Continuation branch failed; saving the remaining reproduction outputs." label error=err
        return nothing
    end
end

function branch_range(branch)
    values = [pt.param for pt in branch.branch.branch]
    isempty(values) && return (NaN, NaN, 0)
    return (minimum(values), maximum(values), length(values))
end

function save_branch_outputs(prefix, bf, branches; sys, params, linked=Symbol[], title="")
    kept = BranchResult[branch for branch in branches if !isnothing(branch)]
    branch_plot = plot_branches(
        kept;
        system=sys,
        params=copy(params),
        linked_param_indices=linked_indices(sys, linked),
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
        min_crossing_time=1e-6,
    )
    title!(branch_plot, isempty(title) ? "$(sys.name) continuation branches" : title)
    save_named(branch_plot, "$(prefix)_branches.png")

    overlay_plot = plot_overlay(
        bf,
        kept;
        system=sys,
        params=copy(params),
        linked_param_indices=linked_indices(sys, linked),
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
        min_crossing_time=1e-6,
    )
    title!(overlay_plot, isempty(title) ? "$(sys.name) continuation overlay" : "$(title) overlay")
    save_named(overlay_plot, "$(prefix)_overlay.png")
end

function simple_beta_branch(sys)
    beta_seed = 120.0
    params = params_for(sys; C1=40e-9, C2=40e-9, beta=beta_seed, V1=5.0, V2=5.0)
    seed_points = DynamicsKit._collect_poincare_points(
        sys,
        params;
        initial_point=copy(sys.default_initial_state),
        crossings=4,
        transient=120,
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
        projected=true,
    )
    isempty(seed_points) && error("No simple-model seed point was found.")
    seed = collect(first(seed_points))
    return continuation_branch(
        sys,
        ContinuationConfig(
            p_min=100.0,
            p_max=135.0,
            ds=0.5,
            dsmax=1.0,
            dsmin=1e-5,
            max_steps=SMOKE ? 120 : 140,
            newton_tol=1e-4,
            newton_max_iter=50,
            detect_bifurcation=1,
            param_index=param_index(sys, :beta),
        ),
        1;
        initial_point=seed,
        params=copy(params),
        search_min=[4.6, -0.95],
        search_max=[5.8, -0.4],
        n_initial=6,
        tol=1e-5,
        max_iter=80,
        fd_step=1e-6,
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
    )
end

function voltage_branches(sys; dynamic_beta::Bool)
    cases = SMOKE ?
        [(period=1, v0=1.5, pmin=1.25, pmax=1.78)] :
        [(period=1, v0=1.5, pmin=1.25, pmax=1.78),
         (period=2, v0=1.85, pmin=1.74, pmax=1.94),
         (period=4, v0=1.95, pmin=1.92, pmax=1.97)]
    primary = dynamic_beta ? :V1 : :V1
    linked = dynamic_beta ? [:V2] : [:V2]
    results = Union{Nothing, BranchResult}[]
    for case in cases
        params = dynamic_beta ?
            params_for(sys; C1=40e-9, C2=40e-9, V1=case.v0, V2=case.v0) :
            params_for(sys; C1=40e-9, C2=40e-9, beta=265.0, V1=case.v0, V2=case.v0)
        println("  continuation $(sys.name): period $(case.period), V=$(case.v0)")
        branch = safe_branch("$(sys.name) period $(case.period)") do
            continuation_branch(
                sys,
                ContinuationConfig(
                    p_min=case.pmin,
                    p_max=case.pmax,
                    ds=0.01,
                    dsmax=0.03,
                    dsmin=1e-5,
                    max_steps=SMOKE ? 140 : 220,
                    newton_tol=1e-6,
                    newton_max_iter=40,
                    detect_bifurcation=1,
                    param_index=param_index(sys, primary),
                    linked_param_indices=linked_indices(sys, linked),
                ),
                case.period;
                params=params,
                search_min=[2.0, -0.82],
                search_max=[2.8, -0.70],
                n_initial=8,
                tol=1e-5,
                max_iter=80,
                fd_step=1e-6,
                solver=Tsit5(),
                reltol=1e-7,
                abstol=1e-7,
            )
        end
        if !isnothing(branch)
            lo, hi, count = branch_range(branch)
            println("    branch points $(count), range [$(round(lo; digits=4)), $(round(hi; digits=4))]")
        end
        push!(results, branch)
    end
    return results
end

function run_simple_study()
    println("\n== Simple Colpitts model ==")
    sys = colpitts_simple_oscillator()
    base = params_for(sys; C1=40e-9, C2=40e-9, beta=265.0, V1=5.0, V2=5.0)

    beta_bf = save_bruteforce(sys, "simple_beta_diagram.png";
        param=:beta, pmin=100.0, pmax=300.0, params=base,
        title="Simple Colpitts beta sweep", xlabel="beta")
    save_bruteforce(sys, "simple_c_equal_diagram.png";
        param=:C1, pmin=30e-9, pmax=60e-9, params=base, linked=[:C2],
        title="Simple Colpitts C1=C2 sweep", xlabel="C1=C2 (F)")
    save_map(sys, "simple_c1_c2_map.png";
        a=:C1, b=:C2, amin=30e-9, amax=60e-9, bmin=30e-9, bmax=60e-9, params=base,
        title="Simple Colpitts C1 vs C2 map", xlabel="C1 (nF)", ylabel="C2 (nF)", xscale=1e9, yscale=1e9)
    save_map(sys, "simple_c_equal_beta_map.png";
        a=:C1, b=:beta, amin=30e-9, amax=60e-9, bmin=100.0, bmax=300.0, params=base,
        a_linked=[:C2], title="Simple Colpitts C1=C2 vs beta map",
        xlabel="C1=C2 (nF)", ylabel="beta", xscale=1e9)

    branch = safe_branch("simple beta") do
        simple_beta_branch(sys)
    end
    save_branch_outputs("simple_beta", beta_bf, [branch]; sys=sys, params=params_for(sys; C1=40e-9, C2=40e-9, beta=120.0, V1=5.0, V2=5.0), title="Simple Colpitts beta branch")
    save_phase(sys, "simple_phase_beta120.png"; params=params_for(sys; C1=40e-9, C2=40e-9, beta=120.0, V1=5.0, V2=5.0), title="Simple Colpitts phase portrait beta=120")
    save_phase(sys, "simple_phase_beta265.png"; params=base, title="Simple Colpitts phase portrait beta=265")
end

function run_exponential_study()
    println("\n== Exponential Colpitts model ==")
    sys = colpitts_exponential_oscillator()
    base = params_for(sys; C1=40e-9, C2=40e-9, beta=265.0, V1=5.0, V2=5.0)
    voltage_base = params_for(sys; C1=40e-9, C2=40e-9, beta=265.0, V1=1.95, V2=1.95)

    voltage_bf = save_bruteforce(sys, "exponential_v_equal_diagram.png";
        param=:V1, pmin=1.25, pmax=1.97, params=voltage_base, linked=[:V2],
        title="Exponential Colpitts V1=V2 sweep", xlabel="V1=V2 (V)")
    save_bruteforce(sys, "exponential_c_equal_diagram.png";
        param=:C1, pmin=30e-9, pmax=60e-9, params=base, linked=[:C2],
        title="Exponential Colpitts C1=C2 sweep", xlabel="C1=C2 (F)")
    save_bruteforce(sys, "exponential_beta_diagram.png";
        param=:beta, pmin=100.0, pmax=300.0, params=base,
        title="Exponential Colpitts beta sweep", xlabel="beta")
    save_map(sys, "exponential_c1_c2_map.png";
        a=:C1, b=:C2, amin=30e-9, amax=60e-9, bmin=30e-9, bmax=60e-9, params=base,
        title="Exponential Colpitts C1 vs C2 map", xlabel="C1 (nF)", ylabel="C2 (nF)", xscale=1e9, yscale=1e9)
    save_map(sys, "exponential_c_equal_beta_map.png";
        a=:C1, b=:beta, amin=30e-9, amax=60e-9, bmin=100.0, bmax=300.0, params=base,
        a_linked=[:C2], title="Exponential Colpitts C1=C2 vs beta map",
        xlabel="C1=C2 (nF)", ylabel="beta", xscale=1e9)

    branches = voltage_branches(sys; dynamic_beta=false)
    save_branch_outputs("exponential_voltage", voltage_bf, branches; sys=sys, params=voltage_base, linked=[:V2], title="Exponential Colpitts voltage cascade")
    save_phase(sys, "exponential_phase_v150.png"; params=params_for(sys; C1=40e-9, C2=40e-9, beta=265.0, V1=1.5, V2=1.5), title="Exponential Colpitts phase portrait V1=V2=1.5")
    save_phase(sys, "exponential_phase_v195.png"; params=voltage_base, title="Exponential Colpitts phase portrait V1=V2=1.95")
end

function run_dynamic_beta_study()
    println("\n== Dynamic-beta Colpitts model ==")
    sys = colpitts_dynamic_beta_oscillator()
    base = params_for(sys; C1=40e-9, C2=40e-9, V1=5.0, V2=5.0)
    voltage_base = params_for(sys; C1=40e-9, C2=40e-9, V1=1.95, V2=1.95)

    voltage_bf = save_bruteforce(sys, "dynamic_v_equal_diagram.png";
        param=:V1, pmin=1.25, pmax=1.97, params=voltage_base, linked=[:V2],
        title="Dynamic-beta Colpitts V1=V2 sweep", xlabel="V1=V2 (V)")
    save_bruteforce(sys, "dynamic_c_equal_diagram.png";
        param=:C1, pmin=30e-9, pmax=60e-9, params=base, linked=[:C2],
        title="Dynamic-beta Colpitts C1=C2 sweep", xlabel="C1=C2 (F)")
    save_map(sys, "dynamic_c1_c2_map.png";
        a=:C1, b=:C2, amin=30e-9, amax=60e-9, bmin=30e-9, bmax=60e-9, params=base,
        title="Dynamic-beta Colpitts C1 vs C2 map", xlabel="C1 (nF)", ylabel="C2 (nF)", xscale=1e9, yscale=1e9)

    branches = voltage_branches(sys; dynamic_beta=true)
    save_branch_outputs("dynamic_voltage", voltage_bf, branches; sys=sys, params=voltage_base, linked=[:V2], title="Dynamic-beta Colpitts voltage cascade")
    save_phase(sys, "dynamic_phase_v150.png"; params=params_for(sys; C1=40e-9, C2=40e-9, V1=1.5, V2=1.5), title="Dynamic-beta Colpitts phase portrait V1=V2=1.5")
    save_phase(sys, "dynamic_phase_v185.png"; params=params_for(sys; C1=40e-9, C2=40e-9, V1=1.85, V2=1.85), title="Dynamic-beta Colpitts phase portrait V1=V2=1.85")
    save_phase(sys, "dynamic_phase_v195.png"; params=voltage_base, title="Dynamic-beta Colpitts phase portrait V1=V2=1.95")
end

println("Colpitts parameter-study mode: $(MODE)")
println("Output directory: $(OUTPUT_DIR)")
println("Julia threads: $(Threads.nthreads())")

run_simple_study()
run_exponential_study()
run_dynamic_beta_study()

missing = filter(path -> !isfile(path), EXPECTED_FILES)
if !isempty(missing)
    error("Missing expected parameter-study outputs: $(join(missing, ", "))")
end

println("\nDone. Generated $(length(EXPECTED_FILES)) plot files.")
