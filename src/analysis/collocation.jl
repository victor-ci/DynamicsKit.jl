"""
Full-orbit periodic-orbit continuation by orthogonal collocation.

An alternative to the default Poincaré return-map shooting (`continuation_branch` on a
`ContinuousODE`): instead of continuing a fixed point of the return map, the whole
time-parameterized orbit and its period are continued as a boundary-value problem via
BifurcationKit's collocation discretization. Shooting stays the library default; this
path is offered for orbits where the return-map formulation is poorly conditioned and
for cross-validation of the shooting branches.

Orbit stability is reported through the return-map monodromy (the nontrivial Floquet
multipliers) computed by the existing variational machinery at a section crossing of the
collocation orbit, rather than BifurcationKit's collocation-Floquet eigenvalues, whose
largest-magnitude entries are dominated by spurious discretization modes.

Autonomous flows only: the periodic-orbit boundary-value problem has no explicit time,
so the vector field is evaluated with `t` frozen at 0. A nonautonomous right-hand side
is silently treated as its `t = 0` autonomous frame.
"""

# Locate a seed orbit: settle onto the attractor, then return a full-state point on the
# Poincaré section together with the flow period spanning `period` section returns.
function _collocation_orbit_seed(sys::ContinuousODE, base_params::Vector{Float64}, period::Int;
                                 initial_point, solver, reltol::Float64, abstol::Float64,
                                 settle_time::Float64)
    autonomous = (du, u, p, t) -> (sys.f(du, u, p, 0.0); nothing)
    u_start = _resolve_initial_state(sys, initial_point)

    if settle_time > 0
        settle = solve(ODEProblem(autonomous, u_start, (0.0, settle_time), base_params),
                       solver; reltol=reltol, abstol=abstol, save_everystep=false, save_start=false)
        SciMLBase.successful_retcode(settle.retcode) || error(
            "Collocation seed: settling integration for $(sys.name) failed with retcode $(settle.retcode).")
        u_start = settle.u[end]
    end

    # Integrate densely and collect section crossings in the configured direction,
    # widening the window until enough are found (re-integrating from the settled state
    # each time so crossing times stay on a single solution).
    direction = sys.section.direction
    g = u -> Float64(sys.section.condition(u, 0.0, nothing))
    window = max(4 * sys.tspan_hint, sys.tspan_hint) * (period + 2)
    for _ in 1:6
        sol = solve(ODEProblem(autonomous, u_start, (0.0, window), base_params),
                    solver; reltol=reltol, abstol=abstol)
        SciMLBase.successful_retcode(sol.retcode) || error(
            "Collocation seed: crossing integration for $(sys.name) failed with retcode $(sol.retcode).")
        grid = range(0.0, window; length=max(2000, 200 * (period + 2)))
        crossing_times = Float64[]
        prev = g(sol(first(grid)))
        for t in Iterators.drop(grid, 1)
            cur = g(sol(t))
            crossed = (direction >= 0 && prev < 0 && cur >= 0) ||
                      (direction <= 0 && prev > 0 && cur <= 0)
            crossed && push!(crossing_times, t)
            prev = cur
        end
        if length(crossing_times) >= period + 1
            u_orbit = collect(Float64, sol(crossing_times[1]))
            period_guess = crossing_times[period + 1] - crossing_times[1]
            return u_orbit, period_guess
        end
        window *= 2
    end
    error("Collocation seed: could not find $(period + 1) section crossing(s) for a period-$period orbit of $(sys.name); check the seed basin or increase settle_time.")
end

"""
    continuation_orbit_collocation(sys::ContinuousODE, config::CollocationConfig; kwargs...) -> OrbitBranchResult

Continue a family of periodic orbits of a continuous-time system by orthogonal
collocation. A seed orbit is located near the base parameter, discretized on the
collocation mesh, and continued in `config.continuation.param_index`.

Autonomous flows only: the vector field is evaluated with `t` frozen at 0 (the
periodic-orbit boundary-value problem has no explicit time).

Keyword arguments:
- `period`: Topological period (number of Poincaré-section returns per cycle).
- `params`: Base parameter vector; falls back to the system defaults.
- `initial_point`: Seed state for locating the orbit; falls back to the system default.
- `solver`, `reltol`, `abstol`: ODE integration controls for seeding.
"""
function continuation_orbit_collocation(sys::ContinuousODE, config::CollocationConfig;
                                        period::Int=1,
                                        params::Vector{Float64}=Float64[],
                                        initial_point::Union{Nothing, AbstractVector}=nothing,
                                        solver=Tsit5(),
                                        reltol::Float64=1e-9,
                                        abstol::Float64=1e-9)
    period >= 1 || throw(ArgumentError("continuation_orbit_collocation period must be >= 1, got $period."))
    cont = config.continuation
    base_params = _resolve_continuous_params(sys, params)
    p0 = base_params[cont.param_index]

    u_orbit, period_guess = _collocation_orbit_seed(
        sys, base_params, period;
        initial_point=initial_point, solver=solver, reltol=reltol, abstol=abstol,
        settle_time=config.settle_time)

    # Vector field carrying the continuation scalar through the parameter-injection mapping.
    # In-place (3-arg) for BifurcationKit so the many collocation RHS evaluations reuse the
    # solver's buffers instead of allocating a result vector each call; the 4-arg wrapper
    # feeds the seed ODE integrator. Parameter injection reuses one preallocated vector on
    # the Float64 path (the RHS is the discretization's hottest loop); a generic fallback
    # covers any Dual-typed parameter, mirroring the defining-system engine.
    write_indices = unique(vcat(cont.param_index, cont.linked_param_indices))
    params_buffer = copy(base_params)
    inject = pt -> if pt isa Float64
        @inbounds for idx in write_indices
            params_buffer[idx] = pt
        end
        params_buffer
    else
        _codim2_inject_any(base_params, cont.param_index, pt, cont.linked_param_indices)
    end
    field! = (du, u, pt) -> (sys.f(du, u, inject(pt.p), 0.0); nothing)
    ode_rhs! = (du, u, pt, t) -> field!(du, u, pt)

    lens = (@optic _.p)
    bifprob = BifurcationProblem(field!, u_orbit, (p = p0,), lens; inplace=true)
    seed_sol = solve(ODEProblem(ode_rhs!, u_orbit, (0.0, period_guess * config.seed_span_factor), (p = p0,)),
                     solver; reltol=reltol, abstol=abstol)

    coll0 = Collocation(config.ntst, config.m; N=sys.dim,
                        jacobian=BifurcationKit.AutoDiffDense(), meshadapt=config.mesh_adapt)
    # BifurcationKit/PreallocationTools emit benign mesh-adaptation and DiffCache-sizing
    # warnings while building and continuing the collocation problem; suppress them (a
    # failed solve still throws and is caught by the emptiness check below).
    coll, ci = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        generate_ci_problem(coll0, bifprob, seed_sol, period_guess; optimal_period=config.optimal_period)
    end

    opts = ContinuationPar(
        p_min=cont.p_min,
        p_max=cont.p_max,
        ds=cont.ds,
        dsmax=cont.dsmax,
        dsmin=cont.dsmin,
        max_steps=cont.max_steps,
        newton_options=NewtonPar(tol=cont.newton_tol, max_iterations=config.newton_max_iter),
        a=cont.a,
        detect_bifurcation=0,  # collocation-Floquet flags are unreliable; stability via return-map monodromy
        # Every orbit accessor reads branch.sol; pin the save cadence rather than relying
        # on BifurcationKit's default (its struct default is 1 but its docstring says 0).
        save_sol_every_step=1
    )
    branch = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        continuation(coll, ci, PALC(), opts; verbosity=0, bothside=config.bothside)
    end
    length(branch) > 0 || error(
        "Collocation continuation recorded no periodic-orbit points for the period-$period branch of $(sys.name) near parameter $p0.")

    return OrbitBranchResult(
        branch, coll, period,
        base_params, cont.param_index, collect(Int, cont.linked_param_indices),
        sys.name, sys.param_names[cont.param_index], :collocation, now()
    )
end

# --- Orbit-branch accessors ---

"Number of continued orbits stored on the branch."
_orbit_branch_count(result::OrbitBranchResult) = length(result.branch.sol)

"""
    orbit_branch_parameters(result::OrbitBranchResult) -> Vector{Float64}

Continuation-parameter value at each stored orbit, ordered along the branch arc.
"""
orbit_branch_parameters(result::OrbitBranchResult) = [Float64(s.p) for s in result.branch.sol]

"""
    orbit_branch_periods(result::OrbitBranchResult) -> Vector{Float64}

Flow period `T` of each stored orbit.
"""
orbit_branch_periods(result::OrbitBranchResult) =
    # BifurcationKit's getperiod/get_periodic_orbit for a Collocation problem read the period
    # from the solution vector and ignore the parameter argument, so `nothing` is passed here.
    [Float64(getperiod(result.coll, s.x, nothing)) for s in result.branch.sol]

# Materialize a decoded orbit's state samples as a dim × L matrix, robust to whether the
# BifurcationKit solution stores them as a (reshaped) matrix or a vector of state vectors.
_orbit_states_matrix(u::AbstractMatrix) = Matrix{Float64}(u)
_orbit_states_matrix(u::AbstractVector) = reduce(hcat, (collect(Float64, s) for s in u))

"""
    orbit_branch_orbit(result::OrbitBranchResult, i::Int) -> (t, states)

Decode the `i`-th stored orbit into its time grid `t` (length `L`) and a `dim × L`
matrix of full-state samples over one period.
"""
function orbit_branch_orbit(result::OrbitBranchResult, i::Int)
    sol = get_periodic_orbit(result.coll, result.branch.sol[i].x, nothing)
    return collect(Float64, sol.t), _orbit_states_matrix(sol.u)
end

"""
    orbit_branch_amplitude(result::OrbitBranchResult; state_index=1) -> Vector{Float64}

Peak-to-peak amplitude of one state coordinate over each stored orbit.
"""
function orbit_branch_amplitude(result::OrbitBranchResult; state_index::Int=1)
    amplitudes = Vector{Float64}(undef, _orbit_branch_count(result))
    for i in eachindex(result.branch.sol)
        _, states = orbit_branch_orbit(result, i)
        row = @view states[state_index, :]
        amplitudes[i] = maximum(row) - minimum(row)
    end
    return amplitudes
end

# Full state at a section crossing of a decoded orbit, linearly interpolated between the
# bracketing samples so the returned point lies on the section (g ≈ 0) rather than at the
# nearest mesh node.
function _orbit_section_point(sys::ContinuousODE, t::AbstractVector, states::AbstractMatrix)
    direction = sys.section.direction
    g = j -> Float64(sys.section.condition(@view(states[:, j]), t[j], nothing))
    npts = size(states, 2)
    prev = g(1)
    for j in 2:npts
        cur = g(j)
        crossed = (direction >= 0 && prev < 0 && cur >= 0) ||
                  (direction <= 0 && prev > 0 && cur <= 0)
        if crossed
            denom = cur - prev
            s = denom == 0 ? 0.0 : clamp(-prev / denom, 0.0, 1.0)
            return collect(Float64, @view(states[:, j - 1]) .+ s .* (@view(states[:, j]) .- @view(states[:, j - 1])))
        end
        prev = cur
    end
    # No sign change resolved on the sample grid: fall back to the sample closest to the
    # section. Multipliers computed from an off-section point degrade, so say so.
    best_j, best_g = 1, abs(g(1))
    for j in 2:npts
        gj = abs(g(j))
        gj < best_g && ((best_j, best_g) = (j, gj))
    end
    @warn "Decoded collocation orbit has no section crossing in the configured direction; " *
          "using the nearest sample (|g| = $(best_g)). Return-map multipliers computed from " *
          "this point may be inaccurate." maxlog=3
    return collect(Float64, states[:, best_j])
end

"""
    orbit_branch_multipliers(result::OrbitBranchResult, sys::ContinuousODE, i::Int; kwargs...) -> Vector{ComplexF64}

Nontrivial Floquet multipliers (return-map monodromy eigenvalues) of the `i`-th stored
orbit, computed from a section crossing of the collocation orbit with the same
variational machinery used by the shooting branches. Pass `ode_jacobian_method`,
`solver`, `reltol`, `abstol`, `fd_step`, `min_crossing_time` as for `branch_stability`.
"""
function orbit_branch_multipliers(result::OrbitBranchResult, sys::ContinuousODE, i::Int;
                                  ode_jacobian_method::Symbol=:variational, kwargs...)
    t, states = orbit_branch_orbit(result, i)
    section_state = _orbit_section_point(sys, t, states)
    projected = _project_section_state(sys.section, section_state)
    pv = _inject_param(result.base_params, result.param_index,
                       Float64(result.branch.sol[i].p), result.linked_param_indices)
    return _map_multipliers(sys, projected, pv, result.period;
                            ode_jacobian_method=ode_jacobian_method, kwargs...)
end

"""
    orbit_branch_stability(result::OrbitBranchResult, sys::ContinuousODE, i::Int; tol=1e-6, kwargs...) -> (Bool, Vector{ComplexF64})

Return whether the `i`-th orbit is stable (all nontrivial multipliers within the unit
circle to `tol`) together with those multipliers.
"""
function orbit_branch_stability(result::OrbitBranchResult, sys::ContinuousODE, i::Int;
                                tol::Float64=1e-6, kwargs...)
    multipliers = orbit_branch_multipliers(result, sys, i; kwargs...)
    return _multipliers_are_stable(multipliers; tol=tol), multipliers
end
