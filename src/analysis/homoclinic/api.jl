# Public connecting-orbit API: labels, orbit accessor, and the homoclinic /
# heteroclinic / saddle-cycle entry points.

const _HOMOCLINIC_EVENT_LABELS = Dict{Symbol, String}(
    :nns => "Neutral saddle",
    :nsf => "Neutral saddle-focus",
    :nff => "Neutral focus-focus",
    :drs => "Double real stable eigenvalue",
    :dru => "Double real unstable eigenvalue",
    :nds => "Neutral double stable",
    :ndu => "Neutral double unstable",
    :tls => "Triple leading stable eigenvalue",
    :tlu => "Triple leading unstable eigenvalue",
    :nch => "Neutral center-homoclinic",
    :sh => "Shilnikov condition",
    :bt => "Bogdanov-Takens point",
    :ofu => "Unstable orbit flip",
    :ofs => "Stable orbit flip",
    :ifu => "Unstable inclination flip",
    :ifs => "Stable inclination flip",
)

"""
    homoclinic_special_point_label(kind) -> String

Human-readable label for a standard connecting-orbit test-function event code.
"""
function homoclinic_special_point_label(kind::Symbol)
    normalized = Symbol(lowercase(String(kind)))
    return get(_HOMOCLINIC_EVENT_LABELS, normalized, uppercase(String(kind)))
end

"""
    homoclinic_orbit(result, branch_index) -> HomoclinicOrbitRecord

Return a stored normalized orbit for the requested full-locus branch index.
Only the bounded orbit subset selected by the configuration is retained.
"""
function homoclinic_orbit(result::HomoclinicBranchResult, branch_index::Int)
    index = findfirst(orbit -> orbit.branch_index == branch_index, result.orbits)
    isnothing(index) && throw(ArgumentError(
        "No stored connecting orbit for branch index $branch_index; available indices are " *
        "$(join(sort([orbit.branch_index for orbit in result.orbits]), ", "))."))
    return result.orbits[index]
end

# --- seed coercion ------------------------------------------------------------

_coerce_orbit_guess(guess::AbstractMatrix, M::Int) =
    _resample_states(collect(1.0:size(guess, 2)), Matrix{Float64}(guess), M)

function _coerce_orbit_guess(guess, M::Int)
    # A callable τ ∈ [0, 1] -> state vector.
    τ = range(0.0, 1.0, length=M + 1)
    cols = [collect(Float64, guess(t)) for t in τ]
    return reduce(hcat, cols)
end

function _build_seed(field, kind::Symbol, base_params::Vector{Float64}, primary_index::Int,
                     secondary_index::Int, M::Int, orbit_guess, saddle_guess,
                     target_guess, T0::Float64)
    primary0 = base_params[primary_index]
    secondary0 = base_params[secondary_index]
    U0 = _coerce_orbit_guess(orbit_guess, M)
    source_guess = if isnothing(saddle_guess)
        kind == :homoclinic || throw(ArgumentError(
            "A $(kind) connection requires a source saddle guess."))
        speeds = [norm(field(collect(Float64, U0[:, j]), base_params))
                  for j in axes(U0, 2)]
        collect(Float64, U0[:, argmin(speeds)])
    else
        collect(Float64, saddle_guess)
    end
    xs0, _, _ = _solve_saddle(field, source_guess, base_params)
    if kind == :heteroclinic
        target_guess === nothing &&
            throw(ArgumentError("A heteroclinic connection requires a target_saddle_guess."))
        xt0, _, _ = _solve_saddle(field, collect(Float64, target_guess), base_params)
    else
        xt0 = xs0
    end
    return _ConnectingSeed(U0, xs0, xt0, T0, primary0, secondary0)
end

function _connecting_base_params(
        sys::ContinuousODE,
        base_params::AbstractVector,
        param_index::Int)
    np = length(sys.param_names)
    1 <= param_index <= np || throw(ArgumentError(
        "continuation parameter index $param_index is out of range 1:$np."))
    length(base_params) == np || throw(ArgumentError(
        "base_params must contain exactly $np values for system '$(sys.name)'; " *
        "received $(length(base_params))."))
    all(isfinite, base_params) || throw(ArgumentError(
        "base_params must contain only finite values."))
    return collect(Float64, base_params)
end

# --- generalized entry point --------------------------------------------------

"""
    connecting_orbit_continuation(sys, config::ConnectingOrbitConfig; kwargs...) -> HomoclinicBranchResult

Continue an equilibrium connecting orbit (homoclinic or heteroclinic) with the
projection boundary-condition method.

# Keyword arguments
- `primary_param_index`: Index of the first free parameter (distinct from
  `config.continuation.param_index`, the secondary parameter).
- `orbit_guess`: Seed trajectory, either a `dim × K` matrix of state samples or a
  callable `τ ∈ [0, 1] -> state`.
- `saddle_guess`: Initial guess for the source saddle equilibrium. It may be
  omitted for a homoclinic connection, in which case the slowest sampled orbit
  state seeds the saddle solve.
- `target_saddle_guess`: Initial guess for the target saddle (heteroclinic only).
- `truncation_time`: Truncation time `T` of the seed orbit.
- `base_params`: Base parameter vector (defaults to `sys.default_params`).
"""
function connecting_orbit_continuation(sys::ContinuousODE, config::ConnectingOrbitConfig;
                                       primary_param_index::Int,
                                       orbit_guess,
                                       saddle_guess=nothing,
                                       target_saddle_guess=nothing,
                                       truncation_time::Real,
                                       base_params::AbstractVector=sys.default_params,
                                       source_period::Int=0, source_index::Int=0,
                                       provenance::String=config.provenance)
    source_period >= 0 || throw(ArgumentError("source_period must be non-negative."))
    source_index >= 0 || throw(ArgumentError("source_index must be non-negative."))
    ((source_period == 0) == (source_index == 0)) || throw(ArgumentError(
        "source_period and source_index must either both be zero or both be positive."))
    config.kind in (:homoclinic, :heteroclinic) ||
        throw(ArgumentError("connecting_orbit_continuation handles :homoclinic and " *
                            ":heteroclinic connections; use saddle_cycle_homoclinic_continuation " *
                            "for :saddle_cycle."))
    secondary_index = config.continuation.param_index
    primary_param_index != secondary_index ||
        throw(ArgumentError("primary and secondary parameter indices must differ " *
                            "(both = $(secondary_index))."))
    np = length(sys.param_names)
    (1 <= primary_param_index <= np) ||
        throw(ArgumentError("primary_param_index $(primary_param_index) is out of range 1:$(np)."))
    field = _ode_field(sys)
    bp = _connecting_base_params(sys, base_params, secondary_index)
    T_seed = Float64(truncation_time)
    (isfinite(T_seed) && T_seed > 0) ||
        throw(ArgumentError(
            "truncation_time must be finite and positive (got $(T_seed))."))
    seed = _build_seed(field, config.kind, bp, primary_param_index, secondary_index,
                       config.n_mesh, orbit_guess, saddle_guess, target_saddle_guess,
                       T_seed)

    # Validate seed endpoint displacements from the saddle and resolve epsilon values.
    # NaN in the config means "derive from the seed's natural endpoint distance";
    # an explicit positive value rescales the seed endpoint to that exact radius before
    # Newton correction, so the BVP boundary condition pins it to the requested distance.
    d0 = seed.U[:, 1] .- seed.xs
    d1 = seed.U[:, end] .- seed.xt
    n0 = norm(d0)
    n1 = norm(d1)
    (isfinite(n0) && n0 > 0) ||
        throw(ArgumentError("Seed orbit start endpoint coincides with the source saddle " *
                            "(or has zero/nonfinite displacement). Provide a truncated orbit " *
                            "whose start endpoint lies at a small positive distance from the saddle."))
    (isfinite(n1) && n1 > 0) ||
        throw(ArgumentError("Seed orbit end endpoint coincides with the target saddle " *
                            "(or has zero/nonfinite displacement). Provide a truncated orbit " *
                            "whose end endpoint lies at a small positive distance from the saddle."))
    eps_start = isnan(config.epsilon_start) ? n0 : config.epsilon_start
    eps_end   = isnan(config.epsilon_end)   ? n1 : config.epsilon_end
    if !isnan(config.epsilon_start) || !isnan(config.epsilon_end)
        seed_U = copy(seed.U)
        if !isnan(config.epsilon_start)
            seed_U[:, 1] = seed.xs .+ (eps_start / n0) .* d0
        end
        if !isnan(config.epsilon_end)
            seed_U[:, end] = seed.xt .+ (eps_end / n1) .* d1
        end
        seed = _ConnectingSeed(seed_U, seed.xs, seed.xt, seed.T, seed.primary, seed.secondary)
    end

    prob = _ConnectingProblem(field, bp, primary_param_index, secondary_index, sys.dim,
                              config.n_mesh, config.kind, eps_start, eps_end)
    return _run_connecting_orbit_continuation(
        sys, prob, seed, config;
        source_period=source_period, source_index=source_index,
        provenance=provenance)
end

# --- homoclinic ---------------------------------------------------------------

"""
    homoclinic_orbit_continuation(sys, config::ConnectingOrbitConfig; kwargs...) -> HomoclinicBranchResult

Continue a homoclinic connection to an equilibrium from an explicit seed. See
[`connecting_orbit_continuation`](@ref) for the keyword arguments
(`primary_param_index`, `orbit_guess`, `saddle_guess`, `truncation_time`,
`base_params`).
"""
function homoclinic_orbit_continuation(sys::ContinuousODE, config::ConnectingOrbitConfig;
                                       primary_param_index::Int, orbit_guess,
                                       saddle_guess=nothing,
                                       truncation_time::Real,
                                       base_params::AbstractVector=sys.default_params,
                                       source_period::Int=0, source_index::Int=0,
                                       provenance::String=config.provenance)
    config.kind == :homoclinic || throw(ArgumentError(
        "homoclinic_orbit_continuation requires config.kind=:homoclinic."))
    return connecting_orbit_continuation(
        sys, config; primary_param_index=primary_param_index,
        orbit_guess=orbit_guess, saddle_guess=saddle_guess,
        truncation_time=truncation_time, base_params=base_params,
        source_period=source_period, source_index=source_index,
        provenance=provenance)
end

"""
    homoclinic_orbit_continuation(sys, source::OrbitBranchResult, config::ConnectingOrbitConfig) -> HomoclinicBranchResult

Continue a homoclinic connection seeded from a long-period collocation orbit. The
stored orbit selected by `config.source_index` (`0` selects the longest-period
orbit) provides the seed trajectory; its slowest sample seeds the saddle, and the
orbit's branch parameter becomes the primary continuation parameter.
"""
function homoclinic_orbit_continuation(sys::ContinuousODE, source::OrbitBranchResult,
                                       config::ConnectingOrbitConfig)
    config.kind == :homoclinic || throw(ArgumentError(
        "homoclinic_orbit_continuation requires config.kind=:homoclinic."))
    source.system_name == sys.name || throw(ArgumentError(
        "Source orbit branch belongs to system '$(source.system_name)', not '$(sys.name)'."))
    length(source.base_params) == length(sys.param_names) || throw(ArgumentError(
        "Source orbit branch has $(length(source.base_params)) parameters, but system " *
        "'$(sys.name)' declares $(length(sys.param_names))."))
    1 <= source.param_index <= length(sys.param_names) || throw(ArgumentError(
        "Source orbit branch parameter index $(source.param_index) is outside the system parameter layout."))
    source.param_name == sys.param_names[source.param_index] || throw(ArgumentError(
        "Source orbit branch parameter '$(source.param_name)' does not match system parameter " *
        "'$(sys.param_names[source.param_index])' at index $(source.param_index)."))
    all(index -> 1 <= index <= length(sys.param_names), source.linked_param_indices) ||
        throw(ArgumentError(
            "Source orbit branch contains linked parameter indices outside the system parameter layout."))
    count = _orbit_branch_count(source)
    count >= 1 || throw(ArgumentError("Source orbit branch is empty."))
    primary_index = source.param_index
    secondary_index = config.continuation.param_index
    primary_index != secondary_index ||
        throw(ArgumentError("Source-orbit primary parameter index $(primary_index) must differ " *
                            "from the secondary continuation parameter index."))
    idx = config.source_index == 0 ? argmax(orbit_branch_periods(source)) :
          clamp(config.source_index, 1, count)
    t, states = orbit_branch_orbit(source, idx)
    size(states, 1) == sys.dim || throw(ArgumentError(
        "Source orbit state dimension $(size(states, 1)) does not match system dimension $(sys.dim)."))
    size(states, 2) == length(t) || throw(ArgumentError(
        "Source orbit state/time sample counts do not match."))
    field = _ode_field(sys)
    primary0 = orbit_branch_parameters(source)[idx]
    bp = collect(Float64, source.base_params)
    bp[primary_index] = primary0
    speeds = [norm(field(collect(Float64, states[:, j]), bp)) for j in axes(states, 2)]
    saddle_guess = collect(Float64, states[:, argmin(speeds)])
    return connecting_orbit_continuation(
        sys, config; primary_param_index=primary_index,
        orbit_guess=states, saddle_guess=saddle_guess,
        truncation_time=Float64(t[end] - t[1]),
        base_params=bp, source_period=source.period,
        source_index=idx, provenance="orbit-branch-seed")
end

# --- heteroclinic -------------------------------------------------------------

"""
    heteroclinic_orbit_continuation(sys, config::ConnectingOrbitConfig; kwargs...) -> HomoclinicBranchResult

Continue a heteroclinic connection between two saddle equilibria. Requires
`source_saddle` and `target_saddle` guesses in addition to the seed orbit.
"""
function heteroclinic_orbit_continuation(sys::ContinuousODE, config::ConnectingOrbitConfig;
                                         primary_param_index::Int, source_saddle, target_saddle,
                                         orbit_guess, truncation_time::Real,
                                         base_params::AbstractVector=sys.default_params,
                                         provenance::String=config.provenance)
    cc = config.kind == :heteroclinic ? config : (Accessors.@set config.kind = :heteroclinic)
    return connecting_orbit_continuation(sys, cc; primary_param_index=primary_param_index,
                                         orbit_guess=orbit_guess, saddle_guess=source_saddle,
                                         target_saddle_guess=target_saddle,
                                         truncation_time=truncation_time, base_params=base_params,
                                         provenance=provenance)
end

# --- saddle-cycle homoclinic --------------------------------------------------

function _assemble_cycle_result(sys::ContinuousODE, prob::_CycleProblem, corr::_CorrectorResult,
                                config::ConnectingOrbitConfig, cycle_period::Float64,
                                provenance::String)
    n = prob.n
    M = prob.M
    U = Matrix{Float64}(reshape(corr.z[1:n * (M + 1)], n, M + 1))
    T = corr.z[n * (M + 1) + 1]
    eps_start = norm(U[:, 1] .- prob.x0)
    eps_end = norm(U[:, end] .- prob.x0)
    split = prob.split
    param_index = config.continuation.param_index
    pval = 1 <= param_index <= length(prob.base_params) ? prob.base_params[param_index] : 0.0
    pname = 1 <= param_index <= length(sys.param_names) ? sys.param_names[param_index] :
            Symbol("p", param_index)

    saddles = reshape(copy(prob.x0), n, 1)
    t = collect(range(0.0, T, length=M + 1))
    orbit = HomoclinicOrbitRecord(1, t, U, copy(prob.x0), pval, pval, T, eps_start, eps_end)
    # A single correction has no continuation direction; still expose two distinct
    # parameter slots so the result satisfies the branch-result invariants.
    nparams = length(prob.base_params)
    secondary_index = param_index == 1 ? (nparams >= 2 ? 2 : 1) : 1
    sname = 1 <= secondary_index <= length(sys.param_names) ? sys.param_names[secondary_index] :
            Symbol("p", secondary_index)
    diagnostics = Dict{String, Any}(
        "kind" => "saddle_cycle",
        "mesh_intervals" => M,
        "epsilon_start" => prob.eps0,
        "epsilon_end" => prob.eps1,
        "cycle_period" => cycle_period,
        "floquet_multipliers_re" => real.(split.multipliers),
        "floquet_multipliers_im" => imag.(split.multipliers),
        "stable_floquet_dim" => split.ns,
        "unstable_floquet_dim" => split.nu,
        "center_floquet_dim" => split.nc,
        "converged" => corr.converged,
        "max_residual" => corr.residual,
        "corrector_path" => String(corr.path),
        "seed_source" => provenance,
        "phase" => "reference cross-section (endpoints pinned to the reference phase point)",
    )
    return HomoclinicBranchResult(
        [pval], [pval], [T], [eps_start], [eps_end],
        saddles, copy(saddles), Dict{Symbol, Vector{Float64}}(),
        Dict{Symbol, Vector{Symbol}}(),
        HomoclinicSpecialPoint[], [orbit],
        [corr.residual], [corr.path], :saddle_cycle,
        0, 0, pval, copy(prob.base_params), param_index, secondary_index,
        sys.name, (pname, sname), diagnostics, now())
end

"""
    saddle_cycle_homoclinic_continuation(sys, config::ConnectingOrbitConfig; kwargs...) -> HomoclinicBranchResult

Correct a homoclinic connection to a saddle periodic orbit. The monodromy matrix
of the sampled cycle is built from the variational equation, its Floquet
multipliers classify the stable/unstable manifolds, the geometry is validated
(the cycle must be a genuine saddle with a single trivial multiplier), and the
truncated orbit's endpoints are pinned to the unstable/stable Floquet subspaces
at the reference cross-section (phase-aware endpoint projection). The Floquet
data and convergence provenance are recorded in `diagnostics`.

# Keyword arguments
- `cycle_states`: `dim × L` samples of one period of the saddle cycle, including
  both endpoints at the same reference phase.
- `cycle_period`: The cycle period.
- `orbit_guess`: Seed connecting trajectory (`dim × K` matrix or callable).
- `truncation_time`: Truncation time `T` of the seed connecting orbit.
- `reference_index`: Cycle sample used as the reference cross-section.
- `base_params`: Base parameter vector (defaults to `sys.default_params`).
"""
function saddle_cycle_homoclinic_continuation(sys::ContinuousODE, config::ConnectingOrbitConfig;
                                              cycle_states::AbstractMatrix, cycle_period::Real,
                                              orbit_guess, truncation_time::Real,
                                              reference_index::Int=1,
                                              base_params::AbstractVector=sys.default_params,
                                              provenance::String=config.provenance)
    config.kind == :saddle_cycle || throw(ArgumentError(
        "saddle_cycle_homoclinic_continuation requires config.kind=:saddle_cycle."))
    n = sys.dim
    size(cycle_states, 1) == n ||
        throw(ArgumentError("cycle_states must have $(n) rows (state dimension)."))
    size(cycle_states, 2) >= 2 ||
        throw(ArgumentError("cycle_states must contain at least two time samples."))
    all(isfinite, cycle_states) ||
        throw(ArgumentError("cycle_states must contain only finite values."))
    cycle_closure_scale = max(
        norm(cycle_states[:, 1]), norm(cycle_states[:, end]), 1.0)
    cycle_closure_error = norm(cycle_states[:, end] .- cycle_states[:, 1])
    cycle_closure_error <= 1e-4 * cycle_closure_scale || throw(ArgumentError(
        "cycle_states must span one closed period with matching first and last " *
        "samples (relative closure error = $(cycle_closure_error / cycle_closure_scale))."))

    T_seed = Float64(truncation_time)
    (isfinite(T_seed) && T_seed > 0) ||
        throw(ArgumentError(
            "truncation_time must be finite and positive (got $(T_seed))."))
    if isfinite(config.max_return_time) && T_seed > config.max_return_time
        throw(ArgumentError(
            "Seed truncation_time $(T_seed) already exceeds max_return_time " *
            "$(config.max_return_time). Reduce truncation_time or increase max_return_time."))
    end
    Tc = Float64(cycle_period)
    (isfinite(Tc) && Tc > 0) ||
        throw(ArgumentError("cycle_period must be finite and positive (got $(Tc))."))

    field = _ode_field(sys)
    bp = _connecting_base_params(
        sys, base_params, config.continuation.param_index)
    Mmono = _cycle_monodromy(field, cycle_states, Tc, bp)
    split = _floquet_split(Mmono)
    _validate_saddle_cycle_geometry(split, n)

    (1 <= reference_index <= size(cycle_states, 2)) ||
        throw(ArgumentError("reference_index out of range."))
    x0 = collect(Float64, cycle_states[:, reference_index])
    U0 = _coerce_orbit_guess(orbit_guess, config.n_mesh)

    # Validate seed endpoint displacements from the reference phase point and resolve
    # epsilon values. NaN in the config means "derive from the seed's natural distance";
    # an explicit positive value rescales the seed endpoint to that exact radius.
    d0 = U0[:, 1] .- x0
    d1 = U0[:, end] .- x0
    n0 = norm(d0)
    n1 = norm(d1)
    (isfinite(n0) && n0 > 0) ||
        throw(ArgumentError("Seed connecting-orbit start endpoint coincides with the reference " *
                            "phase point (or has zero/nonfinite displacement)."))
    (isfinite(n1) && n1 > 0) ||
        throw(ArgumentError("Seed connecting-orbit end endpoint coincides with the reference " *
                            "phase point (or has zero/nonfinite displacement)."))
    eps_start = isnan(config.epsilon_start) ? n0 : config.epsilon_start
    eps_end   = isnan(config.epsilon_end)   ? n1 : config.epsilon_end
    if !isnan(config.epsilon_start)
        U0[:, 1] = x0 .+ (eps_start / n0) .* d0
    end
    if !isnan(config.epsilon_end)
        U0[:, end] = x0 .+ (eps_end / n1) .* d1
    end

    prob = _CycleProblem(field, bp, bp, n, config.n_mesh, x0, eps_start, eps_end, split)
    z0 = vcat(vec(U0), T_seed)
    corr = _correct_cycle_homoclinic(prob, z0; tol=config.continuation.newton_tol,
                                     maxiter=config.continuation.newton_max_iter,
                                     use_fallback=config.use_fallback,
                                     fallback_max_iter=config.fallback_max_iter)

    # Reject unconverged correction consistently with the equilibrium path.
    corr.converged || throw(ErrorException(
        "Saddle-cycle homoclinic corrector did not converge (residual = $(corr.residual)). " *
        "Improve the orbit guess, adjust epsilon_start/epsilon_end, or relax " *
        "newton_tol."))

    # Post-check: validate the corrected T and enforce the cap.
    T_corrected = corr.z[n * (config.n_mesh + 1) + 1]
    (isfinite(T_corrected) && T_corrected > 0) ||
        throw(ErrorException(
            "Saddle-cycle homoclinic corrector converged, but the corrected truncation " *
            "time T = $(T_corrected) is not finite and positive."))
    if isfinite(config.max_return_time)
        if T_corrected > config.max_return_time
            throw(ErrorException(
                "Saddle-cycle homoclinic corrector converged, but the corrected truncation " *
                "time T = $(T_corrected) exceeds max_return_time $(config.max_return_time). " *
                "The seed point cannot be accepted. Reduce truncation_time or increase " *
                "max_return_time."))
        end
    end

    return _assemble_cycle_result(sys, prob, corr, config, Tc, provenance)
end
