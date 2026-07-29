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

"""
    _validate_orbit_guess_variation(U0, label)

Reject a connecting-orbit guess whose mesh points all coincide (to within
numerical tolerance) — e.g. a constant array repeated across the mesh. Such a
seed cannot support a meaningful collocation residual or phase condition
regardless of what the vector field happens to evaluate to at that single
point: evaluating the field at a state does not certify that the *seed
trajectory* spans anything, since a system's field can be nonzero even at a
degenerate/repeated point (for example, near a coordinate-singularity guard
like `max(r^2, eps)`) without the supplied orbit describing real dynamics.

The degeneracy tolerance is relative to `eltype(U0)`'s working precision
(`sqrt(eps)` of the real/underlying float type) rather than a fixed `Float64`
constant, so this remains meaningful if a caller ever supplies a guess in a
different real float type; orbit guesses are physical state samples, so no
`Complex`/dual element type is expected here, but the check stays type-generic
rather than assuming `Float64`. Allocation-free: computes the pairwise spread
with explicit loops instead of broadcasting temporaries, since this runs on
every connecting-orbit continuation call.
"""
function _validate_orbit_guess_variation(U0::AbstractMatrix, label::AbstractString)
    K = size(U0, 2)
    K >= 2 || throw(ArgumentError("$label must contain at least 2 mesh points; got $K."))
    all(isfinite, U0) || throw(ArgumentError("$label must contain only finite values."))
    T = float(real(eltype(U0)))
    n = size(U0, 1)
    scale_sq = zero(T)
    @inbounds for i in 1:n
        scale_sq += abs2(U0[i, 1])
    end
    scale = max(sqrt(scale_sq), one(T))
    spread = zero(T)
    @inbounds for j in 1:K
        d_sq = zero(T)
        for i in 1:n
            d_sq += abs2(U0[i, j] - U0[i, 1])
        end
        spread = max(spread, sqrt(d_sq))
    end
    tol = sqrt(eps(T))
    spread > tol * scale || throw(ArgumentError(
        "$label is degenerate: all mesh points coincide to within numerical " *
        "tolerance (spread=$(spread), scale=$(scale), tol=$(tol)). Provide a " *
        "genuinely time-varying connecting-orbit guess."))
    return nothing
end

function _build_seed(field, kind::Symbol, base_params::Vector{Float64}, primary_index::Int,
                     secondary_index::Int, M::Int, orbit_guess, saddle_guess,
                     target_guess, T0::Float64)
    primary0 = base_params[primary_index]
    secondary0 = base_params[secondary_index]
    U0 = _coerce_orbit_guess(orbit_guess, M)
    _validate_orbit_guess_variation(U0, "orbit_guess")
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

# --- cycle sample validation --------------------------------------------------

function _validate_cycle_samples(label::AbstractString, states::AbstractMatrix,
                                 period::Real, n::Int; min_samples::Int=3)
    stem = isempty(label) ? "cycle" : "$(label)_cycle"
    size(states, 1) == n ||
        throw(ArgumentError("$(stem)_states must have $(n) rows (state dimension)."))
    size(states, 2) >= min_samples ||
        throw(ArgumentError("$(stem)_states must contain at least $(min_samples) time samples."))
    all(isfinite, states) ||
        throw(ArgumentError("$(stem)_states must contain only finite values."))
    T = Float64(period)
    (isfinite(T) && T > 0) ||
        throw(ArgumentError("$(stem)_period must be finite and positive (got $(T))."))
    first_state = view(states, :, 1)
    last_state = view(states, :, size(states, 2))
    closure_scale = max(norm(first_state), norm(last_state), 1.0)
    closure_error = norm(last_state .- first_state)
    closure_error <= 1e-4 * closure_scale || throw(ArgumentError(
        "$(stem)_states must span one closed period with matching first " *
        "and last samples (relative closure error = $(closure_error / closure_scale))."))
    return T
end

function _connection_phase_tangents(field, U::AbstractMatrix, p::AbstractVector)
    tangents = Matrix{Float64}(undef, size(U, 1), size(U, 2))
    for j in axes(U, 2)
        tangents[:, j] = collect(Float64, field(view(U, :, j), p))
    end
    norm(tangents) > 0 || throw(ArgumentError(
        "Connection phase condition is rank deficient: the seed trajectory has " *
        "zero tangent norm. Provide a non-stationary connecting-orbit guess."))
    return tangents
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

    T_seed = Float64(truncation_time)
    (isfinite(T_seed) && T_seed > 0) ||
        throw(ArgumentError(
            "truncation_time must be finite and positive (got $(T_seed))."))
    if isfinite(config.max_return_time) && T_seed > config.max_return_time
        throw(ArgumentError(
            "Seed truncation_time $(T_seed) already exceeds max_return_time " *
            "$(config.max_return_time). Reduce truncation_time or increase max_return_time."))
    end
    Tc = _validate_cycle_samples("", cycle_states, cycle_period, n; min_samples=2)

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
    _validate_orbit_guess_variation(U0, "orbit_guess")

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

# --- cycle-to-cycle connection ------------------------------------------------

"""
    cycle_connection_seed(sys; source_cycle_states, source_cycle_period,
                          target_cycle_states, target_cycle_period, base_params,
                          seed_config=CycleConnectionSeedConfig()) -> CycleConnectionSeedResult

Automatically discover a seed trajectory for a saddle-cycle to saddle-cycle
connection by sampling source-cycle phases and unstable Floquet launch
directions, then integrating forward until a trajectory approaches the target
cycle. Returned cycle samples are rotated so the selected source/target phases
are the reference phases expected by [`cycle_connection_continuation`](@ref).
"""
function cycle_connection_seed(sys::ContinuousODE;
                               source_cycle_states::AbstractMatrix,
                               source_cycle_period::Real,
                               target_cycle_states::AbstractMatrix,
                               target_cycle_period::Real,
                               base_params::AbstractVector=sys.default_params,
                               seed_config::CycleConnectionSeedConfig=CycleConnectionSeedConfig(),
                               solver=Tsit5(),
                               reltol::Float64=1e-9,
                               abstol::Float64=1e-9)
    n = sys.dim
    _validate_cycle_samples("source", source_cycle_states, source_cycle_period, n)
    _validate_cycle_samples("target", target_cycle_states, target_cycle_period, n)
    length(base_params) == length(sys.param_names) || throw(ArgumentError(
        "base_params must contain exactly $(length(sys.param_names)) values for system '$(sys.name)'."))
    all(isfinite, base_params) || throw(ArgumentError(
        "base_params must contain only finite values."))
    return _discover_cycle_connection_seed(
        sys, seed_config;
        source_cycle_states=source_cycle_states,
        source_cycle_period=source_cycle_period,
        target_cycle_states=target_cycle_states,
        target_cycle_period=target_cycle_period,
        base_params=base_params,
        solver=solver,
        reltol=reltol,
        abstol=abstol)
end

"""
    cycle_connection_continuation(sys, config::ConnectingOrbitConfig; kwargs...) -> HomoclinicBranchResult

Continue a saddle-cycle to saddle-cycle connecting orbit. The source and target
periodic orbits are solved alongside the connecting mesh, so their phases are
free and determined by the projection boundary conditions. A single integral
phase condition on the connecting orbit removes the global time-shift
degeneracy, and pseudo-arclength continuation traces the two-parameter locus.

# Keyword arguments
- `primary_param_index`: First free parameter; the secondary parameter is
  `config.continuation.param_index`.
- `source_cycle_states` / `target_cycle_states`: `dim × L` samples of one
  period for each saddle cycle, including matching first/last samples.
- `source_cycle_period` / `target_cycle_period`: Flow periods of those cycles.
- `orbit_guess`: Seed connecting trajectory (`dim × K` matrix or callable).
  Omit or pass `nothing` to run automatic seed discovery.
- `truncation_time`: Seed connecting-orbit truncation time. Required with an
  explicit `orbit_guess`; ignored for automatic discovery.
- `base_params`: Base parameter vector (defaults to `sys.default_params`).
"""
function cycle_connection_continuation(sys::ContinuousODE, config::ConnectingOrbitConfig;
                                       primary_param_index::Int,
                                       source_cycle_states::AbstractMatrix,
                                       source_cycle_period::Real,
                                       target_cycle_states::AbstractMatrix,
                                       target_cycle_period::Real,
                                       orbit_guess=nothing,
                                       truncation_time::Real=NaN,
                                       base_params::AbstractVector=sys.default_params,
                                       seed_config::CycleConnectionSeedConfig=CycleConnectionSeedConfig(),
                                       solver=Tsit5(),
                                       reltol::Float64=1e-9,
                                       abstol::Float64=1e-9,
                                       provenance::String=config.provenance)
    config.kind == :cycle_connection || throw(ArgumentError(
        "cycle_connection_continuation requires config.kind=:cycle_connection."))
    secondary_index = config.continuation.param_index
    primary_param_index != secondary_index ||
        throw(ArgumentError("primary and secondary parameter indices must differ " *
                            "(both = $(secondary_index))."))
    np = length(sys.param_names)
    1 <= primary_param_index <= np ||
        throw(ArgumentError("primary_param_index $(primary_param_index) is out of range 1:$(np)."))
    n = sys.dim
    Tsource = _validate_cycle_samples("source", source_cycle_states, source_cycle_period, n)
    Ttarget = _validate_cycle_samples("target", target_cycle_states, target_cycle_period, n)
    field = _ode_field(sys)
    bp = _connecting_base_params(sys, base_params, secondary_index)
    source = Matrix{Float64}(source_cycle_states)
    target = Matrix{Float64}(target_cycle_states)
    seed_result = nothing
    if isnothing(orbit_guess)
        seed_result = cycle_connection_seed(
            sys; source_cycle_states=source, source_cycle_period=Tsource,
            target_cycle_states=target, target_cycle_period=Ttarget,
            base_params=bp, seed_config=seed_config, solver=solver,
            reltol=reltol, abstol=abstol)
        seed_result.status === :found || throw(ErrorException(
            "Automatic cycle-connection seed discovery did not approach the target " *
            "cycle within distance_tolerance=$(seed_config.distance_tolerance) " *
            "(best distance = $(seed_result.distance)). Increase max_time, sample more " *
            "phases, or provide an explicit orbit_guess."))
        orbit_guess = seed_result.orbit_guess
        truncation_time = seed_result.truncation_time
        source = seed_result.source_cycle_states
        target = seed_result.target_cycle_states
        provenance = isempty(provenance) ? "automatic-cycle-connection-seed" :
                     "$(provenance); automatic-cycle-connection-seed"
    end
    Tseed = Float64(truncation_time)
    (isfinite(Tseed) && Tseed > 0) ||
        throw(ArgumentError("truncation_time must be finite and positive (got $(Tseed))."))
    if isfinite(config.max_return_time) && Tseed > config.max_return_time
        throw(ArgumentError(
            "Seed truncation_time $(Tseed) already exceeds max_return_time " *
            "$(config.max_return_time). Reduce truncation_time or increase max_return_time."))
    end

    U0 = _coerce_orbit_guess(orbit_guess, config.n_mesh)
    size(U0, 1) == n ||
        throw(ArgumentError("orbit_guess state dimension $(size(U0, 1)) does not match system dimension $(n)."))
    _validate_orbit_guess_variation(U0, "orbit_guess")
    d0 = U0[:, 1] .- source[:, 1]
    d1 = U0[:, end] .- target[:, 1]
    n0 = norm(d0)
    n1 = norm(d1)
    (isfinite(n0) && n0 > 0) ||
        throw(ArgumentError("Seed connecting-orbit start endpoint coincides with the source cycle phase point."))
    (isfinite(n1) && n1 > 0) ||
        throw(ArgumentError("Seed connecting-orbit end endpoint coincides with the target cycle phase point."))
    eps_start = isnan(config.epsilon_start) ? n0 : config.epsilon_start
    eps_end = isnan(config.epsilon_end) ? n1 : config.epsilon_end
    if !isnan(config.epsilon_start)
        U0[:, 1] = source[:, 1] .+ (eps_start / n0) .* d0
    end
    if !isnan(config.epsilon_end)
        U0[:, end] = target[:, 1] .+ (eps_end / n1) .* d1
    end
    tangents = _connection_phase_tangents(field, U0, bp)
    prob = _CycleConnectionProblem(
        field, bp, primary_param_index, secondary_index, n, config.n_mesh,
        size(source, 2) - 1, size(target, 2) - 1, copy(U0), tangents,
        eps_start, eps_end)
    seed = _cycle_connection_seed_vector(
        prob, U0, source, target, Tseed, Tsource, Ttarget,
        bp[primary_param_index], bp[secondary_index])
    result = _run_cycle_connection_continuation(sys, prob, seed, config; provenance=provenance)
    if seed_result !== nothing
        result.diagnostics["seed_discovery"] = Dict{String, Any}(
            "status" => String(seed_result.status),
            "distance" => seed_result.distance,
            "source_phase_index" => seed_result.source_phase_index,
            "target_phase_index" => seed_result.target_phase_index,
            "source_direction_index" => seed_result.source_direction_index,
            "source_direction_sign" => seed_result.source_direction_sign,
            "truncation_time" => seed_result.truncation_time,
            "diagnostics" => seed_result.diagnostics,
        )
    end
    return result
end
