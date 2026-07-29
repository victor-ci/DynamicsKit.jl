"""
Switching map generator: construct a `DiscreteMap` from a piecewise-linear
circuit description (per-mode affine state-space data + algebraic switching timing).

The affine primitive supports arbitrary finite state dimension by exponentiating
the augmented Duhamel matrix for `dx/dt = A*x + b`, so singular and defective
mode matrices use the same path as nonsingular matrices.
"""

# ─── Descriptor types ─────────────────────────────────────────────────────────

"""
    AffineModeSpec{A,B,D,G}

Describes one operating mode of a piecewise-linear switching circuit.

Type parameters encode the concrete types of the state-matrix, affine-input,
duration, and optional boundary-state suppliers — no field is annotated `Function`:
- `A`: square matrix constant **or** a callable `p -> matrix`.
- `B`: vector constant **or** a callable `p -> vector`.
- `D`: callable `(x, p) -> Real` for intermediate modes, `Nothing` for
  the final mode (which consumes the remaining clock period).
- `G`: `Nothing` (no override) **or** a callable `(x_flow, p) -> SVector`
  that overrides the state at the end of the mode.  Use this to enforce
  switching conditions — for example, forcing the inductor current to exactly
  `Iref` in a peak-current-mode buck converter, matching the comparator trip.

# Fields
- `A_fn::A`: state matrix supplier.
- `b_fn::B`: affine input supplier.
- `duration_fn::D`: duration function or `nothing` (final mode).
- `boundary_fn::G`: optional state override at the mode boundary.
- `events::Vector{SwitchingEvent}`: guard metadata for the border ending this mode.

Use constant `SMatrix`/`SVector` values when the matrix/vector does not depend on
parameters; pass a callable only when it does.
"""
struct AffineModeSpec{A, B, D, G}
    A_fn::A
    b_fn::B
    duration_fn::D
    boundary_fn::G
    events::Vector{SwitchingEvent}
end

"""
    AffineModeSpec(A, b; duration=nothing, boundary=nothing, events=SwitchingEvent[])

Construct an `AffineModeSpec`. `A` and `b` are either constant
`SMatrix{2,2}` / `SVector{2}` values or callables `p -> matrix` /
`p -> vector`; `duration` is a callable `(x, p) -> Real` for intermediate
modes or `nothing` for the final mode; `boundary` is an optional callable
`(x_flow, p) -> SVector{2}` that overrides the state output of the mode
(useful for enforcing exact switching conditions).
"""
function AffineModeSpec(A, b;
                        duration=nothing,
                        boundary=nothing,
                        events::AbstractVector{SwitchingEvent}=SwitchingEvent[])
    AffineModeSpec{typeof(A), typeof(b), typeof(duration), typeof(boundary)}(
        A, b, duration, boundary, collect(SwitchingEvent, events))
end

"""
    SwitchingCircuitDescription{T}

Ordered list of operating modes making up one clock period of a
piecewise-linear switching circuit, together with the clock period and
parameter metadata needed to construct a `DiscreteMap` via `switching_map`.

Type parameter `T` is the type of `period` — a `Float64` constant or a
callable `p -> Real` for parameter-dependent periods. `M` is the concrete
mode tuple type.

# Fields
- `modes`: ordered mode descriptions, one per operating mode.
- `period::T`: clock period constant or callable.
- `param_names::Vector{Symbol}`: bifurcation-parameter names forwarded to the
  generated `DiscreteMap`.
- `name::String`: human-readable circuit name.
- `state_dim::Int`: state dimension of the generated map.
"""
struct SwitchingCircuitDescription{T, M<:Tuple}
    modes::M
    period::T
    param_names::Vector{Symbol}
    name::String
    state_dim::Int
end

"""
   SwitchingCircuitDescription(modes, period; param_names=Symbol[], name="Switching Circuit", state_dim=nothing)

Construct a `SwitchingCircuitDescription` from an ordered collection of
`AffineModeSpec` values and a clock period (constant `Float64` or callable).
When all mode matrices/vectors are callable, pass `state_dim` explicitly.
"""
function SwitchingCircuitDescription(modes, period;
                                     param_names::AbstractVector{Symbol}=Symbol[],
                                     name::AbstractString="Switching Circuit",
                                     state_dim::Union{Integer,Nothing}=nothing)
    mode_tuple = Tuple(modes)
    inferred_dim = state_dim === nothing ? nothing : Int(state_dim)
    inferred_dim !== nothing && inferred_dim > 0 ||
        inferred_dim === nothing ||
        throw(ArgumentError("SwitchingCircuitDescription: state_dim must be positive; got $state_dim."))
    for (index, mode) in enumerate(mode_tuple)
        mode isa AffineModeSpec ||
           throw(ArgumentError(
               "SwitchingCircuitDescription: mode $index is not an AffineModeSpec."))
        mode_dim = _sw_mode_state_dim(mode)
        if mode_dim !== nothing
           if inferred_dim === nothing
               inferred_dim = mode_dim
           elseif inferred_dim != mode_dim
               throw(ArgumentError(
                   "SwitchingCircuitDescription: mode $index has dimension $mode_dim, expected $inferred_dim."))
           end
        end
    end
    inferred_dim === nothing &&
        throw(ArgumentError(
            "SwitchingCircuitDescription: could not infer state dimension; pass state_dim explicitly."))
    SwitchingCircuitDescription{typeof(period), typeof(mode_tuple)}(
        mode_tuple, period,
        collect(Symbol, param_names), String(name), inferred_dim)
end

# ─── Evaluation helpers ───────────────────────────────────────────────────────

# Retrieve A at params p — dispatch on constant vs callable.
_sw_A(A_fn::AbstractMatrix, p) = A_fn
_sw_A(A_fn, p)                 = A_fn(p)

# Retrieve b at params p — dispatch on constant vs callable.
_sw_b(b_fn::AbstractVector, p) = b_fn
_sw_b(b_fn, p)                 = b_fn(p)

# Retrieve clock period at params p — dispatch on constant vs callable.
_sw_period(T_val::Real, p) = T_val
_sw_period(T_fn, p)        = T_fn(p)

# Retrieve mode duration — callable for intermediate modes; Nothing is a guard.
_sw_duration(fn, x, p) = fn(x, p)
function _sw_duration(::Nothing, x, p)
    error("BUG: _sw_duration called on a final mode (duration_fn = nothing)")
end

function _sw_matrix_dim(A::AbstractMatrix)
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("AffineModeSpec requires a square A matrix; got size $(size(A))."))
    return size(A, 1)
end
_sw_matrix_dim(A) = nothing

_sw_vector_dim(b::AbstractVector) = length(b)
_sw_vector_dim(b) = nothing

function _sw_mode_state_dim(mode::AffineModeSpec)
    a_dim = _sw_matrix_dim(mode.A_fn)
    b_dim = _sw_vector_dim(mode.b_fn)
    if a_dim !== nothing && b_dim !== nothing && a_dim != b_dim
        throw(ArgumentError(
            "AffineModeSpec dimension mismatch: A is $(a_dim)x$(a_dim) but b has length $b_dim."))
    end
    return a_dim === nothing ? b_dim : a_dim
end

_sw_primal(x::Real) = x
_sw_primal(x::ForwardDiff.Dual) = _sw_primal(ForwardDiff.value(x))

function _sw_primal_abs(x)
    return abs(float(_sw_primal(x)))
end

function _sw_one_norm(A::AbstractMatrix)
    isempty(A) && return 0.0
    best = nothing
    for j in axes(A, 2)
        col_sum = nothing
        for i in axes(A, 1)
            value = _sw_primal_abs(A[i, j])
            col_sum = col_sum === nothing ? value : col_sum + value
        end
        best = best === nothing ? col_sum : max(best, col_sum)
    end
    return best
end

function _sw_eye_like(A::AbstractMatrix)
    n = size(A, 1)
    T = eltype(A)
    return Matrix{T}(I, n, n)
end

_sw_eye_like(A::SMatrix{N,N,T}) where {N,T} = SMatrix{N,N,T}(I)

function _sw_coeffs(A::AbstractMatrix, values)
    one_a = one(eltype(A))
    return map(v -> v * one_a, values)
end

const _SW_MATRIX_EXP_PADE13_COEFFS = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)

# Shared Padé(13) rational-approximant + scaling-and-squaring core, given an
# already norm-scaled matrix `B` (norm(B) <= theta13) and its squaring count
# `s`. This is the single source of truth for the algorithm body: both
# `_sw_matrix_exp` methods below differ only in how they obtain `B` and `s`
# (a defensive heap copy for a general `AbstractMatrix`, vs. a direct
# zero-copy pass-through for an immutable `SMatrix`) and delegate the actual
# numerics — coefficient application, U/V construction, and squaring — here,
# so a future coefficient/order change or bugfix cannot update one backend
# and silently miss the other.
function _sw_matrix_exp_pade13(B, s::Int)
    c = _sw_coeffs(B, _SW_MATRIX_EXP_PADE13_COEFFS)
    I_n = _sw_eye_like(B)
    B2 = B * B
    B4 = B2 * B2
    B6 = B4 * B2

    U = B * (B6 * (c[14] * B6 + c[12] * B4 + c[10] * B2) +
             c[8] * B6 + c[6] * B4 + c[4] * B2 + c[2] * I_n)
    V = B6 * (c[13] * B6 + c[11] * B4 + c[9] * B2) +
        c[7] * B6 + c[5] * B4 + c[3] * B2 + c[1] * I_n
    R = (V - U) \ (V + U)
    for _ in 1:s
        R = R * R
    end
    return R
end

# Norm-based scaling exponent `s` and scaled matrix `B` shared by both
# `_sw_matrix_exp` methods (Higham scaling-and-squaring: reduce ||B|| below
# theta13 by repeated halving, undone by `s` squarings in `_sw_matrix_exp_pade13`).
function _sw_matrix_exp_scale(A)
    norm_A = _sw_one_norm(A)
    theta13 = 5.371920351148152
    s = norm_A <= theta13 ? 0 : max(0, ceil(Int, log2(norm_A / theta13)))
    B = A
    if s > 0
        scale = (one(eltype(A)) + one(eltype(A)))^s
        B = B / scale
    end
    return B, s
end

"""
    _sw_matrix_exp(A)

Dense matrix exponential using the Higham scaling-and-squaring Padé(13)
algorithm. Branching uses primal values so ForwardDiff dual payloads flow
through the linear algebra operations.
"""
function _sw_matrix_exp(A::AbstractMatrix)
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("_sw_matrix_exp requires a square matrix; got size $(size(A))."))
    n = size(A, 1)
    n == 0 && return Matrix{eltype(A)}(undef, 0, 0)

    B, s = _sw_matrix_exp_scale(Matrix(A))
    return _sw_matrix_exp_pade13(B, s)
end

"""
    _sw_matrix_exp(A::SMatrix)

Stack-allocated specialization of the Higham scaling-and-squaring Padé(13)
algorithm above, used for the fixed-size augmented Duhamel matrices built by
`_affine_flow_nd`. Shares its algorithm body (`_sw_matrix_exp_pade13`) with the
`AbstractMatrix` method above — only the scale/copy setup and the array
backend differ — so every switching map (including the n-dimensional
generator's runtime-sized modes lifted to a compile-time-known augmented
dimension) gets identical numerics without the heap `Matrix` allocation and
BLAS dispatch overhead that dominates cost when this runs inside a
per-map-iteration bisection loop (e.g. peak-current-mode converters).
"""
function _sw_matrix_exp(A::SMatrix{N,N}) where {N}
    B, s = _sw_matrix_exp_scale(A)
    return _sw_matrix_exp_pade13(B, s)
end

# ─── n-D affine flow ─────────────────────────────────────────────────────────

"""
    _affine_flow_nd(x, A, b, tau)

Exact solution `x(tau)` of `dx/dt = A*x + b` using the augmented Duhamel
matrix `[[A b]; [0 0]]`. This form is valid for singular, nilpotent, and
defective matrices without solving for an equilibrium.
"""
function _affine_flow_nd(x::SVector{N}, A::SMatrix{N,N}, b::SVector{N}, tau) where {N}
    T = promote_type(eltype(x), eltype(A), eltype(b), typeof(tau))
    M = SMatrix{N + 1, N + 1, T}(ntuple(k -> begin
        i = ((k - 1) % (N + 1)) + 1
        j = ((k - 1) ÷ (N + 1)) + 1
        if j <= N
            i <= N ? A[i, j] * tau : zero(T)
        else
            i <= N ? b[i] * tau : zero(T)
        end
    end, Val((N + 1) * (N + 1))))
    y0 = SVector{N + 1, T}(ntuple(i -> i <= N ? T(x[i]) : one(T), N + 1))
    E = _sw_matrix_exp(M)
    y = E * y0
    return SVector{N}(ntuple(i -> y[i], N))
end

_affine_flow_2d(x::SVector{2}, A::SMatrix{2,2}, b::SVector{2}, tau) =
    _affine_flow_nd(x, A, b, tau)

# ─── Validation helpers ───────────────────────────────────────────────────────

function _sw_check_period(T, desc_name)
    (isfinite(T) && T > zero(T)) ||
        throw(ArgumentError(
            "switching_map ($desc_name): period must be finite and positive; got T = $T."))
end

function _sw_check_raw_duration(tau_raw, k, desc_name)
    isnan(tau_raw) &&
        throw(ArgumentError(
            "switching_map ($desc_name): duration_fn for mode $k returned NaN — check circuit parameters."))
end

# ─── Map generator ───────────────────────────────────────────────────────────

"""
    switching_map(desc::SwitchingCircuitDescription; name=nothing) -> DiscreteMap

Construct a `DiscreteMap` from a `SwitchingCircuitDescription`.

The generated map applies the affine flows of each `AffineModeSpec` in
sequence over one clock period:

- **Intermediate modes** run for the duration returned by their
  `duration_fn(x, p)`, clamped to the remaining period `[0, remaining]`.
  Clamping handles both saturation rails (`duration ≤ 0` → mode skipped;
  `duration ≥ remaining` → mode consumes the rest) matching the behaviour
  of the hand-coded `boost_converter` and the `tn ≥ T` guard in
  `buck_converter`.
- **The final mode** consumes the remaining period exactly (its
  `duration_fn` must be `nothing`).

The map is ForwardDiff-compatible away from the switching borders; the
exact Jacobian is obtained via `ForwardDiff.jacobian`.

Switching events from all modes are forwarded to the `DiscreteMap`.

# Errors
- `ArgumentError` if the clock period is non-finite or non-positive.
- `ArgumentError` if any intermediate mode's `duration_fn` returns `NaN`.
- `ArgumentError` if any intermediate mode has `duration_fn = nothing`.
- `ArgumentError` if no modes are provided.
"""
function switching_map(desc::SwitchingCircuitDescription;
                       name::Union{String,Nothing}=nothing)
    sys_name = isnothing(name) ? desc.name : name
    modes    = desc.modes
    n_modes  = length(modes)
    state_dim = desc.state_dim

    n_modes >= 1 ||
        throw(ArgumentError("switching_map: description must contain at least one mode."))

    # Validate duration roles before constructing the map closure.
    for (k, mode) in enumerate(modes)
        if k < n_modes
            mode.duration_fn === nothing &&
                throw(ArgumentError(
                    "switching_map: mode $k is an intermediate mode and must have a " *
                    "duration_fn (got nothing)."))
        elseif mode.duration_fn !== nothing
            throw(ArgumentError(
                "switching_map: final mode $k must have duration_fn = nothing because " *
                "it consumes the remaining clock period."))
        end
    end

    desc_name = desc.name  # capture for error messages inside closure
    f = let modes=modes, desc=desc, desc_name=desc_name, state_dim=state_dim
        function (x::SVector{N}, p) where {N}
            N == state_dim ||
                throw(ArgumentError(
                    "switching_map ($desc_name): state dimension mismatch; expected $state_dim, got $N."))
            T         = _sw_period(desc.period, p)
            _sw_check_period(T, desc_name)
            return _sw_apply_modes(x, T, modes, p, desc_name)
        end
    end

    # Collect switching events from all modes.
    all_events = SwitchingEvent[]
    for mode in modes
        append!(all_events, mode.events)
    end

    DiscreteMap(f, state_dim, desc.param_names, sys_name; switching_events=all_events)
end

function _sw_sum_indices(x, indices)
    total = zero(eltype(x))
    for idx in indices
        total += x[idx]
    end
    return total
end

function _sw_peak_current_residual(x::SVector{N}, A::SMatrix{N,N}, b::SVector{N},
                                   t, current_indices, iref) where {N}
    y = _affine_flow_nd(x, A, b, t)
    return _sw_sum_indices(y, current_indices) - iref
end

function _sw_peak_current_on_time(x::SVector{N}, p, A_fn, b_fn, T;
                                  reference_index::Int=1,
                                  current_indices=(2, 4),
                                  iterations::Int=72) where {N}
    return _sw_peak_current_on_time(x, p, A_fn, b_fn, T, reference_index, current_indices, iterations)
end

function _sw_peak_current_on_time(x::SVector{N}, p, A_fn, b_fn, T,
                                  reference_index::Int,
                                  current_indices,
                                  iterations::Int) where {N}
    A = SMatrix{N,N}(_sw_A(A_fn, p))
    b = SVector{N}(_sw_b(b_fn, p))
    iref = p[reference_index]
    g0 = _sw_sum_indices(x, current_indices) - iref
    _sw_primal(g0) >= 0 && return zero(T)

    gT = _sw_peak_current_residual(x, A, b, T, current_indices, iref)
    _sw_primal(gT) < 0 && return T + one(T)

    lo = 0.0
    hi = Float64(_sw_primal(T))
    for _ in 1:iterations
        mid = (lo + hi) / 2
        if _sw_primal(_sw_peak_current_residual(x, A, b, mid, current_indices, iref)) >= 0
            hi = mid
        else
            lo = mid
        end
    end

    tau0 = hi
    y = _affine_flow_nd(x, A, b, tau0)
    dy = A * y + b
    slope = _sw_sum_indices(dy, current_indices)
    abs(Float64(_sw_primal(slope))) <= 10eps(Float64) && return tau0
    return tau0 - _sw_peak_current_residual(x, A, b, tau0, current_indices, iref) / slope
end

function _sw_apply_final_mode(xc::SVector{N}, remaining, mode::AffineModeSpec, p) where {N}
    A = SMatrix{N,N}(_sw_A(mode.A_fn, p))
    b = SVector{N}(_sw_b(mode.b_fn, p))
    x_flow = _affine_flow_nd(xc, A, b, remaining)
    return mode.boundary_fn === nothing ? x_flow : mode.boundary_fn(x_flow, p)
end

function _sw_apply_intermediate_mode(xc::SVector{N}, remaining, mode::AffineModeSpec, p, desc_name, mode_index::Int) where {N}
    A = SMatrix{N,N}(_sw_A(mode.A_fn, p))
    b = SVector{N}(_sw_b(mode.b_fn, p))
    tau_raw = mode.duration_fn(xc, p)
    _sw_check_raw_duration(tau_raw, mode_index, desc_name)
    tau = clamp(tau_raw, zero(remaining), remaining)
    x_flow = _affine_flow_nd(xc, A, b, tau)
    xc_next = if mode.boundary_fn !== nothing &&
                 tau_raw >= zero(tau_raw) && tau_raw < remaining
        mode.boundary_fn(x_flow, p)
    else
        x_flow
    end
    return xc_next, remaining - tau
end

function _sw_apply_modes(xc::SVector{N}, remaining, modes::Tuple, p, desc_name) where {N}
    return _sw_apply_modes(xc, remaining, modes, p, desc_name, 1)
end

function _sw_apply_modes(xc::SVector{N}, remaining, modes::Tuple{M}, p, desc_name, mode_index::Int) where {N, M}
    return _sw_apply_final_mode(xc, remaining, first(modes), p)
end

function _sw_apply_modes(xc::SVector{N}, remaining, modes::Tuple{M1, M2, Vararg}, p, desc_name, mode_index::Int) where {N, M1, M2}
    xc_next, remaining_next = _sw_apply_intermediate_mode(xc, remaining, first(modes), p, desc_name, mode_index)
    return _sw_apply_modes(xc_next, remaining_next, Base.tail(modes), p, desc_name, mode_index + 1)
end

# ─── Built-in circuit descriptions ───────────────────────────────────────────

"""
    buck_converter_description(; L=2.2e-6, T=1/0.5e6) -> SwitchingCircuitDescription

Return the `SwitchingCircuitDescription` for the peak-current-mode buck
converter with inductor `L` and clock period `T` (defaults: 2.2 µH, 2 µs).
Circuit constants `R = 1.6 Ω` and `C = 39.1 µF` are fixed. Bifurcation
parameters: `[Iref, Ein]`.

Pass the result to `switching_map` to obtain a `DiscreteMap` numerically
identical (to floating-point rounding) to the hand-coded `buck_converter()`:

```julia
sys_gen = switching_map(buck_converter_description())
```

**Circuit structure.**
Both the ON (switch closed, t ∈ [0, tₙ]) and OFF (freewheeling, t ∈ [tₙ, T])
stages share the same underdamped circuit matrix
`A = [[-1/(RC), 1/C], [-1/L, 0]]`. The ON equilibrium is `(Ein, Ein/R)`;
the OFF equilibrium is `(0, 0)`. The switching time
`tₙ = L(Iref − Iₙ)/(Ein − Vₙ)` is clamped to [0, T].

**Boundary condition.**
At the switching instant the comparator trips at exactly `I = Iref`. The ON
mode therefore carries a `boundary_fn` that keeps the voltage computed by the
affine flow but forces the current component to `Iref = p[1]`, matching the
hand-coded map's `k1 = Iref` assignment.
"""
function buck_converter_description(; L::Float64=2.2e-6, T::Float64=1/0.5e6)
    (L > 0 && T > 0) ||
        throw(ArgumentError("buck_converter_description requires positive L and T; got L=$L, T=$T."))
    R = 1.6
    C = 39.1e-6

    # Both modes share the same circuit matrix (same R, L, C).
    # A = [[-1/(RC), 1/C], [-1/L, 0]]  (underdamped at default parameters)
    # SMatrix column-major: SMatrix{2,2}(M[1,1], M[2,1], M[1,2], M[2,2])
    A = SMatrix{2,2}(-1/(R*C), -1/L, 1/C, 0.0)

    # ON mode: switch closed; equilibrium at (Ein, Ein/R).
    # b_on depends on the bifurcation parameter Ein = p[2].
    b_on = p -> SVector(zero(p[2]), p[2] / L)

    # Raw switching time; clamped inside the generator to [0, remaining].
    t_on = (x, p) -> begin
        denom = p[2] - x[1]        # Ein − Vn
        denom == 0 && return Inf
        L * (p[1] - x[2]) / denom  # L*(Iref − In)/(Ein − Vn)
    end

    # At the switching instant the comparator forces I = Iref exactly:
    # keep V from the affine flow, override I with p[1].
    buck_on_boundary = (x_flow, p) -> SVector(x_flow[1], p[1])

    events_on = [
        SwitchingEvent(
            "switch-time-period-border",
            (x, p) -> t_on(x, p) - T;
            description="Switching time tn reaches the clock period T; the map changes between switched and unswitched cycles.",
            tolerance=1e-9,
            scale=T
        )
    ]

    # OFF mode: switch open, freewheeling; equilibrium at (0, 0).
    b_off = SVector(0.0, 0.0)

    mode_on  = AffineModeSpec(A, b_on;  duration=t_on, boundary=buck_on_boundary, events=events_on)
    mode_off = AffineModeSpec(A, b_off)   # final mode: no duration_fn

    SwitchingCircuitDescription(
        (mode_on, mode_off), T;
        param_names=[:Iref, :Ein],
        name="Buck Converter",
        state_dim=2
    )
end

"""
    boost_converter_description(; L=1e-3, C=12e-6, T=100e-6) -> SwitchingCircuitDescription

Return the `SwitchingCircuitDescription` for the peak-current-mode boost
converter with inductor `L`, output capacitor `C`, and clock period `T`
(defaults: 1 mH, 12 µF, 100 µs). Bifurcation parameters:
`[Iref, E, R, Sc]` (with shortened-vector fallbacks `E=10`, `R=20`, `Sc=0`
matching `boost_converter()`).

Pass the result to `switching_map` to obtain a `DiscreteMap` matching the
hand-coded `boost_converter()`:

```julia
sys_gen = switching_map(boost_converter_description())
```

**Circuit structure.**
ON stage (`A` singular: `det = 0`, one zero eigenvalue) — decoupled V decay
+ I ramp; equilibrium does not exist, so the Duhamel integral form is used.
OFF stage — standard underdamped LC matrix with equilibrium `(E, E/R)`.
The on-time `t_on = clamp((Iref − I)/(E/L + Sc), 0, T)` handles both
saturation rails.
"""
function boost_converter_description(; L::Float64=1e-3, C::Float64=12e-6, T::Float64=100e-6)
    (L > 0 && C > 0 && T > 0) ||
        throw(ArgumentError("boost_converter_description requires positive L, C, T; got L=$L, C=$C, T=$T."))

    # ON mode: dV/dt = −V/(RC),  dI/dt = E/L
    # A_on is singular (det = 0): one zero eigenvalue, one eigenvalue −1/(RC).
    # A and b depend on R (p[3]) and E (p[2]).
    A_on = p -> begin
        R = length(p) >= 3 ? p[3] : 20.0
        SMatrix{2,2}(-one(R)/(R*C), zero(R), zero(R), zero(R))
    end
    b_on = p -> begin
        E = length(p) >= 2 ? p[2] : 10.0
        SVector(zero(E), E / L)
    end

    # Switch-on duration: clamp((Iref − I) / (E/L + Sc), 0, T)
    t_on = (x, p) -> begin
        E  = length(p) >= 2 ? p[2] : 10.0
        Sc = length(p) >= 4 ? p[4] : 0.0
        denom = E / L + Sc
        denom == 0 && return Inf
        (p[1] - x[2]) / denom
    end

    events_on = [
        SwitchingEvent(
            "on-time-lower-border",
            t_on;
            description="Unclamped switch-on duration reaches zero; the cycle starts at or above the current reference.",
            tolerance=1e-9,
            scale=T
        ),
        SwitchingEvent(
            "on-time-upper-border",
            (x, p) -> T - t_on(x, p);
            description="Unclamped switch-on duration reaches the full clock period; the comparator never trips in the cycle.",
            tolerance=1e-9,
            scale=T
        )
    ]

    # OFF mode: dV/dt = I/C − V/(RC),  dI/dt = (E − V)/L
    # A_off is non-singular; equilibrium at (E, E/R).
    A_off = p -> begin
        R = length(p) >= 3 ? p[3] : 20.0
        SMatrix{2,2}(-one(R)/(R*C), -one(R)/L, one(R)/C, zero(R))
    end
    b_off = p -> begin
        E = length(p) >= 2 ? p[2] : 10.0
        SVector(zero(E), E / L)
    end

    mode_on  = AffineModeSpec(A_on, b_on;   duration=t_on, events=events_on)
    mode_off = AffineModeSpec(A_off, b_off)  # final mode

    SwitchingCircuitDescription(
        (mode_on, mode_off), T;
        param_names=[:Iref, :E, :R, :Sc],
        name="Boost (peak-current)",
        state_dim=2
    )
end

function _sw_converter_param(p, idx::Int, default)
    return length(p) >= idx ? p[idx] : default
end

"""
    cuk_converter_description(; kwargs...) -> SwitchingCircuitDescription

Return the current-programmed Cuk converter description from Debbat,
El Aroudi, and Bouyadjra (2012). State ordering is
`[vC2, iL2, vC1, iL1]`; parameters are `[Iref, Vin, R]`, with `Vin`
and `R` falling back to the cited operating point. The switch starts ON
each period and opens when `iL1 + iL2 = Iref`.
"""
function cuk_converter_description(; Vin::Float64=15.0, L1::Float64=75e-3, L2::Float64=75e-3,
                                   rL1::Float64=0.02, rL2::Float64=0.02,
                                   C1::Float64=47e-6, C2::Float64=47e-6,
                                   R::Float64=10.0, T::Float64=50e-6)
    (Vin > 0 && L1 > 0 && L2 > 0 && C1 > 0 && C2 > 0 && R > 0 && T > 0) ||
        throw(ArgumentError("cuk_converter_description requires positive Vin, L1, L2, C1, C2, R, and T."))
    (rL1 >= 0 && rL2 >= 0) ||
        throw(ArgumentError("cuk_converter_description requires non-negative inductor resistances."))

    A_on = p -> begin
        Rp = _sw_converter_param(p, 3, R)
        z = zero(Rp); o = one(Rp)
        SMatrix{4,4}(
            -o/(Rp*C2), -o/L2, z, z,
             o/C2,      -rL2/L2, -o/C1, z,
             z,          o/L2, z, z,
             z,          z, z, -rL1/L1)
    end
    A_off = p -> begin
        Rp = _sw_converter_param(p, 3, R)
        z = zero(Rp); o = one(Rp)
        SMatrix{4,4}(
            -o/(Rp*C2), -o/L2, z, z,
             o/C2,      -rL2/L2, z, z,
             z,          z, z, -o/L1,
             z,          z, o/C1, -rL1/L1)
    end
    b = p -> begin
        Vinp = _sw_converter_param(p, 2, Vin)
        SVector(zero(Vinp), zero(Vinp), zero(Vinp), Vinp / L1)
    end
    t_on = (x, p) -> _sw_peak_current_on_time(x, p, A_on, b, T, 1, (2, 4), 72)

    events_on = [
        SwitchingEvent(
            "sum-current-lower-border",
            (x, p) -> x[2] + x[4] - p[1];
            description="The clock-edge sum current is already at or above the reference, so the ON interval collapses.",
            tolerance=1e-8,
            scale=1.0
        ),
        SwitchingEvent(
            "sum-current-upper-border",
            (x, p) -> p[1] - (begin
                A = SMatrix{4,4}(_sw_A(A_on, p))
                bv = SVector{4}(_sw_b(b, p))
                y = _affine_flow_nd(SVector{4}(x), A, bv, T)
                y[2] + y[4]
            end);
            description="The sum current reaches the reference at the end of the clock period; beyond this border the cycle is all-ON.",
            tolerance=1e-8,
            scale=1.0
        )
    ]

    mode_on = AffineModeSpec(A_on, b; duration=t_on, events=events_on)
    mode_off = AffineModeSpec(A_off, b)
    SwitchingCircuitDescription(
        (mode_on, mode_off), T;
        param_names=[:Iref, :Vin, :R],
        name="Cuk (peak-current)",
        state_dim=4
    )
end

"""
    sepic_converter_description(; kwargs...) -> SwitchingCircuitDescription

Return the current-programmed SEPIC converter description from Debbat,
El Aroudi, Giral, and Martinez-Salamero (2002). State ordering is
`[vC2, iL2, vC1, iL1]`; parameters are `[Iref, Vin, R]`, with `Vin`
and `R` falling back to the cited operating point. The switch starts ON
each period and opens when `iL1 + iL2 = Iref`.
"""
function sepic_converter_description(; Vin::Float64=24.0, L1::Float64=17.8e-6, L2::Float64=37e-6,
                                     rL1::Float64=0.13, rL2::Float64=0.019,
                                     C1::Float64=0.4e-6, C2::Float64=47e-6,
                                     R::Float64=10.0, T::Float64=2e-6)
    (Vin > 0 && L1 > 0 && L2 > 0 && C1 > 0 && C2 > 0 && R > 0 && T > 0) ||
        throw(ArgumentError("sepic_converter_description requires positive Vin, L1, L2, C1, C2, R, and T."))
    (rL1 >= 0 && rL2 >= 0) ||
        throw(ArgumentError("sepic_converter_description requires non-negative inductor resistances."))

    A_on = p -> begin
        Rp = _sw_converter_param(p, 3, R)
        z = zero(Rp); o = one(Rp)
        SMatrix{4,4}(
            -o/(Rp*C2), z, z, z,
             z,        -rL2/L2, -o/C1, z,
             z,         o/L2, z, z,
             z,         z, z, -rL1/L1)
    end
    A_off = p -> begin
        Rp = _sw_converter_param(p, 3, R)
        z = zero(Rp); o = one(Rp)
        SMatrix{4,4}(
            -o/(Rp*C2), -o/L2, z, -o/L1,
             o/C2,      -rL2/L2, z, z,
             z,          z, z, -o/L1,
             o/C2,       z, o/C1, -rL1/L1)
    end
    b = p -> begin
        Vinp = _sw_converter_param(p, 2, Vin)
        SVector(zero(Vinp), zero(Vinp), zero(Vinp), Vinp / L1)
    end
    t_on = (x, p) -> _sw_peak_current_on_time(x, p, A_on, b, T, 1, (2, 4), 72)

    events_on = [
        SwitchingEvent(
            "sum-current-lower-border",
            (x, p) -> x[2] + x[4] - p[1];
            description="The clock-edge sum current is already at or above the reference, so the ON interval collapses.",
            tolerance=1e-8,
            scale=1.0
        ),
        SwitchingEvent(
            "sum-current-upper-border",
            (x, p) -> p[1] - (begin
                A = SMatrix{4,4}(_sw_A(A_on, p))
                bv = SVector{4}(_sw_b(b, p))
                y = _affine_flow_nd(SVector{4}(x), A, bv, T)
                y[2] + y[4]
            end);
            description="The sum current reaches the reference at the end of the clock period; beyond this border the cycle is all-ON.",
            tolerance=1e-8,
            scale=1.0
        )
    ]

    mode_on = AffineModeSpec(A_on, b; duration=t_on, events=events_on)
    mode_off = AffineModeSpec(A_off, b)
    SwitchingCircuitDescription(
        (mode_on, mode_off), T;
        param_names=[:Iref, :Vin, :R],
        name="SEPIC (peak-current)",
        state_dim=4
    )
end

"""
    cuk_converter(; kwargs...) -> DiscreteMap

Create the generated 4-D current-programmed Cuk converter map.
"""
function cuk_converter(; kwargs...)
    switching_map(cuk_converter_description(; kwargs...))
end

"""
    sepic_converter(; kwargs...) -> DiscreteMap

Create the generated 4-D current-programmed SEPIC converter map.
"""
function sepic_converter(; kwargs...)
    switching_map(sepic_converter_description(; kwargs...))
end
