"""
Switching map generator: construct a `DiscreteMap` from a piecewise-linear
circuit description (per-mode affine state-space data + algebraic switching timing).

Scope: 2-D systems (state = [V, I] or similar). Sufficient for the DC–DC
converter family (buck, boost, …). The primitive `_affine_flow_2d` handles
under-, over-, and critically-damped 2-D circuit matrices uniformly, as well
as singular `A` matrices (boost ON stage).
"""

# ─── Descriptor types ─────────────────────────────────────────────────────────

"""
    AffineModeSpec{A,B,D,G}

Describes one operating mode of a 2-D piecewise-linear switching circuit.

Type parameters encode the concrete types of the state-matrix, affine-input,
duration, and optional boundary-state suppliers — no field is annotated `Function`:
- `A`: `SMatrix{2,2}` constant **or** a callable `p -> SMatrix{2,2}`.
- `B`: `SVector{2}` constant **or** a callable `p -> SVector{2}`.
- `D`: callable `(x, p) -> Real` for intermediate modes, `Nothing` for
  the final mode (which consumes the remaining clock period).
- `G`: `Nothing` (no override) **or** a callable `(x_flow, p) -> SVector{2}`
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

Ordered list of operating modes making up one clock period of a 2-D
piecewise-linear switching circuit, together with the clock period and
parameter metadata needed to construct a `DiscreteMap` via `switching_map`.

Type parameter `T` is the type of `period` — a `Float64` constant or a
callable `p -> Real` for parameter-dependent periods.

# Fields
- `modes::Vector{AffineModeSpec}`: ordered mode descriptions, one per operating mode.
- `period::T`: clock period constant or callable.
- `param_names::Vector{Symbol}`: bifurcation-parameter names forwarded to the
  generated `DiscreteMap`.
- `name::String`: human-readable circuit name.
"""
struct SwitchingCircuitDescription{T}
    modes::Vector{AffineModeSpec}
    period::T
    param_names::Vector{Symbol}
    name::String
end

"""
    SwitchingCircuitDescription(modes, period; param_names=Symbol[], name="Switching Circuit")

Construct a `SwitchingCircuitDescription` from an ordered collection of
`AffineModeSpec` values and a clock period (constant `Float64` or callable).
"""
function SwitchingCircuitDescription(modes, period;
                                     param_names::AbstractVector{Symbol}=Symbol[],
                                     name::AbstractString="Switching Circuit")
    typed_modes = AffineModeSpec[]
    for (index, mode) in enumerate(modes)
        mode isa AffineModeSpec ||
            throw(ArgumentError(
                "SwitchingCircuitDescription: mode $index is not an AffineModeSpec."))
        push!(typed_modes, mode)
    end
    SwitchingCircuitDescription{typeof(period)}(
        typed_modes, period,
        collect(Symbol, param_names), String(name))
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

# ─── 2-D affine flow ─────────────────────────────────────────────────────────

"""
    _affine_flow_2d(x, A, b, tau)

Exact solution `x(tau)` of the 2-D affine ODE `dx/dt = A x + b` with
initial condition `x(0) = x`, integrated for duration `tau ≥ 0`.

**Matrix exponential.** Uses the psi0/psi1 decomposition valid for all
damping regimes:

    exp(A τ) = e^{aτ} · (ψ₀·I  +  ψ₁·(A − aI))

where `a = tr(A)/2` is the half-trace and the discriminant
`disc = a² − det(A)` selects the regime:
- `disc < 0` (under-damped / complex eigenvalues): `ψ₀ = cos(ωτ)`, `ψ₁ = sin(ωτ)/ω` with `ω = √(−disc)`.
- `disc > 0` (over-damped / real distinct eigenvalues): `ψ₀ = cosh(sτ)`, `ψ₁ = sinh(sτ)/s` with `s = √disc`.
- `disc ≈ 0` (critically damped / repeated eigenvalue): `ψ₀ = 1`, `ψ₁ = τ`.

**Non-singular A.** Uses the equilibrium form
`x(τ) = exp(Aτ)·(x − x_eq) + x_eq` where `x_eq = −A⁻¹ b`.

**Singular A (one or two zero eigenvalues, e.g. boost ON stage).** Uses the
Duhamel integral directly:

    x(τ) = exp(Aτ)·x  +  (τ·I + coeff·A)·b

where `coeff = (exp(λ₂ τ) − 1 − λ₂ τ) / λ₂²` and `λ₂ = tr A` is the
second eigenvalue. Its `λ₂ → 0` limit is evaluated by a local series, so
nilpotent matrices are handled without division by zero. This is exact and
avoids any division by `det A`.

**ForwardDiff compatibility.** All damping-regime and singularity branches
are evaluated via standard comparison operators, which `ForwardDiff.Dual`
types reduce to primal-value comparisons. The function is therefore
ForwardDiff-compatible away from the switching borders and the
discriminant boundary.
"""
function _sw_exprel2(z)
    if abs(z) < 1e-5
        return one(z) / 2 + z / 6 + z^2 / 24 + z^3 / 120 + z^4 / 720
    end
    return (expm1(z) - z) / (z * z)
end

function _affine_flow_2d(x::SVector{2}, A::SMatrix{2,2}, b::SVector{2}, tau)
    a    = (A[1,1] + A[2,2]) / 2
    d    = A[1,1]*A[2,2] - A[1,2]*A[2,1]
    disc = a*a - d
    # Reference scale for relative thresholds: squared Frobenius norm + 1.
    norm_ref = A[1,1]^2 + A[1,2]^2 + A[2,1]^2 + A[2,2]^2 + one(d)
    e = exp(a * tau)

    # Damping-regime branches on primal disc (safe for ForwardDiff.Dual).
    if disc < -1e-10 * norm_ref          # under-damped
        omega = sqrt(-disc)
        psi0  = cos(omega * tau)
        psi1  = sin(omega * tau) / omega
    elseif disc > 1e-10 * norm_ref       # over-damped (includes singular A: one zero eigenvalue)
        s    = sqrt(disc)
        psi0 = cosh(s * tau)
        psi1 = sinh(s * tau) / s
    else                                  # critically damped
        psi0 = one(tau)
        psi1 = tau
    end

    # Shifted circuit matrix A − aI (shared by both flow branches).
    Ash11 = A[1,1] - a;  Ash12 = A[1,2]
    Ash21 = A[2,1];      Ash22 = A[2,2] - a

    if abs(d) > 1e-10 * norm_ref
        # ── Non-singular A: equilibrium form ─────────────────────────────────
        # x_eq = −A⁻¹ b;  2×2 inverse: A⁻¹ = [[A22,−A12],[−A21,A11]] / det
        xeq1 = -(A[2,2]*b[1] - A[1,2]*b[2]) / d
        xeq2 = -(-A[2,1]*b[1] + A[1,1]*b[2]) / d
        w1   = x[1] - xeq1;  w2 = x[2] - xeq2
        Aw1  = Ash11*w1 + Ash12*w2
        Aw2  = Ash21*w1 + Ash22*w2
        return e * SVector(psi0*w1 + psi1*Aw1, psi0*w2 + psi1*Aw2) +
               SVector(xeq1, xeq2)
    else
        # ── Singular A (det ≈ 0, one zero eigenvalue): Duhamel form ──────────
        # exp(Aτ)·x
        Ax1 = Ash11*x[1] + Ash12*x[2]
        Ax2 = Ash21*x[1] + Ash22*x[2]
        ex  = e * SVector(psi0*x[1] + psi1*Ax1, psi0*x[2] + psi1*Ax2)
        # Integral: (τ·I + coeff·A)·b
        # λ₂ = tr A = 2a. The exprel₂ form stays finite for λ₂ = 0.
        lam2  = A[1,1] + A[2,2]
        coeff = tau * tau * _sw_exprel2(lam2 * tau)
        Ab1   = A[1,1]*b[1] + A[1,2]*b[2]
        Ab2   = A[2,1]*b[1] + A[2,2]*b[2]
        return ex + tau*b + SVector(coeff*Ab1, coeff*Ab2)
    end
end

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

Construct a 2-D `DiscreteMap` from a `SwitchingCircuitDescription`.

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
    f = let modes=modes, desc=desc, n_modes=n_modes, desc_name=desc_name
        function (x::SVector{2}, p)
            T         = _sw_period(desc.period, p)
            _sw_check_period(T, desc_name)
            xc        = x
            remaining = T
            for k in 1:n_modes
                mode = modes[k]
                A    = SMatrix{2,2}(_sw_A(mode.A_fn, p))
                b    = SVector{2}(_sw_b(mode.b_fn, p))
                if k < n_modes
                    tau_raw = _sw_duration(mode.duration_fn, xc, p)
                    _sw_check_raw_duration(tau_raw, k, desc_name)
                    tau     = clamp(tau_raw, zero(remaining), remaining)
                    x_flow  = _affine_flow_2d(xc, A, b, tau)
                    # Apply boundary_fn only when the switch actually fired within
                    # the remaining period (tau_raw < remaining, matching the
                    # `tn < T` branch in the hand-coded buck). When clamped to the
                    # full remaining period the event never triggered.
                    if mode.boundary_fn !== nothing &&
                       tau_raw >= zero(tau_raw) && tau_raw < remaining
                        xc = mode.boundary_fn(x_flow, p)
                    else
                        xc = x_flow
                    end
                    remaining = remaining - tau
                else
                    x_flow = _affine_flow_2d(xc, A, b, remaining)
                    xc     = mode.boundary_fn === nothing ? x_flow :
                             mode.boundary_fn(x_flow, p)
                end
            end
            return xc
        end
    end

    # Collect switching events from all modes.
    all_events = SwitchingEvent[]
    for mode in modes
        append!(all_events, mode.events)
    end

    DiscreteMap(f, 2, desc.param_names, sys_name; switching_events=all_events)
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
        name="Buck Converter"
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
        name="Boost (peak-current)"
    )
end
