"""
Colpitts oscillator models used by the package's parameter-study examples.

All three variants share the same state ordering:

    u = [V_C1, V_C2, I_L]

and use the Poincaré section `I_L = 0` with upward crossings for the
brute-force and continuation diagrams.
"""

"""Shared Poincaré section for the Colpitts models (`I_L = 0`, upward crossing)."""
function _colpitts_section()
    PoincareSection(
        (u, t, integrator) -> u[3];
        direction = :up,
        projection = [1, 2],
        template = [0.0, 0.0, 0.0]
    )
end

"""Numerically safe exponential collector-current law for the BJT model."""
function _colpitts_collector_current(vbe::Float64, IS::Float64, VT::Float64)
    return max(IS * expm1(clamp(vbe / VT, -50.0, 50.0)), 0.0)
end

"""Dynamic β approximation fitted from the 2N3904 datasheet curve."""
function _colpitts_dynamic_beta(ic::Float64, a::Float64, b::Float64, c::Float64)
    scaled = ic / b
    return a * scaled / (1 + scaled + (ic / c)^2)
end

"""Shared Colpitts state equations once the transistor base/collector currents are known."""
function _colpitts_core!(du, u, C1, C2, R1, RL, V1, V2, L1, V3, ib, ic)
    vc1, vc2, iL = u
    du[1] = (iL - ic) / C1
    du[2] = ((-V1 + V3 - vc2) / R1 + iL + ib) / C2
    du[3] = (V2 - vc1 + V3 - vc2 - iL * RL) / L1
    nothing
end

"""
    colpitts_simple_oscillator(; kwargs...) -> ContinuousODE

Piecewise-linear Colpitts oscillator model using a threshold BJT approximation.

Defaults mirror one of the reference C-sweep / β=265 diagram configurations:
`C1=C2=40 nF`, `β=265`, `V1=V2=5 V`, `R1=425 Ω`, `RL=33 Ω`, `L1=80 µH`,
`RON=400 Ω`, `VTH=0.75 V`, `V3=0 V`.

Parameter vector order: `[C1, C2, beta, V1, V2]`.
"""
function colpitts_simple_oscillator(;
    C1::Float64 = 40e-9,
    C2::Float64 = 40e-9,
    beta::Float64 = 265.0,
    R1::Float64 = 425.0,
    RL::Float64 = 33.0,
    V1::Float64 = 5.0,
    V2::Float64 = 5.0,
    L1::Float64 = 80e-6,
    RON::Float64 = 400.0,
    VTH::Float64 = 0.75,
    V3::Float64 = 0.0,
    tspan_hint::Float64 = 0.02
)
    f! = function(du, u, p, t)
        C1p = p[1]
        C2p = p[2]
        betap = max(p[3], eps(Float64))
        V1p = p[4]
        V2p = p[5]

        vbe = V3 - u[2]
        ib = vbe <= VTH ? 0.0 : (vbe - VTH) / RON
        ic = betap * ib
        _colpitts_core!(du, u, C1p, C2p, R1, RL, V1p, V2p, L1, V3, ib, ic)
    end

    ContinuousODE(
        f!, 3, _colpitts_section(), [:C1, :C2, :beta, :V1, :V2], "Colpitts (simple)";
        tspan_hint = tspan_hint,
        default_initial_state = [0.0, 0.0, 0.0],
        default_params = [C1, C2, beta, V1, V2]
    )
end

"""
    colpitts_exponential_oscillator(; kwargs...) -> ContinuousODE

Exponential Colpitts oscillator model using an exponential BJT current law.

Defaults mirror one of the reference C-sweep / β=265 diagram configurations:
`C1=C2=40 nF`, `β=265`, `V1=V2=5 V`, `R1=425 Ω`, `RL=33 Ω`, `L1=80 µH`,
`IS=1e-15 A`, `VT=26 mV`, `V3=0 V`.

Parameter vector order: `[C1, C2, beta, V1, V2]`.
"""
function colpitts_exponential_oscillator(;
    C1::Float64 = 40e-9,
    C2::Float64 = 40e-9,
    beta::Float64 = 265.0,
    R1::Float64 = 425.0,
    RL::Float64 = 33.0,
    V1::Float64 = 5.0,
    V2::Float64 = 5.0,
    L1::Float64 = 80e-6,
    IS::Float64 = 1e-15,
    VT::Float64 = 26e-3,
    V3::Float64 = 0.0,
    tspan_hint::Float64 = 0.02
)
    f! = function(du, u, p, t)
        C1p = p[1]
        C2p = p[2]
        betap = max(p[3], eps(Float64))
        V1p = p[4]
        V2p = p[5]

        vbe = V3 - u[2]
        ic = _colpitts_collector_current(vbe, IS, VT)
        ib = ic / betap
        _colpitts_core!(du, u, C1p, C2p, R1, RL, V1p, V2p, L1, V3, ib, ic)
    end

    ContinuousODE(
        f!, 3, _colpitts_section(), [:C1, :C2, :beta, :V1, :V2], "Colpitts (exponential)";
        tspan_hint = tspan_hint,
        default_initial_state = [0.0, 0.0, 0.0],
        default_params = [C1, C2, beta, V1, V2]
    )
end

"""
    colpitts_dynamic_beta_oscillator(; kwargs...) -> ContinuousODE

Exponential Colpitts oscillator model with a dynamic current-dependent transistor gain
fitted from the 2N3904 collector-current curve.

Defaults mirror one of the reference dynamic-β C-sweep configurations:
`C1=C2=40 nF`, `V1=V2=5 V`, `R1=425 Ω`, `RL=33 Ω`, `L1=80 µH`,
`IS=1e-15 A`, `VT=26 mV`, `V3=0 V`, and fitted β-law coefficients
`a=328.82`, `b=0.00025`, `c=0.0034`.

Parameter vector order: `[C1, C2, V1, V2]`.
"""
function colpitts_dynamic_beta_oscillator(;
    C1::Float64 = 40e-9,
    C2::Float64 = 40e-9,
    R1::Float64 = 425.0,
    RL::Float64 = 33.0,
    V1::Float64 = 5.0,
    V2::Float64 = 5.0,
    L1::Float64 = 80e-6,
    IS::Float64 = 1e-15,
    VT::Float64 = 26e-3,
    V3::Float64 = 0.0,
    beta_a::Float64 = 328.82,
    beta_b::Float64 = 0.00025,
    beta_c::Float64 = 0.0034,
    beta_floor::Float64 = 1e-3,
    tspan_hint::Float64 = 0.02
)
    f! = function(du, u, p, t)
        C1p = p[1]
        C2p = p[2]
        V1p = p[3]
        V2p = p[4]

        vbe = V3 - u[2]
        ic = _colpitts_collector_current(vbe, IS, VT)
        beta_dyn = ic <= 0 ? 0.0 : max(beta_floor, _colpitts_dynamic_beta(ic, beta_a, beta_b, beta_c))
        ib = ic <= 0 ? 0.0 : ic / beta_dyn
        _colpitts_core!(du, u, C1p, C2p, R1, RL, V1p, V2p, L1, V3, ib, ic)
    end

    ContinuousODE(
        f!, 3, _colpitts_section(), [:C1, :C2, :V1, :V2], "Colpitts (dynamic beta)";
        tspan_hint = tspan_hint,
        default_initial_state = [0.0, 0.0, 0.0],
        default_params = [C1, C2, V1, V2]
    )
end
