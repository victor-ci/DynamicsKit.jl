"""
Peak-current-mode (current-programmed) PWM boost converter: a 2D discrete-time
stroboscopic map sampled once per switching period.

Circuit: input `E` drives an inductor `L`; a switch chops the inductor current
and an output capacitor `C` feeds a load `R`. A current-mode controller turns
the switch ON at each clock edge and OFF when the inductor current reaches the
(optionally slope-compensated) reference `Iref`. Because the inductor current
ramps linearly during the ON stage, the switch-off instant is algebraic — no
per-iteration root find — so the stroboscopic map `x_{n+1} = F(x_n)` is
genuinely closed-form (and ForwardDiff-friendly away from the saturation
borders).

    State x = [V, I] = [output voltage, inductor current]
    ON  (switch closed, diode off): dV/dt = -V/(RC),       dI/dt = E/L
    OFF (switch open, diode on):     dV/dt = I/C - V/(RC),  dI/dt = (E - V)/L

The ON stage is *decoupled* (voltage decays exponentially while the current
ramps at `E/L`), so it has an elementary closed form. The OFF stage shares the
underdamped LC system matrix `A = [[-1/(RC), 1/C], [-1/L, 0]]` (equilibrium
`(E, E/R)`) and uses an exact `exp(A·τ)` solution.

Switching instant within a clock period of length `T`: the comparator trips
when `I(t)` meets the compensated reference `Iref - Sc·t`. With `I` ramping at
`E/L`, the ON duration is

    t_on = clamp( (Iref - I_n) / (E/L + Sc), 0, T )

(`t_on = T` means the current never reaches the reference and the switch stays
closed for the whole period; `t_on = 0` means it is already at/above the
reference at the clock edge).

Without slope compensation (`Sc = 0`) the period-1 orbit loses stability via the
classic current-mode subharmonic (period-doubling) instability once the duty
ratio exceeds 1/2, producing a period-doubling cascade to chaos as `Iref` rises.
The model assumes continuous conduction (the diode is not modelled, so the OFF
stage may carry `I` negative for very small `Iref`); the documented bifurcation
sweeps stay in continuous-conduction mode.

References:
  C. K. Tse, M. di Bernardo, "Complex behavior in switching power converters",
  Proc. IEEE 90(5):768-781 (2002). doi:10.1109/JPROC.2002.1015011
  A. El Aroudi, L. Benadero, E. Toribio, G. Olivar, "Hopf bifurcation and chaos
  from torus breakdown in a PWM voltage-controlled DC-DC boost converter", IEEE
  Trans. Circuits Syst. I 46(11):1374-1382 (1999). doi:10.1109/81.802837
  W. C. Y. Chan, C. K. Tse, "Study of bifurcations in current-programmed DC/DC
  boost converters", IEEE Trans. Circuits Syst. I 44(12):1129-1142 (1997).
  doi:10.1109/81.645151
"""

"""Exact `exp(A*tau)*(x - x_eq) + x_eq` for the boost OFF-stage LC system matrix
`A = [[-1/(RC), 1/C], [-1/L, 0]]` (equilibrium `(veq, ieq) = (E, E/R)`), valid
for under-, over-, and critically damped regimes. psi0 ("cosine-like") and psi1
("sinc-like": sin/omega, sinh/s, or tau at critical) are smooth as their
argument -> 0; a small relative band around the critical discriminant routes
the near-critical case to the tau series rather than exact equality."""
function _boost_off_flow(v, i, tau, veq, ieq, R, L, C)
    w1 = v - veq
    w2 = i - ieq
    a = -1 / (2 * R * C)
    Aw1 = -w1 / (R * C) + w2 / C
    Aw2 = -w1 / L
    omega_sq = 1 / (L * C)
    disc = a * a - omega_sq          # >0 overdamped, <0 underdamped, ~=0 critical
    e = exp(a * tau)
    if disc < -1e-10 * omega_sq
        omega = sqrt(-disc)
        psi0 = cos(omega * tau)
        psi1 = sin(omega * tau) / omega
    elseif disc > 1e-10 * omega_sq
        s = sqrt(disc)
        psi0 = cosh(s * tau)
        psi1 = sinh(s * tau) / s
    else
        psi0 = one(tau)
        psi1 = tau
    end
    nv = e * (psi0 * w1 + psi1 * (Aw1 - a * w1)) + veq
    ni = e * (psi0 * w2 + psi1 * (Aw2 - a * w2)) + ieq
    return nv, ni
end

function _boost_pcm_rule(x, p; L=1e-3, C=12e-6, T=100e-6)
    V, I = x[1], x[2]
    Iref = p[1]
    E    = length(p) >= 2 ? p[2] : 10.0
    R    = length(p) >= 3 ? p[3] : 20.0
    Sc   = length(p) >= 4 ? p[4] : 0.0

    # Peak-current comparator: switch opens when I meets the compensated
    # reference Iref - Sc*t. During ON, I ramps at E/L, so the crossing is
    # algebraic. clamp handles both rails: t_on = T (never reaches ref) and
    # t_on = 0 (already at/above ref at the clock edge).
    t_on = clamp((Iref - I) / (E / L + Sc), 0.0, T)

    # ON sub-interval: decoupled — V decays, I ramps.
    V1 = V * exp(-t_on / (R * C))
    I1 = I + (E / L) * t_on

    # OFF sub-interval: coupled LC dynamics toward equilibrium (E, E/R).
    V2, I2 = _boost_off_flow(V1, I1, T - t_on, E, E / R, R, L, C)
    return SVector(V2, I2)
end

function _boost_raw_on_time(x, p; L=1e-3)
    I = x[2]
    Iref = p[1]
    E = length(p) >= 2 ? p[2] : 10.0
    Sc = length(p) >= 4 ? p[4] : 0.0
    denom = E / L + Sc
    denom == 0 && return Inf
    return (Iref - I) / denom
end

"""
    boost_converter(; L=1e-3, C=12e-6, T=100e-6) -> DiscreteMap

Create a peak-current-mode PWM boost converter map. Bifurcation parameters are
`[Iref, E, R, Sc]` (peak-current reference, input voltage, load resistance,
slope-compensation rate). The constructor fixes the inductor `L = 1 mH`, output
capacitor `C = 12 µF`, and switching period `T = 100 µs` (10 kHz clock).

When a caller passes a shortened parameter vector the trailing entries fall back
to the documented operating point — `E = 10 V`, `R = 20 Ω`, `Sc = 0` (no slope
compensation) — so a short `p` lands in the same regime the presets and UI
describe. Sweeping `Iref` upward traverses the current-mode subharmonic
instability (period-1 → period-2 once the duty ratio exceeds 1/2, near
`Iref ≈ 1.75 A` at the default `E = 10 V`, `R = 20 Ω`) into a period-doubling
cascade and chaos for `Iref ≳ 2.7 A`, with a period-3 window near
`Iref ≈ 4.8 A`. These thresholds shift with the operating point (`E`, `R`).
"""
function boost_converter(; L::Float64=1e-3, C::Float64=12e-6, T::Float64=100e-6)
    (L > 0 && C > 0 && T > 0) ||
        throw(ArgumentError("boost_converter requires positive circuit constants: L=$L, C=$C, T=$T."))
    f = (x, p) -> _boost_pcm_rule(x, p; L=L, C=C, T=T)
    events = [
        SwitchingEvent(
            "on-time-lower-border",
            (x, p) -> _boost_raw_on_time(x, p; L=L);
            description="Unclamped switch-on duration reaches zero; the cycle starts at or above the current reference.",
            tolerance=1e-9,
            scale=T
        ),
        SwitchingEvent(
            "on-time-upper-border",
            (x, p) -> T - _boost_raw_on_time(x, p; L=L);
            description="Unclamped switch-on duration reaches the full clock period; the comparator never trips in the cycle.",
            tolerance=1e-9,
            scale=T
        )
    ]
    DiscreteMap(f, 2, [:Iref, :E, :R, :Sc], "Boost (peak-current)"; switching_events=events)
end
