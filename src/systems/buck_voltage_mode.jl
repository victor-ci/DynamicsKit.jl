"""
Voltage-mode (regular / uniformly sampled) PWM buck converter: a 2D discrete
stroboscopic map sampled once per switching period.

Circuit: input `E` feeds an LC output filter (`L`, `C`) across a load `R`. A
voltage-mode PWM controller compares the amplified output error
`gain·(Vref - v)` against a sawtooth ramp spanning `[VL, VU]` over each
switching period `T`. Under *regular sampling* the duty cycle for cycle `n` is
fixed by the sampled output voltage `v_n` at the start of the period:

    d_n = clamp( (gain·(Vref - v_n) - VL) / (VU - VL), 0, 1 )

so the stroboscopic map `x_{n+1} = F(x_n)` is genuinely closed-form — no
per-iteration root find. Each clock period is an ON sub-interval (input
connected, length `d_n·T`) followed by an OFF sub-interval (freewheeling,
length `(1 - d_n)·T`); both linear stages have exact `exp(A·τ)` solutions.

    State x = [v, i] = [output voltage, inductor current]
    ON  (switch closed): dv/dt = i/C - v/(RC),  di/dt = (E - v)/L
    OFF (switch open):   dv/dt = i/C - v/(RC),  di/dt =    - v/L

Both stages share the underdamped LC system matrix
`A = [[-1/(RC), 1/C], [-1/L, 0]]`; only the `di/dt` forcing differs (`E/L` vs
`0`). Border-collision and period-doubling bifurcations appear as the input
voltage `E` (or the load `R`, reference `Vref`, or loop gain) is varied.

Implementation note: the classic Deane–Hamill chaos uses *naturally* sampled
PWM, whose ramp-crossing instant is transcendental (no closed form). This
model uses *regular* sampling on the same circuit, which keeps the map
closed-form (and ForwardDiff-friendly away from the d = 0/1 borders) while
still exhibiting the documented border-collision / period-doubling route.

References:
  J. H. B. Deane, D. C. Hamill, "Analysis, simulation and experimental study
  of chaos in the buck converter", IEEE PESC (1990).
  doi:10.1109/PESC.1990.131228
  G. Yuan, S. Banerjee, E. Ott, J. A. Yorke, "Border-collision bifurcations in
  the buck converter", IEEE Trans. Circuits Syst. I 45(7):707-716 (1998).
  doi:10.1109/81.703837
  S. Banerjee, G. C. Verghese (eds.), "Nonlinear Phenomena in Power
  Electronics", IEEE Press (2001), ch. 5.
"""

"""Exact `exp(A*tau)*(x - x_eq) + x_eq` for the buck LC system matrix
`A = [[-1/(RC), 1/C], [-1/L, 0]]`, valid for under-, over-, and critically
damped regimes."""
function _buck_vm_flow(v, i, tau, veq, ieq, R, L, C)
    w1 = v - veq
    w2 = i - ieq
    a = -1 / (2 * R * C)
    # A*w
    Aw1 = -w1 / (R * C) + w2 / C
    Aw2 = -w1 / L
    # disc selects the damping regime. psi0 ("cosine-like") and psi1 ("sinc-like":
    # sin(omega*tau)/omega, sinh(s*tau)/s, or tau at critical) are all smooth as
    # their argument -> 0, so the only care needed is the regime boundary. A small
    # relative tolerance around disc = 0 routes the (near-)critical case to the tau
    # series rather than relying on exact floating-point equality.
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

function _buck_voltage_mode_rule(x, p; L=20e-3, C=47e-6, T=400e-6, VL=3.8, VU=8.2)
    v, i = x[1], x[2]
    E    = p[1]
    Vref = length(p) >= 2 ? p[2] : 11.3
    R    = length(p) >= 3 ? p[3] : 22.0
    gain = length(p) >= 4 ? p[4] : 1.2

    # Regular-sampled duty from the period-start voltage.
    d = (gain * (Vref - v) - VL) / (VU - VL)
    d = clamp(d, 0.0, 1.0)

    # ON sub-interval: equilibrium (E, E/R). OFF sub-interval: equilibrium (0, 0).
    v1, i1 = _buck_vm_flow(v, i, d * T, E, E / R, R, L, C)
    v2, i2 = _buck_vm_flow(v1, i1, (1 - d) * T, 0.0, 0.0, R, L, C)
    return SVector(v2, i2)
end

function _buck_voltage_mode_raw_duty(x, p; VL=3.8, VU=8.2)
    v = x[1]
    Vref = length(p) >= 2 ? p[2] : 11.3
    gain = length(p) >= 4 ? p[4] : 1.2
    return (gain * (Vref - v) - VL) / (VU - VL)
end

"""
    buck_voltage_mode(; L=20e-3, C=47e-6, T=400e-6, VL=3.8, VU=8.2) -> DiscreteMap

Create a regular-sampled voltage-mode PWM buck converter map. Bifurcation
parameters are `[E, Vref, R, gain]` (input voltage, reference voltage, load
resistance, loop gain). The constructor fixes the Deane–Hamill (1990) LC filter
`L = 20 mH`, `C = 47 µF`, switching period `T = 1/2500 s`, and ramp bounds
`[VL, VU] = [3.8, 8.2]`.

When a caller passes a shortened parameter vector the trailing entries fall
back to the *workbench preset regime* — `Vref = 11.3 V`, `R = 22 Ω`,
`gain = 1.2` — rather than a separate "Deane–Hamill chaos" set, so a short `p`
lands in the same low-gain regulated regime the presets and UI describe. At
this circuit the regulated period-1 orbit has a complex-conjugate multiplier
pair that crosses the unit circle in a Neimark–Sacker bifurcation near
`gain ≈ 1.46`; sweep `gain` to traverse period-1 → torus → mode-locking →
chaos. (The `d = clamp(..., 0, 1)` duty saturation can additionally produce
border-collision events at parameters where the duty hits a rail.)
"""
function buck_voltage_mode(; L::Float64=20e-3, C::Float64=47e-6, T::Float64=400e-6,
                           VL::Float64=3.8, VU::Float64=8.2)
    f = (x, p) -> _buck_voltage_mode_rule(x, p; L=L, C=C, T=T, VL=VL, VU=VU)
    events = [
        SwitchingEvent(
            "duty-lower-border",
            (x, p) -> _buck_voltage_mode_raw_duty(x, p; VL=VL, VU=VU);
            description="Unclamped PWM duty cycle reaches the lower rail d=0.",
            tolerance=1e-6,
            scale=1.0
        ),
        SwitchingEvent(
            "duty-upper-border",
            (x, p) -> 1.0 - _buck_voltage_mode_raw_duty(x, p; VL=VL, VU=VU);
            description="Unclamped PWM duty cycle reaches the upper rail d=1.",
            tolerance=1e-6,
            scale=1.0
        )
    ]
    DiscreteMap(f, 2, [:E, :Vref, :R, :gain], "Buck (voltage-mode)"; switching_events=events)
end
