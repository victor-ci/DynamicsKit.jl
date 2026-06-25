"""
Buck converter: a 2D discrete-time piecewise-smooth electronic system.

Models a DC-DC buck converter with switching condition based on
reference current crossing.

Bifurcation parameters: [Iref, Ein]
State variables: [Vn, In] (voltage, current)
"""

function _buck_rule(x, p; L=2.2e-6, T=1/0.5e6)
    Vn, In = x[1], x[2]
    Iref, Ein = p[1], p[2]

    # Circuit constants
    R = 1.6
    C = 39.1e-6

    # Switching condition
    tn = L * (Iref - In) / (Ein - Vn)

    # Substitutions
    a = -1 / (2 * R * C)
    b = sqrt(1 / (L * C) - 1 / (4 * R^2 * C^2))
    c1 = In - Ein / R
    c2 = (1 / b) * ((Ein - Vn) / L - a * (In - Ein / R))
    c3 = Vn - Ein
    c4 = (a / b) * (Vn + Ein) + In / (b * C)
    k1 = Iref
    Vc0 = exp(a * tn) * (c3 * cos(b * tn) + c4 * sin(b * tn)) + Ein
    k2 = (-1 / (b * L)) * (Vc0 + a * L * Iref)

    if tn >= T
        In_next = exp(a * T) * (c1 * cos(b * T) + c2 * sin(b * T)) + Ein / R
        Vn_next = exp(a * T) * (c3 * cos(b * T) + c4 * sin(b * T)) + Ein
    else
        In_next = exp(a * (T - tn)) * (k1 * cos(b * (T - tn)) + k2 * sin(b * (T - tn)))
        Vn_next = -L * exp(a * (T - tn)) * ((k1 * a + k2 * b) * cos(b * (T - tn)) + (k2 * a - k1 * b) * sin(b * (T - tn)))
    end

    SVector(Vn_next, In_next)
end

function _buck_switching_time_guard(x, p; L=2.2e-6, T=1/0.5e6)
    Vn, In = x[1], x[2]
    Iref, Ein = p[1], p[2]
    denom = Ein - Vn
    denom == 0 && return Inf
    tn = L * (Iref - In) / denom
    return tn - T
end

"""
    buck_converter(; L=2.2e-6, T=1/0.5e6) -> DiscreteMap

Create a Buck converter discrete map. Bifurcation parameters are `[Iref, Ein]`.
The constructor fixes the inductor `L = 2.2 µH` and switching period `T = 2 µs`
(500 kHz clock). Other circuit constants (`R = 1.6 Ω`, `C = 39.1 µF`) are internal.
"""
function buck_converter(; L::Float64=2.2e-6, T::Float64=1/0.5e6)
    (L > 0 && T > 0) ||
        throw(ArgumentError("buck_converter requires positive circuit constants: L=$L, T=$T."))
    f = (x, p) -> _buck_rule(x, p; L=L, T=T)
    events = [
        SwitchingEvent(
            "switch-time-period-border",
            (x, p) -> _buck_switching_time_guard(x, p; L=L, T=T);
            description="Switching time tn reaches the clock period T; the map changes between switched and unswitched cycles.",
            tolerance=1e-9,
            scale=T
        )
    ]
    DiscreteMap(f, 2, [:Iref, :Ein], "Buck Converter"; switching_events=events)
end
