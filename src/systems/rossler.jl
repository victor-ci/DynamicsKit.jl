"""
Rössler system: a 3D continuous-time ODE with a single quadratic
nonlinearity, exhibiting a textbook period-doubling cascade.

    dx/dt = -y - z
    dy/dt = x + a * y
    dz/dt = b + z * (x - c)

Original reference:
    O. E. Rössler, "An equation for continuous chaos",
    Phys. Lett. A 57 (1976) 397–398.
    https://doi.org/10.1016/0375-9601(76)90101-8

Canonical parameter sweep (`a = b = 0.2`, vary `c`):
    c ≈ 2.5  → period-1 limit cycle
    c ≈ 3.5  → period-2
    c ≈ 4.0  → period-4
    c ≈ 4.23 → onset of chaos (Feigenbaum accumulation)
    c ≈ 4.6  → period-3 window
    c ≈ 5.7  → fully developed single-scroll chaotic attractor

Documented in e.g. J. C. Sprott, *Chaos and Time-Series Analysis*
(Oxford University Press, 2003) and S. H. Strogatz, *Nonlinear Dynamics
and Chaos*, 2nd ed. (Westview Press, 2015).

Poincaré section: hyperplane `y = 0` taken on upward crossings (the spiral
component of the flow rotates around the z-axis, so `y = 0` is hit twice
per loop; the upward-crossing constraint selects one of them). The
projected (x, z) return map is what brute-force / continuation iterate.
"""

"""
    rossler_oscillator(; a=0.2, b=0.2, c=5.7) -> ContinuousODE

Create the standard Rössler oscillator. Parameter vector order is
`[a, b, c]`. The Poincaré section is `y = 0` (upward crossings), retaining
the `(x, z)` coordinates. The full state at the section is reconstructed
through the section template `(x, 0, z)`.

The default state `[1.0, 1.0, 1.0]` lands inside the chaotic attractor's
basin for the standard `(a, b, c) = (0.2, 0.2, 5.7)` parameters and is
also a stable basin point for the period-1/2/4 windows along the cascade.

When a caller passes a shorter parameter vector (e.g. only `[a]` because
only the first parameter is being swept), the remaining entries fall back
to the constructor's `b` and `c` kwargs — so changing the kwargs actually
changes the model the closure represents.
"""
function rossler_oscillator(; a::Float64=0.2, b::Float64=0.2, c::Float64=5.7)
    f! = function(du, u, p, t)
        ap = p[1]
        bp = length(p) >= 2 ? p[2] : b
        cp = length(p) >= 3 ? p[3] : c
        du[1] = -u[2] - u[3]
        du[2] = u[1] + ap * u[2]
        du[3] = bp + u[3] * (u[1] - cp)
        nothing
    end
    section = PoincareSection(
        (u, t, integrator) -> u[2];   # y = 0
        direction = :up,
        projection = [1, 3],          # keep x and z; y reset to 0 on the section
        template = [0.0, 0.0, 0.0]
    )
    ContinuousODE(
        f!, 3, section, [:a, :b, :c], "Rössler";
        tspan_hint = 6.5,             # one Poincaré return ≈ 2π / ω with ω ≈ 1
        default_initial_state = [1.0, 1.0, 1.0],
        default_params = [a, b, c]
    )
end
