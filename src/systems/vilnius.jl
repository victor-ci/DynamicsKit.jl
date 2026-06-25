"""
Vilnius oscillator: a third-order analog electronic chaos generator analyzed
by Ipatovs et al. (2023).

    dx/dt = y
    dy/dt = a·y - x - z
    ε·dz/dt = b + y - 4e-9·(exp(z) - 1)

Parameter vector order: `[a, b, ε]`. The exponential is clamped to `z ≤ 500`
to keep the in-place ODE numerically stable when the integrator transiently
overshoots during seed exploration.

Reference:
  A. Ipatovs, C. Iheanacho, D. Pikuļins, S. Tjukovs, A. Litviņenko, "Complete
  Bifurcation Analysis of the Vilnius Chaotic Oscillator", Electronics 12(13),
  Article 2861 (2023). doi:10.3390/electronics12132861
"""

function _vilnius_ode!(du, u, p, t)
    x, y, z = u
    a = p[1]
    b = length(p) >= 2 ? p[2] : 30.0
    ε = length(p) >= 3 ? p[3] : 0.2
    du[1] = y
    du[2] = a * y - x - z
    du[3] = (b + y - 4e-9 * (exp(min(z, 500.0)) - 1)) / ε
    nothing
end

"""
    vilnius_oscillator(; b=30.0, ε=0.2) -> ContinuousODE

Create the Vilnius oscillator from Ipatovs et al. (2023). The Poincaré section
is `y = 0` (upward crossings), projecting onto the `(x, z)` coordinates. Use
this constructor for the reference Figure 10-14 diagrams; sweep `a` through
≈ `[0.05, 0.6]` at default `(b, ε)`.
"""
function vilnius_oscillator(; b::Float64=30.0, ε::Float64=0.2)
    section = PoincareSection(
        (u, t, integrator) -> u[2];
        direction = :up,
        projection = [1, 3],
        template = [0.0, 0.0, 0.0]
    )
    ContinuousODE(
        _vilnius_ode!, 3, section, [:a, :b, :ε], "Vilnius";
        tspan_hint = 20.0,
        default_initial_state = [0.0, 0.1, 0.0],
        default_params = [0.2, b, ε]
    )
end
