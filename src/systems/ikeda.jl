"""
Ikeda map: a 2D discrete-time iteration that models the field amplitude in an
optical ring cavity. In the (x, y) real-part / imaginary-part formulation:

    t_n     = a - b / (1 + x_n^2 + y_n^2)
    x_{n+1} = 1 + u * (x_n * cos(t_n) - y_n * sin(t_n))
    y_{n+1} = u * (x_n * sin(t_n) + y_n * cos(t_n))

Standard "chaotic" parameter set: `a = 0.4`, `b = 6.0`, `u = 0.9`.
Standard "regular" set: same `(a, b)` with `u ≈ 0.5–0.7`.

Original optical-cavity reference:
    K. Ikeda, "Multiple-valued stationary state and its instability of the
    transmitted light by a ring cavity system", Opt. Commun. 30 (1979) 257–261.
    K. Ikeda, H. Daido, O. Akimoto, "Optical turbulence: chaotic behavior of
    transmitted light from a ring cavity", Phys. Rev. Lett. 45 (1980) 709–712.

The discrete real-amplitude form used here follows:
    S. M. Hammel, C. K. R. T. Jones, J. V. Moloney, "Global dynamical behavior
    of the optical field in a ring cavity", J. Opt. Soc. Am. B 2 (1985) 552–564.
    https://doi.org/10.1364/JOSAB.2.000552
"""

"""
    ikeda_map(; a=0.4, b=6.0) -> DiscreteMap

Create an Ikeda map with the two phase-law constants held fixed by the
constructor. The bifurcation parameter is the loss/feedback magnitude `u`.

`u` is the primary parameter swept in the literature: the trivial fixed point
loses stability through a period-doubling cascade as `u` is increased from
about `0.5` toward `≈ 0.9` (chaos), with documented periodic windows in
between.

To enable 2-D parameter maps and continuation in either of the phase-law
constants, all three of `(u, a, b)` are exposed as bifurcation parameters in
the returned `DiscreteMap` (the constructor's `a`, `b` just set their default
values).
"""
function ikeda_map(; a::Float64=0.4, b::Float64=6.0)
    f = function(x, p)
        u = p[1]
        ap = length(p) >= 2 ? p[2] : a
        bp = length(p) >= 3 ? p[3] : b
        denom = 1 + x[1]^2 + x[2]^2
        t = ap - bp / denom
        ct = cos(t)
        st = sin(t)
        SVector(1 + u * (x[1] * ct - x[2] * st),
                u * (x[1] * st + x[2] * ct))
    end
    DiscreteMap(f, 2, [:u, :a, :b], "Ikeda")
end
