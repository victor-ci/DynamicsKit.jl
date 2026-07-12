"""
Hénon map: a classic 2D discrete chaotic system.

    x_{n+1} = 1 + y_n - a * x_n^2
    y_{n+1} = b * x_n

Default parameters: a (bifurcation), b = 0.3 (fixed).
"""

function _henon_rule(x, p)
    a = p[1]
    b = length(p) > 1 ? p[2] : 0.3
    SVector(1 + x[2] - a * x[1]^2, b * x[1])
end

"""
    henon_map(; b=0.3) -> DiscreteMap

Create a Hénon map system. Both `a` (nonlinearity) and `b` (dissipation) are
exposed as parameters, so the system supports the classic two-parameter (a, b)
periodicity map in addition to the 1-D `a` cascade. The `b` keyword sets the
default used when a caller passes a single-element parameter vector
(`f(x, [a])` calls); a full `[a, b]` vector overrides it.
"""
function henon_map(; b::Float64=0.3)
    f = (x, p) -> SVector(1 + x[2] - p[1] * x[1]^2, (length(p) >= 2 ? p[2] : b) * x[1])
    DiscreteMap(f, 2, [:a, :b], "Hénon")
end

