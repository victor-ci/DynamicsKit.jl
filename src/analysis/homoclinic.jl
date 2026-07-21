"""
Connecting-orbit analysis: homoclinic and heteroclinic connections to
equilibria, and homoclinic connections to saddle periodic orbits.

The numerics are owned entirely by DynamicsKit. A truncated orbit is discretized
on a uniform trapezoidal mesh and corrected as a projection boundary-value
problem: the endpoints are pinned to the linear stable/unstable subspaces of the
source and target saddles (or the Floquet subspaces of a saddle cycle). The
primary corrector is a line-searched Gauss-Newton; a Levenberg-Marquardt /
pseudo-inverse path is the recorded fallback. Loci are traced by pseudo-arclength
continuation with real eigenvalue and adjoint/variational-transport test
functions marking typed special points.

The subfiles are concatenated in load-bearing order: projection primitives, the
corrector, the saddle-cycle machinery, the continuation driver, and the public
API.
"""

include("homoclinic/projection.jl")
include("homoclinic/corrector.jl")
include("homoclinic/saddle_cycle.jl")
include("homoclinic/continuation.jl")
include("homoclinic/api.jl")
