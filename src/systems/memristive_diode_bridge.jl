"""
Memristive diode-bridge band-pass-filter (BPF) chaotic circuit: a third-order
autonomous continuous-time system. A second-order active BPF has one of its
series resistors replaced by an improved memristive diode-bridge emulator (a
diode bridge cascaded with a single inductor), which supplies the nonlinearity.

The three dynamic elements (two capacitors and the emulator inductor) give three
state variables — `x = ρ·v_C1`, `y = ρ·v_C2`, `z = ρ·R·i_L` — and the
dimensionless state equations are (Xu et al. 2017, eq. 11):

    ẋ = (c + z) tanh(y − x)
    ẏ = k y − (k + 1) x − (c + z) tanh(y − x)
    ż = a [ ln(c·cosh(y − x)) − ln(c + z) ]

The diode-bridge memristor contributes `i = (2 I_S + i_L) tanh(ρ v)` with
`v = v_C2 − v_C1` (so the dimensionless drive is `y − x`), and the inductor
state follows from the bridge's logarithmic v–i law. Dimensionless parameters:

    a = R² C / L        (primary bifurcation parameter; varied via C in hardware)
    c = 2 ρ R I_S        (diode-bridge scale; ρ = 1/(2 n V_T))
    k = R_i / R_f        (BPF feedback ratio)

Typical values (a = 0.005, c = 6.02e-6, k = 0.05) come from C₁=C₂=20 nF,
L=10 mH, R=50 Ω, R_i=50 Ω, R_f=1 kΩ and four 1N4148 diodes (I_S=5.84 nA,
n=1.94, V_T=25 mV). The only equilibrium is the origin, an unstable saddle, so
all attractors are limit cycles or chaos. Sweeping `a` from 0.001 to 0.1
produces period, period-doubling, chaos, periodic windows, and — depending on
initial conditions — coexisting (multistable) attractors. Documented regimes
(initial condition (0, 0.01, 0), Xu et al. Fig. 6): period-1 at a≈0.014,
period-2 at a≈0.022, chaos (spiral) at a≈0.028, period-3 window at a≈0.031.

Poincaré section: hyperplane `y = 0` taken on upward crossings (matching the
paper's `y = 0` Poincaré map, plotted in the x–z plane, Fig. 4d); the projected
`(x, z)` return map is what brute-force / continuation iterate.

Numerical note: the system is stiff near the origin (λ₃ = −a/c ≈ −830 at the
saddle while the in-plane eigenvalues are ≈ 0.05), so a stiff-aware solver
(the workbench `"auto"` = `AutoTsit5(Rosenbrock23())`) is appropriate. The
`ln(c + z)` term requires `c + z > 0`; on the physical attractor `z ≥ 0` and the
flow repels from `z = 0`, but a small positive floor guards the logarithm
against transient integrator overshoot without altering the attractor.

References:
  Q. Xu, Q. Zhang, N. Wang, H. Wu, B. Bao, "An Improved Memristive Diode
  Bridge-Based Band Pass Filter Chaotic Circuit", Mathematical Problems in
  Engineering, vol. 2017, Article ID 2461964 (2017).
  doi:10.1155/2017/2461964
  B. Bao, N. Wang, Q. Xu, H. Wu, Y. Hu, "A simple third-order memristive band
  pass filter chaotic circuit", IEEE Trans. Circuits Syst. II 64(8):977-981
  (2017). doi:10.1109/TCSII.2017.2710953
"""

"""
    memristive_diode_bridge(; c=6.02e-6, k=0.05) -> ContinuousODE

Create the third-order memristive diode-bridge BPF chaotic circuit. Parameter
vector order is `[a, c, k]`; `a = R²C/L` is the primary bifurcation parameter.
The Poincaré section is `y = 0` (upward crossings), retaining the `(x, z)`
coordinates, with the full state reconstructed through the section template
`(x, 0, z)`.

The default state `[0.0, 0.01, 0.0]` is the paper's typical initial condition.
When a caller passes a shorter parameter vector (e.g. only `[a]` while sweeping
`a`), the remaining entries fall back to the constructor's `c` and `k` kwargs,
so changing the kwargs changes the model the closure represents.
"""
function memristive_diode_bridge(; c::Float64=6.02e-6, k::Float64=0.05)
    # c = 2ρRI_S is physically positive; ln(c) below requires it. Validate at the
    # constructor boundary so a bad value fails fast instead of as a mid-integration
    # DomainError. (The param vector's c slot is never swept by any preset.)
    c > 0 || throw(ArgumentError("memristive_diode_bridge requires c > 0 (got c=$c); c = 2ρRI_S is a positive diode-bridge scale."))
    f! = function(du, u, p, t)
        x, y, z = u[1], u[2], u[3]
        a  = p[1]
        cp = length(p) >= 2 ? p[2] : c
        kp = length(p) >= 3 ? p[3] : k
        w = y - x
        g = tanh(w)
        # cz_safe guards the ln domain against transient integrator overshoot
        # (physically z ≥ 0 and the flow repels from z = 0). Use it consistently
        # in every term so an overshoot can't flip the sign of du[1]/du[2] while
        # the strong restoring du[3] pulls z back; on the attractor cz ≫ 1e-12 so
        # this is identical to the faithful model.
        cz = max(cp + z, 1e-12)
        # ln(cosh w) computed overflow-safe: cosh(w) overflows Float64 for
        # |w| ≳ 710, which would turn ln(c·cosh w) into Inf→NaN and abort the
        # integrator on divergent FD-Newton probes. This identity is exact.
        aw = abs(w)
        logcosh = aw + log1p(exp(-2 * aw)) - log(2)
        du[1] = cz * g
        du[2] = kp * y - (kp + 1) * x - cz * g
        du[3] = a * (log(cp) + logcosh - log(cz))
        nothing
    end
    section = PoincareSection(
        (u, t, integrator) -> u[2];   # y = 0
        direction = :up,
        projection = [1, 3],          # keep x and z; y reset to 0 on the section
        template = [0.0, 0.0, 0.0]
    )
    ContinuousODE(
        f!, 3, section, [:a, :c, :k], "Memristive Diode Bridge";
        tspan_hint = 50.0,            # one Poincaré return ≈ 26-30 in dimensionless τ
        default_initial_state = [0.0, 0.01, 0.0],
        default_params = [0.005, c, k]
    )
end
