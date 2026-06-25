"""
    select_ode_solver(solver_key::AbstractString)

Map a solver key to an `OrdinaryDiffEq` solver instance for integrating a `ContinuousODE`:

- `"auto"`         → `AutoTsit5(Rosenbrock23())` (non-stiff/stiff auto-switching; the default)
- `"tsit5"`        → `Tsit5()`
- `"rosenbrock23"` → `Rosenbrock23()`

Throws `ArgumentError` on an unknown key — strict by design, so a typo or a stale config field
surfaces as a clear error instead of a silent fallback to a different algorithm.

This is the public solver-selection entry point shared by the workbench and by scripted analysis
(e.g. reproducibility scripts that drive the public sweeps headlessly). Discrete maps need no
ODE solver, so callers select one only for `ContinuousODE` systems.
"""
function select_ode_solver(solver_key::AbstractString)
    if solver_key == "auto"
        return AutoTsit5(Rosenbrock23())
    elseif solver_key == "tsit5"
        return Tsit5()
    elseif solver_key == "rosenbrock23"
        return Rosenbrock23()
    else
        throw(ArgumentError(
            "Unknown ODE solver key '$(solver_key)'. Expected one of: \"auto\", \"tsit5\", \"rosenbrock23\"."
        ))
    end
end
