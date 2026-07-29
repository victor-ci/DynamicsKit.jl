"""
Shared helpers for the scripts in `validation/`.

Each script in this directory checks specific, named claims about DynamicsKit
against either a closed-form analytic result or a cited external reference
(literature, or a documented convention such as Kuznetsov/MATCONT normal-form
sign conventions), and reports PASS/FAIL/NOTE for each claim via
`record_gate!`. A "required" gate causes the script to exit with a nonzero
status if it fails; an "informational" gate (`required=false`) is reported but
never fails the run — use it for context that cannot be reduced to a strict
pass/fail (e.g. a cost comparison with no fixed target, or an external
cross-tool comparison pending a reference artifact that isn't available yet).

Every script is self-contained and runnable from the repository root:

    julia --project=. validation/<script>.jl

Exit code 0 means every required gate passed; nonzero means at least one
required gate failed (see the printed summary for which).
"""

struct ValidationGate
    name::String
    passed::Bool
    required::Bool
    detail::String
end

"""
    record_gate!(gates, name, passed, detail; required=true) -> Bool

Append a `ValidationGate` to `gates`, print a `PASS`/`FAIL`/`NOTE` line for it,
and return `passed` unchanged (so call sites can inline it in a boolean
expression if useful). `required=false` marks the gate informational: it is
reported but never causes `conclude` to exit with a failure status.
"""
function record_gate!(gates::Vector{ValidationGate}, name::AbstractString, passed::Bool,
                      detail::AbstractString; required::Bool=true)
    push!(gates, ValidationGate(String(name), Bool(passed), Bool(required), String(detail)))
    tag = passed ? "PASS" : (required ? "FAIL" : "NOTE")
    println(tag, " — ", required ? "[required] " : "[informational] ", name, ": ", detail)
    return passed
end

"""
    conclude(gates, label)

Print a summary and `exit(1)` if any required gate in `gates` failed;
otherwise print a one-line success summary and return `nothing`.
"""
function conclude(gates::Vector{ValidationGate}, label::AbstractString)
    failed = [g for g in gates if g.required && !g.passed]
    n_required = count(g -> g.required, gates)
    if !isempty(failed)
        println("\n$(length(failed))/$(n_required) required gate(s) failed for $label:")
        for g in failed
            println("  - ", g.name, ": ", g.detail)
        end
        exit(1)
    end
    println("\nAll required $label gates passed ($(n_required) required, $(length(gates)) total).")
    return nothing
end
