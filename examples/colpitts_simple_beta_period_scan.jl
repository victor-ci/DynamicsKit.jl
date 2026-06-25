#!/usr/bin/env julia

using DynamicsKit
using DifferentialEquations

const BETA_VALUES = begin
    raw = strip(get(ENV, "COLPITTS_BETA_VALUES", "100,105,110,115,120,125,130,135"))
    [parse(Float64, strip(item)) for item in split(raw, ',') if !isempty(strip(item))]
end
const ITERATIONS = parse(Int, get(ENV, "COLPITTS_PERIOD_SCAN_ITER", "8"))
const TRANSIENT = parse(Int, get(ENV, "COLPITTS_PERIOD_SCAN_TRANSIENT", "120"))
const DETECTION_TOL = parse(Float64, get(ENV, "COLPITTS_PERIOD_SCAN_TOL", "1e-4"))

function detect_period(points::AbstractMatrix{<:Real}; tol::Float64=1e-4)
    n = size(points, 1)
    n <= 1 && return n
    for period in 1:(n - 1)
        reference = @view points[1:period, :]
        matches = true
        for i in 1:n
            candidate = @view points[i, :]
            target = @view reference[mod1(i, period), :]
            if maximum(abs.(candidate .- target)) > tol
                matches = false
                break
            end
        end
        matches && return period
    end
    return 0
end

println("═══ Colpitts (simple) — β period scan ═══\n")
println("Using $(ITERATIONS) recorded crossings after $(TRANSIENT) transient crossings.")
println("β values: $(join(BETA_VALUES, ", "))\n")
println(rpad("β", 10), rpad("period", 10), "seed point [V_C1, V_C2]")
println(repeat("-", 56))

sys = colpitts_simple_oscillator()
for beta in BETA_VALUES
    params = [40e-9, 40e-9, beta, 5.0, 5.0]
    points = DynamicsKit._collect_poincare_points(
        sys,
        params;
        initial_point=copy(sys.default_initial_state),
        crossings=ITERATIONS,
        transient=TRANSIENT,
        solver=Tsit5(),
        reltol=1e-7,
        abstol=1e-7,
        projected=true
    )

    point_matrix = isempty(points) ? zeros(0, 2) : reduce(vcat, permutedims.(points))
    period = size(point_matrix, 1) == ITERATIONS ? detect_period(point_matrix; tol=DETECTION_TOL) : -1
    seed_str = isempty(points) ? "(no crossings)" : string(round.(collect(first(points)); digits=6))
    println(rpad(string(round(beta; digits=3)), 10), rpad(string(period), 10), seed_str)
end

println("\nA detected period of 1 across β ≈ 100–130 is the window used by the validated continuation preset.")

