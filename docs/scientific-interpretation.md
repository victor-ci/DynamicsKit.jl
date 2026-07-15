# Scientific interpretation guide

This guide explains how to read DynamicsKit outputs without over-interpreting them.

## Brute-force sweeps vs continuation branches

| Feature | Brute-force sweep | Continuation branch |
| --- | --- | --- |
| Object traced | Attractor reached by simulation | Periodic solution of the map/Poincare map |
| Stability | Usually stable attractors only | Stable and unstable branches can appear |
| Initial condition dependence | Strong | Seed dependence mainly affects which branch is started |
| Good for | Observed behavior and bifurcation diagrams | Branch geometry, folds, bifurcations, unstable structure |
| Can miss | Coexisting attractors, unstable or tiny windows | Attractors not connected to chosen seeds or failed solves |

Use both when possible. Agreement between a brute-force cloud and stable continuation points is strong evidence. Disagreement is also meaningful: it can indicate unstable branches, multistability, insufficient transient, missed windows, or solver issues.

## Period detection and period `0`

Period detection compares the base post-transient point with later points and selects the smallest return whose closure error is below a tolerance-scaled threshold.

The public heatmap-like result matrices use:

- positive integers for detected periods;
- `0` for "no finite period was detected."

`0` is intentionally backward-compatible and broad. Diagnostics distinguish:

| Status | Interpretation |
| --- | --- |
| `aperiodic_or_high_period` | Could be chaos, quasiperiodicity, unresolved high period, or a transient that did not settle |
| `diverged` | State amplitude exceeded a cutoff |
| `insufficient_crossings` | ODE did not cross the Poincare section enough times |
| `integration_failed` | ODE solver failed |
| `invalid_state` | Non-finite state was produced |

Closure confidence is a numerical clue, not a proof. Low confidence around a detected period means the return was close enough to pass the threshold but may deserve finer sampling, a longer transient, or an independent check.

## Fixed-seed vs neighbor-seeded 2D maps

Fixed-seed maps answer:

> Starting every parameter cell from the same initial condition, what finite period is detected?

Neighbor-seeded maps answer a different question:

> If I traverse parameter space and use the previous cell's final state as the next cell's seed, what branch/path do I follow?

| Mode | Meaning |
| --- | --- |
| Fixed initial condition | Reproducible pointwise classification map. |
| Neighbor continuation, full transient | Hysteresis/path-following scan using neighbor final states while keeping the full transient. |
| Neighbor continuation, accelerated transient | Same path-following scan with reduced transient after the first cell/tile. |

Neighbor modes are scientifically meaningful for branch following and hysteresis, but they are traversal-dependent. A mismatch with fixed-seed output is not automatically an error.

## Multistability

Single-seed sweeps can hide coexisting attractors. A cell can look period-1 from one seed and chaotic or period-3 from another.

The multistability map mode samples extra initial conditions per parameter cell and reports:

- dominant period;
- set of periods observed;
- coexistence flags;
- attractor counts;
- basin-fraction entropy;
- selected seed summaries.

Use this mode for systems known to have coexisting attractors, especially the memristive diode bridge and parameter windows near basin boundaries.

## Lyapunov diagnostics

Largest-Lyapunov estimates help split period `0` cells into candidates:

| Exponent behavior | Label |
| --- | --- |
| Positive above tolerance | `chaotic_candidate` |
| Near zero | `quasiperiodic_neutral_candidate` |
| Negative or estimate unavailable but no period found | `unresolved` |

These are estimates. Treat them as diagnostics to guide further analysis, not as final proofs. Increase iterations, compare neighboring cells, and inspect phase portraits or return maps for publication-level claims.

The same estimator is also available as a standalone 1D `lyapunov_diagram` sweep. Use that when you want a route-to-chaos summary along one parameter without running a full 2D map. For direct 2D exponent sweeps, `lyapunov_field(sys, config)` computes the field without first paying for a period map. For combined 2D maps with `lyapunov_enabled=true`, `lyapunov_field(result)` still exposes the co-computed field as a first-class result so plotting the exponent layer is a presentation choice, not a separate computation.

## Power spectra

Power spectra answer a different question from Lyapunov estimates:

| Spectrum shape | Typical interpretation |
| --- | --- |
| One or a few sharp lines | Periodic or frequency-locked regime |
| Many incommensurate sharp lines | Quasiperiodic / torus-like candidate |
| Broadband floor or smeared peaks | Chaotic or strongly modulated regime candidate |

Use spectra together with phase portraits and Lyapunov diagnostics. A broadband spectrum alone is not a proof of chaos, and a near-zero Lyapunov exponent alone is not a proof of quasiperiodicity.

## Multipliers, Floquet spectra, and stability

For a period-N map orbit, multipliers are eigenvalues of the derivative of `F^N` at the orbit. For ODEs, Floquet/Poincare multipliers describe the linearized return to the section.

Rules of thumb:

| Diagnostic | Interpretation |
| --- | --- |
| All multiplier moduli below 1 | Locally attracting periodic orbit |
| One multiplier crosses `-1` | Period-doubling candidate |
| One real multiplier crosses `+1` | Fold/limit-point or branch-point candidate |
| Complex pair crosses unit circle | Neimark-Sacker/torus candidate for maps/Poincare maps |
| Large residual norm | Branch point may not solve the periodicity equation accurately |

Always consider residuals and solver warnings together with multiplier spectra.

## Codimension-2 curves

`codim2_curve` turns those 1D bifurcation clues into a traced locus across a second parameter. Read it with the same caution as any stitched diagnostic:

| Field | Interpretation |
| --- | --- |
| `raw_candidates` | Every candidate bifurcation location found on each secondary slice |
| `primary_values` + `valid_mask` | The one stitched curve chosen by nearest-neighbour tracking |
| `candidate_sources` | Whether a slice came from explicit special points or the PD stability-flip fallback |
| `slice_statuses` / `slice_messages` | Whether continuation succeeded and whether a candidate was actually found |

If a scientifically important segment only appears through the `:stability_flip` fallback, validate it against the multiplier spectra and, when possible, a nearby Lyapunov field or brute-force map. The stitched curve is meant to summarize a branch family, not to replace the underlying slice evidence.

A stability flip is not automatically a period doubling: a multiplier can leave
the unit circle through +1 (fold), −1 (flip), or as a complex pair
(Neimark–Sacker), and the flip fallback cannot distinguish them. The
`:defining_system` engine resolves this ambiguity — it verifies the seed
candidate against the actual multiplier gap to the requested defining value and
records per-sample multipliers on the returned `Codim2ContinuationResult`, so
every point of the locus carries its own evidence that the defining condition
holds. Defining-system curves are solved to Newton tolerance and may fold back
in either parameter; a fold of the locus itself
(`curve_fold_secondary_values`) is an organising-point candidate (for example a
cusp) worth inspecting rather than an artifact.

## Special points

Continuation can record special points such as folds, period doublings, branch points, and Hopf/Neimark-Sacker-like markers depending on system type and BifurcationKit detection level.

The atlas can optionally use special points for limited branch switching. This is budgeted and local: it is meant to discover nearby connected structures without turning the atlas into an exhaustive global continuation engine.

## Continuous-time systems and Poincare sections

ODE analyses reduce continuous trajectories to a Poincare map by recording section crossings.

Important section choices:

- condition function and crossing direction;
- projected state coordinates retained as map coordinates;
- full-state template used to lift projected coordinates back into the ODE state;
- minimum crossing time to avoid counting the initial point as an immediate crossing.

If a continuous analysis returns sparse data or many period `0` cells, inspect crossing diagnostics before interpreting it as dynamics. The trajectory may simply not be hitting the section within the requested time/budget.

## Variational vs finite-difference ODE derivatives

ODE continuation can estimate Poincare-map derivatives by finite differences or, when requested, by variational-equation monodromy.

| Method | Strength | Cost/risk |
| --- | --- | --- |
| Finite difference | Robust fallback, simple assumptions | Many integrations per Jacobian and lower accuracy near stiff regions |
| Variational | Better accuracy and speed for smooth systems | Assumes smooth differentiability across the integrated segment |

For nonsmooth or switching systems, derivative-based smooth bifurcation labels can be incomplete. Use switching-event diagnostics and physical guard interpretation.

## Nonsmooth and switching systems

Power-electronics maps and switching circuits can have border collisions, saturation events, duty clamps, and grazing-like behavior. These may not look like smooth folds or period doublings.

`SwitchingEvent` metadata reports proximity to guard surfaces. A branch or map cell near a guard deserves interpretation as a nonsmooth event candidate even if the smooth multiplier story appears ordinary.

## Practical validation checklist

Before treating a result as scientifically meaningful:

1. Compare brute-force attractor samples with stable continuation branches.
2. Check residuals and multiplier spectra on continuation branches.
3. Inspect period `0` status counts rather than only the period matrix.
4. Use Lyapunov estimates for suspected chaos/quasiperiodicity.
5. Run multiseed maps or basins where coexistence is plausible.
6. For ODEs, inspect crossing diagnostics and phase portraits.
7. For switching systems, inspect guard-distance diagnostics.
8. Rerun important claims with finer steps, more iterations, or independent initial conditions.
