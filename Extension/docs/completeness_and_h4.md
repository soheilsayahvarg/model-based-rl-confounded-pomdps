# Completeness Is the Switch for (H4)

**Status: §1–§3 are population algebra and are complete. §4 states predictions for
the sampling test; §5 is empty and is filled in a later commit, so the
predict-then-measure order stays auditable.**

---

## 1. Why this exists

Round 6 refuted the coverage branch of step (c) by showing that on the Phase 3 toy
`beta_pop = ||P_Nul b_true|| = 5.4e-16` — exactly zero — so the coverage horn has
no instance there. It attributed the zero to the anchor paper's completeness
assumption (3.3): `K0` invertible forces the min-norm bridge out of the population
null.

That attribution is a *prediction*, and it is checkable with no data at all. If it
is right, `beta_pop > 0` exactly when per-action completeness fails, and nowhere
else. This document tests it as linear algebra across 50 cells, and then aims the
sampling test that step (c) still owes.

## 2. The algebra

The min-norm bridge solves `E b = c`, so every block of `b_true` lies in
`range(E^T)`, of dimension `rank(E) = |S|`. The population stage-2 design for
action `a` is spanned by the stage-1 profiles

```text
mu(. | a, x) = E^T v_{a,x},    v_{a,x}[s] ∝ p1[s] * pi_b[s,a] * K0[s,x],   x in O_0
```

so the design's population range is a **subspace of** `range(E^T)`, and

```text
beta_pop > 0   <=>   dim span{ v_{a,x} : x }  <  rank(E) = |S|.
```

Two ways for that inclusion to be strict, and they are qualitatively different:

**Route 1 — knife-edge.** `confound = 1.0` makes `pi_b[:,a]` one-hot in `s`, so
every `v_{a,x}` is supported on the *same* single state and the span collapses to
dimension 1.

**Route 2 — structural.** `|O_0| < |S|`: the negative control is too coarse to
identify the latent state, so `rank(K0) <= |O_0| < |S|` regardless of the policy.

## 3. Result — 50/50, and the two routes behave differently

`poc/run_completeness_beta.py`, no sampling. `beta_pop` is reported relative to
`||b_true||` for the action's reward block.

| `(\|S\|,\|O\|,\|O_0\|)` | confound 0.0 | 0.3 | 0.6 | 0.9 | **1.0** |
|---|---|---|---|---|---|
| (2,6,4) | 0.0% | 0.0% | 0.0% | 0.0% | **66.0%** |
| (2,3,2) | 0.0% | 0.0% | 0.0% | 0.0% | **65.7%** |
| (3,7,5) | 0.0% | 0.0% | 0.0% | 0.0% | **56.9%** |
| (2,6,2) | 0.0% | 0.0% | 0.0% | 0.0% | **64.3%** |
| **(4,6,2)** | **71.3%** | **71.3%** | **71.1%** | **69.4%** | **69.5%** |

**The prediction holds in 50/50 cells**, and the dichotomy is sharp rather than
graded: `beta_pop <= 1.5e-15` in every complete cell and `56.9%`–`77.3%` of the
bridge norm in every incomplete one. Nothing sits in between. Max bridge residual
`4.4e-16`, so the min-norm solve is exact everywhere.

Round 6's attribution is therefore confirmed, as population algebra rather than as
a plausible story about one environment.

**Route 2 is the one worth building on.** Route 1 is a knife-edge — at
`confound = 0.9` completeness is restored and `beta_pop` returns to machine zero —
and a referee would call a deterministic behaviour policy contrived. Route 2 is a
**dimensional** condition: `|O_0| < |S|`, checkable before fitting anything, and it
holds at every confounding level. `(4,6,2)` shows ~70% null share from
`confound = 0.0` to `1.0`.

This composes with step (a)'s surviving result into two pre-fit checks:

| question | condition | from step |
|---|---|---|
| Does the design have an exact null? | `dim >= \|O\| - min(\|O\|,\|O_0\|) > 0` | (a) |
| Does the **truth** have mass in it? | `\|O_0\| < \|S\|`, or `confound = 1` | here |

Both are dimensional, both are answerable before any estimation.

**Scope, and it matters.** `|O_0| < |S|` **violates the anchor paper's Assumption
3.3.** Nothing here contradicts the paper. This is a robustness question — what
does the construction do when its own completeness premise fails — and it must be
labelled that way, not as a defect. Round 6 made this correction once already
(`noncontraction.md` §2) and it applies verbatim here.

## 4. Predictions for the sampling test

The coverage branch of step (c) has never had a confirmed empirical instance: on
the toy the predicted crossing sat at `N* = 1.7e9`, unreachable, and we wrongly
reported it as tested. With `beta_pop ≈ 1.2` instead of an artifactual `0.053`, and
`N* ∝ beta^-4`, the threshold moves into range.

Predicted from population quantities only — `sigma2_pop` is the smallest retained
eigenvalue of the population design, `lam0 = 0.03`, `kappa = 0.5`:

```text
N2* = ( c / (lam0 * beta_pop^2 * sigma2_pop) )^2
```

| config | confound | act | `sigma2_pop` | `beta_pop` | `c` | predicted `N*` |
|---|---|---|---|---|---|---|
| (4,6,2) | 0.0 | 0 | 3.07e-02 | 1.211 | 0.03 | **1,647** |
| (4,6,2) | 0.0 | 0 | 3.07e-02 | 1.211 | 1.0 | **1,830,412** |
| (4,6,2) | 0.0 | 1 | 3.07e-02 | 1.143 | 1.0 | **2,307,166** |
| (4,6,2) | 0.6 | 0 | 6.73e-03 | 1.206 | 0.03 | **34,846** |
| (2,6,4) | 1.0 | 0 | 4.49e-01 | 0.759 | 1.0 | **55,412** |
| (3,7,5) | 1.0 | 1 | 4.43e-01 | 1.051 | 1.0 | **15,455** |

**The test.** Sweep `N` across a grid that straddles these values and locate where
`xi(N)` crosses `xi_needed(N)`. The predictions span three orders of magnitude and
move with `c` as `c^2` and with `beta` as `beta^-4`, so this is falsifiable in a
way the toy version never was: a grid that brackets 1,647 and 1,830,412 cannot
return "covered everywhere" if the prediction is right, and cannot return crossings
at the predicted places by accident.

**What would refute it.** No crossing anywhere on a grid reaching `10^7`; or
crossings that do not move with `c` as `c^2`; or crossings present in the
*complete* configurations too, which would show they are driven by something other
than `beta_pop`. The last is the important control and must be run: `(2,6,4)` at
`confound = 0.6` has `beta_pop = 0` and should show **no** crossing at any `N`.

## 5. Measurements

*(added after §1–§4 were committed)*

## 6. Files

| File | Role |
|---|---|
| `poc/run_completeness_beta.py` | §3, and the population quantities behind §4 |
| `experiments/results_completeness_beta.json` | Raw |
