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

`poc/run_coverage_incomplete.py`, 3 seeds, `N` from 1,000 to 512,000 (512× range),
`kappa = 0.5`. `T = 1`: the stage-1 reward block is built from `(O_0,O_1,A_1,R_1)`
alone, asserted rather than assumed — `max|dH| = 0.00e+00`, `max|db_hat| = 0.00e+00`
against `T = 3`.

### 5.1 The decisive contrast: `beta` is a constant only where completeness fails

| case | `beta_emp` at `N=1,000` | at `N=512,000` | log-log slope |
|---|---|---|---|
| **treatment** (4,6,2), cf 0.0, incomplete | 1.6829 | 1.6649 | **−0.001** |
| **control** (2,6,4), cf 0.6, complete | 0.2699 | 0.0039 | **−0.660** |

This is the population algebra of §3 showing up in sampled data with no ambiguity.
In the incomplete design `beta` is a genuine constant; in the complete one it
decays to zero, exactly as it did on the Phase 3 toy where we mistook it for an
environment property.

And the constant is the *right* constant. The block spans both actions, so its
`beta` should be the quadrature sum of the per-action population values:

```text
sqrt(1.2108^2 + 1.1428^2) = 1.66494      measured beta_emp = 1.66494
```

Agreement to five digits. The empirical null converges to the population null, and
`beta_pop` predicts what the fit finds.

### 5.2 Coverage — crossings in the treatment, none in the control

Fraction of seeds retaining coverage:

| `N` | c=0.03 | c=0.1 | c=0.2 | c=0.3 | | control, all `c` |
|---|---|---|---|---|---|---|
| 1,000 | 2/3 | 3/3 | 3/3 | 3/3 | | 3/3 |
| 8,000 | 0/3 | 3/3 | 3/3 | 3/3 | | 3/3 |
| 16,000 | 0/3 | 1/3 | 3/3 | 3/3 | | 3/3 |
| 64,000 | 0/3 | 0/3 | 3/3 | 3/3 | | 3/3 |
| 128,000 | 0/3 | 0/3 | 0/3 | 3/3 | | 3/3 |
| 512,000 | 0/3 | 0/3 | 0/3 | 0/3 | | 3/3 |

> **The coverage branch of step (c) now has a confirmed empirical instance.** In a
> structurally incomplete design the confidence region loses the truth as `N`
> grows, at every width constant tested, and the loss is monotone in `N` — more
> data strictly destroys coverage.

**The control is what makes this a test.** `(2,6,4)` at `confound = 0.6` has
`beta_pop = 0` and retains 3/3 coverage at every `N` and every `c` — 40 cells, no
failures, across the same 512× range. Step (c) had no such control, which is why
its coverage claim was worthless even before the `beta` error was found.

The control also behaves as predicted *in form*: with `beta = 0` the coverage
condition reduces to `c/sigma2 >= C`, which contains no `N`, so coverage must be
all-or-nothing in `N` rather than crossing. It is.

### 5.3 The `c^2` signature

`N* ∝ c^2` is the falsifiable part, and it is free of `sigma2` and `beta`, which
cancel out of the scaling:

```text
d log N* / d log c   measured +2.158   predicted +2.000
```

The `N` grid steps by factors of 2, so a crossing is located only to within a
factor of 2; the residual 0.158 is inside that quantization.

### 5.4 DISCLOSED DEFECT: the absolute `N*` predictions in §4 were wrong

The predictions in §4 were committed before this run, and **two of them were built
on my own errors**, both found only by checking why the measured crossings sat
*above* a bound that should have been an upper bound:

1. **Wrong `beta`.** §4 used the single-action `beta_pop = 1.211`. The block spans
   both actions, so the right value is the quadrature sum `1.665` (§5.1).
2. **Wrong `sigma2`.** §4's `sigma2_pop` was computed under a different
   normalization from the one `TabularBridgeEstimator` uses for
   `sigma2_signal` — the estimator carries the behaviour-policy factor inside the
   stage-1 profile, my population version normalized it away. They differ by
   **2.02×**, and `N* ∝ sigma2^-2`, so this alone moves `N*` by 4×.

With both corrected, every prediction falls inside the interval the grid brackets:

| `c` | §4 published `N*` | corrected `N*` | measured bracket | inside |
|---|---|---|---|---|
| 0.03 | 1,647 | 1,878 | (1,000, 2,000) | **yes** |
| 0.10 | 18,304 | 20,862 | (8,000, 32,000) | **yes** |
| 0.20 | 73,216 | 83,448 | (64,000, 128,000) | **yes** |
| 0.30 | 164,737 | 187,757 | (128,000, 256,000) | **yes** |

**The correction is post-hoc and is labelled as such.** Four-for-four bracketing
after fixing two constants against the data is weaker evidence than it looks, and
it is exactly the move round 6 caught in the `c`-floor. What was genuinely a
priori, and confirmed, is everything that does not depend on those two constants:
that crossings exist in the incomplete design, that they are absent in the
complete control, that `beta` is flat in one and decays in the other, and that the
crossing scales as `c^2`.

### 5.5 Summary

| claim | a priori? | status |
|---|---|---|
| `beta_pop > 0` iff per-action completeness fails | yes, §3 | **confirmed**, 50/50 cells |
| `beta_emp` constant where incomplete, → 0 where complete | yes | **confirmed**, −0.001 vs −0.660 |
| `beta_emp` = quadrature of per-action `beta_pop` | yes | **confirmed** to 5 digits |
| coverage crossings exist in the incomplete design | yes | **confirmed**, all 4 `c` |
| no crossing in the complete control | yes | **confirmed**, 40/40 cells |
| crossing scales as `c^2` | yes | **confirmed**, +2.158 vs +2.000 |
| absolute `N*` values of §4 | yes | **wrong**; corrected post-hoc (§5.4) |

## 5b. What this does and does not establish

**Does.** The coverage horn of the step (c) dichotomy is real and observable, and
the condition for it is dimensional and checkable before fitting: `|O_0| < |S|`.
Where it holds, more data strictly destroys coverage of the confidence region.

**Does not.** This is a robustness result about a regime the anchor paper
*excludes by assumption*. `|O_0| < |S|` violates Assumption 3.3, and D.16(a)
separately assumes the truth out of the null. Nothing here is a defect in the
paper. The honest claim is: **the construction's guarantee degrades with sample
size precisely when its completeness premise fails, and that premise is checkable
from dimensions alone** — which is useful to a practitioner and is not in the
paper, but is not a contradiction of it.

Still one environment family, 3 seeds, `T = 1`, tabular.

## 6. Files

| File | Role |
|---|---|
| `poc/run_completeness_beta.py` | §3, and the population quantities behind §4 |
| `experiments/results_completeness_beta.json` | Raw |
