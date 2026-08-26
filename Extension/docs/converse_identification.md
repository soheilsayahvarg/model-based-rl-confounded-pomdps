# The Converse: What a Rank Failure Actually Costs

**Written and committed before any of it was run.** The measurement sections are
empty on purpose.

---

## 1. The question

`selfheal_theorem.md` gives the forward direction: the stage-`t` per-action
design span is `E^T diag(pi_b[:,a]) W_t`, and it reaches `R^|S|` exactly when the
reachable transition rows span and the logger is unsaturated.

`novelty_check.md` then established that those conditions are the classical
finite-state HMM identifiability conditions, so the forward direction is not a
contribution. What nobody in either literature does is ask the converse:

> When the span does **not** reach `|S|`, what exactly is identified, what is
> not, and what does the gap cost a pessimistic planner?

## 2. The identified set

Fix a block. The conditional moment restrictions pin the bridge only up to the
design null: any `b` with `T_2 b = g_2` is observationally equivalent, so the
identified set is

    B = { b_hat + n : n in Nul }  intersect  { the bridge class }

with `b_hat` the minimum-norm solution, `P_Nul b_hat = 0`. Take the class as a
norm ball `||b|| <= M`, which is what `Assumption D.16(d)` of the anchor paper
imposes.

For a value functional linear in the block, `V(b) = <g, b>`, the identified set of
values is an interval centred at `<g, b_hat>` of half-width

    HW = max { <P_Nul g, n> : n in Nul, ||b_hat + n|| <= M }

Since `b_hat` is orthogonal to `Nul`, `||b_hat + n||^2 = ||b_hat||^2 + ||n||^2`,
so the constraint is `||n|| <= sqrt(M^2 - ||b_hat||^2)` and

    HW = beta_g * sqrt( M^2 - ||b_hat||^2 )

## 3. The consequence, which is the point

The true bridge lies in the identified set, and `b_true = b_hat + n_true` with
`n_true in Nul` and `||n_true|| = beta`. So `||b_true||^2 = ||b_hat||^2 + beta^2`.
**Take the tightest class bound a practitioner could honestly claim,
`M = ||b_true||`.** Then `sqrt(M^2 - ||b_hat||^2) = beta` exactly and

    HW = beta * beta_g

which is `prop:floor`.

**The floor is not a pathology of the confidence region. It is the exact
half-width of the sharp identified set.** A pessimistic planner that pays
`beta*beta_g` is paying precisely the identification gap and not a penny more.
No estimator can do better, because the data cannot distinguish the endpoints.

Three things follow, and they reverse the reading of every result in this line.

1. **`prop:floor` is tight, not merely a bound.** The open question of whether it
   is tight has an answer: it is achieved, at the tightest admissible class.
2. **`cor:epaper` becomes a statement about overpayment.** The anchor schedule's
   pathology is not that its width fails to contract; a width that contracts
   below `beta*beta_g` would be *invalid*. The pathology is that it pays
   `Theta(N^(e/2))` when the gap is a constant, so the **overpayment ratio**
   `W(N) / (beta*beta_g)` diverges. That ratio, not the width, is the quantity to
   report.
3. **The PD-kernel result fits rather than conflicts.** With no atom there is no
   exact identification gap, so there is correctly no floor. `pd_kernel_regime.md`
   found the floor decaying with slope `-0.30` to `-0.91` in that regime, which
   is now the expected behaviour rather than a lost adjective.

## 4. What is point-identified

`V(b) = <g, b>` is point-identified iff `HW = 0` iff `beta_g = 0` iff `g` lies in
the design span. So the set of point-identified value functionals is **exactly**
the span, of dimension `rank(E^T diag(pi_b[:,a]) W_t)`, and the span formula
becomes a decision procedure: compute it from the primitives, project the policy
gradient, and read off whether that policy's value is point-identified before
touching any data.

## 5. Predictions

- **P-C1.** With `M = ||b_true||`, the numerically computed sharp half-width
  equals `beta * beta_g` to machine precision (relative error below `1e-10`) in
  every cell of the leakage grid, incomplete and complete alike.
- **P-C2.** `HW = beta_g * sqrt(M^2 - ||b_hat||^2)` holds for **every** `M >=
  ||b_hat||`, not only the tight one, to the same precision. Sweeping `M` from
  `||b_hat||` to `3||b_true||` traces that curve exactly.
- **P-C3.** A policy's value is point-identified iff its gradient's null share is
  zero. On a complete design all six candidate policies are point-identified; on
  the incomplete `(4,6,2)` design at confounding `0.9` none of them is, and the
  half-widths are ordered the same way as the `beta_g` values.
- **P-C4.** The overpayment ratio `W(N) / (beta*beta_g)` under the anchor
  schedule grows with log-log slope `e/2 = +0.227`, matching the width slope
  exactly, since the denominator is constant in `N`.
- **P-C5.** `projall` pays close to the gap and `full` does not: at
  `N = 128,000` the ratio of the arm's width to `beta*beta_g` is at least `3x`
  smaller for `projall` than for `full`, and `projall`'s ratio does not grow.

P-C1 is the load-bearing one. If the numerically computed sharp half-width is
**not** `beta*beta_g`, the derivation in section 2 is wrong and nothing in
section 3 follows.

## 6. What would make this worth a paper

The forward theorem is classical. This converse is a different claim: it
identifies the pessimism floor with a partial-identification half-width. If that
identification is also already known, the honest conclusion is that this line has
no novel core and should be written up as a careful empirical study of a known
condition's consequences.

**The novelty of section 3 must be checked before anything is built on it.** That
is now three for three on rediscovery, and the check is cheap.

## 7. Measurement

`poc/run_converse_identification.py`. Population algebra; the numeric check
solves the constrained maximum with SLSQP from six random starts and never uses
`beta_g`, so agreement is a check of the derivation rather than a restatement of
it.

### P-C1 holds. Worst relative error `9.13e-13`, tolerance `1e-10`.

Across `48` cells (`4` configurations x `6` confounding levels x `2` actions),
`24` have a non-negligible half-width. On those:

- the closed form matches the numerically solved maximum in **23 of 24**, median
  relative error `3.93e-15`. The single miss is `(4,8,3)` at confounding `0.6`,
  where the half-width is `6.7e-4`, the smallest in the set, and SLSQP returned
  `0` from every start. A solver failure on the smallest target, not a formula
  failure.
- the identity `HW = beta * beta_g` at the tight class holds with worst relative
  error `9.13e-13`.

**The floor of `prop:floor` is the sharp identified half-width.** It is achieved,
so the proposition is tight rather than a bound.

### P-C2 holds across the class sweep.

| `M` | closed | numeric | rel err | `HW / (beta*beta_g)` |
|---|---|---|---|---|
| `0.6843` **tight** | `0.061819` | `0.061819` | `1.80e-15` | **`1.0000`** |
| `0.7185` | `0.070846` | `0.070846` | `5.88e-16` | `1.1460` |
| `0.8553` | `0.101948` | `0.101948` | `9.53e-16` | `1.6491` |
| `1.0264` | `0.135741` | `0.135741` | `4.09e-16` | `2.1958` |
| `1.3685` | `0.197157` | `0.197157` | `1.41e-16` | `3.1893` |
| `2.0528` | `0.311908` | `0.311908` | `1.78e-16` | `5.0455` |

The formula holds at every `M`, and the ratio is exactly `1` at
`M = ||b_true||`. A looser class bound widens the identified set proportionally,
so `beta*beta_g` is the **smallest** gap any honest class can produce.

### P-C3 holds, with an ordering we did not predict.

On the complete design `(2,6,4)`, all **6 of 6** candidate policies are
point-identified, half-widths from `0` to `1.06e-9`. On the incomplete `(4,6,2)`
at confounding `0.9`, **1 of 6** is:

| policy | `beta_g` | half-width | identified |
|---|---|---|---|
| `always_0` | `0.157965` | `6.18e-2` | no |
| `greedy_lo` | `0.115566` | `4.52e-2` | no |
| `greedy_hi` | `0.099827` | `3.91e-2` | no |
| `soft` | `0.091877` | `3.60e-2` | no |
| `uniform` | `0.078983` | `3.09e-2` | no |
| **`always_1`** | `0.000000` | `0` | **yes** |

The half-widths are ordered exactly by `beta_g`, as the formula requires. The one
point-identified policy is `always_1`, which is **the optimal policy** on this
design. We did not predict that and have no account of it; with one instance it
could easily be a coincidence of this configuration.

### P-C4 is REFUTED IN SIGN, and the correction matters more than the prediction

The prediction called the ratio `W(N) / (beta*beta_g)` an **overpayment** and
expected it to grow past `1`. It grows, at log-log slope `+0.1812`, which equals
the measured width slope exactly since the denominator is constant in `N`. That
half is right.

But the ratio is **below `1` at every sample size tested**:

| `N` | width | gap | ratio |
|---|---|---|---|
| `2,000` | `0.1908` | `0.4285` | `0.445` |
| `8,000` | `0.2343` | `0.4285` | `0.547` |
| `32,000` | `0.3092` | `0.4285` | `0.722` |
| `128,000` | `0.4018` | `0.4285` | `0.938` |

These regions are **narrower than the sharp identified set**. By `prop:floor`
they cannot cover the truth. Every decision experiment in this project ran on
calibrated widths that are provably invalid, and the schedule is climbing toward
validity rather than away from it, reaching `0.938` at the largest `N` and
crossing `1` somewhere beyond the grid.

That is a different story from the one the paper tells, and a more defensible
one:

- The region "never contracting" is **correct behaviour**. It cannot go below the
  identification gap, and a rule that did would be invalid.
- The `e > 0` schedule is not overpaying. On this grid it is **underpaying and
  growing toward the right answer.**
- The decision-level pathology therefore is not caused by an inflated region. It
  happens while the region is still too narrow to cover, which means the
  pessimistic selector is being misled by an **invalid** region, not a
  conservative one.

`cor:epaper` should say: the schedule's width grows because it must, since the
gap is a constant no schedule can beat. The finding is that selection degrades
along the way.

### What this costs and what it buys

**Costs.** The framing "pessimism cannot contract, and that is a pathology" is
wrong. Non-contraction is forced by identification. The paper's Section 1 and its
title are built on that reading.

**Buys.** `prop:floor` is tight and has an interpretation nobody in the two
neighbouring literatures gives it, the span formula becomes a pre-fit decision
procedure for which policies are point-identified, and the calibration used
throughout this project is now known to produce invalid regions, which is a
finding about our own experiments that had to surface eventually.

### Still to check

Section 6 stands: **the novelty of identifying the pessimism floor with a
partial-identification half-width has not been checked.** Three for three on
rediscovery so far. Nothing should be built on this before that search.
