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

*(empty until run)*
