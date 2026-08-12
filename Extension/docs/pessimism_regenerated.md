# Every Pessimism Number, Regenerated Correctly

**Status: definitive for this environment. Supersedes `repair_vs_normball.md`
§3 and the magnitudes in `norm_constraint.md`. Three implementation defects and
two reporting faults are fixed here at once.**

---

## 1. What was wrong, and is now fixed

**Solver.** The norm-ball branch scaled one fixed direction out to the *ellipsoid*
boundary. The KKT family is a two-parameter curve and at large `xi` the ellipsoid
is not active at all — measured up to 56% suboptimal, sometimes outside the ball.
Replaced by a certified two-multiplier solver. Verified: the ellipsoid quadratic
form now comes back strictly *below* `xi` at large widths (`q − xi = −0.87`,
`−1.26`, `−2.83`), which the old branch never produced.

**`M` per block.** It was averaged across blocks and applied to all, which put
some blocks' own centres outside their own ball. Now per block.

**`M` admissible.** It was set to `||b_hat||`, argued to be the smallest
admissible value. Ridge shrinks the estimate, so `||b_hat|| < ||b_true||` in every
block/seed pair tested and that ball *excluded the truth*. Now taken from the
exact oracle bridges: `M_R = 1.803`, `M_D = 1.321`, admissible by construction.

**Common `c` grid.** The phase write-ups quote vanilla at `c = 0.1` and projected
at `c = 0.03`. Everything below is on one grid.

**Plug-in reported.** Our own frozen output always contained it; it was not shown
alongside.

## 2. Result

Toy POMDP, `T = 3`, `N = 20,000` (Phase 3's operating point), 3 seeds.
Optimal policy `greedy_lo`, `V_true = 2.0177`.

| `c` | method | selected | regret | `V_low` | contains truth |
|---|---|---|---|---|---|
| **0.03** | plug-in | greedy_lo | **0.0000** | 2.052 | n/a |
| **0.03** | vanilla | greedy_lo | **0.0000** | 1.272 | yes (no class) |
| **0.03** | norm-ball | greedy_lo | **0.0000** | 1.278 | **yes** |
| **0.03** | projected | greedy_lo | **0.0000** | 1.338 | no |
| 0.10 | plug-in | greedy_lo | **0.0000** | 2.052 | n/a |
| 0.10 | vanilla | soft | 0.2565 | 0.123 | yes (no class) |
| 0.10 | norm-ball | soft | 0.2565 | 0.469 | **yes** |
| 0.10 | projected | greedy_lo | **0.0000** | 0.813 | no |
| 0.30 | plug-in | greedy_lo | **0.0000** | 2.052 | n/a |
| 0.30 | vanilla | uniform | 0.5127 | −3.235 | yes (no class) |
| 0.30 | norm-ball | uniform | 0.5127 | −0.806 | **yes** |
| 0.30 | projected | soft | 0.2565 | −0.298 | no |
| 1.00 | plug-in | greedy_lo | **0.0000** | 2.052 | n/a |
| 1.00 | vanilla | uniform | 0.5127 | −20.983 | yes (no class) |
| 1.00 | norm-ball | uniform | 0.5127 | **−3.026** | **yes** |
| 1.00 | projected | uniform | 0.5127 | −4.015 | no |

## 3. The published advantage is a cross-`c` artifact

**At `c = 0.03` — Phase 3's own operating point — all four methods achieve
`0.000` regret, including vanilla.**

The published claim contrasts vanilla's divergence at one width with the
projection's zero regret at another. Put them on the same width and the contrast
disappears. The projection does not rescue selection at `c = 0.03`, because
nothing needed rescuing there.

The projection does have a genuine advantage in the middle of the grid: at
`c = 0.1` it alone still selects the optimal policy, and at `c = 0.3` its regret
is half the others'. That is a real and defensible result — it is simply not the
result that was published.

## 4. No pessimistic variant earns its complexity here

**The plug-in achieves `0.000` regret at every width**, and is never beaten by any
pessimistic method at any `c`. Its `V_low` column is its point estimate, `2.052`
against a true `2.018`.

This is the finding that matters for the paper. Pessimism is machinery for
handling partial coverage; on this environment it costs selection accuracy and
returns nothing, while a confounding-aware plug-in ranks perfectly. Any write-up
that reports the projection's `0.000` without the plug-in's `0.000` alongside is
claiming credit that the baseline already earns for free.

## 5. What the class bound does

It tames the magnitude substantially without changing the selection: `−20.98` to
`−3.03` at `c = 1.0`, `−3.24` to `−0.81` at `c = 0.3`. Selections are identical to
vanilla at every width.

So the catastrophic figures we published as the method's failure mode are largely
an artifact of omitting the class, but the *ranking* failure they were used to
illustrate is not — it survives the correction unchanged.

At `c = 1.0` the norm ball is **both more informative and sound** (`−3.026`,
contains the truth) than our projection (`−4.015`, provably excludes 21% of the
truth's norm). At `c = 0.3` the projection is more informative but unsound. So the
trade is width-dependent and neither dominates.

## 6. What has to change in the phase write-ups

1. Report regret on a **common `c` grid**. The cross-`c` comparison is the single
   most misleading thing in the current text.
2. Report the **plug-in baseline alongside**, since it dominates.
3. Scope `0.000` to the operating point, and note that vanilla achieves it too
   there.
4. Report the norm-constrained magnitudes next to the `−1755` / `−1779` figures,
   which overstate the failure by roughly 7× on this grid.

Identification, estimation and de-biasing results are untouched — none involve
the pessimism layer.

## 7. Honest limitations

- One environment, one candidate set, 3 seeds. The candidate-set question is
  **not** settled here: the review's 200-random-set sweep is the evidence on that,
  and it found the plug-in ahead in 99–100% of sets.
- `M_R` is imposed as an `L2` ball. The paper's Assumption 4.1(f) is a **sup-norm**
  bound, so the literal class is a box. Under the box the rescue is far weaker
  (the review measured `−92` at `c = 10` against `−9` for `L2`), which means the
  §5 magnitudes are a function of the class *shape* we chose. **The box version
  has not been run here and should be before publication.**
- Coordinate descent gives an upper bound on the exact inner minimum for all
  pessimistic variants equally.

## 8. Files

| File | Role |
|---|---|
| `poc/run_pessimism_regenerated.py` | This table |
| `src/pessimism/ellipsoid_opt.py` | Certified ball solver, per-block `M` |
| `experiments/results_pessimism_regenerated.json` | Raw results |
