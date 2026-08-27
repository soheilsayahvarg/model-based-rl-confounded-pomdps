# Tier 2: Does Any of This Survive a Real Environment, and a Baseline We Did Not Write?

**Written and committed before measurement. The measurement sections are empty
on purpose.**

Two reviewer objections have never been answered in this project:

1. **"It is a toy."** Every result is at `|S| in {2,3,4}` and `T = 3`, with one
   run at `T = 6`.
2. **"Maybe it is your implementation."** Nothing has ever been compared against
   a method we did not write.

## 1. What blocked scale, and why it was not a model problem

`TabularBridgeEstimator` allocates its stage-1 tables densely at `(n_w, n_x)`
with `n_x = |A| (|O||A|)^(t-1) |O_0|`. At `|O| = 10, T = 10` that is `4.1e12`
columns. The cap was never a property of the method. It was an indexing choice.

`x` takes at most `N` distinct values in any sample, so the dense alphabet is
almost entirely empty columns. `ScaledBridgeEstimator` remaps the observed values
to a compact range and changes nothing else.

**Verified, not argued.** `run_estimator_equivalence.py` compares `b_R`, `b_D`,
every block's `H`, `b_hat_vec` and `lam2` across `5` configurations, `2` sample
sizes and `2` seeds, including `T = 4`. Worst absolute difference over `20`
comparisons: **`0.000e+00`**. `TabularBridgeEstimator` is untouched so all `40+`
stored result files stay reproducible.

At `|S| = 8, |O| = 10, T = 10` a fit takes `24-32 s` and carries `19` bridge
blocks against the `5` of every previous run.

## 2. Environment L

`|S| = 8, |O| = 10, |O_0| = 3, |A| = 2, T = 10`, confounding `0.9`. Incomplete by
a wide margin (`3 < 8`). Schedule C (`tau = 0, kappa = 1.5, e = +0.5`).
`N in {4,000, 16,000, 64,000}`, `10` seeds.

Precondition, checked before designing the grid, per the recurring-defect rule:

    value spread (best - worst candidate)  =  0.3979
    beta = 1.779,  beta_g = 0.562,  floor = beta*beta_g = 1.0003

`beta_g > 0`, so the grid can show the effect.

**And immediately a problem for the whole method.** The identification gap is
`1.0003` while the entire spread between the best and worst candidate policy is
`0.3979`. The gap is `2.51x` the spread.

## 3. Predictions

### On scale

- **P-L1.** The anti-monotone failure replicates: `full`'s regret slope is
  positive at `|S| = 8, T = 10`.
- **P-L2.** `plugin` on the identical fits has a negative slope and strictly
  lower regret than `full` at `N = 64,000`.
- **P-L5.** At `T = 10` the empirical null is nonempty at **more than one**
  stage, unlike `T = 3` where only `t = 1` leaked. Reason: `n_x` at late stages
  reaches `15,995` of `16,000`, so late-stage designs are rank-deficient
  everywhere and `cor:selfheal`'s "exactly one block leaks" cannot hold.

### The one that would end the method

- **P-L6.** Because `floor / spread = 2.51 > 1`, **no valid region can separate
  the best candidate from the worst at any sample size.** Pessimistic selection
  at this scale is not degraded, it is impossible. `full`'s regret will therefore
  be flat in `N` at roughly the spread, rather than merely rising.

  If P-L6 holds, the honest headline is not "pessimism degrades with more data".
  It is "**at realistic horizons the identification gap exceeds the entire value
  range, so pessimistic selection carries no information at all**", and the
  `T = 3` results are the mild case.

- **P-L7.** `floor / spread` **grows** with `T` on this configuration, reaching
  `2.51` at `T = 10`. This contradicts `cor:dilution`, which says the floor is
  constant in `T` while the spread grows, so the ratio should **decay** as
  `1/T`. Under `cor:dilution` the `T = 3` ratio would be about `2.51 * (10/3) =
  8.4` and the ratio would fall with `T`.

  P-L7 and `cor:dilution` cannot both be right. Measuring
  `floor/spread` at `T in {3,4,6,8,10}` on one fixed configuration decides it.

### On the baseline

The standard offline-RL pessimism is a count-based lower confidence bound with a
bonus decaying as `N^(-1/2)`, not an ellipsoid driven by a prescribed width
schedule. Adding it answers "maybe it is your implementation" directly.

- **P-L3.** The Hoeffding LCB baseline **converges**: negative regret slope, and
  it does not reproduce the failure. Its bonus has `e < 0` by construction.
- **P-L4.** Therefore the failure is a property of the **schedule** (`e > 0`),
  not of pessimism as such. That scopes the contribution correctly and is a
  weaker claim than the paper currently makes.

### On `projall`

- **P-L8.** `projall` converges at this scale, and its excluded value component
  `c` at `t = 1` is positive, as in the `32` small-environment cells.
- **P-L9.** The concentration explanation is testable here for the first time on
  fresh cells: predict `|cos| ~ 1/sqrt(k_excl)` within a factor of `2` at the
  `t = 1` block of Environment L.

## 4. What would make this a negative result about our own project

If P-L6 holds, then `projall`, `tail`, `proj1` and the entire stage-selective
line are remedies for a regime that does not matter, because at realistic
horizons nothing can be selected pessimistically at all. That would have to be
stated first, not buried.

## 5. Measurement

### P-L7 is REFUTED, 0 of 4 configurations. `cor:dilution` reproduces exactly.

`poc/run_scale_floor_ratio.py`. Population algebra, no fitting.

| config | floor constant in `T` | spread log-log slope | ratio slope |
|---|---|---|---|
| `(8,10,3)` | yes, to `10` decimals | `+1.024` | `-1.024` |
| `(4,6,2)` | yes | `+1.046` | `-1.046` |
| `(6,10,3)` | yes | `+0.974` | `-0.974` |
| `(8,10,4)` | yes | `+1.207` | `-1.207` |

The floor is exactly constant in `T`, the spread grows linearly, and the ratio
decays as `1/T`. That is `cor:dilution`, reproduced at four configurations and
five horizons. Our prediction that it would reverse was wrong.

### P-L6 holds, and it is worse than predicted

| config | `T=3` | `T=4` | `T=6` | `T=8` | `T=10` |
|---|---|---|---|---|---|
| `(8,10,3)` | `8.807` | `6.261` | `4.324` | `3.185` | `2.514` |
| **`(4,6,2)`** | **`1.445`** | `1.047` | `0.687` | `0.512` | `0.408` |
| `(6,10,3)` | `3.421` | `2.567` | `1.732` | `1.312` | `1.057` |
| `(8,10,4)` | `7.595` | `5.130` | `3.209` | `2.273` | `1.757` |

`17` of `20` cells have `floor > spread`. Only `(4,6,2)` at `T >= 6` is separable.

### The consequence for this project's own results

**`(4,6,2)` at `T = 3` is the configuration every decision-level result in this
project used, and its ratio is `1.445`.**

The identification gap is `1.4x` the entire value range between the best and the
worst candidate policy. A valid region cannot be narrower than the gap. So on
that grid **no valid pessimistic rule could have ordered the candidates, at any
sample size.** Not degraded. Impossible.

This is not a small caveat. It is the direct explanation of the anti-monotone
failure, and it is stronger than the explanation the paper currently gives:

- a region wide enough to be valid cannot separate the policies, so it selects
  by tie-break;
- a region narrow enough to separate them is invalid, which is what
  `converse_identification.md` section 8 measured (`pen_t1 / HW_t1` reaching only
  `0.695` at `N = 128,000`);
- there is no width in between.

### But infeasibility is not the whole story, and the `T = 6` run proves it

At `T = 6` the same configuration has ratio `0.687 < 1`. Separation is feasible
there. The stored `T = 6` run (`results_regret_horizon_T6.json`, whose floor
`0.4285` and spread `0.6234` reproduce ours exactly) gives:

| horizon | `floor/spread` | separable | pessimistic regret slope | plug-in slope |
|---|---|---|---|---|
| `T = 3` | `1.445` | **no** | `+0.0220` | `-0.0210` |
| `T = 6` | `0.687` | **yes** | **`+0.0387`** | `-0.0379` |

**Crossing into the feasible regime made the selector worse, not better.** The
slope nearly doubles. So the failure has two separable causes and only one of
them is the identification gap. That is what saves the finding from being an
artefact of an infeasible grid, and it is the sharpest statement this project
has:

> Where separation is impossible, pessimism fails necessarily. Where separation
> becomes possible, pessimism fails anyway, and harder.

`cor:nogovern` said the `1/T` dilution does not reach the decision. This is why:
dilution fixes the **necessary** condition and the selector fails on a different
one.
