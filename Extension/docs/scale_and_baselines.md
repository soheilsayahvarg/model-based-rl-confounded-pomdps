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

## 6. The baseline found a defect in our own regret definition, before it ran

The first thing `run_environment_L.py` printed, before any arm was evaluated:

    behaviour-clone policy value 7.8778  (regret -0.3775)

The behaviour clone, the observation-marginal of the logging policy, is **better
than every candidate policy**, by `0.3775`, which is `95%` of the entire spread
of `0.3979`.

Checked across the project's other configurations:

| config | `T` | best candidate | its value | BC value | BC wins |
|---|---|---|---|---|---|
| `(4,6,2)` | `3` | `always_1` | `2.3157` | `2.3166` | **yes**, by `0.0009` |
| `(4,6,2)` | `6` | `always_1` | `4.6188` | `4.6215` | **yes**, by `0.0027` |
| `(8,10,3)` | `3` | `always_0` | `2.2584` | `2.3662` | **yes**, by `0.1078` |
| `(8,10,3)` | `10` | `greedy_lo` | `7.5004` | `7.8778` | **yes**, by `0.3775` |
| `(6,10,3)` | `10` | `greedy_lo` | `6.7102` | `7.3452` | **yes**, by `0.6350` |
| `(2,6,4)` | `3` | `always_1` | `2.1083` | `2.1657` | **yes**, by `0.0574` |

**Six of six.** The "optimal policy" against which every regret number in this
project is measured is not optimal. It is the best of six hand-made policies, and
a baseline that runs no OPE at all beats it on every configuration tested.

### What this does and does not invalidate

**Does not.** All arms select from the same candidate set, so every
arm-versus-arm comparison is unaffected. The measured facts stand: pessimistic
regret rises, plug-in regret falls, the slopes have opposite signs, `full`
converges to a wrong pick with zero variance.

**Does.** The absolute numbers are regret against a reference that a trivial
baseline beats. On `(4,6,2)` at `T = 3` the margin is `0.0009` against a spread
of `0.2966`, i.e. `0.3%`, so the published small-grid results are numerically
unaffected. At scale it is `95%` of the spread and the defect is fatal to any
absolute claim.

### The fix

`bc` is now in the candidate set, so the arms can select it and regret is
measured against a reference nothing trivial beats. That also makes the baseline
a competitor rather than an unmeasured elephant: an arm that cannot beat "clone
the logs" has not earned its complexity.

This is the seventh entry in `claims.md`'s recurring-defect list, and a fifth
mode: **the reference against which the outcome is measured was never itself
checked against a trivial alternative.** No range guard would have caught it. Only
running a baseline we did not write.

### What adding `bc` to the candidate set did to Environment L

| quantity | six candidates | seven, with `bc` |
|---|---|---|
| optimal candidate | `greedy_lo` at `7.5004` | **`bc` at `7.8778`** |
| spread | `0.3979` | `0.7753` |
| floor | `1.0003` | `1.0003` |
| `floor / spread` | `2.514` | **`1.290`** |

The clone nearly doubles the spread, because it sits `0.3775` above everything
else. The ratio falls from `2.51` to `1.29` and **Environment L is still not
separable.** Even with the best reference the candidate set can supply, the
identification gap exceeds the whole range of achievable values.

That also raises the bar correctly for every arm. The optimal policy is now the
one an analyst gets by cloning the logs, so an arm that does not select it has
lost to a method that runs no OPE at all.

## 7. P-L8 is REFUTED. The sign flips, and `projall`'s safety was an accident

`poc/run_projall_mechanism_L.py`. Excluded value component
`c = <g, P_excl(b_true - b_hat)>` at the t=1 reward block, optimal candidate.

| `N` | seed | `c` | `cos` | `k_excl` | `1/sqrt(k)` |
|---|---|---|---|---|---|
| `4,000` | `0` | **`-0.0627`** | `-0.052` | `280` | `0.060` |
| `4,000` | `1` | **`-0.0572`** | `-0.046` | `280` | `0.060` |
| `4,000` | `2` | **`-0.0476`** | `-0.037` | `280` | `0.060` |
| `16,000` | `0` | **`-0.0571`** | `-0.045` | `280` | `0.060` |
| `16,000` | `1` | **`-0.0540`** | `-0.043` | `280` | `0.060` |
| `16,000` | `2` | **`-0.0565`** | `-0.045` | `280` | `0.060` |

**`c < 0` in 6 of 6 cells**, at both sample sizes and all three seeds, between
`-0.048` and `-0.063`. This is not the numerical zero that the two negative cells
on the small grid were. It is a consistent sign flip.

### This overturns what section 10 of `stage_selective_pessimism.md` concluded

Earlier today that document reported P-C16 refuted, `c > 0` in `30` of `32`
cells, and drew the conclusion "the attack failed, `projall` stands". **That
conclusion was wrong, and the grid was the reason.** `32` cells of one small
family at `T = 3` could not flip a sign that a single larger environment flips in
every cell.

`BlockEllipsoid`'s audit finding A-high said the projected lower bound holds "by
sign-alignment, not by construction". That was right and we under-weighted it.
The correct statement is:

> `projall` is not a certified lower bound and its empirical conservatism does
> not transfer. On `|S| = 8, T = 10` the excluded value component is negative in
> every cell, so the arm's bound sits above where it should and it is
> anti-conservative wherever `|c|` exceeds its own penalty.

### P-L9 holds, and it explains the magnitude but never the sign

All six `|cos|` values fall in `0.037` to `0.052` against `1/sqrt(280) = 0.060`;
mean ratio `0.75`, inside the predicted factor of `2`, on cells the hypothesis
was not fitted to and with `k_excl` spanning `40` to `360` across the `19`
blocks.

So concentration is real: `|c|` is about `1/sqrt(k_excl)` of its Cauchy-Schwarz
bound. **But a concentration argument bounds a magnitude and says nothing about a
sign.** We used it as though it supported both. It does not, and P-L8 is what
that costs.

### The leak spreads with `N`, which bears on P-L5

At `N = 4,000`, seed `0`, the excluded gradient norm `||P_excl g||` is exactly
zero at blocks `7` through `10`. At `N = 16,000` those same blocks carry
`0.046` to `1.02`. More blocks leak as data accumulates, because the late-stage
conditioning alphabet fills in and the design's null structure becomes visible
rather than being masked by empty cells.

## 8. The decision run at `T = 10` is not interpretable, and why that is the finding

The first Environment L run produced

    N=4000  full       regret 0.7076   pen 283,167,900.29
    N=4000  projall    regret 0.7753   pen  87,454,007.84
    N=4000  plugin     regret 0.0000   pen 0.0000        modal bc
    N=4000  hoeffding  regret 0.0000   pen 0.2193        modal bc

A penalty of `2.8e8` against a value scale of `7.9`. Reporting P-L1 from that run
would be the recurring defect again: a grid that cannot show the effect.

### Diagnosis, per block, at `N = 4,000`

| block | `sigma2` | `xi(c=1)` | `\|\|g\|\|_{H^-1}` |
|---|---|---|---|
| `bR_t1` | `7.02e-03` | `1.19e-01` | `1.00e+02` |
| `bR_t4` | `6.19e-03` | `1.35e-01` | `9.38e+01` |
| `bR_t5` | `2.39e-04` | `3.49e+00` | `9.33e+01` |
| `bR_t6` | `5.74e-05` | `1.45e+01` | `1.49e+02` |
| **`bR_t9`** | **`0.00e+00`** | **`8.33e+08`** | `0.00e+00` |
| **`bR_t10`** | **`0.00e+00`** | **`8.33e+08`** | `0.00e+00` |
| **`bD_t9`** | **`0.00e+00`** | **`8.33e+08`** | `0.00e+00` |

`sigma2` is **exactly zero** at the last stages. The anchor paper's `tau = 0`
width rule is `xi ∝ 1/(N_2 · sigma2)`, so it is **undefined** there, not merely
large.

### Our own defect: the floor that hid it

`xi_for` computes `c * N2^(tau-1) / max(sigma2, 1e-12)`. That `max` turns an
undefined quantity into `8.33e8` and lets the run continue. The gradient happens
to be zero at those blocks for the reference policy, so the calibration saw
nothing wrong; but `pessimistic_value` minimizes **jointly**, and once other
blocks move, a region of radius `sqrt(8.33e8) = 2.9e4` at a late block dominates
everything.

**Eighth recurring defect, and a sixth mode: a numerical floor silently converted
an undefined quantity into a finite one.** A guard belongs there, not a floor.

### Why `sigma2` vanishes, which is the actual result

`sigma2` is the smallest kept eigenvalue of the per-action design
`W_a = M_a M_a^T / N_2`, whose columns are `mu_hat(. | x'_n)` for the stage-2
points. The estimator cross-fits: `mu_hat` is built on the stage-1 half and
evaluated at stage-2 histories. At `t = 5` the observed alphabet is already
`3,387` of `N = 4,000`, and by `t = 10` it is `3,999` of `4,000`. Nearly every
stage-2 history was **never seen in the stage-1 half**, so `mu_hat(. | x')` is
identically zero there, `M_a` is zero, and `W_a` is zero.

This is the curse of history hitting the **cross-fitting split**, not the
estimator's capacity. It is exactly the phenomenon Zhang and Jiang's title names,
arriving in the one place a width rule cannot survive it.

### Predictions, committed before measuring

- **P-L10.** `sigma2 = 0` at a block iff its `unseen_x2_frac` is at or near `1`.
  The estimator already records that diagnostic and nothing has ever read it.
- **P-L11.** The breakdown stage `t*`, the first `t` with `sigma2 = 0`, grows
  only **logarithmically** in `N`. Concretely, going from `N = 4,000` to
  `N = 64,000`, a `16x` increase, moves `t*` by at most `2` stages. If so, no
  feasible sample size makes the anchor width rule defined at `T = 10`, and the
  correct statement is that the construction has a horizon ceiling set by the
  split, independent of `N`.
- **P-L12.** Below `t*` the decision experiment is well posed. Running Environment
  L truncated to `T = t* - 1` gives interpretable penalties (order `1`, not
  `1e8`) and a fair test of P-L1 through P-L4.

### Measurement

`poc/run_horizon_ceiling.py`, `3` sample sizes x `2` seeds x `10` blocks.

### P-L10 holds in one direction, not both

`sigma2 = 0` in `6` of `60` block-cells. Among those, `unseen_x2_frac` runs
`0.9975` to `1.0000`, mean `0.9992`. Among the `54` cells with `sigma2 > 0` it
runs `0.0000` to `0.9996`.

So **`sigma2 = 0` implies the stage-2 histories were essentially never seen in
the stage-1 half**, and that direction is sharp. The converse fails: at
`N = 64,000` block `t = 10` has `unseen_x2_frac = 0.9986` and still carries
`sigma2 = 1.23e-05`. Saturation is necessary for collapse, not sufficient.

The diagnostic that predicts it was already being recorded by the estimator and
had never been read.

### P-L11 is REFUTED, and the correction is more useful than the prediction

| `N` | `t*` (first stage with `sigma2 = 0`) |
|---|---|
| `4,000` | `8.5` |
| `16,000` | `10.5` |
| `64,000` | `11.0` (defined at every stage) |

`16x` more data moved `t*` by `+2.5` stages. We predicted at most `2`.

The form of the prediction was right and its conclusion was wrong. `t*` does grow
logarithmically, at **`+0.902` stages per e-fold of `N`**, which means

> each additional stage of usable horizon costs about `3x` the data.

That is exponential in the horizon, but with base `3`, not the base `20` the
alphabet growth `(|O||A|)^(t-1)` would suggest. So `T = 10` is **not** out of
reach: it is undefined at `N = 4,000`, marginal at `N = 16,000`, and fully
defined at `N = 64,000`.

**The honest statement is a cost, not a ceiling.** The construction has a usable
horizon set by `log N` through the cross-fitting split, and buying horizon costs
roughly `3^T` samples. Our claim that no feasible `N` reaches `T = 10` was too
strong.

### Consequence for the decision experiment

A `3`-point `N` grid at `T = 10` would have to start at `N = 64,000` and reach
`N = 10^6`, which is not affordable here. The decision run therefore uses
`T = 5`, where `sigma2` is positive at every block for all three sample sizes,
and the horizon ceiling is reported as its own result rather than folded into it.
