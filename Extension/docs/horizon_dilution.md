# Is the Self-Healing Uniform in the Horizon?

**Predictions committed before measurement; §4 was committed empty.**

---

## 1. The question

Round 8 established that at `t = 2` the stage conditioning set includes history,
the per-action span reaches `|S|`, and `beta_pop ~ 1e-15` in 24/24 cells at every
`confound < 1`. So the truth-leak switch is a **stage-1 statement** and an
inadequate negative control is partially self-healing.

That was measured at `t = 2` only, in one horizon (`T = 3`). Two things follow
that nobody has checked, and a referee will ask both:

1. **Is the restoration uniform?** Does the span reach `|S|` at *every* stage
   `t >= 2`, for every horizon, or does it decay again deeper in the trajectory?
2. **How much does the surviving stage-1 leak matter as `T` grows?** If only
   `t = 1` leaks, the floor in the total value is carried by one block out of
   `T`. The decision-relevant scale — how far apart candidate policies are —
   grows with the horizon. So the floor may be diluted.

Question 2 is the uncomfortable one. If the answer is that the floor washes out,
then the decision-level result of the paper is a **short-horizon phenomenon**,
and saying so is mandatory.

## 2. What is being computed

All population algebra, no sampling. At stage `t` the conditioning cells are
`(a_1..a_{t-1}, o_1..o_{t-1}, o_0)`, so their number grows as
`(|O||A|)^(t-1) |O_0|`. The design span for action `a_t` is spanned by
`E^T v` over those cells, with `v` the forward-propagated posterior over `s_t`.
The count is exponential but the span lives in `R^|O|`, so the basis is
accumulated incrementally and re-orthonormalised, keeping memory at `O(|O|^2)`.

Per stage we report the span rank, `beta_t = ||P_Nul b_true,t||` relative to the
block norm, `beta_g,t`, and the product.

For the dilution question we compare

```
total floor  =  sum_t beta_t * beta_g,t          (only t=1 contributes, if P3 holds)
value spread =  max_pi V_true(pi) - min_pi V_true(pi)
```

over horizons `T = 1..6`, and report their ratio. The spread is the right
denominator because selection regret is decided by differences between candidate
policies, not by the absolute value.

## 3. Predictions

| # | prediction |
|---|---|
| P1 | The span reaches `|S|` at **every** stage `t >= 2`, at every horizon up to `T = 6` and every `confound < 1`. The restoration is uniform, not a `t = 2` accident. |
| P2 | `beta_t <= 1e-12` for every `t >= 2`; `beta_1` stays in the 57–77% band for incomplete configurations, matching the 50/50 table. |
| P3 | The total floor is **constant in `T`** — one leaking block regardless of horizon — so it does not compound. |
| P4 | The value spread grows roughly **linearly** in `T`, so `floor / spread` decays roughly as `1/T`. |
| P5 | **Therefore the decision-level effect is a short-horizon phenomenon.** At `T = 3` (where the paper's 20-seed experiment runs) the floor is a large fraction of the spread; by `T = 6` it should be materially smaller. If P4 holds, the paper's §6 must be scoped to short horizons and the claim "the region does not contract" must be separated from "and therefore selection degrades", because only the first is horizon-free. |

P5 is stated first among the things that would hurt. The floor itself
(Proposition 1) is a per-block statement and is untouched by any of this; what is
at stake is whether its **decision-level consequence** survives a long horizon.

## 4. Measurements

`poc/run_horizon_dilution.py`, population algebra throughout. **All five
predictions confirmed, including P5.**

### 4.1 P1 — the restoration is uniform, and the knife-edge is too

Per-action span rank by stage (worst action shown), horizons to `t = 5`:

| config | `cf` | `\|S\|` | t=1 | t=2 | t=3 | t=4 | t=5 |
|---|---|---|---|---|---|---|---|
| (4,6,2) | 0.6 | 4 | **2** | 4 | 4 | 4 | 4 |
| (4,6,2) | 0.9 | 4 | **2** | 4 | 4 | 4 | 4 |
| (4,8,3) | 0.6 | 4 | **3** | 4 | 4 | 4 | 4 |
| (4,8,3) | 0.9 | 4 | **3** | 4 | 4 | 4 | 4 |
| (4,8,3) | **1.0** | 4 | 2 | **2** | **2** | **2** | **2** |
| (2,6,4) | 1.0 | 2 | 1 | **1** | **1** | **1** | **1** |

The restoration is not a `t=2` accident: the span hits `|S|` at **every** stage
after the first, at every confounding level below saturation. And the
`confound = 1` knife-edge is equally uniform in the other direction — history
never helps there, at any stage.

### 4.2 P2 and P3 — only the first block leaks, and the floor does not compound

| config | `cf` | `t` | span | `beta_t` | `beta_g,t` | floor |
|---|---|---|---|---|---|---|
| (4,6,2) | 0.6 | **1** | 2 | **0.7105** | **0.0928** | **6.60e-02** |
| (4,6,2) | 0.6 | 2 | 4 | 5.7e-16 | 6.1e-16 | 3.5e-31 |
| (4,6,2) | 0.9 | **1** | 2 | **0.6938** | **0.3639** | **2.52e-01** |
| (4,6,2) | 0.9 | 2 | 4 | 8.2e-16 | 1.4e-15 | 1.1e-30 |
| (4,8,3) | 0.9 | **1** | 3 | **0.5116** | **0.1869** | **9.56e-02** |
| (4,8,3) | 0.9 | 4 | 4 | 2.7e-15 | 1.6e-15 | 4.3e-30 |

Max `beta_t` over all `t >= 2` cells: **2.69e-15**. Max floor over `t >= 2`:
**4.26e-30** — machine zero squared, as it must be for a product of two machine
zeros.

> **The floor is carried by exactly one block, whatever the horizon.** It does
> not compound along the trajectory. This is a stronger statement than
> "restoration happens at `t = 2`": the later blocks contribute *nothing* to the
> floor at any depth.

### 4.3 P4 and P5 — and this is the uncomfortable one

The absolute floor is constant in `T` by §4.2. The decision-relevant scale is
not:

| config | `cf` | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 |
|---|---|---|---|---|---|---|---|
| (4,6,2) | 0.6 | 0.425 | 0.161 | **0.099** | 0.072 | 0.056 | 0.046 |
| (4,6,2) | 0.9 | **1.627** | 0.615 | **0.379** | 0.274 | 0.215 | 0.177 |
| (4,8,3) | 0.6 | 0.390 | 0.205 | **0.141** | 0.108 | 0.088 | 0.074 |
| (4,8,3) | 0.9 | 0.505 | 0.266 | **0.183** | 0.140 | 0.114 | 0.096 |

(floor divided by the spread of true values across six candidate policies.)

Log–log slopes against `T`: **−1.234** and **−0.927**. P4 predicted about −1;
both configurations bracket it. The ratio falls by roughly **10×** from `T = 1`
to `T = 6`.

Two readings, and the paper needs both.

1. **At `T = 1` with strong confounding the floor exceeds the entire value
   spread** (1.627). Pessimism there cannot discriminate between candidate
   policies at all — the irreducible penalty is larger than the whole range it
   would have to resolve.
2. **The damage dilutes as `1/T`.** At the `T = 3` where the paper's 20-seed
   decision experiment runs, the floor is still 10–38% of the spread, which is
   why the effect is visible. By `T = 6` it is 5–18%.

> **P5 confirmed: the decision-level consequence is a short-horizon phenomenon.**
> Proposition 1 is a per-block statement and is untouched — the region still does
> not contract, at any horizon. What decays is the *fraction of the decision it
> distorts*. Those two claims must be separated in the paper, because only the
> first is horizon-free.

### 4.4 What this changes

This is partly good news for the method and must be reported that way. An
inadequate negative control is self-healing after the first stage, and its
residual damage is diluted linearly by the horizon. The regime where proximal
pessimism is genuinely unusable is **short-horizon, strongly-confounded,
weak-instrument** — which is narrower than the paper implied before this run.

It also supplies the answer to the first question a referee would ask about
\S6, which previously had none.

---

## 5. Testing the dilution at the decision level

§4 predicts dilution from population algebra. The paper's decision experiment
runs at `T = 3`. **The direct test is to re-run it at `T = 6` and see whether the
effect shrinks by the predicted factor.** That is the check the paper currently
lists as its sharpest open item, so it should be run rather than listed.

### 5.1 What is predicted, committed before the run

From §4.3, for `(4,6,2)` at `confound = 0.9` the floor-to-spread ratio moves
`0.379 -> 0.177` between `T = 3` and `T = 6`, a factor of **0.467**.

At `T = 3`, schedule C (`kappa = 1.5`), `N = 128,000`, the measured pessimistic
regret is `0.1804` against a value spread of `0.6659`, i.e. a **normalized
regret of 0.271**.

| # | prediction |
|---|---|
| P6 | Normalized pessimistic regret at `T = 6`, same schedule and `N`, is about `0.271 * 0.467 = 0.127` — **roughly half**. Accept within a factor of ~1.6 either way, since regret is discrete over six policies and cannot track a continuous ratio exactly. |
| P7 | **RAW regret will not show this.** Since the spread grows `0.666 -> 1.429` (2.15x) while the ratio falls 0.467x, raw regret is predicted at `0.127 * 1.429 ~ 0.18` — essentially unchanged from `0.1804`. Anyone comparing raw regret across horizons would wrongly conclude there is no dilution. This is the trap and we are calling it before seeing the numbers. |
| P8 | The mechanism is unchanged, so the **sign structure survives**: pessimistic and plug-in slopes still have opposite signs under both `e > 0` schedules. |
| P9 | The plug-in still reaches `0.000` regret at large `N` under `kappa = 1.5`. |
| P10 | The crossing under the anchor schedule persists but is weaker — it should occur at a similar or larger `N`, not disappear. |

P7 is the one worth stating loudest. If it holds, then the correct way to report
horizon effects in this paper is exclusively in normalized units, and any raw
regret table is misleading across horizons.

### 5.2 Measurements — P6 and P7 are both REFUTED, in opposite directions

`poc/run_regret_horizon.py`, `HORIZON=6`, 20 seeds, two schedules.

**First, a defect in my own prediction.** §4.3 computed floor/spread using a
*randomly generated* six-policy set, while the decision experiment uses six
hand-built policies. Recomputing on the set the experiment actually uses:

| | `T=1` | `T=2` | `T=3` | `T=4` | `T=5` | `T=6` |
|---|---|---|---|---|---|---|
| spread | 0.0969 | 0.2038 | 0.2966 | 0.4091 | 0.5148 | 0.6234 |
| floor/spread | 4.42 | 2.10 | **1.44** | 1.05 | 0.83 | **0.687** |

The **dilution law survives this**: the `T=3 -> T=6` factor is `0.476` against the
`0.466` predicted from the other set, and the spread still roughly doubles. What
changes is the level, not the exponent. So §4's Corollary stands.

But every number in P6 and P7 was computed from the wrong spread, so both
predictions were mis-stated before the run. Correcting them first, then testing:

| | predicted from the wrong spread | correct prediction |
|---|---|---|
| `T=3` normalized regret | 0.271 | `0.1804/0.2966` = **0.608** |
| `T=6` normalized (P6) | 0.127 | `0.608 * 0.476` = **0.290** |
| `T=6` raw (P7) | 0.180 | `0.290 * 0.6234` = **0.181** |

**Measured, schedule C (`kappa=1.5`):**

| `N` | pessimistic | plug-in |
|---|---|---|
| 2,000 | 0.2257 ± 0.0290 | 0.2128 ± 0.0155 |
| 8,000 | 0.2941 ± 0.0450 | 0.1171 ± 0.0458 |
| 32,000 | 0.3770 ± 0.0000 | 0.0425 ± 0.0639 |
| 128,000 | **0.3770 ± 0.0000** | 0.0623 ± 0.0841 |

> **P7 refuted.** Raw regret at `T=6` is **0.3770**, not `0.181`. It more than
> **doubled** against `T=3`'s `0.1804`.
>
> **P6 refuted.** Normalized regret is `0.3770/0.6234` = **0.605**, against
> `T=3`'s `0.608`. It did not halve — **it did not move at all.**

### 5.3 Why: the same mistake, at a higher price

The modal pick is `uniform` at `N >= 8,000` for **both** horizons. Pessimism makes
the *identical selection error* at `T=3` and `T=6`; `uniform`'s regret against the
optimum is simply larger at the longer horizon (`0.377` vs `0.180`). The
per-block width barely moves either — `0.4018` at `T=6` against `0.4070` at
`T=3`.

So the decision damage is governed by **which policy the penalty promotes**, not
by the floor-to-spread ratio. That ratio halves; the selection error does not
change; the cost of that error scales with the horizon exactly as the spread
does. The two effects cancel and normalized regret is flat.

> **This refutes the decision-level reading of §4, not §4 itself.** The floor is
> constant in `T`, the spread grows, and their ratio decays as `1/T` — all
> measured, all still true. The inference *"therefore the decision-level
> consequence is a short-horizon phenomenon"* does not follow and is false. A
> lower bound on one block's width does not predict which of six discrete
> policies a pessimistic selector will promote.

### 5.4 What survived

| # | prediction | verdict |
|---|---|---|
| P6 | normalized regret halves | **refuted** — flat (0.608 → 0.605) |
| P7 | raw regret unchanged | **refuted** — doubled (0.180 → 0.377) |
| P8 | opposite-signed slopes survive | **holds** — `+0.0387` vs `−0.0379` (sched. C), `+0.0407` vs `−0.0209` (anchor) |
| P9 | plug-in reaches 0.000 at large `N` | **weakened** — `0.0623 ± 0.0841`, interval covers zero but the point does not |
| P10 | crossing persists at similar or larger `N` | **half wrong** — it persists, but moves *earlier*: between `2,000` and `8,000` at `T=6`, against `8,000`–`32,000` at `T=3` |

P8 is the load-bearing survivor: the mechanism is intact at both horizons, and
under `e>0` more data still converges the plug-in while diverging the pessimistic
selection built on the same fits.

### 5.5 Consequence for the paper

Corollary 4 keeps its statement about the floor and **loses its decision-level
sentence**. The honest replacement is stronger, because it is measured rather
than inferred: *the pessimistic selector commits the same error at every horizon
tested, and the cost of that error grows with the horizon, so its relative damage
is horizon-invariant.* The claim we were about to make — that long horizons are
safe — is wrong, and would have been the kind of reassurance a practitioner acts
on.
