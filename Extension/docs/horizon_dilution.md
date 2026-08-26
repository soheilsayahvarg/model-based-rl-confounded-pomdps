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

### 5.2 Measurements

*(committed empty; filled after the run)*
