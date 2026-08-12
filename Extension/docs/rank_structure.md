# Where the Null Space Actually Comes From

**Status: first result complete. The hypothesis we set out to confirm was refuted,
and replaced by a sharper one that we can predict exactly.**

---

## 1. What we expected

The Phase 4 paper diagnosed the anchor paper's pessimism blow-up as a structural
null space, attributed to `|O| > |S|` — the bridge is a function on the
observation space, but every moment restriction factors through the latent state:

```
p(o_t, o_0 | a) = sum_s p(o_t | s) p(o_0 | s) p(s | a)
```

The plan was to show this is a **family-level** property: any proximal method
whose bridge lives on observations inherits it, model-based or model-free. If so,
"the anchor paper's Eq. 17 is broken" becomes "pessimism over proximal bridge
confidence regions is structurally ill-posed", which is a much larger claim.

## 2. Why the existing environment could not test it

The Phase 3 toy fixes `|S| = 2`, `|O| = 3`, `|O_0| = 2`. Three different
explanations — capped by the latent dimension, capped by the instrument
dimension, capped by their minimum — all predict rank 2 there. **Any conclusion
about the mechanism drawn from that environment is unidentified.**

`src/envs/dim_separated_pomdp.py` varies the three dimensions independently while
holding the confounding channel, reward structure and horizon fixed. With
`|S|=2, |O_0|=4` the latent explanation predicts 2 and the instrument explanation
predicts 4; with `|S|=4, |O_0|=2` the predictions swap. A single mechanism has to
fit both.

## 3. The population limit says one thing

`population_cross_moment` computes `E^T diag(p1) K0` in closed form. Its rank is
`min(|S|, |O|, |O_0|)` in all four configurations tested — exactly as the
factorization argument requires.

So in the **population**, the latent bottleneck does cap the rank.

## 4. The finite-sample design says something else

Fitting a log-log slope of each eigenvalue against `N` separates a direction that
converges to a nonzero limit (slope ≈ 0) from one that is sampling noise around
zero (slope ≈ −1). This is scale-free, so it does not depend on a rank threshold.

`|S|=2, |O|=6, |O_0|=4` — population rank 2, shape cap 4:

| index | N=4,000 | N=256,000 | slope | tier |
|---|---|---|---|---|
| 0 | 8.075e-02 | 8.166e-02 | 0.00 | signal |
| 1 | 7.058e-06 | 2.343e-07 | −0.80 | statistically empty |
| 2 | 2.206e-06 | 3.218e-08 | −0.93 | statistically empty |
| 3 | 2.942e-07 | 3.029e-09 | −1.09 | statistically empty |
| 4 | 1.635e-19 | 5.848e-20 | — | exact zero |
| 5 | −1.855e-18 | −9.332e-20 | — | exact zero |

The spectrum has **three tiers, not two**, and the boundary between the bottom
two is where the mechanism lives.

## 5. The finding

**The exact null space is set by the instrument's shape, not by the latent state.**

```
dim(exact null space) = |O| - min(|O|, |O_0|)
```

Confirmed exactly in 3 of 3 configurations: `(2,6,4) -> 2`, `(3,7,5) -> 2`,
`(4,6,2) -> 4`. This is linear algebra, not asymptotics — the cross-moment has
only `|O_0|` columns, so it cannot span more than `|O_0|` directions at **any**
sample size, for any `|S|`.

**The latent bottleneck produces a softer failure.** Directions between the shape
cap and the signal decay at `N^-0.8` to `N^-1.1` — consistent with `N^-1`, the
sampling rate of a squared mean. They are statistically empty rather than
structurally empty. A confidence region built on them is not infinitely wide; it
is finitely but uselessly wide, and it shrinks too slowly to matter.

So the original hypothesis is **refuted in its strong form**. The two effects are
real but distinct, and only the first is exact.

## 6. A second finding we did not go looking for

Population rank overstates usable rank:

| config | population rank | directions above the sampling floor |
|---|---|---|
| `|S|=2, |O|=6, |O_0|=4` | 2 | **1** |
| `|S|=3, |O|=7, |O_0|=5` | 3 | **2** |
| `|S|=4, |O|=6, |O_0|=2` | 2 | 2 |

In the first two, a direction that genuinely exists in the population never rises
above sampling noise at `N` up to 256,000. The latent directions are individually
weak, and counting them is not the same as being able to use them.

## 7. What this costs us — a problem in our own repair

Signal-Projected Pessimism selects its subspace with an **eigengap rank rule**:
take the largest ratio between consecutive eigenvalues. That rule assumes a wall.
Section 4 shows there is no wall in general — there is a smooth decay across three
tiers.

Running it on the separated dimensions, the rule selects:

| config | eigengap rank per action | signal directions actually present |
|---|---|---|
| `|S|=2, |O|=6, |O_0|=4` | `[5, 5]` | 1 |
| `|S|=4, |O|=6, |O_0|=2` | `[4, 4]` | 2 |
| `|S|=3, |O|=7, |O_0|=5` | `[5, 5]` | 2 |
| `|S|=2, |O|=3, |O_0|=2` (Phase 3 toy) | `[2, 2]` | 2 |

It is correct **only** in the Phase 3 toy — the one configuration where the tiers
collapse because `|S| = |O_0|`. Everywhere else it overshoots, admitting
statistically empty directions into the "signal" subspace.

The estimator's own source comment asserts the design has "rank-`|S|` SIGNAL".
That is true of the population matrix and false of the empirical one, and the
distinction was invisible in the environment where the rule was validated.

This does not retract the Phase 4 results: in that environment the rule selected
the right subspace, and the reported pessimism numbers stand. What it retracts is
the **generality** of the rule.

## 8. Where this leaves the larger claim

The family-level argument survives in modified form, and is now sharper:

- The **exact** ill-posedness is a property of the proxy geometry (`|O|` vs
  `|O_0|`), so it applies to any proximal method with a bridge on observations and
  an instrument on a coarser proxy — model-based or model-free. It is checkable
  from the dimensions alone, before fitting anything.
- The **latent** bottleneck adds a second, softer layer that no rank rule can
  cleanly separate, because it decays continuously.
- Therefore a subspace-projection repair cannot be made general by a better
  eigengap rule. The boundary it needs does not exist as a gap.

That last point is a stronger negative result than the one we started with, and it
implicates our own repair rather than only the anchor paper's method.

## 9. Honest limitations

- **Tabular, one environment family.** The dimension-separated POMDP is
  synthetic; the emission and negative-control matrices are Dirichlet draws with a
  diagonal boost. Other spectra may distribute the weak directions differently.
- **The model-free design here is the stage-1 cross-moment**, extracted to mirror
  `MinimaxValueBridgeOPE`'s per-action solve. We have **not yet** run the full
  model-free pessimism to observe a divergence — that is step (b), and until it is
  done the family claim rests on shared geometry rather than a shared blow-up.
- **`N` up to 256,000, 3 seeds, `T=3`, 2 actions.** The "statistically empty"
  label is relative to that budget; a direction invisible at 256,000 samples is
  not proven absent.
- The slope classifier uses a −0.5 divider and a `1e-12` relative floor. Both are
  visible in the raw table above so a reader can apply their own.

## 10. Files

| File | Role |
|---|---|
| `src/envs/dim_separated_pomdp.py` | Environment with independent `|S|`, `|O|`, `|O_0|` |
| `poc/run_rank_diagnostic.py` | Population check, model-free and model-based rank sweep |
| `poc/run_rank_verify.py` | Hard-vs-soft discrimination by log-log slope |
| `experiments/results_rank_diagnostic.json` | Raw results |
| `experiments/results_rank_verify.json` | Raw results |
