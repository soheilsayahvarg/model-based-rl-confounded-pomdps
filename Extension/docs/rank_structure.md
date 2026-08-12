# Where the Null Space Actually Comes From

**Status: one claim survives and is stronger than first reported; two are
withdrawn. Corrections are in §7, and they were found by adversarial review, not
by us.**

---

## 1. What we set out to show

The Phase 4 paper diagnosed the anchor paper's pessimism blow-up as a structural
null space, attributed to `|O| > |S|` — the bridge is a function on the
observation space, but every moment restriction factors through the latent state:

```
p(o_t, o_0 | a) = sum_s p(o_t | s) p(o_0 | s) p(s | a)
```

The goal was to show this is a **family-level** property, so that "the anchor
paper's Eq. 17 is broken" becomes "pessimism over proximal bridge confidence
regions is structurally ill-posed."

## 2. Why a new environment was needed

The Phase 3 toy fixes `|S| = 2`, `|O| = 3`, `|O_0| = 2`. Capped-by-latent,
capped-by-instrument and capped-by-their-minimum all predict rank 2 there, so
the environment cannot identify the mechanism. `src/envs/dim_separated_pomdp.py`
varies the three dimensions independently.

## 3. What survives — the exact null space is set by the instrument's shape

```
dim(exact null space) >= |O| - min(|O|, |O_0|)
```

Confirmed in every configuration: `(2,6,4) -> 2`, `(3,7,5) -> 2`, `(4,6,2) -> 4`.
This is linear algebra rather than asymptotics — the cross-moment has only
`|O_0|` columns, so it cannot span more than `|O_0|` directions at **any** sample
size, for any `|S|`.

The relation is an **inequality**, not an equality. If the sample does not
realize every instrument value the empirical null space is larger. Our first
draft wrote this as an equality "at any sample size", which is wrong in the rare
degenerate case.

## 4. What survives, and is stronger than we first reported — three tiers

Fitting a log-log slope of each eigenvalue against `N` separates a direction
converging to a nonzero limit (slope ≈ 0) from sampling noise around zero
(slope ≈ −1). The spectrum has **three tiers**: signal, statistically empty
(decaying), and exact zero.

We reported the middle tier as decaying at `N^-0.8`, hedged as "consistent with
`N^-1`". With 20 seeds instead of 3 the rate is **exactly `-1`**. The `-0.80` was
seed noise, and the hedge was unnecessary: the prediction is cleaner than the
measurement we published.

## 5. Two failure modes, and only one is exact

- The **instrument shape gap** produces machine-zero directions. Structural,
  present at every `N`, detectable from the dimensions before fitting anything.
- The **latent bottleneck** produces directions decaying at `N^-1`:
  statistically empty rather than structurally empty. A confidence region built
  on them is not infinitely wide, but finitely and uselessly wide.

## 6. What the family-level claim now rests on

The geometric argument holds for any proximal method with a bridge on
observations and an instrument on a coarser proxy, model-based or model-free,
because it depends only on the shape of the cross-moment. We have **not** yet run
the model-free pessimism to observe an actual divergence. Until step (b) is done
the claim rests on shared geometry, not on a shared blow-up.

---

## 7. Corrections

Both were produced by an independent adversarial review that attacked these
claims by running code. Both were then reproduced independently before being
accepted here.

### 7.1 WITHDRAWN: "population rank overstates usable rank"

We reported that at `|S|=2, |O|=6, |O_0|=4` the population rank is 2 but only one
direction rises above the sampling floor, and presented this as a property of
proximal estimation.

**It is an artifact of our own environment.** `default_params` uses
`confound=1.0`, which makes the behavior policy rows exactly one-hot:

```
pi_b = [[1, 0],
        [0, 1]]
```

The behavior policy is then a deterministic function of the latent state. Every
design matrix is built **per action**, so conditioning on the action *conditions
on the latent state*. The relevant population object is not the unconditional
cross-moment we computed but

```
P_a = E^T diag(p1 * pi_b[:, a]) K0
```

whose rank is the number of latent states that can select action `a`:

| config | unconditional rank (what we reported) | per-action rank (correct) | signal directions observed |
|---|---|---|---|
| `\|S\|=2, \|O\|=6, \|O_0\|=4` | 2 | **[1, 1]** | 1 |
| `\|S\|=3, \|O\|=7, \|O_0\|=5` | 3 | **[2, 1]** | 2 |
| `\|S\|=4, \|O\|=6, \|O_0\|=2` | 2 | **[2, 2]** | 2 |

The observed counts match the per-action rank exactly. The direction we described
as "buried below the sampling floor" has population value **exactly zero** in the
per-action population. Nothing was buried; it was absent.

Causal confirmation: at `confound=0.6` and `0.9` the same direction reappears as
clean signal (slope `-0.02`) at the same `N`, while at `1.0` it decays
(slope `-0.73`). The behavior policy, not the estimator, was producing the effect.

Two further consequences:

- Averaging the spectrum over actions hid an asymmetry. In `(3,7,5)` the
  per-action ranks are `[2, 1]`, not a single number.
- The grid row labelled "Phase 3 toy dims" matches the toy's *dimensions* but not
  its behavior policy, which is stochastic (`0.75/0.25`). It is not the Phase 3
  toy and should not have been labelled as such.

### 7.2 WITHDRAWN: "no eigengap rule can work"

We claimed the smooth three-tier decay means the boundary a projection repair
needs "does not exist as a gap", and that no better eigengap rule could fix it.

**The wall exists. The implementation was broken.** In
`bridge_estimator.py:169`:

```python
ratios = desc[:-1] / np.maximum(desc[1:], 1e-300)
```

Trailing eigenvalues of a positive-semidefinite matrix are machine noise around
zero and are frequently **negative**. `np.maximum(negative, 1e-300)` returns
`1e-300`, so every negative entry yields a ratio of order `1e282`, and `argmax`
selects float sign noise rather than the real gap. The ranks we reported
(`[5,5]`, `[4,4]`) are not the rule's judgement — they are an artifact, and they
are not stable across seeds.

Replacing the floor with one relative to the largest eigenvalue
(`1e-9 * lam_max`) makes the rule recover the correct rank in all tested cases.

So §8 of the first draft is wrong: a subspace-projection repair is **not**
impossible in general. The correct statement is narrower and still useful — the
rule as shipped is numerically unsound outside the configuration it was validated
on.

### 7.3 What this does NOT change

The Phase 4 pessimism numbers stand. In the real Phase 3 toy the per-action ranks
are `[2, 2]` and both directions carry genuine signal, so the eigengap rule
selected the right subspace there. Note it did so partly by luck: with only one
trailing eigenvalue, the buggy ratio happens to point at the same index as the
true gap.

---

## 8. Consequence for step (b) — read before running it

`sigma2_signal` (`bridge_estimator.py:169`) is the smallest *kept* eigenvalue and
it drives the pessimism widths downstream. Under the current rule it can land on
a statistically empty direction, whose magnitude falls at `N^-1`. Width scales as
`xi = c / (N_2 * sigma2)`, so a `sigma2` that decays like `1/N` makes the width
**stop shrinking** with data.

If step (b) is run before this is fixed, a model-free "family blow-up" would be
manufactured by the bug rather than observed. **Fix the rank rule first, then run
the comparison.** Otherwise the headline result of this extension would be an
artifact — the same error, in the same place, that §7.1 just corrected.

## 9. Required re-runs

Everything in §3–§5 was measured under `confound=1.0` and per-action averaging.
The geometric results do not depend on the behavior policy, but they were not
*shown* to be independent of it. Before any of this is used in a paper:

- re-run all tables at `confound in {0.6, 0.9, 1.0}`,
- report per-action rather than averaging across actions,
- use `P_a = E^T diag(p1 * pi_b[:, a]) K0` as the population benchmark.

## 10. Honest limitations

- Tabular, one synthetic environment family, Dirichlet emissions.
- `N` up to 256,000, `T=3`, 2 actions. "Statistically empty" is relative to that
  budget.
- The model-free pessimism divergence (step b) has not been run.
- The three-tier structure was verified on the model-free cross-moment. Whether
  the model-based stage-1 object `Ma Ma^T` shares it was examined by the review
  and should be read there rather than assumed here.

## 11. Files

| File | Role |
|---|---|
| `src/envs/dim_separated_pomdp.py` | Environment with independent `\|S\|`, `\|O\|`, `\|O_0\|` |
| `poc/run_rank_diagnostic.py` | Population and empirical rank sweep |
| `poc/run_rank_verify.py` | Hard-vs-soft discrimination by log-log slope |
| `poc/critic_*.py` | Adversarial review scripts, one per attack |
| `docs/critic_findings.md` | The full review that produced §7 |
| `experiments/results_rank_*.json` | Raw results |
