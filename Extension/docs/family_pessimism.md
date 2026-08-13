# Step (b): The Blow-Up Is Shared — But Not for the Reason We First Published

**Status: the load-bearing result (C3, gradient leakage) survives and is now
causal. The headline number was manufactured by our own width rule and is
withdrawn. Two further claims are restated. Corrections in §8, all found by
adversarial review.**

---

## 1. What was being tested

Phase 4 found the anchor paper's model-based pessimism diverges. If that is
geometric rather than a quirk of one estimator, the model-free proximal method of
Shi et al. — minimax GMM rather than two-stage kernel ridge, a value bridge rather
than reward and dynamics bridges, a different solver — should fail the same way.

It ships without a pessimism layer, so `src/pessimism/mf_pessimism.py` adds the
natural analogue of the model-based construction.

## 2. The sharp form of the question

For a linear value functional,

```
V_low = V(b_hat) - sqrt(xi) * ||g||_{H^-1}
```

Divergence needs **both** an ill-conditioned `H` **and** a value gradient `g`
pointing into the ill-conditioned directions. A null space the value never looks
into is harmless. So the measured quantity is the leakage of `g` out of the
identified subspace.

## 3. THE RESULT THAT SURVIVES: leakage is causal

The decisive experiment was not in our first version; the review built it. Sweep
the **emission alignment** while holding `confound = 1.0`, `rank(P_a) = [1,1]`
and `cond(H)` fixed — so the only thing moving is how the gradient sits relative
to the null space:

| leakage of `g` | width ratio unproj/proj | `V_low` |
|---|---|---|
| 0.381 | 15.9× | −63 |
| ... | ... | ... |
| 0.0004 | 1.07× | −4.6 |

Conditioning is constant across that sweep. Leakage moves three orders of
magnitude and the width follows it exactly. **This is calibration-free and
basis-choice-free**, and it is the column the paper should rest on.

> Pessimism over a proximal bridge confidence region diverges when the value
> gradient has mass in directions the data does not identify. Ill-conditioning is
> necessary but not sufficient; the alignment between the value functional and the
> null space decides it.

## 4. THE OTHER RESULT THAT SURVIVES: the bound never tightens

Under our width rule `xi = c / (N * lambda_min(H))`, in an unidentified direction
`H ≈ rho*I` and `lambda_min ≈ rho`, so

```
width = sqrt(c/(N*rho)) * |g_null|/sqrt(rho) = sqrt(c)*|g_null| / (sqrt(N)*rho)
```

and with the schedule `rho = 0.03/sqrt(N)` the `N` **cancels exactly**. Measured:

| `N` | `rho` | `lambda_min` | width unprojected | width projected |
|---|---|---|---|---|
| 4,000 | 4.74e-04 | 4.74e-04 | **7.223** | 0.667 |
| 16,000 | 2.37e-04 | 2.37e-04 | **7.684** | 0.471 |
| 64,000 | 1.19e-04 | 1.19e-04 | **7.636** | 0.334 |
| 256,000 | 5.93e-05 | 5.93e-05 | **7.697** | 0.236 |

Across a 64× increase in data the unprojected width does not move, while the
projected width shrinks at exactly `N^-1/4`. `lambda_min` equals `rho` to every
digit — the rule divides by the regularizer, not by anything the data determines.

> **Correction.** This paragraph originally read `N^-1/2` for the projected
> column. The rate is `N^-1/4`, and the step (c) proposition is what caught it:
> with the null directions removed the width is set by `sqrt(xi)` alone, giving
> slope `(kappa-1)/2 = -1/4` at the implemented `kappa = 1/2`. The published
> numbers already said so — `0.667/0.236 = 2.826` against `64^0.25 = 2.828` — and
> were misread. Found by prediction, not by re-measurement.

**In the unidentified directions the confidence region never contracts.** More
data does not help, at any sample size, ever. This is a stronger and more useful
statement than the value we originally led with.

**Superseded in scope by step (c).** The flatness above is not a property of this
environment or this schedule — it is the boundary case `e = 0` of a one-parameter
dichotomy in which no ridge schedule buys contraction without losing coverage.
See `noncontraction.md`.

## 4b. §3 reproduced independently, and the leakage table at 20 seeds

Both debts this document has carried since it was written are now discharged, and
§3 was reproduced **by a different route than the review used**, which is stronger
than re-running its script.

**The alignment sweep, done in the population.** Since the gradient's profile is
the observation marginal `nu1 = E^T p1` and the design's span is computable in
closed form, the sweep needs no sampling and no rank selection at all. Emission
rows are interpolated toward their mean, holding `confound = 1.0` so the
per-action span stays rank 1 throughout:

| `alpha` | span rank | `cond(H)` | `beta_g` | ratio unproj/proj |
|---|---|---|---|---|
| 0.00 | 1 | 4.53e+02 | **0.7578** | **24.74** |
| 0.40 | 1 | 3.46e+02 | 0.5274 | 11.59 |
| 0.80 | 1 | 2.93e+02 | 0.1926 | 3.50 |
| 0.95 | 1 | 2.86e+02 | 0.0487 | 1.30 |
| 0.99 | 1 | 2.86e+02 | **0.0098** | **1.014** |

Leakage moves **78×**, the width ratio follows it monotonically from `24.7` to
`1.014`, and `cond(H)` moves only **1.58×** across the whole sweep with the span
rank fixed at 1 in every row.

> **§3 survives independent reproduction.** The null space is held fixed and the
> conditioning barely moves; the penalty is set by how the value gradient sits
> relative to that null space, and by nothing else. The review measured
> `0.381 -> 0.0004` with ratio `15.9 -> 1.07`; the constants differ because the
> alignment knob differs, but the structure is identical and here it is exact.

**The leakage table at 20 seeds, against the population truth.** The empirical
PA-based measure agrees with the exact population value in **6 of 8 cells**. The
two disagreements are informative rather than random:

| cell | empirical (20 seeds) | population | direction |
|---|---|---|---|
| (2,6,4) cf 1.0, act 1 | 0.634 ± 0.104 | **0.758** | under-reports |
| (4,6,2) cf 1.0, act 1 | 0.585 ± 0.082 | **0.698** | under-reports |
| (2,6,4) cf 0.6, both | 0.022–0.026 | **exactly 0** | noise floor |

Two lessons, and both change practice rather than just adding error bars:

1. **The empirical measure is biased low exactly where leakage is large**, because
   PA over-selects rank at `confound = 1.0` and an over-selected basis absorbs
   part of the leak. So `family_pessimism.md`'s original PA-based §3 numbers
   understate the effect they were used to demonstrate.
2. **It has a noise floor of about 0.02–0.03** where the true leakage is zero.

**Single-cell empirical leakage should therefore be read as ±0.05 at best, and the
population computation preferred wherever it is available — which is everywhere,
since it is linear algebra.** That supersedes the "re-run at 20 seeds" item: more
seeds do not fix a biased estimator.

## 5. The family claim, correctly scoped

The divergence **is** shared, but the magnitude we published was ours, not the
data's (§8.1). Under the Phase 3 convention (smallest *retained* eigenvalue rather
than the global minimum), on identical fits:

- `(2,6,4)`, non-saturating: `−63.19 → −1.28`. The headline collapses.
- `(4,6,2)`, saturating: divergence **survives both conventions**, `−26.65`
  against a true value of `2.15`, with a clean `1.0` vs `0.6` confounding
  contrast.

So the defensible claim is: **in the saturating regime the model-free method
diverges under either width convention**, and the mechanism (§3) is
convention-independent throughout. The `−63` is withdrawn.

## 6. What the region actually is

Across 40 independent fits, the estimator's true sampling spread of `J` is about
`[1.63, 2.07]` — **two orders of magnitude narrower** than the confidence region
built around it.

That means the object is a **partial-identification region**, not a
sampling-uncertainty region. Both are legitimate, and the anchor paper's
construction is arguably the former, but a paper has to say which it claims. Ours
did not, and the two support very different conclusions about whether the method
is usable.

## 7. Multiplicative compounding is excluded — additive propagation is not

We wrote that there is "no chain to compound through" because
`J = sum_o nu1(o) sum_a b_V^[1](a,o)` depends on the first-stage bridge only, and
linearly. **That is wrong as stated.** The stage-1 response carries the *estimated*
continuation `V_hat_2`, so later stages enter indirectly. Replacing the
continuation with the population bridge (a surgery validated to `0.00e+00` on a
replica) attributes **12% to 45.6%** of the variance of `J` to later stages.

What survives is narrower and still does the work: the **multiplicative**
multi-bridge structure of the model-based chain is absent here. Propagation is
additive and small relative to the identification width. So multiplicative
compounding is excluded as the mechanism; the general phrase is not.

## 8. Corrections

### 8.1 WITHDRAWN: the `−63` headline

`xi = c / (N * lambda_min(H))` uses the **global** minimum eigenvalue, which after
ridging sits at `rho`. Phase 3's model-based rule used the smallest **signal**
eigenvalue, strictly larger. Our choice made `xi` larger and the divergence deeper
than the model-based convention would have produced. The magnitude was set by the
regularizer, not by the data. See §5 for what replaces it.

### 8.2 REFUTED as an identity: "coverage and leakage are one phenomenon"

We claimed the Phase 4 coverage result and this one are the same thing seen from
two directions, via a five-step chain. The alignment sweep of §3 holds coverage
**and** the collapsed subspace fixed while the penalty moves by **14×**.

Coverage collapse creates the degenerate subspace — a necessary condition — but
**gradient alignment is an independent axis that our chain did not contain**, and
it is what sets the penalty. The unification holds only along the
confound-varying slice we happened to look at. Restated as: coverage collapse is
necessary; alignment is what determines the damage.

### 8.3 WITHDRAWN as evidence: "validity 36/36"

This was uninformative. In 8 of 12 cells `V_hat <= V_true` for *every* seed — the
estimator is negatively biased under confounding — so validity is free at any `c`.
In the remaining cells the violation threshold is `c* ≈ 1e-6` against a grid whose
smallest value was `0.1`. **The grid could not have produced a violation.** We will
not report validity again without a grid that could falsify it.

## 9. Honest limitations

- The pessimism layer is **ours, not Shi et al.'s** — they specify none. A
  different confidence-region construction for the same estimator could behave
  differently.
- Rank selection by parallel analysis is 75–100% accurate at `N = 4,000` with
  ±1 noise, so **single-cell leakage values should not be quoted to three
  decimals**, and the §3 table should be re-run at 20 seeds before publication.
- **Provenance:** §4 was derived and measured here. The §3 alignment sweep is the
  review's experiment and its numbers are quoted from `poc/critic_b_*.py`; they
  have not yet been reproduced independently, and must be before publication.
  The §5, §6 and §7 figures likewise come from the review's scripts.
- `nu1` estimated in-sample and `H` built from `t=1` only were both checked and
  make no difference.
- Tabular, `T = 3`, 2 actions, one synthetic environment family.

## 10. What this leaves for the paper

Defensible: the calibration-free causal leakage result (§3), the
convention-robust divergence in the saturating regime (§5), and the
non-contracting width (§4).

Needing rewrite before submission: the `−63` headline, the "no chain" argument,
and the coverage identity. A knowledgeable referee breaks all three with one
experiment each.

## 11. Files

| File | Role |
|---|---|
| `src/pessimism/mf_pessimism.py` | Confidence region, gradient, leakage, PA basis |
| `src/envs/dim_separated_pomdp.py` | `dp_value` oracle, candidate policies |
| `poc/run_family_pessimism.py` | Driver for [B1]–[B4] |
| `poc/critic_b_*.py` | The review's attacks, one per item |
| `experiments/results_family_pessimism.json` | Raw results |
