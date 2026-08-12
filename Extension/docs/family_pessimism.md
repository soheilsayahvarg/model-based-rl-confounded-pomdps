# Step (b): The Blow-Up Is Shared, and the Driver Is Not What We Assumed

**Status: complete. The family claim is confirmed. The mechanism is sharper than
"the design matrix is ill-conditioned" — conditioning barely moves while the
failure varies by an order of magnitude.**

---

## 1. What was being tested

Phase 4 found that the anchor paper's model-based pessimism diverges, and traced
it to confidence-region directions the data cannot constrain. If that is
geometric rather than a quirk of one estimator, the model-free proximal method of
Shi et al. — different loss (minimax GMM rather than two-stage kernel ridge),
different solver, different bridge (a value bridge rather than reward and
dynamics bridges) — should fail the same way.

The model-free estimator ships without a pessimism layer, so `src/pessimism/
mf_pessimism.py` adds one using the same ellipsoid geometry as the model-based
side.

## 2. Why the test is sharp rather than qualitative

For a linear value functional the pessimistic value has a closed form:

```
V_low = V(b_hat) - sqrt(xi) * ||g||_{H^-1}
```

Divergence needs **two** things at once: an ill-conditioned `H`, **and** a value
gradient `g` that points into the ill-conditioned directions. A null space the
value never looks into is harmless. So we measure the leakage of `g` out of the
retained subspace directly, rather than inferring failure from conditioning.

Rank selection uses parallel analysis, not the eigengap rule. At `N = 4,000` the
eigengap rule needs `N` in the millions in the non-saturating regime, so it would
have manufactured exactly the artifact this experiment is trying to observe.

## 3. A structural point that makes the result stronger

For the model-based chain the value is multilinear across every stage's bridge,
which is what forced the coupled multi-block minimisation and what our Phase 4
compounding hypothesis appealed to. Here,

```
J = sum_o nu1(o) sum_a b_V^[1](a, o)
```

depends on the **first stage bridge only, and linearly**. There is no chain to
compound through. So if this diverges, compounding is definitively excluded and
geometry is isolated as the cause.

## 4. Result: it diverges

`N = 4,000` — the sample size Phase 4 actually uses — 5 seeds.

| config | confound | `V_true` | `V_low` unprojected | `V_low` projected |
|---|---|---|---|---|
| `(4,6,2)` | 1.0 | 2.152 | **−51.56** | −4.72 |
| `(2,6,4)` | 1.0 | 2.022 | **−63.19** | −2.22 |
| `(3,7,5)` | 1.0 | 2.537 | **−50.80** | −1.86 |
| `(4,6,2)` | 0.6 | 2.152 | −12.55 | −8.79 |
| `(2,6,4)` | 0.6 | 2.022 | −8.65 | −7.99 |
| `(3,7,5)` | 0.6 | 2.537 | −7.47 | −7.10 |

(at `c = 10`; the pattern is monotone in `c`)

Against true values near `2`, the unprojected pessimistic value reaches `−63`.
This is the same qualitative signature as the model-based `−1779` reported in
Phase 4, produced by an estimator that shares none of its machinery.

**The family claim is confirmed.** The failure is not specific to the anchor
paper's method.

Validity holds throughout: `V_low <= V_true` in **36 of 36** cells. The bound is
correct, just uselessly loose — the same character as the model-based failure.

## 5. The mechanism is gradient leakage, not conditioning

This is the part we did not expect.

| | `cond(H)` | leak of `g` | width ratio unproj/proj |
|---|---|---|---|
| confound = 1.0 | 1.1e2 – 5.9e2 | **0.348** | **10.62×** |
| confound = 0.6 | 1.4e2 – 4.7e2 | **0.0146** | **1.48×** |

**The conditioning of `H` is essentially the same in both rows.** It does not
predict the failure. What moves by a factor of 24 is how much of the value
gradient lies outside the identified subspace, and the width ratio tracks that,
not the conditioning.

So the correct statement is not "proximal designs are ill-conditioned, therefore
pessimism diverges". It is:

> Pessimism over a proximal bridge confidence region diverges when the value
> gradient has mass in directions the data does not identify. Ill-conditioning is
> necessary but not sufficient; the alignment between the value functional and
> the null space is what decides it.

## 6. What drives the leakage — a link to the coverage result

The leakage is controlled by the **behavior policy's determinism**, i.e. the
confounding strength. The chain is:

1. Stronger confounding makes the behavior policy closer to a deterministic
   function of the latent state.
2. Every design is built per action, so conditioning on the action then
   conditions on the latent state, and the per-action population rank collapses
   (`rank(P_a)` falls to 1 in the extreme).
3. The identified subspace shrinks accordingly.
4. But the value gradient is `nu1`, the initial observation marginal, which is
   spread across **all** observations regardless.
5. So more of it falls outside the identified subspace, and the width explodes.

This connects step (b) to the Phase 4 coverage finding rather than sitting beside
it. There, the pessimism penalty was found to be monotonically inverse to
behavior-policy coverage. Here the same driver appears as a geometric quantity:
poor coverage *is* a collapsed per-action subspace, and the penalty it produces
*is* the gradient leaking out of that subspace. The two results are one
phenomenon seen from two directions.

## 7. Honest limitations

- **The absolute values in §4 depend on the width calibration.** We use
  `xi = c / (N * lambda_min(H))` and sweep `c`. Different calibrations shift the
  numbers. The §5 quantities — leakage and the unprojected/projected width ratio
  — involve no `xi` at all and are therefore calibration-independent. **§5 is the
  robust evidence; §4 is the illustration.**
- The model-free pessimism layer is **ours, not Shi et al.'s**. They do not
  specify one. We built the most natural analogue of the model-based construction
  so the comparison is like-for-like, but a different confidence-region
  construction for the same estimator might behave differently.
- Tabular, `T = 3`, 2 actions, `N = 4,000`, 5 seeds, one synthetic environment
  family.
- The projected values are not certified: as in Phase 4, projection buys
  informativeness by giving up guaranteed coverage of the truth.

## 8. Files

| File | Role |
|---|---|
| `src/pessimism/mf_pessimism.py` | Confidence region, gradient, leakage, parallel-analysis basis |
| `src/envs/dim_separated_pomdp.py` | `dp_value` exact oracle and candidate policies |
| `poc/run_family_pessimism.py` | Driver for [B1]–[B4] |
| `experiments/results_family_pessimism.json` | Raw results |
