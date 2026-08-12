# Step (c): A Contraction/Coverage Dichotomy for Proximal Pessimism

**Status: proposition and predictions fixed BEFORE measurement. This document was
committed with §1–§4 complete and §5 empty; the verification numbers were added in
a later commit. The git history is the evidence for that ordering, which is the
point — the two results of ours that survived adversarial review were both derived
analytically before they were measured, and the ones that failed were not.**

---

## 1. What this replaces

Step (b) reported a measured fact: across a 64× increase in data the pessimism
width did not move (`family_pessimism.md` §4). That is a statement about four
numbers in one environment under one ridge schedule. A referee's first question is
whether a different schedule fixes it.

The answer is that no schedule fixes it, and the reason is not a property of the
environment. Both the width and the coverage of the confidence region are governed
by **one exponent**, with opposite signs. Making the region contract is exactly
what makes it drop the truth.

## 2. Setting and hypotheses

Per bridge block, the stage-2 empirical risk is exactly quadratic, so

```
conf(xi) = { b : (b - b_hat)' H (b - b_hat) <= xi },    H = T2_hat + lam * I
```

and for a value functional linear in the block, `V(b) = <g, b>`, the pessimistic
value and its penalty (width) are

```
V_low = <g, b_hat> - W,     W = sqrt(xi) * ||g||_{H^-1}.
```

**(H1) Structural null.** `T2_hat` has an exact null space `Nul` with
`dim >= |O| - min(|O|, |O_0|) > 0`, fixed by the dimensions and independent of `N`.
This is step (a)'s surviving result: it is linear algebra, it holds at every sample
size, and it is checkable before fitting anything.

**(H2) Ridge schedule.** `lam(N) = lam0 * N^-kappa`, `lam0 > 0`, `kappa >= 0`. The
implementation's default is `lam0 = 0.03`, `kappa = 1/2`. The anchor paper's
Theorem 4.2 sets `lam2 = N2^(-gamma/(gamma*c2+1))`, i.e. `kappa = gamma/(gamma*c2+1)`.

**(H3) Width rule.** `xi(N) = c / (N * mu(N))` with `mu(N) = m * N^-tau`. Two
conventions appear in this repository, and both are covered:

| convention | `mu` | `tau` | used by |
|---|---|---|---|
| **global** | `lam_min(H)`, which under (H1) equals `lam(N)` exactly | `tau = kappa` | the model-free layer, `mf_pessimism.py` |
| **signal** | `sigma2_signal`, the smallest *retained* eigenvalue, `-> sigma2 > 0` | `tau = 0` | the model-based layer, `BlockEllipsoid.width_rule` |

**(H4) Non-degenerate leakage.** `g(N) -> g_inf` with `P_Nul g_inf != 0`; write
`beta_g = ||P_Nul g_inf|| > 0`. The truth has a null component
`beta = ||P_Nul b_true|| > 0`.

(H4) is the step (b) result in hypothesis form. When it fails — when the value
gradient is orthogonal to the null space — nothing below applies, and that is the
correct behaviour: a null space the value never looks into is harmless.

## 3. The proposition

Define the **governing exponent**

```
e := tau + kappa - 1.
```

> **Proposition.** Under (H1)–(H4), while the bridge-class constraint is inactive:
>
> **(i) Width.** `W(N) = Theta(N^(e/2))`. Explicitly
> `W(N) >= beta_g * sqrt(c/m) * lam0^(-1/2) * N^(e/2)`.
>
> **(ii) Coverage.** `b_true in conf(xi)` requires `N^e >= K`, where
> `K := lam0 * beta^2 * m / c`.
>
> **(iii) Dichotomy.** Exactly one of the following holds.
> * `e >= 0`: the null-direction coverage requirement is satisfied at every `N`,
>   and `W` is non-decreasing — **the region never contracts**. At `e = 0` it is
>   exactly constant.
> * `e < 0`: `W` contracts at rate `N^(e/2)`, but coverage **fails for every**
>   `N > N* = K^(1/e)`, a finite and computable threshold.
>
> **No choice of `(kappa, tau)` delivers contraction with sustained coverage.**
>
> **(iv) Class-constrained regime.** With `||b|| <= M` imposed, the unconstrained
> minimiser's excursion is `Theta(N^((e+kappa)/2))`. Whenever `e + kappa > 0` the
> ball becomes active at a finite `N`, and thereafter
> `W(N) -> <g_inf, b_inf> + M*||g_inf|| > 0`, a constant. So the penalty has a
> strictly positive limit in every case.

### Proof

**The ridge acts as a scalar on `Nul`.** By (H1), `H|_Nul = lam * I` exactly, so
`(H^-1)|_Nul = lam^-1 * I`. Splitting `g` into signal and null parts,

```
||g||^2_{H^-1} = ||P_S g||^2_{(H|_S)^-1} + ||P_Nul g||^2 / lam  >=  beta_g^2 / lam.
```

The first term is `Theta(1)`: `H|_S` has eigenvalues bounded away from 0 by the
design's own signal spectrum, which converges. So the null term dominates whenever
`lam -> 0`, giving `||g||_{H^-1} = Theta(lam^-1/2)`.

**(i)** `W = sqrt(xi) * ||g||_{H^-1} = sqrt(c/(N*m*N^-tau)) * Theta(lam0^-1/2 * N^(kappa/2))`
`= Theta(N^((tau-1)/2 + kappa/2)) = Theta(N^(e/2))`.

**(ii)** The stage-2 solve is `b_hat = (T2_hat + lam I)^-1 * g2_hat` with
`g2_hat = Psi' p_hat / N2 in range(Psi') = Nul^perp`. Hence

```
P_Nul b_hat = lam^-1 * P_Nul g2_hat = 0    exactly,
```

so `P_Nul (b_true - b_hat) = P_Nul b_true` and

```
xi_needed = (b_true-b_hat)' H (b_true-b_hat) >= lam * beta^2.
```

Coverage therefore requires `c/(N*m*N^-tau) >= lam0*beta^2*N^-kappa`, i.e.
`N^(tau+kappa-1) >= lam0*beta^2*m/c`, i.e. `N^e >= K`.

**(iii)** If `e >= 0` then `N^e` is non-decreasing, so the requirement in (ii) is
sustainable, and by (i) `W` is non-decreasing. If `e < 0` then `N^e -> 0 < K`, and
`N^e >= K` fails exactly for `N > K^(1/e)`. The two cases are exhaustive and
exclusive because they partition the sign of a single real number. **The exponent
that shrinks the width is the same one that breaks coverage.**

**(iv)** The unconstrained minimiser is `b* = b_hat - sqrt(xi) H^-1 g / ||g||_{H^-1}`,
so `||b* - b_hat|| = sqrt(xi) * ||H^-1 g|| / ||g||_{H^-1}`, and on `Nul`,
`||H^-1 g|| >= beta_g/lam`, giving excursion `= Theta(N^(e/2 + kappa/2))`. If
`e + kappa > 0` this diverges, so it exceeds any fixed `M` at a finite `N`, after
which the constraint binds and the ellipsoid is inactive; then
`min_{||b||<=M} <g,b> = -M||g||` and `W = <g,b_hat> + M||g||`, which converges to a
positive constant since `g -> g_inf` and `b_hat -> b_inf`. QED

### Why this is stronger than what step (b) reported

Step (b) said: in this environment, at this schedule, the width was flat. The
proposition says the flatness was **the boundary case `e = 0`**, and that moving
off it in either direction is worse — one way the width grows with data, the other
way the region stops containing the truth at a computable sample size.

It also converts the criticism from an implementation complaint into a statement
about the construction: any method that (a) regularises a rank-deficient bridge
design and (b) calibrates its region against a spectral quantity from that same
design inherits this dichotomy, whatever the estimator.

## 4. Predictions, fixed before measurement

All at `lam0 = 0.03` (the implemented default), `beta_g, beta > 0` by (H4).

**4.1 Width slope.** `d log W / d log N = e/2 = (tau + kappa - 1)/2`.

| convention | `tau` | `kappa` | `e` | predicted slope |
|---|---|---|---|---|
| global (model-free) | `kappa` | 0.25 | −0.50 | **−0.25** |
| global (model-free) | `kappa` | 0.50 | 0.00 | **0.00** (flat) |
| global (model-free) | `kappa` | 0.75 | +0.50 | **+0.25** (grows with data) |
| signal (model-based) | 0 | 0.25 | −0.75 | **−0.375** |
| signal (model-based) | 0 | 0.50 | −0.50 | **−0.25** |
| signal (model-based) | 0 | 1.00 | 0.00 | **0.00** (flat) |

The `kappa = 0.75` global row is the sharpest falsifiable claim: **more data makes
the pessimistic bound strictly worse**, at rate `N^(+1/4)`.

**4.2 Retrodiction, and a correction it forces.** The step (b) table was run under
the global convention at `kappa = 1/2`, so `e = 0` and the width is predicted
exactly flat. Measured: 7.223, 7.684, 7.636, 7.697 across 64× in `N`. Consistent.

The **projected** width removes the null directions, so it is set by `sqrt(xi)`
alone: slope `(tau-1)/2 = (kappa-1)/2 = **-0.25**`, not `-0.5`.
`family_pessimism.md` §4 states the projected width "shrinks at exactly `N^-1/2`".
The measured column is 0.667, 0.471, 0.334, 0.236 over a 64× range, and
`0.667/0.236 = 2.826` against `64^0.25 = 2.828`. **The proposition says `N^-1/4`,
the data says `N^-1/4`, and our own published sentence is wrong.** It is corrected
in that file, by prediction rather than by re-measurement.

**4.3 Coverage crossing.** Under the signal convention at `kappa = 1/2` — which is
what the model-based layer actually runs, at Phase 3's own operating point — the
proposition gives `e = -1/2` and hence a finite

```
N2* = K^(1/e) = ( c / (lam0 * beta^2 * sigma2) )^2.
```

Beyond `N2*` the region provably excludes the truth. This predicts a **crossing of
`xi(N)` and `xi_needed(N)` at a specific `N`**, computable from quantities the fit
already reports. Phase 3 ran at `N = 20,000`, i.e. `N2 = 6,000`. Whether that sits
below or above `N2*` is the test, and the prediction is quantitative, not
directional.

## 5. Measurements

`poc/run_noncontraction.py`, 3 seeds, `N` from 4,000 to 256,000 (64× range).

### 5.1 The two exact structural claims

The proof rests on two statements about the *implementation*, not the mathematics,
and either could have been false:

| claim | predicted | measured |
|---|---|---|
| `lam_min(H) = lam` under (H1) | exact | max relative error **1.25e-12** |
| `P_Nul b_hat = 0` | exact | **7.09e-12** |

Both hold to machine precision, across every fit in the sweep.

### 5.2 Width slope, global convention (model-free)

| `kappa` | `e` | predicted slope | measured | error |
|---|---|---|---|---|
| 0.25 | −0.50 | −0.250 | **−0.2526** | 0.0026 |
| 0.50 | 0.00 | 0.000 | **+0.0040** | 0.0040 |
| 0.75 | +0.50 | +0.250 | **+0.2402** | 0.0098 |

The `kappa = 0.75` row is the claim worth stating on its own. The width goes
72.4 → 100.8 → 139.3 → 197.3 as `N` goes 4,000 → 256,000:

> **More data makes the pessimistic bound strictly worse, at the predicted rate.**

Nothing in the step (b) table suggested that; it falls out of the exponent.

### 5.3 Width slope, signal convention (model-based)

| `kappa` | `e` | predicted | measured | error |
|---|---|---|---|---|
| 0.25 | −0.75 | −0.375 | −0.4141 | 0.0391 |
| 0.50 | −0.50 | −0.250 | −0.2798 | 0.0298 |
| 1.00 | 0.00 | 0.000 | −0.0069 | 0.0069 |

The residuals are larger here, and they are **not** noise: all three are negative,
and they shrink monotonically in `kappa`. The proof says why. The exact finite-`N`
form is `W^2 = xi * (A_N + beta_g^2/lam)` where `A_N = ||P_S g||^2_{(H|_S)^-1}` is
the signal contribution; §3 takes the `lam -> 0` limit, in which `A_N` is
negligible. At finite `lam` the retained `A_N` makes `W` decay slightly *faster*
than the asymptote, and the smaller `kappa` is, the slower `lam` shrinks and the
longer the contamination persists. Predicted ordering: residuals negative, largest
at `kappa = 0.25`, smallest at `kappa = 1.0`. Observed: −0.039, −0.030, −0.007.

**The deviation from the prediction has the shape the prediction implies**, which
is a better check than agreement would have been.

### 5.4 The projected column is NOT verified

| `kappa` | predicted | measured | error |
|---|---|---|---|
| 0.25 | −0.375 | −0.3659 | 0.0091 |
| 0.50 | −0.250 | −0.2137 | 0.0363 |
| 0.75 | −0.125 | −0.0641 | 0.0609 |

Right sign, right order of magnitude, but the underlying series is non-monotone —
at `kappa = 0.75` it runs 2.139, 1.791, 3.565, 1.265. Parallel-analysis rank
selection carries ±1 noise at these sample sizes (`family_pessimism.md` §9), and a
±1 change in the retained dimension moves the projected width discontinuously. The
unprojected column needs no basis at all, which is why it is clean.

**Reported as consistent, not as confirmed.** It needs ~20 seeds. The §4.2
retrodiction is unaffected: it is arithmetic on numbers already published in
`family_pessimism.md`, not a re-measurement.

### 5.5 Coverage — the rate holds, the threshold is out of reach

**The test as designed in §4.3 was vacuous, and is reported as such.** Predicted
crossing at `N* ≈ 1.66e9`; the grid stopped at `N = 1,024,000`, coverage 3/3
everywhere. A grid 1,600× short of the threshold could not have produced a
failure. That is exactly the defect for which step (b) withdrew its "validity
36/36" evidence (`family_pessimism.md` §8.3), and it is not admissible here either.

What *is* falsifiable on a reachable grid is the **rate**. The mechanism is that
the null-direction coverage margin `m_null := xi / (lam * beta^2)` degrades as
`N^e`, hitting 1 when coverage fails:

| `N` | 1,000 | 4,000 | 16,000 | 64,000 | 256,000 | 1,024,000 |
|---|---|---|---|---|---|---|
| `m_null` | 1238.2 | 680.0 | 332.9 | 159.1 | 82.0 | 41.3 |

Slope **−0.4965** against a predicted **−0.5000**, over three decades of `N`.
The margin degrades at exactly the predicted exponent. It is the *level* that puts
the crossing out of reach, and the level is a property of the environment: the
toy's true bridge has only **2.9%** of its norm in the null space
(`beta = 0.0532` against `||b_true|| = 1.803`), so (H4) holds but barely, and
`N* ∝ beta^-4` — an environment with ten times the null share crosses at
`N2* ≈ 50,000`.

**Post-hoc refinement, labelled as such.** The measurements show the *signal*
directions still dominate `xi_needed` by 15× at `N = 1,024,000`, so the §3 bound
`xi_needed >= lam*beta^2` is far from tight here. Retaining the signal term,
`xi_needed ≈ C/N2 + lam0*beta^2*N2^-kappa` with `C = 0.698` measured, gives

```
N2* = ( (c/sigma2 - C) / (lam0*beta^2) )^2,   valid only when c > sigma2*C.
```

That second condition is a **floor on the width constant**: below
`c > sigma2*C = 0.0111` the region fails to cover at *every* `N`, not just large
`N`. Phase 3 operates at `c = 0.03`, a factor of **2.7** above its own floor.
This is more practically relevant than the `N = 1.7e9` threshold — but it was
derived after seeing the data, so it is a hypothesis for the next round, not a
confirmed prediction.

### 5.6 Summary

| claim | status |
|---|---|
| `lam_min(H) = lam`, `P_Nul b_hat = 0` exactly | **confirmed**, machine precision |
| width slope `= e/2`, global convention | **confirmed**, ≤0.010 in all cells |
| width grows with data when `e > 0` | **confirmed**, +0.240 vs +0.250 |
| width slope `= e/2`, signal convention | **confirmed**, with a finite-`lam` residual whose sign and ordering the proof predicts |
| coverage margin degrades as `N^e` | **confirmed**, −0.4965 vs −0.5000 |
| projected slope `= (tau-1)/2` | consistent, **not verified** (PA rank noise) |
| coverage crossing at `N* = K^(1/e)` | **not tested** — grid 1,600× short |
| floor `c > sigma2*C` | **post-hoc**, untested |

## 6. Honest limitations

- **The coverage branch is not the operative failure mode on this environment.**
  It is mathematically real and its rate is confirmed, but with a 2.9% null share
  the crossing sits at `N ≈ 1.7e9`. Any write-up should lead with the width
  branch, which bites at ordinary sample sizes, and state the coverage branch as
  a completeness result with its environment-dependent threshold made explicit.
  Presenting it the other way round would be the same cross-regime flattery the
  regenerated pessimism table already had to correct.
- `beta` is computed from the **exact oracle bridge**. A practitioner cannot
  compute `N*`, because knowing `beta` means knowing the bridge. The proposition
  says the threshold exists and how it scales, not where it is for real data.
- 3 seeds. Adequate for the unprojected slopes (residuals ≤0.010 against a signal
  of 0.25) and inadequate for the projected ones, as §5.4 says.
- Two environments, one per convention: the `(2,6,4)` dimension-separated POMDP
  and the Phase 3 toy. (H1) is dimensional and transfers; (H4) is not, and is the
  hypothesis most likely to fail elsewhere.
- The width rule family `xi = c/(N*mu)` covers both conventions in this repository
  but is not the only possible calibration. A rule that made `mu` depend on the
  *truth* rather than on the design spectrum would escape the dichotomy — and
  would not be implementable.
- The proposition assumes the class constraint is inactive for (i)–(iii); (iv)
  handles the active case but is proved, not measured. The measured evidence for
  the ball-active regime is the `M`-sweep in `norm_constraint.md` §5.

## 7. Files

| File | Role |
|---|---|
| `poc/run_noncontraction.py` | Verification of §4 |
| `experiments/results_noncontraction.json` | Raw results |

## 7. Files

| File | Role |
|---|---|
| `poc/run_noncontraction.py` | Verification of §4 |
| `experiments/results_noncontraction.json` | Raw results |
