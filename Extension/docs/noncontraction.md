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

*(added after §1–§4 were committed)*

## 6. Honest limitations

*(added with §5)*

## 7. Files

| File | Role |
|---|---|
| `poc/run_noncontraction.py` | Verification of §4 |
| `experiments/results_noncontraction.json` | Raw results |
