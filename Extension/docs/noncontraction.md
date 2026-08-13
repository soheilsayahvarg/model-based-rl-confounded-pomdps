# Step (c): A Coverage Floor for Proximal Pessimism

**Status: revised after adversarial round 6 (`critic_findings_stepc.md`). The
width branch survived and is stronger. The coverage branch's quantitative content
on the tested environment is WITHDRAWN. The novelty claim is downgraded to a
restatement with two novel corollaries.**

**Provenance.** §1–§4 of the original version were committed at `53d71a5` with §5
empty, so the predict-then-measure order is auditable; the review verified that
commit and found §1–§4 untouched by the measurement commit. **This revision
post-dates the review and rewrites §3.** The original statement is retrievable at
`53d71a5`; §8 lists exactly what changed and why.

---

## 1. What this is

Step (b) reported that the pessimism width did not move across a 64× increase in
data. That was four numbers, one environment, one ridge schedule. The question a
referee asks first is whether a different schedule fixes it.

None does, and the reason is two lines rather than a rate calculation. But the
result is **not** an indictment of the anchor paper: §3.3 shows the paper's own
schedule already sits on the non-contracting branch, and its Assumption D.16(a)
explicitly excludes the configuration in which the coverage half could bite. The
dichotomy **explains the paper's design choice**; it does not break it.

## 2. Setting and hypotheses

Per bridge block the stage-2 empirical risk is exactly quadratic, so

```text
conf(xi) = { b : (b - b_hat)' H (b - b_hat) <= xi },    H = T2_hat + lam * I
```

and for a value functional linear in the block, `V(b) = <g, b>`,

```text
V_low = <g, b_hat> - W,     W = sqrt(xi) * ||g||_{H^-1}.
```

**(H1) Exact structural null.** `T2_hat` has an exact null space `Nul` with
`dim >= |O| - min(|O|, |O_0|) > 0`, fixed by dimensions, independent of `N`. This
is step (a)'s surviving result. It is exact **because the tabular design is a
finite-rank count matrix**; see §3.4 for what happens with a PD kernel.

**(H2) Ridge schedule.** `lam(N) = lam0 * N^-kappa`, `lam0 > 0`. Implemented
default `lam0 = 0.03`, `kappa = 1/2`.

**(H3) Width rule.** `xi(N) = c / (N * mu(N))`, `mu(N) = m * N^-tau`. Two
conventions appear in this repository:

| convention | `mu` | `tau` | used by |
|---|---|---|---|
| **global** | `lam_min(H)`, `= lam(N)` exactly under (H1) | `tau = kappa` | `mf_pessimism.py` |
| **signal** | `sigma2_signal`, smallest retained eigenvalue, `-> sigma2 > 0` | `tau = 0` | `BlockEllipsoid.width_rule` |

**(H4) Non-degenerate leakage.** `beta_g := ||P_Nul g_inf|| > 0` **and**
`beta := ||P_Nul b_true|| > 0`.

> **(H4) is the binding hypothesis, and it FAILS on the Phase 3 toy.** Round 6
> established that this is structural, not accidental: the anchor paper's
> completeness assumption (3.3, `K0` invertible) forces the min-norm bridge out of
> the population null, giving `beta_pop = 0`; and by its Lemma C.1 population
> gradient leakage is equivalent to `C*_pi = infinity`. **Both halves of (H4) live
> outside the anchor paper's assumptions.** Everything below is therefore a
> statement about what happens when those assumptions fail — which is a legitimate
> question, and is exactly what a robustness study should ask, but it must be
> labelled as such and not presented as a defect in the paper.

## 3. The results

### 3.1 The floor — schedule-free, and the statement that should lead

> **Proposition 1.** Assume (H1) and `P_Nul b_hat = 0`. Then for **any** ridge
> `lam > 0` and **any** width `xi > 0` — power-law, logarithmic, data-adaptive,
> truth-dependent, anything — a region that covers the truth satisfies
>
> ```text
> W  >=  beta * beta_g,   at every N.
> ```

*Proof.* Coverage requires `xi >= xi_needed >= lam*beta^2` (§3.5). And
`W = sqrt(xi)*||g||_{H^-1} >= sqrt(xi/lam) * beta_g >= beta * beta_g`. The `lam`
cancels. ∎

Two lines, no schedule, no rates. **Any width rule that covers the truth pays at
least `beta * beta_g` at every sample size.** This is due to the round-6 review;
our own proof steps gave it and we stated something narrower.

### 3.2 The power-law dichotomy, as a corollary

With (H2)–(H3), define `e := tau + kappa - 1`. Then, while the class constraint is
inactive:

- **(i) Width.** `W(N) = Theta(N^(e/2))`, explicitly
  `W(N) >= beta_g * sqrt(c/m) * lam0^(-1/2) * N^(e/2)`.
- **(ii) Coverage, necessary condition.** `b_true in conf(xi)` **requires**
  `N^e >= K`, `K := lam0*beta^2*m/c`.
- **(iii) Dichotomy.** If `e < 0`, `W` contracts at `N^(e/2)` but the necessary
  condition fails for all `N > K^(1/e)`. If `e >= 0`, `W` is non-decreasing —
  it never contracts.
- **(iv) Class-constrained regime.** The unconstrained excursion is
  `Theta(N^((e+kappa)/2))`; when `e + kappa > 0` the ball binds at finite `N` and
  `W -> <g_inf,b_inf> + M*||g_inf|| > 0`.

**Two corrections to the original statement of (iii), both from round 6.**

1. *The constant was ignored.* "`e >= 0` ⟹ satisfied at every `N`" needs `K <= 1`.
   For `e > 0` the condition **fails** for all `N < K^(1/e)`. True at the
   implemented constants (`K ~ 1e-6`), false as a general statement.
2. *Necessary is not sufficient.* The `e >= 0` branch shows the **null**
   obstruction is absent, not that coverage holds. The signal directions also
   demand width — on the toy they dominate `xi_needed` by 15× — and they can
   break coverage at any `e`. The original wrote "coverage is sustainable", which
   overstates.

### 3.3 Where the anchor paper sits — `e_paper > 0`, always

This is the one genuinely new consequence, and it is purely analytical. The
paper's width is `xi ~ M_R * N2^(-alpha/(2*alpha+2))` and its ridge is
`lam2 = N2^(-alpha/(alpha*c2+1))`, which places it **inside our own family**:

```text
tau_paper = (alpha+2)/(2*alpha+2),    kappa_paper = alpha/(alpha*c2+1)

e_paper = alpha*(2*alpha + 1 - alpha*c2) / ((alpha*c2+1)*(2*alpha+2))
```

Since Assumption D.16 restricts `c2` to `(1, 2]`, the factor `2*alpha+1-alpha*c2`
is positive for every `alpha > 0`. Verified independently over the admissible
grid: **`e_paper > 0` in 30/30 cells**, minimum `0.0217` at `alpha=10, c2=2`.

> **The anchor paper's own schedule never promises a contracting region.** Its
> construction sits strictly on the non-contracting branch for every admissible
> smoothness — and by (i), the penalty it assigns to a leaky policy **grows with
> `N`**. That consequence does not appear in the paper.

This reframes the whole contribution. We are not exhibiting a flaw; we are
locating the paper's schedule on a trade-off curve and naming the price it pays.

> **Scope, added after `regret_vs_schedule.md`.** The algebra above is unaffected,
> but the emphasis on "leaky" is load-bearing and was under-weighted here. On a
> design satisfying the paper's completeness assumption, the value gradient has
> **exactly zero** mass in the design null (`beta_g = 0.000e+00`, measured for
> three policies at the exact bridges), so `||g||_{H^-1}` is signal-dominated, the
> width shrinks under *every* schedule, and selection is unaffected: four
> schedules including this one give `0.000` regret at every `N` tested. So
> `e_paper > 0` is real algebra whose **decision-level consequence requires
> `beta_g > 0`**, i.e. `C*_pi = infinity` by the paper's Lemma C.1 — outside its
> own assumptions. Where the paper's assumptions hold, this result is
> decision-irrelevant. The regime where it could bite is the incomplete design of
> `completeness_and_h4.md`, and the decision-level test has not been run there.

### 3.4 Scope: this is the exact-null specialisation

(H1)'s exactness is a property of the **tabular delta-kernel**, where the design
is a finite-rank count matrix. For a strictly positive-definite kernel the
spectrum decays smoothly and there is no atom at zero. Round 6 ran the classical
calculation (`s_j ~ j^-b`, gradient mass `g_j^2 ~ j^-a`): then
`||g||^2_{H^-1} = Theta(lam^-theta)` with `theta = 1 - (a-1)/b < 1`, and the
governing exponent generalises to

```text
e' = tau + theta*kappa - 1,     theta = 1 exactly iff the null is an atom.
```

The trade-off persists — **that much is Tikhonov source-condition theory, not
ours** — but the threshold `e = 0` and every numeric constant in §4 are the
delta-kernel case. The original §3 claimed "any method that regularises a
rank-deficient bridge design and calibrates against that design inherits this
dichotomy, whatever the estimator." That is **overstated** and is corrected here.

### 3.5 The two exact structural steps

`H|_Nul = lam*I` exactly, so `(H^-1)|_Nul = lam^-1 I` and
`||g||^2_{H^-1} >= beta_g^2/lam`. And the stage-2 solve gives
`g2_hat = Psi' p_hat/N2 in range(Psi') = Nul^perp`, hence

```text
P_Nul b_hat = lam^-1 * P_Nul g2_hat = 0    exactly,
```

so `P_Nul(b_true - b_hat) = P_Nul b_true` and `xi_needed >= lam*beta^2`. Round 6
audited both solver paths (primal and dual) for a route by which `g2_hat` could
acquire a null component and found none.

## 4. Predictions, as committed before measurement

*Unchanged from `53d71a5`. The verdict column is the only addition.*

**4.1 Width slope.** `d log W / d log N = e/2`.

| convention | `tau` | `kappa` | `e` | predicted | verdict |
|---|---|---|---|---|---|
| global | `kappa` | 0.25 | −0.50 | −0.25 | **confirmed** |
| global | `kappa` | 0.50 | 0.00 | 0.00 | **confirmed** |
| global | `kappa` | 0.75 | +0.50 | +0.25 | **confirmed** |
| signal | 0 | 0.25 | −0.75 | −0.375 | **confirmed** (§5.3) |
| signal | 0 | 0.50 | −0.50 | −0.25 | **confirmed** (§5.3) |
| signal | 0 | 1.00 | 0.00 | 0.00 | **confirmed** |

**4.2 Retrodiction and the correction it forced.** Projected width slope
`(tau-1)/2 = -1/4`, not `-1/2`. `family_pessimism.md` §4 said `N^-1/2`; its own
numbers give `0.667/0.236 = 2.826` against `64^0.25 = 2.828`. **Correct, and
confirmed rank-conditionally at 20 seeds** (§5.4).

**4.3 Coverage crossing.** Predicted `N2* = (c/(lam0*beta^2*sigma2))^2`, with the
crossing declared "the test". **Refuted — see §5.5.** Both the number and the
claim that it was untestable were wrong.

## 5. Measurements

`poc/run_noncontraction.py` (3 seeds), `poc/run_stepc_recheck.py` (5 seeds,
independent re-derivation of the round-6 headline), `poc/critic_c_*.py` (review).

### 5.1 The two exact structural claims

| claim | predicted | measured |
|---|---|---|
| `lam_min(H) = lam` under (H1) | exact | max rel. error **1.25e-12** |
| `P_Nul b_hat = 0` | exact | **7.09e-12** |

Both hold to machine precision, and round 6 confirmed they are structural in both
solver paths rather than numerical luck.

### 5.2 Width slope, global convention — the load-bearing result

| `kappa` | `e` | predicted | measured | error |
|---|---|---|---|---|
| 0.25 | −0.50 | −0.250 | **−0.2526** | 0.0026 |
| 0.50 | 0.00 | 0.000 | **+0.0040** | 0.0040 |
| 0.75 | +0.50 | +0.250 | **+0.2402** | 0.0098 |

The width goes 72.4 → 100.8 → 139.3 → 197.3 as `N` goes 4,000 → 256,000:

> **More data makes the pessimistic bound strictly worse, at the predicted rate.**

Round 6 confirms this survives, and adds the point that matters: **this is the
branch the anchor paper itself occupies** (§3.3). A self-calibrated width rule —
one whose `mu` is read off the same regularised design it is calibrating — has
this pathology built in. That pairing is the second of the two novel corollaries.

### 5.3 Width slope, signal convention — residuals explained

| `kappa` | predicted | measured | error |
|---|---|---|---|
| 0.25 | −0.375 | −0.4141 | 0.0391 |
| 0.50 | −0.250 | −0.2798 | 0.0298 |
| 1.00 | 0.000 | −0.0069 | 0.0069 |

The residuals are all negative and shrink monotonically in `kappa`. The exact
finite-`N` form is `W^2 = xi*(A_N + beta_g^2/lam)` with
`A_N = ||P_S g||^2_{(H|_S)^-1}`; §3 takes the `lam -> 0` limit. At finite `lam`
the retained `A_N` makes `W` decay faster than the asymptote, more so at small
`kappa`.

**Round 6 confirmed this by direct decomposition** — the split is exact to 1e-14
and a constant-coefficient model reproduces the slopes to ±0.002. Two honest
caveats it added:

- At `kappa = 0.25` the constant-coefficient model is short, because `A_N` itself
  **drifts upward** toward its limit over the grid; the explanation needs that
  clause to be complete.
- The regression test we proposed (fit `W^2 = xi*(A + B/lam)`) is **ill-posed at
  `kappa = 1`** — fitted `A = -44.8` from collinearity between the two
  regressors, not because the decomposition is wrong. Our proposed test was a bad
  test; the direct decomposition is the right one.

The per-`kappa` gradient redraw was checked and the ordering is robust to it.

### 5.4 Projected slope — geometry confirmed, deployed estimator refuted

Our original text called this "consistent, not verified" at 3 seeds. At 20 seeds
the picture splits in two, and **we under-claimed**:

- **Rank-conditional** (conditioning on parallel analysis selecting the correct
  rank): **−0.374 / −0.250 / −0.125** against predicted `(tau-1)/2` of
  −0.375 / −0.250 / −0.125. Confirmed to 0.001. The geometry is right.
- **As deployed** (the mean over whatever rank PA actually selects): at
  `kappa = 0.75` the mean **grows** at **+0.236** — the unprojected rate. PA
  over-selects about 5% of the time, and an over-selected rank re-admits a null
  direction whose contribution is `Theta(N^(e/2))`, so the mixture is dominated by
  the heavy tail. **More seeds make the mean worse, not better.**

That second bullet is a real finding and it is bad news for the repair: the
projection's contraction is conditional on a rank selector that is right most of
the time, and "most of the time" is not enough when the failure mode is unbounded.

### 5.5 WITHDRAWN — the coverage branch's quantitative content

**Everything the original §5.5 asserted about coverage on this environment is
withdrawn.** The error was not in the honesty labels, which were accurate about
what had and had not been run. It was in what the measurement *meant*.

The original treated `beta = 0.0532` as an environment constant — "the toy's true
bridge has 2.9% of its norm in the null space". It is not a constant. It is a
grid-average of a quantity that decays. Re-derived independently here at 5 seeds:

| `N` | 1,000 | 4,000 | 16,000 | 64,000 | 256,000 | 1,024,000 |
|---|---|---|---|---|---|---|
| `beta` | 0.166 | 0.115 | 0.031 | 0.019 | 0.011 | 0.006 |

**Slope −0.4954.** A constant gives 0.000; `N^(-1/2)` misalignment noise gives
−0.500. And in the population, the design has rank 2 of 3 with a 1-dimensional
null, and

```text
beta_pop = ||P_Nul_pop b_true|| = 5.4e-16     (exactly zero)
```

So `beta` was measuring the empirical null rotating around the population null at
the sampling rate, nothing more. Consequently:

| withdrawn | why |
|---|---|
| `N* = 1.66e9` | computed from a decaying `beta` treated as constant |
| `K = 4.48e-5`, the `beta^-4` sentence | same |
| the `m_null` table as *evidence* | its slope is algebraically forced (`= e - slope(sigma2)`) and its level is the artifact |
| "(H4) holds but barely" | (H4) **fails** in the population on this toy |
| "`c > sigma2*C` fails at EVERY `N` below the floor" | 3/3 coverage measured below the floor at `N = 16,000` |
| "§4.3 is not testable, the grid is 1,600× short" | it *was* testable by moving `c` toward the floor; tested; the refined crossing does not appear |

The `m_null` withdrawal deserves its own sentence, because it is the trap we
walked into twice. We replaced a vacuous test with a *rate* test and reported
−0.4965 against −0.5000 as confirmation. But that slope is forced by the algebra
of how `m_null` is assembled from `xi`, `lam` and a `beta` we held fixed. **An
identity is not evidence.** With the per-fit `beta`, the corrected margin slope is
**+0.41** — the null-direction coverage margin *improves* with data on this
environment, the opposite of what we published.

**Why this happens, structurally.** `K0` invertible — the anchor paper's own
completeness Assumption 3.3 — forces the min-norm bridge out of the population
null. So the coverage horn requires **completeness to fail**. The environment
where it does fail is the dimension-separated `(2,6,4)` at `confound = 1.0`, which
is precisely the environment we did **not** run part C on. That is the experiment
that would decide the coverage branch empirically, and it has not been run.

### 5.6 Summary

| claim | status |
|---|---|
| `W >= beta*beta_g`, schedule-free | **holds**, and subsumes the power-law framing |
| `lam_min(H) = lam`, `P_Nul b_hat = 0` exactly | **confirmed**, machine precision, both solver paths |
| width slope `= e/2`, both conventions | **confirmed**, ≤0.010 global; signal residuals fully decomposed |
| more data ⟹ worse bound when `e > 0` | **confirmed**, +0.240 vs +0.250 |
| `e_paper > 0` for all admissible `(alpha, c2)` | **confirmed**, 30/30 cells, min 0.0217 |
| projected slope `(tau-1)/2`, rank-conditional | **confirmed** to 0.001 at 20 seeds |
| projected slope, as deployed | **refuted** — grows at +0.236 when `e > 0` |
| coverage horn, as instantiated on the toy | **refuted** — `beta_pop = 0`; corrected margin slope +0.41 |
| the `c` floor as a sharp threshold | **refuted**; order-of-magnitude marker only |
| dichotomy is novel | **restatement** of source-condition theory, plus two novel corollaries |
| coverage horn where completeness fails | **not run** |

## 6. Honest limitations

- **The novelty claim is downgraded.** The trade-off is Tikhonov source-condition
  theory specialised to an exact-null design. What survives as ours: the
  placement `e_paper > 0` (§3.3) with its growing-penalty consequence, and the
  self-calibrated width-rule pathology (§5.2). Neither is a theorem about
  pessimism in general.
- **The coverage branch has no confirmed empirical instance.** It is sound as
  mathematics and its premise fails on every environment tested. Until part C is
  run where completeness fails, it should be stated as a conditional result with
  its premise displayed, not as an observed failure mode.
- **The escape claim was wrong.** §6 of the original asserted that a `mu`
  depending on the truth would escape the dichotomy and be unimplementable. By
  Proposition 1 **no width rule escapes**. The real escapes are a prior-informed
  centre (violating `P_Nul b_hat = 0`), non-ellipsoidal geometry, or `beta = 0` —
  and the last is what actually holds here.
- `beta` is computed from the exact oracle bridge; a practitioner cannot compute
  it, and now we know they would get zero under completeness anyway.
- 3 seeds for §5.2–§5.3, 20 for §5.4, 5 for §5.5. Two environments, one per
  convention.
- (iv) is proved, not measured; the measured evidence for the ball-active regime
  is the `M`-sweep in `norm_constraint.md` §5.

## 7. What round 6 changed

| § | original | now |
|---|---|---|
| 3 | power-law dichotomy as the headline | Proposition 1 (`W >= beta*beta_g`) leads; dichotomy is a corollary |
| 3 (iii) | "`e >= 0` ⟹ coverage sustainable at every `N`" | constant `K` restored; necessary ≠ sufficient |
| 3 | "whatever the estimator" | scoped to exact-null designs; `e' = tau + theta*kappa - 1` for PD kernels |
| 3 | (no anchor placement) | **new**: `e_paper > 0` for all admissible `(alpha, c2)` |
| 5.3 | residual explanation post-hoc, untested | confirmed by exact decomposition; `A_N`-drift clause added; our proposed regression shown ill-posed |
| 5.4 | "consistent, not verified" | rank-conditional **confirmed** to 0.001; deployed mean **refuted** |
| 5.5 | `N*`, `K`, `beta^-4`, `m_null`, sharp floor | all withdrawn; `beta_pop = 0`, `beta(N) ~ N^(-1/2)` |
| 6 | truth-dependent `mu` escapes | no width rule escapes |

**Over-correction check.** Round 6 found none, and moved the other way: §5.4
under-claimed a result that was right to three decimals, and §4.3 called
untestable a test that could have been run. The honesty labels were accurate; the
errors were in the meaning assigned to the measurements.

## 8. Files

| File | Role |
|---|---|
| `poc/run_noncontraction.py` | §5.1–§5.4 |
| `poc/run_stepc_recheck.py` | independent re-derivation of `beta` decay and `e_paper` |
| `poc/critic_c_*.py` | round-6 attacks |
| `docs/critic_findings_stepc.md` | the review |
| `experiments/results_noncontraction.json`, `results_stepc_recheck.json` | raw |
