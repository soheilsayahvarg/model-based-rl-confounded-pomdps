# Claim Ledger

One row per claim this extension has made, with its current status and where it
was established or killed. **Read this before any of the other documents.** They
were written in sequence and several contain statements that later work
superseded; where a document and this ledger disagree, the ledger is current.

Status vocabulary: **holds** (survived every review round aimed at it),
**scoped** (true only under stated conditions that were not originally stated),
**withdrawn** (we retracted it), **refuted** (review or measurement killed it),
**open** (stated, not yet tested).

---

## The four legs of what is actually defensible

Everything below rolls up into these. Nothing else should be presented as a
result.

1. **A faithful implementation, audited.** Every defect found across seven review
   rounds is recorded with the measurement that found it.
2. **`e_paper > 0` for every admissible `(alpha, c2)`** — the anchor paper's own
   schedule never promises a contracting region. Pure algebra, 30/30 cells,
   independently re-derived.
3. **The floor `W >= beta * beta_g`, with two switches.** An inadequate negative
   control (`|O_0| < |S|`) makes the truth leak; confounding makes the gradient
   leak. The floor is nonzero exactly when both hold — and there, the width grows
   with data and selection degrades.
4. **Plug-in dominance**, environment-scoped, with the mechanism stated.

---

## Step (a) — rank structure of the design

| claim | status | where |
|---|---|---|
| Exact null `dim >= \|O\| - min(\|O\|,\|O_0\|)`, set by instrument shape, at every `N` | **holds** | `rank_structure.md` |
| Three-tier spectrum; middle tier decays at exactly `N^-1` | **holds** | same (first published as `N^-0.8`, 3-seed noise) |
| "Population rank overstates usable rank" | **withdrawn** | artifact of `confound=1.0` making `pi_b` a deterministic function of the state |
| "No eigengap rule can work" | **withdrawn** | the wall is real, our implementation was broken — a `1e-300` floor made `argmax` select float sign noise |
| Parallel-analysis rank selection, with the `1 + #{i>=2}` correction | **holds** | correct in every cell at every `N` |
| Regime criterion `\|O_0\| < \|O\|` | **refuted** | true in every row including every failing one — zero discriminating power. Real axis is whether `rank(P_a)` saturates `min(\|O\|,\|O_0\|)` |

## Step (b) — model-free pessimism

| claim | status | where |
|---|---|---|
| "Gradient leakage is the causal driver, **not** conditioning" | **corrected — our own dichotomy was false** | `alignment_knob.md` §5.5. Exactly, `ratio = sqrt(1 + tan^2(theta) cond(H))`: conditioning enters as `sqrt(cond)` on equal footing with alignment. The penalty is their **product**. The dichotomy was read off a knob in which `tan(theta)` moves 119× and `sqrt(cond)` moves 1.26× |
| Gradient leakage governs the penalty at fixed conditioning | **holds, as an identity** | `alignment_knob.md` §5.4, worst error `8.45e-15` over 17 cells of two independent knobs. Reliable everywhere, and *not a discovery* — no sweep could have refuted it |
| The non-arithmetic content: real confounded POMDPs sit at `beta_g` = 0.46–0.76 | **holds** | so the penalty inherits essentially all of `sqrt(cond(H))`. *Why* `beta_g > 0` is the two-switch result, which is where §3's weight belongs |
| The `-63` headline | **withdrawn** | our width rule used global `lambda_min` (= the ridge) where Phase 3 used the smallest *retained* eigenvalue |
| The confidence region never contracts | **holds** | §4; and generalised by step (c) |
| Projected width shrinks at `N^-1/2` | **corrected to `N^-1/4`** | our own published numbers gave `0.667/0.236 = 2.826` against `64^0.25 = 2.828`; caught by prediction, not re-measurement |
| "No chain to compound through" | **refuted as stated** | later stages contribute 12–45.6% of the variance. What survives: *multiplicative* compounding is excluded, propagation is additive |
| "Coverage and leakage are one phenomenon" | **refuted as an identity** | alignment sweep holds coverage fixed while the penalty moves 14× |
| "Validity 36/36" | **withdrawn as evidence** | the grid could not have produced a violation |
| §3 reproduced independently | **holds, but the knob was degenerate** | `alignment_knob.md` §5.1–§5.2. Interpolating rows toward their mean scales row separation by `(1-alpha)` and `beta_g` tracks it at slope 0.954 — alignment and channel informativeness are one variable — and the span vector rotates 36.8°, so §4b's "the null space is FIXED" is false. Reproduced on a clean knob (prior tilt, `H` frozen algebraically, 3.09 decades of `beta_g`): the conclusion survives |
| Step (b) §4's flat width is an independent result | **subsumed** | it is the same identity along a different axis: `cond(H) ≈ sigma/rho` gives `ratio ∝ rho^-1/2 ∝ N^1/4` under the implemented schedule; §4's own numbers give slope **0.265** against **0.25** |
| §5/§6 numbers reproduced independently | **open** | still quoted from the review's scripts |
| Leakage table at 20 seeds | **superseded** | 6/8 cells agree with the population; the empirical PA measure **under-reports** where leakage is large and has a 0.02–0.03 noise floor where it is zero. More seeds do not fix a biased estimator — use the population computation |

## The bridge class

| claim | status | where |
|---|---|---|
| Our implementation omitted `M_R`, the class bound Theorem 4.2 carries | **holds** | `norm_constraint.md` §1. Excursion 14× the a priori bound |
| The divergence survives imposing the class | **holds** | §4, §6b |
| "The paper's own norm ball beats our repair" | **withdrawn** | held only for the `l2` ball we chose |
| "The paper's literal class is the sup-norm box" | **refuted — our over-correction** | Assumption D.16(d) also bounds the class in RKHS norm (`\|\|b\|\| <= 1`), which is coefficient `l2` here. The class is the **intersection**; both our ball and our box relax it |
| Magnitudes shrink 4–7× under the class (~200× at `c=10`), `l2` selections unchanged | **holds** | verified against the raw JSON |

## Pessimism, regenerated on a common grid

| claim | status | where |
|---|---|---|
| The published projection advantage is a cross-`c` artifact | **holds** | `pessimism_regenerated.md` §3. At `c=0.03` all four methods reach 0.000 |
| The projection has a genuine mid-grid advantage (`c=0.1`, `0.3`) | **holds** | §3 |
| **Plug-in dominance** | **holds, and extended** | never beaten on the toy; and in the *incomplete* design it reaches regret 0.000 where no pessimistic schedule does |
| "The plug-in also fails in the incomplete design" | **refuted — near miss** | measured at one `kappa` only; `b_hat` depends on `lambda_2`, so it was not a control. Caught before write-up |

## Step (c) — the floor and the dichotomy

| claim | status | where |
|---|---|---|
| `W >= beta * beta_g` for any ridge and any width rule | **holds** | `noncontraction.md` §3.1. Two lines; the regularizer cancels |
| Power-law dichotomy `e = tau + kappa - 1` | **holds as a corollary** | §3.2, with the constant `K` restored and necessary-vs-sufficient separated |
| `lambda_min(H) = lambda` and `P_Nul b_hat = 0`, exactly | **holds** | machine precision, both solver paths |
| Width slope `= e/2` | **scoped** | holds where `beta_g > 0`; where it is zero the width is signal-dominated and tracks `(tau-1)/2` |
| **`e_paper > 0` for every admissible `(alpha, c2)`** | **holds** | §3.3, 30/30 cells, min 0.0217, independently re-derived |
| The dichotomy is novel | **refuted** | it is Tikhonov source-condition theory specialised to an exact-null design (`e' = tau + theta*kappa - 1`). Novel corollaries: the placement above, and the self-calibrated width-rule pathology |
| "A truth-dependent `mu` would escape" | **refuted** | no width rule escapes; the escapes are a prior-informed centre, non-ellipsoidal geometry, or `beta = 0` |
| Coverage branch's quantitative content on the toy (`N* = 1.7e9`, `K`, `beta^-4`, the `m_null` rate) | **withdrawn** | `beta_pop = 5.4e-16`; the 0.0532 was a grid average of a quantity decaying at `-0.495` |
| Projected slope `(tau-1)/2`, rank-conditional | **holds** | to 0.001 at 20 seeds |
| Projected slope as deployed | **refuted** | the mean over PA-selected ranks grows at +0.236 when `e > 0`; 5% over-selection re-admits an unbounded direction |

## Completeness, and the two switches

| claim | status | where |
|---|---|---|
| `beta_pop > 0` iff per-action completeness fails | **holds** | `completeness_and_h4.md` §3, 50/50 cells, population algebra. Sharp: `<=1.5e-15` vs 57–77% |
| Coverage crossings appear in an incomplete design and not in the complete control | **holds** | §5.2, 40/40 control cells clean |
| Crossing scales as `c^2` | **holds** | +2.158 vs +2.000, inside grid quantization |
| The absolute `N*` predicted in §4 | **wrong** | two errors of ours; corrected post-hoc and labelled as such |
| "`confound=1` is a contrived knife-edge, `\|O_0\|<\|S\|` is the robust route" | **corrected** | true for `beta`; wrong for the floor. In an incomplete design `beta_g` is **graded** in confounding |
| **Incompleteness leaks the truth; confounding leaks the gradient; the floor needs both** | **holds** | §5c, four configurations |
| The two-switch claim, unscoped in `t` | **scoped to stage-1 blocks** | round 8 B3, `poc/critic8_t2span.py`. At `t=2` the conditioning set includes history; the per-action span reaches `\|S\|` and `beta_pop ~ 1e-15` in 24/24 cells at every confound < 1. History is a partial substitute for the instrument; only the `t=1` blocks leak |
| Stage-2+ restoration is uniform in `t` | **holds** | `horizon_dilution.md` §4.1. Span reaches `\|S\|` at every `t` in 2..5 for every confound < 1, and the knife-edge at confound 1 is equally uniform the other way. Only the first block leaks: `beta_t <= 2.7e-15`, per-stage floor `<= 4.3e-30` |
| The floor compounds along the horizon | **refuted** | §4.2. Exactly one block leaks at any `T`, so the absolute floor is constant in `T` |
| **The decision-level consequence is short-horizon** | **holds** | §4.3. Floor constant while the value spread grows; ratio decays with log–log slope −1.23 and −0.93, about 10× from `T=1` to `T=6`. At `T=1` under strong confounding the floor **exceeds the entire value spread** (1.63). Proposition 1 is horizon-free; its decision consequence is not |
| Prop. 2 (unconfounded gradients cannot leak) is novel | **refuted** | derivable from the anchor’s own Lemma C.1; restates the folk fact that without unmeasured confounding a negative control is unnecessary. The novel part is the conjunction |
| The floor should use the angle, not the product | **refuted** | round 8: coverage forces `xi >= lam*beta^2` and `W >= sqrt(xi/lam)*beta_g` independently, so the product is the right object for a **floor** and is achieved at the minimal covering `xi` for any angle. The angle would enter only a matching upper bound |

## Decision level

| claim | status | where |
|---|---|---|
| The width schedule changes the decision, under completeness | **refuted** | `regret_vs_schedule.md` §4.2. `beta_g = 0` exactly, 0.000 regret under all four schedules |
| `beta_g = 0` under completeness | **holds** | exactly, three policies |
| The width schedule changes the decision where both switches are on | **holds, 20 seeds** | §6.2b. At `kappa=1.5`, `N=128k`: pessimism 0.1804 and plug-in 0.0000, both with zero interval across 20 seeds |
| Under `e > 0`, the plug-in converges while pessimism diverges on identical fits | **holds, 20 seeds** | §6.2b. Opposite-signed slopes in both `e>0` schedules; under the paper's own schedule they **cross** — pessimism better at `N=2k`, worse at `N=32k`+ |
| Effect size quotable | **holds** | 20 seeds with 95% intervals; the `kappa=1.5` separation is complete |
| Schedule B is a flat `e=0` control | **refuted** | 3-seed artifact; at 20 seeds it runs 0.1230 → 0.0644 and plateaus, slope −0.0127 |
| Width slope *magnitudes* equal `e/2` | **scoped** | signs right in all four, magnitudes compressed 9–28% toward the signal-dominated value; post-hoc reading |
| Pessimism never finds the optimum here, under any schedule | **holds** | unexplained, and separate from the schedule story |

---

## The recurring defect

Four times a measurement's meaning turned on a design detail rather than on the
number, and each was caught only by a check made *after* the result looked clean:

1. The coverage crossing predicted at `N = 1.7e9` on a grid reaching `10^6`.
2. "Validity 36/36" on a grid whose smallest `c` was four orders above the
   violation threshold.
3. The first decision sweep, three of four schedules starting inside the
   attractor.
4. The plug-in control run at one `kappa`, which produced a clean and false
   reversal of a claim that had survived seven rounds.

All four are the same failure: **not checking that the measurement range
straddles the predicted transition before running.** The guard now in
`run_regret_incomplete.py` — abort if the precondition fails — is the first
structural fix rather than another individual correction.

**A fifth incident, and it breaks that generalisation.** The step (b) alignment
knob (`alignment_knob.md`) is an experiment that also could not fail, but for a
*different* reason: its knob moved two variables at once, and the quantity it
measured turned out to be an identity. No range check would have caught either.
So the honest statement is not "one defect, one fix" but:

| failure mode | instances | fix |
|---|---|---|
| the range cannot straddle the transition | 1, 2, 3, 4 | precondition guard, in place |
| the knob confounds the treatment with something else | 5 | **none yet** — needs an explicit list of what each sweep holds fixed, verified numerically, not asserted in prose |
| the measured relation is arithmetic | 5 | **none yet** — derive the closed form before sweeping |

Two of the three modes have no structural fix. The claim that the guard is *the*
fix was itself over-general.
