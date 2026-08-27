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
| Stage-2+ restoration is uniform in `t` | **scoped -- needs a hypothesis we did not state** | `horizon_dilution.md` §4.1 measured it; `selfheal_theorem.md` §6 proves it and bounds it. True **iff the reachable transition rows span `R^\|S\|`**. Our generator draws transition rows from a Dirichlet, which is full-rank a.s., so the hypothesis was invisible. On two engineered families it fails at confound **0.0** with `beta_2 ~ 0.71` |
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

**A sixth incident, and it is a fourth mode.** P-C4 divided a value-unit penalty
(`sqrt(xi)*||g||_{H^-1}`, empirical coefficient space `R^144`) by a bridge-unit
floor (`beta*beta_g`, population per-action null in `R^6`) and read the quotient
as an overpayment ratio. Both objects are called "the width of the gap" in
prose, so the mismatch is invisible at the level the prose operates on. This is
the same collision as P11 -- two null spaces sharing a name -- one layer up.

| failure mode | instances | fix |
|---|---|---|
| the range cannot straddle the transition | 1, 2, 3, 4 | precondition guard, in place |
| the knob confounds the treatment with something else | 5 | **none yet** |
| the measured relation is arithmetic | 5 | **none yet** |
| two quantities share a name and not a space | P11, 6 | **none yet** -- a ratio needs its numerator and denominator stated with space and units before it is computed |

The correction is in `converse_identification.md` section 8, and it strengthens
the conclusion rather than reversing it.

## Self-healing, proved and re-scoped

`selfheal_theorem.md`. Population algebra, 360 cells, closed form checked
against brute-force enumeration on the **subspace**, not only the dimension.

| claim | status | where |
|---|---|---|
| The stage-`t` per-action span is `E^T diag(pi_b[:,a]) W_t`, with `W_1 = span{p1*K0[:,o0]}` and `W_{k+1} = sum_a P[a]^T diag(pi_b[:,a]) span{z*E[:,o]}` | **holds** | §6, **360/360** cells, dimension and subspace, `\|\|U_c U_c^T - U_b U_b^T\|\|_2 < 1e-8`. Closed form is `O(\|S\|^3)`; brute force enumerates up to 12,288 profiles |
| Self-healing is *the appearance of a free observation index* in the conditioning set | **holds** | §7. At `t=1` there is no free `o`, so the span is at most `\|O_0\|`; from `t=2` the span is the image of a full `R^\|S\|` |
| The confound-1 knife-edge is `dim = \|S\|/\|A\|` | **holds** | §6, 6/6 cells. `diag(pi_b[:,a])` loses rank. We had measured the collapse and called the point "contrived" without saying what breaks |
| **`cor:selfheal` as published** | **refuted -- false as stated** | §6. `sparse_support` and `dense_lowrank` both lose completeness at `t=2` at confounding **0.0**, `beta_2` = 0.705-0.730 of the bridge norm. Needs "provided the reachable transition rows span `R^\|S\|`" |
| The span is monotone nondecreasing in `t` (our P-S5) | **refuted** | §6, 16 sequences. In `(4,8,3)` the stage-1 span is **3** and history collapses it to **2**, the transition row rank. History is not a *partial substitute* for the instrument; it is a **different** instrument, usually better and sometimes strictly worse |
| The support version of the span proof | **refuted -- our own, caught before measuring** | §7. The free observation multiplies *before* the transition, so the support argument sits on the wrong side of `P`. The discriminating cell is `dense_lowrank`, whose rows have full support: support proof predicts 4, rank proof predicts 2, **measured 2** |
| `cor:dilution` and the stage-selective remedy | **inherit the missing hypothesis** | both assume exactly one block leaks. Under a rank failure every stage leaks, the floor compounds, and the remedy targets the wrong blocks |

## Stage-selective pessimism

`stage_selective_pessimism.md`. Four arms off one fit per `(N, seed)`, 20 seeds.

| claim | status | where |
|---|---|---|
| The divergence is concentrated in the `t=1` blocks | **holds** | Part A, schedule C. `beta_g` = 0.326 at both `t=1` blocks and **exactly 0** at every `t>=2` block; empirical null dim 96/0/0/288/0. Penalty slopes `+0.181/+0.151` at `t=1` against `-0.417/-0.429/-0.438` at `t>=2`. `t=1` share of total penalty 0.908 -> 0.991 |
| Dropping pessimism on the `t=1` blocks recovers selection | **holds, schedule C** | Part B. `full` diverges (slope +0.022, regret 0.1804 at `N=128k`); `tail` and `proj1` both reach **0.0000** from `N=32k`, matching the plug-in |
| `proj1` keeps the layer and still converges | **holds, schedule C** | restricting the `t=1` region to the identified span, rather than dropping it, gives slope **-0.034** and 0.0000 regret. The guarantee is not abandoned |
| Our P7/P8: the remedy trades away conservatism | **refuted** | Part C. All arms stay conservative in **every** cell; `plugin` is the only arm that ever fails (0.99 at `N=32k`). `full` pays a mean gap of `-2.004` and **growing**; `proj1` pays `-0.335` and **shrinking**. Against `full` this is not a trade |
| The remedy holds under a rank failure | **holds -- our prediction that it would fail was wrong** | §8. We predicted `tail` and `proj1` slopes `>= 0` where their targeting is wrong. Measured `-0.0149` and `-0.0254`, against `full` stuck at 0.0997 from `N=32k` |
| The paper schedule reproduces schedule C's pattern | **open** | run in progress |

## The two null spaces

`stage_selective_pessimism.md` §9. Found because Part D refuted a prediction the
theorem work caused us to make.

| claim | status | where |
|---|---|---|
| **`projall` is the arm to carry** | **holds, both families** | slope `-0.0290` dense and `-0.0300` rank-failure; reaches the plug-in's regret within the interval; conservative in **every** cell while its gap shrinks `-1.125 -> -0.399` and `full`'s grows to `-2.085`. It needs no knowledge of which stage leaks |
| The population per-action span and the empirical stage-2 design are the same object | **refuted -- a caveat on our own theorem** | §9. `beta_g` and the empirical null at `t>=2` are **identical** in the rank-failure family and the dense one (`0.0000`, `0`). The span `selfheal_theorem.md` proves lives in `R^\|O\|` and is cut by the transition rank; the design the floor uses is the empirical `T_2`, whose cells are counted by the history alphabet. A rank-deficient `P` leaves it full rank and merely ill-conditioned |
| The `t=1` empirical null is `(\|A\|\|O\| - \|A\|min(\|O_0\|,\|O\|)) x n_y`, a pure cell count | **holds** | §9. Exact in 8/15 at moderate `N` and in **every** case at `N = 256,000`; the excess at small `N` is near-threshold eigenvalues in sparse cells. It involves neither `\|S\|` nor `P` |
| `as:null` and `sec:switch-beta` state the same condition | **refuted** | `as:null` is `\|O_0\| < \|O\|` (cells); `sec:switch-beta` is `\|O_0\| < \|S\|` (latent states). The paper moves between them without saying so |
| `cor:conjunction`, as a **population** statement | **holds** | §9. Empirical `beta` in a complete design decays at the parametric rate: slopes `-0.662`, `-0.501`, `-0.527`, `-0.410`, `-0.500`, `-0.505`, **5 of 6** within `-0.5 +- 0.15` and the miss is *faster* than predicted. Null dimension reaches the exact cell count at `N = 256,000` in **6 of 6** |
| `cor:conjunction`, at finite `N` | **scoped** | a complete design carries a measured floor of order `N^-1/2`: `beta_emp` up to `0.1200` in `(3,7,5)` at `N = 8k..32k`. A footnote with a rate, not a defect |
| The rank failure does not reach the later blocks | **refuted** | §8. `bR_t2` and `bD_t2` penalty slopes flip from `-0.417` and `-0.438` to `+0.090` and `+0.121`. It propagates as conditioning, not as a null, so `beta_g` cannot see it |

## Horizon dilution, re-scoped

`selfheal_theorem.md` §8.

| claim | status | where |
|---|---|---|
| Pipeline agrees with the published dilution numbers | **holds** | `ratio_t1` slope `-1.2340` against the paper's `-1.23`; dense aggregate exactly constant for `T>=2`, slope `+0.00000` |
| Every stage leaks under a rank failure | **holds** | `beta_t` at `t>=2` is `5e-16` in dense and **`0.67` to `0.73`** in both rank-failure families, per-stage floors `0.25` to `0.52` |
| **`cor:dilution` merely needs a side condition** | **refuted -- it REVERSES without one** | ratio slope in `T`: `-0.89`/`-0.58` dense, **`+0.90` to `+1.93`** rank-failure. The floor compounds faster than the spread widens, so a longer horizon is strictly worse, not neutral |
| Our P-D3 ("flat under a rank failure") | **refuted** | the ratio grows. Worse than we predicted |
| Our P-D4 control, as stated | **refuted -- the control was mis-specified** | it compared a two-block aggregate against a one-block published number, so the "failure" was exactly `1.00`. Corrected, it passes at `0.00e+00` at every `T` |

## The PD-kernel regime

`pd_kernel_regime.md`. Fills the gap `sec:scope` admits.

| claim | status | where |
|---|---|---|
| `theta* = (alpha c_2 + 1)/(2 alpha + 2)`, and `= 1/2` at `c_2 = 1` for every `alpha` | **holds** | §6, 0 mismatches of 25. Equivalently `b >= 2(a-1)` |
| **`prop:floor` is schedule-free because of the ATOM, not the method** | **holds** | §6. Atom spectrum: floor slope `-0.009`, `-0.000`. PD spectrum: `-0.302`, `-0.911`, decaying four orders. The paper's central adjective must be scoped to the exact-null case |
| `sec:not-new`'s `e' = tau + theta kappa - 1` | **corrected** | needs `clip(theta, 0, 1)`. For `theta <= 0` the exponent saturates at `(tau-1)/2`, measured to `0.0007` at `theta = -0.60` and `0.0000` at `theta = -2.33` |
| The clipped law, out of sample | **holds** | §7, 17/20 within 0.02, 9 within 0.001; the three misses are two `theta = 0` log boundaries and one truncation |
| `theta = 0` residual is a log correction | **holds** | §7, 3/3. Extending the `N` grid moves the slope toward the prediction every time |
| Our P-K2 and P-K5 | **refuted, both our fault** | truncation (`j* = 2.4e4` against a `2e4` cut, the **fifth** instance) and fitting a power law to a Gaussian spectrum |
| Our `j*` truncation guard | **refuted, both directions** | over-conservative on 4 of 5 rejections, and it missed `(1.2,1.2)`. Replaced by an exact tail bound `J^(1-a)/((a-1) lambda)`, which admits **0** truncated cells and whose conservatism is a provable slackness |
| `cor:epaper`, unconditional | **scoped** | conditional on `b >= 2(a-1)` at `c_2 = 1`; at `c_2 = 2` the threshold is `0.955` at `alpha = 10`, so almost any smoothness rescues the schedule |

## Novelty, checked against the literature

`novelty_check.md`, run 2026-08-27 **before** rebuilding the paper around the
span theorem. Third rediscovery in this project, and the first one caught before
we built on it.

| claim | status | where |
|---|---|---|
| The `\|O_0\| >= \|S\|` completeness switch is ours | **refuted -- known** | Ying, Miao, Shi & Tchetgen Tchetgen JRSS-B 2023, eq. (6): `min(d_z, d_w) >= d_u`, attributed to four earlier papers |
| Sequential completeness is characterized in the longitudinal proximal literature | **no -- it is assumed** | their Assumptions 4 and 5 state it per time point; "rank" appears once in the whole paper, in an unrelated remark |
| The stage-wise rank condition is ours | **refuted -- known and assumed** | Zhang & Jiang arXiv:2402.14703 Assumption 2: `rank(M_H,h) = rank(M_F,h) = S` at every `h`; they also note Uehara et al. 2022a had pointed out its role |
| **"History substitutes for a negative control" is ours** | **refuted -- classical** | finite-state nonparametric HMMs are identifiable "as soon as the transition matrix has full rank and the emission distributions are linearly independent" (Allman-Matias-Rhodes 2009 via Kruskal; Gassiat et al. 2016). Those are our two conditions verbatim |
| The exact span formula with **partial** rank and the behaviour-policy factor | **plausibly new** | HMM identifiability has no actions and no confounded logger; the `\|S\|/\|A\|` saturation knife-edge is specific to a confounded POMDP |
| Non-monotonicity in `t` (span `3 -> 2`) | **no hit found, weak evidence** | once the rank condition is stated it is close to a one-line corollary |
| **What a violated rank assumption costs a pessimistic planner** | **not found in either literature** | this is where every measurement in this project lives, and it is the defensible framing |
| The `projall` remedy | **not found** | |

## The converse, and what the floor actually is

`converse_identification.md`. Derived and committed before measurement, then
corrected once.

| claim | status | evidence |
|---|---|---|
| The identified set of values is an interval of half-width `HW = beta_g*sqrt(M^2 - \|\|b_hat\|\|^2)` | **holds** | closed form vs SLSQP from six starts, worst rel err `9.13e-13`; 23 of 24 non-degenerate cells, the miss a solver failure on the smallest target (`6.7e-4`) |
| At the tight class `M = \|\|b_true\|\|` the half-width is exactly `beta*beta_g` | **holds** | so `prop:floor` is **tight**, achieved rather than bounded |
| `V(b)` is point-identified iff `beta_g = 0` | **holds** | 6/6 policies on the complete design, 1/6 on `(4,6,2)` at cf `0.9`; half-widths ordered exactly by `beta_g` |
| The one point-identified policy is the optimal one | **unexplained** | single instance; could be a coincidence of this configuration, and we have no account of it |
| P-C4: the anchor schedule **overpays** the identification gap | **refuted in sign** | the ratio is below `1` at every `N`; the schedule underpays and climbs |
| P-C4's ratio is a ratio of widths | **refuted -- sixth defect** | value units over bridge units, across two different null spaces |
| P-C6: on the coherent comparison the t=1 region still fails to cover | **holds, 4 of 4** | `HW_t1 = \|\|U'b_true\|\| * \|\|U'g\|\|`, ratio `0.220 -> 0.695`, crossing extrapolated to `N ~ 4.1e5` |
| P-C7: the corrected ratio is below `0.5` at `N = 128,000` | **refuted** | it is `0.695`. The direction was supported by the argument; the level was not |
| P-C8: `HW_t1` is near-constant | **holds at its threshold and is misleading** | spread `0.110 < 0.20`, but monotone decreasing, `-23%` over the grid |
| P-C9: the corrected ratio's slope matches the width slope | **refuted by P-C8's slack** | `+0.2852 = +0.2209 - (-0.0643)`; an `11%` denominator drift supplied a third of the slope |

Consequence for the writing: "pessimism cannot contract, and that is a pathology"
is **wrong**. Non-contraction is forced by identification, and a width that
contracted below the gap would be invalid. The finding is that selection degrades
while the region is still too narrow to cover.

## Second novelty check: the converse is not new either

`novelty_check.md`, run before anything was built on the converse.

| claim | status | where |
|---|---|---|
| The identified set is `b_hat + Nul` | **refuted -- known** | Florens, Simoni et al. arXiv:1709.03473: "the identified set becomes a closed linear manifold, which is denoted by `I_0 = phi + N(K)`" |
| Point-identification iff `g` is orthogonal to the null | **refuted -- known, with more attached** | same paper, `N(K)^perp = closure(R(K*))`; Severini & Tripathi (2006, 2012) also give the efficiency bound, which we do not |
| Confidence regions failing to shrink under identification failure | **refuted -- known** | same paper, stated as an already-understood consequence |
| The sharp half-width arithmetic and the tight-class identity | **a remark, not a contribution** | one line of convex computation once the two above are in hand |

**Four for four on rediscovery. This line of work has no novel theoretical core.**
Both neighbouring literatures do inference on a functional; neither does policy
**selection**, and every measured finding here is on the selection side.

## Decision level, dense schedule C, 20 seeds

`results_stage_selective_dense_kappa15.json`.

| claim | status | evidence |
|---|---|---|
| Pessimism with full regions is anti-monotone in `N` | **holds** | regret `0.0952 -> 0.1804`, slope `+0.0220`, the only positive slope among five arms |
| It converges to a **wrong** policy with zero variance | **holds** | at `N = 128,000` the CI half-width is exactly `0`; all 20 seeds pick `uniform`, the optimum is `always_1` |
| The fit is not the problem | **holds** | the plug-in on the identical estimates reaches zero regret by `N = 32,000` |
| The failure is drift toward maximum entropy | **holds** | modal pick `greedy_lo -> soft -> uniform -> uniform`; the penalty charges gradient null-space mass and `uniform` minimises it |
| Every remedy arm converges | **holds** | `tail`, `proj1`, `projall` all reach zero regret and `always_1` |
| Part 4's t=1 penalties can discriminate between families | **no** | `dense` and `dense_lowrank` differ only in `P`, which does not enter the t=1 design, so the stage-1 penalties are **identical to four decimals**. Structural, not a coincidence, and it means that comparison carries no information |

## `projall`, attacked and still standing

`stage_selective_pessimism.md` section 10. The predictions were written to kill
our own arm.

| claim | status | evidence |
|---|---|---|
| `projall` targets a point-identified surrogate functional | **refuted -- it does not** | `projected=True` restricts the perturbation direction and leaves the centre `b_hat` alone. Same functional, smaller region |
| `projall`'s region excludes most of the truth | **holds** | `\|\|P_excl b_true\|\| = 1.6344` against `\|\|b_true\|\| = 2.3590`, i.e. **69%** by norm, at t=1 |
| Its lower bound is therefore certified | **no, and never was** | `BlockEllipsoid` audit finding A-high already said so: coverage holds "by sign-alignment, not by construction" |
| P-C14: the excluded value component `c` is positive where `projall` works | **holds in sign** | `c > 0` at `(4,6,2)` cf `0.9` at every `N` |
| P-C14: alignment cosine above `0.3` | **refuted** | `0.111`. Nearly orthogonal, so `c` is a tenth of its Cauchy-Schwarz bound |
| P-C15: `c` is constant in `N` | **refuted** | spread `0.237 > 0.20`, drifting `0.0738 -> 0.0588`. The quantity is population; its **subspace** is estimated |
| **P-C16: the sign of `c` flips somewhere** | **REFUTED -- the attack failed** | `c > 0` in `30` of `32` cells, `~0` in the other `2`, no real flip |
| P-C17: anti-conservative cells exist | **untested** | conditional on P-C16; no cell qualified. Recorded untested, not passed |
| The two constructions of "unidentified subspace" agree | **verified, not assumed** | `\|\|P_excl - P_null\|\|_F = 4e-13`, principal-angle sines all `1.000` |
| P-C6 needs a centre-offset term | **no** | `\|\|P_Nul b_hat\|\| = 0.0000` at every `N`, structurally: `rhs` lies in `Range(T_2) ⊥ Nul(T_2)` |

`projall` survives an attack designed to break it, on `32` cells. **The mechanism
is unexplained.** Two candidates (concentration at `1/sqrt(k_excl)`, and a shared
positive component surviving `P_excl`) were both formed after seeing these
numbers and neither is tested. `(3,7,5)` at cf `0.9` also shows an unexplained
five-fold penalty jump between `N = 8,000` and `32,000`.
