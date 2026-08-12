# Critic Findings — Adversarial Review of `docs/rank_structure.md`

Reviewer: independent adversarial pass (Fable 5, code-executing).
Target claims: C1 (exact null space = instrument shape), C2 (three-tier spectrum),
C3 (population rank overstates usable rank), C4 (eigengap rule wrong outside the
Phase 3 toy).

## Checklist

- [x] A0 — reproduce the author's numbers by running the shipped scripts
- [x] A1 — is C4 comparing two different matrices? (tier analysis on `Ma @ Ma.T`)
- [x] A2 — is the environment rigged? (flat-spectrum emissions; attack C1 directly)
- [x] A3 — is "statistically empty" an artifact of the classifier constants?
- [x] A4 — seeds / actions / horizon / per-action averaging
- [x] A5 — anything the author did not think to check

Findings are appended below as each attack completes. Verdict vocabulary:
REFUTED | UNPROVEN | OVERSTATED | SURVIVES.

---

## A0 — Reproduction

**What I ran:** `poc/run_rank_verify.py` and `poc/run_rank_diagnostic.py`, unmodified
(Python 3.13.1, NumPy 2.2.2).

**Result:** Every number in `docs/rank_structure.md` sections 4, 6 and 7 reproduces
exactly — the (2,6,4) table, the tier counts (1/3/2, 2/3/2, 2/0/4), the population
ranks, and the eigengap ranks per action. The results JSONs are regenerated
byte-equivalent in content.

**One thing the author's own output already reveals:** in the diagnostic's
model-free table, the Phase 3 toy rows show `ratio ~ 7.9e+295`. Ratios of that
magnitude can only come from the `max(denominator, 1e-300)` clamp in the eigengap
code, i.e. from a **negative** trailing eigenvalue. This is pulled on in A1.

**Verdict: reproduction SURVIVES.** The numbers are real; the fight is over what
they mean.

---

## A1 — C4's cross-matrix comparison, and what the eigengap rule actually computes

**Claim attacked:** C4 — "the eigengap rank rule is correct only in the Phase 3
toy; elsewhere it admits statistically empty directions into the signal subspace."
The author flagged that the "signal present" column comes from the model-free
cross-moment while the eigengap ranks come from `Wa = Ma @ Ma.T / N2` inside the
estimator — two different matrices.

**What I ran:** `poc/critic_a1_mb_tiers.py`. It monkeypatches
`TabularBridgeEstimator._stage2_solve` to capture the estimator's own per-action
`Wa` spectra (byte-identical to what drives the rule), runs the author's exact
three-tier slope classification on `Wa` across N = 4k..256k × 3 seeds for all
four grid configs, and instruments the eigengap argmax to see which ratio wins
and whether its denominator was clamped by `max(., 1e-300)`. A truncated-fit
trick (T=1 on one-step data) reproduces the full fit's `bR_t1` solve exactly.
Raw output: `experiments/critic_a1_mb_tiers.json`.

**Result 1 — the matrix swap does NOT change the tier story.** `Wa` has exactly
the same tier counts as the cross-moment in all four configs: (2,6,4) → 1
signal / 3 statistically-empty / 2 exact-zero; (3,7,5) → 2/3/2; (4,6,2) → 2/0/4;
(2,3,2) → 1/1/1. Middle-tier slopes on `Wa` are −0.93 to −1.08. The three-tier
geometry is a shared property of both designs, so the author's self-flagged
worry is, on this axis, unfounded.

**Result 2 — the eigengap ranks in the doc's Section 7 table are floating-point
sign noise, not measurements.** Instrumenting the argmax shows the winning ratio
is essentially always at the machine-zero cliff, and in most cases its
denominator is a NEGATIVE eigenvalue clamped to 1e-300, manufacturing ratios of
1e+280–1e+294 that beat every genuine gap. The rule as implemented returns "the
number of eigenvalues with positive sign", i.e. the empirical shape cap plus
noise-sign flips. Direct evidence:

- (2,6,4): ranks per (seed × action) flap between [4,4], [5,5], [4,5] across N
  with no trend — the doc's reported `[5, 5]` is one draw of that noise.
- (4,6,2): ranks flap among 3, 4, 5. `Wa` has rank ≤ 2 there (its columns take
  at most |O_0| = 2 distinct values at t=1), so the doc's `[4, 4]` row is not
  "admitting statistically empty directions" — there are none; it is admitting
  exact-zero machine noise. The doc's mechanism story is wrong in detail for
  this row.
- The author's own diagnostic output contains ratios ~7.9e+295 (Phase-3-toy
  rows), which is the clamp's fingerprint, visible before any new experiment.

**Result 3 — the doc's table uses two different definitions of "signal directions
actually present" in the same column.** For (2,6,4)/(3,7,5)/(4,6,2) it uses the
slope-tier count from `run_rank_verify.py`. For the (2,3,2) toy row it uses the
population rank (2) — the toy config is not even in `run_rank_verify.py`'s CONFIG
list, so that cell was never measured. Measuring it (both on `Wa` and on the
cross-moment): at toy dims the spectrum is 1 signal / 1 statistically-empty / 1
exact-zero — index 1 decays with slope −0.84 (cross-moment) / −1.07 (`Wa`). By
the author's own criterion the eigengap rule's k=2 admits a statistically empty
direction IN THE TOY CONFIG TOO. The claimed contrast "correct only in the Phase
3 toy" is therefore unsupported by the doc's own classifier. (Caveat: this is
the dim-separated construction at toy dimensions, not the literal Phase 3
`toy_pomdp` matrices — checked separately in A4/A5.)

**Verdict on C4: OVERSTATED and misdiagnosed; the headline direction SURVIVES in
a stronger form.** What survives: the eigengap rule cannot select the usable
subspace, anywhere — it reads the machine-precision cliff (the shape cap), not a
signal/noise wall, so it "works" at toy dims only in the sense that
min(|O|,|O_0|) happens to equal |S| there. What dies: (a) the specific rank
numbers [5,5]/[4,4]/[5,5] as meaningful measurements — they are sign-noise and
do not replicate across seeds; (b) the claim that the rule is correct in the
Phase 3 toy — under the doc's own slope criterion it is not; (c) the implied
mechanism for (4,6,2).

**What it would take to fix:** re-state C4 on `Wa` (this script already provides
the numbers); replace the eigengap implementation's `1e-300` clamp with a floor
relative to `lam_max` (e.g. `max(desc[1:], eps * desc[0])`) before quoting any
eigengap rank; rewrite the Section 7 table with one definition per column and a
measured toy row; and re-examine the sentence "in that environment the rule
selected the right subspace" — k=2 at toy dims includes a direction that is
statistically empty by the extension's own standard, which touches the Phase 4
interpretation (see A5).

---

## A2 — The environment is rigged, but not where the author looked: `confound=1.0` makes C3's missing direction *exactly* absent

**Claims attacked:** C3 (population rank overstates usable rank), C2 (the middle
tier is "the latent bottleneck's softer effect"), C1 ("at any sample size").

**What I ran:** `poc/critic_a2_flat_spectrum.py` (flat-spectrum construction +
rare-symbol C1 attack) and `poc/critic_a2b_confound.py` /
`poc/critic_a2c_confound_continuum.py` (per-action population design
`P_a = E^T diag(p1 * pi_b[:,a]) K0`, its exact rank and eigenvalues, and a
confound sweep). Raw outputs in `experiments/critic_a2_*.json`.

### A2-1. C3 is REFUTED. The "population direction below the sampling floor" does not exist in the population the estimator sees.

`default_params` uses `confound=1.0`, which makes `pi_b` rows **one-hot**: the
behavior action is a deterministic function of the latent state. Every design
matrix in this study is per-action. Conditioning on the action therefore pins the
latent state to `{s : s % n_a == a}`, and the population limit of the per-action
design has rank `|{s : s % n_a == a}|`, not `min(|S|, |O|, |O_0|)`:

| config | action-marginal pop rank (doc's benchmark) | rank(P_a) per action (correct benchmark) | "usable" directions the doc reports |
|---|---|---|---|
| (2,6,4) | 2 | **[1, 1]** | 1 |
| (3,7,5) | 3 | **[2, 1]** | 2 |
| (4,6,2) | 2 | [2, 2] (shape-capped) | 2 |

The "signal directions actually present" column of the doc equals rank(P_a)
exactly, in every configuration. The per-action population eigenvalue of the
"missing" direction is ~1e-19 (computed in closed form — see the `pop limit`
columns in the critic scripts' output). Section 6's sentence "a direction that
genuinely exists in the population never rises above sampling noise" is false:
conditioned on the action — which is how every matrix in the study is built —
the direction does not exist. Its observed N^-1 decay is ordinary sampling noise
around an exactly-zero population value, i.e. by the author's own Section 4
criterion it is the signature of an absent direction, not a hidden one.

**Causal proof:** setting `confound=0.6` (still confounded, merely stochastic)
restores rank(P_a) = [2, 2] and the "invisible" direction appears as clean
SIGNAL (slope 0.00, population limit 2.1e-3) **already at N = 4,000** — with the
author's own Dirichlet matrices, no spectral surgery needed. At (3,7,5),
confound=0.6 gives 3 signal directions. The N-budget was never the constraint.

**Continuum salvage attempt (author's best repair):** at confound = 0.9 / 0.95 /
0.99 the per-action population eigenvalue of direction 1 is 1.5e-4 / 3.7e-5 /
1.5e-6, and even at 0.99 it emerges above the sampling floor by N = 256,000
(slope −0.37, classified SIGNAL). So a "genuinely present but statistically
invisible at budget N" regime exists only for extremely deterministic policies,
and where it exists it is *quantifiable in advance* from the closed-form
`P_a` spectrum — which is a constructive result, not the reported mystery.

**Verdict: C3 REFUTED as stated.** Fix: compute the per-action population
design's rank/eigenvalues (closed form, three lines) and restate the finding as:
near-deterministic confounding continuously drives per-action signal eigenvalues
toward zero, with exact collapse at determinism. That is a statement about the
behavior policy, not about proximal estimation overstating rank.

### A2-2. C2's tier taxonomy survives; its causal attribution does not (at the tested configs).

At `confound=1.0`, the middle (N^-1) tier has size `min(|O|,|O_0|) − rank(P_a)`,
which conflates two causes: the latent bottleneck AND policy determinism. For
(2,6,4) the doc attributes 3 middle directions to the latent bottleneck; in fact
only 2 of them are latent-capped (4 − |S|) and 1 is policy-capped. At
`confound=0.6` the clean claim — middle tier = `min(|O|,|O_0|) − |S|` directions
decaying at N^-1 — holds exactly (2 signal / 2 empty / 2 zero at (2,6,4);
3/2/2 at (3,7,5)). **C2 SURVIVES after restatement**, and is cleaner in the
restated form. But the "dimension-separated" environment failed to separate a
fourth dimension that moves the tier boundary: the stochasticity of the behavior
policy. The doc's Section 9 worries about Dirichlet spectra; it never varied
`confound`, which is the knob that actually moves the answer.

### A2-3. The flat-spectrum attack FAILS (good news for the author).

Rebuilding E and K0 with near-disjoint block supports (well-separated, comparable
singular values) changes no tier count in any configuration at either confound
level. The tier structure is set by rank(P_a) and the shape cap, not by the
Dirichlet+boost conditioning. The Section 9 limitation ("other spectra may
distribute the weak directions differently") is aimed at the wrong knob — the
spectral profile moves *magnitudes*, not *counts*, at these budgets.

### A2-4. C1's "at any sample size" is falsified by unobserved instrument symbols.

C1 predicts exactly `|O| − min(|O|,|O_0|)` = 2 exact zeros at (2,6,4). Give the
last instrument symbol probability 1e-6 (K0 still full row rank) and at N = 4,000
only 3 of 4 symbols are observed: the empirical design has **3** exact zeros.
At N = 4 it has 4. The correct statement is an inequality —
`dim(exact null) ≥ |O| − min(|O|,|O_0|)` — with equality holding almost surely
once every instrument symbol in the support has been observed *within each
action bin*. **C1 OVERSTATED (minor):** the linear-algebra core (the cross-moment
cannot span more than |O_0| directions at any N) is correct and survives; the
claimed exact equality "at any sample size" does not.

---

## A3 — The classifier is NOT the artifact-maker; the author under-sold their own rate hypothesis

**Claim attacked:** C2's "N^-0.8 to N^-1.1, consistent with N^-1" — is −0.80 a
distinct rate being rounded toward the hypothesis? And are the tier counts an
artifact of the −0.5 threshold and 1e-12 floor?

**What I ran:** `poc/critic_a3_classifier.py` on (2,6,4), confound=1.0: 20 seeds
under (a) the author's convention, where the environment varies with the seed,
and (b) a fixed environment with 20 independent data draws; per-seed slope fits
with error bars; and a 5 × 3 grid of slope thresholds × exact floors.

**Result 1 — the middle-tier rate is −1, cleanly.** Index 1 slope: −1.02 ± 0.18
(author's convention), −1.03 ± 0.13 (fixed env); indices 2–3 similar; t-tests
against −1 give |t| < 1.6 throughout. The doc's −0.80 is 3-seed sampling noise.
Given A2 (the per-action population value is exactly zero) this is the expected
sampling rate of a squared mean, and the author's hedge "N^-0.8 ... consistent
with N^-1" was, if anything, too timid. The 3-seed budget produced a misleading
point estimate; the mechanism was right.

**Result 2 — tier counts are completely insensitive to the classifier
constants.** All 15 combinations of threshold ∈ {−0.2, −0.35, −0.5, −0.65,
−0.8} × floor ∈ {1e-10, 1e-12, 1e-14} give 1 signal / 3 empty / 2 exact-zero.
The three-tier boundary is not an artifact of the two constants.

**Result 3 (method hygiene, minor):** the author's scripts pass `seed=sd` to
`default_params` AND tie the data rng to `sd`, so each "seed" is a different
environment — slopes are fit on spectra averaged across three different
populations, with no replication inside any single environment. It happens not
to change conclusions (fixed-env numbers agree), but it should be fixed before
these numbers go in a paper. Also, the exact-zero tier's fitted "slopes" are
meaningless noise (±16, ±65 across seeds); the doc prints "—" for them, which
is correct — the JSON stores them, which is a trap for later readers.

**Verdict: SURVIVES.** The slope classifier and both constants are doing their
job; C2's rate claim is stronger than the author stated. The −0.5 / 1e-12
choices are not load-bearing.

---

## A4 — Actions, horizon, per-action averaging, and the real toy

**What I ran:** `poc/critic_a4_scope.py`: per-action (unaveraged) tier
classification at (3,7,5); n_a=4 configurations; stages t=2,3; and the actual
`toy_pomdp` environment (the real Phase 3/4 substrate).

**Result 1 — per-action averaging hides real structure.** At (3,7,5),
confound=1.0, the per-action tier counts are action 0: 2 signal, action 1:
**1 signal** — matching rank(P_a) = [2, 1] from A2. The doc's action-averaged
"2 signal" is the pointwise mean of a rank-2 spectrum and a rank-1 spectrum and
describes neither. Any downstream per-action machinery (the estimator's
per-action eigengap, per-action pessimism widths) sees the unaveraged objects.

**Result 2 — more actions expose a structural degeneracy of the testbed.** With
n_s=2, n_a=4, confound=1.0, actions 2 and 3 are NEVER taken (behavior support is
{s % 4} ⊆ {0,1}): their design matrices are identically zero. The
dimension-separated environment cannot even populate all action bins once
n_a > n_s at full confounding. With n_s = n_a = 4 each action pins one state and
per-action signal counts are ~1, as A2's formula predicts (11 of 12 action-cells
match; one cell flips S/E on a tiny eigenvalue — the 3-seed slope fit is fragile
at per-bin sample sizes, consistent with A3's error bars).

**Result 3 — horizon changes nothing.** At t=2 and t=3 the (2,6,4) tiers stay
1/3/2 per action: pi_b reads s_t at every stage, so the determinism collapse is
stage-invariant. The doc's t=1-only measurement is representative for THIS
policy class.

**Result 4 — the REAL Phase 3 toy is not the doc's "(2,3,2) toy dims" row.**
`toy_pomdp` has stochastic pi_b = (0.75, 0.25), so rank(P_a) = [2, 2] with
per-action population eigenvalues (8.7e-2, 2.5e-3) and (4.1e-2, 5.3e-3) — and
the measured spectrum has 2 genuine signal directions per action, no
statistically-empty tier at all (2/0/1). Two consequences:

- The doc's Section 7 sentence "in that environment the rule selected the right
  subspace, and the reported pessimism numbers stand" is **SUPPORTED** — in the
  real toy both directions are honest signal, so k=2 spans the right subspace.
  (A1's Result 3 concern resolves in the author's favor for the real toy; it
  stands against the grid's (2,3,2) row, which is a different,
  deterministically-confounded environment that happens to share dimensions.)
- But the reason the doc gives — "|S| = |O_0| collapses the tiers" — is not what
  saves the toy. What saves it is the stochastic behavior policy plus
  well-conditioned hand-picked matrices. A (2,3,2) environment with
  confound=1.0 has tiers 1/1/1 and the rule's k=2 is wrong there (A1).

**Verdict: C2/C3's evidence base OVERSTATED in scope** (single averaged-action
view, t=1 only, one policy class), though the missing checks mostly confirm the
corrected (A2) picture rather than adding new failures.

---

## A5 — What the author did not check: the wall exists, and a three-line fix finds it

**Claim attacked:** Section 8's capstone — "a subspace-projection repair cannot
be made general by a better eigengap rule. The boundary it needs does not exist
as a gap." This is the sentence that turns the finding into a strong negative
result about the method family. It is refutable by exhibiting a better rule.

**What I ran:** the estimator's eigengap rule with one change — the ratio
denominator floored at `1e-9 * lambda_max` instead of `1e-300` — applied to the
exact spectra captured in A1 (both `Wa` and the cross-moment, seed 0, action 0):

| config | N | k (shipped rule) | k (floored rule) | correct rank(P_a) |
|---|---|---|---|---|
| (2,6,4) | 256k | 4 (Wa) / 5 (mf) | **1 / 1** | 1 |
| (3,7,5) | 256k | 6 / 6 | **2 / 2** | 2 |
| (4,6,2) | 256k | 4 / 5 | **2 / 2** | 2 |
| (2,3,2) | 256k | 2 / 2 | **1 / 1** | 1 |

At N = 256,000 the floored rule recovers the correct per-action population rank
in ALL eight matrix instances. At N = 4,000 it still overshoots (the
signal/noise gap has not yet opened) — so what the evidence supports is "the
eigengap rule fails at small N, and its shipped implementation is dominated by
a numerical artifact at every N", NOT "the boundary does not exist". The
boundary exists and widens like N: signal eigenvalues converge to nonzero
constants while the middle tier decays at N^-1 (A3), so ANY fixed threshold
between them eventually separates the tiers — and the author's own slope
classifier already separates them at the budgets in this study. Section 8's
impossibility claim is **REFUTED as stated**; the defensible version is a
finite-sample statement: below a computable N threshold (obtainable in advance
from the closed-form P_a spectrum), no spectral rule can distinguish weak
signal from noise.

**Second — the downstream victim nobody audited:** `_stage2_solve` sets
`sigma2_signal` to the smallest KEPT eigenvalue under the shipped rule
(`bridge_estimator.py:169`). Wherever the rule overshoots into the
statistically-empty tier, `sigma2_signal` decays like N^-1 (and on exact-zero
picks it sits at ~1e-18); anything downstream that scales pessimism widths by
1/sigma2 inherits a spurious N-divergence that has nothing to do with proximal
geometry. This matters for step (b) of the extension (model-free pessimism):
run it with the shipped rule and a "family-level blow-up" will appear — as an
artifact of the 1e-300 clamp. Step (b) MUST fix the rule first or it will
manufacture its own headline result.

**Third (bookkeeping):** the two shipped scripts use two different objects under
one name — `run_rank_diagnostic.model_free_design` weights by inverse instrument
counts (`(Acr * Wa) @ Acr.T`), while `run_rank_verify.raw_spectrum` uses the
plain normalized cross-moment. Tier counts agree, but eigenvalue magnitudes
differ; a paper table mixing the two would be internally inconsistent.

---

## Summary — what survives, what dies

| claim | verdict | one line |
|---|---|---|
| C1 (exact null = shape) | **SURVIVES, minor fix** | Core linear algebra correct; "at any sample size" false — unobserved instrument symbols enlarge the null space (A2-4). State as ≥ with a.s. equality. |
| C2 (three tiers, N^-1 middle) | **SURVIVES restated** | Tiers and the N^-1 rate are real and robust (A3); but the middle tier is "per-action-population-null", set by rank(P_a) — policy determinism + latent cap together, not the latent cap alone (A2-2). |
| C3 (pop rank > usable rank) | **REFUTED** | The "hidden" direction is exactly absent from the per-action population: confound=1.0 makes pi_b one-hot, rank(P_a)=1. At confound=0.6 it is clean SIGNAL already at N=4,000. Wrong population benchmark, not a statistical ceiling (A2-1). |
| C4 (eigengap wrong outside toy) | **OVERSTATED; direction survives** | The matrix swap does not matter (tiers match on Wa, A1). But the reported ranks are 1e-300-clamp sign-noise artifacts; the rule is also wrong at toy dims under confound=1; the real toy is saved by its stochastic policy, not by |S| = |O_0| (A1, A4-4). |
| §8 (no rank rule can work) | **REFUTED** | A floored eigengap recovers the correct rank in 8/8 instances at N=256k; the author's own slope classifier is a working rule. The impossibility is finite-sample, not structural (A5). |
| §7 ("Phase 4 numbers stand") | **SURVIVES** | Real toy_pomdp has rank(P_a)=[2,2], both directions genuine signal (A4-4) — though for a different reason than the doc gives. |

**The single most consequential correction:** every per-action design in this
study was built under a behavior policy that is a deterministic function of the
latent state (`confound=1.0`). That collapses the per-action population rank
below |S| and manufactures the "statistically empty but genuinely present"
story. The dimension-separated environment separated |S|, |O|, |O_0| — and left
the fourth dimension that controls the answer, policy stochasticity, pinned at
its most degenerate value. Every headline table should be re-run at
confound ∈ {0.6, 0.9, 1.0} with per-action (unaveraged) reporting, and the
population benchmark should be the closed-form per-action spectrum of
`P_a = E^T diag(p1 * pi_b[:, a]) K0`, which this review's scripts already
compute.

**Critic scripts:** `poc/critic_a1_mb_tiers.py`, `poc/critic_a2_flat_spectrum.py`,
`poc/critic_a2b_confound.py`, `poc/critic_a2c_confound_continuum.py`,
`poc/critic_a3_classifier.py`, `poc/critic_a4_scope.py`.
Raw outputs: `experiments/critic_a1_mb_tiers.json`,
`experiments/critic_a2_flat_spectrum.json`, `experiments/critic_a2b_confound.json`.

