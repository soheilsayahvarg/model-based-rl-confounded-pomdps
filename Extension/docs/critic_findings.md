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


---

# Verification Round

Second pass: verifying the applied corrections, not re-opening the originals.
Same rules — every number checked by running code.

## Checklist

- [x] V0 — reproduce `run_rank_corrected.py` and confirm the fix is confined to Extension/
- [x] V1 — 16/18 vs 0/18; floor sensitivity sweep; autopsy of the 2 misses
- [x] V2 — middle tier decays at exactly N^-1 (more N points, per-seed error bars)
- [x] V3 — "shape-capped regime => correct at every N" (construct hostile shape-capped configs)
- [x] V4 — does (3,7,5) cf=0.6 ever converge? (push N past 512k)
- [x] V5 — is the step-(b) plan circular?
- [x] V6 — anything else

## V1 — The floor constant and the two misses

**Claim:** fixed rule correct in 16/18 (old 0/18); the 2 misses are at
confound=0.9 "where the weakest population direction genuinely sits near the
noise floor". Question: is 1e-9 a tuned constant?

**What I ran:** `poc/critic_v1_floor.py` — the author's R3 protocol (18 cells,
20 seeds, N=256,000, modal vs rank(P_a)) with the relative floor swept over
1e-3..1e-15, spectra collected once and reused so the floor is the only
variable; plus a per-seed autopsy of the miss cells that records each
environment draw's population ratio lambda_truth/lambda_1 (the author's seed
convention varies the environment with the seed).

**Result 1 — 16/18 reproduces at 1e-9, and the floor sits on a plateau, not a
knife edge.** Cells correct: 8 (1e-3), 14 (1e-5), 15 (1e-6), **16 (1e-7), 16
(1e-8), 16 (1e-9)**, 15 (1e-10), 12 (1e-11), 9 (1e-12), 6 (1e-15). A ~3-decade
plateau at 1e-7..1e-9 means the fix is not a tuned constant. But "any relative
floor" is also false: too high clips the weakest genuine signal ratios
(cf=0.9 environments have lambda_2/lambda_1 down to ~1e-4); too low resurrects
the exact-zero cliff (middle-tier over clamped-zero ratios win again — the same
disease as 1e-300, in miniature). The plateau's edges are set by identifiable
quantities: min signal ratio above, middle-tier relative magnitude at the
largest N below. V4 adds that the choice trades off against required N as
N* ∝ rel^{-1/2}, so the upper plateau edge is actually the better operating
point, not 1e-9.

**Result 2 — the miss attribution is correct, and sharper than stated.** Both
miss cells are at cf=0.9 (as claimed). Within each, the wrong seeds are
precisely the environment draws whose population ratio is small: median
2.6e-4 (wrong) vs 1.1e-2 (correct) at (2,6,4)a0; 9.1e-5 vs 3.2e-3 at (3,7,5)a0.
So "genuinely weak population direction" is right — but it is a property of
individual environment draws, not of the cf=0.9 cell as a whole: roughly half
the draws are resolvable at N=256k and half are not, which is exactly what 55%
stability means. Two refinements the doc should absorb: (a) the failure mode is
UNDER-selection (modal k=1 vs truth 2, k=2 vs truth 3) — for pessimism this
projects away real signal, a bias failure, unlike the width blow-up caused by
over-selection; (b) one (3,7,5) seed fails the other way (picks the cliff, k=5),
so the cf=0.9 cell mixes both failure modes.

**Verdict: SURVIVES** (16/18 verified; attribution verified), with the floor's
operating band and the under- vs over-selection asymmetry as required additions.

**Fix:** report the plateau (1e-7..1e-9) instead of the bare constant; note the
failure-mode asymmetry in §7.4; consider rel=1e-7 as the default.

---

## V4 — The unconverged cell converges at N ≈ 2M, and the threshold is predictable in closed form

**Claim:** (3,7,5) cf=0.6 inside the estimator "has still not converged at
N=512,000"; V4 asks whether it converges at all.

**What I ran:** `poc/critic_v4_converge.py`. No shipped script produces the
§7.4 Wa table's 128k/512k rows (R4 in `run_rank_corrected.py` only runs N=32k),
so I rebuilt Wa = Ma Ma^T / N2 exactly as `TabularBridgeEstimator.fit` does
(same permutation, same lam1 = 1/N1 ridge, same stage-2 columns), validated the
replica against the real estimator at N=32k (selects [5,5], identical), then
pushed N to 4,096,000 with 3 seeds. Also computed the population Wa spectrum in
closed form.

**Result 1 — it converges.** Selection per seed: [5,5] at 32k/128k (matches the
author), mixed at 512k–1M ([5,5]/[3,5]/[3,3] — the transition), and **[3,3]
unanimously at N=2M and 4M**. The "necessary but not sufficient" framing
survives; "simply wrong in that regime" is not the case. The rule is
data-hungry, not broken.

**Result 2 — the convergence point is predictable, which upgrades the finding
from an observation to a formula.** Population Wa (seed 0, action 0) has signal
eigenvalues (1.72e-1, 4.59e-2, 1.89e-3); the middle tier decays as c/N with
c ≈ 0.9 (measured: 1.77e-6 at 512k, 3.4e-7 at 2M). The rule picks the truth
once the signal-to-middle ratio lambda_3 N / c exceeds the middle-to-floor
ratio c / (N * rel * lambda_1), i.e.

    N* = c / sqrt(lambda_3 * lambda_1 * rel)  ≈  1.6e6

for rel=1e-9 — exactly between the observed flip at 512k–2M. Everything on the
right-hand side is computable in advance from P_a and the confound level. This
formula also exposes the rel trade-off noted in V1: N* shrinks like rel^{-1/2}
as the floor is raised toward the smallest signal ratio.

**Result 3 (reproducibility, minor but real):** the §7.4 Wa table cannot be
regenerated from any shipped script, and its cells are single data draws — my
seed-0 draw gives [5,5] at 512k where the table says [5,3]; at the transition N
the cell value is draw-dependent. The table should be produced by a committed
script with seeds shown, or the transition marked as such.

**Verdict: SURVIVES — and understated.** The author had a stronger, sharper
result available (a closed-form N* that validates against experiment) and
published a weaker one ("has still not converged").

**Fix:** add the N* formula and the 2M/4M convergence to §7.4; commit the
script that generates the Wa table.

## V2 — "Exactly N^-1" survives more N points and honest error bars

**Claim:** middle tier decays at exactly N^-1 (mean −0.983, sd 0.070, n=29).

**What I ran:** `poc/critic_v2_slope.py` — 5 N points {4k, 16k, 64k, 256k,
1.024M} instead of 3, per-seed fits (n=420) alongside seed-averaged fits, and a
3-point fit on the same data as a like-for-like control; 21 middle-tier
directions across (2,6,4)/(3,7,5) × confound {1.0, 0.6} × both actions.

**Result:** seed-averaged 5-point slope **−1.005 ± 0.039** (n=21). No curvature:
3-point and 5-point fits agree to two decimals everywhere. Per-seed fits are
noisier (−0.92 ± 0.66) with heavy tails for the smallest directions (single-draw
eigenvalues near the exact floor produce garbage slopes, sd up to 2.0) — so the
seed-averaged number is the right one to quote, and it is even closer to −1
than the author's −0.983. One presentational trap: per-seed slope
distributions should not be used for error bars on the rate; the averaging is
doing necessary variance reduction on order statistics.

**Verdict: SURVIVES**, tightened (−1.005 ± 0.039 with 5 N points up to 1M).

---

## V3 — The "shape-capped regime" claim is REFUTED as stated; the real axis is whether the middle tier is empty

**Claim (the main new one):** "the rule is correct at every sample size when the
instrument shape forces exact zeros (|O_0| < |O|), and needs very large N when
the rank deficiency is only statistical."

**Before running anything:** the criterion is already contradicted by the
author's own §7.4 table. (2,6,4) has |O_0| = 4 < |O| = 6 — the shape forces two
exact zeros — yet the table shows it failing at N=32k. What is special about
(4,6,2) is not |O_0| < |O| but that rank(P_a) = min(|O|,|O_0|) = 2: there is
NO statistically-empty tier, so the machine-zero cliff sits exactly at the
truth.

**What I ran:** `poc/critic_v3_regime.py` — 7 configurations, all with
|O_0| < |O| (all "shape-capped" by the doc's criterion), split by the predicted
axis: 3 controls with middle tier = 0 (rank(P_a) = cliff) and 4 hostile cases
with middle tier > 0 (rank(P_a) < cliff), including (2,6,2) cf=1.0 — more
shape-capped than the author's control — and (2,8,4) cf=0.6. Fixed rule,
N ∈ {2k, 8k, 32k, 128k, 512k}, 20 seeds.

**Result — clean separation along the predicted axis, not the doc's:**

| type | config | truth vs cliff | N=2k | N=8k | N=32k | N=512k |
|---|---|---|---|---|---|---|
| control | (4,6,2) cf=1.0 | 2 = 2 | 2 (95%) | 2 (95%) | 2 (95%) | 2 (95%) |
| control | (4,6,2) cf=0.6 | 2 = 2 | 2 (100%) | 2 (100%) | 2 (100%) | 2 (100%) |
| control | (2,6,2) cf=0.6 | 2 = 2 | 2 (100%) | 2 (100%) | 2 (100%) | 2 (100%) |
| HOSTILE | (2,6,2) cf=1.0 | 1 < 2 | 2 (10%) | 2 (40%) | 1 (65%) | 1 (100%) |
| HOSTILE | (3,8,4) cf=1.0 a0 | 2 < 4 | 4 (0%) | 4 (30%) | 2 (60%) | 2 (100%) |
| HOSTILE | (2,8,4) cf=0.6 a1 | 2 < 4 | 4 (0%) | 4 (5%) | 4 (30%) | 2 (95%) |
| HOSTILE | (4,7,3) cf=1.0 | 2 < 3 | 3 (5%) | 3 (10%) | 3 (45%) | 2 (100%) |

Every control is correct at every N including 2,000. Every hostile config —
each one satisfying the doc's "|O_0| < |O|" criterion — fails at small N with
the modal selection pinned at the cliff. The regime claim as stated is
**REFUTED**; one supporting config was never a regime, and the first
counterexample was already sitting in the author's own table.

**The correct statement** (verified by all 14 cells): the rule is correct at
every sample size iff the statistically-empty tier is empty, i.e.
**rank(P_a) = min(|O|, |O_0|)** — the per-action population rank saturates the
shape cap. Within the other class, difficulty is continuous and quantified by
V4's N* formula; "shape-capped vs statistically-deficient" is not the axis,
since both of those labels apply simultaneously to most of the failing configs.

**Consequence for step (b) — this is the load-bearing part.** The README's
instruction "run step (b) with |O_0| strictly smaller than |O|" is UNSAFE as
written: (2,8,4) cf=0.6 satisfies it and misselects in 95–100% of seeds at the
project's N ≤ 4,000. If step (b) had been run under that guidance in this
configuration, the rank misselection would have manufactured exactly the
divergence the study is looking for. The safe precondition is
rank(P_a) = min(|O|,|O_0|), which is checkable a priori in a synthetic
environment (three lines from E, K0, p1, pi_b).

**Verdict: REFUTED as stated; SURVIVES after replacing the axis.** Fix: rewrite
§7.4's regime paragraph and the README's step-(b) precondition in terms of
rank(P_a) vs min(|O|,|O_0|); note explicitly that (2,6,4) was always a
counterexample to the dimensional phrasing.

## V0 — Reproduction and confinement

**What I ran:** `poc/run_rank_corrected.py` unmodified; grep for the old floor
across the tree.

**Result:** Every headline number reproduces: R1 18/18 OK (signal = rank(P_a),
exact zeros = |O| − min(|O|,|O_0|), at all three confound levels); R2 mean
−0.983, sd 0.070, n=29; R3 old rule 0/18, fixed rule 16/18, with the two misses
at (2,6,4) cf=0.9 action 0 and (3,7,5) cf=0.9 action 0, both 55% stability, as
published. R4 inside the estimator at N=32k: correct only in the three (4,6,2)
cells, wrong in the other six — consistent with §7.4's "necessary, not
sufficient", and the sigma2 values in the wrong cells (1e-7..1e-6, the
statistically-empty tier) concretely confirm the §8 hazard. Confinement:
`Extension/src/estimation/bridge_estimator.py` has the relative floor;
`Phase_3/src/estimation/bridge_estimator.py:167` still has `1e-300` — the
graded phase is untouched, as intended.

**Verdict: SURVIVES.**

---

## V5 — The step-(b) plan: not circular, but unsafe as written; a gap-free selector now exists and beats the eigengap

**Claim:** step (b) should run "in the shape-capped regime, or with a rank
selection that does not depend on finding a spectral gap".

**Assessment of circularity.** Measuring the baseline's blow-up in a regime
where exact zeros exist is not circular: the exact null space IS the phenomenon
under study, and it is present in that regime by construction, not by selection
bias. But two genuine problems remain. First, V3 shows the regime as *written*
(|O_0| < |O|) includes configurations — (2,8,4) cf=0.6 among them — where the
fixed rule misselects in ~all seeds at the project's N ≤ 4,000; following the
README's instruction literally can land exactly on the misselection-manufactured
divergence the plan is trying to avoid. The safe precondition is
rank(P_a) = min(|O|,|O_0|). Second, an asymmetry should be stated in the paper:
the safe regime is also precisely where the projection repair is reliable, so
step (b) run there can validate the repair only in its easiest regime. The
family-blow-up conclusion would be scoped to exact-tier ill-posedness; nothing
about the repair's behavior in the statistical-tier regime follows.

**The gap-free selector, built and tested:** `poc/critic_v5_parallel.py`
implements parallel analysis (Horn 1965) adapted to the per-action
cross-moment: B=40 permutations of O0 within the action bin give the eigenvalue
null under independence; since the permutation preserves both marginals, the
null itself contains the rank-1 product-of-marginals direction, so the
estimator is rank = 1 + #{i ≥ 2 : λ_i > q95(λ_i^null)}. (A first draft that
also tested index 1 returned 0 whenever the top direction was
marginal-dominated — wrong semantics, kept in the script as `naive` for the
record.) Results, modal over 20 seeds (per-seed accuracy in parens), parallel
analysis (PA) vs the fixed eigengap (EG):

| config | truth | N=8k | N=32k | N=128k |
|---|---|---|---|---|
| (3,7,5) cf=0.6 a0 | 3 | **PA 3 (90%)** vs EG 5 (5%) | PA 3 (65%) vs EG 5 (15%) | PA 3 (85%) vs EG 3 (75%) |
| (2,6,4) cf=1.0 a0 | 1 | PA 1 (85%) vs EG 1 (100%) | PA 1 (100%) vs EG 1 (100%) | PA 1 (95%) vs EG 1 (100%) |
| (2,6,4) cf=0.6 a0 | 2 | **PA 2 (75%)** vs EG 4 (15%) | PA 2 (90%) vs EG 2 (50%) | PA 2 (90%) vs EG 2 (95%) |
| (4,6,2) cf=1.0 a0 | 2 | PA 2 (80%) vs EG 2 (95%) | PA 2 (95%) vs EG 2 (95%) | PA 2 (95%) vs EG 2 (95%) |

The modal PA selection is correct in every cell at every N — including at
N=8,000 on the cell that needs N ≈ 2,000,000 inside the estimator (V4) — and PA
never selects the cliff. It needs no floor constant and no gap. Caveats:
per-run accuracy is 65–100%, not 100%; B permutations cost B spectra per fit
(trivial in the tabular setting); adapting it to the estimator's internal Wa
requires permuting O0 before the stage-1 fit (nulling the CME dependence),
which is straightforward but not yet implemented.

**Verdict: the plan as written is UNSAFE (wrong regime criterion — V3); the
corrected plan is sound and no longer needs the regime restriction at all.**
Recommended step (b): use rank(P_a) as the a-priori truth (synthetic
environment), select with parallel analysis, and run BOTH regimes; report the
repair's behavior separately per regime.

---

## V6 — Everything else

1. **README defect:** the row "| (c) The rank cap as a proposition | not
   started | |" appears twice — in the Status table (line 28) and stranded
   mid-document after the "Why (b) is still blocked" section (line 47).
2. **N-budget inconsistency:** §10 says "N up to 256,000" while §7.4's table
   and the README discuss N = 512,000 runs.
3. **Provenance gap:** no shipped script generates the §7.4 Wa table's
   128k/512k rows (R4 only runs N=32k), and V4 shows its transition-region
   cells are single-draw noise ([5,3] vs my [5,5] at 512k, seed 0).
4. **Withdrawn numbers now unreproducible:** the estimator was fixed in place,
   so `run_rank_diagnostic.py` now silently produces eigengap ranks that
   disagree with the committed `results_rank_diagnostic.json` (old rule), with
   no code path that reproduces the archived output. Add a `rule="legacy"`
   switch or a provenance note in the JSON.
5. **No over-correction found.** Both withdrawals (C3, the §8 impossibility)
   were warranted; §7.4's replacement of §7.3's "partly by luck" is also right
   for the real toy — (2,3,2) with the toy's stochastic policy has
   rank(P_a) = 2 = min(|O|,|O_0|), i.e. it sits in the genuinely-safe class
   identified in V3. What is wrong is only the dimensional phrasing of that
   class (per V3).
6. Cosmetic: R1's match criterion accepts `n_zero >= pred_zero` (should be
   equality); R2's slope helper takes abs() of eigenvalues, which can turn a
   negative machine-noise eigenvalue into a finite log — harmless here because
   the exact floor filters first, but a trap if reused.

---

## Verification Round — summary

| item | verdict | one line |
|---|---|---|
| V0 reproduction | **SURVIVES** | All numbers reproduce; fix confined to Extension/. |
| V1 floor + misses | **SURVIVES** | 16/18 verified; 1e-7..1e-9 plateau, not a knife edge; misses = weak-λ environment draws; failure mode is UNDER-selection. |
| V2 N^-1 | **SURVIVES, tightened** | −1.005 ± 0.039 with 5 N points to 1M; no curvature. |
| V3 regime claim | **REFUTED as stated** | 4/4 hostile shape-capped configs fail at small N; correct axis is rank(P_a) = min(|O|,|O_0|); (2,6,4) already contradicted the doc's own criterion. |
| V4 convergence | **SURVIVES, understated** | Converges to [3,3] at N=2M–4M; closed-form N* = c/√(λ_r·λ_1·rel) ≈ 1.6e6 matches observation. |
| V5 step-(b) plan | **UNSAFE as written; fixed** | Not circular, but the regime criterion is V3-wrong and the safe regime favors the repair; parallel analysis selects correctly in all regimes at N=8k and removes the restriction. |
| V6 misc | — | README dup row; N-budget mismatch; §7.4 table unreproducible; legacy rule unreproducible; no over-correction found. |

**Bottom line:** the two accepted corrections were applied correctly, and every
quantitative claim built on them reproduces or tightens. The new interpretive
layer contains one real error with consequences — the "shape-capped regime"
criterion — which, if followed as written in step (b), can reintroduce exactly
the artifact the correction was meant to remove. The fix is one substitution
(rank(P_a) = min(|O|,|O_0|) in place of |O_0| < |O|) plus, preferably, dropping
the regime restriction in favor of the parallel-analysis selector.

**Critic scripts this round:** `poc/critic_v1_floor.py`,
`poc/critic_v2_slope.py`, `poc/critic_v3_regime.py`,
`poc/critic_v4_converge.py`, `poc/critic_v5_parallel.py`.
Raw outputs: `experiments/critic_v1_floor.json`,
`experiments/critic_v3_regime.json`, `experiments/critic_v5_parallel.json`.

---

# Step (b) Round

Third pass: the family-pessimism experiment (`docs/family_pessimism.md`). Claims
C1 (shared blow-up), C2 (compounding excluded), C3 (gradient leakage, not
conditioning), C4 (unification with the Phase 4 coverage result). The
parallel-analysis selector I contributed last round is now load-bearing and gets
attacked like everything else.

## Checklist

- [x] A0 -- reproduce `run_family_pessimism.py`
- [x] A1 -- the pessimism layer is ours: xi convention (a) and ellipsoid shape (b)
- [x] A2 -- is leakage causal, or confounded with rank(P_a)?
- [x] A3 -- does later-stage uncertainty enter through the response? (C2)
- [x] A4 -- t=1-only H; nu1 estimated from the same sample
- [x] A5 -- is 36/36 validity informative or trivial?
- [x] A6 -- can any link of C4's five-step chain be tested directly?
- [x] A7 -- anything else, incl. the parallel-analysis selector in its new role

## A0 -- Reproduction

**What I ran:** `poc/run_family_pessimism.py` unmodified.

**Result:** every number in `docs/family_pessimism.md` reproduces: -63.19 at
(2,6,4) cf=1.0 c=10; the section-5 means (leak 0.348 vs 0.0146, ratio 10.62x vs
1.48x, cond(H) 1.1e2-5.9e2); validity 36/36. Two details visible in the raw
table that the doc's means hide, picked up later: (i) V_hat is BELOW V_true in
22 of 24 policy cells (the estimator is biased downward under confounding) --
relevant to A5; (ii) leakage varies 4x WITHIN confound=1.0 across actions
(0.157 vs 0.627 at (3,7,5)) -- relevant to A2's causality question.

**Verdict: reproduction SURVIVES.**

---

## A1 — The calibration did manufacture the headline numbers; the geometry survives without them

**Claim attacked:** C1 — "V_low reaches −63 against V_true ~ 2; the same
qualitative signature as the model-based −1779."

**What I ran:** `poc/critic_b_a1.py`. (a) B4 recomputed on identical fits under
three xi conventions: shipped (`c/(N·λ_min(H))`), Phase-3 (`c/(N·σ2)` with σ2 =
smallest RETAINED eigenvalue of H under the PA rank), and a convention-free
reference (`c/N`). (b) 40 independent fits at (2,6,4) cf=1.0 to measure the
estimator's ACTUAL sampling spread along H's eigendirections, against the
region's width.

**Result (a) — the flagship number is a calibration artifact.** At c=10,
policy always_0, identical fits:

| config | cf | shipped V_low | Phase-3 convention | c/N reference |
|---|---|---|---|---|
| (2,6,4) | 1.0 | **−63.19** | **−1.28** | 0.46 |
| (3,7,5) | 1.0 | −50.80 | −2.35 | 1.11 |
| (4,6,2) | 1.0 | −51.56 | **−26.65** | 0.57 |
| (4,6,2) | 0.6 | −12.55 | −2.81 | 1.68 |
| (2,6,4) | 0.6 | −8.65 | +0.12 | 1.77 |
| (3,7,5) | 0.6 | −7.47 | −1.12 | 2.17 |

Under the model-based side's own convention, the (2,6,4) headline collapses
from −63 to −1.3: the shipped rule divides xi by λ_min(H) ≈ ρ (the ridge), so
the divergence magnitude was set by the regularizer, not the data. **However**,
the blow-up survives the convention change in the saturating config (4,6,2):
−26.65 vs truth 2.15, an order of magnitude, with a clean 1.0-vs-0.6 contrast
(−26.65 vs −2.81). So C1 is not dead — it is mislocated. The honest family
claim lives in the exact-cliff regime (where the smallest retained eigenvalue
is itself tiny), not in the statistical regime the headline row came from.

**Result (b) — the region measures identification width, not sampling
uncertainty.** Over 40 independent fits: std(J) = 0.11, |bias| = 0.17 — a
sampling-based interval is ≈ [1.63, 2.07], against a shipped penalty of ~65.
Along H's null eigendirections the stage-1 bridge moves with std ≈ 0.35 while
the region's half-width there is ≈ 90–110: the ellipsoid is ~300× wider than
the estimator's actual variability in exactly the directions that produce the
blow-up. Additional structure: for the deterministic target policy always_0,
the action-1 response is identically zero (`pi_e(1|o) = 0`), so the action-1
bridge is 0 by construction with zero sampling variance — yet contributes half
the penalty. And the unprojected null-direction width is
√(ξ/ρ) = √c/(√N·ρ) = √c/0.03 — **constant in N**: the bound never tightens
with data, which is the sharpest statement of the failure and is nowhere in
the doc.

**On (b)'s "right object" question:** H = design + ρI is the loss level set —
the same construction as the model-based side (T2 + λ2·I), so like-for-like
holds. But a loss level set is a *partial-identification* region: any bridge
inside it fits the moments equally well. Its width in null directions is set
by ρ, a free parameter. The defensible reading of both Phase 4 and step (b) is
therefore "identification-region pessimism cannot shrink in unidentified
directions", not "the estimator is wildly variable". A sampling-covariance
(sandwich/bootstrap) region would be ~two orders of magnitude tighter and
would NOT diverge — but it would also silently ignore the identification gap,
which is the phenomenon under study. The doc should say which uncertainty it
means; at present §4 reads as if it were sampling uncertainty.

**Verdict on C1: OVERSTATED.** The family-level failure is real but its
correct statement is: (i) the identification-region width in unidentified
directions is Ω(1) in N under any calibration (the √c/0.03 constant); (ii)
under the model-based side's own convention the divergence survives only where
the smallest retained eigenvalue is small (the saturating config). The −63
headline and the "−63 vs −1779" comparison should be replaced by the
convention-matched table above.

**Fix:** re-run B4 under the σ2 convention as the primary table; move the
shipped-convention numbers to an appendix with the ρ-dependence stated; add
the width-is-constant-in-N observation, which is stronger than any single
number.

---

## A2 (+A6) — Leakage is CAUSAL at fixed rank; C3 survives and is strengthened; C4's identity claim breaks

**Claim attacked:** C3 — leakage, not conditioning, drives the failure. The
worry: leakage and rank(P_a) both move with `confound`, so C3 could be a
passenger.

**What I ran:** `poc/critic_b_a2.py`. Emission-alignment sweep at FIXED
confound=1.0, (2,6,4): E's second row interpolated toward the first
(E_β = [E0, norm((1−β)E0 + βE1)], full row rank for all β > 0), so rank(P_a)
stays [1,1], coverage/policy/K0/rewards stay fixed, and only the alignment
between nu1 and the identified direction moves. Plus a p1-skew variant.

**Result — width tracks leakage at fixed everything-else:**

| β | rank(P_a) | cond(H) | leak | width ratio | V_low (c=10) |
|---|---|---|---|---|---|
| 1.00 | [1,1] | 4.8e2 | 0.381 | 15.9 | −63.2 |
| 0.50 | [1,1] | 4.1e2 | 0.108 | 4.1 | −33.5 |
| 0.25 | [1,1] | 4.3e2 | 0.024 | 2.7 | −15.8 |
| 0.10 | [1,1] | 4.5e2 | 0.003 | 1.4 | −7.0 |
| 0.02 | [1,1] | 4.7e2 | 0.0004 | 1.07 | −4.6 |

Conditioning flat, rank fixed, coverage fixed — leakage falls 1000× and the
width ratio falls to 1. The p1-skew variant shows the same per action
(leak 0.83 vs 0.03 for the two actions at p1 = (0.85, 0.15), fixed confound).
Leakage is causal. **C3 SURVIVES** — and this sweep is the experiment the doc
needed and did not run.

**But the same result kills C4 as an identity (A6).** C4 says "poor coverage
IS a collapsed per-action subspace, and the penalty IS the gradient leaking
out" — a five-step chain asserted, not measured. Steps 1–3 (confound →
determinism → rank collapse) were established last round (V3). Steps 4–5 fail
as an identity: along the β sweep, coverage and the collapsed subspace are
CONSTANT while the penalty moves by an order of magnitude; alignment between
the value functional and the subspace is an independent axis set by the
emission geometry and initial distribution. Coverage collapse is *necessary*
for a large penalty; it does not *produce* it. The Phase 4 coverage-penalty
monotonicity is one slice through a two-axis phenomenon.

**Verdict: C3 SURVIVES (now with causal evidence); C4 REFUTED as an identity,
salvageable as "coverage sets the subspace; alignment decides the penalty".**

**Fix:** fold the β sweep into the doc as the causal test of C3; restate §6's
chain with the alignment axis explicit; drop "the two results are one
phenomenon seen from two directions" or qualify it to the confound-driven
slice.

---

## A3 — There IS a chain to compound through; C2's "definitively excluded" is false

**Claim attacked:** C2 — "the model-free value depends on the first-stage
bridge only and linearly... there is no chain to compound through. So if this
diverges, compounding is definitively excluded."

**What I ran:** `poc/critic_b_a3.py`. The stage-1 response is
y = (r_1 + V̂_2(o_2))·π_e(a_1|o_1), where V̂_2 is the FITTED stage-2 bridge —
which used the fitted stage-3 bridge. I computed the population bridge (the
estimator run on expected counts at the same N-ridge, all stages) and, on the
same 20 datasets, fitted stage 1 twice: with the shipped estimated
continuation, and with the population continuation (later-stage error
surgically removed). A verbatim replica of the per-action solve reproduces
`bV_hat[0]` to 0.00e+00 before being trusted.

**Result — later-stage estimation error propagates into J:**

| config | cf | std(J) shipped | std of later-stage contribution | share of Var(J) |
|---|---|---|---|---|
| (2,6,4) | 1.0 | 0.072 | 0.025 | 12.2% |
| (2,6,4) | 0.6 | 0.064 | 0.043 | **45.6%** |
| (4,6,2) | 1.0 | 0.050 | 0.018 | 13.4% |

Nearly half of J's sampling variance at cf=0.6 comes from stages 2–3 flowing
through the response. What IS true: the dependence is additive-linear (the
response enters linearly), not the multiplicative bridge-product structure of
the model-based chain; and the propagation is small relative to the
identification width that drives C1 (0.04 vs a penalty of 65), so C1's
direction is safe — the stage-1-only ellipsoid UNDERSTATES uncertainty, it
does not inflate it.

**Verdict: C2 REFUTED as stated.** "Depends on the first-stage bridge only" is
false as a statement about the ESTIMATE, true only conditionally on the fitted
continuation. The defensible claim: the multiplicative compounding structure
of the model-based chain is absent, so the divergence cannot be attributed to
error compounding across bridge products; uncertainty still propagates
additively through the estimated continuation (12–46% of Var(J)) and the
stage-1-only confidence region ignores it.

**Fix:** rewrite §3 and the C2 sentence with "multiplicative compounding" in
place of "any chain"; note the stage-1-only region understates total
uncertainty and quantify with the population-continuation experiment (script
provided).

---

## A4 — Same-sample nu1 and t=1-only H: immaterial

**What I ran:** `poc/critic_b_a45.py` part 1 — every B1 cell recomputed with
the exact population nu1 = p1 @ E in place of the same-sample empirical
estimate.

**Result:** leakage and width move in the third decimal everywhere (e.g.
0.3813 → 0.3827, 14.171 → 14.153). The same-sample nu1 is a non-issue at
N=4,000. The t=1-only choice of H is correct for the DIRECT dependence (J
touches only the stage-1 bridge); the indirect later-stage channel it misses
is exactly the one quantified in A3 and should be cited there rather than
fixed here.

**Verdict: SURVIVES.**

---

## A5 — 36/36 validity is structurally guaranteed, not evidence

**Claim attacked:** "Validity holds throughout: V_low <= V_true in 36 of 36
cells."

**What I ran:** `poc/critic_b_a45.py` part 2 — per-seed margins (validity in
the driver is checked on seed MEANS), and for each cell the critical c* below
which the unprojected bound could first violate.

**Result:** in 8 of 12 (config, confound, policy) cells, V_hat <= V_true in
every single seed — the estimator is biased downward under confounding, so
validity holds there for ANY c >= 0 before a penalty is even subtracted. In
the remaining 4 cells the worst per-seed overshoot yields critical c* between
4.5e-07 and 8.0e-05 — five orders of magnitude below the smallest c on the
grid (0.1). The grid was structurally incapable of producing a violation;
per-seed violations are 0/180 for the same reason. "36/36" is a consequence of
(penalty >> estimation error), not a test of calibration.

**Verdict: OVERSTATED as reported.** The doc already says the bound is
"uselessly loose", but still presents 36/36 as a checked property. Fix: report
min c* (~1e-6) alongside, which states precisely how far from informative the
check is — or drop the validity column.

---

## A7 — The parallel-analysis selector survives its audit in the load-bearing role

**What I ran:** `poc/critic_b_a7.py` — (1) principal angles between the PA
basis (eigenvectors of the UNWEIGHTED cross-moment) and the top-k eigenbasis
of H (the WEIGHTED design that defines the widths), with leakage and projected
width recomputed in H's own basis; (2) selected-rank distributions over 20
seeds per cell at the experiment's actual N=4,000; (3) percentile sensitivity.

**Result 1 — the geometry mismatch I suspected is immaterial.** Max principal
angle is <= 4 degrees in 11 of 12 cells; the one outlier (50.7 degrees,
(4,6,2) cf=0.6 action 1 — a rotation inside a near-degenerate eigenvalue pair)
changes leakage by 0.0002 and the projected width by 0.018. Every C3 number is
basis-robust.

**Result 2 — rank selection at N=4,000 is right 75–100% of the time, with ±1
errors in both directions.** Notable cells: (4,6,2) cf=1.0 action 0 picks k=1
in 4/20 seeds (under-selection inflates measured leakage by counting a real
signal direction as leaked); (3,7,5) cf=0.6 action 0 picks k=4 in 5/20
(over-selection deflates it). With 5 seeds per cell in the driver, the doc's
leakage means carry this selection noise. It does not threaten the 24×
contrast (the misclassification effects are second-order against it) but
single-cell leakage values should not be quoted to three decimals.

**Result 3 — q=95 vs q=99 identical; q=90 slightly looser.** Not a tuned knob.

**Verdict: SURVIVES**, with the caveat that per-cell leakage numbers inherit
±1 rank-selection noise at this N and seed count. Fix: 20 seeds in B1, and
report leakage alongside the modal-k-conditioned value.

---

## Step (b) Round — summary

| claim | verdict | one line |
|---|---|---|
| C1 (shared blow-up, −63 vs −1779) | **OVERSTATED** | The −63 is manufactured by dividing xi by the ridge; under the model-based side's own σ2 convention it becomes −1.3. What survives: a genuine convention-robust blow-up in the saturating config (−26.7 vs truth 2.15), and the width in unidentified directions is √c/0.03 — constant in N — under the shipped rule. The family claim is real but lives elsewhere than the headline (A1). |
| C2 (compounding definitively excluded) | **REFUTED as stated** | The stage-1 response carries the fitted continuation: later stages contribute 12–46% of Var(J). True core: the MULTIPLICATIVE product structure is absent; propagation is additive and small vs the identification width (A3). |
| C3 (leakage, not conditioning) | **SURVIVES — strengthened** | Causal now: at fixed confound, rank, coverage and conditioning, an emission-alignment sweep moves leakage 1000× and the width ratio 15.9 → 1.07 tracks it (A2). Calibration-free, basis-robust (A7). |
| C4 (unification with Phase 4 coverage) | **REFUTED as identity; salvageable** | Coverage constant along the A2 sweep while the penalty moves 14×: coverage collapse sets the subspace (necessary), alignment decides the penalty. "One phenomenon seen from two directions" overclaims (A2/A6). |
| 36/36 validity | **OVERSTATED** | Structurally guaranteed: 8/12 cells valid before any penalty; elsewhere c* ≈ 1e-6 vs grid min 0.1 (A5). |
| nu1 / t=1-only H | SURVIVES | Third-decimal effects (A4). |
| PA selector (mine) | SURVIVES | Basis mismatch immaterial; 75–100% rank accuracy at N=4k with ±1 noise; q-insensitive (A7). |

**Bottom line for the paper:** the durable results of step (b) are (i) the
calibration-independent geometry — leakage causally drives the width ratio,
now with a fixed-rank causal experiment to prove it — and (ii) the
convention-robust blow-up in the saturating regime plus the width's
N-independence in unidentified directions. The −63 headline, the "no chain to
compound through" argument, and the coverage-identity story should all be
rewritten; each currently overclaims in a way an informed referee can puncture
with one experiment. My own contribution (the PA selector) held up, with a
quantified seed-noise caveat.

**Critic scripts this round:** `poc/critic_b_a1.py`, `poc/critic_b_a2.py`,
`poc/critic_b_a3.py`, `poc/critic_b_a45.py`, `poc/critic_b_a7.py`.
