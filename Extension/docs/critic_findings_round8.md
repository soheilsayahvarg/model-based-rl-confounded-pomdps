# Critic Findings — Round 8

Targets: `claims.md` (the ledger), the two-switch result (§5c), the 20-seed
decision test (§6.2b–c), the §4b reproduction, and the three new scripts.
Verdicts: REFUTED | UNPROVEN | OVERSTATED | SCOPED | SURVIVES.

## Checklist (resume from first unticked)

- [x] B0. Read claims.md, completeness_and_h4.md, regret_vs_schedule.md,
      family_pessimism.md §4b, the three scripts, run_completeness_beta.py
- [x] B1. Recompute all quoted numbers from experiments/*.json (five results)
- [x] B2. P2: is the alignment sweep tautological? Build and run a
      non-degenerate knob (sweep p1 with E fixed: rows stay maximally distinct,
      rank fixed, only the gradient's angle to the span moves)
- [x] B3. P3: two-switch derivation — K0 row sums, t > 1 (X_t includes
      history: does per-action completeness get restored at t >= 2?), angle vs
      product in the floor, novelty vs Lemma C.1 / proximal folklore
- [x] B4. P4 from JSON: schedule B flat-after-transient, plug-in-beaten cells
      (dominance vs crossing contradiction), reported widths vs 0.12 target
- [x] B5. P4 rerun: random candidate sets + calibration on a different policy +
      continuous margin, schedules C and paper
- [x] B6. P5: the kappa=0.5 buried claim
- [x] B7. P1: ledger audit — row statuses, missing rows, are four legs four,
      is "the recurring defect" a real generalisation
- [x] B8. Round summary

## State notes (for resume)

Queued observations, to verify in B1–B7:
 (1) Ledger "Plug-in dominance — never beaten ... reaches 0.000 where no
     pessimistic schedule does" CONTRADICTS its own crossing row: §6.2b paper
     schedule has plug-in 0.1641 vs pessimism 0.0694 at N=2k — the plug-in IS
     beaten in the incomplete design at small N. Dominance row needs scoping.
 (2) "The recurring defect: all four are the same failure" — #4 (plug-in at
     one kappa) is a missing-control error, not a range-straddle error; the
     precondition guard would not have caught it. 3/4, not 4/4.
 (3) §4b population sweep: given W = sqrt(xi)*||g||_{H^-1}, the ratio
     unproj/proj is a near-algebraic monotone function of beta_g at fixed rho
     — the population "reproduction" verifies algebra, not the phenomenon; its
     legitimate content is the existence counterexample against cond(H).
     The alpha-knob endpoint is degenerate (rows identical at alpha=1).
     Non-degenerate knob: sweep p1 with confound=1.0 — span = E^T e_{s_a} is
     p1-independent, nu1 = E^T p1 moves, E untouched (informativeness fixed).
 (4) Two-switch at t>1: X_t includes history, so the per-action conditioning
     cells multiply (|O|*|A|*|O_0| at t=2); the population span plausibly
     reaches |S| and beta_pop(t>=2) = 0 even when |O_0| < |S| — the truth
     switch may be a stage-1 phenomenon. Check in population.
 (5) beta*beta_g floor: the product needs no angle — two independent
     inequalities; achievable at xi = lam*beta^2 regardless of the angle
     between the two null projections. Should SURVIVE with a proof note.
 (6) Schedule B: 0.1230 -> 0.0644 -> 0.0644 -> 0.0644 with plug-in also
     improving early under B (slope −0.037) — the transient is plausibly the
     shared estimator, not the penalty; "flat after transient" may be the
     right statement and the ledger's "refuted" an over-correction.
 (7) Missing ledger row candidates: schedule A non-monotone (e<0 prediction
     not confirmed, §6.3); kappa=0.5 plug-in stuck at 0.164 while pessimism
     reaches 0.084–0.098 (P5's buried claim).

---
## B1 — Recomputation from raw JSONs: the quoted numbers are right; several
## things the JSONs contain are not quoted, and they matter

All checked against `experiments/*.json`, not prose: §6.2b's C and paper tables
reproduce cell-for-cell; the slope table (+0.0056/−0.0127/+0.0220/+0.0135 pess,
+0.0022/−0.0370/−0.0210/−0.0288 plug) matches; §4b's alignment rows (78×,
cond 1.58×, ratio 24.74→1.014) match; the leakage-20 disagreement cells match;
the (4,6,2) leakage-map row (beta 1.178, beta_g 0.364, floor 0.4285) matches;
completeness 50/50 with sharp separation confirmed (complete cells max rel
1.2e-15; incomplete 0.569–0.773); coverage `c_slope = 2.158` confirmed;
`T_invariance max_dH = max_db = 0.0` confirmed. **No fabricated or drifted
number found anywhere. The problems below are all about what the numbers mean
or what was left unreported.**

What the decision-test JSON contains that the doc does not say:

1. **The calibration start-point guarantee FAILED for schedules A and B.**
   §4.5b/§6 claim every schedule "starts with the correct selection and the
   sweep tests slope alone". The stored pick distributions at N=2,000:
   A = {soft: 10, uniform: 9, greedy_lo: 1}; B = {always_0: 10, greedy_lo: 7,
   soft: 3}. Neither starts at greedy_lo; C starts there only 13/20. The
   realized starting widths (best-policy gradient, 20-seed mean) are
   0.171–0.200 against the 0.12 calibration target — i.e. AT or ABOVE the
   ~0.19 boundary the calibration was supposed to stay below. The width was
   calibrated on seed 0 with greedy_lo's gradient and verified on neither.
   **This is the recurring defect's fifth occurrence, inside the very run
   whose precondition guard was billed as the structural fix** — the guard
   checks `beta_g > 0` but never checks the thing that actually broke run 1
   (the starting point). The C/paper conclusions survive (both start at
   greedy_lo and degrade monotonically from there); the A and B rows' early
   points are start-point artifacts.

2. **"Schedule B is not flat" is an OVER-CORRECTION** (the failure mode I was
   asked to watch). B's 0.1230 at N=2k is the mis-calibrated start (modal pick
   always_0, width 0.1995 > boundary); from the first correctly-started point
   B is 0.0644 → 0.0644 → 0.0644, flat to the digit — and the plug-in on the
   SAME fits moves 0.0943 → 0.0644 → 0.0032 over that range, so the moving
   part of B's curve is the shared estimator transient, not the penalty. The
   right statement is the one the round-8 prompt suspected: **flat after a
   transient that the run's own design caused**, which is the e = 0
   prediction behaving, not failing. The ledger row "Schedule B is a flat
   e=0 control — refuted" should be REINSTATED as "holds, from the first
   correctly-calibrated point"; the §6.2c correction traded a 3-seed artifact
   for a calibration artifact.

3. **Pessimism BEATS the plug-in in 4 of 16 cells, with separated intervals.**
   A@8k (0.0964±0.0154 vs 0.1641±0.0000), A@32k (0.1193±0.0223 vs 0.1641),
   paper@2k (0.0694±0.0098 vs 0.1641), paper@8k (0.0943±0.0205 vs 0.1641).
   Under schedule A the plug-in is STUCK at 0.1641 at every N — at
   kappa = 0.5 the ridge decays too slowly for `b_hat` to escape its bias on
   this environment, and the pessimism layer partially compensates at mid N.
   This is Priority 5's buried claim, and it is real at 20 seeds (see B6).

4. **"Pessimism never finds the optimum, under any schedule" is false at the
   seed level**: individual seeds select `always_1` under C at N=8k and 32k
   (and the plug-in reaches it under B@128k, C@8k+). True as a modal/mean
   statement; the ledger's absolute phrasing overstates.

5. **"The plug-in finds it: 0.000 at kappa = 1.0 and 1.5" is a 3-seed relic.**
   At 20 seeds, kappa=1.0/N=128k gives 0.0032 ± 0.0063 (not all seeds arrive);
   0.000 exactly holds only at kappa = 1.5 (N ≥ 32k). Ledger should carry the
   20-seed value.

---

## B2 — Priority 2: the population "reproduction" is an identity — and §3
## still SURVIVES, on a knob with nothing degenerate about it

**The tautology, made exact.** With `confound = 1.0` the per-action span is
rank 1, so `H = w*mm' + rho*I`, and the support-function algebra gives, exactly:

```
ratio^2 - 1  =  [beta_g^2 / (1 - beta_g^2)] * cond(H).
```

`poc/critic8_align_p1.py` verifies this on every row of the §4b sweep to
rel. err ≤ 4e-14. Consequences:

- **The §4b population sweep could not have failed.** Given the formula, the
  ratio is an algebraic function of (beta_g, cond); computing both and
  observing they co-move is not a test. "§3 survives independent
  reproduction" is therefore OVERSTATED as written — §4b is a *consistency
  computation*, and the only §3 evidence that could have failed remains the
  review's empirical sweep (sampled fits, PA-selected bases).
- **The 1.58× question has an exact answer, and it favours the claim.** The
  identity factorises the ratio into a conditioning factor and a leakage
  factor, so their contributions separate multiplicatively: across the sweep
  cond contributes 1.58× while `beta_g^2/(1-beta_g^2)` contributes ~22,000×
  (0.758 → 0.00975). "Barely moves" is not a judgement call here; it is a
  factorisation. Conditioning is excluded as the driver *by algebra*, which
  is stronger than the doc's phrasing — but for the honest reason (identity),
  not the stated one (reproduction).

**The degeneracy suspicion is right about the knob, wrong about the
conclusion.** The alpha knob does drive the environment to zero observational
information at its endpoint, so `beta_g -> 0` there is forced. The
non-degenerate knob exists and is cleaner than a rotation: **sweep `p1` with
`confound = 1.0`**. The span direction is the acting state's own emission row
(p1-independent); `nu1 = E^T p1` moves; `E`, `K0`, `rank(E)` and the row
separation (0.8136) are untouched, so the POMDP stays fully informative at
every point. Measured (`critic8_align_p1.py` part 2):

| p1[0] | beta_g | ratio | cond(H) |
|---|---|---|---|
| 0.05 | 0.9499 | 64.4 | 4.498e+02 |
| 0.50 | 0.6073 | 16.2 | 4.498e+02 |
| 0.95 | 0.9498 | 64.6 | 4.529e+02 |

`beta_g` sweeps 0.607–0.950 in both directions, the ratio follows it
monotonically 16→65, and **cond(H) moves 1.01×** — tighter than the 1.58× of
the alpha sweep, on an environment that never loses informativeness. (Range
caveat: this knob cannot reach beta_g ≈ 0, since `nu1` is a convex combination
of emission rows; the low-leakage end is covered by the alpha sweep's
non-degenerate midrange.)

**Verdicts:** §3 (leakage drives the penalty, conditioning does not)
**SURVIVES**, now on a fully-informative knob and with the driver question
settled by factorisation rather than by sweep. "§3 survives independent
reproduction" (§4b, ledger row) is **OVERSTATED — reword**: the population
sweep is an identity computation, not a reproduction; the ledger row should
say "holds — population identity + informative-knob check (round 8); the only
empirical test remains the review's §3 sweep". The alpha-knob table itself is
fine to keep as illustration.

---

## B3 — Priority 3: the two-switch derivation survives at t = 1, and t > 1
## is where its scope actually lives

**The span identity.** `sum_x v_{a,x}[s] = p1[s]*pi_b[s,a]*sum_x K0[s,x]` and
K0's rows sum to 1 in the code (verified: `[1,1,1,1]`), so with `pi_b`
constant in `s` (verified: exactly uniform at `confound = 0`) the sum is
`c_a * p1` and `nu1 = E^T p1` lies in the span. SURVIVES — with one
dependency the doc omits: the argument needs K0 row-normalised. If O_0 were
subject to state-dependent missingness (rows summing to different constants),
the sum would return `E^T(p1 ∘ rowsum)` instead and the gradient could leak
WITHOUT confounding. Worth one sentence in §5c; not a defect of the code.

**t > 1 REFUTES the unscoped claim** (`poc/critic8_t2span.py`, population,
exact). At t = 2 the conditioning set is `(A_2, O_1, A_1, O_0)` —
`|O|*|A|*|O_0|` cells per action — and the o_1-conditioned posteriors span all
of R^|S| at every `confound < 1`:

| config | confound | t1 span | t2 span | t2 beta_pop (rel) |
|---|---|---|---|---|
| (4,6,2) | 0.0–0.9 | 2 | **4 = \|S\|** | ~1e-15 (0.0%) |
| (4,8,3) | 0.0–0.9 | 3 | **4 = \|S\|** | ~1e-15 (0.0%) |
| (4,6,2) | 1.0 | 2 | 2 | 1.179 (69.5%) |

**History restores per-action completeness at every stage after the first**,
except on the confound = 1 knife-edge. So "incompleteness (`|O_0| < |S|`)
makes the truth leak" is exactly a **stage-1-block statement**: `bR_t1`/`bD_t1`
(whose conditioning is `(A_1, O_0)` alone) leak; every t ≥ 2 block has
`beta_pop = 0`. Nothing measured is wrong — `completeness_and_h4.md` §5 ran
`T = 1` explicitly, and the decision test's floor is still carried by the
t = 1 blocks inside the T = 3 value — but the ledger row and §5c state the
switch without the stage scope, and the practitioner pitch ("checkable from
dimensions alone") silently concerns the first stage only. **SCOPED, not
refuted.** One corollary worth stating: with T > 1, an inadequate O_0 is
partially self-healing — the history takes over the instrument's job from
stage 2 on.

**Angle vs product.** The floor needs no joint geometry: coverage forces
`xi >= lam*beta^2` (truth's null offset), and `W >= sqrt(xi/lam)*beta_g`
(gradient's null mass) — two independent inequalities whose product cancels
`lam`, valid for ANY angle between the two null projections, and achieved at
the minimal covering `xi` regardless of that angle. The product is the right
object for a floor; the angle would only enter a matching upper bound.
SURVIVES.

**Novelty.** The gradient half is derivable from the anchor's own machinery:
no confounding ⇒ `pi_b` is observable ⇒ importance weights
`pi_e(a|o)/pi_b(a)` exist on observables for every observation-based policy ⇒
`C*_pi < inf` ⇒ (the round-6 Lemma C.1 argument) the population gradient
cannot leak. In proximal-inference terms it is the folk statement that without
unmeasured confounding the negative control is unnecessary. So: the
*mechanism* is not new; the **conjunction** — truth-leak switched by a
dimensional condition, gradient-leak switched by confounding, floor = the
product, all computable pre-fit — is a tidy packaging I have not seen stated,
and the graded-in-confounding measurement is new. The ledger row is fine if
"holds" means the result, not its novelty; a novelty column entry like the
dichotomy's would say "mechanism folk / packaging new".

---

## B6 — Priority 5: the reversal was right, and it buried a two-sided result
## that is better than the one-sided one being led with

The per-kappa control was the correct design — `b_hat` depends on `lambda_2`,
so the one-kappa comparison was confounded, and reversing it was not an
over-correction. But the 20-seed JSON shows the corrected comparison is
genuinely **two-sided**, and the current presentation keeps only one side:

- **Where pessimism wins, with separated 95% intervals (4 of 16 cells):**
  A@8k 0.0964±0.0154 vs plug-in 0.1641±0.0000; A@32k 0.1193±0.0223 vs 0.1641;
  paper@2k 0.0694±0.0098 vs 0.1641; paper@8k 0.0943±0.0205 vs 0.1641.
- **The mechanism is legible.** At `kappa = 0.5` the ridge decays slowly, the
  plug-in is pinned to the ridge-biased pick (`always_0`, regret 0.1641 at
  every N — its "slope +0.0022" is a flat line at the biased value, not
  convergence), and the pessimism penalty happens to demote that pick. Under
  the paper schedule at small N the same thing happens before the estimator
  escapes.
- **Where the plug-in wins:** everywhere at the largest N (strict in 3 of 4
  schedules, tie at A), with the complete separation at kappa=1.5/128k.

So the defensible statement is not "plug-in dominance, extended" but:
**the pessimism layer helps exactly where its sales pitch says it should —
scarce data or a heavily-regularised estimator — and turns strictly harmful as
data accumulates under `e > 0`; the plug-in wins every asymptote.** The
crossing row of the ledger already says half of this; the dominance row
contradicts it. Verdict: reversal SURVIVES; the dominance row's status
**"holds, and extended" is WRONG as stated** — it is refuted cellwise by its
own experiment's JSON and must absorb the crossing.

---

## B7 — Priority 1: the ledger audit

**Wrong or mis-stated statuses (5 rows):**

1. *"Plug-in dominance — holds, and extended | never beaten on the toy; and in
   the incomplete design it reaches regret 0.000 where no pessimistic schedule
   does."* **WRONG as stated.** Toy half: fine. Incomplete half: beaten in 4
   cells with separated CIs (B6), contradicting the ledger's own crossing row
   two lines below; and "reaches 0.000" holds exactly only at kappa=1.5 (at
   kappa=1.0, 20 seeds give 0.0032±0.0063). Rewrite: "dominates every
   asymptote (strict in 3/4 schedules, tie under A at the biased value);
   beaten by pessimism at small-N/heavy-ridge in 4/16 cells; 0.000 at
   kappa=1.5, 0.0032±0.0063 at kappa=1.0."
2. *"Schedule B is a flat e=0 control — refuted."* **OVER-CORRECTION;
   reinstate.** The N=2k point started above the decision boundary (calibration
   failure, B1.1); from the first correctly-started point B is flat to the
   digit while the plug-in on the same fits still moves. The e=0 prediction
   behaved; the run's start did not.
3. *"§3 reproduced independently — holds, by a different route."*
   **OVERSTATED.** The population sweep is an identity
   (`ratio^2-1 = [beta_g^2/(1-beta_g^2)]*cond(H)`, verified to 4e-14) and
   could not have failed. Reword: "population identity + informative-knob
   check (round 8); the only reproduction-that-could-fail of the *empirical*
   §3 sweep has still never been run." The companion row "§5/§6 numbers
   reproduced independently — open" is honest; §3's row should not claim more
   than it.
4. *"Incompleteness leaks the truth ... the floor needs both — holds."*
   **SCOPED by t>1** (B3): history restores per-action completeness at every
   stage ≥ 2 for confound < 1 (span = |S|, beta_pop ~ 1e-15, 24/24 cells), so
   the switch governs the stage-1 blocks; `|O_0| < |S|` is a stage-1
   criterion. The iff itself generalises (t=2 spans are complete AND leak-free
   — the 50/50 dichotomy survives per-design); the dimensional shortcut does
   not.
5. *"Pessimism never finds the optimum here, under any schedule — holds."*
   **Overstated absolute**: seeds select `always_1` under C at N=8k/32k.
   Reword to "never the modal selection at any (schedule, N)".

**Missing rows (all inconvenient, which is the point):**
- Schedule A under `e < 0`: prediction "stays at its start" NOT confirmed
  (non-monotone, ends at the plug-in's biased value). §6.5 records it;
  the ledger's decision table silently omits it.
- The 4-cell pessimism-beats-plug-in result (B6) — the single most
  inconvenient fact for leg 4, present in the JSON, absent everywhere.
- The §4.5b start-point guarantee ("every schedule starts with the correct
  selection") — now known FALSE for A and B; needs a refuted row.
- The `e_paper > 0` leg does not carry regret_vs_schedule §4.4's own
  scope-down ("practical bite conditional on `C*_pi = infinity`") — the
  ledger states it in the step-(b) row's scope but not on the leg.

**Are the four legs four? No — three legs and an artifact.** Leg 1 is a
reproducibility/artifact contribution (oracle-verified implementation + defect
taxonomy), legitimate but a different genus from a finding; it is not "we
fixed our bugs" (the audit trail has independent value), but it belongs in a
different column than legs 2–4. Leg 2 is small and must carry its §4.4 scope
inline. Leg 3 is the strongest and needs the stage-1 scope. Leg 4 as worded is
refuted by its own JSON and must become the two-sided statement (B6) — which
is, in fact, a *better* leg: "helps where pessimism should help, harms
asymptotically, crossing included" survives referee contact in a way blanket
dominance does not.

**"The recurring defect": 3/4 real, the fourth is a different genus, and the
"structural fix" is already known-partial.** Incidents 1–3 are genuinely one
failure (measurement range does not straddle the predicted transition).
Incident 4 (plug-in at one kappa) is an uncontrolled-covariate error —
`lambda_2` changed along with the thing being compared — which no
range-straddling check would catch. And the guard in
`run_regret_incomplete.py` checks exactly one precondition (`beta_g > 0`)
while the SAME run violated a second stated precondition (start below the
decision boundary, B1.1) that the guard does not check. Calling it "the first
structural fix" is generous by one round: it is a fix for incident-class
1–3 applied to one script, already outflanked in its own run. The honest
generalisation: *every incident is a measurement whose meaning turned on an
unverified design premise; the fix is to assert premises in-run, and the
round-8 finding shows the assertion list was incomplete the first time it was
tried.*

---

## B5 — Priority 4, the robustness rerun: schedule C's degradation is robust;
## the paper-schedule CROSSING is not — it is calibration- and set-dependent

`poc/critic8_decision_robust.py` — 10 seeds, schedules C and paper, all
candidate sets evaluated on the SAME fits; two calibrations (the run's
greedy_lo/seed-0, and always_1 averaged over seeds 0–2, which halves `c`);
regret plus the continuous margin `M = V_low(best_true) − max V_low` (≤ 0,
more negative = the optimum further from being selected). Raw:
`experiments/critic8_decision_robust.json`.

**What survives everything:**
- **Schedule C degrades under both calibrations** (pess slope +0.0242 orig,
  +0.0185 alt) while the plug-in improves (−0.0223), reproducing the 20-seed
  pattern at 10 seeds.
- **The continuous margin worsens with N under `e > 0` in 6 of 8
  non-degenerate (schedule, set) cells** — including rand42, where the
  discrete regret is FROZEN at 0.0837 at every N for every method while the
  optimum's V_low deficit still grows (−0.36 → −0.56): the margin metric
  shows motion the 6-policy regret cannot. The discreteness worry was
  justified, and resolving it *helps* the mechanism claim.

**What does not survive:**
- **The crossing is calibration-dependent.** Under the paper schedule with the
  alternative calibration (c halved), pessimism is better at N=2k–8k and then
  TIES the plug-in at 32k–128k (0.0644 = 0.0644); its slope flips sign
  (+0.0160 → −0.0194). No crossing in the tested range. This is what the
  `e > 0` mechanism itself predicts — the degradation onset N scales with the
  width constant — but it means "under the paper's own schedule the two curves
  cross" is a statement about ONE (c, N-range) pair, not about the schedule.
- **The crossing is candidate-set-dependent.** On rand17 the paper-schedule
  plug-in is at 0.0000 from N=2k (nothing to be better than) and pessimism
  only degrades at 128k; on rand42 every method picks the same policy at every
  N and the comparison is blind.
- **NEW, and inconvenient for the OTHER side: the plug-in converges to a
  WRONG policy on rand17 under C** — 0.0205 → 0.0922 → 0.1025 → 0.1025,
  degrading with N and settling 0.1025 below the best (value spread only
  0.117). In an incomplete design `b_hat`'s asymptote is not a valid bridge,
  so the plug-in carries an asymptotic bias; when candidate values are spaced
  more tightly than that bias, "the plug-in finds the optimum" fails. The same
  set at the paper kappa gives 0.0000 everywhere — so plug-in reliability in
  incomplete designs depends on BOTH the ridge schedule and the candidate
  spacing. Leg 4 loses its last unconditional clause.

**Verdicts:** "the width schedule changes the decision where both switches are
on" **SURVIVES** (carried by schedule C + the margin metric, robust across
calibrations and sets); "under the paper's own schedule they cross"
**SCOPED** — true at the tested (c, candidate set, N-range), moves or
disappears under either perturbation, and should be stated as "the degradation
onset lies inside/outside the data range depending on the width constant";
plug-in-finds-the-optimum **SCOPED** to candidate sets whose value gaps exceed
the incompleteness bias.

---

## B8 — Round 8 summary

| target | verdict | one line |
|---|---|---|
| Ledger numbers | **SURVIVE** | every quoted number recomputes from the JSONs; no drift anywhere |
| Ledger as a ledger | **6 rows wrong or mis-stated, 4 rows missing** | see B7; worst: "plug-in dominance — holds, and extended" is contradicted by its own experiment's JSON |
| Four legs | **three legs + an artifact** | leg 1 is a reproducibility artifact, not a finding; leg 4 must become the two-sided statement |
| "Recurring defect, same failure ×4" | **OVERSTATED (3/4)** | incident 4 is an uncontrolled covariate, not a range failure; and the "structural fix" run itself violated a second unchecked premise (start-point, B1.1) — the fifth occurrence |
| §4b "independent reproduction" of §3 | **OVERSTATED** | it is an identity (`ratio²−1 = [βg²/(1−βg²)]·cond`, err ≤ 4e-14) and could not have failed |
| §3 itself | **SURVIVES** | p1 knob: fully informative environment, cond moves 1.01×, ratio tracks beta_g 16→65; and the 1.58× question is settled by factorisation, not judgement |
| Two-switch derivation (t=1) | **SURVIVES** | K0 rows sum to 1 in code; pi_b exactly constant at cf=0; product needs no angle |
| Two-switch scope (t>1) | **SCOPED** | history restores per-action completeness at every t ≥ 2 for confound < 1 (span = \|S\|, beta_pop ≈ 1e-15, 24/24 cells); the switch is a stage-1-block statement |
| Two-switch novelty | mechanism folk, packaging new | gradient half derivable from Lemma C.1 + observable importance weights |
| 20-seed decision test, schedule C | **SURVIVES** | degradation robust to calibration, candidate sets, and the continuous metric |
| 20-seed decision test, the crossing | **SCOPED** | disappears at half the width constant (ties instead), and on both random candidate sets |
| Schedule B "not flat — refuted" | **OVER-CORRECTION, reinstate** | the N=2k point started above the decision boundary; flat to the digit from the first correct start |
| Calibration start-point guarantee (§4.5b) | **REFUTED** | A starts at soft/uniform, B at always_0; realized widths 0.171–0.200 vs the 0.12 target |
| Plug-in reversal (P5) | **SURVIVES; incomplete** | the reversal was right; the buried 4-cell pessimism-wins result (separated CIs) and the rand17 plug-in failure must both surface |

**Over-correction check (standing rule): two found** — the schedule-B
"refuted" row (B1.2), and the §4b claim demoting the review's empirical sweep
to "quoted, not reproduced" while promoting an unfalsifiable identity in its
place (the empirical sweep remains the only §3 test that could fail; the
ledger treats it as the weaker evidence).

**What the ledger needs (minimum edit set):**
1. Rewrite the dominance row two-sided (B6): asymptotic dominance + the four
   small-N/heavy-ridge cells where pessimism wins + the rand17 caveat.
2. Reinstate schedule B as a flat control from the first correct start; add a
   refuted row for the §4.5b start-point guarantee.
3. Rescope the crossing row to its (c, candidate set, N-range); keep schedule
   C as the load-bearing decision-level cell.
4. Stage-scope the two-switch rows (`|O_0| < |S|` is a stage-1 criterion;
   history heals later stages); add the K0-row-sum dependency to §5c.
5. Reword "§3 reproduced independently" per B2; add the p1-knob run as the
   informative-environment check.
6. Reword "never finds the optimum" to "never the modal selection"; update
   kappa=1.0 plug-in to 0.0032 ± 0.0063.
7. Recount the legs: three findings + one artifact, with leg 2 carrying its
   §4.4 scope inline.
8. Restate the recurring-defect note as 3-of-4 + a partial guard, citing the
   fifth occurrence as evidence the premise-assertion list was incomplete.

**Critic scripts this round:** `poc/critic8_align_p1.py`,
`poc/critic8_t2span.py`, `poc/critic8_decision_robust.py`; raw output in
`experiments/critic8_decision_robust.json`.
