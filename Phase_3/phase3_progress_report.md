# Phase 3 Progress Checkpoint

## Confounded POMDPs: Dual-Bridge Function Identification & Signal-Projected Pessimism

**Team:** Soheil Sayah Varg, Mahdi Shirinbayan, Mahdi Ostadmohammadi
**Deep Reinforcement Learning — Progress Checkpoint (Phase 3)**

---

## How to Present This (suggested 10–12 minute flow)

1. **Recap the question** (30s) — does the Hong–Qi–Xu confounded-POMDP framework actually
   work end-to-end, and can its intractable pessimism step be made to run at all?
2. **What's built and verified** (2 min) — the full pipeline is implemented and matches
   exact ground truth to machine precision.
3. **Finding #1 — the theory detonates** (2 min) — the paper's own pessimism formula
   returns `-1779` when actually run.
4. **Finding #2 — our repair, and its honest limit** (2–3 min) — Signal-Projected
   Pessimism fixes the numbers and gets zero selection regret, but we found and disclose
   a real gap in its guarantee.
5. **Finding #3 — the benchmark boundary** (2 min) — at realistic scale, a dumb baseline
   beats our method, and we know exactly why. Worth saying out loud: two of these numbers
   changed in our final rigor pass when we added more seeds (§3.5) — lead with that as a
   sign of process, not as a mistake to downplay.
6. **What changed since the proposal, and why** (1–2 min).
7. **Roadmap to Phase 4 + risks** (1–2 min).
8. **Specific asks for the mentor** (§8 below).

---

## 1. Executive Summary

We set out to build the first empirical implementation of the Hong, Qi & Xu (ICML 2024)
model-based bridge-function framework for confounded POMDPs — a purely theoretical paper
with no released code or experiments — and to make its intractable pessimistic
optimization step tractable. We have a complete, oracle-verified pipeline: environment →
two-stage bridge estimation → policy-value identification → pessimistic policy selection,
running on both a controlled toy environment and a 720-observation benchmark built from a
public clinical simulator.

Three things came out of actually running the theory that were not visible on paper:
the paper's own pessimism step is numerically unusable as written (values diverge to
`-1779`); our fix for it (an SVD-based signal-subspace projection) works empirically but
is a heuristic, not a certified bound, and we can quantify exactly how much guarantee it
gives up; and at realistic sample sizes and observation-space size, the whole model-based
approach is beaten by a confounding-blind baseline, because estimation variance
outweighs the bias it is designed to remove. We have since implemented and tested a
variance-reduction upgrade and a competing model-free baseline to map that landscape
precisely. All of this is backed by machine-precision correctness checks and is reported
honestly, including where the news is negative.

In a final pre-checkpoint rigor pass, every multi-seed result in this report was moved to
5 seeds (up from 2–3) with
95%-CI reporting — we prioritized this pass specifically because two of our headline
claims rested on a 3-seed standard deviation, which we no longer trust as a stable
number. That pass **changed two of our own headline claims**: the model-based
estimator's worst-case harm at zero confounding was `23–46×` naive's error on 3 seeds and
is `2.8–6.8×` on 5; a claimed "6× variance cut" from shrinkage does not replicate and is
now retracted. Both are flagged explicitly in §3.5 and §4 — we would rather correct our
own numbers before the meeting than after.

---

## 2. Completed Work

### 2.1 Core pipeline (environments + exact oracles)

- A minimal tabular confounded POMDP (`|S|=2, |A|=2, |O|=3, T=3`) with **exact** oracle
  bridge functions (closed-form pseudo-inverse) and exact dynamic-programming policy
  values — this is our ground-truth substrate for every correctness check.
- A 720-observation benchmark environment built on the public `gumbel-max-scm` clinical
  simulator, augmented with an absorbing-state structure, a confounding-strength knob
  (`kappa`), and a **manufactured pre-decision negative control `O_0`** constructed so
  the paper's Assumption 3.1 holds by construction.
- A naive, confounding-blind observation-MDP evaluator as the baseline every method is
  measured against.

### 2.2 Estimation layer

- The paper's two-stage bridge estimator (Stage-1 ridge conditional-mean embedding,
  Stage-2 closed-form solve), implemented in **both** the dual (representer) and primal
  (operator) forms, verified algebraically identical.
- A factored, benchmark-scale version of the same estimator that avoids ever
  materializing the combinatorial conditioning-set size (which would otherwise explode
  to billions of cells) via three exact structural factorizations, so the full 720-state
  benchmark runs on a laptop, no GPU.
- Grid-calibrated regularization schedules (§4/§5 below), replacing the paper's
  theoretical rates, which do not transfer to the tabular setting.

### 2.3 Identification and pessimism layer

- An exact tensor-contraction chain (Theorem 3.5) that turns estimated bridges into a
  policy value, verified against dynamic programming to machine precision.
- A closed-form ellipsoidal-pessimism solver for the paper's confidence-region
  optimization, plus our own **Signal-Projected Pessimism** repair for the failure mode
  we discovered (§3, §4).

### 2.4 Benchmark-scale extensions (beyond the original 2-page proposal)

In direct response to the benchmark boundary result (§3), we added:

- **Variance-reduction estimator** — a backward-compatible subclass adding James–Stein
  shrinkage and K-fold cross-fitting to the benchmark estimator.
- **Model-free proximal baseline** — an independent implementation of a minimax
  value-bridge estimator (Shi et al., 2022) as a genuine competing method, not just a
  strawman, for a head-to-head comparison.
- **A comprehensive sweep** across 5 confounding strengths × 3 sample sizes × 5 seeds ×
  4 methods (naive, model-based plain, model-based shrunk, model-free), producing a full
  empirical landscape with 95% confidence intervals rather than isolated data points.

### 2.5 Continuous state/action extension (toy-scale PoC)

Beyond the discrete pipeline above, we built a self-contained continuous state/action
extension of the same dual-bridge identification and pessimism machinery: a kernel/RKHS
bridge estimator on a linear-Gaussian confounded POMDP with an exact closed-form oracle,
reusing `ellipsoid_opt.py`'s pessimism engine unmodified in coefficient space, plus a
dedicated 5-seed driver script. Oracle self-consistency holds to `1.665e-16`; plug-in
de-biasing beats naive in **4/4** candidate policies; pessimism validity (`V_low ≤
V_true`) holds in 100% of tests — though, reported honestly rather than smoothed over,
pessimistic *selection quality* is currently worse than plain plug-in selection there, an
open item. Full derivation, code map, and results: `docs/continuous_extension.md`.

### 2.6 Validation infrastructure

Four independent regression suites (environment/oracle, estimator, plug-in/pessimism,
benchmark), all currently green, plus three rounds of adversarial self-audit (below) that
materially changed how we report two of our own headline claims.

---

## 3. Preliminary Results & Key Findings

### 3.1 Correctness (the machine-precision handshake)

| Check | Result |
|---|---|
| Policy-value chain vs. exact dynamic programming | `4.4 × 10⁻¹⁶` max gap |
| Estimator vs. exact oracle bridges (population limit) | `3.0 × 10⁻¹⁵` |
| Dual vs. primal closed-form solve (independent derivations) | `2.4 × 10⁻¹⁵` |
| Analytic gradients vs. finite differences | `4.9 × 10⁻¹⁰` |

Every subsequent result is therefore a fact about the *method*, not an artifact of the
code — this is the bedrock the rest of the report stands on.

### 3.2 Convergence (where the method works as intended)

On the toy environment, bridge-recovery error decreases **monotonically at every
step** as sample size grows:

| `N` | 1,000 | 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|
| Reward-bridge error | 1.360 | 0.775 | 0.581 | **0.459** |
| Dynamics-bridge error | 0.732 | 0.422 | 0.312 | **0.223** |

(log–log OLS slopes `N^{-0.36}` and `N^{-0.39}`). The de-biased evaluator beats the
naive baseline in **12/12** toy test cases — confirmed on 5 seeds (up from 3) — with
worst-policy bias shrinking `0.104 → 0.053 → 0.036` as `N` grows from 5k to 50k, against
a naive bias that stays essentially flat near `0.13–0.33` regardless of `N` — the
textbook signature of removed *bias* versus irreducible *bias*.

### 3.3 The theory detonates: vanilla pessimism

The paper's own confidence-region pessimism (its Eq. 17), run as written, is not merely
weak — it is numerically unusable:

| Region size `c` | 0.1 | 0.3 | 1.0 | 3.0 | 10.0 |
|---|---|---|---|---|---|
| Best-policy `V_low` | `-1.15` | `-10.4` | `-62.2` | `-305.2` | **`-1779`** |

(5-seed re-run; the 3-seed numbers were `-1.12 → -1755` — same story, confirms this
isn't a low-seed-count artifact.) True policy values lie between roughly `1.5` and `2.2`;
a "pessimistic lower bound" of `-1.15` is already outside the physically feasible range,
and it only gets worse. We traced the mechanism precisely: because the observation space
is richer than the latent
state, the estimator's design matrix has a structural null space that costs almost
nothing in the confidence-region geometry, and the paper's *joint* multi-block
minimization exploits exactly those directions.

### 3.4 Our repair, and its honest limit

**Signal-Projected Pessimism** — an SVD/eigengap decomposition that restricts the
pessimistic search to the empirically identified signal subspace — repairs the numbers:
best-policy `V_low = 1.35` (sane, in-range) and **0.000 selection regret** across every
seed, versus vanilla's `0.68` regret and physically impossible values.

We then stress-tested our own fix, as part of an internal adversarial audit, and found
a real limitation: the true bridge functions leak up to **21% of their norm** outside
the projected signal subspace. This means the projected confidence region does not
actually contain the true bridge — its honest coverage is zero, not the "covers the
truth" claim we initially reported. We now describe this precisely as a
**coverage/informativeness trade-off**: vanilla pessimism covers the truth but is
uninformative; our projection is informative and empirically correct but does not
provably cover the truth. `V_low ≤ V_true` still held in every test we ran, but by
empirical sign-alignment, not by a certified guarantee. We report this as our honest
finding rather than an overclaimed fix (§5).

### 3.5 The benchmark boundary (the most important negative result)

At the full 720-observation scale, across the entire confounding-strength sweep (5
confounding levels × 3 sample sizes × 5 seeds, 15 cells), the **naive, confounding-blind
baseline beats every bridge-based method in every single cell**:

| | MAE range | Behavior with `N` |
|---|---|---|
| Naive | **0.013 – 0.074** | improves with `N` |
| Model-based (plain) | 0.072 – 0.230 | beats naive in **0/15** cells |
| Model-based (shrunk) | 0.063 – 0.240 | beats naive in **0/15** cells |
| Model-free proximal | 0.164 – 0.180 | **flat — does not improve with `N`** |

![Model-based vs. model-free MAE under hidden confounding, 5-seed 95% CIs. Left: MAE vs.
confounding strength kappa at N=20,000. Right: MAE vs. sample size N at kappa=1 (strong
confounding). Naive (black) stays lowest across both panels; error bars on MB-plain/MB-js
are wide and cross each other repeatedly, the visual signature behind the retracted
"shrinkage cuts variance 6x" claim.](results/figures/fig6_mb_vs_mf.png)

*Figure: naive vs. model-based (plain/shrunk) vs. model-free, 5-seed 95% CIs. Regenerated
from the corrected 5-seed sweep — see §3.5 text and Appendix A for the exact numbers.*

The de-biasing method's *worst* performance is at zero confounding (`kappa=0`), where
it is **2.8–6.8× worse** than naive — precisely where there is nothing to correct. This
is the clearest possible signature that the dominant error source at this scale is
**estimation variance**, not the confounding bias the method targets: inverting many
small, noisy per-cell matrices injects more noise than the bias it removes.

**Correction from our 5-seed re-run:** on the original 3-seed sweep we had reported
this `kappa=0` harm ratio as `23–46×` and claimed James–Stein shrinkage "cuts run-to-run
variance sixfold under strong confounding." Neither survives more seeds. The harm ratio
is `2.8–6.8×` — still a real, clear negative result, just a smaller one than we said. The
variance claim does not replicate at all: across the full 15-cell sweep, shrinkage has
*lower* variance than the plain estimator in only 7 of 15 cells (roughly a coin flip), and
at kappa=1 specifically — the exact condition we cited — its variance is *higher* than
plain's at `N=5,000` and `N=20,000`, and only about `2.7×` lower at `N=60,000`. We are
retracting the "6× variance cut" claim rather than carry it into the final paper; the
honest summary is that shrinkage gives a modest, noisy average MAE reduction (~7% across
the sweep), not a reliable variance-reduction guarantee. This is precisely why we
prioritized adding seeds and CIs before this checkpoint rather than after — a 3-seed standard
deviation is itself barely more than a point estimate, and we were citing it as if it
were a stable number.

Our model-free baseline is under-identified by the benchmark's binary proxy variable
(confirmed: it *is* consistent on the toy, where identification is clean, so this is a
genuine identification limit, not a bug).

### 3.6 Continuous state and action spaces (the paper's own generality, tested)

The anchor paper's framework is **not** inherently tabular. Re-reading it to check rather
than assume, it states that *"both S and O are continuous"*, that *"function approximation
is inevitable when state and observation spaces are continuous"*, and that its two-stage
procedure is built on kernel methods (Singh et al. 2019; Mastouri et al. 2021). Our
discrete implementation is therefore one *instantiation* of the theory, not its full scope.
To test that generality directly, we built a continuous proof of concept: real-valued
state, action, observation and reward, with a kernel/RKHS two-stage bridge estimator
replacing the tabular one, on a linear-Gaussian confounded POMDP that admits an **exact
closed-form oracle** — preserving the same "verify against ground truth to machine
precision" bar as §3.1.

**One correction to our own framing, from that same re-reading.** The paper's sentence
continues: *"...while the action space A is finite."* Our environment uses a real-valued
action, so it varies **two** things at once relative to the paper — continuous
observations (which the paper covers) and continuous actions (which it does not). The
identification chain still reproduced the exact oracle to `1.665e-16`, so nothing below is
numerically wrong, but this is **empirical extrapolation beyond the stated theorem, not an
instantiation of it**, and we do not claim the paper's guarantees for it. The
paper-faithful configuration — continuous `S`/`O`, *finite* `A` — is the top item of our
Phase 4 program (§6G). We flag this rather than relabel the work, on the same principle
that produced the 21% leakage disclosure in §3.4.

| Check | Result |
|---|---|
| Oracle self-consistency (true bridges vs. closed-form value) | `1.665 × 10⁻¹⁶` |
| Value-gradient check (analytic/hybrid vs. numerical) | `~1 × 10⁻¹¹` |
| Bridge recovery, reward / dynamics (`N`: 300 → 4,000, 5 seeds) | `0.220 → 0.055` / `0.240 → 0.102` |
| De-biasing vs. naive (`N=2,000`, 5 seeds, 95% CIs) | plug-in wins **4/4** policies, `3–11×` smaller bias |
| Pessimism validity `V_low ≤ V_true` (5 widths × 5 seeds) | **100%** |

The de-biasing result is the strongest here and the direct continuous analogue of §3.2's
12/12 story: naive bias is tight but systematically wrong (e.g. `-1.255 ± 0.018` on
`never_treat`), plug-in bias is much smaller but noisier (`-0.401 ± 0.216`) — the same
"removed bias versus irreducible bias" signature, now visible in the CIs themselves.

**Honest caveat, held to the same standard as §3.4: validity is not selection quality.**
`V_low ≤ V_true` holds in every test, but *ranking* policies by `V_low` is consistently
**worse** than ranking by the plain plug-in value across the entire width grid we tested
(regret `0.40–0.48` versus plug-in's `0.241`) — unlike the discrete toy's clean `0.000`
regret. We have a hypothesis (chain-compounding across the `T=3` stages, corroborated by
coordinate-descent restart gaps growing from `2×10⁻¹⁶` to `62.2` as the region widens) but
no confirmed root cause, so we report it as an open item rather than a solved one.

Two bottlenecks appear here that the tabular version never had: dense kernel ridge costs
`O(N₁³)` (capping this PoC at `N ≈ 4,000`, not tens of thousands, and requiring low-rank
approximations to go further), and a *finite-dimensional* kernel silently re-creates the
same rank-deficiency pathology §3.3 diagnosed — caught by checking the Gram matrix's rank
directly before trusting it, not by any downstream symptom, since the recovery numbers
looked fine either way and only the pessimism geometry would have misbehaved. Full
derivation, code map and results: `docs/continuous_extension.md`.

---

## 4. Difficulties & Challenges

1. **The paper's regularization theory does not transfer to a tabular implementation.**
   Its theoretical rate schedules (derived for infinite-dimensional RKHS estimators)
   over-shrink our tabular estimator by roughly 80% by the third time-step when applied
   literally. We resolved this with an empirical grid search (3 seeds), landing on
   `lambda_1 = 1/N_1` and `lambda_2 = 0.03/sqrt(N_2)` with a `0.7:0.3` sample split —
   values that work, but were measured, not derived from the paper.

2. **The standard ill-posedness diagnostic (matrix condition number) is misleading
   here.** Because of the same structural null space that causes the pessimism blow-up,
   the condition number stays artificially flat even as the true estimation quality
   degrades tenfold. We had to switch to monitoring the smallest *signal* eigenvalue of
   the restricted design instead — a subtler, non-obvious diagnostic choice.

3. **Diagnosing the pessimism blow-up.** Understanding *why* Eq. 17 produces values in
   the `-1700`-to-`-1800` range (exact number is seed-dependent; `-1779` on our current
   5-seed run) required tracing the interaction between the null-space geometry and the
   joint multi-block optimization — not visible from the paper, only from watching the
   number diverge and working backward.

4. **Catching our own overclaim.** Our first framing of Signal-Projected Pessimism was
   "the repair." Only a subsequent adversarial self-audit — deliberately trying to break
   our own result — surfaced the 21% leakage that invalidates the coverage claim. This
   was a genuine research-process difficulty: distinguishing "it produces good numbers"
   from "it is provably sound" required active, structured skepticism, not just more
   testing.

5. **The benchmark variance wall.** Scaling to 720 observations exposed that per-cell
   estimation variance can dominate the structural bias entirely — the opposite of the
   toy environment's story. This was the single most consequential empirical finding and
   took a full sweep across confounding strengths to characterize properly rather than
   dismiss as a tuning issue.

6. **Two real bugs in the model-free baseline, both caught only by oracle checks.**
   First, a per-stage direct-method implementation was silently biased off-policy (a
   policy with true value 1.865 evaluated to 1.61, with the error *not* shrinking with
   more data — the fingerprint of a derivation bug, not noise); fixed with a proper
   backward recursion. Second, the corrected version, naively scaled to the benchmark,
   tried to build and invert an ~11,500 × 11,500 dense matrix per policy per fit (about
   2 GB), which stalled the run entirely; fixed by exploiting a block-diagonal structure
   the naive implementation missed. Neither bug was visible from reading the derivation —
   both required running against exact ground truth to surface.

7. **Combinatorial blow-up in the benchmark's conditioning set.** A direct
   implementation of the paper's history-conditioning would require on the order of
   billions of table cells at benchmark scale. We resolved this with three structural
   factorizations specific to the environment's observation design (hashed encoding,
   core-diagonal bridge support, and closed-form 2×2 local inversions), keeping the
   entire pipeline laptop-scale.

8. **Balancing rigor against timeline.** Every one of the findings above came from
   deliberately adversarial, repeated self-review — which is expensive. We have had to
   make judgment calls about when a result is "verified enough" to report versus where
   further scrutiny would likely surface more issues; see §6 for how we're prioritizing
   the remaining time.

9. **Our own 3-seed statistics were not stable enough to report as claims.** Bumping the
   benchmark sweep from 3 to 5 seeds (§3.5) changed the `kappa=0` harm-ratio
   number from `23–46×` to `2.8–6.8×` and fully retracted a "shrinkage cuts variance
   sixfold" claim that turned out not to replicate (shrinkage has lower variance than the
   plain estimator in only 7 of 15 sweep cells). Neither was a coding bug — both were
   correctly computed `std()` values over an `n=3` sample, which is simply too few draws
   for a stable variance estimate. The lesson we're taking forward: any claim derived from
   a *standard deviation*, not just a mean, needs more seeds than a claim about the mean
   alone, and we should have flagged the original numbers as preliminary rather than
   headline results.

10. **We overstated our own continuous extension's scope, and caught it by re-reading
    the paper rather than trusting our summary of it.** We had described the extension
    as testing "the paper's own generality." Going back to the source showed the paper
    assumes continuous states and observations but an explicitly **finite action
    space**, while our environment uses a continuous action — so the PoC tests a
    strictly larger claim than the paper makes (§3.6). This is the same failure mode as
    item 4, in a different place: a plausible-sounding summary of someone else's result,
    never checked against the text. The fix was cheap here only because we checked
    before building anything further on top of it; the corrected, paper-faithful
    configuration is now the first item of the Phase 4 program (§6G).

---

## 5. Changes From the Original Proposal

The original 2-page proposal's core research question and method are unchanged. What
changed is how precisely we can state the contribution, based on what running the code
actually showed us:

- **Novelty claim narrowed.** The proposal's "first end-to-end empirical implementation"
  framing implicitly leaned on ranking ourselves against a concurrent submission we had
  not read. We now state a **capability claim** instead — "first to carry both bridge
  families through to a tractable pessimistic selection step" — which we can actually
  verify ourselves, rather than a primacy claim we cannot.
- **The pessimism "reduction" is now correctly attributed.** The per-block closed-form
  minimization is the standard ellipsoid support-function identity, not new mathematics.
  Our genuinely new content is the *diagnosis* of the blow-up mechanism and the
  coverage/informativeness finding, and we now say so explicitly rather than implying
  the closed form itself was the contribution.
- **Signal-Projected Pessimism reframed from "repair" to "heuristic with a measured
  price."** As described in §3.4/§4, this is the most significant change: we now report
  the 21% leakage finding as a first-class result rather than omitting it.
- **The benchmark result reframed from a soft "boundary condition" to an explicit,
  quantified negative result** (naive wins in the large majority of cells, worst at zero
  confounding), and generalized from two confounding levels to the full sweep.
- **Scope addition, not a backbone change:** the variance-reduction estimator and
  model-free baseline (§2.4) were not in the original proposal. We added them
  specifically to characterize the benchmark boundary rather than leave it as an
  unexamined limitation, and to strengthen the empirical comparison the course rewards.
  We believe this is an extension of the original method, not a change to it, but flag
  it for the mentor's awareness per the course's backbone-change policy (§8).

---

## 6. Remaining Tasks & Roadmap to Phase 4

We have not pinned an exact Phase 4 date to this roadmap since our current understanding
is that the course timeline has shifted; we would like to confirm the actual deadline
with the mentor (§8) and calendar-anchor these stages accordingly. The stages are ordered
by priority, not necessarily by equal time allocation.

**A. Immediate.** Finalize and deliver this checkpoint; no code changes required.

**B. Strengthen empirical rigor.** *Done for the benchmark sweep and the toy
pessimism/de-biasing checks*: both now run 5 seeds (up from 2–3) with per-seed std and
95%-CI reporting, which caught and corrected two overstated claims (§3.5, and §4 item 9).
The toy `N`-scaling sweep (log–log slopes) remains at 3 seeds — it already carried full std/CI
infrastructure but is the one script too slow (~12 min for 3 seeds, dominated by an
exhaustive-enumeration population check, not the seed loop) to safely re-run again before
this checkpoint; a candidate for post-checkpoint follow-up, not a blocker. Still open: a second,
*confounding-aware* baseline beyond naive (e.g., a simplified spectral/tabular OPE
method) — right now our only comparator is confounding-blind, which is a fair but
incomplete picture.

**C. Decide the pessimism narrative for the final paper.** Two options: (i) report the
heuristic-with-honest-caveat finding as the contribution — lowest risk, already complete,
and fits the course's explicitly accepted "critical analysis" / "empirical study" project
forms; or (ii) attempt a certified alternative (e.g., explicitly bounding the
excluded-subspace value swing and adding it back as a correction term). We recommend
(i) given the timeline, with (ii) as a stretch goal only if time permits after (B).

**D. Draft the ICML-format final paper.** Nearly all technical content already exists in
our proposal and internal technical write-ups; the remaining work is compression into
the required structure (abstract, introduction, related work, method, experiments,
limitations, conclusion) within the page budget, using the unlimited appendix for the
full derivations and audit history rather than the main body.

**E. Mentor-feedback incorporation pass.** Budget a revision cycle after mentor comments
on the near-final draft, per the course's stated Phase 4 process.

**F. Final polish and presentation prep.** Figure/table cleanup (we already have six
figures and five result tables to draw from), proofreading, and page-budget trimming.

**G. Continuous-space program (scoped for Phase 4).** Re-reading the paper to verify
its own scope produced a concrete work plan, and one correction to our framing that we
are recording rather than quietly fixing (§3.6, §4 item 10). In priority order:

1. **A paper-faithful continuous environment — continuous `S`/`O` but *finite* `A`.**
   This is the highest-value item, because it is the only configuration where the
   paper's Theorem 3.5 and its rate guarantees actually apply; our current PoC uses a
   real-valued action and so tests a strictly larger claim than the paper makes.
   Target: a partially-observed continuous control task (hidden velocity, noisy
   position observation) with a small discrete action set, a constructed pre-decision
   negative control, and ground truth by Monte-Carlo rollout.
2. **Diagnose the [C4] pessimistic-selection gap** (§3.6) — restart quality versus a
   genuine chain-compounding property needing different width calibration.
3. **Sweep the horizon `T`.** The paper is finite-horizon but imposes no small-`T`
   restriction; our `T=3` only mirrors the discrete pipeline. Because the
   chain-compounding hypothesis predicts pessimism degrades as `T` grows, this is a
   direct test of that hypothesis rather than a scaling exercise.
4. **More realistic data**, subject to a hard constraint worth stating plainly:
   measuring de-biasing needs *ground-truth* policy values, which logged real-world
   datasets cannot supply, and Assumption 3.1 needs a valid pre-decision negative
   control, which they rarely record. Standard offline-RL suites (e.g. D4RL) fail on
   both counts and are additionally *unconfounded by construction*, since the
   behaviour policy's inputs are fully logged — removing the very effect the method
   corrects. The realistic target is therefore a simulator or semi-synthetic
   construction in the medical setting the paper itself uses as motivation.

None of this gates items A–F; it is the substantive research program for the final
checkpoint.

---

## 7. Risks & Fallback Plan

- **Risk:** attempting a certified pessimism bound (§6C-ii) consumes time without
  converging. **Fallback:** it is explicitly scoped as optional; the honest-heuristic
  narrative (§6C-i) is already a complete, defensible deliverable on its own.
- **Risk:** a second baseline (§6B) turns out to need more derivation time than
  expected, as our first model-free baseline did. **Fallback:** report the current
  naive-only comparison, which is already informative and honestly scoped, and note the
  missing comparator as a stated limitation rather than a gap we pretend isn't there.
- **Risk:** further adversarial review of new code surfaces additional issues close to
  the deadline. **Fallback:** our validation suites are fast (minutes) and modular, so
  we can re-verify incrementally rather than needing a final big-bang check.

---

## 8. Specific Questions for the Mentor

1. Given the honest negative benchmark result, is "characterizing precisely where
   model-based de-biasing helps versus where estimation variance dominates" an
   acceptable core contribution for the final paper, or should we prioritize further
   closing the gap to naive before Phase 4?
2. Is the narrowed, capability-scoped novelty claim (§5) sufficiently defensible, or
   should we invest time trying to obtain and read the concurrent submission we
   currently cannot verify ourselves against?
3. Do the benchmark-scale extensions (variance reduction, model-free baseline) count as
   a backbone change requiring approval, or as an acceptable extension of the original
   proposal's method?
4. Given remaining time, should we prioritize (a) a second confounding-aware baseline,
   (b) attempting a certified pessimism bound, or (c) neither, and instead spend the
   time on writing/rigor (error bars, more seeds)? We currently lean toward (c) with
   (a) as a stretch goal.

---

## Appendix A: Key Numbers at a Glance

| Metric | Value |
|---|---|
| Plug-in vs. exact DP | `4.4e-16` |
| Estimator vs. oracle bridges (population limit) | `3.0e-15` |
| Dual vs. primal solve agreement | `2.4e-15` |
| Gradient check (analytic vs. finite-difference) | `4.9e-10` |
| Toy de-biasing win rate | `12/12` cells (confirmed on 5 seeds) |
| Toy naive bias | `+0.13` to `+0.33`, flat in `N` |
| Vanilla pessimism, best-policy `V_low` | `-1.15` (`c=0.1`) → `-1779` (`c=10`), 5 seeds |
| Signal-projected pessimism, best-policy `V_low` | `1.35` at `c=0.03` |
| Signal-projected selection regret | `0.000` (all seeds) |
| Signal-subspace leakage of the true bridge | up to `21%` of its norm |
| Benchmark: naive MAE (15-cell sweep, 5 seeds) | `0.013` – `0.074` |
| Benchmark: model-based (plain) MAE | `0.072` – `0.230`; beats naive in `0/15` cells |
| Benchmark: model-based (shrunk) MAE | `0.063` – `0.240`; beats naive in `0/15` cells |
| Benchmark: shrinkage variance-reduction claim | **retracted** — lower std in only `7/15` cells (§4, item 9) |
| Benchmark: worst-case harm at `kappa=0` | `2.8–6.8x` naive's error (was `23–46x` on 3 seeds — corrected, §3.5) |
| Model-free baseline, toy consistency | `0.0076` max error at `N=200k` |
| Model-free baseline, benchmark behavior | flat MAE `0.164–0.180` (under-identified by the proxy) |
| Continuous PoC: oracle self-consistency | `1.665e-16` |
| Continuous PoC: value-gradient check | `~1e-11` |
| Continuous PoC: de-biasing win rate | `4/4` policies, `3–11×` smaller bias (5 seeds) |
| Continuous PoC: pessimism validity `V_low ≤ V_true` | `100%` (5 widths × 5 seeds) |
| Continuous PoC: pessimistic vs. plug-in selection regret | `0.40–0.48` vs. `0.241` — **open item** (§3.6) |

## Appendix B: Repository Map

This progress report and every result and figure cited above were produced entirely
inside the top-level `Phase_3/` directory, which is fully self-contained:

- `phase3_progress_report.md` — this document.
- `src/envs/` — environments + exact oracles.
- `src/estimation/` — bridge estimators (toy-scale and benchmark-scale).
- `src/pessimism/` — confidence regions + Signal-Projected Pessimism.
- `src/baselines/` — the model-free comparator.
- `poc/` — all validation and benchmark driver scripts (`run_phase1_check.py` through
  `run_comprehensive_benchmark.py`, `plot_results.py`), now with 5-seed CI reporting and
  output directories created automatically on first run.
- `experiments/` — raw result JSON, one file per driver script.
- `results/figures/` — generated figures, including `fig6_mb_vs_mf.png` above.
- `external/` — the shared `gumbel-max-scm` benchmark data (not duplicated on disk; the
  original data lives alongside the frozen Phase 2 submission and is not modified by
  anything in this directory).

The `Phase_2/` directory (our original, separately submitted checkpoint) is kept fully
intact elsewhere as a frozen historical baseline and is not referenced by, or required to
read, this report.
