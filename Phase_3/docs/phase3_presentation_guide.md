# Phase 3 Presentation & Self-Study Guide

**Purpose of this file:** this is *not* a submission document. It's for us — a guide
detailed enough that any of the three of us could explain any part of this project,
in plain language, without notes, and answer a follow-up question about it. If you
only read one section before the meeting, read Section 5 (the cheat-sheet).

Everything technical in here is grounded in the actual code in `Phase_3/src/` and
`Phase_3/poc/`, and every number matches `Phase_3/phase3_progress_report.md` and
`Phase_3/docs/proposal_updated.pdf`. If something here ever looks inconsistent with
those two files, trust them over this one and flag it — this file is the explainer,
not the source of truth.

---

## 1. Core Problem & Big Picture (Simplified)

### 1.1 The problem, explained to a peer

Picture a doctor deciding how aggressively to treat a patient. The doctor looks at
the chart (vitals, labs) — but also uses things that never get written down: how the
patient looks, gut instinct from experience. Now imagine you're a data scientist who
only has the chart, and you want to learn "does this treatment actually work?" from
historical records.

Here's the trap: the doctor tends to give the strong treatment to patients who *look*
sicker, even among patients whose charts are identical. So in your data, "received
the strong treatment" is tangled up with "was secretly sicker in a way your chart
doesn't capture." If you naively compare outcomes between treated and untreated
patients, the treatment will look worse than it really is — not because it doesn't
work, but because it was disproportionately given to the sicker patients. Collecting
more data does **not** fix this. It's not noise, it's a systematic bias in *who got
what*, and more data just makes you more confidently wrong.

This is **confounding**, and in RL terms: the doctor's "gut instinct" is a piece of
**hidden (latent) state** that drives both the logged actions and the outcomes,
while you, the learner, only see the **observation** (the chart). A learner that
treats the observation as if it were the full state — which is what almost all
offline RL does by default — is systematically biased, permanently.

Our benchmark environment is, quite literally, this scenario: it's built on a public
sepsis-treatment simulator (`gumbel-max-scm`), where a hidden patient-severity index
drives both the logged treatment decisions and the outcomes, exactly like the doctor
example above.

### 1.2 What we're actually testing

A 2024 ICML paper (Hong, Qi & Xu) proposes a fix: use a **negative control** — some
piece of information from *before* the decision was made, that's correlated with the
hidden state but doesn't itself change anything once you already know the current
situation. In our doctor example, that's something like "how did the patient look
when they first walked in, before any treatment was decided" — informative about
true severity, but not a cause of anything downstream by itself.

Using this negative control, the paper shows you can mathematically recover
unbiased treatment-effect estimates, purely from observable data. It's an elegant
theory. But it had **no code, no experiments, no implementation** — we don't know if
it actually works when you try to run it. That's what we built and tested.

### 1.3 The core ideas, in plain language (with the real math alongside)

**The negative control assumption.** We construct a signal `O_0` — a noisy readout
of the patient's *initial* hidden state, using information from before any action
was taken. The key property (Assumption 3.1 in the paper) is:

> Once you know the current state and action, `O_0` tells you nothing more about
> what happens next. It only matters through what it reveals about the hidden state.

**Bridge functions — the central trick.** A "bridge function" is a translation
table: it lets you compute a quantity that technically depends on the *hidden*
state, using only *observable* quantities instead. There are two of them:

- `b_R` (reward bridge) — recovers reward information.
- `b_D` (dynamics bridge) — recovers transition information.

They're defined as the solution to a system of equations (the paper calls these
*bridge moment restrictions*):

```
sum_o~  p(o~ | s) * b_R(a, o~, r, o)   =  p(r, o | s, a)         for every hidden state s
sum_o~  p(o~ | s) * b_D(a, o~, o~', o) =  p(o|s) * sum_s' p(s'|s,a) p(o~'|s')
```

In words: "if you average the bridge function over the hidden state's view of the
world, you get back the true reward/transition probability." If you can solve this
system, you never need to observe the hidden state directly again.

**Chaining bridges into a policy value (Theorem 3.5).** Once you have `b_R` and
`b_D` for every timestep, there's a clean recursive formula that strings them
together into the actual thing you care about — how good is policy `π`:

```
g_1(o~)     = p(o~_1)                                   # start: distribution over "hidden view"
g_{j+1}(o~') = sum_o~  b_D(a_j, o~, o~', o_j) * g_j(o~)  # propagate forward one step
f_t(r, h_t) = sum_o~  b_R(a_t, o~, r, o_t) * g_t(o~)     # read off reward probability
V(π)        = sum_t E[R_t]                               # sum over the horizon
```

This is exactly a forward-recursion Markov-chain computation, except every step
runs through a bridge function instead of a raw transition matrix. `value_plugin.py`
(Section 2) is a direct, exact implementation of this formula.

**Why the theory needs pessimism.** Because bridge functions are *estimated* from
finite data, any policy-value number you compute has estimation error. The paper's
answer is to be deliberately pessimistic: don't report your best-guess value, report
a defensible *lower bound* under the estimator's own uncertainty, and pick the
policy with the best lower bound rather than the best point estimate. This is the
part that, as we found out, is much harder to actually run than to write down on
paper (Section 3).

---

## 2. What We Have Built & Code Architecture

### 2.1 The shape of the pipeline

```
   src/envs/           src/estimation/          src/pessimism/       src/baselines/
 (world + truth)  -->  (learn bridges     ) --> (be appropriately --  (competing
                        from logged data)         cautious)            method)
        |                      |                       |                  |
        +----------------------+-----------+-----------+------------------+
                                            |
                                       poc/*.py
                          (drives everything above, checks
                           it against exact ground truth,
                           writes experiments/*.json + figures)
```

`src/` contains no scripts that run on their own — it's a library. `poc/` is where
everything actually executes: every script constructs an environment, generates
data, fits estimators, and checks the result against an *exact* oracle answer
computed independently (dynamic programming or closed-form matrix inversion). This
oracle-first structure is why we can say "verified to machine precision" rather than
"looked plausible."

### 2.2 `src/envs/` — the world, and the ground truth against it

| File | Plain-language role |
|---|---|
| `toy_pomdp.py` | A tiny, hand-built confounded POMDP (2 hidden states, 2 actions, 3 observations, horizon 3). Small enough that we can compute the *true* bridge functions by direct matrix inversion — this is the only place we can measure "how far off is our estimate from the actual right answer," because everywhere else the right answer isn't directly computable. |
| `simulated_env.py` | The realistic 720-observable-state × 2-latent-state sepsis-simulator benchmark. Builds the confounded logging policy, the negative control `O_0`, and generates trajectory data. This is "the real test" — the toy is for correctness, this is for realism. |
| `oracle_module.py` | All the *exact* ground-truth machinery: dynamic programming for true policy values, closed-form true bridge functions (by matrix pseudo-inverse), and the naive (confounding-blind) baseline evaluator that everything else is compared against. Nothing in this file is estimated from data — it's either exact math or a simple count-based baseline. |

**`toy_pomdp.py` in detail.**
- `default_params(T=3)` — builds the toy POMDP's parameters: emission matrix `E`
  (`p(o|s)`), transition tensor `P`, reward table `pR`, initial distribution `p1`,
  the confounded behavior policy `pi_b` (deliberately correlated with the hidden
  state), and the negative-control matrix `K0`.
- `sample_trajectories(params, N, rng, ...)` — generates `N` logged trajectories.
  Returns what an estimator is allowed to see (`O0, O, A, R`) *and*, separately, the
  private hidden state `S` — kept apart so nothing downstream can accidentally cheat
  by peeking at it.

**`simulated_env.py` in detail.**
- Constants at the top (`N_X=720` core states, `N_U=2` latent index, `N_A=8`
  actions, `S_AUG=1442` states once you add two absorbing terminal states) define
  the whole benchmark's shape.
- `load_params()` / `build_augmented()` — load the public simulator's raw tensors
  and reshape them into one clean Markov chain with absorbing "the episode ended
  positively/negatively" terminal states.
- `optimal_Q()` / `behavior_policy()` — compute the optimal Q-function on the full
  (state-aware) chain, then build a *confounded* logging policy that's a mixture of
  "sees the hidden state" and "doesn't" behavior, controlled by a knob `kappa` (0 =
  no confounding, 1 = fully confounded). This `kappa` knob is what we sweep across
  in the benchmark experiments.
- `sample_trajectories()` — generates logged data, including the observation
  channel `o = 2x + m` where `m` is a noisy binary marker whose emission rate
  depends on the hidden state (this noise is what makes bridge functions solvable
  at all — see the "spectral monitor fix" story in Section 3).
- `make_env()` / `generate_dataset()` — the one-call convenience functions every
  `poc/` script actually calls.

**`oracle_module.py` in detail** (organized into 7 numbered blocks in the file
itself):
1. `dp_value_obs_policy()` — exact dynamic programming for any candidate policy on
   the 720×2 benchmark.
2. `dp_value_toy()` / `dp_value_toy_latent()` — the same, for the toy.
3. `toy_true_bridges()` — solves for the *exact* bridge functions on the toy by
   direct pseudo-inverse. This is the "answer key" every toy-scale estimator check
   is graded against.
4. `toy_chain_value()` — an independent, brute-force enumeration implementation of
   the Theorem-3.5 chain (loops over every possible history explicitly, rather than
   the fast tensor-contraction version in `value_plugin.py`). Having two
   *independently coded* implementations of the same formula is what lets us say
   "these agree to machine precision" and mean it as a real check, not a tautology.
5. `big_bridge_R()` / `big_bridge_selftest()` — the true bridge functions for the
   720×2 benchmark, computed on-demand via a small 2×2 matrix inversion per query.
6. `fit_naive_mdp()` / `naive_value()` — **the baseline everything is measured
   against**: pretend the observation *is* the state, fit a standard MDP by
   counting, evaluate by forward DP. This is deliberately the "wrong" way to do it —
   it's supposed to be biased, and demonstrating that bias is Phase 1's whole job.
   `population_naive_mdp()` is the same idea but with infinite data (no sampling
   noise at all), which isolates the *structural* bias from ordinary statistical
   noise.
7. `mc_value_big()` / `mc_value_toy()` — plain Monte-Carlo rollout value estimates,
   used only as a sanity check that the DP oracle itself is right.

### 2.3 `src/estimation/` — learning the bridge functions from data

This is the heart of "Phase 2" — turning the theory into a runnable estimator.

| File | Plain-language role |
|---|---|
| `bridge_estimator.py` | The paper's two-stage estimator, implemented exactly, for the small toy POMDP. |
| `big_bridge_estimator.py` | The *same* two-stage idea, but re-derived into a form that scales to 720 states without ever building an astronomically large table. |
| `shrinked_big_bridge_estimator.py` | A variance-reduction add-on for the benchmark estimator (added in Phase 3, in response to the benchmark going badly — see Sections 3–4). |
| `value_plugin.py` | Turns a set of bridge functions into an actual policy-value number, via the Theorem-3.5 chain from Section 1.3 — implemented as fast tensor contractions (`np.einsum`), not the slow brute-force enumeration. |

**`bridge_estimator.py` — `TabularBridgeEstimator`.** Two stages, matching the
paper exactly:
- **Stage 1** (`_stage1_tables`): from a chunk of the data, build a lookup table
  estimate of "given this context `x`, what's the distribution over `(action,
  observation)`" — this is just counting, with a small ridge correction so it stays
  well-defined even for contexts barely seen in the data.
- **Stage 2** (`_stage2_solve`): from a *different* chunk of data, solve a
  regularized least-squares problem to get the actual bridge function values. This
  is implemented in **two mathematically equivalent ways** — a "dual" form (small
  matrix, good when there's little Stage-2 data) and a "primal" form (bigger matrix,
  good when there's a lot) — and the code checks at runtime that both forms agree,
  logging the gap (it typically agrees to ~$10^{-15}$, i.e., floating-point noise).
- `fit(O0, O, A, R)` — runs both stages for every timestep, producing `bR_hat` and
  `bD_hat` tables.
- Also computes and stores, per stage, the diagnostic information the pessimism
  layer will need later (`stage_store`): the quadratic-risk matrix `H`, the
  estimate itself, and — this is the "spectral monitor fix" from Section 3 — a
  `sigma2_signal` number that turned out to be the *only* reliable way to tell if
  an estimate is actually trustworthy (the ordinary "condition number" diagnostic
  is misleading here; explained in Section 3).

**`big_bridge_estimator.py` — `BigBridgeEstimator`.** Same two-stage *idea*, but
re-derived around a structural fact of our benchmark: the observable "core" part of
the observation is a deterministic function of the hidden state, and the only
genuinely noisy, hidden-state-dependent piece is one binary marker. That collapses
the whole estimation problem, per (timestep, core-state, action) cell, into
inverting one $2\times2$ matrix — cheap, and it means memory usage scales with how
much data you have, never with the size of the observation space (which would
otherwise be billions of cells; see Section 3's "combinatorial blow-up" story).
Key pieces:
- `fit()` — accumulates counts per cell, inverts the per-cell $2\times2$ system,
  and falls back to a simpler "pooled" estimate for any cell with too little data
  to invert safely (logged, never silent — `fallback_cell_frac` /
  `fallback_mass_frac`).
- `folded_operator()` / `_reward_head()` / `plugin_value()` — the benchmark-scale
  version of the Theorem-3.5 chain from Section 1.3, specialized to this
  estimator's storage format.

**`shrinked_big_bridge_estimator.py` — `ShrinkedBigBridgeEstimator`.** A subclass
of `BigBridgeEstimator` (same parent class, same interface) that adds two optional
variance-reduction techniques, both off by default so the class is byte-for-byte
identical to its parent unless you turn them on:
- **Shrinkage** (`shrinkage='pool'` or `'js'`) — instead of trusting each cell's own
  noisy $2\times2$ matrix, blend it toward a more stable, pooled estimate from
  similar cells, weighted by how much data that cell actually has.
- **Cross-fitting** (`k_folds >= 2`) — estimate the numerator and denominator of the
  matrix-inversion from *different* data splits, which removes a subtle statistical
  bias that comes from reusing the same counts for both (the "ratio bias" every
  statistics course warns about).

**`value_plugin.py` — `plugin_value()`.** The fast, exact tensor-contraction
implementation of the Theorem-3.5 chain (Section 1.3), used by every toy-scale
check. Also computes **exact gradients** of the policy value with respect to every
bridge coefficient — needed downstream because the pessimism step (Section 2.4) has
to *minimize* the policy value over a region, and having the exact gradient in
closed form is what makes that minimization fast and exact rather than approximate.

### 2.4 `src/pessimism/` — being appropriately cautious

| File | Plain-language role |
|---|---|
| `ellipsoid_opt.py` | The geometry of "how uncertain is this estimate" and the machinery to find the worst-case (most pessimistic) value inside that uncertainty region. |
| `pessimistic_optimizer.py` | Wraps the geometry into "pick the best policy under pessimism," across a whole panel of candidate policies. |

**`ellipsoid_opt.py` — `BlockEllipsoid` and `pessimistic_value()`.** The estimator's
own statistical theory says its uncertainty region is exactly an ellipsoid (a
stretched sphere) around the estimate — not an approximation, an exact consequence
of the estimator being a regularized least-squares solve. `BlockEllipsoid` holds
that geometry for one bridge block; `pessimistic_value()` finds the minimum policy
value across the *combination* of all the blocks' ellipsoids simultaneously, using
repeated closed-form steps (coordinate descent: fix everything but one block, solve
that block exactly, move to the next, repeat until it stops improving). This file
also holds the honest caveat about that procedure — see Section 3's "joint
minimization" story — and the **signal-subspace projection**, our fix for the
failure mode.

**`pessimistic_optimizer.py` — `pessimistic_selection()`.** The user-facing
function: given a fitted estimator and a handful of candidate policies, compute
each one's pessimistic value and pick the best. Also computes **coverage** (does the
uncertainty region actually contain the true bridge function, checked against the
oracle — only possible on the toy, where we know the true answer) and **validity**
(does the pessimistic value actually stay below the true value, as it's supposed
to).

### 2.5 `src/baselines/` — the competing method

**`model_free_proximal.py` — `MinimaxValueBridgeOPE`.** An independent
implementation of a *different* published method (Shi, Uehara, Huang & Jiang, ICML
2022) that solves a related but distinct problem: instead of modeling the whole
environment (bridge functions for reward and dynamics separately, our main
approach), it directly estimates a single "value bridge" that already bakes in a
specific target policy. It's "model-free" in the sense that it never builds a
transition model — it estimates policy value more directly. We built this
specifically to have a second, honest opinion to compare against, not a strawman.
This file **imports nothing** from the main pipeline, by design — it's a fully
independent implementation, so if it agrees with the model-based approach on the
toy (where we know the truth), that's real evidence, not circular reasoning.

### 2.6 `poc/` — the scripts that run everything and check it

Each script below writes its results to `Phase_3/experiments/<name>.json` and, for
`run_comprehensive_benchmark.py`, also a figure. **Every script is a self-contained
regression test**: it asserts specific numeric thresholds and crashes loudly if
they're violated — nothing is "eyeballed."

| Script | What it checks, in plain language |
|---|---|
| `run_phase1_check.py` | Sets up both environments and their oracles, and demonstrates the core problem: the naive (confounding-blind) evaluator is measurably wrong, on both the toy and the benchmark, and the bias does **not** shrink with more data — the thing this whole project exists to fix. |
| `run_phase2_check.py` | Validates the estimator itself, independent of anything downstream: does it recover the exact right answer with infinite data (`[P0]`)? Do its two internal solve methods agree (`[P1]`)? Does its error shrink as `N` grows (`[P2]`)? Does it degrade sensibly when the negative control is made less informative (`[P3]`)? |
| `run_phase34_check.py` | Validates the value chain and the pessimism layer: does the plug-in formula match exact dynamic programming with true bridges (`[V1]`)? Does it beat the naive baseline with estimated bridges (`[V2]`)? What happens when you sweep the pessimism width (`[V3]`) — this is where the vanilla blow-up and the Signal-Projected repair both show up. Does the core guarantee (pessimistic value ≤ true value) hold whenever the region provably covers the truth (`[V4]`)? |
| `run_bigenv_check.py` | The same de-biasing check as Phase 1/2, but at full 720×2 benchmark scale instead of the toy — this is where we first saw the benchmark go badly. |
| `run_comprehensive_benchmark.py` | The headline experiment: naive vs. model-based-plain vs. model-based-shrunk vs. model-free, across 5 confounding levels × 3 sample sizes × 5 seeds. Produces `fig6_mb_vs_mf.png`. |
| `plot_results.py` | Regenerates all the earlier (Phase 1–4, non-comprehensive) figures and CSV tables from the saved JSON results — a convenience script, not an independent check. |

---

## 3. Step-by-Step Story of What We Did (Phase 1 → Phase 3)

This is the narrative version of the difficulties list in
`phase3_progress_report.md` §4 — same facts, told as a story.

**Phase 1 — build the world, prove the problem is real.** We built both
environments and their exact oracles first, before writing a single line of
estimation code. Why: if we can't compute a trustworthy "right answer" independently
of our own estimator, we can never tell a real bug from a believable-looking bug.
The Phase 1 exit criterion was simple: show that the naive, confounding-blind
evaluator is measurably and persistently wrong. It was — that's the whole reason
this project exists.

**Phase 2 — build the estimator, hit two real problems immediately.**

- *Problem 1: the paper's regularization schedule doesn't work here.* The paper's
  theory is written for infinite-dimensional function spaces (RKHS) and prescribes
  regularization strength that shrinks at a specific rate as data grows. Applied
  literally to our finite tabular setting, it over-shrinks the estimate by roughly
  80% by the third timestep — badly under-using the data we have. *Fix:* we ran a
  small grid search instead and found values that actually work
  (`lambda_1 = 1/N_1`, `lambda_2 = 0.03/sqrt(N_2)`), and we say plainly that these
  are measured, not derived from the paper's theory.

- *Problem 2 ("the spectral monitor fix"): the standard way to check "is this
  estimate trustworthy" lies to you here.* The usual diagnostic for this kind of
  regularized least-squares problem is the matrix **condition number** — big means
  "untrustworthy," small means "fine." We ran an experiment where we deliberately
  degraded the negative control's informativeness (making `O_0` a worse and worse
  signal about the hidden state) and watched the condition number... not move. At
  all. Meanwhile the actual estimation error got 10× worse. The reason: our
  observation space is structurally *larger* than the hidden-state space, so there
  are directions in the estimation problem that are **never identifiable no matter
  what**, and the ordinary condition number gets permanently pinned near zero by
  those directions — it can't distinguish "healthy but has an unidentifiable
  direction" from "genuinely falling apart." *Fix:* we built a different diagnostic,
  `sigma2_signal`, that only looks at the eigenvalues of the part of the problem
  that's actually identifiable (found via an eigengap: keep the eigenvalues that are
  meaningfully bigger than the next one down, throw away the rest as structural
  noise). This one degrades exactly the way you'd want when we ran the same
  stress-test. This same "structural null space" turned out to be the root cause of
  the much bigger problem below — same disease, two different symptoms.

**Phase 3/4 — the value chain works perfectly; the pessimism step detonates.**

- The value-chain formula itself (Section 1.3) checked out immediately: plugging in
  the *true* bridge functions reproduces exact dynamic programming to
  $4.4\times10^{-16}$ — as close to "identical" as floating-point arithmetic gets.
  Good news, but the easy part.

- *Problem 3 ("the joint-minimization repair"): the paper's pessimism formula, run
  literally, is not just weak — it's unusable.* The pessimism step asks: "across
  every value of the bridge functions that's still statistically plausible given
  our uncertainty, what's the worst-case (lowest) policy value?" On paper this is a
  well-posed optimization. When we actually ran it, at every region size that's
  supposed to *guarantee* it contains the true bridge, the answer came out as
  `-1779` — a number with no physical meaning (true values are around `1.5`–`2.2`;
  a policy literally cannot score `-1779`). *Diagnosis:* remember those structurally
  unidentifiable directions from Problem 2? They cost almost nothing in the
  uncertainty-region geometry (the model is honestly saying "I have no information
  here, so I won't penalize you for guessing wildly in this direction"). But the
  *optimizer* doesn't know those directions are meaningless — it happily walks
  arbitrarily far into them because doing so makes the "worst case" value arbitrarily
  bad, for free. *Fix ("Signal-Projected Pessimism"):* we restrict the pessimistic
  search to *only* the directions we actually have information about — the same
  signal subspace `sigma2_signal` was already identifying — using an SVD-based
  projection. This repairs the numbers completely: the pessimistic value becomes
  sane (`1.35`, in the right range) and the method picks the truly best policy every
  single time (`0.000` selection regret, vs. the vanilla version's `0.68`).

- *Problem 4 (catching our own overclaim): the fix isn't actually a certified fix.*
  We initially described Signal-Projected Pessimism as "the repair" — full stop. It
  took a deliberate, adversarial self-review (actively trying to break our own
  result, not just re-checking it) to find the catch: the *true* bridge function
  isn't perfectly contained in that signal subspace either — it leaks up to 21% of
  its norm outside it. That means the "safe" region we built doesn't actually,
  honestly contain the truth. We now describe this precisely as a
  **coverage/informativeness trade-off**, not a repair: the original version covers
  the truth but is useless (values in the thousands of magnitude off); our version
  is useful but doesn't provably cover the truth. Both are disclosed, with numbers.

**Scaling up — the benchmark, and a second kind of failure.**

- *Problem 5 (combinatorial blow-up): a literal implementation of the estimator at
  benchmark scale would need billions of table cells* (the conditioning set grows
  combinatorially with the trajectory history). We didn't hit this as a crash so
  much as recognize it in advance from the toy estimator's design and route around
  it: `big_bridge_estimator.py` exploits three structural facts about our specific
  observation design (the "core" of the observation is deterministic given the
  hidden state, so the estimation problem factors into independent, tiny per-cell
  systems) to keep everything laptop-scale.

- *Problem 6 (the benchmark's honest negative result): at realistic scale, the
  naive baseline wins.* Across every confounding level and every sample size we
  tested, our model-based method's estimation *variance* — noise from inverting many
  small, data-thin matrices — outweighs the confounding *bias* it's designed to
  remove. Worst of all at zero confounding, where there's nothing to correct in the
  first place. We didn't hide this; we characterized it with a full sweep instead
  of dismissing it as a tuning problem (details in Section 4).

- *Problem 7 (two real bugs in the model-free baseline, both invisible without an
  oracle).* First bug: an early version of the model-free estimator summed each
  timestep's reward against the *behavior* policy's state distribution instead of
  correctly propagating the *target* policy's continuation value backward through
  time. The symptom was subtle — a specific policy's true value of `1.865`
  evaluated to `1.61`, and critically, that error did **not** shrink as we added
  more data, which is the fingerprint of a wrong formula, not statistical noise.
  Fixed by implementing a proper backward recursion (see `model_free_proximal.py`'s
  stage-by-stage loop in Section 2.5). Second bug ("the 2GB matrix crash"): the
  corrected version, naively scaled up to the benchmark's 1442 observations × 8
  actions, tried to build and invert an ~11,500×11,500 dense matrix *per policy, per
  fit* — about 2GB, and the run simply stalled. We noticed the instrument
  (`O_0`, only 2 values) and the bridge argument (`(action, observation)`) share the
  action index, meaning the giant system is actually block-diagonal across the 8
  actions. Solving each small block separately (with a Woodbury-identity shortcut
  for the inner inverse) brought the fit time down to about 0.02 seconds. Neither
  bug was visible by reading the math — both only showed up by running the code
  against a known-correct oracle answer.

**The Phase 3 checkpoint rigor pass — how it caught our own numbers.** We
had been running our benchmark and pessimism sweeps on only 2–3 random seeds. A
3-seed standard deviation is a genuinely unstable number — barely more informative
than a single extra data point — and two of our claims turned out to be built on
exactly that instability. We reran everything relevant at 5 seeds with proper 95%
confidence intervals, and: the "zero-confounding harm ratio" changed from
`23–46×` to a still-bad-but-smaller `2.8–6.8×`, and a claimed "shrinkage cuts
variance sixfold" finding evaporated entirely — on the fuller check, shrinkage has
*lower* variance than the plain estimator in only 7 of 15 sweep cells, barely better
than a coin flip. We retracted that claim outright rather than soften it. Full
details, with exact numbers, are in Section 4 below and
`phase3_progress_report.md` §3.5 / §4 item 9.

---

## 4. Experiments & Key Findings Explained Simply

### 4.1 What "5 seeds and a 95% CI" actually means (in one paragraph)

Every experiment that involves randomly generated data (which is all of them) gets
run several times with a different random number generator seed each time, so we get
several independent estimates of the same quantity instead of one. The **mean** is
our best guess; the **standard deviation** tells us how much that guess would wobble
if we reran the experiment again; the **95% confidence interval** (mean ±
`1.96 × std / sqrt(n_seeds)`) is a rough range we're fairly confident the "true"
average result falls within. The critical catch: with only `n=3` seeds, that
standard deviation is itself estimated from just 3 numbers — it can be wildly wrong
by chance. That's exactly what happened to us (Section 3, "the final rigor pass"),
which is why we bumped to 5 seeds for the benchmark and toy pessimism/de-biasing
checks before finalizing this checkpoint.

### 4.2 Walking through each experiment

**Oracle verification (machine precision).** We check that our fast tensor-chain
formula and an independently-coded brute-force enumeration give the same answer,
and that both match exact dynamic programming. All agree to somewhere between
$10^{-10}$ and $10^{-16}$ — floating-point noise, nothing more. **What this proves:**
every later result is a fact about the *method itself*, not a bug we're
misinterpreting as a finding.

**Bridge recovery (toy, 3 seeds).** As we feed the estimator more data (`N` from
1,000 to 20,000 trajectories), its distance from the *true* bridge function
shrinks steadily — from `1.36` down to `0.46` for the reward bridge, `0.73` down to
`0.22` for the dynamics bridge. **What this means:** the estimator is doing exactly
what an estimator should do — converging toward the truth as data grows. (This
specific check is still at 3 seeds; it already stores per-seed spread, but its
population-limit sanity check is too slow — about 12 minutes — to safely rerun at
5 seeds before the checkpoint. Queued for later, not a blocker.)

**De-biasing (toy, 5 seeds).** Compare the naive evaluator against our bridge-based
plug-in, across 4 candidate policies × 3 sample sizes = 12 total comparisons. The
plug-in wins all 12, and the size of its remaining error shrinks as `N` grows
(`0.104 → 0.053 → 0.036`), while the naive evaluator's error stays essentially flat
(`0.13`–`0.33`) no matter how much data it gets. **What this means:** this is
exactly the "systematic bias vs. shrinking noise" signature from Section 1 —
concrete proof the method removes the *right kind* of error.

**Pessimism sweep (toy, 5 seeds) — the honest headline finding.** We sweep a
"region size" knob `c` that controls how cautious the pessimism step is. The
vanilla (paper-literal) version's pessimistic value gets *worse* as `c` grows —
`-1.15` at the smallest width tested up to `-1779` at the largest — which is
backwards; a wider, more honest uncertainty region should give a more conservative
but still *sane* number, not an increasingly meaningless one. Our Signal-Projected
repair, evaluated at a modest region size, gives `1.35` (a real, in-range value) and
picks the actually-best policy with zero regret. **What this means, and the honest
caveat:** the repair produces trustworthy-looking numbers, but the true bridge
function leaks ~21% of its norm outside the region we search over, so we cannot
*prove* the region contains the truth — we can only report that, empirically, in
every test we ran, the pessimistic value still stayed below the true value. That's
evidence, not a certificate.

**Benchmark boundary (5 seeds) — the most important negative result.** This is
`fig6_mb_vs_mf.png` (embedded in `proposal_updated.pdf` §5.5). Two panels:

- *Left panel:* mean-absolute-error of each method's value estimate, plotted against
  confounding strength `κ` (x-axis), at a fixed sample size. Each point has a
  vertical error bar showing the 95% CI across seeds.
- *Right panel:* the same MAE, but plotted against sample size `N` (x-axis, log
  scale), at fixed strong confounding.

**What the picture shows:** the naive baseline (black line) sits visibly *below*
every other method in both panels — lower error is better, so naive is winning.
The model-based lines (red = plain, orange = shrunk) have wide, overlapping error
bars that cross each other repeatedly from point to point — that visual noise *is*
the finding: their behavior is not stable enough to reliably rank against each
other, and definitely not stable enough to beat naive. The model-free line (blue)
is nearly flat across both panels — it isn't improving no matter how much data or
how much confounding you throw at it (see "why the model-free baseline goes flat,"
below).

**The numbers behind the picture** (5-seed, 15-cell sweep: 5 confounding levels ×
3 sample sizes):

| Method | MAE range | Beats naive? |
|---|---|---|
| Naive | 0.013 – 0.074 | — |
| Model-based (plain) | 0.072 – 0.230 | 0 / 15 cells |
| Model-based (shrunk) | 0.063 – 0.240 | 0 / 15 cells |
| Model-free proximal | 0.164 – 0.180 (flat) | 0 / 15 cells |

Worst case is zero confounding (`κ=0`), where the model-based method is `2.8–6.8×`
*worse* than naive — precisely the setting where there's nothing to correct in the
first place, so every bit of that error is pure estimation noise, not a fixed bias.

**Why does this happen?** The benchmark estimator inverts a small $2\times2$ matrix
*per (timestep, core-state, action) cell*. With 720 core states and limited data,
most individual cells only see a handful of visits — and inverting a matrix built
from a handful of noisy counts is itself noisy. That per-cell noise adds up to more
total error than the confounding bias it was trying to remove. This is the opposite
failure mode from the toy: on the toy, bias dominates and the fix helps; at
benchmark scale, variance dominates and the fix backfires.

**Why did we retract the James-Stein shrinkage claim?** Shrinkage was supposed to
help exactly this problem, by blending each noisy cell's estimate toward a more
stable, pooled estimate from similar cells. On our original 3-seed run, it looked
like it cut run-to-run variance roughly sixfold under strong confounding. On the
5-seed re-check, that doesn't hold up: across the full 15-cell sweep, shrinkage has
*lower* variance than the plain estimator in only 7 of 15 cells — essentially a coin
flip — and at the exact condition we'd originally highlighted (`κ=1`), it actually
has *higher* variance at two of the three sample sizes tested, and only about
`2.7×` lower (not `6×`) at the largest. Neither the original nor the corrected
number was a coding bug — both were correctly computed statistics over too few
seeds to be a stable estimate of a *standard deviation* specifically (means
stabilize faster than standard deviations do, as a general statistical fact). We
retracted the claim rather than quietly soften it, and we say so explicitly in the
report. Shrinkage's real, defensible benefit is a modest ~7% average MAE reduction
— real, just much smaller and noisier than we first said.

**Why does the model-free baseline go flat instead of improving?** It's
under-identified at benchmark scale: its only "instrument" for recovering hidden-
state information is the binary negative control `O_0` (just 2 possible values),
but it's trying to identify a bridge function defined over 1,442 possible
observations. Two values of information cannot pin down that many unknowns, no
matter how much data you feed it — this is a structural identification limit of
*that specific baseline*, not a bug, confirmed by the fact that this same baseline
*is* consistent (converges correctly) on the toy, where the instrument-to-unknown
ratio is reasonable.

---

## 5. Cheat-Sheet for Mentor Presentation

### 5.1 If the mentor asks X, answer Y

- **"What's your actual contribution, in one sentence?"**
  → "We built the first working, oracle-verified, end-to-end implementation of a
  purely theoretical ICML 2024 paper, found that its pessimism step is numerically
  broken as written, built and honestly characterized a repair for it, and mapped
  out exactly where the whole approach does and doesn't beat a naive baseline."

- **"Is the pessimism fix a real fix, or are you overselling it?"**
  → "It's a disclosed heuristic, not a certified bound. It repairs the numbers
  completely (sane values, zero selection regret) but the true bridge function
  leaks ~21% of its norm outside the region we search, so we can't *prove* the
  region contains the truth — only report that it held empirically in every test we
  ran. We say this explicitly rather than claim more than we can back up."

- **"Why does your method lose to a naive baseline at benchmark scale?"**
  → "Estimation variance, not model error. We invert many small, data-thin
  matrices — one per (time, core-state, action) cell — and with limited data per
  cell, that noise outweighs the confounding bias we're trying to remove. It's
  worst exactly where there's nothing to correct (zero confounding), which is the
  clearest possible signature that it's a variance problem, not a bias problem."

- **"Why did some of your numbers change since the last update?"**
  → "We deliberately re-ran our benchmark and pessimism experiments with more
  random seeds (5 instead of 2–3) specifically because two of our claims rested on
  an unstably small sample. Both changed: a harm ratio came down from `23–46×` to
  `2.8–6.8×`, and a 'shrinkage cuts variance sixfold' claim didn't replicate at all
  and is now retracted. We see this as the verification process working as
  intended, not as an embarrassing mistake — we'd rather catch it ourselves."

- **"Is adding the shrinkage estimator and the model-free baseline a change to your
  proposal's backbone?"**
  → "We consider it a scope addition, not a backbone change — same research
  question, same core method, added specifically to characterize the benchmark
  result rather than leave it unexamined. Flagging it for your judgment per the
  course's backbone-change policy."

- **"What's your novelty claim, exactly?"**
  → "A capability claim, not a primacy claim: we can verify that ours is the first
  implementation to carry both bridge families through to a tractable pessimistic
  *selection* step. We deliberately don't claim to be first overall, since we
  haven't reviewed every concurrent paper well enough to make that claim honestly."

- **"What's left before the final submission?"**
  → See the roadmap below (5.3) — short version: decide the pessimism narrative,
  compress existing material into the ICML paper format, incorporate your feedback,
  polish. Almost no new technical work is required; nearly everything left is
  writing.

- **"The paper is stated for continuous spaces — did you only do the tabular case?"**
  → "No. We built a continuous proof of concept too: real-valued state, action,
  observation and reward, with a kernel/RKHS bridge estimator replacing the tabular one,
  on a linear-Gaussian environment that still gives us an exact closed-form oracle. It
  hits the same machine-precision bar (`1.665e-16`), plug-in beats naive in 4/4 policies,
  and pessimism validity holds 100% of the time."
  → **Then volunteer the caveat before they find it**: "Two honest points. First, when we
  went back and re-read the paper to check its exact scope, it assumes continuous states
  and observations but an explicitly *finite action space* — and our environment uses a
  continuous action. So our PoC actually tests a *larger* claim than the paper makes; the
  identification still held exactly against the oracle, but we don't claim the paper's
  guarantees for it, and the paper-faithful version (continuous S/O, finite A) is the
  first thing on our Phase 4 list. Second, pessimistic *selection* there is currently
  worse than plain plug-in selection — validity isn't the same as ranking quality — and
  that's an open item, not something we've solved."

  ⚠️ **Do not say** "we tested the paper's continuous setting" without that first caveat —
  it isn't accurate, and a mentor who knows the paper will catch it. Saying it yourself
  first is the stronger move anyway.

- **"Why kernel/RKHS and not a neural network for the continuous case?"**
  → "Two reasons, both about being able to check ourselves. A neural bridge has no
  closed-form confidence region, so the pessimism step would need a completely different
  theory (ensemble-disagreement penalties), and there's no exact oracle to verify a
  trained network against. The kernel path kept both: our existing pessimism engine works
  unmodified because it operates in coefficient space, and the linear-Gaussian environment
  gives exact ground truth."

### 5.2 Known limitations (say these before you're asked)

- The pessimism guarantee is empirical, not certified (the 21% leak).
- In the continuous extension, pessimistic *selection* underperforms plain plug-in
  selection across every confidence width we tested (regret `0.40–0.48` vs. `0.241`),
  unlike the discrete toy's clean `0.000`. Validity (`V_low ≤ V_true`) still holds 100%
  of the time. We have a hypothesis (chain-compounding across the 3 stages) but no
  confirmed root cause — say "open item," not "solved."
- The continuous PoC is toy-scale only (`N ≈ 4,000` ceiling, from an `O(N₁³)` dense kernel
  solve) and uses scalar, not vector-valued, state/action/observation.
- The benchmark-scale result is a **negative** result for the model-based method —
  naive wins in all 15 sweep cells tested.
- Only one confounding-aware comparator exists (the under-identified model-free
  baseline); the naive comparator is confounding-*blind*, which is a fair but
  incomplete picture.
- The toy bridge-recovery convergence check is still at 3 seeds (not yet re-run at
  5), because its slowest component (~12 minutes) is a population-limit sanity
  check, not the actual seed loop.
- `plot_results.py`'s `fig5_bigenv_failure.png` still carries an old caption
  ("15/16 cells") from before the corrected 5-seed sweep — the current,
  authoritative benchmark figure and numbers are `fig6_mb_vs_mf.png` and the 15/15
  figures in Section 4 above; worth knowing so a sharp-eyed mentor question about
  that specific PNG doesn't catch us flat-footed.

### 5.3 Roadmap to Phase 4 (condensed — full version in `phase3_progress_report.md` §6)

1. Finalize and deliver this checkpoint (done).
2. Optionally strengthen rigor further: re-run the toy bridge-recovery sweep at 5
   seeds; consider a second, confounding-*aware* baseline.
3. Decide the pessimism narrative: report the honest-heuristic finding as the
   contribution (recommended, lowest risk, already complete), or attempt a
   certified bound as a stretch goal only.
4. Compress existing material (this guide, the progress report, the updated
   proposal) into the required ICML paper structure — mostly writing, not new
   derivation.
5. Incorporate mentor feedback.
6. Final polish and submission.
