# Continuous State/Action Extension

**Status: proof-of-concept, toy-scale only** (explicit scope decision — see §1).
This document is the companion to `phase3_presentation_guide.md`, written in the
same voice, but scoped entirely to the new continuous-space modules added under
`Phase_3/src/` and `Phase_3/poc/run_continuous_benchmark.py`. It does not
replace or modify anything about the discrete pipeline, which remains exactly
as it was for this checkpoint.

---

## 0. What the paper actually licenses (verified by re-reading it)

Before treating this extension as "the paper's own generality," we went back to the
source and checked, rather than relying on recollection. Three statements, quoted:

- **Continuous states and observations: explicitly assumed.** *"Without loss of
  generality, we assume that both S and O are continuous..."* (§2). The
  identification result is stated for *"confounded POMDPs with continuous state and
  observation spaces"* (§3.1).
- **General function approximation: explicitly required, not merely permitted.**
  *"our approach accommodates general function approximation, which is necessary
  given continuous state spaces and observation spaces"* (§1); *"function
  approximation is inevitable when state and observation spaces are continuous"*
  (§3). The two-stage nonparametric procedure is explicitly *"inspired by Singh et
  al. (2019); Mastouri et al. (2021)"* — i.e. **kernel** methods, which is exactly
  the backend chosen in §1 below. Our kernel/RKHS choice is therefore the paper's
  own instantiation, not a substitute for it.
- **Actions: FINITE.** The same sentence continues: *"...while the action space A is
  finite."*

> **Honest correction to our own framing.** That last point matters and cuts against
> the title of this document. Our environment (`continuous_env.py`) uses a
> **real-valued action** (`a = K·o + c`), so it varies *two* things at once relative
> to the paper: continuous observations (which the paper covers) **and** continuous
> actions (which it explicitly does not). The identification chain still reproduced
> the exact oracle value to `1.665e-16` in our linear-Gaussian instance, so nothing
> here is numerically wrong — but that is **empirical extrapolation beyond the stated
> theorem, not an instantiation of it**, and the paper's guarantees (Theorem 3.5 and
> the downstream rates) are not claimed for it. A paper-faithful continuous test
> would keep `S`/`O` continuous and make `A` **finite**; that is now the first item
> in §8. We flag this rather than quietly relabel the work, consistent with how the
> 21% signal-leakage finding was handled in the discrete pipeline.

Also confirmed while re-reading: the paper contains **no experiments section** of any
kind, which is the premise this whole project rests on.

---

## 1. Scope decisions made before writing any code

Two forks were resolved before implementation started, both toward the
lower-risk, more-continuous-with-existing-work option:

- **Backend: kernel / RKHS, not a neural network.** A neural bridge function
  has no closed-form confidence region, so "pessimism" would need an entirely
  different theory (ensemble-disagreement penalties, in the style of
  MOPO/PBRL) and no exact oracle check is possible for a trained network. The
  kernel/RKHS path keeps both intact: `ellipsoid_opt.py`'s pessimism engine
  operates purely in coefficient space and doesn't care whether those
  coefficients came from a tabular or kernel Stage-2 solve, and a
  linear-Gaussian toy environment gives an *exact* closed-form oracle in
  continuous space, preserving the project's whole "verify against ground
  truth to machine precision" discipline.
- **PoC scope: toy-scale only, no benchmark-scale continuous environment.**
  Mirrors how the discrete project itself started (toy first; the 720×2
  benchmark came later, as a separate, deliberate step). A benchmark-scale
  continuous environment is a natural, larger follow-up, not attempted here.

---

## 2. Theoretical & Mathematical Adaptation

### 2.1 Mean-value bridges (a deliberate simplification)

The anchor paper's bridge functions are *distributional*: `b(W_t, y)` must
match `p(y | X_t)` for every value of `y`. Reproducing that exactly in
continuous `y` needs conditional density estimation, which is a much larger
undertaking on its own. Since this project's actual downstream need is a
policy-value **mean** (`V(π) = Σ_t E[R_t]`), and the pessimism machinery is
about *estimation* uncertainty in the bridge coefficients — not about the
environment's own intrinsic reward noise — a **conditional-mean moment
restriction** is sufficient and exact for that purpose:

```
E[ b(A_t, O_t) | X_t = x ] = E[ Y_t | X_t = x ]     for all x,
```

with `Y_t = R_t` (reward bridge) or `O_{t+1}` (dynamics bridge, t<T). This is
exactly a (nonparametric) **instrumental-variable regression**: `W=(A,O)` is
confounded, `X=(A,H_{t-1},O_0)` is the instrument (informative via the
negative control, independent of the residual by Assumption 3.1), `Y` is the
outcome. Well-precedented in the proximal-causal-inference literature this
project already cites (Mastouri et al. 2021, already in the bibliography).

### 2.2 The two-stage kernel solve, derived (not recalled) from first principles

We deliberately **derived this closed form directly from a stated objective**
rather than pattern-matching a remembered paper formula — a mis-remembered
index convention would silently produce a wrong-but-plausible estimator,
exactly the failure mode this whole project exists to catch via oracle
verification. Full derivation is in `continuous_bridge_estimator.py`'s module
docstring; summary:

```
Stage 1:  Gamma = (K_X(X1,X1) + N1*lam1*I)^{-1} K_X(X1,X2)     (N1 x N2)
Stage 2:  minimize  (1/N2)||Z^T beta - y2||^2 + lam2*beta^T K_W(W1,W1) beta
          Z = K_W(W1,W1) @ Gamma
   =>     H = Z@Z.T/N2 + lam2*K_W(W1,W1),   g = Z@y2/N2,   beta = solve(H, g)
```

**Unifying check:** when `K_W` is a one-hot/delta kernel (the tabular case),
`K_W(W1,W1) = I` exactly and this reduces to the tabular estimator's own
primal solve — a correctness check on the derivation itself, not just an
analogy.

### 2.3 A real numerical pitfall caught before it corrupted anything

`K_W` was chosen as a **pure linear kernel** `k(w,w')=w·w'` for the primary
verified path (see §2.4). This has a **finite (2-D) feature space**, so the
dual Gram matrix `K_W(W1,W1)` (N1×N1) has **rank ≤ 2** — confirmed
empirically: eigenvalues at the ~1e-13 floor for N1 in the hundreds. The dual
system `H·beta=g` is therefore numerically unstable in those near-null
directions, even though the specific *projected* quantity we originally
extracted from it (`theta_hat = W1ᵀ·beta`, the primal linear coefficients)
turned out to be invariant to that instability and reproduced identical
recovery numbers before and after the fix. The instability still had to be
fixed, because it would have corrupted the **pessimism ellipsoid geometry**
directly (`H` and `b_hat` are used as-is there, not through an invariant
projection) — `np.linalg.solve`/Cholesky on a near-singular `H` would have
been numerically meaningless. **Fix:** solve directly in the 2-D **primal**
coordinates for the linear-kernel case (see the corrected derivation in
`continuous_bridge_estimator.py`) — mathematically identical solution, exactly
well-conditioned, and incidentally ~1000× cheaper (2×2 solve vs. N1×N1).

### 2.4 Kernel choices

- `K_X` (instrument/history side): RBF, median-heuristic bandwidth. Just a
  smoothing operator — no realizability requirement.
- `K_W` (bridge-argument side): **linear** (primary, verified path) — the
  true bridges (§2.5) are exactly linear with zero intercept, so a linear
  kernel's RKHS contains them exactly, giving a clean "recovery error → 0 as
  N grows" check. An RBF option is implemented (`w_kernel='rbf'`) for future
  nonlinear extensions but is **not** the primary verified path this pass.

### 2.5 Spectral monitor: effective dimension, not a hard eigengap

The tabular project's fix for a misleading `cond(G)` was a hard eigengap on
the smallest **signal** eigenvalue — correct there because the null space was
*structural* (`|O|>|S|`, exactly zero signal beyond it). A continuous kernel
Gram matrix has **smooth spectral decay**, not a structural wall, so a hard
eigengap rule doesn't transfer. The honest continuous analogue, implemented
here, is the standard kernel-ridge **effective dimension** (Caponnetto & De
Vito 2007): `N_eff(λ) = trace(K(K+Nλ I)⁻¹)` — "how many directions are
actually well-identified," continuously rather than discretely.

### 2.6 Pessimism: reused, not reinvented — and why signal-projection doesn't apply here

`ellipsoid_opt.BlockEllipsoid`/`pessimistic_value` operate purely on `(H,
b_hat)` in coefficient space; nothing about them assumes a tabular origin. For
the linear-kernel primal path, that coefficient space is a genuine 2-D
`(H, theta)` with **no structural null space** (§2.3 confirmed `H` here is a
proper, well-conditioned 2×2 matrix), so Signal-Projected Pessimism — built
specifically to repair a structural null space — has nothing to repair here
and is not built for this pass. `continuous_pessimism.py` is a thin adapter
that builds `BlockEllipsoid` objects from the continuous estimator's stage
output and otherwise imports the discrete pessimism engine unmodified.

### 2.7 Gradients: exact for reward blocks, a documented finite-difference for dynamics blocks

The chain is multilinear in each stage's coefficients (same structural
property as the tabular chain, for the same reason: holding every other block
fixed pins every evaluation point as a constant). Reward-block gradients are
exactly linear in their own coefficients — exact, no approximation. Dynamics
blocks are harder: a stage's coefficients affect the *evaluation point* of
every later stage, and that dependency runs through the kernel function
nonlinearly. We keep the coefficient-linear part exact and get the one
remaining scalar sensitivity (`d V_rest / d m_{j+1}`) via a single 1-D central
finite difference — **O(T) extra work per gradient call, not O(N1)** — verified
against plain numerical differentiation of the whole chain to ~1e-11 (see
`continuous_value_plugin.py`'s module docstring). This is a real,
explicitly-documented departure from the tabular pipeline's fully-exact
gradients, not an oversight.

---

## 3. Environment (`src/envs/continuous_env.py`)

Scalar linear-Gaussian confounded POMDP: latent state, action, observation,
and reward are all real-valued. The observation `o_t = s_t + noise` is a
*noisy proxy* for the hidden state; the logging policy can act on the true
`s_t` (confounded) or only on `o_t` (not confounded), mixed by a `kappa`
knob — a continuous relaxation of `simulated_env.py`'s aware/blind mixture.
**Why this is genuine confounding with a single noisy latent variable:** this
is the classical *measurement-error* setting proximal causal inference was
designed for — fitting a model that treats the noisy proxy as if it were the
true state induces errors-in-variables bias whenever `kappa>0`.

Because reward is **linear** (not quadratic) and candidate policies are
restricted to **affine, observation-only** functions (`a=K·o+c` — matching
every candidate policy already in this project, which are likewise pure
functions of the observation), the mean trajectory of the closed-loop system
is an *exact* scalar recursion — no covariance propagation needed. This gives
an exact closed-form oracle (`true_value`) and exact closed-form bridge
coefficients (`true_bridge_coeffs`), verified to self-consistently reproduce
each other to **machine precision** (see §5, [C1]) — the same quality bar as
`toy_pomdp.py`'s oracle.

---

## 4. Code Architecture

| File | Role |
|---|---|
| `src/envs/continuous_env.py` | Linear-Gaussian confounded POMDP; exact closed-form oracle values and bridge coefficients. |
| `src/estimation/continuous_bridge_estimator.py` | Two-stage kernel bridge estimator (`ContinuousBridgeEstimator`); primal solve for the linear-kernel path, dual solve for the RBF path; effective-dimension diagnostic. |
| `src/estimation/continuous_value_plugin.py` | Scalar mean-chain (continuous Theorem-3.5); exact reward gradients, hybrid exact/finite-difference dynamics gradients. |
| `src/pessimism/continuous_pessimism.py` | Thin adapter: builds `BlockEllipsoid`s from the continuous estimator and reuses `ellipsoid_opt.py`'s pessimism engine unmodified. |
| `poc/run_continuous_benchmark.py` | Driver: oracle self-consistency, N-scaling recovery, naive-vs-plug-in de-biasing, pessimism sweep — 5 seeds and 95% CIs from the start. |

None of these import from or modify any existing discrete file; the discrete
`poc/run_phase1_check.py` was re-run after adding these and still passes
unchanged.

---

## 5. Preliminary Results

*(Captured directly from `experiments/results_continuous.json`, produced by
`poc/run_continuous_benchmark.py`. Every number here is machine-generated, not
estimated in advance.)*

### [C1] Oracle self-consistency

`1.665e-16` max gap between `true_value` and `chain_value_from_bridges` using
the true bridge coefficients, across all 4 candidate policies — machine
precision, the same quality bar as the discrete toy's own oracle checks.

### [C2] N-scaling bridge recovery (5-seed mean)

| `N` | 300 | 800 | 2,000 | 4,000 |
|---|---|---|---|---|
| Reward-bridge error | 0.220 | 0.090 | 0.103 | **0.055** |
| Dynamics-bridge error | 0.240 | 0.214 | 0.175 | **0.102** |

Overall downward trend confirmed from smallest to largest `N` (the passing
assertion); not perfectly monotonic at every intermediate point (reward error
ticks up slightly from N=800 to N=2,000) — expected 5-seed noise, reported
as measured rather than smoothed over. `N` is capped at 4,000 specifically
because of the O(N1³) dense-solve cost documented in §6 — a real, different
scaling ceiling than the tabular estimator's.

### [C3] De-biasing at N=2,000 (5-seed mean, 95% CI) — a clean result

| Policy | Naive bias | Plug-in bias |
|---|---|---|
| never_treat | `-1.255 ± 0.018` | `-0.401 ± 0.216` |
| constant_dose | `+0.093 ± 0.013` | `+0.008 ± 0.032` |
| proportional | `-0.435 ± 0.013` | `-0.129 ± 0.061` |
| aggressive | `+1.357 ± 0.038` | `+0.365 ± 0.178` |

Plug-in beats naive in **4/4** policies, by a wide margin in every case
(roughly 3–11× smaller absolute bias) — the clearest result in this whole
extension, and the direct continuous analogue of the discrete toy's 12/12
de-biasing story. The naive evaluator's bias is tight (small CI) but
*systematically wrong*; the plug-in's bias is smaller but noisier (wider CI)
— the same "removed bias vs. irreducible bias" signature, now visible in the
CIs themselves.

### [C4] Pessimism sweep at N=2,000 (5 seeds) — validity holds; selection quality is an honest mixed result

| `c` | all valid | pessimistic regret | plug-in regret | restart gap |
|---|---|---|---|---|
| 0.03 | 1.00 | 0.479 | 0.241 | `2.2e-16` |
| 0.1 | 1.00 | 0.438 | 0.241 | `1.36` |
| 0.3 | 1.00 | 0.397 | 0.241 | `5.26` |
| 1.0 | 1.00 | 0.397 | 0.241 | `20.4` |
| 3.0 | 1.00 | 0.397 | 0.241 | `62.2` |

**The core guarantee holds throughout:** `V_low ≤ V_true` for every policy, at
every width tested, all 5 seeds — `all_valid=1.00` across the board.

**The honest part:** unlike the discrete toy, where Signal-Projected
Pessimism achieved *both* validity and zero selection regret, here
**pessimistic selection is consistently worse than plain plug-in selection**
(regret 0.40–0.48 vs. plug-in's flat 0.241) across the entire grid. This is
not a contradiction of the core guarantee — `V_low ≤ V_true` is a per-policy
statement, not a promise that *ranking by* `V_low` reproduces the true
ranking, and here it apparently doesn't as cleanly as it did in the discrete
case. The growing restart gap (`2.2e-16 → 62.2` as `c` grows) is corroborating
evidence: coordinate descent's sensitivity to its starting point grows
sharply with width, consistent with the same chain-compounding mechanism
described in §6 item 3 — at wider confidence regions, the optimization
landscape genuinely gets harder to solve exactly, not just more conservative.
This is reported as-is rather than tuned away; it's a real, useful finding
about where this PoC's pessimism step needs more work before it matches the
discrete pipeline's cleaner story.

---

## 6. New Continuous-Domain Challenges Encountered

1. **Dense kernel ridge is O(N1³) time / O(N1²) memory** — a fundamentally
   different scaling bottleneck than the tabular estimator's O(N) counting.
   Measured directly: a single fit at N1≈4,200 took ~2 minutes; the same
   estimator design would not reach the discrete benchmark's N=60,000 without
   a low-rank approximation (Nyström / random features / inducing points —
   standard fixes, not implemented in this pass). This is the direct
   continuous analogue of the discrete project's "combinatorial blow-up"
   story, but from a different root cause (dense linear algebra, not
   combinatorial index growth) — and it's why the N-scaling sweep below tops
   out in the low thousands, not tens of thousands.
2. **A finite-dimensional kernel silently reintroduces a structural null
   space** (§2.3) — an unexpected echo of the exact pathology the tabular
   project spent real effort diagnosing, this time self-inflicted by the
   choice of a linear kernel rather than an intrinsic property of the
   environment. Caught by deliberately checking the Gram matrix's rank before
   trusting it, not by a downstream symptom — the recovery numbers would have
   looked fine regardless (§2.3's invariance finding), and only the pessimism
   step would have silently misbehaved.
3. **Joint multi-block pessimism still degrades at wide confidence widths,
   via a different mechanism than the tabular blow-up.** There is no
   structural null space here (§2.3), yet `V_low` still grows substantially
   more negative as the width multiplier `c` grows, even once the region
   already covers the true coefficients. The mechanism is chain
   *compounding*: a dynamics block's worst-case perturbation shifts every
   later stage's evaluation point, and those shifts accumulate multiplicatively
   across T=3 stages. A milder, mechanistically distinct, but real relative of
   the tabular project's headline finding — see §5 for the actual numbers.
4. **A tabular-calibrated regularization schedule doesn't have an obvious
   right answer here either.** We hypothesized the tabular `lambda1=1/N1`
   schedule might be too weak for a genuine kernel CME (the effective
   dimension came out very close to N1 itself, suggesting little smoothing).
   Tested `1/sqrt(N1)` and `5/sqrt(N1)` as stronger alternatives with 5-seed
   averaging — both performed *worse*, not better. Kept `1/N1`. Notable
   because it's the mirror image of the original tabular finding (there, the
   paper's RKHS schedule didn't transfer to tabular; here, the
   tabular-calibrated schedule turned out fine for a genuine kernel) — a
   reminder to test rather than assume in either direction.

---

## 7. Known Simplifications / Limitations (explicit, not hidden)

- **Continuous actions go beyond the paper's stated assumptions** (§0) — the paper
  assumes a *finite* action space; our environment uses a real-valued action. The
  oracle check still passes exactly, but the paper's theorem is not being tested
  as stated. The paper-faithful variant (continuous `S`/`O`, finite `A`) is the
  first item in §8.
- **Pessimistic selection quality, not just validity, is an open item** — §5
  [C4] found `V_low ≤ V_true` holds throughout (the core guarantee), but
  pessimistic selection is consistently *worse* than plain plug-in selection
  across the whole tested width grid, unlike the discrete toy's zero-regret
  result. Not yet root-caused beyond the chain-compounding hypothesis in §6
  item 3 — the top-priority next step for this extension (§8).
- **Mean-value bridges, not the full distributional bridge** (§2.1) — a
  deliberate scope reduction, not a fundamental barrier.
- **No dual-form cross-check.** The tabular estimator verifies its dual and
  primal solves agree; this pass only derives and uses one closed form per
  kernel choice. A natural follow-up, not done here.
- **No exact truth-coverage diagnostic in the pessimism sweep.** The tabular
  pipeline's `coverage_report` checks whether the confidence region contains
  the *true* bridge table. Expressing the true (or, in the RBF case,
  infinite-dimensional) bridge in this estimator's own data-dependent
  coefficient coordinates is not needed for the core claim — the discrete
  pipeline's own ultimate guarantee is also empirical (`ellipsoid_opt.py`
  says so explicitly), not a coverage certificate — so we check
  `V_low <= V_true` directly instead and skip the intermediate diagnostic.
- **Linear kernel is the only verified path.** RBF is implemented and usable
  but not the primary reported/verified path this pass (§2.4).
- **Deterministic, affine, observation-only candidate policies only** — no
  stochastic or nonlinear candidate policies evaluated.
- **Scalar state/action/observation**, not vector-valued.
- **No benchmark-scale continuous environment** (§1) — toy-scale only, by
  deliberate scope decision.

---

## 8. Natural Next Steps (deferred to Phase 4, not started)

These are **scoped for the final checkpoint**, not this one. Ordered by priority:

1. **A paper-faithful continuous environment: continuous `S`/`O`, *finite* `A`**
   (§0). This is the single highest-value follow-up, because it is the only
   configuration in which the paper's own Theorem 3.5 and its rate guarantees
   actually apply. Our current PoC deliberately or not tests a strictly larger
   claim than the paper makes. Concretely: a partially-observed continuous
   control task (hidden velocity, noisy position observation) with a small
   discrete action set, a constructed pre-decision negative control satisfying
   Assumption 3.1, and ground truth available by Monte-Carlo rollout.
2. **Diagnose why pessimistic selection underperforms plug-in selection**
   (§5 [C4], §7) — check first whether restart quality (more restarts, or a
   smarter start than "center"/"random") closes the gap, or whether it is a
   genuine property of chain compounding needing a different width calibration
   or joint-minimization strategy.
3. **Longer horizons.** The paper is finite-horizon in `T` but imposes no small-`T`
   restriction; our PoC uses `T=3` purely to mirror the discrete pipeline. Since
   the chain-compounding hypothesis in §6 item 3 predicts *worse* pessimism
   behaviour as `T` grows, sweeping `T` is a direct test of that hypothesis, not
   just a scaling exercise.
4. **Realistic data.** Note the hard constraint that shapes this: measuring
   de-biasing requires *ground-truth* policy values, which logged real-world
   datasets cannot provide, and Assumption 3.1 requires a valid pre-decision
   negative control, which they rarely record. So the realistic target is a
   simulator or semi-synthetic construction (the medical scenario the paper
   itself uses as motivation), not an off-the-shelf offline-RL dataset such as
   D4RL — those have neither ground truth nor a valid negative control, and
   their behaviour policies are fully observed, i.e. *unconfounded by
   construction*, which removes the very effect the method targets.
5. **Neural function approximation** as an alternative backend, accepting that it
   forfeits both the closed-form confidence region and the exact oracle check
   (§1) — worth it only once the kernel path's open items above are settled.
6. Low-rank kernel approximation (Nyström or random features) to get past the
   `O(N₁³)` wall; vector-valued state/observation; exact (non-finite-difference)
   dynamics-block gradients if profiling ever shows the FD step matters (it does
   not currently).
