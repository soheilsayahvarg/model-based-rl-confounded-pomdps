# Confounded POMDPs: Dual-Bridge Function Identification & Signal-Projected Pessimism

An end-to-end, oracle-verified implementation of model-based off-policy evaluation and
pessimistic policy selection for **confounded partially observable MDPs**, together with a
diagnosis (and practical repair) of a failure mode in the theory's pessimism step.

> Graduate course project — Deep Reinforcement Learning. This repository is the Phase-2
> deliverable: proposal, implementation, experiments, and write-ups.

---

## Overview

Offline reinforcement learning is unreliable when the logged behavior policy acted on
information the learner never sees. When a **latent state drives both the recorded actions
and the outcomes**, the data is *confounded*, and a learner that treats the observation as
the state is **systematically biased** — a bias that does not vanish with more data.

This repository implements the model-based **bridge-function** framework of
**Hong, Qi & Xu (ICML 2024), *Model-based Reinforcement Learning for Confounded POMDPs*** —
which, to our knowledge, had no public implementation. Using a pre-decision negative-control
observation `O_0`, we estimate two families of bridge functions (a reward bridge `b_R` and a
dynamic bridge `b_D`) via the paper's two-stage procedure, chain them through the
Theorem-3.5 identification formula to compute policy values, and perform pessimistic policy
selection over risk-level-set confidence regions.

Along the way we found that the paper's **joint-minimization pessimism (Eq. 17) blows up in
practice** — because the observation space is richer than the latent space (`|O| > |S|`),
the confidence ellipsoid stretches unboundedly into the design matrix's **null space**, and
the minimization escapes into those data-blind directions (pessimistic values diverge to
absurd magnitudes such as `-1755`). We repair this with **Signal-Projected Pessimism**: an
SVD/eigen-decomposition of the design isolates the informative *signal subspace* from the
empty *null space*, and the pessimism is restricted to the signal subspace only. This keeps
the pessimistic value informative and yields **0.000 selection regret** on the identified
regime.

The codebase is validated against exact oracles throughout (dynamic-programming ground truth
and closed-form bridges), with agreements at the `1e-15` machine-precision level.

---

## Directory & File Structure

```text
Phase_2/
├── README.md                          # this file
├── src/
│   ├── envs/
│   │   ├── simulated_env.py            # Tabular confounded POMDP (720x2 benchmark, built from the
│   │   │                               #   gumbel-max SCM). Latent index masked from the log; pre-decision
│   │   │                               #   negative control O_0 constructed to satisfy Assumption 3.1;
│   │   │                               #   per-step noisy marker guarantees bridge existence. kappa knob
│   │   │                               #   controls confounding strength.
│   │   ├── toy_pomdp.py                # Minimal tabular POMDP (|S|=2, |A|=2, |O|=3, T=3) with exact
│   │   │                               #   oracle bridges — the substrate where recovery error is measurable.
│   │   └── oracle_module.py            # Exact dynamic-programming value evaluator V(pi), closed-form
│   │                                   #   pseudo-inverse bridge solver, and the naive (confounding-blind)
│   │                                   #   baseline evaluator.
│   ├── estimation/
│   │   ├── bridge_estimator.py         # Core two-stage estimator: Stage-1 ridge conditional-mean embedding
│   │   │                               #   (CME) + Stage-2 closed-form solve, implemented in BOTH the dual
│   │   │                               #   representer form and the primal operator form (verified equal).
│   │   ├── big_bridge_estimator.py     # Benchmark-scale estimator: hashed state-history factorization,
│   │   │                               #   core-diagonal bridge support, and per-cell 2x2 local matrix
│   │   │                               #   inversions (memory scales with data, not with |X_t|).
│   │   ├── shrinked_big_bridge_estimator.py  # Backward-compatible subclass adding James-Stein / pooling
│   │   │                               #   shrinkage and K-fold cross-fitting for variance reduction
│   │   │                               #   (default flags reproduce the parent byte-for-byte).
│   │   └── value_plugin.py             # Theorem-3.5 identification as an exact tensor-contraction chain
│   │                                   #   (np.einsum). Forward/backward recursions give V(pi) and the exact
│   │                                   #   per-block gradients used by the pessimism layer.
│   ├── pessimism/
│   │   ├── ellipsoid_opt.py            # Confidence-region ellipsoid geometry (H = T2 + lambda2*I), the
│   │   │                               #   closed-form support-function minimum, and the SVD signal-subspace
│   │   │                               #   projection (the null-space-blow-up repair).
│   │   └── pessimistic_optimizer.py    # Pessimistic policy selection over the confidence regions:
│   │                                   #   restricted-spectrum width calibration, blockwise coordinate
│   │                                   #   descent, and V_low <= V_true coverage instrumentation.
│   └── baselines/
│       └── model_free_proximal.py      # Model-free proximal OPE baseline (minimax value bridge, Shi et al.
│                                       #   2022) for a head-to-head model-based vs. model-free comparison.
├── poc/
│   ├── run_phase1_check.py             # Environments + exact oracles; naive-bias exit criterion.
│   ├── run_phase2_check.py             # Two-stage estimator: population-limit + primal/dual equivalence +
│   │                                   #   N-scaling sweep + O_0-noise ill-posedness sweep.
│   ├── run_phase34_check.py            # Value plug-in vs. DP, de-biasing, and the pessimism layer
│   │                                   #   (vanilla vs. signal-projected) — the master regression suite.
│   ├── run_bigenv_check.py             # Benchmark-scale bridge recovery + de-biasing vs. naive.
│   ├── run_comprehensive_benchmark.py  # Unified model-based vs. model-free sweep across N and kappa.
│   └── plot_results.py                 # Regenerates all figures and CSV tables from the results JSON.
├── results/
│   ├── table1_naive_bias.csv ... table5_bigenv_debias.csv   # Machine-readable result tables.
│   └── figures/                        # Colorblind-safe PNG plots (naive bias, bridge recovery, value
│                                       #   de-biasing, pessimism coverage/regret, benchmark comparison).
├── experiments/                        # Raw result JSON produced by the poc scripts (source of truth).
├── proposal/
│   ├── proposal.tex / proposal.pdf                 # The 2-page research proposal.
│   ├── detailed_project_guide.tex / .pdf           # Long-form, first-principles technical guide.
├── theory/                             # Supplementary analysis notes.
├── external/                           # The public gumbel-max-scm benchmark parameters (submodule/clone).
├── checkpoint_log.md                   # Chronological development log / milestone tracker.
└── workspace_checkpoint.json           # Structured project state and results snapshot.
```

---

## Quick Start

**Requirements:** Python 3.9+ with `numpy` (core) and `matplotlib` (figures). A LaTeX
distribution is needed only to rebuild the PDFs in `proposal/`. No GPU required — everything
runs on a laptop in NumPy.

```bash
# from the Phase_2/ directory
pip install numpy matplotlib

# 1. Environments + exact oracles + naive-bias exit criterion
python poc/run_phase1_check.py

# 2. Two-stage estimator: population-limit exactness, primal/dual equivalence,
#    N-scaling convergence, and the O_0-noise ill-posedness sweep
python poc/run_phase2_check.py

# 3. MASTER regression suite: plug-in vs. exact DP, de-biasing, and the
#    vanilla vs. signal-projected pessimism (asserts V_low <= V_true)
python poc/run_phase34_check.py

# 4. Benchmark-scale bridge recovery and de-biasing
python poc/run_bigenv_check.py

# 5. Unified model-based vs. model-free comparison across N and kappa
python poc/run_comprehensive_benchmark.py

# Regenerate all figures + CSV tables from the saved JSON
python poc/plot_results.py
```

Each script prints its own pass/fail checks. The headline assertions to look for:

- `run_phase34_check.py`: `[V1] ... max gap = 4.4e-16` (plug-in equals exact DP),
  `12/12` de-biasing cells improved, and `[V4] ... PASS` for both pessimism variants.
- `run_phase2_check.py`: monotone bridge-recovery error decay and the population-limit /
  primal-dual agreements at `~1e-15`.

---

## Core Mathematical Highlights

**Two-stage estimation.** Each bridge solves a conditional moment restriction (a first-kind
Fredholm equation) `E[b(W_t, y) | X_t] = p(y | X_t)`, where `W_t = (A_t, O_t)` are observed
and `X_t = (A_t, H_{t-1}, O_0)` carries the negative control. Stage 1 learns the conditional
mean embedding by ridge regression; Stage 2 solves the closed form
`alpha = (G + N_2 * lambda_2 * I)^{-1} * p_hat`, with `G = (Gamma^T K_W Gamma) ⊙ K_{Y''}`.

**Grid-calibrated regularization.** The paper's RKHS rate schedules over-shrink the tabular
class (an `N^{-1/2}` stage-1 schedule injects `√N_1` pseudo-counts, ≈80% shrinkage by `t=3`).
A small grid search selected instead:

- data split `N_1 : N_2 = 0.7 : 0.3` (Stage 1, over a growing history, needs more data),
- **`lambda_1 = 1 / N_1`** — gentle, parametric-rate stiffness,
- **`lambda_2 = 0.03 / sqrt(N_2)`** — decays *slower* than `1/N_2` so it dominates the
  stage-1 noise that leaks into the weakly-identified directions.

**Restricted-spectrum width rule.** The confidence-region width is
**`xi_t = c / (N_2 * sigma2_t)`**, where `sigma2_t` is the smallest *signal* eigenvalue of
the per-stage design (the weakest direction that still carries real information). This is the
key stability choice: the ordinary condition number `cond(G)` is *uninformative* here because
the structural `|O| > |S|` null space pins its smallest eigenvalue near zero regardless of
identification quality (verified — an `O_0`-noise sweep degrades stage-1 error 10x while
`cond(G)` stays flat). Calibrating pessimism to `sigma2` instead of `cond(G)`, and restricting
the pessimism to the SVD signal subspace, is what keeps the layer stable and informative.

---

## Key Results

- **Correctness.** The Theorem-3.5 value chain matches exact dynamic programming to
  `4.4e-16`; the estimator reaches the oracle pseudo-inverse bridges to `3.0e-15`; the primal
  and dual Stage-2 forms agree to `2.4e-15`.
- **Convergence.** On the toy, bridge-recovery error decays monotonically
  (`err_bR: 1.36 → 0.46`, `err_bD: 0.73 → 0.22` over `N = 1k → 20k`; log-log slopes
  `N^{-0.36}` / `N^{-0.39}`), and the model-based plug-in beats the naive baseline in
  **12/12** cases where identification is clean.
- **Pessimism.** Vanilla Eq.-17 pessimism is uninformative at coverage-guaranteeing widths
  (best-policy `V_low` diverges `-1.12 → -1755`); Signal-Projected Pessimism stays
  informative (`V_low = 1.35`) and selects the optimal policy with **0.000 regret**.
- **Honest boundary.** At full `720x2` benchmark scale, per-cell estimation *variance* can
  exceed the small structural confounding *bias* it removes, so at feasible sample sizes the
  bridge estimators do not uniformly beat the naive baseline. A James-Stein shrinkage variant
  cuts the model-based run-to-run variance ~6x under strong confounding but does not overturn
  this boundary. We report this variance boundary explicitly rather than around it.

---

## References

1. M. Hong, Z. Qi, Y. Xu. *Model-based Reinforcement Learning for Confounded POMDPs.* ICML, 2024.
2. C. Shi, M. Uehara, J. Huang, N. Jiang. *A Minimax Learning Approach to Off-Policy Evaluation in Confounded POMDPs.* ICML, 2022.
3. X. Guo, M. Cai, Z. Yang, Z. Wang. *Provably Efficient Offline Reinforcement Learning for Partially Observable Markov Decision Processes.* ICML, 2022.
4. A. Mastouri et al. *Proximal Causal Learning with Kernels.* ICML, 2021.
5. M. Oberst, D. Sontag. *Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models.* ICML, 2019.

---

*See `proposal/detailed_project_guide.pdf` for a first-principles walkthrough of the problem,
the method, the null-space breakdown, and the empirical findings.*
