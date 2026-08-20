# The Paper-Faithful Experiment: Continuous States, Finite Actions

**Status: complete, 5 seeds, all checks pass.** This was the top open item coming
out of the Phase 3 checkpoint, and it resolved an open question by *refuting* our
own hypothesis.

---

## 1. Why this experiment exists

Phase 3 built a continuous extension with a **real-valued action**. Re-reading the
anchor paper showed its setting is stated as:

> "we assume that both S and O are continuous, while the action space A is finite."

So the Phase 3 study varied **two** things at once relative to the paper —
continuous observations (covered by the theory) and continuous actions (not
covered). It was extrapolation beyond Theorem 3.5, not an instantiation of it.

This experiment fixes exactly that one axis and holds everything else as close to
Phase 3 as possible — same transition/reward/observation parameters, same
regularization schedule, same pessimism engine — so any difference in results is
attributable to the action space rather than to retuning.

**It also tests a specific hypothesis.** Phase 3 found pessimistic selection
consistently *worse* than plain plug-in selection and hypothesised chain
compounding through the T=3 recursion, citing growing coordinate-descent restart
gaps as corroboration. If the continuous action space was the cause, the anomaly
should disappear here.

---

## 2. Design

| Component | Choice |
|---|---|
| State / observation | Continuous scalar, `o_t = s_t + noise` |
| **Action** | **Finite: `A = {0.0, 0.5, 1.0}`** |
| Reward / transition | Linear-Gaussian (unchanged from Phase 3) |
| Behavior policy | Softmax over dose levels driven by the **true hidden state** (confounded); softmax guarantees overlap |
| Negative control | `O_0 = s_1 + noise`, satisfies Assumption 3.1 by construction |
| Candidate policies | Observation-independent distributions over `A` |

**Bridge parameterisation.** The finite action coordinate gives each action level
its own free coefficient:

```
b_R(a, o) = alpha_a + beta_o * o        (K+1 parameters)
b_D(a, o) = gamma_a + phi_o * o         (K+1 parameters)
```

i.e. a **delta kernel on the action coordinate tensored with a linear kernel on
the observation** — strictly richer in the action coordinate than Phase 3's pure
linear kernel, since it never assumes the effect is linear in the dose.

**Why observation-independent policies.** That restriction is what preserves the
exact closed-form oracle: under such a policy the expected bridge feature is
exactly `[pi, m_t]`, so the chain stays affine in the mean and needs no covariance
propagation. A threshold policy would make the closed loop a Gaussian mixture and
forfeit the machine-precision verification the rest of this project depends on.
Stated as a limitation, not hidden.

**Two things improved over Phase 3.** The finite feature space makes the primal
solve mandatory from the start (the rank-deficiency lesson, applied rather than
rediscovered), and because the chain is affine in the mean, **dynamics-block
gradients are now exact** — a backward recursion `D_t = bR_t[-1] + bD_t[-1]·D_{t+1}`
replaces Phase 3's finite-difference approximation. Verified against numerical
differentiation to `1.16e-10`.

---

## 3. Results

### [F1] Oracle self-consistency: `0.000e+00`

Exactly zero, across all four candidate policies. True values:
`never_treat 1.5435`, `mixed 1.3330`, `low_dose 1.2427`, `high_dose 0.9420`.

### [F2] Bridge recovery improves with N (5 seeds)

| `N` | 300 | 800 | 2,000 | 4,000 |
|---|---|---|---|---|
| Reward bridge | 0.147 | 0.141 | 0.083 | **0.071** |
| Dynamics bridge | 0.271 | 0.251 | 0.156 | **0.086** |

Monotone in both families — cleaner than Phase 3's continuous-action run, which
was non-monotone at an intermediate point.

### [F3] De-biasing: 2/4 — worse than Phase 3, and the reason is informative

| Policy | Naive bias | Plug-in bias | Winner |
|---|---|---|---|
| never_treat | `-0.103 ± 0.032` | `+0.003 ± 0.092` | plug-in |
| low_dose | `-0.008 ± 0.020` | `-0.063 ± 0.054` | naive |
| high_dose | `+0.009 ± 0.026` | `+0.040 ± 0.049` | naive |
| mixed | `-0.052 ± 0.019` | `-0.009 ± 0.049` | plug-in |

This looks like a regression from Phase 3's 4/4, but it is the **same phenomenon
as the discrete benchmark's headline negative result, reproduced in miniature**:
where the naive estimator is *already nearly unbiased* (`low_dose`, `high_dose`:
bias `0.008`, `0.009`), there is nothing to correct, and the two-stage
correction only contributes variance. Plug-in wins exactly where naive bias is
large enough to dominate.

### [F4] Pessimism: validity holds, selection still fails

| `c` | valid | pessimistic regret | plug-in regret | restart gap |
|---|---|---|---|---|
| 0.03 | 1.00 | 0.211 | **0.000** | 0.0 |
| 0.1 | 1.00 | 0.211 | **0.000** | 1.1e-16 |
| 0.3 | 1.00 | 0.289 | **0.000** | 2.2e-16 |
| 1.0 | 1.00 | 0.602 | **0.000** | 4.4e-16 |
| 3.0 | 1.00 | 0.602 | **0.000** | 5.3 |

`V_low ≤ V_true` holds in 100% of tests. Plug-in selection is **perfect** (regret
0.000 at every width); pessimistic selection is not.

**Our Phase 3 hypothesis is refuted.** The anomaly persists with a finite action
space, so the continuous action space was not the cause. Worse for the original
hypothesis: restart gaps here are **essentially zero** up to `c=1`, meaning
coordinate descent is finding the global optimum reliably. The anomaly is
therefore not an optimization failure either — which was the specific mechanism
"chain compounding" proposed.

---

## 4. What is actually going on

A direct diagnostic settles it. At `N=2,000`, `c=0.3`:

| Policy | `V_true` | Pessimism penalty | Behavior coverage |
|---|---|---|---|
| **never_treat** (best) | **1.5435** | **2.058** (largest) | **0.121** (worst) |
| low_dose | 1.2427 | 1.350 | 0.260 |
| mixed | 1.3330 | 1.092 | 0.262 |
| high_dose (worst) | 0.9420 | 0.842 (smallest) | 0.619 (best) |

The pessimism penalty is **monotonically inverse to how well the behavior policy
covers each candidate**. And in this environment the *best* policy is the *least
covered* one: the confounded logging policy prefers high doses (62% of actions),
while the optimal policy never treats (12% coverage).

So pessimism is not malfunctioning — **it is working exactly as designed**. It
penalises policies the data does not support, and here the optimal policy is
precisely the unsupported one. The result is the textbook partial-coverage
trade-off of pessimistic offline RL, stated sharply: *pessimism buys a valid
lower bound at the cost of systematically avoiding the best policy whenever the
behavior policy under-covers it.*

This replaces the Phase 3 chain-compounding hypothesis with an explanation that
is both mechanistically evidenced and predictive: it says the gap should shrink
as behavior coverage of the optimal policy improves, and vanish under full
coverage.

---

## 5. Honest limitations of this experiment

- **Observation-independent candidate policies only.** Feedback policies would
  forfeit the exact oracle (§2). The confounding, the estimation problem and the
  pessimism step are all fully exercised regardless, but the policy class is
  restricted.
- **Scalar spaces, `T=3`, `N ≤ 4,000`.** The `O(N₁³)` dense-solve wall is
  unchanged from Phase 3.
- **The coverage explanation is demonstrated in one environment.** It is
  well-evidenced here (a monotone four-point ordering plus a mechanism) but a
  coverage sweep — deliberately varying how much the behavior policy covers the
  optimal action — would test it properly. That is the natural follow-up.
- **De-biasing win rate (2/4) is environment-specific**, driven by how much bias
  the naive estimator has per policy. It is not evidence that the method is worse
  here in general.

---

## 6. Files

| File | Role |
|---|---|
| `src/envs/finite_action_env.py` | Environment + exact oracle |
| `src/estimation/fa_bridge_estimator.py` | Two-stage estimator, `[onehot(a), o]` features, primal solve |
| `src/estimation/fa_value_plugin.py` | Mean chain with **exact** gradients |
| `src/pessimism/fa_pessimism.py` | Adapter over the unmodified `ellipsoid_opt.py` |
| `poc/run_finite_action_benchmark.py` | Driver: [F1]–[F4], 5 seeds |
| `experiments/results_finite_action.json` | Raw results |
