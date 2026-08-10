# Model-Based RL for Confounded POMDPs

**An end-to-end empirical implementation of a theory-only ICML 2024 paper — including a
diagnosis of a failure mode in its pessimism step, a repair, and an honest account of
where the whole approach stops working.**

Offline reinforcement learning breaks down when the policy that generated the data acted
on information the learner never sees. If a **latent state drives both the logged actions
and the outcomes**, the data is *confounded*: a learner that treats the observation as the
state is systematically biased, and no amount of extra data removes that bias.

Hong, Qi & Xu (ICML 2024), *Model-based Reinforcement Learning for Confounded POMDPs*,
give an elegant theory for exactly this setting — identifying the reward and transition
models through two families of **bridge functions** anchored by a pre-decision negative
control, then planning pessimistically over a confidence region. The paper is **theory
only**: no algorithm, no code, no experiments.

This repository builds that pipeline, verifies it against exact ground truth, and reports
what happens when the theory is actually run.

---

## Headline results

Every number below is machine-generated from the tracked experiment scripts, verified on
5 random seeds with 95% confidence intervals.

| | Result |
|---|---|
| Policy-value chain vs. exact dynamic programming | `4.4 × 10⁻¹⁶` |
| Estimator vs. exact oracle bridges (population limit) | `3.0 × 10⁻¹⁵` |
| Dual vs. primal closed-form solve (independent derivations) | `2.4 × 10⁻¹⁵` |
| De-biasing vs. confounding-blind baseline (toy scale) | wins **12/12** cells |
| De-biasing vs. baseline (continuous extension) | wins **4/4** policies |
| Oracle self-consistency (continuous / finite-action) | `1.7 × 10⁻¹⁶` / `0.0` |

### Four findings that only appear when you run the theory

**1. The paper's pessimism step is numerically unusable as written.** Run literally, its
confidence-region optimization returns policy values that diverge to `-1779`, against true
values in the range `1.5`–`2.2`. We traced the mechanism precisely: because the observation
space is structurally richer than the latent state, the design matrix carries a null space
that costs the confidence-region geometry almost nothing, and the paper's *joint*
multi-block minimization walks arbitrarily far into exactly those directions.

**2. Our repair works — and we quantify what it gives up.** *Signal-Projected Pessimism*
restricts the search to the empirically identified signal subspace, restoring sane values
(`V_low = 1.35`) and **zero selection regret** across every seed. But an adversarial
self-audit showed the true bridge functions leak up to **21% of their norm** outside that
subspace, so the projected region does not provably contain the truth. We report this as a
**coverage/informativeness trade-off**, not as a solved problem.

**3. At realistic scale, the method loses to a naive baseline — and we know why.** Across a
full sweep (5 confounding strengths × 3 sample sizes × 5 seeds, 720-observation benchmark),
the confounding-blind baseline beats every bridge-based method in **all 15 cells**. The
model-based estimator is *worst* at zero confounding, precisely where there is nothing to
correct — the clearest possible signature that **estimation variance**, not confounding
bias, dominates at this scale.

**4. Pessimistic *selection* fails for a reason that turned out not to be our first
hypothesis.** Pessimism reliably produces valid lower bounds (`V_low ≤ V_true` in 100% of
tests) yet ranks policies worse than a plain plug-in estimate. We hypothesised that
worst-case perturbations compound through the horizon, and designed an experiment to test
it — a paper-faithful configuration with continuous states and a *finite* action space, the
setting the anchor paper actually assumes. The hypothesis was **refuted**: the anomaly
persists, and coordinate-descent restart gaps there are `≤ 4e-16`, so the optimizer is
finding the global minimum. A direct diagnostic found the real mechanism — the pessimism
penalty is monotonically inverse to **behavior-policy coverage**, and in that environment
the optimal policy is the least-covered one (12% vs 62%). Pessimism is not malfunctioning;
it is penalising policies the data does not support, and the best policy is the unsupported
one.

These negative results are reported in full rather than buried. So are corrections we made
to our own work: two earlier headline numbers rested on a 3-seed standard deviation, and
when re-checked at 5 seeds one changed materially while the other did not replicate at all
and was retracted; and the compounding hypothesis above was retracted in favour of the
coverage explanation that replaced it.

---

## Repository layout

```
Literature Review/    Survey of the surrounding literature (proximal causal inference,
                      model-free OPE in POMDPs, pessimism in offline RL)

Phase_2/              First complete implementation
  src/                Environments + exact oracles, bridge estimators, pessimism
  poc/                Validation and benchmark driver scripts
  experiments/        Raw result JSON, one file per driver
  results/            Generated figures
  proposal/           Research proposal (LaTeX + PDF)

Phase_3/              Progress checkpoint — extended pipeline and rigor pass
  src/envs/           Toy POMDP, 720-observation benchmark, continuous environment
  src/estimation/     Tabular, benchmark-scale, and kernel/RKHS bridge estimators
  src/pessimism/      Confidence regions, Signal-Projected Pessimism
  src/baselines/      Independent model-free proximal comparator
  poc/                Driver scripts (5-seed, 95% CI reporting)
  docs/               Updated proposal, progress report, continuous extension writeup

Phase_4/              Final paper draft (ICML format) and the paper-faithful experiment
  paper/              main.tex, appendix.tex, refs.bib, compiled main.pdf
  src/                Continuous-state / finite-action environment, estimator, pessimism
  poc/                Driver for the finite-action study
  docs/               Writeup of the finite-action experiment and its diagnostic
```

---

## Method

**Identification.** Two families of bridge functions — reward-emission `b_R` and
dynamic-emission `b_D` — are recovered from conditional moment restrictions anchored by a
pre-decision negative control `O₀`. An exact tensor-contraction chain then turns estimated
bridges into a policy value.

**Estimation.** The paper's two-stage nonparametric procedure: a Stage-1 conditional mean
embedding by kernel ridge regression, then a Stage-2 closed-form solve. Implemented in both
dual (representer) and primal (operator) forms and verified algebraically identical.

**Scaling.** A direct implementation of the paper's history-conditioning would need on the
order of billions of table cells at benchmark scale. Three exact structural factorizations
(hashed encoding, core-diagonal bridge support, closed-form 2×2 local inversions) keep the
entire pipeline laptop-scale — no GPU.

**Pessimism.** Confidence regions are exact ellipsoids; the per-block inner minimum is the
standard ellipsoid support-function identity. The hard part is the coupled multilinear
minimization across blocks, which is where the failure mode above appears.

**Continuous extension.** A kernel/RKHS instantiation on a linear-Gaussian confounded POMDP
with an exact closed-form oracle, reusing the pessimism engine unmodified in coefficient
space. Documented in `Phase_3/docs/continuous_extension.md`, including an explicit note
that its continuous action space goes beyond the anchor paper's stated assumptions — the
paper assumes a *finite* action space.

---

## Reproducing the results

Requires Python 3 with NumPy, SciPy and Matplotlib. No GPU.

```bash
cd Phase_3
python poc/run_phase1_check.py            # environment + oracle regression suite
python poc/run_phase2_check.py            # estimator convergence
python poc/run_phase34_check.py           # plug-in identification + pessimism
python poc/run_comprehensive_benchmark.py # full sweep (long-running)
python poc/run_continuous_benchmark.py    # continuous kernel/RKHS extension

cd ../Phase_4
python poc/run_finite_action_benchmark.py # paper-faithful: continuous states, finite actions
```

Results are written to `experiments/*.json`; figures to `results/figures/`.

The final paper draft (ICML format, 4-page main body plus appendix) is
`Phase_4/paper/main.pdf`; rebuild it with
`cd Phase_4/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

The 720-observation benchmark builds on the public
[`clinicalml/gumbel-max-scm`](https://github.com/clinicalml/gumbel-max-scm) clinical
simulator, which is **not vendored here**. Clone it into `Phase_2/external/gumbel-max-scm`
to run the benchmark drivers; the toy-scale and continuous suites run without it.

---

## References

- M. Hong, Z. Qi, Y. Xu. *Model-based Reinforcement Learning for Confounded POMDPs.* ICML, 2024. — anchor paper
- C. Shi, M. Uehara, J. Huang, N. Jiang. *A Minimax Learning Approach to Off-Policy Evaluation in Confounded POMDPs.* ICML, 2022. — model-free comparator
- A. Mastouri et al. *Proximal Causal Learning with Kernels.* ICML, 2021. — kernel two-stage estimation
- X. Guo, M. Cai, Z. Yang, Z. Wang. *Provably Efficient Offline RL for POMDPs.* ICML, 2022. — pessimism precedent
- M. Oberst, D. Sontag. *Counterfactual Off-Policy Evaluation with Gumbel-Max SCMs.* ICML, 2019. — benchmark simulator
- A. Bennett, N. Kallus. *Proximal Reinforcement Learning.* Operations Research, 2024.

---

## Team

Soheil Sayah Varg · Mahdi Shirinbayan · Mahdi Ostadmohammadi

Graduate Deep Reinforcement Learning course project, Sharif University of Technology.
