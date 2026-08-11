# Phase 3 — Progress Checkpoint

**Deliverable per the course requirements (§4.3):** present current progress to
the mentor — what has been completed, difficulties encountered, changes from the
original proposal, and the remaining tasks before the final checkpoint.

---

## What is here

```
phase3_progress_report.md      The progress report — read this first
docs/
  phase3_progress_report.pdf   Compiled version of the report
  proposal_updated.tex/.pdf    The proposal, revised to match what was actually built
  phase3_presentation_guide.md Speaking notes and anticipated questions
  continuous_extension.md      Derivation and results for the kernel/RKHS extension
src/
  envs/                        Toy POMDP, 720-observation benchmark, continuous env,
                               and the exact oracles for each
  estimation/                  Tabular, benchmark-scale, and kernel bridge estimators;
                               the Theorem-3.5 value chain
  pessimism/                   Confidence-region geometry and Signal-Projected Pessimism
  baselines/                   Independent model-free proximal comparator
poc/                           Driver scripts, 5-seed with 95% confidence intervals
experiments/                   Raw result JSON, one file per driver
results/figures/               Generated figures
```

## Compliance with §4.3

| Requirement | Where |
|---|---|
| What has been completed | `phase3_progress_report.md` §1–§3 |
| Difficulties encountered | §4 — the pessimism blow-up, the variance boundary, the scope correction |
| Changes from the original proposal | §5, and `docs/proposal_updated.tex` against the frozen Phase 2 proposal |
| Remaining tasks before the final checkpoint | §6 |
| Presented to the mentor | `docs/phase3_presentation_guide.md` |

## Reproducing

Requires Python 3.9+ with NumPy, SciPy and Matplotlib. No GPU.

```bash
# from the Phase_3/ directory
python poc/run_phase1_check.py            # environments + exact oracles
python poc/run_phase2_check.py            # estimator convergence, primal/dual equivalence
python poc/run_phase34_check.py           # value plug-in, de-biasing, pessimism
python poc/run_bigenv_check.py            # benchmark-scale recovery
python poc/run_comprehensive_benchmark.py # full sweep (long-running)
python poc/run_continuous_benchmark.py    # kernel/RKHS continuous extension
python poc/plot_results.py                # regenerate figures from the saved JSON
```

Results are written to `experiments/*.json`, figures to `results/figures/`.

The benchmark drivers additionally need the public
[`clinicalml/gumbel-max-scm`](https://github.com/clinicalml/gumbel-max-scm)
simulator cloned into `external/gumbel-max-scm` — it is third-party code and is
not redistributed here. The toy-scale and continuous suites run without it.

## Headline numbers

| | Result |
|---|---|
| Value chain vs. exact dynamic programming | `4.4e-16` |
| Estimator vs. oracle bridges (population limit) | `3.0e-15` |
| De-biasing vs. confounding-blind baseline (toy scale) | `12/12` cells |
| Vanilla Eq.-17 pessimism, run literally | diverges to `-1755` |
| Signal-Projected Pessimism | `V_low = 1.35`, `0.000` selection regret |
| Continuous extension: oracle self-consistency | `1.665e-16` |
| Continuous extension: de-biasing win rate | `4/4` policies, `3–11x` smaller bias |

## Two things this phase reported against itself

**A scope correction.** The continuous extension uses a real-valued action, while
the anchor paper assumes a *finite* action space. That makes the experiment
extrapolation beyond the stated theorem rather than an instantiation of it. This
is recorded verbatim in `docs/continuous_extension.md` §0 rather than relabelled.

**A retraction.** Two earlier headline numbers rested on 3-seed standard
deviations. Re-checking at 5 seeds changed one materially, and the other did not
replicate and was withdrawn.

## Isolation

This directory is self-contained and frozen as submitted. It does not read from
or write to `Phase_2/`, and nothing in it is required to read `Phase_4/`.
