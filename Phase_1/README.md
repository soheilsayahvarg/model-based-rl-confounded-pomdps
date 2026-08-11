# Phase 1 — Literature Review

**Deliverable per the course requirements (§4.1):** find and study relevant papers
on the selected topic, and present the selected papers to the mentor with an
explanation of why each was chosen.

**Topic:** causal reinforcement learning with hidden confounders and partial
observability — specifically, offline policy evaluation and selection when a
latent state drives both the logged actions and the outcomes.

---

## What is here

```
Literature Review.pdf   The written review — selection table, per-paper summary,
                        comparison of assumptions and methods, and the direction
                        they motivate
Slides.pdf              The slide deck used for the mentor presentation
```

## Compliance with §4.1

| Requirement | Status |
|---|---|
| At least six relevant papers | Eight |
| A short summary of each paper | §2, one subsection per paper |
| Comparison of ideas, assumptions, methods, limitations | §3 |
| How the papers motivate the project direction | §4 |
| Presented to the mentor | `Slides.pdf` |

## The eight papers

| # | Paper | Venue | Role in the project |
|---|---|---|---|
| 1 | Model-based RL for Confounded POMDPs | ICML 2024 | **Anchor paper** — the one we implement |
| 2 | Pessimism in the Face of Confounders (P3O) | arXiv 2024 | Model-free counterpart using the same pessimism idea |
| 3 | A Minimax Learning Approach to OPE in Confounded POMDPs | arXiv 2021 | Continuous spaces via minimax rather than tabular |
| 4 | Off-Policy Evaluation in Partially Observable Environments | AAAI 2020 | Foundational: why standard OPE fails under partial observability |
| 5 | Proximal RL: Efficient OPE in POMDPs | arXiv 2021 | Proxy variables and the bridge-function idea |
| 6 | A Spectral Approach to OPE for POMDPs | arXiv 2021 | A different toolkit (spectral/moment matrices) for the same problem |
| 7 | Confounding-Robust Policy Evaluation in Infinite-Horizon RL | NeurIPS 2020 | Bounds instead of point identification when identification fails |
| 8 | Deconfounding RL in Observational Settings | arXiv 2018 | Latent-variable/deep approach to recovering the confounder |

The PDFs of these papers are not redistributed in this repository.

## What this phase decided

Paper 1 was selected as the anchor because it is **theory only** — no algorithm,
no code, no experiments — which makes an end-to-end implementation a genuine
contribution rather than a reproduction. Paper 2 was retained as the model-free
comparator, and Papers 5 and 6 as alternative identification strategies to
contrast against the bridge-function route.

The paper list was treated as provisional, as §4.1 permits. Later phases added
Mastouri et al., *Proximal Causal Learning with Kernels* (ICML 2021), when the
continuous extension required a kernel two-stage estimator.

## Isolation

This directory is self-contained and frozen as submitted. It does not read from
or write to any other phase.
