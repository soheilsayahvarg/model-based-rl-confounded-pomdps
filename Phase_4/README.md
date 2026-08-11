# Phase 4 — Final Checkpoint

**Deliverable per the course requirements (§4.4):** a complete draft of the final
paper, plus a near-final presentation of the project to the mentor.

**Paper format per §5:** ICML template, main body limited to 4 pages excluding
references, unlimited appendix. Required sections: abstract, introduction, related
work, method or analysis, experiments or evidence, discussion of limitations,
conclusion.

---

## What is here

```
paper/
  main.tex        Paper source (main body + appendix include)
  appendix.tex    Appendix: setup, continuous derivation, corrections, bugs
  refs.bib        Bibliography
  main.pdf        Compiled draft — 10 pages (4 main + references + appendices)
  figures/        Figures used by the paper
  icml2024.sty    Official ICML 2024 style files (unmodified)
  icml2024.bst
src/              Continuous-state / finite-action environment, estimator, pessimism
poc/              Drivers: finite-action benchmark, coverage sweep, negative-control sweep
experiments/      Raw result JSON, one file per driver
docs/             Writeups of the three studies run in this phase
```

## Compliance with §5

| Requirement | Status |
|---|---|
| ICML template | Official `icml2024.sty`, unmodified |
| Main body ≤ 4 pages excluding references | Main body ends on page 4; references begin at the top of page 5 |
| Abstract | Yes |
| Introduction | §1 |
| Related work | §2 |
| Method or analysis | §3 (four subsections) |
| Experiments or evidence | §4 (seven subsections) |
| Discussion of limitations | §5 (four named limitations) |
| Conclusion | §6 |
| Appendix (unlimited) | Appendices A–H, pages 6–10 |

## Building

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires a TeX distribution with `natbib`, `booktabs`, `cleveref`, `hyperref`,
`microtype`. No network access needed — the ICML style files are vendored here.

## Isolation

This directory is self-contained. It does not read from or write to `Phase_2/` or
`Phase_3/`, which remain frozen as submitted checkpoints. The one figure reused
from earlier work was copied into `paper/figures/` rather than referenced across
phase boundaries.

## Notes for the mentor meeting

The paper deliberately leads with what did *not* work, because that is where the
empirical content is:

1. The anchor paper's pessimism step is numerically unusable as written (diverges
   to `-1779`), and we identify the mechanism.
2. Our repair achieves zero selection regret but forfeits coverage — the true
   bridge leaks 21% of its norm outside the projected subspace. Reported as a
   disclosed heuristic, not a certified bound.
3. At benchmark scale a naive confounding-blind baseline beats every bridge-based
   method in all 15 sweep cells, worst at zero confounding.
4. Pessimistic *selection* fails for a mechanism we first got wrong. We
   hypothesised that worst-case perturbations compound through the horizon, built
   the paper-faithful configuration to test it, and the hypothesis was refuted.
   A coverage sweep then confirmed the real mechanism quantitatively
   (correlation `-0.944`; regret falls to exactly `0.000` at 73% coverage).

Self-corrections are recorded in Appendix E, including one to the scope of our
own continuous study: the anchor paper assumes a *finite* action space, while the
Phase 3 continuous environment uses a real-valued action, so that experiment is
extrapolation beyond the stated theorem rather than an instantiation of it. The
paper-faithful configuration built here (continuous states and observations,
three discrete action levels) closes that gap.

Open items suitable for mentor input: whether the honest-negative-result framing
is the right spine for the final version, whether a second confounding-aware
baseline is worth the remaining time, and whether to spend it instead on a
horizon sweep (`T = 1..6`) or vector-valued state and observation spaces.
