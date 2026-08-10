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
  main.pdf        Compiled draft — 8 pages (4 main + references + appendix)
  figures/        Figures used by the paper
  icml2024.sty    Official ICML 2024 style files (unmodified)
  icml2024.bst
```

## Compliance with §5

| Requirement | Status |
|---|---|
| ICML template | Official `icml2024.sty`, unmodified |
| Main body ≤ 4 pages excluding references | Main body ends on page 4; references begin there |
| Abstract | Yes |
| Introduction | §1 |
| Related work | §2 |
| Method or analysis | §3 |
| Experiments or evidence | §4 (five subsections) |
| Discussion of limitations | §5 (five named limitations) |
| Conclusion | §6 |
| Appendix (unlimited) | Appendices A–D |

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

Three self-corrections are recorded in Appendix C, including one to the scope of
our own continuous study: the anchor paper assumes a *finite* action space, while
our continuous environment uses a real-valued action, so that experiment is
extrapolation beyond the stated theorem rather than an instantiation of it.

Open items suitable for mentor input: whether the honest-negative-result framing
is the right spine for the final version, whether to prioritize a paper-faithful
continuous configuration (continuous states/observations, finite actions) before
the final submission, and whether a second confounding-aware baseline is worth
the remaining time.
