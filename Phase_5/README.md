# Final Submission — Running the Theory: Model-Based RL for Confounded POMDPs

Deep Reinforcement Learning, Spring 2026 · Professor Mohammad Hossein Rohban ·
Sharif University of Technology.

Soheil Sayah Varg · Mahdi Shirinbayan · Mahdi Ostadmohammadi

**The paper is `paper/main.pdf`.** Everything else in this bundle is the code and
raw data behind it.

---

## Compliance with the Final Paper Requirements (§5)

| Requirement | Where / status |
|---|---|
| Short workshop-style paper, ICML template | `paper/main.pdf`; official `icml2024.sty`, unmodified |
| Main paper ≤ 4 pages **excluding references** | Body ends on page 4; **references begin at the top of page 5** |
| Unlimited appendix | Appendices A–I, pages 6–11 |
| Main 4 pages self-contained | Every headline claim is stated and qualified in the body; appendices add full grids, derivations and corrections, but the body does not depend on them to be read correctly |
| Abstract | Page 1 |
| Introduction | §1 |
| Related work | §2 (proximal causal inference · model-free OPE · pessimism in offline RL) |
| Method or analysis | §3 (identification, estimation, pessimism + failure mode, continuous extension) |
| Experiments or evidence | §4 (seven experiments) |
| Discussion of limitations | §5 (four named limitations) |
| Conclusion | §6 |

Verified mechanically on the shipped PDF: 0 overfull boxes, 0 unresolved
references or citations, no page overflowing its text column.

## The research question, and the answer

**Question.** Hong, Qi & Xu (ICML 2024) give a model-based theory for offline RL in
confounded POMDPs — bridge-function identification plus pessimistic planning —
with no algorithm, no code and no experiments. Built end-to-end, does it de-bias
policy evaluation at practical sample sizes, and can its pessimistic max–min step
be made to run at all?

**Answer, in four parts.**

1. **Identification works exactly.** Both bridge families estimate and carry
   through to a tractable pessimistic selection step, verified against exact
   oracles to ~1e-15.
2. **The pessimism step does not.** Run without the bridge-class bound the theory
   carries, it diverges to −20.98 against a true value of 2.018. Restoring the
   class caps it at −3.026 but does not cure it, and a confounding-blind plug-in
   is never beaten at any confidence width.
3. **At benchmark scale a naive baseline wins all 15 cells**, worst at *zero*
   confounding — estimation variance, not confounding bias, is what binds.
4. **Pessimistic selection tracks behavior-policy coverage, not value.** We
   turned that into a prediction and tested it: raising coverage of the optimal
   action drives regret to exactly zero. This is the manipulable causal result
   and the one we would build on.

A negative-control quality sweep then quantifies how good the framework's central
assumption has to be — the question the theory cannot ask of itself, because that
assumption is what it conditions on.

## What is here

```
paper/
  main.pdf        The submission
  main.tex        Source (body + appendix include)
  appendix.tex    Appendices A–I
  refs.bib        Bibliography
  figures/        Benchmark sweep figure
  *.sty, *.bst    Official ICML 2024 style files, vendored, unmodified
src/              Environment, bridge estimator, plug-in, pessimism engine
poc/              Drivers: finite-action benchmark, coverage sweep, negative-control sweep
experiments/      Raw result JSON, one file per driver
docs/             Full write-ups of the three studies
```

## Reproducing

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
python poc/run_finite_action_benchmark.py
python poc/run_coverage_sweep.py
python poc/run_negative_control_sweep.py
```

Needs NumPy/SciPy and a TeX distribution with `natbib`, `booktabs`, `cleveref`,
`hyperref`, `microtype`. No network access — the style files are vendored.

## A note on self-correction

The paper reports several results that overturned our own earlier claims, and
says so in place rather than quietly: Appendix E records two headline numbers
retracted after re-checking at more seeds, and Appendix I records three
implementation defects and two reporting errors found in our own pessimism study
— including that an advantage we had published was an artifact of comparing
across confidence-region sizes. Where our repair does not beat the trivial
baseline, we say that too.
