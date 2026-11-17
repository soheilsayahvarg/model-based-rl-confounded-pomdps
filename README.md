# Candidate-Set Sensitivity of Proximal Pessimistic Policy Selection

Soheil Sayah Varg, Mahdi Shirinbayan, and Mahdi Ostadmohammadi

[Preprint source](Extension/paper/workshop_draft.tex) ·
[Reproduction instructions](Extension/paper/REPRODUCE.md) ·
[Background paper (Hong, Qi, and Xu, ICML 2024)](https://proceedings.mlr.press/v235/hong24d.html)

This study examines how candidate construction affects a heuristic offline
policy selector in confounded partially observable environments. Adding a
behavior clone improves selection in the original synthetic design and under
transferred calibration at the largest sample size. Reward reversal makes the
same addition harmful in four of the five tested environments. A separate
single-decision illustration using public International Stroke Trial outcomes
finds no resolved survival-value difference.

The findings concern a specific implementation with an incomplete instrument,
heuristic calibration, and an unrestricted bridge class. The manuscript
details these conditions and the scope of its conclusions. The reproduction
package includes code and per-fit results. It downloads the public IST source
separately from its original repository.

The `Phase_*` directories preserve the earlier course project. Use the
preprint above for the current scientific claims and results.
