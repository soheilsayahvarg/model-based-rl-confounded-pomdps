# Extension — Post-Checkpoint Research

Work aimed at turning the course project into a publishable paper. This is **not
a course phase**: `Phase_1/` through `Phase_4/` are the graded deliverables and
stay frozen as submitted. `tools/make_bundles.py` only discovers `Phase_N`
directories, so nothing here is bundled into a course submission.

Like every phase directory, this one is self-contained. The modules it needs from
`Phase_3/src/` were copied in rather than imported across the boundary.

## Goal

The Phase 4 paper diagnoses a failure mode in one paper's pessimism step. The
question here is whether that failure is a property of the **method family** —
any proximal approach whose bridge is a function on the observation space — which
would change the unit of contribution from a critique of one paper to a result
about a research line.

## Status

| Step | State | Outcome |
|---|---|---|
| (a) Rank structure of the design matrices | **done** | Original hypothesis refuted; replaced by an exact prediction that holds 3/3 |
| (b) Model-free pessimism, to observe the divergence directly | not started | |
| (c) The rank cap as a proposition | not started | |

## What (a) found

The exact null space is set by the **instrument's shape**, not by the latent
state: `dim = |O| - min(|O|, |O_0|)`, confirmed exactly in every configuration.
The latent bottleneck produces a *separate, softer* effect — directions decaying
at `N^-1`, statistically rather than structurally empty.

The consequence lands on our own repair. Signal-Projected Pessimism picks its
subspace with an eigengap rank rule, which assumes a wall in the spectrum. There
are three tiers with smooth decay, so the rule is correct only in the Phase 3 toy
— the one configuration where `|S| = |O_0|` collapses them. That is a limitation
on the repair's generality, not on the Phase 4 numbers, which stand.

Full writeup with the tables: `docs/rank_structure.md`.

## Running

Python 3.9+ with NumPy. No GPU, no external data.

```bash
python poc/run_rank_diagnostic.py   # population + empirical rank across the grid
python poc/run_rank_verify.py       # hard-vs-soft null space by log-log slope
```

Results are written to `experiments/*.json`.

## Why a new environment was needed

`src/envs/dim_separated_pomdp.py` exists because the Phase 3 toy fixes
`|S| = |O_0| = 2`. Every competing explanation for the rank cap predicts the same
number there, so that environment cannot identify the mechanism. Separating the
dimensions is what makes the question answerable.
