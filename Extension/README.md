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
| (a) Rank structure of the design matrices | **done** | One claim survives and is stronger than first reported; two withdrawn |
| Adversarial review of (a) | **done** | Found both errors; see `docs/critic_findings.md` |
| Fix the eigengap rank rule | **required before (b)** | Numerical bug at `bridge_estimator.py:169` |
| Re-run (a) at `confound < 1.0`, per-action | **required** | See `docs/rank_structure.md` §9 |
| (b) Model-free pessimism, to observe the divergence | blocked on the fix | |
| (c) The rank cap as a proposition | not started | |

## What (a) found, after review

**Survives.** The exact null space is set by the **instrument's shape**, not the
latent state: `dim >= |O| - min(|O|, |O_0|)`. It is linear algebra, holds at every
sample size, and is detectable from the dimensions before fitting anything. The
three-tier spectrum is real, and the middle tier's decay rate is exactly `N^-1` —
cleaner than the `N^-0.8` we first published, which was 3-seed noise.

**Withdrawn.** Two claims did not survive adversarial review:

1. *"Population rank overstates usable rank."* An artifact of our own environment.
   `confound=1.0` makes the behavior policy a deterministic function of the latent
   state, and since every design is built per action, conditioning on the action
   conditions on the latent state. The direction we called "buried below the
   sampling floor" is exactly zero in the per-action population.
2. *"No eigengap rule can work."* The wall exists; the implementation was broken.
   A `1e-300` floor turns machine-negative eigenvalues into `~1e282` ratios, so
   `argmax` selects float sign noise. With a relative floor the rule recovers the
   correct rank.

The Phase 4 numbers stand — in the real Phase 3 toy the rule selected the right
subspace, though partly by luck.

Full writeup and corrections: `docs/rank_structure.md` §7. The review that found
them: `docs/critic_findings.md`.

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
