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
| Fix the eigengap rank rule | **done** | Correct in 16/18 cells vs 0/18 for the shipped rule |
| Re-run (a) at `confound < 1.0`, per-action, 20 seeds | **done** | Surviving results confirmed; middle tier decays at exactly `N^-1` |
| Gap-free rank selection (parallel analysis) | **done** | Correct in every cell at every `N`; supersedes the eigengap rule |
| (b) Model-free pessimism, to observe the divergence | **done, reviewed** | Leakage mechanism survives and is causal; headline number withdrawn |
| Reproduce the review's §3/§5/§6 numbers independently | **required** | Quoted, not yet verified here |
| Re-run the leakage table at 20 seeds | **required** | PA rank carries ±1 noise at `N=4,000` |
| Check the omitted bridge-class bound `M_R` | **done** | Omission real (excursion 14x the a priori bound); divergence survives it |
| Same check on the model-based ellipsoid | **done** | Same verdict; and the norm ball beats our own repair at `c=1` |
| Compare the repair against the norm-constrained baseline | **done, reviewed** | Magnitudes invalid (solver + class-shape defects); D1/D2 survive |
| Fix the norm-ball solver, per-block admissible `M` | **done** | Certified two-multiplier solver; `M` from the exact bridges |
| Regret-vs-`c` on a common grid, with the plug-in baseline | **done** | Published advantage is a cross-`c` artifact; plug-in wins at every `c` |
| Re-run under the paper's sup-norm (box) class | **done** | Box is the WEAKEST class; our "norm ball beats the repair" claim reverses |
| (c) A proposition on non-contraction | **done** | One exponent governs both width and coverage; predictions committed before the run |
| Re-run the projected-width slope at 20 seeds | **required** | PA rank noise; slope consistent but not verified |

## How (b) got unblocked

The corrected eigengap rule recovers the right rank inside the estimator, but
only as `N` grows. We first attributed the difference to whether the instrument
shape forces exact zeros (`|O_0| < |O|`). **That criterion was wrong**, and our
own table refuted it — `|O_0| < |O|` holds in every row, including every failing
one, so it has no discriminating power at all.

The real condition is whether the per-action population rank **saturates the
shape cap**:

| | `rank(P_a) = min(\|O\|,\|O_0\|)` | `rank(P_a) < min(\|O\|,\|O_0\|)` |
|---|---|---|
| eigengap rule | correct at every `N` | needs `N` in the millions |

When the rank saturates, the deficiency is entirely a machine-zero cliff and the
gap exists immediately. When it does not, a statistically-empty tier sits between
signal and cliff, and the rule must wait for it to fall away. The convergence
threshold has a closed form, `N* ≈ c / sqrt(lambda_r * lambda_1 * rel)`, giving
`≈1.6M` for the slowest cell — which is where it converges.

This is why the Phase 3 toy worked: `|O|=3, |O_0|=2` with `rank(P_a)=[2,2]`
saturates the cap.

**The fix is not to restrict the regime but to stop relying on a spectral gap.**
A parallel-analysis selector — permute `O_0` within each action bin to destroy the
`(o_t, o_0)` dependence while preserving marginals, then keep directions above the
95th percentile of the permutation null — selects the correct rank in every cell
at every `N` tested, including 90% at `N=8,000` on the cell where the eigengap
rule needs `N≈2M`. It never selects the cliff.

One subtlety makes it work: the null's own outer product of marginals is rank 1,
so the permutation test can only detect dependence *beyond* the first direction,
and the count must be `1 + #{i >= 2 : lambda_i > q95(null)}`. Without that
correction it returns 0 whenever the behavior policy is deterministic.

Step (b) can therefore run in **both** regimes with this selector, and the regime
restriction is dropped.

## What (c) established

Step (b) reported that the pessimism width did not shrink across a 64× increase in
data. Step (c) shows that was **the boundary case of a dichotomy**, not a quirk of
one schedule. With ridge `lam = lam0*N^-kappa` and width `xi = c/(N*mu)`,
`mu = m*N^-tau`, one exponent `e = tau + kappa - 1` governs both sides:

- `e >= 0`: coverage is sustainable at every `N`, and the width **never contracts**
  — flat at `e = 0`, **growing with data** for `e > 0`.
- `e < 0`: the width contracts at `N^(e/2)`, but coverage fails beyond a finite
  `N* = K^(1/e)`.

No schedule gives both. The exponent that shrinks the width is the one that makes
the region drop the truth.

Predictions were written and **committed before the verification script existed**
(commit "stated before it is tested"), because the two results that survived
adversarial review were both derived analytically first and the withdrawn ones
were not. Measured: width slopes match to ≤0.010 in all three cells of the global
convention — including `+0.240` against a predicted `+0.250`, i.e. **more data
making the bound strictly worse** — and the coverage margin degrades at `−0.4965`
against a predicted `−0.5000`.

What did **not** work is reported in `docs/noncontraction.md` §5.4–§5.5: the
projected-width slope is too noisy at 3 seeds to call, and the coverage-crossing
test was vacuous — the grid stopped 1,600× short of the predicted threshold, so it
could not have failed. On this environment the true bridge has only 2.9% of its
norm in the null space, so the coverage branch is real but inoperative at any
realistic `N`; the width branch is the one that bites.

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
