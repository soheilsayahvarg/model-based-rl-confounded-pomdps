# Extension — Post-Checkpoint Research

Work aimed at turning the course project into a publishable paper. This is **not
a course phase**: `Phase_1/` through `Phase_4/` are the graded deliverables and
stay frozen as submitted. `tools/make_bundles.py` only discovers `Phase_N`
directories, so nothing here is bundled into a course submission.

Like every phase directory, this one is self-contained. The modules it needs from
`Phase_3/src/` were copied in rather than imported across the boundary.

## Start here

`docs/claims.md` is the ledger: one row per claim, its current status, and where
it was established or killed. The other documents were written in sequence and
several contain statements later work superseded — where a document and the
ledger disagree, the ledger is current.

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
| (c) A proposition on non-contraction | **done, reviewed** | Width branch survives and locates the anchor paper; coverage branch refuted on this environment |
| Re-run the projected-width slope at 20 seeds | **done** | Geometry confirmed to 0.001; the *deployed* estimator refuted |
| Run part (c) coverage where completeness FAILS | **done** | Coverage branch confirmed; complete-design control clean at 40/40 |
| Does the width schedule change the *decision*? | **done** | **No**, where completeness holds: `beta_g = 0` exactly, 0.000 regret under all four schedules |
| Map `beta_g`, the gradient's leakage | **done** | Two independent switches: incompleteness leaks the truth, confounding leaks the gradient |
| Decision-level test at a nonzero floor | **done** | Width tracks `e/2` and **grows**; regret degrades under the paper's own schedule |
| Repeat the decision sweep at 20 seeds | **required** | 3 seeds; effect real but small, not yet quotable |
| Does the plug-in find `always_1`? | **done** | **Yes**, regret 0.000 at `kappa` 1.0/1.5; a one-`kappa` version of this check nearly produced a false reversal |

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

## What (c) established, after review

The clean statement is two lines and needs no schedule. Under the structural null
and `P_Nul b_hat = 0`, **any** confidence region that covers the truth pays

```text
W  >=  beta * beta_g          at every N,
```

for any ridge and any width rule — the regularizer cancels. The power-law
dichotomy we originally led with (`e = tau + kappa - 1`; contraction and coverage
governed by one exponent with opposite signs) is a corollary of that.

**Predictions were committed before the verification script existed** (`53d71a5`,
§1–§4 with §5 empty; the review verified that commit). Width slopes matched to
≤0.010 in every cell, including `+0.240` against a predicted `+0.250` — **more
data making the pessimistic bound strictly worse**.

The one genuinely new consequence is where it puts the anchor paper. Its width
`N2^(-alpha/(2alpha+2))` and ridge `N2^(-alpha/(alpha*c2+1))` sit inside our own
family, and

```text
e_paper = alpha*(2*alpha + 1 - alpha*c2) / ((alpha*c2+1)*(2*alpha+2))  >  0
```

for every admissible `(alpha, c2)` — 30/30 cells, minimum 0.0217. **The paper's
own schedule never promises a contracting region**, and the penalty it assigns a
leaky policy grows with `N`. The dichotomy explains the paper's design; it does
not indict it.

**But the word "leaky" is load-bearing, and we under-weighted it.** Asking whether
the schedule changes the *decision* rather than the interval, the answer on a
complete design is **no**: the value gradient has exactly zero mass in the design
null (`beta_g = 0.000e+00`, three policies, at the exact bridges), so the width is
signal-driven and shrinks under every schedule, and all four — including the
paper's own — select the optimal policy at every `N`. Completeness kills both
halves of (H4) at once, the truth's and the gradient's. `e_paper > 0` is real
algebra whose practical bite needs `C*_pi = infinity`. See
`docs/regret_vs_schedule.md`.

**Where both switches are on, it does bite.** `beta` is switched by an inadequate
negative control (`|O_0| < |S|`) and `beta_g` by confounding, so the floor
`W >= beta*beta_g` is nonzero exactly when the two meet — and there the `N^(e/2)`
law applies, the width **grows** with data under `e > 0` (+0.207 measured), and
**selection degrades**: under the paper's own schedule regret runs
`0.064 -> 0.098 -> 0.123` while the `e = 0` control stays flat at `0.064` to the
digit. That is the first confirmed decision-level instance of the branch, at 3
20 seeds with 95% intervals, and under the paper's own schedule the two curves
**cross**: pessimism is better at `N = 2,000` (0.069 vs 0.164) and worse from
`N = 32,000` on (0.123 vs 0.064). The layer helps when data is scarce, which is
what it is for, and then actively hurts as data accumulates.

The cleanest comparison the extension has produced sits in the same run. At
`kappa = 1.5`, on identical data and identical fits, the only difference being
whether the pessimism layer is applied:

```text
                N =    2,000     8,000    32,000   128,000
plug-in            0.0894    0.0226    0.0000    0.0000
pessimistic        0.0952    0.1218    0.1714    0.1804
```

20 seeds, 95% intervals. At `N = 128,000` all twenty seeds put pessimism at
`0.1804` and all twenty put the plug-in at the optimum — zero interval on both,
complete separation, from a start where they are statistically
indistinguishable.

So it is not that pessimism is merely worse — the layer actively destroys a
selection the underlying estimates get right, faster the more data it is given.
Plug-in dominance therefore survives, and now holds in an incomplete design as
well as the complete one where it was first found.

**What round 6 refuted**, in full in `docs/noncontraction.md` §5.5 and §7:

- The coverage branch's every quantitative claim on this environment. We treated
  `beta = 0.053` as an environment constant; it decays at **−0.495**, and
  `beta_pop = 5.4e-16` — exactly zero. `N* = 1.7e9`, `K`, the `beta^-4` scaling
  and the `m_null` rate test are all withdrawn. The `m_null` slope we reported as
  confirmation was algebraically forced; with the per-fit `beta` the margin slope
  is **+0.41**, the opposite sign.
- The novelty claim. The trade-off is Tikhonov source-condition theory
  specialised to an exact-null design (`e' = tau + theta*kappa - 1` in general).
- Our escape claim: **no** width rule escapes, not even a truth-dependent one.
- The projection, as deployed: rank-conditionally its slope is right to 0.001,
  but the mean over parallel-analysis-selected ranks **grows** at +0.236 when
  `e > 0`, because a 5% over-selection rate re-admits a null direction whose
  contribution is unbounded.

Structurally, the coverage branch requires the anchor paper's **completeness
assumption to fail** — `K0` invertible forces the min-norm bridge out of the
population null. The environment where it does fail, `(2,6,4)` at `confound=1.0`,
is the one we never ran part C on. **That experiment has now been run** — see
`docs/completeness_and_h4.md`.

## Completeness is the switch (the step after (c))

Round 6's attribution — that `beta_pop = 0` on the toy is caused by the anchor
paper's completeness assumption — is confirmed as **population algebra, 50/50
cells**: `beta_pop <= 1.5e-15` wherever per-action completeness holds, and 57–77%
of the bridge norm wherever it fails. Sharp, with no intermediate cells.

Two routes to failure, and only one is worth building on. `confound = 1.0` is a
knife-edge, restored at 0.9. `|O_0| < |S|` is **dimensional**, holds at every
confounding level, and is checkable before fitting — composing with step (a) into
two pre-fit questions: does the design have an exact null, and does the truth have
mass in it.

Running the coverage test there gives step (c)'s coverage branch its first
confirmed instance, with the control it never had:

| | `beta_emp` slope | coverage over 512x in `N` |
|---|---|---|
| **incomplete** (4,6,2) | **−0.001** (constant) | crossings at all four `c`, monotone loss |
| **complete control** (2,6,4) | **−0.660** (→ 0) | **3/3 at every `N` and `c`**, 40/40 cells |

The crossing scales as `c^2` (+2.158 measured vs +2.000). The absolute `N*`
predictions were **wrong** and are corrected post-hoc in §5.4 — two errors in our
own prediction, disclosed there.

Scope, unchanged from round 6: `|O_0| < |S|` **violates the paper's Assumption
3.3**. This is a robustness result about a regime the paper excludes, not a defect
in it.

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
