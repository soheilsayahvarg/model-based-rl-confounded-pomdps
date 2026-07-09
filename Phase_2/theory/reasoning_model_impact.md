# Where Advanced-Reasoning Models Help (and Where They Do Not) in This Repo

*An analytical assessment. We did not use an o1/o3-class model in this project; nothing below is a measured comparison. "Long-form hidden chain-of-thought" here means only deliberate, extended reasoning, with no vendor performance claim attached.*

## The one-line thesis

Reasoning proposes; the oracle disposes. Long-form reasoning would have genuinely shortened the **derivation and synthesis** path in this math-heavy tabular-RL codebase. But every load-bearing correctness result — `V1 = 4.4e-16`, `V4`, the row-space leak, the negative benchmark — was established by **running code against an exact oracle**, and in several cases reasoning-from-theory produced the *wrong* prior that only the run corrected.

## Where extended reasoning genuinely helps: derivation and synthesis

These are tasks where the answer is a symbolic object whose correctness is largely a matter of not slipping across many mechanical steps — exactly what a long CoT is good at.

- **Multilinear tensor-dimension bookkeeping.** The Theorem-3.5 chain is a stack of einsum contractions producing `bR_hat (3,2,3,2,3)` `[t,a,o~,r,o]` and `bD_hat (2,2,3,3,3)` `[t,a,o~,o~',o]`, with forward/backward recursions `V = G_1 · Hb_1`. Tracking index alignment across this many axes is error-prone for a person and well-suited to deliberate reasoning.
- **The joint coordinate-descent structure.** Deducing that all reward blocks minimize *jointly* because `gR[t]` depends only on `b_D` (linear → one-shot), while the `T-1` dynamic blocks are coupled and must be refreshed, is a chain-of-dependencies argument.
- **The vec-ordering invariant `w·n_y + y`.** A mechanical convention that must stay identical across estimator ↔ ellipsoid ↔ value-plugin. A CoT can hold this invariant in view across modules.
- **The primal/dual push-through identity.** Showing *symbolically* that the dual representer `α = (G + N₂λ₂I)⁻¹p̂`, `G = (M̂ᵀM̂) ⊙ K_Y″`, equals the primal `(T̂₂+λ₂I)⁻¹ĝ₂` — before the numerical `2.4e-15` check confirmed it.
- **Spectral-structure reasoning.** Explaining *why* `cond(T̂₂+λ₂I)` stays flat under the O₀-noise sweep — the structural `|O|>|S|` null space pins `eig_min ≈ 0`, so the regularized condition number is not a sufficient ill-posedness monitor and the restricted-spectrum `σ_min` is the right statistic.
- **Surfacing the leak concern as a hypothesis.** Careful reasoning *could* have asked, up front, "does the true bridge actually lie inside the projected signal subspace?" — the exact question the projected-pessimism heuristic hinges on.

## Where reasoning does not help: empirical verification no amount of thinking replaces

- **`V1 = 4.4e-16`** (plug-in with true bridges == DP == enumeration). No reasoning certifies floating-point agreement at this scale; running the three implementations against each other does.
- **The 21% row-space leak.** Reasoning can *raise* the hypothesis that projection excludes the truth; only **measuring** `bR_t3` leak `0.38 / norm 1.80` forced the Entry-9 reframe from "sound repair" to "heuristic." The number is decisive, not the argument.
- **The 15/16 negative benchmark.** Plug-in is worse than naive in 15/16 cells, *worst* at `kappa=0` (23–46× the naive bias, where there is nothing to correct), and non-monotone in N. Reasoning-from-theory had produced the **opposite** optimistic prior ("de-biasing pays off"); the run over the 720×2 env reversed it.
- **`V4` holding by sign-alignment, not containment.** Because the truth is not even in the feasible set post-projection, `V_low ≤ V_true` cannot be certified by derivation — only the sweep across all policies/seeds/`c` shows it (and honestly, `V4:projected` is now *vacuous*).
- **Coordinate descent upper-bounding the exact inner min.** The `restart_gap` growing `1e-13 → 38` is a runtime observable exposing a non-global stall (canonical `min x·y` returns 0, not −1); no proof was going to hand us that.
- **The hyperparameter-schedule finding.** RKHS `N^{-1/2}` λ₁ schedules — the "reasoned" default from the anchor theory — over-shrink the tabular CME 80% at `t=3`. Grid calibration, not reasoning, produced `λ₁ = 1/N₁`, `λ₂ = 0.03/√N₂`.

## A fresh, dated instance (2026-07-09, the model-free-baseline build)

While adding the model-free proximal baseline this session, the *first* implementation
followed a perfectly reasonable derivation: estimate a per-stage value bridge and sum
`E_{o~ν_t}[Σ_a b_V(a,o)]` over stages. It compiled, ran, and looked plausible. The oracle
check immediately exposed it as **systematically biased for the deterministic policies**
(`always_a0` converged to `1.61 ≠ 1.865`, error *flat* in N — a bias, not variance). The
diagnosis: summing immediate rewards against the **behavior** observation marginal is wrong
off-policy whenever the target shifts state occupancy; the correct Shi value bridge needs a
**backward recursion** that carries the target-policy continuation value and reads `J` off
the *initial* (policy-independent) marginal. After the fix, `max|MF − V_true| = 0.0076` at
N=200k, converging cleanly. This is the thesis in miniature: the derivation was confident and
wrong in a way that *only the run against DP caught*, and the fix was itself a second
derivation that again had to be verified, not trusted.

## The honest boundary between the two

The gains from reasoning are real but **bounded by how good the oracle harness already is**. Here the harness is excellent: an exact `pinv` oracle, DP ground truth, finite-difference gradient checks (worst `4.9e-10`), and population-limit tests (`P0 = 3.0e-15`). Anything a longer derivation would have caught, these cheap exact checks *also* catch — so the marginal value of extended reasoning is real for getting to a correct symbolic form *faster*, and near-zero for *establishing* correctness that the oracle establishes anyway.

And repeatedly, theory-driven reasoning was simply **wrong** (the λ schedule; "de-biasing pays off"; the per-stage model-free value). A reasoning model given the same premises could have repeated those errors with more confidence. The run is what corrected them.

**Bottom line:** for this repo, use extended reasoning to derive and cross-check symbolic structure — tensor dims, the vec-ordering, the primal/dual identity, the null-space spectral argument, and to *pre-register* concerns like the row-space leak. Then treat every such output as a hypothesis and settle it against the exact oracle. Reasoning and empirical verification are complementary; neither substitutes for the other, and in this codebase the oracle has the last word.
