# Phase 2 Checkpoint Log (append-only)

## [2026-07-08] Entry 1 — Phase 0 start
- Created workspace scaffold: `proposal/`, `theory/`, `src/envs/`, `src/estimation/`, `src/pessimism/`, `poc/`, `experiments/`, `external/`.
- Toolchain verified: Python 3.13.1, NumPy 2.2.2, git 2.49 on Windows 11.
- Locked substrate decisions (see `workspace_checkpoint.json` → `substrate_decisions`), including the resolution of the
  bridge-existence landmine: sepsis observations get a per-step noisy metabolic-marker channel `m_t` (P(m=1|diab)=q1 ≠ q0)
  so the emission block per vitals-index is an invertible 2×2 — without it, deterministic masking makes one-step bridges
  nonexistent (rank 720 < 1440).
- Next: proposal.tex skeleton, then simulator acquisition.

## [2026-07-08] Entry 2 — Environment wrapper complete
- Cloned clinicalml/gumbel-max-scm; extracted + validated parameter pickle: tx (2,8,720,720) row-stochastic,
  reward provably destination-only -> rdest[720] (416 neg terminals, 1 pos terminal), p0 (2,720), pmix=[0.8,0.2].
- `src/envs/simulated_env.py` SMOKE-TESTED: augmented chain P (8,1442,1442) with absorbing terminals,
  exact finite-horizon optimal Q, confounded behavior (kappa knob), noisy-marker observation channel
  (bridge existence guaranteed for q0!=q1), O_0 negative control (Assumption 3.1 by construction).
- Smoke run: 2000 trajectories, T=3, behavior mean return -0.0465 (kappa=1).
- Next: toy_pomdp.py, oracle_module.py (DP values, true bridges by pinv, Theorem-3.5 chain check, naive baseline).

## [2026-07-08] Entry 3 — PHASE 1 COMPLETE (all oracle checks passed, exit criterion met)
- `toy_pomdp.py` + `oracle_module.py` + `poc/run_phase1_check.py` written and green:
  - Toy true bridges solve latent moment equations to residual ~1e-16 (direct pinv inversion).
  - **Theorem-3.5 sequential bridge integration reproduces DP V(π) to 4.4e-16** across all 4 candidate
    policies — the identification oracle is VERIFIED.
  - Big-env on-demand bridge solver (per-x 2×2 inversion) residual ~1e-16; MC-vs-DP sanity passed both envs.
- **Exit criterion (naive uncorrected evaluator vs truth): MET.**
  - Toy (N=200k): naive bias +0.13…+0.32 on values ~1.5–2.2 (7–22% relative), all policies.
  - Big env, POPULATION (infinite-data) naive → pure structural bias: κ=1 biases all positive
    (+0.0036…+0.0114), 2.4–5.1× larger in magnitude than κ=0 (−0.0021…+0.0048) [ratio corrected in audit,
    Entry 7]. Confounding signature isolated from noise.
  - Finite-N (20k) naive additionally shows rare-action sampling noise (never_treat bias flips sign across
    seeds) — coverage finding to report.
- Results: `experiments/results_phase1.json`. Phase 2 estimation NOT started (per instruction).

## [2026-07-08] Entry 4 — PHASE 2 ESTIMATION CORE COMPLETE (all checks passed)
- `src/estimation/bridge_estimator.py`: two-stage estimator, exact tabular instantiation.
  Stage-1 ridge CME `μ̂(w|x) = count(w,x)/(count(x)+N₁λ₁)` + MLE pmf targets; Stage-2 closed forms in BOTH
  the paper's dual representer form `α = (G + N₂λ₂I)⁻¹p̂`, `G = (M̂ᵀM̂) ⊙ K_Y″` and the primal operator
  form (Eq. 12) — **identical to 2.4e-15**.
- **Population-limit test (P0): 3.0e-15** — with exact μ, p, design, the solver reproduces the oracle pinv
  bridges at machine precision for all stages/families. Estimator target structurally EXACT (the
  min-Frobenius-norm solution decouples per (a,y)-slice into the oracle's min-ℓ₂ pinv solutions).
- Hyperparameter calibration (coarse grid, 3 seeds): split 0.7 ≻ 0.5 (confirms anchor's N₁ ≫ N₂);
  λ₁ = 1/N₁ (N^{-1/2} schedules inject √N₁ pseudo-counts → 80% CME shrinkage at t=3 — RKHS schedules
  don't transfer to strictly tabular classes); λ₂ = 0.03/√N₂ (must dominate stage-1 noise in
  weakly-identified directions; 1/N₂ under-regularizes → t≥2 error plateau).
- **N-scaling sweep (P2, exit criterion): MONOTONE decrease** at every step, both families:
  errR 1.360→0.459, errD 0.732→0.223 over N = 1k→20k (~N^{-0.36..-0.40}).
- **O_0-noise sweep (P3)**: stage-1 error degrades 10× as ν→0.45 (negative control saturates; X₁ has no
  other latent channel); later stages compensated by history — matches completeness theory.
  ⚠ DIAGNOSTIC FINDING: cond(T̂₂+λ₂I) stays FLAT across the ν sweep (eig_min pinned ≈0 by the structural
  |O|>|S| null space) — the regularized condition number is NOT a sufficient ill-posedness monitor; the
  pessimism layer must use the restricted-spectrum signal σ_min (per-(a)-slice).
- Results: `experiments/results_phase2.json`. Pessimism layer NOT started (per instruction).

## [2026-07-08] Entry 5 — PHASES 3+4 COMPLETE: end-to-end pipeline operational (toy, oracle-verified)
- **Phase 3** `src/estimation/value_plugin.py`: Theorem-3.5 chain as exact einsum contractions with
  forward/backward recursions (V = G₁·Hb₁) yielding exact per-block gradients for free.
  - V1: plug-in with true bridges == DP == enumeration to **4.4e-16**.
  - V2: **plug-in beats naive in 12/12 (N, policy) cells** — naive bias flat (+0.13…+0.33, structural),
    plug-in bias shrinks with N (worst policy: 0.11 → 0.057 → 0.031 at N = 5k/20k/50k).
- **Phase 4** `src/pessimism/ellipsoid_opt.py` + `pessimistic_optimizer.py`: exact ellipsoid regions
  (H = T̂₂+λ₂I incl. ridge, anchor Eq. 121), closed-form per-block min, blockwise coordinate descent
  (R-blocks jointly linear → one-shot; D-blocks refreshed), multi-restart; widths ξ_t = c/(N₂σ̂₂,t)
  via the restricted-spectrum rule; full coverage instrumentation.
  - ⚠ **HEADLINE FINDING — vanilla Eq.-17 blow-up**: at the smallest coverage-guaranteeing width
    (c=0.1) the joint minimum is already uninformative: V_low(best) = −1.12 (outside feasible [0,3]),
    regret 0.684; diverges to −1755 at c=10. Mechanism: structural null directions (H-weight ~λ₂)
    couple across blocks under the joint multilinear min.
  - ✅ **REPAIR — signal-projected pessimism** (per-action eigengap design eigenvectors ⊗ I_index;
    Nair–Jiang-style rank restriction, purely data-driven): coverage onset improves 0.1 → **0.03**, and
    at onset: V_low(best) = 1.346 (informative), **zero selection regret (all seeds)**, restart gaps
    ~1e-12. Vanilla has NO width with coverage=1 AND regret=0; projected does.
  - V4: **V_low ≤ V_true PASS** for all policies/seeds/c in both variants.
  - Ranking at operating point (projected, c=0.03, N=20k): obs_dependent 1.346 ≻ uniform 0.956 ≻
    always_a0 0.942 ≻ always_a1 0.845 (true best obs_dependent = 2.189 — correctly selected).
- Results: `experiments/results_phase34.json`. Next: scale to the 720×2 benchmark env; headline figures.

## [2026-07-08] Entry 6 — PHASE 2 COMPLETE (handoff)
- **Benchmark scaling** (`src/estimation/big_bridge_estimator.py`): three exact factorizations — hashed X_t,
  core-diagonal bridge support, per-cell square 2×2 closed forms; sparse b_D → dense policy-folded chain.
  ⚠ Honest boundary result: at 720×2 scale, N ≤ 60k, per-cell estimation variance (bR RMSE ~1.0 on ok-cells,
  plug-in |bias| 0.01–0.09) exceeds the small structural naive bias (0.004–0.011) — de-biasing pays off in the
  strong-confounding toy regime, not yet at mild confounding + feasible N. Fix identified (cross-fitting /
  partial pooling), stated as future work.
- **Figures/tables**: `results/figures/fig1..fig4.png` (validated palette, rendered + inspected) +
  `results/table1..table5.csv`.
- **Proposal**: `proposal/proposal.pdf` COMPILED, exactly 2 pages — claim + differentiation, calibrated
  hyperparameters, σ₂ spectrum rule, Eq.-17 blow-up + signal-projection repair as contributions, boundary
  result disclosed.
- All four validation suites green. This file + `workspace_checkpoint.json` are the handoff artifacts.

## [2026-07-08] Entry 7 — MASTER AUDIT (adversarial code + proposal review)
**Quantitative corrections (claims did not match raw numbers):**
- κ-contrast ratio "2–7×" was WRONG → actual per-policy ratios 2.4–5.1× (computed from raw
  `results_phase1.json` B5). Fixed in proposal.tex, README.md, checkpoint files, Entry 3 above.
- "≈N^{-0.4}" recovery rate was generous → actual empirical rates N^{-0.36} (b_R), N^{-0.40} (b_D). Fixed.
- Big-env plug-in |bias| range "0.01–0.09" understated the max → actual 0.006–0.12. Fixed.
- "coverage onset 3× earlier" → "at least 3× earlier on our width grid" (onset is grid-bracketed).
- Proposal environment description said T ≤ 4 → actual T = 3. Fixed; also added the bridge-nonexistence
  rank fact (720 < 1440) to the environment bullet.
**Code audit findings (no functional bugs; hygiene + documentation fixed):**
- Line-level cross-check of all contractions/encodings against the anchor's formulations: stage-2 dual
  G = (Γᵀ K_W Γ)⊙K_{Y″} vs primal (T̂₂+λ₂I)⁻¹ĝ₂ agree to 2.4e-15; vec-ordering (w·n_y + y) consistent
  across estimator ↔ ellipsoid ↔ value-plugin; N₁/N₂ split leak-free (stage-1 tables from i1 only,
  targets evaluated at stage-2 x but estimated on stage-1); ellipsoid H = T̂₂+λ₂I is the exact quadratic
  form (linear term vanishes at b̂); Cholesky args PD by λ₂ > 0; all divisions guarded (no NaN paths).
- NEW independent test: finite-difference gradient check of the value plug-in at a generic random bridge
  point — worst error 4.9e-10 (exact at FD truncation scale). Gradients drive all pessimism steps.
- big_bridge_estimator: documented the no-split deviation explicitly (square-case direct inversion);
  cleaned a garbled comment; run_bigenv_check: removed an unused variable, simplified a redundant stack
  (p_m ≡ Em2). Cell-conditioning validity argument sharpened in the docstring (cell event is
  (S_t, A_t)-measurable since the core channel is deterministic).
**Re-validation after fixes:** all suites re-run green (see Entry 7 addendum), proposal recompiled at 2 pages.

## [2026-07-09] Entry 8 — FINAL POLISH: independent 4-reviewer adversarial audit
- Ran workflow of 4 refute-oriented reviewers (each cross-referencing the anchor digest), then verified
  their findings on the merits. **No functional bug with a concrete failing input on any real code path.**
- **value_plugin.py — CONFIRMED_CORRECT** (reviewer's own FD gradient test <4e-10 + brute-force nested
  Thm-3.5 contraction 1e-15). Fixed 2 low doc issues: reward_levels is ORDER-coupled to bR's r-axis (not
  just the value SET); the `V==sum(per_step)` assert is self-consistency only (convention validated
  externally by run_phase34 V1 = 4.4e-16).
- **big_bridge_estimator.py — CONFIRMED_CORRECT** (chain ops hand-traced; tower-property validity of the
  no-history conditioning confirmed). Fixed: `fallback_cell_frac` docstring claimed "value mass" but computed
  an UNWEIGHTED cell fraction → added visit-weighted `fallback_mass_frac`; made the Thm-4.2 rate-void caveat
  explicit for the no-split deviation. Smoke test (N=3k): cell_frac 0.68 vs **mass_frac 0.32** — deconfoundable
  cells are the high-traffic ones, a more favorable boundary picture than the cell fraction implied.
- **ellipsoid_opt.py — 1 MEDIUM + 1 low + 1 nit.** MEDIUM (valid, already honestly reflected): V_low is the
  min over coordinate-descent endpoints, which UPPER-bounds the exact inner min → not an algorithmically
  *certified* lower bound on V(π); the code docstring + proposal challenges section already framed this as an
  *empirical* check. Hardened: added GUARANTEE-SCOPE docstring block; converted the unchecked U-orthonormality
  assumption in `_build_U` into an ASSERT (passes — eigh vectors; confirmed by V4:projected completing).
- **proposal.tex — 2 number-bracket errors + 1 code/proposal drift + phrasing**, all fixed: "7–22%"→
  "7.8–21.5%" (raw range); errD rate "N^-0.40"→"N^-0.39" with OLS method stated (2-pt-endpoint was the only
  source of -0.40); "N1≫N2"→"2.3:1, consistent with N1≥N2"; **removed a false "Woodbury correction" claim**
  (code uses a dense Cholesky solve; off-span handled by the signal projection); (iv) scoped to "toy";
  "catastrophically loose"→"loose enough to be uninformative" + true-value anchor; (vi) reworded qualitative +
  backed with numbers; superiority puffery + "first" both hedged "to our knowledge".
- **Re-validation: run_phase34 GREEN** (V1 4.4e-16, V2 12/12, V4 both PASS); proposal recompiled to **2 pages**.
- Confidence: HIGH. Two of three math files independently confirmed correct with concrete numerical tests; the
  third's one substantive finding is a framing clarification already honestly reflected. Remaining known
  limitations (coordinate-descent non-global min; no-split finite-sample dependence; benchmark-scale variance)
  are all documented, not hidden.

## [2026-07-09] Entry 9 — 4-LENS ADVERSARIAL AUDIT (theory / software / empirical / peer-review)
Deepest round yet. Every high finding independently RE-VERIFIED by direct computation before acting.
**TWO major reframes — the artifacts are now substantially more honest:**
- **Projected pessimism is a HEURISTIC, not a sound repair (Agent A, high, verified).** The signal-projected
  region structurally excludes the true bridge when it leaks out of the subspace; on the real toy the true
  bridges leak up to **21%** of their norm (bR_t3: 0.38/1.80). So the Entry-5 headline "projected achieves
  coverage=1 at c=0.03" was an artifact of checking only the truth's *projection*. With honest leak-gating,
  **projected joint coverage = 0 at all widths**. Reframed everywhere as a **coverage/informativeness tension**:
  vanilla covers the truth (c≥0.1) but V_low blows up and mis-selects (regret 0.68); projected stays informative
  and selects correctly (regret 0 at c=0.03) but never covers — V_low ≤ V_true holds empirically by
  sign-alignment, not containment. Fixed: `coverage_report` leak-gating + `max_subspace_leakage`; docstring
  soundness caveat; proposal PoC(v) rewritten; fig4 relabeled. Distinct from Entry-8's coordinate-descent point.
- **Benchmark result is more negative than disclosed (Agent C, high, verified).** Plug-in is **worse than naive
  in 15/16 cells** (14/16 vs empirical-naive), worst under **κ=0** (23–46× the naive bias, where there's nothing
  to correct); plug-in bias is non-monotone in N. Entry-6/8's "variance merely exceeds the small bias" understated
  it. Reframed proposal (vi) as an explicit negative result; research question now poses "characterize where it
  helps"; new **fig5_bigenv_failure.png** makes the failure visible. Entry-8's "mass_frac ⇒ more favorable" was a
  red herring (Agent C, is_new=false) — reconciled/dropped.
**Peer-review (Agent D):** novelty reframed from "first by any method" (depended on unread DPC-MBRL) to a
  **capability claim**; the per-block reduction stated as the **standard ellipsoid support-function** identity
  (novelty relocated to the empirical diagnosis); toy naive-bias number corrected "+0.13–0.32 (7.8–21.5%)" →
  **"+0.14–0.33 (8.1–22.0%)"** (Entry-8's was wrong on both ends).
**Software (Agent B):** `fallback_mass_frac` was computed-but-never-read → plumbed into results; feasible-range
  assert added to the big-env chain; eigengap empty-ratios crash-guarded (did NOT add the k_floor change that
  would alter real results); per-seed std persisted (C-med).
**Theory (Agent A) confirmations:** primal/dual 2.8e-16, value-plugin FD gradients 3e-10, big-env chain hand-traced
  correct, NaN reachability clear; half-Hessian wording fixed.
**Re-validation:** all suites GREEN (V1 4.4e-16, V2 12/12, V4 both PASS — V4:projected now vacuous, the honest
  state), big-env feasible-range assert holds; figures regenerated; proposal recompiled to 2 pages.
- Net: the pessimism contribution is **weaker but far more defensible** (a verified diagnosis + an honest
  tension, not an overclaimed repair); the de-biasing story is honestly bounded to the toy. No functional bug.

## [2026-07-09] Entry 10 — 3-AGENT UPGRADE ROUND (variance reduction / model-free baseline / reasoning doc)
3 specialist agents PROPOSED; I implemented each backward-compatibly and verified against oracles. Pinned
oracle math untouched; **master regression GREEN after the round** (V1 4.4e-16, 12/12, V4 both PASS).
- **Agent 1 — variance reduction** `src/estimation/shrinked_big_bridge_estimator.py`
  `ShrinkedBigBridgeEstimator(BigBridgeEstimator)`, modes `shrinkage='none'|'pool'|'js'`, `k_folds>=2`
  cross-fitting. Principled: the true M2 factors through a **cell-independent emission**, so pooling per-(t,a)
  toward a group prior is a confounding-aware, low-variance target. **Default byte-identical** (bR diff 0.0).
  **js halves MAE (0.22→0.08) and cuts across-seed std 6× (0.20→0.03) at κ=1** — a real variance win; shrink
  weight rises 0.26→0.46 with N (consistency preserved). Honest boundary: no variant beats naive at benchmark N.
- **Agent 2 — model-free baseline** `src/baselines/model_free_proximal.py` `MinimaxValueBridgeOPE` (Shi 2022,
  chosen over Nair–Jiang for tractability). Value bridge via tabular minimax GMM with O_0 as instrument.
  **Two bugs caught by the oracle**, both fixed: (1) per-stage direct method was off-policy biased →
  backward-recursion fix (max|MF−Vtrue|=0.0076 @200k, consistent); (2) benchmark solve built an 11536² ≈ 2 GB
  matrix (killed the run) → action-block-diagonal Woodbury solve (0.02s/fit).
- **Agent 3 — reasoning doc** `theory/reasoning_model_impact.md`: honest "reasoning proposes, the oracle
  disposes" analysis, with the live MF off-policy bug as a dated case study. No o1/o3 used; no fabricated numbers.
- **Comprehensive sweep** `poc/run_comprehensive_benchmark.py` (5 κ × 3 N × 3 seeds × 4 methods) →
  `results_comprehensive.json` + **fig6_mb_vs_mf.png**. Landscape: **naive dominates all bridge methods** at
  benchmark scale; **MB > MF** (MF is under-identified — binary O_0 can't pin a 1442-dim value bridge, MAE flat
  in N); **js narrows but doesn't close** the model-based gap. Variance, not identification, is the wall — the
  project's through-line holds.

