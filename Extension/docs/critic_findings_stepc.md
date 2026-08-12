# Critic Findings — Step (c), the Contraction/Coverage Dichotomy

Adversarial review of `docs/noncontraction.md` + `poc/run_noncontraction.py`.
Sixth round. Verdict vocabulary: REFUTED | UNPROVEN | OVERSTATED | SURVIVES.

## Checklist (resume from the first unticked item)

- [x] A0. Read noncontraction.md, run_noncontraction.py, ellipsoid_opt.py,
      mf_pessimism.py, bridge_estimator.py, family_pessimism.md §4/§8
- [x] A1. Reproduce the quoted numbers (rerun run_noncontraction.py verbatim)
- [x] A2. Priority 2c/2a first (they feed Priority 1): exhaustiveness of the
      dichotomy — is there a schedule-free statement that subsumes it? — and
      whether the delta-kernel exactness is load-bearing (smooth-spectrum demo)
- [x] A3. Priority 1b: anchor paper Theorem 4.2 — compute kappa =
      gamma/(gamma*c2+1) ranges, find the paper's own xi rate, decide which
      side of the dichotomy the paper's own schedule sits on
- [x] A4. Priority 1a: steelman the "this is just non-identification" reading,
      with the literature verdict
- [x] A5. Priority 3a: direct test of the §5.3 residual explanation — exact
      finite-lam decomposition W^2 = xi*(A_N + beta_g^2/lam), A_N > 0, slopes
      recomputed from the exact formula vs measured
- [x] A6. Priority 3b: projected slope at 20 seeds
- [x] A7. Priority 3c: is the m_null slope an algebraic identity? (circularity
      audit) + Priority 2d: does sigma2_signal / A_N drift with N?
- [x] A8. Priority 3d: the c floor — test coverage just below and just above
      c = sigma2*C, including whether the refined N2* formula puts a crossing
      INSIDE the reachable grid (which would un-vacuate §4.3)
- [x] A9. Priority 2b: routes by which g2_hat could pick up a null component
      (dual path, uniform y'' draw, unseen-x fallback)
- [x] A10. Priority 4: framing verdict + round summary

## State notes (for resume)

- Files read. Key observations queued for verification:
  (1) The proposition's (iii) e>=0 branch says the coverage requirement is
      "satisfied at every N" — as written this ignores the constant K; for
      e=0 it needs K<=1 and for e>0 it fails for N < K^(1/e). Check.
  (2) There may be a schedule-free bound W >= beta_g * beta that subsumes the
      power-law dichotomy entirely (coverage forces xi >= lam*beta^2; width
      contains sqrt(xi/lam)*beta_g; the lam cancels). If true, §6's claim that
      a truth-dependent mu "would escape" is REFUTED, and the power-law family
      framing is scoped too narrowly.
  (3) m_null = xi/(lam*beta^2) is built from xi and lam (inputs, not
      measurements) and a constant beta — its slope may be forced except for
      the flatness of sigma2_signal. Check what the -0.4965 actually tests.
  (4) The refined N2* formula may put a coverage crossing inside the existing
      grid for c slightly above the floor (approx 0.0111 < c < 0.0118) —
      compute exactly, then run it.
  (5) noncontraction.md has a duplicated "## 7. Files" section (lines 343-355).
  (6) Doc claims "git history is the evidence" for predictions-before-
      measurement. Verify a git repo actually exists and shows that ordering.

---
## A1 — Reproduction: every quoted number reproduces exactly

**What I ran:** `poc/run_noncontraction.py` verbatim (same seeds; deterministic).

All of §5 reproduces digit-for-digit: slopes −0.2526/+0.0040/+0.2402 (global),
−0.4141/−0.2798/−0.0069 (signal), (H1) max rel err 1.248e-12, `P_Nul b_hat`
7.086e-12, m_null slope −0.4965, beta 0.0532, C = 0.6981, floor 0.0111,
N2* = 498,167,859. The git evidence also checks out: commit `53d71a5` contains
§1–§4 complete with the prediction tables and §5 as a placeholder; commit
`c0e696f` touches ONLY §5 (the diff removes two placeholder lines and adds
measurements; not one line of §1–§4 changes). The predict-then-measure claim
**SURVIVES** and is auditable exactly as advertised.

One doc defect: `noncontraction.md` ends with a duplicated "## 7. Files"
section (lines 343–355 repeat verbatim).

---

## A2 — Exhaustiveness: the power-law family is scaffolding; a two-line
## schedule-free bound subsumes the dichotomy, and §6's "escape" claim is wrong

**Claim attacked:** (iii)'s exhaustiveness ("the two cases partition the sign of
a single real number") is only exhaustive WITHIN the power-law family, and §6
asserts the only escape is a `mu` depending on the truth, "which would not be
implementable."

**The subsuming bound.** Under (H1) + `P_Nul b_hat = 0`, for ANY `xi > 0` and
ANY `lam > 0` — power law, logarithmic, data-adaptive, truth-dependent,
anything:

```
coverage  =>  xi >= lam * beta^2          (the proof's own step (ii))
W = sqrt(xi)*||g||_{H^-1} >= sqrt(xi/lam) * beta_g >= beta * beta_g.
```

The `lam` cancels. **Any width rule whatsoever that covers the truth pays at
least `beta * beta_g`, at every N.** The dichotomy (iii) is the one-line
corollary for power laws (contraction to below `beta*beta_g` implies coverage
failure), and (i)+(ii) are the rate-decorated version. Consequences:

1. The exhaustiveness worry is moot — but so is the power-law framing. The
   proposition is stated more narrowly than what its own proof steps give.
2. §6's assertion that a truth-dependent `mu` "would escape the dichotomy" is
   **REFUTED**: no width rule escapes. The actual escape routes are (a) an
   estimator whose CENTER has a null component (prior-informed `b_hat`, so
   `P_Nul(b_true - b_hat)` shrinks — violating the `P_Nul b_hat = 0` premise,
   not the width rule), (b) non-ellipsoidal geometry, or (c) `beta = 0` — and
   (c) is what actually happens on the test environment (see A7).
3. Also a precision defect in (iii) as stated: "`e >= 0`: the requirement is
   satisfied at every N" ignores the constant. `N^e >= K` at every `N >= 1`
   needs `K <= 1`; for `e > 0` it FAILS for all `N < K^(1/e)`, and at `e = 0`
   it is satisfied at every N iff `K <= 1`. True at the implemented constants
   (global convention: `K = lam0^2 * beta^2 / c ~ 1e-6`), false as a general
   statement. And the requirement is NECESSARY, not sufficient — the e >= 0
   branch shows the null obstruction is absent, not that coverage holds (the
   doc's own §5.5 signal term, which dominates xi_needed by 15x, decides
   coverage there; under the signal convention the floor `c > sigma2*C` binds
   at EVERY e, including e >= 0). "Coverage sustainable" overstates.

**Delta-kernel exactness (Priority 2).** `lam_min(H) = lam` and the Theta(1/lam)
null term are exact ONLY because the tabular design has an exact finite-rank
null. `poc/critic_c_smooth.py` runs the classical calculus for a smoothly
decaying spectrum (`s_j ~ j^-b`, gradient mass `g_j^2 ~ j^-a`):

| b | a | predicted theta | measured |
|---|---|---|---|
| 2.0 | 1.2 | 0.900 | 0.865 |
| 2.0 | 1.5 | 0.750 | 0.741 |
| 2.0 | 2.0 | 0.500 | 0.501 |
| 2.0 | 3.5 | 0.000 (source condition) | 0.023 |
| atom at 0 (exact null) | | 1.000 | 0.975 |

`||g||^2_{H^-1} = Theta(lam^-theta)` with `theta = 1 - (a-1)/b < 1` for any
strictly PD kernel; `theta = 1` exactly requires the atom (exact null). So in
general the governing exponent is `e' = tau + theta*kappa - 1`: the trade-off
persists (that is Tikhonov rate theory), but the exact threshold `e = 0` and
every numeric constant in §4 are the delta-kernel specialisation. The
proposition should say so; as written, "any method that regularises a
rank-deficient bridge design ... inherits this dichotomy" (§3) is
**OVERSTATED** — with a PD kernel the same construction inherits a
source-condition-dependent version whose threshold sits elsewhere.

**Verdicts:** exhaustiveness within power laws SURVIVES (trivially); the §6
escape claim REFUTED; the (iii) e>=0 branch as stated OVERSTATED (constant K
ignored, necessary vs sufficient conflated); generality beyond delta kernels
OVERSTATED (theta < 1); and the proposition is simultaneously SCOPED TOO
NARROWLY — the schedule-free `W >= beta*beta_g` is stronger, shorter, and
referee-proof against the "you only checked power laws" objection.

---

## A3 — The anchor paper's own schedule sits STRICTLY on the no-contraction
## side, for every admissible smoothness — the dichotomy explains the paper,
## it does not indict it

**What I checked:** Theorem 4.2 and Lemmas C.2/C.3/D.4/D.5, Assumption D.16,
extracted from the PDF (pp. 8–9, 19, 29–30, 34–36, 40).

**The paper's schedule.** Theorem 4.2 sets `lam2 = N2^(-alpha/(alpha*c2+1))`
(alpha = the bridge-RKHS eigen-decay exponent of Assumption 4.1(e); the repo's
H2 calls it gamma) and — the piece the doc never uses — its confidence width is

```
xi_paper = c * log(T/delta) * M_R * N2^(-alpha/(2*alpha+2)).
```

That is IN the proposition's own family `xi = (c/m) * N^(tau-1)` with
`tau_paper = (alpha+2)/(2*alpha+2)`, so the proposition applies to it directly:

```
kappa_paper = alpha/(alpha*c2+1),
e_paper = kappa + tau - 1 = alpha*(2*alpha + 1 - alpha*c2)
                            / ((alpha*c2+1)*(2*alpha+2)).
```

`e_paper >= 0  <=>  c2 <= 2 + 1/alpha`. Assumption D.16 fixes **c2 in (1, 2]**.
Since `2 < 2 + 1/alpha` for every `alpha > 0`, **`e_paper > 0 strictly, for
every admissible (alpha, c2)`** — e.g. alpha=1, c2=2 gives e = 1/12; the
infimum over the assumption set is 0 (approached as alpha -> inf at c2 = 2) and
is never attained.

**Reading.** The paper's own (lam, xi) pair lands on the `e > 0` branch: its
confidence region never contracts in a leaked null direction — the width there
GROWS like `N^(e/2)` — and what that buys is exactly Lemma C.2, coverage of the
true bridges. The paper never promises the region contracts; it promises
coverage, and its SubOpt bound still decays (`N^(-alpha/(4alpha+4))`) because
Lemma C.1 converts region width into value error through the concentrability
coefficients `C*_t`, i.e. only along covered directions. So:

1. The answer to "is the paper already on the no-contraction side of its own
   theorem?" is **YES, unconditionally** — computable in three lines from
   quantities the doc's own H2 already cites. The doc quotes the paper's kappa
   in (H2) and then never evaluates the paper's own cell in its §4 tables.
   That is the single most consequential omission in `noncontraction.md`.
2. The dichotomy therefore does not contradict the anchor paper — it DERIVES
   why the paper's schedule had to be non-contracting. As a criticism of the
   paper it is empty; as an explanation of the paper's design it is real.
3. The side that fails — `e < 0` — is occupied not by the paper but by the
   REPO's Phase-3 signal-convention rule (`tau = 0`, `kappa = 1/2`,
   `e = -1/2`). Step (c)'s sharpest legitimate conclusion about prior work is
   about the repository's own width rule, not the anchor's.
4. A genuinely new and publishable corollary of the paper-side computation:
   under the paper's own schedule, the pessimism penalty for policies whose
   value gradient leaks into unidentified directions (the uncovered ones)
   GROWS with N — the paper's algorithm becomes monotonically more
   conservative toward uncovered policies as data accumulates. Nothing in the
   paper states this; it follows from the e > 0 placement.

**One more consequence the doc missed (feeds A4).** Lemma C.1 bounds
`|V(pi) - V(pi, b)| <= sum_t C*_t sqrt(L_R(b))` for ALL b in the class.
Population-null directions leave the population risk `L_R` unchanged, so for a
policy with `C*_pi < inf` the value functional CANNOT move along them — i.e.
population gradient leakage `beta_g > 0` implies `C*_pi = inf`. (H4) is
therefore not a mild regularity condition: within the paper's assumption set
it holds only for policies the paper's guarantee already excludes and its
pessimism is designed to penalize. Combined with A7's `beta_pop = 0` under
completeness, BOTH halves of (H4) live outside the anchor's assumptions.

**Verdict:** the highest-value check REFUTES the framing "no schedule gives
both" as a criticism of the anchor method (the paper achieves coverage + a
decaying SubOpt bound by not asking for width contraction), while CONFIRMING
the calculus itself — the paper is a worked example of the e > 0 branch.

---

## A5 — §5.3's post-hoc residual explanation: CONFIRMED by direct
## decomposition, with one honest caveat about the test the author proposed

**What I ran:** `poc/critic_c_decomp.py` — per fit, the exact spectral split
`Q := ||g||^2_{H^-1} = A_N + beta_g^2/lam` with `A_N = ||P_S g||^2_{(H|_S)^-1}`,
`beta_g^2 = ||P_Nul g||^2`; the requested least-squares fit `Q = A + B/lam`
with constant coefficients; and a frozen-coefficient slope reconstruction.

**The identity is exact** (worst rel. error 3e-15 / 6e-14 / 1.6e-11 across the
three kappas), null dim = 12 in every fit. Measured quantities (means over the
grid): beta_g^2 = 0.2450 essentially constant in N (drift slope −0.003);
A_N > 0 always.

**The requested regression, honestly reported.** At kappa=0.5: fitted
B = 0.24569 vs measured beta_g^2 = 0.24505 (0.3% off), A = 80.3 vs measured
mean A_N = 85.3 — the explanation's parameters are recovered. At kappa=0.25:
B = 0.297 vs 0.245 (21% off) because A_N itself drifts upward (+0.086/decade,
45.5 -> 65.3 — the 1/(s_i+lam) weights grow as lam shrinks) and the constant-A
model dumps that drift into B. At kappa=1.0 the fitted **A = −44.8 is
negative** — the letter of my instruction ("if A comes out negative the
explanation is a rationalisation") triggers, but the reason is instrument
failure, not a wrong explanation: at kappa=1 the regressor 1/lam spans 1e4–2.5e6
while A ~ 93, so the intercept is unidentifiable collinear noise. The exact
decomposition, which needs no regression, gives A_N = 91–95 > 0 in every fit.
The regression I was told to run is the WRONG TEST at exactly the kappa where
it superficially fails; the right test is the identity, and it passes.

**Slope reconstruction.** Frozen-coefficient model (A_bar + B_bar/lam(N),
nothing fitted) vs measured W slope:

| kappa | measured | asymptote e/2 | frozen-coef model |
|---|---|---|---|
| 0.25 | −0.4141 | −0.375 | −0.4316 |
| 0.50 | −0.2813 | −0.250 | −0.2824 |
| 1.00 | −0.0079 | 0.000 | −0.0066 |

At kappa = 0.5 and 1.0 the finite-lam story reproduces the residual to
±0.001. At kappa = 0.25 it over-explains by 0.017, and the gap is exactly the
A_N upward drift the doc does not mention (its contribution: +0.086/2 x
mixture weight). sigma2 drift contributes only −0.003, beta_g drift −0.002.

**Note:** part B redraws `g_fixed` from the same rng stream per kappa, so the
three published rows use three different gradients; rerun with ONE g the
residual ordering (−0.039, −0.031, −0.008) persists, so the ordering claim is
robust to the draw — worth one sentence in the doc.

**Verdict: §5.3 SURVIVES.** The explanation is not a rationalisation: the
mechanism is an exact identity, the recovered B matches beta_g^2, and the
frozen-coefficient model lands within 0.002 of the measured slope at two of
three kappas. The doc should add the A_N-drift clause for kappa = 0.25, where
the pure mixture story over-shoots.

---

## A6 — §5.4 projected slope at 20 seeds: the GEOMETRY is confirmed to
## 3 decimal places; the DEPLOYED estimator is refuted, and more seeds make
## the mean worse, not better

**What I ran:** `poc/critic_c_proj20.py` — part A with 20 seeds, reporting
mean, median, and rank==1-conditional slopes plus PA rank distributions.

| kappa | predicted (tau−1)/2 | mean slope | median slope | rank==1 slope |
|---|---|---|---|---|
| 0.25 | −0.375 | −0.3176 | −0.3706 | **−0.3739** |
| 0.50 | −0.250 | −0.0515 | −0.2470 | **−0.2502** |
| 0.75 | −0.125 | **+0.2360** | −0.1222 | **−0.1254** |

Conditional on the PA selector returning the correct rank 1 (95–100% of fits),
the prediction is verified to |err| <= 0.0011 — far beyond what §5.4 hoped.
But the MEAN projected width at kappa = 0.75 GROWS at +0.236, i.e. it inherits
the unprojected `e/2 = +0.25`: in the 3–5% of fits where PA over-selects
(rank 2–3), the retained subspace admits a near-null direction whose
contribution scales like the UNPROJECTED width, and those events dominate the
mean. The per-N standard errors at kappa=0.75 (0.06 -> 4.7) show the
contamination growing with N exactly as the unprojected rate implies.

So §5.4's own remedy ("needs ~20 seeds") is wrong in an instructive way: at 20
seeds the mean moves AWAY from the prediction (kappa=0.75: 3-seed −0.064,
20-seed +0.236). Averaging does not fix a heavy-tail mixture; conditioning or
medians do.

**Verdict:** the (tau−1)/2 prediction **SURVIVES for the geometry** (median /
correct-rank slopes, err <= 0.001 — this round's cleanest confirmation), and
is **REFUTED for the projected estimator as deployed** (mean over PA-selected
ranks): with e > 0 the deployed projected width is asymptotically NOT
`Theta(N^((tau-1)/2))`; it is a mixture whose mean grows at the unprojected
rate with weight P(over-select). §5.4 should report the conditional slope and
state the mixture behaviour; "consistent, not verified" was the right label
for the wrong reason.

---

## A7 — THE ROUND'S HEADLINE: beta is not an environment constant. It is a
## sampling artifact that decays to an exact population zero, and with it the
## coverage branch's quantitative content on this environment collapses

**Claims attacked:** (H4) "the truth has a null component beta = 0.0532 > 0";
§5.5 "the level is a property of the ENVIRONMENT: the toy's true bridge has
2.9% of its norm in the null space — (H4) holds but barely"; the crossing
N* = 1.7e9; "N2* scales as beta^-4: an environment with 10x the null share
crosses at N2* = 50,000"; and the m_null rate test (−0.4965 vs −0.5000).

**What I ran:** `poc/critic_c_beta.py`.

**Result 1 — beta_pop = 0, exactly, by construction.** The population t=1
design profiles `mu(.|a, o0) = sum_s P(s|a,o0) E[s,.]` have rank 2 = |S| per
action, and their per-action null direction coincides with
`perp(range(E^T))` to machine precision (inner product 1.0). The toy's true
bridge is the MIN-NORM solution (`oracle_module` builds it with `pinv(E)`),
which lies in `range(E^T)` slice by slice. Hence

```
beta_pop = ||P_Nul_pop b_true|| = 4.996e-16.
```

This is not a numerical accident. The population design span equals
`range(E^T)` exactly when the O_0-conditional latent posteriors span R^|S| —
i.e. when K0 is invertible — which is the anchor paper's own completeness
condition (Assumption 3.3) and is how the toy was DESIGNED ("K0 invertible",
toy_pomdp.py docstring). **Under completeness, the min-norm true bridge has
zero population null component; (H4)'s beta > 0 requires completeness to
fail.** The one environment used to test the coverage branch is an
environment where the coverage branch is provably vacuous.

**Result 2 — the measured beta is null-space misalignment noise.** Per-fit
beta (part C's own construction, kappa = 0.5): 0.113, 0.132, 0.033, 0.021,
0.013, 0.006 across N = 1k..1M — slope **−0.4544**, i.e. beta ~ N^(-1/2), the
rate of the empirical null's rotation around the population null (stage-1
tables converge at root-N). The doc's "beta = 0.0532" is the GRID-AVERAGE of
this decaying sequence (my recomputed average: 0.0532, matching exactly), and
"2.9% null share" is that average dressed as an environment property.

**Result 3 — the corrected coverage margin IMPROVES with N.** m_null with the
frozen average beta reproduces the doc: slope −0.4965. m_null with the per-fit
beta(N): slope **+0.4123**. The quantity the doc says "degrades at exactly the
predicted exponent" in fact grows: null-direction coverage gets EASIER with
data on this environment, because the truth's offset from the region's null
directions dies faster than the region tightens. Every downstream number
built on constant beta — K = 4.48e-5, N2* = 4.98e8, N* = 1.66e9, the beta^-4
scaling sentence, "an environment with 10x the null share crosses at 50,000"
— describes a threshold that does not exist here at any N.

**The circularity finding (what −0.4965 actually tested).** m_null =
xi/(lam*beta^2) with frozen beta is algebra in the INPUTS: under the signal
convention m_null = (c/(lam0*beta^2)) * N2^(kappa-1) / sigma2(N), so its
log-slope is identically `e − slope(sigma2)`. The only measured quantity in it
is sigma2's flatness; part C's own sigma2 column has slope −0.0035, and
−0.5 − (−0.0035) = −0.4965. The "rate confirmation" is forced by construction
up to sigma2 drift — I flagged this before finding the beta decay, and the
beta decay then shows the frozen constant is also the WRONG constant. §5.5's
falsifiable content reduces to "sigma2_signal is flat," which is the same
content as the A_N = Theta(1) check. **The m_null table is not evidence about
coverage.**

**What survives.** The proof's conditional statement (ii) is correct algebra:
IF beta were a positive constant, coverage would require N^e >= K. The width
branch (i) is untouched — it uses beta_g (the GRADIENT's null mass), which is
measured stable (0.245 across the grid, drift −0.003; see A5), and gradient
leakage is a property of the value functional, not of the truth. And on
environments where completeness genuinely fails — the (2,6,4) config at
confound = 1.0, where step (a) proved the population rank collapse — every
valid bridge plausibly has a nonvanishing null component and beta_pop > 0 is
real. But that is exactly the environment the doc did NOT run part C on.

**Verdicts:** (H4)-for-b_true on the toy **REFUTED** (beta_pop = 5e-16; "holds
but barely" is a measurement error — it fails); §5.5's rate test **REFUTED as
evidence** (algebraically forced, and its constant is a decaying artifact);
the crossing threshold N*, the beta^-4 scaling sentence, and "the threshold is
a property of the environment" **REFUTED on this environment** (the measured
threshold is a property of the SAMPLE SIZE); proposition (ii)-(iii)
**UNPROVEN empirically** — no environment in the repository currently
instantiates the coverage branch; the candidate that could (dim-separated,
confound = 1.0) is untested for coverage. A_N = Theta(1) SURVIVES (A5 table:
smallest retained DESIGN eigenvalue flat at ~2.5e-3; sigma2 flat at ~1.55e-2;
null dim constant at 12).

---

## A8 — The c floor: qualitatively real, quantitatively not a threshold; and
## the "untestable" crossing WAS testable — it just fails when tested

**Claim attacked:** §5.5's post-hoc refinement `N2* = ((c/sigma2 - C)/
(lam0*beta^2))^2, valid when c > sigma2*C = 0.0111`, "below it the region
fails to cover at EVERY N"; and §4.3/§5.6's "coverage crossing: not tested —
grid 1,600x short."

**What I ran:** `poc/critic_c_floor.py` — 18 fits (N = 1k..1M, 3 seeds, fits
are c-independent), floor and crossings PRE-REGISTERED from the N=1,000 fits
only (floor_hat = 0.01001), then coverage evaluated on a c-grid spanning it.

**The overlooked test.** The doc declares the crossing untestable because at
c = 0.03 it sits at N* = 1.7e9. But N2* is steered by c: for c close to the
floor the predicted crossing lands INSIDE the existing grid (c = 0.01047 ->
first fail at N = 64,000; c = 0.01093 -> 256,000; c = 0.01185 -> 1,024,000).
§5.5 derived the formula that un-vacuates its own §4.3 test and did not
notice. Measured coverage (fraction of 3 seeds, by N):

| c | predicted first-fail N | 1k | 4k | 16k | 64k | 256k | 1M |
|---|---|---|---|---|---|---|---|
| 0.00701 (0.7x floor) | ALL N | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.00981 (0.98x floor) | ALL N | 0.33 | 0.67 | **1.00** | 0.00 | 0.67 | 0.00 |
| 0.01021 | 4,000 | 0.67 | 0.67 | **1.00** | 0.00 | 0.67 | 0.33 |
| 0.01047 | 64,000 | 1.00 | 0.67 | 1.00 | 0.33 | 0.67 | 0.33 |
| 0.01093 | 256,000 | 1.00 | 0.67 | 1.00 | 0.33 | 0.67 | 0.67 |
| 0.01185 | 1,024,000 | 1.00 | 0.67 | 1.00 | 0.67 | 0.67 | 0.67 |
| 0.03 | beyond grid | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

There is no crossing structure: coverage near the floor is non-monotone in N
in every row (1.00 at N=16k sitting between 0.67 and 0.00), and just BELOW the
floor — where the refinement says coverage fails at EVERY N — it holds 3/3 at
N=16,000. The mechanism is in the per-fit table: xi_needed is dominated by its
signal term C_sig, and C_sig scatters 0.46–0.90 ACROSS SEEDS (per-N means
0.51–0.74, sd ~0.08–0.11). The refinement freezes C at one number (0.698,
measured at the largest N); near the floor the coverage indicator is a coin
flip on C_sig's seed noise, not a threshold. The floor survives only as an
order-of-magnitude marker: at 0.7x floor coverage is 1/18 fits; at 3x floor
(c = 0.03) it is 18/18.

**Verdicts:** "below the floor the region fails to cover at EVERY N"
**REFUTED** (3/3 coverage measured below the floor); the refined N2* formula's
crossing predictions **REFUTED** (no monotone failure pattern at any tested
c); the floor as an order-of-magnitude statement SURVIVES; and §5.6's
"coverage crossing: not tested — grid 1,600x short" is itself **OVERSTATED**
— the crossing is testable on the existing grid by moving c, and fails. Note
the doc's honesty labels ("post-hoc, untested") are accurate; what this round
adds is that when tested, the refinement does not survive. Since A7 shows beta
is an artifact anyway, the right fix is not to patch the formula but to drop
the quantitative coverage story for this environment entirely.

---

## A4 — The steelman: "this is non-identification with extra steps." Verdict:
## mostly right about the mathematics, wrong about two specific pieces

**The strongest form of the hostile reading.** Tikhonov regularisation of an
operator with a null space converges to the minimum-norm solution; components
of the truth in the null space are unrecoverable at any sample size; that is
the definition of non-identification. A confidence region centred at a
Tikhonov estimate covers the truth only if its null-direction half-width,
`sqrt(xi/lam)`, exceeds the truth's null offset `beta` — so a covering region
pays `beta * |g_null|` in any linear functional forever. That is partial
identification: the penalty converges to the identified-set width, not zero —
the standard picture in pessimistic offline RL under partial coverage
(Uehara & Sun 2021's model-based partial-coverage analysis; Rashidinejad et
al. 2021's lower bounds) and in ill-posed inverse problems (no rate without a
source condition — Caponnetto & De Vito 2007, the anchor's own citation;
Carrasco–Florens–Renault 2007 for the econometrics form; Chen & Reiss 2011
for NPIV ill-posedness measures). [Citations from memory — verify before
quoting them in any manuscript.] The "coupling" of width and coverage through
one exponent is the bias–variance coupling of ridge regression: `lam` enters
the variance-like factor as `1/lam` and the bias-like factor as `lam`; their
product being schedule-free is exactly why the schedule-free bound in A2 is
two lines. On this reading the proposition renames known facts.

**Where the steelman is simply right.**
- The general trade-off IS in the source-condition literature. The smooth-
  spectrum demo (A2) recovers the classical exponent `theta = 1 - (a-1)/b`;
  the proposition is its delta-kernel endpoint (`theta = 1`). Step (c) as
  "a new dichotomy theorem" will not survive a referee who knows Caponnetto–
  De Vito. It should be presented as a specialisation, with the classical
  literature cited.
- Decisively: **the anchor paper's Assumption D.16(a) is a source condition
  that already forces `beta_pop = 0`.** It requires
  `b*_R = T2^((c2-1)/2) G2` with `c2 > 1`, i.e. the true bridge lies in the
  range of a positive power of the stage-2 covariance operator — hence
  orthogonal to `null(T2)` BY ASSUMPTION. Within the paper's assumption set
  the coverage branch of the dichotomy cannot arise, not because of a
  schedule choice but because the truth is assumed to have no null component.
  Together with A3 (gradient leakage implies infinite concentrability) and A7
  (the toy satisfies completeness, so its min-norm bridge measures
  `beta_pop = 5e-16`), all of (H4) sits outside the paper's assumptions, and
  the doc's §3 sentence "any method that (a) regularises a rank-deficient
  bridge design and (b) calibrates against its spectrum inherits this
  dichotomy, whatever the estimator" is **OVERSTATED**: the anchor method,
  under its stated assumptions, does not inherit the coverage horn.

**What survives the steelman.** Two things, and they are worth keeping:
1. **The placement computation (A3).** "Where does the anchor's own
   (lam, xi) pair sit in (kappa, tau) space, and what does that imply for its
   penalty in leaked directions" — `e_paper > 0` for every admissible
   (alpha, c2), so the paper's penalty for leaky/uncovered policies GROWS
   with N — is not in the paper, not in the source-condition literature (it
   is about the paper's specific pairing of schedules), and is checkable.
   This is the novel content of step (c).
2. **The self-calibration pathology.** The `kappa = 0.75` global-convention
   row — more data makes the bound strictly worse at rate N^(+1/4), verified
   +0.2402 — is a statement about width rules of the form
   `xi = c/(N * mu(design))` interacting with the ridge, i.e. about how
   PRACTITIONERS (including Phase 3) calibrate, not about the theory. As an
   implementation-audit finding it is new and useful; nothing in the standard
   theory predicts a specific code base's width rule grows its own penalty.

**Verdict on Priority 1:** the dichotomy as theory is a RESTATEMENT
(specialisation of source-condition facts, and the paper's D.16(a) already
implies the coverage horn is vacuous under its assumptions); the placement of
the paper's schedule at `e > 0` and the self-calibrated width-rule pathology
are NOVEL and survive. Step (c) should be rewritten around those two, with
the classical literature acknowledged, or a referee will do it for you.

---

## A9 — Routes into the null space for g2_hat: none found; the mechanism of
## (ii) survives code audit

**Checked against `bridge_estimator._stage2_solve` line by line:**
- Primal path (used by every run here: `fit_toy` passes `mode="primal"`):
  `g2 = Psi^T p_vec / N2` is in `range(Psi^T) = Nul(T2)^perp` by construction,
  whatever `p_vec` contains — the unseen-x marginal fallback and stage-1 ridge
  shrinkage change p_vec's VALUES, never the subspace. `H = T2 + lam I` is
  block-diagonal wrt the split, so `b_hat = H^-1 g2` stays in `Nul^perp` up to
  solve roundoff. Measured 7.09e-12, consistent with `1/lam` amplification of
  1e-16-level solve error at `lam ~ 5e-5`.
- Dual path (would engage at `mode="auto"` with N2 <= 3000):
  `b_dual = sum_n alpha_n psi_n` is in `range(Psi^T)` even more directly, and
  the primal–dual gap is logged. No route.
- The uniform y'' draw shapes WHICH coordinates Psi touches, i.e. it shapes
  `Nul` itself — at small N2 an undrawn y-slice joins the empirical null. It
  cannot put mass INTO the null of the same Psi. Side effect worth noting: an
  undrawn y-slice has `b_hat = 0` there while `b_true != 0`, which inflates
  the measured beta at small N2 — a second contributor to A7's artifact
  (negligible on this grid: n_y = 6, N2 >= 300).

**Verdict: `P_Nul b_hat = 0` SURVIVES** — it is structural in both solver
paths; the only failure route is linear-solve roundoff, and the measured
7e-12 is exactly that.

---

## A10 — Priority 4: which framing survives referee contact

The regenerated table's fact pattern: plug-in selection achieves 0.000 regret
at every width on this environment and is never beaten by any pessimistic
variant; projection's only genuine win is mid-grid (c = 0.1–0.3). After this
round, the two proposed framings look like this:

- **"A sharper theorem about why pessimism fails."** Weakened by this round:
  the theorem's coverage horn is uninstantiated on the only environment
  tested (A7), excluded by the anchor's own assumptions (A4/D.16a), and the
  width horn is classical rate theory in the smooth case (A2). What is left
  of "the theorem" that is genuinely yours is the paper-placement corollary
  and the self-calibration pathology. Not enough to carry a paper alone.
- **"Proximal pessimism is a cost, not a benefit, in this regime."** True on
  the evidence, but the evidence is one 3-observation toy and one (2,6,4)
  synthetic family, T = 3, tabular, where the estimator's negative bias makes
  validity free (step (b) §8.3) and plug-in ranking is preserved through
  vacuous widths. A referee will call the environments too easy to indict
  pessimism and will be right.

**Both are too thin alone. The package that survives is the audit:** (1) a
faithful tabular instantiation of the anchor method with every implementation
defect found and fixed across six adversarial rounds; (2) the derived
placement `e_paper > 0` — the paper's construction never promises contracting
regions, and its penalty for leaky policies grows with N (new); (3) practical
width rules that DO contract (the repo's own signal convention) provably pay
with coverage, with the schedule-free floor `W >= beta * beta_g` in genuinely
incomplete designs (confound = 1.0) — the honest scope of the "dichotomy";
(4) empirically, on every tested operating point of these environments,
pessimistic selection never beats plug-in — reported as an environment-scoped
finding with the "why these environments are easy" mechanism stated, not as a
general indictment. That is a reproducibility/audit contribution; it is
coherent, defensible, and none of its four legs was broken by six rounds of
attack. Attempting to dress leg (3) up as a general theorem is the one move
this round shows a referee can break.

---

## Step (c) Round — summary

| claim | verdict | one line |
|---|---|---|
| Predict-then-measure provenance (git) | **SURVIVES** | Commit 53d71a5 has §1–§4 + empty §5; the second commit touches only §5. |
| (i) width slope = e/2, both conventions | **SURVIVES** | Reproduced exactly; §5.3 residuals fully explained by the exact finite-lam decomposition (A5). |
| "More data makes the bound worse" (e > 0) | **SURVIVES** | +0.2402 vs +0.25; and it is the branch the anchor paper itself occupies. |
| `lam_min(H) = lam`, `P_Nul b_hat = 0` | **SURVIVES** | Structural in both solver paths (A9); 1.2e-12 / 7.1e-12. |
| A_N = Theta(1) | **SURVIVES** | Smallest retained design eigenvalue flat (~2.5e-3); null dim constant 12; A_N drifts up toward its limit at small kappa (add a clause). |
| (ii)+(iii) coverage horn, as instantiated | **REFUTED on this environment** | beta_pop = 5.0e-16; measured beta is N^(-1/2) null-misalignment noise (slope −0.45); the corrected margin IMPROVES with N (slope +0.41). No tested environment instantiates the horn. |
| (H4) "holds but barely" on the toy | **REFUTED** | It fails: completeness (K0 invertible) forces the min-norm bridge out of the population null. 2.9% is a grid-average of a decaying artifact. |
| §5.5 m_null rate "confirmed −0.4965" | **REFUTED as evidence** | Algebraically forced (= e − slope(sigma2)); its constant is the artifact above. |
| §5.5 floor c > sigma2*C | **OVERSTATED** | Order-of-magnitude marker survives; "fails at EVERY N below it" is false (3/3 coverage measured below the floor); refined N2* crossings do not appear (A8). |
| §4.3 crossing "not testable, grid 1600x short" | **OVERSTATED** | Testable on the existing grid by moving c toward the floor; tested; the refinement fails. |
| §5.4 projected slope (tau−1)/2 | **SURVIVES (geometry) / REFUTED (deployed)** | Rank-conditional slopes match to 0.001; the mean over PA-selected ranks grows at the unprojected rate when e > 0 (A6). |
| (iii) exhaustiveness / §6 escape bullet | **OVERSTATED / REFUTED** | Schedule-free `W >= beta*beta_g` subsumes the power-law family; a truth-dependent mu does NOT escape; the e>=0 branch ignores the constant K (A2). |
| "Any method that regularises ... inherits this dichotomy" | **OVERSTATED** | Delta-kernel-specific (theta < 1 for PD kernels); and the anchor's D.16(a)+C.1 exclude both halves of (H4) (A2, A3, A4). |
| Novelty of the dichotomy | **RESTATEMENT, with two novel corollaries** | Source-condition literature + partial-identification; what is new: e_paper > 0 for all admissible (alpha, c2), and the self-calibrated width-rule pathology (A4). |

**Over-correction check (the standing rule):** none found — this round moved
in the opposite direction. §5.4's "consistent, not verified" under-claimed a
result that is actually verified to 0.001 (conditionally); §4.3's "not
tested" mislabeled a test that could have been run (and fails). The doc's
honesty labels were accurate about what was and wasn't done; the errors are
in what the measurements were said to MEAN (beta as environment constant,
m_null as coverage evidence).

**What noncontraction.md needs:**
1. Lead with the schedule-free floor `W >= beta * beta_g` (coverage implies
   `xi >= lam*beta^2`; the lam cancels); make the power-law dichotomy its
   corollary. Fix (iii)'s constant-level statement (K <= 1 at e = 0; first-N
   threshold at e > 0) and the necessary-vs-sufficient conflation.
2. Add the anchor-placement cell: `e_paper = alpha(2alpha+1-alpha*c2) /
   ((alpha*c2+1)(2alpha+2)) > 0` for c2 in (1,2] — and reframe §1: the paper
   does not try to contract; D.16(a) additionally assumes the truth out of
   the null. The dichotomy explains the paper's schedule; it does not break it.
3. Withdraw the toy's quantitative coverage content: N* = 1.7e9, K, the
   beta^-4 sentence, the m_null table as evidence, the sharp floor. State
   beta_pop = 0 under completeness and that beta(N) ~ N^(-1/2) is
   misalignment noise. If the coverage horn is to be kept as an empirical
   claim, instantiate it where completeness fails (dim-separated,
   confound = 1.0) and run part C there.
4. §5.3: add the A_N-drift clause at kappa = 0.25; note the per-kappa g
   redraw (ordering robust to it — verified with a single g).
5. §5.4: replace with rank-conditional/median slopes (verified to 0.001) plus
   the mixture statement for the deployed mean.
6. Scope §3's "whatever the estimator" to exact-null designs, citing the
   source-condition generalisation `e' = tau + theta*kappa - 1`.
7. Delete the duplicated "## 7. Files"; fix §6's escape bullet per A2.

**Critic scripts this round:** `poc/critic_c_decomp.py`, `poc/critic_c_floor.py`,
`poc/critic_c_proj20.py`, `poc/critic_c_smooth.py`, `poc/critic_c_beta.py`.
Anchor-paper extracts in the session scratchpad; page references in A3/A4.
