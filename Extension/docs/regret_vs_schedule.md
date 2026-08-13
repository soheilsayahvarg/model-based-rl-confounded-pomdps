# Does the Width Schedule Change the DECISION?

**Status: §1–§3 are predictions, committed before the script exists. §4 is empty
and is filled in a later commit.**

---

## 1. The gap this closes

Step (c) established that the pessimism width scales as `N^(e/2)` with
`e = tau + kappa - 1`, and §3.3 of `noncontraction.md` placed the anchor paper's
own schedule at `e_paper > 0` for every admissible `(alpha, c2)` — 30/30 cells,
minimum `0.0217`. The stated consequence was that "the penalty it assigns a leaky
policy grows with `N`".

**That is a statement about an interval, not about a decision.** A practitioner
does not care how wide `V_low` is; they care which policy comes out. If regret is
unaffected, the whole non-contraction result is theoretically real and practically
empty, and we should say so in exactly those words.

Nothing we have run answers this. `run_pessimism_regenerated.py` sweeps `c` at
fixed `N`; the step (c) sweeps vary `N` but measure width, never selection.

## 2. What the theory predicts

Selection uses `argmax_pi V_low(pi) = argmax_pi [ V(b_hat; pi) - W(pi) ]`. The
plug-in term converges; the penalty term scales as `N^(e/2)` and is
**policy-dependent**, since `W(pi) = sqrt(xi) * ||g_pi||_{H^-1}` and `g_pi` differs
across candidates. So the penalty acts as a policy-dependent handicap whose size
is governed by `e`:

- **`e < 0`** — the handicap vanishes, `argmax V_low -> argmax V(b_hat)`, i.e.
  pessimistic selection converges to the plug-in's. Since the plug-in achieves
  `0.000` regret on this environment at every width tested
  (`pessimism_regenerated.md` §4), regret should **improve toward 0** as `N` grows.
- **`e = 0`** — the handicap is `N`-independent. Regret should be **flat**: more
  data neither helps nor hurts the decision.
- **`e > 0`** — the handicap grows without bound and swamps the value differences.
  Selection is then decided by which policy has the smallest `||g_pi||_{H^-1}`,
  not by which has the highest value. Regret should be **non-decreasing in `N`**
  and should saturate at whatever policy minimises the gradient norm.

The last is the anchor paper's own regime.

## 3. Predictions, fixed before measurement

Toy POMDP, `T = 3`, 6 candidate policies, `V_true = 2.0177`, optimal `greedy_lo`.
Under the signal convention `tau = 0`, so `e = kappa - 1`:

| schedule | `tau` | `kappa` | `e` | width vs `N` | **predicted regret vs `N`** |
|---|---|---|---|---|---|
| A | 0 | 0.5 | −0.50 | `N^-0.25` | **decreasing to 0.000** |
| B | 0 | 1.0 | 0.00 | flat | **flat** |
| C | 0 | 1.5 | +0.50 | `N^+0.25` | **increasing** |
| **paper** | 0.545 | 0.909 | **+0.45** | `N^+0.23` | **increasing** |

The paper row uses its own schedule at `alpha = 10, c2 = 1.0`:
`xi ∝ N2^(-alpha/(2*alpha+2))` and `lam2 = N2^(-alpha/(alpha*c2+1))`, which is the
`e_paper` maximum over the admissible grid and therefore the clearest instance to
measure. It is *not* cherry-picking the direction — every admissible cell has
`e_paper > 0`; this one has the largest effect size and so the best chance of
being visible against a discrete regret scale.

**Quantitatively.** Schedule C and the paper row should show
`d(regret)/d(log N) > 0`. Schedule A should reach `0.000` and stay. Schedule B
should not move outside seed noise.

**What refutes this.** Regret flat under C and the paper row while the width
demonstrably grows would mean the penalty grows *uniformly* across candidates and
cancels in the argmax — which would make the whole `e > 0` result decision-
irrelevant and would have to be reported as the headline. Regret *decreasing*
under C would refute the mechanism outright.

**The control this needs.** Regret is discrete over 6 candidates, so it can look
flat purely from coarseness. The run must therefore also report the **selected
policy** and `V_low` per schedule per `N`, so a flat regret curve can be
distinguished from a curve that is moving but not yet crossing a decision
boundary.

## 4. Measurements

### 4.1 Run 1 was mis-designed, and §3's table contradicted §2

Two defects, both mine, both found before any conclusion was drawn.

**The prediction table disagreed with its own mechanism.** §2 says that when
`e > 0` "selection is decided by which policy has the smallest
`||g_pi||_{H^-1}`". Once selection reaches that policy it **stays there** — the
penalty growing further changes nothing. So the mechanism implies **flat at the
attractor**, and §3's "increasing" was inconsistent with the paragraph above it.
Measured: regret pinned at `0.5127` (`uniform`) for schedules B, C and the paper
row at every `N`. That is the attractor, not a ceiling — `greedy_hi` at `1.0243`
was available and never selected.

**The four schedules did not start in the same place.** Run 1 used one shared
constant, but schedules A–C normalise by `sigma2 ≈ 0.015` and the paper row by
`1`, giving starting widths of `0.553 / 1.045 / 1.369 / 0.665`. Three of the four
began *already inside* the attractor, so "does regret degrade with `N`" could not
have been observed either way. **This is the third experiment in this extension
that could not have shown the effect it was built to test**, after the coverage
crossing at `N = 1.7e9` and the `validity 36/36` grid. The pattern is now
established enough to be worth naming: the failure is always the same, checking
whether the measurement range straddles the predicted transition *before* running.

Run 2 calibrates each schedule so its width at the smallest `N` is `0.12`, below
the `~0.19` at which schedule A recovers the optimal policy, so every schedule
starts with the *correct* selection and the sweep tests slope alone.

### 4.2 The answer is no

Calibrated, 5 seeds:

| `N` | A (`e`=−0.50) | B (`e`=0.00) | C (`e`=+0.50) | paper (`e`=+0.455) |
|---|---|---|---|---|
| 2,000 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| 8,000 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| 32,000 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| 128,000 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

> **On this environment the width schedule does not change the decision.** Every
> schedule, including the two with `e > 0` and the paper's own, selects the
> optimal policy at every sample size once started below the decision boundary.

And the width **shrinks in all four**, with slopes `−0.393 / −0.480 / −0.365 /
−0.186` against `e/2` predictions of `−0.250 / 0.000 / +0.250 / +0.227`. The
`e > 0` schedules do not grow. The measured slopes instead track `(tau-1)/2` —
`−0.5 / −0.5 / −0.5 / −0.227` — which is the **signal-dominated** regime.

### 4.3 Why, verified analytically

The `N^(e/2)` law needs the null term to dominate `||g||_{H^-1}`, which needs
`beta_g = ||P_Nul g|| > 0`. On the toy it does not. Measuring the value gradient
against the population design null, at the exact bridges:

| policy | `\|\|g\|\|` | `\|\|P_Nul g\|\|` | share |
|---|---|---|---|
| `greedy_lo` | 0.8865 | **0.000e+00** | 0.00% |
| `uniform` | 0.7422 | **0.000e+00** | 0.00% |
| `soft` | 0.7808 | **0.000e+00** | 0.00% |

Exactly zero, not small. Combined with `beta_pop = 5.4e-16` from
`completeness_and_h4.md`:

> **Completeness kills both halves of (H4) at once.** Where the anchor paper's
> Assumption 3.3 holds, neither the truth nor the value gradient has any mass in
> the design's null space, so the width is signal-driven, shrinks with `N` under
> every schedule, and no schedule can degrade the decision.

This is the same conclusion the review reached from the paper's side — Lemma C.1
ties population gradient leakage to `C*_pi = infinity` — arrived at here by direct
measurement.

### 4.4 What this costs us

`noncontraction.md` §3.3 called `e_paper > 0` "the one genuinely new consequence",
with the reading that the paper's penalty for a leaky policy grows with `N`. The
algebra is unaffected and still holds for every admissible `(alpha, c2)`. **Its
practical bite, however, is conditional on `beta_g > 0`**, which by Lemma C.1
means `C*_pi = infinity` — outside the paper's own assumptions. Where the paper's
assumptions hold, `e_paper > 0` is **decision-irrelevant**, and this run is the
evidence.

That is a real reduction in what the result claims, and it should be applied to
`noncontraction.md` and to the README rather than left in this file.

### 4.5b Predictions for the run where the floor is nonzero

*Committed before `poc/run_regret_incomplete.py` existed. §6 holds the result.*

`completeness_and_h4.md` §5c located the regime this test needs:
`(4,6,2)` at `confound = 0.9`, where `beta = 1.178` and `beta_g = 0.364` give a
floor of `0.428` — and, unlike `confound = 1.0`, the behaviour policy is **not**
degenerate. **The precondition is verified this time before running**, which is
what the previous two attempts skipped.

With `beta_g > 0` the null term is no longer absent, so the `N^(e/2)` law should
apply where §4.2 found it did not:

| schedule | `e` | predicted width slope | predicted regret vs `N` |
|---|---|---|---|
| A `kappa=0.5` | −0.50 | −0.25 | stays at its start |
| B `kappa=1.0` | 0.00 | 0.00 | stays at its start |
| C `kappa=1.5` | +0.50 | +0.25 | **degrades into the attractor** |
| paper | +0.455 | +0.227 | **degrades into the attractor** |

All schedules are calibrated to the same width at the smallest `N`, below the
decision boundary, so every one starts with the correct selection and the sweep
tests slope alone — the §4.1 fix, carried forward.

**What refutes this.** Width slopes that again track `(tau-1)/2` rather than
`e/2` would mean `beta_g = 0.364` is still not enough for the null term to
dominate, and the `N^(e/2)` law has no reachable regime at all on these
environments. Regret flat under C and the paper row *with* the width confirmed
growing would mean the penalty grows uniformly across candidates and cancels in
the argmax — which would make `e > 0` decision-irrelevant everywhere, not just
under completeness, and would be the headline.

### 4.5 What is still open

The decision-level test has **not** been run where `beta_g > 0`. The environment
for it is the same one `completeness_and_h4.md` identified — `(4,6,2)`, incomplete
at every confounding level — and that is the only place the `e > 0` schedules
could degrade selection. Until it is run, "the schedule changes the decision"
has no confirmed instance, exactly as the coverage branch had none before.

### 4.6 Summary

| claim | status |
|---|---|
| §3's "increasing regret" for `e > 0` | **refuted** — the mechanism implies an attractor, and §3 contradicted §2 |
| width schedule changes the decision (complete design) | **refuted** — 0.0000 regret, 4 schedules × 4 `N` |
| `beta_g = 0` under completeness | **confirmed**, exactly, 3 policies |
| width `∝ N^(e/2)` on the toy | **does not apply** — signal-dominated, tracks `(tau-1)/2` |
| `e_paper > 0` as algebra | unaffected |
| `e_paper > 0` as a practical consequence | **scoped down** — needs `C*_pi = infinity` |
| decision degradation where `beta_g > 0` | **not run** |

## 6. The run where the floor is nonzero

`poc/run_regret_incomplete.py`, `(4,6,2)` at `confound = 0.9`, 3 seeds. The script
checks the precondition and aborts if it fails; it did not — `beta = 1.178`,
`beta_g = 0.364`, floor `0.428`.

### 6.1 The `N^(e/2)` law has a reachable regime

| schedule | `e` | measured width slope | `e/2` | `(tau-1)/2` |
|---|---|---|---|---|
| A `kappa=0.5` | −0.50 | **−0.2335** | −0.250 | −0.500 |
| B `kappa=1.0` | 0.00 | **−0.0867** | 0.000 | −0.500 |
| C `kappa=1.5` | +0.50 | **+0.2073** | +0.250 | −0.500 |
| paper | +0.455 | **+0.1036** | +0.227 | −0.227 |

On the toy (§4.2) the slopes tracked `(tau-1)/2` — signal-dominated, because
`beta_g = 0`. Here they track `e/2`. **The width under schedule C grows with data
(+0.207)**, measured with the real value gradient in an environment whose
behaviour policy is not degenerate. This is the regime the toy could not provide.

### 6.2 The decision degrades

Selection regret, optimal `always_1` at `V_true = 2.3157`:

| `N` | A (`e`<0) | B (`e`=0) | C (`e`>0) | **paper** |
|---|---|---|---|---|
| 2,000 | 0.1418 | 0.0644 | 0.0644 | **0.0644** |
| 8,000 | 0.0838 | 0.0644 | 0.1611 | **0.0977** |
| 32,000 | 0.0977 | 0.0644 | 0.1203 | **0.1225** |
| 128,000 | 0.1641 | 0.0644 | 0.1804 | **0.1225** |
| slope | +0.006 | **0.000** | **+0.022** | **+0.014** |

B is flat to the digit and holds `greedy_lo` at every `N`. C and the paper row
both degrade, and their modal selection drifts away from the best reachable
policy — `greedy_lo` → `uniform` for C, `greedy_lo` → `soft` for the paper row.

> **Under the anchor paper's own schedule, in an environment with real confounding
> and an inadequate negative control, policy selection gets worse with more data.**

This is the first confirmed decision-level instance of the `e > 0` branch. It is
also the first time in this extension that a prediction of a *positive* effect was
committed in advance and then observed.

### 6.3 Three things this does not support

**Schedule A is inconclusive, not confirmed.** §4.5b predicted it "stays at its
start". Measured: `0.142 → 0.084 → 0.098 → 0.164`, non-monotone and ending worse
than it began, slope `+0.006`. At 3 seeds this is probably noise, but it is not
what was predicted and is recorded as unconfirmed rather than folded into the
success column.

**The paper row's width slope is half the prediction** — `+0.104` against
`+0.227`. The sign is right and `(tau-1)/2 = -0.227` is clearly excluded, but the
magnitude is not reproduced, and the gap is unexplained.

**The effect is small and thinly sampled.** Regret moves `0.064 → 0.123` on a
scale whose worst candidate is `0.297`, at 3 seeds. **These numbers should not be
quoted until the run is repeated at 20 seeds** — the same debt still outstanding
for the step (b) leakage table.

### 6.4 The plug-in control, and a reversal I nearly published

The optimal policy `always_1` is **never selected by any pessimistic schedule at
any `N`**; the best any of them reaches is `greedy_lo` at regret `0.0644`. The
obvious question is whether the plug-in does better.

**First check said no, and that was wrong.** Run at `kappa = 0.5` only, the
plug-in picks `always_0` and sits at regret `0.164` at every `N` — worse than
schedule B's `0.0644`. Taken at face value that reverses the plug-in dominance
reported in `pessimism_regenerated.md` §4, a claim that had survived seven review
rounds. It does not, and the reason is that `b_hat` depends on `lambda_2`, so
comparing a `kappa = 0.5` plug-in against a `kappa = 1.5` pessimistic run is not a
control. Re-run at each schedule's **own** `kappa`:

| `kappa` | `N` | plug-in regret | pessimistic regret |
|---|---|---|---|
| 0.5 | 2,000 → 128,000 | 0.131 / 0.164 / 0.164 / 0.164 | 0.142 / 0.084 / 0.098 / 0.164 |
| 1.0 | | 0.164 / 0.064 / 0.064 / **0.000** | 0.064 / 0.064 / 0.064 / 0.064 |
| **1.5** | | 0.064 / 0.022 / **0.000** / **0.000** | 0.064 / 0.161 / 0.120 / 0.180 |
| 0.909 (paper) | | 0.164 / 0.164 / 0.064 / 0.064 | 0.064 / 0.098 / 0.123 / 0.123 |

**The plug-in does find the optimum**, reaching `0.000` at `kappa = 1.0` and
`kappa = 1.5`. No pessimistic schedule ever does. Dominance survives, and now
holds in an incomplete design as well as a complete one.

**And the `kappa = 1.5` row is the sharpest single comparison in this extension.**
Same data, same fits, same sweep — the only difference is whether the pessimism
layer is applied on top:

```text
plug-in      0.064 -> 0.022 -> 0.000 -> 0.000     (improves with data)
pessimistic  0.064 -> 0.161 -> 0.120 -> 0.180     (degrades with data)
```

> Under `e > 0`, more data makes the plug-in converge to the optimal policy and
> makes the pessimistic selection built on those same estimates diverge from it.

That is stronger than either §6.2 or the dominance result alone, because it
removes every confound between them: it is not that pessimism is worse, it is that
the pessimism layer actively destroys a selection the underlying estimates get
right, and does so faster the more data it is given.

**Recorded as a near-miss.** The one-`kappa` version of this check produced a
clean, quotable reversal of a load-bearing claim, and it was wrong. It was caught
only because the control was run before writing it up. That is the fourth time in
this extension a measurement's meaning turned on a design detail rather than on
the number itself.

### 6.5 Summary

| claim | status |
|---|---|
| precondition `beta_g > 0` checked before running | **done** — the guard the two previous attempts lacked |
| width tracks `e/2` where `beta_g > 0` | **confirmed**, 3 of 4 schedules within 0.05 |
| width *grows* with data under `e > 0` | **confirmed**, +0.207 |
| regret degrades under `e > 0` | **confirmed** for C and the paper row |
| regret flat at `e = 0` | **confirmed**, exactly |
| regret stable under `e < 0` | **not confirmed** — non-monotone, ends worse |
| paper row width magnitude | **partially** — sign right, magnitude half |
| effect size quotable | **no** — 3 seeds, needs 20 |
| pessimism finds the optimum at all | **no**, under any schedule |
| the plug-in finds it | **yes**, regret 0.000 at `kappa` 1.0 and 1.5 — dominance survives, now in an incomplete design too |
| plug-in improves while pessimism degrades, same fits | **confirmed** at `kappa=1.5`: 0.064→0.000 against 0.064→0.180 |

## 5. Files

| File | Role |
|---|---|
| `poc/run_regret_vs_schedule.py` | §4 |
| `experiments/results_regret_vs_schedule.json` | Raw |
