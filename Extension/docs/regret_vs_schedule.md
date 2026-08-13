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

*(added after §1–§3 were committed)*

## 5. Files

| File | Role |
|---|---|
| `poc/run_regret_vs_schedule.py` | §4 |
| `experiments/results_regret_vs_schedule.json` | Raw |
