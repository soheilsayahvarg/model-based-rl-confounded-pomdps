# Coverage Sweep: Turning an Observation into a Tested Mechanism

**Status: complete, 5 seeds, prediction confirmed.**

---

## 1. What this tests

The finite-action experiment found pessimistic selection consistently worse than
plain plug-in selection, and a diagnostic suggested the reason: the pessimism
penalty was monotonically inverse to behavior-policy coverage, and the *optimal*
policy happened to be the *least covered* one.

That was four points in a single configuration. Suggestive, not established. It
also had a competing explanation we had already been wrong about once (chain
compounding), so the bar for accepting a second story without testing it should
have been high.

So we turned it into a falsifiable prediction:

> If coverage is the mechanism, then increasing how well the logging policy
> covers the optimal action should shrink pessimistic-selection regret toward
> zero, and should flip the optimal policy from being penalised *more* than the
> alternatives to being penalised *less*.

If that failed, the coverage story would have been wrong and we would have
reported it as such.

## 2. Design

We vary `b_pref`, a dose-preference offset added to the behavior policy's logits:

```
logits_k = (eta * driver + b_pref) * a_k
```

Negative `b_pref` shifts the logging policy toward low doses, which increases
coverage of the optimal (never-treat) policy. Two properties make this a clean
instrument:

- It leaves the **confounding channel** (`eta * driver`, which reads the true
  hidden state) untouched. The identification problem is unchanged.
- It leaves the **environment dynamics** untouched, so every candidate policy's
  true value is invariant. The driver asserts this at every grid point. The
  ranking target is therefore fixed across the whole sweep, and any movement in
  regret is attributable to the logged data alone.

`N=2,000`, 5 seeds, regret averaged over confidence widths `c ∈ {0.1, 0.3, 1.0}`.

## 3. Result: prediction confirmed

| `b_pref` | Coverage of optimal | Pessimistic regret | Plug-in regret | Penalty ratio (best / others) |
|---|---|---|---|---|
| 0.0 | 0.124 | 0.367 ± 0.051 | 0.000 | 1.846 |
| −1.0 | 0.244 | 0.211 ± 0.000 | 0.000 | 1.400 |
| −2.0 | 0.407 | 0.211 ± 0.000 | 0.000 | 1.038 |
| −3.0 | 0.582 | 0.140 ± 0.000 | 0.000 | 0.731 |
| −4.0 | **0.727** | **0.000 ± 0.000** | 0.000 | **0.486** |

- **Correlation(coverage, pessimistic regret) = −0.944.**
- Regret falls from `0.367` at 12% coverage to **exactly `0.000`** at 73%.
- The penalty ratio crosses `1.0` at roughly 40% coverage: below it the optimal
  policy is penalised *more* than the alternatives, above it *less*. That
  crossing is the mechanism made visible.

Plug-in selection is perfect throughout, which is the control: the estimation
problem is not what changes across the sweep.

## 4. What this establishes

Pessimistic selection failure in this setting is **not** a defect of the
optimizer, the estimator, the action space, or the horizon. It is the intended
behavior of pessimism operating in a regime where the logging policy
under-covers the optimal policy. The method is answering a different question
than the one selection asks: *"which policy can I most confidently guarantee?"*
rather than *"which policy is best?"* Those coincide only under adequate
coverage.

This is the partial-coverage trade-off of pessimistic offline RL, exhibited
quantitatively with a knob and a prediction rather than asserted.

## 5. Honest limitations

- **One environment, one action set.** The mechanism is now tested rather than
  merely observed, but it is tested in a single linear-Gaussian POMDP with three
  dose levels.
- **Coverage is varied through one specific channel** (a dose-preference offset).
  Other ways of degrading coverage — fewer actions represented at all, or
  state-dependent coverage gaps — are not tested.
- **Two grid points share a regret value** (`0.211` at 24% and 41% coverage),
  so the decrease is monotone but not strictly so at every step. With four
  candidate policies, regret can only take a few discrete values, which
  coarsens the curve.
- The zero-variance intervals at most grid points reflect that all 5 seeds
  selected the same policy, not unusually precise estimation.

## 6. Files

| File | Role |
|---|---|
| `poc/run_coverage_sweep.py` | Driver |
| `experiments/results_coverage_sweep.json` | Raw results |
| `src/envs/finite_action_env.py` | `b_pref` knob (default `0.0` leaves the base configuration unchanged) |
