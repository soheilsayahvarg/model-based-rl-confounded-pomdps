# How Good Does the Negative Control Have to Be?

**Status: complete, 5 seeds, two failure modes swept separately.**

---

## 1. Why this is worth doing

Every method in the proximal causal inference line — including the anchor paper —
assumes a valid negative control satisfying Assumption 3.1, and proceeds from
there. No theory paper can say what happens when that assumption is only
approximately true, because the assumption is precisely what the theory is
conditioned on.

A simulator where the negative control's quality is a knob *can* say. This is a
contribution an implementation can make and a proof cannot, and it directly
addresses the most obvious objection to the rest of this project: that our
benchmark constructs `O_0` so that Assumption 3.1 holds *by construction*.

## 2. Two failure modes, deliberately not conflated

They are different questions and behave differently:

**[N1] Instrument STRENGTH (`sigma_0`).** `O_0 = s_1 + eps_0` with
`eps_0 ~ N(0, sigma_0²)`. Assumption 3.1 holds **exactly** for every `sigma_0` —
fresh independent noise on the initial state cannot create dependence on later
outcomes given the state. What degrades is informativeness: as `sigma_0` grows,
`O_0` carries less signal about `s_1`, stage 1's conditional mean embedding
weakens, and the two-stage solve inherits it. This is a weak-instrument study.

**[N2] Assumption VALIDITY (`delta_nc`).** The reward gains a term
`r_t += delta_nc * eps_0`. Now `O_0` carries information about `R_t` beyond
`(S_t, A_t, H_{t-1})`, so **Assumption 3.1 is false**. Because `E[eps_0] = 0` the
mean reward is unchanged, so the exact oracle stays valid and the damage to
identification is measurable against it. The driver asserts the invariance of
every candidate's true value at each grid point.

## 3. A metric correction we had to make first

Our first version of this sweep reported only mean absolute error and concluded
the plug-in was worse everywhere — which contradicted the finite-action
experiment, where the plug-in has *smaller* bias. The two disagree because they
measure different things:

```
systematic bias := mean over policies of | mean over seeds of (V_hat - V) |
total error     := mean over policies and seeds of | V_hat - V |
```

The plug-in's purpose is to remove bias; it pays in variance. On the first metric
it wins, on the second it loses. Reporting only the second would have inverted
the conclusion. **Both are now reported at every grid point.** (The base
configuration: systematic bias `0.029` plug-in vs `0.043` naive; total error
`0.067` plug-in vs `0.049` naive.)

## 4. Results

### [N1] Strength — Assumption 3.1 holds throughout

| `sigma_0` | Recovery err | Sys. bias plug-in | Sys. bias naive | Total err plug-in | Total err naive |
|---|---|---|---|---|---|
| 0.10 | 0.074 | **0.027** | 0.043 | 0.061 | 0.049 |
| 0.25 (base) | 0.083 | **0.029** | 0.043 | 0.067 | 0.049 |
| 0.50 | 0.113 | **0.033** | 0.043 | 0.076 | 0.049 |
| 1.00 | 0.191 | 0.046 | 0.043 | 0.084 | 0.049 |
| 2.00 | 0.423 | 0.068 | 0.043 | 0.102 | 0.049 |

Naive is flat at `0.043` throughout, as it must be — it never touches `O_0`. That
makes it a clean control: everything that moves is attributable to the negative
control's quality.

**Threshold: the plug-in's bias advantage survives to `sigma_0 ≈ 1.0`.** For
reference the observation noise is `sigma_o = 0.35` and the initial-state spread
is `sigma_1 = 0.40`, so the negative control can be roughly **three times noisier
than the observation channel** before it stops helping.

### [N2] Validity — `delta_nc > 0` breaks Assumption 3.1

| `delta_nc` | Recovery err | Sys. bias plug-in | Sys. bias naive | Total err plug-in | Total err naive |
|---|---|---|---|---|---|
| 0.00 | 0.083 | **0.029** | 0.043 | 0.067 | 0.049 |
| 0.25 | 0.190 | **0.032** | 0.041 | 0.067 | 0.047 |
| 0.50 | 0.378 | 0.039 | 0.038 | 0.068 | 0.047 |
| 1.00 | 0.767 | 0.055 | 0.034 | 0.071 | 0.046 |

**Threshold: the advantage is gone by `delta_nc = 0.5`** — a leakage coefficient
comparable to the true reward coefficients themselves (`beta_s = 0.60`,
`beta_a = -0.40`).

## 5. Three findings

**1. Violating the assumption is worse than weakening the instrument.** Bridge
recovery degrades `9.23×` across the validity sweep against `5.72×` across the
strength sweep, and the plug-in's advantage is lost sooner in relative terms.
This matches intuition — a weak instrument costs efficiency, an invalid one costs
identification — but it is now quantified rather than asserted.

**2. Bridge recovery error is a poor proxy for policy-value bias.** At
`delta_nc = 1.0` the reward-bridge recovery error is `9×` its baseline while the
plug-in's systematic value bias grows only `1.9×`. The value chain is
substantially more robust than the bridge estimates feeding it. Practically: you
cannot diagnose the quality of your value estimate by inspecting how well your
bridges fit, in either direction.

**3. Total error favours naive everywhere.** Across both sweeps, at every grid
point, including the best-case negative control. This is the project's central
finding reproduced under yet another lens: the bias the method removes is smaller
than the variance it adds, in this regime.

## 6. Honest limitations

- **`delta_nc` is one specific violation** — leakage of the negative control's
  own noise into the reward. Other violations (e.g. `O_0` depending on the
  behavior policy, or on later states) are untested and may behave differently.
- **Thresholds are environment-specific.** `sigma_0 ≈ 1.0` and
  `delta_nc ≈ 0.5` are properties of this linear-Gaussian POMDP with these
  parameters, not universal constants. The qualitative ordering (validity matters
  more than strength) is the transferable part.
- **`N = 2,000`, `T = 3`, 5 seeds**, as elsewhere.
- Naive's bias drifts slightly downward as `delta_nc` grows (`0.043 → 0.034`).
  We have not chased this; it is small and not what the sweep is about, but it
  does mean the widening gap is not purely the plug-in degrading.

## 7. Files

| File | Role |
|---|---|
| `poc/run_negative_control_sweep.py` | Driver, both sweeps |
| `experiments/results_negative_control.json` | Raw results |
| `src/envs/finite_action_env.py` | `sigma_0` and `delta_nc` knobs (defaults leave the base configuration unchanged) |
