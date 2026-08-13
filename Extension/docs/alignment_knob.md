# Is the Alignment Result a Discovery or an Identity?

**Self-initiated audit of `family_pessimism.md` §3 / §4b, the load-bearing result
of step (b). Written before measurement; §5 was committed empty.**

---

## 1. The suspicion

§3 is the column the extension rests on: *the penalty is set by how the value
gradient sits relative to the null space, and by nothing else.* Both the review's
version and my population reproduction (§4b) establish it with the same kind of
knob — interpolate every emission row toward the mean,

```
E(alpha) = (1 - alpha) * E + alpha * mean_row.
```

At `alpha = 1` every latent state emits identically. An environment whose emission
rows are equal carries **no observational information about the latent state at
all**. So the endpoint of the sweep is not "a well-identified problem whose
gradient happens to align" — it is "a problem with nothing to identify".

If leakage falls only because the channel is being destroyed, the sweep
demonstrates its conclusion rather than testing it, and the ledger row
`§3 reproduced independently | holds` is wrong.

## 2. Why the knob is algebraically suspect

Write `m` for the mean emission row. Because `p1` sums to one,

```
nu1(alpha)  = p1 E(alpha) = (1-alpha) nu1 + alpha m
u(alpha)   ∝ E(alpha)[s_a] = (1-alpha) E[s_a] + alpha m
```

Both the gradient and the span vector are affine in `alpha` toward the **same
point** `m`. Their difference is `(1-alpha)(nu1 - E[s_a])`, while their common
part grows. The angle between them must close as `alpha -> 1` for a reason that
has nothing to do with identification geometry.

Meanwhile the row separation is

```
|| E(alpha)[0] - E(alpha)[1] || = (1-alpha) * || E[0] - E[1] ||
```

— exactly proportional to `(1-alpha)`. **Alignment and channel informativeness are
the same variable under this knob.** That is the definition of a confounded
experiment, and it is the one design flaw the extension has hit four times.

## 3. A second, non-degenerate knob

At `confound = 1.0` and `n_s = n_a = 2` the behaviour policy is the deterministic
indicator `pi_b[s, s] = 1`, so the stage-2 design weight `v_{a,x} = p1 * 1{s = a} *
K0[:,x]` is supported on the single state `s_a` **whatever `p1` is**. Normalising
kills `p1` entirely:

```
m_x = E^T (v / v.sum()) = E[s_a, :]     for every x
D   = outer(E[s_a], E[s_a])             exactly, independent of p1
```

So tilting the state prior,

```
p1(t) = (1-t) * p1_0 + t * delta_{s_a},
```

moves `nu1 = E^T p1` — and therefore its alignment — while leaving `E`, `K0`,
`P`, `pi_b`, the span, and `H` **literally untouched**. This is the knob §3 should
have used: it varies alignment at a fixed channel, whereas the interpolation knob
varies both at once.

## 4. Predictions, committed before running

| # | prediction |
|---|---|
| P1 | Knob 1 is degenerate: `beta_g` and row separation are both proportional to `(1-alpha)`; a log–log fit of `beta_g` against separation has slope `1.00 ± 0.05`. |
| P2 | Knob 1's **span vector rotates**. §4b says "span rank is 1 in every row, so the null space is FIXED" — the *dimension* is fixed, the *subspace* is not. Predicted rotation across the sweep: `> 5` degrees. |
| P3 | Knob 2 holds `H` constant to `max\|dH\| = 0.00e+00` and the span angle to `0.0` degrees, and still moves `beta_g` over at least one decade. |
| P4 | Under **both** knobs the width ratio obeys the closed form to machine precision: `ratio = sqrt(1 + (beta_g^2/(1-beta_g^2)) * (sigma+rho)/rho)`, `sigma = \|\|E[s_a]\|\|^2`. |
| P5 | Therefore §3's content **survives but changes character**: it is not an empirical finding about this environment, it is an identity about ellipsoidal confidence regions with an exact null space. That makes it more reliable and less novel at the same time. |

P5 is the uncomfortable one and is stated first on purpose. If P4 holds, the
honest report is that the sweep was never capable of failing, and the ledger row
should read **holds as an identity** rather than **holds** on the strength of a
measurement.

## 5. Measurements

*(committed empty; filled after the run)*
