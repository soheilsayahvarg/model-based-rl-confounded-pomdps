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

`poc/run_alignment_nondegenerate.py`, population algebra throughout, no sampling.
All five predictions confirmed, one of them literally wrong in a way worth
recording (§5.3). The outcome is the uncomfortable one: **§3's conclusion
survives on a clean knob, and its headline sentence is wrong.**

### 5.1 P1 confirmed — knob 1 is degenerate

| `alpha` | row separation | `sep/(1-a)` | `beta_g` | `bg/(1-a)` | span rotation |
|---|---|---|---|---|---|
| 0.00 | 0.81364 | 0.81364 | 0.7578 | 0.758 | 0.00° |
| 0.40 | 0.48818 | 0.81364 | 0.5274 | 0.879 | 12.70° |
| 0.80 | 0.16273 | 0.81364 | 0.1926 | 0.963 | 28.58° |
| 0.99 | 0.00814 | 0.81364 | 0.0098 | 0.975 | 36.81° |

Log–log fit of `beta_g` against row separation: **slope 0.9538, `R^2` = 0.9984**,
inside the predicted `1.00 ± 0.05` band by 0.004. The `bg/(1-a)` column shows the
proportionality is near-exact rather than exact — leakage falls slightly slower
than separation because it is normalised by `||nu1(alpha)||`, which also shrinks.

> **The knob varies channel informativeness and gradient alignment as a single
> variable.** At `alpha = 0.99` the emission rows are 100× closer together than at
> `alpha = 0`; the environment is nearly uninformative about the latent state. The
> sweep could not have produced a large leakage at a well-separated channel,
> because in this family there is no such point.

### 5.2 P2 confirmed — §4b contains a false sentence

The span vector rotates **36.81°** across the sweep. `family_pessimism.md` §4b
says

> "span rank is 1 in every row, so the null space is FIXED and only the gradient's
> alignment with it moves"

Only the **dimension** is fixed. The subspace itself rotates by more than a third
of a right angle, so the sweep moves the gradient *and* the null space *and* the
channel together. Corrected in §4b.

### 5.3 P3 confirmed in substance, and my own check was wrong

| `t` | row sep | `max\|dH\|` | span rot | `beta_g` | ratio |
|---|---|---|---|---|---|
| 0.000 | 0.81364 | 0.00e+00 | 0.00° | 0.7578 | 24.738 |
| 0.400 | 0.81364 | 5.55e-17 | 0.00° | 0.4572 | 10.984 |
| 0.800 | 0.81364 | 5.55e-17 | 0.00° | 0.1345 | 3.056 |
| 0.990 | 0.81364 | 1.39e-17 | 0.00° | 0.0062 | 1.009 |
| 0.999 | 0.81364 | 5.55e-17 | 0.00° | 0.00061 | 1.000 |

Row separation constant to every digit, span rotation exactly `0.00e+00`,
`beta_g` spanning **3.09 decades** — a full decade more than knob 1 achieves, at a
channel that is never touched.

**My prediction said `max|dH| = 0.00e+00` and the measurement returned `5.55e-17`,
so the script printed `CONTROL IMPERFECT`.** That verdict is wrong and the
prediction was over-stated: `H` is *algebraically* independent of `p1` (the
normalisation `v/v.sum()` collapses to `delta_{s_a}` whatever `p1` is), and
`5.55e-17` is one unit in the last place on a matrix with entries of order `10^-1`
— it comes from reordering a floating-point sum, not from a dependence. Predicting
exact zero for a quantity assembled by summation was the error. The control is
clean.

### 5.4 P4 confirmed — it is an identity

Over **17 cells across both knobs**, the closed form

```
ratio = sqrt( 1 + (beta_g^2 / (1 - beta_g^2)) * (sigma + rho) / rho )
```

matches the measured ratio with a worst relative error of **8.45e-15**. Not
approximately: to machine precision, at every point of both sweeps.

> **No alignment sweep could ever have refuted §3.** In a design with an exact
> null space, the ratio of unprojected to projected width is an arithmetic
> function of the leakage and the conditioning. Sweeping alignment and observing
> that the width follows is a re-measurement of an identity.

That does not make §3 false. It makes it *reliable* — it holds in every
environment, not just this one — and *not a discovery*. The ledger row is updated
accordingly.

### 5.5 P5 confirmed, and §3's headline sentence is wrong

Writing `theta` for the angle between the gradient and the identified span
(`sin theta = beta_g`), and noting that the null eigenvalue of `H` is exactly
`rho` while the top is `sigma + rho`, the closed form is

```
ratio = sqrt( 1 + tan^2(theta) * cond(H) )
      ≈ tan(theta) * sqrt(cond(H))          where the ratio is large
```

The asymptotic form is accurate to **0.08% at ratio 24.7, 1% at ratio 7, and is
useless below ratio ≈ 2** (87% error at ratio 1.01) — which is the correct
behaviour, since the `+1` is what keeps the ratio at one when the gradient is
fully identified. In the regime the paper is about, the large-penalty regime, the
penalty **factorises into an alignment term and a conditioning term, multiplied**.

§3 currently reads:

> "Ill-conditioning is necessary but not sufficient; the alignment between the
> value functional and the null space decides it."

**The second clause is wrong as a general statement.** Conditioning is not a mere
precondition that alignment then resolves — it enters as `sqrt(cond(H))`, on
exactly equal footing with `tan(theta)`. Neither factor "decides" anything alone;
the penalty is their product, and either one at zero kills it.

§3 reached its dichotomy because *in its own sweep* `tan(theta)` moves **119×**
while `sqrt(cond(H))` moves **1.26×** — a 95:1 imbalance built into the knob. The
observation "conditioning barely moves while the penalty moves with alignment" is
a true description of that sweep and a false description of the mechanism.

### 5.6 What the identity buys back

Three things survive, and one of them is new.

1. **Non-arithmetic content of §3:** not the functional form, but that real
   confounded POMDPs land at `beta_g` between **0.46 and 0.76**, i.e.
   `tan(theta)` between 0.5 and 1.2. The gradient is not incidentally misaligned,
   it is misaligned by order one, so the penalty inherits essentially the whole
   of `sqrt(cond(H))`. *Why* `beta_g > 0` at all is the two-switch result, which
   is a genuine finding and is where §3's weight should be transferred.
2. **Knob 2 is the experiment §3 should have run**, and it gives the same curve
   (24.7 → 1.00) with the channel frozen. So the conclusion is not an artifact of
   channel destruction — it just was not being tested by the sweep that
   established it.
3. **The identity subsumes step (b) §4.** Since the null eigenvalue is the ridge,
   `cond(H) ≈ sigma/rho`, so `ratio ∝ rho^-1/2`. Under the implemented schedule
   `rho = 0.03/sqrt(N)` this predicts `ratio ∝ N^1/4`. §4's published numbers give
   ratios `10.83, 16.31, 22.86, 32.61` over a 64× range in `N` — a log–log slope
   of **0.265** against the predicted **0.25**, the 6% gap being the residual
   drift in the unprojected column §4 calls flat. The flat-width result and the
   alignment result are the same identity read along two different axes.

## 6. Consequences

| document | change |
|---|---|
| `family_pessimism.md` §3 | headline sentence corrected — the penalty is a **product** of alignment and conditioning, not alignment alone |
| `family_pessimism.md` §4b | "the null space is FIXED" corrected to "its dimension is fixed"; knob disclosed as degenerate; knob 2 added |
| `claims.md` | "gradient leakage is the causal driver, **not** conditioning" → **scoped/corrected**; "§3 reproduced independently" → **holds, as an identity** |
| step (b) §4 | recorded as a corollary of the same identity rather than an independent result |

**Found without the critic.** The suspicion was written into the round-8 prompt
and then tested here rather than handed over; the round-8 review should still
check it, since the point of an adversarial round is that self-audit is exactly
what fails.

