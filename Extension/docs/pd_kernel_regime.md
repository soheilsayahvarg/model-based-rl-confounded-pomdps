# The PD-Kernel Regime: What the Floor Costs When There Is No Atom

**Written and committed before any of it was run.** The measurement sections are
empty on purpose.

---

## 1. The gap this fills

`sec:scope` of the paper says it outright:

> `as:null` is exact for a tabular delta-kernel, where the design is a
> finite-rank count matrix. For a strictly positive-definite kernel the spectrum
> decays smoothly with no atom at zero, and `prop:floor` degrades to a statement
> about the effective null at the ridge scale rather than an exact one. **We have
> not characterized that regime.**

Every referee who cares about this paper will stop there. The tabular exact null
is the easy case; the whole proximal literature works in RKHS. Leaving it
uncharacterized means the headline holds on the one design where it is cheapest
to get.

## 2. What the source-condition calculation actually gives

`sec:not-new` already records that `cor:dichotomy` is Tikhonov source-condition
theory. Redo that calculation as a measurement target rather than a citation.

Take singular values `s_j ~ j^-b` and gradient mass `g_j^2 ~ j^-a`. Split
`||g||^2_{H^-1} = sum_j g_j^2 / (s_j + lambda)` at `j* = lambda^(-1/b)`, where
`s_j = lambda`:

    j < j*   terms ~ j^(b-a),     partial sum ~ j*^(b-a+1) = lambda^((a-1)/b - 1)
    j > j*   terms ~ j^-a / lam,  partial sum ~ j*^(1-a)/lam = lambda^((a-1)/b - 1)

Both halves give the same power, so

    ||g||_{H^-1} ~ lambda^(-theta/2),      theta = 1 - (a-1)/b

and with `xi = c N^(tau-1)` and `lambda = lambda_0 N^-kappa`,

    W(N) ~ N^(e'/2),      e' = tau + theta*kappa - 1

The exact-null case is `b = infinity`, giving `theta = 1` and recovering
`e = tau + kappa - 1`. So the tabular result is the `theta = 1` endpoint of a
one-parameter family, and the question is how far from that endpoint a real
kernel sits.

## 3. The threshold, in closed form

The anchor schedule is non-contracting exactly when `e' >= 0`, i.e. when

    theta >= theta*(alpha, c_2) := (1 - tau) * (alpha*c_2 + 1) / alpha

Substituting `tau = (alpha+2)/(2*alpha+2)` gives `1 - tau = alpha/(2*alpha+2)`
and therefore

    theta* = (alpha*c_2 + 1) / (2*alpha + 2)

Two readings of that, and the second is the useful one.

At `c_2 = 2`, the top of the admissible range, `theta* = (2*alpha+1)/(2*alpha+2)`,
which is `0.955` at `alpha = 10` and tends to `1`. Almost any spectral smoothness
rescues the schedule.

At `c_2 = 1`, `theta* = (alpha+1)/(2*(alpha+1)) = 1/2` **exactly, for every
`alpha`**. The smoothness parameter cancels. In terms of the spectral exponents,
`theta >= 1/2` iff

    b >= 2*(a - 1)

**The schedule diverges when the design spectrum decays at least twice as fast as
the gradient's mass concentrates.** That is a condition on two numbers a
practitioner can estimate from a Gram matrix, and it is the PD-kernel analogue of
the two switches.

## 4. What this does to `prop:floor`

`prop:floor` is schedule-free because the regularizer cancels: coverage forces
`xi >= lambda*beta^2` and the support norm gives
`||g||_{H^-1} >= beta_g/sqrt(lambda)`, and the two `lambda` factors annihilate.

That cancellation is exactly what an atom at zero buys. Without one, the
effective null at scale `lambda` is `{j : s_j < lambda}`, whose mass shrinks as
`lambda` does. The floor becomes `lambda`-dependent, hence schedule-dependent,
hence **not a floor**. The honest expectation is that the paper's central
adjective is the first casualty of leaving the tabular case.

## 5. Predictions

- **P-K1.** `theta* = (alpha*c_2 + 1)/(2*alpha + 2)` agrees with a direct
  evaluation of the sign of `tau + theta*kappa - 1` in every cell of the
  admissible grid, and equals `1/2` at `c_2 = 1` for every `alpha` tested.
- **P-K2.** On a synthetic spectrum `s_j = j^-b`, `g_j^2 = j^-a`, the measured
  log-log slope of `W(N)` equals `e'/2 = (tau + theta*kappa - 1)/2` with
  `theta = 1 - (a-1)/b`, to within `0.02`, across the whole `(a, b)` grid.
- **P-K3.** With an atom inserted at zero, the measured slope returns to `e/2`
  and is **insensitive** to `(a, b)`: the atom dominates whatever the tail does.
  This is the `theta = 1` endpoint appearing as a measurement.
- **P-K4.** `prop:floor` fails on the PD spectrum in the specific way section 4
  predicts: the quantity `beta_eff(lambda) * beta_g,eff(lambda)`, computed from
  the mass below the ridge scale, **decays with `N`** rather than sitting at a
  constant, while on the atom spectrum it is constant to machine precision.
- **P-K5.** On a real Gaussian-kernel Gram matrix over the observation space,
  the fitted spectral exponent `b` and the fitted gradient exponent `a` predict
  the measured width slope through `e'/2` to within `0.05`.

P-K4 is the one that costs us something. If it holds, the paper must say that
schedule-freeness is a property of the atom and not of the method, and
`cor:epaper` becomes conditional on `b >= 2(a-1)` rather than unconditional.

## 6. Measurement

`poc/run_pd_kernel.py`. Exact linear algebra on a diagonal spectrum, no
sampling.

### P-K1 holds. 0 mismatches of 25.

`theta* = (alpha*c_2 + 1)/(2*alpha + 2)` agrees with a direct sign evaluation of
`tau + theta*kappa - 1` in every admissible cell, and at `c_2 = 1` the value is
`0.5` **exactly** for every `alpha` in `{0.5, 1, 2, 5, 10}`. The smoothness
parameter cancels, as derived.

### P-K3 holds. The atom dominates and does not care about the tail.

With an atom inserted, the measured slope is `0.207` to `0.227` against a
predicted `e/2 = 0.2273`, worst error `0.0195`, and the **spread across the
entire `(a, b)` grid is `0.0195`**. Whatever the tail does, one exactly-zero
eigenvalue sets the exponent.

### P-K4 holds, and it takes `prop:floor`'s central adjective.

| spectrum | floor across the `N` grid | log-log slope |
|---|---|---|
| atom, `a=2 b=3` | `1.091` -> `1.006` | `-0.0085` |
| atom, `a=3 b=2` | `1.0005` -> `1.0000` | `-0.0000` |
| PD, `a=2 b=3` | `9.51e-2` -> `6.42e-3` | `-0.302` |
| PD, `a=3 b=2` | `5.04e-4` -> `1.34e-7` | `-0.911` |

On the atom spectrum the floor is constant, which is `prop:floor`. On the PD
spectrum the same quantity decays by up to four orders of magnitude. **It is not
a floor.** Schedule-freeness is a property of the atom, not of the method.

### P-K2 is REFUTED as stated, and splits into two causes.

Worst error `0.4547` against a `0.02` tolerance. The failures are not one
phenomenon.

**Cause 1, our own defect: the grid could not straddle the transition.** The
split point is `j* = lambda^(-1/b)`, and at the smallest `lambda` on the grid
(`2.70e-7`) that is `2.39e4` for `b = 1.5`, against a truncation at
`J_MAX = 20000`. The spectrum ended before the ridge scale reached it. Re-run at
`J_MAX = 4e6`:

| `(a, b)` | `theta` | predicted | at `J_MAX=2e4` | at `J_MAX=4e6` |
|---|---|---|---|---|
| `(1.5, 1.5)` | `0.667` | `0.0758` | `0.0147` | **`0.0727`** |
| `(1.5, 2.0)` | `0.750` | `0.1136` | `0.0987` | **`0.1127`** |
| `(2.0, 1.5)` | `0.333` | `-0.0758` | `-0.0921` | **`-0.0729`** |

Errors fall to `0.003`, `0.001`, `0.003`. This is the **fifth** instance of the
failure mode `claims.md` already names, and the one place we did not install the
precondition guard that exists for it. The guard is now in the driver.

**Cause 2, a real correction to the law.** Cells with `theta <= 0` stay wrong at
any truncation. `(3.0, 1.5)` has `theta = -0.333` and a predicted `-0.3788`; the
measured value is `-0.2227` at both truncations. That number is
`(tau - 1)/2 = -0.2273`. When `theta <= 0` the sum `sum_j g_j^2 / s_j`
**converges**, `||g||_{H^-1}` tends to a constant, and the width is governed by
`sqrt(xi)` alone. The exponent saturates:

    slope = ( tau + clip(theta, 0, 1) * kappa - 1 ) / 2

At `theta = 0` exactly the sum diverges logarithmically, and the two boundary
cells `(3,2)` and `(4,3)` sit at `-0.189` against `-0.2273`, a residual of
`0.038` that a log factor would explain.

`sec:not-new` of the paper states `e' = tau + theta*kappa - 1` without the clip.
That is wrong for `theta <= 0` and should be corrected.

### P-K5 is REFUTED, and the fault is our experiment, not the theory.

Worst error `0.1064` against `0.05`. We fitted a power law to a **Gaussian**
kernel spectrum, whose eigenvalues decay exponentially. A log-log fit to an
exponential is meaningless, so `b_fit` between `2.77` and `8.37` is an artifact
of the fitting window rather than a spectral exponent. The Gaussian kernel is
not in the polynomial-decay class the theory addresses; it behaves like
`theta = 1` with a logarithmic drag, and `400` modes truncate it besides. Part 5
is redone with a kernel that actually has polynomial decay.

## 7. The corrected law, and a fresh out-of-sample test

Stated post-hoc, so it is not evidence yet. The grid below shares no `(a, b)`
pair with the run that produced the correction.

    slope(a, b) = ( tau + clip(1 - (a-1)/b, 0, 1) * kappa - 1 ) / 2

- **P-K6.** On the fresh grid `a in {1.2, 1.8, 2.5, 3.5, 5.0}` and
  `b in {1.2, 2.5, 4.0, 8.0}`, all `20` cells, the measured slope matches the
  clipped law to within `0.02`, **provided** the truncation guard passes.
- **P-K7.** Every cell the guard rejects is a cell where `j*` exceeds a tenth of
  the mode count, and none of the accepted cells miss. If a rejected cell would
  have passed anyway, the guard is over-conservative and we say so.
- **P-K8.** At `theta = 0` exactly, the residual shrinks as the `N` grid extends,
  consistent with a logarithmic correction rather than a wrong exponent. Measured
  as the slope of the residual against `log N` over a doubled grid.
- **P-K9.** With a Laplacian kernel, whose eigenvalues decay polynomially, the
  fitted `b` predicts the measured slope through the clipped law to within
  `0.05`, where the Gaussian kernel failed at `0.106`.

### Measurement

*(empty until run)*
