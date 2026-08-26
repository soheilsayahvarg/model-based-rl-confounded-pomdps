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

*(empty until run)*
