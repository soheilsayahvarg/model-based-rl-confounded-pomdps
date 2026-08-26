# Stage-Selective Pessimism: Can the Diagnosis Be Turned Into a Remedy?

**Written and committed before any of it was run.** The measurement sections are
empty on purpose; they are filled in a later commit. Everything above them is a
prediction, so a wrong prediction stays in the history rather than being quietly
edited into a right one.

---

## 1. Where this comes from

`cor:selfheal` (docs/horizon_dilution.md 4.1, poc/critic8_t2span.py) established
that an inadequate negative control leaks **only at the first stage**. At `t = 1`
the conditioning set is `(A_1, O_0)`, so a per-action design has `|O_0|` profiles
and `|O_0| < |S|` forces an incomplete span. From `t = 2` on the conditioning set
includes history, the per-action span reaches `|S|`, and we measure
`beta_t <= 2.7e-15` with a per-stage floor `<= 4.3e-30`.

The floor `W >= beta * beta_g` is therefore **not** a property of the estimator.
It is a property of `2` of the `2T-1` blocks.

That has an obvious operational reading which we have never tested:

> Apply the pessimism layer only to the blocks whose region can actually contract.

This is implementable, not an oracle rule. Which blocks leak is decided by
`|O_0| < |S|` and by whether the logging policy saw latent state, both of which a
practitioner knows **before fitting anything**. That is the whole point of
calling the two switches *checkable*.

## 2. Why dropping the layer at stage 1 is not obviously cheating

The objection writes itself: pessimism exists to produce a lower bound, and a
block with no penalty contributes no conservatism, so `tail` is not the same
method.

Correct, and the honest answer is that the guarantee being given up was already
empty. Under an `e > 0` schedule on a leaky block, `W(N) = Theta(N^(e/2))`. A
lower bound that diverges is not a weak bound, it is a vacuous one: it certifies
`V(pi) >= V_hat - W` where `W` grows without limit. Nothing is lost by declining
to compute it, and `sec:decision` already shows something is gained.

What *is* genuinely given up is the finite-sample validity of the combined bound,
and Part C measures the price rather than asserting it is small.

## 3. Arms

All four share one fit per `(N, seed)`. They differ only in which blocks get a
nonzero region, so nothing but the pessimism layer varies.

| arm | `t = 1` blocks | `t >= 2` blocks |
|---|---|---|
| `full` | full region | full region |
| `tail` | `xi = 0` (plug-in) | full region |
| `proj1` | region restricted to the identified span | full region |
| `plugin` | `xi = 0` | `xi = 0` |

`full` and `plugin` are the two columns already in `tab:decision`, recomputed
inside the same loop so the four arms are exactly comparable.

`proj1` is the interesting middle: it keeps a penalty at stage 1 but refuses to
pay for directions the design cannot see. It is the variant a referee will ask
about, because it is the one that does not abandon the guarantee outright.

## 4. Part A: is stage 1 actually the culprit?

Before proposing a remedy, check the diagnosis. The per-block penalty is

    p_k = sqrt(xi_k) * ||g_k||_{H_k^{-1}}

for each of the `2T-1 = 5` blocks at `T = 3`. If the divergence is not
concentrated in the `t = 1` blocks, the remedy is aimed at the wrong place and
Part B is not worth running.

### Predictions

- **P1.** At `t >= 2`, the per-block penalty **decreases** in `N`, with a log-log
  slope near the signal-dominated value `(tau - 1) / 2`: `-0.227` for the paper
  schedule, `-0.500` for `kappa = 1.5`. Reason: `beta_g,t = 0` there, so the
  gradient has no mass in the null and the width is set by signal directions.
- **P2.** At `t = 1`, the per-block penalty **increases**, with slope near `e/2`:
  `+0.227` for the paper schedule, `+0.250` for `kappa = 1.5`.
- **P3.** The `t = 1` reward block is the single largest penalty at
  `N = 128,000` under both `e > 0` schedules, and its share of the total penalty
  is **above 50%** there and rising in `N`.

P1 and P2 together are the sharp version: the same fit, the same schedule,
opposite-signed slopes across stages. If that shows up, Cor 3 has a measurable
fingerprint at the level of the estimator and not just in population algebra.

### Measurement

*(empty until run)*

## 5. Part B: does the remedy recover selection?

Four arms, `20` seeds, `N` in `{2000, 8000, 32000, 128000}`, config `(4,6,2)` at
confounding `0.9`, the same grid as `tab:decision`. Schedules `C (kappa=1.5)` and
the anchor paper's.

### Predictions

- **P4.** `tail` regret has a **negative** slope in `log N`, against `full`'s
  positive one, under both `e > 0` schedules.
- **P5.** At `N = 128,000`, `tail` lands closer to `plugin` than to `full`:
  `|tail - plugin| < |tail - full|` under both schedules.
- **P6.** `proj1` also has a negative slope, but is **more conservative** than
  `tail`: higher regret at `N = 2,000` and a lower mean `V_low` at every `N`.

### The refutation this could produce

The value is multilinear across blocks and the coordinate descent couples them.
It is entirely possible that a clean `t >= 2` block still selects badly because
its gradient `g_t` was computed from a stage-1 estimate that the unpenalized
block leaves uncorrected. If so:

- **P4 fails**, `tail`'s slope is `>= 0`, and the finding is that the leak
  propagates through the *coupling* even though the block itself is clean.

That would be a genuinely more interesting result than the remedy working, and it
would scope `cor:selfheal` a second time: block-level self-healing would not
imply decision-level self-healing. We are recording it here so that outcome
cannot be presented afterwards as what we expected.

### Measurement

*(empty until run)*

## 6. Part C: what the remedy costs

Conservatism is the property the layer exists to provide. For every
`(arm, N, seed, policy)` we record whether `V_low <= V_true(pi)`, the empirical
version of the check `ellipsoid_opt.py` documents as V4.

### Predictions

- **P7.** `full` stays conservative in essentially every cell (its penalty
  grows), `plugin` is anti-conservative roughly half the time (it is a point
  estimate), and `tail` sits **strictly between** them at every `N`.
- **P8.** `tail`'s anti-conservative fraction **rises** with `N`, exceeding
  `10%` at `N = 128,000` under at least one schedule. This is the price: the
  arm converges in selection precisely because it stops over-paying, and
  over-paying is what kept the bound valid.

If P7 holds, the honest summary is a trade rather than a free lunch, and the
paper should present it that way.

### Measurement

*(empty until run)*

## 7. What a positive result would mean, and what it would not

It would **not** repair the anchor paper's method. `cor:epaper` is about a
schedule; this is about which blocks a schedule is applied to, which is an
orthogonal choice.

It would mean the two switches are a *pre-fit test with an action attached*:
check `|O_0|` against `|S|`, check whether the logger saw latent state, and if
both fire, do not pay pessimism at the first stage because that payment cannot
be earned back.

That converts the contribution from a diagnosis into a diagnosis plus a
remedy, which is the difference between a paper that reports a limitation and a
paper that does something about it.

## 8. Part D: does the remedy survive its own refutation?

**Written and committed before Part D was run.** Disclosure on what was already
seen: an interrupted 20-seed run completed schedule C on the `dense` family
before it was killed, so the `full` / `tail` / `proj1` / `plugin` numbers for
that one schedule are **already observed** and are not predictions. The
`projall` arm is new, the `dense_lowrank` family is new, and the paper schedule
is unobserved. P9 through P13 concern only those.

### Why Part D exists

`selfheal_theorem.md` section 6 refuted `cor:selfheal` as published. With
rank-deficient transitions, every stage leaks, not only the first. The whole
premise of `tail` and `proj1` is that the leaky blocks are exactly the `t = 1`
ones, and that premise is now known to be false on a class of problems.

So the remedy has to be tested where its targeting is wrong. `dense_lowrank`
keeps `p_1`, `K_0`, `E` and `pi_b` and changes only `P`, so the stage-1 floor is
**identical** (`beta = 1.1775`, `beta_g = 0.3639`, floor `0.4285`) while the
per-action span is `2` at every stage instead of `2, 4, 4`. One variable moves.

### A fifth arm

`projall` restricts **every** block's region to its identified span, not only
the first. It is the arm that does not need to know which stage leaks: it
refuses to pay for directions the design cannot see, wherever they occur. If it
works in both families, the refutation has produced a better remedy than the one
it killed.

### Predictions

- **P9.** `tail` has a slope `>= 0` under `dense_lowrank`, against `-0.026`
  measured under `dense`. The remedy fails when its targeting is wrong.
- **P10.** `proj1` also fails there, slope `>= 0`, for the same reason: it
  restricts the `t = 1` region only.
- **P11.** In `dense_lowrank`, `beta_g` at the `t >= 2` blocks exceeds `0.1` and
  the empirical null dimension there is nonzero, against exactly `0` and `0` in
  `dense`. This is the mechanism, and it should be visible in Part A before Part
  B is consulted.
- **P12.** `projall` has a **negative** slope in **both** families. This is the
  one that matters.
- **P13.** `projall` stays conservative in every cell of both families, so it
  buys convergence without giving up the property the layer exists to provide.

### If P12 fails

Then there is no remedy that survives without knowing the leak structure, and
the honest conclusion is that the two switches are a diagnosis and not a
prescription. We would report that, drop the remedy section, and keep the
refutation, which is a real result on its own.

### Measurement

*(empty until run)*
