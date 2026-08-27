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

`FAMILY=dense_lowrank SCHED=kappa=1.5`, 20 seeds, against the same slice of the
`dense` family. Only `P` differs, so the stage-1 floor is identical
(`beta = 1.1775`, `beta_g = 0.3639`, floor `0.4285`) and the per-action span is
`2` at every stage instead of `2, 4, 4`.

#### P9 and P10 are REFUTED. The remedy does not fail.

| arm | slope, `dense` | slope, `dense_lowrank` |
|---|---|---|
| `full` | `+0.0220` | `+0.0045` |
| `tail` | `-0.0255` | **`-0.0149`** |
| `proj1` | `-0.0345` | **`-0.0254`** |
| `projall` | `-0.0290` | `-0.0300` |
| `plugin` | `-0.0210` | `-0.0162` |

We predicted `tail` and `proj1` would have slopes `>= 0` where their targeting is
wrong. Both stay negative. `tail` runs `0.0855 -> 0.0096` and `proj1`
`0.1256 -> 0.0257`, while `full` sticks at `0.0997` from `N = 32,000` on.

The `dense` slice is now stored (`results_stage_selective_dense_kappa15.json`,
`20` seeds), and its per-`N` levels are the sharpest form of the finding:

| arm | `2,000` | `8,000` | `32,000` | `128,000` | slope | modal pick at `128,000` |
|---|---|---|---|---|---|---|
| `full` | `0.0952` | `0.1218` | `0.1714` | **`0.1804 +- 0.0000`** | `+0.0220` | `uniform` |
| `plugin` | `0.0894` | `0.0226` | `0.0000` | `0.0000` | `-0.0210` | `always_1` |
| `tail` | `0.1093` | `0.0258` | `0.0000` | `0.0000` | `-0.0255` | `always_1` |
| `proj1` | `0.1292` | `0.0904` | `0.0000` | `0.0000` | `-0.0345` | `always_1` |
| `projall` | `0.1093` | `0.0746` | `0.0000` | `0.0000` | `-0.0290` | `always_1` |

Three things are worth stating separately.

1. `full` is the **only** arm whose regret rises, and it is the only arm that
   fails to find `always_1`. Every other arm, including the plug-in, is at zero
   regret by `N = 32,000`.
2. At `N = 128,000` `full` has a confidence half-width of **exactly zero**: all
   `20` seeds pick the same wrong policy. This is not variance, it is convergence
   to the wrong answer.
3. `full`'s modal pick drifts `greedy_lo -> soft -> uniform -> uniform`, away
   from the optimum and toward maximum entropy. That is what a penalty growing on
   the leaky `t = 1` block should do: it charges every policy in proportion to
   its gradient's null-space mass, and `uniform` minimises that. The selector is
   not being cautious, it is optimising the penalty instead of the value.

The plug-in column is the control that makes this a statement about the
**selector** and not about the fit: both are built from the identical estimates.

#### P12 and P13 hold.

`projall` is negative in both families, `-0.0290` and `-0.0300`, and at
`N = 128,000` in the rank-failure family it reaches `0.0096 +- 0.0086`, which is
the plug-in's `0.0072 +- 0.0077` within the interval. It stays conservative in
**every** cell of both families, and its mean gap shrinks `-1.125 -> -0.399`
while `full`'s grows `-1.424 -> -2.085`.

The arm that needs no knowledge of which stage leaks is the one that works
everywhere. That is the remedy to carry.

#### P11 is REFUTED, and the reason matters more than the prediction.

We predicted `beta_g > 0.1` and a nonzero empirical null at the `t >= 2` blocks
under a rank failure. Measured, they are **exactly what the dense family gives**:

    beta_g    bR_t1 0.3259   bR_t2 0.0000   bR_t3 0.0000   bD_t1 0.3259   bD_t2 0.0000
    nulldim   bR_t1 96       bR_t2 0        bR_t3 0        bD_t1 288      bD_t2 0

**The population per-action design span and the empirical stage-2 design are
different objects.** The span that `selfheal_theorem.md` proves lives in
`R^|O|` and is cut to dimension `2` by the transition rank. The design the floor
actually uses is the empirical `T_2` in coefficient space, whose conditioning
cells are counted by the history alphabet, not by the transition rank. A
rank-deficient `P` leaves that matrix **full rank** and merely ill-conditioned.

So `cor:selfheal`'s link to `prop:floor` is looser than the paper states. The
corollary is about identification of the bridge; the floor is about an exact null
in a coefficient-space design. They coincide at `t = 1` and come apart after.
This is a caveat on our own theorem, found by a prediction it caused us to make.

#### The rank failure does propagate, as conditioning rather than as a null.

Per-block penalty slopes, same schedule, same seeds:

| block | `dense` | `dense_lowrank` |
|---|---|---|
| `bR_t1` | `+0.181` | `+0.181` |
| `bR_t2` | `-0.417` | **`+0.090`** |
| `bR_t3` | `-0.429` | `-0.144` |
| `bD_t1` | `+0.151` | `+0.147` |
| `bD_t2` | `-0.438` | **`+0.121`** |

The `t = 1` blocks are untouched, as they must be since `P` does not enter their
design. Two of the three later blocks **flip sign** and the third shrinks three
times more slowly. So the effect is real and it reaches the later stages; it
simply is not an exact null, and `beta_g` cannot see it.

That is why `tail` still converges: the later blocks' penalties grow, but from
`0.0135` and `0.0369` rather than from `0.19` and `0.85`, so they never dominate
the decision on this grid. A longer `N` grid or a stronger rank failure could
change that, and we have not tested either.

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

`FAMILY=dense_lowrank SCHED=kappa=1.5`, 20 seeds, against the same slice of the
`dense` family. Only `P` differs, so the stage-1 floor is identical
(`beta = 1.1775`, `beta_g = 0.3639`, floor `0.4285`) and the per-action span is
`2` at every stage instead of `2, 4, 4`.

One run answers Parts A through D, so the measurement is reported once, in
section 4. **P9 and P10** is the part that answers this section's
predictions.

This block was duplicated verbatim into all four Part sections. Removed here
rather than left in place: four copies of one table read as four
independent confirmations, which is exactly the impression it should not
give.

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

`FAMILY=dense_lowrank SCHED=kappa=1.5`, 20 seeds, against the same slice of the
`dense` family. Only `P` differs, so the stage-1 floor is identical
(`beta = 1.1775`, `beta_g = 0.3639`, floor `0.4285`) and the per-action span is
`2` at every stage instead of `2, 4, 4`.

One run answers Parts A through D, so the measurement is reported once, in
section 4. **P12 and P13** is the part that answers this section's
predictions.

This block was duplicated verbatim into all four Part sections. Removed here
rather than left in place: four copies of one table read as four
independent confirmations, which is exactly the impression it should not
give.

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

`FAMILY=dense_lowrank SCHED=kappa=1.5`, 20 seeds, against the same slice of the
`dense` family. Only `P` differs, so the stage-1 floor is identical
(`beta = 1.1775`, `beta_g = 0.3639`, floor `0.4285`) and the per-action span is
`2` at every stage instead of `2, 4, 4`.

One run answers Parts A through D, so the measurement is reported once, in
section 4. **P11 and the block-penalty slopes** is the part that answers this section's
predictions.

This block was duplicated verbatim into all four Part sections. Removed here
rather than left in place: four copies of one table read as four
independent confirmations, which is exactly the impression it should not
give.

## 9. Two different null spaces, and which one the floor actually uses

**Written and committed before it was run.**

Part D refuted P11 in a way that exposed something the paper does not
distinguish. Two objects have been used interchangeably:

    POPULATION per-action design span     lives in R^|O|
                                          rank = min(|O_0|, |S|) generically
                                          governs whether the BRIDGE is identified
                                          this is what tab:beta and cor:selfheal measure

    EMPIRICAL stage-2 design T_2          lives in coefficient space, dim |A||O| x n_y
                                          rank = (distinct conditioning cells) x n_y
                                          governs the FLOOR, since Nul is its null
                                          this is what the pessimism layer uses

At `t = 1` the conditioning set is `(A_1, O_0)`, giving `|A||O_0| = 4` cells
against `|A||O| = 12` coefficient profiles, so the measured null dimension is
`(12 - 4) x 12 = 96`, exactly what Part A printed. **That count involves neither
`|S|` nor `P`.** It is driven by `|O_0| < |O|`.

`sec:switch-beta` says the truth leaks when `|O_0| < |S|`. `as:null` says the
design has a null of dimension at least `|O| - min(|O|, |O_0|)`, which is
`|O_0| < |O|`. Those are **different conditions**, and the paper moves between
them without saying so.

`(2, 6, 4)` separates them: `|O_0| = 4 >= |S| = 2`, so the population span
resolves the latent state and `tab:beta` reports `beta = 0`. But
`|O_0| = 4 < |O| = 6`, so the empirical design still carries a null of dimension
`(12 - 8) x 12 = 48`. The question the paper never asks is whether the true
bridge has mass in **that** null.

### Predictions

- **P-E1.** In `(4,6,2)`, empirical null dimension is `96` at `t = 1` and `0` at
  `t >= 2`, matching Part A, and empirical `beta_hat` is within `10%` of the
  population `1.1775`. The two notions agree where both switches are on.
- **P-E2.** In `(2,6,4)` at confounding below `1`, the empirical null dimension
  at `t = 1` is `48` and **nonzero**, yet empirical `beta_hat` is below `0.05`
  relative to the bridge norm. The truth lies inside the empirical span even
  though that span is not everything, so the two conditions agree in effect and
  the paper's switch is the right one.
- **P-E3.** Empirical `beta_g` in `(2,6,4)` is below `0.05` at every confounding
  below `1`, matching `prop:betag`.
- **P-E4.** The empirical null dimension is exactly
  `(|A||O| - |A|min(|O_0|,|O|)) x n_y` at `t = 1` in every configuration tested,
  i.e. a pure cell count, independent of `|S|`, of `P`, and of the confounding
  level.

### If P-E2 fails

Then a design the paper calls complete still has a nonzero floor, the two
switches do not characterize it, and `cor:conjunction` is false as stated. That
would be the most serious defect found in this line of work, and it would have
been sitting inside the gap between `as:null` and `sec:switch-beta` the whole
time.

### Measurement

*(empty until run)*

### The decisive follow-up: is it finite-sample or structural?

**Disclosure: `(2, 6, 4)` at confounding `0.9` was already run before this was
written.** It gave

    N =   4,000   nulldim 52.3   beta_emp 0.1282
    N =  16,000   nulldim 48.3   beta_emp 0.0437
    N =  64,000   nulldim 48.0   beta_emp 0.0338
    N = 256,000   nulldim 48.0   beta_emp 0.0138

so `beta_emp` falls by a factor of `9` over a `64x` increase in `N`, about
`N^(-1/2)`, and the null dimension converges to the structural `48`. The
remaining configurations are unobserved and P-E5 and P-E6 are predictions about
them.

- **P-E5.** In every complete design (`|O_0| >= |S|`, confounding below `1`),
  `beta_emp` has a log-log slope in `N` of `-0.5 +- 0.15`. It is sampling noise
  projected onto a structurally empty direction, so it should carry the
  parametric rate.
- **P-E6.** The measured null dimension converges to the exact cell count
  `(|A||O| - |A|min(|O_0|,|O|)) x n_y` as `N` grows, so P-E4's misses were
  near-threshold eigenvalues in sparsely sampled cells and not a wrong formula.

If P-E5 holds, `cor:conjunction` is **correct as a population statement** and
gains a finite-sample caveat with an explicit rate: a complete design has a
measured floor of order `N^(-1/2)`, which is not zero at any finite `N` but does
not survive the limit. That is a footnote, not a defect.

If `beta_emp` were flat in `N` instead, the corollary would be false as stated.

### Measurement of the rate

`poc/run_empirical_null_rate.py`, complete designs only, `N` from `4,000` to
`256,000`, 3 seeds.

**P-E5 holds in 5 of 6.** Log-log slopes of `beta_emp` in `N`:

    -0.662   (2,6,4) cf 0.6      <- the one miss
    -0.501   (2,6,4) cf 0.9
    -0.527   (2,3,2) cf 0.6
    -0.410   (2,3,2) cf 0.9
    -0.500   (3,7,5) cf 0.6
    -0.505   (3,7,5) cf 0.9

Four of the six sit within `0.03` of the parametric rate. The single miss is
faster than predicted, not slower, so it does not threaten the conclusion.

**P-E6 holds in 6 of 6.** The measured null dimension reaches the exact cell
count at the largest `N` in every case: `48`, `48`, `12`, `12`, `56`, `56`. The
excess at small `N` was near-threshold eigenvalues in sparsely sampled cells, as
suspected, and not a wrong formula.

The largest `beta_emp` at the largest `N` is `0.0273`, down from `0.2356` at
`N = 4,000` in the same cell.

### Verdict

`cor:conjunction` is **correct as a population statement**. What it gains is a
finite-sample footnote with a rate: a design the paper calls complete carries a
measured floor of order `N^(-1/2)`, which reaches `0.12` at the sample sizes used
in `tab:decision` and is not zero at any finite `N`.

The gap between `as:null` and `sec:switch-beta` is real and the paper should name
it. `as:null` counts conditioning cells, `sec:switch-beta` counts latent states,
and the two coincide only in the limit. Since every decision experiment in this
work runs at `N <= 128,000`, the distinction is not academic for our own numbers,
even though it is for the theorem.

---

## 10. Is `projall` a remedy or a coin flip that landed right?

**Written and committed before measurement. Adversarial against our own arm.**

### What `projall` actually does

Reading `ellipsoid_opt.py` rather than the arm's name: `projected=True` restricts
the **perturbation direction** to the signal subspace `U_sig` and leaves the
centre `b_hat` untouched. The penalty becomes
`sqrt(xi) * ||U_sig' g||_{(U_sig' H U_sig)^-1}`.

So `projall` does **not** switch to a point-identified surrogate functional. It
keeps `<g, b>` and shrinks the region until the region no longer contains the
truth. `BlockEllipsoid`'s docstring already carries this as audit finding
A-high: the projected `V_low` "is NOT a certified lower bound", and coverage held
on the tested grid "by sign-alignment, not by construction".

Section 8 of `converse_identification.md` now puts a number on how much is
excluded. Since `b_hat` is minimum-norm, `P_Nul b_hat = 0`, so the excluded part
of the truth is `P_Nul b_true`, and at the t=1 block

    ||P_Nul b_true|| = 1.6344   against   ||b_true|| = 2.3590

**`projall`'s region structurally excludes 69% of the true bridge's norm.**

### The decomposition

    V_low - V_true  =  -penalty  -  c ,      c = <U_null' g, U_null' b_true>

`projall` is conservative iff `c >= -penalty`, which for a small penalty means
iff `c >= 0`. And `|c| <= HW_t1 = 0.5777` by Cauchy-Schwarz, with the sign fixed
entirely by alignment. That is the "sign-alignment" caveat, made computable.

### Predictions

- **P-C14.** On `(4,6,2)` at confounding `0.9`, where `projall` was reported to
  work, `c > 0` at the t=1 block, and the alignment cosine
  `c / (||U'g|| * ||U'b_true||)` exceeds `0.3` in magnitude.
- **P-C15.** `c` is near-constant in `N`, relative spread below `20%` for
  `N >= 8,000`, because it is a population quantity while the penalty is not.
- **P-C16.** The sign of `c` **flips** somewhere on the grid
  `{(4,6,2), (4,8,3), (2,6,2), (3,7,5)} x {0.0, 0.3, 0.6, 0.9}`. At least one
  cell with a nonempty null has `c < 0`.
- **P-C17.** Where `c < 0`, `projall`'s margin `-penalty - c` turns positive at
  large `N`: the penalty shrinks on the identified subspace while `|c|` stays
  fixed, so the arm becomes **anti-conservative**.

**P-C16 is load-bearing and it is aimed at our own arm.** If the sign flips,
`projall` is not a remedy. It is a rule whose validity depends on an alignment
nobody checked, and it happened to land right on the one grid we ran. That would
have to be the headline about it, not "the arm that works everywhere".

If the sign does **not** flip anywhere, then the conservatism has a structural
reason we have not identified, and finding it becomes the interesting question.
Either outcome is worth more than the current claim.

### Measurement

