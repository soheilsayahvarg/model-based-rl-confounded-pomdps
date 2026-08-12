# We Did Drop a Constraint — And It Does Not Rescue the Method

**Status: the finding survives; the MAGNITUDES in this document do not. Three
defects in our own implementation were found in review — see §0. The qualitative
conclusions (§1, §4, §5) hold and one of them is now stronger; every specific
number below must be regenerated with the corrected solver.**

---

## 0. Defects in this document's numbers

**0.1 The norm-ball solver is wrong.** `linear_min`'s bisection scales a single
fixed direction out to the *ellipsoid* boundary, but the KKT solution family is a
two-parameter curve, and in the large-`xi` regime — `c = 1..10`, exactly where all
the rescue numbers live — the ellipsoid should not be active at all. Measured up
to **56% suboptimal**, and on dynamics blocks at `c = 10` it returns points
violating the ball by `+2.9`. Confirmed here: the solver forces the ellipsoid
quadratic form to equal `xi` in every instance tested, which is the bug.

**0.2 The averaged `M` is incoherent.** The driver sets `M` to the mean of
`||b_hat||` across reward blocks and applies it to all of them. Confirmed here on
the toy: `bR_t3` has `||b_hat|| = 1.760` against `M = 1.660`, so **that block's own
centre lies outside the ball** and its constrained problem is not the one we meant
to pose.

**0.3 "The smallest admissible `M`" is not admissible.** We set `M = ||b_hat||`
arguing that `M_R >= ||b_true||`. But ridge *shrinks* the estimate: in **15 of 15**
block/seed pairs `||b_hat|| < ||b_true||`. So `M = ||b_hat||` **excludes the
truth**, and the "contains truth: yes" column in `repair_vs_normball.md` §3 was
false — that comparison pitted two unsound methods against each other.

**0.4 `M_R` is a sup-norm, not an `L2`/RKHS norm** (Assumption 4.1(f) of the
anchor paper). We imposed the wrong geometry. With the paper's literal box class
at its tightest realizable size, the rescue nearly vanishes: `−92` at `c = 10`
against the `−9` an `L2` ball gives. **Every magnitude here is a function of the
class shape we chose**, and the §6b conclusion that the norm ball beats our repair
reverses under the paper's actual class.

**What survives all four.** The omission is real (§3). The divergence is not
rescued by any bounded class (§4, §6b) — and with the corrected solver and an
honest `M` this came out *stronger*, `−9.06` at `c = 10` against a true value of
`1.89`. What does not survive is the claim in §6b that the paper's constraint
beats our repair, and the decomposition in §5 of how much of `−1755` is
attributable to the omission (see §5 note).

---

## 1. The suspicion

The anchor paper minimises over `conf(alpha)` **intersected with the bridge class**
`B_{R,t}`, and Theorem 4.2 carries `M_R` — a uniform upper bound on that class —
right through the suboptimality bound.

Our implementation minimises over the ellipsoid **alone**. `linear_min` in
`ellipsoid_opt.py` has no norm ball, no clipping, no `M_R`; neither does the
model-free layer added for step (b). Verified by reading both.

If the class bound is what stops the excursion into unidentified directions, then
every divergence we have reported is an artifact of omitting a constraint the
method actually has. That had to be settled before writing any proposition.

## 2. The bound used

Not fitted, and not tuned to make the answer come out either way. The model-free
value bridge satisfies `V_t(o) = sum_a b_V(a,o)` with `V` a value function and
Bernoulli rewards, so `|b_V(a,o)| <= T` and per action

```
||b_V(a,.)||_2 <= sqrt(n_o) * T = sqrt(6) * 3 = 7.35
```

This is available *a priori*, before any data. Multiples of it are swept so the
conclusion does not rest on one choice.

## 3. The omission is real, and it is large

With no ball, the minimiser walks to `||b*|| = 98` to `105`.

Against an a priori bound of `7.35`. The unconstrained excursion is roughly
**14× beyond anything a value bridge bounded by `T = 3` could be**. So the
constraint was not decorative, and dropping it was a genuine implementation error
on our side.

## 4. But the divergence survives it

`N = 4,000`, 5 seeds, `confound = 1.0`, `c = 1`:

| config | `V_true` | `V_hat` | `V_low` at `M = M0` | `V_low` unbounded |
|---|---|---|---|---|
| `(4,6,2)` saturating | 2.152 | 1.730 | **−3.15** | −15.12 |
| `(2,6,4)` non-saturating | 2.022 | 1.879 | **−4.14** | −18.70 |

At `c = 10` the constrained values are `−5.08` and `−6.12`.

The headline number shrinks by roughly a factor of 4.5, but the pessimistic value
is still **negative against true values near +2**, and still useless for ranking
policies. The bound is bounded, not repaired.

**So the step (b) conclusion stands, with a corrected magnitude.** The divergence
is not an artifact of the omitted constraint.

## 5. What replaces it is arguably worse

Look at how `V_low` moves with `M`, at `c = 1`, `(2,6,4)`:

| `M / M0` | `V_low` | drop from `V_hat` |
|---|---|---|
| 0.5 | −1.58 | 3.46 |
| 1.0 | −4.14 | 6.02 |
| 2.0 | −8.89 | 10.77 |
| 10 | −18.70 | 20.58 (ball inactive) |

While the ball is active the drop scales with `M` and is **essentially
independent of the ellipsoid**. In that regime the pessimism is not performing
statistical inference at all — it is reporting the worst case over the *prior norm
bound you assumed*, and the confidence region contributes nothing.

This joins up with the non-contracting-width result:

- **Ball inactive:** the width is set by the regularizer and does not shrink with
  `N` (measured: flat across a 64× increase in data).
- **Ball active:** the width is set by `M` and does not shrink with `N` either,
  because `M` is a constant.

**Either way, the pessimistic bound does not contract with data.** That is a
single statement covering both regimes, and it is stronger than what we had
before this check.

## 6. What this changes

- **Corrected:** our implementation omitted `M_R`. All reported pessimistic
  magnitudes are affected and the `−63` is now doubly withdrawn — once for the
  width convention (§8.1 of `family_pessimism.md`) and once for this.
- **Unchanged:** the divergence itself, the causal leakage mechanism, and the
  non-contraction. None of them depend on the omission.
- **New:** when the class bound binds, the method's output is determined by an
  assumption rather than by the data. That is a design-level criticism of
  conservative optimisation over a bounded class, not a bug report.

## 6b. The model-based side, where `M_R` is the paper's own symbol

The check above was on our model-free layer. Repeating it on the **model-based**
ellipsoid, on the **toy POMDP where our published `−1755` / `−1779` divergence was
measured**, `N = 5,000`, 3 seeds, `V_true = 1.8894`:

| `c` | `M / \|\|b_hat\|\|` | `V_low` | `V_low` projected |
|---|---|---|---|
| 0.1 | 0.5 (inadmissible) | 0.77 | −0.37 |
| 0.1 | **1.0** | **−0.79** | −0.37 |
| 0.1 | 2.0 | −2.45 | −0.37 |
| 0.1 | no ball | −2.45 | −0.37 |
| 1.0 | 0.5 (inadmissible) | −1.65 | −20.34 |
| 1.0 | **1.0** | **−8.57** | −20.34 |
| 1.0 | 5.0 | −71.68 | −20.34 |
| 1.0 | no ball | −71.68 | −20.34 |

`M_R` must be at least `||b_true||` for the paper's assumption to hold, so
`M / ||b_hat|| = 1` is the smallest defensible row; `0.5` excludes the truth and
is shown only to bracket the behaviour.

**Same verdict as the model-free side.** The ball binds — the unconstrained
excursion is 2–5× the fitted bridge norm, and the ball is inactive above that —
but at the smallest admissible bound `V_low` is `−0.79` and `−8.57` against a true
value of `1.89`. The divergence survives.

### A constructive by-product

At `c = 1.0`, imposing the paper's own norm ball gives `−8.57` while our
Signal-Projected Pessimism gives `−20.34`. The norm ball is both **tighter** and
**sound**: `M_R >= ||b_true||` holds by assumption, whereas the projection
provably excludes part of the truth (the 21% norm leakage reported in Phase 4).

This does not hold uniformly — at `c = 0.1` the projection is tighter (`−0.37`
against `−0.79`). But it means **part of what our repair was doing, the paper's
own constraint already does, without giving up coverage**. Any write-up of
Signal-Projected Pessimism should compare against the norm-constrained baseline
rather than against the unconstrained one, which is the comparison we have been
making and which flatters the repair.

## 7. Honest limitations

- `M_R` bounds an **RKHS norm** in the paper; we impose a Euclidean norm on the
  tabular coefficient vector. These agree up to the kernel's metric, which is the
  identity in the tabular delta-kernel case we run, but the correspondence should
  be stated rather than assumed in continuous settings.
- The a priori bound `sqrt(n_o) * T` is loose. A tighter admissible bound would
  shrink the drop further; nothing here establishes how far. What it cannot do is
  restore contraction with `N`, since any fixed `M` is `N`-independent.
- The constrained minimisation is solved numerically (SLSQP, dimension 6) and
  checked against the closed form whenever the ball is inactive.
- Model-free layer only. The same check should be run on the model-based
  ellipsoid, where `M_R` is the paper's own symbol, before this is written up.

## 8. Files

| File | Role |
|---|---|
| `poc/run_norm_constraint.py` | The check |
| `experiments/results_norm_constraint.json` | Raw results |
