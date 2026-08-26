# Is the Self-Healing Uniform in the Horizon?

**Predictions committed before measurement; §4 was committed empty.**

---

## 1. The question

Round 8 established that at `t = 2` the stage conditioning set includes history,
the per-action span reaches `|S|`, and `beta_pop ~ 1e-15` in 24/24 cells at every
`confound < 1`. So the truth-leak switch is a **stage-1 statement** and an
inadequate negative control is partially self-healing.

That was measured at `t = 2` only, in one horizon (`T = 3`). Two things follow
that nobody has checked, and a referee will ask both:

1. **Is the restoration uniform?** Does the span reach `|S|` at *every* stage
   `t >= 2`, for every horizon, or does it decay again deeper in the trajectory?
2. **How much does the surviving stage-1 leak matter as `T` grows?** If only
   `t = 1` leaks, the floor in the total value is carried by one block out of
   `T`. The decision-relevant scale — how far apart candidate policies are —
   grows with the horizon. So the floor may be diluted.

Question 2 is the uncomfortable one. If the answer is that the floor washes out,
then the decision-level result of the paper is a **short-horizon phenomenon**,
and saying so is mandatory.

## 2. What is being computed

All population algebra, no sampling. At stage `t` the conditioning cells are
`(a_1..a_{t-1}, o_1..o_{t-1}, o_0)`, so their number grows as
`(|O||A|)^(t-1) |O_0|`. The design span for action `a_t` is spanned by
`E^T v` over those cells, with `v` the forward-propagated posterior over `s_t`.
The count is exponential but the span lives in `R^|O|`, so the basis is
accumulated incrementally and re-orthonormalised, keeping memory at `O(|O|^2)`.

Per stage we report the span rank, `beta_t = ||P_Nul b_true,t||` relative to the
block norm, `beta_g,t`, and the product.

For the dilution question we compare

```
total floor  =  sum_t beta_t * beta_g,t          (only t=1 contributes, if P3 holds)
value spread =  max_pi V_true(pi) - min_pi V_true(pi)
```

over horizons `T = 1..6`, and report their ratio. The spread is the right
denominator because selection regret is decided by differences between candidate
policies, not by the absolute value.

## 3. Predictions

| # | prediction |
|---|---|
| P1 | The span reaches `|S|` at **every** stage `t >= 2`, at every horizon up to `T = 6` and every `confound < 1`. The restoration is uniform, not a `t = 2` accident. |
| P2 | `beta_t <= 1e-12` for every `t >= 2`; `beta_1` stays in the 57–77% band for incomplete configurations, matching the 50/50 table. |
| P3 | The total floor is **constant in `T`** — one leaking block regardless of horizon — so it does not compound. |
| P4 | The value spread grows roughly **linearly** in `T`, so `floor / spread` decays roughly as `1/T`. |
| P5 | **Therefore the decision-level effect is a short-horizon phenomenon.** At `T = 3` (where the paper's 20-seed experiment runs) the floor is a large fraction of the spread; by `T = 6` it should be materially smaller. If P4 holds, the paper's §6 must be scoped to short horizons and the claim "the region does not contract" must be separated from "and therefore selection degrades", because only the first is horizon-free. |

P5 is stated first among the things that would hurt. The floor itself
(Proposition 1) is a per-block statement and is untouched by any of this; what is
at stake is whether its **decision-level consequence** survives a long horizon.

## 4. Measurements

*(committed empty; filled after the run)*
