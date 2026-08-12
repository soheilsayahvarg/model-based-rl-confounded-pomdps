# Is Our Repair Still Needed Once the Paper's Own Constraint Is Imposed?

**Status: complete. The repair survives on regret but loses its main
justification. And the published `0.000 selection regret` does not reproduce on a
different candidate set — which is a separate and more serious problem.**

---

## 1. The debt being paid

Phase 3 and Phase 4 both report Signal-Projected Pessimism against the **vanilla**
baseline: vanilla diverges to `−1755`, the projection restores informative values
and achieves `0.000` selection regret.

That vanilla baseline omitted the anchor paper's bridge-class bound `M_R`. So we
have been comparing our repair against a version of the paper's method with a
constraint removed — which flatters the repair. The honest comparison is against
the paper's construction implemented correctly.

## 2. Setup

Toy POMDP, `T = 3`, `N = 5,000`, 3 seeds, six observation-based candidate
policies with exact DP values:

| policy | `V_true` |
|---|---|
| **greedy_lo** | **2.0177** (optimal) |
| always_0 | 1.8648 |
| soft | 1.7612 |
| always_1 | 1.6944 |
| uniform | 1.5050 |
| greedy_hi | 0.9934 |

`M_R` is set to the fitted bridge norm — the smallest defensible admissible value,
since the paper requires `M_R >= ||b_true||`.

## 3. Result

| `c` | method | selected | regret | `V_low` of selection | contains truth |
|---|---|---|---|---|---|
| 0.1 | vanilla | uniform | 0.5127 | −1.81 | yes, but constraint dropped |
| 0.1 | norm-ball | uniform | 0.5127 | **−0.58** | **yes** |
| 0.1 | projected | soft | **0.3419** | −0.25 | **no** |
| 0.5 | vanilla | uniform | 0.5127 | −20.37 | yes, but constraint dropped |
| 0.5 | norm-ball | uniform | 0.5127 | **−3.55** | **yes** |
| 0.5 | projected | uniform | 0.5127 | −6.03 | **no** |
| 1.0 | vanilla | uniform | 0.5127 | −52.78 | yes, but constraint dropped |
| 1.0 | norm-ball | uniform | 0.5127 | **−6.43** | **yes** |
| 1.0 | projected | uniform | **0.3928** | −13.72 | **no** |

## 4. The headline claim does not reproduce

**No method achieves zero regret on this candidate set.** The best any of them
manages is `0.342`, and all three usually select `uniform`, whose true value is
`1.505` against an optimal `2.018`.

Phase 3's `0.000` was measured on its own candidate set. It is not wrong there,
but it **does not generalise**: swap the candidates and the same method on the
same environment selects a policy that is 25% below optimal. A result that
survives only for one choice of candidates is a property of the candidates, not of
the method, and we have been presenting it as the latter.

This is the most consequential finding here, and it is independent of the norm
ball entirely.

## 5. What the norm ball changes

**Most of the "vanilla diverges" story was the missing constraint.** Imposing
`M_R` moves `V_low` from `−52.78` to `−6.43` at `c = 1.0`, and from `−20.37` to
`−3.55` at `c = 0.5`. The catastrophic numbers we have been reporting as the
paper's failure mode are substantially our omission.

**On informativeness the norm ball beats our repair** at `c >= 0.5`: `−3.55`
against `−6.03`, and `−6.43` against `−13.72`. And it does so while **containing
the truth**, which the projection provably does not.

**On regret the projection is better** at `c = 0.1` (`0.342` vs `0.513`) and
`c = 1.0` (`0.393` vs `0.513`), and ties at `c = 0.5`.

So the repair is not worthless, but its justification has to change. It cannot be
"vanilla diverges, we fix it" — the paper's own constraint fixes most of that. The
remaining case for it is a modest regret improvement, bought by giving up
guaranteed coverage of the truth. That is a much weaker claim than the one in the
current write-ups, and whether the trade is worth making is now genuinely unclear.

## 6. What has to change in the phase write-ups

- Every comparison of Signal-Projected Pessimism against "vanilla" must be
  relabelled: that baseline is the paper's method **with a constraint removed**,
  not the paper's method.
- The `−1755` / `−1779` divergence figures must be accompanied by the
  norm-constrained values, or they overstate the failure they diagnose.
- The `0.000 selection regret` claim must be scoped to its candidate set, and
  ideally re-run across several.

None of this touches the identification, estimation or de-biasing results, which
do not involve the pessimism layer.

## 7. Honest limitations

- **One candidate set here too.** This set is not privileged over Phase 3's; the
  point is precisely that the answer moves between them, so neither is
  authoritative. The right fix is a distribution over candidate sets.
- 3 seeds, one environment, `N = 5,000`, `T = 3`.
- `M_R` set to the fitted bridge norm. A larger admissible `M_R` weakens the norm
  ball; `M/||b_hat|| >= 2` makes it inactive entirely (see `norm_constraint.md`).
- Coordinate descent gives an upper bound on the exact inner minimum, so all three
  `V_low` values are approximations from above — as documented in
  `ellipsoid_opt.py`. That caveat applies equally to all three, so the comparison
  is fair even though the absolute values are not certified.

## 8. Files

| File | Role |
|---|---|
| `poc/run_repair_vs_normball.py` | The comparison |
| `experiments/results_repair_vs_normball.json` | Raw results |
| `docs/norm_constraint.md` | The omission that prompted this |
