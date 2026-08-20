"""
run_coverage_sweep.py -- a predictive test of the coverage explanation.

BACKGROUND. The finite-action study found pessimistic selection consistently
worse than plug-in selection, refuting our earlier chain-compounding hypothesis
(restart gaps there were <= 4e-16, so the optimizer was not at fault). A
diagnostic instead found the pessimism penalty to be monotonically inverse to
behavior-policy coverage, with the OPTIMAL policy the least covered one.

That was a four-point observation in a single configuration. This driver turns it
into a falsifiable prediction and tests it: if coverage is the mechanism, then
increasing how well the logging policy covers the optimal action should shrink
the pessimistic-selection regret toward zero, and shrink the optimal policy's
pessimism penalty relative to the others.

DESIGN. We vary b_pref, a dose-preference offset in the behavior policy's
logits, which shifts the logging policy toward low doses WITHOUT touching the
confounding channel (eta*driver) or the environment dynamics. Crucially, every
candidate policy's TRUE value is invariant to b_pref -- only the logged data
changes. So the target ranking is fixed across the entire sweep and any change in
regret is attributable to coverage alone.

FALSIFIABLE PREDICTION: regret_pessimistic decreases monotonically (up to seed
noise) as cov(optimal action) increases, and reaches 0 once coverage is adequate.
If it does not, the coverage explanation is wrong and we report that.

Writes experiments/results_coverage_sweep.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from finite_action_env import (default_params, sample_trajectories, true_value,
                               candidate_policies, N_ACTIONS)
from fa_bridge_estimator import FiniteActionBridgeEstimator
from fa_value_plugin import empirical_initial_mean
from fa_pessimism import fa_pessimistic_selection

SEEDS = [0, 1, 2, 3, 4]
B_PREF_GRID = [0.0, -1.0, -2.0, -3.0, -4.0]
C_GRID = [0.1, 0.3, 1.0]
N_MAIN = 2000
RESULTS = {}


def ci95(v):
    v = np.asarray(v, float)
    return float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def main():
    base = default_params(T=3, kappa=1.0)
    T = base["T"]
    cands = candidate_policies()
    v_true = {n: true_value(base, pi) for n, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    print("candidate true values (invariant across the sweep):",
          {k: round(v, 4) for k, v in v_true.items()})
    print("optimal policy:", best, "\n")

    rows = []
    for b in B_PREF_GRID:
        p = default_params(T=3, kappa=1.0, b_pref=b)
        # true values must not move: the sweep changes data only
        assert all(abs(true_value(p, pi) - v_true[n]) < 1e-12
                   for n, pi in cands.items()), "b_pref changed a true value"

        covs, reg_p, reg_g, pen_best, pen_rest = [], [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            d = sample_trajectories(p, N_MAIN, rng)
            cov_a = np.bincount(d["A_idx"].ravel(), minlength=N_ACTIONS) / d["A_idx"].size
            covs.append(float(np.dot(cands[best], cov_a)))

            est = FiniteActionBridgeEstimator(T=T, n_actions=N_ACTIONS, seed=sd).fit(
                d["O0"], d["O"], d["A_idx"], d["R"])
            m1 = empirical_initial_mean(d["O"])

            rp, rg, pb, pr = [], [], [], []
            for c in C_GRID:
                r = fa_pessimistic_selection(est, cands, m1, c, v_true=v_true,
                                             n_restarts=3, seed=sd)
                rp.append(r["suboptimality_pessimistic"])
                rg.append(r["suboptimality_plugin"])
                pens = {n: q["V_plugin"] - q["V_low"] for n, q in r["policies"].items()}
                pb.append(pens[best])
                pr.append(np.mean([v for n, v in pens.items() if n != best]))
            reg_p.append(np.mean(rp)); reg_g.append(np.mean(rg))
            pen_best.append(np.mean(pb)); pen_rest.append(np.mean(pr))

        rows.append(dict(
            b_pref=b,
            coverage=float(np.mean(covs)),
            regret_pess=float(np.mean(reg_p)), regret_pess_ci=ci95(reg_p),
            regret_plugin=float(np.mean(reg_g)),
            penalty_best=float(np.mean(pen_best)),
            penalty_others=float(np.mean(pen_rest)),
            penalty_ratio=float(np.mean(pen_best) / np.mean(pen_rest))))
        r = rows[-1]
        print(f"b_pref={b:>5.1f}  cov(opt)={r['coverage']:.3f}  "
              f"regret_pess={r['regret_pess']:.4f}+/-{r['regret_pess_ci']:.3f}  "
              f"regret_plug={r['regret_plugin']:.4f}  "
              f"penalty(best)/penalty(others)={r['penalty_ratio']:.3f}")

    RESULTS["coverage_sweep"] = rows
    RESULTS["true_values"] = {k: float(v) for k, v in v_true.items()}
    RESULTS["optimal_policy"] = best

    cov = np.array([r["coverage"] for r in rows])
    reg = np.array([r["regret_pess"] for r in rows])
    ratio = np.array([r["penalty_ratio"] for r in rows])
    corr = float(np.corrcoef(cov, reg)[0, 1])
    RESULTS["corr_coverage_vs_regret"] = corr
    RESULTS["regret_at_lowest_coverage"] = float(reg[0])
    RESULTS["regret_at_highest_coverage"] = float(reg[-1])
    RESULTS["prediction_confirmed"] = bool(reg[-1] < reg[0] - 1e-12)

    print(f"\ncorrelation(coverage, pessimistic regret) = {corr:+.3f}")
    print(f"regret at lowest coverage ({cov[0]:.3f}) = {reg[0]:.4f}")
    print(f"regret at highest coverage ({cov[-1]:.3f}) = {reg[-1]:.4f}")
    print(f"penalty ratio best/others: {ratio[0]:.3f} -> {ratio[-1]:.3f}")
    print("PREDICTION CONFIRMED" if RESULTS["prediction_confirmed"]
          else "PREDICTION NOT CONFIRMED -- coverage explanation is wrong")

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_coverage_sweep.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
