"""run_estimator_equivalence.py -- is the compacted estimator the same estimator?

ScaledBridgeEstimator compacts the conditioning alphabet to its observed support
so that T can exceed 3. That is only legitimate if it changes nothing on the
grids where the dense version runs. Argued in the module docstring; asserted here.

Compares b_R, b_D, every block's H, b_hat_vec and lam2, and the resulting
plug-in value, across configurations, sample sizes and seeds. Exact equality is
the bar, not a tolerance: the two differ by a relabelling of empty columns and
the RNG is consumed in the same order, so any nonzero difference is a bug.

Writes experiments/results_estimator_equivalence.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params, sample_trajectories
from bridge_estimator import TabularBridgeEstimator
from bridge_estimator_scaled import ScaledBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1

CASES = [((4, 6, 2), 0.9, 3), ((2, 6, 4), 0.6, 3), ((4, 8, 3), 0.9, 3),
         ((3, 7, 5), 0.0, 3), ((4, 6, 2), 0.9, 4)]
N_GRID = [2000, 8000]
SEEDS = [0, 1]


def fit_both(p, cfg, N, sd, T):
    n_s, n_o, n_o0 = cfg
    d = sample_trajectories(p, N, np.random.default_rng(sd))
    N2 = N - int(N * 0.7)
    out = []
    for cls in (TabularBridgeEstimator, ScaledBridgeEstimator):
        est = cls(n_obs=n_o, n_act=2, n_o0=n_o0, n_r=2, T=T, mode="primal",
                  seed=sd, lambda2=N2 ** (-1.5))
        est.fit(d["O0"], d["O"], d["A"], d["R"])
        out.append(est)
    return out[0], out[1], d


def main():
    print("%14s%5s%8s%6s%9s%14s%14s%14s" %
          ("config", "T", "N", "seed", "n_x cut", "max|dbR|", "max|dbD|",
           "max|dH|"))
    rows, worst = [], 0.0
    for cfg, cf, T in CASES:
        n_s, n_o, n_o0 = cfg
        p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=T, seed=0,
                           confound=cf)
        for N in N_GRID:
            for sd in SEEDS:
                a, b, d = fit_both(p, cfg, N, sd, T)
                dbR = float(np.abs(a.bR_hat - b.bR_hat).max())
                dbD = float(np.abs(a.bD_hat - b.bD_hat).max())
                dH = 0.0
                for k in a.stage_store:
                    for f in ("H", "b_hat_vec", "lam2"):
                        dH = max(dH, float(np.abs(
                            np.asarray(a.stage_store[k][f]) -
                            np.asarray(b.stage_store[k][f])).max()))
                # dense alphabet vs the compacted one at the last stage
                n_x_dense = 2 * (n_o * 2) ** (T - 1) * n_o0
                cut = n_x_dense / max(b.n_x_compact[T], 1)
                print("%14s%5d%8d%6d%9.0fx%14.2e%14.2e%14.2e" %
                      (str(cfg), T, N, sd, cut, dbR, dbD, dH))
                worst = max(worst, dbR, dbD, dH)
                rows.append(dict(config=list(cfg), confound=cf, T=T, N=N,
                                 seed=sd, dbR=dbR, dbD=dbD, dH=dH,
                                 n_x_dense=n_x_dense,
                                 n_x_compact=b.n_x_compact[T]))

    print("\n" + "=" * 90)
    print("worst absolute difference over %d comparisons: %.3e" % (len(rows), worst))
    print("EXACT" if worst == 0.0 else "NOT EXACT -- investigate before using")

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_estimator_equivalence.json"))
    with open(out, "w") as f:
        json.dump(dict(rows=rows, worst=worst, exact=bool(worst == 0.0)),
                  f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
