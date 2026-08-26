"""run_empirical_null_rate.py -- is the complete-design floor finite-sample?

Predictions: docs/stage_selective_pessimism.md section 9, P-E5 and P-E6,
committed at faee092 with the disclosure that (2,6,4) at confounding 0.9 had
already been run.

WHAT IS AT STAKE. run_empirical_null.py refuted P-E2: designs the paper calls
complete, where tab:beta reports beta = 0 EXACTLY, carry an empirical beta up to
0.1200. The empirical null there is structurally nonempty because as:null counts
CONDITIONING CELLS (|O_0| < |O|) while sec:switch-beta counts LATENT STATES
(|O_0| < |S|), and those are different conditions.

Everything turns on the N-dependence. If beta_emp decays at the parametric rate
it is sampling noise projected onto a structurally empty direction, and
cor:conjunction is right in the population with a finite-sample footnote. If it
is flat, a complete design has a real floor and cor:conjunction is false as
stated.

Writes experiments/results_empirical_null_rate.json.
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
from run_empirical_null import null_basis, true_bR_vec

# complete designs only: |O_0| >= |S|, where the population beta is exactly 0
CASES = [((2, 6, 4), 0.6), ((2, 6, 4), 0.9),
         ((2, 3, 2), 0.6), ((2, 3, 2), 0.9),
         ((3, 7, 5), 0.6), ((3, 7, 5), 0.9)]
N_GRID = [4000, 16000, 64000, 256000]
SEEDS = [0, 1, 2]
T, N_A = 3, 2
RESULTS = {}


def run_case(cfg, cf):
    n_s, n_o, n_o0 = cfg
    n_y = 2 * n_o
    pred_null = (N_A * n_o - N_A * min(n_o0, n_o)) * n_y
    p = default_params(n_s=n_s, n_a=N_A, n_o=n_o, n_o0=n_o0, T=T,
                       seed=0, confound=cf)
    b_true, _ = true_bR_vec(p, n_o, N_A)
    out = []
    for N in N_GRID:
        bs, nds = [], []
        for sd in SEEDS:
            d = sample_trajectories(p, N, np.random.default_rng(sd))
            N2 = N - int(N * 0.7)
            est = TabularBridgeEstimator(n_obs=n_o, n_act=N_A, n_o0=n_o0,
                                         n_r=2, T=T, mode="primal", seed=sd,
                                         lambda2=N2 ** (-1.5))
            est.fit(d["O0"], d["O"], d["A"], d["R"])
            st = est.stage_store["bR_t1"]
            U = null_basis(st["H"], st["lam2"])
            nds.append(U.shape[1])
            bs.append(0.0 if U.shape[1] == 0 else
                      float(np.linalg.norm(U.T @ b_true) /
                            np.linalg.norm(b_true)))
        out.append((N, float(np.mean(nds)), float(np.mean(bs))))
    return pred_null, out


def main():
    print("complete designs only (|O_0| >= |S|), where the POPULATION beta is 0")
    print("%12s%6s%10s%10s%12s" % ("config", "cf", "N", "nulldim", "beta_emp"))
    rows, slopes = [], []
    for cfg, cf in CASES:
        pred_null, out = run_case(cfg, cf)
        for N, nd, be in out:
            print("%12s%6.1f%10d%10.1f%12.4f" % (str(cfg), cf, N, nd, be))
        Ns = [o[0] for o in out]
        bs = [max(o[2], 1e-300) for o in out]
        sl = float(np.polyfit(np.log(Ns), np.log(bs), 1)[0])
        conv = abs(out[-1][1] - pred_null) < 1e-9
        print("%12s%6s%10s%10s%12s   slope %+.4f   nulldim -> %s (%s)" %
              ("", "", "", "", "", sl, pred_null,
               "converged" if conv else "NOT converged"))
        slopes.append(sl)
        rows.append(dict(config=list(cfg), confound=cf, pred_nulldim=pred_null,
                         N=Ns, nulldim=[o[1] for o in out],
                         beta_emp=[o[2] for o in out], slope=sl,
                         nulldim_converged=bool(conv)))
    print("\n" + "=" * 76)
    ok5 = [abs(s + 0.5) <= 0.15 for s in slopes]
    print("P-E5  slopes: %s" % ["%+.3f" % s for s in slopes])
    print("      within -0.5 +- 0.15 in %d of %d cases" % (sum(ok5), len(ok5)))
    ok6 = [r["nulldim_converged"] for r in rows]
    print("P-E6  null dimension reached the exact cell count at the largest N "
          "in %d of %d" % (sum(ok6), len(ok6)))
    for r in rows:
        if not r["nulldim_converged"]:
            print("      %s cf=%.1f  ended at %.1f, predicted %d"
                  % (r["config"], r["confound"], r["nulldim"][-1],
                     r["pred_nulldim"]))
    print("\n      largest beta_emp at the largest N: %.4f"
          % max(r["beta_emp"][-1] for r in rows))
    RESULTS.update(rows=rows, slopes=slopes, N_grid=N_GRID, seeds=SEEDS)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_empirical_null_rate.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
