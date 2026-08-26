"""run_empirical_null.py -- which null space does the floor actually use?

Predictions: docs/stage_selective_pessimism.md section 9 (P-E1..P-E4), committed
at 8b5f6d9 before this file existed.

THE GAP. Two objects have been used interchangeably in this work:

  POPULATION per-action span   in R^|O|, rank min(|O_0|, |S|) generically.
                               tab:beta and cor:selfheal measure this.
  EMPIRICAL stage-2 design     in coefficient space, rank = (distinct
                               conditioning cells) x n_y. prop:floor's Nul is
                               THIS one, because the pessimism layer inverts it.

At t = 1 the conditioning set is (A_1, O_0), so |A||O_0| = 4 cells against
|A||O| = 12 coefficient profiles gives a null of (12-4) x 12 = 96 -- a pure cell
count involving neither |S| nor P. as:null states |O_0| < |O|;
sec:switch-beta states |O_0| < |S|. Different conditions.

(2,6,4) separates them. It is complete by the population criterion, tab:beta
reports beta = 0, and yet the empirical design still carries a 48-dimensional
null. Whether the TRUE BRIDGE has mass in that null has never been asked, and if
it does then cor:conjunction is false as stated.

Writes experiments/results_empirical_null.json.
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
from value_plugin import plugin_value, empirical_p_o1
from run_completeness_beta import true_bridges_generic, beta_pop_for

CONFIGS = [(4, 6, 2), (2, 6, 4), (2, 3, 2), (3, 7, 5), (4, 8, 3)]
CONFOUNDS = [0.0, 0.6, 0.9]
N_GRID = [8000, 32000]
SEEDS = [0, 1, 2]
T = 3
N_A = 2
TOL = 1e-9
RESULTS = {}


def null_basis(H, lam, tol=TOL):
    """Orthonormal basis of the numerical null of T_2 = H - lam I."""
    w, V = np.linalg.eigh(H - lam * np.eye(H.shape[0]))
    cut = tol * max(float(w[-1]), 1e-300)
    return V[:, w <= cut]


def true_bR_vec(p, n_o, n_a):
    """The true reward bridge as the estimator's flat coefficient vector.

    Layout must match build_blocks: reshape(n_a, n_o, 2, n_o), i.e. axes are
    (action, o-tilde, reward, next observation). true_bridges_generic returns
    bR[a] with axes (o-tilde, reward, next observation), so stacking over a and
    ravelling reproduces the estimator's convention.
    """
    bR, res = true_bridges_generic(p)
    return np.stack([bR[a] for a in range(n_a)], axis=0).ravel(), float(res)


def main():
    print("empirical vs population null, per (config, confound)")
    print("%12s%6s%6s%9s%10s%12s%12s%12s%12s" %
          ("config", "cf", "t", "nulldim", "predicted", "beta_emp",
           "beta_pop", "betag_emp", "resid"))
    rows = []
    for cfg in CONFIGS:
        n_s, n_o, n_o0 = cfg
        n_y = 2 * n_o
        # P-E4: a pure cell count, |A||O| profiles minus |A|min(|O_0|,|O|) seen
        pred_null = (N_A * n_o - N_A * min(n_o0, n_o)) * n_y
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_A, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            b_true, resid = true_bR_vec(p, n_o, N_A)
            b_pop = max(beta_pop_for(p, a)[0] for a in range(N_A))
            nb_pop = max(beta_pop_for(p, a)[1] for a in range(N_A))
            nu1 = p["p1"] @ p["E"]
            for t in (1, 2):
                acc_nd, acc_b, acc_bg = [], [], []
                for N in N_GRID:
                    for sd in SEEDS:
                        d = sample_trajectories(p, N, np.random.default_rng(sd))
                        N2 = N - int(N * 0.7)
                        est = TabularBridgeEstimator(
                            n_obs=n_o, n_act=N_A, n_o0=n_o0, n_r=2, T=T,
                            mode="primal", seed=sd, lambda2=N2 ** (-1.5))
                        est.fit(d["O0"], d["O"], d["A"], d["R"])
                        st = est.stage_store["bR_t%d" % t]
                        U = null_basis(st["H"], st["lam2"])
                        acc_nd.append(U.shape[1])
                        if U.shape[1] == 0:
                            acc_b.append(0.0)
                            acc_bg.append(0.0)
                            continue
                        acc_b.append(float(np.linalg.norm(U.T @ b_true) /
                                           max(np.linalg.norm(b_true), 1e-300)))
                        # gradient of the plug-in value in this block
                        bRl = [est.stage_store["bR_t%d" % k]["b_hat_vec"]
                               for k in range(1, T + 1)]
                        bDl = [est.stage_store["bD_t%d" % k]["b_hat_vec"]
                               for k in range(1, T)]
                        BR = np.stack([b.reshape(N_A, n_o, 2, n_o)
                                       for b in bRl])
                        BD = np.stack([b.reshape(N_A, n_o, n_o, n_o)
                                       for b in bDl])
                        pi = np.full((n_o, N_A), 1.0 / N_A)
                        _V, _pp, (gR, _gD) = plugin_value(
                            BR, BD, pi, empirical_p_o1(d["O"], n_o),
                            return_grads=True)
                        g = gR[t - 1].ravel()
                        acc_bg.append(float(np.linalg.norm(U.T @ g) /
                                            max(np.linalg.norm(g), 1e-300)))
                nd = float(np.mean(acc_nd))
                be = float(np.mean(acc_b))
                bg = float(np.mean(acc_bg))
                print("%12s%6.1f%6d%9.1f%10d%12.4f%12.4f%12.4f%12.2e" %
                      (str(cfg), cf, t, nd, pred_null if t == 1 else 0,
                       be, b_pop / max(nb_pop, 1e-300) if t == 1 else 0.0,
                       bg, resid))
                rows.append(dict(config=list(cfg), confound=cf, stage=t,
                                 nulldim=nd, pred_nulldim=pred_null,
                                 beta_emp=be, beta_pop_rel=b_pop /
                                 max(nb_pop, 1e-300), betag_emp=bg,
                                 bridge_resid=resid, n_s=n_s, n_o=n_o,
                                 n_o0=n_o0))

    print("\n%s" % ("=" * 92))
    t1 = [r for r in rows if r["stage"] == 1]
    t2 = [r for r in rows if r["stage"] == 2]

    bad4 = [r for r in t1 if abs(r["nulldim"] - r["pred_nulldim"]) > 1e-9]
    print("P-E4  t=1 null dimension == the cell count in %d of %d cells"
          % (len(t1) - len(bad4), len(t1)))
    for r in bad4:
        print("      %s cf=%.1f  measured %.1f  predicted %d"
              % (r["config"], r["confound"], r["nulldim"], r["pred_nulldim"]))
    same = {}
    for r in t1:
        same.setdefault(tuple(r["config"]), set()).add(r["nulldim"])
    print("      independent of confounding: %s"
          % all(len(v) == 1 for v in same.values()))

    inc = [r for r in t1 if r["n_o0"] < r["n_s"]]
    com = [r for r in t1 if r["n_o0"] >= r["n_s"] and r["confound"] < 1.0]
    print("\nP-E1  INCOMPLETE (|O_0| < |S|), t=1:")
    for r in inc:
        print("      %s cf=%.1f  nulldim %.0f  beta_emp %.4f  beta_pop %.4f"
              % (r["config"], r["confound"], r["nulldim"], r["beta_emp"],
                 r["beta_pop_rel"]))
    print("\nP-E2  COMPLETE (|O_0| >= |S|) but with a NONEMPTY empirical null:")
    hits = [r for r in com if r["nulldim"] > 0]
    for r in hits:
        flag = "" if r["beta_emp"] < 0.05 else "   <-- ABOVE 0.05"
        print("      %s cf=%.1f  nulldim %.0f  beta_emp %.4f  "
              "beta_pop %.4f%s" % (r["config"], r["confound"], r["nulldim"],
                                   r["beta_emp"], r["beta_pop_rel"], flag))
    worst = max((r["beta_emp"] for r in hits), default=0.0)
    print("      worst beta_emp among them: %.4f   [predicted < 0.05]" % worst)

    print("\nP-E3  empirical beta_g in complete designs, worst: %.4f"
          % max((r["betag_emp"] for r in com), default=0.0))
    print("\n      t>=2 null dimension, all configs: %s"
          % sorted({r["nulldim"] for r in t2}))

    RESULTS.update(rows=rows, tol=TOL, N_grid=N_GRID, seeds=SEEDS,
                   worst_complete_beta=float(worst))
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_empirical_null.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
