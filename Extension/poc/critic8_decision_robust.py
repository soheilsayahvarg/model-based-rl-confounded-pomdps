"""critic8_decision_robust.py -- round 8, Priority 4: robustness of the 20-seed
decision test.

Three questions, all on schedules C and paper (the load-bearing rows):

 (1) CANDIDATE SET. Six hand-built policies produced the crossing (pessimism
     better at N=2k under the paper schedule, worse at 32k+). Evaluate two
     RANDOM candidate sets on the SAME fits and see whether the qualitative
     pattern (opposite pess/plug slopes; the crossing) survives.
 (2) CALIBRATION. The run calibrated every schedule on seed 0 + greedy_lo. Re-
     calibrate on a different policy (always_1, the true optimum) averaged over
     seeds 0-2, and see whether the ordering changes on the original set.
 (3) CONTINUOUS METRIC. Regret is discrete over 6 policies. Also report the
     continuous margin M = V_low(best-true) - max_pi V_low(pi) (<= 0, zero iff
     the optimum is selected) and mean V_true of the selected policy.

Fits depend only on (kappa, N, seed), so all candidate sets and calibrations
share them. 10 seeds. Writes experiments/critic8_decision_robust.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params, sample_trajectories, dp_value
from value_plugin import empirical_p_o1
from run_regret_incomplete import (candidates, build_blocks, xi_for, fit_once,
                                   make_vg, N_S, N_O, N_O0, N_A, T, CONFOUND,
                                   W_TARGET, SCHEDULES)
from ellipsoid_opt import pessimistic_value

SEEDS = list(range(10))
N_GRID = [2000, 8000, 32000, 128000]
SCHED = [s for s in SCHEDULES if s[0].startswith(("C", "paper"))]
RESULTS = {}


def random_candidates(tag_seed):
    rng = np.random.default_rng(tag_seed)
    out = {}
    for i in range(6):
        out[f"r{tag_seed}_{i}"] = rng.dirichlet((1.0, 1.0), size=N_O)
    return out


def calibrate_variant(p, pol, cal_seeds):
    """c multiplier per schedule: width W_TARGET at N_GRID[0] using policy
    `pol`'s stage-1 gradient, averaged over cal_seeds."""
    cal = {}
    for tag, tau, kap in SCHED:
        ws = []
        for sd in cal_seeds:
            est, d = fit_once(p, N_GRID[0], sd, kap)
            bR, bD = build_blocks(est)
            p_o1 = empirical_p_o1(d["O"], N_O)
            vg = make_vg(bR, bD, pol, p_o1)
            _, g, _ = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
            ws.append(np.sqrt(xi_for(tau, bR[0].N2, bR[0].sigma2, 1.0))
                      * bR[0].h_inv_norm(g[0]))
        cal[tag] = (W_TARGET / float(np.mean(ws))) ** 2
    return cal


def main():
    p = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T,
                       seed=0, confound=CONFOUND)
    sets = {"original": candidates(),
            "rand17": random_candidates(17),
            "rand42": random_candidates(42)}
    vt = {sn: {k: dp_value(p, pi) for k, pi in cs.items()}
          for sn, cs in sets.items()}
    for sn in sets:
        best = max(vt[sn], key=vt[sn].get)
        spread = sorted(vt[sn].values())
        print(f"set {sn}: best {best} = {vt[sn][best]:.4f}, "
              f"range [{spread[0]:.4f}, {spread[-1]:.4f}]")

    print("\ncalibrations:")
    cal_orig = calibrate_variant(p, sets["original"]["greedy_lo"], [0])
    cal_alt = calibrate_variant(p, sets["original"]["always_1"], [0, 1, 2])
    for tag, _, _ in SCHED:
        print(f"  {tag:<18} c_orig {cal_orig[tag]:.3e}   c_alt {cal_alt[tag]:.3e}"
              f"   ratio {cal_alt[tag]/cal_orig[tag]:.3f}")

    rows = []
    for tag, tau, kap in SCHED:
        for N in N_GRID:
            acc = {}
            for sd in SEEDS:
                est, d = fit_once(p, N, sd, kap)
                bR, bD = build_blocks(est)
                p_o1 = empirical_p_o1(d["O"], N_O)
                bh = [b.b_hat for b in bR]
                dh = [b.b_hat for b in bD]
                for sn, cs in sets.items():
                    for cal_name, cal in (("orig", cal_orig),
                                          ("alt", cal_alt)):
                        if cal_name == "alt" and sn != "original":
                            continue
                        cm = cal[tag]
                        xR = [xi_for(tau, b.N2, b.sigma2, cm) for b in bR]
                        xD = [xi_for(tau, b.N2, b.sigma2, cm) for b in bD]
                        vals, pvals = {}, {}
                        for name, pi in cs.items():
                            vg = make_vg(bR, bD, pi, p_o1)
                            pvals[name] = vg(bh, dh)[0]
                            vals[name] = pessimistic_value(
                                bR, bD, xR, xD, vg,
                                rng=np.random.default_rng(sd))["V_low"]
                        best = max(vt[sn], key=vt[sn].get)
                        pick = max(vals, key=vals.get)
                        ppick = max(pvals, key=pvals.get)
                        key = (sn, cal_name)
                        a = acc.setdefault(key, dict(reg=[], preg=[], marg=[]))
                        a["reg"].append(vt[sn][best] - vt[sn][pick])
                        a["preg"].append(vt[sn][best] - vt[sn][ppick])
                        a["marg"].append(vals[best] - max(vals.values()))
            for (sn, cal_name), a in acc.items():
                r = dict(schedule=tag, N=N, set=sn, cal=cal_name,
                         regret=float(np.mean(a["reg"])),
                         plug_regret=float(np.mean(a["preg"])),
                         margin=float(np.mean(a["marg"])),
                         n=len(a["reg"]))
                rows.append(r)
                print(f"{tag:<16} N={N:>7} {sn:>9}/{cal_name:<5}"
                      f" pess {r['regret']:.4f}  plug {r['plug_regret']:.4f}"
                      f"  margin {r['margin']:+.4f}")

    print("\nslopes d(regret)/d(logN) per (schedule, set, cal):")
    for tag, _, _ in SCHED:
        for sn in sets:
            for cal_name in ("orig", "alt"):
                rs = [r for r in rows if r["schedule"] == tag
                      and r["set"] == sn and r["cal"] == cal_name]
                if not rs:
                    continue
                ns = np.log([r["N"] for r in rs])
                gp = float(np.polyfit(ns, [r["regret"] for r in rs], 1)[0])
                gq = float(np.polyfit(ns, [r["plug_regret"] for r in rs], 1)[0])
                gm = float(np.polyfit(ns, [r["margin"] for r in rs], 1)[0])
                print(f"  {tag:<16}{sn:>9}/{cal_name:<5} pess {gp:+.4f}  "
                      f"plug {gq:+.4f}  margin {gm:+.4f}")

    RESULTS["rows"] = rows
    RESULTS["v_true"] = {sn: vt[sn] for sn in sets}
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic8_decision_robust.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
