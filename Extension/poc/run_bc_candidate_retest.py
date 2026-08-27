"""run_bc_candidate_retest.py -- does the headline failure survive adding bc?

THE QUESTION. This project's central decision result is that on (4,6,2) at T=3,
confounding 0.9, schedule C, the pessimistic selector's regret RISES with N
(slope +0.0220) while the plug-in on the identical fits falls (-0.0210), and that
`full` converges to a wrong policy with zero variance across 20 seeds.

scale_and_baselines.md section 6 then found that a behaviour clone beats the best
of the six candidates on 6 of 6 configurations, so every one of those numbers is
regret against a reference a no-OPE baseline already beats.

On Environment L at T = 5, once bc was in the candidate set, EVERY arm including
`full` selected it and every regret was 0. If the same happens at (4,6,2), the
headline failure is an artifact of a candidate set too narrow to contain a good
policy, and the paper's central claim has to be withdrawn or rescoped.

This is the decisive test and it is aimed at the project's main result.

PREDICTION, fixed before running (see scale_and_baselines.md section 10):
  P-L13. The failure SURVIVES. `full`'s slope stays positive with bc present,
         because bc's margin over always_1 at (4,6,2) T=3 is 0.0009 against a
         spread of 0.2966 -- three tenths of a percent, far too small to change
         a ranking that a penalty of order 0.2 is driving. Environment L differed
         because there bc's margin was 95% of the spread.

Writes experiments/results_bc_candidate_retest.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params, dp_value
from value_plugin import empirical_p_o1
from ellipsoid_opt import pessimistic_value
import run_stage_selective as D

TAU, KAP = 0.0, 1.5
N_GRID = [int(x) for x in
          os.environ.get("NGRID", "2000,8000,32000,128000").split(",")]
SEEDS = list(range(int(os.environ.get("SEEDS", "20"))))
ARMS = ["full", "plugin", "projall"]
W_TARGET = 0.12


def bc_policy(p):
    d = p["p1"]
    joint = (d[:, None] * p["E"]).T
    denom = joint.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (joint / denom) @ p["pi_b"]


def arm_value(arm, bR, bD, xR, xD, vg):
    if arm == "plugin":
        return float(vg([b.b_hat for b in bR], [b.b_hat for b in bD])[0])
    out = pessimistic_value(bR, bD, xR, xD, vg, projected=(arm == "projall"),
                            rng=np.random.default_rng(0))
    return float(out["V_low"])


def run(with_bc):
    p = default_params(n_s=D.N_S, n_a=D.N_A, n_o=D.N_O, n_o0=D.N_O0, T=D.T,
                       seed=0, confound=D.CONFOUND)
    cands = dict(D.candidates())
    if with_bc:
        cands["bc"] = bc_policy(p)
    v_true = {k: dp_value(p, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    spread = max(v_true.values()) - min(v_true.values())

    est, d = D.fit_once(p, N_GRID[0], 0, KAP)
    bR, bD = D.build_blocks(est)
    vg = D.make_vg(cands["greedy_lo"], empirical_p_o1(d["O"], D.N_O))
    _, g0, _ = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
    w1 = np.sqrt(D.xi_for(TAU, bR[0].N2, bR[0].sigma2, 1.0)) \
        * bR[0].h_inv_norm(g0[0])
    cm = (W_TARGET / w1) ** 2

    out = {a: [] for a in ARMS}
    picks = {a: [] for a in ARMS}
    for N in N_GRID:
        acc = {a: [] for a in ARMS}
        pk = {a: [] for a in ARMS}
        for sd in SEEDS:
            est, d = D.fit_once(p, N, sd, KAP)
            bR, bD = D.build_blocks(est)
            p_o1 = empirical_p_o1(d["O"], D.N_O)
            xR = [D.xi_for(TAU, b.N2, b.sigma2, cm) for b in bR]
            xD = [D.xi_for(TAU, b.N2, b.sigma2, cm) for b in bD]
            for arm in ARMS:
                xr = xR if arm != "plugin" else [0.0] * len(xR)
                xd = xD if arm != "plugin" else [0.0] * len(xD)
                vals = {n: arm_value(arm, bR, bD, xr, xd, D.make_vg(pi, p_o1))
                        for n, pi in cands.items()}
                pick = max(vals, key=vals.get)
                pk[arm].append(pick)
                acc[arm].append(v_true[best] - v_true[pick])
        for arm in ARMS:
            r = np.array(acc[arm])
            out[arm].append((float(r.mean()),
                             float(1.96 * r.std() / np.sqrt(len(r)))))
            picks[arm].append(max(set(pk[arm]), key=pk[arm].count))
    return dict(best=best, spread=spread, v_true=v_true, regret=out,
                modal=picks)


def slope(ys):
    y = np.maximum(ys, 1e-12)
    return float(np.polyfit(np.log(N_GRID), np.log(y), 1)[0])


def main():
    print("(%d,%d,%d) cf=%s T=%d, schedule C (e=%+.1f), %d seeds"
          % (D.N_S, D.N_O, D.N_O0, D.CONFOUND, D.T, TAU + KAP - 1, len(SEEDS)))
    res = {}
    for tag, wb in (("six candidates", False), ("seven, with bc", True)):
        r = run(wb)
        res[tag] = r
        print("\n--- %s ---   optimal %s   spread %.4f"
              % (tag, r["best"], r["spread"]))
        print("%10s%12s%12s%12s%12s%10s%14s"
              % ("arm", *["N=%d" % n for n in N_GRID], "slope", "modal@max"))
        for arm in ARMS:
            ys = [v[0] for v in r["regret"][arm]]
            print("%10s%12s%10.4f%14s"
                  % (arm, "".join("%12.4f" % y for y in ys), slope(ys),
                     r["modal"][arm][-1]))

    print("\n" + "=" * 88)
    a = res["six candidates"]["regret"]["full"]
    b = res["seven, with bc"]["regret"]["full"]
    sa, sb = slope([v[0] for v in a]), slope([v[0] for v in b])
    print("P-L13  full's slope: %+.4f without bc,  %+.4f with bc" % (sa, sb))
    print("       the failure survives: %s" % (sb > 0))
    print("       bc margin over the old best: %.4f  against spread %.4f (%.2f%%)"
          % (res["seven, with bc"]["v_true"]["bc"]
             - res["six candidates"]["v_true"][res["six candidates"]["best"]],
             res["six candidates"]["spread"],
             100 * (res["seven, with bc"]["v_true"]["bc"]
                    - res["six candidates"]["v_true"][
                        res["six candidates"]["best"]])
             / res["six candidates"]["spread"]))

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_bc_candidate_retest.json"))
    with open(out, "w") as f:
        json.dump(dict(results=res, N_grid=N_GRID, seeds=SEEDS,
                       slope_without_bc=sa, slope_with_bc=sb),
                  f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
