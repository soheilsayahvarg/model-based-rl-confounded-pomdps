"""critic_p_a1b.py -- A1 follow-up: is the 0.000 non-reproduction an N mismatch?

Phase 3's frozen run_phase34_check.py uses N_PESS = 20,000 and 5 seeds; the
Extension comparison (and critic_p_a1) ran N = 5,000 and 3 seeds. Re-run the
original 4-candidate set with the Extension code at N = 20,000, 5 seeds.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid, pessimistic_value
from critic_p_a1 import phase3_candidates, dp_value_toy, build_blocks

N = 20000
SEEDS = [0, 1, 2, 3, 4]
T = 3


def main():
    params = toy.default_params(T=3)
    cands = phase3_candidates()
    v_true = {k: dp_value_toy(params, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    print("original Phase 3 candidates, Extension code, N=20,000, 5 seeds")
    print("   " + "  ".join(f"{k}={v:.4f}" for k, v in v_true.items()))
    for c in (0.1, 1.0):
        sel = {m: [] for m in ("vanilla", "norm-ball", "projected", "plug-in")}
        for sd in SEEDS:
            d = toy.sample_trajectories(params, N, np.random.default_rng(sd))
            est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                         n_o0=toy.N_O0, n_r=2, T=T,
                                         mode="primal", seed=sd)
            est.fit(d["O0"], d["O"], d["A"], d["R"])
            blocksR, blocksD = build_blocks(est, T)
            p_o1 = empirical_p_o1(d["O"], toy.N_O)
            M = float(np.mean([np.linalg.norm(b.b_hat) for b in blocksR]))
            vlows = {m: {} for m in sel}
            for name, pi in cands.items():
                def vg(bRl, bDl, pi=pi):
                    R = np.stack([b.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                                  for b in bRl])
                    D = np.stack([b.reshape(toy.N_A, toy.N_O, toy.N_O, toy.N_O)
                                  for b in bDl])
                    V, _p, (gR, gD) = plugin_value(R, D, pi, p_o1,
                                                   return_grads=True)
                    return (V, [gR[t].ravel() for t in range(T)],
                            [gD[j].ravel() for j in range(T - 1)])

                xR = [b.width_rule(c) for b in blocksR]
                xD = [b.width_rule(c) for b in blocksD]
                vlows["vanilla"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd))["V_low"]
                vlows["norm-ball"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd), M_R=M, M_D=M)["V_low"]
                vlows["projected"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd), projected=True)["V_low"]
                vlows["plug-in"][name] = vg([b.b_hat for b in blocksR],
                                            [b.b_hat for b in blocksD])[0]
            for m in sel:
                sel[m].append(max(vlows[m], key=vlows[m].get))
        print(f"  c={c}:")
        for m, picks in sel.items():
            regs = [v_true[best] - v_true[p] for p in picks]
            print(f"    {m:>10}: picks={picks}  mean regret={np.mean(regs):.4f}")


if __name__ == "__main__":
    main()
