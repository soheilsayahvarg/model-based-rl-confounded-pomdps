"""critic_p_a1c.py -- A1 close-out: run BOTH candidate sets at the paper's own
operating point (c = 0.03) and at the Extension's (c = 0.1, 1.0), N = 20,000
and 5,000. If the Extension's 6-candidate set also yields ~0.000 regret at
c = 0.03, the 'does not reproduce on a different candidate set' claim was a
c-grid artifact, not candidate-set fragility.
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

T = 3
SEEDS = [0, 1, 2, 3, 4]


def extension_candidates():
    return {
        "greedy_lo":  np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        "greedy_hi":  np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]),
        "always_0":   np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        "always_1":   np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        "uniform":    np.full((3, 2), 0.5),
        "soft":       np.array([[0.7, 0.3], [0.5, 0.5], [0.3, 0.7]]),
    }


def main():
    params = toy.default_params(T=3)
    for set_name, cands in (("phase3-original", phase3_candidates()),
                            ("extension-new", extension_candidates())):
        v_true = {k: dp_value_toy(params, pi) for k, pi in cands.items()}
        best = max(v_true, key=v_true.get)
        for N in (20000, 5000):
            for c in (0.03, 0.1, 1.0):
                regs = {m: [] for m in ("vanilla", "projected", "plug-in")}
                for sd in SEEDS:
                    d = toy.sample_trajectories(params, N,
                                                np.random.default_rng(sd))
                    est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                                 n_o0=toy.N_O0, n_r=2, T=T,
                                                 mode="primal", seed=sd)
                    est.fit(d["O0"], d["O"], d["A"], d["R"])
                    blocksR, blocksD = build_blocks(est, T)
                    p_o1 = empirical_p_o1(d["O"], toy.N_O)
                    vl = {m: {} for m in regs}
                    for name, pi in cands.items():
                        def vg(bRl, bDl, pi=pi):
                            R = np.stack([b.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                                          for b in bRl])
                            D = np.stack([b.reshape(toy.N_A, toy.N_O, toy.N_O,
                                                    toy.N_O) for b in bDl])
                            V, _p, (gR, gD) = plugin_value(R, D, pi, p_o1,
                                                           return_grads=True)
                            return (V, [gR[t].ravel() for t in range(T)],
                                    [gD[j].ravel() for j in range(T - 1)])

                        xR = [b.width_rule(c) for b in blocksR]
                        xD = [b.width_rule(c) for b in blocksD]
                        vl["vanilla"][name] = pessimistic_value(
                            blocksR, blocksD, xR, xD, vg,
                            rng=np.random.default_rng(sd))["V_low"]
                        vl["projected"][name] = pessimistic_value(
                            blocksR, blocksD, xR, xD, vg,
                            rng=np.random.default_rng(sd),
                            projected=True)["V_low"]
                        vl["plug-in"][name] = vg(
                            [b.b_hat for b in blocksR],
                            [b.b_hat for b in blocksD])[0]
                    for m in regs:
                        pick = max(vl[m], key=vl[m].get)
                        regs[m].append(v_true[best] - v_true[pick])
                print(f"{set_name:>17}  N={N:>6}  c={c:<5}"
                      + "".join(f"  {m}={np.mean(v):.4f}"
                                for m, v in regs.items()))


if __name__ == "__main__":
    main()
