"""run_repair_vs_normball.py -- is our repair still needed once M_R is imposed?

THE DEBT. Phase 3 and Phase 4 both report Signal-Projected Pessimism against the
VANILLA baseline: vanilla diverges to -1755, the projection restores informative
values and achieves 0.000 selection regret. But that vanilla baseline omitted the
anchor paper's bridge-class bound M_R. Comparing a repair against a broken
baseline flatters the repair.

The honest comparison is against the paper's construction implemented correctly:
the confidence region INTERSECTED with the bridge class. This script runs all
three on identical fits.

WHY SELECTION REGRET IS THE TEST THAT MATTERS. Informativeness of V_low is a
means; the published claim is about ranking policies. If the norm-constrained
baseline also achieves zero regret, the projection contributes nothing to the
headline result, and a component we present as a contribution is unnecessary.

SOUNDNESS ASYMMETRY, worth keeping in view while reading the output. The norm
ball provably CONTAINS the truth, since M_R >= ||b_true|| is exactly the paper's
assumption. Our projection provably EXCLUDES part of it -- Phase 4 measured 21%
of the true bridge's norm outside the retained subspace. So equal regret is not a
tie: the sound method wins.

Writes experiments/results_repair_vs_normball.json.
"""

import json
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

N_MAIN = 5000
SEEDS = [0, 1, 2]
C_GRID = [0.1, 0.5, 1.0]
RESULTS = {}


def dp_value_toy(params, pi_obs, T=None):
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    T = T or params["T"]
    V = np.zeros(toy.N_S)
    for _ in range(T):
        Q = pR + np.einsum("asz,z->sa", P, V)
        V = np.einsum("sa,sa->s", E @ pi_obs, Q)
    return float(p1 @ V)


def candidates():
    """Observation-based policies spanning a range of true values."""
    return {
        "greedy_lo":  np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        "greedy_hi":  np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]),
        "always_0":   np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        "always_1":   np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        "uniform":    np.full((3, 2), 0.5),
        "soft":       np.array([[0.7, 0.3], [0.5, 0.5], [0.3, 0.7]]),
    }


def build_blocks(est, T):
    bR, bD = [], []
    for t in range(1, T + 1):
        s = est.stage_store[f"bR_t{t}"]
        bR.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                 s["N2"], f"bR_t{t}", s["signal_basis"],
                                 n_obs=toy.N_O, n_act=toy.N_A, n_y=s["n_y"]))
        if t < T:
            s = est.stage_store[f"bD_t{t}"]
            bD.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                     s["N2"], f"bD_t{t}", s["signal_basis"],
                                     n_obs=toy.N_O, n_act=toy.N_A, n_y=s["n_y"]))
    return bR, bD


def main():
    params = toy.default_params(T=3)
    T = params["T"]
    cands = candidates()
    v_true = {k: dp_value_toy(params, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    print("toy POMDP; exact policy values")
    for k, v in sorted(v_true.items(), key=lambda x: -x[1]):
        print(f"   {k:>11}  {v:.4f}" + ("   <- optimal" if k == best else ""))
    print()
    print(f"{'c':>6}{'method':>16}{'selected':>12}{'regret':>9}"
          f"{'V_low(best)':>13}{'sound?':>9}")

    rows = []
    for c in C_GRID:
        sel = {m: [] for m in ("vanilla", "norm-ball", "projected")}
        vlow_best = {m: [] for m in sel}
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            d = toy.sample_trajectories(params, N_MAIN, rng)
            est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                         n_o0=toy.N_O0, n_r=2, T=T,
                                         mode="primal", seed=sd)
            est.fit(d["O0"], d["O"], d["A"], d["R"])
            blocksR, blocksD = build_blocks(est, T)
            p_o1 = empirical_p_o1(d["O"], toy.N_O)
            M = float(np.mean([np.linalg.norm(b.b_hat) for b in blocksR]))
            xR = [b.width_rule(c) for b in blocksR]
            xD = [b.width_rule(c) for b in blocksD]

            vlows = {m: {} for m in sel}
            for name, pi in cands.items():
                def vg(bRl, bDl):
                    bR = np.stack([b.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                                   for b in bRl])
                    bD = np.stack([b.reshape(toy.N_A, toy.N_O, toy.N_O, toy.N_O)
                                   for b in bDl])
                    V, _p, (gR, gD) = plugin_value(bR, bD, pi, p_o1,
                                                   return_grads=True)
                    return (V, [gR[t].ravel() for t in range(T)],
                            [gD[j].ravel() for j in range(T - 1)])

                vlows["vanilla"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd))["V_low"]
                vlows["norm-ball"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd), M_R=M, M_D=M)["V_low"]
                vlows["projected"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd), projected=True)["V_low"]

            for m in sel:
                pick = max(vlows[m], key=vlows[m].get)
                sel[m].append(pick)
                vlow_best[m].append(vlows[m][pick])

        for m in ("vanilla", "norm-ball", "projected"):
            regrets = [v_true[best] - v_true[p] for p in sel[m]]
            modal = max(set(sel[m]), key=sel[m].count)
            sound = {"vanilla": "yes*", "norm-ball": "yes",
                     "projected": "NO"}[m]
            print(f"{c:>6.1f}{m:>16}{modal:>12}{np.mean(regrets):>9.4f}"
                  f"{np.mean(vlow_best[m]):>13.2f}{sound:>9}")
            rows.append(dict(c=c, method=m, modal_selection=modal,
                             regret=float(np.mean(regrets)),
                             vlow_selected=float(np.mean(vlow_best[m])),
                             contains_truth=sound))
    print("\n* vanilla contains the truth but omits the paper's bridge class,")
    print("  so it is the paper's method with a constraint dropped, not the")
    print("  paper's method. 'projected' provably excludes 21% of the truth's norm.")

    RESULTS["comparison"] = rows
    RESULTS["v_true"] = v_true
    RESULTS["optimal"] = best
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_repair_vs_normball.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
