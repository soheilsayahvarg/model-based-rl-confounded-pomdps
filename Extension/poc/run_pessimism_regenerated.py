"""run_pessimism_regenerated.py -- every pessimism number, regenerated correctly.

This supersedes run_repair_vs_normball.py, whose magnitudes were invalidated by
three defects found in review:

  * the norm-ball solver scaled one fixed direction to the ELLIPSOID boundary,
    up to 56% suboptimal and occasionally outside the ball. Replaced by the
    certified two-multiplier solver now in ellipsoid_opt.
  * M was averaged across blocks and applied to all of them, putting some blocks'
    own centres outside their ball. Now PER BLOCK.
  * M was set to ||b_hat||, argued to be the smallest admissible value. Ridge
    shrinks the estimate, so ||b_hat|| < ||b_true|| in every block/seed pair
    tested and that ball EXCLUDED the truth. Now set from ||b_true|| via the
    exact oracle bridges, which is admissible by construction.

It also fixes two reporting faults in the phase write-ups:

  * they compare across different c (vanilla at 0.1 against projected at 0.03).
    Everything here runs on a COMMON c grid.
  * they omit the plug-in baseline, which our own frozen output shows has 0.000
    regret. It is reported alongside, since a pessimistic method that loses to
    the plug-in has not earned its complexity.

Writes experiments/results_pessimism_regenerated.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
import oracle_module as om
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid, pessimistic_value

N_MAIN = 20000                      # Phase 3's operating point
SEEDS = [0, 1, 2]
C_GRID = [0.03, 0.1, 0.3, 1.0]      # common grid, spans Phase 3's 0.03
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
    return {
        "greedy_lo": np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        "greedy_hi": np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]),
        "always_0":  np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        "always_1":  np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        "uniform":   np.full((3, 2), 0.5),
        "soft":      np.array([[0.7, 0.3], [0.5, 0.5], [0.3, 0.7]]),
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

    # The toy's true bridges are time-homogeneous: one (A,O,r,O) tensor for the
    # reward family and one (A,O,O,O) tensor for the dynamics family, shared by
    # every stage. So every R block gets the same admissible bound, and likewise
    # for D. M >= ||b_true|| then holds per block by construction.
    bR_true, bD_true, _ = om.toy_true_bridges(params)
    MR_true = [float(np.linalg.norm(np.asarray(bR_true).ravel()))] * T
    MD_true = [float(np.linalg.norm(np.asarray(bD_true).ravel()))] * (T - 1)
    print("admissible per-block M from the EXACT bridges (M >= ||b_true|| holds "
          "by construction)")
    print("   R blocks:", [f"{m:.3f}" for m in MR_true])
    print("   D blocks:", [f"{m:.3f}" for m in MD_true])
    print(f"\noptimal policy: {best}  V_true = {v_true[best]:.4f}\n")

    print(f"{'c':>6}{'method':>13}{'selected':>11}{'regret':>9}"
          f"{'V_low':>11}{'contains truth':>16}")
    rows = []
    for c in C_GRID:
        sel = {m: [] for m in ("plug-in", "vanilla", "norm-ball", "projected")}
        vlows_best = {m: [] for m in sel}
        for sd in SEEDS:
            d = toy.sample_trajectories(params, N_MAIN, np.random.default_rng(sd))
            est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                         n_o0=toy.N_O0, n_r=2, T=T,
                                         mode="primal", seed=sd)
            est.fit(d["O0"], d["O"], d["A"], d["R"])
            blocksR, blocksD = build_blocks(est, T)
            p_o1 = empirical_p_o1(d["O"], toy.N_O)
            xR = [b.width_rule(c) for b in blocksR]
            xD = [b.width_rule(c) for b in blocksD]

            v = {m: {} for m in sel}
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

                bRh = [b.b_hat for b in blocksR]
                bDh = [b.b_hat for b in blocksD]
                v["plug-in"][name] = vg(bRh, bDh)[0]
                v["vanilla"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd))["V_low"]
                try:
                    v["norm-ball"][name] = pessimistic_value(
                        blocksR, blocksD, xR, xD, vg,
                        rng=np.random.default_rng(sd),
                        M_R=MR_true, M_D=MD_true)["V_low"]
                except (ValueError, TypeError):
                    v["norm-ball"][name] = float("nan")
                v["projected"][name] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd), projected=True)["V_low"]

            for m in sel:
                if any(np.isnan(x) for x in v[m].values()):
                    sel[m].append(None)
                    vlows_best[m].append(float("nan"))
                    continue
                pick = max(v[m], key=v[m].get)
                sel[m].append(pick)
                vlows_best[m].append(v[m][pick])

        for m in ("plug-in", "vanilla", "norm-ball", "projected"):
            picks = [p for p in sel[m] if p is not None]
            if not picks:
                print(f"{c:>6.2f}{m:>13}{'n/a':>11}")
                continue
            reg = float(np.mean([v_true[best] - v_true[p] for p in picks]))
            modal = max(set(picks), key=picks.count)
            truth = {"plug-in": "n/a", "vanilla": "yes (no class)",
                     "norm-ball": "yes", "projected": "NO"}[m]
            print(f"{c:>6.2f}{m:>13}{modal:>11}{reg:>9.4f}"
                  f"{np.nanmean(vlows_best[m]):>11.3f}{truth:>16}")
            rows.append(dict(c=c, method=m, modal=modal, regret=reg,
                             vlow=float(np.nanmean(vlows_best[m])),
                             contains_truth=truth))

    RESULTS["rows"] = rows
    RESULTS["v_true"] = v_true
    RESULTS["optimal"] = best
    RESULTS["M_R_true"] = MR_true
    RESULTS["M_D_true"] = MD_true
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_pessimism_regenerated.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
