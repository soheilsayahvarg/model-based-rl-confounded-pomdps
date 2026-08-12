"""run_mb_norm_constraint.py -- the omitted M_R, on the MODEL-BASED side.

The model-free check (run_norm_constraint.py) showed we omit the paper's bridge-
class bound and that the divergence survives imposing it. But that layer was ours.
Here M_R is the ANCHOR PAPER'S OWN SYMBOL, carried through Theorem 4.2, and the
environment is the toy POMDP where our published pessimism divergence (-1755,
Phase 3; -1779, Phase 4) was measured. This is where the omission, if it matters,
would invalidate results we have already reported.

WHAT M_R SHOULD BE. The paper requires the true bridge to lie inside the class, so
any admissible M_R is at least ||b_true||. We sweep multiples of the fitted bridge
norm, which estimates that scale, and report where the ball becomes active. Using
a multiple below 1.0 is deliberately too tight -- it excludes the truth and is
shown only to bracket the behaviour.

Writes experiments/results_mb_norm_constraint.json.
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
C_GRID = [0.1, 1.0]
M_MULT = [0.5, 1.0, 2.0, 5.0, None]      # None = no ball (what we shipped)
RESULTS = {}


def dp_value_toy(params, pi_obs, T=None):
    """Exact V(pi) for an observation-based policy on the toy POMDP."""
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    T = T or params["T"]
    V = np.zeros(toy.N_S)
    for _ in range(T):
        Q = pR + np.einsum("asz,z->sa", P, V)
        V = np.einsum("sa,sa->s", E @ pi_obs, Q)
    return float(p1 @ V)


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
    pi = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])     # observation policy
    v_true = dp_value_toy(params, pi)
    print(f"toy POMDP, |S|={toy.N_S} |O|={toy.N_O} |O_0|={toy.N_O0} T={T}")
    print(f"V_true = {v_true:.4f}\n")
    print(f"{'c':>6}{'M/||b_hat||':>13}{'V_low':>12}{'V_low proj':>13}"
          f"{'gap to truth':>14}")

    rows = []
    for c in C_GRID:
        for mm in M_MULT:
            vl, vlp, active = [], [], []
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                d = toy.sample_trajectories(params, N_MAIN, rng)
                est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                             n_o0=toy.N_O0, n_r=2, T=T,
                                             mode="primal", seed=sd)
                est.fit(d["O0"], d["O"], d["A"], d["R"])
                blocksR, blocksD = build_blocks(est, T)
                p_o1 = empirical_p_o1(d["O"], toy.N_O)

                def vg(bRl, bDl):
                    bR = np.stack([b.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                                   for b in bRl])
                    bD = np.stack([b.reshape(toy.N_A, toy.N_O, toy.N_O, toy.N_O)
                                   for b in bDl])
                    V, _per, (gR, gD) = plugin_value(bR, bD, pi, p_o1,
                                                     return_grads=True)
                    return (V, [gR[t].ravel() for t in range(T)],
                            [gD[j].ravel() for j in range(T - 1)])

                nb = float(np.mean([np.linalg.norm(b.b_hat) for b in blocksR]))
                M = None if mm is None else mm * nb
                xR = [b.width_rule(c) for b in blocksR]
                xD = [b.width_rule(c) for b in blocksD]
                r = pessimistic_value(blocksR, blocksD, xR, xD, vg,
                                      rng=np.random.default_rng(sd),
                                      M_R=M, M_D=M)
                rp = pessimistic_value(blocksR, blocksD, xR, xD, vg,
                                       rng=np.random.default_rng(sd),
                                       projected=True)
                vl.append(r["V_low"])
                vlp.append(rp["V_low"])
                active.append(M is not None)
            lbl = "no ball" if mm is None else f"{mm:g}"
            mv = float(np.mean(vl))
            print(f"{c:>6.1f}{lbl:>13}{mv:>12.2f}{np.mean(vlp):>13.2f}"
                  f"{v_true - mv:>14.2f}")
            rows.append(dict(c=c, M_mult=mm, v_true=v_true, v_low=mv,
                             v_low_projected=float(np.mean(vlp))))
    RESULTS["mb_norm_constraint"] = rows
    RESULTS["v_true"] = v_true

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_mb_norm_constraint.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
