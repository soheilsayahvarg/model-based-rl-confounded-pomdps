"""critic_p_a1.py -- ATTACK A1: is D4's candidate set adversarial, and does
Phase 3's 0.000 reproduce under current code?

Three measurements, all on the toy POMDP (N=5,000, 3 seeds, same fits as the
published comparison):

1. Phase 3's ORIGINAL candidate set {always_a0, always_a1, obs_dependent,
   uniform} run through the current comparison code. If projected selection is
   regret-0 there, the Extension's non-zero regret is candidate-set dependence,
   not a code change.

2. The DISTRIBUTION of regret over 200 random candidate sets (5 policies drawn
   from a pool of 8 deterministic obs->action maps + 40 Dirichlet-soft
   policies), for vanilla / norm-ball / projected pessimistic selection AND
   plain plug-in selection (argmax V_hat), which every write-up so far omits.
   V_low is computed once per (policy, method, c, seed) and reused across sets,
   so the 200 sets cost nothing extra.

3. The candidate-set-free statement: mean regret per method across the
   distribution, and head-to-head win/tie/loss counts.
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
N_SOFT = 40
N_SETS = 200
SET_SIZE = 5
RESULTS = {}


def dp_value_toy(params, pi_obs, T=None):
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    T = T or params["T"]
    V = np.zeros(toy.N_S)
    for _ in range(T):
        Q = pR + np.einsum("asz,z->sa", P, V)
        V = np.einsum("sa,sa->s", E @ pi_obs, Q)
    return float(p1 @ V)


def phase3_candidates():
    always0 = np.zeros((toy.N_O, toy.N_A)); always0[:, 0] = 1.0
    always1 = np.zeros((toy.N_O, toy.N_A)); always1[:, 1] = 1.0
    obs_dep = np.zeros((toy.N_O, toy.N_A))
    obs_dep[[0, 1], 0] = 1.0
    obs_dep[2, 1] = 1.0
    uniform = np.full((toy.N_O, toy.N_A), 0.5)
    return {"always_a0": always0, "always_a1": always1,
            "obs_dependent": obs_dep, "uniform": uniform}


def policy_pool(rng):
    pool = {}
    for code in range(toy.N_A ** toy.N_O):              # 8 deterministic maps
        pi = np.zeros((toy.N_O, toy.N_A))
        c = code
        for o in range(toy.N_O):
            pi[o, c % toy.N_A] = 1.0
            c //= toy.N_A
        pool[f"det_{code}"] = pi
    for i in range(N_SOFT):
        pool[f"soft_{i}"] = rng.dirichlet(np.ones(toy.N_A), size=toy.N_O)
    return pool


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
    rng = np.random.default_rng(7)
    pool = policy_pool(rng)
    pool.update(phase3_candidates())                    # share the cache
    v_true = {k: dp_value_toy(params, pi) for k, pi in pool.items()}

    # ---------- V_low / V_hat cache over the whole pool
    print(f"caching V_low for {len(pool)} policies x 3 methods x "
          f"{len(C_GRID)} c x {len(SEEDS)} seeds ...")
    cache = {}                    # (name, method, c, seed) -> V_low
    vhat = {}                     # (name, seed) -> plug-in value
    for sd in SEEDS:
        d = toy.sample_trajectories(params, N_MAIN, np.random.default_rng(sd))
        est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                     n_o0=toy.N_O0, n_r=2, T=T,
                                     mode="primal", seed=sd)
        est.fit(d["O0"], d["O"], d["A"], d["R"])
        blocksR, blocksD = build_blocks(est, T)
        p_o1 = empirical_p_o1(d["O"], toy.N_O)
        M = float(np.mean([np.linalg.norm(b.b_hat) for b in blocksR]))
        for name, pi in pool.items():
            def vg(bRl, bDl, pi=pi):
                bR = np.stack([b.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                               for b in bRl])
                bD = np.stack([b.reshape(toy.N_A, toy.N_O, toy.N_O, toy.N_O)
                               for b in bDl])
                V, _p, (gR, gD) = plugin_value(bR, bD, pi, p_o1,
                                               return_grads=True)
                return (V, [gR[t].ravel() for t in range(T)],
                        [gD[j].ravel() for j in range(T - 1)])

            vhat[(name, sd)] = vg([b.b_hat for b in blocksR],
                                  [b.b_hat for b in blocksD])[0]
            for c in C_GRID:
                xR = [b.width_rule(c) for b in blocksR]
                xD = [b.width_rule(c) for b in blocksD]
                cache[(name, "vanilla", c, sd)] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd))["V_low"]
                cache[(name, "norm-ball", c, sd)] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd), M_R=M, M_D=M)["V_low"]
                cache[(name, "projected", c, sd)] = pessimistic_value(
                    blocksR, blocksD, xR, xD, vg,
                    rng=np.random.default_rng(sd), projected=True)["V_low"]
        print(f"  seed {sd} done")

    def regret_on(cand_names, method, c):
        best = max(cand_names, key=lambda k: v_true[k])
        regs = []
        for sd in SEEDS:
            if method == "plug-in":
                pick = max(cand_names, key=lambda k: vhat[(k, sd)])
            else:
                pick = max(cand_names, key=lambda k: cache[(k, method, c, sd)])
            regs.append(v_true[best] - v_true[pick])
        return float(np.mean(regs))

    # ---------- 1. Phase 3's original candidate set
    p3 = list(phase3_candidates())
    print("\n[1] Phase 3's ORIGINAL candidate set under current code")
    print("    true values: " + "  ".join(f"{k}={v_true[k]:.4f}" for k in p3))
    print(f"    {'c':>5}{'vanilla':>10}{'norm-ball':>11}{'projected':>11}"
          f"{'plug-in':>10}")
    for c in C_GRID:
        row = [regret_on(p3, m, c)
               for m in ("vanilla", "norm-ball", "projected", "plug-in")]
        print(f"    {c:>5.1f}" + "".join(f"{r:>10.4f}" for r in row[:3])
              + f"{row[3]:>10.4f}")
        RESULTS[f"phase3_set_c{c}"] = dict(zip(
            ("vanilla", "norm-ball", "projected", "plug-in"), row))

    # ---------- 2. distribution over random candidate sets
    print(f"\n[2] {N_SETS} random candidate sets of {SET_SIZE} "
          f"(8 deterministic + {N_SOFT} soft pool)")
    pool_names = [k for k in pool if k.startswith(("det_", "soft_"))]
    sets = [list(rng.choice(pool_names, size=SET_SIZE, replace=False))
            for _ in range(N_SETS)]
    methods = ("vanilla", "norm-ball", "projected", "plug-in")
    dist = {}
    for c in C_GRID:
        print(f"\n    c={c}: {'method':>10}{'mean':>8}{'median':>8}"
              f"{'P(=0)':>8}{'P(>0.2)':>9}")
        for m in methods:
            rs = np.array([regret_on(s, m, c) for s in sets])
            dist[(m, c)] = rs
            print(f"         {m:>10}{rs.mean():>8.4f}{np.median(rs):>8.4f}"
                  f"{np.mean(rs < 1e-9):>8.2%}{np.mean(rs > 0.2):>9.2%}")
            RESULTS[f"dist_{m}_c{c}"] = dict(
                mean=float(rs.mean()), median=float(np.median(rs)),
                p_zero=float(np.mean(rs < 1e-9)),
                p_gt02=float(np.mean(rs > 0.2)))
        pv = dist[("projected", c)]
        nb = dist[("norm-ball", c)]
        pl = dist[("plug-in", c)]
        print(f"         projected vs norm-ball: wins "
              f"{np.mean(pv < nb - 1e-9):.0%}, ties "
              f"{np.mean(abs(pv - nb) < 1e-9):.0%}, losses "
              f"{np.mean(pv > nb + 1e-9):.0%}")
        print(f"         plug-in beats ALL pessimism: "
              f"{np.mean((pl <= pv + 1e-9) & (pl <= nb + 1e-9)):.0%} of sets")

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic_p_a1.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
