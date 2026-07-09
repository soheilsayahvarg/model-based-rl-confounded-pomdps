"""
run_phase2_check.py — Phase 2 estimation-core validation (NO pessimism here).

Checks, in order:
  [P0] POPULATION-LIMIT identification test: with exact mu(w|x), p(y|x) and the
       exact uniform-index design, the stage-2 minimum-norm solve must reproduce
       the oracle pinv bridges at machine precision — separates structural bugs
       from finite-N statistics. (All stages, both families.)
  [P1] PRIMAL/DUAL equivalence: alpha = (G + N2 lam2 I)^{-1} p_hat representer
       solution equals the operator solution (T2 + lam2 I)^{-1} g2 to ~1e-12.
  [P2] N-SCALING SWEEP (N = 1k, 5k, 10k, 20k; 3 seeds): L2 recovery error of
       b_R, b_D vs oracle bridges must DECREASE with N; condition numbers of the
       regularized design logged per stage/family.
  [P3] O_0-NOISE (ill-posedness) SWEEP at N = 10k: K0(nu) = [[1-nu, nu],[nu, 1-nu]];
       as nu -> 0.5 the negative control loses latent information, completeness
       degrades (stage 1 is hit hardest: X_1 = (a_1, o_0) has no other latent
       channel), and errors/conditioning must reflect it. Oracle bridges do NOT
       depend on K0, so the reference is unchanged across the sweep.

Writes experiments/results_phase2.json.
"""

import itertools
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "estimation")))

import toy_pomdp as toy
import oracle_module as oracle
from bridge_estimator import TabularBridgeEstimator

RESULTS = {}
N_LIST = [1000, 5000, 10000, 20000]
SEEDS = [0, 1, 2]
NU_LIST = [0.05, 0.15, 0.30, 0.45]


# ------------------------------------------------------------------ error helper
def bridge_errors(est, bRs, bDs, T=3):
    bR, bD = est.bridges()
    eR = [float(np.linalg.norm(bR[t] - bRs)) for t in range(T)]
    eD = [float(np.linalg.norm(bD[t] - bDs)) for t in range(T - 1)]
    return dict(errR_per_stage=eR, errD_per_stage=eD,
                errR_total=float(np.sqrt(np.sum(np.square(eR)))),
                errD_total=float(np.sqrt(np.sum(np.square(eD)))))


# ------------------------------------------------------------------ [P0] population
def population_limit_check(params):
    """Exact-population stage-2 solve vs oracle pinv bridges (all stages/families)."""
    E, P, pR, p1, pi_b, K0, T = (params[k] for k in
                                 ("E", "P", "pR", "p1", "pi_b", "K0", "T"))
    nS, nA, nO, nO0, nR = toy.N_S, toy.N_A, toy.N_O, toy.N_O0, 2
    bRs, bDs, _ = oracle.toy_true_bridges(params)

    paths = {}
    for s1, o0, o1, a1 in itertools.product(range(nS), range(nO0), range(nO), range(nA)):
        pr0 = p1[s1] * K0[s1, o0] * E[s1, o1] * pi_b[s1, a1]
        for s2, o2, a2 in itertools.product(range(nS), range(nO), range(nA)):
            pr1 = pr0 * P[a1][s1, s2] * E[s2, o2] * pi_b[s2, a2]
            for s3, o3, a3 in itertools.product(range(nS), range(nO), range(nA)):
                paths[(s1, o0, o1, a1, s2, o2, a2, s3, o3, a3)] = \
                    pr1 * P[a2][s2, s3] * E[s3, o3] * pi_b[s3, a3]
    assert abs(sum(paths.values()) - 1.0) < 1e-12

    w_list = list(itertools.product(range(nA), range(nO)))
    y_list = list(itertools.product(range(nR), range(nO)))
    z_list = list(itertools.product(range(nO), range(nO)))

    def joint(t):
        Px, Pwx, Pyx, Pzx = {}, {}, {}, {}
        for path, pr in paths.items():
            s1, o0, o1, a1, s2, o2, a2, s3, o3, a3 = path
            st = (s1, s2, s3)[t - 1]; ot = (o1, o2, o3)[t - 1]; at = (a1, a2, a3)[t - 1]
            hist = () if t == 1 else ((o1, a1) if t == 2 else (o1, a1, o2, a2))
            x = (at, hist, o0)
            Px[x] = Px.get(x, 0.0) + pr
            Pwx[((at, ot), x)] = Pwx.get(((at, ot), x), 0.0) + pr
            for r in range(nR):
                pr_r = pR[st, at] if r == 1 else 1 - pR[st, at]
                Pyx[((r, ot), x)] = Pyx.get(((r, ot), x), 0.0) + pr * pr_r
            if t < T:
                snext = (s2, s3)[t - 1]
                for onext in range(nO):
                    Pzx[((onext, ot), x)] = Pzx.get(((onext, ot), x), 0.0) \
                        + pr * E[snext, onext]
        return Px, Pwx, Pyx, Pzx

    def pop_solve(Px, Pwx, Ptgt, tgt_list):
        nw, ny = len(w_list), len(tgt_list)
        w_id = {w: i for i, w in enumerate(w_list)}
        t_id = {y: i for i, y in enumerate(tgt_list)}
        D = nw * ny
        T2, g = np.zeros((D, D)), np.zeros(D)
        for x, px in Px.items():
            mu = np.zeros(nw); tgt = np.zeros(ny)
            for (w, xx), p in Pwx.items():
                if xx == x: mu[w_id[w]] = p / px
            for (y, xx), p in Ptgt.items():
                if xx == x: tgt[t_id[y]] = p / px
            for yi in range(ny):
                psi = np.zeros(D)
                psi[np.arange(nw) * ny + yi] = mu
                T2 += px / ny * np.outer(psi, psi)
                g += px / ny * psi * tgt[yi]
        return (np.linalg.pinv(T2, rcond=1e-9) @ g).reshape(nw, ny)

    worst = 0.0
    for t in range(1, T + 1):
        Px, Pwx, Pyx, Pzx = joint(t)
        bR_pop = pop_solve(Px, Pwx, Pyx, y_list)
        ref = np.array([[bRs[a, ot, r, o] for (r, o) in y_list] for (a, ot) in w_list])
        worst = max(worst, float(np.abs(bR_pop - ref).max()))
        if t < T:
            bD_pop = pop_solve(Px, Pwx, Pzx, z_list)
            refD = np.array([[bDs[a, ot, op, o] for (op, o) in z_list]
                             for (a, ot) in w_list])
            worst = max(worst, float(np.abs(bD_pop - refD).max()))
    return worst


# ------------------------------------------------------------------ main
def main():
    params = toy.default_params(T=3)
    bRs, bDs, _ = oracle.toy_true_bridges(params)

    # [P0]
    worst_pop = population_limit_check(params)
    RESULTS["P0_population_limit_max_gap"] = worst_pop
    print(f"[P0] population-limit identification: max |b_pop - b_oracle| = {worst_pop:.3e}")
    assert worst_pop < 1e-12, "population-limit identification FAILED"

    # [P1] primal/dual equivalence at N=1000
    data = toy.sample_trajectories(params, 1000, np.random.default_rng(42))
    est = TabularBridgeEstimator(3, 2, 2, 2, 3, mode="dual", seed=42)
    est.fit(data["O0"], data["O"], data["A"], data["R"])
    gaps = [d["primal_dual_gap"] for d in est.diagnostics]
    RESULTS["P1_primal_dual_max_gap"] = max(gaps)
    print(f"[P1] primal/dual equivalence: max gap = {max(gaps):.3e}")
    assert max(gaps) < 1e-10, "primal/dual closed forms disagree"

    # [P2] N-scaling sweep
    sweep = {}
    for N in N_LIST:
        per_seed = []
        conds = []
        for seed in SEEDS:
            data = toy.sample_trajectories(params, N, np.random.default_rng(1000 + seed))
            est = TabularBridgeEstimator(3, 2, 2, 2, 3, mode="auto", seed=seed)
            est.fit(data["O0"], data["O"], data["A"], data["R"])
            per_seed.append(bridge_errors(est, bRs, bDs))
            conds.append({d["label"]: dict(cond_T2_reg=d["cond_T2_reg"],
                                           cond_G_reg=d.get("cond_G_reg"),
                                           eig_min_T2=d["eig_min_T2"],
                                           eig_max_T2=d["eig_max_T2"],
                                           solver=d["solver"],
                                           unseen_x2_frac=d["unseen_x2_frac"])
                          for d in est.diagnostics})
        agg = {k: float(np.mean([p[k] for p in per_seed]))
               for k in ("errR_total", "errD_total")}
        # persist per-seed spread (audit C-med: few-seed point estimates need CIs)
        agg["errR_total_std"] = float(np.std([p["errR_total"] for p in per_seed]))
        agg["errD_total_std"] = float(np.std([p["errD_total"] for p in per_seed]))
        agg["errR_total_per_seed"] = [float(p["errR_total"]) for p in per_seed]
        agg["errD_total_per_seed"] = [float(p["errD_total"]) for p in per_seed]
        agg["n_seeds"] = len(per_seed)
        agg["errR_per_stage"] = np.mean([p["errR_per_stage"] for p in per_seed],
                                        axis=0).tolist()
        agg["errD_per_stage"] = np.mean([p["errD_per_stage"] for p in per_seed],
                                        axis=0).tolist()
        agg["hyperparams"] = est.fitted_
        agg["condition_numbers_seed0"] = conds[0]
        sweep[N] = agg
    RESULTS["P2_scaling_sweep"] = sweep

    print("\n[P2] N-scaling sweep (mean over 3 seeds; oracle norms: "
          f"|bR*|={np.linalg.norm(bRs):.3f}, |bD*|={np.linalg.norm(bDs):.3f})")
    print(f"{'N':>7}{'errR_total':>12}{'errD_total':>12}"
          f"{'condT2(bR_t1)':>15}{'condT2(bR_t3)':>15}")
    for N in N_LIST:
        s = sweep[N]
        c = s["condition_numbers_seed0"]
        print(f"{N:>7}{s['errR_total']:>12.4f}{s['errD_total']:>12.4f}"
              f"{c['bR_t1']['cond_T2_reg']:>15.2e}{c['bR_t3']['cond_T2_reg']:>15.2e}")
    assert sweep[N_LIST[-1]]["errR_total"] < sweep[N_LIST[0]]["errR_total"], \
        "b_R error did not decrease from N=1k to N=20k"
    assert sweep[N_LIST[-1]]["errD_total"] < sweep[N_LIST[0]]["errD_total"], \
        "b_D error did not decrease from N=1k to N=20k"
    print("PASS: L2 recovery error decreases with N for both bridge families")

    # [P3] O_0-noise ill-posedness sweep at N = 10k
    noise_sweep = {}
    for nu in NU_LIST:
        p_nu = toy.default_params(T=3)
        p_nu["K0"] = np.array([[1 - nu, nu], [nu, 1 - nu]])
        errs, cond1, condG1, eigmin1 = [], [], [], []
        for seed in SEEDS:
            data = toy.sample_trajectories(p_nu, 10_000, np.random.default_rng(2000 + seed))
            est = TabularBridgeEstimator(3, 2, 2, 2, 3, mode="auto", seed=seed)
            est.fit(data["O0"], data["O"], data["A"], data["R"])
            errs.append(bridge_errors(est, bRs, bDs))
            d1 = next(d for d in est.diagnostics if d["label"] == "bR_t1")
            cond1.append(d1["cond_T2_reg"])
            eigmin1.append(d1["eig_min_T2"])
        noise_sweep[nu] = dict(
            errR_total=float(np.mean([e["errR_total"] for e in errs])),
            errD_total=float(np.mean([e["errD_total"] for e in errs])),
            errR_stage1=float(np.mean([e["errR_per_stage"][0] for e in errs])),
            errR_stage3=float(np.mean([e["errR_per_stage"][2] for e in errs])),
            cond_T2_reg_bR_t1=float(np.mean(cond1)),
            eig_min_T2_bR_t1=float(np.mean(eigmin1)))
    RESULTS["P3_O0_noise_sweep_N10k"] = noise_sweep

    print("\n[P3] O_0-noise sweep at N=10k (nu -> 0.5 destroys the negative control)")
    print(f"{'nu':>6}{'errR_tot':>10}{'errR_t1':>10}{'errR_t3':>10}"
          f"{'errD_tot':>10}{'cond(bR_t1)':>13}{'eigmin(bR_t1)':>14}")
    for nu in NU_LIST:
        s = noise_sweep[nu]
        print(f"{nu:>6}{s['errR_total']:>10.4f}{s['errR_stage1']:>10.4f}"
              f"{s['errR_stage3']:>10.4f}{s['errD_total']:>10.4f}"
              f"{s['cond_T2_reg_bR_t1']:>13.2e}{s['eig_min_T2_bR_t1']:>14.3e}")

    res_path = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                             "results_phase2.json"))
    with open(res_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", res_path)


if __name__ == "__main__":
    main()
