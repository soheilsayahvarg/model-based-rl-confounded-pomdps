"""
run_continuous_benchmark.py -- continuous-state/action extension PoC: the
analogue of run_phase34_check.py + run_phase2_check.py's [P2], combined, for
the linear-Gaussian confounded POMDP (continuous_env.py) and the kernel
two-stage estimator (continuous_bridge_estimator.py).

Checks (5 seeds, 95% CIs from the start -- applying tonight's lesson rather
than needing a later correction pass):
  [C1] Oracle self-consistency: continuous_env.true_value vs chain_value_from_
       bridges using the TRUE bridge coefficients must agree to machine
       precision, for every candidate policy (the continuous analogue of
       run_phase34_check.py's [V1]).
  [C2] N-scaling bridge recovery: ||theta_hat - theta_true|| for both bridge
       families must (on average, across 5 seeds) decrease as N grows (the
       continuous analogue of run_phase2_check.py's [P2]). NOTE: dense kernel
       ridge is O(N1^3) per solve (see docs/continuous_extension.md) -- N is
       kept to a toy-appropriate range for this reason, not accuracy.
  [C3] De-biasing: naive (plain OLS, ignores the negative control) vs
       bridge-based plug-in, both evaluated against the exact closed-form
       V_true, across all 4 candidate policies.
  [C4] Pessimism sweep: V_low per policy across a width-multiplier grid,
       validity check V_low <= V_true, and selection regret (pessimistic vs
       plug-in-based selection) vs the true best policy.

Writes experiments/results_continuous.json.
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from continuous_env import (default_params, sample_trajectories, true_value,
                            true_bridge_coeffs, chain_value_from_bridges,
                            candidate_policies)
from continuous_bridge_estimator import ContinuousBridgeEstimator
from continuous_value_plugin import chain_value, empirical_initial_mean
from continuous_pessimism import continuous_pessimistic_selection

RESULTS = {}
SEEDS = [0, 1, 2, 3, 4]
N_SCALING_LIST = [300, 800, 2000, 4000]
N_MAIN = 2000
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]


def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    return float(1.96 * vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def main():
    p = default_params(T=3, kappa=1.0)
    T = p["T"]
    cands = candidate_policies()
    true_b = true_bridge_coeffs(p)
    bR_true, bD_true = np.array(true_b["bR"]), np.array(true_b["bD"])
    v_true = {name: true_value(p, K, c) for name, (K, c) in cands.items()}

    # ---------------- [C1] oracle self-consistency ==
    worst = 0.0
    for name, (K, c) in cands.items():
        v_chain = chain_value_from_bridges(p, true_b["bR"], true_b["bD"], K, c)
        worst = max(worst, abs(v_chain - v_true[name]))
    RESULTS["C1_oracle_self_consistency_max_gap"] = worst
    print(f"[C1] oracle self-consistency (true bridges vs closed-form): "
          f"max gap = {worst:.3e}")
    assert worst < 1e-9, "continuous oracle self-consistency FAILED"

    # ---------------- [C2] N-scaling bridge recovery ==
    scaling = {}
    for N in N_SCALING_LIST:
        errR_list, errD_list = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            data = sample_trajectories(p, N, rng)
            est = ContinuousBridgeEstimator(T=T, seed=seed).fit(
                data["O0"], data["O"], data["A"], data["R"])
            errR_list.append(float(np.linalg.norm(est.stage_R[1]["theta_hat"] - bR_true)))
            errD_list.append(float(np.linalg.norm(est.stage_D[1]["theta_hat"] - bD_true)))
        scaling[N] = dict(
            errR_mean=float(np.mean(errR_list)), errR_std=float(np.std(errR_list)),
            errD_mean=float(np.mean(errD_list)), errD_std=float(np.std(errD_list)),
            n_seeds=len(SEEDS))
    RESULTS["C2_bridge_recovery_scaling"] = {str(k): v for k, v in scaling.items()}
    print("\n[C2] N-scaling bridge recovery (5-seed mean +/- std)")
    print(f"{'N':>6}{'errR':>10}{'errD':>10}")
    for N in N_SCALING_LIST:
        s = scaling[N]
        print(f"{N:>6}{s['errR_mean']:>10.4f}{s['errD_mean']:>10.4f}")
    assert (scaling[N_SCALING_LIST[-1]]["errR_mean"]
            < scaling[N_SCALING_LIST[0]]["errR_mean"]), \
        "reward-bridge recovery error did not decrease from smallest to largest N"
    print("PASS: recovery error decreases from smallest to largest N (reward bridge)")

    # ---------------- fit once at N_MAIN for [C3]/[C4], 5 seeds ==
    fits = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        data = sample_trajectories(p, N_MAIN, rng)
        est = ContinuousBridgeEstimator(T=T, seed=seed).fit(
            data["O0"], data["O"], data["A"], data["R"])
        m1_hat = empirical_initial_mean(data["O"])
        fits.append((est, data, m1_hat, seed))

    # ---------------- [C3] de-biasing: naive OLS vs bridge plug-in ==
    debias = {name: dict(bias_naive=[], bias_plugin=[]) for name in cands}
    for est, data, m1_hat, seed in fits:
        O, A, R = data["O"], data["A"], data["R"]
        naive_R, naive_D = {}, {}
        for t in range(1, T + 1):
            W = np.stack([A[:, t - 1], O[:, t - 1]], axis=1)
            naive_R[t], *_ = np.linalg.lstsq(W, R[:, t - 1], rcond=None)
            if t < T:
                naive_D[t], *_ = np.linalg.lstsq(W, O[:, t], rcond=None)
        bR_est = [est.stage_R[t]["theta"] for t in range(1, T + 1)]
        bD_est = [est.stage_D[t]["theta"] for t in range(1, T)]
        bR_nv = [naive_R[t] for t in range(1, T + 1)]
        bD_nv = [naive_D[t] for t in range(1, T)]
        for name, (K, c) in cands.items():
            v_pl, _ = chain_value(est, K, c, m1_hat, bR_est, bD_est)
            v_nv, _ = chain_value(est, K, c, m1_hat, bR_nv, bD_nv)
            debias[name]["bias_plugin"].append(v_pl - v_true[name])
            debias[name]["bias_naive"].append(v_nv - v_true[name])
    RESULTS["C3_debiasing"] = {
        name: dict(
            bias_naive_mean=float(np.mean(r["bias_naive"])),
            bias_naive_ci95=ci95(r["bias_naive"]),
            bias_plugin_mean=float(np.mean(r["bias_plugin"])),
            bias_plugin_ci95=ci95(r["bias_plugin"]),
            abs_bias_naive=float(np.mean(np.abs(r["bias_naive"]))),
            abs_bias_plugin=float(np.mean(np.abs(r["bias_plugin"]))))
        for name, r in debias.items()}
    print(f"\n[C3] de-biasing at N={N_MAIN} (5-seed mean, 95% CI)")
    print(f"{'policy':<16}{'bias_naive':>14}{'bias_plugin':>16}")
    n_better = 0
    for name, r in RESULTS["C3_debiasing"].items():
        print(f"{name:<16}{r['bias_naive_mean']:>9.4f}+/-{r['bias_naive_ci95']:.3f}"
              f"   {r['bias_plugin_mean']:>9.4f}+/-{r['bias_plugin_ci95']:.3f}")
        n_better += r["abs_bias_plugin"] < r["abs_bias_naive"]
    RESULTS["C3_plugin_beats_naive_count"] = f"{n_better}/{len(cands)}"
    print(f"plug-in |bias| < naive |bias| in {n_better}/{len(cands)} policies")

    # ---------------- [C4] pessimism sweep ==
    sweep = {}
    for c in C_GRID:
        recs = []
        for est, data, m1_hat, seed in fits:
            rec = continuous_pessimistic_selection(
                est, cands, m1_hat, c, v_true=v_true, n_restarts=3, seed=seed)
            recs.append(rec)
        sweep[c] = dict(
            all_valid=float(np.mean([r["all_lower_bounds_valid"] for r in recs])),
            subopt_pess_mean=float(np.mean([r["suboptimality_pessimistic"] for r in recs])),
            subopt_plugin_mean=float(np.mean([r["suboptimality_plugin"] for r in recs])),
            max_restart_gap=float(np.max([p["restart_gap"] for r in recs
                                          for p in r["policies"].values()])),
            n_seeds=len(recs))
    RESULTS["C4_pessimism_sweep"] = {str(k): v for k, v in sweep.items()}
    print(f"\n[C4] pessimism sweep at N={N_MAIN} (5 seeds)")
    print(f"{'c':>8}{'all_valid':>11}{'subopt_pess':>13}{'subopt_plug':>13}{'restart_gap':>13}")
    for c in C_GRID:
        s = sweep[c]
        print(f"{c:>8g}{s['all_valid']:>11.2f}{s['subopt_pess_mean']:>13.4f}"
              f"{s['subopt_plugin_mean']:>13.4f}{s['max_restart_gap']:>13.2e}")
    assert all(sweep[c]["all_valid"] == 1.0 for c in C_GRID), \
        "V_low <= V_true violated at some width in the continuous pessimism sweep"
    print("PASS: V_low <= V_true holds at every width tested, all seeds")

    res_path = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                             "results_continuous.json"))
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", res_path)


if __name__ == "__main__":
    main()
