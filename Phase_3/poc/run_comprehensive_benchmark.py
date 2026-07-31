"""
run_comprehensive_benchmark.py — unified Model-Based vs Model-Free comparison
across sample size N and confounding intensity kappa on the 720x2 benchmark.

Methods compared (all evaluated by MAE of V-hat vs the exact DP value, over the
candidate policies; across-seed std reported for stability):
  - naive       : confounding-blind observation-MDP evaluator (oracle.naive_value)
  - MB-plain    : model-based plug-in (BigBridgeEstimator + value_plugin)
  - MB-js       : model-based plug-in with James-Stein shrinkage (variance reduction)
  - MF-proximal : model-free Shi minimax value-bridge OPE (backward recursion)

Writes experiments/results_comprehensive.json and
results/figures/fig6_mb_vs_mf.png (MAE vs kappa at fixed N; MAE vs N at fixed kappa).

This script imports the pinned oracle only for GROUND TRUTH; it does not modify any
pinned module, so the regression suite (V1/V4) is unaffected.
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

import simulated_env as senv
import oracle_module as oracle
from big_bridge_estimator import BigBridgeEstimator, N_O0 as BIG_N_O0
from shrinked_big_bridge_estimator import ShrinkedBigBridgeEstimator
from value_plugin import empirical_p_o1  # noqa: F401 (kept for parity)
from model_free_proximal import MinimaxValueBridgeOPE
from run_bigenv_check import big_candidates

KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_LIST = [5000, 20000, 60000]
SEEDS = [0, 1, 2, 3, 4]
REWARD_LEVELS = [-1, 0, 1]


def mb_plugin_mae(EstClass, kwargs, env, data, cands, v_true):
    est = EstClass(T=env["T"], **kwargs).fit(
        data["O0"], data["O"], data["A"], data["R"])
    p_o1 = np.bincount(data["O"][:, 0], minlength=senv.N_OBS).astype(float)
    p_o1 /= p_o1.sum()
    return np.mean([abs(est.plugin_value(pi, p_o1)[0] - v_true[n])
                    for n, pi in cands.items()])


def mf_mae(env, data, cands, v_true):
    errs = []
    for n, pi in cands.items():
        mf = MinimaxValueBridgeOPE(senv.N_OBS, senv.N_A, BIG_N_O0, env["T"],
                                   reward_levels=REWARD_LEVELS)
        errs.append(abs(mf.estimate_value(data, pi) - v_true[n]))
    return np.mean(errs)


def naive_mae(env, data, cands, v_true):
    P_hat, r_hat, p1_hat, _ = oracle.fit_naive_mdp(
        data["O"], data["A"], data["R"], senv.N_OBS, senv.N_A)
    return np.mean([abs(oracle.naive_value(P_hat, r_hat, p1_hat, pi, env["T"])
                        - v_true[n]) for n, pi in cands.items()])


METHODS = [
    ("naive", lambda env, d, c, vt: naive_mae(env, d, c, vt)),
    ("MB-plain", lambda env, d, c, vt: mb_plugin_mae(BigBridgeEstimator, {}, env, d, c, vt)),
    ("MB-js", lambda env, d, c, vt: mb_plugin_mae(
        ShrinkedBigBridgeEstimator, dict(shrinkage="js"), env, d, c, vt)),
    ("MF-proximal", lambda env, d, c, vt: mf_mae(env, d, c, vt)),
]


def main():
    cands = big_candidates()
    results = {}   # (kappa, N) -> method -> {mae_mean, mae_std}
    for kappa in KAPPAS:
        env = senv.make_env(kappa=kappa)
        v_true = {n: oracle.dp_value_obs_policy(env["aug"], pi, env["T"],
                                                env["q0"], env["q1"])
                  for n, pi in cands.items()}
        for N in N_LIST:
            per_method = {m: [] for m, _ in METHODS}
            for seed in SEEDS:
                data = senv.generate_dataset(env, N=N, seed=1000 + seed)
                for m, fn in METHODS:
                    per_method[m].append(float(fn(env, data, cands, v_true)))
            n_seeds = len(SEEDS)
            results[f"{kappa}|{N}"] = {
                m: dict(mae_mean=float(np.mean(v)), mae_std=float(np.std(v)),
                        mae_ci95_halfwidth=float(
                            1.96 * np.std(v, ddof=1) / np.sqrt(n_seeds)),
                        n_seeds=n_seeds)
                for m, v in per_method.items()}
            row = "  ".join(f"{m}={np.mean(per_method[m]):.4f}"
                            for m, _ in METHODS)
            print(f"kappa={kappa:<4} N={N:<6} | {row}")

    res_path = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                             "results_comprehensive.json"))
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(dict(kappas=KAPPAS, n_list=N_LIST, seeds=len(SEEDS),
                       results=results), f, indent=2)
    print("results ->", res_path)
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"naive": "#52514e", "MB-plain": "#e34948",
           "MB-js": "#eb6834", "MF-proximal": "#2a78d6"}
    MK = {"naive": "o", "MB-plain": "s", "MB-js": "^", "MF-proximal": "D"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "font.size": 9, "legend.frameon": False,
                         "axes.grid": True, "grid.color": "#e6e5e0"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # (left) MAE vs kappa at N = 20000
    Nf = 20000
    for m in COL:
        ys = [results[f"{k}|{Nf}"][m]["mae_mean"] for k in KAPPAS]
        es = [results[f"{k}|{Nf}"][m]["mae_std"] for k in KAPPAS]
        ax1.errorbar(KAPPAS, ys, yerr=es, marker=MK[m], color=COL[m], lw=1.8,
                     ms=5, capsize=2, label=m)
    ax1.set_xlabel(r"confounding intensity $\kappa$")
    ax1.set_ylabel("MAE of $\\hat V(\\pi)$ vs true (lower better)")
    ax1.set_title(f"Robustness to confounding (N={Nf:,})")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=8)

    # (right) MAE vs N at kappa = 1.0 (strong confounding)
    kf = 1.0
    for m in COL:
        ys = [results[f"{kf}|{N}"][m]["mae_mean"] for N in N_LIST]
        es = [results[f"{kf}|{N}"][m]["mae_std"] for N in N_LIST]
        ax2.errorbar(N_LIST, ys, yerr=es, marker=MK[m], color=COL[m], lw=1.8,
                     ms=5, capsize=2, label=m)
    ax2.set_xlabel("N (trajectories)")
    ax2.set_ylabel("MAE")
    ax2.set_title(r"Sample efficiency ($\kappa$=1, strong confounding)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Model-Based vs Model-Free under hidden confounding (720x2 benchmark)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.normpath(os.path.join(HERE, "..", "results", "figures",
                                        "fig6_mb_vs_mf.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("figure ->", out)


if __name__ == "__main__":
    main()
