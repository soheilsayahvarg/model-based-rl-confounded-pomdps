"""
run_finite_action_benchmark.py -- driver for the PAPER-FAITHFUL configuration:
continuous state/observation, FINITE action space.

This is the experiment Phase 3 flagged as its top open item. Phase 3's continuous
study used a real-valued action, which is outside the anchor paper's stated
setting ("both S and O are continuous, while the action space A is finite"), so
its results were extrapolation beyond Theorem 3.5. Everything here is held as
close to Phase 3 as possible except the action space, so differences are
attributable.

Checks (5 seeds, 95% CIs):
  [F1] Oracle self-consistency: true_value vs chain_value_from_bridges at the
       true coefficients, all candidate policies.
  [F2] N-scaling bridge recovery for both bridge families.
  [F3] De-biasing: naive (confounding-blind OLS) vs two-stage plug-in, against
       the exact closed-form value.
  [F4] Pessimism sweep: validity (V_low <= V_true) and selection regret.
       THE SCIENTIFIC QUESTION: Phase 3 found pessimistic selection consistently
       WORSE than plug-in selection in continuous-action space, and hypothesised
       chain compounding. If that anomaly persists here, the continuous action
       space was not the cause; if it disappears, it was.

Writes experiments/results_finite_action.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from finite_action_env import (default_params, sample_trajectories, true_value,
                               true_bridge_coeffs, chain_value_from_bridges,
                               candidate_policies, N_ACTIONS)
from fa_bridge_estimator import FiniteActionBridgeEstimator, bridge_features
from fa_value_plugin import chain_value, empirical_initial_mean
from fa_pessimism import fa_pessimistic_selection

SEEDS = [0, 1, 2, 3, 4]
N_SCALING = [300, 800, 2000, 4000]
N_MAIN = 2000
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
RESULTS = {}


def ci95(v):
    v = np.asarray(v, float)
    return float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def naive_bridges(data, T, n_actions):
    """Confounding-blind baseline: plain least squares of the outcome on
    [onehot(a), o], ignoring the negative control entirely (no instrument, no
    two-stage). This is what a learner does if it treats the noisy observation
    as if it were the state."""
    O, A_idx, R = data["O"], data["A_idx"], data["R"]
    bR, bD = [], []
    for t in range(1, T + 1):
        W = bridge_features(A_idx[:, t - 1], O[:, t - 1], n_actions)
        bR.append(np.linalg.lstsq(W, R[:, t - 1], rcond=None)[0])
        if t < T:
            bD.append(np.linalg.lstsq(W, O[:, t], rcond=None)[0])
    return bR, bD


def main():
    p = default_params(T=3, kappa=1.0)
    T = p["T"]
    cands = candidate_policies()
    tb = true_bridge_coeffs(p)
    v_true = {n: true_value(p, pi) for n, pi in cands.items()}

    # ---------------- [F1] oracle self-consistency ----------------
    worst = max(abs(chain_value_from_bridges(p, tb["bR"], tb["bD"], pi) - v_true[n])
                for n, pi in cands.items())
    RESULTS["F1_oracle_max_gap"] = worst
    print(f"[F1] oracle self-consistency: max gap = {worst:.3e}")
    assert worst < 1e-9, "finite-action oracle self-consistency FAILED"
    print("     true values:", {k: round(v, 4) for k, v in v_true.items()})

    # ---------------- [F2] N-scaling recovery ----------------
    scaling = {}
    for N in N_SCALING:
        eR, eD = [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            d = sample_trajectories(p, N, rng)
            est = FiniteActionBridgeEstimator(T=T, n_actions=N_ACTIONS, seed=sd).fit(
                d["O0"], d["O"], d["A_idx"], d["R"])
            eR.append(float(np.linalg.norm(est.stage_R[1]["theta"] - tb["bR"])))
            eD.append(float(np.linalg.norm(est.stage_D[1]["theta"] - tb["bD"])))
        scaling[N] = dict(errR_mean=float(np.mean(eR)), errR_std=float(np.std(eR)),
                          errD_mean=float(np.mean(eD)), errD_std=float(np.std(eD)))
    RESULTS["F2_recovery_scaling"] = {str(k): v for k, v in scaling.items()}
    print("\n[F2] N-scaling bridge recovery (5-seed mean)")
    print(f"{'N':>6}{'errR':>10}{'errD':>10}")
    for N in N_SCALING:
        print(f"{N:>6}{scaling[N]['errR_mean']:>10.4f}{scaling[N]['errD_mean']:>10.4f}")
    assert scaling[N_SCALING[-1]]["errR_mean"] < scaling[N_SCALING[0]]["errR_mean"], \
        "reward-bridge recovery did not improve from smallest to largest N"
    print("PASS: recovery error decreases from smallest to largest N")

    # ---------------- fits at N_MAIN ----------------
    fits = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        d = sample_trajectories(p, N_MAIN, rng)
        est = FiniteActionBridgeEstimator(T=T, n_actions=N_ACTIONS, seed=sd).fit(
            d["O0"], d["O"], d["A_idx"], d["R"])
        fits.append((est, d, empirical_initial_mean(d["O"]), sd))

    # ---------------- [F3] de-biasing ----------------
    deb = {n: dict(naive=[], plugin=[]) for n in cands}
    for est, d, m1, sd in fits:
        bR_e = [est.stage_R[t]["theta"] for t in range(1, T + 1)]
        bD_e = [est.stage_D[t]["theta"] for t in range(1, T)]
        bR_n, bD_n = naive_bridges(d, T, N_ACTIONS)
        for n, pi in cands.items():
            deb[n]["plugin"].append(chain_value(pi, m1, bR_e, bD_e)[0] - v_true[n])
            deb[n]["naive"].append(chain_value(pi, m1, bR_n, bD_n)[0] - v_true[n])
    RESULTS["F3_debiasing"] = {
        n: dict(bias_naive_mean=float(np.mean(r["naive"])), bias_naive_ci95=ci95(r["naive"]),
                bias_plugin_mean=float(np.mean(r["plugin"])), bias_plugin_ci95=ci95(r["plugin"]),
                abs_naive=float(np.mean(np.abs(r["naive"]))),
                abs_plugin=float(np.mean(np.abs(r["plugin"]))))
        for n, r in deb.items()}
    print(f"\n[F3] de-biasing at N={N_MAIN} (5-seed mean, 95% CI)")
    print(f"{'policy':<14}{'naive':>18}{'plug-in':>18}")
    nb = 0
    for n, r in RESULTS["F3_debiasing"].items():
        print(f"{n:<14}{r['bias_naive_mean']:>10.4f}+/-{r['bias_naive_ci95']:.3f}"
              f"{r['bias_plugin_mean']:>10.4f}+/-{r['bias_plugin_ci95']:.3f}")
        nb += r["abs_plugin"] < r["abs_naive"]
    RESULTS["F3_plugin_beats_naive"] = f"{nb}/{len(cands)}"
    print(f"plug-in |bias| < naive |bias| in {nb}/{len(cands)} policies")

    # ---------------- [F4] pessimism sweep ----------------
    sweep = {}
    for c in C_GRID:
        recs = [fa_pessimistic_selection(est, cands, m1, c, v_true=v_true,
                                         n_restarts=3, seed=sd)
                for est, d, m1, sd in fits]
        sweep[c] = dict(
            all_valid=float(np.mean([r["all_lower_bounds_valid"] for r in recs])),
            subopt_pess=float(np.mean([r["suboptimality_pessimistic"] for r in recs])),
            subopt_plugin=float(np.mean([r["suboptimality_plugin"] for r in recs])),
            max_restart_gap=float(np.max([q["restart_gap"] for r in recs
                                          for q in r["policies"].values()])))
    RESULTS["F4_pessimism_sweep"] = {str(k): v for k, v in sweep.items()}
    print(f"\n[F4] pessimism sweep at N={N_MAIN} (5 seeds)")
    print(f"{'c':>7}{'all_valid':>11}{'regret_pess':>13}{'regret_plug':>13}{'restart_gap':>13}")
    for c in C_GRID:
        s = sweep[c]
        print(f"{c:>7g}{s['all_valid']:>11.2f}{s['subopt_pess']:>13.4f}"
              f"{s['subopt_plugin']:>13.4f}{s['max_restart_gap']:>13.2e}")
    assert all(sweep[c]["all_valid"] == 1.0 for c in C_GRID), \
        "V_low <= V_true violated somewhere in the finite-action sweep"
    print("PASS: V_low <= V_true holds at every width tested, all seeds")

    # the comparison this experiment exists to make
    pess = np.mean([sweep[c]["subopt_pess"] for c in C_GRID])
    plug = np.mean([sweep[c]["subopt_plugin"] for c in C_GRID])
    RESULTS["F4_mean_regret_pessimistic"] = float(pess)
    RESULTS["F4_mean_regret_plugin"] = float(plug)
    verdict = ("pessimistic selection MATCHES OR BEATS plug-in"
               if pess <= plug + 1e-12 else
               "pessimistic selection still WORSE than plug-in")
    RESULTS["F4_verdict"] = verdict
    print(f"\nmean regret over width grid: pessimistic={pess:.4f}  plug-in={plug:.4f}")
    print(f"VERDICT: {verdict}")

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_finite_action.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
