"""
run_phase34_check.py — Phase 3 (value plug-in) + Phase 4 (ellipsoidal pessimism)
validation on the toy substrate (oracle-verified end to end).

Checks:
  [V1] Plug-in with TRUE bridges + exact p(o_1): the einsum chain equals exact DP
       (and the Phase-1 enumeration evaluator) to machine precision, per policy.
  [V2] Plug-in with ESTIMATED bridges + empirical p_hat(o_1): the plug-in removes
       the naive baseline's systematic bias — |bias_plugin| < |bias_naive| —
       and shrinks with N (N = 5k, 20k, 50k; 3 seeds).
  [V3] Pessimism sweep over width multipliers c: coverage curves (block + joint),
       V_low per policy, explicit validity check V_low <= V_true, restart gaps,
       selection + true suboptimality (pessimistic vs plug-in selection).
  [V4] Summary verdict: at every c with joint coverage = 1 (all seeds), all
       V_low <= V_true must hold (the foundational pessimistic guarantee,
       checked against the oracle).

Writes experiments/results_phase34.json.
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
import oracle_module as oracle
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from pessimistic_optimizer import pessimistic_selection
from run_phase1_check import toy_candidates

RESULTS = {}
SEEDS = [0, 1, 2]
C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
N_PESS = 20_000


def main():
    params = toy.default_params(T=3)
    T = params["T"]
    cands = toy_candidates()
    bRs, bDs, _ = oracle.toy_true_bridges(params)
    bR_true = np.stack([bRs] * T)
    bD_true = np.stack([bDs] * (T - 1))
    p_o1_exact = params["p1"] @ params["E"]
    v_true = {n: oracle.dp_value_toy(params, pi) for n, pi in cands.items()}

    # ---------------- [V1] plug-in with true bridges == DP ==
    worst = 0.0
    for name, pi in cands.items():
        v_chain, _ = plugin_value(bR_true, bD_true, pi, p_o1_exact)
        v_enum, _ = oracle.toy_chain_value(params, bRs, bDs, pi)
        worst = max(worst, abs(v_chain - v_true[name]), abs(v_chain - v_enum))
    RESULTS["V1_plugin_true_bridges_max_gap"] = worst
    print(f"[V1] einsum plug-in vs DP / enumeration (true bridges): "
          f"max gap = {worst:.3e}")
    assert worst < 1e-10

    # ---------------- [V2] plug-in de-biasing vs naive ==
    debias = {}
    for N in (5_000, 20_000, 50_000):
        rows = {n: dict(bias_naive=[], bias_plugin=[]) for n in cands}
        for seed in SEEDS:
            data = toy.sample_trajectories(params, N, np.random.default_rng(300 + seed))
            est = TabularBridgeEstimator(3, 2, 2, 2, T, seed=seed)
            est.fit(data["O0"], data["O"], data["A"], data["R"])
            bR_hat, bD_hat = est.bridges()
            p_o1_hat = empirical_p_o1(data["O"], toy.N_O)
            P_hat, r_hat, p1_hat, _ = oracle.fit_naive_mdp(
                data["O"], data["A"], data["R"], toy.N_O, toy.N_A)
            for name, pi in cands.items():
                v_pl, _ = plugin_value(bR_hat, bD_hat, pi, p_o1_hat)
                v_nv = oracle.naive_value(P_hat, r_hat, p1_hat, pi, T)
                rows[name]["bias_plugin"].append(v_pl - v_true[name])
                rows[name]["bias_naive"].append(v_nv - v_true[name])
        debias[N] = {n: dict(
            bias_naive=float(np.mean(r["bias_naive"])),
            bias_plugin=float(np.mean(r["bias_plugin"])),
            abs_bias_naive=float(np.mean(np.abs(r["bias_naive"]))),
            abs_bias_plugin=float(np.mean(np.abs(r["bias_plugin"]))))
            for n, r in rows.items()}
    RESULTS["V2_plugin_vs_naive_bias"] = debias

    print("\n[V2] plug-in de-biasing (mean over 3 seeds)")
    print(f"{'N':>7}{'policy':<18}{'bias_naive':>12}{'bias_plugin':>13}")
    n_better = 0
    for N, tab in debias.items():
        for name, v in tab.items():
            print(f"{N:>7}{name:<18}{v['bias_naive']:>12.4f}{v['bias_plugin']:>13.4f}")
            n_better += v["abs_bias_plugin"] < v["abs_bias_naive"]
    frac_better = n_better / (len(debias) * len(cands))
    RESULTS["V2_frac_plugin_beats_naive"] = float(frac_better)
    print(f"plug-in |bias| < naive |bias| in {n_better}/"
          f"{len(debias) * len(cands)} (N, policy) cells")
    assert debias[50_000][max(cands, key=lambda n: abs(debias[50_000][n]['bias_naive']))
                          ]["abs_bias_plugin"] < 0.5 * max(
        v["abs_bias_naive"] for v in debias[50_000].values()), \
        "plug-in fails to substantially reduce the worst naive bias at N=50k"

    # ---------------- [V3] pessimism sweep: vanilla Eq.-17 vs signal-projected ==
    fits = []
    for seed in SEEDS:
        data = toy.sample_trajectories(params, N_PESS, np.random.default_rng(400 + seed))
        est = TabularBridgeEstimator(3, 2, 2, 2, T, seed=seed)
        est.fit(data["O0"], data["O"], data["A"], data["R"])
        fits.append((est, empirical_p_o1(data["O"], toy.N_O), seed))

    best_name = max(v_true, key=v_true.get)
    all_agg = {}
    for variant, proj in (("vanilla", False), ("projected", True)):
        sweep = {}
        for est, p_o1_hat, seed in fits:
            for c in C_GRID:
                rec = pessimistic_selection(est, cands, p_o1_hat, c,
                                            bR_true=bRs, bD_true=bDs,
                                            v_true=v_true, n_restarts=3, seed=seed,
                                            signal_projection=proj)
                sweep.setdefault(c, []).append(rec)
        agg = {}
        for c, recs in sweep.items():
            agg[c] = dict(
                block_coverage=float(np.mean([r["coverage"]["block_coverage"]
                                              for r in recs])),
                joint_coverage=float(np.mean([r["coverage"]["joint_coverage"]
                                              for r in recs])),
                all_bounds_valid=float(np.mean([r["all_lower_bounds_valid"]
                                                for r in recs])),
                max_restart_gap=float(np.max([p["restart_gap"] for r in recs
                                              for p in r["policies"].values()])),
                subopt_pess=float(np.mean([r["suboptimality_pessimistic"]
                                           for r in recs])),
                subopt_plugin=float(np.mean([r["suboptimality_plugin"]
                                             for r in recs])),
                selected=[r["selected_policy"] for r in recs],
                V_low_mean={n: float(np.mean([r["policies"][n]["V_low"]
                                              for r in recs])) for n in cands})
        all_agg[variant] = agg

        print(f"\n[V3:{variant}] pessimism sweep at N={N_PESS} "
              f"(xi_t = c/(N2*sigma2_t); true best {best_name}="
              f"{v_true[best_name]:.3f})")
        print(f"{'c':>8}{'blk_cov':>9}{'joint_cov':>10}{'valid':>7}{'sub_pess':>10}"
              f"{'sub_plug':>10}{'V_low(best_pol)':>16}{'restart_gap':>12}")
        for c in C_GRID:
            a = agg[c]
            print(f"{c:>8g}{a['block_coverage']:>9.2f}{a['joint_coverage']:>10.2f}"
                  f"{a['all_bounds_valid']:>7.2f}{a['subopt_pess']:>10.4f}"
                  f"{a['subopt_plugin']:>10.4f}{a['V_low_mean'][best_name]:>16.4f}"
                  f"{a['max_restart_gap']:>12.2e}")

    RESULTS["V3_pessimism_sweep_N20k"] = {
        variant: {str(c): v for c, v in agg.items()}
        for variant, agg in all_agg.items()}
    RESULTS["V3_true_values"] = {n: float(v) for n, v in v_true.items()}

    # ---------------- [V4] foundational guarantee (both variants) ==
    v4 = {}
    for variant, agg in all_agg.items():
        violations = [c for c, a in agg.items()
                      if a["joint_coverage"] == 1.0 and a["all_bounds_valid"] < 1.0]
        v4[variant] = dict(
            n_c_with_full_joint_coverage=int(sum(1 for a in agg.values()
                                                 if a["joint_coverage"] == 1.0)),
            violations=violations)
        print(f"\n[V4:{variant}] V_low <= V_true whenever truth jointly covered: "
              f"{'PASS' if not violations else f'VIOLATED at c={violations}'}")
    RESULTS["V4_validity_when_covered"] = v4
    assert not v4["vanilla"]["violations"], \
        "vanilla pessimistic lower bound violated under joint coverage"
    assert not v4["projected"]["violations"], \
        "projected pessimistic lower bound violated under (projected) joint coverage"

    res_path = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                             "results_phase34.json"))
    with open(res_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", res_path)


if __name__ == "__main__":
    main()
