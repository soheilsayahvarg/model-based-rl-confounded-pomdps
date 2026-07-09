"""
run_bigenv_check.py — scaled validation on the full 720x2 benchmark environment.

Checks:
  [B1] Consistency spot-check: the per-cell estimator's population target equals
       the oracle per-core marker inversion (vectorized oracle b_R*, compared on
       well-visited cells; L2 recovery decreasing with N).
  [B2] End-to-end de-biasing at benchmark scale: V_plugin vs V_naive vs V_true
       for the 4 candidate policies, kappa in {1.0, 0.0}, with seeds.
       Success criterion: plug-in tracks V_true and removes the bulk of the
       naive structural bias (population-naive from Phase 1) at kappa = 1.

Writes experiments/results_bigenv.json.
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import simulated_env as senv
import oracle_module as oracle
from big_bridge_estimator import BigBridgeEstimator, N_X, N_A, N_M, N_R
from run_phase1_check import big_candidates

SEEDS = [0, 1]
N_LIST = [20_000, 60_000]
RESULTS = {}


def oracle_bR_all(env):
    """Vectorized oracle reward bridges b*_R[x, a, m~, r, m_o] via per-core
    marker inversion with TRUE quantities (time-homogeneous)."""
    aug, q0, q1 = env["aug"], env["q0"], env["q1"]
    P = aug["P"]
    Em2 = np.array([[1 - q0, q0], [1 - q1, q1]])          # [u, m]
    Em2inv = np.linalg.inv(Em2)                           # [m~, u]
    # P(r | s=(u,x), a): from live rows into terminals
    p_neg = P[:, :1440, senv.TERM_NEG]                    # (A, 1440)
    p_pos = P[:, :1440, senv.TERM_POS]
    p_r = np.stack([p_neg, 1.0 - p_neg - p_pos, p_pos], axis=2)   # (A, 1440, 3)
    p_r = p_r.reshape(N_A, 2, N_X, N_R)                   # [a, u, x, r]
    # target[u, x, a, r, m_o] = P(m_o|u) * P(r|(u,x),a);  P(m_o|u) = Em2[u, m_o]
    tgt = np.einsum("auxr,un->uxarn", p_r, Em2)
    bR = np.einsum("mu,uxarn->xamrn", Em2inv, tgt)        # [x, a, m~, r, m_o]
    return bR


def main():
    cands = big_candidates()
    exit_tab = {}
    recovery = {}

    for kappa in (1.0, 0.0):
        env = senv.make_env(kappa=kappa)
        bR_star = oracle_bR_all(env)
        v_true = {n: oracle.dp_value_obs_policy(env["aug"], pi, env["T"],
                                                env["q0"], env["q1"])
                  for n, pi in cands.items()}
        pop = oracle.population_naive_mdp(env)
        v_pop_naive = {n: oracle.naive_value(pop[0], pop[1], pop[2], pi, env["T"])
                       for n, pi in cands.items()}

        vmax = env["T"]                         # rewards in [-1,1] over T steps
        for N in N_LIST:
            rows = {n: dict(pl=[], nv=[]) for n in cands}
            rec_err, ok_frac, mass_frac = [], [], []
            for seed in SEEDS:
                data = senv.generate_dataset(env, N=N, seed=500 + seed)
                est = BigBridgeEstimator(T=env["T"]).fit(
                    data["O0"], data["O"], data["A"], data["R"])
                # [B1] recovery on well-estimated cells (mean over stages)
                errs = []
                for t in range(env["T"]):
                    ok = est.cell_ok.reshape(env["T"], N_X, N_A)[t]
                    bR_t = est.bR[t]                       # [x, a, m~, r, m_o]
                    diff = (bR_t - bR_star)[ok]
                    if diff.size:
                        errs.append(float(np.sqrt(np.mean(diff ** 2))))
                rec_err.append(float(np.mean(errs)))
                ok_frac.append(1.0 - est.fallback_cell_frac)
                mass_frac.append(1.0 - est.fallback_mass_frac)   # visit-weighted
                # [B2] values
                p_o1 = np.bincount(data["O"][:, 0], minlength=senv.N_OBS
                                   ).astype(float)
                p_o1 /= p_o1.sum()
                P_hat, r_hat, p1_hat, _ = oracle.fit_naive_mdp(
                    data["O"], data["A"], data["R"], senv.N_OBS, senv.N_A)
                for n, pi in cands.items():
                    vpl = est.plugin_value(pi, p_o1)[0]
                    # regression guard (audit B): a gross chain transpose/aliasing
                    # bug in folded_operator/_reward_head yields values outside the
                    # feasible reward range [-T, T]; assert plug-in stays feasible.
                    assert -vmax - 1e-6 <= vpl <= vmax + 1e-6, \
                        f"plug-in value {vpl} outside feasible [-{vmax},{vmax}]"
                    rows[n]["pl"].append(vpl)
                    rows[n]["nv"].append(oracle.naive_value(
                        P_hat, r_hat, p1_hat, pi, env["T"]))
            recovery.setdefault(f"kappa={kappa}", {})[N] = dict(
                bR_rmse_ok_cells=float(np.mean(rec_err)),
                ok_cell_frac=float(np.mean(ok_frac)),
                ok_mass_frac=float(np.mean(mass_frac)))     # audit B: honest metric
            tab = {}
            for n in cands:
                tab[n] = dict(
                    V_true=round(v_true[n], 6),
                    V_naive_pop=round(v_pop_naive[n], 6),
                    V_naive_emp=round(float(np.mean(rows[n]["nv"])), 6),
                    V_plugin=round(float(np.mean(rows[n]["pl"])), 6),
                    bias_naive_pop=round(v_pop_naive[n] - v_true[n], 6),
                    bias_plugin=round(float(np.mean(rows[n]["pl"])) - v_true[n], 6))
            exit_tab.setdefault(f"kappa={kappa}", {})[N] = tab

    RESULTS["B1_bridge_recovery"] = recovery
    RESULTS["B2_debiasing"] = exit_tab

    print("=== BIG-ENV (720x2) scaled validation ===")
    for kap, byN in recovery.items():
        for N, r in byN.items():
            print(f"[B1] {kap} N={N}: bR RMSE (ok cells) = "
                  f"{r['bR_rmse_ok_cells']:.4f}, ok-cell frac = "
                  f"{r['ok_cell_frac']:.3f}")
    for kap, byN in exit_tab.items():
        for N, tab in byN.items():
            print(f"\n[B2] {kap}  N={N}")
            print(f"{'policy':<20}{'V_true':>9}{'naive_pop':>11}{'plugin':>9}"
                  f"{'bias_nv':>9}{'bias_pl':>9}")
            for n, v in tab.items():
                print(f"{n:<20}{v['V_true']:>9.4f}{v['V_naive_pop']:>11.4f}"
                      f"{v['V_plugin']:>9.4f}{v['bias_naive_pop']:>9.4f}"
                      f"{v['bias_plugin']:>9.4f}")

    res_path = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                             "results_bigenv.json"))
    with open(res_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", res_path)


if __name__ == "__main__":
    main()
