"""critic_b_a45.py -- ATTACKS A4 and A5.

A4: nu1 is estimated from the same sample used to fit the bridge and build H.
    Swap in the exact population nu1 = p1 @ E and measure how much leakage and
    width move. (The t=1-only choice of H is addressed analytically: J touches
    only the stage-1 bridge, so stage-1 H is the right DIRECT object; the
    later-stage channel it misses is quantified in A3.)

A5: is 36/36 validity informative? Three measurements:
    - how many cells have V_hat <= V_true even before any penalty (validity
      then holds for ANY c >= 0 -- trivially);
    - per-seed (not seed-averaged) violation counts, since the driver reports
      validity on means;
    - for cells with V_hat > V_true, the critical c* at which the unprojected
      bound would first touch the truth: c* = ((V_hat - V_true)/K)^2 with
      K = sum_a ||g||_{H_a^-1} / sqrt(N * lambda_min(H_a)). If c* is far below
      the smallest c on the grid, the grid could never have produced a
      violation and 36/36 was structurally guaranteed.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import (default_params, sample_trajectories, dp_value,
                                 candidate_policies)
from model_free_proximal import MinimaxValueBridgeOPE
from mf_pessimism import (mf_design_and_gradient, h_inv_norm, null_leakage,
                          parallel_analysis_basis)

CONFIGS = [(4, 6, 2), (2, 6, 4), (3, 7, 5)]
CONFOUNDS = [1.0, 0.6]
N = 4000
SEEDS = [0, 1, 2, 3, 4]
N_ACT = 2
T = 3
CS = [0.1, 1.0, 10.0]


def main():
    # ------------------------------- A4: empirical vs exact nu1
    print("[A4] leakage and width with empirical vs exact population nu1.\n")
    print(f"{'config':>10}{'cf':>5}{'leak emp':>10}{'leak pop':>10}"
          f"{'w_u emp':>9}{'w_u pop':>9}")
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            nu_pop = p["p1"] @ p["E"]
            le, lp, we, wp_ = [], [], [], []
            for sd in SEEDS:
                d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
                nu_emp = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
                nu_emp /= nu_emp.sum()
                He, ge = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu_emp)
                Hp, gp = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu_pop)
                for a in range(N_ACT):
                    k, U = parallel_analysis_basis(
                        d, n_o, n_o0, a, np.random.default_rng(900 + sd))
                    le.append(null_leakage(He[a], ge[a], U))
                    lp.append(null_leakage(Hp[a], gp[a], U))
                    we.append(h_inv_norm(He[a], ge[a]))
                    wp_.append(h_inv_norm(Hp[a], gp[a]))
            print(f"{str((n_s,n_o,n_o0)):>10}{cf:>5.1f}{np.mean(le):>10.4f}"
                  f"{np.mean(lp):>10.4f}{np.mean(we):>9.3f}{np.mean(wp_):>9.3f}")

    # ------------------------------- A5: validity triviality audit
    print("\n[A5] Validity audit: per-seed margins and critical c*.\n")
    print(f"{'config':>10}{'cf':>5}{'policy':>10}{'seeds V_hat>V_true':>19}"
          f"{'min c*':>10}{'per-seed viol (c grid)':>24}")
    n_trivial = n_cells = 0
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            pols = candidate_policies(n_o, N_ACT, seed=0)
            for name in list(pols)[:2]:
                pi = pols[name]
                v_true = dp_value(p, pi)
                over, cstars, viol = 0, [], 0
                for sd in SEEDS:
                    d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
                    nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
                    nu1 /= nu1.sum()
                    est = MinimaxValueBridgeOPE(n_obs=n_o, n_act=N_ACT,
                                                n_o0=n_o0, T=T, seed=sd)
                    est.fit(d["O0"], d["O"], d["A"], d["R"], pi)
                    v_hat = est.value()
                    Hs, gs = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu1)
                    K = sum(h_inv_norm(Hs[a], gs[a])
                            / np.sqrt(N * max(np.linalg.eigvalsh(Hs[a])[0],
                                              1e-300))
                            for a in range(N_ACT))
                    gap = v_hat - v_true
                    if gap > 0:
                        over += 1
                        cstars.append((gap / K) ** 2)
                    for c in CS:
                        if v_hat - np.sqrt(c) * K > v_true:
                            viol += 1
                n_cells += 1
                if over == 0:
                    n_trivial += 1
                mc = f"{min(cstars):.2e}" if cstars else "--"
                print(f"{str((n_s,n_o,n_o0)):>10}{cf:>5.1f}{name:>10}"
                      f"{over:>13}/5{mc:>12}{viol:>18}/15")
    print(f"\n  cells where V_hat <= V_true in every seed (validity free): "
          f"{n_trivial}/{n_cells}")


if __name__ == "__main__":
    main()
