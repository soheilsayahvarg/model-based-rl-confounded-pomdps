"""critic_c_proj20.py -- step (c) critic, Priority 3b.

Section 5.4 reports the projected slope as "consistent, not verified": 3 seeds,
non-monotone series, PA rank noise. Re-run part A's projected column at 20
seeds and decide whether the (tau-1)/2 prediction is confirmed or refuted.
Everything else identical to run_noncontraction.part_a.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params, sample_trajectories
from mf_pessimism import mf_design_and_gradient, h_inv_norm, parallel_analysis_basis
from run_noncontraction import loglog_slope, LAM0, N_GRID, C_WIDTH

SEEDS20 = list(range(20))


def main():
    n_s, n_o, n_o0, n_act, T = 2, 6, 4, 2, 3
    print("PART A projected column, 20 seeds")
    for kappa in (0.25, 0.5, 0.75):
        wu_by_n, wp_by_n, wp_sd, ranks = [], [], [], []
        wp_med_by_n, wp_r1_by_n, frac_r1 = [], [], []
        for N in N_GRID:
            wu, wp, ks = [], [], []
            for sd in SEEDS20:
                p = default_params(n_s=n_s, n_a=n_act, n_o=n_o, n_o0=n_o0,
                                   T=T, seed=sd, confound=1.0)
                rng = np.random.default_rng(1000 + sd)
                d = sample_trajectories(p, N, rng)
                nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float) / N
                lam = LAM0 * N ** (-kappa)
                Hs, gs = mf_design_and_gradient(d, n_o, n_o0, n_act, nu1,
                                                t=1, ridge=lam)
                for a in range(n_act):
                    ev = np.linalg.eigvalsh(Hs[a])
                    xi = C_WIDTH / (N * ev[0])
                    wu.append(np.sqrt(xi) * h_inv_norm(Hs[a], gs[a]))
                    k, U = parallel_analysis_basis(d, n_o, n_o0, a, rng, t=1)
                    ks.append(k)
                    wp.append(np.sqrt(xi) * h_inv_norm(Hs[a], gs[a], basis=U))
            wu_by_n.append(float(np.mean(wu)))
            wp_by_n.append(float(np.mean(wp)))
            wp_sd.append(float(np.std(wp) / np.sqrt(len(wp))))
            ranks.append((min(ks), max(ks), float(np.mean(ks))))
            wp_med_by_n.append(float(np.median(wp)))
            r1 = [w for w, k in zip(wp, ks) if k == 1]
            wp_r1_by_n.append(float(np.mean(r1)) if r1 else np.nan)
            frac_r1.append(len(r1) / len(wp))
        s_u, s_p = loglog_slope(N_GRID, wu_by_n), loglog_slope(N_GRID, wp_by_n)
        s_med = loglog_slope(N_GRID, wp_med_by_n)
        s_r1 = loglog_slope(N_GRID, wp_r1_by_n)
        pred_u, pred_p = (2 * kappa - 1) / 2, (kappa - 1) / 2
        print(f"\n kappa={kappa}")
        print(f"   {'N':>9}{'W unproj':>12}{'W proj':>12}{'se':>9}"
              f"{'rank min/max/mean':>20}")
        for N, a_, b_, se, rk in zip(N_GRID, wu_by_n, wp_by_n, wp_sd, ranks):
            print(f"   {N:>9}{a_:>12.4f}{b_:>12.4f}{se:>9.4f}"
                  f"{str(rk):>20}")
        print(f"   slope unproj  measured {s_u:+.4f}   predicted {pred_u:+.4f}"
              f"   |err| {abs(s_u-pred_u):.4f}")
        print(f"   slope proj MEAN   {s_p:+.4f}   predicted {pred_p:+.4f}"
              f"   |err| {abs(s_p-pred_p):.4f}")
        print(f"   slope proj MEDIAN {s_med:+.4f}   |err| {abs(s_med-pred_p):.4f}"
              f"   proj rank==1 only {s_r1:+.4f}   |err| {abs(s_r1-pred_p):.4f}")
        print(f"   rank==1 fraction by N: "
              + " ".join(f"{f:.2f}" for f in frac_r1))


if __name__ == "__main__":
    main()
