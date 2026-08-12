"""critic_b_a7.py -- ATTACK A7: the parallel-analysis selector in its new, load-bearing role.

Three worries about my own selector now that it defines the retained subspace:

1. GEOMETRY MISMATCH. The PA basis consists of eigenvectors of the UNWEIGHTED
   cross-moment Acr Acr^T, but H (which defines the widths) is the WEIGHTED
   design (Acr * Wa) Acr^T / N + rho I. Two different matrices, two different
   eigenbases. Measure the principal angles between the PA basis and the top-k
   eigenvectors of H, and recompute leakage/projected width in H's own basis.
   If the numbers move materially, C3's leakage table depends on an arbitrary
   basis choice.

2. RANK STABILITY AT N=4,000. Last round's tests ran at N >= 8,000. The doc's
   experiment runs at 4,000 with 5 seeds. Distribution of selected k over 20
   seeds per cell, against rank(P_a).

3. PERCENTILE SENSITIVITY. q = 90 / 95 / 99 at the least stable cell.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params, sample_trajectories
from mf_pessimism import (mf_design_and_gradient, h_inv_norm, null_leakage,
                          parallel_analysis_basis)

CONFIGS = [(4, 6, 2), (2, 6, 4), (3, 7, 5)]
CONFOUNDS = [1.0, 0.6]
N = 4000
N_ACT = 2
T = 3


def pop_rank(p, a):
    return int(np.linalg.matrix_rank(
        p["E"].T @ np.diag(p["p1"] * p["pi_b"][:, a]) @ p["K0"]))


def main():
    print("[A7.1] PA basis (unweighted cross-moment) vs H's own top-k basis.\n")
    print(f"{'config':>10}{'cf':>5}{'act':>4}{'max angle':>10}"
          f"{'leak PA':>9}{'leak H':>9}{'w_proj PA':>10}{'w_proj H':>10}")
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            for a in range(N_ACT):
                angs, lpa, lh, wpa, wh = [], [], [], [], []
                for sd in range(5):
                    d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
                    nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
                    nu1 /= nu1.sum()
                    Hs, gs = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu1)
                    k, U = parallel_analysis_basis(
                        d, n_o, n_o0, a, np.random.default_rng(900 + sd))
                    ev, evec = np.linalg.eigh(Hs[a])
                    UH = evec[:, ::-1][:, :k]
                    sv = np.linalg.svd(U.T @ UH, compute_uv=False)
                    angs.append(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))
                    lpa.append(null_leakage(Hs[a], gs[a], U))
                    lh.append(null_leakage(Hs[a], gs[a], UH))
                    wpa.append(h_inv_norm(Hs[a], gs[a], U))
                    wh.append(h_inv_norm(Hs[a], gs[a], UH))
                print(f"{str((n_s,n_o,n_o0)):>10}{cf:>5.1f}{a:>4}"
                      f"{np.max(angs):>9.1f}d{np.mean(lpa):>9.4f}"
                      f"{np.mean(lh):>9.4f}{np.mean(wpa):>10.3f}"
                      f"{np.mean(wh):>10.3f}")

    print("\n[A7.2] PA rank distribution at N=4,000 (20 seeds), vs rank(P_a).\n")
    print(f"{'config':>10}{'cf':>5}{'act':>4}{'rank(P_a)':>10}   k histogram")
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            for a in range(N_ACT):
                ks = []
                for sd in range(20):
                    d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
                    k, _ = parallel_analysis_basis(
                        d, n_o, n_o0, a, np.random.default_rng(900 + sd))
                    ks.append(k)
                hist = {v: ks.count(v) for v in sorted(set(ks))}
                tr = pop_rank(p, a)
                acc = ks.count(tr) / len(ks)
                print(f"{str((n_s,n_o,n_o0)):>10}{cf:>5.1f}{a:>4}{tr:>10}"
                      f"   {hist}   acc={acc:.0%}")

    print("\n[A7.3] percentile sensitivity, (3,7,5) cf=0.6 action 0, 20 seeds.\n")
    n_s, n_o, n_o0, cf = 3, 7, 5, 0.6
    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=0,
                       confound=cf)
    for q in (90.0, 95.0, 99.0):
        ks = []
        for sd in range(20):
            d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
            k, _ = parallel_analysis_basis(
                d, n_o, n_o0, 0, np.random.default_rng(900 + sd), q=q)
            ks.append(k)
        print(f"  q={q:.0f}: {dict((v, ks.count(v)) for v in sorted(set(ks)))}")


if __name__ == "__main__":
    main()
