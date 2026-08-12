"""critic_v2_slope.py -- VERIFY V2: does 'exactly N^-1' survive more N points?

The author's R2 fits slopes on THREE N points {4k, 32k, 256k} of seed-averaged
spectra (20 seeds), reporting mean -0.983, sd 0.070, n=29. Two ways that could
be flattering:

  - 3 points is the minimum for a line; curvature is invisible.
  - averaging across 20 different ENVIRONMENTS (seed varies the env) before
    fitting can cancel per-env curvature.

Here: 5 N points {4k, 16k, 64k, 256k, 1.024M}, per-seed fits (each seed = one
env + one data draw, the author's convention) AND fits on seed-averaged spectra;
middle-tier directions only (indices in [rank(P_a), min(|O|,|O_0|))).
Also reports the 3-point fit on the same data for a like-for-like check.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

from dim_separated_pomdp import default_params, sample_trajectories

CELLS = [
    (2, 6, 4, 1.0), (2, 6, 4, 0.6),
    (3, 7, 5, 1.0), (3, 7, 5, 0.6),
]
N5 = [4000, 16000, 64000, 256000, 1024000]
N3 = [4000, 64000, 1024000]
SEEDS = list(range(20))
N_ACT = 2
T = 3


def spectra_all_actions(d, n_o, n_o0, t=1):
    o, act, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(act)
    out = []
    for a in range(N_ACT):
        sel = act == a
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o[sel], O0[sel]), 1.0)
        Acr /= N
        out.append(np.linalg.eigvalsh(Acr @ Acr.T)[::-1])
    return out


def fit(ns, vals):
    v = np.maximum(np.abs(np.asarray(vals, float)), 1e-300)
    return float(np.polyfit(np.log(np.asarray(ns, float)), np.log(v), 1)[0])


def main():
    all5_seed, all3_seed, all5_avg = [], [], []
    for (n_s, n_o, n_o0, cf) in CELLS:
        # spectra[sd][N][a]
        spectra = {sd: {} for sd in SEEDS}
        for sd in SEEDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=sd, confound=cf)
            for N in N5:
                d = sample_trajectories(p, N, np.random.default_rng(1000 + sd))
                spectra[sd][N] = spectra_all_actions(d, n_o, n_o0)

        cliff = min(n_o, n_o0)
        for a in range(N_ACT):
            p0 = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                                seed=0, confound=cf)
            Pa = p0["E"].T @ np.diag(p0["p1"] * p0["pi_b"][:, a]) @ p0["K0"]
            truth = int(np.linalg.matrix_rank(Pa))
            mids = range(truth, cliff)
            if not mids:
                continue
            for i in mids:
                s5 = [fit(N5, [spectra[sd][N][a][i] for N in N5]) for sd in SEEDS]
                s3 = [fit(N3, [spectra[sd][N][a][i] for N in N3]) for sd in SEEDS]
                avg5 = fit(N5, [np.mean([spectra[sd][N][a][i] for sd in SEEDS])
                                for N in N5])
                all5_seed.extend(s5)
                all3_seed.extend(s3)
                all5_avg.append(avg5)
                print(f"({n_s},{n_o},{n_o0}) cf={cf} a={a} idx={i}: "
                      f"5pt per-seed {np.mean(s5):+.3f}+/-{np.std(s5):.3f}   "
                      f"3pt per-seed {np.mean(s3):+.3f}+/-{np.std(s3):.3f}   "
                      f"5pt seed-avg {avg5:+.3f}")

    print(f"\nOverall middle-tier slope:")
    print(f"  5-point, per-seed fits : {np.mean(all5_seed):+.3f}"
          f" +/- {np.std(all5_seed):.3f}   (n={len(all5_seed)})")
    print(f"  3-point, per-seed fits : {np.mean(all3_seed):+.3f}"
          f" +/- {np.std(all3_seed):.3f}   (n={len(all3_seed)})")
    print(f"  5-point, seed-averaged : {np.mean(all5_avg):+.3f}"
          f" +/- {np.std(all5_avg):.3f}   (n={len(all5_avg)})")


if __name__ == "__main__":
    main()
