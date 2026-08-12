"""critic_a3_classifier.py -- ATTACK A3: is the tier classifier doing the work?

Three sub-attacks on (2,6,4), confound=1.0 (the doc's flagship config):

1. SEED SEMANTICS. The author's scripts pass seed=sd to default_params AND tie
   the data rng to sd, so each "seed" is a DIFFERENT ENVIRONMENT. Slopes are then
   fit on spectra averaged across three different populations. Here: 20 seeds
   under (a) the author's convention and (b) a fixed environment (env seed 0)
   with 20 independent data draws — per-seed slope fits, mean +/- sd per index.

2. IS -0.80 A DISTINCT RATE? With per-action population values known to be
   exactly zero (A2), the truth is N^-1 sampling decay. Test whether the
   author's -0.80 for index 1 is compatible with -1 given seed noise, or a
   genuinely slower rate.

3. THRESHOLD SENSITIVITY. Tier counts across slope thresholds
   {-0.2, -0.35, -0.5, -0.65, -0.8} x exact floors {1e-10, 1e-12, 1e-14}.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

from dim_separated_pomdp import default_params, sample_trajectories

N_SWEEP = [4000, 16000, 64000, 256000]
N_SEEDS = 20
N_ACT = 2
T = 3
N_S, N_O, N_O0 = 2, 6, 4


def raw_spectrum(d, n_o, n_o0, n_act, t=1):
    o_t, a_t, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(a_t)
    specs = []
    for a in range(n_act):
        sel = a_t == a
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o_t[sel], O0[sel]), 1.0)
        Acr /= N
        specs.append(np.linalg.eigvalsh(Acr @ Acr.T)[::-1])
    return np.mean(specs, axis=0)


def slope(vals):
    v = np.maximum(np.asarray(vals, float), 1e-300)
    return float(np.polyfit(np.log(np.asarray(N_SWEEP, float)), np.log(v), 1)[0])


def collect(fixed_env):
    """spectra[seed][N] -> (n_o,) eigenvalues."""
    out = []
    for sd in range(N_SEEDS):
        env_seed = 0 if fixed_env else sd
        p = default_params(n_s=N_S, n_a=N_ACT, n_o=N_O, n_o0=N_O0, T=T,
                           seed=env_seed)
        per_n = {}
        for N in N_SWEEP:
            d = sample_trajectories(p, N, np.random.default_rng(100 + sd))
            per_n[N] = raw_spectrum(d, N_O, N_O0, N_ACT)
        out.append(per_n)
    return out


def report(tag, runs):
    print(f"\n=== {tag} ({N_SEEDS} seeds)")
    print(f"  {'i':>3}{'slope mean':>12}{'slope sd':>10}{'t vs -1':>9}"
          f"{'mean@N=256k':>14}")
    slopes_by_index = {}
    for i in range(N_O):
        sls = [slope([r[N][i] for N in N_SWEEP]) for r in runs]
        sls = np.array(sls)
        slopes_by_index[i] = sls
        mean_last = np.mean([r[N_SWEEP[-1]][i] for r in runs])
        tstat = (sls.mean() + 1.0) / (sls.std(ddof=1) / np.sqrt(len(sls)))
        print(f"  {i:>3}{sls.mean():>12.3f}{sls.std(ddof=1):>10.3f}"
              f"{tstat:>9.2f}{mean_last:>14.3e}")
    return slopes_by_index


def threshold_sensitivity(runs):
    print("\n=== Tier counts vs classifier constants (seed-averaged spectra)")
    spec = {N: np.mean([r[N] for r in runs], axis=0) for N in N_SWEEP}
    print(f"  {'floor':>8} | " + "  ".join(f"thr={t:>5}" for t in
                                           (-0.2, -0.35, -0.5, -0.65, -0.8)))
    for floor in (1e-10, 1e-12, 1e-14):
        lam_max = spec[N_SWEEP[-1]][0]
        cells = []
        for thr in (-0.2, -0.35, -0.5, -0.65, -0.8):
            t_counts = [0, 0, 0]                     # signal, empty, zero
            for i in range(N_O):
                vals = [spec[N][i] for N in N_SWEEP]
                if abs(vals[-1]) < floor * lam_max:
                    t_counts[2] += 1
                elif slope(vals) > thr:
                    t_counts[0] += 1
                else:
                    t_counts[1] += 1
            cells.append(f"{t_counts[0]}/{t_counts[1]}/{t_counts[2]}")
        print(f"  {floor:>8.0e} | " + "  ".join(f"{c:>9}" for c in cells))


def main():
    author = collect(fixed_env=False)
    fixed = collect(fixed_env=True)
    report("author's convention: environment varies with seed", author)
    report("fixed environment (env seed 0), 20 data draws", fixed)
    threshold_sensitivity(fixed)


if __name__ == "__main__":
    main()
