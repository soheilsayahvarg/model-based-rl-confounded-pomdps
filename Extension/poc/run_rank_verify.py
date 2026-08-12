"""run_rank_verify.py -- does the latent bottleneck produce a HARD null space, or a soft one?

The first diagnostic reported normalized eigenvalues, which cannot distinguish
"this direction is structurally empty" from "this direction is weak and shrinking
at the same rate as everything else". That distinction is the whole question, so
it gets its own check.

Two regimes are separated here:

  |O_0| < |O| : the instrument has fewer columns than the bridge has rows, so
                the cross-moment CANNOT have more than |O_0| nonzero directions
                at ANY sample size. This is exact linear algebra, not asymptotics.

  |O_0| >= |O|: no such shape constraint. The population matrix still has rank
                min(|S|,...) because it factors through the latent state, but the
                empirical matrix has no reason to be exactly rank-deficient. The
                question is whether the excess directions COLLAPSE (structural,
                vanishing faster than noise) or merely shrink at the sampling rate
                (soft, indistinguishable from noise).

DISCRIMINATOR. Fit a log-log slope of each eigenvalue against N. A direction that
is genuinely absent from the population decays at the sampling rate of a squared
mean, ~N^-1. A direction carrying real signal flattens to a constant, slope ~0.
Comparing slopes is scale-free, so it is immune to the normalization problem that
made the first pass ambiguous.

Raw (unnormalized) eigenvalues are reported throughout.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import (default_params, sample_trajectories,
                                 population_cross_moment)

N_SWEEP = [4000, 16000, 64000, 256000]
SEEDS = [0, 1, 2]
N_ACT = 2
T = 3
CONFIGS = [
    (2, 6, 4, "|O_0| > |S|: is the latent cap hard or soft?"),
    (3, 7, 5, "|O_0| > |S|: same question, different dims"),
    (4, 6, 2, "|O_0| < |O|: shape forces exact rank deficiency"),
]
RESULTS = {}


def raw_spectrum(d, n_o, n_o0, n_act, t=1):
    """Unnormalized eigenvalues of the per-action bridge-space design, averaged
    over actions. Counts are divided by N so the matrix has a population limit."""
    o_t, a_t, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(a_t)
    specs = []
    for a in range(n_act):
        sel = a_t == a
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o_t[sel], O0[sel]), 1.0)
        Acr /= N                                   # -> p(o, o_0, a), has a limit
        design = Acr @ Acr.T                       # (n_o, n_o), bridge space
        specs.append(np.linalg.eigvalsh(design)[::-1])
    return np.mean(specs, axis=0)


def loglog_slope(ns, vals):
    """Slope of log(eigenvalue) against log(N). ~0 means it converges to a
    nonzero limit; ~-1 means it is pure sampling noise around zero."""
    v = np.maximum(np.asarray(vals, float), 1e-300)
    return float(np.polyfit(np.log(np.asarray(ns, float)), np.log(v), 1)[0])


def main():
    rows = []
    for (n_s, n_o, n_o0, note) in CONFIGS:
        p0 = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=0)
        pop = population_cross_moment(p0)
        pop_rank = int(np.linalg.matrix_rank(pop))
        shape_cap = min(n_o, n_o0)

        print(f"\n=== |S|={n_s} |O|={n_o} |O_0|={n_o0} -- {note}")
        print(f"    population rank = {pop_rank}   shape cap min(|O|,|O_0|) = {shape_cap}")

        spectra = {}
        for N in N_SWEEP:
            per_seed = []
            for sd in SEEDS:
                p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                   T=T, seed=sd)
                d = sample_trajectories(p, N, np.random.default_rng(100 + sd))
                per_seed.append(raw_spectrum(d, n_o, n_o0, N_ACT))
            spectra[N] = np.mean(per_seed, axis=0)

        # A flat line of floating-point noise also has slope ~0, so the slope
        # alone cannot separate "real signal" from "exact zero". Classify on both
        # magnitude and slope. EXACT_FLOOR is relative to the top eigenvalue.
        EXACT_FLOOR = 1e-12
        lam_max = spectra[N_SWEEP[-1]][0]

        def classify(vals, slope):
            if abs(vals[-1]) < EXACT_FLOOR * lam_max:
                return "exact zero (shape)"
            return "SIGNAL" if slope > -0.5 else "statistically empty"

        print(f"    {'index':>6}" + "".join(f"{('N=%d' % N):>13}" for N in N_SWEEP)
              + f"{'slope':>9}   verdict")
        tiers = {"SIGNAL": 0, "statistically empty": 0, "exact zero (shape)": 0}
        for i in range(n_o):
            vals = [spectra[N][i] for N in N_SWEEP]
            sl = loglog_slope(N_SWEEP, vals)
            verdict = classify(vals, sl)
            tiers[verdict] += 1
            print(f"    {i:>6}" + "".join(f"{v:>13.3e}" for v in vals)
                  + f"{sl:>9.2f}   {verdict}")
            rows.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, index=i,
                             values=[float(v) for v in vals], slope=sl,
                             tier=verdict, pop_rank=pop_rank,
                             shape_cap=shape_cap))

        print(f"    -> {tiers['SIGNAL']} signal, "
              f"{tiers['statistically empty']} statistically empty, "
              f"{tiers['exact zero (shape)']} exact zero")
        print(f"       predicted exact zeros = |O| - min(|O|,|O_0|) = "
              f"{n_o - shape_cap}  "
              f"[{'OK' if tiers['exact zero (shape)'] == n_o - shape_cap else 'MISMATCH'}]")
        print(f"       population rank = {pop_rank}, but only {tiers['SIGNAL']} "
              f"direction(s) rise above the sampling floor")
        RESULTS[f"S{n_s}_O{n_o}_O0{n_o0}"] = dict(
            note=note, pop_rank=pop_rank, shape_cap=shape_cap,
            n_signal=tiers["SIGNAL"],
            n_statistically_empty=tiers["statistically empty"],
            n_exact_zero=tiers["exact zero (shape)"],
            predicted_exact_zeros=n_o - shape_cap,
            exact_zero_prediction_ok=bool(
                tiers["exact zero (shape)"] == n_o - shape_cap))

    RESULTS["per_direction"] = rows
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_rank_verify.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
