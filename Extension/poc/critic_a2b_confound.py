"""critic_a2b_confound.py -- ATTACK A2 (continued): the per-action population rank.

Discovery from critic_a2_flat_spectrum.py: with the default confound=1.0 the
behavior policy is DETERMINISTIC in the latent state (pi_b rows are one-hot), so
conditioning on the action pins the state to {s : s % n_a == a}. The population
limit of the per-action design P_a = E^T diag(p1 * pi_b[:,a]) K0 then has rank
|{s : s % n_a == a}|, NOT min(|S|, |O|, |O_0|).

Hypothesis to verify: C3's "population rank 2 but only 1 usable direction" is
fully explained by rank(P_a) = 1 -- nothing is statistically hidden; the second
direction is absent from the population object the empirical spectrum estimates.
If so, lowering confound below 1 (stochastic behavior policy, still confounded)
must bring the second direction back as genuine SIGNAL with a nonzero population
limit.

Runs (2,6,4) and toy-dims (2,3,2), Dirichlet and flat constructions, confound in
{1.0, 0.6}. Prints rank(P_a) per action, population eigenvalues, and the author's
empirical tier classification.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

from dim_separated_pomdp import default_params, sample_trajectories

N_SWEEP = [4000, 16000, 64000, 256000]
SEEDS = [0, 1, 2]
N_ACT = 2
T = 3
RESULTS = {}


def block_stochastic(n_rows, n_cols, concentration=0.9):
    M = np.full((n_rows, n_cols), (1.0 - concentration) / n_cols)
    edges = np.linspace(0, n_cols, n_rows + 1).astype(int)
    for s in range(n_rows):
        lo, hi = edges[s], max(edges[s + 1], edges[s] + 1)
        M[s, lo:hi] += concentration / (hi - lo)
    M /= M.sum(axis=1, keepdims=True)
    return M


def make_params(n_s, n_o, n_o0, seed, confound, flat):
    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=seed,
                       confound=confound)
    if flat:
        p["E"] = block_stochastic(n_s, n_o)
        p["K0"] = block_stochastic(n_s, n_o0)
    return p


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


def loglog_slope(ns, vals):
    v = np.maximum(np.asarray(vals, float), 1e-300)
    return float(np.polyfit(np.log(np.asarray(ns, float)), np.log(v), 1)[0])


def run_case(n_s, n_o, n_o0, confound, flat):
    tag = (f"|S|={n_s} |O|={n_o} |O_0|={n_o0} confound={confound} "
           f"{'flat' if flat else 'dirichlet'}")
    print(f"\n  --- {tag}")

    # population per-action design and its rank (seed 0)
    p0 = make_params(n_s, n_o, n_o0, 0, confound, flat)
    pop_specs, pa_ranks = [], []
    for a in range(N_ACT):
        Pa = p0["E"].T @ np.diag(p0["p1"] * p0["pi_b"][:, a]) @ p0["K0"]
        pa_ranks.append(int(np.linalg.matrix_rank(Pa)))
        pop_specs.append(np.linalg.eigvalsh(Pa @ Pa.T)[::-1])
    pop = np.mean(pop_specs, axis=0)
    marg_rank = int(np.linalg.matrix_rank(
        p0["E"].T @ np.diag(p0["p1"]) @ p0["K0"]))
    print(f"      action-marginal pop rank = {marg_rank}   "
          f"rank(P_a) per action = {pa_ranks}")

    spectra = {}
    for N in N_SWEEP:
        per_seed = []
        for sd in SEEDS:
            p = make_params(n_s, n_o, n_o0, sd, confound, flat)
            d = sample_trajectories(p, N, np.random.default_rng(100 + sd))
            per_seed.append(raw_spectrum(d, n_o, n_o0, N_ACT))
        spectra[N] = np.mean(per_seed, axis=0)

    EXACT_FLOOR = 1e-12
    lam_max = spectra[N_SWEEP[-1]][0]
    tiers = {"SIGNAL": 0, "statistically empty": 0, "exact zero (shape)": 0}
    rows = []
    print(f"      {'i':>3}" + "".join(f"{('N=%d' % N):>13}" for N in N_SWEEP)
          + f"{'pop limit':>13}{'slope':>9}   verdict")
    for i in range(n_o):
        vals = [spectra[N][i] for N in N_SWEEP]
        sl = loglog_slope(N_SWEEP, vals)
        if abs(vals[-1]) < EXACT_FLOOR * lam_max:
            v = "exact zero (shape)"
        else:
            v = "SIGNAL" if sl > -0.5 else "statistically empty"
        tiers[v] += 1
        rows.append(dict(index=i, values=[float(x) for x in vals],
                         pop_limit=float(pop[i]), slope=sl, tier=v))
        print(f"      {i:>3}" + "".join(f"{x:>13.3e}" for x in vals)
              + f"{pop[i]:>13.3e}{sl:>9.2f}   {v}")
    print(f"      -> {tiers['SIGNAL']} signal / {tiers['statistically empty']} "
          f"stat-empty / {tiers['exact zero (shape)']} exact zero")
    return dict(tag=tag, marginal_pop_rank=marg_rank, pa_ranks=pa_ranks,
                tiers=tiers, rows=rows)


def main():
    for (n_s, n_o, n_o0) in [(2, 6, 4), (2, 3, 2), (3, 7, 5)]:
        for flat in (False, True):
            for confound in (1.0, 0.6):
                key = f"S{n_s}_O{n_o}_O0{n_o0}_{'flat' if flat else 'dir'}_c{confound}"
                RESULTS[key] = run_case(n_s, n_o, n_o0, confound, flat)

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic_a2b_confound.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
