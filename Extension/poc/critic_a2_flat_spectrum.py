"""critic_a2_flat_spectrum.py -- ATTACK A2: is the environment rigged in the finding's favour?

Two attacks:

1. FLAT SPECTRUM. default_params draws E and K0 from Dirichlet(0.7) with a +1.5
   diagonal boost. That produces rows that are similar mixtures, hence a badly
   conditioned population cross-moment whose 2nd..|S|th singular values are tiny.
   C3 ("population rank 2 but only 1 usable direction at N up to 256k") may be a
   property of THAT conditioning, not of proximal estimation. Here E and K0 are
   rebuilt with near-disjoint block supports (90% mass on the block, 10% spread),
   giving well-separated singular values of comparable magnitude. Everything else
   (transitions, rewards, behavior policy, confounding) is untouched. The author's
   exact N-sweep + slope classifier is then re-run.

2. C1 AT SMALL EFFECTIVE N. C1 says the exact null space has dimension EXACTLY
   |O| - min(|O|, |O_0|) "at any sample size". The rank of the empirical
   cross-moment is also capped by the number of DISTINCT instrument symbols
   actually observed. An instrument symbol with probability 1e-6 is unobserved at
   N = 4000 with probability ~ e^{-0.004}, so the empirical matrix loses a column
   and the exact null space is LARGER than the formula. Same story trivially when
   N < |O_0|.

Also computed: the exact population per-action design P_a = E^T diag(p1*pi_b[:,a]) K0
and the eigenvalues of P_a P_a^T (the true limits of the author's raw_spectrum),
which say where each "statistically empty" direction would eventually flatten.
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


def block_stochastic(n_rows, n_cols, rng, concentration=0.9):
    """Rows with near-disjoint block supports: 90% of mass uniform on the row's
    own block, 10% uniform everywhere. Singular values are comparable by design."""
    M = np.full((n_rows, n_cols), (1.0 - concentration) / n_cols)
    edges = np.linspace(0, n_cols, n_rows + 1).astype(int)
    for s in range(n_rows):
        lo, hi = edges[s], max(edges[s + 1], edges[s] + 1)
        M[s, lo:hi] += concentration / (hi - lo)
    M /= M.sum(axis=1, keepdims=True)
    return M


def flat_params(n_s, n_o, n_o0, seed):
    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    p["E"] = block_stochastic(n_s, n_o, rng)
    p["K0"] = block_stochastic(n_s, n_o0, rng)
    assert np.linalg.matrix_rank(p["E"]) == min(n_s, n_o)
    assert np.linalg.matrix_rank(p["K0"]) == min(n_s, n_o0)
    return p


def raw_spectrum(d, n_o, n_o0, n_act, t=1):
    """Verbatim from run_rank_verify.py."""
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


def pop_design_spectrum(p):
    """Exact population limit of raw_spectrum: P_a = E^T diag(p1 * pi_b[:,a]) K0,
    eigenvalues of P_a P_a^T averaged over actions (matching the author's mean)."""
    E, K0, p1, pi_b = p["E"], p["K0"], p["p1"], p["pi_b"]
    specs = []
    for a in range(p["n_a"]):
        Pa = E.T @ np.diag(p1 * pi_b[:, a]) @ K0
        specs.append(np.linalg.eigvalsh(Pa @ Pa.T)[::-1])
    return np.mean(specs, axis=0)


def sweep_and_classify(param_fn, n_s, n_o, n_o0, tag):
    spectra = {}
    for N in N_SWEEP:
        per_seed = []
        for sd in SEEDS:
            p = param_fn(n_s, n_o, n_o0, sd)
            d = sample_trajectories(p, N, np.random.default_rng(100 + sd))
            per_seed.append(raw_spectrum(d, n_o, n_o0, N_ACT))
        spectra[N] = np.mean(per_seed, axis=0)

    pop = np.mean([pop_design_spectrum(param_fn(n_s, n_o, n_o0, sd))
                   for sd in SEEDS], axis=0)

    EXACT_FLOOR = 1e-12
    lam_max = spectra[N_SWEEP[-1]][0]
    tiers = {"SIGNAL": 0, "statistically empty": 0, "exact zero (shape)": 0}
    rows = []
    print(f"\n  [{tag}] |S|={n_s} |O|={n_o} |O_0|={n_o0}")
    print(f"    {'i':>3}" + "".join(f"{('N=%d' % N):>13}" for N in N_SWEEP)
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
        print(f"    {i:>3}" + "".join(f"{x:>13.3e}" for x in vals)
              + f"{pop[i]:>13.3e}{sl:>9.2f}   {v}")
    print(f"    -> {tiers['SIGNAL']} signal / {tiers['statistically empty']} "
          f"stat-empty / {tiers['exact zero (shape)']} exact zero"
          f"   (population rank = "
          f"{int(np.linalg.matrix_rank(param_fn(n_s, n_o, n_o0, 0)['E'].T @ np.diag(param_fn(n_s, n_o, n_o0, 0)['p1']) @ param_fn(n_s, n_o, n_o0, 0)['K0']))})")
    return dict(tiers=tiers, rows=rows)


def dirichlet_params(n_s, n_o, n_o0, seed):
    return default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=seed)


def attack_c1_rare_symbol():
    """C1 claims exactly |O| - min(|O|,|O_0|) exact zeros at ANY sample size.
    Make the last instrument symbol rare (p ~ 1e-6) and count exact zeros."""
    print("\n  [C1 attack] rare instrument symbol, |S|=2 |O|=6 |O_0|=4, N=4000")
    p = default_params(n_s=2, n_a=N_ACT, n_o=6, n_o0=4, T=T, seed=0)
    K0 = p["K0"].copy()
    K0[:, -1] = 1e-6
    K0 /= K0.sum(axis=1, keepdims=True)
    p["K0"] = K0
    assert np.linalg.matrix_rank(K0) == 2          # still full row rank
    out = []
    for N in (4000, 4):
        d = sample_trajectories(p, N, np.random.default_rng(7))
        seen_o0 = len(np.unique(d["O0"]))
        spec = raw_spectrum(d, 6, 4, N_ACT)
        n_exact = int((np.abs(spec) < 1e-12 * spec[0]).sum())
        pred = 6 - min(6, 4)
        print(f"    N={N:>5}: distinct o0 observed = {seen_o0}/4, "
              f"exact zeros = {n_exact}, C1 predicts {pred}  "
              f"[{'VIOLATED' if n_exact != pred else 'holds'}]")
        out.append(dict(N=N, distinct_o0=seen_o0, exact_zeros=n_exact,
                        predicted=pred, violated=bool(n_exact != pred)))
    return out


def main():
    print("=== A2.1: Dirichlet construction vs flat-spectrum construction")
    for (n_s, n_o, n_o0) in [(2, 6, 4), (3, 7, 5)]:
        RESULTS[f"dirichlet_S{n_s}"] = sweep_and_classify(
            dirichlet_params, n_s, n_o, n_o0, "dirichlet (author)")
        RESULTS[f"flat_S{n_s}"] = sweep_and_classify(
            flat_params, n_s, n_o, n_o0, "flat spectrum (critic)")

    print("\n=== A2.2: C1 'at any sample size' counterexample")
    RESULTS["c1_rare_symbol"] = attack_c1_rare_symbol()

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic_a2_flat_spectrum.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
