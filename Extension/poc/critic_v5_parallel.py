"""critic_v5_parallel.py -- VERIFY V5: a gap-free rank selection, built and tested.

The step-(b) plan restricts to the regime where the eigengap rule happens to be
reliable. A rank selector that does not look for a gap at all removes the
restriction. Candidate: PARALLEL ANALYSIS (Horn 1965, standard in factor
analysis), adapted to the per-action cross-moment design:

  per action a:
    1. form the design eigenvalues lambda_1 >= ... >= lambda_{|O|} as usual;
    2. B times: permute O0 within the action-a rows (destroys the (o_t, o_0)
       dependence, preserves both marginals and the bin size), recompute the
       spectrum -> null distribution of each ordered eigenvalue under
       "no signal";
    3. keep k = number of leading eigenvalues exceeding the 95th percentile of
       their permutation-null counterpart (stop at the first failure).

  The middle tier is sampling noise around a zero population value decaying at
  N^-1 (V2/A3) -- exactly what the permutation null reproduces -- so the
  selector needs no gap, no floor constant, and no N-threshold.

Test on the two regimes at SMALL N, where the fixed eigengap rule fails:
  (3,7,5) cf=0.6  truth [3,3]  (eigengap needs N >> 512k inside the estimator)
  (2,6,4) cf=1.0  truth [1,1]
  (2,6,4) cf=0.6  truth [2,2]
  (4,6,2) cf=1.0  truth [2,2]  (control: both should work here)
20 seeds; report modal selection and accuracy vs rank(P_a), for parallel
analysis AND the fixed eigengap rule, at N in {8000, 32000, 128000}.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

from dim_separated_pomdp import default_params, sample_trajectories

CELLS = [
    (3, 7, 5, 0.6),
    (2, 6, 4, 1.0),
    (2, 6, 4, 0.6),
    (4, 6, 2, 1.0),
]
N_SWEEP = [8000, 32000, 128000]
SEEDS = list(range(20))
N_ACT = 2
T = 3
B_PERM = 40
Q = 95


def design_spectrum(o_sel, O0_sel, n_o, n_o0, N_total):
    Acr = np.zeros((n_o, n_o0))
    np.add.at(Acr, (o_sel, O0_sel), 1.0)
    Acr /= N_total
    return np.linalg.eigvalsh(Acr @ Acr.T)[::-1]


def rule_eigengap(desc, rel=1e-9):
    if desc.size < 2:
        return desc.size
    floor = max(desc[0], 0.0) * rel
    if floor <= 0.0:
        return 1
    return int(np.argmax(np.maximum(desc[:-1], floor)
                         / np.maximum(desc[1:], floor))) + 1


def rule_parallel(o_sel, O0_sel, n_o, n_o0, N_total, rng):
    """Permutation null preserves BOTH marginals, so the null design itself has
    the rank-1 product-of-marginals direction: lambda_1^null ~ lambda_1^obs by
    construction, and the permutation test can only detect DEPENDENCE
    directions, of which a rank-r joint has exactly r - 1 (the marginal
    direction lies in the span, and P_a minus its marginal outer product has
    rank r - 1). Hence rank = 1 + #{i >= 2 : lambda_i > q95(lambda_i^null)}.
    A first draft compared index 1 too and returned k=0 whenever the top
    direction was marginal-dominated -- wrong semantics, kept here as
    rule_parallel_naive for the record."""
    obs = design_spectrum(o_sel, O0_sel, n_o, n_o0, N_total)
    null = np.empty((B_PERM, n_o))
    for b in range(B_PERM):
        null[b] = design_spectrum(o_sel, rng.permutation(O0_sel),
                                  n_o, n_o0, N_total)
    thresh = np.percentile(null, Q, axis=0)
    if len(o_sel) == 0:
        return 0
    k = 1
    for i in range(1, n_o):
        if obs[i] > thresh[i]:
            k += 1
        else:
            break
    naive = 0
    for i in range(n_o):
        if obs[i] > thresh[i]:
            naive += 1
        else:
            break
    return k, naive


def main():
    results = []
    for (n_s, n_o, n_o0, cf) in CELLS:
        p0 = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                            seed=0, confound=cf)
        truth = [int(np.linalg.matrix_rank(
            p0["E"].T @ np.diag(p0["p1"] * p0["pi_b"][:, a]) @ p0["K0"]))
            for a in range(N_ACT)]
        print(f"\n=== ({n_s},{n_o},{n_o0}) cf={cf}  truth={truth}")
        for N in N_SWEEP:
            pa_sel = {a: [] for a in range(N_ACT)}
            nv_sel = {a: [] for a in range(N_ACT)}
            eg_sel = {a: [] for a in range(N_ACT)}
            for sd in SEEDS:
                p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                   T=T, seed=sd, confound=cf)
                d = sample_trajectories(p, N, np.random.default_rng(1000 + sd))
                o, act, O0 = d["O"][:, 0], d["A"][:, 0], d["O0"]
                prng = np.random.default_rng(50000 + sd)
                for a in range(N_ACT):
                    sel = act == a
                    k, naive = rule_parallel(o[sel], O0[sel], n_o, n_o0,
                                             N, prng)
                    pa_sel[a].append(k)
                    nv_sel[a].append(naive)
                    eg_sel[a].append(rule_eigengap(
                        design_spectrum(o[sel], O0[sel], n_o, n_o0, N)))
            line = f"  N={N:>7}:"
            for a in range(N_ACT):
                mp = max(set(pa_sel[a]), key=pa_sel[a].count)
                ap = np.mean([k == truth[a] for k in pa_sel[a]])
                mn = max(set(nv_sel[a]), key=nv_sel[a].count)
                an = np.mean([k == truth[a] for k in nv_sel[a]])
                me = max(set(eg_sel[a]), key=eg_sel[a].count)
                ae = np.mean([k == truth[a] for k in eg_sel[a]])
                line += (f"   a{a}: PA {mp}({ap:.0%})"
                         f" naive {mn}({an:.0%})"
                         f" EG {me}({ae:.0%})")
                results.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, cf=cf, N=N,
                                    action=a, truth=truth[a],
                                    parallel_modal=int(mp),
                                    parallel_acc=float(ap),
                                    naive_modal=int(mn),
                                    naive_acc=float(an),
                                    eigengap_modal=int(me),
                                    eigengap_acc=float(ae)))
            print(line)

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic_v5_parallel.json"))
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
