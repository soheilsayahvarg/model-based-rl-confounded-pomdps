"""critic_v4_converge.py -- VERIFY V4: does the (3,7,5) cf=0.6 cell EVER converge?

The 7.4 table says the fixed rule inside the estimator gives [5,3] at N=512,000
for (3,7,5) confound=0.6 (truth [3,3]). Two tasks:

1. REPRODUCE the Wa table independently. run_rank_corrected.py's R4 only runs
   N=32,000 -- no shipped script produces the 128k/512k rows, so they are
   unreproducible as published. Rebuild Wa = Ma Ma^T / N2 exactly as
   TabularBridgeEstimator does (same split, same ridge, same stage-2 columns),
   validate against the real estimator at N=32k, then extend to N = 4M.

2. PREDICT the convergence N from the population Wa spectrum. The stage-2
   population design is

     Wa_pop = sum_o0 p(a, o0) mu(.|a,o0) mu(.|a,o0)^T,
     mu(o|a,o0) = sum_s p(s|a,o0) E[s,o]

   The rule picks the truth once lambda_truth^pop / (c/N) exceeds both the
   largest within-signal ratio and the middle-to-floor ratio, where c is the
   middle tier's N^-1 constant (estimated from the empirical middle eigenvalue).
   Solve for N and compare with what actually happens.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params, sample_trajectories
from bridge_estimator import TabularBridgeEstimator

N_ACT = 2
T = 3
REL = 1e-9


def rule_new(desc):
    if desc.size < 2:
        return desc.size
    floor = max(desc[0], 0.0) * REL
    if floor <= 0.0:
        return 1
    return int(np.argmax(np.maximum(desc[:-1], floor)
                         / np.maximum(desc[1:], floor))) + 1


def wa_spectra_fast(d, n_o, n_o0, est_seed=0):
    """Replicates TabularBridgeEstimator.fit's bR_t1 stage-1 CME + stage-2 Wa
    without building Psi. Same permutation, same ridge (lam1 = 1/N1)."""
    O0, O, A = d["O0"], d["O"], d["A"]
    N = O.shape[0]
    rng = np.random.default_rng(est_seed)
    perm = rng.permutation(N)
    N1 = int(N * 0.7)
    i1, i2 = perm[:N1], perm[N1:]
    N2 = N - N1

    w = A[:, 0] * n_o + O[:, 0]
    x = A[:, 0] * n_o0 + O0
    n_w, n_x = N_ACT * n_o, N_ACT * n_o0

    cnt_wx = np.zeros((n_w, n_x))
    np.add.at(cnt_wx, (w[i1], x[i1]), 1.0)
    cnt_x = np.bincount(x[i1], minlength=n_x).astype(float)
    mu = cnt_wx / (cnt_x + 1.0)[None, :]          # N1 * lam1 = 1

    M = mu[:, x[i2]]
    specs = []
    for a in range(N_ACT):
        Ma = M[a * n_o:(a + 1) * n_o, :]
        Wa = Ma @ Ma.T / N2
        specs.append(np.linalg.eigvalsh(Wa)[::-1])
    return specs


def wa_pop_spectrum(p, a, n_o, n_o0):
    E, K0, p1, pi_b = p["E"], p["K0"], p["p1"], p["pi_b"]
    Wa = np.zeros((n_o, n_o))
    for o0 in range(n_o0):
        joint_s = p1 * pi_b[:, a] * K0[:, o0]     # p(s, a, o0)
        p_x = joint_s.sum()
        if p_x <= 0:
            continue
        mu = E.T @ (joint_s / p_x)                # p(.|a, o0)
        Wa += p_x * np.outer(mu, mu)
    return np.linalg.eigvalsh(Wa)[::-1]


def main():
    n_s, n_o, n_o0, cf = 3, 7, 5, 0.6

    # ---- validation against the real estimator at N=32k
    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=0,
                       confound=cf)
    d = sample_trajectories(p, 32000, np.random.default_rng(0))
    est = TabularBridgeEstimator(n_obs=n_o, n_act=N_ACT, n_o0=n_o0, n_r=2,
                                 T=1, mode="primal", seed=0)
    est.fit(d["O0"], d["O"][:, :1], d["A"][:, :1], d["R"][:, :1])
    sel_est = [b.shape[1] for b in est.stage_store["bR_t1"]["signal_basis"]]
    sel_fast = [rule_new(s) for s in wa_spectra_fast(d, n_o, n_o0)]
    print(f"validation at N=32k: estimator selects {sel_est}, "
          f"fast replica selects {sel_fast}  "
          f"[{'MATCH' if sel_est == sel_fast else 'MISMATCH -- STOP'}]")
    if sel_est != sel_fast:
        return

    # ---- population Wa spectrum and predicted convergence N
    print("\npopulation Wa spectra (seed-0 environment):")
    for a in range(N_ACT):
        ev = wa_pop_spectrum(p, a, n_o, n_o0)
        print(f"  action {a}: " + " ".join(f"{v:.3e}" for v in ev))

    # ---- the table, reproduced and extended
    print(f"\n(3,7,5) cf=0.6, truth [3,3]; author's table says [5,5] at 32k/128k,"
          f" [5,3] at 512k")
    print(f"{'N':>10}  seed0    seed1    seed2")
    for N in (32000, 128000, 512000, 1024000, 2048000, 4096000):
        row = []
        for sd in (0, 1, 2):
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=sd, confound=cf)
            d = sample_trajectories(p, N, np.random.default_rng(sd))
            row.append([rule_new(s) for s in wa_spectra_fast(d, n_o, n_o0)])
        print(f"{N:>10}  " + "  ".join(str(r) for r in row))

    # ---- where does the middle tier sit vs the smallest signal eigenvalue?
    print("\nWa spectrum detail, seed 0, action 0:")
    for N in (512000, 2048000, 4096000):
        p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=0,
                           confound=cf)
        d = sample_trajectories(p, N, np.random.default_rng(0))
        desc = wa_spectra_fast(d, n_o, n_o0)[0]
        print(f"  N={N:>8}: " + " ".join(f"{v:.3e}" for v in desc))


if __name__ == "__main__":
    main()
