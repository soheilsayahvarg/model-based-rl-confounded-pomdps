"""critic_c_beta.py -- step (c) critic: is beta an environment constant or a
finite-sample artifact?

(H4) asserts beta = ||P_Nul b_true|| > 0 and treats it as a property of the
environment; 5.5 measures 0.0532 by AVERAGING per-fit values over the N grid,
and every threshold statement (N* = 1.7e9, N2* ~ beta^-4, "10x the null share
crosses at 50,000", the m_null level) treats it as constant.

The floor-test fits show per-fit beta FALLING with N (0.13 at N=1,000 to 0.007
at N=1,024,000). Hypothesis: the toy's true bridge is the MIN-NORM solution
(oracle_module builds it by pinv), which lies in range(E^T); the POPULATION
design span per action equals range(E^T) whenever the two O_0-profiles are
independent; hence the population null is range(E^T)-perp and beta_pop = 0
EXACTLY. The measured beta is then only the O(N^-1/2) misalignment of the
EMPIRICAL null with the population null -- a sampling artifact, not an
environment property -- and the true null-direction coverage margin
xi/(lam*beta(N)^2) does not degrade at N^e at all.

Test: (1) build the population t=1 design profiles exactly from toy params,
check their span, compute beta_pop; (2) measure the slope of per-fit beta vs N;
(3) recompute the m_null slope with the PER-FIT beta instead of the frozen one.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
import oracle_module as om
from run_noncontraction import fit_toy, loglog_slope, LAM0, SEEDS

KAPPA = 0.5
C_PHASE3 = 0.03
GRID = [1000, 4000, 16000, 64000, 256000, 1024000]


def population_null_basis(params):
    """Exact population t=1 design profiles per action and the population null.

    x = (a1, o0). mu(w=(a,o~)|x=(a',o0)) = delta(a,a') * sum_s P(s|a',o0) E[s,o~]
    with P(s|a,o0) prop p1(s) K0[s,o0] pi_b(a|s). Per action the o~-profiles over
    o0 span a subspace of range(E^T); if that span is 2-dim it IS range(E^T).
    """
    E, p1, pi_b, K0 = (params[k] for k in ("E", "p1", "pi_b", "K0"))
    n_o, n_s = toy.N_O, toy.N_S
    profiles = {}
    for a in range(toy.N_A):
        vs = []
        for o0 in range(toy.N_O0):
            w = p1 * K0[:, o0] * pi_b[:, a]
            w = w / w.sum()                         # P(s | a, o0)
            vs.append(w @ E)                        # (n_o,) profile
        profiles[a] = np.stack(vs)                  # (n_o0, n_o)
    # per-action population null of the o~-profile span
    nulls = {}
    for a, V in profiles.items():
        rank = np.linalg.matrix_rank(V, tol=1e-12)
        # orthonormal complement of span(V.T)
        q, _ = np.linalg.qr(V.T, mode="complete")
        nulls[a] = q[:, rank:]                      # (n_o, n_o - rank)
        print(f"  action {a}: profile rank = {rank}  "
              f"null dim = {n_o - rank}")
    # range(E^T) and its complement, for comparison
    qE, _ = np.linalg.qr(E.T, mode="complete")
    perpE = qE[:, np.linalg.matrix_rank(params['E'], tol=1e-12):]
    for a in range(toy.N_A):
        align = np.abs(nulls[a].T @ perpE)
        print(f"  action {a}: |<null_a, perp(range E^T)>| = {align.ravel()}")
    return nulls


def beta_of(b_true_mat, nulls, n_y):
    """||P_Nul b_true|| with Nul = per-action null (x) I_y (flat w*n_y+y)."""
    total = 0.0
    n_obs = toy.N_O
    B = b_true_mat.reshape(toy.N_A * n_obs, n_y)
    for a, Un in nulls.items():
        blockrows = B[a * n_obs:(a + 1) * n_obs, :]     # (n_o, n_y)
        total += float(np.sum((Un.T @ blockrows) ** 2))
    return np.sqrt(total)


def main():
    params = toy.default_params(T=3)
    bR_true, _, _ = om.toy_true_bridges(params)
    b_true = np.asarray(bR_true).ravel()

    print("population design span / null (t=1, per action):")
    nulls = population_null_basis(params)
    n_y = 2 * toy.N_O
    beta_pop = beta_of(b_true, nulls, n_y)
    print(f"\n  beta_pop = ||P_Nul_pop b_true|| = {beta_pop:.3e}   "
          f"(H4 claims an environment constant > 0)")

    print("\nempirical beta per fit (same construction as part C):")
    rows = []
    for N in GRID:
        for sd in SEEDS:
            _, blk, lam2, N2 = fit_toy(N, KAPPA, sd)
            ev, evec = np.linalg.eigh(blk.H)
            null_mask = (ev - lam2) <= 1e-9 * max(ev[-1], 1.0)
            Un = evec[:, null_mask]
            beta = float(np.linalg.norm(Un.T @ b_true))
            xi = blk.width_rule(C_PHASE3)
            rows.append(dict(N=N, sd=sd, beta=beta, lam2=lam2, xi=xi,
                             dnull=int(null_mask.sum())))
    byN = lambda k: [float(np.mean([r[k] for r in rows if r["N"] == N]))
                     for N in GRID]
    beta_n = byN("beta")
    s_beta = loglog_slope(GRID, beta_n)
    print(f"  {'N':>9}{'beta (mean)':>13}{'null dim':>10}")
    for N, b in zip(GRID, beta_n):
        d = sorted({r["dnull"] for r in rows if r["N"] == N})
        print(f"  {N:>9}{b:>13.4f}{str(d):>10}")
    print(f"  slope of beta vs N: {s_beta:+.4f}   "
          f"(a constant would be 0; pure null-misalignment noise is -0.5)")
    print(f"  grid-average beta = {np.mean([r['beta'] for r in rows]):.4f}   "
          f"(the doc's 'environment constant' 0.0532)")

    # the m_null slope with per-fit beta instead of the frozen average
    m_frozen = [byN("xi")[i] / (byN("lam2")[i] * np.mean([r['beta'] for r in rows]) ** 2)
                for i in range(len(GRID))]
    m_true = [byN("xi")[i] / (byN("lam2")[i] * max(beta_n[i], 1e-12) ** 2)
              for i in range(len(GRID))]
    print(f"\n  m_null slope, frozen beta: {loglog_slope(GRID, m_frozen):+.4f}  "
          f"(doc: -0.4965, predicted e = -0.5)")
    print(f"  m_null slope, per-fit beta: {loglog_slope(GRID, m_true):+.4f}  "
          f"(positive = the true null margin IMPROVES with N)")


if __name__ == "__main__":
    main()
