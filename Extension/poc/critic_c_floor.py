"""critic_c_floor.py -- step (c) critic, Priority 3d (+ un-vacuating 4.3).

The doc's post-hoc refinement says xi_needed ~ C/N2 + lam0*beta^2*N2^-kappa,
which implies (a) a FLOOR c > sigma2*C below which coverage fails at EVERY N,
and (b) a refined crossing N2* = ((c/sigma2 - C)/(lam0*beta^2))^2 for c above
the floor. Both were derived after seeing the data and never tested.

Key observation: for c only slightly above the floor, the refined N2* lands
INSIDE the reachable grid -- so the coverage-crossing prediction, reported in
5.5 as untestable (grid 1600x short), becomes testable after all. The doc
declared its own headline crossing test vacuous when a c 2.7x above the floor
made N* ~ 1.7e9; it did not notice that moving c toward the floor pulls the
crossing onto the grid.

Protocol: fit once per (N, seed) (fits do not depend on c), measure xi_needed,
sigma2, beta, C per fit; PRE-REGISTER predicted crossings from the SMALLEST-N
fits only (so the prediction does not use the data it is tested on); then
evaluate coverage for a c-grid spanning the floor.
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
from run_noncontraction import fit_toy, LAM0, SEEDS

KAPPA = 0.5
GRID = [1000, 4000, 16000, 64000, 256000, 1024000]


def main():
    params = toy.default_params(T=3)
    bR_true, _, _ = om.toy_true_bridges(params)
    b_true = np.asarray(bR_true).ravel()

    fits = []
    for N in GRID:
        for sd in SEEDS:
            _, blk, lam2, N2 = fit_toy(N, KAPPA, sd)
            ev, evec = np.linalg.eigh(blk.H)
            null_mask = (ev - lam2) <= 1e-9 * max(ev[-1], 1.0)
            Un = evec[:, null_mask]
            beta = float(np.linalg.norm(Un.T @ b_true))
            need = blk.xi_needed(b_true)
            C_sig = max(need - lam2 * beta ** 2, 0.0) * N2
            fits.append(dict(N=N, sd=sd, N2=N2, lam2=lam2, need=need,
                             sigma2=blk.sigma2, beta=beta, C_sig=C_sig))
            print(f"  fit N={N:>8} sd={sd}  sigma2={blk.sigma2:.4e}  "
                  f"beta={beta:.4f}  C_sig={C_sig:.4f}  xi_needed={need:.4e}")

    # ---- prediction calibrated ONLY on the smallest N (out-of-sample forward)
    small = [f for f in fits if f["N"] == GRID[0]]
    sig2_hat = float(np.mean([f["sigma2"] for f in small]))
    beta_hat = float(np.mean([f["beta"] for f in small]))
    C_hat = float(np.mean([f["C_sig"] for f in small]))
    floor = sig2_hat * C_hat
    print(f"\ncalibrated on N={GRID[0]} only: sigma2={sig2_hat:.4e}  "
          f"beta={beta_hat:.4f}  C={C_hat:.4f}  ->  floor c = {floor:.5f}")

    def n2_star(c):
        x = c / sig2_hat - C_hat
        if x <= 0:
            return 0.0                      # fails at every N
        return (x / (LAM0 * beta_hat ** 2)) ** 2

    # c targets: below floor, at floor, and crossings targeted at grid points
    n2_of = {N: [f["N2"] for f in fits if f["N"] == N][0] for N in GRID}
    targets = [0.7 * floor, 0.98 * floor, 1.02 * floor]
    for N_t in (16000, 64000, 256000):
        c_t = sig2_hat * (C_hat + LAM0 * beta_hat ** 2 * np.sqrt(n2_of[N_t])
                          / sig2_hat)
        # NB solve c/sig2 - C = lam0*beta^2*sqrt(N2_t)  =>  c = sig2*(C + ...)
        c_t = sig2_hat * C_hat + sig2_hat * LAM0 * beta_hat ** 2 * np.sqrt(
            n2_of[N_t]) / 1.0
        targets.append(float(c_t))
    targets.append(0.03)

    print(f"\n{'c':>10}{'pred N2*':>12}{'pred first-fail N':>18}   "
          f"coverage by N (frac of seeds covered)")
    header = "".join(f"{N:>10}" for N in GRID)
    print(f"{'':>40}{header}")
    for c in targets:
        ns = n2_star(c)
        if ns == 0.0:
            pred = "ALL N"
        else:
            fail_N = [N for N in GRID if n2_of[N] > ns]
            pred = str(fail_N[0]) if fail_N else "beyond grid"
        cov = []
        for N in GRID:
            fl = [f for f in fits if f["N"] == N]
            cov.append(sum(c / (f["N2"] * max(f["sigma2"], 1e-12)) >= f["need"]
                           for f in fl) / len(fl))
        cov_s = "".join(f"{v:>10.2f}" for v in cov)
        print(f"{c:>10.5f}{ns:>12.3e}{pred:>18}   {cov_s}")

    # constancy of C_sig across the grid (the refinement treats it as constant)
    print("\nC_sig by N (mean over seeds):")
    for N in GRID:
        cs = [f["C_sig"] for f in fits if f["N"] == N]
        print(f"  N={N:>8}  C_sig = {np.mean(cs):.4f} +- {np.std(cs):.4f}")


if __name__ == "__main__":
    main()
