"""critic_c_decomp.py -- step (c) critic, Priority 3a + 2d.

Direct test of the Section 5.3 residual explanation and the A_N = Theta(1) claim.

The doc explains the negative signal-convention residuals as finite-lam
contamination from the retained signal term: W^2 = xi * (A_N + beta_g^2 / lam).
That explanation was written post hoc and never tested. Here:

  (1) verify the decomposition identity per fit (it is exact spectral algebra;
      if it fails the null-mask construction is wrong),
  (2) fit W^2/xi = A + B/lam with A, B CONSTANT across the N grid (the test the
      author asked for) and compare fitted B to beta_g^2, fitted A to A_N,
  (3) decompose the measured slope residual into its actual sources:
      d log W / d log N = e/2 + 0.5*d log(sigma2)/dlogN... explicitly,
        slope(W) = 0.5*slope(xi) + 0.5*slope(Q),  Q := ||g||^2_{H^-1}
        slope(xi) = -1 - slope(sigma2)   (signal convention, N2 prop N)
        slope(Q)  = mixture of the A-term and the B/lam term + drift of A_N,
                    beta_g^2 across N.
      The doc's story blames ONLY the A-term mixture. If sigma2 drift or
      beta_g drift contributes materially, the story is incomplete.
  (4) A_N = Theta(1) check: slope of A_N vs N, slope of sigma2_signal vs N,
      and the smallest RETAINED eigenvalue of H|_S vs N (does it drift down as
      more directions clear the rank threshold?).

Everything uses the same fits, g_fixed, and conventions as run_noncontraction
part B (rng seed 7 for g, seeds 0-2, LAM0=0.03, C_WIDTH=1.0).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from run_noncontraction import fit_toy, loglog_slope, LAM0, N_GRID, SEEDS, C_WIDTH


def spectral_split(H, lam2, g):
    """Exact split of Q = ||g||^2_{H^-1} into signal and null parts."""
    ev, evec = np.linalg.eigh(H)
    null_mask = (ev - lam2) <= 1e-9 * max(ev[-1], 1.0)
    proj = evec.T @ g
    Q_null = float(np.sum(proj[null_mask] ** 2 / ev[null_mask]))
    Q_sig = float(np.sum(proj[~null_mask] ** 2 / ev[~null_mask]))
    beta_g2 = float(np.sum(proj[null_mask] ** 2))
    dim_null = int(null_mask.sum())
    # smallest retained eigenvalue of H|_S (minus lam2 = design eigenvalue)
    sig_ev = ev[~null_mask]
    lam_min_sig = float(sig_ev.min()) if sig_ev.size else np.nan
    return Q_sig, Q_null, beta_g2, dim_null, lam_min_sig


def main():
    rng0 = np.random.default_rng(7)
    g_fixed = None
    print("=" * 78)
    print("DECOMPOSITION TEST  signal convention, toy POMDP, same fits as part B")
    print("=" * 78)
    for kappa in (0.25, 0.5, 1.0):
        rows = []
        for N in N_GRID:
            for sd in SEEDS:
                _, blk, lam2, N2 = fit_toy(N, kappa, sd)
                if g_fixed is None:
                    g_fixed = rng0.standard_normal(blk.b_hat.size)
                    g_fixed /= np.linalg.norm(g_fixed)
                Q = blk.h_inv_norm(g_fixed) ** 2
                Q_sig, Q_null, beta_g2, dnull, lam_min_sig = spectral_split(
                    blk.H, lam2, g_fixed)
                xi = blk.width_rule(C_WIDTH)
                rows.append(dict(N=N, sd=sd, N2=N2, lam2=lam2, xi=xi, Q=Q,
                                 Q_sig=Q_sig, Q_null=Q_null, beta_g2=beta_g2,
                                 dnull=dnull, sigma2=blk.sigma2,
                                 lam_min_sig=lam_min_sig,
                                 ident_err=abs(Q - Q_sig - Q_null) / Q))
        R = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
        byN = lambda v: np.array([np.mean(v[R["N"] == N]) for N in N_GRID])

        W = np.sqrt(byN(R["xi"])) * np.sqrt(byN(R["Q"]))
        # NB: part B averages W over seeds, then slopes; replicate that:
        Wb = []
        for N in N_GRID:
            m = R["N"] == N
            Wb.append(np.mean(np.sqrt(R["xi"][m] * R["Q"][m])))
        s_meas = loglog_slope(N_GRID, Wb)

        # constant-coefficient fit  Q = A + B/lam  (least squares over all fits)
        X = np.stack([np.ones_like(R["lam2"]), 1.0 / R["lam2"]], axis=1)
        coef, res, *_ = np.linalg.lstsq(X, R["Q"], rcond=None)
        A_fit, B_fit = float(coef[0]), float(coef[1])
        Q_model = X @ coef
        rel_resid = float(np.sqrt(np.mean((Q_model - R["Q"]) ** 2))
                          / np.mean(R["Q"]))

        # slope decomposition
        s_xi = loglog_slope(N_GRID, byN(R["xi"]))
        s_Q = loglog_slope(N_GRID, byN(R["Q"]))
        s_sig2 = loglog_slope(N_GRID, byN(R["sigma2"]))
        s_A = loglog_slope(N_GRID, byN(R["Q_sig"]))
        s_bg2 = loglog_slope(N_GRID, byN(R["beta_g2"]))
        s_lms = loglog_slope(N_GRID, byN(R["lam_min_sig"]))
        # frozen-coefficient model: Q_f(N) = A_bar + B_bar/lam(N) with A_bar,
        # B_bar the mean measured values -- the doc's story with NO drift
        A_bar, B_bar = float(np.mean(R["Q_sig"])), float(np.mean(R["beta_g2"]))
        lam_grid = byN(R["lam2"])
        Q_frozen = A_bar + B_bar / lam_grid
        W_frozen = np.sqrt(byN(R["xi"]) * Q_frozen)
        s_frozen = loglog_slope(N_GRID, W_frozen)
        # pure-asymptote slope for reference
        e = kappa - 1.0

        print(f"\nkappa = {kappa}")
        print(f"  identity |Q - Qsig - Qnull|/Q worst: {R['ident_err'].max():.2e}")
        print(f"  null dim per fit: {sorted(set(R['dnull'].astype(int)))}")
        print(f"  {'N':>9}{'Q_sig(A_N)':>12}{'beta_g^2':>12}{'sigma2':>11}"
              f"{'lam_min_S':>11}{'W':>10}")
        for i, N in enumerate(N_GRID):
            print(f"  {N:>9}{byN(R['Q_sig'])[i]:>12.4f}{byN(R['beta_g2'])[i]:>12.6f}"
                  f"{byN(R['sigma2'])[i]:>11.4e}{byN(R['lam_min_sig'])[i]:>11.4e}"
                  f"{Wb[i]:>10.4f}")
        print(f"  ls-fit Q = A + B/lam:  A = {A_fit:.4f}  B = {B_fit:.6f}  "
              f"rel rms resid = {rel_resid:.3e}")
        print(f"     A vs measured A_N mean {A_bar:.4f}   "
              f"B vs measured beta_g^2 mean {B_bar:.6f}")
        print(f"  slopes:  W meas {s_meas:+.4f} | asymptote e/2 {e/2:+.4f} | "
              f"frozen-coef model {s_frozen:+.4f}")
        print(f"  drifts:  A_N {s_A:+.4f}  beta_g^2 {s_bg2:+.4f}  "
              f"sigma2 {s_sig2:+.4f}  lam_min(H|S) {s_lms:+.4f}")
        print(f"  residual accounting: meas - asymptote = {s_meas-e/2:+.4f};"
              f"  frozen-A contamination alone = {s_frozen-e/2:+.4f};"
              f"  sigma2-drift contribution = {-0.5*s_sig2:+.4f};"
              f"  beta_g^2-drift contribution = {0.5*s_bg2:+.4f} (approx)")


if __name__ == "__main__":
    main()
