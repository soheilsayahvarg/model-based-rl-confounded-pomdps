"""run_pd_kernel2.py -- the corrected width law, tested out of sample.

Predictions: docs/pd_kernel_regime.md section 7 (P-K6..P-K9), committed at
44e0cb5 before this file existed.

WHY A SECOND FILE. run_pd_kernel.py's results are committed and referenced. The
corrections it produced are stated post-hoc there, so they are not evidence.
This driver tests them on a grid that shares no (a, b) pair with the first.

WHAT CHANGED, AND WHY EACH CHANGE EXISTS

  the truncation guard      j* = lambda^(-1/b) is where the ridge meets the
                            spectrum. In the first run j* reached 2.4e4 against
                            a truncation at 2e4, so three cells measured a
                            spectrum that ended before the transition. This is
                            the fifth instance of that defect in this project.
                            J_MAX is now sized per cell as 20*j*(lambda_min) and
                            a cell whose requirement exceeds the cap is REJECTED
                            rather than measured.

  the clip                  for theta <= 0 the sum sum_j g_j^2/s_j converges,
                            ||g||_{H^-1} tends to a constant, and the width is
                            governed by sqrt(xi) alone. The exponent saturates at
                            (tau-1)/2, so the law needs clip(theta, 0, 1).
                            sec:not-new of the paper states it without the clip.

  a Laplacian kernel        the first run fitted a power law to a GAUSSIAN
                            spectrum, whose eigenvalues decay exponentially. That
                            was our error, not the theory's. exp(-|x-y|/bw) in 1D
                            is Sobolev with polynomial decay, which is the class
                            the theory addresses.

Writes experiments/results_pd_kernel2.json.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ALPHA, C2 = 10.0, 1.0
N_GRID = np.array([2.0 ** k for k in range(11, 25)])
LAM0 = 1.0
XI_C = 1.0
J_CAP = 8_000_000

# fresh grid: shares no (a, b) pair with run_pd_kernel.py
A_GRID = [1.2, 1.8, 2.5, 3.5, 5.0]
B_GRID = [1.2, 2.5, 4.0, 8.0]
B_STRESS = 1.0          # not part of P-K6; exists to give the guard something
RESULTS = {}

TAU = (ALPHA + 2.0) / (2.0 * ALPHA + 2.0)
KAP = ALPHA / (ALPHA * C2 + 1.0)


def clipped_slope(a, b):
    """The corrected law: theta is clipped to [0, 1] before entering e'."""
    th = 1.0 - (a - 1.0) / b
    return (TAU + min(max(th, 0.0), 1.0) * KAP - 1.0) / 2.0, th


def j_star(b, N=None):
    lam = LAM0 * (N_GRID[-1] if N is None else N) ** (-KAP)
    return lam ** (-1.0 / b)


def measure(a, b, j_max, N=N_GRID):
    """Exact W(N) on s_j = j^-b, g_j^2 = j^-a, truncated at j_max."""
    j = np.arange(1, int(j_max) + 1, dtype=np.float64)
    s = j ** (-b)
    g2 = j ** (-a)
    W = np.empty(len(N))
    for i, n in enumerate(N):
        lam = LAM0 * n ** (-KAP)
        W[i] = np.sqrt(XI_C * n ** (TAU - 1.0)) * np.sqrt((g2 / (s + lam)).sum())
    return float(np.polyfit(np.log(N), np.log(W), 1)[0])


def part6():
    print("\nPART 6  corrected law on a FRESH grid   tau=%.4f kappa=%.4f"
          % (TAU, KAP))
    print("%6s%6s%9s%11s%12s%12s%9s%10s" %
          ("a", "b", "theta", "clipped", "pred", "meas", "err", "guard"))
    rows, acc_bad, rej = [], 0, []
    for b in B_GRID + [B_STRESS]:
        need = 20.0 * j_star(b)
        ok = need <= J_CAP
        for a in A_GRID:
            pred, th = clipped_slope(a, b)
            jm = min(max(20000.0, need), J_CAP)
            meas = measure(a, b, jm)
            err = abs(meas - pred)
            tag = ("ok" if ok else "REJECT") + ("" if b != B_STRESS
                                                else "/stress")
            if ok and b != B_STRESS:
                acc_bad += int(err > 0.02)
            if not ok:
                rej.append((a, b, pred, meas, err))
            print("%6.1f%6.1f%9.4f%11.4f%12.4f%12.4f%9.4f%10s" %
                  (a, b, th, min(max(th, 0.0), 1.0), pred, meas, err, tag))
            rows.append(dict(a=a, b=b, theta=th, pred=pred, meas=meas, err=err,
                             j_star=float(j_star(b)), j_max=float(jm),
                             guard_ok=bool(ok), stress=bool(b == B_STRESS)))
    n_acc = len([r for r in rows if r["guard_ok"] and not r["stress"]])
    print("\n   P-K6: accepted cells failing the 0.02 tolerance: %d of %d"
          % (acc_bad, n_acc))
    print("   P-K7: cells the guard rejected: %d" % len(rej))
    for a, b, pred, meas, err in rej:
        print("      a=%.1f b=%.1f  pred %+.4f  meas %+.4f  err %.4f  %s"
              % (a, b, pred, meas, err,
                 "guard was RIGHT" if err > 0.02 else "guard over-conservative"))
    RESULTS["part6"] = rows
    return acc_bad, n_acc, rej


def part7():
    """theta = 0 exactly (a = b + 1): is the residual a log factor?"""
    print("\nPART 7  the theta = 0 boundary, on a doubled N grid")
    print("   if the residual is a log correction it shrinks as the grid "
          "extends; a wrong exponent would not")
    base = (TAU - 1.0) / 2.0
    print("%6s%6s%14s%14s%14s%10s" %
          ("a", "b", "pred", "meas short", "meas long", "shrink"))
    rows = []
    long_grid = np.array([2.0 ** k for k in range(11, 39)])
    for b in (2.0, 4.0, 8.0):
        a = b + 1.0
        jm = min(max(20000.0, 20.0 * j_star(b)), J_CAP)
        m_short = measure(a, b, jm)
        jm_l = min(max(20000.0, 20.0 *
                       (LAM0 * long_grid[-1] ** (-KAP)) ** (-1.0 / b)), J_CAP)
        m_long = measure(a, b, jm_l, N=long_grid)
        r_s, r_l = abs(m_short - base), abs(m_long - base)
        print("%6.1f%6.1f%14.4f%14.4f%14.4f%10s" %
              (a, b, base, m_short, m_long,
               "yes" if r_l < r_s else "NO"))
        rows.append(dict(a=a, b=b, pred=base, meas_short=m_short,
                         meas_long=m_long, res_short=r_s, res_long=r_l,
                         shrinks=bool(r_l < r_s)))
    print("   P-K8: residual shrinks in %d of %d"
          % (sum(r["shrinks"] for r in rows), len(rows)))
    RESULTS["part7"] = rows
    return rows


def part8():
    """A kernel with genuinely polynomial eigenvalue decay."""
    print("\nPART 8  Laplacian kernel exp(-|x-y|/bw) in 1D (Sobolev, "
          "polynomial decay)")
    rng = np.random.default_rng(0)
    n = 3000
    x = np.sort(rng.uniform(0.0, 1.0, n))
    D = np.abs(x[:, None] - x[None, :])
    print("%10s%10s%9s%11s%12s%12s%9s" %
          ("bandwidth", "fit b", "theta", "clipped", "pred", "meas", "err"))
    rows = []
    a = 2.0
    for bw in (0.05, 0.1, 0.25, 0.5):
        K = np.exp(-D / bw)
        s = np.linalg.eigvalsh(K)[::-1]
        s = np.maximum(s, 0.0)
        s = s / s[0]
        s = s[s > 1e-13]
        j = np.arange(1, len(s) + 1, dtype=float)
        # fit the decay on the middle decades, away from the top few modes and
        # the numerical floor
        lo, hi = max(3, len(s) // 20), len(s) // 2
        fit = np.polyfit(np.log(j[lo:hi]), np.log(s[lo:hi]), 1)
        b_fit = -float(fit[0])
        g2 = j ** (-a)
        W = np.empty(len(N_GRID))
        for i, nn in enumerate(N_GRID):
            lam = LAM0 * nn ** (-KAP)
            W[i] = np.sqrt(XI_C * nn ** (TAU - 1.0)) * \
                np.sqrt((g2 / (s + lam)).sum())
        meas = float(np.polyfit(np.log(N_GRID), np.log(W), 1)[0])
        pred, th = clipped_slope(a, b_fit)
        print("%10.2f%10.4f%9.4f%11.4f%12.4f%12.4f%9.4f" %
              (bw, b_fit, th, min(max(th, 0.0), 1.0), pred, meas,
               abs(pred - meas)))
        rows.append(dict(bandwidth=bw, b_fit=b_fit, theta=th, pred=pred,
                         meas=meas, err=abs(pred - meas), n_modes=int(len(s)),
                         j_star=float(j_star(b_fit))))
    worst = max(r["err"] for r in rows)
    print("   P-K9 tol 0.05; worst |error| = %.4f  (Gaussian gave 0.1064)"
          % worst)
    print("   note: j* against the mode count is the same guard as Part 6 --")
    for r in rows:
        print("      bw=%.2f  j* = %.1f  modes = %d  %s"
              % (r["bandwidth"], r["j_star"], r["n_modes"],
                 "ok" if r["j_star"] < r["n_modes"] / 10 else "TRUNCATED"))
    RESULTS["part8"] = rows
    return worst, rows


def main():
    acc_bad, n_acc, rej = part6()
    part7()
    worst, _ = part8()
    print("\n%s\nSUMMARY\n%s" % ("=" * 76, "=" * 76))
    print("P-K6 clipped law, fresh grid   %d of %d accepted cells fail"
          % (acc_bad, n_acc))
    print("P-K7 guard rejected            %d cells" % len(rej))
    print("P-K9 Laplacian kernel          worst err %.4f (tol 0.05)" % worst)
    RESULTS.update(tau=TAU, kappa=KAP, alpha=ALPHA, c2=C2, j_cap=J_CAP,
                   a_grid=A_GRID, b_grid=B_GRID, b_stress=B_STRESS,
                   N_grid=N_GRID.tolist())
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_pd_kernel2.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
