"""run_pd_kernel3.py -- a truncation guard built on the quantity, not a proxy.

Follows run_pd_kernel2.py, whose results are committed at ac68996. That run's
guard failed in both directions: over-conservative on 4 of 5 rejections and
wrong on the one cell it accepted and lost.

WHY THE OLD GUARD FAILED. It rejected when j* = lambda^(-1/b) exceeded a tenth
of the mode budget. j* is where the ridge meets the spectrum. It is NOT where the
truncation error lives, and the two come apart in opposite directions:

    theta <= 0    the sum is head-dominated, the tail is irrelevant however far
                  j* sits, so rejecting was pointless
    a near 1      sum_{j>J} j^-a converges so slowly that J = 20 j* is not
                  enough, so accepting was wrong

THE REPLACEMENT. Bound the discarded mass directly. For j > J the spectrum obeys
s_j <= J^-b, so whenever J^-b <= lambda,

    tail = sum_{j>J} j^-a / (s_j + lambda)  <=  J^(1-a) / ((a-1) * lambda)

an exact upper bound computable in O(1). A cell is admissible when that bound is
below 1% of the sum actually computed, at EVERY N on the grid. No case analysis,
no proxy, and it covers both failure modes by construction: a head-dominated sum
makes the ratio small, and a slowly converging tail makes it large.

This is the structural fix claims.md says the range-cannot-straddle-the-
transition mode needs. The previous three attempts patched individual cells.

Part 3 also separates two questions the earlier Laplacian test confounded:
whether the kernel is in the polynomial-decay class (empirical, and n = 3000 is
plenty for that) and whether the law holds on such a spectrum (analytic, needing
J far beyond any Gram matrix we can diagonalize).

Writes experiments/results_pd_kernel3.json.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ALPHA, C2 = 10.0, 1.0
TAU = (ALPHA + 2.0) / (2.0 * ALPHA + 2.0)
KAP = ALPHA / (ALPHA * C2 + 1.0)
N_GRID = np.array([2.0 ** k for k in range(11, 25)])
LAM0, XI_C = 1.0, 1.0
J_CAP = 6_000_000
TAIL_TOL = 0.01
A_GRID = [1.2, 1.8, 2.5, 3.5, 5.0]
B_GRID = [1.0, 1.2, 2.5, 4.0, 8.0]
RESULTS = {}


def clipped(a, b):
    th = 1.0 - (a - 1.0) / b
    return (TAU + min(max(th, 0.0), 1.0) * KAP - 1.0) / 2.0, th


def sums_and_tail(a, b, j_max, N):
    """Exact W(N) plus the exact upper bound on what truncation discarded."""
    j = np.arange(1, int(j_max) + 1, dtype=np.float64)
    s = j ** (-b)
    g2 = j ** (-a)
    W = np.empty(len(N))
    worst = 0.0
    for i, n in enumerate(N):
        lam = LAM0 * n ** (-KAP)
        tot = float((g2 / (s + lam)).sum())
        W[i] = np.sqrt(XI_C * n ** (TAU - 1.0)) * np.sqrt(tot)
        if a > 1.0 and j_max ** (-b) <= lam:
            bound = j_max ** (1.0 - a) / ((a - 1.0) * lam)
        else:
            bound = np.inf              # bound not valid -> treat as failing
        worst = max(worst, bound / max(tot, 1e-300))
    return float(np.polyfit(np.log(N), np.log(W), 1)[0]), worst


def admissible(a, b, j_max=J_CAP, N=N_GRID):
    slope, ratio = sums_and_tail(a, b, j_max, N)
    return slope, ratio, ratio <= TAIL_TOL


def part1():
    print("\nPART 1  the tail-bound guard on the full (a, b) grid")
    print("   admissible when  max_N [ J^(1-a) / ((a-1) lambda) ] / sum  <= %g"
          % TAIL_TOL)
    print("%6s%6s%9s%12s%12s%9s%12s%9s" %
          ("a", "b", "theta", "pred", "meas", "err", "tail/sum", "verdict"))
    rows, bad_acc, bad_rej = [], [], []
    for b in B_GRID:
        for a in A_GRID:
            pred, th = clipped(a, b)
            meas, ratio, ok = admissible(a, b)
            err = abs(meas - pred)
            v = "ok" if ok else "REJECT"
            if ok and err > 0.02:
                bad_acc.append((a, b, th, pred, meas, err))
            if not ok and err <= 0.02:
                bad_rej.append((a, b, th, pred, meas, err))
            print("%6.1f%6.1f%9.4f%12.4f%12.4f%9.4f%12.2e%9s" %
                  (a, b, th, pred, meas, err, ratio, v))
            rows.append(dict(a=a, b=b, theta=th, pred=pred, meas=meas,
                             err=err, tail_ratio=float(ratio),
                             admissible=bool(ok)))
    n_ok = sum(r["admissible"] for r in rows)
    print("\n   admissible cells: %d of %d" % (n_ok, len(rows)))
    print("   admissible cells failing tol 0.02: %d" % len(bad_acc))
    for a, b, th, pred, meas, err in bad_acc:
        print("      a=%.1f b=%.1f theta=%+.4f pred %+.4f meas %+.4f err %.4f"
              % (a, b, th, pred, meas, err))
    print("   rejected cells that would have PASSED (over-conservative): %d"
          % len(bad_rej))
    for a, b, th, pred, meas, err in bad_rej:
        print("      a=%.1f b=%.1f theta=%+.4f err %.4f" % (a, b, th, err))
    RESULTS["part1"] = rows
    return bad_acc, bad_rej, n_ok, len(rows)


def part2():
    """Does raising J rescue a rejected cell? If not, the cell is hopeless."""
    print("\nPART 2  rejected cells, re-measured as J grows")
    print("%6s%6s%12s%14s%14s%14s" %
          ("a", "b", "pred", "J=1e5", "J=1e6", "J=6e6"))
    rows = []
    for b in B_GRID:
        for a in A_GRID:
            _, _, ok = admissible(a, b)
            if ok:
                continue
            pred, th = clipped(a, b)
            ms = [admissible(a, b, j_max=j)[0] for j in (1e5, 1e6, 6e6)]
            print("%6.1f%6.1f%12.4f%14.4f%14.4f%14.4f" %
                  (a, b, pred, ms[0], ms[1], ms[2]))
            rows.append(dict(a=a, b=b, theta=th, pred=pred,
                             meas=[float(m) for m in ms],
                             converging=bool(abs(ms[2] - pred)
                                             < abs(ms[0] - pred))))
    print("   moving toward the prediction as J grows: %d of %d"
          % (sum(r["converging"] for r in rows), len(rows)))
    RESULTS["part2"] = rows
    return rows


def part3():
    """Split the Laplacian question in two."""
    print("\nPART 3a  is the Laplacian kernel in the polynomial-decay class?")
    rng = np.random.default_rng(0)
    n = 3000
    x = np.sort(rng.uniform(0.0, 1.0, n))
    D = np.abs(x[:, None] - x[None, :])
    fits = []
    print("%12s%10s%12s" % ("bandwidth", "fit b", "modes kept"))
    for bw in (0.05, 0.1, 0.25, 0.5):
        K = np.exp(-D / bw)
        s = np.linalg.eigvalsh(K)[::-1]
        s = np.maximum(s, 0.0)
        s = s / s[0]
        s = s[s > 1e-13]
        j = np.arange(1, len(s) + 1, dtype=float)
        lo, hi = max(3, len(s) // 20), len(s) // 2
        b_fit = -float(np.polyfit(np.log(j[lo:hi]), np.log(s[lo:hi]), 1)[0])
        fits.append(b_fit)
        print("%12.2f%10.4f%12d" % (bw, b_fit, len(s)))
    spread = max(fits) - min(fits)
    print("   fitted b over a 10x bandwidth range: spread %.4f  "
          "(Sobolev/Matern-1/2 predicts exactly 2)" % spread)

    print("\nPART 3b  does the law hold on that spectrum, with J set by the "
          "guard?")
    print("%8s%9s%12s%12s%9s%12s%9s" %
          ("a", "theta", "pred", "meas", "err", "tail/sum", "verdict"))
    b = float(np.mean(fits))
    rows = []
    for a in (1.5, 2.0, 2.5, 3.0):
        pred, th = clipped(a, b)
        meas, ratio, ok = admissible(a, b)
        print("%8.1f%9.4f%12.4f%12.4f%9.4f%12.2e%9s" %
              (a, th, pred, meas, abs(meas - pred), ratio,
               "ok" if ok else "REJECT"))
        rows.append(dict(a=a, b_fit=b, theta=th, pred=pred, meas=meas,
                         err=abs(meas - pred), tail_ratio=float(ratio),
                         admissible=bool(ok)))
    adm = [r for r in rows if r["admissible"]]
    worst = max((r["err"] for r in adm), default=float("nan"))
    print("   P-K9 on admissible cells: %d of %d, worst err %.4f (tol 0.05); "
          "Gaussian gave 0.1064" % (len(adm), len(rows), worst))
    RESULTS["part3"] = dict(b_fits=fits, b_spread=spread, rows=rows)
    return spread, worst, len(adm), len(rows)


def main():
    bad_acc, bad_rej, n_ok, n_tot = part1()
    part2()
    spread, worst, n_adm, n_rows = part3()
    print("\n%s\nSUMMARY\n%s" % ("=" * 76, "=" * 76))
    print("guard admitted            %d of %d cells" % (n_ok, n_tot))
    print("admitted and wrong        %d   (old guard: 1)" % len(bad_acc))
    print("rejected but fine         %d   (old guard: 4 of 5)" % len(bad_rej))
    print("Laplacian b spread        %.4f over a 10x bandwidth range" % spread)
    print("P-K9 worst err            %.4f on %d admissible cells (tol 0.05)"
          % (worst, n_adm))
    RESULTS.update(tau=TAU, kappa=KAP, tail_tol=TAIL_TOL, j_cap=J_CAP,
                   a_grid=A_GRID, b_grid=B_GRID, N_grid=N_GRID.tolist())
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_pd_kernel3.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
