"""run_pd_kernel.py -- what the floor costs when the spectrum has no atom.

Predictions: docs/pd_kernel_regime.md, committed before this file existed
(commit 877927b).

WHY. sec:scope of the paper concedes that as:null is exact only for a tabular
delta-kernel and that "we have not characterized" the strictly positive-definite
case. That is the first thing a referee will stop on, because the proximal
literature works in RKHS and the tabular exact null is the cheapest place to get
the result.

WHAT IS BEING MEASURED. Four things, in order of what they cost us:

  Part 1  the closed-form threshold theta* = (alpha*c2 + 1)/(2*alpha + 2),
          against a direct evaluation of sign(tau + theta*kappa - 1)
  Part 2  the width exponent on a synthetic spectrum s_j = j^-b with gradient
          mass g_j^2 = j^-a, against e'/2 with theta = 1 - (a-1)/b
  Part 3  the same with an ATOM inserted at zero, which should snap back to e/2
          and stop caring about (a, b)
  Part 4  whether prop:floor survives. On the atom spectrum the floor
          beta*beta_g is constant in N. On a PD spectrum the analogous quantity
          is the mass below the ridge scale, and it has no reason to be constant.

Part 4 is the one that costs something. prop:floor is schedule-free because
coverage forces xi >= lambda*beta^2 while the support norm gives
||g||_{H^-1} >= beta_g/sqrt(lambda), and the two lambda factors cancel. That
cancellation is bought by the atom. Without one, both sides move with lambda.

Everything here is exact linear algebra on a diagonal spectrum, plus one real
Gaussian-kernel Gram matrix in Part 5. No sampling.

Writes experiments/results_pd_kernel.json.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
C2_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
N_GRID = np.array([2 ** k for k in range(11, 25)], dtype=float)   # 2e3 .. 1.7e7
J_MAX = 20000
LAM0 = 1.0
XI_C = 1.0
RESULTS = {}


def tau_kappa(alpha, c2):
    return (alpha + 2.0) / (2.0 * alpha + 2.0), alpha / (alpha * c2 + 1.0)


def theta_star(alpha, c2):
    """Closed form from docs/pd_kernel_regime.md section 3."""
    return (alpha * c2 + 1.0) / (2.0 * alpha + 2.0)


def spectrum(a, b, atom=False, j_max=J_MAX):
    """s_j = j^-b and g_j^2 = j^-a, optionally with an exact zero eigenvalue.

    The atom carries the same gradient mass as the j = 1 mode, so the two
    spectra are comparable: the only structural difference is whether one
    direction has s = 0 exactly.
    """
    j = np.arange(1, j_max + 1, dtype=float)
    s = j ** (-b)
    g2 = j ** (-a)
    if atom:
        s = np.concatenate([[0.0], s])
        g2 = np.concatenate([[g2[0]], g2])
    return s, g2


def width(s, g2, xi, lam):
    """W = sqrt(xi) * ||g||_{H^-1} with H = diag(s) + lam I, exactly."""
    return float(np.sqrt(xi) * np.sqrt((g2 / (s + lam)).sum()))


def slope(N, W):
    return float(np.polyfit(np.log(N), np.log(W), 1)[0])


def eff_floor(s, g2, b_true2, lam):
    """The effective-null analogue of beta*beta_g at ridge scale lam.

    On the atom spectrum this is exactly the beta*beta_g of prop:floor. On a PD
    spectrum it is the mass in directions the ridge cannot resolve, which is what
    the proof's null projector degrades to.
    """
    m = s < lam
    if not m.any():
        return 0.0
    return float(np.sqrt(b_true2[m].sum()) * np.sqrt(g2[m].sum()))


def part1():
    print("\nPART 1  threshold formula vs direct sign evaluation")
    print("%8s%8s%12s%12s%14s%10s" %
          ("alpha", "c2", "theta*", "closed", "direct xover", "match"))
    rows, bad = [], 0
    for alpha in ALPHA_GRID:
        for c2 in C2_GRID:
            tau, kap = tau_kappa(alpha, c2)
            ts = theta_star(alpha, c2)
            # direct: smallest theta on a fine grid with tau + theta*kap - 1 >= 0
            th = np.linspace(0.0, 1.0, 200001)
            ok = tau + th * kap - 1.0 >= 0.0
            direct = float(th[ok][0]) if ok.any() else float("nan")
            match = abs(direct - ts) < 1e-4
            bad += int(not match)
            rows.append(dict(alpha=alpha, c2=c2, theta_star=ts,
                             direct=direct, match=bool(match),
                             tau=tau, kappa=kap))
            print("%8.2f%8.2f%12.6f%12.6f%14.6f%10s" %
                  (alpha, c2, ts, ts, direct, "yes" if match else "NO"))
    print("   mismatches: %d of %d" % (bad, len(rows)))
    half = [r for r in rows if r["c2"] == 1.0]
    print("   at c2 = 1.0, theta* = %s  (predicted 0.5 for every alpha)" %
          sorted({round(r["theta_star"], 12) for r in half}))
    RESULTS["part1"] = rows
    return bad


def part2_3():
    print("\nPART 2/3  measured width exponent vs e'/2")
    print("   alpha = 10, c2 = 1  ->  tau = %.4f, kappa = %.4f"
          % tau_kappa(10.0, 1.0))
    alpha, c2 = 10.0, 1.0
    tau, kap = tau_kappa(alpha, c2)
    e_exact = tau + kap - 1.0
    print("%6s%6s%9s%12s%12s%9s%12s%12s%9s" %
          ("a", "b", "theta", "pred e'/2", "meas PD", "err",
           "pred e/2", "meas atom", "err"))
    rows, worst_pd, worst_atom = [], 0.0, 0.0
    for a in (1.5, 2.0, 3.0, 4.0):
        for b in (1.5, 2.0, 3.0, 6.0):
            th = 1.0 - (a - 1.0) / b
            e_pd = tau + th * kap - 1.0
            for atom in (False, True):
                s, g2 = spectrum(a, b, atom=atom)
                W = np.array([width(s, g2, XI_C * n ** (tau - 1.0),
                                    LAM0 * n ** (-kap)) for n in N_GRID])
                m = slope(N_GRID, W)
                if atom:
                    meas_atom, err_atom = m, abs(m - e_exact / 2.0)
                else:
                    meas_pd, err_pd = m, abs(m - e_pd / 2.0)
            worst_pd = max(worst_pd, err_pd)
            worst_atom = max(worst_atom, err_atom)
            print("%6.1f%6.1f%9.4f%12.4f%12.4f%9.4f%12.4f%12.4f%9.4f" %
                  (a, b, th, e_pd / 2.0, meas_pd, err_pd,
                   e_exact / 2.0, meas_atom, err_atom))
            rows.append(dict(a=a, b=b, theta=th, pred_pd=e_pd / 2.0,
                             meas_pd=meas_pd, err_pd=err_pd,
                             pred_atom=e_exact / 2.0, meas_atom=meas_atom,
                             err_atom=err_atom))
    print("   worst |error|  PD %.4f (P-K2 tol 0.02)   atom %.4f (P-K3)"
          % (worst_pd, worst_atom))
    spread = max(r["meas_atom"] for r in rows) - min(r["meas_atom"]
                                                     for r in rows)
    print("   P-K3 insensitivity: atom slope spread over the (a,b) grid = %.5f"
          % spread)
    RESULTS["part2_3"] = rows
    return worst_pd, worst_atom, spread


def part4():
    print("\nPART 4  does prop:floor survive without an atom?")
    print("   the floor's analogue at ridge scale lambda, across N")
    alpha, c2 = 10.0, 1.0
    tau, kap = tau_kappa(alpha, c2)
    rows = []
    for a, b in ((2.0, 3.0), (3.0, 2.0)):
        for atom in (True, False):
            s, g2 = spectrum(a, b, atom=atom)
            # a true bridge with source-condition mass, same on both spectra
            b_true2 = np.arange(1, len(s) + 1, dtype=float) ** (-a)
            fl = np.array([eff_floor(s, g2, b_true2, LAM0 * n ** (-kap))
                           for n in N_GRID])
            sl = slope(N_GRID, np.maximum(fl, 1e-300))
            print("   a=%.1f b=%.1f %-6s floor %.4e -> %.4e   log-log slope "
                  "%+.4f" % (a, b, "atom" if atom else "PD",
                             fl[0], fl[-1], sl))
            rows.append(dict(a=a, b=b, atom=atom, first=float(fl[0]),
                             last=float(fl[-1]), slope=sl))
    print("   P-K4: atom slopes should be ~0 (a genuine floor); PD slopes < 0")
    RESULTS["part4"] = rows
    return rows


def part5():
    """A real PD Gram matrix rather than a hand-built spectrum."""
    print("\nPART 5  Gaussian kernel over the observation space")
    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 3))
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    alpha, c2 = 10.0, 1.0
    tau, kap = tau_kappa(alpha, c2)
    rows = []
    print("%10s%10s%10s%12s%12s%9s" %
          ("bandwidth", "fit b", "fit a", "theta", "pred e'/2", "meas"))
    for bw in (0.5, 1.0, 2.0, 4.0):
        K = np.exp(-D / (2.0 * bw ** 2))
        s = np.linalg.eigvalsh(K)[::-1]
        s = np.maximum(s, 0.0) / s[0]
        keep = s > 1e-14
        s = s[keep]
        j = np.arange(1, len(s) + 1, dtype=float)
        # gradient mass drawn once and shared, so only the spectrum varies
        g2 = j ** (-2.0)
        fit = np.polyfit(np.log(j[1:]), np.log(np.maximum(s[1:], 1e-300)), 1)
        b_fit = -float(fit[0])
        a_fit = 2.0
        th = min(1.0, 1.0 - (a_fit - 1.0) / max(b_fit, 1e-9))
        pred = (tau + th * kap - 1.0) / 2.0
        W = np.array([width(s, g2, XI_C * nn ** (tau - 1.0),
                            LAM0 * nn ** (-kap)) for nn in N_GRID])
        meas = slope(N_GRID, W)
        print("%10.2f%10.4f%10.2f%12.4f%12.4f%9.4f" %
              (bw, b_fit, a_fit, th, pred, meas))
        rows.append(dict(bandwidth=bw, b_fit=b_fit, a=a_fit, theta=th,
                         pred=pred, meas=meas, err=abs(pred - meas),
                         n_kept=int(len(s))))
    print("   P-K5 tol 0.05; worst |error| = %.4f"
          % max(r["err"] for r in rows))
    RESULTS["part5"] = rows
    return rows


def main():
    bad = part1()
    wpd, watom, spread = part2_3()
    part4()
    part5()
    print("\n%s\nSUMMARY\n%s" % ("=" * 76, "=" * 76))
    print("P-K1 threshold formula      mismatches %d" % bad)
    print("P-K2 PD slope == e'/2       worst err  %.4f  (tol 0.02)" % wpd)
    print("P-K3 atom slope == e/2      worst err  %.4f, spread %.5f" %
          (watom, spread))
    print("P-K4 see Part 4 slopes")
    print("P-K5 see Part 5")
    RESULTS.update(N_grid=N_GRID.tolist(), j_max=J_MAX, lam0=LAM0, xi_c=XI_C)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_pd_kernel.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
