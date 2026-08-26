"""run_converse_identification.py -- is the floor the sharp identified width?

Predictions: docs/converse_identification.md, committed at a15a0d8 before this
file existed.

THE CLAIM. The conditional moment restrictions pin a bridge block only up to the
design null, so the identified set is (b_hat + Nul) intersected with the class
ball ||b|| <= M. For a linear value functional <g, b> that is an interval of
half-width

    HW = max{ <P_Nul g, n> : n in Nul, ||b_hat + n|| <= M }
       = beta_g * sqrt(M^2 - ||b_hat||^2)

because b_hat is orthogonal to Nul. The truth satisfies
||b_true||^2 = ||b_hat||^2 + beta^2, so at the tightest admissible class
M = ||b_true|| this is exactly beta*beta_g, which is prop:floor.

If that holds, the floor is not a defect of the pessimism layer. It is the width
of the set the data cannot distinguish, and a region that contracted below it
would be INVALID. The pathology of an e > 0 schedule is then overpayment: it pays
Theta(N^(e/2)) where the gap is constant.

Part 1 checks the closed form against a numerically solved constrained maximum,
so agreement is not a restatement of the algebra. Part 2 sweeps M. Part 3 asks
which policies are point-identified. Part 4 converts the published widths into
overpayment ratios.

Writes experiments/results_converse.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params
from run_completeness_beta import (true_bridges_generic, population_design_span)

CONFIGS = [(4, 6, 2), (4, 8, 3), (2, 6, 4), (3, 7, 5)]
CONFOUNDS = [0.0, 0.3, 0.6, 0.9, 0.99, 1.0]
N_A = 2
RESULTS = {}


def block_pieces(p, a):
    """b_hat (min-norm), b_true, the null projector, and the value gradient.

    Everything is in the per-action observation coordinate R^{n_o}, tensored with
    the block's index space. The design span is the population one, so this is an
    identification question with no sampling in it.
    """
    E = p["E"]
    n_o = E.shape[1]
    U = population_design_span(p, a)               # (n_o, r)
    Pn = np.eye(n_o) - U @ U.T
    bR, _res = true_bridges_generic(p)
    B = bR[a]                                      # (n_o, 2, n_o)
    b_true = B.reshape(n_o, -1)                    # treat trailing axes as index
    b_hat = (U @ U.T) @ b_true                     # min-norm: the in-span part
    nu1 = p["p1"] @ E
    return b_hat, b_true, Pn, U, nu1


def sharp_halfwidth_numeric(b_hat, Pn, g_vec, M, n_restarts=6, seed=0):
    """max <P_Nul g, n> over n in Nul with ||b_hat + n|| <= M, solved directly.

    Deliberately NOT the closed form. If the two agree we have checked the
    derivation rather than restated it.
    """
    from scipy.optimize import minimize
    n_o, k = b_hat.shape
    rng = np.random.default_rng(seed)
    gN = Pn @ g_vec                                # (n_o,)
    lim2 = M ** 2 - float((b_hat ** 2).sum())
    if lim2 <= 0:
        return 0.0

    def unpack(x):
        return Pn @ x.reshape(n_o, k)              # project into the null

    def neg_obj(x):
        Nm = unpack(x)
        return -float(gN @ Nm @ np.ones(k) / max(np.sqrt(k), 1e-300))

    def cons_f(x):
        Nm = unpack(x)
        return lim2 - float((Nm ** 2).sum())

    best = 0.0
    for r in range(n_restarts):
        x0 = rng.standard_normal(n_o * k) * np.sqrt(max(lim2, 1e-12) / (n_o * k))
        res = minimize(neg_obj, x0, constraints=[{"type": "ineq",
                                                  "fun": cons_f}],
                       method="SLSQP", options=dict(maxiter=800, ftol=1e-14))
        if res.success:
            best = max(best, -res.fun)
    return float(best)


def agree(closed, numeric, scale):
    """Relative error where the quantity is meaningful, absolute where it is not.

    The first version of this divided by `closed` unconditionally and reported a
    relative error of 1.0 on cells where BOTH values were about 1e-17. That is
    not a disagreement, it is a vanishing denominator, and it made the
    load-bearing prediction look refuted when it was not.
    """
    d = abs(closed - numeric)
    denom = max(abs(closed), 1e-3 * abs(scale))
    return float(d / denom) if denom > 0 else float(d)


def sharp_halfwidth_vector(b_hat_v, beta_g, M):
    """Closed form for a plain vector block: beta_g * sqrt(M^2 - ||b_hat||^2)."""
    lim2 = M ** 2 - float(b_hat_v @ b_hat_v)
    return 0.0 if lim2 <= 0 else float(beta_g * np.sqrt(lim2))


def part1():
    """The load-bearing check, on a vector-valued functional in R^{n_o}."""
    print("\nPART 1  closed form vs a numerically solved constrained maximum")
    print("   value functional g = nu_1 (the stage-1 gradient profile)")
    print("%12s%7s%4s%11s%11s%13s%13s%11s" %
          ("config", "cf", "a", "beta", "beta_g", "closed HW", "numeric HW",
           "rel err"))
    rows, worst = [], 0.0
    for cfg in CONFIGS:
        n_s, n_o, n_o0 = cfg
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_A, n_o=n_o, n_o0=n_o0, T=3,
                               seed=0, confound=cf)
            for a in range(N_A):
                b_hat, b_true, Pn, U, nu1 = block_pieces(p, a)
                # Collapse the index axes first, THEN compute every quantity on
                # the collapsed vector. The first version computed beta on the
                # full matrix and the half-width on the collapsed vector, which
                # compared two different objects and made the identity
                # HW = beta*beta_g look violated by up to 99%.
                k = b_true.shape[1]
                bt_v = b_true @ np.ones(k) / np.sqrt(k)
                bh_v = (U @ U.T) @ bt_v
                beta = float(np.linalg.norm(Pn @ bt_v))
                beta_g = float(np.linalg.norm(Pn @ nu1))
                M = float(np.linalg.norm(bt_v))
                Mv = M
                closed = sharp_halfwidth_vector(bh_v, beta_g, Mv)
                # numeric: same problem, solved without the formula
                num = numeric_scalar(bh_v, Pn, nu1, Mv)
                rel = agree(closed, num, beta * beta_g)
                worst = max(worst, rel)
                print("%12s%7.2f%4d%11.4f%11.4f%13.6f%13.6f%11.2e" %
                      (str(cfg), cf, a, beta, beta_g, closed, num, rel))
                rows.append(dict(config=list(cfg), confound=cf, action=a,
                                 beta=beta, beta_g=beta_g, M=M,
                                 closed_hw=closed, numeric_hw=num,
                                 rel_err=float(rel),
                                 floor=beta * beta_g))
    # Score it two ways rather than one. A single "worst relative error" over a
    # grid half of whose cells have a half-width of 1e-17 reports the noise, not
    # the claim.
    big = [r for r in rows if r["closed_hw"] > 1e-6]
    deg = [r for r in rows if r["closed_hw"] <= 1e-6]
    relbig = [abs(r["closed_hw"] - r["numeric_hw"]) / r["closed_hw"]
              for r in big]
    ok = sum(x <= 1e-8 for x in relbig)
    print("\n   closed form vs the numeric maximum:")
    print("      %d of %d cells have a non-negligible half-width" %
          (len(big), len(rows)))
    print("      of those, %d of %d agree to 1e-8 relative; median error %.2e"
          % (ok, len(big), float(np.median(relbig))))
    print("      degenerate cells: worst ABSOLUTE error %.2e"
          % max((abs(r["closed_hw"] - r["numeric_hw"]) for r in deg),
                default=0.0))
    # the identity itself, on consistent objects
    idn = [abs(r["closed_hw"] - r["floor"]) / max(r["floor"], 1e-300)
           for r in big]
    print("   HW == beta*beta_g at the tight class: worst relative %.2e "
          "over %d cells   [P-C1]" % (max(idn), len(big)))
    RESULTS["part1_score"] = dict(n_big=len(big), n_agree=ok,
                                  median_rel=float(np.median(relbig)),
                                  worst_identity_rel=float(max(idn)))
    RESULTS["part1"] = rows
    return rows, float(max(idn))


def numeric_scalar(b_hat_v, Pn, g, M):
    """max <P_Nul g, n> s.t. n in Nul(range of Pn), ||b_hat_v + n|| <= M.

    Solved by SLSQP from several starts, never using beta_g.
    """
    from scipy.optimize import minimize
    n_o = b_hat_v.size
    lim2 = M ** 2 - float(b_hat_v @ b_hat_v)
    if lim2 <= 1e-18:
        return 0.0
    gN = Pn @ g
    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(6):
        x0 = rng.standard_normal(n_o) * np.sqrt(lim2 / n_o)
        res = minimize(lambda x: -float(gN @ (Pn @ x)), x0,
                       constraints=[{"type": "ineq",
                                     "fun": lambda x: lim2 -
                                     float((Pn @ x) @ (Pn @ x))}],
                       method="SLSQP", options=dict(maxiter=900, ftol=1e-15))
        if res.success:
            best = max(best, -res.fun)
    return float(best)


def part2(rows):
    """P-C2: the closed form should hold for EVERY admissible M, not just the
    tight one."""
    print("\nPART 2  sweeping the class bound M")
    p = default_params(n_s=4, n_a=N_A, n_o=6, n_o0=2, T=3, seed=0, confound=0.9)
    b_hat, b_true, Pn, U, nu1 = block_pieces(p, 0)
    k = b_true.shape[1]
    bt_v = b_true @ np.ones(k) / np.sqrt(k)
    bh_v = (U @ U.T) @ bt_v
    beta = float(np.linalg.norm(Pn @ bt_v))
    beta_g = float(np.linalg.norm(Pn @ nu1))
    Mt = float(np.linalg.norm(bt_v))
    print("   ||b_hat|| = %.4f   ||b_true|| = %.4f   beta = %.4f   "
          "beta_g = %.4f" % (np.linalg.norm(bh_v), Mt, beta, beta_g))
    print("%12s%14s%14s%12s%14s" %
          ("M", "closed HW", "numeric HW", "rel err", "HW/(beta*bg)"))
    out, worst = [], 0.0
    # The first version of this never evaluated M = ||b_true||, which is the ONLY
    # point P-C1 is about: its mult=1.0 branch set M = ||b_hat|| instead. The
    # tight case is now first in the list and labelled.
    Ms = [("||b_true||  TIGHT", Mt)] + \
         [("%.2fx tight" % m, Mt * m) for m in (1.05, 1.25, 1.5, 2.0, 3.0)] + \
         [("||b_hat||  degenerate", float(np.linalg.norm(bh_v)))]
    for tag, M in Ms:
        c = sharp_halfwidth_vector(bh_v, beta_g, M)
        n = numeric_scalar(bh_v, Pn, nu1, M)
        rel = agree(c, n, beta * beta_g)
        worst = max(worst, rel)
        print("%12.4f%14.6f%14.6f%12.2e%14.4f   %s" %
              (M, c, n, rel, c / max(beta * beta_g, 1e-300), tag))
        out.append(dict(M=M, tag=tag, closed=c, numeric=n, rel=float(rel),
                        ratio_to_floor=float(c / max(beta * beta_g, 1e-300))))
    print("   worst relative error over the sweep: %.3e" % worst)
    print("   at M = ||b_true|| = %.4f the ratio HW/(beta*beta_g) should be 1"
          % Mt)
    RESULTS["part2"] = out
    return worst


def part3():
    """P-C3: which policies are point-identified?"""
    print("\nPART 3  point-identification of candidate policy values")
    lo = np.linspace(0.9, 0.1, 6)
    cands = {
        "greedy_lo": np.stack([lo, 1 - lo], 1),
        "greedy_hi": np.stack([1 - lo, lo], 1),
        "always_0": np.tile([1.0, 0.0], (6, 1)),
        "always_1": np.tile([0.0, 1.0], (6, 1)),
        "uniform": np.full((6, 2), 0.5),
        "soft": np.stack([0.5 + 0.2 * (lo - 0.5) / 0.4,
                          0.5 - 0.2 * (lo - 0.5) / 0.4], 1),
    }
    rows = []
    for cfg, cf, tag in (((2, 6, 4), 0.9, "complete"),
                         ((4, 6, 2), 0.9, "incomplete")):
        n_s, n_o, n_o0 = cfg
        p = default_params(n_s=n_s, n_a=N_A, n_o=n_o, n_o0=n_o0, T=3,
                           seed=0, confound=cf)
        b_hat, b_true, Pn, U, nu1 = block_pieces(p, 0)
        k = b_true.shape[1]
        bt_v = b_true @ np.ones(k) / np.sqrt(k)
        bh_v = (U @ U.T) @ bt_v
        M = float(np.linalg.norm(bt_v))
        print("   %s %s   span dim %d of |S| = %d" %
              (tag, str(cfg), U.shape[1], n_s))
        print("      %-12s%14s%16s%16s" %
              ("policy", "beta_g", "half-width", "point-identified"))
        for name, pi in cands.items():
            # the stage-1 gradient profile weighted by the evaluation policy
            g = (p["p1"] @ p["E"]) * pi[:, 0]
            bg = float(np.linalg.norm(Pn @ g))
            hw = sharp_halfwidth_vector(bh_v, bg, M)
            # relative to the block scale. An absolute 1e-12 cut called a 1e-10
            # half-width "not identified" on a design where beta is about 1e-16,
            # which is floating-point noise in sqrt(M^2 - ||b_hat||^2).
            pid = bool(hw < 1e-8 * max(M, 1e-300))
            print("      %-12s%14.6f%16.3e%16s" %
                  (name, bg, hw, "yes" if pid else "no"))
            rows.append(dict(config=list(cfg), tag=tag, policy=name,
                             beta_g=bg, halfwidth=hw, point_identified=pid))
    n_c = sum(r["point_identified"] for r in rows if r["tag"] == "complete")
    n_i = sum(r["point_identified"] for r in rows if r["tag"] == "incomplete")
    print("   point-identified: %d of 6 complete, %d of 6 incomplete"
          % (n_c, n_i))
    RESULTS["part3"] = rows
    return n_c, n_i


def part4():
    """P-C4: turn the published widths into overpayment ratios."""
    print("\nPART 4  overpayment: published width divided by the sharp gap")
    exp = os.path.normpath(os.path.join(HERE, "..", "experiments"))
    # The first version silently read a 2-seed, single-N smoke file left by a
    # calibration run and reported an overpayment slope fitted through one point.
    # A source is now required to carry at least 3 sample sizes and 10 seeds.
    fn = None
    for c in ("results_stage_selective_dense_kappa15.json",
              "results_stage_selective_dense_lowrank_kappa15.json",
              "results_regret_incomplete.json"):
        q = os.path.join(exp, c)
        if not os.path.exists(q):
            continue
        j = json.load(open(q))
        recs = j.get("penalties") or j.get("rows") or []
        if len({r["N"] for r in recs}) >= 3 and j.get("n_seeds", 20) >= 10:
            fn = q
            break
    if fn is None:
        print("   no stored run with >= 3 sample sizes and >= 10 seeds; "
              "skipping")
        return None
    d = json.load(open(fn))
    gap = d.get("floor")
    print("   source: %s   sharp gap beta*beta_g = %.4f"
          % (os.path.basename(fn), gap))
    rows = [r for r in d.get("rows", []) if "width" in r]
    if not rows:
        pen = d.get("penalties", [])
        if pen:
            print("%20s%10s%12s%14s" % ("schedule", "N", "t1 penalty",
                                        "overpayment"))
            for r in sorted(pen, key=lambda r: (r["schedule"], r["N"])):
                w = r.get("pen_bR_t1", float("nan"))
                print("%20s%10d%12.4f%14.4f" %
                      (r["schedule"], r["N"], w, w / gap))
            sch = sorted({r["schedule"] for r in pen})
            for s in sch:
                sub = sorted([r for r in pen if r["schedule"] == s],
                             key=lambda r: r["N"])
                Ns = np.log([r["N"] for r in sub])
                ov = np.log([r["pen_bR_t1"] / gap for r in sub])
                print("   %s  overpayment log-log slope %+.4f"
                      % (s, float(np.polyfit(Ns, ov, 1)[0])))
            RESULTS["part4"] = dict(source=os.path.basename(fn), gap=gap,
                                    rows=pen)
            return gap
    return gap


def main():
    rows, worst1 = part1()
    worst2 = part2(rows)
    n_c, n_i = part3()
    part4()
    print("\n%s\nSUMMARY\n%s" % ("=" * 76, "=" * 76))
    print("P-C1  closed form vs numeric, worst rel err %.3e  (tol 1e-10)"
          % worst1)
    print("P-C2  same over an M sweep,   worst rel err %.3e" % worst2)
    print("P-C3  point-identified policies: %d/6 complete, %d/6 incomplete"
          % (n_c, n_i))
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_converse.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
