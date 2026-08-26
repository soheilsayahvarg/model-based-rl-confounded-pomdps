"""run_selfheal_formula.py -- self-healing as a theorem, not a measurement.

Predictions: docs/selfheal_theorem.md, committed before this file existed
(commits d0e9cab, and d2beed2 which corrects the proof BEFORE any measurement).

WHAT IS BEING TESTED. cor:selfheal is currently a measurement: 24/24 population
cells, span reaches |S| at every t >= 2 for confounding < 1. Section 7 of the doc
derives a closed form for that span. Writing
    w_{a1,o0}[s] = p1[s] K0[s,o0] pi_b[s,a1]
the t=2 profiles are E^T M E[:,o1] with M = diag(pi_b[:,a]) P[a1]^T diag(w), so
the span over o1 is E^T range(M) and the whole per-action span is

    W_1 = span{ p1 * K0[:,o0] : o0 }
    W_{k+1} = sum over a_k of  P[a_k]^T diag(pi_b[:,a_k]) span{ z * E[:,o] :
                                                    z in W_k, o }
    stage-t span = E^T diag(pi_b[:,a]) W_t

That recursion is O(|S|^2 |O| |A|) per stage. The brute force enumerates
(|O||A|)^(t-1) |O0| profiles and takes an SVD. If the two agree in every cell,
the closed form IS the span and the corollary is proved rather than observed.

THE SIDE CONDITION, WHICH IS THE POINT. The closed form fails to reach |S| when
diag(pi_b[:,a]) is rank-deficient (saturated confounding, the knife-edge we
measured and never explained) or when the reachable transition ROWS fail to span
R^|S|. Our environment family draws every transition row from a Dirichlet, so
that second case has never been generated. Two families are added here to
generate it:

    sparse_support   rows supported on {0,1} only        support fails AND rank fails
    dense_lowrank    rows have FULL support but lie in   support holds, rank fails
                     a 2-dimensional row space

dense_lowrank is the discriminating cell. The support version of the proof
(section 2 of the doc, wrong) predicts a full span there because the supports
cover S. The rank version (section 7, corrected) predicts 2. They disagree, so
the cell decides which proof is right.

Writes experiments/results_selfheal_formula.json.
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
from run_completeness_beta import true_bridges_generic, population_design_span

TOL = 1e-10
CONFIGS = [(4, 6, 2), (4, 8, 3), (2, 6, 4)]
CONFOUNDS = [0.0, 0.3, 0.6, 0.9, 1.0]
STAGES = [1, 2, 3, 4]
FAMILIES = ["dense", "sparse_support", "dense_lowrank"]
RESULTS = {}


def orth(M, tol=TOL):
    """Orthonormal basis of the column space of M, by SVD with a relative cut."""
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    if sv[0] <= 0:
        return np.zeros((M.shape[0], 0))
    r = int((sv > tol * sv[0]).sum())
    return U[:, :r]


def make_params(cfg, confound, family, seed=0, T=3):
    n_s, n_o, n_o0 = cfg
    p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=T,
                       seed=seed, confound=confound)
    if family == "dense":
        return p
    n_a = p["P"].shape[0]
    rng = np.random.default_rng(1234 + seed)
    if family == "sparse_support":
        # Every transition lands in {0, 1}. Rows are supported on two states, so
        # both the support union and the row rank are deficient for n_s > 2.
        P = np.zeros_like(p["P"])
        for a in range(n_a):
            for s in range(n_s):
                q = rng.uniform(0.2, 0.8)
                P[a, s, 0] = q
                P[a, s, 1] = 1.0 - q
        p = dict(p)
        p["P"] = P
        return p
    if family == "dense_lowrank":
        # Every row is a mixture of the SAME two full-support distributions, so
        # every row has full support while the row space has dimension 2. This is
        # the cell where the support proof and the rank proof disagree.
        d0 = rng.dirichlet(np.ones(n_s) * 2.0)
        d1 = rng.dirichlet(np.ones(n_s) * 2.0)
        P = np.zeros_like(p["P"])
        for a in range(n_a):
            for s in range(n_s):
                th = 0.15 + 0.7 * ((a * n_s + s + 1) / (n_a * n_s + 1))
                P[a, s] = th * d0 + (1.0 - th) * d1
        p = dict(p)
        p["P"] = P
        assert (P > 0).all(), "dense_lowrank must have full support"
        return p
    raise ValueError(family)


def span_closedform(p, a, t, tol=TOL):
    """Orthonormal basis of the stage-t per-action design span, by the recursion.

    Returns (basis in R^{n_o}, dim of W_t before the final E^T).
    """
    E, P, p1, K0, pi_b = p["E"], p["P"], p["p1"], p["K0"], p["pi_b"]
    n_s, n_o = E.shape
    n_o0, n_a = K0.shape[1], P.shape[0]

    W = orth(np.stack([p1 * K0[:, x] for x in range(n_o0)], axis=1))
    for _ in range(t - 1):
        # free observation index: {z * E[:,o] : z in W, o} spans the image of the
        # bilinear map, then one action step
        cols = []
        for k in range(W.shape[1]):
            for o in range(n_o):
                cols.append(W[:, k] * E[:, o])
        Z = orth(np.stack(cols, axis=1)) if cols else np.zeros((n_s, 0))
        cols = []
        for ak in range(n_a):
            Mk = P[ak].T @ (Z * pi_b[:, ak][:, None])
            for k in range(Mk.shape[1]):
                cols.append(Mk[:, k])
        W = orth(np.stack(cols, axis=1)) if cols else np.zeros((n_s, 0))
        if W.shape[1] == 0:
            break
    dimW = W.shape[1]
    if dimW == 0:
        return np.zeros((n_o, 0)), 0
    return orth(E.T @ (W * pi_b[:, a][:, None])), dimW


def span_bruteforce(p, a, t, tol=TOL):
    """Orthonormal basis of the same span, by enumerating every conditioning cell.

    Independent of span_closedform: it never forms W, it normalizes each profile
    exactly as the estimator's design does, and it takes one SVD at the end.
    """
    E, P, p1, K0, pi_b = p["E"], p["P"], p["p1"], p["K0"], p["pi_b"]
    n_s, n_o = E.shape
    n_o0, n_a = K0.shape[1], P.shape[0]

    fronts = [p1 * K0[:, x] for x in range(n_o0)]
    for _ in range(t - 1):
        nxt = []
        for w in fronts:
            for o in range(n_o):
                for ak in range(n_a):
                    wn = (w * E[:, o] * pi_b[:, ak]) @ P[ak]
                    if wn.sum() > 1e-300:
                        nxt.append(wn)
        fronts = nxt
        if not fronts:
            break
    cols = []
    for w in fronts:
        v = w * pi_b[:, a]
        if v.sum() > 1e-300:
            cols.append(E.T @ (v / v.sum()))
    if not cols:
        return np.zeros((n_o, 0))
    return orth(np.stack(cols, axis=1), tol)


def beta_at(p, a, basis):
    """||P_Nul b_true|| for the reward block, against a given design span."""
    E = p["E"]
    n_o = E.shape[1]
    bR, _res = true_bridges_generic(p)
    Pn = np.eye(n_o) - basis @ basis.T if basis.shape[1] else np.eye(n_o)
    leak = np.einsum("ij,jro->iro", Pn, bR[a])
    nb = float(np.linalg.norm(bR[a]))
    return float(np.linalg.norm(leak)), nb


def main():
    rows = []
    agree = disagree = 0
    print("closed form vs brute force, per (family, config, confound, action, t)")
    print("%16s%10s%6s%4s%4s%9s%9s%8s%12s" %
          ("family", "config", "cf", "a", "t", "closed", "brute", "|S|",
           "beta_t/|b|"))
    for family in FAMILIES:
        for cfg in CONFIGS:
            n_s = cfg[0]
            for cf in CONFOUNDS:
                p = make_params(cfg, cf, family)
                for a in range(2):
                    for t in STAGES:
                        Uc, dimW = span_closedform(p, a, t)
                        Ub = span_bruteforce(p, a, t)
                        dc, db = Uc.shape[1], Ub.shape[1]
                        ok = (dc == db)
                        if ok and dc > 0:
                            # same dimension is necessary, same SUBSPACE is the
                            # claim; check the principal angles too
                            gap = float(np.linalg.norm(
                                Uc @ Uc.T - Ub @ Ub.T, 2))
                            ok = gap < 1e-8
                        else:
                            gap = float("nan")
                        agree += int(ok)
                        disagree += int(not ok)
                        leak, nb = beta_at(p, a, Ub)
                        rows.append(dict(
                            family=family, config=list(cfg), confound=cf,
                            action=a, stage=t, dim_closed=dc, dim_brute=db,
                            dim_W=dimW, n_s=n_s, subspace_gap=gap,
                            beta=leak, beta_rel=leak / max(nb, 1e-300),
                            agree=bool(ok)))
                        if t <= 2 or dc != n_s:
                            print("%16s%10s%6.1f%4d%4d%9d%9d%8d%12.2e%s" %
                                  (family, str(cfg), cf, a, t, dc, db, n_s,
                                   leak / max(nb, 1e-300),
                                   "" if ok else "   <-- DISAGREE"))

    print("\nP-S4' closed form == brute force (dimension AND subspace): "
          "%d/%d cells" % (agree, agree + disagree))

    def sel(**kw):
        out = rows
        for k, v in kw.items():
            out = [r for r in out if r[k] == v]
        return out

    print("\nP-S1 control: dense family, confound < 1, t >= 2, span == |S|")
    bad = [r for r in sel(family="dense")
           if r["confound"] < 1.0 and r["stage"] >= 2
           and r["dim_brute"] != r["n_s"]]
    print("   violations: %d of %d" %
          (len(bad), len([r for r in sel(family="dense")
                          if r["confound"] < 1.0 and r["stage"] >= 2])))

    print("\nP-S2 knife-edge: dense family, confound == 1, t >= 2")
    for r in sel(family="dense", confound=1.0):
        if r["stage"] == 2:
            print("   %s a=%d  span %d  |S| %d  predicted |S|/|A| = %d" %
                  (r["config"], r["action"], r["dim_brute"], r["n_s"],
                   r["n_s"] // 2))

    print("\nP-S3' reachability: sparse and low-rank families at confound 0.0")
    for family in ("sparse_support", "dense_lowrank"):
        for r in sel(family=family, confound=0.0, stage=2, action=0):
            print("   %-15s %s  t=2 span %d of |S|=%d   beta_rel %.3f" %
                  (family, r["config"], r["dim_brute"], r["n_s"],
                   r["beta_rel"]))

    print("\nDISCRIMINATING CELL. dense_lowrank has FULL-support rows, so the")
    print("support proof (doc section 2, wrong) predicts span = |S|; the rank")
    print("proof (section 7, corrected) predicts the row-space dimension, 2.")
    for r in sel(family="dense_lowrank", confound=0.0, stage=2):
        print("   %s a=%d  measured span %d   support proof %d   rank proof 2"
              % (r["config"], r["action"], r["dim_brute"], r["n_s"]))

    print("\nP-S5 monotone in t: span never decreases with stage "
          "(confound < 1)")
    viol = 0
    for family in FAMILIES:
        for cfg in CONFIGS:
            for cf in CONFOUNDS:
                if cf >= 1.0:
                    continue
                for a in range(2):
                    ds = [r["dim_brute"] for r in
                          sorted(sel(family=family, config=list(cfg),
                                     confound=cf, action=a),
                                 key=lambda r: r["stage"])]
                    viol += int(any(ds[i + 1] < ds[i]
                                    for i in range(len(ds) - 1)))
    print("   sequences that decrease somewhere: %d" % viol)

    RESULTS.update(rows=rows, agree=agree, disagree=disagree, tol=TOL,
                   families=FAMILIES, configs=[list(c) for c in CONFIGS],
                   confounds=CONFOUNDS, stages=STAGES)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_selfheal_formula.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
