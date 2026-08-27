"""run_projall_signflip.py -- is projall's conservatism structural or an accident?

Predictions: docs/stage_selective_pessimism.md section 10, P-C14..P-C17,
committed at 0e1ffde before this file existed.

WHAT IS AT STAKE. projected=True restricts the perturbation direction to the
signal subspace U_sig and leaves the centre b_hat alone. It keeps the functional
<g,b> and shrinks the region until the region no longer holds the truth. So

    V_low - V_true  =  -penalty  -  c ,    c = <g, (I - U_sig U_sig') (b_true - b_hat)>

and projall is conservative iff c >= -penalty. |c| <= ||P_excl g||*||P_excl db||
by Cauchy-Schwarz with the sign fixed purely by alignment. BlockEllipsoid's own
docstring flags this as audit finding A-high and says coverage held "by
sign-alignment, not by construction". This measures the sign.

If it flips anywhere, projall is not a remedy, it is a rule that happened to land
right on the one grid we ran.

Writes experiments/results_projall_signflip.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params, dp_value
from value_plugin import empirical_p_o1
import run_stage_selective as D
from run_empirical_null import true_bR_vec

TAU, KAP = 0.0, 1.5
CONFIGS = [(4, 6, 2), (4, 8, 3), (2, 6, 2), (3, 7, 5)]
CONFOUNDS = [0.0, 0.3, 0.6, 0.9]
SWEEP_N = [int(x) for x in os.environ.get("SWEEPN","8000,32000").split(",")]
SWEEP_SEEDS = [0, 1, 2]
DEEP_CFG, DEEP_CF = (4, 6, 2), 0.9
DEEP_N = [int(x) for x in os.environ.get("DEEPN","2000,8000,32000,128000").split(",")]
DEEP_SEEDS = [0, 1, 2, 3, 4]


def cell(cfg, cf, N, seeds, cm=None):
    """Excluded value component c, projected penalty, and the margin, at bR_t1."""
    n_s, n_o, n_o0 = cfg
    D.N_S, D.N_O, D.N_O0 = n_s, n_o, n_o0
    p = default_params(n_s=n_s, n_a=D.N_A, n_o=n_o, n_o0=n_o0, T=D.T,
                       seed=0, confound=cf)
    cands = D.candidates()
    best = max(cands, key=lambda k: dp_value(p, cands[k]))
    b_true, _ = true_bR_vec(p, n_o, D.N_A)
    acc = {k: [] for k in ("c", "pen", "cos", "ng", "ndb", "kexcl")}
    for sd in seeds:
        est, d = D.fit_once(p, N, sd, KAP)
        bR, bD = D.build_blocks(est)
        b0 = bR[0]
        _, gR, _ = D.make_vg(cands[best], empirical_p_o1(d["O"], n_o))(
            [b.b_hat for b in bR], [b.b_hat for b in bD])
        g = gR[0]
        db = b_true - b0.b_hat
        if b0.U is None:                       # nothing excluded
            gx = np.zeros_like(g)
            dbx = np.zeros_like(db)
        else:
            gx = g - b0.U @ (b0.U.T @ g)
            dbx = db - b0.U @ (b0.U.T @ db)
        c = float(g @ dbx)                     # = <gx, dbx>, P_excl idempotent
        xi = D.xi_for(TAU, b0.N2, b0.sigma2, cm if cm else 1.0)
        pen = 0.0 if b0.U is None else \
            float(b0.linear_min(g, xi, projected=True)[1])
        ng, ndb = float(np.linalg.norm(gx)), float(np.linalg.norm(dbx))
        acc["c"].append(c)
        acc["pen"].append(pen)
        acc["cos"].append(c / (ng * ndb) if ng * ndb > 1e-300 else 0.0)
        acc["ng"].append(ng)
        acc["ndb"].append(ndb)
        acc["kexcl"].append(0 if b0.U is None else g.size - b0.U.shape[1])
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    m["margin"] = -m["pen"] - m["c"]           # <= 0 means conservative
    m["optimal"] = best
    return m


def main():
    p0 = default_params(n_s=DEEP_CFG[0], n_a=D.N_A, n_o=DEEP_CFG[1],
                        n_o0=DEEP_CFG[2], T=D.T, seed=0, confound=DEEP_CF)
    D.N_S, D.N_O, D.N_O0 = DEEP_CFG
    D.SCHEDULES = [("C  kappa=1.5", TAU, KAP)]
    D.N_GRID = DEEP_N
    D.calibrate(p0, D.candidates())
    cm = D.C_CAL["C  kappa=1.5"]

    print("\nP-C14/P-C15  %s at confounding %s, %d seeds" %
          (DEEP_CFG, DEEP_CF, len(DEEP_SEEDS)))
    print("%10s%12s%12s%10s%12s%12s" %
          ("N", "c", "pen_proj", "cos", "margin", "conserv"))
    deep = []
    for N in DEEP_N:
        m = cell(DEEP_CFG, DEEP_CF, N, DEEP_SEEDS, cm)
        print("%10d%12.4f%12.4f%10.3f%12.4f%12s"
              % (N, m["c"], m["pen"], m["cos"], m["margin"],
                 "yes" if m["margin"] <= 0 else "NO"))
        m["N"] = N
        deep.append(m)

    print("\nP-C16  sign sweep, %s at N in %s, %d seeds"
          % (CONFIGS, SWEEP_N, len(SWEEP_SEEDS)))
    print("%12s%7s%9s%12s%12s%10s%12s%9s" %
          ("config", "cf", "N", "c", "pen_proj", "cos", "margin", "conserv"))
    sweep = []
    for cfg in CONFIGS:
        for cf in CONFOUNDS:
            for N in SWEEP_N:
                m = cell(cfg, cf, N, SWEEP_SEEDS, cm)
                print("%12s%7.1f%9d%12.4f%12.4f%10.3f%12.4f%9s"
                      % (str(cfg), cf, N, m["c"], m["pen"], m["cos"],
                         m["margin"], "yes" if m["margin"] <= 0 else "NO"))
                m.update(config=list(cfg), confound=cf, N=N)
                sweep.append(m)

    print("\n" + "=" * 88)
    d0 = deep[0]
    print("P-C14  c > 0 at (4,6,2) cf 0.9: %s   |cos| = %.3f  [predicted > 0.3]"
          % (d0["c"] > 0, abs(d0["cos"])))
    cs = [r["c"] for r in deep if r["N"] >= 8000]
    spread = (max(cs) - min(cs)) / abs(np.mean(cs)) if cs else float("nan")
    print("P-C15  c relative spread over N >= 8000: %.3f  [predicted < 0.20]"
          % spread)
    live = [r for r in sweep if abs(r["c"]) > 1e-9 or r["pen"] > 1e-9]
    neg = [r for r in live if r["c"] < 0]
    print("P-C16  cells with a live exclusion: %d of %d;  c < 0 in %d"
          % (len(live), len(sweep), len(neg)))
    for r in neg:
        print("       %s cf=%.1f N=%d   c = %+.4f  margin = %+.4f  %s"
              % (r["config"], r["confound"], r["N"], r["c"], r["margin"],
                 "ANTI-CONSERVATIVE" if r["margin"] > 0 else "still conservative"))
    bad = [r for r in sweep if r["margin"] > 0]
    print("P-C17  anti-conservative cells: %d of %d" % (len(bad), len(sweep)))
    for r in bad:
        print("       %s cf=%.1f N=%d   margin = %+.4f"
              % (r["config"], r["confound"], r["N"], r["margin"]))

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_projall_signflip.json"))
    with open(out, "w") as f:
        json.dump(dict(deep=deep, sweep=sweep, tau=TAU, kappa=KAP, c_mult=cm,
                       deep_cfg=list(DEEP_CFG), deep_cf=DEEP_CF,
                       n_negative=len(neg), n_anticonservative=len(bad)),
                  f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
