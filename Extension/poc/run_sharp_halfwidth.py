"""run_sharp_halfwidth.py -- is the t=1 region actually invalid?

Predictions: docs/converse_identification.md section 8, P-C6..P-C9, committed at
773b0cc before this file existed.

WHAT IS AT STAKE. P-C4 divided pen_bR_t1 by floor = beta*beta_g and read the
result as an overpayment ratio. Those two are not commensurable:

  pen_bR_t1   value units, empirical coefficient space R^144, empirical T_2 null
  beta        BRIDGE units, per-action tensor R^{6x2x6}, POPULATION null in R^6
  beta_g      dimensionless, same population null

So the published ratio mixes units and mixes null spaces, and the conclusion
"the regions are invalid" does not follow from it. This is the sixth instance of
the same failure mode: two objects sharing a name and not a space.

The coherent gap puts both factors in the null the pessimism layer inverts and
leaves both absolute:

    HW_t1 = || U^T b_true_t1 || * || U^T g_t1 ||

which is section 2's HW = beta_g * sqrt(M^2 - ||b_hat||^2) at the tight class
M = ||b_true||, restricted to one block. Validity needs pen_bR_t1 >= HW_t1.

g_t1 is the OPTIMAL policy's stage-1 gradient, the same one the driver penalises,
so the numerator and denominator are built from one object.

Writes experiments/results_sharp_halfwidth.json.
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
from run_empirical_null import null_basis, true_bR_vec

N_GRID = [int(x) for x in
          os.environ.get("NGRID", "2000,8000,32000,128000").split(",")]
SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
TAU, KAP = 0.0, 1.5                              # schedule C, e = +0.5
STORED = "results_stage_selective_dense_kappa15.json"


def main():
    p = default_params(n_s=D.N_S, n_a=D.N_A, n_o=D.N_O, n_o0=D.N_O0, T=D.T,
                       seed=0, confound=D.CONFOUND)
    cands = D.candidates()
    v_true = {k: dp_value(p, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    b_true, resid = true_bR_vec(p, D.N_O, D.N_A)
    print("(%d,%d,%d) cf=%s  schedule C (tau=%.1f, kappa=%.1f, e=%+.1f)"
          % (D.N_S, D.N_O, D.N_O0, D.CONFOUND, TAU, KAP, TAU + KAP - 1))
    print("optimal %s   ||b_true_t1|| = %.4f   bridge residual %.2e\n"
          % (best, np.linalg.norm(b_true), resid))

    # Same calibration path as the driver, so pen_bR_t1 is comparable to the
    # stored 20-seed run rather than to a differently-scaled region.
    D.SCHEDULES = [("C  kappa=1.5", TAU, KAP)]
    D.N_GRID = N_GRID
    D.calibrate(p, cands)
    cm = D.C_CAL["C  kappa=1.5"]

    print("\n%10s%12s%12s%12s%10s%12s%12s" %
          ("N", "pen_t1", "HW_t1", "ratio", "nulldim", "|U'b_true|", "|U'g|"))
    rows = []
    for N in N_GRID:
        acc = {k: [] for k in ("pen", "hw", "nb", "ng", "nd")}
        for sd in SEEDS:
            est, d = D.fit_once(p, N, sd, KAP)
            bR, bD = D.build_blocks(est)
            p_o1 = empirical_p_o1(d["O"], D.N_O)
            xR = [D.xi_for(TAU, b.N2, b.sigma2, cm) for b in bR]
            _, gR, _gD = D.make_vg(cands[best], p_o1)(
                [b.b_hat for b in bR], [b.b_hat for b in bD])
            pen = float(np.sqrt(xR[0]) * bR[0].h_inv_norm(gR[0]))
            U = null_basis(bR[0].H, est.stage_store["bR_t1"]["lam2"])
            if U.shape[1] == 0:
                nb = ng = 0.0
            else:
                nb = float(np.linalg.norm(U.T @ b_true))
                ng = float(np.linalg.norm(U.T @ gR[0]))
            acc["pen"].append(pen)
            acc["hw"].append(nb * ng)
            acc["nb"].append(nb)
            acc["ng"].append(ng)
            acc["nd"].append(U.shape[1])
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        r = m["pen"] / m["hw"] if m["hw"] > 0 else float("inf")
        print("%10d%12.4f%12.4f%12.4f%10.1f%12.4f%12.4f" %
              (N, m["pen"], m["hw"], r, m["nd"], m["nb"], m["ng"]))
        rows.append(dict(N=N, pen=m["pen"], hw=m["hw"], ratio=r,
                         nulldim=m["nd"], nb=m["nb"], ng=m["ng"],
                         pen_sd=float(np.std(acc["pen"])),
                         hw_sd=float(np.std(acc["hw"]))))

    print("\n" + "=" * 82)
    # sanity: does this 5-seed pen reproduce the stored 20-seed pen?
    q = os.path.normpath(os.path.join(HERE, "..", "experiments", STORED))
    stored = None
    if os.path.exists(q):
        j = json.load(open(q))
        ps = {r["N"]: r["pen_bR_t1"] for r in j["penalties"]}
        stored = j
        print("cross-check against the stored %d-seed run:" % j["n_seeds"])
        print("%10s%12s%12s%10s" % ("N", "here", "stored", "rel diff"))
        for r in rows:
            s = ps.get(r["N"])
            if s:
                print("%10d%12.4f%12.4f%10.3f"
                      % (r["N"], r["pen"], s, abs(r["pen"] - s) / s))

    Ns = np.log([r["N"] for r in rows])
    print("\nP-C6  HW_t1 > pen_t1 at %d of %d sample sizes"
          % (sum(1 for r in rows if r["hw"] > r["pen"]), len(rows)))
    if stored:
        old = {r["N"]: r["pen_bR_t1"] / stored["floor"]
               for r in stored["penalties"]}
        n_sm = sum(1 for r in rows if r["ratio"] < old.get(r["N"], np.inf))
        print("P-C7  corrected ratio below the incoherent one at %d of %d; "
              "at N=%d it is %.4f against %.4f"
              % (n_sm, len(rows), rows[-1]["N"], rows[-1]["ratio"],
                 old.get(rows[-1]["N"], float("nan"))))
    settled = [r for r in rows if r["N"] >= 8000]
    hw = [r["hw"] for r in settled]
    spread = (max(hw) - min(hw)) / np.mean(hw)
    print("P-C8  HW_t1 relative spread over N >= 8000: %.3f  [predicted < 0.20]"
          % spread)
    sl = float(np.polyfit(Ns, np.log([r["ratio"] for r in rows]), 1)[0])
    print("P-C9  corrected ratio log-log slope %+.4f  [width slope +0.1812]"
          % sl)

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_sharp_halfwidth.json"))
    with open(out, "w") as f:
        json.dump(dict(rows=rows, seeds=SEEDS, tau=TAU, kappa=KAP,
                       optimal=best, b_true_norm=float(np.linalg.norm(b_true)),
                       ratio_slope=sl, hw_spread=float(spread)),
                  f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
