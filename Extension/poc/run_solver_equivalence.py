"""run_solver_equivalence.py -- did swapping in solve_triangular change anything?

ellipsoid_opt.py used np.linalg.solve on Cholesky factors, i.e. a full LU solve
on a triangular matrix. At the D = 144 blocks of every previous run that costs
nothing measurable. At Environment L's D = 2000 dynamic blocks it is 542 ms per
call against 14 ms for a triangular solve, and pessimistic_value makes
19 x 30 x 3 = 1710 of them per candidate per arm. That is the difference between
6 minutes and 10 hours per seed.

The seven calls are now solve_triangular. Mathematically identical; the point of
this file is to show that numerically it is too, on the grids where results are
already stored.

Compares h_inv_norm, linear_min (plain and projected), xi_needed and the full
pessimistic_value against the pre-change module loaded from a backup copy.

Writes experiments/results_solver_equivalence.json.
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

BACKUP = os.environ.get("BACKUP", "/tmp/ellipsoid_opt.bak")


def load_backup(path):
    spec = importlib.util.spec_from_file_location("ellipsoid_opt_old", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    from dim_separated_pomdp import default_params, dp_value
    from value_plugin import empirical_p_o1
    import run_stage_selective as D
    import ellipsoid_opt as new

    if not os.path.exists(BACKUP):
        raise SystemExit("backup module not found at %s; set BACKUP" % BACKUP)
    old = load_backup(BACKUP)

    p = default_params(n_s=4, n_a=2, n_o=6, n_o0=2, T=3, seed=0, confound=0.9)
    cands = D.candidates()
    best = max(cands, key=lambda k: dp_value(p, cands[k]))
    rows, worst = [], {}

    def track(name, a, b):
        d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
        worst[name] = max(worst.get(name, 0.0), d)
        return d

    for N in (2000, 8000, 32000):
        for sd in (0, 1):
            est, d = D.fit_once(p, N, sd, 1.5)
            p_o1 = empirical_p_o1(d["O"], D.N_O)
            vg = D.make_vg(cands[best], p_o1)

            def blocks(mod):
                bR, bD = [], []
                for t in range(1, D.T + 1):
                    s = est.stage_store["bR_t%d" % t]
                    bR.append(mod.BlockEllipsoid(
                        s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"],
                        "bR_t%d" % t, s["signal_basis"], n_obs=D.N_O,
                        n_act=D.N_A, n_y=s["n_y"]))
                    if t < D.T:
                        s = est.stage_store["bD_t%d" % t]
                        bD.append(mod.BlockEllipsoid(
                            s["H"], s["b_hat_vec"], s["sigma2_signal"],
                            s["N2"], "bD_t%d" % t, s["signal_basis"],
                            n_obs=D.N_O, n_act=D.N_A, n_y=s["n_y"]))
                return bR, bD

            bRn, bDn = blocks(new)
            bRo, bDo = blocks(old)
            _, gR, gD = vg([b.b_hat for b in bRn], [b.b_hat for b in bDn])
            xR = [D.xi_for(0.0, b.N2, b.sigma2, 1e-6) for b in bRn]
            xD = [D.xi_for(0.0, b.N2, b.sigma2, 1e-6) for b in bDn]

            for t, (bn, bo, g) in enumerate(zip(bRn, bRo, gR)):
                track("h_inv_norm", bn.h_inv_norm(g), bo.h_inv_norm(g))
                pn, qn = bn.linear_min(g, xR[t])
                po, qo = bo.linear_min(g, xR[t])
                track("linear_min_pt", pn, po)
                track("linear_min_pen", qn, qo)
                if bn.U is not None:
                    pn, qn = bn.linear_min(g, xR[t], projected=True)
                    po, qo = bo.linear_min(g, xR[t], projected=True)
                    track("linear_min_proj_pt", pn, po)
                    track("linear_min_proj_pen", qn, qo)

            for proj in (False, True):
                vn = new.pessimistic_value(bRn, bDn, xR, xD, vg,
                                           projected=proj,
                                           rng=np.random.default_rng(0))
                vo = old.pessimistic_value(bRo, bDo, xR, xD, vg,
                                           projected=proj,
                                           rng=np.random.default_rng(0))
                dv = track("V_low_proj" if proj else "V_low",
                           vn["V_low"], vo["V_low"])
                rows.append(dict(N=N, seed=sd, projected=proj,
                                 V_low_new=vn["V_low"], V_low_old=vo["V_low"],
                                 diff=dv))
            print("N=%-7d seed=%d   V_low diff %.3e (plain)  %.3e (projected)"
                  % (N, sd, rows[-2]["diff"], rows[-1]["diff"]))

    print("\n" + "=" * 78)
    for k in sorted(worst):
        print("%-22s worst absolute difference  %.3e" % (k, worst[k]))
    w = max(worst.values())
    print("\noverall worst: %.3e   %s"
          % (w, "MACHINE PRECISION" if w < 1e-9 else "TOO LARGE -- investigate"))

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_solver_equivalence.json"))
    with open(out, "w") as f:
        json.dump(dict(rows=rows, worst=worst, overall=w), f, indent=2,
                  default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
