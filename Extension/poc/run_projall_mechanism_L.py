"""run_projall_mechanism_L.py -- projall's excluded component on fresh cells.

Predictions: docs/scale_and_baselines.md P-L8 and P-L9, committed at f709bba.

WHY THIS GRID. run_projall_signflip.py found c > 0 in 30 of 32 cells and could
not break the arm. Two explanations were then formed AFTER seeing those numbers
and are therefore untested:

  concentration   |cos| ~ 1/sqrt(k_excl), so c is small by dimension
  sign structure  g and b_true keep a common positive component through P_excl

Environment L is a fresh grid: |S|=8, |O|=10, |O_0|=3, T=10, with k_excl an order
of magnitude larger and NINETEEN blocks instead of five. P-L9 is the first test
of the concentration hypothesis on cells it was not fitted to.

This measures c and cos at EVERY block, not only t=1, which the small-environment
run never did. If the sign is positive at t=1 and mixed later, the arm's safety
is a stage-1 accident and does not transfer to a long horizon.

No pessimistic_value call, so this is cheap: one fit per (N, seed).

Writes experiments/results_projall_mechanism_L.json.
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
import run_environment_L as L

N_GRID = [int(x) for x in os.environ.get("NGRID", "4000,16000").split(",")]
SEEDS = list(range(int(os.environ.get("SEEDS", "3"))))


def true_bR_block(p, t):
    """True reward bridge at stage t as the estimator's flat coefficient vector.

    Only available in closed form for the reward blocks; the dynamic blocks are
    left out rather than approximated.
    """
    from run_completeness_beta import true_bridges_generic
    bR, res = true_bridges_generic(p)
    return np.stack([bR[a] for a in range(L.N_A)], axis=0).ravel(), float(res)


def main():
    p = default_params(n_s=L.N_S, n_a=L.N_A, n_o=L.N_O, n_o0=L.N_O0, T=L.T,
                       seed=0, confound=L.CONFOUND)
    cands = L.candidates(p)
    v_true = {k: dp_value(p, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    b_true_t1, resid = true_bR_block(p, 1)
    print("Environment L, optimal candidate %s, bridge residual %.2e" %
          (best, resid))
    print("%8s%6s%8s%12s%12s%10s%10s%12s" %
          ("N", "seed", "block", "c", "|P_x g|", "cos", "k_excl",
           "1/sqrt(k)"))

    rows = []
    for N in N_GRID:
        for sd in SEEDS:
            est, d = L.fit_once(p, N, sd)
            bR, bD = L.build_blocks(est)
            p_o1 = empirical_p_o1(d["O"], L.N_O)
            _, gR, gD = L.make_vg(cands[best], p_o1)(
                [b.b_hat for b in bR], [b.b_hat for b in bD])
            for t, (b, g) in enumerate(zip(bR, gR), start=1):
                if b.U is None:
                    continue
                gx = g - b.U @ (b.U.T @ g)
                k = int(b.b_hat.size - b.U.shape[1])
                # the truth is only known in closed form at t = 1
                if t == 1:
                    db = b_true_t1 - b.b_hat
                    dbx = db - b.U @ (b.U.T @ db)
                    c = float(g @ dbx)
                    ndb = float(np.linalg.norm(dbx))
                    cos = c / (np.linalg.norm(gx) * ndb) if ndb > 0 else 0.0
                else:
                    c, cos = float("nan"), float("nan")
                print("%8d%6d%8d%12.4f%12.4f%10.3f%10d%12.3f"
                      % (N, sd, t, c, np.linalg.norm(gx), cos, k,
                         1.0 / np.sqrt(max(k, 1))))
                rows.append(dict(N=N, seed=sd, block=t, c=c, cos=cos,
                                 k_excl=k, gx=float(np.linalg.norm(gx)),
                                 inv_sqrt_k=1.0 / np.sqrt(max(k, 1))))

    print("\n" + "=" * 82)
    t1 = [r for r in rows if r["block"] == 1 and not np.isnan(r["c"])]
    pos = sum(1 for r in t1 if r["c"] > 0)
    print("P-L8  c > 0 at t=1 in %d of %d cells" % (pos, len(t1)))
    if t1:
        cm = float(np.mean([abs(r["cos"]) for r in t1]))
        ik = float(np.mean([r["inv_sqrt_k"] for r in t1]))
        ratio = cm / ik if ik > 0 else float("nan")
        print("P-L9  mean |cos| = %.4f   mean 1/sqrt(k_excl) = %.4f   "
              "ratio = %.2f  [predicted within 2x]" % (cm, ik, ratio))
        print("      within a factor of 2: %s" % (0.5 <= ratio <= 2.0))
    ks = sorted({r["k_excl"] for r in rows})
    print("\n      k_excl across the 19 blocks: %s" % ks)

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_projall_mechanism_L.json"))
    with open(out, "w") as f:
        json.dump(dict(rows=rows, N_grid=N_GRID, seeds=SEEDS, optimal=best),
                  f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
