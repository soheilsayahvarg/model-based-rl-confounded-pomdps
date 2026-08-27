"""run_horizon_ceiling.py -- where does the anchor width rule stop being defined?

Predictions: docs/scale_and_baselines.md P-L10 and P-L11, committed at 2474443.

The tau = 0 width rule is xi = c / (N_2 * sigma2), with sigma2 the smallest kept
eigenvalue of the per-action design W_a = M_a M_a^T / N_2. At Environment L,
sigma2 is EXACTLY zero from t = 9 on, so the rule is undefined, not merely large.

The cause is the cross-fitting split, not estimator capacity: mu_hat is built on
the stage-1 half and evaluated at stage-2 histories, and at late t essentially no
stage-2 history was ever seen in the stage-1 half. The estimator already records
unseen_x2_frac per block and nothing has ever read it.

t* is the first stage with sigma2 = 0. If t* grows only logarithmically in N,
there is a horizon ceiling no feasible sample size lifts, and that is a property
of the construction rather than of our grid.

Writes experiments/results_horizon_ceiling.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params, sample_trajectories
from bridge_estimator_scaled import ScaledBridgeEstimator

N_S, N_O, N_O0, N_A, T = 8, 10, 3, 2, 10
CONFOUND = 0.9
KAP = 1.5
SPLIT = 0.7
N_GRID = [int(x) for x in
          os.environ.get("NGRID", "4000,16000,64000,256000").split(",")]
SEEDS = list(range(int(os.environ.get("SEEDS", "2"))))
ZERO = 0.0          # sigma2 == 0 exactly; no tolerance, that is the point


def main():
    p = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T, seed=0,
                       confound=CONFOUND)
    print("Environment L, |S|=%d |O|=%d |O0|=%d T=%d, %d seeds"
          % (N_S, N_O, N_O0, T, len(SEEDS)))
    print("%9s%6s%8s%14s%14s%16s%12s" %
          ("N", "seed", "block", "sigma2", "unseen_x2", "n_x/N", "xi(c=1)"))
    rows, tstars = [], {}
    for N in N_GRID:
        ts = []
        for sd in SEEDS:
            d = sample_trajectories(p, N, np.random.default_rng(sd))
            N2 = N - int(N * SPLIT)
            est = ScaledBridgeEstimator(n_obs=N_O, n_act=N_A, n_o0=N_O0, n_r=2,
                                        T=T, mode="primal", seed=sd,
                                        lambda2=N2 ** (-KAP))
            est.fit(d["O0"], d["O"], d["A"], d["R"])
            diag = {x["label"]: x for x in est.diagnostics}
            tstar = None
            for t in range(1, T + 1):
                lab = "bR_t%d" % t
                s2 = float(diag[lab]["sigma2_signal"])
                uf = float(diag[lab]["unseen_x2_frac"])
                nx = est.n_x_compact[t] / N
                xi = 1.0 / (N2 * max(s2, 1e-12))
                if s2 <= ZERO and tstar is None:
                    tstar = t
                if sd == 0:
                    print("%9d%6d%8d%14.3e%14.4f%16.4f%12.3e"
                          % (N, sd, t, s2, uf, nx, xi))
                rows.append(dict(N=N, seed=sd, block=t, sigma2=s2,
                                 unseen_x2_frac=uf, n_x_over_N=nx,
                                 xi_unit=xi, is_zero=bool(s2 <= ZERO)))
            ts.append(tstar if tstar is not None else T + 1)
        tstars[N] = float(np.mean(ts))
        print("%9d   t* = %.1f   (T+1 means the rule stayed defined)\n"
              % (N, tstars[N]))

    print("=" * 84)
    z = [r for r in rows if r["is_zero"]]
    nz = [r for r in rows if not r["is_zero"]]
    print("P-L10  sigma2 == 0 in %d of %d block-cells" % (len(z), len(rows)))
    if z:
        print("       among those, unseen_x2_frac: min %.4f  mean %.4f  max %.4f"
              % (min(r["unseen_x2_frac"] for r in z),
                 float(np.mean([r["unseen_x2_frac"] for r in z])),
                 max(r["unseen_x2_frac"] for r in z)))
    if nz:
        print("       among sigma2 > 0:            min %.4f  mean %.4f  max %.4f"
              % (min(r["unseen_x2_frac"] for r in nz),
                 float(np.mean([r["unseen_x2_frac"] for r in nz])),
                 max(r["unseen_x2_frac"] for r in nz)))

    Ns = sorted(tstars)
    print("\nP-L11  t* by N: %s" % {n: tstars[n] for n in Ns})
    if len(Ns) >= 2:
        shift = tstars[Ns[-1]] - tstars[Ns[0]]
        fold = Ns[-1] / Ns[0]
        print("       %gx more data moved t* by %+.1f stages  [predicted <= 2]"
              % (fold, shift))
        sl = float(np.polyfit(np.log(Ns), [tstars[n] for n in Ns], 1)[0])
        print("       t* vs log N slope %+.3f stages per e-fold" % sl)

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_horizon_ceiling.json"))
    with open(out, "w") as f:
        json.dump(dict(rows=rows, tstar=tstars, N_grid=N_GRID, seeds=SEEDS,
                       config=[N_S, N_O, N_O0], T=T), f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
