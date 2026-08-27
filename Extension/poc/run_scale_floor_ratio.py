"""run_scale_floor_ratio.py -- does the identification gap exceed the value range?

Predictions: docs/scale_and_baselines.md P-L6 and P-L7, committed at f709bba.

Population algebra only, no fitting. beta and beta_g are stage-1 quantities built
from p1, K0, E and pi_b, none of which depend on T, so the floor is constant in T
by construction. cor:dilution says the value spread grows with T, so the ratio
floor/spread should DECAY as 1/T. P-L7 says it grows. They cannot both hold.

The ratio is the quantity that matters for the method: a valid region cannot be
narrower than the gap, so once floor > spread no valid pessimistic rule can order
the candidates at all, at any sample size.

Writes experiments/results_scale_floor_ratio.json.
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
from run_completeness_beta import population_design_span, beta_pop_for
import run_stage_selective as D

CONFIGS = [(8, 10, 3), (4, 6, 2), (6, 10, 3), (8, 10, 4)]
T_GRID = [3, 4, 6, 8, 10]
CONFOUND = 0.9


def floor_and_spread(cfg, T, cf=CONFOUND):
    n_s, n_o, n_o0 = cfg
    p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=T, seed=0,
                       confound=cf)
    D.N_O, D.N_A = n_o, 2
    cands = D.candidates()
    v = {k: dp_value(p, pi) for k, pi in cands.items()}
    spread = max(v.values()) - min(v.values())
    nu1 = p["p1"] @ p["E"]
    bg = max(float(np.linalg.norm(
        nu1 - population_design_span(p, a) @
        (population_design_span(p, a).T @ nu1))) / float(np.linalg.norm(nu1))
        for a in range(2))
    b = max(beta_pop_for(p, a)[0] for a in range(2))
    return b * bg, spread, b, bg, max(v, key=v.get)


def main():
    print("%14s%5s%12s%12s%10s%10s%14s" %
          ("config", "T", "floor", "spread", "ratio", "separable", "best"))
    rows = []
    for cfg in CONFIGS:
        for T in T_GRID:
            fl, sp, b, bg, best = floor_and_spread(cfg, T)
            r = fl / sp if sp > 0 else float("inf")
            print("%14s%5d%12.4f%12.4f%10.3f%10s%14s"
                  % (str(cfg), T, fl, sp, r, "yes" if r < 1 else "NO", best))
            rows.append(dict(config=list(cfg), T=T, floor=fl, spread=sp,
                             ratio=r, beta=b, beta_g=bg, best=best,
                             separable=bool(r < 1)))
        print()

    print("=" * 82)
    for cfg in CONFIGS:
        sub = [r for r in rows if r["config"] == list(cfg)]
        fl = {round(r["floor"], 10) for r in sub}
        Ts = np.log([r["T"] for r in sub])
        sl_sp = float(np.polyfit(Ts, np.log([r["spread"] for r in sub]), 1)[0])
        sl_r = float(np.polyfit(Ts, np.log([r["ratio"] for r in sub]), 1)[0])
        print("%14s  floor constant in T: %-5s  spread log-log slope %+.3f  "
              "ratio slope %+.3f  separable at: %s"
              % (str(cfg), len(fl) == 1, sl_sp, sl_r,
                 [r["T"] for r in sub if r["separable"]] or "no T"))
    print("\nP-L6  cells with floor > spread: %d of %d"
          % (sum(1 for r in rows if not r["separable"]), len(rows)))
    print("P-L7  ratio grows with T in %d of %d configs"
          % (sum(1 for cfg in CONFIGS
                 if float(np.polyfit(
                     np.log([r["T"] for r in rows if r["config"] == list(cfg)]),
                     np.log([r["ratio"] for r in rows
                             if r["config"] == list(cfg)]), 1)[0]) > 0),
             len(CONFIGS)))

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_scale_floor_ratio.json"))
    with open(out, "w") as f:
        json.dump(dict(rows=rows, T_grid=T_GRID, confound=CONFOUND),
                  f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
