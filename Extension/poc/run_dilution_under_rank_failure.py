"""run_dilution_under_rank_failure.py -- does the 1/T law survive?

Predictions: docs/selfheal_theorem.md section 8 (P-D1..P-D4), committed at
96b61e0 before this file existed.

WHY. Section 6 refuted cor:selfheal as published: with rank-deficient transitions
every stage leaks, not only the first. cor:dilution is built directly on top of
that. run_horizon_dilution.py part 3 computes

    floor1 = max over actions of the STAGE-1 floor
    ratio  = floor1 / value spread at horizon T

and takes floor1 as the entire floor because exactly one block leaks. Remove that
premise and the right quantity is the aggregate over all 2T-1 blocks, which has
no reason to be constant in T.

claims.md currently ASSERTS that cor:dilution inherits the hypothesis. This
measures it.

P-D4 IS THE CONTROL AND SHOULD BE READ FIRST. In the dense family the aggregate
floor must reproduce the published stage-1-only ratios to within 1%, because the
t >= 2 per-stage floors there are around 1e-30. If it does not, the aggregate is
not the right generalization and the other three predictions mean nothing.

Population algebra throughout. No sampling.

Writes experiments/results_dilution_rank.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from run_completeness_beta import true_bridges_generic
from run_horizon_dilution import (stage_span, block_leak, leak_of, dp_values,
                                  candidates)
from run_selfheal_formula import make_params

CONFIGS = [(4, 6, 2), (4, 8, 3)]
CONFOUNDS = [0.6, 0.9]
FAMILIES = ["dense", "dense_lowrank", "sparse_support"]
T_GRID = [1, 2, 3, 4, 5, 6]
RESULTS = {}


def per_stage_floor(p, t):
    """max over actions of beta_t * beta_g,t at stage t, exactly as
    run_horizon_dilution.py part 2 computes it."""
    bR, _ = true_bridges_generic(p)
    nu1 = p["p1"] @ p["E"]
    best = (0.0, 0.0, 0.0, 0)
    for a in range(p["P"].shape[0]):
        U = stage_span(p, t, a)
        b = block_leak(bR[a], U)
        g = leak_of(nu1, U)
        if b * g >= best[2]:
            best = (b, g, b * g, U.shape[1])
    return best


def main():
    print("per-stage leakage by family   (population algebra)")
    print("%16s%10s%6s%4s%6s%12s%12s%12s" %
          ("family", "config", "cf", "t", "span", "beta_t", "beta_g,t",
           "floor_t"))
    stages = []
    for family in FAMILIES:
        for cfg in CONFIGS:
            for cf in CONFOUNDS:
                p = make_params(cfg, cf, family, T=3)
                for t in T_GRID:
                    b, g, fl, r = per_stage_floor(p, t)
                    print("%16s%10s%6.1f%4d%6d%12.3e%12.3e%12.3e" %
                          (family, str(cfg), cf, t, r, b, g, fl))
                    stages.append(dict(family=family, config=list(cfg),
                                       confound=cf, t=t, span=r, beta=b,
                                       beta_g=g, floor=fl, n_s=cfg[0]))

    print("\nP-D1  beta_t at t >= 2, by family")
    for family in FAMILIES:
        lat = [r["beta"] for r in stages if r["family"] == family
               and r["t"] >= 2]
        print("   %-16s min %.3e   max %.3e   [dense predicted <= 2.7e-15, "
              "rank-failure predicted > 0.1]" % (family, min(lat), max(lat)))

    print("\n" + "=" * 100)
    print("DILUTION  stage-1-only floor (as published) vs aggregate over all "
          "2T-1 blocks")
    print("=" * 100)
    print("%16s%10s%6s%4s%11s%12s%11s%12s%12s" %
          ("family", "config", "cf", "T", "floor_t1", "floor_aggr",
           "spread", "ratio_t1", "ratio_aggr"))
    rows = []
    for family in FAMILIES:
        for cfg in CONFIGS:
            n_o = cfg[1]
            for cf in CONFOUNDS:
                sub = {r["t"]: r for r in stages if r["family"] == family
                       and r["config"] == list(cfg) and r["confound"] == cf}
                f1 = sub[1]["floor"]
                pols = candidates(n_o, 2)
                for T in T_GRID:
                    p = make_params(cfg, cf, family, T=T)
                    v = dp_values(p, pols)
                    spread = float(v.max() - v.min())
                    # 2T-1 blocks: reward at t=1..T, dynamics at t=1..T-1. Both
                    # families of block share a stage's conditioning set, so the
                    # per-stage floor is used for both.
                    aggr = sum(sub[t]["floor"] for t in range(1, T + 1)) + \
                        sum(sub[t]["floor"] for t in range(1, T))
                    r1 = f1 / spread if spread > 0 else float("inf")
                    ra = aggr / spread if spread > 0 else float("inf")
                    print("%16s%10s%6.1f%4d%11.4f%12.4f%11.4f%12.4f%12.4f" %
                          (family, str(cfg), cf, T, f1, aggr, spread, r1, ra))
                    rows.append(dict(family=family, config=list(cfg),
                                     confound=cf, T=T, floor_t1=f1,
                                     floor_aggr=aggr, spread=spread,
                                     ratio_t1=r1, ratio_aggr=ra))

    def sl(xs, ys):
        y = np.maximum(np.asarray(ys, float), 1e-300)
        return float(np.polyfit(np.log(xs), np.log(y), 1)[0])

    print("\n%16s%10s%6s%14s%14s%14s%14s" %
          ("family", "config", "cf", "slope f_aggr", "slope r_t1",
           "slope r_aggr", "P-D4 max rel"))
    summ, worst_ctrl = [], 0.0
    for family in FAMILIES:
        for cfg in CONFIGS:
            for cf in CONFOUNDS:
                sub = sorted([r for r in rows if r["family"] == family
                              and r["config"] == list(cfg)
                              and r["confound"] == cf],
                             key=lambda r: r["T"])
                Ts = [r["T"] for r in sub]
                s_f = sl(Ts, [r["floor_aggr"] for r in sub])
                s_1 = sl(Ts, [r["ratio_t1"] for r in sub])
                s_a = sl(Ts, [r["ratio_aggr"] for r in sub])
                rel = max(abs(r["ratio_aggr"] - r["ratio_t1"]) /
                          max(r["ratio_t1"], 1e-300) for r in sub)
                if family == "dense":
                    worst_ctrl = max(worst_ctrl, rel)
                print("%16s%10s%6.1f%14.4f%14.4f%14.4f%14.2e" %
                      (family, str(cfg), cf, s_f, s_1, s_a, rel))
                summ.append(dict(family=family, config=list(cfg), confound=cf,
                                 slope_floor_aggr=s_f, slope_ratio_t1=s_1,
                                 slope_ratio_aggr=s_a, control_rel=float(rel)))

    print("\nP-D4 (control, read first)  dense: aggregate vs stage-1-only "
          "ratios differ by at most %.2e   [predicted < 1%%]" % worst_ctrl)
    for tag, pred in (("dense", "about 0"), ("dense_lowrank", "about +1"),
                      ("sparse_support", "about +1")):
        s = [r["slope_floor_aggr"] for r in summ if r["family"] == tag]
        print("P-D2  %-16s aggregate-floor slope in T: %s   [predicted %s]"
              % (tag, ["%+.3f" % x for x in s], pred))
    for tag, pred in (("dense", "about -1"), ("dense_lowrank", "about 0"),
                      ("sparse_support", "about 0")):
        s = [r["slope_ratio_aggr"] for r in summ if r["family"] == tag]
        print("P-D3  %-16s ratio slope in T:            %s   [predicted %s]"
              % (tag, ["%+.3f" % x for x in s], pred))

    RESULTS.update(stages=stages, rows=rows, summary=summ,
                   control_worst_rel=float(worst_ctrl),
                   families=FAMILIES, T_grid=T_GRID)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_dilution_rank.json"))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
