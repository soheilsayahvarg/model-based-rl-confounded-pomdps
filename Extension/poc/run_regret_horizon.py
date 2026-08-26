"""run_regret_incomplete.py -- the decision-level test, where the floor is nonzero.

Predictions: docs/regret_vs_schedule.md 4.5b, committed before this file existed.

WHY HERE AND NOT ON THE TOY. run_regret_vs_schedule.py found that the width
schedule does not change the decision on the toy, because beta_g = 0 exactly
there: the value gradient has no mass in the design null, so the N^(e/2) law has
nothing to act through. run_gradient_leakage_map.py then showed beta_g is switched
by CONFOUNDING while beta is switched by INCOMPLETENESS, so the floor
W >= beta*beta_g needs both. (4,6,2) at confound = 0.9 has beta = 1.178,
beta_g = 0.364, floor = 0.428, and a NON-degenerate behaviour policy.

The precondition is checked in-run before the sweep, and the sweep aborts if it
fails. The two previous attempts at this test each measured on a grid that could
not have shown the effect; that is the defect this guard exists to prevent.

Writes experiments/results_regret_incomplete.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params, sample_trajectories, dp_value
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid, pessimistic_value
from run_completeness_beta import population_design_span, beta_pop_for

N_S, N_O, N_O0, N_A = 4, 6, 2, 2
T = int(os.environ.get("HORIZON", "3"))
CONFOUND = 0.9
N_GRID = [2000, 8000, 32000, 128000]
# 3 seeds gave a real but small effect that docs/regret_vs_schedule.md 6.3 refused
# to quote. Default raised to 20; override with argv[1].
SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
ONLY = [t for t in os.environ.get("SCHEDULES", "").split(",") if t]
SPLIT = 0.7
W_TARGET = 0.12
RESULTS = {}

ALPHA_P, C2_P = 10.0, 1.0
SCHEDULES = [
    ("A  kappa=0.5", 0.0, 0.5),
    ("B  kappa=1.0", 0.0, 1.0),
    ("C  kappa=1.5", 0.0, 1.5),
    ("paper a=10,c2=1", (ALPHA_P + 2) / (2 * ALPHA_P + 2),
     ALPHA_P / (ALPHA_P * C2_P + 1)),
]
C_CAL = {}


def candidates():
    """Six observation-only policies, matching the toy sweep's structure so the
    two runs are comparable."""
    lo = np.linspace(0.9, 0.1, N_O)
    return {
        "greedy_lo": np.stack([lo, 1 - lo], 1),
        "greedy_hi": np.stack([1 - lo, lo], 1),
        "always_0":  np.tile([1.0, 0.0], (N_O, 1)),
        "always_1":  np.tile([0.0, 1.0], (N_O, 1)),
        "uniform":   np.full((N_O, 2), 0.5),
        "soft":      np.stack([0.5 + 0.2 * (lo - 0.5) / 0.4,
                               0.5 - 0.2 * (lo - 0.5) / 0.4], 1),
    }


def build_blocks(est):
    bR, bD = [], []
    for t in range(1, T + 1):
        s = est.stage_store[f"bR_t{t}"]
        bR.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                 s["N2"], f"bR_t{t}", s["signal_basis"],
                                 n_obs=N_O, n_act=N_A, n_y=s["n_y"]))
        if t < T:
            s = est.stage_store[f"bD_t{t}"]
            bD.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                     s["N2"], f"bD_t{t}", s["signal_basis"],
                                     n_obs=N_O, n_act=N_A, n_y=s["n_y"]))
    return bR, bD


def xi_for(tau, N2, sigma2, c_mult):
    m = sigma2 if tau == 0.0 else 1.0
    return c_mult * N2 ** (tau - 1.0) / max(m, 1e-12)


def fit_once(p, N, sd, kap):
    N2 = N - int(N * SPLIT)
    d = sample_trajectories(p, N, np.random.default_rng(sd))
    est = TabularBridgeEstimator(n_obs=N_O, n_act=N_A, n_o0=N_O0, n_r=2, T=T,
                                 mode="primal", seed=sd, lambda2=N2 ** (-kap))
    est.fit(d["O0"], d["O"], d["A"], d["R"])
    return est, d


def make_vg(bR_blocks, bD_blocks, pi, p_o1):
    def vg(bRl, bDl):
        BR = np.stack([b.reshape(N_A, N_O, 2, N_O) for b in bRl])
        BD = np.stack([b.reshape(N_A, N_O, N_O, N_O) for b in bDl])
        V, _p, (gR, gD) = plugin_value(BR, BD, pi, p_o1, return_grads=True)
        return (V, [gR[t].ravel() for t in range(T)],
                [gD[j].ravel() for j in range(T - 1)])
    return vg


def precondition(p):
    """beta_g > 0 must hold or the sweep has no mechanism to detect. Abort if not."""
    nu1 = p["p1"] @ p["E"]
    bg = max(float(np.linalg.norm(nu1 - population_design_span(p, a) @
                                  (population_design_span(p, a).T @ nu1)))
             / float(np.linalg.norm(nu1)) for a in range(N_A))
    b = max(beta_pop_for(p, a)[0] for a in range(N_A))
    print(f"precondition: beta = {b:.4f}, beta_g = {bg:.4f}, "
          f"floor = {b*bg:.4f}")
    if bg < 1e-6:
        raise SystemExit("ABORT: beta_g = 0, this grid cannot show the effect.")
    return b, bg


def calibrate(p, cands, p_o1_cache):
    print(f"\ncalibrating every schedule to width {W_TARGET} at N={N_GRID[0]}")
    for tag, tau, kap in SCHEDULES:
        est, d = fit_once(p, N_GRID[0], 0, kap)
        bR, bD = build_blocks(est)
        p_o1 = empirical_p_o1(d["O"], N_O)
        vg = make_vg(bR, bD, cands["greedy_lo"], p_o1)
        _, g, _ = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
        w1 = np.sqrt(xi_for(tau, bR[0].N2, bR[0].sigma2, 1.0)) * \
            bR[0].h_inv_norm(g[0])
        C_CAL[tag] = (W_TARGET / w1) ** 2
        print(f"   {tag:<18} c = {C_CAL[tag]:.3e}")


def main():
    p = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T,
                       seed=0, confound=CONFOUND)
    print(f"({N_S},{N_O},{N_O0}) confound={CONFOUND}  INCOMPLETE\n")
    beta, beta_g = precondition(p)

    cands = candidates()
    v_true = {k: dp_value(p, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    print(f"\noptimal {best}  V_true = {v_true[best]:.4f}")
    print("   regret levels:", {k: round(v_true[best] - v, 4)
                                for k, v in sorted(v_true.items(),
                                                   key=lambda kv: -kv[1])})
    spread = float(max(v_true.values()) - min(v_true.values()))
    print(f"   VALUE SPREAD (T={T}) = {spread:.4f}   "
          f"-- regret must be normalised by this to compare across horizons")
    RESULTS["T"] = T
    RESULTS["spread"] = spread
    calibrate(p, cands, None)

    rows = []
    for tag, tau, kap in SCHEDULES:
        if ONLY and not any(o in tag for o in ONLY):
            continue
        print(f"\n{'='*76}\n{tag}   e = {tau+kap-1:+.3f}   "
              f"predicted width slope {(tau+kap-1)/2:+.3f}\n{'='*76}")
        print(f"{'N':>8}{'pess regret':>13}{'+-95%':>9}{'plug-in':>10}{'+-95%':>9}"
              f"{'modal pick':>13}{'width':>9}")
        for N in N_GRID:
            picks, plug_picks, widths = [], [], []
            for sd in SEEDS:
                est, d = fit_once(p, N, sd, kap)
                bR, bD = build_blocks(est)
                p_o1 = empirical_p_o1(d["O"], N_O)
                cm = C_CAL[tag]
                xR = [xi_for(tau, b.N2, b.sigma2, cm) for b in bR]
                xD = [xi_for(tau, b.N2, b.sigma2, cm) for b in bD]
                bh = [b.b_hat for b in bR]
                dh = [b.b_hat for b in bD]
                vals, pvals = {}, {}
                for name, pi in cands.items():
                    vg = make_vg(bR, bD, pi, p_o1)
                    # plug-in and pessimistic share the SAME fit, so the only
                    # difference between the two columns is the pessimism layer
                    pvals[name] = vg(bh, dh)[0]
                    vals[name] = pessimistic_value(
                        bR, bD, xR, xD, vg,
                        rng=np.random.default_rng(sd))["V_low"]
                picks.append(max(vals, key=vals.get))
                plug_picks.append(max(pvals, key=pvals.get))
                vg0 = make_vg(bR, bD, cands[best], p_o1)
                widths.append(np.sqrt(xR[0]) *
                              bR[0].h_inv_norm(vg0(bh, dh)[1][0]))

            def stat(ps):
                r = np.array([v_true[best] - v_true[q] for q in ps])
                return float(r.mean()), float(1.96 * r.std(ddof=1) /
                                              np.sqrt(len(r)))
            reg, ci = stat(picks)
            preg, pci = stat(plug_picks)
            modal = max(set(picks), key=picks.count)
            print(f"{N:>8}{reg:>13.4f}{ci:>9.4f}{preg:>10.4f}{pci:>9.4f}"
                  f"{modal:>13}{np.mean(widths):>9.4f}")
            rows.append(dict(schedule=tag, tau=tau, kappa=kap, e=tau + kap - 1,
                             N=N, regret=reg, regret_ci=ci,
                             plugin_regret=preg, plugin_ci=pci, modal=modal,
                             width=float(np.mean(widths)), picks=picks,
                             plug_picks=plug_picks, n_seeds=len(SEEDS)))

    print(f"\n{'='*76}\nTREND\n{'='*76}")
    print(f"{'schedule':>18}{'e':>8}{'pess slope':>12}{'plugin slope':>14}"
          f"{'width slope':>13}{'pred e/2':>10}")
    summary = []
    for tag, tau, kap in SCHEDULES:
        rs = [r for r in rows if r["schedule"] == tag]
        if not rs:
            continue
        ns = np.log([r["N"] for r in rs])
        gr = float(np.polyfit(ns, [r["regret"] for r in rs], 1)[0])
        gp = float(np.polyfit(ns, [r["plugin_regret"] for r in rs], 1)[0])
        gw = float(np.polyfit(ns, np.log([r["width"] for r in rs]), 1)[0])
        e = tau + kap - 1
        print(f"{tag:>18}{e:>8.3f}{gr:>12.5f}{gp:>14.5f}{gw:>13.4f}"
              f"{e/2:>10.3f}")
        summary.append(dict(schedule=tag, e=e, regret_slope=gr,
                            plugin_slope=gp, width_slope=gw,
                            pred_e2=e / 2, pred_signal=(tau - 1) / 2))

    RESULTS.update(rows=rows, summary=summary, v_true=v_true, optimal=best,
                   beta=beta, beta_g=beta_g, floor=beta * beta_g,
                   confound=CONFOUND, config=[N_S, N_O, N_O0])
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        f"results_regret_horizon_T{T}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
