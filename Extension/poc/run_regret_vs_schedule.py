"""run_regret_vs_schedule.py -- does the width schedule change the DECISION?

Predictions are in docs/regret_vs_schedule.md sections 1-3, committed before this
file existed. Nothing here is exploratory.

THE GAP. Step (c) showed the pessimism width scales as N^(e/2), e = tau+kappa-1,
and that the anchor paper's own schedule sits at e_paper > 0. That is a statement
about the width of an interval. Whether it changes which policy is selected has
never been measured. If it does not, the branch is practically empty and must be
reported as such.

PREDICTED, per schedule:
    e < 0  ->  penalty vanishes, selection converges to the plug-in's, regret -> 0
    e = 0  ->  penalty is N-independent, regret flat
    e > 0  ->  penalty swamps value differences, regret non-decreasing in N

CONTROL. Regret is discrete over 6 candidates and can look flat from coarseness
alone, so the selected policy and V_low are reported alongside: a flat regret
curve that is really moving shows up as a moving V_low gap.

Writes experiments/results_regret_vs_schedule.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid, pessimistic_value

N_GRID = [2000, 8000, 32000, 128000]
SEEDS = [0, 1, 2, 3, 4]
SPLIT = 0.7
RESULTS = {}

# CALIBRATION (added after run 1 -- see docs/regret_vs_schedule.md 4.1).
# Run 1 used one shared constant, which gave the four schedules starting widths
# of 0.55/1.05/1.37/0.67 and different m normalizations (sigma2 vs 1). Three of
# them began already inside the pessimism attractor, so "does regret degrade with
# N" could not be observed: the selection was pinned before the sweep started.
# Each schedule is now calibrated so its width at the SMALLEST N equals a common
# target below the decision boundary, making the sweep a test of SLOPE alone.
W_TARGET = 0.12         # below the ~0.19 at which schedule A recovers greedy_lo
C_CAL = {}              # filled by calibrate()

# (tag, tau, kappa) -- signal convention has tau = 0; the paper row carries its own
ALPHA_P, C2_P = 10.0, 1.0
TAU_P = (ALPHA_P + 2) / (2 * ALPHA_P + 2)
KAPPA_P = ALPHA_P / (ALPHA_P * C2_P + 1)
SCHEDULES = [
    ("A  kappa=0.5", 0.0, 0.5),
    ("B  kappa=1.0", 0.0, 1.0),
    ("C  kappa=1.5", 0.0, 1.5),
    ("paper a=10,c2=1", TAU_P, KAPPA_P),
]


def dp_value_toy(params, pi_obs, T=None):
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    T = T or params["T"]
    V = np.zeros(toy.N_S)
    for _ in range(T):
        Q = pR + np.einsum("asz,z->sa", P, V)
        V = np.einsum("sa,sa->s", E @ pi_obs, Q)
    return float(p1 @ V)


def candidates():
    return {
        "greedy_lo": np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        "greedy_hi": np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]),
        "always_0":  np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        "always_1":  np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        "uniform":   np.full((3, 2), 0.5),
        "soft":      np.array([[0.7, 0.3], [0.5, 0.5], [0.3, 0.7]]),
    }


def build_blocks(est, T):
    bR, bD = [], []
    for t in range(1, T + 1):
        s = est.stage_store[f"bR_t{t}"]
        bR.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                 s["N2"], f"bR_t{t}", s["signal_basis"],
                                 n_obs=toy.N_O, n_act=toy.N_A, n_y=s["n_y"]))
        if t < T:
            s = est.stage_store[f"bD_t{t}"]
            bD.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                     s["N2"], f"bD_t{t}", s["signal_basis"],
                                     n_obs=toy.N_O, n_act=toy.N_A, n_y=s["n_y"]))
    return bR, bD


def xi_for(tau, N2, sigma2, c_mult):
    """xi = c * N2^(tau-1) / m.  tau = 0 with m = sigma2 reproduces the repo's
    signal-convention width_rule exactly; the paper row uses its own tau."""
    m = sigma2 if tau == 0.0 else 1.0
    return c_mult * N2 ** (tau - 1.0) / max(m, 1e-12)


def fit_once(params, N, sd, kap, T):
    N2 = N - int(N * SPLIT)
    d = toy.sample_trajectories(params, N, np.random.default_rng(sd))
    est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A, n_o0=toy.N_O0,
                                 n_r=2, T=T, mode="primal", seed=sd,
                                 lambda2=N2 ** (-kap))
    est.fit(d["O0"], d["O"], d["A"], d["R"])
    return est, d


def calibrate(params, T, cands):
    """Set each schedule's constant so its width at the smallest N matches
    W_TARGET. W ∝ sqrt(c_mult), so one probe fit per schedule suffices."""
    print(f"calibrating every schedule to width {W_TARGET} at N={N_GRID[0]}")
    for tag, tau, kap in SCHEDULES:
        est, d = fit_once(params, N_GRID[0], 0, kap, T)
        bR, bD = build_blocks(est, T)
        p_o1 = empirical_p_o1(d["O"], toy.N_O)
        pi = cands["greedy_lo"]
        BR = np.stack([b.b_hat.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                       for b in bR])
        BD = np.stack([b.b_hat.reshape(toy.N_A, toy.N_O, toy.N_O, toy.N_O)
                       for b in bD])
        _, _, (gR, _) = plugin_value(BR, BD, pi, p_o1, return_grads=True)
        w1 = np.sqrt(xi_for(tau, bR[0].N2, bR[0].sigma2, 1.0)) * \
            bR[0].h_inv_norm(gR[0].ravel())
        C_CAL[tag] = (W_TARGET / w1) ** 2
        print(f"   {tag:<18} c = {C_CAL[tag]:.3e}   (width at c=1 was {w1:.3f})")


def main():
    params = toy.default_params(T=3)
    T = params["T"]
    cands = candidates()
    v_true = {k: dp_value_toy(params, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    print(f"optimal {best}  V_true = {v_true[best]:.4f}\n")
    print("schedules (e = tau + kappa - 1):")
    for tag, tau, kap in SCHEDULES:
        print(f"   {tag:<18} tau={tau:.3f} kappa={kap:.3f}  e={tau+kap-1:+.3f}"
              f"   width ~ N^{(tau+kap-1)/2:+.3f}")
    print()
    calibrate(params, T, cands)

    rows = []
    for tag, tau, kap in SCHEDULES:
        print(f"\n{'='*74}\n{tag}   e = {tau+kap-1:+.3f}\n{'='*74}")
        print(f"{'N':>8}{'regret':>10}{'modal pick':>13}{'V_low(pick)':>13}"
              f"{'width':>10}")
        for N in N_GRID:
            picks, vlows, widths = [], [], []
            for sd in SEEDS:
                est, d = fit_once(params, N, sd, kap, T)
                bR, bD = build_blocks(est, T)
                p_o1 = empirical_p_o1(d["O"], toy.N_O)
                cm = C_CAL[tag]
                xR = [xi_for(tau, b.N2, b.sigma2, cm) for b in bR]
                xD = [xi_for(tau, b.N2, b.sigma2, cm) for b in bD]

                vals = {}
                for name, pi in cands.items():
                    def vg(bRl, bDl):
                        BR = np.stack([b.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                                       for b in bRl])
                        BD = np.stack([b.reshape(toy.N_A, toy.N_O, toy.N_O,
                                                 toy.N_O) for b in bDl])
                        V, _p, (gR, gD) = plugin_value(BR, BD, pi, p_o1,
                                                       return_grads=True)
                        return (V, [gR[t].ravel() for t in range(T)],
                                [gD[j].ravel() for j in range(T - 1)])
                    vals[name] = pessimistic_value(
                        bR, bD, xR, xD, vg,
                        rng=np.random.default_rng(sd))["V_low"]
                pick = max(vals, key=vals.get)
                picks.append(pick)
                vlows.append(vals[pick])
                # width of the t=1 reward block at the plug-in gradient, as a
                # scale readout independent of which policy won
                _, g1, _ = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
                widths.append(np.sqrt(xR[0]) * bR[0].h_inv_norm(g1[0]))

            reg = float(np.mean([v_true[best] - v_true[p] for p in picks]))
            modal = max(set(picks), key=picks.count)
            print(f"{N:>8}{reg:>10.4f}{modal:>13}{np.mean(vlows):>13.3f}"
                  f"{np.mean(widths):>10.3f}")
            rows.append(dict(schedule=tag, tau=tau, kappa=kap,
                             e=tau + kap - 1, N=N, regret=reg, modal=modal,
                             vlow=float(np.mean(vlows)),
                             width=float(np.mean(widths)),
                             picks=picks))

    print(f"\n{'='*74}\nTREND: d(regret)/d(log N), and the width slope as a check")
    print(f"{'='*74}")
    print(f"{'schedule':>18}{'e':>8}{'regret slope':>15}{'predicted':>14}"
          f"{'width slope':>14}{'pred e/2':>10}")
    summary = []
    for tag, tau, kap in SCHEDULES:
        rs = [r for r in rows if r["schedule"] == tag]
        ns = np.log([r["N"] for r in rs])
        gr = float(np.polyfit(ns, [r["regret"] for r in rs], 1)[0])
        gw = float(np.polyfit(ns, np.log([r["width"] for r in rs]), 1)[0])
        e = tau + kap - 1
        pred = "decreasing" if e < -1e-9 else ("flat" if abs(e) < 1e-9
                                               else "increasing")
        print(f"{tag:>18}{e:>8.3f}{gr:>15.5f}{pred:>14}{gw:>14.4f}{e/2:>10.3f}")
        summary.append(dict(schedule=tag, e=e, regret_slope=gr,
                            predicted=pred, width_slope=gw, width_pred=e / 2))

    RESULTS["rows"] = rows
    RESULTS["summary"] = summary
    RESULTS["v_true"] = v_true
    RESULTS["optimal"] = best
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_regret_vs_schedule.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
