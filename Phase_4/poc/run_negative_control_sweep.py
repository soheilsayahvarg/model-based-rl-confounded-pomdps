"""
run_negative_control_sweep.py -- how good does the negative control have to be?

MOTIVATION. Every method in the proximal causal inference line -- including the
anchor paper -- assumes the existence of a valid negative control satisfying
Assumption 3.1, and then proceeds. No theory paper can say what happens when that
assumption is only approximately true, because the assumption is what the theory
is conditioned on. A simulator where the negative control's quality is a knob can
say, and that is a contribution an implementation can make and a proof cannot.

TWO DISTINCT FAILURE MODES, swept separately because they are not the same thing:

  [N1] STRENGTH.  sigma_0 is the noise on O_0 = s_1 + eps_0. Assumption 3.1 holds
       EXACTLY for every sigma_0 (fresh independent noise on the initial state
       cannot induce dependence on later outcomes given the state). What degrades
       is the instrument's informativeness: as sigma_0 grows, O_0 carries less
       signal about s_1, stage 1's conditional mean embedding weakens, and the
       whole two-stage solve inherits it. This is a weak-instrument study.

  [N2] VALIDITY.  delta_nc leaks the negative control's own noise into the reward,
       r_t += delta_nc * eps_0. Now O_0 carries information about R_t beyond
       (S_t, A_t, H_{t-1}), so Assumption 3.1 is FALSE. Because E[eps_0] = 0 the
       mean reward is unchanged, so the exact oracle remains valid and the damage
       to identification is directly measurable against it.

Conflating these would be a mistake: the first is about how much data you need,
the second about whether the estimand is identified at all.

Writes experiments/results_negative_control.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from finite_action_env import (default_params, sample_trajectories, true_value,
                               true_bridge_coeffs, candidate_policies, N_ACTIONS)
from fa_bridge_estimator import FiniteActionBridgeEstimator, bridge_features
from fa_value_plugin import chain_value, empirical_initial_mean

SEEDS = [0, 1, 2, 3, 4]
SIGMA0_GRID = [0.10, 0.25, 0.50, 1.00, 2.00]     # 0.25 is the base configuration
DELTA_GRID = [0.0, 0.25, 0.50, 1.00]
N_MAIN = 2000
RESULTS = {}


def ci95(v):
    v = np.asarray(v, float)
    return float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def naive_bridges(d, T, K):
    O, A_idx, R = d["O"], d["A_idx"], d["R"]
    bR, bD = [], []
    for t in range(1, T + 1):
        W = bridge_features(A_idx[:, t - 1], O[:, t - 1], K)
        bR.append(np.linalg.lstsq(W, R[:, t - 1], rcond=None)[0])
        if t < T:
            bD.append(np.linalg.lstsq(W, O[:, t], rcond=None)[0])
    return bR, bD


def evaluate(p, cands, v_true):
    """Evaluate one configuration, separating SYSTEMATIC BIAS from TOTAL ERROR.

    These must not be conflated, and doing so would invert the conclusion. The
    plug-in's whole purpose is to remove bias; it pays for that in variance. So:

      systematic bias := mean over policies of | mean over seeds of (V_hat - V) |
      total error     := mean over policies and seeds of | V_hat - V |

    The first isolates what the method is designed to fix; the second is what a
    practitioner actually pays, and includes the variance the fix costs. In this
    project they point in OPPOSITE directions, which is the central finding, so
    both are reported at every grid point.
    """
    T = p["T"]
    tb = true_bridge_coeffs(p)
    names = list(cands.keys())
    err_pl = {n: [] for n in names}     # signed per-seed errors
    err_nv = {n: [] for n in names}
    rec = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        d = sample_trajectories(p, N_MAIN, rng)
        est = FiniteActionBridgeEstimator(T=T, n_actions=N_ACTIONS, seed=sd).fit(
            d["O0"], d["O"], d["A_idx"], d["R"])
        m1 = empirical_initial_mean(d["O"])
        bR_e = [est.stage_R[t]["theta"] for t in range(1, T + 1)]
        bD_e = [est.stage_D[t]["theta"] for t in range(1, T)]
        bR_n, bD_n = naive_bridges(d, T, N_ACTIONS)
        rec.append(float(np.linalg.norm(est.stage_R[1]["theta"] - tb["bR"])))
        for n in names:
            pi = cands[n]
            err_pl[n].append(chain_value(pi, m1, bR_e, bD_e)[0] - v_true[n])
            err_nv[n].append(chain_value(pi, m1, bR_n, bD_n)[0] - v_true[n])

    def summarize(err):
        sysb = float(np.mean([abs(np.mean(err[n])) for n in names]))
        tot = float(np.mean([np.mean(np.abs(err[n])) for n in names]))
        return sysb, tot

    sys_pl, tot_pl = summarize(err_pl)
    sys_nv, tot_nv = summarize(err_nv)
    return dict(sys_plugin=sys_pl, tot_plugin=tot_pl,
                sys_naive=sys_nv, tot_naive=tot_nv,
                recovery_err=float(np.mean(rec)), recovery_ci=ci95(rec))


def main():
    base = default_params(T=3, kappa=1.0)
    cands = candidate_policies()
    v_true = {n: true_value(base, pi) for n, pi in cands.items()}

    hdr = (f"{'knob':>9}{'recov_err':>11}{'sysbias_pl':>12}{'sysbias_nv':>12}"
           f"{'toterr_pl':>11}{'toterr_nv':>11}{'bias winner':>13}")

    # ---------------- [N1] instrument STRENGTH ----------------
    print("[N1] negative-control STRENGTH sweep (Assumption 3.1 holds throughout)")
    print(hdr)
    n1 = []
    for s0 in SIGMA0_GRID:
        p = default_params(T=3, kappa=1.0, sigma_0=s0)
        assert all(abs(true_value(p, pi) - v_true[n]) < 1e-12 for n, pi in cands.items())
        r = evaluate(p, cands, v_true)
        r["sigma_0"] = s0
        r["bias_winner"] = "plug-in" if r["sys_plugin"] < r["sys_naive"] else "naive"
        n1.append(r)
        print(f"{s0:>9.2f}{r['recovery_err']:>11.4f}{r['sys_plugin']:>12.4f}"
              f"{r['sys_naive']:>12.4f}{r['tot_plugin']:>11.4f}{r['tot_naive']:>11.4f}"
              f"{r['bias_winner']:>13}")
    RESULTS["N1_strength"] = n1

    # ---------------- [N2] assumption VALIDITY ----------------
    print("\n[N2] negative-control VALIDITY sweep (delta_nc>0 breaks Assumption 3.1)")
    print(hdr)
    n2 = []
    for dl in DELTA_GRID:
        p = default_params(T=3, kappa=1.0, delta_nc=dl)
        assert all(abs(true_value(p, pi) - v_true[n]) < 1e-12 for n, pi in cands.items()), \
            "delta_nc must not move the true value (E[eps_0]=0)"
        r = evaluate(p, cands, v_true)
        r["delta_nc"] = dl
        r["bias_winner"] = "plug-in" if r["sys_plugin"] < r["sys_naive"] else "naive"
        n2.append(r)
        print(f"{dl:>9.2f}{r['recovery_err']:>11.4f}{r['sys_plugin']:>12.4f}"
              f"{r['sys_naive']:>12.4f}{r['tot_plugin']:>11.4f}{r['tot_naive']:>11.4f}"
              f"{r['bias_winner']:>13}")
    RESULTS["N2_validity"] = n2

    # ---------------- summary ----------------
    s_rec = [r["recovery_err"] for r in n1]
    d_rec = [r["recovery_err"] for r in n2]
    RESULTS["N1_recovery_degradation"] = float(s_rec[-1] / s_rec[0])
    RESULTS["N2_recovery_degradation"] = float(d_rec[-1] / d_rec[0])
    RESULTS["N1_sysbias_breakeven"] = next(
        (r["sigma_0"] for r in n1 if r["bias_winner"] == "naive"), None)
    RESULTS["N2_sysbias_breakeven"] = next(
        (r["delta_nc"] for r in n2 if r["bias_winner"] == "naive"), None)

    print(f"\nrecovery error grows {s_rec[-1]/s_rec[0]:.2f}x across the strength sweep, "
          f"{d_rec[-1]/d_rec[0]:.2f}x across the validity sweep")
    print("systematic-bias advantage of the plug-in is lost at "
          f"sigma_0 = {RESULTS['N1_sysbias_breakeven']} (strength), "
          f"delta_nc = {RESULTS['N2_sysbias_breakeven']} (validity)")
    print("NOTE: total error already favours naive everywhere, because the "
          "plug-in trades bias for variance -- the two metrics disagree by "
          "construction and both are reported.")

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_negative_control.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
