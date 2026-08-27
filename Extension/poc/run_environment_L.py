"""run_environment_L.py -- does any of this survive |S|=8, T=10, and a baseline?

Predictions: docs/scale_and_baselines.md P-L1..P-L4, P-L8, P-L9, committed at
f709bba before this file existed.

Environment L: |S|=8, |O|=10, |O_0|=3, |A|=2, T=10, confounding 0.9. Nineteen
bridge blocks against the five of every previous run in this project. Uses
ScaledBridgeEstimator, verified identical to TabularBridgeEstimator at
0.000e+00 over 20 comparisons (run_estimator_equivalence.py).

ARMS
  full        every block, full width, no projection      -- the anchor construction
  plugin      xi = 0 everywhere                           -- control on the same fits
  projall     every block restricted to its signal span   -- our remedy
  hoeffding   count-based LCB, bonus ~ N^(-1/2)           -- STANDARD offline-RL
                                                             pessimism, not ours

hoeffding is the baseline that answers "maybe it is your implementation". It is a
different penalty FAMILY: no ellipsoid, no design geometry, no prescribed width
schedule. Its bonus decays by construction (e < 0), so if it converges where the
full arm does not, the failure belongs to the SCHEDULE and not to pessimism, and
the paper's claim has to be scoped accordingly.

Both full and hoeffding are calibrated to the same TOTAL penalty at the smallest
N, on the same reference policy, so at N_min they are the same size and differ
only in how they scale.

A behaviour-cloned observation policy is also reported: a fixed policy, constant
in N, which is what an analyst gets by cloning the logs with no OPE at all.

Writes experiments/results_environment_L.json.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params, sample_trajectories, dp_value
from bridge_estimator_scaled import ScaledBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid, pessimistic_value
from run_completeness_beta import population_design_span, beta_pop_for

N_S, N_O, N_O0, N_A = 8, 10, 3, 2
T = int(os.environ.get("HORIZON", "10"))
CONFOUND = 0.9
TAU, KAP = 0.0, 1.5                      # schedule C, e = +0.5
SPLIT = 0.7
W_TARGET = 0.12
N_GRID = [int(x) for x in os.environ.get("NGRID", "4000,16000,64000").split(",")]
SEEDS = list(range(int(os.environ.get("SEEDS", "10"))))
ARMS = ["full", "plugin", "projall", "hoeffding"]
DELTA = 0.05


def candidates(p=None):
    """The six hand-made candidates, PLUS the behaviour clone.

    bc is in the candidate set because the smoke test found it beats the best of
    the original six on every configuration tested, including (4,6,2) at T=3 --
    the grid every decision result in this project used. Measuring regret against
    a reference that a no-OPE baseline already beats is not a regret. See
    scale_and_baselines.md section 6.
    """
    lo = np.linspace(0.9, 0.1, N_O)
    C = {"always_0": np.stack([np.ones(N_O), np.zeros(N_O)], 1),
         "always_1": np.stack([np.zeros(N_O), np.ones(N_O)], 1),
         "uniform": np.full((N_O, N_A), 1.0 / N_A),
         "greedy_lo": np.stack([lo, 1 - lo], 1),
         "greedy_hi": np.stack([1 - lo, lo], 1),
         "soft": np.stack([0.5 + 0.2 * (lo - .5), 0.5 - 0.2 * (lo - .5)], 1)}
    if p is not None:
        C["bc"] = bc_policy(p)
    return C


def bc_policy(p):
    """Observation-marginal of the confounded behaviour policy: what BC would learn.

    d(s) is the stage-1 latent marginal, so this is pi_b marginalized over the
    states that produce each observation. Fixed, constant in N.
    """
    d = p["p1"]
    joint = (d[:, None] * p["E"]).T               # (n_o, n_s), unnormalised
    denom = joint.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return (joint / denom) @ p["pi_b"]            # (n_o, n_a)


def build_blocks(est):
    bR, bD = [], []
    for t in range(1, T + 1):
        s = est.stage_store["bR_t%d" % t]
        bR.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                 s["N2"], "bR_t%d" % t, s["signal_basis"],
                                 n_obs=N_O, n_act=N_A, n_y=s["n_y"]))
        if t < T:
            s = est.stage_store["bD_t%d" % t]
            bD.append(BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"],
                                     s["N2"], "bD_t%d" % t, s["signal_basis"],
                                     n_obs=N_O, n_act=N_A, n_y=s["n_y"]))
    return bR, bD


def xi_for(tau, N2, sigma2, c_mult, label="?"):
    """Width parameter. tau = 0 divides by the design signal eigenvalue.

    The original helper used max(sigma2, 1e-12), which silently converted an
    UNDEFINED width into 8.33e8 and let a run continue to a penalty of 2.8e8
    against a value scale of 7.9. sigma2 = 0 is not a small number, it is the
    statement that the block carries no signal, and the tau = 0 rule has no value
    there. Guard, do not floor. See scale_and_baselines.md section 8.
    """
    if tau == 0.0:
        if not (sigma2 > 0.0):
            raise ValueError(
                "sigma2 = %r at block %s: the tau=0 width rule xi = c/(N2*sigma2) "
                "is undefined here. The block's design has collapsed, which at "
                "long horizons is the cross-fitting split failing, not noise. "
                "Truncate the horizon or change the width rule; do not floor."
                % (sigma2, label))
        return c_mult * N2 ** (tau - 1.0) / sigma2
    return c_mult * N2 ** (tau - 1.0)


def fit_once(p, N, sd):
    d = sample_trajectories(p, N, np.random.default_rng(sd))
    N2 = N - int(N * SPLIT)
    est = ScaledBridgeEstimator(n_obs=N_O, n_act=N_A, n_o0=N_O0, n_r=2, T=T,
                                mode="primal", seed=sd, lambda2=N2 ** (-KAP))
    est.fit(d["O0"], d["O"], d["A"], d["R"])
    return est, d


def make_vg(pi, p_o1):
    def vg(bRl, bDl):
        BR = np.stack([b.reshape(N_A, N_O, 2, N_O) for b in bRl])
        BD = np.stack([b.reshape(N_A, N_O, N_O, N_O) for b in bDl])
        V, _p, (gR, gD) = plugin_value(BR, BD, pi, p_o1, return_grads=True)
        return (V, [gR[t].ravel() for t in range(T)],
                [gD[j].ravel() for j in range(T - 1)])
    return vg


def hoeffding_pen(bR, bD, gR, gD, N2, c_h):
    """Count-based LCB bonus: C * ||g||_2 * sqrt(log(D/delta)/N2), summed over blocks.

    No design geometry, no width schedule. Decays as N^(-1/2) by construction.
    """
    tot = 0.0
    for b, g in list(zip(bR, gR)) + list(zip(bD, gD)):
        tot += float(np.linalg.norm(g)) * np.sqrt(
            np.log(b.b_hat.size / DELTA) / N2)
    return c_h * tot


def arm_value(arm, bR, bD, xR, xD, vg, N2, c_h):
    if arm == "plugin":
        V, _, _ = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
        return float(V), 0.0
    if arm == "hoeffding":
        V, gR, gD = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
        pen = hoeffding_pen(bR, bD, gR, gD, N2, c_h)
        return float(V - pen), float(pen)
    proj = (arm == "projall")
    V0, _, _ = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
    out = pessimistic_value(bR, bD, xR, xD, vg, projected=proj,
                            rng=np.random.default_rng(0))
    return float(out["V_low"]), float(V0 - out["V_low"])


def main():
    p = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T, seed=0,
                       confound=CONFOUND)
    cands = candidates(p)
    v_true = {k: dp_value(p, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    spread = max(v_true.values()) - min(v_true.values())
    nu1 = p["p1"] @ p["E"]
    beta_g = max(float(np.linalg.norm(
        nu1 - population_design_span(p, a) @
        (population_design_span(p, a).T @ nu1))) / float(np.linalg.norm(nu1))
        for a in range(N_A))
    beta = max(beta_pop_for(p, a)[0] for a in range(N_A))
    v_bc = dp_value(p, bc_policy(p))
    print("Environment L: |S|=%d |O|=%d |O0|=%d T=%d cf=%s, %d blocks, %d seeds"
          % (N_S, N_O, N_O0, T, CONFOUND, 2 * T - 1, len(SEEDS)))
    print("optimal %s  V=%.4f   spread %.4f   floor %.4f   ratio %.3f  %s"
          % (best, v_true[best], spread, beta * beta_g,
             beta * beta_g / spread,
             "SEPARABLE" if beta * beta_g < spread else "NOT SEPARABLE"))
    print("behaviour-clone policy value %.4f  (regret %.4f)"
          % (v_bc, v_true[best] - v_bc))
    if beta_g < 1e-6:
        raise SystemExit("ABORT: beta_g = 0, this grid cannot show the effect.")

    est, d = fit_once(p, N_GRID[0], 0)
    bR, bD = build_blocks(est)
    vg = make_vg(cands["greedy_lo"], empirical_p_o1(d["O"], N_O))
    _, gR0, gD0 = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
    w_unit = sum(np.sqrt(xi_for(TAU, b.N2, b.sigma2, 1.0, b.label)) * b.h_inv_norm(g)
                 for b, g in list(zip(bR, gR0)) + list(zip(bD, gD0)))
    c_mult = (W_TARGET / w_unit) ** 2
    h_unit = hoeffding_pen(bR, bD, gR0, gD0, bR[0].N2, 1.0)
    c_h = W_TARGET / h_unit
    print("calibration: c_ellipsoid = %.3e   c_hoeffding = %.3e\n"
          % (c_mult, c_h))

    rows = []
    for N in N_GRID:
        acc = {a: [] for a in ARMS}
        pens = {a: [] for a in ARMS}
        picks = {a: [] for a in ARMS}
        cons = {a: [0, 0] for a in ARMS}
        t0 = time.time()
        for sd in SEEDS:
            est, d = fit_once(p, N, sd)
            bR, bD = build_blocks(est)
            p_o1 = empirical_p_o1(d["O"], N_O)
            xR = [xi_for(TAU, b.N2, b.sigma2, c_mult, b.label) for b in bR]
            xD = [xi_for(TAU, b.N2, b.sigma2, c_mult, b.label) for b in bD]
            for arm in ARMS:
                vals, pn = {}, {}
                for name, pi in cands.items():
                    vgc = make_vg(pi, p_o1)
                    xr = xR if arm != "plugin" else [0.0] * len(xR)
                    xd = xD if arm != "plugin" else [0.0] * len(xD)
                    v, q = arm_value(arm, bR, bD, xr, xd, vgc, bR[0].N2, c_h)
                    vals[name], pn[name] = v, q
                    cons[arm][1] += 1
                    cons[arm][0] += int(v <= v_true[name] + 1e-12)
                pick = max(vals, key=vals.get)
                picks[arm].append(pick)
                acc[arm].append(v_true[best] - v_true[pick])
                pens[arm].append(pn[best])
        el = time.time() - t0
        for arm in ARMS:
            r = np.array(acc[arm])
            md = max(set(picks[arm]), key=picks[arm].count)
            print("N=%-7d %-10s regret %.4f +- %.4f   pen %.4f   modal %-10s"
                  % (N, arm, r.mean(), 1.96 * r.std() / np.sqrt(len(r)),
                     np.mean(pens[arm]), md))
            rows.append(dict(N=N, arm=arm, regret=float(r.mean()),
                             regret_ci=float(1.96 * r.std() / np.sqrt(len(r))),
                             penalty=float(np.mean(pens[arm])), modal=md,
                             picks=picks[arm],
                             conservative_frac=cons[arm][0]
                             / max(cons[arm][1], 1)))
        print("   (%.0f s)\n" % el)

    print("=" * 88)
    lg = np.log(N_GRID)
    slopes = {}
    for arm in ARMS:
        y = [r["regret"] for r in rows if r["arm"] == arm]
        if min(y) > 0:
            slopes[arm] = float(np.polyfit(lg, np.log(y), 1)[0])
        else:
            # a zero-regret arm has no log-log slope; report the linear one
            # normalised by the spread so the sign is still meaningful
            slopes[arm] = float(np.polyfit(lg, y, 1)[0]) / max(spread, 1e-12)
        print("%-10s regret slope %+.4f   regret %s"
              % (arm, slopes[arm], ["%.4f" % v for v in y]))
    rf = [r["regret"] for r in rows if r["arm"] == "full"]
    rp = [r["regret"] for r in rows if r["arm"] == "plugin"]
    print("\nP-L1  full slope positive: %s (%+.4f)"
          % (slopes["full"] > 0, slopes["full"]))
    print("P-L2  plugin negative and below full at N_max: %s / %s"
          % (slopes["plugin"] < 0, rp[-1] < rf[-1]))
    print("P-L3  hoeffding converges (negative slope): %s (%+.4f)"
          % (slopes["hoeffding"] < 0, slopes["hoeffding"]))
    print("P-L4  failure belongs to the SCHEDULE: %s"
          % (slopes["full"] > 0 and slopes["hoeffding"] < 0))
    print("      spread %.4f; regret at the spread means no information" % spread)

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_environment_L.json"))
    with open(out, "w") as f:
        json.dump(dict(rows=rows, slopes=slopes, v_true=v_true, optimal=best,
                       spread=spread, beta=beta, beta_g=beta_g,
                       floor=beta * beta_g, v_bc=v_bc, config=[N_S, N_O, N_O0],
                       T=T, confound=CONFOUND, n_seeds=len(SEEDS),
                       N_grid=N_GRID, arms=ARMS, c_mult=c_mult, c_h=c_h),
                  f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
