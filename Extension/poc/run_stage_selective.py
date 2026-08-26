"""run_stage_selective.py -- can the diagnosis be turned into a remedy?

Predictions: docs/stage_selective_pessimism.md, committed before this file
existed (commit 07e4c50).

THE IDEA. cor:selfheal says an inadequate negative control leaks only at the
FIRST stage: at t = 1 the conditioning set is (A_1, O_0), so |O_0| < |S| forces
an incomplete span, while from t = 2 on the conditioning set includes history and
the per-action span reaches |S| (beta_t <= 2.7e-15, 24/24 cells). So the floor
W >= beta*beta_g binds on 2 of the 2T-1 = 5 blocks, not on the estimator.

That is checkable before fitting -- |O_0| against |S|, and whether the logger saw
latent state -- so "apply pessimism only where the region can contract" is an
implementable rule rather than an oracle one.

FOUR ARMS, ONE FIT. Every arm reads the same fitted bridge at each (N, seed), so
the only thing that varies is which blocks get a nonzero region:

    full    all blocks    full region          (the tab:decision column)
    tail    t = 1         xi = 0               t >= 2 full region
    proj1   t = 1         restricted to span   t >= 2 full region
    plugin  all blocks    xi = 0               (the other tab:decision column)

A block with xi = 0 returns its centre and zero penalty, which IS the plug-in for
that block, so the on/off switch needs no special case in the optimizer.

PART A checks the diagnosis before trusting the remedy: per-block penalties
sqrt(xi_k)*||g_k||_{H_k^-1} across the N grid. If the divergence is not
concentrated at t = 1, Part B is aimed at the wrong place.

PART C prices it. Pessimism buys conservatism; an arm that stops paying may stop
being a lower bound. We record V_low <= V_true per (arm, N, seed, policy) rather
than asserting the cost is small.

Writes experiments/results_stage_selective.json.
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

N_S, N_O, N_O0, N_A, T = 4, 6, 2, 2, 3
CONFOUND = 0.9
N_GRID = [int(x) for x in
          os.environ.get("NGRID", "2000,8000,32000,128000").split(",")]
SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
SPLIT = 0.7
W_TARGET = 0.12

ALPHA_P, C2_P = 10.0, 1.0
SCHEDULES = [
    ("C  kappa=1.5", 0.0, 1.5),
    ("paper a=10,c2=1", (ALPHA_P + 2) / (2 * ALPHA_P + 2),
     ALPHA_P / (ALPHA_P * C2_P + 1)),
]
ARMS = ["full", "tail", "proj1", "projall", "plugin"]
# selfheal_theorem.md refuted the assumption that only the t=1 blocks leak: with
# rank-deficient transitions every stage leaks, and "tail" then drops the penalty
# on the wrong blocks. "projall" is the arm that does not need to know which
# stage leaks -- it refuses to pay for unidentified directions at EVERY block.
FAMILY = os.environ.get("FAMILY", "dense")
# Long background jobs here get killed before two families x two schedules
# finish, so the driver runs one slice at a time and the JSON is keyed by both.
_ONLY = [t for t in os.environ.get("SCHED", "").split(",") if t]
if _ONLY:
    SCHEDULES = [s for s in SCHEDULES if any(o in s[0] for o in _ONLY)]
C_CAL = {}
RESULTS = {}


def candidates():
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


def xi_for(tau, N2, sigma2, c_mult):
    m = sigma2 if tau == 0.0 else 1.0
    return c_mult * N2 ** (tau - 1.0) / max(m, 1e-12)


def fit_once(p, N, sd, kap):
    d = sample_trajectories(p, N, np.random.default_rng(sd))
    N2 = N - int(N * SPLIT)
    est = TabularBridgeEstimator(n_obs=N_O, n_act=N_A, n_o0=N_O0, n_r=2, T=T,
                                 mode="primal", seed=sd, lambda2=N2 ** (-kap))
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


def arm_regions(arm, xR_full, xD_full):
    """(xis_R, xis_D, projected_R, projected_D) for one arm.

    Index 0 in each list IS t = 1, the only stage cor:selfheal says leaks.
    """
    nR, nD = len(xR_full), len(xD_full)
    if arm == "full":
        return xR_full, xD_full, [False] * nR, [False] * nD
    if arm == "plugin":
        return [0.0] * nR, [0.0] * nD, [False] * nR, [False] * nD
    if arm == "tail":
        return ([0.0] + list(xR_full[1:]), [0.0] + list(xD_full[1:]),
                [False] * nR, [False] * nD)
    if arm == "proj1":
        return (xR_full, xD_full,
                [True] + [False] * (nR - 1), [True] + [False] * (nD - 1))
    if arm == "projall":
        return xR_full, xD_full, [True] * nR, [True] * nD
    raise ValueError(arm)


def precondition(p):
    nu1 = p["p1"] @ p["E"]
    bg = max(float(np.linalg.norm(nu1 - population_design_span(p, a) @
                                  (population_design_span(p, a).T @ nu1)))
             / float(np.linalg.norm(nu1)) for a in range(N_A))
    b = max(beta_pop_for(p, a)[0] for a in range(N_A))
    print("precondition: beta = %.4f, beta_g = %.4f, floor = %.4f"
          % (b, bg, b * bg))
    if bg < 1e-6:
        raise SystemExit("ABORT: beta_g = 0, this grid cannot show the effect.")
    return b, bg


def calibrate(p, cands):
    print("\ncalibrating every schedule to width %s at N=%d"
          % (W_TARGET, N_GRID[0]))
    for tag, tau, kap in SCHEDULES:
        est, d = fit_once(p, N_GRID[0], 0, kap)
        bR, bD = build_blocks(est)
        vg = make_vg(cands["greedy_lo"], empirical_p_o1(d["O"], N_O))
        _, g, _ = vg([b.b_hat for b in bR], [b.b_hat for b in bD])
        w1 = np.sqrt(xi_for(tau, bR[0].N2, bR[0].sigma2, 1.0)) * \
            bR[0].h_inv_norm(g[0])
        C_CAL[tag] = (W_TARGET / w1) ** 2
        print("   %-18s c = %.3e" % (tag, C_CAL[tag]))


def grad_null_share(H, lam, g, tol=1e-9):
    """beta_g for one block, measured on the EMPIRICAL design.

    Added after the smoke test, which exposed a hole in the reasoning that led to
    this experiment. cor:selfheal establishes beta_t (the TRUTH's null share) is
    ~1e-15 for t >= 2, and the per-stage FLOOR beta_t*beta_g,t is therefore ~0.
    A zero floor does not make the penalty small: the penalty is
    sqrt(xi)*||g||_{H^-1}, and ||g||_{H^-1} >= beta_g / sqrt(lambda) whatever
    beta is. So a t >= 2 block can have no floor and still have a diverging
    penalty, if its GRADIENT leaks even though its truth does not.

    Nothing measured so far separates those two cases, so we measure it here.
    """
    T2 = H - lam * np.eye(H.shape[0])
    w, V = np.linalg.eigh(T2)
    cut = tol * max(float(w[-1]), 1e-300)
    null = V[:, w <= cut]
    ng = float(np.linalg.norm(g))
    if ng < 1e-300 or null.shape[1] == 0:
        return 0.0, int(null.shape[1])
    return float(np.linalg.norm(null.T @ g) / ng), int(null.shape[1])


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.array(ys, dtype=float), 1)[0])


def logslope(xs, ys):
    y = np.maximum(np.asarray(ys, dtype=float), 1e-300)
    return float(np.polyfit(np.log(xs), np.log(y), 1)[0])


def main():
    if FAMILY == "dense":
        p = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T,
                           seed=0, confound=CONFOUND)
    else:
        # Same p1, K0, E and pi_b, so the stage-1 floor is IDENTICAL; only the
        # transition changes. That makes the comparison a controlled one.
        from run_selfheal_formula import make_params, span_bruteforce
        p = make_params((N_S, N_O, N_O0), CONFOUND, FAMILY)
        spans = [span_bruteforce(p, a, t).shape[1]
                 for a in range(N_A) for t in (1, 2, 3)]
        print("family=%s   per-action spans (a,t) = %s against |S| = %d"
              % (FAMILY, spans, N_S))
    print("(%d,%d,%d) confound=%s  INCOMPLETE   %d seeds   family=%s\n"
          % (N_S, N_O, N_O0, CONFOUND, len(SEEDS), FAMILY))
    beta, beta_g = precondition(p)

    cands = candidates()
    v_true = {k: dp_value(p, pi) for k, pi in cands.items()}
    best = max(v_true, key=v_true.get)
    print("\noptimal %s  V_true = %.4f" % (best, v_true[best]))
    print("   regret levels:", {k: round(v_true[best] - v, 4)
                                for k, v in sorted(v_true.items(),
                                                   key=lambda kv: -kv[1])})
    calibrate(p, cands)

    labels = ["bR_t%d" % t for t in range(1, T + 1)] + \
             ["bD_t%d" % t for t in range(1, T)]
    rows, pen_rows, cons_rows = [], [], []

    for tag, tau, kap in SCHEDULES:
        e = tau + kap - 1
        print("\n" + "=" * 94)
        print("%s   e = %+.3f" % (tag, e))
        print("=" * 94)

        # ------------------------------------------------------- Parts A/B/C
        # One fit per (N, seed) feeds all three parts. Sampling 128k
        # trajectories twice per seed was the dominant cost in the smoke test.
        pen_by_N = {L: [] for L in labels}
        bg_by_N = {L: [] for L in labels}
        nd_by_N = {L: [] for L in labels}
        share_by_N = []
        arm_reg = {a: [] for a in ARMS}
        for N in N_GRID:
            acc = {L: [] for L in labels}
            accg = {L: [] for L in labels}
            accnd = {L: [] for L in labels}
            picks = {a: [] for a in ARMS}
            cons = {a: [0, 0] for a in ARMS}       # [n_conservative, n_total]
            # The fraction saturated at 1.000 for every arm in the smoke test,
            # so it cannot by itself distinguish a barely-valid bound from a
            # hugely over-paid one. The signed gap V_low - V_true can.
            gaps = {a: [] for a in ARMS}
            for sd in SEEDS:
                est, d = fit_once(p, N, sd, kap)
                bR, bD = build_blocks(est)
                p_o1 = empirical_p_o1(d["O"], N_O)
                cm = C_CAL[tag]
                xR = [xi_for(tau, b.N2, b.sigma2, cm) for b in bR]
                xD = [xi_for(tau, b.N2, b.sigma2, cm) for b in bD]

                # -- Part A: per-block penalty and gradient leak at the optimum
                vg0 = make_vg(cands[best], p_o1)
                _, gR, gD = vg0([b.b_hat for b in bR], [b.b_hat for b in bD])
                for t, b in enumerate(bR):
                    L = "bR_t%d" % (t + 1)
                    acc[L].append(np.sqrt(xR[t]) * b.h_inv_norm(gR[t]))
                    bg, nd = grad_null_share(b.H, est.stage_store[L]["lam2"],
                                             gR[t])
                    accg[L].append(bg)
                    accnd[L].append(nd)
                for j, b in enumerate(bD):
                    L = "bD_t%d" % (j + 1)
                    acc[L].append(np.sqrt(xD[j]) * b.h_inv_norm(gD[j]))
                    bg, nd = grad_null_share(b.H, est.stage_store[L]["lam2"],
                                             gD[j])
                    accg[L].append(bg)
                    accnd[L].append(nd)

                # -- Parts B and C: the four arms, off the same fit
                for arm in ARMS:
                    xr, xd, pr, pd = arm_regions(arm, xR, xD)
                    vals = {}
                    for name, pi in cands.items():
                        vg = make_vg(pi, p_o1)
                        if arm == "plugin":
                            v = vg([b.b_hat for b in bR],
                                   [b.b_hat for b in bD])[0]
                        else:
                            v = pessimistic_value(
                                bR, bD, xr, xd, vg,
                                rng=np.random.default_rng(sd),
                                projected_R=pr, projected_D=pd)["V_low"]
                        vals[name] = float(v)
                        cons[arm][1] += 1
                        gaps[arm].append(float(v) - v_true[name])
                        if v <= v_true[name] + 1e-9:
                            cons[arm][0] += 1
                    picks[arm].append(max(vals, key=vals.get))

            means = {L: float(np.mean(acc[L])) for L in labels}
            tot = sum(means.values())
            share = (means["bR_t1"] + means["bD_t1"]) / max(tot, 1e-300)
            share_by_N.append(float(share))
            row = dict(schedule=tag, e=e, N=N, share_t1=float(share))
            for L in labels:
                pen_by_N[L].append(means[L])
                bg_by_N[L].append(float(np.mean(accg[L])))
                nd_by_N[L].append(float(np.mean(accnd[L])))
                row["pen_" + L] = means[L]
                row["betag_" + L] = float(np.mean(accg[L]))
                row["nulldim_" + L] = float(np.mean(accnd[L]))
            pen_rows.append(row)

            for arm in ARMS:
                r = np.array([v_true[best] - v_true[q] for q in picks[arm]])
                m = float(r.mean())
                ci = float(1.96 * r.std(ddof=1) / np.sqrt(len(r)))
                arm_reg[arm].append(m)
                modal = max(set(picks[arm]), key=picks[arm].count)
                frac = cons[arm][0] / max(cons[arm][1], 1)
                gp = np.array(gaps[arm])
                rows.append(dict(schedule=tag, e=e, N=N, arm=arm, regret=m,
                                 regret_ci=ci, modal=modal, picks=picks[arm]))
                cons_rows.append(dict(schedule=tag, N=N, arm=arm,
                                      conservative_frac=float(frac),
                                      mean_gap=float(gp.mean()),
                                      max_gap=float(gp.max()),
                                      n_cells=cons[arm][1]))
            print("   N=%d done" % N)

        # ---------------------------------------------------------- Part A
        print("\nPART A  per-block penalty sqrt(xi_k)*||g_k||_Hinv at the "
              "optimal policy, mean over %d seeds" % len(SEEDS))
        print("%8s" % "N" + "".join("%11s" % L for L in labels)
              + "%11s" % "t1 share")
        for i, N in enumerate(N_GRID):
            print("%8d" % N + "".join("%11.4f" % pen_by_N[L][i]
                                      for L in labels)
                  + "%11.3f" % share_by_N[i])
        print("%8s" % "slope"
              + "".join("%11.4f" % logslope(N_GRID, pen_by_N[L])
                        for L in labels)
              + "   pred t1 %+.3f / t>=2 %+.3f" % (e / 2, (tau - 1) / 2))
        # The quantity that decides whether a block's penalty CAN contract.
        # cor:selfheal measured only the TRUTH's leak; this is the gradient's,
        # and a block with no design null has neither.
        print("%8s" % "beta_g"
              + "".join("%11.4f" % bg_by_N[L][-1] for L in labels))
        print("%8s" % "nulldim"
              + "".join("%11.1f" % nd_by_N[L][-1] for L in labels))
        for L in labels:
            RESULTS.setdefault("penalty_slopes", []).append(
                dict(schedule=tag, block=L,
                     slope=logslope(N_GRID, pen_by_N[L]),
                     pen_by_N=pen_by_N[L], betag_by_N=bg_by_N[L],
                     nulldim_by_N=nd_by_N[L],
                     pred_leaky=e / 2, pred_clean=(tau - 1) / 2))

        # ---------------------------------------------------------- Part B
        print("\nPART B  selection regret by arm, %d seeds" % len(SEEDS))
        print("%8s" % "N" + "".join("%20s" % a for a in ARMS))
        for i, N in enumerate(N_GRID):
            line = "%8d" % N
            for arm in ARMS:
                c = [r for r in rows if r["schedule"] == tag
                     and r["N"] == N and r["arm"] == arm][0]
                line += "%9.4f +-%-8.4f" % (c["regret"], c["regret_ci"])
            print(line)
        print("\n%8s" % "slope"
              + "".join("%20.5f" % slope(N_GRID, arm_reg[a]) for a in ARMS))
        print("\nPART C  conservatism.  frac = cells with V_low <= V_true;")
        print("        gap = mean V_low - V_true (more negative = more slack "
              "paid); worst = max over cells")
        print("%8s" % "N" + "".join("%22s" % a for a in ARMS))
        for N in N_GRID:
            line = "%8d" % N
            for arm in ARMS:
                c = [r for r in cons_rows if r["schedule"] == tag
                     and r["N"] == N and r["arm"] == arm][0]
                line += "%7.2f%8.3f%7.3f" % (c["conservative_frac"],
                                             c["mean_gap"], c["max_gap"])
            print(line)
        print("%8s" % "" + "".join("%22s" % "frac   gap  worst"
                                   for _ in ARMS))
        for arm in ARMS:
            RESULTS.setdefault("arm_slopes", []).append(
                dict(schedule=tag, arm=arm,
                     regret_slope=slope(N_GRID, arm_reg[arm])))

    RESULTS.update(rows=rows, penalties=pen_rows, conservatism=cons_rows,
                   v_true=v_true, optimal=best, beta=beta, beta_g=beta_g,
                   floor=beta * beta_g, confound=CONFOUND,
                   config=[N_S, N_O, N_O0], n_seeds=len(SEEDS), T=T,
                   family=FAMILY, arms=ARMS)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_stage_selective_%s%s.json"
                                        % (FAMILY,
                                           "_" + "_".join(_ONLY).replace(
                                               "=", "").replace(".", "")
                                           if _ONLY else "")))
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
