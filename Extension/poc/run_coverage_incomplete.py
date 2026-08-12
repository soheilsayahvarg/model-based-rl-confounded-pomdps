"""run_coverage_incomplete.py -- the coverage test, where completeness FAILS.

Step (c)'s coverage branch has never had a confirmed empirical instance. On the
Phase 3 toy the predicted crossing sat at N* = 1.7e9 because beta_pop = 0 there
(completeness holds), and we wrongly reported an unreachable grid as a test.

docs/completeness_and_h4.md established, as population algebra over 50 cells, that
beta_pop > 0 exactly when per-action completeness fails, and predicted the
crossing from population quantities alone. Those predictions were committed before
this script existed. This runs them.

THE DESIGN.

  * treatment: (4,6,2) at confound = 0.0 -- |O_0| < |S|, so completeness fails
    STRUCTURALLY, at every confounding level, without a degenerate behaviour
    policy. beta_pop = 1.211 (act 0).
  * control:   (2,6,4) at confound = 0.6 -- completeness holds, beta_pop = 0.
    Predicted to show NO crossing at any N or any c. This is the control that
    decides whether a measured crossing is driven by beta_pop or by something
    else, and step (c) had no such control.

  * The signature. N* = (c/(lam0*beta_pop^2*sigma2_pop))^2 scales as c^2, so
    sweeping c moves the predicted crossing across two orders of magnitude on one
    N grid. A crossing that does not move as c^2 refutes the mechanism even if a
    crossing exists.

  * The prediction is an UPPER BOUND, not an equality. Coverage needs
    xi >= xi_needed and the null term gives only xi_needed >= lam*beta^2, so the
    signal directions can break coverage earlier. Measured crossing <= predicted.

T = 1 is used: the stage-1 reward block's H and b_hat are built from (O_0, O_1,
A_1, R_1) alone, so the block is identical at any T, and the dynamic blocks (whose
index space is |O|^2) would cost ~4x the memory for nothing. Asserted, not assumed.

Writes experiments/results_coverage_incomplete.json.
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
from bridge_estimator import TabularBridgeEstimator
from ellipsoid_opt import BlockEllipsoid
from run_completeness_beta import (true_bridges_generic, population_design_span,
                                   beta_pop_for)

LAM0 = 0.03
KAPPA = 0.5
C_GRID = [0.03, 0.1, 0.2, 0.3]
N_GRID = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
SEEDS = [0, 1, 2]
SPLIT = 0.7
RESULTS = {}

CASES = [
    dict(tag="treatment (4,6,2) cf=0.0  INCOMPLETE", n_s=4, n_o=6, n_o0=2, cf=0.0),
    dict(tag="control   (2,6,4) cf=0.6  complete",   n_s=2, n_o=6, n_o0=4, cf=0.6),
]


def sigma2_pop_of(p, a):
    E, p1, K0, pi_b = p["E"], p["p1"], p["K0"], p["pi_b"]
    cols, ws = [], []
    for x in range(K0.shape[1]):
        v = p1 * pi_b[:, a] * K0[:, x]
        if v.sum() <= 0:
            continue
        ws.append(v.sum())
        cols.append(E.T @ (v / v.sum()))
    D = sum(w * np.outer(m, m) for w, m in zip(ws, cols)) / sum(ws)
    ev = np.linalg.eigvalsh(D)[::-1]
    r = population_design_span(p, a).shape[1]
    return float(ev[r - 1])


def fit_block(p, N, sd, n_o, n_o0):
    N2 = N - int(N * SPLIT)
    lam2 = LAM0 * N2 ** (-KAPPA)
    d = sample_trajectories(p, N, np.random.default_rng(sd))
    est = TabularBridgeEstimator(n_obs=n_o, n_act=2, n_o0=n_o0, n_r=2, T=1,
                                 mode="primal", seed=sd, lambda2=lam2)
    est.fit(d["O0"], d["O"], d["A"], d["R"])
    s = est.stage_store["bR_t1"]
    blk = BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"],
                         "bR_t1", s["signal_basis"], n_obs=n_o, n_act=2,
                         n_y=s["n_y"])
    return blk, lam2, N2


def check_T_invariance():
    """The stage-1 reward block must not depend on T. Assert rather than assume."""
    p = default_params(n_s=4, n_a=2, n_o=6, n_o0=2, T=3, seed=0, confound=0.0)
    d = sample_trajectories(p, 4000, np.random.default_rng(0))
    out = {}
    for T in (1, 3):
        est = TabularBridgeEstimator(n_obs=6, n_act=2, n_o0=2, n_r=2, T=T,
                                     mode="primal", seed=0, lambda2=1e-3)
        est.fit(d["O0"], d["O"][:, :T], d["A"][:, :T], d["R"][:, :T])
        out[T] = est.stage_store["bR_t1"]
    dh = float(np.abs(out[1]["H"] - out[3]["H"]).max())
    db = float(np.abs(out[1]["b_hat_vec"] - out[3]["b_hat_vec"]).max())
    print(f"T-invariance of the stage-1 reward block: max|dH| = {dh:.2e}, "
          f"max|db_hat| = {db:.2e}")
    assert dh < 1e-12 and db < 1e-12, "stage-1 block depends on T; rerun at T=3"
    return dh, db


def main():
    dh, db = check_T_invariance()
    RESULTS["T_invariance"] = dict(max_dH=dh, max_db=db)

    all_rows = []
    for case in CASES:
        p = default_params(n_s=case["n_s"], n_a=2, n_o=case["n_o"],
                           n_o0=case["n_o0"], T=1, seed=0, confound=case["cf"])
        bR_true, res = true_bridges_generic(p)
        b_true = bR_true.ravel()
        bp0, _, _, _ = beta_pop_for(p, 0)
        bp1, _, _, _ = beta_pop_for(p, 1)
        beta_pop = max(bp0, bp1)
        s2p = min(sigma2_pop_of(p, 0), sigma2_pop_of(p, 1))

        print("\n" + "=" * 92)
        print(f"{case['tag']}    beta_pop = {beta_pop:.4f}  "
              f"sigma2_pop = {s2p:.4e}  (bridge residual {res:.1e})")
        print("=" * 92)

        # one fit per (N, seed); xi_needed is c-independent so all c reuse it
        needed, sig2, lam2s, n2s, beta_emp = {}, {}, {}, {}, {}
        for N in N_GRID:
            nd, sg, be = [], [], []
            for sd in SEEDS:
                blk, lam2, N2 = fit_block(p, N, sd, case["n_o"], case["n_o0"])
                nd.append(blk.xi_needed(b_true))
                sg.append(blk.sigma2)
                ev, evec = np.linalg.eigh(blk.H)
                mask = (ev - lam2) <= 1e-9 * max(ev[-1], 1.0)
                be.append(float(np.linalg.norm(evec[:, mask].T @ b_true)))
                lam2s[N], n2s[N] = lam2, N2
            needed[N], sig2[N], beta_emp[N] = nd, sg, float(np.mean(be))

        print(f"\n{'N':>9}{'N2':>9}{'sigma2':>11}{'beta_emp':>10}"
              f"{'xi_needed':>12}" + "".join(f"{'c='+str(c):>12}" for c in C_GRID))
        for N in N_GRID:
            xi_row = ""
            for c in C_GRID:
                xis = [c / (n2s[N] * max(s, 1e-12)) for s in sig2[N]]
                cov = sum(1 for x, nd in zip(xis, needed[N]) if x >= nd)
                xi_row += f"{str(cov)+'/'+str(len(SEEDS)):>12}"
            print(f"{N:>9}{n2s[N]:>9}{np.mean(sig2[N]):>11.3e}"
                  f"{beta_emp[N]:>10.4f}{np.mean(needed[N]):>12.3e}{xi_row}")

        print(f"\n{'c':>8}{'predicted N* (upper bd)':>26}{'measured crossing':>20}"
              f"{'ratio':>10}")
        case_rows = []
        for c in C_GRID:
            if beta_pop > 1e-10:
                n2s_pred = (c / (LAM0 * beta_pop ** 2 * s2p)) ** 2
                pred = n2s_pred / (1.0 - SPLIT)
            else:
                pred = float("inf")
            cross = None
            for N in N_GRID:
                xis = [c / (n2s[N] * max(s, 1e-12)) for s in sig2[N]]
                cov = sum(1 for x, nd in zip(xis, needed[N]) if x >= nd)
                if cov < len(SEEDS) - len(SEEDS) // 2:      # majority lost
                    cross = N
                    break
            ratio = (cross / pred) if (cross and np.isfinite(pred)) else float("nan")
            print(f"{c:>8}{(f'{pred:,.0f}' if np.isfinite(pred) else 'none'):>26}"
                  f"{(f'{cross:,}' if cross else 'none on grid'):>20}"
                  f"{ratio:>10.3f}")
            case_rows.append(dict(c=c, predicted_N=pred, measured_N=cross,
                                  ratio=ratio))

        # the c^2 signature: crossing should scale as c^2 across the sweep
        got = [(r["c"], r["measured_N"]) for r in case_rows if r["measured_N"]]
        if len(got) >= 2:
            cs = np.log([g[0] for g in got])
            ns = np.log([g[1] for g in got])
            slope = float(np.polyfit(cs, ns, 1)[0])
            print(f"\n   crossing scaling: d log N* / d log c = {slope:+.3f}   "
                  f"predicted +2.000   |err| {abs(slope-2):.3f}")
        else:
            slope = float("nan")
            print("\n   crossing scaling: not enough crossings to fit "
                  "(expected for the control)")

        all_rows.append(dict(tag=case["tag"], beta_pop=beta_pop, sigma2_pop=s2p,
                             rows=case_rows, c_slope=slope,
                             beta_emp={str(k): v for k, v in beta_emp.items()}))

    RESULTS["cases"] = all_rows
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_coverage_incomplete.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
