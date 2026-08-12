"""run_noncontraction.py -- verification of the step (c) proposition.

The proposition (docs/noncontraction.md) was committed BEFORE this script existed,
with its predictions filled in and its measurement section empty. Nothing here is
exploratory: every number produced is compared against a value that was written
down first. That ordering is the point -- of everything this extension produced,
the two results that survived five adversarial review rounds were both derived
analytically before they were measured, and the ones that were withdrawn were not.

WHAT IS PREDICTED. With lam = lam0*N^-kappa and xi = c/(N*mu), mu = m*N^-tau, the
governing exponent is e = tau + kappa - 1, and

    [P1] width slope    d log W / d log N = e/2
    [P2] projected slope                  = (tau-1)/2      (null directions removed)
    [P3] coverage fails for N > N* = K^(1/e),  K = lam0*beta^2*m/c   (when e < 0)

Part A tests [P1]/[P2] on the model-free layer (global convention, tau = kappa).
Part B tests [P1] on the model-based layer (signal convention, tau = 0).
Part C tests [P3], and the two exact structural claims the proof rests on:
lam_min(H) = lam exactly, and P_Nul b_hat = 0 exactly.

Writes experiments/results_noncontraction.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params, sample_trajectories
from mf_pessimism import mf_design_and_gradient, h_inv_norm, parallel_analysis_basis
import toy_pomdp as toy
import oracle_module as om
from bridge_estimator import TabularBridgeEstimator
from ellipsoid_opt import BlockEllipsoid

LAM0 = 0.03                                  # the implemented default
N_GRID = [4000, 16000, 64000, 256000]        # 64x range, as in family_pessimism §4
SEEDS = [0, 1, 2]
C_WIDTH = 1.0
RESULTS = {}


def loglog_slope(ns, ws):
    ns, ws = np.asarray(ns, float), np.asarray(ws, float)
    ok = ws > 0
    return float(np.polyfit(np.log(ns[ok]), np.log(ws[ok]), 1)[0])


# --------------------------------------------------------------------------
# Part A -- model-free layer, GLOBAL convention (mu = lam_min(H), so tau = kappa)
# --------------------------------------------------------------------------

def part_a():
    print("=" * 78)
    print("PART A  model-free layer, global convention (tau = kappa), config (2,6,4)")
    print("        predicted width slope = e/2 = (2*kappa-1)/2")
    print("        predicted proj. slope = (tau-1)/2 = (kappa-1)/2")
    print("=" * 78)
    n_s, n_o, n_o0, n_act, T = 2, 6, 4, 2, 3
    rows, h1_err = [], []
    for kappa in (0.25, 0.5, 0.75):
        wu_by_n, wp_by_n = [], []
        for N in N_GRID:
            wu, wp = [], []
            for sd in SEEDS:
                p = default_params(n_s=n_s, n_a=n_act, n_o=n_o, n_o0=n_o0,
                                   T=T, seed=sd, confound=1.0)
                rng = np.random.default_rng(1000 + sd)
                d = sample_trajectories(p, N, rng)
                nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float) / N
                lam = LAM0 * N ** (-kappa)
                Hs, gs = mf_design_and_gradient(d, n_o, n_o0, n_act, nu1,
                                                t=1, ridge=lam)
                for a in range(n_act):
                    ev = np.linalg.eigvalsh(Hs[a])
                    # (H1): the structural null makes lam_min(H) EXACTLY the ridge
                    h1_err.append(abs(ev[0] - lam) / lam)
                    xi = C_WIDTH / (N * ev[0])
                    wu.append(np.sqrt(xi) * h_inv_norm(Hs[a], gs[a]))
                    _, U = parallel_analysis_basis(d, n_o, n_o0, a, rng, t=1)
                    wp.append(np.sqrt(xi) * h_inv_norm(Hs[a], gs[a], basis=U))
            wu_by_n.append(float(np.mean(wu)))
            wp_by_n.append(float(np.mean(wp)))
        s_u, s_p = loglog_slope(N_GRID, wu_by_n), loglog_slope(N_GRID, wp_by_n)
        pred_u, pred_p = (2 * kappa - 1) / 2, (kappa - 1) / 2
        print(f"\n kappa={kappa}   e = {2*kappa-1:+.2f}")
        print(f"   {'N':>9}{'W unproj':>12}{'W proj':>12}")
        for N, a_, b_ in zip(N_GRID, wu_by_n, wp_by_n):
            print(f"   {N:>9}{a_:>12.4f}{b_:>12.4f}")
        print(f"   slope unproj  measured {s_u:+.4f}   predicted {pred_u:+.4f}"
              f"   |err| {abs(s_u-pred_u):.4f}")
        print(f"   slope proj    measured {s_p:+.4f}   predicted {pred_p:+.4f}"
              f"   |err| {abs(s_p-pred_p):.4f}")
        rows.append(dict(kappa=kappa, e=2 * kappa - 1, W=wu_by_n, W_proj=wp_by_n,
                         slope=s_u, pred=pred_u, slope_proj=s_p, pred_proj=pred_p))
    print(f"\n (H1) max relative |lam_min(H) - lam| / lam over all fits: "
          f"{max(h1_err):.3e}")
    RESULTS["part_a"] = rows
    RESULTS["h1_max_rel_err"] = float(max(h1_err))


# --------------------------------------------------------------------------
# Part B -- model-based layer, SIGNAL convention (mu = sigma2_signal, tau = 0)
# --------------------------------------------------------------------------

def fit_toy(N, kappa, sd, T=3):
    params = toy.default_params(T=T)
    N2 = N - int(N * 0.7)
    lam2 = LAM0 * N2 ** (-kappa)
    d = toy.sample_trajectories(params, N, np.random.default_rng(sd))
    est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A, n_o0=toy.N_O0,
                                 n_r=2, T=T, mode="primal", seed=sd,
                                 lambda2=lam2)
    est.fit(d["O0"], d["O"], d["A"], d["R"])
    s = est.stage_store["bR_t1"]
    blk = BlockEllipsoid(s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"],
                         "bR_t1", s["signal_basis"], n_obs=toy.N_O,
                         n_act=toy.N_A, n_y=s["n_y"])
    return params, blk, lam2, N2


def part_b():
    print("\n" + "=" * 78)
    print("PART B  model-based layer, signal convention (tau = 0), toy POMDP")
    print("        predicted width slope = e/2 = (kappa-1)/2")
    print("=" * 78)
    # A fixed direction is enough: W = sqrt(xi)*||g||_{H^-1} and the proposition is
    # about the N-dependence of that scalar, not about which g.
    rng0 = np.random.default_rng(7)
    rows = []
    for kappa in (0.25, 0.5, 1.0):
        w_by_n = []
        g_fixed = None
        for N in N_GRID:
            ws = []
            for sd in SEEDS:
                _, blk, lam2, N2 = fit_toy(N, kappa, sd)
                if g_fixed is None:
                    g_fixed = rng0.standard_normal(blk.b_hat.size)
                    g_fixed /= np.linalg.norm(g_fixed)
                xi = blk.width_rule(C_WIDTH)         # = c / (N2 * sigma2_signal)
                ws.append(np.sqrt(xi) * blk.h_inv_norm(g_fixed))
            w_by_n.append(float(np.mean(ws)))
        s = loglog_slope(N_GRID, w_by_n)
        pred = (kappa - 1) / 2
        print(f"\n kappa={kappa}   e = {kappa-1:+.2f}")
        print(f"   {'N':>9}{'W':>12}")
        for N, a_ in zip(N_GRID, w_by_n):
            print(f"   {N:>9}{a_:>12.4f}")
        print(f"   slope  measured {s:+.4f}   predicted {pred:+.4f}"
              f"   |err| {abs(s-pred):.4f}")
        rows.append(dict(kappa=kappa, e=kappa - 1, W=w_by_n, slope=s, pred=pred))
    RESULTS["part_b"] = rows


# --------------------------------------------------------------------------
# Part C -- the coverage crossing, at Phase 3's own operating point
# --------------------------------------------------------------------------

def part_c(c_width=0.03, kappa=0.5):
    print("\n" + "=" * 78)
    print(f"PART C  coverage crossing, signal convention, kappa={kappa}, "
          f"c={c_width} (Phase 3's own c)")
    print("        predicted N2* = ( c / (lam0 * beta^2 * sigma2) )^2")
    print("=" * 78)
    params = toy.default_params(T=3)
    bR_true, _, _ = om.toy_true_bridges(params)
    b_true = np.asarray(bR_true).ravel()

    grid = [1000, 4000, 16000, 64000, 256000, 1024000]
    rows, betas, sig2s, pnb = [], [], [], []
    print(f"   {'N':>9}{'N2':>8}{'lam2':>11}{'sigma2':>11}"
          f"{'xi':>12}{'xi_needed':>12}{'covered':>9}")
    for N in grid:
        xi_l, need_l, cov_l = [], [], []
        for sd in SEEDS:
            _, blk, lam2, N2 = fit_toy(N, kappa, sd)
            # null space of T2 = H - lam2*I : eigenvalues of H sitting at lam2
            ev, evec = np.linalg.eigh(blk.H)
            null_mask = (ev - lam2) <= 1e-9 * max(ev[-1], 1.0)
            Un = evec[:, null_mask]
            betas.append(float(np.linalg.norm(Un.T @ b_true)))
            pnb.append(float(np.linalg.norm(Un.T @ blk.b_hat)))
            sig2s.append(blk.sigma2)
            xi = blk.width_rule(c_width)
            need = blk.xi_needed(b_true)
            xi_l.append(xi)
            need_l.append(need)
            cov_l.append(xi >= need)
        xi_m, need_m = float(np.mean(xi_l)), float(np.mean(need_l))
        cov = f"{sum(cov_l)}/{len(cov_l)}"
        print(f"   {N:>9}{N2:>8}{lam2:>11.3e}{np.mean(sig2s[-len(SEEDS):]):>11.3e}"
              f"{xi_m:>12.4e}{need_m:>12.4e}{cov:>9}")
        rows.append(dict(N=N, N2=N2, lam2=lam2, xi=xi_m, xi_needed=need_m,
                         covered=int(sum(cov_l)), n_seeds=len(cov_l)))

    beta = float(np.mean(betas))
    sigma2 = float(np.mean(sig2s))
    K = LAM0 * beta ** 2 * sigma2 / c_width
    e = kappa - 1.0
    N2_star = K ** (1.0 / e)
    N_star = N2_star / 0.3
    print(f"\n   beta = ||P_Nul b_true|| = {beta:.4f}   "
          f"(P_Nul b_hat = {max(pnb):.3e}, predicted exactly 0)")
    print(f"   sigma2 (mean over grid) = {sigma2:.4e}   K = {K:.4e}   e = {e:+.2f}")
    print(f"   PREDICTED crossing at N2* = {N2_star:,.0f}  (N* = {N_star:,.0f})")
    first_fail = next((r["N"] for r in rows if r["covered"] < r["n_seeds"]), None)
    print(f"   MEASURED  coverage first fails at N = "
          f"{first_fail if first_fail else 'not on this grid'}")

    # ---------------------------------------------------------------
    # The test above is VACUOUS: the grid stops 1600x short of N*, so it could
    # not have produced a failure. That is precisely the defect we withdrew
    # evidence for in family_pessimism.md 8.3, and we are not repeating it.
    #
    # What IS falsifiable on a reachable grid is the RATE. The proposition's
    # mechanism is that the null-direction coverage margin
    #
    #     m_null(N) := xi(N) / (lam(N) * beta^2)
    #
    # degrades as N^e, and coverage in those directions fails when it reaches 1.
    # The exponent is testable here even though the absolute level is not.
    # ---------------------------------------------------------------
    m_null = [r["xi"] / (r["lam2"] * beta ** 2) for r in rows]
    s_m = loglog_slope([r["N"] for r in rows], m_null)
    print(f"\n   null-direction margin m_null = xi / (lam * beta^2), "
          f"predicted slope {e:+.2f}")
    print(f"   {'N':>9}{'m_null':>12}")
    for r, m in zip(rows, m_null):
        print(f"   {r['N']:>9}{m:>12.1f}")
    print(f"   slope  measured {s_m:+.4f}   predicted {e:+.4f}   "
          f"|err| {abs(s_m-e):.4f}")

    # The margin starts at ~1e3, so reaching 1 needs ~1e6 more data. The reason
    # is that beta is small here (3% of ||b_true||), and N* scales as beta^-4:
    # the threshold is a property of the ENVIRONMENT, not of the construction.
    nb = float(np.linalg.norm(b_true))
    print(f"\n   ||b_true|| = {nb:.4f}, so the null share is "
          f"{beta/nb:.1%} -- (H4) holds but barely.")
    print(f"   N2* scales as beta^-4: an environment with 10x the null share "
          f"has N2* = {N2_star*1e-4:,.0f}.")

    # The signal directions also demand width, and at these N they DOMINATE
    # xi_needed. Retaining that term sharpens the threshold and adds a floor on c.
    big = rows[-1]
    C_sig = max(big["xi_needed"] - big["lam2"] * beta ** 2, 0.0) * big["N2"]
    c_floor = sigma2 * C_sig
    print(f"\n   signal term of xi_needed ~ C/N2 with C = {C_sig:.4f}")
    print(f"   at N = {big['N']:,} the null term is {big['lam2']*beta**2:.3e} "
          f"vs signal {C_sig/big['N2']:.3e} -- signal still dominates "
          f"{C_sig/big['N2']/(big['lam2']*beta**2):.0f}x")
    print(f"   retaining it: N2* = ((c/sigma2 - C)/(lam0*beta^2))^2, which also "
          f"implies a FLOOR c > sigma2*C = {c_floor:.4f}")
    print(f"   Phase 3 runs at c = 0.03, only {0.03/c_floor:.1f}x above that "
          f"floor; below it the region fails to cover at EVERY N.")

    RESULTS["part_c"] = dict(rows=rows, beta=beta, sigma2=sigma2, K=K,
                             N2_star=N2_star, N_star=N_star,
                             first_fail_N=first_fail,
                             max_P_null_b_hat=float(max(pnb)),
                             m_null=m_null, m_null_slope=s_m, m_null_pred=e,
                             norm_b_true=nb, null_share=beta / nb,
                             C_sig=C_sig, c_floor=c_floor,
                             c=c_width, kappa=kappa)


def main():
    want = set(sys.argv[1:]) or {"a", "b", "c"}
    if "a" in want:
        part_a()
    if "b" in want:
        part_b()
    if "c" in want:
        part_c()
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_noncontraction.json"))
    if want != {"a", "b", "c"} and os.path.exists(out):
        with open(out) as f:                      # keep parts not re-run
            RESULTS.update({k: v for k, v in json.load(f).items()
                            if k not in RESULTS})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
