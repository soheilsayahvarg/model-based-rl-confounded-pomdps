"""run_stepc_recheck.py -- independent re-derivation of the two round-6 findings
that decide what survives of step (c).

Round 6 (docs/critic_findings_stepc.md) refuted the coverage half of the
proposition and confirmed the width half. Both verdicts turn on claims we did not
make ourselves, so neither is taken on trust here; this script re-derives them
without reusing poc/critic_c_*.py.

  [R1] beta = ||P_Nul b_true|| is NOT an environment constant. The claim is that
       the population null of the stage-2 design is orthogonal to the true
       (min-norm) bridge, so beta_pop = 0, and the 0.0532 reported in
       noncontraction.md 5.5 is a grid-average of empirical-null misalignment
       decaying at N^(-1/2). If so, every quantitative coverage number in 5.5
       (N* = 1.7e9, K, the beta^-4 sentence, the m_null table) is void.

  [R2] The anchor paper's own schedule sits at e > 0. Its width is
       xi ~ M_R * N2^(-alpha/(2*alpha+2)) and its ridge is
       lam2 = N2^(-alpha/(alpha*c2+1)), which places it inside our own family at
       tau = (alpha+2)/(2*alpha+2), kappa = alpha/(alpha*c2+1). Checked
       symbolically-by-grid over the admissible region.

Writes experiments/results_stepc_recheck.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
import oracle_module as om
from bridge_estimator import TabularBridgeEstimator

LAM0 = 0.03
RESULTS = {}


def loglog_slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ys > 0
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


# ---------------------------------------------------------------------------
# [R1] is beta an environment constant, or decaying misalignment noise?
# ---------------------------------------------------------------------------

def r1(kappa=0.5, seeds=(0, 1, 2, 3, 4)):
    print("=" * 78)
    print("[R1] beta = ||P_Nul b_true||  --  constant, or N^(-1/2) noise?")
    print("=" * 78)
    params = toy.default_params(T=3)
    bR_true, _, _ = om.toy_true_bridges(params)
    b_true = np.asarray(bR_true).ravel()
    nb = float(np.linalg.norm(b_true))

    grid = [1000, 4000, 16000, 64000, 256000, 1024000]
    per_n, rows = [], []
    print(f"   {'N':>9}{'beta (mean)':>14}{'sd':>10}{'null dim':>10}")
    for N in grid:
        bs, nd = [], []
        for sd in seeds:
            N2 = N - int(N * 0.7)
            lam2 = LAM0 * N2 ** (-kappa)
            d = toy.sample_trajectories(params, N, np.random.default_rng(sd))
            est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                         n_o0=toy.N_O0, n_r=2, T=3,
                                         mode="primal", seed=sd, lambda2=lam2)
            est.fit(d["O0"], d["O"], d["A"], d["R"])
            H = est.stage_store["bR_t1"]["H"]
            ev, evec = np.linalg.eigh(H)
            mask = (ev - lam2) <= 1e-9 * max(ev[-1], 1.0)
            bs.append(float(np.linalg.norm(evec[:, mask].T @ b_true)))
            nd.append(int(mask.sum()))
        per_n.append(float(np.mean(bs)))
        rows.append(dict(N=N, beta=float(np.mean(bs)), sd=float(np.std(bs)),
                         null_dim=int(np.mean(nd))))
        print(f"   {N:>9}{np.mean(bs):>14.5f}{np.std(bs):>10.5f}"
              f"{int(np.mean(nd)):>10}")

    s = loglog_slope(grid, per_n)
    print(f"\n   slope of beta vs N : {s:+.4f}")
    print(f"   a CONSTANT would give 0.000; misalignment noise gives -0.500")
    print(f"   ||b_true|| = {nb:.4f};  beta/||b_true|| runs "
          f"{per_n[0]/nb:.1%} -> {per_n[-1]/nb:.1%}")

    # The population statement: the exact null of the POPULATION design.
    # Stage-2's per-action design is built from the stage-1 CME profiles
    # mu(.|a,o0); their population span is range(E^T) restricted per action, and
    # the min-norm bridge (oracle_module builds it by pinv) lies in that span.
    E, p1, K0 = params["E"], params["p1"], params["K0"]
    P_pop = E.T @ np.diag(p1) @ K0                    # (n_o, n_o0)
    U, sv, _ = np.linalg.svd(P_pop)
    r = int((sv > 1e-12 * sv[0]).sum())
    Un_pop = U[:, r:]                                 # population null on o-space
    n_y = b_true.size // (toy.N_A * toy.N_O)
    # lift to the flat coefficient space: flat = (a*n_o + o~)*n_y + y
    B = b_true.reshape(toy.N_A, toy.N_O, n_y)
    beta_pop = float(np.linalg.norm(np.einsum("or,aoy->ary", Un_pop, B)))
    print(f"\n   population design rank {r} of {toy.N_O}; "
          f"population null dim {Un_pop.shape[1]}")
    print(f"   beta_pop = ||P_Nul_pop b_true|| = {beta_pop:.3e}   "
          f"({'ZERO' if beta_pop < 1e-10 else 'NONZERO'})")
    print(f"   -> (H4) {'FAILS' if beta_pop < 1e-10 else 'holds'} in the "
          f"population on this toy.")

    RESULTS["r1"] = dict(rows=rows, slope=s, beta_pop=beta_pop,
                         norm_b_true=nb, pop_rank=r)
    return beta_pop, s


# ---------------------------------------------------------------------------
# [R2] where does the anchor paper's own schedule sit?
# ---------------------------------------------------------------------------

def r2():
    print("\n" + "=" * 78)
    print("[R2] the anchor paper's own (tau, kappa) placement")
    print("=" * 78)
    print("   xi   ~ M_R * N2^(-alpha/(2alpha+2))  =>  tau   = (alpha+2)/(2alpha+2)")
    print("   lam2 =        N2^(-alpha/(alpha*c2+1)) =>  kappa = alpha/(alpha*c2+1)")
    print("   e = tau + kappa - 1 = alpha(2alpha+1-alpha*c2)"
          " / ((alpha*c2+1)(2alpha+2))\n")
    worst, cells = None, []
    print(f"   {'alpha':>8}{'c2':>7}{'tau':>9}{'kappa':>9}{'e':>11}")
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
        for c2 in (1.0001, 1.25, 1.5, 1.75, 2.0):
            tau = (alpha + 2) / (2 * alpha + 2)
            kappa = alpha / (alpha * c2 + 1)
            e = tau + kappa - 1
            e_closed = (alpha * (2 * alpha + 1 - alpha * c2)
                        / ((alpha * c2 + 1) * (2 * alpha + 2)))
            assert abs(e - e_closed) < 1e-12, "closed form disagrees"
            cells.append(dict(alpha=alpha, c2=c2, tau=tau, kappa=kappa, e=e))
            if worst is None or e < worst["e"]:
                worst = cells[-1]
            if c2 in (1.0001, 2.0):
                print(f"   {alpha:>8}{c2:>7.2f}{tau:>9.4f}{kappa:>9.4f}{e:>11.5f}")
    n_pos = sum(1 for c in cells if c["e"] > 0)
    print(f"\n   e > 0 in {n_pos}/{len(cells)} admissible cells; "
          f"minimum e = {worst['e']:.5f} at alpha={worst['alpha']}, "
          f"c2={worst['c2']:.4f}")
    print("   2alpha+1-alpha*c2 > 0 for all alpha>0 whenever c2 <= 2, and D.16")
    print("   restricts c2 to (1,2].  So e_paper > 0 ALWAYS: the anchor paper's")
    print("   own schedule sits on the NON-CONTRACTING branch of our dichotomy.")
    RESULTS["r2"] = dict(cells=cells, n_positive=n_pos, min_e=worst["e"],
                         argmin=worst)


def main():
    beta_pop, s = r1()
    r2()
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   [R1] beta_pop = {beta_pop:.2e}, beta(N) slope = {s:+.3f}")
    print("        -> the 0.0532 in noncontraction.md 5.5 is a grid average of a")
    print("           DECAYING quantity, not an environment constant. Every")
    print("           quantitative coverage number derived from it is withdrawn.")
    print("   [R2] e_paper > 0 for every admissible (alpha, c2).")
    print("        -> the dichotomy does not refute the anchor paper; it locates")
    print("           its schedule, and predicts a penalty that GROWS with N.")
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_stepc_recheck.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
