"""run_completeness_beta.py -- WHERE does (H4) hold? Population algebra, no sampling.

Round 6 refuted the coverage branch of step (c) on the Phase 3 toy by showing
beta_pop = ||P_Nul b_true|| is exactly zero there, and that the 0.0532 we reported
was empirical-null misalignment decaying at N^(-1/2). It attributed the zero to
the anchor paper's completeness assumption (3.3): K0 invertible forces the
min-norm bridge out of the population null.

That attribution predicts something specific and checkable WITHOUT any data:
beta_pop > 0 exactly when per-action completeness fails. This script tests it as
pure linear algebra, and does so BEFORE running the sampling experiment, so the
sampling grid can be aimed by a prediction rather than swept blindly.

THE ALGEBRA. The min-norm bridge solves E b = c, so every block of b_true lies in
range(E^T), of dimension rank(E) = |S|. The population stage-2 design for action a
is spanned by the stage-1 profiles

    mu(. | a, x)  =  E^T v_{a,x},     v_{a,x}[s] ∝ p1[s] * pi_b[s,a] * K0[s,x]

over the instrument values x in O_0. So the design's population range is a
SUBSPACE of range(E^T), and beta_pop > 0 iff that inclusion is strict:

    dim span{v_{a,x} : x}  <  rank(E) = |S|.

With an unconfounded behaviour policy, pi_b[:,a] is constant in s, so the v_{a,x}
inherit the rank of K0 and completeness (rank(K0) = |S|) makes the span all of
range(E^T) -- beta_pop = 0, which is the toy. With confound = 1, pi_b[:,a] is
one-hot in s, every v_{a,x} is supported on the SAME single state, the span
collapses to dimension 1, and for |S| >= 2 the inclusion is strict.

PREDICTION, fixed here before the measurement script exists:
    beta_pop = 0    whenever  per-action span dim = |S|   (completeness holds)
    beta_pop > 0    whenever  per-action span dim < |S|   (completeness fails)
and the transition is driven by `confound`, not by the observation dimensions.

Writes experiments/results_completeness_beta.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params

RESULTS = {}


def true_bridges_generic(params):
    """Min-norm true bridges for ANY shape. oracle_module.toy_true_bridges is the
    same construction with the toy's dimensions hard-coded as module constants."""
    E, P, pR = params["E"], params["P"], params["pR"]
    n_s, n_o = E.shape
    n_a = P.shape[0]
    Epinv = np.linalg.pinv(E)

    bR = np.zeros((n_a, n_o, 2, n_o))
    res = 0.0
    for a in range(n_a):
        for r in range(2):
            p_r = pR[:, a] if r == 1 else 1.0 - pR[:, a]
            for o in range(n_o):
                c = E[:, o] * p_r
                u = Epinv @ c
                bR[a, :, r, o] = u
                res = max(res, float(np.abs(E @ u - c).max()))
    return bR, res


def population_design_span(params, a, tol=1e-12):
    """Orthonormal basis of the population stage-2 design range for action a."""
    E, p1, K0, pi_b = params["E"], params["p1"], params["K0"], params["pi_b"]
    n_o0 = K0.shape[1]
    cols = []
    for x in range(n_o0):
        v = p1 * pi_b[:, a] * K0[:, x]          # unnormalised P(s | o0=x, a)
        if v.sum() <= 0:
            continue
        cols.append(E.T @ (v / v.sum()))        # mu(. | a, x) in R^{n_o}
    if not cols:
        return np.zeros((E.shape[1], 0))
    Mx = np.stack(cols, axis=1)
    U, sv, _ = np.linalg.svd(Mx, full_matrices=False)
    r = int((sv > tol * max(sv[0], 1e-300)).sum())
    return U[:, :r]


def beta_pop_for(params, a):
    """||P_Nul_pop b_true|| for action a's reward block, and the span dimension."""
    E = params["E"]
    n_o = E.shape[1]
    bR, res = true_bridges_generic(params)
    U = population_design_span(params, a)
    Pn = np.eye(n_o) - U @ U.T                  # projector onto the population null
    B = bR[a]                                   # (n_o, 2, n_o), first axis is o~
    leak = np.einsum("ij,jro->iro", Pn, B)
    return (float(np.linalg.norm(leak)), float(np.linalg.norm(B)),
            U.shape[1], res)


def main():
    print("=" * 88)
    print("Population beta: does (H4) hold exactly where per-action completeness")
    print("fails?  PREDICTION (fixed before any measurement):")
    print("    span dim = |S|  ->  beta_pop = 0        (completeness holds)")
    print("    span dim <  |S| ->  beta_pop > 0        (completeness fails)")
    print("=" * 88)

    configs = [(2, 6, 4), (2, 3, 2), (4, 6, 2), (3, 7, 5), (2, 6, 2)]
    confounds = [0.0, 0.3, 0.6, 0.9, 1.0]
    rows = []
    print(f"\n{'(|S|,|O|,|O0|)':>16}{'confound':>10}{'act':>5}{'span dim':>10}"
          f"{'|S|':>5}{'complete':>10}{'beta_pop':>12}{'rel':>9}")
    for (n_s, n_o, n_o0) in configs:
        for cf in confounds:
            p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=3,
                               seed=0, confound=cf)
            for a in range(2):
                bp, nb, dim, res = beta_pop_for(p, a)
                complete = dim >= n_s
                ok = (bp < 1e-10) == complete
                flag = "" if ok else "   <-- PREDICTION VIOLATED"
                print(f"{str((n_s,n_o,n_o0)):>16}{cf:>10.1f}{a:>5}{dim:>10}"
                      f"{n_s:>5}{str(complete):>10}{bp:>12.3e}{bp/nb:>9.1%}{flag}")
                rows.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf, act=a,
                                 span_dim=dim, complete=bool(complete),
                                 beta_pop=bp, rel=bp / nb, bridge_residual=res,
                                 prediction_holds=bool(ok)))

    bad = [r for r in rows if not r["prediction_holds"]]
    print(f"\nprediction holds in {len(rows)-len(bad)}/{len(rows)} cells")
    if bad:
        print("VIOLATIONS:")
        for r in bad:
            print("   ", r)
    else:
        print("(H4) holds exactly where per-action completeness fails. "
              "The round-6 attribution is confirmed as population algebra.")

    max_res = max(r["bridge_residual"] for r in rows)
    print(f"max bridge residual over all configs: {max_res:.2e} "
          f"(min-norm solve is exact)")

    cand = [r for r in rows if r["beta_pop"] > 1e-10]
    if cand:
        best = max(cand, key=lambda r: r["rel"])
        print(f"\nlargest null share: {best['rel']:.1%} at "
              f"(|S|,|O|,|O0|)={best['n_s'],best['n_o'],best['n_o0']}, "
              f"confound={best['confound']}, action {best['act']}, "
              f"beta_pop={best['beta_pop']:.4f}")
        print("-> this is the operating point at which to run the part C "
              "coverage test, which has no confirmed instance yet.")

    RESULTS["rows"] = rows
    RESULTS["n_violations"] = len(bad)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_completeness_beta.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
