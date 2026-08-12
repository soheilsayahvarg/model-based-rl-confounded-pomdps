"""run_family_pessimism.py -- step (b): is the pessimism blow-up shared by the family?

THE QUESTION. Phase 4 found that the anchor paper's model-based pessimism diverges,
and traced it to confidence-region directions the data cannot constrain. If that is
geometric rather than specific to one estimator, the model-free proximal method --
different loss, different solver, different bridge -- should fail the same way.

THE TEST. For a linear value functional the pessimistic value has a closed form,

    V_low = V(b_hat) - sqrt(xi) * ||g||_{H^-1}

so divergence needs BOTH an ill-conditioned H and a value gradient g that points
into the ill-conditioned directions. Reporting only the conditioning of H would
prove nothing: a null space the value never looks into is harmless. So we measure

    [B1] the conditioning of H per action,
    [B2] the leakage of g into the unconstrained subspace,
    [B3] the resulting width, unprojected vs projected,
    [B4] whether V_low stays below V_true (validity) and by how much.

REGIMES. Both are run, deliberately. Restricting to the regime where the cliff is
sharpest would bias the comparison toward finding the blow-up we expect:

    (4,6,2)  rank(P_a) saturates min(|O|,|O_0|)  -- exact cliff
    (2,6,4)  does not saturate                   -- statistical deficiency

RANK SELECTION uses parallel analysis, not the eigengap rule: the eigengap rule
needs N in the millions in the non-saturating regime, so at these sample sizes it
would manufacture exactly the artifact this experiment is trying to observe.

Writes experiments/results_family_pessimism.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import (default_params, sample_trajectories, dp_value,
                                 candidate_policies, population_cross_moment)
from model_free_proximal import MinimaxValueBridgeOPE
from mf_pessimism import (mf_design_and_gradient, h_inv_norm, null_leakage,
                          parallel_analysis_basis)

CONFIGS = [
    (4, 6, 2, "saturating (exact cliff)"),
    (2, 6, 4, "non-saturating (statistical)"),
    (3, 7, 5, "non-saturating (statistical)"),
]
CONFOUNDS = [1.0, 0.6]
N_MAIN = 4000            # the sample size Phase 4 actually uses
SEEDS = [0, 1, 2, 3, 4]
N_ACT = 2
T = 3
XI_C = [0.1, 1.0, 10.0]  # width calibration constants
RESULTS = {}


def pop_rank_per_action(p, a):
    return int(np.linalg.matrix_rank(
        p["E"].T @ np.diag(p["p1"] * p["pi_b"][:, a]) @ p["K0"]))


def main():
    print("[B1-B3] Model-free confidence region: conditioning, gradient leakage, width.\n")
    print(f"{'config':>10}{'cf':>5}{'act':>4}{'rank(P_a)':>10}{'PA rank':>9}"
          f"{'cond(H)':>12}{'leak(g)':>10}{'w_unproj':>11}{'w_proj':>10}{'ratio':>10}")
    b1 = []
    for (n_s, n_o, n_o0, note) in CONFIGS:
        for cf in CONFOUNDS:
            for a in range(N_ACT):
                conds, leaks, wu, wp, pas = [], [], [], [], []
                for sd in SEEDS:
                    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                       T=T, seed=sd, confound=cf)
                    d = sample_trajectories(p, N_MAIN, np.random.default_rng(500 + sd))
                    nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
                    nu1 /= nu1.sum()
                    Hs, gs = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu1)
                    k, U = parallel_analysis_basis(d, n_o, n_o0, a,
                                                   np.random.default_rng(900 + sd))
                    H, g = Hs[a], gs[a]
                    ev = np.linalg.eigvalsh(H)
                    conds.append(ev[-1] / max(ev[0], 1e-300))
                    leaks.append(null_leakage(H, g, U))
                    wu.append(h_inv_norm(H, g))
                    wp.append(h_inv_norm(H, g, U))
                    pas.append(k)
                p0 = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                    T=T, seed=0, confound=cf)
                tr = pop_rank_per_action(p0, a)
                row = dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf, action=a,
                           pop_rank=tr, pa_rank=float(np.mean(pas)),
                           cond_H=float(np.median(conds)),
                           leak=float(np.mean(leaks)),
                           w_unproj=float(np.mean(wu)),
                           w_proj=float(np.mean(wp)), note=note)
                row["ratio"] = row["w_unproj"] / max(row["w_proj"], 1e-300)
                b1.append(row)
                print(f"{str((n_s,n_o,n_o0)):>10}{cf:>5.1f}{a:>4}{tr:>10}"
                      f"{row['pa_rank']:>9.1f}{row['cond_H']:>12.2e}"
                      f"{row['leak']:>10.4f}{row['w_unproj']:>11.3f}"
                      f"{row['w_proj']:>10.3f}{row['ratio']:>10.2f}")
    RESULTS["B1_geometry"] = b1

    # ---------------- [B4] does V_low diverge, and does it stay valid? -------
    print("\n\n[B4] Model-free pessimistic value vs exact truth.")
    print("     V_low must satisfy V_low <= V_true. Divergence shows as a huge gap.\n")
    print(f"{'config':>10}{'cf':>5}{'c':>6}{'policy':>11}{'V_true':>9}"
          f"{'V_hat':>9}{'Vlow_unproj':>13}{'Vlow_proj':>11}   valid")
    b4 = []
    for (n_s, n_o, n_o0, note) in CONFIGS:
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            pols = candidate_policies(n_o, N_ACT, seed=0)
            for c in XI_C:
                for name, pi in list(pols.items())[:2]:
                    v_true = dp_value(p, pi)
                    vh, vlu, vlp = [], [], []
                    for sd in SEEDS:
                        d = sample_trajectories(p, N_MAIN, np.random.default_rng(500 + sd))
                        nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
                        nu1 /= nu1.sum()
                        est = MinimaxValueBridgeOPE(n_obs=n_o, n_act=N_ACT,
                                                    n_o0=n_o0, T=T, seed=sd)
                        est.fit(d["O0"], d["O"], d["A"], d["R"], pi)
                        v_hat = est.value()
                        Hs, gs = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu1)
                        pen_u = pen_p = 0.0
                        for a in range(N_ACT):
                            k, U = parallel_analysis_basis(
                                d, n_o, n_o0, a, np.random.default_rng(900 + sd))
                            xi = c / (N_MAIN * max(np.linalg.eigvalsh(Hs[a])[0], 1e-300))
                            pen_u += np.sqrt(xi) * h_inv_norm(Hs[a], gs[a])
                            pen_p += np.sqrt(xi) * h_inv_norm(Hs[a], gs[a], U)
                        vh.append(v_hat)
                        vlu.append(v_hat - pen_u)
                        vlp.append(v_hat - pen_p)
                    mvh, mvu, mvp = np.mean(vh), np.mean(vlu), np.mean(vlp)
                    valid = "OK" if (mvu <= v_true and mvp <= v_true) else "VIOLATED"
                    print(f"{str((n_s,n_o,n_o0)):>10}{cf:>5.1f}{c:>6.1f}{name:>11}"
                          f"{v_true:>9.3f}{mvh:>9.3f}{mvu:>13.3f}{mvp:>11.3f}   {valid}")
                    b4.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf, c=c,
                                   policy=name, v_true=float(v_true),
                                   v_hat=float(mvh), vlow_unproj=float(mvu),
                                   vlow_proj=float(mvp), valid=(valid == "OK")))
    RESULTS["B4_values"] = b4

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_family_pessimism.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
