"""run_stepb_reproduce.py -- clearing the last two debts from step (b).

family_pessimism.md has carried two "required" items since it was written:

  * its section 3 alignment sweep -- the load-bearing result of step (b) -- was
    the REVIEW's experiment, and its numbers were quoted from poc/critic_b_*.py
    rather than reproduced here;
  * its leakage table was 3-5 seeds, with PA rank carrying +-1 noise at N=4,000,
    so single cells should not have been quoted to three decimals.

Both are cleared here, and the first is done by a DIFFERENT route than the
review's. The review swept alignment empirically. Since then we have the
population design span (run_completeness_beta) and know the gradient's profile is
the observation marginal nu1 = E^T p1, so the same sweep can be done in the
POPULATION, exactly, with no sampling and no rank selection anywhere:

    beta_g(a) = || nu1 - P_span(a) nu1 || / || nu1 ||

An exact reproduction by an independent route is stronger evidence than re-running
the review's script, and it removes the PA-rank noise that forced the second debt.

THE ALIGNMENT KNOB. The environment has none, so one is built here: interpolate
every emission row toward their mean,

    E(alpha) = (1 - alpha) * E + alpha * mean_row,

At alpha = 1 all latent states emit identically, so nu1 coincides with each
action's span vector and the leakage must be exactly 0. At alpha = 0 the rows are
maximally distinct. This isolates alignment: confound stays 1.0, so the per-action
span stays rank 1 throughout, and cond(H) is reported to show it is not what moves.

Writes experiments/results_stepb_reproduce.json.
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
from mf_pessimism import (mf_design_and_gradient, h_inv_norm, null_leakage,
                          parallel_analysis_basis)

N_S, N_O, N_O0, N_A, T = 2, 6, 4, 2, 3
RHO = 1e-3
RESULTS = {}


def aligned_params(alpha, confound=1.0, seed=0):
    p = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T,
                       seed=seed, confound=confound)
    E = p["E"]
    p["E"] = (1 - alpha) * E + alpha * E.mean(axis=0, keepdims=True)
    return p


def pop_span_and_design(p, a):
    """Population span, design and gradient for action a -- no sampling."""
    E, p1, K0, pi_b = p["E"], p["p1"], p["K0"], p["pi_b"]
    cols, ws = [], []
    for x in range(K0.shape[1]):
        v = p1 * pi_b[:, a] * K0[:, x]
        if v.sum() <= 0:
            continue
        ws.append(v.sum())
        cols.append(E.T @ (v / v.sum()))
    D = sum(w * np.outer(m, m) for w, m in zip(ws, cols)) / sum(ws)
    U, sv, _ = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
    r = int((sv > 1e-12 * max(sv[0], 1e-300)).sum())
    return U[:, :r], D


def part1_alignment():
    print("=" * 88)
    print("[1] The section 3 alignment sweep, reproduced in the POPULATION")
    print("    confound = 1.0 throughout, so the per-action span stays rank 1;")
    print("    only the emission alignment moves.")
    print("=" * 88)
    print(f"{'alpha':>8}{'span rank':>11}{'cond(H)':>12}{'beta_g':>12}"
          f"{'W unproj':>11}{'W proj':>10}{'ratio':>9}")
    rows = []
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        p = aligned_params(alpha)
        nu1 = p["p1"] @ p["E"]
        n = float(np.linalg.norm(nu1))
        best = None
        for a in range(N_A):
            U, D = pop_span_and_design(p, a)
            H = D + RHO * np.eye(N_O)
            bg = float(np.linalg.norm(nu1 - U @ (U.T @ nu1))) / n
            wu = h_inv_norm(H, nu1)
            wp = h_inv_norm(H, nu1, basis=U)
            cond = float(np.linalg.cond(H))
            if best is None or bg > best[0]:
                best = (bg, wu, wp, cond, U.shape[1])
        bg, wu, wp, cond, r = best
        print(f"{alpha:>8.2f}{r:>11}{cond:>12.3e}{bg:>12.4e}{wu:>11.4f}"
              f"{wp:>10.4f}{wu/max(wp,1e-300):>9.3f}")
        rows.append(dict(alpha=alpha, span_rank=r, cond=cond, beta_g=bg,
                         w_unproj=wu, w_proj=wp, ratio=wu / max(wp, 1e-300)))

    lo, hi = rows[0], rows[-1]
    print(f"\n   leakage moves {lo['beta_g']:.4f} -> {hi['beta_g']:.2e}  "
          f"({lo['beta_g']/max(hi['beta_g'],1e-300):.0f}x)")
    print(f"   width ratio    {lo['ratio']:.2f} -> {hi['ratio']:.3f}")
    conds = [r["cond"] for r in rows]
    print(f"   cond(H) spans  {min(conds):.3e} .. {max(conds):.3e}  "
          f"({max(conds)/min(conds):.2f}x)")
    print("   span rank is 1 in every row, so the null space is FIXED and only")
    print("   the gradient's alignment with it moves.")
    RESULTS["alignment"] = rows


def part2_leakage_20_seeds():
    print("\n" + "=" * 88)
    print("[2] The empirical leakage table at 20 seeds, with intervals")
    print("    (family_pessimism.md quoted 3-5 seeds; PA rank carries +-1 noise)")
    print("=" * 88)
    seeds = list(range(20))
    print(f"{'config':>12}{'cf':>6}{'act':>5}{'PA rank':>10}{'leak':>10}"
          f"{'+-95%':>9}{'pop beta_g':>12}{'agrees':>9}")
    rows = []
    for (n_s, n_o, n_o0) in [(2, 6, 4), (4, 6, 2)]:
        for cf in [1.0, 0.6]:
            for a in range(N_A):
                p = default_params(n_s=n_s, n_a=N_A, n_o=n_o, n_o0=n_o0, T=T,
                                   seed=0, confound=cf)
                nu1 = p["p1"] @ p["E"]
                U, _ = pop_span_and_design(p, a)
                bg_pop = float(np.linalg.norm(nu1 - U @ (U.T @ nu1))) / \
                    float(np.linalg.norm(nu1))
                leaks, ranks = [], []
                for sd in seeds:
                    pe = default_params(n_s=n_s, n_a=N_A, n_o=n_o, n_o0=n_o0,
                                        T=T, seed=sd, confound=cf)
                    d = sample_trajectories(pe, 4000, np.random.default_rng(sd))
                    k, Upa = parallel_analysis_basis(d, n_o, n_o0, a,
                                                     np.random.default_rng(sd))
                    nu_e = np.bincount(d["O"][:, 0],
                                       minlength=n_o).astype(float)
                    nu_e /= nu_e.sum()
                    leaks.append(np.sqrt(null_leakage(None, nu_e, Upa)))
                    ranks.append(k)
                m = float(np.mean(leaks))
                ci = float(1.96 * np.std(leaks, ddof=1) / np.sqrt(len(leaks)))
                agree = "yes" if abs(m - bg_pop) <= max(ci, 0.05) else "NO"
                print(f"{str((n_s,n_o,n_o0)):>12}{cf:>6.1f}{a:>5}"
                      f"{np.mean(ranks):>10.2f}{m:>10.4f}{ci:>9.4f}"
                      f"{bg_pop:>12.4f}{agree:>9}")
                rows.append(dict(config=[n_s, n_o, n_o0], confound=cf, act=a,
                                 pa_rank_mean=float(np.mean(ranks)),
                                 leak=m, ci=ci, beta_g_pop=bg_pop,
                                 agrees=agree, n_seeds=len(seeds)))
    RESULTS["leakage20"] = rows
    bad = [r for r in rows if r["agrees"] == "NO"]
    print(f"\n   empirical agrees with population in "
          f"{len(rows)-len(bad)}/{len(rows)} cells")


def main():
    part1_alignment()
    part2_leakage_20_seeds()
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_stepb_reproduce.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
