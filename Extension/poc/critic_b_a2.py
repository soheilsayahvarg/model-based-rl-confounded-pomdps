"""critic_b_a2.py -- ATTACK A2 (and A6): is gradient leakage CAUSAL at fixed rank?

C3's evidence moves `confound`, but that knob moves rank(P_a) and leakage
together, so leakage could be a passenger. Here the confound stays at 1.0 and
rank(P_a) stays [1, 1]; only the ALIGNMENT between the value gradient nu1 and
the per-action identified subspace is varied, by interpolating the second
emission row toward the first:

    E_beta = [E0, normalize((1 - beta) * E0 + beta * E1)],  beta in (0, 1]

As beta -> 0 the two emissions converge, so nu1 = p1(0) E0 + p1(1) E1_beta
approaches the (single) identified direction of BOTH actions: leakage -> 0 with
rank, confound, |O|, |O_0|, K0, transitions, rewards, policy all held fixed.
E_beta keeps full row rank for every beta > 0.

If C3 is causal, width ratio and V_low must track leakage along beta. If they
stay put, C3 is an artifact of the confound knob. Either way this tests C4's
step 4-5 directly (A6): coverage/confounding is CONSTANT along beta, so if the
penalty moves, "poor coverage IS the penalty" is too strong.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import (default_params, sample_trajectories, dp_value,
                                 candidate_policies)
from model_free_proximal import MinimaxValueBridgeOPE
from mf_pessimism import (mf_design_and_gradient, h_inv_norm, null_leakage,
                          parallel_analysis_basis)

N = 4000
SEEDS = [0, 1, 2, 3, 4]
N_ACT = 2
T = 3
C = 10.0
BETAS = [1.0, 0.5, 0.25, 0.1, 0.02]


def variant_params(beta):
    p = default_params(n_s=2, n_a=N_ACT, n_o=6, n_o0=4, T=T, seed=0,
                       confound=1.0)
    E = p["E"].copy()
    row1 = (1.0 - beta) * E[0] + beta * E[1]
    E[1] = row1 / row1.sum()
    p["E"] = E
    assert np.linalg.matrix_rank(E) == 2
    return p


def pop_rank(p, a):
    return int(np.linalg.matrix_rank(
        p["E"].T @ np.diag(p["p1"] * p["pi_b"][:, a]) @ p["K0"]))


def main():
    print("[A2] Emission-alignment sweep at confound=1.0, (2,6,4). rank(P_a) and"
          " coverage are FIXED; only gradient/subspace alignment moves.\n")
    print(f"{'beta':>6}{'rank(P_a)':>10}{'cond(H)':>11}{'leak':>9}"
          f"{'w_unproj':>10}{'w_proj':>9}{'ratio':>8}{'V_true':>8}"
          f"{'V_hat':>8}{'Vlow_u':>9}")
    pi_name = "always_0"
    for beta in BETAS:
        p = variant_params(beta)
        ranks = [pop_rank(p, a) for a in range(N_ACT)]
        pi = candidate_policies(6, N_ACT, seed=0)[pi_name]
        v_true = dp_value(p, pi)
        conds, leaks, wu, wp, vh, vlu = [], [], [], [], [], []
        for sd in SEEDS:
            d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
            nu1 = np.bincount(d["O"][:, 0], minlength=6).astype(float)
            nu1 /= nu1.sum()
            est = MinimaxValueBridgeOPE(n_obs=6, n_act=N_ACT, n_o0=4, T=T,
                                        seed=sd)
            est.fit(d["O0"], d["O"], d["A"], d["R"], pi)
            Hs, gs = mf_design_and_gradient(d, 6, 4, N_ACT, nu1)
            pen = 0.0
            for a in range(N_ACT):
                k, U = parallel_analysis_basis(
                    d, 6, 4, a, np.random.default_rng(900 + sd))
                ev = np.linalg.eigvalsh(Hs[a])
                conds.append(ev[-1] / max(ev[0], 1e-300))
                leaks.append(null_leakage(Hs[a], gs[a], U))
                nu_ = h_inv_norm(Hs[a], gs[a])
                np_ = h_inv_norm(Hs[a], gs[a], U)
                wu.append(nu_)
                wp.append(np_)
                pen += np.sqrt(C / (N * max(ev[0], 1e-300))) * nu_
            vh.append(est.value())
            vlu.append(est.value() - pen)
        ratio = np.mean(wu) / max(np.mean(wp), 1e-300)
        print(f"{beta:>6.2f}{str(ranks):>10}{np.median(conds):>11.2e}"
              f"{np.mean(leaks):>9.4f}{np.mean(wu):>10.3f}{np.mean(wp):>9.3f}"
              f"{ratio:>8.2f}{v_true:>8.3f}{np.mean(vh):>8.3f}"
              f"{np.mean(vlu):>9.2f}")

    print("\n[A2-reverse] p1-skew variant at beta=1: skewing the initial"
          " distribution moves per-action alignment asymmetrically.\n")
    print(f"{'p1':>14}{'act':>4}{'leak':>9}{'w_unproj':>10}")
    for p1v in ([0.5, 0.5], [0.85, 0.15], [0.15, 0.85]):
        p = default_params(n_s=2, n_a=N_ACT, n_o=6, n_o0=4, T=T, seed=0,
                           confound=1.0)
        p["p1"] = np.array(p1v)
        leak_a = {a: [] for a in range(N_ACT)}
        wu_a = {a: [] for a in range(N_ACT)}
        for sd in SEEDS:
            d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
            nu1 = np.bincount(d["O"][:, 0], minlength=6).astype(float)
            nu1 /= nu1.sum()
            Hs, gs = mf_design_and_gradient(d, 6, 4, N_ACT, nu1)
            for a in range(N_ACT):
                k, U = parallel_analysis_basis(
                    d, 6, 4, a, np.random.default_rng(900 + sd))
                leak_a[a].append(null_leakage(Hs[a], gs[a], U))
                wu_a[a].append(h_inv_norm(Hs[a], gs[a]))
        for a in range(N_ACT):
            print(f"{str(p1v):>14}{a:>4}{np.mean(leak_a[a]):>9.4f}"
                  f"{np.mean(wu_a[a]):>10.3f}")


if __name__ == "__main__":
    main()
