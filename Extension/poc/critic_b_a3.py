"""critic_b_a3.py -- ATTACK A3: does later-stage error reach stage 1 through the response?

C2 says: "J depends on the first-stage bridge only, and linearly. There is no
chain to compound through." But fit() builds the stage-1 response as

    y = (r_1 + V_hat_2(o_2)) * pi_e(a_1 | o_1)

where V_hat_2 comes from the FITTED stage-2 bridge, which used the fitted
stage-3 bridge. J is linear in b_V^[1] GIVEN the response, but b_hat^[1] itself
carries every later stage's estimation error. The stage-1-only ellipsoid treats
the response as fixed.

TEST. Compute the POPULATION bridge (the estimator run on expected counts at
the same N-ridge, backward through all stages) to get the population
continuation V_pop_2. Then on the SAME 20 datasets fit stage 1 twice:

    (a) shipped: cont = V_hat_2 (estimated, as in fit())
    (b) surgical: cont = V_pop_2 (later-stage error removed)

Var(J_a - J_b) is the variance J inherits from later stages alone. If it is a
material share of Var(J_a), C2's "definitively excluded" is wrong as stated.

A replica of the stage-1 solve is validated against the estimator to machine
precision before being trusted.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "baselines"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import (default_params, sample_trajectories,
                                 candidate_policies, dp_value)
from model_free_proximal import MinimaxValueBridgeOPE

N = 4000
N_ACT = 2
T = 3
SEEDS = list(range(20))


def stage_solve(o_t, a_t, O0, y, n_o, n_o0, N_total):
    """Verbatim per-action GLS solve from MinimaxValueBridgeOPE.fit."""
    eps, rho = 1.0 / N_total, 0.03 / np.sqrt(N_total)
    bV = np.zeros((N_ACT, n_o))
    for a in range(N_ACT):
        sel = a_t == a
        if not sel.any():
            continue
        wa, xa, ya = o_t[sel], O0[sel], y[sel]
        Ga = np.bincount(xa, minlength=n_o0).astype(float)
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (wa, xa), 1.0)
        cXy = np.zeros(n_o0)
        np.add.at(cXy, xa, ya)
        Wa = 1.0 / (Ga + N_total * eps)
        AW = Acr * Wa[None, :]
        M = Acr.T @ AW + N_total * rho * np.eye(n_o0)
        rhs = AW @ cXy
        bV[a] = (rhs - AW @ np.linalg.solve(M, Acr.T @ rhs)) / (N_total * rho)
    return bV


def population_bridge(p, pi_e, n_o, n_o0):
    """The estimator applied to expected counts at the same N and ridge,
    backward through all stages. Returns per-stage V_pop_t (n_o,)."""
    E, K0, p1, pi_b, P, pR = (p[k] for k in ("E", "K0", "p1", "pi_b", "P", "pR"))
    n_s = p["n_s"]
    eps, rho = 1.0 / N, 0.03 / np.sqrt(N)
    # forward joint q_t(s, o0)
    qs = [p1[:, None] * K0]                                  # t = 1
    for t in range(1, T):
        q = qs[-1]
        nxt = np.zeros_like(q)
        for a in range(N_ACT):
            nxt += P[a].T @ (q * pi_b[:, [a]])
        qs.append(nxt)
    V_by_stage = [None] * (T + 2)
    cont = np.zeros(n_o)
    V_by_stage[T + 1] = cont.copy()
    for t in range(T, 0, -1):
        q = qs[t - 1]
        bV = np.zeros((N_ACT, n_o))
        for a in range(N_ACT):
            w_sa = q * pi_b[:, [a]]                          # (n_s, n_o0)
            Pa = E.T @ w_sa                                  # p(o, a, o0)
            qa = w_sa.sum(axis=0)                            # p(a, o0)
            pie_bar = E @ pi_e[:, a]                         # E_o[pi_e(a|o)|s]
            cont_exp = P[a] @ (E @ cont) if t < T else np.zeros(n_s)
            ma = ((pie_bar * (pR[:, a] + cont_exp))[:, None] * w_sa).sum(axis=0)
            Acr, Ga, cXy = N * Pa, N * qa, N * ma
            Wa = 1.0 / (Ga + N * eps)
            AW = Acr * Wa[None, :]
            M = Acr.T @ AW + N * rho * np.eye(n_o0)
            rhs = AW @ cXy
            bV[a] = (rhs - AW @ np.linalg.solve(M, Acr.T @ rhs)) / (N * rho)
        cont = bV.sum(axis=0)
        V_by_stage[t] = cont.copy()
    return V_by_stage


def main():
    for (n_s, n_o, n_o0, cf) in [(2, 6, 4, 1.0), (2, 6, 4, 0.6), (4, 6, 2, 1.0)]:
        p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=0,
                           confound=cf)
        pi = candidate_policies(n_o, N_ACT, seed=0)["always_0"]
        v_true = dp_value(p, pi)
        Vpop = population_bridge(p, pi, n_o, n_o0)

        Ja, Jb, worst_replica = [], [], 0.0
        for sd in SEEDS:
            d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
            est = MinimaxValueBridgeOPE(n_obs=n_o, n_act=N_ACT, n_o0=n_o0, T=T,
                                        seed=sd)
            est.fit(d["O0"], d["O"], d["A"], d["R"], pi)
            nu1 = est.nu1
            o1, a1, r1 = d["O"][:, 0], d["A"][:, 0], d["R"][:, 0]
            o2 = d["O"][:, 1]

            # replica validation: shipped continuation must reproduce bV_hat[0]
            V2_hat = est.bV_hat[1].sum(axis=0)
            y_ship = (r1 + V2_hat[o2]) * pi[o1, a1]
            b_rep = stage_solve(o1, a1, d["O0"], y_ship, n_o, n_o0, N)
            worst_replica = max(worst_replica,
                                float(np.abs(b_rep - est.bV_hat[0]).max()))
            Ja.append(float((b_rep.sum(axis=0) * nu1).sum()))

            # surgical: population continuation on the same data
            y_pop = (r1 + Vpop[2][o2]) * pi[o1, a1]
            b_srg = stage_solve(o1, a1, d["O0"], y_pop, n_o, n_o0, N)
            Jb.append(float((b_srg.sum(axis=0) * nu1).sum()))

        Ja, Jb = np.array(Ja), np.array(Jb)
        delta = Ja - Jb
        share = delta.var() / max(Ja.var(), 1e-300)
        print(f"({n_s},{n_o},{n_o0}) cf={cf}  V_true={v_true:.3f}  "
              f"replica max err={worst_replica:.2e}")
        print(f"    J shipped   : mean {Ja.mean():+.4f}  std {Ja.std():.4f}")
        print(f"    J pop-cont  : mean {Jb.mean():+.4f}  std {Jb.std():.4f}")
        print(f"    delta (later-stage contribution): mean {delta.mean():+.4f}"
              f"  std {delta.std():.4f}   var share of J: {share:.1%}")
        print(f"    corr(Ja, Jb) = {np.corrcoef(Ja, Jb)[0,1]:.3f}\n")


if __name__ == "__main__":
    main()
