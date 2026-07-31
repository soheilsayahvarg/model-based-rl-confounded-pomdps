"""
toy_pomdp.py — Minimal tabular confounded POMDP (Phase 1 companion substrate).

Purpose: the ONLY environment where the true bridge functions b_R / b_D are small
enough to materialize densely and verify by direct matrix (pseudo)inversion, and
where the Theorem-3.5 sequential-integration identity can be checked against exact
dynamic programming to machine precision.

Model (time-homogeneous, finite horizon T):
    latent state   s in {0, 1}
    action         a in {0, 1}
    observation    o in {0, 1, 2}   emitted from s BEFORE the action: p(o|s) = E[s, o]
    reward         r in {0, 1}      Bernoulli given (s, a): P(r=1|s,a) = pR[s, a]
    transition     p(s'|s, a) = P[a][s, s']
    initial        p1[s]
    behavior       pi_b(a|s) — depends on the LATENT state => confounded log
    neg. control   O_0 in {0, 1}: p(O_0 = k | s_1) = K0[s_1, k], fresh independent
                   noise => Assumption 3.1 holds by the Markov structure.

Design notes:
  * |O| = 3 > |S| = 2 and E has full row rank => the latent bridge equations are
    exactly solvable (min-norm solution via pinv), i.e., Assumption 3.2 holds.
  * K0 is invertible (rows distinct) => O_0 is informative about s_1; with the
    latent chain mixing slowly, completeness (Assumption 3.3) is plausible and,
    in this 2-state model, reduces to rank conditions that hold by construction.
  * pR depends strongly on s, and pi_b prefers the good action per latent state:
    this makes the naive observation-MDP evaluator visibly biased.

Log format matches simulated_env: (o_0, o_1, a_1, r_1, ..., o_T, a_T, r_T).
"""

import numpy as np

N_S = 2
N_A = 2
N_O = 3
N_O0 = 2


def default_params(T=3):
    E = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.2, 0.7]])            # p(o|s), full row rank
    P = np.array([                              # P[a][s, s']
        [[0.9, 0.1],
         [0.3, 0.7]],
        [[0.5, 0.5],
         [0.1, 0.9]],
    ])
    pR = np.array([[0.9, 0.2],                  # P(r=1|s,a)
                   [0.1, 0.8]])
    p1 = np.array([0.6, 0.4])
    pi_b = np.array([[0.75, 0.25],              # behavior pi_b(a|s): prefers the
                     [0.25, 0.75]])             # latent-appropriate action
    K0 = np.array([[0.85, 0.15],                # p(O_0|s_1), invertible
                   [0.20, 0.80]])
    return dict(E=E, P=P, pR=pR, p1=p1, pi_b=pi_b, K0=K0, T=T)


def sample_trajectories(params, N, rng, policy=None, policy_type="latent"):
    """Generate N trajectories of length T.

    policy=None      -> use the confounded behavior pi_b (latent-indexed).
    policy_type='obs' with policy (N_O, N_A) -> candidate-policy rollouts (MC checks).
    Returns estimator-visible {O0, O, A, R} + PRIVATE {S}.
    """
    E, P, pR, p1, pi_b, K0, T = (params[k] for k in
                                 ("E", "P", "pR", "p1", "pi_b", "K0", "T"))
    if policy is None:
        policy, policy_type = pi_b, "latent"

    S = np.zeros((N, T + 1), dtype=np.int64)
    O = np.zeros((N, T), dtype=np.int64)
    A = np.zeros((N, T), dtype=np.int64)
    R = np.zeros((N, T), dtype=np.float64)

    s = rng.choice(N_S, size=N, p=p1)
    S[:, 0] = s
    # pre-decision negative control: O_0 ~ K0[s_1, .], fresh independent noise
    cdf0 = np.cumsum(K0, axis=1)
    O0 = (cdf0[s] < rng.random((N, 1))).sum(axis=1)

    cdfE = np.cumsum(E, axis=1)
    for t in range(T):
        o_t = (cdfE[s] < rng.random((N, 1))).sum(axis=1)
        O[:, t] = o_t
        pol_idx = s if policy_type == "latent" else o_t
        cdf_pi = np.cumsum(policy, axis=1)
        a_t = (cdf_pi[pol_idx] < rng.random((N, 1))).sum(axis=1)
        A[:, t] = a_t
        R[:, t] = (rng.random(N) < pR[s, a_t]).astype(np.float64)
        # transition
        s_next = np.empty(N, dtype=np.int64)
        for a in range(N_A):
            ia = a_t == a
            cdfP = np.cumsum(P[a], axis=1)
            s_next[ia] = (cdfP[s[ia]] < rng.random((ia.sum(), 1))).sum(axis=1)
        s = s_next
        S[:, t + 1] = s

    return dict(O0=O0, O=O, A=A, R=R, S=S)


if __name__ == "__main__":
    params = default_params()
    data = sample_trajectories(params, 5000, np.random.default_rng(0))
    print("toy log shapes:", data["O0"].shape, data["O"].shape,
          data["A"].shape, data["R"].shape)
    print("mean episode return (behavior):", data["R"].sum(axis=1).mean())
