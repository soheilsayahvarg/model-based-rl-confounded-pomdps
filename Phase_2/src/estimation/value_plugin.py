"""
value_plugin.py — Phase 3: plug-in policy value V(pi, b_R, b_D) via Theorem 3.5,
implemented as EXACT tabular tensor contractions (np.einsum). No Monte Carlo.

For a MEMORYLESS observation policy pi(a|o), the sequential bridge integration
collapses into a chain of |O| x |O| transfer operators (the history sums over
(o_j, a_j) factorize because the policy weight is per-step):

    M_j[o~, o~']  = sum_{o,a} pi(a|o) * b_D^[j](a, o~, o~', o)      (policy-folded
                                                                     dynamic bridge)
    B_t[o~]       = sum_{o,a} pi(a|o) * sum_r r * b_R^[t](a, o~, r, o)
                                                                     (reward head; for
                                                                     r in {0,1} only
                                                                     the r=1 slice)
    G_1 = p(o~_1);   G_{t} = G_{t-1} M_{t-1}          (forward chain vectors)
    V(pi, b_R, b_D) = sum_t G_t . B_t

Backward recursion  Hb_T = B_T,  Hb_t = B_t + M_t Hb_{t+1}  gives V = G_1 . Hb_1 and
the EXACT gradients of the multilinear value w.r.t. each bridge block (needed by the
ellipsoidal-pessimism inner minimization, where V is linear in each block):

    dV / d b_R^[t][a, o~, r=1, o] = pi(a|o) * G_t[o~]
    dV / d b_D^[j][a, o~, o~', o] = pi(a|o) * G_j[o~] * Hb_{j+1}[o~']

p(o~_1) is either the exact initial-observation marginal (oracle checks) or its
empirical version from the log (the anchor's sanctioned data-driven choice).
"""

import numpy as np


def plugin_value(bR, bD, pi_obs, p_o1, return_grads=False, reward_levels=None):
    """V(pi, b_R, b_D) by exact chain contraction.

    bR: (T, A, O, n_r, O); bD: (T-1, A, O, O, O); pi_obs: (O, A); p_o1: (O,).
    reward_levels: the real reward VALUES for the n_r bridge slices, POSITIONALLY
        aligned to bR's r-axis (reward_levels[i] must be the reward value of bR
        slice i in the estimator's own r-axis ordering, not merely the correct
        value SET -- a permuted order silently mis-weights the value). Defaults to
        np.arange(n_r), correct for the toy (r-axis = {0,1} in order). A caller
        whose alphabet is not 0..n_r-1 (e.g. the 720x2 benchmark, r_idx = R+1 so
        slice i has value i-1 = [-1,0,+1]) MUST pass the explicit, correctly-ordered
        levels. (The benchmark path uses big_bridge_estimator._reward_head, which
        hard-codes [-1,0,1] matching its own r_idx layout; this entry point is the
        toy one.)
    Returns (V, per_step) or (V, per_step, grads) with grads = (gR, gD) matching
    the bridge tensor shapes (gR nonzero only on reward-weighted slices).
    """
    T = bR.shape[0]
    n_r = bR.shape[3]
    if reward_levels is None:
        r_weights = np.arange(n_r, dtype=np.float64)
    else:
        r_weights = np.asarray(reward_levels, dtype=np.float64)
        assert r_weights.shape == (n_r,), \
            f"reward_levels must have shape ({n_r},), got {r_weights.shape}"

    # policy-folded operators
    Ms = [np.einsum("oa,auvo->uv", pi_obs, bD[j]) for j in range(T - 1)]
    Bs = [np.einsum("oa,r,auro->u", pi_obs, r_weights, bR[t]) for t in range(T)]

    # forward chain
    G = [None] * T
    G[0] = np.asarray(p_o1, dtype=np.float64)
    for t in range(1, T):
        G[t] = G[t - 1] @ Ms[t - 1]

    # backward chain
    Hb = [None] * T
    Hb[T - 1] = Bs[T - 1]
    for t in range(T - 2, -1, -1):
        Hb[t] = Bs[t] + Ms[t] @ Hb[t + 1]

    per_step = [float(G[t] @ Bs[t]) for t in range(T)]
    V = float(G[0] @ Hb[0])
    # Self-consistency only: forward (G) and backward (Hb) are built from the SAME
    # Ms/Bs tensors, so this checks the two contraction ORDERS agree -- it does NOT
    # detect a mis-oriented bridge tensor (e.g. swapped o~_j/o~_{j+1} axes). The
    # convention itself is validated externally by run_phase34_check [V1]: plug-in
    # on ORACLE bridges == exact DP to 4.4e-16.
    assert abs(V - sum(per_step)) < 1e-9 * max(1.0, abs(V))

    if not return_grads:
        return V, per_step

    gR = np.zeros_like(bR)
    gD = np.zeros_like(bD)
    for t in range(T):
        # dV/dbR[t][a, u, r, o] = pi(a|o) * r * G_t[u]
        gR[t] = np.einsum("oa,r,u->auro", pi_obs, r_weights, G[t])
    for j in range(T - 1):
        # dV/dbD[j][a, u, v, o] = pi(a|o) * G_j[u] * Hb_{j+1}[v]
        gD[j] = np.einsum("oa,u,v->auvo", pi_obs, G[j], Hb[j + 1])
    return V, per_step, (gR, gD)


def empirical_p_o1(O, n_obs):
    """Empirical initial-observation marginal p_hat(o_1) from the log."""
    p = np.bincount(O[:, 0], minlength=n_obs).astype(np.float64)
    return p / p.sum()
