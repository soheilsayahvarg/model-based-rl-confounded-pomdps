"""dim_separated_pomdp.py -- a confounded POMDP whose three dimensions move independently.

WHY THIS ENVIRONMENT EXISTS.

The Phase 3 toy POMDP fixes |S| = 2, |O| = 3, |O_0| = 2. That is fine for
verifying correctness, but it cannot answer the question this extension is about.

The claim under test is that a proximal bridge, which is parameterized over the
OBSERVATION space, is only identified through the LATENT bottleneck, so its
design matrix carries a structural null space. There are three candidate
explanations for what caps the identifiable rank:

    H_latent : rank is capped by |S|    -- the latent state mediates everything
    H_instr  : rank is capped by |O_0|  -- the instrument cannot resolve more
    H_min    : rank is capped by min(|S|, |O_0|, |O|)

In the Phase 3 toy, |S| = |O_0| = 2. All three hypotheses predict rank 2, so the
environment is silent on which is right. Any conclusion drawn there about the
mechanism would be an artifact of the design.

This module separates them. With |S| = 2, |O_0| = 4, |O| = 6, H_latent predicts a
rank of 2 and H_instr predicts 4 -- a decisive gap. Reversing to |S| = 4,
|O_0| = 2 flips the prediction, which guards against a coincidence in one
direction.

WHAT IS HELD FIXED. The confounding channel (the behavior policy reads the latent
state), the reward structure, and the horizon are unchanged as the dimensions
move, so a difference in the spectrum is attributable to dimension alone.

CONSTRUCTION. Emission and negative-control matrices are drawn from a Dirichlet
with a diagonal boost, then verified to have full row rank -- if the proxy
channels were rank-deficient the identification would fail for an uninteresting
reason and confound the test. `default_params` asserts this rather than hoping.

Log format matches toy_pomdp: (O0, O, A, R) visible, S private.
"""

import numpy as np


def default_params(n_s=2, n_a=2, n_o=6, n_o0=4, T=3, seed=0, confound=1.0):
    """Build a confounded POMDP with independently specified dimensions.

    confound : 0 -> behavior ignores the latent state (no confounding)
               1 -> behavior strongly prefers the latent-appropriate action
    """
    rng = np.random.default_rng(seed)

    # Emission p(o|s): each latent state gets a distinct, concentrated profile.
    # The diagonal boost keeps the rows separated so E has full row rank.
    E = rng.dirichlet(np.ones(n_o) * 0.7, size=n_s)
    for s in range(n_s):
        E[s, s % n_o] += 1.5
    E /= E.sum(axis=1, keepdims=True)

    # Negative control p(O_0|s_1): same construction, independent draw.
    K0 = rng.dirichlet(np.ones(n_o0) * 0.7, size=n_s)
    for s in range(n_s):
        K0[s, s % n_o0] += 1.5
    K0 /= K0.sum(axis=1, keepdims=True)

    # Transitions P[a][s, s'] -- action a nudges the chain toward state (s+a).
    P = np.zeros((n_a, n_s, n_s))
    for a in range(n_a):
        for s in range(n_s):
            row = rng.dirichlet(np.ones(n_s) * 0.8)
            row[(s + a) % n_s] += 1.2
            P[a, s] = row / row.sum()

    # Reward P(r=1|s,a): depends strongly on the latent state, so a
    # confounding-blind evaluator is visibly biased.
    pR = rng.uniform(0.15, 0.85, size=(n_s, n_a))
    for s in range(n_s):
        pR[s, s % n_a] = 0.85

    p1 = rng.dirichlet(np.ones(n_s) * 2.0)

    # Behavior policy reads the LATENT state -- this is the confounding channel.
    pi_b = np.full((n_s, n_a), 1.0 / n_a)
    if confound > 0:
        for s in range(n_s):
            pi_b[s] = (1.0 - confound) / n_a
            pi_b[s, s % n_a] += confound
        pi_b /= pi_b.sum(axis=1, keepdims=True)

    # The proxy channels must be able to resolve the latent state, or
    # identification fails for a reason unrelated to what we are testing.
    assert np.linalg.matrix_rank(E) == min(n_s, n_o), \
        f"emission rank {np.linalg.matrix_rank(E)} < min({n_s},{n_o})"
    assert np.linalg.matrix_rank(K0) == min(n_s, n_o0), \
        f"negative-control rank {np.linalg.matrix_rank(K0)} < min({n_s},{n_o0})"

    return dict(E=E, P=P, pR=pR, p1=p1, pi_b=pi_b, K0=K0, T=T,
                n_s=n_s, n_a=n_a, n_o=n_o, n_o0=n_o0)


def sample_trajectories(params, N, rng, policy=None, policy_type="latent"):
    """Generate N trajectories. Mirrors toy_pomdp.sample_trajectories."""
    E, P, pR, p1, pi_b, K0, T = (params[k] for k in
                                 ("E", "P", "pR", "p1", "pi_b", "K0", "T"))
    n_s, n_a = params["n_s"], params["n_a"]
    if policy is None:
        policy, policy_type = pi_b, "latent"

    S = np.zeros((N, T + 1), dtype=np.int64)
    O = np.zeros((N, T), dtype=np.int64)
    A = np.zeros((N, T), dtype=np.int64)
    R = np.zeros((N, T), dtype=np.float64)

    s = rng.choice(n_s, size=N, p=p1)
    S[:, 0] = s
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
        s_next = np.empty(N, dtype=np.int64)
        for a in range(n_a):
            ia = a_t == a
            if not ia.any():
                continue
            cdfP = np.cumsum(P[a], axis=1)
            s_next[ia] = (cdfP[s[ia]] < rng.random((ia.sum(), 1))).sum(axis=1)
        s = s_next
        S[:, t + 1] = s

    return dict(O0=O0, O=O, A=A, R=R, S=S)


def dp_value(params, pi_obs, T=None):
    """Exact V(pi) by dynamic programming, for an observation-based policy.

    pi_obs : (n_o, n_a) stationary policy over observations.

    The latent chain is what actually evolves, so the recursion runs on states;
    the policy enters through the emission, since the agent picks its action from
    the observation it happens to see:

        Q_t(s,a) = pR[s,a] + sum_s' P[a][s,s'] V_{t+1}(s')
        V_t(s)   = sum_o E[s,o] sum_a pi(a|o) Q_t(s,a)

    This is ground truth: no estimation, no sampling. It is the reference every
    pessimistic lower bound must stay below.
    """
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    n_s, n_a = params["n_s"], params["n_a"]
    T = T or params["T"]
    V_next = np.zeros(n_s)
    for _ in range(T):
        Q = pR + np.einsum("asz,z->sa", P, V_next)      # (n_s, n_a)
        eff = E @ pi_obs                                # (n_s, n_a): p(a|s)
        V_next = np.einsum("sa,sa->s", eff, Q)
    return float(p1 @ V_next)


def candidate_policies(n_o, n_a, seed=0):
    """A small set of observation-based candidate policies to rank."""
    rng = np.random.default_rng(seed)
    pols = {}
    for a in range(n_a):
        pi = np.zeros((n_o, n_a))
        pi[:, a] = 1.0
        pols[f"always_{a}"] = pi
    pols["uniform"] = np.full((n_o, n_a), 1.0 / n_a)
    pi = rng.dirichlet(np.ones(n_a) * 1.5, size=n_o)
    pols["mixed"] = pi
    return pols


def population_cross_moment(params, a=None):
    """The exact population limit of the model-free cross-moment, p(o_t, o_0).

    This is the object the empirical design matrix converges to:

        p(o, o_0) = sum_s p(s) p(o | s) p(o_0 | s)   =   E^T diag(p(s)) K0

    Written this way the factorization through the latent state is explicit, and
    the rank is bounded by |S| no matter how large |O| and |O_0| are. Computing
    it in closed form lets the diagnostic separate a genuine structural rank cap
    from finite-sample noise: at N -> infinity the empirical matrix must approach
    THIS, so any rank above rank(this) is sampling artifact.

    The marginal over s used here is the stationary-free t=1 marginal p1, which
    is the correct one for the first stage; later stages mix but the rank bound
    is unchanged because every stage factors through s in the same way.
    """
    E, K0, p1 = params["E"], params["K0"], params["p1"]
    return E.T @ np.diag(p1) @ K0        # (n_o, n_o0)


if __name__ == "__main__":
    for (ns, no, no0) in [(2, 6, 4), (4, 6, 2), (3, 7, 5)]:
        p = default_params(n_s=ns, n_o=no, n_o0=no0)
        M = population_cross_moment(p)
        print(f"|S|={ns} |O|={no} |O_0|={no0}  cross-moment shape={M.shape}  "
              f"rank={np.linalg.matrix_rank(M)}  (min of dims = {min(ns, no, no0)})")
