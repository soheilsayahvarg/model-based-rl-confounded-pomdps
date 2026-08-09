"""
finite_action_env.py -- PAPER-FAITHFUL continuous confounded POMDP:
continuous state and observation, FINITE action space.

WHY THIS ENVIRONMENT EXISTS. Phase 3's continuous_env.py used a real-valued
action. Re-reading the anchor paper showed its setting is stated as: "we assume
that both S and O are continuous, while the action space A is finite." So the
Phase-3 study varied TWO things at once relative to the paper -- continuous
observations (covered by the theory) and continuous actions (not covered) -- and
was therefore extrapolation beyond Theorem 3.5 rather than an instantiation of
it. This module fixes exactly that one axis and holds everything else as close
to Phase 3 as possible, so any change in the results is attributable.

MODEL (time-homogeneous, horizon T, scalar latent/observation):
    latent state    s_t in R                       (never observed)
    action          a_t in A = {a^1, ..., a^K}      FINITE set of dose levels
    observation     o_t = s_t + eps_o               noisy proxy, emitted pre-action
    reward          r_t = beta_s*s_t + beta_a*a_t + eps_r
    transition      s_{t+1} = phi*s_t + psi*a_t + eps_s
    initial         s_1 ~ N(mu1, sigma1^2)
    neg. control    O_0 = s_1 + eps_0               (Assumption 3.1 by construction)
    behavior        softmax over dose levels driven by kappa*s_t + (1-kappa)*o_t,
                    i.e. logits_k = eta * driver * a^k. At kappa=1 the logging
                    policy reads the TRUE hidden state => confounded; at kappa=0
                    it reads only o_t => unconfounded. Softmax guarantees every
                    action keeps positive probability, so overlap holds.

WHAT CHANGES STRUCTURALLY WITH A FINITE ACTION SPACE. The bridge argument w =
(a, o) now has a finite coordinate. The natural (and paper-faithful)
parameterisation gives each action level its OWN free coefficient rather than
forcing a single slope in a:

    b_R(a, o) = alpha_{a} + beta_o * o          (K + 1 free parameters)
    b_D(a, o) = gamma_{a} + phi_o * o           (K + 1 free parameters)

i.e. the feature map is [onehot(a), o]. This is a delta kernel on the action
coordinate tensored with a linear kernel on the observation coordinate -- the
finite-action analogue of Phase 3's pure linear kernel, and a strictly richer
model in the action coordinate (it does not assume the effect is linear in the
dose; it lets the data say so).

EXACT ORACLE. Candidate policies are distributions over the finite action set
that do NOT depend on the observation. That restriction is what preserves an
exact closed-form oracle: E[b(A,O)] under such a policy is
sum_k pi_k * alpha_k + beta_o * E[O], which is affine in the chain mean, so the
mean recursion stays exact and no covariance propagation is needed. A
threshold/greedy policy would make the closed loop a Gaussian MIXTURE and forfeit
the closed form -- deliberately out of scope here, since the point of this
experiment is to isolate the finite-action axis while KEEPING the
machine-precision verification discipline the rest of the project relies on.
"""

import numpy as np

ACTION_LEVELS = np.array([0.0, 0.5, 1.0])   # the finite action set A
N_ACTIONS = len(ACTION_LEVELS)


def default_params(T=3, kappa=1.0):
    return dict(
        T=T,
        phi=0.85, psi=0.35, sigma_s=0.30,          # transition
        beta_s=0.60, beta_a=-0.40, sigma_r=0.25,   # reward
        sigma_o=0.35,                              # observation noise
        sigma_0=0.25,                              # negative-control noise
        mu1=1.0, sigma1=0.40,                      # initial state
        eta=1.6,                                   # behavior-policy sharpness
        kappa=kappa,
        actions=ACTION_LEVELS.copy(),
    )


def candidate_policies():
    """Observation-independent distributions over the finite action set.

    Each entry is a probability vector over ACTION_LEVELS (see module docstring
    for why observation-independence is what buys the exact oracle).
    """
    return {
        "never_treat":  np.array([1.0, 0.0, 0.0]),
        "low_dose":     np.array([0.0, 1.0, 0.0]),
        "high_dose":    np.array([0.0, 0.0, 1.0]),
        "mixed":        np.array([0.5, 0.3, 0.2]),
    }


def _behavior_probs(driver, params):
    """Softmax over dose levels: logits_k = eta * driver * a^k. (N, K)."""
    logits = params["eta"] * np.outer(driver, params["actions"])
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ sampling
def sample_trajectories(params, N, rng):
    """N logged trajectories under the confounded behavior policy.

    Returns estimator-visible {O0, O, A, A_idx, R} plus the PRIVATE hidden state
    S, which must never be passed to an estimator.
    """
    T = params["T"]
    acts = params["actions"]
    S = np.zeros((N, T + 1))
    O = np.zeros((N, T))
    A = np.zeros((N, T))
    A_idx = np.zeros((N, T), dtype=int)
    R = np.zeros((N, T))

    s = rng.normal(params["mu1"], params["sigma1"], size=N)
    S[:, 0] = s
    O0 = s + rng.normal(0.0, params["sigma_0"], size=N)

    for t in range(T):
        o_t = s + rng.normal(0.0, params["sigma_o"], size=N)
        O[:, t] = o_t
        driver = params["kappa"] * s + (1.0 - params["kappa"]) * o_t
        probs = _behavior_probs(driver, params)
        u = rng.random(N)
        k = (u[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
        k = np.clip(k, 0, len(acts) - 1)
        a_t = acts[k]
        A_idx[:, t] = k
        A[:, t] = a_t
        R[:, t] = (params["beta_s"] * s + params["beta_a"] * a_t
                   + rng.normal(0.0, params["sigma_r"], size=N))
        s = (params["phi"] * s + params["psi"] * a_t
             + rng.normal(0.0, params["sigma_s"], size=N))
        S[:, t + 1] = s

    return dict(O0=O0, O=O, A=A, A_idx=A_idx, R=R, S=S)


# ------------------------------------------------------------------ exact oracle
def true_value(params, pi):
    """Exact V(pi) for an observation-independent action distribution pi.

    abar = sum_k pi_k a^k; m_1 = mu1;
    V = sum_t (beta_s*m_t + beta_a*abar);  m_{t+1} = phi*m_t + psi*abar.
    """
    abar = float(np.dot(pi, params["actions"]))
    m = params["mu1"]
    V = 0.0
    for _ in range(params["T"]):
        V += params["beta_s"] * m + params["beta_a"] * abar
        m = params["phi"] * m + params["psi"] * abar
    return float(V)


def true_bridge_coeffs(params):
    """Exact bridge coefficients in the [onehot(a), o] parameterisation.

    b_R(a,o) = alpha_a + beta_o*o must satisfy E_{o|s}[b_R(a,o)] = E[R|s,a] =
    beta_s*s + beta_a*a for all s. With E[o|s] = s this forces beta_o = beta_s
    and alpha_k = beta_a * a^k. Same argument for the dynamics bridge against
    E[S_{t+1}|s,a] = phi*s + psi*a.

    Returns dict(bR=(K+1,), bD=(K+1,)), last entry being the o-coefficient.
    """
    acts = params["actions"]
    bR = np.concatenate([params["beta_a"] * acts, [params["beta_s"]]])
    bD = np.concatenate([params["psi"] * acts, [params["phi"]]])
    return dict(bR=bR, bD=bD)


def chain_value_from_bridges(params, bR, bD, pi, m1=None):
    """Recompute V(pi) from bridge coefficients ONLY (never the raw env
    parameters) -- the finite-action Theorem-3.5 mean chain.

    Under an observation-independent pi the expected feature entering each stage
    is exactly [pi, m_t], so the chain is affine in m_t and stays closed-form.
    Must equal true_value(...) to machine precision at the true coefficients.
    """
    T = params["T"]
    m = params["mu1"] if m1 is None else m1
    V = 0.0
    for t in range(T):
        feat = np.concatenate([pi, [m]])
        V += float(feat @ bR)
        if t < T - 1:
            m = float(feat @ bD)
    return float(V)


if __name__ == "__main__":
    p = default_params()
    rng = np.random.default_rng(0)
    d = sample_trajectories(p, 4000, rng)
    print("action histogram:", np.bincount(d["A_idx"].ravel(), minlength=N_ACTIONS))
    tb = true_bridge_coeffs(p)
    print("true bR:", tb["bR"], " true bD:", tb["bD"])
    for name, pi in candidate_policies().items():
        v = true_value(p, pi)
        vc = chain_value_from_bridges(p, tb["bR"], tb["bD"], pi)
        print(f"{name:<12} V_true={v:+.6f}  chain={vc:+.6f}  gap={abs(v-vc):.3e}")
