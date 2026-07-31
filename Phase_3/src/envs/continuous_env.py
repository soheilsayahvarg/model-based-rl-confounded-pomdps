"""
continuous_env.py -- Continuous-state/action confounded POMDP (linear-Gaussian),
the continuous analogue of toy_pomdp.py.

Model (time-homogeneous, finite horizon T, all scalars for this first pass):
    latent state    s_t in R          (never observed)
    action           a_t in R          (real-valued "dose")
    observation      o_t = s_t + eps_o,           eps_o ~ N(0, sigma_o^2)
                     -- a NOISY proxy for the hidden state, emitted BEFORE the
                        action (mirrors toy_pomdp's o emitted from s pre-action).
    reward           r_t = beta_s*s_t + beta_a*a_t + eps_r,   eps_r ~ N(0, sigma_r^2)
    transition       s_{t+1} = phi*s_t + psi*a_t + eps_s,     eps_s ~ N(0, sigma_s^2)
    initial          s_1 ~ N(mu1, sigma1^2)
    neg. control     O_0 = s_1 + eps_0,            eps_0 ~ N(0, sigma_0^2)
                     -- fresh independent noise on the INITIAL state only =>
                        Assumption 3.1 holds by the same Markov argument as the
                        discrete envs (O_0 depends only on S_1 + fresh noise).
    behavior policy  A_t | s_t, o_t ~ N( kappa*K_b*s_t + (1-kappa)*K_b*o_t + c_b,
                                         sigma_b^2 )
                     kappa in [0,1]: at kappa=0 the logging policy is a function
                     of o_t ONLY (what a naive learner also conditions on) =>
                     NO confounding; at kappa=1 it depends on the true hidden
                     s_t directly => full confounding. This is a continuous
                     relaxation of simulated_env.py's aware/blind kappa-mixture.

WHY THIS COUNTS AS CONFOUNDING even with a single noisy-latent variable: this is
the classical MEASUREMENT-ERROR setting proximal causal inference was designed
for. The logging policy reacts to the true s_t; any learner only ever sees the
noisy proxy o_t = s_t + eps_o. Fitting a model that treats o_t as if it were the
true state induces errors-in-variables bias whenever kappa>0, because the same
noise eps_o that corrupts the "state" also correlates (through s_t) with what
determined the action -- exactly the discrete environment's u/x split, collapsed
onto a single noisy channel.

CANDIDATE (evaluation) policies are restricted to be AFFINE and OBSERVATION-only
(never a function of the hidden s_t): a_t = K_pi*o_t + c_pi. This mirrors
toy_pomdp/simulated_env's candidate policies, which are likewise pure functions
of the observation, never the latent state.

WHY LINEAR-GAUSSIAN, WHY AFFINE POLICIES: with affine candidate policies and a
linear-Gaussian system, E[O_t|S_t] = S_t exactly, so the MEAN trajectory of the
closed-loop system is an exact, closed-form scalar recursion (no covariance
propagation needed, since reward is LINEAR not quadratic -- only first moments
ever enter the value). This gives an EXACT oracle in continuous space, the
direct analogue of toy_pomdp's dynamic-programming oracle, and is what makes
machine-precision self-consistency checks possible here too (see
true_value / chain_value_from_bridges below, and run_continuous_benchmark.py's
[C1] check).
"""

import numpy as np

STATE_DIM = 1
ACTION_DIM = 1
OBS_DIM = 1


def default_params(T=3, kappa=1.0):
    return dict(
        T=T,
        phi=0.85, psi=0.35, sigma_s=0.30,          # transition
        beta_s=0.60, beta_a=-0.40, sigma_r=0.25,   # reward
        sigma_o=0.35,                              # observation noise
        sigma_0=0.25,                              # negative-control noise
        mu1=1.0, sigma1=0.40,                      # initial state
        K_b=0.70, c_b=0.10, sigma_b=0.20,           # behavior policy
        kappa=kappa,
    )


def candidate_policies():
    """Affine, observation-only candidate policies a = K*o + c (name -> (K, c))."""
    return {
        "never_treat": (0.0, 0.0),
        "constant_dose": (0.0, 1.0),
        "proportional": (0.6, 0.0),
        "aggressive": (1.0, 0.4),
    }


# ------------------------------------------------------------------ sampling
def sample_trajectories(params, N, rng):
    """Generate N logged trajectories under the confounded behavior policy.

    Returns estimator-visible {O0, O, A, R} (each (N,T) except O0 which is
    (N,)) and the PRIVATE hidden state S (N, T+1), never to be passed to
    estimators.
    """
    T = params["T"]
    S = np.zeros((N, T + 1))
    O = np.zeros((N, T))
    A = np.zeros((N, T))
    R = np.zeros((N, T))

    s = rng.normal(params["mu1"], params["sigma1"], size=N)
    S[:, 0] = s
    O0 = s + rng.normal(0.0, params["sigma_0"], size=N)

    for t in range(T):
        o_t = s + rng.normal(0.0, params["sigma_o"], size=N)
        O[:, t] = o_t
        mean_a = (params["kappa"] * params["K_b"] * s
                  + (1.0 - params["kappa"]) * params["K_b"] * o_t
                  + params["c_b"])
        a_t = mean_a + rng.normal(0.0, params["sigma_b"], size=N)
        A[:, t] = a_t
        r_t = (params["beta_s"] * s + params["beta_a"] * a_t
               + rng.normal(0.0, params["sigma_r"], size=N))
        R[:, t] = r_t
        s = (params["phi"] * s + params["psi"] * a_t
             + rng.normal(0.0, params["sigma_s"], size=N))
        S[:, t + 1] = s

    return dict(O0=O0, O=O, A=A, R=R, S=S)


# ------------------------------------------------------------------ exact oracle
def true_value(params, K_pi, c_pi):
    """Exact V(pi) for an affine observation-only policy a = K_pi*o + c_pi.

    Closed-loop mean recursion (exact because reward is linear and E[O_t|S_t]
    = S_t): m_1 = E[S_1]; m_{t+1} = (phi + psi*K_pi)*m_t + psi*c_pi;
    V = sum_t (beta_s + beta_a*K_pi)*m_t + beta_a*c_pi.
    """
    phi, psi = params["phi"], params["psi"]
    beta_s, beta_a = params["beta_s"], params["beta_a"]
    T = params["T"]
    m = params["mu1"]
    V = 0.0
    for _ in range(T):
        V += (beta_s + beta_a * K_pi) * m + beta_a * c_pi
        m = (phi + psi * K_pi) * m + psi * c_pi
    return float(V)


def true_bridge_coeffs(params):
    """Exact (time-homogeneous) bridge coefficients, solved in closed form.

    Reward bridge b_R*(a, o~) = beta_a*a + beta_s*o~ solves
        E_{o~|s}[ b_R*(a,o~) ] = beta_a*a + beta_s*s = E[R_t | s, a]   for all s,
    using E[o~|s] = s (unbiased noisy proxy). Matching coefficients on s gives
    this exactly -- no data, no estimation, pure algebra (the continuous
    analogue of oracle_module.toy_true_bridges' pseudo-inverse solve).

    Dynamics bridge b_D*(a, o~) = psi*a + phi*o~ solves, by the same argument,
        E_{o~|s}[ b_D*(a,o~) ] = psi*a + phi*s = E[S_{t+1} | s, a].

    Returns dict(bR=(beta_a, beta_s), bD=(psi, phi)).
    """
    return dict(bR=(params["beta_a"], params["beta_s"]),
                bD=(params["psi"], params["phi"]))


def chain_value_from_bridges(params, bR_coeffs, bD_coeffs, K_pi, c_pi, m1=None):
    """Independent recomputation of V(pi) via the continuous Theorem-3.5 mean
    chain, using ONLY bridge coefficients (not the true env parameters phi/psi/
    beta_s/beta_a directly) -- the continuous analogue of oracle_module.
    toy_chain_value. Must equal true_value(...) to machine precision when
    bR_coeffs/bD_coeffs are the TRUE coefficients (checked in [C1]).

    m_1 defaults to E[S_1] = mu1 (the exact initial mean); a fitted estimator
    instead passes the empirical mean of O[:, 0] (see continuous_value_plugin).
    """
    a_R, s_R = bR_coeffs      # b_R(a, o~) = a_R*a + s_R*o~
    a_D, s_D = bD_coeffs      # b_D(a, o~) = a_D*a + s_D*o~
    T = params["T"]
    m = params["mu1"] if m1 is None else m1
    V = 0.0
    for t in range(T):
        a_t = K_pi * m + c_pi
        V += a_R * a_t + s_R * m
        if t < T - 1:
            m = a_D * a_t + s_D * m
    return float(V)


if __name__ == "__main__":
    p = default_params()
    rng = np.random.default_rng(0)
    data = sample_trajectories(p, 5000, rng)
    print("continuous log shapes:", data["O0"].shape, data["O"].shape,
          data["A"].shape, data["R"].shape)
    print("mean episode return (behavior):", data["R"].sum(axis=1).mean())
