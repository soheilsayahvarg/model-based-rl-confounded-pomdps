"""
simulated_env.py — Tabular confounded-POMDP benchmark environment (Phase 1).

Ground-truth SCM: generic discrete tabular tensors reconstructed from the
clinicalml/gumbel-max-scm benchmark logic (external/gumbel-max-scm), loaded as
plain numerical arrays:

    tx[u, a, x, x'] : (2, 8, 720, 720)  transition of the observable-core index x,
                                        modulated by a static binary LATENT index u
    rdest[x']       : (720,) in {-1,0,+1}  reward granted on ENTERING core state x'
    p0[u, x]        : (2, 720)          initial core-state distribution per latent u
    pmix[u]         : (2,)              latent mixture P(u)

Augmented latent chain (S_AUG = 1442):
    live latent state  s = u*720 + x            (u in {0,1}, x in {0..719})
    absorbing          TERM_NEG = 1440 (entered with r = -1)
                       TERM_POS = 1441 (entered with r = +1)
    Reward is emitted exactly once, on the live -> terminal transition; terminals
    self-loop with r = 0 so every trajectory has a fixed length T.

Observation channel visible in the log (N_OBS = 1442):
    live:      o_t = 2*x_t + m_t,  m_t ~ Bernoulli(q_u)  — a per-step noisy binary
               marker whose rate depends on the latent u.  With q0 != q1 the per-x
               emission block  Em2 = [[1-q0, q0], [1-q1, q1]]  is invertible, which
               guarantees one-step bridge existence (Assumption 3.2).  With a purely
               deterministic emission o = x, bridges would NOT exist (rank 720 < 1440).
    terminal:  o = OBS_NEG = 1440  /  o = OBS_POS = 1441   (deterministic).

Pre-decision negative control (Assumption 3.1 by construction):
    O_0 = u XOR Bernoulli(eps0) — a noisy readout of the INITIAL latent index using
    fresh independent noise. Since O_0 = f(S_1, independent noise) and future
    observables depend only on (S_t, A_t) plus fresh noise, the Markov structure gives
    O_0  _||_  (O_t, O_{t+1}, R_t) | (S_t, A_t, H_{t-1})  exactly.

Confounded behavior policy (the latent index drives the logged actions):
    pi_b,t(a|s) = (1-eps_u) * [ kappa * softmax(Q_t(s,·)/tau)
                                + (1-kappa) * softmax(Qbar_t(x,·)/tau) ] + eps_u/8
    Q_t   = exact finite-horizon optimal Q on the augmented LATENT chain (latent-aware),
    Qbar_t(x,·) = sum_u pmix[u] * Q_t(u*720+x, ·)      (latent-blind reference),
    kappa in [0,1] = confounding-strength knob (kappa=0 -> behavior measurable in x).

Data record per trajectory (the estimator-visible log):  (o_0, o_1, a_1, r_1, ..., o_T, a_T, r_T).
Hidden diagnostics (S, U) are returned separately and must never be passed to estimators.
"""

import os
import pickle
import numpy as np

# ----------------------------------------------------------------------------- constants
N_X = 720          # observable-core states
N_U = 2            # latent binary index
N_A = 8            # discrete actions
N_LIVE = N_X * N_U
TERM_NEG = 1440    # absorbing, entered with r = -1
TERM_POS = 1441    # absorbing, entered with r = +1
S_AUG = 1442
OBS_NEG = 1440
OBS_POS = 1441
N_OBS = 1442

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKL = os.path.normpath(os.path.join(
    _HERE, "..", "..", "external", "gumbel-max-scm", "data", "extracted",
    "diab_txr_mats-replication.pkl"))

DEFAULTS = dict(T=3, q0=0.30, q1=0.70, eps0=0.15, kappa=1.0, tau=0.15, eps_u=0.05)


# ----------------------------------------------------------------------------- parameters
def load_params(pkl_path=DEFAULT_PKL):
    """Load the benchmark tensors and extract the destination-reward vector.

    Validates the property (verified in Phase 0 inspection) that the raw reward
    tensor depends only on the destination core state x'.
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    tx = np.asarray(d["tx_mat"], dtype=np.float64)        # (2, 8, 720, 720)
    r_raw = np.asarray(d["r_mat"], dtype=np.float64)      # (2, 8, 720, 720)
    p0 = np.asarray(d["p_initial_state"], dtype=np.float64)  # (2, 720)
    pmix = np.asarray(d["p_mixture"], dtype=np.float64)      # (2,)

    assert tx.shape == (N_U, N_A, N_X, N_X)
    assert np.allclose(tx.sum(axis=3), 1.0), "transition rows must sum to 1"

    # reward depends only on destination: collapse columns
    rdest = np.zeros(N_X)
    for xp in range(N_X):
        col = r_raw[:, :, :, xp]
        nz = col[np.abs(col) > 0]
        if nz.size:
            assert np.allclose(nz, nz[0]), "reward must depend only on destination"
            rdest[xp] = nz[0]
    return dict(tx=tx, rdest=rdest, p0=p0, pmix=pmix)


def build_augmented(params):
    """Build the augmented latent chain P[a, s, s'] (8, 1442, 1442) with absorbing
    terminals, plus the expected one-step reward ER[a, s] and initial distribution.

    Transition-reward convention: entering a destination x' with rdest[x'] = -1/+1
    redirects that probability mass to TERM_NEG/TERM_POS and grants the reward once.
    """
    tx, rdest, p0, pmix = params["tx"], params["rdest"], params["p0"], params["pmix"]
    neg_cols = np.where(rdest < 0)[0]
    pos_cols = np.where(rdest > 0)[0]
    live_cols = np.where(rdest == 0)[0]

    P = np.zeros((N_A, S_AUG, S_AUG))
    row_idx = {u: np.arange(u * N_X, (u + 1) * N_X) for u in range(N_U)}
    for u in range(N_U):
        rows = slice(u * N_X, (u + 1) * N_X)
        for a in range(N_A):
            M = tx[u, a]                                   # (720, 720)
            P[a][np.ix_(row_idx[u], u * N_X + live_cols)] = M[:, live_cols]
            P[a, rows, TERM_NEG] = M[:, neg_cols].sum(axis=1)
            P[a, rows, TERM_POS] = M[:, pos_cols].sum(axis=1)
    P[:, TERM_NEG, TERM_NEG] = 1.0
    P[:, TERM_POS, TERM_POS] = 1.0
    assert np.allclose(P.sum(axis=2), 1.0)

    # expected one-step reward from live states (reward only on live -> terminal entry)
    ER = np.zeros((N_A, S_AUG))
    ER[:, :N_LIVE] = P[:, :N_LIVE, TERM_POS] - P[:, :N_LIVE, TERM_NEG]

    p_init = np.zeros(S_AUG)
    for u in range(N_U):
        p_init[u * N_X:(u + 1) * N_X] = pmix[u] * p0[u]
    assert np.isclose(p_init.sum(), 1.0)

    return dict(P=P, ER=ER, p_init=p_init, pmix=pmix,
                neg_cols=neg_cols, pos_cols=pos_cols, live_cols=live_cols)


# ----------------------------------------------------------------------------- behavior
def optimal_Q(aug, T):
    """Exact finite-horizon optimal Q on the augmented latent chain.

    Returns list Qts of length T; Qts[t-1] has shape (S_AUG, N_A) for stage t.
    Q_t(s,a) = ER(a,s) + sum_s' P[a,s,s'] max_a' Q_{t+1}(s',a').
    """
    P, ER = aug["P"], aug["ER"]
    Qts = [None] * T
    V_next = np.zeros(S_AUG)
    for t in range(T, 0, -1):
        Q = ER.T + np.einsum("asz,z->sa", P, V_next)
        Qts[t - 1] = Q
        V_next = Q.max(axis=1)
    return Qts


def _softmax(Q, tau):
    z = (Q - Q.max(axis=1, keepdims=True)) / tau
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def behavior_policy(aug, Qts, kappa, tau, eps_u):
    """Confounded behavior pi_b,t(a|s): latent-aware/blind softmax mixture + uniform floor.

    Returns list of (S_AUG, N_A) row-stochastic arrays, one per stage t = 1..T.
    """
    pmix = aug["pmix"]
    pis = []
    for Q in Qts:
        aware = _softmax(Q, tau)                                   # (S_AUG, 8)
        Qbar_x = pmix[0] * Q[:N_X] + pmix[1] * Q[N_X:N_LIVE]       # (720, 8)
        blind_x = _softmax(Qbar_x, tau)
        blind = aware.copy()                       # terminals keep aware (irrelevant: absorbing)
        blind[:N_X] = blind_x
        blind[N_X:N_LIVE] = blind_x                # same for both latent blocks: blind in u
        pi = (1.0 - eps_u) * (kappa * aware + (1.0 - kappa) * blind) + eps_u / N_A
        assert np.allclose(pi.sum(axis=1), 1.0)
        pis.append(pi)
    return pis


# ----------------------------------------------------------------------------- sampling
def _sample_rows(P_a_rows, rng):
    """Sample one destination index per row of a row-stochastic matrix (n, S)."""
    cdf = np.cumsum(P_a_rows, axis=1)
    rr = rng.random((P_a_rows.shape[0], 1))
    return (cdf < rr).sum(axis=1)


def _emit_obs(s, rng, q0, q1):
    """Vectorized observation emission o(s): live -> 2x+m with m ~ Bern(q_u)."""
    o = np.empty(s.shape, dtype=np.int64)
    term_neg = s == TERM_NEG
    term_pos = s == TERM_POS
    live = ~(term_neg | term_pos)
    u = s[live] // N_X
    x = s[live] % N_X
    q = np.where(u == 1, q1, q0)
    m = (rng.random(x.shape) < q).astype(np.int64)
    o[live] = 2 * x + m
    o[term_neg] = OBS_NEG
    o[term_pos] = OBS_POS
    return o


def sample_trajectories(aug, policy, N, T, rng, q0, q1, eps0,
                        policy_type="latent", chunk=4096):
    """Generate N trajectories of length T.

    policy: list of T arrays — (S_AUG, N_A) if policy_type='latent' (behavior),
            (N_OBS, N_A) if policy_type='obs' (candidate policies, for MC checks).
    Returns dict with the estimator-visible log {O0, O, A, R} and the PRIVATE
    diagnostics {S, U} (never to be passed to estimators).
    """
    P, p_init = aug["P"], aug["p_init"]
    S = np.zeros((N, T + 1), dtype=np.int64)   # S[:, t-1] = s_t; S[:, T] = s_{T+1}
    O = np.zeros((N, T), dtype=np.int64)
    A = np.zeros((N, T), dtype=np.int64)
    R = np.zeros((N, T), dtype=np.float64)

    s = rng.choice(S_AUG, size=N, p=p_init)
    S[:, 0] = s
    U = np.where(s < N_LIVE, s // N_X, -1)     # initial latent index (static on live paths)

    # pre-decision negative control O_0: noisy readout of initial latent, fresh noise
    flip = rng.random(N) < eps0
    O0 = np.where(flip, 1 - U, U)
    O0 = np.where(U >= 0, O0, 0)               # (defensive; initial states are always live)

    for t in range(T):
        o_t = _emit_obs(s, rng, q0, q1)
        O[:, t] = o_t
        pi_t = policy[t]
        idx = s if policy_type == "latent" else o_t
        # sample actions row-wise
        a_t = np.empty(N, dtype=np.int64)
        for lo in range(0, N, chunk):
            sl = slice(lo, min(lo + chunk, N))
            a_t[sl] = _sample_rows(pi_t[idx[sl]], rng)
        A[:, t] = a_t
        # transition per action group, chunked
        s_next = np.empty(N, dtype=np.int64)
        for a in range(N_A):
            ia = np.where(a_t == a)[0]
            for lo in range(0, ia.size, chunk):
                sub = ia[lo:lo + chunk]
                s_next[sub] = _sample_rows(P[a][s[sub]], rng)
        # reward on live -> terminal entry only
        was_live = s < N_LIVE
        R[:, t] = np.where(was_live & (s_next == TERM_POS), 1.0,
                  np.where(was_live & (s_next == TERM_NEG), -1.0, 0.0))
        s = s_next
        S[:, t + 1] = s

    return dict(O0=O0, O=O, A=A, R=R, S=S, U=U)


# ----------------------------------------------------------------------------- assembly
def make_env(pkl_path=DEFAULT_PKL, T=DEFAULTS["T"], q0=DEFAULTS["q0"], q1=DEFAULTS["q1"],
             eps0=DEFAULTS["eps0"], kappa=DEFAULTS["kappa"], tau=DEFAULTS["tau"],
             eps_u=DEFAULTS["eps_u"]):
    """One-call construction: parameters -> augmented chain -> behavior policy."""
    params = load_params(pkl_path)
    aug = build_augmented(params)
    Qts = optimal_Q(aug, T)
    pi_b = behavior_policy(aug, Qts, kappa=kappa, tau=tau, eps_u=eps_u)
    return dict(params=params, aug=aug, Qts=Qts, pi_b=pi_b,
                T=T, q0=q0, q1=q1, eps0=eps0, kappa=kappa, tau=tau, eps_u=eps_u)


def generate_dataset(env, N, seed=0):
    rng = np.random.default_rng(seed)
    return sample_trajectories(env["aug"], env["pi_b"], N, env["T"], rng,
                               q0=env["q0"], q1=env["q1"], eps0=env["eps0"],
                               policy_type="latent")


# ----------------------------------------------------------------------------- obs decode
def decode_core(x):
    """Decode core index x (0..719) into its 7 mixed-radix components
    [c0:3, c1:3, c2:2, c3:5, c4:2, c5:2, c6:2] (most significant first)."""
    c0 = x // 240
    c1 = (x // 80) % 3
    c2 = (x // 40) % 2
    c3 = (x // 8) % 5
    c4 = (x // 4) % 2
    c5 = (x // 2) % 2
    c6 = x % 2
    return c0, c1, c2, c3, c4, c5, c6


if __name__ == "__main__":
    env = make_env()
    data = generate_dataset(env, N=2000, seed=0)
    print("O0 shape", data["O0"].shape, "O", data["O"].shape,
          "A", data["A"].shape, "R", data["R"].shape)
    print("mean episode return (behavior, kappa=%.1f): %.4f"
          % (env["kappa"], data["R"].sum(axis=1).mean()))
