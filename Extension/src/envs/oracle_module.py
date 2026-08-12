"""
oracle_module.py — Exact oracles for Phase 1 (NO estimation code here).

Contents:
  (1) dp_value_obs_policy      exact finite-horizon V(pi) on the augmented latent
                               chain of simulated_env, for observation-based
                               (memoryless, possibly time-varying) candidate policies
  (2) dp_value_toy             exact V(pi) for the toy POMDP
  (3) toy_true_bridges         true b_R, b_D by direct matrix (pseudo)inversion,
                               with residuals of the latent moment equations
  (4) toy_chain_value          Theorem-3.5 sequential-integration value using the
                               true bridges (must equal DP to machine precision)
  (5) big_bridge_* helpers     on-demand true-bridge solvers for simulated_env via
                               per-x 2x2 inversion of the marker emission block,
                               plus a residual self-test
  (6) fit_naive_mdp /          the NAIVE UNCORRECTED baseline: count-based
      naive_value              observation-MDP estimation + forward DP — the thing
                               confounding provably biases (Phase 1 exit criterion)
  (7) mc_value                 Monte-Carlo on-policy value (DP sanity check)

Mathematical conventions follow the anchor paper (Hong-Qi-Xu ICML 2024):
  model factors      p(r_t, o_t | s_t, a_t) = p(o_t|s_t) * p(r_t|s_t,a_t)
                     p(s_{j+1}, o_j | s_j, a_j) = p(o_j|s_j) * p(s_{j+1}|s_j,a_j)
  latent bridge equations (sufficient conditions; solvable by construction here):
      sum_o~ p(o~|s) b_R(a, o~, r, o)      = p(r, o | s, a)                for all s
      sum_o~ p(o~|s) b_D(a, o~, o~', o)    = p(o|s) * sum_s' p(s'|s,a) p(o~'|s')
  Theorem-3.5 chain:
      g_1(o~) = p(o~_1);  g_{j+1}(o~') = sum_o~ b_D(a_j, o~, o~', o_j) g_j(o~)
      f_t(r, h_t) = sum_o~ b_R(a_t, o~, r, o_t) g_t(o~)
      p^pi(r_t) = sum_{h_t} prod_j pi(a_j|o_j) f_t(r, h_t);   V = sum_t E[R_t]
"""

import itertools
import numpy as np

# simulated_env is the 720-observation benchmark and needs the third-party
# gumbel-max-scm clone. The toy-scale oracles below do not, so the import is made
# optional here: without it the big-env helpers are unavailable but
# toy_true_bridges / dp_value_toy still work.
try:
    from simulated_env import (N_X, N_U, N_A, N_LIVE, S_AUG, N_OBS,
                               TERM_NEG, TERM_POS, OBS_NEG, OBS_POS)
    _HAS_BIG_ENV = True
except ModuleNotFoundError:
    _HAS_BIG_ENV = False
import toy_pomdp as toy


# ============================================================== (1) big-env DP oracle
def _effective_policy_latent(pi_obs_t, q0, q1):
    """Fold the marker emission into an obs-policy: pi_eff(a|s) for live s=(u,x).

    pi_eff(a | s=(u,x)) = (1-q_u) * pi(a | o=2x) + q_u * pi(a | o=2x+1).
    Returns (S_AUG, N_A); terminal rows are irrelevant (absorbing, zero reward).
    """
    pi_eff = np.zeros((S_AUG, N_A))
    even = pi_obs_t[0:2 * N_X:2]      # (720, A) rows for o = 2x
    odd = pi_obs_t[1:2 * N_X:2]       # (720, A) rows for o = 2x+1
    for u, q in ((0, q0), (1, q1)):
        pi_eff[u * N_X:(u + 1) * N_X] = (1 - q) * even + q * odd
    pi_eff[TERM_NEG] = pi_obs_t[OBS_NEG]
    pi_eff[TERM_POS] = pi_obs_t[OBS_POS]
    return pi_eff


def dp_value_obs_policy(aug, pi_obs, T, q0, q1):
    """Exact V(pi) for an observation-based candidate policy on the augmented chain.

    pi_obs: (N_OBS, N_A) stationary, or list of T such arrays (time-varying).
    V_t(s) = sum_a pi_eff,t(a|s) [ ER(a,s) + sum_s' P[a,s,s'] V_{t+1}(s') ].
    """
    P, ER, p_init = aug["P"], aug["ER"], aug["p_init"]
    pis = pi_obs if isinstance(pi_obs, list) else [pi_obs] * T
    V_next = np.zeros(S_AUG)
    for t in range(T, 0, -1):
        pi_eff = _effective_policy_latent(pis[t - 1], q0, q1)
        Qpi = ER.T + np.einsum("asz,z->sa", P, V_next)     # (S_AUG, N_A)
        V_next = (pi_eff * Qpi).sum(axis=1)
    return float(p_init @ V_next)


# ============================================================== (2) toy DP oracle
def dp_value_toy(params, pi_obs, T=None):
    """Exact V(pi) for the toy POMDP, observation-based policy (N_O, N_A)."""
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    T = T or params["T"]
    pi_eff = E @ pi_obs                        # (S, A): sum_o p(o|s) pi(a|o)
    r_sa = pR                                  # expected reward given (s, a)
    V_next = np.zeros(toy.N_S)
    for _ in range(T, 0, -1):
        Q = r_sa + np.einsum("asz,z->sa", P, V_next)
        V_next = (pi_eff * Q).sum(axis=1)
    return float(p1 @ V_next)


def dp_value_toy_latent(params, pi_lat, T=None):
    """Exact V for a LATENT-indexed policy (used for the behavior policy itself)."""
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    T = T or params["T"]
    V_next = np.zeros(toy.N_S)
    for _ in range(T, 0, -1):
        Q = pR + np.einsum("asz,z->sa", P, V_next)
        V_next = (pi_lat * Q).sum(axis=1)
    return float(p1 @ V_next)


# ============================================================== (3) toy true bridges
def toy_true_bridges(params):
    """True bridges by direct matrix (pseudo)inversion + residuals.

    b_R[a, o~, r, o]  solves  E b = c,  c[s] = E[s,o] * P(r|s,a)
    b_D[a, o~, o~', o] solves E B = C,  C[s,o~'] = E[s,o] * (P[a] E)[s,o~']
    (min-norm solutions via pinv(E); residuals must be ~0 since rank(E) = |S|).
    """
    E, P, pR = params["E"], params["P"], params["pR"]
    Epinv = np.linalg.pinv(E)                                  # (O, S)

    bR = np.zeros((toy.N_A, toy.N_O, 2, toy.N_O))
    resR = 0.0
    for a in range(toy.N_A):
        for r in range(2):
            p_r = pR[:, a] if r == 1 else 1.0 - pR[:, a]       # P(r|s,a), (S,)
            for o in range(toy.N_O):
                c = E[:, o] * p_r                              # (S,)
                u = Epinv @ c
                bR[a, :, r, o] = u
                resR = max(resR, float(np.abs(E @ u - c).max()))

    bD = np.zeros((toy.N_A, toy.N_O, toy.N_O, toy.N_O))
    resD = 0.0
    for a in range(toy.N_A):
        PE = P[a] @ E                                          # (S, O'): sum_s' p(s'|s,a) p(o~'|s')
        for o in range(toy.N_O):
            C = E[:, o][:, None] * PE                          # (S, O')
            B = Epinv @ C                                      # (O, O')
            bD[a, :, :, o] = B
            resD = max(resD, float(np.abs(E @ B - C).max()))

    return bR, bD, dict(residual_bR=resR, residual_bD=resD)


# ============================================================== (4) Theorem-3.5 chain
def toy_chain_value(params, bR, bD, pi_obs, T=None):
    """V(pi) via the Theorem-3.5 sequential bridge integration (exact enumeration).

    Enumerates all histories h_t = (o_1, a_1, ..., o_t, a_t); the policy weight uses
    pi(a_j | o_j) (memoryless candidate). Returns V_chain = sum_t sum_r r * p^pi(r_t).
    """
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    T = T or params["T"]
    g1 = p1 @ E                                # p(o~_1), (O,)

    V = 0.0
    per_step = []
    for t in range(1, T + 1):
        p_rt1 = 0.0                            # p^pi(r_t = 1)
        for hist in itertools.product(range(toy.N_O), range(toy.N_A), repeat=t):
            obs = hist[0::2]
            acts = hist[1::2]
            w = 1.0
            for o_j, a_j in zip(obs, acts):
                w *= pi_obs[o_j, a_j]
            if w == 0.0:
                continue
            g = g1
            for j in range(t - 1):
                g = g @ bD[acts[j], :, :, obs[j]]              # g_{j+1}(o~')
            f_t = float(g @ bR[acts[t - 1], :, 1, obs[t - 1]])  # r = 1 slice
            p_rt1 += w * f_t
        per_step.append(p_rt1)
        V += 1.0 * p_rt1                       # rewards in {0,1}: E[R_t] = p(r_t=1)
    return float(V), per_step


# ============================================================== (5) big-env bridges
def big_emission_block(q0, q1):
    """Per-x marker emission block Em2[u, m] = P(m | u); invertible iff q0 != q1."""
    return np.array([[1 - q0, q0],
                     [1 - q1, q1]])


def big_bridge_R(env, a, r_val, o, x_grid=None):
    """On-demand true reward bridge b_R(a, o~=(x,m~), r, o) for live queries.

    For each core x, solve Em2 @ b[(x,·)] = target where
    target[u] = p(o | s=(u,x)) * P(R = r_val | s=(u,x), a).
    Returns (len(x_grid), 2) array over (x, m~). Terminal o handled via 1x1 blocks
    upstream (not needed in Phase 1 checks).
    """
    aug, q0, q1 = env["aug"], env["q0"], env["q1"]
    P = aug["P"]
    Em2inv = np.linalg.inv(big_emission_block(q0, q1))
    x_grid = np.arange(N_X) if x_grid is None else np.asarray(x_grid)

    # p(o | s=(u,x)): nonzero only if o is a live obs with core(o) == x, or terminal
    if o >= 2 * N_X:
        raise ValueError("Phase 1 bridge queries restricted to live observation o")
    xo, mo = o // 2, o % 2
    p_o_ux = np.zeros((N_U, x_grid.size))
    match = x_grid == xo
    p_o_ux[0, match] = q0 if mo == 1 else (1 - q0)
    p_o_ux[1, match] = q1 if mo == 1 else (1 - q1)

    # P(R = r_val | s, a) on live states
    out = np.zeros((x_grid.size, 2))
    for i, x in enumerate(x_grid):
        tgt = np.zeros(N_U)
        for u in range(N_U):
            s = u * N_X + x
            if r_val == 1:
                p_r = P[a, s, TERM_POS]
            elif r_val == -1:
                p_r = P[a, s, TERM_NEG]
            else:
                p_r = 1.0 - P[a, s, TERM_POS] - P[a, s, TERM_NEG]
            tgt[u] = p_o_ux[u, i] * p_r
        out[i] = Em2inv @ tgt
    return out


def big_bridge_selftest(env, rng, n_queries=64):
    """Residual self-test: verify Em2 @ b == target at random (a, r, o, x) queries."""
    aug, q0, q1 = env["aug"], env["q0"], env["q1"]
    Em2 = big_emission_block(q0, q1)
    P = aug["P"]
    worst = 0.0
    for _ in range(n_queries):
        a = int(rng.integers(N_A))
        r_val = int(rng.choice([-1, 0, 1]))
        x = int(rng.integers(N_X))
        o = 2 * x + int(rng.integers(2))       # query at a live obs matching x
        b = big_bridge_R(env, a, r_val, o, x_grid=[x])[0]      # (2,)
        xo, mo = o // 2, o % 2
        for u in range(N_U):
            s = u * N_X + x
            p_o = (q1 if u == 1 else q0) if mo == 1 else ((1 - q1) if u == 1 else (1 - q0))
            p_o = p_o if x == xo else 0.0
            if r_val == 1:
                p_r = P[a, s, TERM_POS]
            elif r_val == -1:
                p_r = P[a, s, TERM_NEG]
            else:
                p_r = 1.0 - P[a, s, TERM_POS] - P[a, s, TERM_NEG]
            lhs = float(Em2[u] @ b)
            worst = max(worst, abs(lhs - p_o * p_r))
    return worst


# ============================================================== (6) naive baseline
def fit_naive_mdp(O, A, R, n_obs, n_act, smooth=0.0):
    """Count-based observation-MDP fit from confounded logs (pooled over t).

    Returns P_hat (n_act, n_obs, n_obs), r_hat (n_obs, n_act), p1_hat (n_obs,),
    visit_mask (n_obs, n_act). Unvisited (o, a): self-loop, r_hat = 0.
    """
    N, T = O.shape
    counts = np.zeros((n_act, n_obs, n_obs))
    r_sum = np.zeros((n_obs, n_act))
    sa_cnt = np.zeros((n_obs, n_act))

    for t in range(T):
        o_t, a_t, r_t = O[:, t], A[:, t], R[:, t]
        np.add.at(r_sum, (o_t, a_t), r_t)
        np.add.at(sa_cnt, (o_t, a_t), 1.0)
        if t < T - 1:
            o_next = O[:, t + 1]
            np.add.at(counts, (a_t, o_t, o_next), 1.0)

    P_hat = counts + smooth
    row = P_hat.sum(axis=2)
    visit_tx = row > 0
    for a in range(n_act):
        z = ~visit_tx[a]
        P_hat[a][z] = 0.0
        P_hat[a][z, np.arange(n_obs)[z]] = 1.0             # unvisited: self-loop
        nz = visit_tx[a]
        P_hat[a][nz] /= P_hat[a][nz].sum(axis=1, keepdims=True)

    r_hat = np.divide(r_sum, sa_cnt, out=np.zeros_like(r_sum), where=sa_cnt > 0)
    p1_hat = np.bincount(O[:, 0], minlength=n_obs).astype(float)
    p1_hat /= p1_hat.sum()
    return P_hat, r_hat, p1_hat, sa_cnt > 0


def naive_value(P_hat, r_hat, p1_hat, pi_obs, T):
    """Forward DP on the fitted observation-MDP: the naive uncorrected evaluator."""
    mu = p1_hat.copy()
    V = 0.0
    for t in range(T):
        V += float((mu[:, None] * pi_obs * r_hat).sum())
        if t < T - 1:
            flow = np.einsum("o,oa,aoz->z", mu, pi_obs, P_hat)
            mu = flow
    return V


def population_naive_mdp(env):
    """POPULATION (infinite-data) naive observation-MDP: exact expected values of the
    count-based fit under the behavior measure — isolates STRUCTURAL bias from
    sampling noise.

    Exact behavior-measure quantities (pooled over t, matching fit_naive_mdp):
      J_{t+1}(s') = sum_s J_t(s) sum_a pi_b,t(a|s) P[a,s,s']       (latent occupancy)
      n(o,a)      = sum_t sum_s J_t(s) W[s,o] pi_b,t(a|s)          (visit weight)
      r~(o,a)     = sum_t sum_s J_t(s) W[s,o] pi_b,t(a|s) ER(a,s) / n(o,a)
      C(o,a,o')   = sum_{t<T} sum_{s,s'} J_t(s) W[s,o] pi_b,t(a|s) P[a,s,s'] W[s',o']
      P~(o'|o,a)  = C / row-sums;   p~_1(o) = sum_s J_1(s) W[s,o]
    where W[s,o] = p(o|s) is the (dense) emission matrix of the marker channel.
    Behavior acts on the LATENT state and never sees the marker, so
    P(o,a|s) = W[s,o] * pi_b(a|s) factorizes exactly.
    """
    aug, T, q0, q1 = env["aug"], env["T"], env["q0"], env["q1"]
    P, ER, p_init, pi_b = aug["P"], aug["ER"], aug["p_init"], env["pi_b"]

    # dense emission matrix W[s, o]
    W = np.zeros((S_AUG, N_OBS))
    x = np.arange(N_X)
    for u, q in ((0, q0), (1, q1)):
        W[u * N_X + x, 2 * x] = 1 - q
        W[u * N_X + x, 2 * x + 1] = q
    W[TERM_NEG, OBS_NEG] = 1.0
    W[TERM_POS, OBS_POS] = 1.0

    J = p_init.copy()
    n_oa = np.zeros((N_OBS, N_A))
    r_sum = np.zeros((N_OBS, N_A))
    C = np.zeros((N_A, N_OBS, N_OBS))
    p1_pop = W.T @ J

    for t in range(T):
        B = J[:, None] * pi_b[t]                       # (S, A): J_t(s) pi_b,t(a|s)
        n_oa += W.T @ B
        r_sum += W.T @ (B * ER.T)
        if t < T - 1:
            for a in range(N_A):
                C[a] += (W * B[:, a][:, None]).T @ (P[a] @ W)
        J = np.einsum("s,sa,asz->z", J, pi_b[t], P)

    P_pop = C.copy()
    row = P_pop.sum(axis=2)
    for a in range(N_A):
        z = row[a] <= 0
        P_pop[a][z] = 0.0
        P_pop[a][z, np.arange(N_OBS)[z]] = 1.0
        nz = ~z
        P_pop[a][nz] /= P_pop[a][nz].sum(axis=1, keepdims=True)
    r_pop = np.divide(r_sum, n_oa, out=np.zeros_like(r_sum), where=n_oa > 0)
    return P_pop, r_pop, p1_pop, n_oa


# ============================================================== (7) MC sanity
def mc_value_big(env, pi_obs, N, seed):
    """On-policy Monte-Carlo value of an obs-policy in the true environment."""
    from simulated_env import sample_trajectories
    rng = np.random.default_rng(seed)
    pis = pi_obs if isinstance(pi_obs, list) else [pi_obs] * env["T"]
    data = sample_trajectories(env["aug"], pis, N, env["T"], rng,
                               q0=env["q0"], q1=env["q1"], eps0=env["eps0"],
                               policy_type="obs")
    ret = data["R"].sum(axis=1)
    return float(ret.mean()), float(ret.std(ddof=1) / np.sqrt(N))


def mc_value_toy(params, pi_obs, N, seed):
    rng = np.random.default_rng(seed)
    data = toy.sample_trajectories(params, N, rng, policy=pi_obs, policy_type="obs")
    ret = data["R"].sum(axis=1)
    return float(ret.mean()), float(ret.std(ddof=1) / np.sqrt(N))
