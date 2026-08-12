"""
model_free_proximal.py — model-FREE proximal OPE baseline (Agent 2 design),
the policy-DEPENDENT value-bridge counterpart to the anchor's model bridges.

Grounded in Shi, Uehara, Huang & Jiang (ICML 2022), "A Minimax Learning Approach
to Off-Policy Evaluation in Confounded POMDPs" (Option A). We estimate a per-step
VALUE bridge b_V(a,o) via a minimax/adversarial GMM conditional-moment restriction
that uses the pre-decision negative control O_0 as the past-proxy instrument, then
read policy value directly off the bridge. This imports NOTHING from the pinned
oracle/toy/bridge/value_plugin/ellipsoid modules, so V1/V4 are byte-identically
preserved.

NOTATION MAP (Shi -> ours). O_0 is Shi's pre-decision proxy O_{-1}:
  W (bridge argument)   = (A_t, O_t)              treatment-side proxy
  X (instrument)        = (A_t, O_0)              O_0 is the PAST proxy
  Y (response)          = R_t * pi_e(A_t | O_t)   reward x on-policy prob
Value bridge b_V^[t](A_t,O_t) solves the conditional moment restriction with a
BACKWARD-RECURSIVE response that carries the target-policy continuation value:
  E[ b_V^[t](A_t,O_t) | A_t, O_0 ] = E[ (R_t + V_{t+1}(O_{t+1}))*pi_e(A_t|O_t) | A_t,O_0 ]
  V_t(o) := sum_a b_V^[t](a, o),   V_{T+1} := 0.                                   (VB-t)
The residual  Y - b_V^[t](W)  is mean-zero given the instrument X. Shi's minimax form
  min_g max_f  E[(Y - g(W)) f(X)] - (lam/2) E[f(X)^2]
has, for a tabular one-hot critic class f indexed by X, a closed-form inner max,
collapsing to the regularized GLS (2SLS/GMM) solve
  beta = (A Wreg A^T + N rho I)^{-1} A Wreg c,
A = Phi_W^T Phi_X (|W| x |X|), Wreg the critic-Gram weighting, c = Phi_X^T Y.
Finite-horizon value uses ONLY the (policy-independent) INITIAL observation marginal:
  J(pi_e) = E_{o ~ nu_1}[ V_1(o) ] = sum_o nu_1(o) sum_a b_V^[1](a, o).
The recursion propagates the target policy's future value backward, so unlike a
per-stage direct method it is correct OFF-policy (a per-stage sum against the
behavior marginal is biased whenever pi_e shifts state occupancy -- verified). The
bridge is POLICY-DEPENDENT: fit() takes the target policy and re-fits per policy.

Why this is the right model-free counterweight: the shared tabular critic pools the
reward signal across all observations (plus Tikhonov ridge), rather than inverting
an isolated 2x2 per cell like the model-based estimator -- a structurally different
variance profile to compare head-to-head across N and confounding intensity.
"""

import numpy as np


def _infer_reward_levels(R):
    return np.unique(R)


class MinimaxValueBridgeOPE:
    """Model-free proximal (value-bridge) OPE via tabular minimax GMM.

    Parameters
    ----------
    n_obs, n_act, n_o0, T : discrete space sizes / horizon
    epsilon : critic-Gram ridge; default 1/N (parametric rate)
    rho     : bridge Tikhonov ridge; default 0.03/sqrt(N) (mirrors the MB schedule)
    instrument_weighting : 'gram' (default) | 'invvar' | 'identity'
    """

    def __init__(self, n_obs, n_act, n_o0, T, reward_levels=None,
                 epsilon=None, rho=None, instrument_weighting="gram", seed=0):
        self.n_obs, self.n_act, self.n_o0, self.T = n_obs, n_act, n_o0, T
        self.reward_levels = reward_levels
        self.epsilon, self.rho = epsilon, rho
        self.instrument_weighting = instrument_weighting
        self.seed = seed
        self.bV_hat = None
        self.nu_O = None
        self.diagnostics = []

    def fit(self, O0, O, A, R, target_policy):
        n_obs, n_act, n_o0, T = self.n_obs, self.n_act, self.n_o0, self.T
        N = O.shape[0]
        pi_e = np.asarray(target_policy, dtype=np.float64)   # (n_obs, n_act)
        eps = self.epsilon if self.epsilon is not None else 1.0 / N
        rho = self.rho if self.rho is not None else 0.03 / np.sqrt(N)
        nW, nX = n_act * n_obs, n_act * n_o0
        self.bV_hat = np.zeros((T, n_act, n_obs))
        self.diagnostics = [None] * T

        # BACKWARD recursion: V_{T+1}=0; stage t response carries V_{t+1}(O_{t+1}).
        V_next = np.zeros(n_obs)                   # V_{t+1}(o'); zero at t=T
        for t in range(T, 0, -1):
            o_t, a_t, r_t = O[:, t - 1], A[:, t - 1], R[:, t - 1]
            w = a_t * n_obs + o_t                 # W = (A_t, O_t)
            x = a_t * n_o0 + O0                    # X = (A_t, O_0)  (instrument)
            cont = V_next[O[:, t]] if t < T else np.zeros(N)   # V_{t+1}(O_{t+1})
            y = (r_t + cont) * pi_e[o_t, a_t]     # response with continuation

            # The instrument X=(A_t,O_0) and bridge argument W=(A_t,O_t) SHARE the
            # action, so the GLS system is BLOCK-DIAGONAL across actions: solve each
            # (n_obs x n_obs) action block separately instead of a
            # (n_act*n_obs)^2 dense matrix (which is ~1GB at benchmark scale). Per
            # action the instrument has only n_o0 columns, so LHS_a = low-rank +
            # ridge; the ridge min-norm solution is unique. NOTE: with a small
            # (binary) proxy this under-identifies a large-|O| bridge -- an honest
            # limitation surfaced at 720x2 scale (see the benchmark discussion).
            bV_t = np.zeros((n_act, n_obs))
            for a in range(n_act):
                sel = a_t == a
                if not sel.any():
                    continue
                wa = o_t[sel]                       # W-index within action a: O_t
                xa = O0[sel]                         # X-index within action a: O_0
                ya = y[sel]
                Ga = np.bincount(xa, minlength=n_o0).astype(float)
                Acr_a = np.zeros((n_obs, n_o0))
                np.add.at(Acr_a, (wa, xa), 1.0)
                cXy_a = np.zeros(n_o0)
                np.add.at(cXy_a, xa, ya)
                if self.instrument_weighting == "invvar":
                    Wa = 1.0 / (Ga + eps)
                elif self.instrument_weighting == "identity":
                    Wa = np.ones(n_o0)
                else:  # 'gram'
                    Wa = 1.0 / (Ga + N * eps)
                AW_a = Acr_a * Wa[None, :]                       # (n_obs, n_o0)
                # Woodbury: (AW_a Acr_a^T + Nrho I)^{-1} rhs, inner inverse is n_o0xn_o0
                M = Acr_a.T @ AW_a + N * rho * np.eye(n_o0)      # (n_o0, n_o0)
                rhs = AW_a @ cXy_a                               # (n_obs,)
                inner = Acr_a.T @ rhs                            # (n_o0,)
                bV_t[a] = (rhs - AW_a @ np.linalg.solve(M, inner)) / (N * rho)
            self.bV_hat[t - 1] = bV_t
            V_next = bV_t.sum(axis=0)             # V_t(o) = sum_a b_V^[t](a,o)
            # diagnostic: instrument coverage (binary O_0 vs |O| bridge -> the
            # under-identification that limits this baseline at benchmark scale)
            Gx = np.bincount(a_t * n_o0 + O0, minlength=nX).astype(float)
            self.diagnostics[t - 1] = dict(
                stage=t, unseen_instrument_frac=float((Gx == 0).mean()),
                proxy_dim=n_o0, bridge_dim_per_action=n_obs)

        # J = E_{o ~ nu_1}[ V_1(o) ], nu_1 = policy-INDEPENDENT initial-obs marginal
        nu1 = np.bincount(O[:, 0], minlength=n_obs).astype(float)
        self.nu1 = nu1 / max(nu1.sum(), 1.0)
        self._V1 = self.bV_hat[0].sum(axis=0)
        return self

    def value(self):
        """J(pi_e) = E_{o~nu_1}[ V_1(o) ]  (initial-marginal, off-policy-correct)."""
        return float((self._V1 * self.nu1).sum())

    def estimate_value(self, dataset, target_policy):
        """Convenience: fit on a dataset dict {O0,O,A,R} for a target policy and
        return the OPE value (bridge re-fit per policy)."""
        self.fit(dataset["O0"], dataset["O"], dataset["A"], dataset["R"],
                 target_policy)
        return self.value()
