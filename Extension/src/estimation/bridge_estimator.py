"""
bridge_estimator.py — Phase 2 estimation core: the anchor's two-stage bridge
estimator, instantiated exactly in the tabular setting (one-hot/delta kernels).

Per stage t = 1..T and bridge family (R: reward-emission, D: dynamic-emission):

  variables    W_t = (A_t, O_t)                       w-index, |W| = n_act * n_obs
               X_t = (A_t, H_{t-1}, O_0)              conditioning; |X_t| grows with t
               Y_t = (R_t, O_t)                       reward-bridge index, |Y| = n_r * n_obs
               Z_t = (O_{t+1}, O_t)                   dynamic-bridge index, |Z| = n_obs^2
  moment       E[ b(W_t, y) | X_t ] = p(y | X_t)      for ALL index values y

STAGE 1 (on the N1 split) — with one-hot features the anchor's closed forms become:
  ridge CME    mu_hat(w | x) = count1(w, x) / (count1(x) + N1 * lambda1)
               [push-through-equivalent to Gamma = (K_X + N1 lambda1 I)^{-1} K_{XX'};
                columns of M_hat = Phi_W Gamma are exactly mu_hat(.|x'_n)]
  pmf target   p_hat(y | x) = count1(y, x) / count1(x)   (MLE; marginal fallback for
               x unseen in stage 1 — such rows have mu_hat = 0 and are inert)

STAGE 2 (on the N2 split) — uniform index draws y''_n ~ Unif(Y_t), paired 1:1 with
stage-2 points x'_n; design vectors psi_n = mu_hat(.|x'_n) (x) e_{y''_n}:
  DUAL (paper's representer form):
      G = (Gamma^T K_W Gamma) hadamard K_{Y''} = (M_hat^T M_hat) hadamard K_{Y''}
      alpha = (G + N2 * lambda2 * I)^{-1} p_hat_vec          [(N2 x N2) solve]
      b_hat = sum_n alpha_n psi_n
  PRIMAL (paper's operator form, Eq. 12; identical by the push-through identity):
      (T_hat_2 + lambda2 I) vec(b_hat) = g_hat_2,
      T_hat_2 = Psi^T Psi / N2,  g_hat_2 = Psi^T p_hat_vec / N2   [(D x D) solve,
      D = |W| * |Y| — the right form when N2 >> D, as in the tabular setting]

Diagnostics logged per stage/family: eigenvalue extremes of T_hat_2, regularized
condition numbers cond(T_hat_2 + lambda2 I) and (when the dual is formed)
cond(G + N2 lambda2 I), plus the fraction of stage-2 points whose x was unseen in
stage 1 (coverage/ill-posedness monitor).

Consistency target: as N -> inf (lambda -> 0) the estimator converges to the
MINIMUM-FROBENIUS-NORM solution of the moment equations, which decouples per
(a, y)-slice into minimum-l2 solutions — exactly the pinv bridges produced by
oracle_module.toy_true_bridges. Hence L2(b_hat, b_oracle) -> 0 is the correct
recovery metric.

NO pessimism code here (Phase 2 later step).
"""

import numpy as np


# --------------------------------------------------------------------- encodings
def history_index(O, A, t, n_obs, n_act):
    """Flat index of h_{t-1} = (o_1, a_1, ..., o_{t-1}, a_{t-1}); t is 1-indexed."""
    N = O.shape[0]
    h = np.zeros(N, dtype=np.int64)
    for j in range(t - 1):
        h = h * (n_obs * n_act) + (O[:, j] * n_act + A[:, j])
    return h, (n_obs * n_act) ** (t - 1)


def stage_indices(O0, O, A, R, t, n_obs, n_act, n_o0, n_r):
    """Return (w, x, y, z, sizes) flat indices for stage t (1-indexed).
    z is None at t = T (no dynamic bridge at the last stage)."""
    a_t = A[:, t - 1]
    o_t = O[:, t - 1]
    r_t = R[:, t - 1].astype(np.int64)
    h, n_hist = history_index(O, A, t, n_obs, n_act)

    w = a_t * n_obs + o_t
    x = (a_t * n_hist + h) * n_o0 + O0
    y = r_t * n_obs + o_t
    z = (O[:, t] * n_obs + o_t) if t < O.shape[1] else None
    sizes = dict(n_w=n_act * n_obs, n_x=n_act * n_hist * n_o0,
                 n_y=n_r * n_obs, n_z=n_obs * n_obs)
    return w, x, y, z, sizes


# --------------------------------------------------------------------- estimator
class TabularBridgeEstimator:
    """Two-stage bridge estimator (tabular one-hot instantiation of the anchor).

    Parameters
    ----------
    n_obs, n_act, n_o0, n_r : discrete space sizes
    T          : horizon
    lambda1    : stage-1 CME ridge; default 1/N1  (the ridge enters the CME as
                 count(w,x)/(count(x) + N1*lambda1), i.e. N1*lambda1 pseudo-counts:
                 the anchor's N^{-1/(c+1)} schedules target infinite-dimensional
                 RKHSs; in the finite-dimensional well-specified tabular class the
                 parametric-rate choice N1*lambda1 = O(1) avoids gross shrinkage
                 while keeping the estimator well-defined on unseen cells)
    lambda2    : stage-2 ridge; default 0.03/sqrt(N2). Calibrated on the toy via a
                 coarse grid (see checkpoint_log entry): lambda2 must decay SLOWER
                 than 1/N so it dominates the stage-1 CME noise that leaks into the
                 weakly-identified (near-null) design directions — matching the
                 anchor's prescription that the stage-2 width dominates stage-1
                 error — while staying below the O(1/|Y|) signal eigenvalues.
    split_frac : N1 fraction (default 0.7; the grid confirms the anchor's N1 >> N2
                 prescription — stage-1 tables over the growing X_t need more data)
    mode       : 'primal' | 'dual' | 'auto' (auto = dual iff N2 <= dual_cap,
                 else primal; both are exactly equivalent — verified in tests)
    dual_cap   : max N2 for forming the (N2 x N2) Gram matrix G
    """

    def __init__(self, n_obs, n_act, n_o0, n_r, T,
                 lambda1=None, lambda2=None, split_frac=0.7,
                 mode="auto", dual_cap=3000, seed=0):
        self.n_obs, self.n_act, self.n_o0, self.n_r = n_obs, n_act, n_o0, n_r
        self.T = T
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.split_frac = split_frac
        self.mode, self.dual_cap = mode, dual_cap
        self.seed = seed
        self.bR_hat = None      # (T,  n_act, n_obs, n_r,  n_obs)  [a, o~, r,  o]
        self.bD_hat = None      # (T-1,n_act, n_obs, n_obs, n_obs) [a, o~, o~', o]
        self.diagnostics = []
        self.stage_store = {}   # per-stage ellipsoid geometry (H, b_hat, sigma2)

    # ---------------------------------------------------------------- stage 1
    @staticmethod
    def _stage1_tables(w1, x1, tgt1, n_w, n_x, n_tgt, N1, lam1):
        """Ridge CME table mu_hat (n_w, n_x) and MLE pmf table (n_tgt, n_x)."""
        cnt_wx = np.zeros((n_w, n_x))
        np.add.at(cnt_wx, (w1, x1), 1.0)
        cnt_x = np.bincount(x1, minlength=n_x).astype(np.float64)
        mu = cnt_wx / (cnt_x + N1 * lam1)[None, :]

        cnt_tx = np.zeros((n_tgt, n_x))
        np.add.at(cnt_tx, (tgt1, x1), 1.0)
        marg = cnt_tx.sum(axis=1) / max(N1, 1)
        seen = cnt_x > 0
        pmf = np.empty((n_tgt, n_x))
        pmf[:, seen] = cnt_tx[:, seen] / cnt_x[seen][None, :]
        pmf[:, ~seen] = marg[:, None]
        return mu, pmf, seen

    # ---------------------------------------------------------------- stage 2
    def _stage2_solve(self, mu, pmf, seen_x, x2, n_w, n_y, N2, lam2, rng, label):
        """Closed-form bridge solve. Returns b_hat (n_w, n_y) + diagnostics."""
        yidx = rng.integers(0, n_y, size=N2)              # y'' ~ Unif(Y), one per point
        M = mu[:, x2]                                     # (n_w, N2): mu_hat(.|x'_n)
        p_vec = pmf[yidx, x2]                             # p_hat(y''_n | x'_n)

        # primal design Psi (N2, n_w * n_y): psi_n = mu_n (x) e_{y''_n}
        D = n_w * n_y
        Psi = np.zeros((N2, D))
        cols = (np.arange(n_w)[None, :] * n_y + yidx[:, None])   # (N2, n_w)
        np.put_along_axis(Psi, cols, M.T, axis=1)

        T2 = Psi.T @ Psi / N2
        g2 = Psi.T @ p_vec / N2
        evals = np.linalg.eigvalsh(T2)
        cond_primal = (evals[-1] + lam2) / (evals[0] + lam2)

        # restricted-spectrum ill-posedness monitor (Phase-2 finding): per action a,
        # the occupancy-weighted design of mu_hat a-blocks has rank-|S| SIGNAL; its
        # 2nd-largest eigenvalue sigma2 is the smallest signal direction — the global
        # eig_min is pinned ~0 by the structural |O|>|S| null space and is blind to
        # identification collapse. sigma2 drives the pessimism widths downstream.
        sigma2_per_a = []
        signal_basis = []                      # per-action top-k eigvecs (eigengap rank)
        for a in range(self.n_act):
            Ma = M[a * self.n_obs:(a + 1) * self.n_obs, :]        # (n_obs, N2)
            Wa = Ma @ Ma.T / N2
            ev, evec = np.linalg.eigh(Wa)                         # ascending
            desc = ev[::-1]
            if desc.size < 2:                                    # guard empty ratios
                k_a = desc.size                                  # (audit A-low: crash guard)
            else:
                # The original rule divided by max(desc[1:], 1e-300). Trailing
                # eigenvalues of a PSD matrix are machine noise around zero and
                # are frequently NEGATIVE, so that floor turned each of them into
                # a ratio of order 1e282 and argmax selected float sign noise
                # rather than the real spectral gap. The selected ranks were not
                # the rule's judgement and did not repeat across seeds.
                #
                # Floor relative to the largest eigenvalue instead. Anything below
                # it is numerically zero, so consecutive zeros give ratio ~1 and
                # cannot win the argmax; the largest genuine gap does.
                floor = max(desc[0], 0.0) * 1e-9
                if floor <= 0.0:                                 # degenerate: all ~0
                    k_a = 1
                else:
                    ratios = np.maximum(desc[:-1], floor) / np.maximum(desc[1:], floor)
                    k_a = int(np.argmax(ratios)) + 1             # eigengap rank rule
            sigma2_per_a.append(float(desc[k_a - 1]))            # smallest KEPT eig
            signal_basis.append(evec[:, ::-1][:, :k_a].copy())
        sigma2_signal = float(min(sigma2_per_a))

        use_dual = (self.mode == "dual" or
                    (self.mode == "auto" and N2 <= self.dual_cap))
        diag = dict(label=label, N2=int(N2), D=int(D), lambda2=float(lam2),
                    eig_min_T2=float(evals[0]), eig_max_T2=float(evals[-1]),
                    cond_T2_reg=float(cond_primal),
                    sigma2_signal=sigma2_signal,
                    sigma2_per_action=sigma2_per_a,
                    unseen_x2_frac=float(1.0 - seen_x[x2].mean()),
                    solver="dual" if use_dual else "primal")

        H = T2 + lam2 * np.eye(D)
        b_primal = np.linalg.solve(H, g2).reshape(n_w, n_y)

        if use_dual:
            # G = (M^T M) hadamard K_{Y''};  alpha = (G + N2 lam2 I)^{-1} p_hat
            KY = (yidx[:, None] == yidx[None, :]).astype(np.float64)
            G = (M.T @ M) * KY
            gev = np.linalg.eigvalsh(G + N2 * lam2 * np.eye(N2))
            diag["cond_G_reg"] = float(gev[-1] / gev[0])
            alpha = np.linalg.solve(G + N2 * lam2 * np.eye(N2), p_vec)
            b_dual = np.zeros((n_w, n_y))
            np.add.at(b_dual.T, yidx, (alpha[None, :] * M).T)    # sum_n alpha_n psi_n
            diag["primal_dual_gap"] = float(np.abs(b_dual - b_primal).max())
            b_hat = b_dual
        else:
            b_hat = b_primal

        # ellipsoid geometry for the pessimism layer: L_hat(b) - L_hat(b_hat) =
        # (b - b_hat)^T H (b - b_hat) with H = T_hat_2 + lam2 I (ridge term included,
        # anchor Eq. 121); region conf(xi) is the exact ellipsoid {quadform <= xi}.
        self.stage_store[label] = dict(
            H=H, b_hat_vec=b_hat.ravel().copy(), n_w=n_w, n_y=n_y,
            N2=int(N2), lam2=float(lam2), sigma2_signal=sigma2_signal,
            signal_basis=signal_basis)

        self.diagnostics.append(diag)
        return b_hat

    # ---------------------------------------------------------------- fit
    def fit(self, O0, O, A, R):
        n_obs, n_act, n_o0, n_r, T = (self.n_obs, self.n_act, self.n_o0,
                                      self.n_r, self.T)
        N = O.shape[0]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(N)
        N1 = int(N * self.split_frac)
        i1, i2 = perm[:N1], perm[N1:]
        N2 = N - N1
        lam1 = self.lambda1 if self.lambda1 is not None else 1.0 / N1
        lam2 = self.lambda2 if self.lambda2 is not None else 0.03 / np.sqrt(N2)
        self.fitted_ = dict(N=N, N1=N1, N2=N2, lambda1=lam1, lambda2=lam2)
        self.diagnostics = []
        self.stage_store = {}

        self.bR_hat = np.zeros((T, n_act, n_obs, n_r, n_obs))
        self.bD_hat = np.zeros((T - 1, n_act, n_obs, n_obs, n_obs))

        for t in range(1, T + 1):
            w, x, y, z, sz = stage_indices(O0, O, A, R, t, n_obs, n_act, n_o0, n_r)
            w1, x1, y1 = w[i1], x[i1], y[i1]
            x2 = x[i2]

            # ---- reward bridge b_R^[t]
            muR, pmfR, seenR = self._stage1_tables(
                w1, x1, y1, sz["n_w"], sz["n_x"], sz["n_y"], N1, lam1)
            bR = self._stage2_solve(muR, pmfR, seenR, x2, sz["n_w"], sz["n_y"],
                                    N2, lam2, rng, label=f"bR_t{t}")
            self.bR_hat[t - 1] = bR.reshape(n_act, n_obs, n_r, n_obs)

            # ---- dynamic bridge b_D^[t] (t <= T-1)
            if z is not None:
                z1 = z[i1]
                muD, pmfD, seenD = self._stage1_tables(
                    w1, x1, z1, sz["n_w"], sz["n_x"], sz["n_z"], N1, lam1)
                bD = self._stage2_solve(muD, pmfD, seenD, x2, sz["n_w"], sz["n_z"],
                                        N2, lam2, rng, label=f"bD_t{t}")
                self.bD_hat[t - 1] = bD.reshape(n_act, n_obs, n_obs, n_obs)
        return self

    # ---------------------------------------------------------------- accessors
    def bridges(self):
        """b_R (T, a, o~, r, o) and b_D (T-1, a, o~, o~', o) tables."""
        return self.bR_hat, self.bD_hat
