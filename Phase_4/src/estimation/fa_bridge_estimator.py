"""
fa_bridge_estimator.py -- two-stage bridge estimator for the PAPER-FAITHFUL
setting: continuous state/observation, FINITE action space.

Same two-stage construction as Phase 3's continuous estimator; the only change
is the bridge-side feature map, which is what the finite action space dictates:

    phi_W(a, o) = [ onehot(a) , o ]        (K + 1 dimensional)

i.e. a DELTA kernel on the finite action coordinate tensored with a LINEAR
kernel on the continuous observation coordinate. Each action level gets its own
free intercept, so the estimator never assumes the reward/transition effect is
linear in the dose -- it lets the data decide, which is the whole point of
treating A as a finite set rather than a real interval.

WHY THE PRIMAL SOLVE IS MANDATORY HERE. The feature space above is finite
((K+1)-dimensional), so the N1 x N1 dual Gram matrix has rank <= K+1 and its
remaining eigenvalues sit at the numerical floor. Phase 3 hit exactly this with
a linear kernel: the dual system is meaningless in those null directions, and
although one particular projected quantity happened to be invariant to the
instability, H and b_hat feed the pessimism ellipsoid directly, where nothing
protects them. We therefore solve in the (K+1)-dimensional primal throughout --
not an approximation, just the well-posed way to solve the same ridge problem
when the feature space is finite.

STAGE-1 instrument side keeps the RBF kernel on the history vector
X_t = [A_t, O_{t-1}, A_{t-1}, ..., O_1, A_1, O_0], unchanged from Phase 3.
"""

import numpy as np


def _median_bandwidth(X):
    X = np.atleast_2d(X)
    if X.shape[0] < 2:
        return 1.0
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    iu = np.triu_indices_from(d2, k=1)
    return float(np.sqrt(max(float(np.median(d2[iu])), 1e-12)))


def rbf_kernel(X, Y, bandwidth):
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2.0 * bandwidth ** 2))


def effective_dimension(K, N, lam):
    n = K.shape[0]
    return float(np.trace(K @ np.linalg.solve(K + N * lam * np.eye(n), K)))


def action_features(a_idx, n_actions):
    """onehot(a) -> (N, K)."""
    N = len(a_idx)
    F = np.zeros((N, n_actions))
    F[np.arange(N), a_idx] = 1.0
    return F


def bridge_features(a_idx, o, n_actions):
    """phi_W(a,o) = [onehot(a), o] -> (N, K+1)."""
    return np.concatenate([action_features(a_idx, n_actions),
                           np.asarray(o).reshape(-1, 1)], axis=1)


def _history_vec(O, A, O0, t):
    """X_t = [A_t, O_{t-1}, A_{t-1}, ..., O_1, A_1, O_0]; shape (N, 2t)."""
    cols = [A[:, t - 1]]
    for j in range(t - 1, 0, -1):
        cols.append(O[:, j - 1])
        cols.append(A[:, j - 1])
    cols.append(O0)
    return np.stack(cols, axis=1)


class FiniteActionBridgeEstimator:
    """Two-stage estimator with a finite-action bridge feature map.

    lambda1, lambda2 default to 1/N1 and 0.03/sqrt(N2) -- the same calibration
    Phase 3 arrived at and re-tested, kept here so that any difference in
    results is attributable to the action space and not to retuning.
    """

    def __init__(self, T, n_actions, lambda1=None, lambda2=None,
                 split_frac=0.7, seed=0):
        self.T = T
        self.n_actions = n_actions
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.split_frac = split_frac
        self.seed = seed
        self.stage_R, self.stage_D = {}, {}

    def _solve_stage(self, X1, X2, W1, y2, lam1, lam2, N1, N2):
        bw = _median_bandwidth(X1)
        K_X11 = rbf_kernel(X1, X1, bw)
        K_X12 = rbf_kernel(X1, X2, bw)
        Gamma = np.linalg.solve(K_X11 + N1 * lam1 * np.eye(N1), K_X12)  # (N1,N2)
        n_eff = effective_dimension(K_X11, N1, lam1)

        # PRIMAL: project the stage-1 CME weights onto the finite feature space.
        Wt = W1.T @ Gamma                                # (K+1, N2)
        H = (Wt @ Wt.T) / N2 + lam2 * np.eye(W1.shape[1])
        g = (Wt @ y2) / N2
        theta = np.linalg.solve(H, g)
        return dict(theta=theta, theta_hat=theta, H=H, N1=int(N1), N2=int(N2),
                    lambda2=float(lam2), n_eff_x=float(n_eff))

    def fit(self, O0, O, A_idx, R):
        N = O.shape[0]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(N)
        N1 = int(N * self.split_frac)
        i1, i2 = perm[:N1], perm[N1:]
        N2 = N - N1
        lam1 = self.lambda1 if self.lambda1 is not None else 1.0 / N1
        lam2 = self.lambda2 if self.lambda2 is not None else 0.03 / np.sqrt(N2)
        self.fitted_ = dict(N=N, N1=N1, N2=N2, lambda1=lam1, lambda2=lam2)
        self.stage_R, self.stage_D = {}, {}

        A_real = A_idx.astype(float)
        for t in range(1, self.T + 1):
            X = _history_vec(O, A_real, O0, t)
            W = bridge_features(A_idx[:, t - 1], O[:, t - 1], self.n_actions)
            X1, X2, W1 = X[i1], X[i2], W[i1]
            self.stage_R[t] = self._solve_stage(X1, X2, W1, R[i2, t - 1],
                                                lam1, lam2, N1, N2)
            if t < self.T:
                self.stage_D[t] = self._solve_stage(X1, X2, W1, O[i2, t],
                                                    lam1, lam2, N1, N2)
        return self

    def eval_and_feature(self, family, t, coeffs, feat):
        """Evaluate b^[t] at a given (already-built) feature row using the GIVEN
        coefficient vector -- needed so the pessimism coordinate descent can
        evaluate trial points inside the confidence ellipsoid."""
        feat = np.atleast_2d(feat)
        return (feat @ coeffs), feat

    def bridges(self):
        return self.stage_R, self.stage_D
