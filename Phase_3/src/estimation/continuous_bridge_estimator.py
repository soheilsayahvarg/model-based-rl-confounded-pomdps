"""
continuous_bridge_estimator.py -- kernel two-stage bridge estimator for
CONTINUOUS state/action spaces, the RKHS analogue of bridge_estimator.py.

DERIVATION (worked from first principles below, not copied from a remembered
formula -- see docs/continuous_extension.md for why: a mis-remembered index
convention here would silently produce a wrong-but-plausible-looking estimator,
exactly the failure mode this whole project is built to catch via oracle
verification, so it is derived and re-checked directly instead).

MEAN-VALUE MOMENT RESTRICTION (a deliberate simplification of the anchor
paper's full distributional bridge -- see docs/continuous_extension.md
"Theoretical & Mathematical Adaptation" for the justification): find b(w),
w=(a, o), such that

    E[ b(A_t, O_t) | X_t = x ]  =  E[ Y_t | X_t = x ]      for all x,

where Y_t = R_t (reward bridge) or O_{t+1} (dynamics bridge, t < T), and
X_t = (A_t, H_{t-1}, O_0) is the same conditioning set as the tabular
estimator, just a real-valued concatenated vector instead of a hashed index
(no combinatorial blow-up here -- a kernel takes a longer feature vector
just as easily as a short one; the discrete estimator's hashing trick was
specifically a fix for a problem continuous features don't have).

This is exactly a (nonparametric) instrumental-variable regression: W=(A,O) is
confounded (correlated with the hidden state through both channels), X is the
instrument (informative via the negative control O_0, independent of the
residual by Assumption 3.1), Y is the outcome.

TWO-STAGE KERNEL RIDGE SOLVE. Let K_X = kernel Gram matrix on X-points, K_W =
kernel Gram matrix on W-points (see kernels below). Stage 1 (on N1 points):

    Gamma = (K_X(X1,X1) + N1*lambda1*I)^{-1} K_X(X1, X2)          (N1 x N2)

-- the usual kernel-ridge conditional-mean-embedding weights (this matches
bridge_estimator.py's own docstring, which already describes this exact
"push-through" form as the general RKHS case the tabular code specializes).

Stage 2 (on N2 points): represent b(w) = sum_i beta_i k_W(w, w1_i) (representer
theorem, basis = STAGE-1 W points) and minimize the empirical, ridge-penalized
version of E_X[ (E[b(W)|X] - E[Y|X])^2 ]:

    J(beta) = (1/N2) || Z^T beta - y2 ||^2  +  lambda2 * beta^T K_W(W1,W1) beta,
    Z := K_W(W1,W1) @ Gamma                                        (N1 x N2)

(the j-th column of Z^T beta is exactly E_hat[b(W)|X=x2_j], since Gamma's j-th
column gives the stage-1 CME weights for query x2_j, and b(w1_i)=(K_W11 @ beta)_i
for the representer basis). Setting the gradient to zero gives the closed form

    H = Z @ Z.T / N2  +  lambda2 * K_W(W1,W1),     g = Z @ y2 / N2
    beta = H^{-1} g                                                (N1,)

H is exactly the "T_hat_2 + lambda2*I" object ellipsoid_opt.py expects, with
K_W(W1,W1) playing the role of "I": when K_W is chosen as a one-hot/delta
kernel (the tabular case), K_W(W1,W1) = I exactly and this formula reduces to
the tabular estimator's own primal solve -- a useful correctness check on the
derivation itself, not just an analogy.

KERNELS. K_X: RBF (median-heuristic bandwidth) on the concatenated history
vector -- a smoothing operator, no realizability requirement. K_W: a PURE
LINEAR kernel k(w,w') = w . w' on w=(a,o). This is a deliberate choice for
this first pass: the true bridges (continuous_env.true_bridge_coeffs) are
EXACTLY linear with zero intercept, so a linear kernel's RKHS contains them
exactly, giving a clean "recovery error -> 0 as N -> infinity" check
(run_continuous_benchmark.py [C3]) analogous to run_phase2_check.py's [P2].
An RBF option is exposed for w_kernel='rbf' but is not the primary verified
path in this pass (see docs/continuous_extension.md limitations).

SPECTRAL MONITOR (continuous analogue of the tabular sigma2_signal fix). The
tabular project's fix for a MISLEADING cond(G) was to monitor the smallest
SIGNAL eigenvalue via a hard eigengap -- correct there because the null space
was STRUCTURAL (|O|>|S|, exactly zero signal beyond it). A continuous kernel's
Gram matrix has smooth spectral decay, not a structural wall, so a hard
eigengap rule does not transfer. The honest continuous analogue is the
EFFECTIVE DIMENSION (Caponnetto & De Vito 2007; standard in kernel ridge
regression theory): N_eff(lambda) = trace( K_X (K_X + N1*lambda*I)^{-1} ),
computed below as `n_eff_x`. It measures "how many directions are actually
well-identified" continuously instead of discretely.
"""

import numpy as np


def linear_kernel(X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    return X @ Y.T


def _median_bandwidth(X):
    X = np.atleast_2d(X)
    if X.shape[0] < 2:
        return 1.0
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    iu = np.triu_indices_from(d2, k=1)
    med = float(np.median(d2[iu]))
    return float(np.sqrt(max(med, 1e-12)))


def rbf_kernel(X, Y, bandwidth):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2.0 * bandwidth ** 2))


def effective_dimension(K, N, lam):
    """N_eff(lambda) = trace( K (K + N*lambda*I)^{-1} ) -- the continuous
    spectral-identification monitor (see module docstring)."""
    n = K.shape[0]
    return float(np.trace(K @ np.linalg.solve(K + N * lam * np.eye(n), K)))


def _history_vec(O, A, O0, t):
    """X_t = [A_t, O_{t-1}, A_{t-1}, ..., O_1, A_1, O_0], t is 1-indexed.
    Shape (N, 2t): grows LINEARLY with t (no combinatorial blow-up)."""
    N = O.shape[0]
    cols = [A[:, t - 1]]
    for j in range(t - 1, 0, -1):
        cols.append(O[:, j - 1])
        cols.append(A[:, j - 1])
    cols.append(O0)
    return np.stack(cols, axis=1)


class ContinuousBridgeEstimator:
    """Two-stage kernel bridge estimator (see module docstring for the solve).

    Parameters
    ----------
    T          : horizon
    lambda1, lambda2 : ridge strengths; default 1/N1 and 0.03/sqrt(N2), the
                 SAME calibration the tabular estimator uses (bridge_estimator.
                 py), kept for continuity -- not re-derived for this PoC.
    split_frac : Stage-1 fraction of N (default 0.7, matches the tabular default)
    w_kernel   : 'linear' (default, exactly realizable -- see module docstring)
                 or 'rbf'
    """

    def __init__(self, T, lambda1=None, lambda2=None, split_frac=0.7,
                 w_kernel="linear", seed=0):
        self.T = T
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.split_frac = split_frac
        self.w_kernel = w_kernel
        self.seed = seed
        self.stage_R = {}   # t -> dict(beta, W1, H, K_W11, N2, theta_hat, n_eff_x)
        self.stage_D = {}

    def _kw(self, X, Y, bw=None):
        if self.w_kernel == "linear":
            return linear_kernel(X, Y)
        return rbf_kernel(X, Y, bw if bw is not None else _median_bandwidth(X))

    def _solve_stage(self, X1, X2, W1, y2, lam1, lam2, N1, N2):
        bw_x = _median_bandwidth(X1)
        K_X11 = rbf_kernel(X1, X1, bw_x)
        K_X12 = rbf_kernel(X1, X2, bw_x)
        Gamma = np.linalg.solve(K_X11 + N1 * lam1 * np.eye(N1), K_X12)   # (N1,N2)
        n_eff_x = effective_dimension(K_X11, N1, lam1)

        if self.w_kernel == "linear":
            # PRIMAL solve. A linear kernel's feature map is finite (2-D: the
            # raw w=(a,o) coordinates), so the dual (N1 x N1) Gram K_W11=W1@W1.T
            # has rank <= 2 -- a large near-zero-eigenvalue null space that
            # makes the dual ridge system numerically unstable (confirmed
            # empirically: eigenvalues at the 1e-13 floor for N1 in the
            # hundreds; see docs/continuous_extension.md). Solving directly in
            # the 2-D primal coordinates sidesteps this entirely and is also
            # far cheaper (2x2 solve instead of N1xN1) -- this is not an
            # approximation, it is the numerically well-posed way to solve the
            # SAME ridge problem when the kernel's feature space is finite.
            Wtilde = W1.T @ Gamma                          # (2, N2)
            H = (Wtilde @ Wtilde.T) / N2 + lam2 * np.eye(W1.shape[1])
            g = (Wtilde @ y2) / N2
            theta = np.linalg.solve(H, g)
            return dict(theta=theta, W1=W1, H=H, N1=int(N1), N2=int(N2),
                        lambda2=float(lam2), theta_hat=theta,
                        n_eff_x=float(n_eff_x))

        # DUAL solve (RBF path): K_W11 is strictly PD for distinct points (a
        # proper infinite-dimensional-feature kernel), so no null space and
        # the dual system is well-conditioned -- the dual representation is
        # also the ONLY option here, since RBF has no finite primal form.
        K_W11 = self._kw(W1, W1)
        Z = K_W11 @ Gamma                                                # (N1,N2)
        H = (Z @ Z.T) / N2 + lam2 * K_W11
        g = (Z @ y2) / N2
        beta = np.linalg.solve(H, g)
        return dict(beta=beta, W1=W1, H=H, K_W11=K_W11, N1=int(N1), N2=int(N2),
                    lambda2=float(lam2), theta_hat=None,
                    n_eff_x=float(n_eff_x))

    def fit(self, O0, O, A, R):
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

        for t in range(1, self.T + 1):
            X = _history_vec(O, A, O0, t)
            W = np.stack([A[:, t - 1], O[:, t - 1]], axis=1)
            X1, X2 = X[i1], X[i2]
            W1 = W[i1]

            self.stage_R[t] = self._solve_stage(
                X1, X2, W1, R[i2, t - 1], lam1, lam2, N1, N2)

            if t < self.T:
                self.stage_D[t] = self._solve_stage(
                    X1, X2, W1, O[i2, t], lam1, lam2, N1, N2)
        return self

    def predict(self, family, t, w_query):
        """b_R^[t](w) or b_D^[t](w) at query points w_query (n,2) -> (n,)."""
        s = (self.stage_R if family == "R" else self.stage_D)[t]
        w_query = np.atleast_2d(w_query)
        if self.w_kernel == "linear":
            return w_query @ s["theta"]
        K = self._kw(w_query, s["W1"])
        return K @ s["beta"]

    def eval_and_feature(self, family, t, coeffs, w_query):
        """Evaluate b^[t](w) using the GIVEN (possibly perturbed) coefficient
        vector rather than the fitted one, plus the raw linear feature/kernel
        row -- used by continuous_value_plugin/continuous_pessimism, where the
        pessimism coordinate descent evaluates the chain at trial points
        inside the confidence ellipsoid, not just at the fitted center."""
        w_query = np.atleast_2d(w_query)
        if self.w_kernel == "linear":
            feat = w_query                          # (n, 2): the primal feature IS w itself
        else:
            s = (self.stage_R if family == "R" else self.stage_D)[t]
            feat = self._kw(w_query, s["W1"])        # (n, N1)
        return (feat @ coeffs), feat

    def bridges(self):
        return self.stage_R, self.stage_D
