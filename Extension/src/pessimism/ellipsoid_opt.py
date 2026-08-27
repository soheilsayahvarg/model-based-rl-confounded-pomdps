"""
ellipsoid_opt.py — Phase 4: confidence-region ellipsoid geometry and the
closed-form blockwise pessimistic minimization.

GEOMETRY. Per stage/family block, the stage-2 empirical risk (INCLUDING the ridge
term, anchor Eq. 121) is exactly quadratic in the coefficient vector, so

    L_hat(b) - L_hat(b_hat) = (b - b_hat)^T H (b - b_hat),   H = T_hat_2 + lam2 I.

(NB: H is HALF the Hessian of L_hat -- the true Hessian is 2H -- but the quadratic
form above is exact because L_hat is exactly quadratic, so the factor-1/2 Taylor
term cancels the factor-2 Hessian. It is the quadratic form, not "H = Hessian",
that the geometry relies on.) conf(xi) = { b : (b - b_hat)^T H (b - b_hat) <= xi }
is an EXACT ellipsoid
centered at the estimator b_hat. H and b_hat come from
TabularBridgeEstimator.stage_store (computed once, policy-independent — the
anchor's Remark 3.7 advantage over P3O's per-policy regions).

BLOCK MINIMIZATION. The plug-in value is MULTILINEAR in the 2T-1 bridge blocks:
linear in each block with the others frozen. For a linear objective <g, b> over an
ellipsoid the minimizer is closed-form:

    b_min = b_hat - sqrt(xi) * H^{-1} g / ||g||_{H^{-1}},
    min    = <g, b_hat> - sqrt(xi) * ||g||_{H^{-1}},  ||g||_{H^{-1}} = sqrt(g^T H^{-1} g).

Structure exploited: V = sum_t <gR_t, bR_t> where every gR_t depends only on the
DYNAMIC blocks — so all reward blocks minimize simultaneously and EXACTLY from one
gradient evaluation; dynamic-block gradients depend on the other blocks, so they
refresh after each move. Every step is an exact descent step, so V never increases.

GUARANTEE SCOPE (audited): the returned V_low is the min over coordinate-descent
endpoints; each endpoint is FEASIBLE, so V_low >= the exact joint min of the
multilinear objective. It is therefore an UPPER bound on the exact inner min, NOT
a certified lower bound on V(pi): coordinate descent on the (T-1) coupled dynamic
blocks can stall above the global min (canonical case: min of x*y over [-1,1]^2
started at the origin returns 0, not -1). Under region coverage the EXACT inner min
is <= V(pi) (the truth is feasible), but our approximate V_low may exceed it. We do
NOT claim a certified bound: V_low <= V_true is checked EMPIRICALLY (V4 holds for
all policies/seeds/widths here) and restart_gap flags non-global stalls. The reward
blocks — the majority of the coordinates — are solved globally; only the dynamic
coupling is heuristic.

WIDTH CALIBRATION (restricted-spectrum rule, Phase-2 finding): the global eig_min
of H is pinned at ~lam2 by the STRUCTURAL |O|>|S| null space and is blind to
identification collapse; the informative statistic is the smallest SIGNAL
eigenvalue sigma2 (per-action design blocks, min over actions). Widths scale as

    xi = c / (N2 * sigma2)

— the H-weighted distance of the truth from the center concentrates in weak-signal
directions with magnitude ~ (stage-1 noise)^2 / sigma2, so 1/(N2*sigma2) is the
right shape; the multiplier c is swept with coverage curves (never a claimed
"theoretical xi").
"""

import numpy as np
from scipy.linalg import solve_triangular


def _inner(H, b_hat, g, xi, nu):
    """Exact min of <g,b> + nu*||b||^2 over the ellipsoid, by a root find in mu."""
    from scipy.optimize import brentq
    n = H.shape[0]
    if nu > 0:
        b_u = -g / (2.0 * nu)
        d0 = b_u - b_hat
        if float(d0 @ H @ d0) <= xi:
            return b_u                          # ellipsoid inactive
    v = g + 2.0 * nu * b_hat

    def delta(mu):
        return np.linalg.solve(2.0 * mu * H + 2.0 * nu * np.eye(n), -v)

    def h(mu):
        d = delta(mu)
        return float(d @ H @ d) - xi

    lo, hi = 1e-14, 1.0
    for _ in range(400):
        if h(hi) < 0:
            break
        hi *= 2.0
    mu = brentq(h, lo, hi, maxiter=500, xtol=1e-15, rtol=1e-14)
    return b_hat + delta(mu)


def _linear_min_box(H, b_hat, g, xi, M_inf):
    """min <g,b> s.t. (b-b_hat)'H(b-b_hat) <= xi and |b_i| <= M_inf for all i.

    Assumption 4.1(f) of the anchor paper bounds the bridge class in SUP-norm, so
    the literal class is a BOX, not an L2 ball. A box is not rotationally
    symmetric, so there is no single scalar multiplier and the ball solver's root
    find does not apply. Solved directly (linear objective, one quadratic
    constraint, simple bounds), from two starts to reduce the chance of a local
    stall.
    """
    from scipy.optimize import minimize
    n = H.shape[0]
    cons = [{"type": "ineq",
             "fun": lambda b: xi - float((b - b_hat) @ H @ (b - b_hat)),
             "jac": lambda b: -2.0 * H @ (b - b_hat)}]
    bounds = [(-M_inf, M_inf)] * n
    best, best_val = None, np.inf
    for start in (np.clip(b_hat, -M_inf, M_inf), np.zeros(n)):
        r = minimize(lambda b: float(g @ b), start, jac=lambda b: g,
                     constraints=cons, bounds=bounds, method="SLSQP",
                     options=dict(maxiter=400, ftol=1e-12))
        if r.success and r.fun < best_val:
            best, best_val = r.x, r.fun
    return best if best is not None else np.clip(b_hat, -M_inf, M_inf)


def _exact_linear_min_ball(H, b_hat, g, xi, M):
    """Exact min of <g,b> over conf(xi) INTERSECT {||b|| <= M}.

    Replaces an earlier bisection that scaled a single fixed direction out to the
    ELLIPSOID boundary. The KKT family is a two-parameter curve, and in the
    large-xi regime the ellipsoid is not active at all, so that shortcut returned
    points up to 56% suboptimal and occasionally outside the ball. Here the ball
    multiplier nu is found by a root find, with the ellipsoid handled exactly
    inside _inner for each nu.
    """
    from scipy.optimize import brentq
    b0 = _inner(H, b_hat, g, xi, 0.0)
    if np.linalg.norm(b0) <= M + 1e-12:
        return b0, "ball inactive"
    b_min_norm = _inner(H, b_hat, np.zeros_like(g), xi, 1.0)
    if np.linalg.norm(b_min_norm) > M + 1e-9:
        return None, "EMPTY intersection"

    def norm_at(nu):
        return np.linalg.norm(_inner(H, b_hat, g, xi, nu)) - M

    lo, hi = 0.0, 1.0
    for _ in range(400):
        if norm_at(hi) < 0:
            break
        hi *= 2.0
    nu = brentq(norm_at, lo, hi, maxiter=500, xtol=1e-15, rtol=1e-14)
    return _inner(H, b_hat, g, xi, nu), "both active"


class BlockEllipsoid:
    """One bridge block's confidence-region geometry (flat coefficient coords).

    Optionally carries the empirical SIGNAL-SUBSPACE basis (per-action eigengap
    eigenvectors of the design, tensored with the identity on the index space):
    the |O|>|S| structural null directions are unidentified, and under the JOINT
    multilinear minimization they couple across blocks and blow the vanilla Eq.-17
    minimum out of the feasible value range. Restricting the region to the
    identified subspace (Nair-Jiang style rank restriction) contains the blow-up.

    SOUNDNESS CAVEAT (audit finding A-high): the projected region is a SUBSET that
    structurally excludes any out-of-subspace component of the true bridge (leak =
    ||(I-UU^T)(b_true - b_hat)|| > 0). Empirically the true bridges DO leak (up to
    ~0.38 on the toy), so the projected V_low is NOT a certified lower bound even
    at its exact inner min: if an excluded direction is value-relevant with the
    wrong sign, projected V_low can exceed V_true. The premise that excluded
    directions are value-irrelevant is NOT verified by the geometry -- it is only
    supported empirically (V4 held on the tested grid, by sign-alignment, not by
    construction). coverage_report now gates "covered" on small leakage so this is
    surfaced rather than masked. Both variants are exposed; treat projected V_low
    as a heuristic estimate, not a guarantee.
    """

    def __init__(self, H, b_hat_vec, sigma2_signal, N2, label,
                 signal_basis=None, n_obs=None, n_act=None, n_y=None):
        self.H = H
        self.b_hat = np.asarray(b_hat_vec, dtype=np.float64)
        self.sigma2 = float(sigma2_signal)
        self.N2 = int(N2)
        self.label = label
        self.L = np.linalg.cholesky(H)          # H = L L^T (PD since lam2 > 0)
        self.U = None
        if signal_basis is not None:
            self.U = self._build_U(signal_basis, n_act, n_obs, n_y)
            H_sub = self.U.T @ H @ self.U
            self.L_sub = np.linalg.cholesky(H_sub)
            self.H_sub = H_sub

    @staticmethod
    def _build_U(signal_basis, n_act, n_obs, n_y):
        """Orthonormal basis (D, k) of the identified subspace: per action a, the
        kept design eigenvectors on the o~-profile, tensored with I on the index y
        (vec convention: flat = w * n_y + y, w = a * n_obs + o~).

        Correctness of the projected coverage geometry (xi_needed's ds = U^T d and
        the leak residual) requires U^T U = I, which in turn requires each per-action
        Va to have ORTHONORMAL columns. The estimator supplies eigenvectors from
        np.linalg.eigh (orthonormal), but we assert it rather than assume it: a
        non-orthonormal Va would silently corrupt reported subspace coverage."""
        n_w = n_act * n_obs
        cols = []
        for a, Va in enumerate(signal_basis):
            k_a = Va.shape[1]
            if k_a:
                assert np.allclose(Va.T @ Va, np.eye(k_a), atol=1e-8), (
                    f"signal_basis[{a}] columns are not orthonormal; projected "
                    f"coverage geometry would be wrong. Re-orthonormalize (QR).")
            for i in range(k_a):
                w_vec = np.zeros(n_w)
                w_vec[a * n_obs:(a + 1) * n_obs] = Va[:, i]
                for y in range(n_y):
                    col = np.zeros(n_w * n_y)
                    col[np.arange(n_w) * n_y + y] = w_vec
                    cols.append(col)
        return np.stack(cols, axis=1)

    def h_inv_norm(self, g):
        """||g||_{H^{-1}} = sqrt(g^T H^{-1} g)."""
        y = solve_triangular(self.L, g, lower=True)
        return float(np.sqrt(y @ y))

    def linear_min(self, g, xi, projected=False, M=None, M_inf=None):
        """argmin/min of <g, b> over conf(xi) (optionally restricted to the
        identified signal subspace). Returns (b_min, penalty).

        M adds the bridge-class norm ball ||b||_2 <= M, which the anchor paper
        carries as M_R through Theorem 4.2 and which this implementation
        originally omitted. Without it the minimiser is free to run off along
        unidentified directions; the paper's feasible set is the INTERSECTION of
        the confidence region and the bridge class.

        The intersection of an ellipsoid and a ball with a linear objective has
        no one-line closed form, so it is solved by bisection on the ball's KKT
        multiplier: for each nu, minimising <g,b> + nu*||b||^2 over the ellipsoid
        alone IS closed-form, and ||b(nu)|| decreases monotonically in nu, so a
        scalar bisection lands on the active-ball solution.
        """
        if M_inf is not None and not projected:
            b = _linear_min_box(self.H, self.b_hat, g, xi, M_inf)
            return b, float(g @ self.b_hat - g @ b)

        if M is not None and not projected:
            b, status = _exact_linear_min_ball(self.H, self.b_hat, g, xi, M)
            if b is None:                      # ball and ellipsoid do not meet
                raise ValueError(
                    f"{self.label}: conf(xi) and the class ball are disjoint "
                    f"(||b_hat||={np.linalg.norm(self.b_hat):.3f} > M={M:.3f}). "
                    f"M must be per-block and at least ||b_true||.")
            return b, float(g @ self.b_hat - g @ b)

        return self._linear_min_plain(g, xi, projected)

    def _linear_min_plain(self, g, xi, projected=False):
        if projected:
            gs = self.U.T @ g
            y = solve_triangular(self.L_sub, gs, lower=True)
            ng = float(np.sqrt(y @ y))
            if ng < 1e-14:
                return self.b_hat.copy(), 0.0
            ds = solve_triangular(self.L_sub.T, y, lower=False)
            return self.b_hat - (np.sqrt(xi) / ng) * (self.U @ ds), \
                float(np.sqrt(xi) * ng)
        y = solve_triangular(self.L, g, lower=True)
        ng = float(np.sqrt(y @ y))
        if ng < 1e-14:
            return self.b_hat.copy(), 0.0
        Hinv_g = solve_triangular(self.L.T, y, lower=False)
        return self.b_hat - (np.sqrt(xi) / ng) * Hinv_g, float(np.sqrt(xi) * ng)

    def random_boundary_point(self, xi, rng, projected=False):
        if projected:
            u = rng.standard_normal(self.U.shape[1])
            d = solve_triangular(self.L_sub.T, u, lower=False)
            return self.b_hat + (self.U @ d) * (np.sqrt(xi) / np.sqrt(u @ u))
        u = rng.standard_normal(self.b_hat.size)
        d = solve_triangular(self.L.T, u, lower=False)
        return self.b_hat + d * (np.sqrt(xi) / np.sqrt(u @ u))

    def xi_needed(self, b_vec, projected=False):
        """Minimal width for b_vec (or its subspace projection) to lie inside the
        region. For the projected variant, also returns the out-of-subspace
        leakage norm of (b_vec - b_hat)."""
        d = b_vec - self.b_hat
        if projected:
            ds = self.U.T @ d
            leak = float(np.linalg.norm(d - self.U @ ds))
            return float(ds @ self.H_sub @ ds), leak
        return float(d @ self.H @ d)

    def width_rule(self, c):
        """Restricted-spectrum width: xi = c / (N2 * sigma2_signal)."""
        return c / (self.N2 * max(self.sigma2, 1e-12))


def _block_M(M, i):
    """M may be None, a scalar shared by all blocks, or one value per block.

    Per-block is the correct form: a single averaged M put some blocks' own
    centres outside their own ball, which is not the constraint the paper poses.
    """
    if M is None:
        return None
    if np.isscalar(M):
        return float(M)
    return float(M[i])


def _block_flag(flag, per_block, i):
    """Resolve a possibly per-block boolean. per_block, when given, wins."""
    if per_block is None:
        return bool(flag)
    return bool(per_block[i])


def pessimistic_value(blocks_R, blocks_D, xis_R, xis_D, value_and_grads,
                      max_sweeps=30, tol=1e-11, n_restarts=3, rng=None,
                      projected=False, M_R=None, M_D=None,
                      M_R_inf=None, M_D_inf=None,
                      projected_R=None, projected_D=None):
    """min over conf_R x conf_D of the multilinear plug-in value, by blockwise
    closed-form coordinate descent with multi-restart.

    value_and_grads(bR_flat_list, bD_flat_list) -> (V, gR_flat_list, gD_flat_list)
    projected=True restricts every region to its identified signal subspace.
    projected_R / projected_D override it per block, which is what a
    STAGE-SELECTIVE variant needs: cor:selfheal says only the t=1 blocks leak, so
    restricting every block is heavier than the diagnosis calls for.

    A block with xi = 0 contributes its centre and zero penalty, which is exactly
    the plug-in for that block -- so the per-block on/off switch needs no code
    here, only a zero in xis_R / xis_D.

    Returns dict(V_low, V_from_center, restart_gap, n_sweeps_center).
    """
    rng = rng or np.random.default_rng(0)
    T = len(blocks_R)
    pR = [_block_flag(projected, projected_R, t) for t in range(T)]
    pD = [_block_flag(projected, projected_D, j) for j in range(T - 1)]

    def run(start):
        bR = [blocks_R[t].b_hat.copy() for t in range(T)]
        bD = [blocks_D[j].b_hat.copy() for j in range(T - 1)]
        if start == "random":
            bR = [blocks_R[t].random_boundary_point(xis_R[t], rng, pR[t])
                  for t in range(T)]
            bD = [blocks_D[j].random_boundary_point(xis_D[j], rng, pD[j])
                  for j in range(T - 1)]
        V_prev = np.inf
        sweeps = 0
        for sweep in range(max_sweeps):
            sweeps = sweep + 1
            # reward blocks: V is jointly linear in them, gradients depend only on bD
            _, gR, _ = value_and_grads(bR, bD)
            for t in range(T):
                bR[t], _ = blocks_R[t].linear_min(
                    gR[t], xis_R[t], pR[t], M=_block_M(M_R, t),
                    M_inf=_block_M(M_R_inf, t))
            # dynamic blocks: gradients depend on the other blocks -> refresh each
            for j in range(T - 1):
                _, _, gD = value_and_grads(bR, bD)
                bD[j], _ = blocks_D[j].linear_min(
                    gD[j], xis_D[j], pD[j], M=_block_M(M_D, j),
                    M_inf=_block_M(M_D_inf, j))
            V_now = value_and_grads(bR, bD)[0]
            if V_prev - V_now < tol:
                V_prev = V_now
                break
            V_prev = V_now
        return float(V_prev), sweeps

    V0, s0 = run("center")
    vals = [V0]
    for _ in range(n_restarts):
        vals.append(run("random")[0])
    return dict(V_low=float(min(vals)), V_from_center=V0,
                restart_gap=float(max(vals) - min(vals)),
                n_sweeps_center=int(s0))
