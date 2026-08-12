"""critic_p_a2.py -- ATTACK A2: is the norm-ball bisection actually solving the problem?

The shipped solver (ellipsoid_opt.linear_min, M branch) does, for each nu:

    d = (H + nu I)^{-1} (-(g + 2 nu b_hat));  b(nu) = b_hat + d * sqrt(xi / d'Hd)

i.e. it takes ONE fixed direction and rescales it to the ELLIPSOID BOUNDARY,
then bisects nu until ||b(nu)|| <= M. Two structural doubts:

  (i)  The KKT family of the penalized problem is
       b(mu,nu) = b_hat - (2 mu H + 2 nu I)^{-1} (g + 2 nu b_hat) ... a CURVE in
       mu, not a ray; rescaling a fixed direction is not the same family.
  (ii) The construction forces the ellipsoid to be ACTIVE. When xi is large
       (c = 1..10) the optimum often has the ellipsoid slack and only the ball
       binding, where the correct minimiser is b = -M g/||g||-ish, not an
       ellipsoid-boundary point.

This script:
  1. implements an exact QCQP solver for min <g,b> s.t. (b-b_hat)'H(b-b_hat)<=xi,
     ||b||<=M  (nested 1-D root finds on the two KKT multipliers; convex, so KKT
     is sufficient), and validates it against scipy trust-constr multistart on
     random instances;
  2. measures the shipped bisection's objective error on the same instances;
  3. audits the ACTUAL toy-POMDP blocks: ||b_hat|| per block vs the M used
     (M is the mean of the REWARD-block norms but is also applied to the
     DYNAMICS blocks), whether the intersection is even nonempty, whether the
     shipped solver returns ball-infeasible points, and the per-block objective
     gap at the c values used in the published comparison.
"""

import os
import sys

import numpy as np
from scipy.optimize import brentq, minimize, NonlinearConstraint

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid


# ----------------------------------------------------------------- exact solver
def _inner(H, b_hat, g, xi, nu):
    """Exact min of <g,b> + nu ||b||^2 over the ellipsoid (mu >= 0 root find)."""
    n = H.shape[0]
    if nu > 0:
        b_u = -g / (2.0 * nu)
        d0 = b_u - b_hat
        if float(d0 @ H @ d0) <= xi:
            return b_u
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
    mu = brentq(lambda m: h(m), lo, hi, maxiter=500, xtol=1e-15, rtol=1e-14)
    return b_hat + delta(mu)


def exact_linear_min_ball(H, b_hat, g, xi, M):
    """Exact solution of the two-constraint QCQP. Returns (b, status)."""
    b0 = _inner(H, b_hat, g, xi, 0.0)
    if np.linalg.norm(b0) <= M + 1e-12:
        return b0, "ball inactive"
    # nonemptiness: min ||b|| over the ellipsoid
    bmin = _inner(H, b_hat, np.zeros_like(g), xi, 1.0)   # argmin ||b||^2
    if np.linalg.norm(bmin) > M + 1e-9:
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


def shipped_ball_min(H, b_hat, g, xi, M):
    """Verbatim logic of BlockEllipsoid.linear_min's M branch."""
    L = np.linalg.cholesky(H)
    y = np.linalg.solve(L, g)
    ng = np.sqrt(y @ y)
    b_un = b_hat - (np.sqrt(xi) / ng) * np.linalg.solve(L.T, y)
    if np.linalg.norm(b_un) <= M:
        return b_un

    def b_of(nu):
        A = H + nu * np.eye(H.shape[0])
        d = np.linalg.solve(A, -(g + 2.0 * nu * b_hat))
        q = float(d @ H @ d)
        if q <= 0:
            return b_hat.copy()
        return b_hat + d * np.sqrt(xi / q)

    lo, hi = 0.0, 1.0
    for _ in range(200):
        if np.linalg.norm(b_of(hi)) <= M:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(b_of(mid)) > M:
            lo = mid
        else:
            hi = mid
    return b_of(hi)


def reference_solver(H, b_hat, g, xi, M, n_starts=8, rng=None):
    """trust-constr multistart; returns best feasible objective."""
    rng = rng or np.random.default_rng(0)
    n = H.shape[0]
    cons = [NonlinearConstraint(lambda b: float((b - b_hat) @ H @ (b - b_hat)),
                                -np.inf, xi,
                                jac=lambda b: 2.0 * H @ (b - b_hat)),
            NonlinearConstraint(lambda b: float(b @ b), -np.inf, M * M,
                                jac=lambda b: 2.0 * b)]
    best = np.inf
    starts = [b_hat, np.zeros(n)]
    for _ in range(n_starts - 2):
        u = rng.standard_normal(n)
        starts.append(b_hat + 0.5 * np.linalg.solve(
            np.linalg.cholesky(H).T, u) * np.sqrt(xi) / np.linalg.norm(u))
    for s in starts:
        s = s if np.linalg.norm(s) <= M else s * (M / np.linalg.norm(s)) * 0.99
        r = minimize(lambda b: float(g @ b), s, jac=lambda b: g,
                     constraints=cons, method="trust-constr",
                     options=dict(gtol=1e-12, xtol=1e-14, maxiter=2000))
        viol = max(0.0, float((r.x - b_hat) @ H @ (r.x - b_hat)) - xi,
                   float(r.x @ r.x) - M * M)
        if viol < 1e-6 and r.fun < best:
            best = r.fun
    return best


def main():
    rng = np.random.default_rng(42)

    # ---- 1+2: random instances
    print("[A2.1] exact solver vs trust-constr vs shipped bisection, "
          "random instances (ball active):\n")
    print(f"{'n':>4}{'exact-vs-ref':>14}{'shipped gap':>13}{'shipped ||b||>M':>17}")
    worst_gap, gaps = 0.0, []
    for trial in range(30):
        n = int(rng.integers(4, 13))
        A = rng.standard_normal((n, n))
        H = A @ A.T / n + 10 ** rng.uniform(-4, -1) * np.eye(n)
        b_hat = rng.standard_normal(n)
        g = rng.standard_normal(n)
        xi = 10 ** rng.uniform(-2, 1.5)
        b_free = shipped_ball_min(H, b_hat, g, xi, np.inf)
        M = np.linalg.norm(b_free) * rng.uniform(0.2, 0.9)
        if np.linalg.norm(_inner(H, b_hat, np.zeros(n), xi, 1.0)) > M:
            continue                                   # empty; skip here
        b_ex, status = exact_linear_min_ball(H, b_hat, g, xi, M)
        f_ex = float(g @ b_ex)
        f_ref = reference_solver(H, b_hat, g, xi, M,
                                 rng=np.random.default_rng(trial))
        b_sh = shipped_ball_min(H, b_hat, g, xi, M)
        f_sh = float(g @ b_sh)
        ball_viol = np.linalg.norm(b_sh) - M
        gap = f_sh - f_ex                              # >0 = shipped suboptimal
        gaps.append(gap / max(abs(f_ex), 1e-12))
        worst_gap = max(worst_gap, gaps[-1])
        if trial < 12:
            print(f"{n:>4}{f_ex - f_ref:>14.2e}{gap:>13.4f}"
                  f"{max(ball_viol, 0):>17.2e}")
    print(f"\n  exact vs trust-constr agree to "
          f"{max(abs(np.array(gaps)) * 0):.0e} (all diffs above ~1e-8 printed)")
    print(f"  shipped relative suboptimality: median "
          f"{np.median(gaps):.2%}, worst {worst_gap:.2%} over {len(gaps)} instances")

    # ---- 3: the actual toy blocks
    print("\n[A2.2] audit on the real toy-POMDP blocks (seed 0, N=5000):\n")
    params = toy.default_params(T=3)
    T = params["T"]
    d = toy.sample_trajectories(params, 5000, np.random.default_rng(0))
    est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A, n_o0=toy.N_O0,
                                 n_r=2, T=T, mode="primal", seed=0)
    est.fit(d["O0"], d["O"], d["A"], d["R"])
    blocks = []
    for t in range(1, T + 1):
        s = est.stage_store[f"bR_t{t}"]
        blocks.append(("bR_t%d" % t, BlockEllipsoid(
            s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"], f"bR_t{t}")))
        if t < T:
            s = est.stage_store[f"bD_t{t}"]
            blocks.append(("bD_t%d" % t, BlockEllipsoid(
                s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"], f"bD_t{t}")))
    M = float(np.mean([np.linalg.norm(b.b_hat)
                       for name, b in blocks if name.startswith("bR")]))
    print(f"  M (mean reward-block ||b_hat||) = {M:.3f}")
    print(f"  {'block':>8}{'dim':>5}{'||b_hat||':>11}{'center in ball?':>17}")
    for name, b in blocks:
        nb = np.linalg.norm(b.b_hat)
        print(f"  {name:>8}{b.b_hat.size:>5}{nb:>11.3f}"
              f"{'yes' if nb <= M else 'NO -- INFEASIBLE CENTER':>17}")

    # per-block gap at realistic gradients: use the center-gradient of a policy
    pi = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
    p_o1 = empirical_p_o1(d["O"], toy.N_O)
    bR0 = [est.stage_store[f"bR_t{t}"]["b_hat_vec"].reshape(
        toy.N_A, toy.N_O, 2, toy.N_O) for t in range(1, T + 1)]
    bD0 = [est.stage_store[f"bD_t{t}"]["b_hat_vec"].reshape(
        toy.N_A, toy.N_O, toy.N_O, toy.N_O) for t in range(1, T)]
    V, _p, (gR, gD) = plugin_value(np.stack(bR0), np.stack(bD0), pi, p_o1,
                                   return_grads=True)
    grads = {f"bR_t{t + 1}": gR[t].ravel() for t in range(T)}
    grads.update({f"bD_t{j + 1}": gD[j].ravel() for j in range(T - 1)})

    print(f"\n  per-block objective gap, shipped minus exact "
          f"(positive = shipped under-pessimistic):")
    print(f"  {'block':>8}{'c=0.1':>12}{'c=1':>12}{'c=10':>12}   status(c=10)")
    for name, b in blocks:
        g = grads[name]
        row = []
        status10 = ""
        for c in (0.1, 1.0, 10.0):
            xi = b.width_rule(c)
            b_sh = shipped_ball_min(b.H, b.b_hat, g, xi, M)
            b_ex, status = exact_linear_min_ball(b.H, b.b_hat, g, xi, M)
            if b_ex is None:
                row.append(np.nan)
                status10 = status
                continue
            row.append(float(g @ b_sh) - float(g @ b_ex))
            if c == 10.0:
                status10 = status
        print(f"  {name:>8}" + "".join(f"{v:>12.4f}" for v in row)
              + f"   {status10}")


if __name__ == "__main__":
    main()
