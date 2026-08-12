"""critic_p_a6_box.py -- ATTACK A6: the paper's M_R is a SUP-norm bound, not an L2 ball.

Assumption 4.1(f) of the anchor paper (verified from the PDF): "There exist
M_R, M_D > 0 such that max_t sup_{b in B_{R,t}} ||b||_INFTY <= M_R" -- uniform
boundedness of the bridge FUNCTIONS, i.e. in the tabular case a BOX constraint
|b_i| <= M_R, not a Euclidean ball. norm_constraint.md 7 claims M_R "bounds an
RKHS norm ... the identity in the tabular delta-kernel case", which is wrong as
a description of 4.1(f). An L2 ball is still a legitimate CHOICE of class B
(then M_R = its radius bounds the sup-norm too), but the tightest realizable
class of each shape differs, and the constrained pessimism depends on the
shape. Here: the same coordinate descent with the box class
|b_i| <= M_inf := max_i |b_true,i| per block (the tightest realizable box),
solved per block by SLSQP (linear objective, ellipsoid constraint, box bounds),
against the L2-ball results from critic_p_a345.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid, pessimistic_value
from critic_p_a345 import toy_true_bridges, dp_value_toy, build_blocks

N_MAIN = 5000
SEEDS = [0, 1, 2]
T = 3
PI = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])


def box_linear_min(H, b_hat, g, xi, Minf):
    """min <g,b> s.t. (b-b_hat)'H(b-b_hat) <= xi, |b_i| <= Minf."""
    n = b_hat.size
    cons = [dict(type="ineq",
                 fun=lambda b: xi - float((b - b_hat) @ H @ (b - b_hat)),
                 jac=lambda b: -2.0 * H @ (b - b_hat))]
    bounds = [(-Minf, Minf)] * n
    best, fbest = None, np.inf
    for start in (np.clip(b_hat, -Minf, Minf), np.zeros(n)):
        r = minimize(lambda b: float(g @ b), start, jac=lambda b: g,
                     bounds=bounds, constraints=cons, method="SLSQP",
                     options=dict(maxiter=300, ftol=1e-12))
        viol = max(0.0, float((r.x - b_hat) @ H @ (r.x - b_hat)) - xi * (1 + 1e-8))
        if viol < 1e-7 and r.fun < fbest:
            best, fbest = r.x, r.fun
    if best is None:                                   # SLSQP failed; stay put
        best = np.clip(b_hat, -Minf, Minf)
    return best


_orig = BlockEllipsoid.linear_min


def _patched(self, g, xi, projected=False, M=None):
    Minf = getattr(self, "Minf_override", None)
    if Minf is not None and not projected:
        b = box_linear_min(self.H, self.b_hat, g, xi, Minf)
        return b, float(g @ self.b_hat - g @ b)
    return _orig(self, g, xi, projected=projected, M=None)


BlockEllipsoid.linear_min = _patched


def main():
    params = toy.default_params(T=3)
    v_true = dp_value_toy(params, PI)
    bR_true, bD_true, _ = toy_true_bridges(params)
    minf = {}
    for t in range(1, T + 1):
        minf[f"bR_t{t}"] = float(np.abs(bR_true).max())
        if t < T:
            minf[f"bD_t{t}"] = float(np.abs(bD_true).max())
    print(f"V_true = {v_true:.4f}")
    print(f"tightest realizable box: max|bR_true| = {np.abs(bR_true).max():.3f}"
          f"   max|bD_true| = {np.abs(bD_true).max():.3f}")
    print(f"(for scale: ||bR_true||_2 = {np.linalg.norm(bR_true.ravel()):.3f})\n")

    print(f"{'c':>6}{'box M=true':>12}{'box M=2xtrue':>14}"
          f"{'(L2 M=true from A4)':>21}")
    for c in (1.0, 10.0):
        vals = {1.0: [], 2.0: []}
        for sd in SEEDS:
            d = toy.sample_trajectories(params, N_MAIN, np.random.default_rng(sd))
            est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                         n_o0=toy.N_O0, n_r=2, T=T,
                                         mode="primal", seed=sd)
            est.fit(d["O0"], d["O"], d["A"], d["R"])
            blocks = build_blocks(est)
            p_o1 = empirical_p_o1(d["O"], toy.N_O)
            bR = [b for n, b in blocks if n.startswith("bR")]
            bD = [b for n, b in blocks if n.startswith("bD")]

            def vg(bRl, bDl):
                R = np.stack([x.reshape(toy.N_A, toy.N_O, 2, toy.N_O)
                              for x in bRl])
                D = np.stack([x.reshape(toy.N_A, toy.N_O, toy.N_O, toy.N_O)
                              for x in bDl])
                V, _p, (gR, gD) = plugin_value(R, D, PI, p_o1,
                                               return_grads=True)
                return (V, [gR[t].ravel() for t in range(T)],
                        [gD[j].ravel() for j in range(T - 1)])

            xR = [b.width_rule(c) for b in bR]
            xD = [b.width_rule(c) for b in bD]
            for mult in (1.0, 2.0):
                for n, b in blocks:
                    b.Minf_override = mult * minf[n]
                r = pessimistic_value(bR, bD, xR, xD, vg,
                                      rng=np.random.default_rng(sd))
                vals[mult].append(r["V_low"])
                for n, b in blocks:
                    b.Minf_override = None
        ref = {1.0: -7.53, 10.0: -9.06}[c]
        print(f"{c:>6.1f}{np.mean(vals[1.0]):>12.2f}{np.mean(vals[2.0]):>14.2f}"
              f"{ref:>21.2f}")


if __name__ == "__main__":
    main()
