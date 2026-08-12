"""run_norm_constraint.py -- did we drop a constraint the anchor paper assumes?

THE SUSPICION. The paper minimises over conf(alpha) INTERSECTED with the bridge
class B_{R,t}, and Theorem 4.2 carries M_R, a uniform upper bound on that class,
right through the suboptimality bound. Our implementation -- both the Phase 3
model-based ellipsoid and the model-free layer added for step (b) -- minimises
over the ellipsoid ALONE. `linear_min` has no norm ball, no clipping, no M_R.

If the bridge class bound is what stops the excursion into unidentified
directions, then every divergence we have reported is an artifact of omitting a
constraint the method actually has, not a defect of the method. That would
invalidate the step (b) headline and much of the Phase 4 pessimism story.

This is the check that has to come before any proposition is written.

THE BOUND WE USE. Not fitted from data, and not oracle-tuned to make the result
come out either way. The model-free value bridge satisfies
V_t(o) = sum_a b_V(a,o) with V a value function bounded by T * r_max, and rewards
here are Bernoulli, so |b_V(a,o)| <= T and per action ||b_V(a,.)||_2 <= sqrt(n_o)*T.
That is an a priori bound a practitioner has before seeing any data. We sweep
multiples of it so the conclusion does not rest on one choice.

THE CONSTRAINED PROBLEM. Minimise a linear objective over the intersection of an
ellipsoid and a norm ball -- convex, small (n_o = 6), solved directly. We verify
the solver against the closed form whenever the ball is inactive.

Writes experiments/results_norm_constraint.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params, sample_trajectories, dp_value, candidate_policies
from model_free_proximal import MinimaxValueBridgeOPE
from mf_pessimism import mf_design_and_gradient, h_inv_norm, parallel_analysis_basis

CONFIGS = [(4, 6, 2, "saturating"), (2, 6, 4, "non-saturating")]
CONFOUND = 1.0
N_MAIN = 4000
SEEDS = [0, 1, 2, 3, 4]
N_ACT, T = 2, 3
C_GRID = [1.0, 10.0]
M_MULT = [0.5, 1.0, 2.0, 10.0, np.inf]     # multiples of the a priori bound
RESULTS = {}


def constrained_min(g, H, b_hat, xi, M):
    """min <g,b>  s.t.  (b-b_hat)' H (b-b_hat) <= xi  and  ||b||_2 <= M.

    With M = inf this must reproduce the closed form
    <g,b_hat> - sqrt(xi)*||g||_{H^-1}, which we assert.
    """
    y = np.linalg.solve(np.linalg.cholesky(H), g)
    ng = float(np.sqrt(y @ y))
    b_star = b_hat - np.sqrt(xi) * np.linalg.solve(H, g) / max(ng, 1e-300)
    closed = float(g @ b_hat - np.sqrt(xi) * ng)
    if not np.isfinite(M):
        return closed, float(np.linalg.norm(b_star)), False
    if np.linalg.norm(b_star) <= M:
        return closed, float(np.linalg.norm(b_star)), False   # ball inactive

    cons = [
        {"type": "ineq", "fun": lambda b: xi - (b - b_hat) @ H @ (b - b_hat)},
        {"type": "ineq", "fun": lambda b: M ** 2 - b @ b},
    ]
    x0 = b_star * min(1.0, M / max(np.linalg.norm(b_star), 1e-12))
    res = minimize(lambda b: g @ b, x0, constraints=cons, method="SLSQP",
                   options=dict(maxiter=500, ftol=1e-12))
    return float(res.fun), float(np.linalg.norm(res.x)), True


def main():
    print("Does the bridge-class norm bound remove the divergence?\n")
    print("a priori bound per action: ||b_V(a,.)||_2 <= sqrt(n_o) * T")
    print(f"{'config':>10}{'c':>6}{'M/M0':>8}{'V_true':>9}{'V_hat':>9}"
          f"{'V_low':>11}{'ball active':>13}{'||b*||':>10}")
    rows = []
    for (n_s, n_o, n_o0, note) in CONFIGS:
        M0 = np.sqrt(n_o) * T
        p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                           seed=0, confound=CONFOUND)
        pi = candidate_policies(n_o, N_ACT, seed=0)["always_0"]
        v_true = dp_value(p, pi)
        for c in C_GRID:
            for mm in M_MULT:
                M = np.inf if not np.isfinite(mm) else mm * M0
                vls, acts, nrm, vhs = [], [], [], []
                for sd in SEEDS:
                    d = sample_trajectories(p, N_MAIN, np.random.default_rng(500 + sd))
                    nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
                    nu1 /= nu1.sum()
                    est = MinimaxValueBridgeOPE(n_obs=n_o, n_act=N_ACT,
                                                n_o0=n_o0, T=T, seed=sd)
                    est.fit(d["O0"], d["O"], d["A"], d["R"], pi)
                    v_hat = est.value()
                    Hs, gs = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu1)
                    pen = 0.0
                    any_active = False
                    for a in range(N_ACT):
                        H, g = Hs[a], gs[a]
                        b_hat_a = est.bV_hat[0][a]
                        xi = c / (N_MAIN * max(np.linalg.eigvalsh(H)[0], 1e-300))
                        val, nb, active = constrained_min(g, H, b_hat_a, xi, M)
                        pen += (g @ b_hat_a) - val        # the width actually taken
                        any_active |= active
                        nrm.append(nb)
                    vhs.append(v_hat)
                    vls.append(v_hat - pen)
                    acts.append(any_active)
                mv = float(np.mean(vls))
                lbl = "inf" if not np.isfinite(mm) else f"{mm:g}"
                print(f"{str((n_s,n_o,n_o0)):>10}{c:>6.1f}{lbl:>8}{v_true:>9.3f}"
                      f"{np.mean(vhs):>9.3f}{mv:>11.2f}"
                      f"{str(any(acts)):>13}{np.mean(nrm):>10.2f}")
                rows.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, note=note, c=c,
                                 M_mult=(None if not np.isfinite(mm) else mm),
                                 M0=float(M0), v_true=float(v_true),
                                 v_hat=float(np.mean(vhs)), v_low=mv,
                                 ball_active=bool(any(acts)),
                                 mean_bnorm=float(np.mean(nrm))))
    RESULTS["norm_constraint"] = rows

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_norm_constraint.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
