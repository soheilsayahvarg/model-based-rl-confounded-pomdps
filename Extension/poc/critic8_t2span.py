"""critic8_t2span.py -- round 8, Priority 3: does the two-switch result survive
at t > 1?

Everything measured in completeness_and_h4.md is t = 1, where the stage-1
conditioning set is X_1 = (A_1, O_0): per action, only |O_0| profiles, so
|O_0| < |S| forces an incomplete span. At t = 2 the conditioning set is
X_2 = (A_2, H_1, O_0) with H_1 = (O_1, A_1): per action a2 there are
|O|*|A|*|O_0| cells, and the o1-conditioned posteriors are generically rich
enough to span all of R^|S| even when |O_0| < |S|. If so, beta_pop = 0 for
every t >= 2 block, and "incompleteness makes the truth leak" is a claim about
the STAGE-1 blocks only.

Population t=2 profiles, exact:
  v[s2 | a2, o1, a1, o0] ∝ sum_s1 p1[s1] K0[s1,o0] E[s1,o1] pi_b(a1|s1)
                                 P[a1][s1,s2] * pi_b(a2|s2)
  profile = E^T (v / |v|_1),  span over the (o1, a1, o0) cells per a2.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params
from run_completeness_beta import true_bridges_generic


def t2_span_and_beta(p, a2, tol=1e-10):
    E, P, p1, K0, pi_b = p["E"], p["P"], p["p1"], p["K0"], p["pi_b"]
    n_s, n_o = E.shape
    n_o0 = K0.shape[1]
    n_a = P.shape[0]
    cols = []
    for o1 in range(n_o):
        for a1 in range(n_a):
            for o0 in range(n_o0):
                w1 = p1 * K0[:, o0] * E[:, o1] * pi_b[:, a1]     # over s1
                v = (w1 @ P[a1]) * pi_b[:, a2]                   # over s2
                if v.sum() <= 1e-300:
                    continue
                cols.append(E.T @ (v / v.sum()))
    M = np.stack(cols, 1)
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    r = int((sv > tol * sv[0]).sum())
    U = U[:, :r]
    bR, _ = true_bridges_generic(p)
    Pn = np.eye(n_o) - U @ U.T
    leak = np.einsum("ij,jro->iro", Pn, bR[a2])
    return r, float(np.linalg.norm(leak)), float(np.linalg.norm(bR[a2])), sv


def main():
    print("t=2 population span vs t=1, per action  (tol 1e-10 on singular values)")
    print(f"{'config':>12}{'cf':>6}{'act':>5}{'t1 span':>9}{'t2 span':>9}"
          f"{'|S|':>5}{'t2 beta_pop':>13}{'rel':>8}")
    from run_completeness_beta import population_design_span, beta_pop_for
    for (n_s, n_o, n_o0) in [(4, 6, 2), (4, 8, 3), (2, 6, 4)]:
        for cf in [0.0, 0.6, 0.9, 1.0]:
            p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=3,
                               seed=0, confound=cf)
            for a in range(2):
                r1 = population_design_span(p, a).shape[1]
                r2, leak, nb, sv = t2_span_and_beta(p, a)
                print(f"{str((n_s,n_o,n_o0)):>12}{cf:>6.1f}{a:>5}{r1:>9}{r2:>9}"
                      f"{n_s:>5}{leak:>13.3e}{leak/nb:>8.1%}")
    print("\nIf t2 span = |S| wherever cf < 1, the truth-leak switch is a")
    print("stage-1 phenomenon and the two-switch claim must be scoped to the")
    print("blocks whose conditioning set is (A, O_0) alone.")


if __name__ == "__main__":
    main()
