"""critic8_align_p1.py -- round 8, Priority 2: a NON-degenerate alignment knob.

The 4b sweep interpolates emission rows toward their mean, so at alpha -> 1 the
environment carries no observational information at all and beta_g -> 0 is
forced by construction. Two questions:

 (1) Is the population "reproduction" a tautology? With confound = 1.0 the
     per-action span is rank 1 (direction m), and H = w*mm' + rho*I, so

         ratio^2 - 1  =  [beta_g^2 / (1 - beta_g^2)] * cond(H)      (exact)

     -- the ratio is an algebraic function of (beta_g, cond); the sweep could
     not have failed. Verify the identity on the 4b rows. Its real content is
     that the KNOB moves beta_g 78x and cond only 1.58x, so the two factor
     contributions separate exactly: 1.58x vs ~14,000x.

 (2) Does the conclusion survive a knob that keeps the POMDP fully informative?
     Sweep p1 instead: with confound = 1.0 the span direction is the emission
     row of the action's own state (p1-independent), while nu1 = E^T p1 moves.
     E, K0, rank(E), row separation all FIXED -- only the gradient's angle to
     the fixed null moves. If the ratio still tracks beta_g here, section 3
     survives on an informative environment.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params
from mf_pessimism import h_inv_norm
from run_stepb_reproduce import pop_span_and_design, aligned_params

N_S, N_O, N_O0, N_A = 2, 6, 4, 2
RHO = 1e-3


def measures(p):
    nu1 = p["p1"] @ p["E"]
    n = float(np.linalg.norm(nu1))
    best = None
    for a in range(N_A):
        U, D = pop_span_and_design(p, a)
        H = D + RHO * np.eye(N_O)
        bg = float(np.linalg.norm(nu1 - U @ (U.T @ nu1))) / n
        wu = h_inv_norm(H, nu1)
        wp = h_inv_norm(H, nu1, basis=U)
        cond = float(np.linalg.cond(H))
        if best is None or bg > best[0]:
            best = (bg, wu / max(wp, 1e-300), cond, U.shape[1])
    return best


def main():
    print("[1] identity check on the 4b alpha-sweep rows:")
    print(f"{'alpha':>7}{'beta_g':>11}{'cond':>12}{'ratio^2-1':>12}"
          f"{'bg2/(1-bg2)*cond':>18}{'rel err':>10}")
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        p = aligned_params(alpha)
        bg, ratio, cond, r = measures(p)
        lhs = ratio ** 2 - 1.0
        rhs = bg ** 2 / (1.0 - bg ** 2) * cond
        print(f"{alpha:>7.2f}{bg:>11.4e}{cond:>12.3e}{lhs:>12.4e}"
              f"{rhs:>18.4e}{abs(lhs-rhs)/max(lhs,1e-300):>10.2e}")

    print("\n[2] the p1 knob: E, K0, rank, row separation all FIXED;")
    print("    only the initial-state distribution (hence nu1's angle) moves.")
    p0 = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=3,
                        seed=0, confound=1.0)
    rowsep = float(np.linalg.norm(p0["E"][0] - p0["E"][1]))
    print(f"    emission row separation (constant): {rowsep:.4f}")
    print(f"{'p1[0]':>7}{'span rank':>10}{'cond':>12}{'beta_g':>11}"
          f"{'ratio':>9}{'identity rel err':>18}")
    res = []
    for q in [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]:
        p = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=3,
                           seed=0, confound=1.0)
        p["p1"] = np.array([q, 1.0 - q])
        bg, ratio, cond, r = measures(p)
        lhs, rhs = ratio ** 2 - 1, bg ** 2 / (1 - bg ** 2) * cond
        res.append((q, bg, ratio, cond))
        print(f"{q:>7.2f}{r:>10}{cond:>12.3e}{bg:>11.4e}{ratio:>9.3f}"
              f"{abs(lhs-rhs)/max(lhs,1e-300):>18.2e}")
    bgs = [x[1] for x in res]
    rats = [x[2] for x in res]
    conds = [x[3] for x in res]
    print(f"\n    beta_g range: {min(bgs):.4f} .. {max(bgs):.4f} "
          f"({max(bgs)/max(min(bgs),1e-300):.0f}x)")
    print(f"    ratio  range: {min(rats):.3f} .. {max(rats):.3f}")
    print(f"    cond   range: {min(conds):.3e} .. {max(conds):.3e} "
          f"({max(conds)/min(conds):.2f}x)")
    mono = all((bgs[i+1]-bgs[i])*(rats[i+1]-rats[i]) >= 0
               for i in range(len(res)-1))
    print(f"    ratio monotone in beta_g along the sweep: {mono}")


if __name__ == "__main__":
    main()
