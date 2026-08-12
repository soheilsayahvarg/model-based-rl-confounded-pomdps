"""critic_c_smooth.py -- step (c) critic, Priority 2a.

Is the Theta(1/lam) null term -- hence the exact exponent e = tau + kappa - 1 --
an artifact of the tabular delta kernel's EXACT finite-rank null space?

With a strictly positive-definite kernel the design spectrum decays smoothly
(say s_j ~ j^-b) and the "null space" is a spectral tail. Then

    Q(lam) = ||g||^2_{(T+lam I)^-1} = sum_j g_j^2 / (s_j + lam)

and the classical source-condition calculus gives Q(lam) ~ lam^-theta with
theta = 1 - (a-1)/b  when g_j^2 ~ j^-a and 1 < a < b+1  (theta in (0,1)),
Q = Theta(1/lam) ONLY when g has an atom on the exact null (a delta at s=0),
and Q = Theta(1) when g is smooth enough (a > b+1: the "source condition").

Numerical check of that calculus, then the consequence for the dichotomy: with
theta < 1 the width exponent becomes e' = tau + theta*kappa - 1 and the sharp
threshold moves off e = 0; the trade-off persists but its CONSTANT is kernel-
and source-dependent -- i.e. the general form lives in Tikhonov rate theory,
and the proposition's exact exponent is the delta-kernel specialisation.
"""

import numpy as np


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main():
    J = 200000
    j = np.arange(1, J + 1, dtype=float)
    lams = np.logspace(-2, -7, 11)
    print(f"{'b':>5}{'a':>6}{'pred theta':>12}{'meas theta':>12}")
    for b in (1.5, 2.0, 3.0):
        s = j ** -b
        for a in (1.2, 1.5, 2.0, b + 1.5):
            g2 = j ** -a
            Q = [float(np.sum(g2 / (s + lam))) for lam in lams]
            th = -slope(lams, Q)
            pred = max(0.0, min(1.0, 1.0 - (a - 1.0) / b))
            print(f"{b:>5.1f}{a:>6.1f}{pred:>12.3f}{th:>12.3f}")
    # the delta-kernel case: atom at zero (exact null) -> theta = 1 exactly
    s_atom = np.concatenate([j[:50] ** -2.0, np.zeros(10)])
    g2_atom = np.concatenate([j[:50] ** -1.5, np.full(10, 0.1)])
    Q = [float(np.sum(g2_atom / (s_atom + lam))) for lam in lams]
    print(f"{'atom':>5}{'':>6}{1.0:>12.3f}{-slope(lams, Q):>12.3f}"
          f"   (exact null: the proposition's case)")


if __name__ == "__main__":
    main()
