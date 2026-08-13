"""run_gradient_leakage_map.py -- the OTHER half of (H4), mapped in the population.

completeness_and_h4.md mapped beta = ||P_Nul b_true||, the truth's leakage, and
found it is switched by completeness. The schedule-free floor of step (c),

    W >= beta * beta_g,

needs BOTH factors, and beta_g = ||P_Nul g|| had never been mapped at all. It was
assumed to travel with beta. It does not.

Run after regret_vs_schedule.md 4.3 measured beta_g = 0 exactly on the toy and the
decision-level test came back null. The precondition for that test to be
meaningful is beta_g > 0 somewhere, so this locates where.

WHY beta_g CAN VANISH WHILE beta DOES NOT. The stage-1 value gradient's
observation profile is the marginal nu1 = E^T p1. The design is spanned by
E^T v_{a,x} with v_{a,x}[s] ∝ p1[s] * pi_b[s,a] * K0[s,x]. When pi_b is constant
in s, summing over x gives exactly E^T p1 up to scale, so nu1 lies IN the span and
cannot leak, however incomplete the instrument is. Confounding is what breaks the
sum, so the gradient leaks only when the behaviour policy depends on the state.

Writes experiments/results_gradient_leakage_map.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params
from run_completeness_beta import population_design_span, beta_pop_for

CONFIGS = [(2, 6, 4), (4, 6, 2), (3, 7, 5), (4, 8, 3)]
CONFOUNDS = [0.0, 0.3, 0.6, 0.9, 0.99, 1.0]
RESULTS = {}


def beta_g_pop(p):
    """Relative population leakage of the stage-1 value gradient, max over actions.

    The gradient's o~-profile is the observation marginal nu1 = E^T p1; the design
    null is the complement of span{E^T v_{a,x}}. Both are population objects, so
    this is exact linear algebra with no sampling.
    """
    nu1 = p["p1"] @ p["E"]
    n = float(np.linalg.norm(nu1))
    out = []
    for a in range(p["pi_b"].shape[1]):
        U = population_design_span(p, a)
        out.append(float(np.linalg.norm(nu1 - U @ (U.T @ nu1))) / max(n, 1e-300))
    return max(out), [U.shape[1] for U in
                      (population_design_span(p, a)
                       for a in range(p["pi_b"].shape[1]))]


def main():
    print("=" * 84)
    print("Population leakage: the TRUTH (beta) and the GRADIENT (beta_g)")
    print("The step (c) floor is W >= beta * beta_g, so it needs BOTH.")
    print("=" * 84)
    rows = []
    for (n_s, n_o, n_o0) in CONFIGS:
        print(f"\n({n_s},{n_o},{n_o0})   "
              f"{'INCOMPLETE (|O_0| < |S|)' if n_o0 < n_s else 'complete'}")
        print(f"   {'confound':>10}{'span':>10}{'beta':>10}{'beta_g':>12}"
              f"{'floor b*bg':>12}")
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=3,
                               seed=0, confound=cf)
            bg, spans = beta_g_pop(p)
            b = max(beta_pop_for(p, a)[0] for a in range(2))
            floor = b * bg
            print(f"   {cf:>10.2f}{str(spans):>10}{b:>10.4f}{bg:>12.3e}"
                  f"{floor:>12.4f}")
            rows.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf,
                             spans=spans, beta=b, beta_g=bg, floor=floor,
                             incomplete=bool(n_o0 < n_s)))

    print("\n" + "=" * 84)
    print("READING")
    print("=" * 84)
    print("  incompleteness (|O_0| < |S|)  ->  the TRUTH leaks      (beta > 0)")
    print("  confounding    (pi_b on S)    ->  the GRADIENT leaks   (beta_g > 0)")
    print("  the floor W >= beta*beta_g needs BOTH, so it is nonzero exactly")
    print("  when an inadequate negative control meets real confounding.")

    nz = [r for r in rows if r["floor"] > 1e-6]
    if nz:
        best = max(nz, key=lambda r: r["floor"])
        print(f"\n  largest floor: {best['floor']:.4f} at "
              f"({best['n_s']},{best['n_o']},{best['n_o0']}) "
              f"confound={best['confound']}")
        graded = [r for r in nz if r["incomplete"] and 0 < r["confound"] < 1]
        print(f"  nonzero floors at NON-degenerate confounding: {len(graded)}"
              f"  -> the floor is not a knife-edge in incomplete designs")

    RESULTS["rows"] = rows
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_gradient_leakage_map.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
