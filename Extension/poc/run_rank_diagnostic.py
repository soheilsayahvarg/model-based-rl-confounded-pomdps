"""run_rank_diagnostic.py -- is the null space a property of one method, or of the family?

THE CLAIM UNDER TEST. A proximal bridge is parameterized over the OBSERVATION
space but is only identified through the LATENT state, because every moment
restriction available factors through it:

    p(o_t, o_0 | a) = sum_s p(o_t | s) p(o_0 | s) p(s | a)

The right-hand side is a product through an |S|-dimensional bottleneck, so the
matrix has rank at most min(|S|, |O|, |O_0|) however large |O| gets. Every
observation direction beyond that bottleneck is unconstrained by the data -- a
structural null space, not an artifact of conditioning.

If that argument is right it says nothing about model-based versus model-free.
The Phase 4 paper diagnosed the blow-up in the anchor paper's model-based
pessimism. This script asks whether the model-FREE proximal estimator of
Shi et al., which has an entirely different loss, solver and bridge definition,
carries the same structural deficiency.

WHY THE PHASE 3 ENVIRONMENT COULD NOT ANSWER THIS. It fixes |S| = |O_0| = 2, so
"capped by the latent dimension" and "capped by the instrument dimension" make
identical predictions. The grid below separates them:

    |S|=2, |O_0|=4  -> latent-capped predicts 2, instrument-capped predicts 4
    |S|=4, |O_0|=2  -> latent-capped predicts 4, instrument-capped predicts 2

The two configurations point in opposite directions, so a single shared
explanation has to fit both or be abandoned.

WHY AN N-SWEEP RATHER THAN A RANK AT ONE N. At finite N every empirical matrix
is generically full rank; the structure shows up as eigenvalues that COLLAPSE
toward zero as N grows while the first r* stay O(1). A rank number at a single
sample size is a thresholding choice. The ratio

    lambda_{r*} / lambda_{r*+1}     (last signal over first null)

diverging with N is the actual signature, and it cannot be produced by a
threshold.

Writes experiments/results_rank_diagnostic.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import (default_params, sample_trajectories,
                                 population_cross_moment)
from bridge_estimator import TabularBridgeEstimator

# (n_s, n_o, n_o0, label). The first two are the discriminating pair.
GRID = [
    (2, 6, 4, "latent < instrument"),
    (4, 6, 2, "instrument < latent"),
    (3, 7, 5, "both above latent"),
    (2, 3, 2, "Phase 3 toy dims (degenerate: |S|=|O_0|)"),
]
N_SWEEP = [2000, 8000, 32000]
N_ACT = 2
T = 3
SEEDS = [0, 1, 2]
RESULTS = {}


def numerical_rank(evals_desc, rel_tol=1e-8):
    """Count eigenvalues above a relative tolerance. Reported alongside the raw
    spectrum so the reader can apply their own threshold."""
    if evals_desc[0] <= 0:
        return 0
    return int((evals_desc > evals_desc[0] * rel_tol).sum())


def model_free_design(d, n_o, n_o0, n_act, t=1):
    """Per-action design matrix of the model-free value bridge, in bridge space.

    Mirrors MinimaxValueBridgeOPE.fit exactly: within action a it accumulates the
    (|O| x |O_0|) cross-count between the bridge argument W=(A_t,O_t) and the
    instrument X=(A_t,O_0), then forms the bridge-space Gram A W A^T whose null
    space is what the confidence region would inherit.
    """
    o_t, a_t, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(a_t)
    eps = 1.0 / N
    out = []
    for a in range(n_act):
        sel = a_t == a
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o_t[sel], O0[sel]), 1.0)
        Ga = np.bincount(O0[sel], minlength=n_o0).astype(float)
        Wa = 1.0 / (Ga + N * eps)                 # 'gram' weighting, as in the baseline
        design = (Acr * Wa[None, :]) @ Acr.T      # (n_o, n_o) in BRIDGE space
        ev = np.linalg.eigvalsh(design)[::-1]
        out.append(ev / max(ev[0], 1e-300))       # normalized: scale-free comparison
    return out


def model_based_design(d, n_o, n_o0, n_act, seed):
    """Per-action design of the anchor paper's model-based stage-2 solve.

    The estimator already forms Wa = Ma Ma^T / N2 per action, the same object in
    the same bridge space. We read its spectrum and the eigengap rank the
    estimator itself selects, so this is the quantity that actually drives the
    pessimism widths downstream, not a re-derivation.
    """
    est = TabularBridgeEstimator(n_obs=n_o, n_act=n_act, n_o0=n_o0, n_r=2, T=T,
                                 mode="primal", seed=seed)
    est.fit(d["O0"], d["O"], d["A"], d["R"])
    store = est.stage_store["bR_t1"]
    ranks = [b.shape[1] for b in store["signal_basis"]]
    return ranks, store


def main():
    print("Population check: rank of p(o, o_0) = E^T diag(p1) K0\n")
    print(f"{'|S|':>4}{'|O|':>5}{'|O_0|':>7}{'pop rank':>10}{'predicted':>11}   note")
    pop_rows = []
    for (n_s, n_o, n_o0, note) in GRID:
        p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T)
        M = population_cross_moment(p)
        r = int(np.linalg.matrix_rank(M))
        pred = min(n_s, n_o, n_o0)
        flag = "OK" if r == pred else "MISMATCH"
        print(f"{n_s:>4}{n_o:>5}{n_o0:>7}{r:>10}{pred:>11}   {note} [{flag}]")
        pop_rows.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, pop_rank=r,
                             predicted=pred, note=note, ok=(r == pred)))
    RESULTS["population"] = pop_rows

    print("\n\nEmpirical model-FREE design, bridge space (normalized eigenvalues)")
    print("r* = min(|S|,|O|,|O_0|) is the predicted number of signal directions.\n")
    hdr = (f"{'|S|':>4}{'|O|':>5}{'|O_0|':>7}{'r*':>4}{'N':>8}"
           f"{'lam_r*':>12}{'lam_r*+1':>12}{'ratio':>12}{'num rank':>10}")
    print(hdr)
    mf_rows = []
    for (n_s, n_o, n_o0, note) in GRID:
        r_star = min(n_s, n_o, n_o0)
        if r_star >= n_o:
            continue                               # no null space to speak of
        for N in N_SWEEP:
            ratios, lam_a, lam_b, nranks = [], [], [], []
            for sd in SEEDS:
                p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                                   seed=sd)
                d = sample_trajectories(p, N, np.random.default_rng(sd))
                for ev in model_free_design(d, n_o, n_o0, N_ACT):
                    last_sig = ev[r_star - 1]
                    first_null = ev[r_star] if r_star < len(ev) else 0.0
                    lam_a.append(last_sig)
                    lam_b.append(first_null)
                    ratios.append(last_sig / max(first_null, 1e-300))
                    nranks.append(numerical_rank(ev))
            row = dict(n_s=n_s, n_o=n_o, n_o0=n_o0, r_star=r_star, N=N,
                       lam_last_signal=float(np.mean(lam_a)),
                       lam_first_null=float(np.mean(lam_b)),
                       ratio=float(np.median(ratios)),
                       num_rank=float(np.mean(nranks)), note=note)
            mf_rows.append(row)
            print(f"{n_s:>4}{n_o:>5}{n_o0:>7}{r_star:>4}{N:>8}"
                  f"{row['lam_last_signal']:>12.3e}{row['lam_first_null']:>12.3e}"
                  f"{row['ratio']:>12.3e}{row['num_rank']:>10.1f}")
    RESULTS["model_free"] = mf_rows

    print("\n\nModel-BASED stage-2 design: eigengap rank the estimator itself selects\n")
    print(f"{'|S|':>4}{'|O|':>5}{'|O_0|':>7}{'r*':>4}{'N':>8}{'ranks/action':>16}   match")
    mb_rows = []
    for (n_s, n_o, n_o0, note) in GRID:
        r_star = min(n_s, n_o, n_o0)
        for N in N_SWEEP[:2]:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=0)
            d = sample_trajectories(p, N, np.random.default_rng(0))
            try:
                ranks, _ = model_based_design(d, n_o, n_o0, N_ACT, seed=0)
            except Exception as exc:
                print(f"{n_s:>4}{n_o:>5}{n_o0:>7}{r_star:>4}{N:>8}"
                      f"{'ERROR':>16}   {type(exc).__name__}: {exc}")
                mb_rows.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, r_star=r_star,
                                    N=N, error=f"{type(exc).__name__}: {exc}"))
                continue
            ok = all(r == r_star for r in ranks)
            print(f"{n_s:>4}{n_o:>5}{n_o0:>7}{r_star:>4}{N:>8}{str(ranks):>16}"
                  f"   {'OK' if ok else 'DIFFERS'}")
            mb_rows.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, r_star=r_star, N=N,
                                ranks=[int(r) for r in ranks], matches=bool(ok)))
    RESULTS["model_based"] = mb_rows

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_rank_diagnostic.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
