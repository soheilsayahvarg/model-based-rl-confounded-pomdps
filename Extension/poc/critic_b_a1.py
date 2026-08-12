"""critic_b_a1.py -- ATTACK A1: did the width calibration manufacture the blow-up?

(a) The shipped rule is xi = c / (N * lambda_min(H)); after ridging
    lambda_min(H) ~ rho = 0.03/sqrt(N), so xi is set by the RIDGE, not the data.
    Phase 3's model-based rule used the smallest SIGNAL eigenvalue (smallest
    retained). Re-run B4 with both conventions on identical fits, plus a
    convention-free reference. If the divergence signature (order-of-magnitude
    gap, cf=1.0 >> cf=0.6) does not survive the Phase 3 convention, C1 was
    manufactured by calibration.

(b) Is the level-set ellipsoid {(b - b_hat)^T H (b - b_hat) <= xi} even a
    sampling-uncertainty region for this ridged GMM estimator? Empirically: fit
    the estimator on 40 independent datasets and measure the spread of the
    stage-1 bridge along H's eigendirections, and the spread of J itself. If
    the estimator barely moves in the null directions (the ridge pins them),
    the region's width there is measuring identification uncertainty, not
    sampling error -- which changes what C1 is evidence OF.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import (default_params, sample_trajectories, dp_value,
                                 candidate_policies)
from model_free_proximal import MinimaxValueBridgeOPE
from mf_pessimism import (mf_design_and_gradient, h_inv_norm,
                          parallel_analysis_basis)

CONFIGS = [(4, 6, 2), (2, 6, 4), (3, 7, 5)]
CONFOUNDS = [1.0, 0.6]
N = 4000
SEEDS = [0, 1, 2, 3, 4]
N_ACT = 2
T = 3
C = 10.0


def main():
    print("[A1a] xi conventions on identical fits, c=10, policy=always_0.\n")
    print(f"{'config':>10}{'cf':>5}{'V_true':>8}{'V_hat':>8}"
          f"{'shipped':>10}{'sigma2':>10}{'xi=c/N':>10}")
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            pi = candidate_policies(n_o, N_ACT, seed=0)["always_0"]
            v_true = dp_value(p, pi)
            rows = []
            for sd in SEEDS:
                d = sample_trajectories(p, N, np.random.default_rng(500 + sd))
                nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
                nu1 /= nu1.sum()
                est = MinimaxValueBridgeOPE(n_obs=n_o, n_act=N_ACT, n_o0=n_o0,
                                            T=T, seed=sd)
                est.fit(d["O0"], d["O"], d["A"], d["R"], pi)
                v_hat = est.value()
                Hs, gs = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu1)
                pens = dict(shipped=0.0, sigma2=0.0, fixed=0.0)
                for a in range(N_ACT):
                    k, U = parallel_analysis_basis(
                        d, n_o, n_o0, a, np.random.default_rng(900 + sd))
                    ev = np.linalg.eigvalsh(Hs[a])[::-1]      # descending
                    lam_min = max(ev[-1], 1e-300)
                    lam_sig = max(ev[k - 1], 1e-300)          # smallest RETAINED
                    nrm = h_inv_norm(Hs[a], gs[a])
                    pens["shipped"] += np.sqrt(C / (N * lam_min)) * nrm
                    pens["sigma2"] += np.sqrt(C / (N * lam_sig)) * nrm
                    pens["fixed"] += np.sqrt(C / N) * nrm
                rows.append((v_hat, pens))
            mvh = np.mean([r[0] for r in rows])
            out = {kk: np.mean([r[0] - r[1][kk] for r in rows])
                   for kk in ("shipped", "sigma2", "fixed")}
            print(f"{str((n_s,n_o,n_o0)):>10}{cf:>5.1f}{v_true:>8.3f}{mvh:>8.3f}"
                  f"{out['shipped']:>10.2f}{out['sigma2']:>10.2f}"
                  f"{out['fixed']:>10.2f}")

    # ---------------- (b) what does the estimator's ACTUAL sampling spread look like?
    print("\n[A1b] Estimator sampling spread vs region width, (2,6,4) cf=1.0, 40 fits.")
    n_s, n_o, n_o0, cf = 2, 6, 4, 1.0
    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T, seed=0,
                       confound=cf)
    pi = candidate_policies(n_o, N_ACT, seed=0)["always_0"]
    v_true = dp_value(p, pi)
    Js, b1s = [], []
    H_ref = None
    for sd in range(40):
        d = sample_trajectories(p, N, np.random.default_rng(3000 + sd))
        nu1 = np.bincount(d["O"][:, 0], minlength=n_o).astype(float)
        nu1 /= nu1.sum()
        est = MinimaxValueBridgeOPE(n_obs=n_o, n_act=N_ACT, n_o0=n_o0, T=T,
                                    seed=sd)
        est.fit(d["O0"], d["O"], d["A"], d["R"], pi)
        Js.append(est.value())
        b1s.append(est.bV_hat[0].copy())          # (n_act, n_obs)
        if sd == 0:
            Hs, _ = mf_design_and_gradient(d, n_o, n_o0, N_ACT, nu1)
            H_ref = Hs
    Js = np.array(Js)
    b1s = np.array(b1s)                            # (40, n_act, n_obs)
    print(f"  J: mean {Js.mean():.3f}  std {Js.std():.3f}  "
          f"V_true {v_true:.3f}  |bias| {abs(Js.mean()-v_true):.3f}")
    print(f"  -> a sampling-based interval ~ mean +/- 2 std = "
          f"[{Js.mean()-2*Js.std():.2f}, {Js.mean()+2*Js.std():.2f}]  "
          f"vs shipped penalty at c=10 ~ 65 (V_low ~ -63)")
    for a in range(N_ACT):
        ev, evec = np.linalg.eigh(H_ref[a])
        ev, evec = ev[::-1], evec[:, ::-1]
        coords = b1s[:, a, :] @ evec               # (40, n_o)
        stds = coords.std(axis=0)
        print(f"  action {a}: eig(H) = " + " ".join(f"{v:.1e}" for v in ev))
        print(f"            std(b1 along eigvec) = "
              + " ".join(f"{v:.1e}" for v in stds))
        print(f"            region half-width sqrt(xi/lam) at c=10 = "
              + " ".join(f"{np.sqrt((C/(N*max(ev[-1],1e-300)))/v):.1e}"
                         for v in ev))


if __name__ == "__main__":
    main()
