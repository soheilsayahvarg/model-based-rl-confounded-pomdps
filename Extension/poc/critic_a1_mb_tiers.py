"""critic_a1_mb_tiers.py -- ATTACK A1: is C4 comparing two different matrices?

The doc's "signal directions actually present" column comes from the model-free
cross-moment design (Acr @ Acr.T, run_rank_verify.py). The "eigengap rank" column
comes from Wa = Ma @ Ma.T / N2 inside TabularBridgeEstimator._stage2_solve, where
Ma is the stage-1 CME table evaluated at stage-2 points. Different objects.

This script:
  1. Captures the estimator's OWN Wa spectra (monkeypatched, so the object is
     byte-identical to what drives the eigengap rule) across an N sweep and runs
     the author's exact three-tier slope classification on them.
  2. Instruments the eigengap rule: prints the descending eigenvalues, all
     consecutive ratios, where argmax lands, and whether the winning ratio's
     denominator was clamped by max(., 1e-300) because the eigenvalue is <= 0.
  3. Applies the same (instrumented) eigengap rule to the model-free design, so
     the two matrices can be compared like-for-like.

Speed note: fitting with T=1 on data truncated to the first timestep reproduces
the full fit's bR_t1 solve exactly (same seed -> same permutation -> the bR_t1
solve is the first rng consumer), while skipping the expensive later stages.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params, sample_trajectories
import bridge_estimator as be

N_SWEEP = [4000, 16000, 64000, 256000]
SEEDS = [0, 1, 2]
N_ACT = 2
T_DATA = 3
CONFIGS = [(2, 6, 4), (3, 7, 5), (4, 6, 2), (2, 3, 2)]
RESULTS = {}

# ------------------------------------------------ capture Wa inside the estimator
CAPTURE = {}
_orig = be.TabularBridgeEstimator._stage2_solve

def _patched(self, mu, pmf, seen_x, x2, n_w, n_y, N2, lam2, rng, label):
    M = mu[:, x2]
    specs = []
    for a in range(self.n_act):
        Ma = M[a * self.n_obs:(a + 1) * self.n_obs, :]
        Wa = Ma @ Ma.T / N2
        specs.append(np.linalg.eigvalsh(Wa)[::-1])
    CAPTURE[label] = specs
    return _orig(self, mu, pmf, seen_x, x2, n_w, n_y, N2, lam2, rng, label)

be.TabularBridgeEstimator._stage2_solve = _patched


def fit_bR_t1(d, n_o, n_o0, seed):
    """Truncated-fit trick: T=1 on 1-step data == full fit's bR_t1 solve."""
    est = be.TabularBridgeEstimator(n_obs=n_o, n_act=N_ACT, n_o0=n_o0, n_r=2,
                                    T=1, mode="primal", seed=seed)
    CAPTURE.clear()
    est.fit(d["O0"], d["O"][:, :1], d["A"][:, :1], d["R"][:, :1])
    specs = [s.copy() for s in CAPTURE["bR_t1"]]
    ranks = [b.shape[1] for b in est.stage_store["bR_t1"]["signal_basis"]]
    return specs, ranks


def eigengap_instrumented(desc):
    """Replicates the estimator's rule and reports WHY argmax landed there."""
    ratios = desc[:-1] / np.maximum(desc[1:], 1e-300)
    k = int(np.argmax(ratios)) + 1
    clamped = bool(desc[k] <= 0) if k < len(desc) else False
    return k, ratios, clamped


def mf_design_per_action(d, n_o, n_o0, t=1):
    o_t, a_t, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(a_t)
    specs = []
    for a in range(N_ACT):
        sel = a_t == a
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o_t[sel], O0[sel]), 1.0)
        Acr /= N
        specs.append(np.linalg.eigvalsh(Acr @ Acr.T)[::-1])
    return specs


def loglog_slope(ns, vals):
    v = np.maximum(np.asarray(vals, float), 1e-300)
    return float(np.polyfit(np.log(np.asarray(ns, float)), np.log(v), 1)[0])


def main():
    for (n_s, n_o, n_o0) in CONFIGS:
        key = f"S{n_s}_O{n_o}_O0{n_o0}"
        print(f"\n=== |S|={n_s} |O|={n_o} |O_0|={n_o0} " + "=" * 40)

        # ---- 1. Wa spectra across N (author's seed/data conventions from
        #         run_rank_verify: params seed=sd, data rng 100+sd)
        wa_spec, mf_spec, gap_detail = {}, {}, {}
        for N in N_SWEEP:
            wa_seed, mf_seed, ranks_seed = [], [], []
            for sd in SEEDS:
                p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                   T=T_DATA, seed=sd)
                d = sample_trajectories(p, N, np.random.default_rng(100 + sd))
                specs, ranks = fit_bR_t1(d, n_o, n_o0, seed=sd)
                wa_seed.append(np.mean(specs, axis=0))
                ranks_seed.append(ranks)
                mf_seed.append(np.mean(mf_design_per_action(d, n_o, n_o0), axis=0))
                if sd == 0:
                    gap_detail[N] = dict(wa=specs, mf=mf_design_per_action(d, n_o, n_o0))
            wa_spec[N] = np.mean(wa_seed, axis=0)
            mf_spec[N] = np.mean(mf_seed, axis=0)
            print(f"  N={N:>7}  eigengap ranks per (seed x action): {ranks_seed}")

        # ---- author's exact classifier, applied to Wa and to the cross-moment
        EXACT_FLOOR = 1e-12
        out = {}
        for name, spec in (("Wa_model_based", wa_spec), ("Acr_model_free", mf_spec)):
            lam_max = spec[N_SWEEP[-1]][0]
            tiers = {"SIGNAL": 0, "statistically empty": 0, "exact zero (shape)": 0}
            rows = []
            print(f"\n  {name}: three-tier classification (author's constants)")
            print(f"    {'i':>3}" + "".join(f"{('N=%d' % N):>13}" for N in N_SWEEP)
                  + f"{'slope':>9}   verdict")
            for i in range(n_o):
                vals = [spec[N][i] for N in N_SWEEP]
                sl = loglog_slope(N_SWEEP, vals)
                if abs(vals[-1]) < EXACT_FLOOR * lam_max:
                    v = "exact zero (shape)"
                else:
                    v = "SIGNAL" if sl > -0.5 else "statistically empty"
                tiers[v] += 1
                rows.append(dict(index=i, values=[float(x) for x in vals],
                                 slope=sl, tier=v))
                print(f"    {i:>3}" + "".join(f"{x:>13.3e}" for x in vals)
                      + f"{sl:>9.2f}   {v}")
            print(f"    -> {tiers['SIGNAL']} signal / "
                  f"{tiers['statistically empty']} stat-empty / "
                  f"{tiers['exact zero (shape)']} exact zero")
            out[name] = dict(tiers=tiers, rows=rows)

        # ---- 2. why does the eigengap rule land where it lands? (seed 0)
        print("\n  Eigengap rule autopsy (seed 0, action 0):")
        for N in (N_SWEEP[0], N_SWEEP[-1]):
            for name in ("wa", "mf"):
                desc = gap_detail[N][name][0]
                k, ratios, clamped = eigengap_instrumented(desc)
                n_nonpos = int((desc <= 0).sum())
                print(f"    N={N:>7} {name:>3}: k={k}  clamped_denominator={clamped}"
                      f"  #nonpositive eigs={n_nonpos}")
                print("            eigs   = " + " ".join(f"{v:>10.2e}" for v in desc))
                print("            ratios = " + " ".join(f"{r:>10.2e}" for r in ratios))
                out.setdefault("gap_autopsy", []).append(dict(
                    N=N, matrix=name, k=k, clamped=clamped,
                    eigs=[float(v) for v in desc],
                    ratios=[float(r) for r in ratios]))
        RESULTS[key] = out

    out_path = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                             "critic_a1_mb_tiers.json"))
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nwritten to", out_path)


if __name__ == "__main__":
    main()
