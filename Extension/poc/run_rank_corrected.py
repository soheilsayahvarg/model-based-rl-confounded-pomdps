"""run_rank_corrected.py -- the rank study, re-run with the review's corrections applied.

WHAT CHANGED FROM run_rank_verify.py, and why each change was forced:

1. POPULATION BENCHMARK. The old script compared against the unconditional
   cross-moment E^T diag(p1) K0. But every design matrix is built PER ACTION, and
   when the behavior policy depends on the latent state, conditioning on the
   action conditions on the latent state too. The correct object is

       P_a = E^T diag(p1 * pi_b[:, a]) K0

   Using the unconditional version is what produced the withdrawn claim that
   "population rank overstates usable rank": the direction we called buried was
   exactly zero in the per-action population.

2. CONFOUNDING SWEPT, NOT FIXED. default_params used confound=1.0, which makes
   pi_b exactly one-hot -- a deterministic function of the latent state, the most
   extreme possible case. Sweeping {0.6, 0.9, 1.0} shows which findings are
   geometric and which were properties of that corner.

3. NO AVERAGING ACROSS ACTIONS. In (3,7,5) the per-action ranks are [2,1]; the
   old per-action mean hid the asymmetry.

4. 20 SEEDS, NOT 3. The middle tier was reported as decaying at N^-0.8 and hedged
   as "consistent with N^-1". At 3 seeds that estimate was noise.

5. BOTH EIGENGAP RULES. The shipped rule divides by max(desc[1:], 1e-300).
   Trailing eigenvalues of a PSD matrix are machine noise around zero and often
   negative, so each yields a ratio ~1e282 and argmax selects float sign noise.
   Both rules are run on identical spectra so the difference is attributable to
   the rule alone.

Writes experiments/results_rank_corrected.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

from dim_separated_pomdp import default_params, sample_trajectories
from bridge_estimator import TabularBridgeEstimator

CONFIGS = [(2, 6, 4), (3, 7, 5), (4, 6, 2)]
CONFOUNDS = [1.0, 0.9, 0.6]
N_SWEEP = [4000, 32000, 256000]
SEEDS = list(range(20))
N_ACT = 2
T = 3
RESULTS = {}


def population_per_action(p, a):
    """P_a = E^T diag(p1 * pi_b[:,a]) K0 -- the object the per-action design
    actually converges to. Its rank is the number of latent states that can
    select action a, weighted by how often they do."""
    return p["E"].T @ np.diag(p["p1"] * p["pi_b"][:, a]) @ p["K0"]


def per_action_spectrum(d, n_o, n_o0, a, t=1):
    """Raw (unnormalized) eigenvalues of one action's bridge-space design."""
    o, act, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(act)
    sel = act == a
    Acr = np.zeros((n_o, n_o0))
    np.add.at(Acr, (o[sel], O0[sel]), 1.0)
    Acr /= N
    return np.linalg.eigvalsh(Acr @ Acr.T)[::-1]


def rule_old(desc):
    if desc.size < 2:
        return desc.size
    return int(np.argmax(desc[:-1] / np.maximum(desc[1:], 1e-300))) + 1


def rule_new(desc):
    if desc.size < 2:
        return desc.size
    floor = max(desc[0], 0.0) * 1e-9
    if floor <= 0.0:
        return 1
    return int(np.argmax(np.maximum(desc[:-1], floor)
                         / np.maximum(desc[1:], floor))) + 1


def slope(ns, vals):
    v = np.maximum(np.abs(np.asarray(vals, float)), 1e-300)
    return float(np.polyfit(np.log(np.asarray(ns, float)), np.log(v), 1)[0])


def main():
    # ---------------- [R1] tiers per action, per confound ----------------
    print("[R1] Per-action spectrum vs the per-action population rank.")
    print("     'signal' = slope > -0.5 and magnitude above 1e-12 * lam_max.\n")
    print(f"{'|S|':>4}{'|O|':>4}{'|O0|':>5}{'cf':>5}{'act':>4}"
          f"{'rank(P_a)':>10}{'signal':>8}{'stat.empty':>11}{'exact0':>8}"
          f"{'pred exact0':>12}   match")
    r1 = []
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            for a in range(N_ACT):
                spectra = {}
                for N in N_SWEEP:
                    acc = []
                    for sd in SEEDS:
                        p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o,
                                           n_o0=n_o0, T=T, seed=sd, confound=cf)
                        d = sample_trajectories(p, N, np.random.default_rng(1000 + sd))
                        acc.append(per_action_spectrum(d, n_o, n_o0, a))
                    spectra[N] = np.mean(acc, axis=0)

                p0 = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                    T=T, seed=0, confound=cf)
                pop_rank = int(np.linalg.matrix_rank(population_per_action(p0, a)))
                lam_max = spectra[N_SWEEP[-1]][0]

                n_sig = n_emp = n_zero = 0
                for i in range(n_o):
                    vals = [spectra[N][i] for N in N_SWEEP]
                    if abs(vals[-1]) < 1e-12 * lam_max:
                        n_zero += 1
                    elif slope(N_SWEEP, vals) > -0.5:
                        n_sig += 1
                    else:
                        n_emp += 1

                pred_zero = n_o - min(n_o, n_o0)
                ok = "OK" if (n_sig == pop_rank and n_zero >= pred_zero) else "DIFFERS"
                print(f"{n_s:>4}{n_o:>4}{n_o0:>5}{cf:>5.1f}{a:>4}"
                      f"{pop_rank:>10}{n_sig:>8}{n_emp:>11}{n_zero:>8}"
                      f"{pred_zero:>12}   {ok}")
                r1.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf, action=a,
                               pop_rank_per_action=pop_rank, n_signal=n_sig,
                               n_stat_empty=n_emp, n_exact_zero=n_zero,
                               predicted_exact_zero=pred_zero,
                               signal_matches_pop_rank=bool(n_sig == pop_rank)))
    RESULTS["R1_tiers"] = r1

    # ---------------- [R2] middle-tier decay rate, 20 seeds ----------------
    print("\n\n[R2] Decay rate of the statistically-empty tier (20 seeds).")
    print("     Prediction: exactly -1, the sampling rate of a squared mean.\n")
    print(f"{'|S|':>4}{'|O|':>4}{'|O0|':>5}{'cf':>5}{'act':>4}{'idx':>5}{'slope':>9}")
    r2, slopes = [], []
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            for a in range(N_ACT):
                spectra = {}
                for N in N_SWEEP:
                    acc = []
                    for sd in SEEDS:
                        p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o,
                                           n_o0=n_o0, T=T, seed=sd, confound=cf)
                        d = sample_trajectories(p, N, np.random.default_rng(1000 + sd))
                        acc.append(per_action_spectrum(d, n_o, n_o0, a))
                    spectra[N] = np.mean(acc, axis=0)
                lam_max = spectra[N_SWEEP[-1]][0]
                for i in range(n_o):
                    vals = [spectra[N][i] for N in N_SWEEP]
                    if abs(vals[-1]) < 1e-12 * lam_max:
                        continue
                    sl = slope(N_SWEEP, vals)
                    if sl <= -0.5:
                        print(f"{n_s:>4}{n_o:>4}{n_o0:>5}{cf:>5.1f}{a:>4}{i:>5}{sl:>9.2f}")
                        slopes.append(sl)
                        r2.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf,
                                       action=a, index=i, slope=sl))
    if slopes:
        print(f"\n     mean slope = {np.mean(slopes):.3f}  "
              f"sd = {np.std(slopes):.3f}  over {len(slopes)} directions")
        RESULTS["R2_mean_slope"] = float(np.mean(slopes))
        RESULTS["R2_sd_slope"] = float(np.std(slopes))
    RESULTS["R2_slopes"] = r2

    # ---------------- [R3] old vs fixed eigengap rule ----------------
    print("\n\n[R3] Eigengap rank rule, old vs fixed, on identical spectra.")
    print("     Truth = rank(P_a). Stability = how many of 20 seeds agree with"
          " the modal answer.\n")
    print(f"{'|S|':>4}{'|O|':>4}{'|O0|':>5}{'cf':>5}{'act':>4}{'truth':>7}"
          f"{'old':>6}{'old stab':>10}{'new':>6}{'new stab':>10}")
    r3 = []
    N_BIG = N_SWEEP[-1]
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            for a in range(N_ACT):
                olds, news = [], []
                for sd in SEEDS:
                    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                       T=T, seed=sd, confound=cf)
                    d = sample_trajectories(p, N_BIG, np.random.default_rng(1000 + sd))
                    desc = per_action_spectrum(d, n_o, n_o0, a)
                    olds.append(rule_old(desc))
                    news.append(rule_new(desc))
                p0 = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                    T=T, seed=0, confound=cf)
                truth = int(np.linalg.matrix_rank(population_per_action(p0, a)))
                mo, mn = max(set(olds), key=olds.count), max(set(news), key=news.count)
                so, sn = olds.count(mo) / len(olds), news.count(mn) / len(news)
                print(f"{n_s:>4}{n_o:>4}{n_o0:>5}{cf:>5.1f}{a:>4}{truth:>7}"
                      f"{mo:>6}{so:>10.0%}{mn:>6}{sn:>10.0%}")
                r3.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf, action=a,
                               truth=truth, old_modal=mo, old_stability=so,
                               new_modal=mn, new_stability=sn,
                               old_correct=bool(mo == truth),
                               new_correct=bool(mn == truth)))
    RESULTS["R3_rules"] = r3
    n_old = sum(1 for r in r3 if r["old_correct"])
    n_new = sum(1 for r in r3 if r["new_correct"])
    print(f"\n     old rule correct in {n_old}/{len(r3)} cells; "
          f"fixed rule correct in {n_new}/{len(r3)}")
    RESULTS["R3_old_correct"] = n_old
    RESULTS["R3_new_correct"] = n_new
    RESULTS["R3_cells"] = len(r3)

    # ---------------- [R4] the fixed rule inside the real estimator ----------
    print("\n\n[R4] Ranks selected inside TabularBridgeEstimator with the fix applied.\n")
    print(f"{'|S|':>4}{'|O|':>4}{'|O0|':>5}{'cf':>5}{'truth/action':>14}"
          f"{'selected':>10}{'sigma2':>12}   match")
    r4 = []
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                               seed=0, confound=cf)
            d = sample_trajectories(p, 32000, np.random.default_rng(0))
            truth = [int(np.linalg.matrix_rank(population_per_action(p, a)))
                     for a in range(N_ACT)]
            try:
                est = TabularBridgeEstimator(n_obs=n_o, n_act=N_ACT, n_o0=n_o0,
                                             n_r=2, T=T, mode="primal", seed=0)
                est.fit(d["O0"], d["O"], d["A"], d["R"])
                store = est.stage_store["bR_t1"]
                sel = [b.shape[1] for b in store["signal_basis"]]
                s2 = store["sigma2_signal"]
                ok = "OK" if sel == truth else "DIFFERS"
                print(f"{n_s:>4}{n_o:>4}{n_o0:>5}{cf:>5.1f}{str(truth):>14}"
                      f"{str(sel):>10}{s2:>12.3e}   {ok}")
                r4.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf,
                               truth=truth, selected=sel, sigma2_signal=float(s2),
                               matches=bool(sel == truth)))
            except Exception as exc:
                print(f"{n_s:>4}{n_o:>4}{n_o0:>5}{cf:>5.1f}{str(truth):>14}"
                      f"{'ERROR':>10}   {type(exc).__name__}: {exc}")
                r4.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, confound=cf,
                               error=f"{type(exc).__name__}: {exc}"))
    RESULTS["R4_estimator"] = r4

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_rank_corrected.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
