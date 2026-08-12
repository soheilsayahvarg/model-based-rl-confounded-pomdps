"""critic_a4_scope.py -- ATTACK A4: actions, horizon, per-action averaging, and the REAL toy.

1. PER-ACTION AVERAGING. At (3,7,5) confound=1.0, A2 showed rank(P_a) = [2, 1].
   The author's raw_spectrum() averages spectra over actions before classifying,
   which can only report one tier count. Classify each action separately.

2. MORE ACTIONS. n_a=4: with n_s=4 each action pins one state (predict 1 signal
   per action); with n_s=2 actions 2,3 are NEVER taken under confound=1.0
   (pi_b column support is {s % 4} = {0,1}), so two design matrices are all-zero.

3. HORIZON. Tier counts at t=2 and t=3 (the doc only ever measures t=1).

4. THE REAL PHASE 3 TOY. toy_pomdp has a STOCHASTIC behavior policy
   (0.75/0.25), unlike dim_separated_pomdp's confound=1.0 one-hot policy. The
   doc's "(2,3,2) Phase 3 toy dims" grid row is therefore NOT the Phase 3 toy.
   Run the tier analysis on the actual toy_pomdp environment.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

from dim_separated_pomdp import default_params, sample_trajectories
import toy_pomdp

N_SWEEP = [4000, 16000, 64000, 256000]
SEEDS = [0, 1, 2]
T = 3


def spectra_per_action(d, n_o, n_o0, n_act, t=1):
    o_t, a_t, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(a_t)
    specs = []
    for a in range(n_act):
        sel = a_t == a
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o_t[sel], O0[sel]), 1.0)
        Acr /= N
        specs.append(np.linalg.eigvalsh(Acr @ Acr.T)[::-1])
    return specs                                   # NOT averaged


def slope(vals):
    v = np.maximum(np.asarray(vals, float), 1e-300)
    return float(np.polyfit(np.log(np.asarray(N_SWEEP, float)), np.log(v), 1)[0])


def classify_tiers(spec_by_n, n_o, label):
    lam_max = spec_by_n[N_SWEEP[-1]][0]
    tiers = [0, 0, 0]
    detail = []
    for i in range(n_o):
        vals = [spec_by_n[N][i] for N in N_SWEEP]
        if abs(vals[-1]) < 1e-12 * lam_max:
            tiers[2] += 1; detail.append("Z")
        elif slope(vals) > -0.5:
            tiers[0] += 1; detail.append("S")
        else:
            tiers[1] += 1; detail.append("E")
    print(f"    {label}: {tiers[0]} signal / {tiers[1]} empty / {tiers[2]} zero"
          f"   [{','.join(detail)}]")
    return tiers


def run_env(sample_fn, n_o, n_o0, n_act, t, label):
    """sample_fn(seed, N) -> data dict. Classifies per action."""
    per_action = {a: {} for a in range(n_act)}
    for N in N_SWEEP:
        acc = {a: [] for a in range(n_act)}
        for sd in SEEDS:
            d = sample_fn(sd, N)
            for a, s in enumerate(spectra_per_action(d, n_o, n_o0, n_act, t)):
                acc[a].append(s)
        for a in range(n_act):
            per_action[a][N] = np.mean(acc[a], axis=0)
    print(f"\n  {label} (t={t})")
    return [classify_tiers(per_action[a], n_o, f"action {a}")
            for a in range(n_act)]


def main():
    # 1. per-action classification at (3,7,5), confound=1.0
    def mk(n_s, n_o, n_o0, n_a, confound):
        def fn(sd, N):
            p = default_params(n_s=n_s, n_a=n_a, n_o=n_o, n_o0=n_o0, T=T,
                               seed=sd, confound=confound)
            return sample_trajectories(p, N, np.random.default_rng(100 + sd))
        return fn
    print("=== A4.1: per-action tiers, (3,7,5) confound=1.0 "
          "(author averages these)")
    run_env(mk(3, 7, 5, 2, 1.0), 7, 5, 2, 1, "|S|=3 |O|=7 |O_0|=5")

    print("\n=== A4.2: four actions")
    run_env(mk(4, 8, 6, 4, 1.0), 8, 6, 4, 1,
            "|S|=4 |O|=8 |O_0|=6, n_a=4 (each action pins one state)")
    p = default_params(n_s=2, n_a=4, n_o=6, n_o0=4, T=T, seed=0, confound=1.0)
    d = sample_trajectories(p, 4000, np.random.default_rng(0))
    counts = np.bincount(d["A"][:, 0], minlength=4)
    print(f"    n_s=2, n_a=4, confound=1.0: action counts at t=1 = {counts}"
          f"  -> actions 2,3 unvisited; their designs are identically zero")

    print("\n=== A4.3: later stages, (2,6,4) confound=1.0")
    for t in (2, 3):
        run_env(mk(2, 6, 4, 2, 1.0), 6, 4, 2, t, "|S|=2 |O|=6 |O_0|=4")

    print("\n=== A4.4: the ACTUAL Phase 3 toy (stochastic pi_b = 0.75/0.25)")
    tp = toy_pomdp.default_params(T=T)
    E, K0, p1, pi_b = tp["E"], tp["K0"], tp["p1"], tp["pi_b"]
    for a in range(2):
        Pa = E.T @ np.diag(p1 * pi_b[:, a]) @ K0
        print(f"    rank(P_a) action {a} = {np.linalg.matrix_rank(Pa)}   "
              f"eigs(Pa Pa^T) = "
              + " ".join(f"{v:.2e}" for v in np.linalg.eigvalsh(Pa @ Pa.T)[::-1]))
    def toy_fn(sd, N):
        return toy_pomdp.sample_trajectories(tp, N, np.random.default_rng(100 + sd))
    run_env(toy_fn, 3, 2, 2, 1, "toy_pomdp (real Phase 3/4 environment)")


if __name__ == "__main__":
    main()
