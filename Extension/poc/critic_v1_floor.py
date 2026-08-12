"""critic_v1_floor.py -- VERIFY V1: is 1e-9 a tuned constant, and why do 2 cells miss?

1. FLOOR SWEEP. The fixed rule floors ratio terms at rel * lam_max with
   rel = 1e-9. Re-run the author's R3 protocol (18 cells, 20 seeds, N = 256,000,
   modal rank vs rank(P_a)) for rel across 13 orders of magnitude. If correctness
   only holds in a narrow band around 1e-9, the fix is a tuned constant.

2. MISS AUTOPSY. The author attributes the 2 misses (both confound=0.9, 55%
   stability) to "a genuinely weak population direction near the noise floor".
   But the author's seed convention varies the ENVIRONMENT with the seed, so
   each of the 20 draws has a different weak-direction magnitude. Per seed:
   which k was selected, and what is that environment's population ratio
   lambda_truth / lambda_1. If misses concentrate in environment draws whose
   population ratio is smallest, the attribution is right; if they are spread
   evenly, it is not.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

from dim_separated_pomdp import default_params, sample_trajectories

CONFIGS = [(2, 6, 4), (3, 7, 5), (4, 6, 2)]
CONFOUNDS = [1.0, 0.9, 0.6]
SEEDS = list(range(20))
N_BIG = 256000
N_ACT = 2
T = 3
FLOORS = [10.0 ** -e for e in range(3, 16)]      # 1e-3 .. 1e-15


def per_action_spectrum(d, n_o, n_o0, a, t=1):
    o, act, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(act)
    sel = act == a
    Acr = np.zeros((n_o, n_o0))
    np.add.at(Acr, (o[sel], O0[sel]), 1.0)
    Acr /= N
    return np.linalg.eigvalsh(Acr @ Acr.T)[::-1]


def rule_floored(desc, rel):
    if desc.size < 2:
        return desc.size
    floor = max(desc[0], 0.0) * rel
    if floor <= 0.0:
        return 1
    return int(np.argmax(np.maximum(desc[:-1], floor)
                         / np.maximum(desc[1:], floor))) + 1


def pop_matrix(p, a):
    return p["E"].T @ np.diag(p["p1"] * p["pi_b"][:, a]) @ p["K0"]


def main():
    # -------- collect spectra once (18 cells x 20 seeds), reuse for all floors
    cells = []
    for (n_s, n_o, n_o0) in CONFIGS:
        for cf in CONFOUNDS:
            for a in range(N_ACT):
                specs, pop_ratios, truths = [], [], []
                for sd in SEEDS:
                    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                       T=T, seed=sd, confound=cf)
                    d = sample_trajectories(p, N_BIG,
                                            np.random.default_rng(1000 + sd))
                    specs.append(per_action_spectrum(d, n_o, n_o0, a))
                    Pa = pop_matrix(p, a)
                    ev = np.linalg.eigvalsh(Pa @ Pa.T)[::-1]
                    tr = int(np.linalg.matrix_rank(Pa))
                    truths.append(tr)
                    pop_ratios.append(ev[tr - 1] / ev[0] if tr >= 1 else 0.0)
                cells.append(dict(n_s=n_s, n_o=n_o, n_o0=n_o0, cf=cf, a=a,
                                  specs=specs, truths=truths,
                                  pop_ratios=pop_ratios))

    # -------- 1. floor sweep
    print("Floor sweep: cells correct (modal over 20 seeds) out of 18, and mean"
          " per-seed accuracy\n")
    print(f"{'rel floor':>10}{'cells correct':>15}{'seed accuracy':>15}")
    sweep = []
    for rel in FLOORS:
        n_cell_ok, accs = 0, []
        for c in cells:
            ks = [rule_floored(s, rel) for s in c["specs"]]
            modal = max(set(ks), key=ks.count)
            truth_modal = max(set(c["truths"]), key=c["truths"].count)
            n_cell_ok += int(modal == truth_modal)
            accs.append(np.mean([k == t for k, t in zip(ks, c["truths"])]))
        print(f"{rel:>10.0e}{n_cell_ok:>15}{np.mean(accs):>15.2%}")
        sweep.append(dict(rel=rel, cells_correct=n_cell_ok,
                          seed_accuracy=float(np.mean(accs))))

    # -------- 2. autopsy of cells that miss at rel=1e-9
    print("\nMiss autopsy at rel=1e-9 (cells whose modal selection != truth):")
    for c in cells:
        ks = [rule_floored(s, 1e-9) for s in c["specs"]]
        modal = max(set(ks), key=ks.count)
        truth_modal = max(set(c["truths"]), key=c["truths"].count)
        if modal == truth_modal:
            continue
        print(f"\n  ({c['n_s']},{c['n_o']},{c['n_o0']}) cf={c['cf']} action={c['a']}"
              f": modal={modal} truth={truth_modal}")
        print(f"    {'seed':>5}{'k':>4}{'truth':>7}{'pop ratio':>12}   correct?")
        for sd, (k, tr, pr) in enumerate(zip(ks, c["truths"], c["pop_ratios"])):
            print(f"    {sd:>5}{k:>4}{tr:>7}{pr:>12.2e}   {'yes' if k == tr else 'NO'}")
        wrong = [c["pop_ratios"][i] for i in range(len(ks)) if ks[i] != c["truths"][i]]
        right = [c["pop_ratios"][i] for i in range(len(ks)) if ks[i] == c["truths"][i]]
        if wrong and right:
            print(f"    median pop ratio | wrong seeds: {np.median(wrong):.2e}"
                  f" | correct seeds: {np.median(right):.2e}")

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic_v1_floor.json"))
    with open(out, "w") as f:
        json.dump(dict(sweep=sweep), f, indent=2, default=float)
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
