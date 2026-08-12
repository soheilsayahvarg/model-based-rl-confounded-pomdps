"""critic_v3_regime.py -- VERIFY V3: is 'shape-capped => correct at every N' a real regime?

The claim (rank_structure.md 7.4, README): "When the instrument shape forces
exact zeros -- (4,6,2), where |O_0| < |O| -- the rule is correct at EVERY sample
size." It rests on one config.

Two observations motivate the attack:

  a) (2,6,4) ALSO has |O_0| = 4 < |O| = 6 -- the shape forces 2 exact zeros --
     yet the author's own table shows it failing at N=32k. So the criterion
     "|O_0| < |O|" cannot be the regime boundary as written.

  b) What is special about (4,6,2) is not the shape cap but that
     rank(P_a) = min(|O|,|O_0|): there is NO statistically-empty tier, so the
     machine-zero cliff sits exactly at the truth. The predicted regime axis is
     "middle tier empty or not", i.e. rank(P_a) vs min(|O|,|O_0|).

Hostile configs (all with |O_0| < |O|, i.e. shape-capped by the doc's criterion)
where rank(P_a) < min(|O|,|O_0|), so the cliff sits ABOVE the truth:

    (2,6,2) cf=1.0 : truth 1 per action, cliff at 2
    (3,8,4) cf=1.0 : truth [2,1],        cliff at 4
    (2,8,4) cf=0.6 : truth 2,            cliff at 4

Controls where rank(P_a) = min(|O|,|O_0|) (prediction: correct at every N):

    (4,6,2) cf=1.0 (author's), (4,6,2) cf=0.6, (2,6,2) cf=0.6 (truth 2 = cliff),
    (4,7,3) cf=1.0 (truth [2,2]... only if rank(P_a)=3? no: per-action support
                    {0,2} and {1,3} -> truth [2,2] < 3, hostile as well)

Protocol: rule_new (rel floor 1e-9) on the per-action cross-moment, N in
{2000, 8000, 32000, 128000, 512000}, 20 seeds, modal + accuracy. If any hostile
config fails at small N, the regime claim as stated is false, and the correct
axis is the middle-tier one.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

from dim_separated_pomdp import default_params, sample_trajectories

CASES = [
    # (n_s, n_o, n_o0, confound, label)
    (4, 6, 2, 1.0, "author's control"),
    (4, 6, 2, 0.6, "control, stochastic policy"),
    (2, 6, 2, 0.6, "control: truth 2 = cliff 2"),
    (2, 6, 2, 1.0, "HOSTILE: truth 1 < cliff 2"),
    (3, 8, 4, 1.0, "HOSTILE: truth [2,1] < cliff 4"),
    (2, 8, 4, 0.6, "HOSTILE: truth 2 < cliff 4"),
    (4, 7, 3, 1.0, "HOSTILE: truth [2,2] < cliff 3"),
]
N_SWEEP = [2000, 8000, 32000, 128000, 512000]
SEEDS = list(range(20))
N_ACT = 2
T = 3


def per_action_spectrum(d, n_o, n_o0, a, t=1):
    o, act, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(act)
    sel = act == a
    Acr = np.zeros((n_o, n_o0))
    np.add.at(Acr, (o[sel], O0[sel]), 1.0)
    Acr /= N
    return np.linalg.eigvalsh(Acr @ Acr.T)[::-1]


def rule_new(desc, rel=1e-9):
    if desc.size < 2:
        return desc.size
    floor = max(desc[0], 0.0) * rel
    if floor <= 0.0:
        return 1
    return int(np.argmax(np.maximum(desc[:-1], floor)
                         / np.maximum(desc[1:], floor))) + 1


def main():
    results = []
    for (n_s, n_o, n_o0, cf, label) in CASES:
        cliff = min(n_o, n_o0)
        print(f"\n=== ({n_s},{n_o},{n_o0}) cf={cf} -- {label}")
        for a in range(N_ACT):
            p0 = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0, T=T,
                                seed=0, confound=cf)
            Pa = p0["E"].T @ np.diag(p0["p1"] * p0["pi_b"][:, a]) @ p0["K0"]
            truth = int(np.linalg.matrix_rank(Pa))
            middle = cliff - truth
            row = dict(n_s=n_s, n_o=n_o, n_o0=n_o0, cf=cf, action=a,
                       truth=truth, cliff=cliff, middle_tier=middle, by_N={})
            line = (f"  action {a}: truth={truth} cliff={cliff} "
                    f"middle tier={middle}  |")
            for N in N_SWEEP:
                ks = []
                for sd in SEEDS:
                    p = default_params(n_s=n_s, n_a=N_ACT, n_o=n_o, n_o0=n_o0,
                                       T=T, seed=sd, confound=cf)
                    d = sample_trajectories(p, N, np.random.default_rng(1000 + sd))
                    ks.append(rule_new(per_action_spectrum(d, n_o, n_o0, a)))
                modal = max(set(ks), key=ks.count)
                acc = np.mean([k == truth for k in ks])
                row["by_N"][N] = dict(modal=int(modal), accuracy=float(acc))
                line += f"  N={N//1000}k:{modal}({acc:.0%})"
            print(line)
            results.append(row)

    print("\nSummary: 'correct at every N' should hold for controls "
          "(middle tier = 0) and fail for HOSTILE (middle tier > 0).")
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic_v3_regime.json"))
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("written to", out)


if __name__ == "__main__":
    main()
