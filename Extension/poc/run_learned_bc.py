"""Matched-fit, logged-data behaviour-cloning control with raw seed records.

Default: 20 seeds, four sample sizes, dense (4,6,2), T=3, confounding .9.
No legacy result is overwritten. Candidates are built from the stage-1 training
split only. Regret is against the best policy in each seed's augmented set.
All pessimism arms here are class-free heuristics, not certified lower bounds.
"""
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
for folder in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, str(HERE.parent / "src" / folder))
sys.path.insert(0, str(HERE))
from dim_separated_pomdp import default_params, dp_value
from value_plugin import empirical_p_o1
from ellipsoid_opt import pessimistic_value
import run_regret_incomplete as D


def learned_bc(O, A, n_obs, n_act, indices=None, pseudocount=.5):
    """Stage-1 observation policy; no latent state or environment primitives."""
    if indices is None:
        indices = np.arange(len(O))
    counts = np.full((n_obs, n_act), pseudocount, dtype=float)
    np.add.at(counts, (O[indices, 0], A[indices, 0]), 1.)
    return counts / counts.sum(axis=1, keepdims=True)


def summary(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    ci = float(student_t.ppf(.975, n - 1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(mean=float(values.mean()), ci95=ci, n=n)


def main():
    ns = [int(x) for x in os.environ.get("NGRID", "2000,8000,32000,128000").split(",")]
    seeds = list(range(int(os.environ.get("SEEDS", "20"))))
    p = default_params(n_s=4, n_o=6, n_o0=2, T=3, confound=.9, seed=0)
    original = D.candidates()
    # Match the original C schedule calibration, including its first seed.
    est, data = D.fit_once(p, ns[0], seeds[0], 1.5)
    br, bd = D.build_blocks(est)
    vg = D.make_vg(br, bd, original["greedy_lo"], empirical_p_o1(data["O"], 6))
    _, gr, _ = vg([b.b_hat for b in br], [b.b_hat for b in bd])
    unit = np.sqrt(D.xi_for(0., br[0].N2, br[0].sigma2, 1.)) * br[0].h_inv_norm(gr[0])
    cm = (.12 / unit) ** 2
    records, aggregates = [], []
    for N in ns:
        local = []
        for seed in seeds:
            est, data = D.fit_once(p, N, seed, 1.5)
            br, bd = D.build_blocks(est)
            train = np.random.default_rng(seed).permutation(N)[:est.fitted_["N1"]]
            policies = dict(original, learned_bc=learned_bc(data["O"], data["A"], 6, 2, train))
            truth = {name: dp_value(p, pi) for name, pi in policies.items()}
            best = max(truth.values())
            p0 = empirical_p_o1(data["O"], 6)
            xr = [b.width_rule(cm) for b in br]
            xd = [b.width_rule(cm) for b in bd]
            scores, gaps = {}, {}
            for name, pi in policies.items():
                vg = D.make_vg(br, bd, pi, p0)
                pv = vg([b.b_hat for b in br], [b.b_hat for b in bd])[0]
                out = pessimistic_value(br, bd, xr, xd, vg, rng=np.random.default_rng(0))
                scores[name] = dict(plugin=float(pv), full=out["V_low"])
                gaps[name] = out["restart_gap"]
            row = dict(N=N, seed=seed, truth=truth, scores=scores, restart_gaps=gaps,
                       bc_policy=policies["learned_bc"].tolist(), regrets={}, picks={})
            for arm in ("full", "plugin"):
                for label, pool in (("six", list(original)), ("learned", list(policies))):
                    pick = max(pool, key=lambda name: scores[name][arm])
                    key = f"{arm}_{label}"
                    # Common augmented reference lets candidate-set comparisons
                    # measure gains on one scale, including the learned candidate.
                    row["regrets"][key] = best - truth[pick]
                    row["picks"][key] = pick
            row["regrets"]["bc_only"] = best - truth["learned_bc"]
            local.append(row)
            print(f"N={N} seed={seed} full={row['regrets']['full_learned']:.4f} plugin={row['regrets']['plugin_learned']:.4f}", flush=True)
        records.extend(local)
        agg = dict(N=N, arms={key: summary([r["regrets"][key] for r in local])
                              for key in local[0]["regrets"]})
        agg["paired_full_minus_plugin"] = summary([r["regrets"]["full_learned"] - r["regrets"]["plugin_learned"] for r in local])
        aggregates.append(agg)
    result = dict(config=[4, 6, 2], T=3, confound=.9, environment_seed=0,
                  N_grid=ns, seeds=seeds, tau=0., kappa=1.5, c_multiplier=cm,
                  bc_training="stage-1 split, t=1 observations/actions, Jeffreys pseudocount .5",
                  regret_reference="best of six fixed and one learned candidate, per seed",
                  class_constraint="none", optimizer="coordinate descent, three random restarts",
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  summary=aggregates, records=records)
    out = HERE.parent / "experiments" / "results_learned_bc.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
