"""Recheck the 20 largest-N fits with a larger coordinate-descent budget.

Regenerates fits and the training-only clone. Does not modify the original
80-fit record. Agreement between starts is diagnostic, not global optimality.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
for folder in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, str(HERE.parent / "src" / folder))
sys.path.insert(0, str(HERE))

from dim_separated_pomdp import default_params, dp_value
from ellipsoid_opt import pessimistic_value
from run_learned_bc import learned_bc
from value_plugin import empirical_p_o1
import run_regret_incomplete as D


def main():
    source = HERE.parent / "experiments" / "results_learned_bc.json"
    recorded = json.loads(source.read_text(encoding="utf-8"))
    n = 128000
    rows = [r for r in recorded["records"] if r["N"] == n]
    if [r["seed"] for r in rows] != list(range(20)):
        raise ValueError("Expected the complete original 20-seed largest-N grid")
    params = default_params(n_s=4, n_o=6, n_o0=2, T=3, confound=.9, seed=0)
    original = D.candidates()
    results = []
    for row in rows:
        seed = row["seed"]
        est, logs = D.fit_once(params, n, seed, 1.5)
        br, bd = D.build_blocks(est)
        train = np.random.default_rng(seed).permutation(n)[:est.fitted_["N1"]]
        policies = dict(original, learned_bc=learned_bc(logs["O"], logs["A"], 6, 2, train))
        np.testing.assert_allclose(policies["learned_bc"], row["bc_policy"], rtol=0., atol=1e-14)
        xr = [b.width_rule(recorded["c_multiplier"]) for b in br]
        xd = [b.width_rule(recorded["c_multiplier"]) for b in bd]
        p0 = empirical_p_o1(logs["O"], 6)
        enhanced, gaps, changes = {}, {}, {}
        for name, pi in policies.items():
            if abs(dp_value(params, pi) - row["truth"][name]) > 1e-12:
                raise ValueError("True value differs from the original record")
            vg = D.make_vg(br, bd, pi, p0)
            plugin = vg([b.b_hat for b in br], [b.b_hat for b in bd])[0]
            if abs(plugin - row["scores"][name]["plugin"]) > 1e-8:
                raise ValueError("Regenerated fit differs from the original record")
            out = pessimistic_value(br, bd, xr, xd, vg, max_sweeps=150,
                                    n_restarts=30, rng=np.random.default_rng(seed + 1000))
            enhanced[name] = float(out["V_low"])
            gaps[name] = float(out["restart_gap"])
            changes[name] = enhanced[name] - row["scores"][name]["full"]
        picks = {label: max(pool, key=lambda name: enhanced[name])
                 for label, pool in (("six", list(original)), ("learned", list(policies)))}
        changed = {label: picks[label] != row["picks"]["full_" + label] for label in picks}
        results.append(dict(N=n, seed=seed, scores=enhanced,
                            original_scores={name: row["scores"][name]["full"] for name in policies},
                            score_changes=changes, restart_gaps=gaps, picks=picks,
                            original_picks={label: row["picks"]["full_" + label] for label in picks},
                            changed=changed, max_abs_score_change=max(abs(v) for v in changes.values())))
        print(f"seed={seed} enhanced_pick={picks['learned']} changed={changed['learned']}", flush=True)
    result = dict(
        source_records=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        environment_seed=0, N=n, seeds=list(range(20)), candidates=list(policies),
        c_multiplier=recorded["c_multiplier"],
        baseline_budget=dict(max_sweeps=30, n_restarts=3, tol=1e-11, rng_seed=0),
        enhanced_budget=dict(max_sweeps=150, n_restarts=30, tol=1e-11,
                             rng_seed="trajectory seed + 1000"),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        interpretation="Optimization sensitivity diagnostic; no certificate of global optimality.",
        summary=dict(fits=len(results), candidate_evaluations=len(results) * len(policies),
                     changed_six=sum(r["changed"]["six"] for r in results),
                     changed_learned=sum(r["changed"]["learned"] for r in results),
                     max_abs_score_change=max(r["max_abs_score_change"] for r in results)),
        records=results)
    output = HERE.parent / "experiments" / "results_optimizer_budget.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
