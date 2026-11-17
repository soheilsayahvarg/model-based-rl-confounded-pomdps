"""Fixed-calibration extension to five new environments and fresh data seeds.

Design fixed before inspecting the results: environments 1..5, data seeds 20..29,
N in {8000,128000}, the original q=.9 and (S,O,O0,A,T)=(4,6,2,2,3).
The calibration multiplier is transferred unchanged from the original run.
Every seed uses shared fitted bridges and a common augmented regret reference.
This is exploratory robustness within one generator, not a broad benchmark.
Writes only results_external_validity.json; never changes the original record.
"""
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy

import run_learned_bc as L

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "experiments/results_learned_bc.json"
OUTPUT = HERE.parent / "experiments/results_external_validity.json"
ENVIRONMENTS = list(range(1, 6))
DATA_SEEDS = list(range(20, 30))
N_GRID = [8000, 128000]
ARMS = ["full_six", "full_learned", "plugin_six", "plugin_learned", "bc_only"]


def forward_value(params, policy):
    """Independent forward distribution calculation, checked against backward DP."""
    action_given_state = params["E"] @ policy
    distribution = params["p1"].copy()
    total = 0.
    for _ in range(params["T"]):
        state_action = distribution[:, None] * action_given_state
        total += float(np.sum(state_action * params["pR"]))
        distribution = sum(state_action[:, a] @ params["P"][a]
                           for a in range(params["n_a"]))
    return total


def paired_summary(rows, first, second):
    return L.summary([r["regrets"][first] - r["regrets"][second] for r in rows])


def summarize(rows):
    return dict(arms={arm: L.summary([r["regrets"][arm] for r in rows]) for arm in ARMS},
                candidate_gain=paired_summary(rows, "full_six", "full_learned"),
                full_minus_plugin=paired_summary(rows, "full_learned", "plugin_learned"),
                full_minus_clone=paired_summary(rows, "full_learned", "bc_only"))


def verify(result):
    """Recompute all choices, regrets, summaries and independent true values."""
    expected = {(e, n, s) for e in ENVIRONMENTS for n in N_GRID for s in DATA_SEEDS}
    observed = [(r["environment_seed"], r["N"], r["seed"]) for r in result["records"]]
    assert set(observed) == expected and len(observed) == len(expected)
    assert set(result["environment_seeds"]).isdisjoint({0})
    assert set(result["data_seeds"]).isdisjoint(set(range(20)))
    reference = json.loads(REFERENCE.read_text())
    assert result["c_multiplier"] == reference["c_multiplier"]
    pools = {"six": list(L.D.candidates()), "learned": list(L.D.candidates()) + ["learned_bc"]}
    for row in result["records"]:
        p = L.default_params(n_s=4, n_o=6, n_o0=2, T=3, confound=.9,
                             seed=row["environment_seed"])
        policies = dict(L.D.candidates(), learned_bc=np.asarray(row["bc_policy"]))
        np.testing.assert_allclose(policies["learned_bc"].sum(axis=1), 1.)
        assert np.all(policies["learned_bc"] >= 0)
        for name, policy in policies.items():
            np.testing.assert_allclose(row["truth"][name], forward_value(p, policy), atol=1e-12, rtol=0)
        for score in ("full", "plugin"):
            for label, pool in pools.items():
                pick = max(pool, key=lambda name: row["scores"][name][score])
                key = score + "_" + label
                assert row["picks"][key] == pick
                np.testing.assert_allclose(row["regrets"][key],
                                           max(row["truth"].values()) - row["truth"][pick], atol=1e-12)
        np.testing.assert_allclose(row["regrets"]["bc_only"],
                                   max(row["truth"].values()) - row["truth"]["learned_bc"], atol=1e-12)
    for item in result["per_environment"]:
        rows = [r for r in result["records"] if (r["environment_seed"], r["N"]) ==
                (item["environment_seed"], item["N"])]
        assert item["statistics"] == summarize(rows)
    # Summaries intentionally use environment means, not 50 pseudo-independent runs.
    for item in result["across_environments"]:
        means = [r for r in result["per_environment"] if r["N"] == item["N"]]
        for key in ("candidate_gain", "full_minus_plugin", "full_minus_clone"):
            assert item[key] == L.summary([r["statistics"][key]["mean"] for r in means])
        for arm in ARMS:
            assert item["arms"][arm] == L.summary([r["statistics"]["arms"][arm]["mean"] for r in means])
    assert all(np.isfinite(value) for r in result["records"] for value in r["regrets"].values())
    print("Verified grid, calibration separation, true values, scores/picks, regrets and summaries.", flush=True)


def main():
    if os.environ.get("VERIFY_ONLY") == "1":
        verify(json.loads(OUTPUT.read_text()))
        return
    reference_bytes = REFERENCE.read_bytes()
    calibration = json.loads(reference_bytes)["c_multiplier"]
    records = []
    per_environment = []
    for environment_seed in ENVIRONMENTS:
        p = L.default_params(n_s=4, n_o=6, n_o0=2, T=3, confound=.9, seed=environment_seed)
        original = L.D.candidates()
        for N in N_GRID:
            local = []
            for seed in DATA_SEEDS:
                est, data = L.D.fit_once(p, N, seed, 1.5)
                br, bd = L.D.build_blocks(est)
                train = np.random.default_rng(seed).permutation(N)[:est.fitted_["N1"]]
                policies = dict(original, learned_bc=L.learned_bc(data["O"], data["A"], 6, 2, train))
                truth = {name: L.dp_value(p, pi) for name, pi in policies.items()}
                p0 = L.empirical_p_o1(data["O"], 6)
                xr = [b.width_rule(calibration) for b in br]
                xd = [b.width_rule(calibration) for b in bd]
                scores, gaps = {}, {}
                for name, pi in policies.items():
                    vg = L.D.make_vg(br, bd, pi, p0)
                    plug = vg([b.b_hat for b in br], [b.b_hat for b in bd])[0]
                    pessimistic = L.pessimistic_value(br, bd, xr, xd, vg, rng=np.random.default_rng(0))
                    scores[name] = dict(plugin=float(plug), full=float(pessimistic["V_low"]))
                    gaps[name] = float(pessimistic["restart_gap"])
                row = dict(environment_seed=environment_seed, N=N, seed=seed,
                           truth=truth, scores=scores, restart_gaps=gaps,
                           bc_policy=policies["learned_bc"].tolist(), picks={}, regrets={},
                           signal_estimates=[float(b.sigma2) for b in br + bd])
                best = max(truth.values())
                for score in ("full", "plugin"):
                    for label, pool in (("six", list(original)), ("learned", list(policies))):
                        pick = max(pool, key=lambda name: scores[name][score])
                        key = score + "_" + label
                        row["picks"][key] = pick
                        row["regrets"][key] = best - truth[pick]
                row["regrets"]["bc_only"] = best - truth["learned_bc"]
                local.append(row)
                print(f"env={environment_seed} N={N} seed={seed} six={row['regrets']['full_six']:.5f} learned={row['regrets']['full_learned']:.5f}", flush=True)
            records.extend(local)
            per_environment.append(dict(environment_seed=environment_seed, N=N, statistics=summarize(local)))
    across = []
    for N in N_GRID:
        rows = [r for r in per_environment if r["N"] == N]
        item = dict(N=N, environment_count=len(rows),
                    arms={arm: L.summary([r["statistics"]["arms"][arm]["mean"] for r in rows]) for arm in ARMS})
        for key in ("candidate_gain", "full_minus_plugin", "full_minus_clone"):
            item[key] = L.summary([r["statistics"][key]["mean"] for r in rows])
        gain = [r["statistics"]["candidate_gain"]["mean"] for r in rows]
        item["environments_candidate_helped"] = sum(g > 1e-12 for g in gain)
        item["environments_candidate_hurt"] = sum(g < -1e-12 for g in gain)
        item["environments_candidate_tied"] = sum(abs(g) <= 1e-12 for g in gain)
        across.append(item)
    result = dict(design="Fixed before inspecting results; exploratory transfer within one generator",
                  config=[4, 6, 2], T=3, confound=.9, N_grid=N_GRID,
                  environment_seeds=ENVIRONMENTS, data_seeds=DATA_SEEDS,
                  calibration_environment=0, calibration_data_seed=0, calibration_N=2000,
                  c_multiplier=calibration, calibration_tuned_on_new_environments=False,
                  calibration_reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  tau=0., kappa=1.5, optimizer="30 coordinate sweeps, three random restarts plus centre",
                  class_constraint="none; heuristic scores with no coverage claim",
                  regret_reference="same per-seed best of six fixed plus learned clone for all arms",
                  intervals="95% Student-t descriptive half-intervals: n=10 within environment; n=5 across environment means",
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  per_environment=per_environment, across_environments=across, records=records)
    verify(result)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}", flush=True)
    print(json.dumps(across, indent=2), flush=True)


if __name__ == "__main__":
    main()
