"""Predeclared reward-reversal stress control, separate from original results.

Keep emissions, transitions and q=.9 logger; swap the two reward-action columns
so the logger's preferred latent-state action is no longer the rewarded one.
Fixed original calibration, environments 1..5, fresh data seeds 30..34, N=128000.
This is a deliberately adverse synthetic control, not a claim of real-world
external validity. All matched-fit choices share the same augmented reference.
"""
import hashlib
import json
import os
import platform

import numpy as np
import scipy

import run_external_validity as X

L = X.L
OUTPUT = X.HERE.parent / "experiments/results_reward_reversal.json"
ENVIRONMENTS = list(range(1, 6))
DATA_SEEDS = list(range(30, 35))
N = 128000


def parameters(environment_seed):
    p = L.default_params(n_s=4, n_o=6, n_o0=2, T=3, confound=.9, seed=environment_seed)
    p["pR"] = p["pR"][:, ::-1].copy()
    return p


def verify(result):
    expected = {(e, N, s) for e in ENVIRONMENTS for s in DATA_SEEDS}
    actual = [(r["environment_seed"], r["N"], r["seed"]) for r in result["records"]]
    assert len(actual) == len(expected) and set(actual) == expected
    assert set(DATA_SEEDS).isdisjoint(set(range(30)))
    reference = json.loads(X.REFERENCE.read_text())
    assert result["c_multiplier"] == reference["c_multiplier"]
    for row in result["records"]:
        p = parameters(row["environment_seed"])
        policies = dict(L.D.candidates(), learned_bc=np.asarray(row["bc_policy"]))
        np.testing.assert_allclose(policies["learned_bc"].sum(axis=1), 1.)
        assert np.all(policies["learned_bc"] >= 0.)
        for name, pi in policies.items():
            np.testing.assert_allclose(row["truth"][name], X.forward_value(p, pi), atol=1e-12, rtol=0)
        for arm in ("full", "plugin"):
            for label, pool in (("six", list(L.D.candidates())), ("learned", list(policies))):
                pick = max(pool, key=lambda name: row["scores"][name][arm])
                key = arm + "_" + label
                assert row["picks"][key] == pick
                np.testing.assert_allclose(row["regrets"][key], max(row["truth"].values()) - row["truth"][pick], atol=1e-12)
        np.testing.assert_allclose(row["regrets"]["bc_only"], max(row["truth"].values()) - row["truth"]["learned_bc"], atol=1e-12)
    for item in result["per_environment"]:
        rows = [r for r in result["records"] if r["environment_seed"] == item["environment_seed"]]
        assert item["statistics"] == X.summarize(rows)
    means = result["per_environment"]
    for key in ("candidate_gain", "full_minus_plugin", "full_minus_clone"):
        assert result["across_environments"][key] == L.summary([r["statistics"][key]["mean"] for r in means])
    for arm in X.ARMS:
        assert result["across_environments"]["arms"][arm] == L.summary([r["statistics"]["arms"][arm]["mean"] for r in means])
    print("Verified reward-reversed grid, true values, picks, regrets and summaries.", flush=True)


def main():
    if os.environ.get("VERIFY_ONLY") == "1":
        verify(json.loads(OUTPUT.read_text()))
        return
    reference_bytes = X.REFERENCE.read_bytes()
    calibration = json.loads(reference_bytes)["c_multiplier"]
    records, per_environment = [], []
    for environment_seed in ENVIRONMENTS:
        p = parameters(environment_seed)
        original = L.D.candidates()
        local = []
        for seed in DATA_SEEDS:
            est, data = L.D.fit_once(p, N, seed, 1.5)
            br, bd = L.D.build_blocks(est)
            train = np.random.default_rng(seed).permutation(N)[:est.fitted_["N1"]]
            policies = dict(original, learned_bc=L.learned_bc(data["O"], data["A"], 6, 2, train))
            truth = {name: L.dp_value(p, pi) for name, pi in policies.items()}
            p0 = L.empirical_p_o1(data["O"], 6)
            xr, xd = [b.width_rule(calibration) for b in br], [b.width_rule(calibration) for b in bd]
            scores, gaps = {}, {}
            for name, pi in policies.items():
                vg = L.D.make_vg(br, bd, pi, p0)
                plugin = vg([b.b_hat for b in br], [b.b_hat for b in bd])[0]
                out = L.pessimistic_value(br, bd, xr, xd, vg, rng=np.random.default_rng(0))
                scores[name] = dict(plugin=float(plugin), full=float(out["V_low"]))
                gaps[name] = float(out["restart_gap"])
            row = dict(environment_seed=environment_seed, N=N, seed=seed,
                       truth=truth, scores=scores, restart_gaps=gaps,
                       bc_policy=policies["learned_bc"].tolist(), picks={}, regrets={},
                       signal_estimates=[float(b.sigma2) for b in br + bd])
            best = max(truth.values())
            for arm in ("full", "plugin"):
                for label, pool in (("six", list(original)), ("learned", list(policies))):
                    pick = max(pool, key=lambda name: scores[name][arm])
                    key = arm + "_" + label
                    row["picks"][key] = pick
                    row["regrets"][key] = best - truth[pick]
            row["regrets"]["bc_only"] = best - truth["learned_bc"]
            local.append(row)
            print(f"reward-reversed env={environment_seed} N={N} seed={seed} six={row['regrets']['full_six']:.5f} learned={row['regrets']['full_learned']:.5f} bc={row['regrets']['bc_only']:.5f}", flush=True)
        records.extend(local)
        per_environment.append(dict(environment_seed=environment_seed, N=N, statistics=X.summarize(local)))
    across = dict(N=N, environment_count=5,
                  arms={arm: L.summary([r["statistics"]["arms"][arm]["mean"] for r in per_environment]) for arm in X.ARMS})
    for key in ("candidate_gain", "full_minus_plugin", "full_minus_clone"):
        across[key] = L.summary([r["statistics"][key]["mean"] for r in per_environment])
    gain = [r["statistics"]["candidate_gain"]["mean"] for r in per_environment]
    across.update(environments_candidate_helped=sum(g > 1e-12 for g in gain),
                  environments_candidate_hurt=sum(g < -1e-12 for g in gain),
                  environments_candidate_tied=sum(abs(g) <= 1e-12 for g in gain))
    result = dict(design="Reward-reversal stress control fixed before inspecting its outcomes",
                  reward_intervention="After default_params, pR = pR[:, ::-1].copy(); logger unchanged",
                  config=[4, 6, 2], T=3, confound=.9, environment_seeds=ENVIRONMENTS,
                  data_seeds=DATA_SEEDS, N_grid=[N], c_multiplier=calibration,
                  calibration_environment=0, calibration_data_seed=0, calibration_N=2000,
                  calibration_tuned_on_new_environments=False,
                  calibration_reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  tau=0., kappa=1.5, class_constraint="none; heuristic scores without coverage claim",
                  optimizer="30 coordinate sweeps, three random restarts plus centre",
                  regret_reference="same per-seed best of six fixed plus learned clone for all arms",
                  intervals="95% Student-t descriptive half-intervals; n=5 within and across environments",
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  per_environment=per_environment, across_environments=across, records=records)
    verify(result)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}", flush=True)
    print(json.dumps(across, indent=2), flush=True)


if __name__ == "__main__":
    main()
