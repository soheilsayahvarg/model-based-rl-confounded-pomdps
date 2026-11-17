"""IST v2 real-outcome, T=1 boundary case with artificial observationalisation.

Fixed design: outcome-complete patients only; one 50/50 patient split using
20261111; 20 training perturbations (100..119), no held-out outcome tuning.
O = sex x consciousness; hidden-to-learner U = age>=75. The proxy and training
selection are synthetic. Randomized aspirin ASSIGNMENT is the action. Trial
evaluation uses a working .5 propensity, with a baseline-stratified sensitivity.
IST used minimisation, so .5 is NOT asserted to be each subject's known
randomisation probability. Values are held-out estimates, not exact truth or
clinical recommendations. Output contains no patient records or identifiers.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

import run_learned_bc as L

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data/ist/IST_corrected.csv"
OUTPUT = HERE.parent / "experiments/results_ist_policy_selection.json"
REFERENCE = HERE.parent / "experiments/results_learned_bc.json"
EXPECTED_MD5 = "c233a72eef0aa30b23569b2017ed17dd"
SEEDS = list(range(100, 120))
SPLIT_SEED = 20261111
ARMS = ["full_six", "full_learned", "plugin_six", "plugin_learned", "bc_only"]


def retention_mask(action, hidden_age, rng):
    """No outcome argument: acceptance depends only on baseline age and action."""
    probability = np.where(action == hidden_age, .8, .2)
    return rng.random(len(action)) < probability


def gradient(policy, marginal):
    # a, o-tilde, r, o ordering matches TabularBridgeEstimator's vectorization.
    return np.einsum("i,r,oa->airo", marginal, np.array([0., 1.]), policy).ravel()


def load_arrays():
    raw = DATA.read_bytes()
    assert hashlib.md5(raw).hexdigest() == EXPECTED_MD5, "Use the corrected official IST v2 CSV"
    rows = list(csv.DictReader(raw.decode("latin1").splitlines()))
    kept = [i for i, r in enumerate(rows) if r["SET14D"] == "1"]
    assert len(rows) == 19435 and len(kept) == 19433
    sex = {"F": 0, "M": 1}
    consciousness = {"F": 0, "D": 1, "U": 2}
    stroke = {"PACS": 0, "LACS": 1, "TACS": 2, "POCS": 3, "OTH": 4}
    age = np.array([int(rows[i]["AGE"]) for i in kept])
    o = np.array([3 * sex[rows[i]["SEX"]] + consciousness[rows[i]["RCONSC"]] for i in kept])
    a = np.array([{"N": 0, "Y": 1}[rows[i]["RXASP"]] for i in kept])
    y = 1 - np.array([int(rows[i]["ID14"]) for i in kept])
    assert set(y) == {0, 1}
    stype = np.array([stroke[rows[i]["STYPE"]] for i in kept])
    # 3 age bands x 3 consciousness levels x 5 stroke types x 2 sexes.
    age_band = np.where(age < 65, 0, np.where(age < 75, 1, 2))
    strata = (((age_band * 3 + o % 3) * 5 + stype) * 2 + o // 3)
    return dict(raw=raw, o=o, a=a, y=y, u=(age >= 75).astype(int), strata=strata,
                eligible_source_rows=np.asarray(kept, dtype=np.int64), excluded=len(rows)-len(kept), total=len(rows))


def normal_summary(contribution):
    contribution = np.asarray(contribution)
    estimate = float(contribution.mean())
    se = float(contribution.std(ddof=1) / np.sqrt(len(contribution)))
    return dict(estimate=estimate, standard_error=se, half_interval95=1.959963984540054*se,
                interval95=[estimate-1.959963984540054*se, estimate+1.959963984540054*se],
                n_patients=len(contribution))


def exact_score_checks(block, xi, policies, marginal):
    """Check tensor order, support optimizer and sampled feasible competitors."""
    rng = np.random.default_rng(81)
    for policy in policies.values():
        g = gradient(policy, marginal)
        value, _, gradients = L.D.plugin_value(block.b_hat.reshape(1, 2, 6, 2, 6),
                                              np.zeros((0, 2, 6, 6, 6)), policy,
                                              marginal, return_grads=True)
        np.testing.assert_allclose(g, gradients[0][0].ravel(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(g @ block.b_hat, value, atol=1e-12)
        minimum = float(g @ block.b_hat - np.sqrt(xi)*block.h_inv_norm(g))
        bmin, _ = block.linear_min(g, xi)
        np.testing.assert_allclose(g @ bmin, minimum, atol=1e-9)
        d = bmin - block.b_hat
        assert d @ block.H @ d <= xi * (1+1e-7) + 1e-12
        for _ in range(4):
            # H=L L^T; delta=L^{-T}z*sqrt(xi) has ellipsoid norm sqrt(xi).
            z = rng.normal(size=len(g))
            z /= np.linalg.norm(z)
            d = solve_triangular(block.L.T, z, lower=False)*np.sqrt(xi)
            assert g @ (block.b_hat+d) >= minimum-1e-9


def evaluate(mean_policy, o, a, y, propensity):
    probability_of_observed_action = np.where(a == 1, propensity, 1-propensity)
    weight = mean_policy[o, a] / probability_of_observed_action
    return normal_summary(y*weight), y*weight


def main():
    d = load_arrays()
    n = len(d["o"])
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    train, test = perm[:n//2], perm[n//2:]
    assert not set(train) & set(test) and len(train)+len(test) == n
    assert max(d["eligible_source_rows"][train]) < d["total"]
    # Retained data train the bridge; the unfiltered training covariates define
    # the target population marginal and never expose held-out covariates/outcomes.
    marginal = np.bincount(d["o"][train], minlength=6).astype(float) / len(train)
    reference = REFERENCE.read_bytes()
    calibration = json.loads(reference)["c_multiplier"]
    original = L.D.candidates()
    records = []
    selected = {regime: {arm: [] for arm in ARMS} for regime in ("unfiltered", "confounded_retention")}
    score_checks = 0
    for seed in SEEDS:
        proxy_rng, selection_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        proxy = d["u"][train] ^ (proxy_rng.random(len(train)) < .2).astype(int)
        accept = retention_mask(d["a"][train], d["u"][train], selection_rng)
        # Same mask results with any outcomes, because no outcomes enter this function.
        check_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        assert np.array_equal(accept, retention_mask(d["a"][train], d["u"][train], check_rng))
        for regime, mask in (("unfiltered", np.ones(len(train), dtype=bool)), ("confounded_retention", accept)):
            indices = train[mask]
            O, A, R, O0 = d["o"][indices, None], d["a"][indices, None], d["y"][indices, None], proxy[mask]
            N = len(indices)
            N2 = N-int(.7*N)
            est = L.D.TabularBridgeEstimator(n_obs=6, n_act=2, n_o0=2, n_r=2, T=1,
                                             mode="primal", seed=seed, lambda2=N2**(-1.5))
            est.fit(O0, O, A, R)
            stage = est.stage_store["bR_t1"]
            block = L.D.BlockEllipsoid(stage["H"], stage["b_hat_vec"], stage["sigma2_signal"],
                                      stage["N2"], "bR_t1", stage["signal_basis"], n_obs=6, n_act=2, n_y=12)
            xi = block.width_rule(calibration)
            fit_train = np.random.default_rng(seed).permutation(N)[:est.fitted_["N1"]]
            policies = dict(original, learned_bc=L.learned_bc(O, A, 6, 2, fit_train))
            scores = {}
            for name, policy in policies.items():
                g = gradient(policy, marginal)
                plugin = float(g @ block.b_hat)
                scores[name] = dict(plugin=plugin, full=plugin-float(np.sqrt(xi)*block.h_inv_norm(g)))
            exact_score_checks(block, xi, policies, marginal)
            score_checks += len(policies)
            picks = {}
            for score in ("full", "plugin"):
                for label, pool in (("six", list(original)), ("learned", list(policies))):
                    arm = score+"_"+label
                    picks[arm] = max(pool, key=lambda name: scores[name][score])
                    selected[regime][arm].append(policies[picks[arm]])
            selected[regime]["bc_only"].append(policies["learned_bc"])
            records.append(dict(seed=seed, regime=regime, n_retained=N, N1=est.fitted_["N1"], N2=N2,
                                proxy_flip_count=int(np.sum(proxy != d["u"][train])),
                                retained_action_by_hidden_age=[[int(np.sum((d["a"][indices] == a)&(d["u"][indices] == u)))
                                                               for a in (0, 1)] for u in (0, 1)],
                                scores=scores, picks=picks, policies={k:v.tolist() for k,v in policies.items()},
                                selected_policy_probabilities={arm: selected[regime][arm][-1].tolist() for arm in ARMS},
                                sigma2_signal=block.sigma2, xi=xi))
            print(f"IST seed={seed} regime={regime} N={N} picks={picks}", flush=True)
    # Sensitivity propensity fitted only to unfiltered training A/baseline, no Y.
    count = np.bincount(d["strata"][train], minlength=90)
    treated = np.bincount(d["strata"][train], weights=d["a"][train], minlength=90)
    propensity_table = np.clip((treated+1)/(count+2), .1, .9)
    sensitivities = {"working_half": np.full(len(test), .5),
                     "baseline_stratified": propensity_table[d["strata"][test]]}
    evaluation = {}
    for regime in selected:
        mean_policies = {arm: np.mean(selected[regime][arm], axis=0) for arm in ARMS}
        evaluation[regime] = dict(mean_selected_policy={arm: pi.tolist() for arm,pi in mean_policies.items()}, estimators={})
        for label, propensity in sensitivities.items():
            contributions, values = {}, {}
            for arm, policy in mean_policies.items():
                values[arm], contributions[arm] = evaluate(policy, d["o"][test], d["a"][test], d["y"][test], propensity)
            contrasts = {"candidate_addition_full": ("full_learned", "full_six"),
                         "candidate_addition_plugin": ("plugin_learned", "plugin_six"),
                         "full_minus_plugin_augmented": ("full_learned", "plugin_learned"),
                         "full_minus_clone": ("full_learned", "bc_only")}
            paired = {name: normal_summary(contributions[a]-contributions[b]) for name,(a,b) in contrasts.items()}
            observed_probability = np.where(d["a"][test], propensity, 1-propensity)
            evaluation[regime]["estimators"][label] = dict(values=values, paired_value_differences=paired,
                propensity_min=float(propensity.min()), propensity_max=float(propensity.max()),
                inverse_observed_propensity_weight_max=float(np.max(1/observed_probability)))
    result = dict(dataset="International Stroke Trial corrected v2; DOI 10.7488/ds/104",
                  dataset_license="Open Data Commons Attribution License (ODC-By) v1.0; separate from code license",
                  source_url="https://datashare.ed.ac.uk/items/91709c88-d913-45d8-8019-380027133d54",
                  data_sha256=hashlib.sha256(d["raw"]).hexdigest(), data_md5=EXPECTED_MD5,
                  n_original=d["total"], n_eligible=n, n_excluded_missing_14day=d["excluded"],
                  n_train=len(train), n_test=len(test), split_seed=SPLIT_SEED,
                  train_source_row_indices_sha256=hashlib.sha256(d["eligible_source_rows"][train].astype("<i8").tobytes()).hexdigest(),
                  test_source_row_indices_sha256=hashlib.sha256(d["eligible_source_rows"][test].astype("<i8").tobytes()).hexdigest(),
                  feature_definition="O=3*sex+consciousness; sex F=0,M=1; consciousness F=0,D=1,U=2",
                  hidden_baseline="U=(AGE>=75); available to artificial selector/proxy and evaluation propensity only",
                  action="RXASP assigned aspirin Y=1,N=0; intention-to-treat, marginal over factorial heparin assignment",
                  reward="1-ID14; SET14D=0 omitted before fixed split, no other exclusions",
                  proxy="synthetic O0=U XOR independent Bernoulli(.2)",
                  retention="synthetic probability .8 if A==U, .2 otherwise; outcome unused",
                  repeats=SEEDS, repetition_interpretation="training perturbations of one fixed dataset, not independent clinical trials",
                  T=1, n_obs=6, n_act=2, n_o0=2, n_r=2, kappa=1.5,
                  c_multiplier=calibration, calibration_source_sha256=hashlib.sha256(reference).hexdigest(),
                  calibration="unchanged synthetic-environment multiplier; not selected with IST outcomes",
                  target_observation_distribution="unfiltered training patient covariates only",
                  baseline_propensity="training-only Beta(1,1) strata age(<65,65-74,>=75) x consciousness x stroke type x sex; clip .1,.9",
                  evaluation="held-out unnormalised IPW estimated value of averaged selected policy; not ground truth or causal regret",
                  uncertainty="normal patient-level paired standard errors conditional on the training-derived mean policy; minimisation/adaptive assignment not reconstructed",
                  missing_outcome_bound_working_half=dict(full_trial_value_difference_scaling=n/d["total"],
                                                         additional_absolute_contribution_bound=2*d["excluded"]/d["total"]),
                  validation=dict(disjoint_split=True, exact_linear_support_and_random_feasible_points=True,
                                  gradient_order_checked_against_plugin=True, policy_score_checks=score_checks,
                                  retention_function_has_no_outcome_input=True, all_40_fits_complete=len(records)==40),
                  limitations=["single-decision semi-synthetic observationalisation, not sequential clinical validation",
                               "working .5 propensity is an approximation: original trial used minimisation",
                               "baseline-stratified sensitivity does not reconstruct allocation-history probabilities",
                               "artificial proxy does not establish existence/identification of a valid clinical bridge",
                               "early trial outcome and historic treatment population; no clinical recommendations",
                               "conditional patient-level normal intervals omit training and adaptive-randomisation uncertainty"],
                  evaluation_results=evaluation, records=records)
    assert len(records) == 40 and all(r["regime"] in selected for r in records)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}", flush=True)
    for regime in evaluation:
        for label, item in evaluation[regime]["estimators"].items():
            print(regime, label, json.dumps(item["paired_value_differences"]), flush=True)


if __name__ == "__main__":
    main()
