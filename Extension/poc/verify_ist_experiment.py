"""Independent checks of the IST boundary case, with no patient-level output.

Reconstructs the split, artificial training mechanisms, small ridge systems,
exact ellipsoid supports and held-out paired evaluation directly from the
official corrected CSV. Does not import the experiment or its estimator.
Passing validates arithmetic/provenance, not the working propensity assumption.
"""
import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ist/IST_corrected.csv"
RESULT = ROOT / "experiments/results_ist_policy_selection.json"
Z975 = 1.959963984540054
ARMS = ("full_six", "full_learned", "plugin_six", "plugin_learned", "bc_only")


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON literal: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def fixed_policies():
    lo = np.linspace(.9, .1, 6)
    p0 = dict(greedy_lo=lo, greedy_hi=1-lo, always_0=np.ones(6),
              always_1=np.zeros(6), uniform=np.full(6, .5), soft=.5+.5*(lo-.5))
    return {name: np.column_stack([prob, 1-prob]) for name, prob in p0.items()}


def replay_training(o, a, y, proxy, seed, marginal, c):
    """Independent per-target 12x12 solves instead of a dense 144x144 design."""
    n = len(o)
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n)
    n1 = int(.7*n)
    first, second = permutation[:n1], permutation[n1:]
    n2 = len(second)
    # At T=1: X=(A,O0), W=(A,O), Y=(R,O).
    x = 2*a + proxy
    w = 6*a + o
    target = 6*y + o
    counts_x = np.bincount(x[first], minlength=4).astype(float)
    counts_w = np.bincount(4*w[first]+x[first], minlength=48).reshape(12, 4)
    counts_y = np.bincount(4*target[first]+x[first], minlength=48).reshape(12, 4)
    mu = counts_w / (counts_x+1)[None, :]
    pmf = np.divide(counts_y, counts_x[None, :],
                    out=np.broadcast_to(counts_y.sum(axis=1)[:, None]/n1, (12, 4)).copy(),
                    where=counts_x[None, :] > 0)
    random_targets = generator.integers(12, size=n2)
    design = mu[:, x[second]]
    rhs_values = pmf[random_targets, x[second]]
    matrices, coefficients = [], np.zeros((12, 12))
    for target_index in range(12):
        mask = random_targets == target_index
        columns = design[:, mask]
        h = columns @ columns.T/n2 + n2**(-1.5)*np.eye(12)
        coefficients[:, target_index] = np.linalg.solve(h, columns @ rhs_values[mask]/n2)
        matrices.append(h)
    signals = []
    for action in range(2):
        columns = design[6*action:6*(action+1)]
        eig = np.linalg.eigvalsh(columns @ columns.T/n2)[::-1]
        floor = max(eig[0], 0.)*1e-9
        rank = 1 if floor <= 0 else 1+int(np.argmax(np.maximum(eig[:-1], floor)/np.maximum(eig[1:], floor)))
        signals.append(float(eig[rank-1]))
    signal = min(signals)
    if not np.isfinite(signal) or signal <= 0:
        raise ValueError("Independent replay found an undefined width")
    xi = c/(n2*signal)
    policies = fixed_policies()
    clone_counts = np.bincount(2*o[first]+a[first], minlength=12).reshape(6, 2)+.5
    policies["learned_bc"] = clone_counts/clone_counts.sum(axis=1, keepdims=True)
    scores, gradients = {}, {}
    for name, policy in policies.items():
        g = np.zeros((12, 12))
        for action in range(2):
            g[6*action:6*(action+1), 6:] = marginal[:, None]*policy[:, action][None, :]
        plugin = float(np.sum(g*coefficients))
        squared_norm = sum(float(g[:, k] @ np.linalg.solve(matrices[k], g[:, k])) for k in range(12))
        scores[name] = dict(plugin=plugin, full=plugin-np.sqrt(xi*squared_norm))
        gradients[name] = g
    return dict(n1=n1, n2=n2, signal=signal, xi=xi, policies=policies, scores=scores,
                coefficients=coefficients, matrices=matrices, gradients=gradients)


class ISTChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = read_json(RESULT)
        cls.raw = DATA.read_bytes()
        rows = list(csv.DictReader(cls.raw.decode("latin1").splitlines()))
        cls.total = len(rows)
        eligible = [i for i, row in enumerate(rows) if row["SET14D"] == "1"]
        cls.source_rows = np.array(eligible, dtype="<i8")
        cls.o = np.array([3*(rows[i]["SEX"] == "M")+{"F": 0, "D": 1, "U": 2}[rows[i]["RCONSC"]] for i in eligible])
        cls.a = np.array([int(rows[i]["RXASP"] == "Y") for i in eligible])
        cls.y = np.array([1-int(rows[i]["ID14"]) for i in eligible])
        age = np.array([int(rows[i]["AGE"]) for i in eligible])
        cls.u = (age >= 75).astype(int)
        stroke = {"PACS": 0, "LACS": 1, "TACS": 2, "POCS": 3, "OTH": 4}
        stype = np.array([stroke[rows[i]["STYPE"]] for i in eligible])
        bands = np.searchsorted([65, 75], age, side="right")
        cls.strata = (((bands*3+cls.o % 3)*5+stype)*2+cls.o//3)
        permutation = np.random.default_rng(20261111).permutation(len(eligible))
        cls.train, cls.test = permutation[:len(eligible)//2], permutation[len(eligible)//2:]
        cls.marginal = np.bincount(cls.o[cls.train], minlength=6)/len(cls.train)
        cls.replays = {}

    def test_provenance_endpoints_and_patient_split(self):
        r = self.result
        self.assertEqual(hashlib.md5(self.raw).hexdigest(), "c233a72eef0aa30b23569b2017ed17dd")
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), r["data_sha256"])
        provenance = read_json(ROOT / "data/ist/provenance.json")
        self.assertEqual(provenance["files"]["IST_corrected.csv"]["sha256"], r["data_sha256"])
        self.assertEqual((self.total, len(self.o), len(self.train), len(self.test)), (19435, 19433, 9716, 9717))
        self.assertEqual((r["n_original"], r["n_eligible"], r["n_train"], r["n_test"]), (19435, 19433, 9716, 9717))
        self.assertEqual(r["n_excluded_missing_14day"], 2)
        self.assertEqual(np.intersect1d(self.train, self.test).size, 0)
        self.assertEqual(len(np.union1d(self.train, self.test)), len(self.o))
        for label, indices in (("train", self.train), ("test", self.test)):
            digest = hashlib.sha256(self.source_rows[indices].astype("<i8").tobytes()).hexdigest()
            self.assertEqual(digest, r[label+"_source_row_indices_sha256"])
        self.assertEqual(set(self.y), {0, 1})
        self.assertIn("RXASP", r["action"])
        self.assertIn("1-ID14", r["reward"])
        self.assertNotIn("DASP14", r["action"])
        self.assertEqual(r["T"], 1)
        reference = ROOT / "experiments/results_learned_bc.json"
        self.assertEqual(r["calibration_source_sha256"], hashlib.sha256(reference.read_bytes()).hexdigest())
        self.assertEqual(r["c_multiplier"], read_json(reference)["c_multiplier"])

    @classmethod
    def get_replay(cls, row):
        key = (row["seed"], row["regime"])
        if key in cls.replays:
            return cls.replays[key]
        proxy_seed, accept_seed = np.random.SeedSequence(row["seed"]).spawn(2)
        flipped = np.random.default_rng(proxy_seed).random(len(cls.train)) < .2
        proxy = cls.u[cls.train] ^ flipped.astype(int)
        probability = np.where(cls.a[cls.train] == cls.u[cls.train], .8, .2)
        accepted = np.random.default_rng(accept_seed).random(len(cls.train)) < probability
        mask = accepted if row["regime"] == "confounded_retention" else np.ones(len(cls.train), dtype=bool)
        patients = cls.train[mask]
        replay = replay_training(cls.o[patients], cls.a[patients], cls.y[patients], proxy[mask],
                                 row["seed"], cls.marginal, cls.result["c_multiplier"])
        replay.update(patients=patients, proxy_flip_count=int(flipped.sum()))
        cls.replays[key] = replay
        return replay

    def test_training_only_mechanisms_clones_and_exact_scores(self):
        keys = [(r["seed"], r["regime"]) for r in self.result["records"]]
        expected = {(s, regime) for s in range(100, 120) for regime in ("unfiltered", "confounded_retention")}
        self.assertEqual(set(keys), expected)
        self.assertEqual(len(keys), len(expected))
        for row in self.result["records"]:
            replay = self.get_replay(row)
            patients = replay["patients"]
            self.assertTrue(np.isin(patients, self.train).all())
            self.assertFalse(np.isin(patients, self.test).any())
            self.assertEqual(row["n_retained"], len(patients))
            self.assertEqual((row["N1"], row["N2"]), (replay["n1"], replay["n2"]))
            self.assertEqual(row["proxy_flip_count"], replay["proxy_flip_count"])
            cross_counts = [[int(np.sum((self.a[patients] == a) & (self.u[patients] == u))) for a in (0, 1)] for u in (0, 1)]
            self.assertEqual(row["retained_action_by_hidden_age"], cross_counts)
            self.assertAlmostEqual(row["sigma2_signal"], replay["signal"], delta=1e-12)
            self.assertAlmostEqual(row["xi"], replay["xi"], delta=1e-12)
            for name, policy in replay["policies"].items():
                np.testing.assert_allclose(policy, row["policies"][name], rtol=0., atol=1e-14)
                np.testing.assert_allclose(policy.sum(axis=1), 1., rtol=0., atol=1e-14)
                self.assertTrue((policy >= 0).all() and (policy <= 1).all())
                for method in ("plugin", "full"):
                    self.assertAlmostEqual(row["scores"][name][method], replay["scores"][name][method], delta=1e-8)
            for method in ("full", "plugin"):
                for label, pool in (("six", list(fixed_policies())), ("learned", list(replay["policies"]))):
                    choice = max(pool, key=lambda name: replay["scores"][name][method])
                    arm = method+"_"+label
                    self.assertEqual(row["picks"][arm], choice)
                    np.testing.assert_allclose(row["selected_policy_probabilities"][arm], replay["policies"][choice], atol=1e-14)
            np.testing.assert_allclose(row["selected_policy_probabilities"]["bc_only"], replay["policies"]["learned_bc"], atol=1e-14)

    def test_support_minimum_against_feasible_two_dimensional_grid(self):
        row = self.result["records"][0]
        replay = self.get_replay(row)
        g = replay["gradients"]["greedy_lo"].ravel()
        h = np.zeros((144, 144))
        for k, block in enumerate(replay["matrices"]):
            indices = np.arange(12)*12+k
            h[np.ix_(indices, indices)] = block
        l = np.linalg.cholesky(h)
        q = np.linalg.solve(l, g)
        first = -q/np.linalg.norm(q)
        second = np.random.default_rng(765).normal(size=144)
        second -= (second @ first)*first
        second /= np.linalg.norm(second)
        d1, d2 = np.linalg.solve(l.T, np.column_stack([first, second])).T*np.sqrt(replay["xi"])
        center = replay["coefficients"].ravel()
        minimum = replay["scores"]["greedy_lo"]["full"]
        values = []
        for radius in np.linspace(0., 1., 5):
            for angle in np.linspace(0., 2*np.pi, 65):
                delta = radius*(np.cos(angle)*d1+np.sin(angle)*d2)
                self.assertLessEqual(float(delta @ h @ delta), replay["xi"]*(1+1e-8))
                values.append(float(g @ (center+delta)))
        self.assertAlmostEqual(min(values), minimum, delta=1e-8)
        self.assertGreaterEqual(min(values), minimum-1e-8)

    def check_summary(self, contributions, stored):
        estimate = float(np.mean(contributions))
        se = float(np.std(contributions, ddof=1)/np.sqrt(len(contributions)))
        self.assertEqual(stored["n_patients"], len(contributions))
        self.assertAlmostEqual(stored["estimate"], estimate, places=12)
        self.assertAlmostEqual(stored["standard_error"], se, places=12)
        self.assertAlmostEqual(stored["half_interval95"], Z975*se, places=12)
        np.testing.assert_allclose(stored["interval95"], [estimate-Z975*se, estimate+Z975*se], rtol=0., atol=1e-12)

    def test_held_out_policy_values_and_patient_paired_intervals(self):
        count = np.bincount(self.strata[self.train], minlength=90)
        treated = np.bincount(self.strata[self.train], weights=self.a[self.train], minlength=90)
        sensitivity = np.clip((treated+1)/(count+2), .1, .9)[self.strata[self.test]]
        contrasts = dict(candidate_addition_full=("full_learned", "full_six"),
                         candidate_addition_plugin=("plugin_learned", "plugin_six"),
                         full_minus_plugin_augmented=("full_learned", "plugin_learned"),
                         full_minus_clone=("full_learned", "bc_only"))
        for regime, evaluation in self.result["evaluation_results"].items():
            rows = [row for row in self.result["records"] if row["regime"] == regime]
            averages = {arm: np.mean([row["selected_policy_probabilities"][arm] for row in rows], axis=0) for arm in ARMS}
            for arm in ARMS:
                np.testing.assert_allclose(averages[arm], evaluation["mean_selected_policy"][arm], atol=1e-14)
            for label, p in (("working_half", np.full(len(self.test), .5)), ("baseline_stratified", sensitivity)):
                info = evaluation["estimators"][label]
                a, y, o = self.a[self.test], self.y[self.test], self.o[self.test]
                denominator = np.where(a == 1, p, 1-p)
                contributions = {arm: y*policy[o, a]/denominator for arm, policy in averages.items()}
                for arm in ARMS:
                    self.check_summary(contributions[arm], info["values"][arm])
                for name, (first, second) in contrasts.items():
                    self.check_summary(contributions[first]-contributions[second], info["paired_value_differences"][name])
                self.assertAlmostEqual(info["propensity_min"], float(p.min()), places=14)
                self.assertAlmostEqual(info["propensity_max"], float(p.max()), places=14)
                self.assertAlmostEqual(info["inverse_observed_propensity_weight_max"], float(np.max(1/denominator)), places=12)

    def test_manuscript_clinical_numbers(self):
        tex = (ROOT / "paper/workshop_draft.tex").read_text(encoding="utf-8")
        for regime, label in (("unfiltered", "Unfiltered"), ("confounded_retention", "Selection-biased")):
            for weight, description in (("working_half", "Working .5"), ("baseline_stratified", "Baseline sensitivity")):
                stats = self.result["evaluation_results"][regime]["estimators"][weight]["paired_value_differences"]["candidate_addition_full"]
                cell = f"${stats['estimate']:.6f}\\pm{stats['half_interval95']:.6f}$".replace("0.", ".")
                self.assertIn(f"{label} & {description} & {cell}", tex)
        self.assertIn("9,716", tex)
        self.assertIn("9,717", tex)
        self.assertIn("19,433", tex)

    def test_interpretation_missingness_and_no_patient_records(self):
        r = self.result
        self.assertIn("not ground truth", r["evaluation"])
        self.assertIn("conditional", r["uncertainty"])
        self.assertTrue(any(".5 propensity is an approximation" in line for line in r["limitations"]))
        self.assertTrue(any("not sequential clinical validation" in line for line in r["limitations"]))
        self.assertIn("not independent clinical trials", r["repetition_interpretation"])
        bound = r["missing_outcome_bound_working_half"]
        self.assertAlmostEqual(bound["full_trial_value_difference_scaling"], 19433/19435, places=14)
        self.assertAlmostEqual(bound["additional_absolute_contribution_bound"], 2*2/19435, places=14)
        self.assertEqual(r["validation"]["policy_score_checks"], 40*7)
        # Only 40 aggregate fit records/policies are published, not patient rows.
        forbidden = {"patient_id", "patient_ids", "trial_patient_number", "HOSPNUM", "AGE", "ID14", "RXASP"}
        def inspect(item):
            if isinstance(item, dict):
                self.assertFalse(forbidden & set(item))
                for value in item.values():
                    inspect(value)
            elif isinstance(item, list):
                self.assertLess(len(item), 100)
                for value in item:
                    inspect(value)
        inspect(r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
