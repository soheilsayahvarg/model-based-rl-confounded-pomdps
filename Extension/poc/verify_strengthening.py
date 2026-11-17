"""Independent numeric provenance for the November sensitivity experiments.

No experiment is rerun here. Raw selections, common references, clustered
summaries, score-error algebra and manuscript cells are checked independently.
"""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T95 = {4: 2.7764451051977987, 9: 2.2621571628540993}


def load(name):
    def invalid(value):
        raise ValueError(f"Non-finite JSON: {value}")
    return json.loads((ROOT / "experiments" / name).read_text(), parse_constant=invalid)


class StrengtheningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load("results_learned_bc.json")
        cls.transfer = load("results_external_validity.json")
        cls.reversal = load("results_reward_reversal.json")
        cls.tex = (ROOT / "paper/workshop_draft.tex").read_text(encoding="utf-8")

    def test_fresh_grids_and_fixed_calibration(self):
        for data, sizes, seeds in ((self.transfer, [8000, 128000], range(20, 30)),
                                   (self.reversal, [128000], range(30, 35))):
            expected = {(e, n, s) for e in range(1, 6) for n in sizes for s in seeds}
            keys = [(r["environment_seed"], r["N"], r["seed"]) for r in data["records"]]
            self.assertEqual(set(keys), expected)
            self.assertEqual(len(keys), len(expected))
            self.assertEqual(data["c_multiplier"], self.original["c_multiplier"])
            self.assertFalse(data["calibration_tuned_on_new_environments"])
            self.assertEqual(data["calibration_reference_sha256"], hashlib.sha256(
                (ROOT / "experiments/results_learned_bc.json").read_bytes()).hexdigest())

    def test_raw_selection_and_score_error_comparison(self):
        for data in (self.original, self.transfer, self.reversal):
            for r in data["records"]:
                names = list(r["scores"])
                best = max(r["truth"].values())
                for method in ("full", "plugin"):
                    for label, pool in (("six", names[:6]), ("learned", names)):
                        arm = method + "_" + label
                        pick = max(pool, key=lambda k: r["scores"][k][method])
                        self.assertEqual(pick, r["picks"][arm])
                        self.assertAlmostEqual(r["regrets"][arm], best - r["truth"][pick], places=12)
                    i, j = [r["picks"][method + "_" + s] for s in ("six", "learned")]
                    error = {k: r["scores"][k][method] - r["truth"][k] for k in names}
                    change = r["regrets"][method + "_six"] - r["regrets"][method + "_learned"]
                    identity = r["scores"][j][method] - r["scores"][i][method] + error[i] - error[j]
                    self.assertAlmostEqual(change, identity, places=12)
                    self.assertGreaterEqual(change + 1e-12, -(max(error.values()) - min(error.values())))
                self.assertAlmostEqual(r["regrets"]["bc_only"], best - r["truth"]["learned_bc"], places=12)

    def check_stats(self, values, stats):
        self.assertEqual(len(values), stats["n"])
        self.assertAlmostEqual(np.mean(values), stats["mean"], places=12)
        half = T95[len(values) - 1] * np.std(values, ddof=1) / np.sqrt(len(values))
        self.assertAlmostEqual(half, stats["ci95"], places=10)

    def test_clustered_summaries(self):
        contrasts = {"candidate_gain": ("full_six", "full_learned"),
                     "full_minus_plugin": ("full_learned", "plugin_learned"),
                     "full_minus_clone": ("full_learned", "bc_only")}
        for data in (self.transfer, self.reversal):
            for env in data["per_environment"]:
                rows = [r for r in data["records"] if r["environment_seed"] == env["environment_seed"] and r["N"] == env["N"]]
                stats = env["statistics"]
                for arm, summary in stats["arms"].items():
                    self.check_stats([r["regrets"][arm] for r in rows], summary)
                for name, (a, b) in contrasts.items():
                    self.check_stats([r["regrets"][a] - r["regrets"][b] for r in rows], stats[name])
            across = data["across_environments"]
            across = across if isinstance(across, list) else [across]
            for summary in across:
                envs = [e["statistics"] for e in data["per_environment"] if e["N"] == summary["N"]]
                for arm, stats in summary["arms"].items():
                    self.check_stats([e["arms"][arm]["mean"] for e in envs], stats)
                for name in contrasts:
                    self.check_stats([e[name]["mean"] for e in envs], summary[name])
                gains = [e["candidate_gain"]["mean"] for e in envs]
                for label, count in (("helped", sum(v > 1e-12 for v in gains)),
                                     ("hurt", sum(v < -1e-12 for v in gains)),
                                     ("tied", sum(abs(v) <= 1e-12 for v in gains))):
                    self.assertEqual(summary["environments_candidate_" + label], count)

    def test_manuscript_tables(self):
        def cell(s):
            return f"${s['mean']:.5f}\\pm{s['ci95']:.5f}$".replace("0.", ".")
        transfer = self.transfer["across_environments"]
        for label, key in (("Pessimistic, six", "full_six"), ("Pessimistic, augmented", "full_learned"),
                           ("Plug-in, augmented", "plugin_learned"), ("Clone alone", "bc_only"),
                           ("Candidate gain", "candidate_gain")):
            values = [s[key] if key == "candidate_gain" else s["arms"][key] for s in transfer]
            self.assertIn(label + " & " + " & ".join(map(cell, values)), self.tex)
            r = self.reversal["across_environments"]
            stats = r[key] if key == "candidate_gain" else r["arms"][key]
            self.assertIn(label + " & " + cell(stats), self.tex)

    def test_optimizer_records(self):
        data = load("results_optimizer_budget.json")
        self.assertEqual(data["source_sha256"], hashlib.sha256(
            (ROOT / "experiments/results_learned_bc.json").read_bytes()).hexdigest())
        self.assertEqual(len(data["records"]), 20)
        self.assertEqual({r["seed"] for r in data["records"]}, set(range(20)))
        maximum = 0.
        for row in data["records"]:
            original = next(r for r in self.original["records"] if r["seed"] == row["seed"] and r["N"] == 128000)
            self.assertEqual(len(row["scores"]), 7)
            for k, value in row["scores"].items():
                change = value - original["scores"][k]["full"]
                self.assertAlmostEqual(change, row["score_changes"][k], places=12)
                maximum = max(maximum, abs(change))
            self.assertFalse(row["changed"]["learned"])
        self.assertAlmostEqual(maximum, data["summary"]["max_abs_score_change"], places=14)
        self.assertLess(maximum, 1.1365e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
