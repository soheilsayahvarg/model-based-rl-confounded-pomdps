"""Fail-closed numeric provenance for workshop_draft.tex; not scientific certification.

Checks source macros, table entries, every raw regret and paired summary.
No SciPy required: new runs are fixed to n=20 (t_.975,19 below).
"""
import json
from pathlib import Path
import re
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T95_19 = 2.093024054408263


def load(name):
    def invalid(value):
        raise ValueError(f"Non-finite JSON literal: {value}")
    return json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"), parse_constant=invalid)


class ProvenanceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = (ROOT / "paper" / "workshop_draft.tex").read_text(encoding="utf-8")
        cls.macros = {k: float(v) for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([0-9.]+)\}", cls.tex)}
        cls.new = load("results_learned_bc.json")

    def test_legacy_numbers_in_current_source(self):
        d = load("results_regret_incomplete.json")
        rows = [r for r in d["rows"] if "kappa=1.5" in r["schedule"]]
        self.assertEqual(len(rows), 4)
        last = next(r for r in rows if r["N"] == 128000)
        self.assertAlmostEqual(self.macros["PessAtLarge"], last["regret"], delta=.00005)
        self.assertAlmostEqual(self.macros["PluginAtLarge"], last["plugin_regret"], delta=.00005)
        for r in rows:
            self.assertEqual(len(r["picks"]), 20)
            value = max(d["v_true"].values())
            for field, pick_field in (("regret", "picks"), ("plugin_regret", "plug_picks")):
                actual = np.mean([value - d["v_true"][k] for k in r[pick_field]])
                self.assertAlmostEqual(actual, r[field], places=12)
            values = [np.array([value - d["v_true"][k] for k in r[field]])
                      for field in ("picks", "plug_picks")]
            values.append(values[0] - values[1])
            cells = [f"${v.mean():.5f}\\pm{T95_19*v.std(ddof=1)/np.sqrt(20):.5f}$".replace("0.", ".") for v in values]
            self.assertIn(f"{r['N']:,} & " + " & ".join(cells), self.tex)
        oracle = load("results_bc_candidate_retest.json")
        mean = oracle["results"]["seven, with bc"]["regret"]["full"][-1][0]
        self.assertAlmostEqual(mean, self.macros["OracleBCAtLarge"], delta=.00005)
        margin = oracle["results"]["seven, with bc"]["v_true"]["bc"] - max(d["v_true"].values())
        self.assertAlmostEqual(margin, .000926, delta=.0000005)

    def test_complete_new_grid(self):
        d = self.new
        self.assertEqual(d["N_grid"], [2000, 8000, 32000, 128000])
        self.assertEqual(d["seeds"], list(range(20)))
        expected = {(n, s) for n in d["N_grid"] for s in d["seeds"]}
        observed = [(r["N"], r["seed"]) for r in d["records"]]
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(expected))

    def test_raw_regrets_and_summaries(self):
        for agg in self.new["summary"]:
            records = [r for r in self.new["records"] if r["N"] == agg["N"]]
            for r in records:
                best = max(r["truth"].values())
                for arm, pick in r["picks"].items():
                    score_type, pool_type = arm.split("_", 1)
                    pool = list(r["scores"])[:6] if pool_type == "six" else list(r["scores"])
                    self.assertEqual(pick, max(pool, key=lambda key: r["scores"][key][score_type]))
                    self.assertAlmostEqual(r["regrets"][arm], best - r["truth"][pick], places=12)
                self.assertAlmostEqual(r["regrets"]["bc_only"], best - r["truth"]["learned_bc"], places=12)
                pi = np.array(r["bc_policy"])
                np.testing.assert_allclose(pi.sum(axis=1), 1.)
                self.assertTrue((pi >= 0.).all())
            for arm, stats in agg["arms"].items():
                values = np.array([r["regrets"][arm] for r in records])
                self.assertAlmostEqual(stats["mean"], values.mean(), places=12)
                self.assertAlmostEqual(stats["ci95"], T95_19 * values.std(ddof=1) / np.sqrt(20), places=10)
            delta = np.array([r["regrets"]["full_learned"] - r["regrets"]["plugin_learned"] for r in records])
            stats = agg["paired_full_minus_plugin"]
            self.assertAlmostEqual(stats["mean"], delta.mean(), places=12)
            self.assertAlmostEqual(stats["ci95"], T95_19 * delta.std(ddof=1) / np.sqrt(20), places=10)

    def test_new_numbers_in_current_source(self):
        last = self.new["summary"][-1]
        self.assertAlmostEqual(self.macros["LearnedPessAtLarge"], last["arms"]["full_learned"]["mean"], delta=.00005)
        self.assertAlmostEqual(self.macros["LearnedPluginAtLarge"], last["arms"]["plugin_learned"]["mean"], delta=.000005)
        records = [r for r in self.new["records"] if r["N"] == 128000]
        gain = np.array([r["regrets"]["full_six"] - r["regrets"]["full_learned"] for r in records])
        self.assertAlmostEqual(self.macros["CandidateGainAtLarge"], gain.mean(), delta=.000005)
        self.assertAlmostEqual(self.macros["CandidateGainHalfInterval"],
                               T95_19 * gain.std(ddof=1) / np.sqrt(20), delta=.000005)
        self.assertEqual(sum(r["picks"]["full_learned"] == "uniform" for r in records), 4)
        self.assertEqual(sum(r["picks"]["full_learned"] == "learned_bc" for r in records), 16)
        gaps = [v for r in self.new["records"] for v in r["restart_gaps"].values()]
        self.assertEqual(len(gaps), 560)
        self.assertAlmostEqual(self.macros["MaximumRestartGap"], max(gaps), delta=.00000005)
        for row in self.new["summary"]:
            cells = [f"${row['arms'][arm]['mean']:.5f}\\pm{row['arms'][arm]['ci95']:.5f}$".replace("0.", ".")
                     for arm in ("full_learned", "plugin_learned", "bc_only")]
            self.assertIn(f"{row['N']:,} & " + " & ".join(cells), self.tex)


if __name__ == "__main__":
    unittest.main(verbosity=2)
