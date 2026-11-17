"""Tests for main.py: the one-command pipeline must run on every dataset and produce valid JSON."""

import contextlib
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as pipeline_main
from data_generator import all_edge_cases, generate_instance, small_instance
from algorithms.greedy import GreedyParams


def run_quiet(inst, exact=False, params=GreedyParams()):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = pipeline_main.run(inst, params, 1000, exact)
    return result, buf.getvalue()


class TestMainPipeline(unittest.TestCase):
    def test_runs_on_main_and_all_edge_cases_and_is_json_serialisable(self):
        for inst in [generate_instance()] + list(all_edge_cases().values()):
            result, out = run_quiet(inst)
            self.assertTrue(result["feasible"], inst.name)
            json.dumps(result)                                    # must not raise
            self.assertIn("Independent checker on the final plan: PASS", out)
            self.assertEqual(result["dataset"], inst.name)

    def test_main_dataset_headline_results(self):
        result, _ = run_quiet(generate_instance())
        self.assertEqual(result["metrics_final"]["W"], 26927)
        self.assertEqual(result["metrics_final"]["n_collected"], 13)
        self.assertAlmostEqual(result["metrics_final"]["distance_km"], 173.09, places=2)
        self.assertEqual(result["classification"]["P16"]["status"], "unreachable")
        self.assertEqual(set(result["uncollected"]), {"P01", "P07"})
        self.assertEqual(set(result["uncollected"].values()), {"capacity"})
        self.assertIsNone(result["exact"])

    def test_exact_flag_on_a_small_instance(self):
        result, out = run_quiet(small_instance(seed=1), exact=True)
        ex = result["exact"]
        self.assertEqual(ex["W"], 8709)
        self.assertGreaterEqual(ex["W"], result["metrics_final"]["W"])
        self.assertIn("Exact optimum", out)
        self.assertIsInstance(ex["optimal"], bool)

    def test_exact_flag_is_skipped_politely_on_a_large_instance(self):
        result, out = run_quiet(generate_instance(), exact=True)
        self.assertIsNone(result["exact"])
        self.assertIn("Exact comparison skipped", out)

    def test_parameters_change_the_plan(self):
        a, _ = run_quiet(generate_instance(), params=GreedyParams(1.0, 0.5))
        b, _ = run_quiet(generate_instance(), params=GreedyParams(0.0, 1.0))
        self.assertNotEqual(a["metrics_final"]["W"], b["metrics_final"]["W"])
        self.assertEqual((b["alpha"], b["beta"]), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
