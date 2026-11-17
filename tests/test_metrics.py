"""Tests for metrics.py (run from zero_waste/:  python -m unittest discover -s tests -v)"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import _manual, all_edge_cases, edge_no_2opt_gain, edge_unreachable, generate_instance
from algorithms.greedy import greedy_construct
from metrics import compute_metrics, format_metrics
from algorithms.shortest_path import ShortestPaths


class TestKnownValues(unittest.TestCase):
    def test_hand_checked_plan(self):
        inst = edge_no_2opt_gain()                      # 4 points x 100 kg, urgency 5, 1 truck of 1000 kg
        sp = ShortestPaths(inst)
        m = compute_metrics(inst, sp, {1: [1, 2, 3, 4]}, "hand")
        self.assertTrue(m.feasible)
        self.assertEqual((m.n_collected, m.n_reachable, m.kg_collected), (4, 4, 400))
        self.assertEqual(m.W, 2000)
        self.assertAlmostEqual(m.W_pct, 100.0)
        self.assertAlmostEqual(m.distance_km, 16.0)
        self.assertAlmostEqual(m.km_per_100kg, 4.0)
        self.assertAlmostEqual(m.coverage_pct, 100.0)
        self.assertAlmostEqual(m.fleet_utilization_pct, 40.0)
        self.assertAlmostEqual(m.used_utilization_pct, 40.0)
        self.assertEqual((m.vehicles_used, m.n_vehicles), (1, 1))
        self.assertIsNone(m.urgent_coverage_pct)        # no urgent (>= 8) points exist
        self.assertAlmostEqual(m.deadline_compliance_pct, 100.0)
        self.assertAlmostEqual(m.min_slack_min, 300 - 36.0)   # last stop completes at 36 min

    def test_empty_plan(self):
        inst = edge_no_2opt_gain()
        m = compute_metrics(inst, ShortestPaths(inst), {}, "empty")
        self.assertTrue(m.feasible)
        self.assertEqual((m.n_collected, m.kg_collected, m.W, m.vehicles_used), (0, 0, 0, 0))
        self.assertEqual(m.coverage_pct, 0.0)
        self.assertEqual(m.fleet_utilization_pct, 0.0)
        self.assertIsNone(m.km_per_100kg)
        self.assertIsNone(m.min_slack_min)
        self.assertIsNone(m.deadline_compliance_pct)

    def test_infeasible_plan_is_flagged(self):
        inst = _manual("tiny", [(2, 0, 400, 5, 5, 300), (4, 0, 400, 5, 5, 300)],
                       [(0, 1), (1, 2)], (500,), (300,))
        m = compute_metrics(inst, ShortestPaths(inst), {1: [1, 2]}, "too heavy")   # 800 kg on a 500 kg truck
        self.assertFalse(m.feasible)

    def test_urgent_coverage(self):
        # two urgent points (urgency 9), only one fits the truck
        inst = _manual("urgent", [(2, 0, 400, 9, 5, 300), (0, 2, 400, 9, 5, 300)],
                       [(0, 1), (0, 2)], (500,), (300,))
        m = compute_metrics(inst, ShortestPaths(inst), {1: [1]}, "one of two")
        self.assertEqual((m.urgent_eligible, m.urgent_collected), (2, 1))
        self.assertAlmostEqual(m.urgent_coverage_pct, 50.0)


class TestAccounting(unittest.TestCase):
    def test_waste_categories_partition_total_demand(self):
        cases = [generate_instance()] + list(all_edge_cases().values())
        for seed in range(1, 21):
            n, iso = 8 + seed % 15, seed % 3
            cases.append(generate_instance(n_points=n, n_isolated=iso, usable_roads=int(1.8 * (n - iso)),
                                           demand_ratio=0.5 + 0.1 * (seed % 12), seed=seed, name=f"r{seed}"))
        for inst in cases:
            sp = ShortestPaths(inst)
            g = greedy_construct(inst, sp)
            m = compute_metrics(inst, sp, g.plan(), inst.name)
            self.assertEqual(m.kg_unreachable + m.kg_infeasible + m.kg_uncollected_eligible + m.kg_collected,
                             m.kg_total, inst.name)
            self.assertEqual(m.n_unreachable + m.n_infeasible + m.n_uncollected_eligible + m.n_collected,
                             m.n_points, inst.name)
            self.assertLessEqual(m.W_pct, 100.0 + 1e-9)
            self.assertEqual(m.n_collected, len(g.collected()))

    def test_main_dataset_headline_numbers(self):
        inst = generate_instance()
        sp = ShortestPaths(inst)
        g = greedy_construct(inst, sp)
        m = compute_metrics(inst, sp, g.plan(), "main")
        self.assertEqual((m.n_collected, m.n_reachable, m.kg_collected), (13, 15, 3986))
        self.assertEqual((m.kg_unreachable, m.kg_uncollected_eligible), (382, 873))
        self.assertEqual(m.W, 26927)
        self.assertEqual((m.urgent_collected, m.urgent_eligible), (6, 6))
        self.assertAlmostEqual(m.distance_km, sum(r.distance for r in g.routes.values()), places=6)
        self.assertAlmostEqual(m.deadline_compliance_pct, 100.0)

    def test_everything_collected_gives_full_w(self):
        inst = edge_unreachable()
        sp = ShortestPaths(inst)
        m = compute_metrics(inst, sp, greedy_construct(inst, sp).plan(), "x")
        self.assertAlmostEqual(m.W_pct, 100.0)
        self.assertAlmostEqual(m.coverage_pct, 100.0)        # unreachable P06 is not in the denominator

    def test_report_text_contains_key_lines(self):
        inst = generate_instance()
        sp = ShortestPaths(inst)
        txt = format_metrics(compute_metrics(inst, sp, greedy_construct(inst, sp).plan(), "main", 1.23))
        for needle in ("Distance", "Coverage", "13/15", "unreachable", "Urgent coverage", "Execution time"):
            self.assertIn(needle, txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
