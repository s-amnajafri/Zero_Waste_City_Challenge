"""End-to-end edge-case tests: dataset -> Greedy -> 2-opt -> independent checker -> metrics.

Each test states the expected behaviour in plain words (these are the demonstrations for the report).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import (_manual, edge_deadline_violation, edge_no_2opt_gain, edge_over_demand,
                            edge_oversized, edge_unreachable)
from feasibility import check_plan
from algorithms.greedy import greedy_construct
from metrics import compute_metrics
from problem import ELIGIBLE, INFEASIBLE, UNREACHABLE
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import improve_plan


def pipeline(inst):
    sp = ShortestPaths(inst)
    g = greedy_construct(inst, sp)
    t = improve_plan(inst, sp, g)
    chk = check_plan(inst, sp, t.plan())
    assert chk.ok, chk.violations                     # every edge case must end in a feasible plan
    return sp, g, t, compute_metrics(inst, sp, t.plan(), inst.name)


class TestEdgeCases(unittest.TestCase):
    def test_unreachable_point_is_reported_and_everything_else_is_served(self):
        sp, g, t, m = pipeline(edge_unreachable())
        self.assertEqual(g.classification[6].status, UNREACHABLE)
        self.assertEqual(g.collected(), [1, 2, 3, 4, 5])
        self.assertEqual(g.uncollected, {})
        self.assertEqual((m.n_unreachable, m.kg_unreachable), (1, 200))
        self.assertAlmostEqual(m.coverage_pct, 100.0)     # unreachable waste is not "reachable but uncollected"

    def test_over_demand_gives_partial_coverage_not_failure(self):
        inst = edge_over_demand()                          # 1800 kg requested, 1500 kg of trucks
        sp, g, t, m = pipeline(inst)
        self.assertEqual((m.n_collected, m.kg_collected), (4, 1200))
        self.assertEqual(set(g.uncollected.values()), {"capacity"})
        self.assertEqual(m.kg_uncollected_eligible, 600)
        self.assertEqual(m.W, 8700)
        self.assertGreater(m.W, 0)

    def test_oversized_point_is_excluded_automatically(self):
        sp, g, t, m = pipeline(edge_oversized())
        self.assertEqual(g.classification[3].status, INFEASIBLE)
        self.assertEqual(g.classification[3].reasons, ["oversized"])
        self.assertEqual(g.collected(), [1, 2, 4, 5])
        self.assertEqual((m.n_infeasible, m.kg_infeasible), (1, 900))

    def test_point_with_impossible_deadline_is_excluded(self):
        sp, g, t, m = pipeline(edge_deadline_violation())
        self.assertEqual(g.classification[2].reasons, ["deadline"])
        self.assertNotIn(2, g.collected())
        self.assertEqual(m.n_infeasible, 1)
        self.assertAlmostEqual(t.total_before, 14.32)      # 2-opt then fixes the zig-zag
        self.assertAlmostEqual(t.total_after, 10.32)

    def test_route_with_no_improvement_is_left_alone(self):
        sp, g, t, m = pipeline(edge_no_2opt_gain())
        self.assertTrue(all(len(i.moves) == 0 for i in t.improvements.values()))
        self.assertTrue(t.all_converged)
        self.assertAlmostEqual(m.distance_km, 16.0)

    def test_no_reachable_points_at_all(self):
        inst = _manual("all_closed", [(3, 0, 100, 5, 5, 300), (0, 3, 100, 5, 5, 300)],
                       [(0, 1, False), (0, 2, False)], (500,), (300,))
        sp, g, t, m = pipeline(inst)
        self.assertEqual(g.plan(), {})
        self.assertEqual(g.d_max, 0.0)
        self.assertEqual((m.n_unreachable, m.n_collected, m.vehicles_used), (2, 0, 0))

    def test_every_reachable_point_individually_infeasible(self):
        inst = _manual("all_late", [(6, 0, 100, 5, 5, 1), (0, 6, 100, 5, 5, 1)],
                       [(0, 1), (0, 2)], (500,), (300,))
        sp, g, t, m = pipeline(inst)
        self.assertEqual(g.plan(), {})
        self.assertEqual({c.status for c in g.classification.values()}, {INFEASIBLE})
        self.assertEqual(m.n_infeasible, 2)

    def test_more_vehicles_than_points(self):
        inst = _manual("many_trucks", [(3, 0, 100, 5, 5, 300)], [(0, 1)], (500, 500, 500, 500), (300,) * 4)
        sp, g, t, m = pipeline(inst)
        self.assertEqual((m.n_collected, m.vehicles_used, m.n_vehicles), (1, 1, 4))

    def test_two_opt_blocked_by_deadline_end_to_end(self):
        # untangling would save distance but make P3 late; the plan must stay feasible
        pts = [(4, 0, 100, 5, 5, 300), (4, 4, 100, 5, 5, 300), (0, 4, 100, 5, 5, 35)]
        inst = _manual("blocked", pts, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], (1000,), (300,))
        sp = ShortestPaths(inst)
        g = greedy_construct(inst, sp)
        t = improve_plan(inst, sp, g)
        self.assertTrue(check_plan(inst, sp, t.plan()).ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
