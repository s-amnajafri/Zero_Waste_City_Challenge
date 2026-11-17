"""Tests for feasibility.py: the checker must ACCEPT valid plans and CATCH every violation type."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import _manual, edge_no_2opt_gain, edge_unreachable
from feasibility import check_plan
from algorithms.shortest_path import ShortestPaths


def two_point_instance(cap=(1000,), dur=(300,), q=(100, 100), dl=(300, 300), service=10):
    """P1 at (6,0), P2 at (0,6); roads only 0-1 and 0-2 (so P1->P2 goes via the depot, 12 km)."""
    pts = [(6, 0, q[0], 5, service, dl[0]), (0, 6, q[1], 5, service, dl[1])]
    return _manual("two_point", pts, [(0, 1), (0, 2)], cap, dur)


def codes(chk):
    return {v.split()[0] for v in chk.violations}


class TestCheckerAccepts(unittest.TestCase):
    def test_valid_plan_and_totals(self):
        inst = edge_no_2opt_gain()
        chk = check_plan(inst, ShortestPaths(inst), {1: [1, 2, 3, 4]})
        self.assertTrue(chk.ok, chk.violations)
        self.assertAlmostEqual(chk.distance, 16.0)          # 0 -> x=8 and back
        self.assertEqual(chk.waste_value, 5 * 100 * 4)      # urgency 5 x 100 kg x 4 points
        self.assertEqual(chk.routes[1].load, 400)

    def test_empty_plan_and_empty_routes_are_feasible(self):
        inst = edge_no_2opt_gain()
        sp = ShortestPaths(inst)
        self.assertTrue(check_plan(inst, sp, {}).ok)
        self.assertTrue(check_plan(inst, sp, {1: []}).ok)

    def test_times_are_recomputed_correctly(self):
        inst = two_point_instance()
        chk = check_plan(inst, ShortestPaths(inst), {1: [1]})
        rc = chk.routes[1]
        self.assertAlmostEqual(rc.arrivals[0], 12.0)         # 6 km at 30 km/h
        self.assertAlmostEqual(rc.completions[0], 22.0)      # + 10 min service
        self.assertAlmostEqual(rc.return_time, 34.0)         # + 12 min back


class TestCheckerCatchesViolations(unittest.TestCase):
    def test_c1_capacity(self):
        inst = two_point_instance(cap=(150,), q=(100, 100))
        chk = check_plan(inst, ShortestPaths(inst), {1: [1, 2]})
        self.assertIn("C1", codes(chk))
        self.assertFalse(chk.ok)

    def test_c2_deadline_uses_running_clock_not_direct_reachability(self):
        # Each point alone meets its deadline (completion 22 <= 30), but serving P1 then
        # P2 makes P2 complete at 22 + 24 + 10 = 56 > 30.
        inst = two_point_instance(dl=(30, 30))
        sp = ShortestPaths(inst)
        self.assertTrue(check_plan(inst, sp, {1: [1]}).ok)
        self.assertTrue(check_plan(inst, sp, {1: [2]}).ok)
        chk = check_plan(inst, sp, {1: [1, 2]})
        self.assertIn("C2", codes(chk))

    def test_c2_is_a_completion_deadline(self):
        # arrival 12 <= 20 but arrival + service = 22 > 20
        inst = two_point_instance(dl=(20, 300))
        self.assertIn("C2", codes(check_plan(inst, ShortestPaths(inst), {1: [1]})))
        inst2 = two_point_instance(dl=(22, 300))             # completion == deadline is allowed
        self.assertTrue(check_plan(inst2, ShortestPaths(inst2), {1: [1]}).ok)

    def test_c3_duration(self):
        inst = two_point_instance(dur=(60,))
        chk = check_plan(inst, ShortestPaths(inst), {1: [1, 2]})   # returns at 68 > 60
        self.assertIn("C3", codes(chk))
        inst2 = two_point_instance(dur=(34,))
        self.assertTrue(check_plan(inst2, ShortestPaths(inst2), {1: [1]}).ok)   # exactly 34

    def test_c4_unreachable_point(self):
        inst = edge_unreachable()
        chk = check_plan(inst, ShortestPaths(inst), {1: [6]})
        self.assertIn("C4", codes(chk))

    def test_c5_duplicates(self):
        inst = two_point_instance(cap=(1000, 1000), dur=(300, 300))
        sp = ShortestPaths(inst)
        self.assertIn("C5", codes(check_plan(inst, sp, {1: [1], 2: [1]})))      # two vehicles
        self.assertIn("C5", codes(check_plan(inst, sp, {1: [1, 1]})))           # same route twice

    def test_c5_invalid_point_id(self):
        inst = two_point_instance()
        sp = ShortestPaths(inst)
        self.assertIn("C5", codes(check_plan(inst, sp, {1: [0]})))              # depot is not a stop
        self.assertIn("C5", codes(check_plan(inst, sp, {1: [9]})))

    def test_c6_unknown_vehicle(self):
        inst = two_point_instance()
        self.assertIn("C6", codes(check_plan(inst, ShortestPaths(inst), {7: [1]})))


if __name__ == "__main__":
    unittest.main(verbosity=2)
