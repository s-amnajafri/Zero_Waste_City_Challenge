"""Tests for greedy.py and problem.py (run from zero_waste/:  python -m unittest discover -s tests -v)"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import (_manual, all_edge_cases, edge_deadline_violation, edge_over_demand,
                            edge_oversized, edge_unreachable, generate_instance)
from feasibility import check_plan
from algorithms.greedy import GreedyParams, extra_distance, greedy_construct, priority_score
from problem import ELIGIBLE, INFEASIBLE, UNREACHABLE, classify_points, is_better, objective
from algorithms.shortest_path import ShortestPaths


def two_point_instance(cap=(1000,), dur=(300,), q=(100, 100), dl=(300, 300), service=10):
    """P1 at (6,0), P2 at (0,6); P1->P2 must go through the depot (12 km = 24 min)."""
    pts = [(6, 0, q[0], 5, service, dl[0]), (0, 6, q[1], 5, service, dl[1])]
    return _manual("two_point", pts, [(0, 1), (0, 2)], cap, dur)


def run(inst, **kw):
    sp = ShortestPaths(inst)
    return sp, greedy_construct(inst, sp, GreedyParams(**kw))


# --------------------------------------------------------------------------
class TestScore(unittest.TestCase):
    def test_formula(self):
        # 1.0 * 5/10 - 0.5 * 4/8 = 0.25
        self.assertAlmostEqual(priority_score(5, 4.0, 8.0), 0.25)
        self.assertAlmostEqual(priority_score(10, 0.0, 8.0, alpha=2.0), 2.0)

    def test_dmax_zero_safeguard(self):
        self.assertAlmostEqual(priority_score(8, 3.0, 0.0), 0.8)     # distance term dropped

    def test_extra_distance(self):
        inst = generate_instance()
        sp = ShortestPaths(inst)
        for j in sp.reachable_points():
            self.assertAlmostEqual(extra_distance(sp, 0, j.id), 2 * sp.dist(0, j.id))
        reach = [p.id for p in sp.reachable_points()]
        for i in reach:
            for j in reach:
                if i != j:
                    self.assertGreaterEqual(extra_distance(sp, i, j), 0.0)


class TestClassification(unittest.TestCase):
    def test_main_dataset(self):
        inst = generate_instance()
        cls = classify_points(inst, ShortestPaths(inst))
        self.assertEqual([j for j, c in cls.items() if c.status == UNREACHABLE], [16])
        self.assertEqual(sum(c.status == ELIGIBLE for c in cls.values()), 15)

    def test_edge_datasets(self):
        sp = lambda i: (i, ShortestPaths(i))
        i, s = sp(edge_oversized())
        self.assertEqual(classify_points(i, s)[3].reasons, ["oversized"])
        i, s = sp(edge_deadline_violation())
        self.assertEqual(classify_points(i, s)[2].reasons, ["deadline"])
        i, s = sp(edge_unreachable())
        self.assertEqual(classify_points(i, s)[6].status, UNREACHABLE)

    def test_completion_deadline_boundary(self):
        # P1: arrival 12, service 10 -> completion 22
        for dl, expected in ((20, INFEASIBLE), (22, ELIGIBLE)):
            inst = two_point_instance(dl=(dl, 300))
            self.assertEqual(classify_points(inst, ShortestPaths(inst))[1].status, expected)

    def test_duration_boundary(self):
        # round trip 12 + 10 + 12 = 34
        for dur, expected in ((33, INFEASIBLE), (34, ELIGIBLE)):
            inst = two_point_instance(dur=(dur,))
            self.assertEqual(classify_points(inst, ShortestPaths(inst))[1].status, expected)

    def test_no_single_vehicle_meets_all_limits(self):
        # truck A fits the 500 kg but cannot finish in time; truck B is fast enough but too small
        inst = _manual("joint", [(6, 0, 500, 5, 10, 300)], [(0, 1)], (1000, 300), (20, 300))
        c = classify_points(inst, ShortestPaths(inst))[1]
        self.assertEqual((c.status, c.reasons), (INFEASIBLE, ["no_vehicle_meets_all_limits"]))


class TestFeasibilityOfOutput(unittest.TestCase):
    def _assert_valid(self, inst):
        sp, res = run(inst)
        chk = check_plan(inst, sp, res.plan())
        self.assertTrue(chk.ok, f"{inst.name}: {chk.violations}")
        W, D = objective(inst, sp, res.plan())
        self.assertEqual(chk.waste_value, W)
        self.assertAlmostEqual(chk.distance, D, places=6)
        self.assertAlmostEqual(chk.distance, sum(r.distance for r in res.routes.values()), places=6)
        for vid, rc in chk.routes.items():
            self.assertAlmostEqual(rc.return_time, res.routes[vid].return_time, places=6)
        # outcome partition: every eligible point is either collected or uncollected, never both
        eligible = {j for j, c in res.classification.items() if c.status == ELIGIBLE}
        self.assertEqual(set(res.collected()) | set(res.uncollected), eligible)
        self.assertFalse(set(res.collected()) & set(res.uncollected))
        self.assertLessEqual(res.evaluations, len(eligible) * (len(eligible) + 1) // 2 * len(inst.vehicles))

    def test_main_and_edge_cases(self):
        for inst in [generate_instance()] + list(all_edge_cases().values()):
            self._assert_valid(inst)

    def test_random_instances(self):
        for seed in range(1, 31):
            n, iso = 8 + seed % 15, seed % 3
            inst = generate_instance(n_points=n, n_isolated=iso, usable_roads=int(1.8 * (n - iso)),
                                     demand_ratio=0.5 + 0.1 * (seed % 12), seed=seed, name=f"rand{seed}")
            self._assert_valid(inst)

    def test_deterministic(self):
        inst = generate_instance()
        a, b = run(inst)[1], run(inst)[1]
        self.assertEqual(a.plan(), b.plan())
        self.assertEqual(a.log, b.log)


class TestBehaviour(unittest.TestCase):
    def test_unreachable_oversized_and_deadline_points_never_served(self):
        for inst, bad in ((edge_unreachable(), 6), (edge_oversized(), 3), (edge_deadline_violation(), 2)):
            _, res = run(inst)
            self.assertNotIn(bad, res.collected())
            self.assertNotIn(bad, res.uncollected)          # excluded before Greedy, not "uncollected"

    def test_over_demand_gives_partial_coverage_in_urgency_order(self):
        inst = edge_over_demand()
        _, res = run(inst)
        got = res.collected()
        self.assertEqual(len(got), 4)                       # 1200 kg fit, 1800 kg requested
        self.assertEqual(set(res.uncollected.values()), {"capacity"})
        urg = lambda j: inst.point(j).urgency
        self.assertLess(max(urg(j) for j in res.uncollected), min(urg(j) for j in got))
        self.assertGreater(sum(inst.point(j).waste for j in got), 0)

    def test_negative_scores_are_still_selected(self):
        inst = _manual("neg", [(10, 0, 100, 1, 5, 300)], [(0, 1)], (500,), (300,))
        _, res = run(inst)
        self.assertEqual(res.collected(), [1])
        self.assertLess(res.log[0][2], 0)                   # score = 0.1 - 0.5*2 = -0.9

    def test_dmax_zero_instance_runs(self):
        inst = _manual("dmax0", [(0, 0, 100, 3, 5, 300), (0, 0, 100, 8, 5, 300)],
                       [(0, 1), (0, 2)], (500,), (300,))
        _, res = run(inst)
        self.assertEqual(res.d_max, 0.0)
        self.assertEqual(res.plan(), {1: [2, 1]})           # higher urgency first

    # ---- tie-breaking -----------------------------------------------------
    def test_tie_smaller_point_id(self):
        inst = _manual("tie_id", [(3, 0, 100, 5, 5, 300), (0, 3, 100, 5, 5, 300)],
                       [(0, 1), (0, 2)], (500,), (300,))
        self.assertEqual(run(inst)[1].plan()[1][0], 1)

    def test_tie_higher_urgency_beats_smaller_id(self):
        # P1 (d=4, u=5) and P2 (d=5, u=7) have the same score (-0.3) with alpha=1, beta=0.5
        inst = _manual("tie_urg", [(4, 0, 100, 5, 5, 300), (0, 5, 100, 7, 5, 300)],
                       [(0, 1), (0, 2)], (500,), (300,))
        # confirm the premise: with D_max = 5 both candidates score exactly -0.3
        self.assertAlmostEqual(priority_score(5, 8.0, 5.0), priority_score(7, 10.0, 5.0))
        res = run(inst)[1]
        self.assertEqual(res.plan()[1][0], 2)

    def test_tie_smaller_additional_distance_when_beta_zero(self):
        # beta = 0 -> scores equal for equal urgency; the nearer point (id 2) must go first
        inst = _manual("tie_dist", [(0, 6, 100, 5, 5, 300), (3, 0, 100, 5, 5, 300)],
                       [(0, 1), (0, 2)], (500,), (300,))
        self.assertEqual(run(inst, alpha=1.0, beta=0.0)[1].plan()[1][0], 2)

    def test_vehicle_best_fit(self):
        inst = _manual("best_fit", [(3, 0, 400, 5, 5, 300)], [(0, 1)], (1000, 500), (300, 300))
        _, res = run(inst)
        self.assertEqual(res.plan(), {2: [1]})              # leftover 100 kg beats leftover 600 kg

    def test_vehicle_tie_falls_back_to_smaller_id(self):
        inst = _manual("veh_tie", [(3, 0, 100, 5, 5, 300)], [(0, 1)], (500, 500), (300, 300))
        self.assertEqual(run(inst)[1].plan(), {1: [1]})

    def test_selection_is_global_not_sequential(self):
        # Sequential construction would never switch back to an earlier vehicle.
        _, res = run(generate_instance())
        seq = [v for v, *_ in res.log]
        switches_back = sum(1 for a, b in zip(seq, seq[1:]) if b != a)
        self.assertGreater(switches_back, len(set(seq)) - 1)

    # ---- reasons for uncollected points ----------------------------------
    def test_reason_capacity(self):
        _, res = run(two_point_instance(cap=(150,)))
        self.assertEqual(res.uncollected, {2: "capacity"})

    def test_reason_deadline(self):
        # P2 alone is fine (completion 22 <= 30) but after P1 it would complete at 56
        _, res = run(two_point_instance(dl=(30, 30)))
        self.assertEqual(res.collected(), [1])
        self.assertEqual(res.uncollected, {2: "deadline"})

    def test_reason_route_duration(self):
        _, res = run(two_point_instance(dur=(60,)))        # P2 would end at 56 but return at 68
        self.assertEqual(res.uncollected, {2: "route_duration"})

    def test_second_vehicle_rescues_blocked_point(self):
        _, res = run(two_point_instance(cap=(1000, 1000), dur=(300, 300), dl=(30, 30)))
        self.assertEqual(res.uncollected, {})
        self.assertEqual(len(res.plan()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
