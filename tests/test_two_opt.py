"""Tests for two_opt.py (run from zero_waste/:  python -m unittest discover -s tests -v)"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import _manual, all_edge_cases, edge_no_2opt_gain, generate_instance
from feasibility import check_plan
from algorithms.greedy import greedy_construct
from problem import EPS, objective
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import improve_plan, two_opt_route


def square_instance(dl3=300.0, dur=300.0):
    """Depot (0,0); P1 (4,0), P2 (4,4), P3 (0,4); complete road graph (distances are Euclidean).
    Optimal tour 0-1-2-3-0 = 16 km.  The order [1,3,2] has crossing edges (19.32 km)."""
    pts = [(4, 0, 100, 5, 5, 300), (4, 4, 100, 5, 5, 300), (0, 4, 100, 5, 5, dl3)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return _manual("square", pts, edges, (1000,), (dur,))


def seeded(seed=7, n=17):
    return generate_instance(n_points=n, n_isolated=0, usable_roads=int(2.2 * n),
                             demand_ratio=0.8, seed=seed, name=f"r{seed}")


def all_cases():
    cases = [generate_instance()] + list(all_edge_cases().values())
    for seed in range(1, 31):
        n, iso = 8 + seed % 15, seed % 3
        cases.append(generate_instance(n_points=n, n_isolated=iso, usable_roads=int(1.8 * (n - iso)),
                                       demand_ratio=0.5 + 0.1 * (seed % 12), seed=seed, name=f"rand{seed}"))
    return cases


class TestCorrectMoves(unittest.TestCase):
    def test_crossing_route_is_untangled_to_the_optimum(self):
        inst = square_instance()
        sp = ShortestPaths(inst)
        new, imp = two_opt_route(inst, sp, inst.vehicles[0], [1, 3, 2])
        self.assertAlmostEqual(imp.before_km, 4 + 5.66 + 4 + 5.66)   # roads are stored rounded to 0.01 km
        self.assertEqual(new, [1, 2, 3])
        self.assertAlmostEqual(imp.after_km, 16.0)
        self.assertEqual(len(imp.moves), 1)
        self.assertTrue(imp.converged)

    def test_no_gain_route_is_left_alone(self):
        inst = edge_no_2opt_gain()
        sp = ShortestPaths(inst)
        new, imp = two_opt_route(inst, sp, inst.vehicles[0], [1, 2, 3, 4])
        self.assertEqual(new, [1, 2, 3, 4])
        self.assertEqual(imp.moves, [])
        self.assertTrue(imp.converged)

    def test_tiny_routes(self):
        inst = square_instance()
        sp = ShortestPaths(inst)
        for stops in ([], [1], [1, 2]):
            new, imp = two_opt_route(inst, sp, inst.vehicles[0], stops)
            self.assertEqual(new, stops)
            self.assertEqual(imp.moves, [])
            self.assertTrue(imp.converged)

    def test_infeasible_input_is_rejected(self):
        inst = square_instance(dl3=5.0)                    # P3 can never meet a 5-minute deadline
        with self.assertRaises(ValueError):
            two_opt_route(inst, ShortestPaths(inst), inst.vehicles[0], [1, 3, 2])


class TestFeasibilityIsPreserved(unittest.TestCase):
    def test_improving_move_blocked_by_deadline(self):
        # Untangling [1,3,2] -> [1,2,3] saves 3.31 km but P3 would then complete at 39 min.
        # With deadline 35 the move must be rejected (scrambled order completes P3 at 29.3).
        inst = square_instance(dl3=35.0)
        sp = ShortestPaths(inst)
        new, imp = two_opt_route(inst, sp, inst.vehicles[0], [1, 3, 2])
        self.assertEqual(new, [1, 3, 2])
        self.assertEqual(imp.moves, [])
        self.assertEqual(imp.rejected_deadline, 1)
        self.assertTrue(imp.converged)
        # the same move is accepted once the deadline allows it
        inst2 = square_instance(dl3=45.0)
        new2, _ = two_opt_route(inst2, ShortestPaths(inst2), inst2.vehicles[0], [1, 3, 2])
        self.assertEqual(new2, [1, 2, 3])

    def test_duration_can_never_block_an_improving_move(self):
        # return time = travel + service; an improving move lowers travel and keeps service.
        for inst in all_cases():
            sp = ShortestPaths(inst)
            g = greedy_construct(inst, sp)
            res = improve_plan(inst, sp, g)
            for vid, imp in res.improvements.items():
                self.assertEqual(imp.rejected_duration, 0)
                self.assertLessEqual(res.routes[vid].return_time, g.routes[vid].return_time + EPS)

    def test_outputs_pass_the_independent_checker(self):
        for inst in all_cases():
            sp = ShortestPaths(inst)
            g = greedy_construct(inst, sp)
            res = improve_plan(inst, sp, g)
            chk = check_plan(inst, sp, res.plan())
            self.assertTrue(chk.ok, f"{inst.name}: {chk.violations}")
            self.assertAlmostEqual(chk.distance, res.total_after + sum(
                r.distance for vid, r in res.routes.items() if vid not in res.improvements), places=6)

    def test_served_set_load_and_W_are_unchanged(self):
        for inst in all_cases():
            sp = ShortestPaths(inst)
            g = greedy_construct(inst, sp)
            res = improve_plan(inst, sp, g)
            self.assertEqual(res.plan().keys(), g.plan().keys())
            for vid in g.plan():
                self.assertEqual(sorted(res.plan()[vid]), sorted(g.plan()[vid]))
                self.assertEqual(res.routes[vid].load, g.routes[vid].load)
            self.assertEqual(objective(inst, sp, res.plan())[0], objective(inst, sp, g.plan())[0])

    def test_distance_never_increases(self):
        for inst in all_cases():
            sp = ShortestPaths(inst)
            g = greedy_construct(inst, sp)
            res = improve_plan(inst, sp, g)
            self.assertLessEqual(res.total_after, res.total_before + EPS)
            for vid, imp in res.improvements.items():
                self.assertLessEqual(imp.after_km, imp.before_km + EPS)
            saved = sum(i.saved_km for i in res.improvements.values())
            self.assertAlmostEqual(res.total_before - res.total_after, saved, places=6)


class TestTwoOptimality(unittest.TestCase):
    def test_no_feasible_improving_reversal_remains(self):
        """Exhaustively try every reversal on every output route; any that shortens the route
        must be infeasible according to the independent checker."""
        for inst in all_cases():
            sp = ShortestPaths(inst)
            g = greedy_construct(inst, sp)
            res = improve_plan(inst, sp, g)
            plan = res.plan()
            for vid, stops in plan.items():
                base = sp.route_distance([0] + stops + [0])
                for a in range(len(stops)):
                    for b in range(a + 1, len(stops)):
                        alt = stops[:a] + stops[a:b + 1][::-1] + stops[b + 1:]
                        if sp.route_distance([0] + alt + [0]) < base - 1e-7:
                            trial = dict(plan)
                            trial[vid] = alt
                            self.assertFalse(check_plan(inst, sp, trial).ok,
                                             f"{inst.name} V{vid}: feasible improving move left over")

    def test_idempotent(self):
        for inst in all_cases():
            sp = ShortestPaths(inst)
            res = improve_plan(inst, sp, greedy_construct(inst, sp))
            again = improve_plan(inst, sp, res_as_greedy(res))
            self.assertEqual(again.plan(), res.plan())
            self.assertTrue(all(len(i.moves) == 0 for i in again.improvements.values()))

    def test_deterministic(self):
        inst = seeded()
        sp = ShortestPaths(inst)
        a = improve_plan(inst, sp, greedy_construct(inst, sp))
        b = improve_plan(inst, sp, greedy_construct(inst, sp))
        self.assertEqual(a.plan(), b.plan())
        self.assertEqual({v: i.moves for v, i in a.improvements.items()},
                         {v: i.moves for v, i in b.improvements.items()})


class TestIterationCap(unittest.TestCase):
    def setUp(self):
        self.inst = seeded(7, 17)
        self.sp = ShortestPaths(self.inst)
        g = greedy_construct(self.inst, self.sp)
        full = improve_plan(self.inst, self.sp, g)
        self.vid = max(full.improvements, key=lambda v: len(full.improvements[v].moves))
        self.needed = len(full.improvements[self.vid].moves)
        self.stops = g.routes[self.vid].stops
        self.veh = next(v for v in self.inst.vehicles if v.id == self.vid)

    def test_premise_route_needs_several_moves(self):
        self.assertGreaterEqual(self.needed, 2)

    def test_cap_stops_the_search_and_is_reported(self):
        _, imp = two_opt_route(self.inst, self.sp, self.veh, self.stops, max_iter=1)
        self.assertEqual(len(imp.moves), 1)
        self.assertFalse(imp.converged)

    def test_cap_equal_to_needed_moves_still_reports_convergence(self):
        _, imp = two_opt_route(self.inst, self.sp, self.veh, self.stops, max_iter=self.needed)
        self.assertEqual(len(imp.moves), self.needed)
        self.assertTrue(imp.converged)

    def test_capped_result_is_still_feasible_and_not_worse(self):
        new, imp = two_opt_route(self.inst, self.sp, self.veh, self.stops, max_iter=1)
        self.assertLess(imp.after_km, imp.before_km)
        plan = {self.vid: new}
        self.assertTrue(check_plan(self.inst, self.sp, plan).ok)


def res_as_greedy(res):
    """Wrap a TwoOptResult so improve_plan can run on it again (it only needs .routes)."""
    class _Wrapper:
        routes = res.routes
    return _Wrapper()


if __name__ == "__main__":
    unittest.main(verbosity=2)
