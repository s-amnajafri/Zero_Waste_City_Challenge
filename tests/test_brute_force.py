"""Tests for brute_force.py and optimality_gap.py (run from zero_waste/:  python -m unittest discover -s tests -v)"""

import itertools
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.baselines import compare_methods
from experiments.brute_force import solve_exact
from data_generator import (_manual, edge_deadline_violation, edge_no_2opt_gain, edge_over_demand,
                            edge_oversized, edge_unreachable, generate_instance)
from feasibility import check_plan
from algorithms.greedy import greedy_construct
from experiments.optimality_gap import gap_instance, gap_rows, greedy_counterexample
from problem import EPS, classify_points, is_better, objective
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import improve_plan


def naive_optimum(inst, sp):
    """Fully independent oracle: try EVERY assignment of points to {no vehicle, v1, v2, ...} and, per
    vehicle, EVERY ordering, judging feasibility only with the independent checker."""
    vs = sorted(inst.vehicles, key=lambda v: v.id)
    pts = [p.id for p in inst.points]
    best = (0, 0.0)
    for assign in itertools.product(range(len(vs) + 1), repeat=len(pts)):
        groups = {vi: [] for vi in range(len(vs))}
        for pid, a in zip(pts, assign):
            if a < len(vs):
                groups[a].append(pid)
        tot_w, tot_d, ok = 0, 0.0, True
        for vi, g in groups.items():
            if not g:
                continue
            shortest = None
            for perm in itertools.permutations(g):
                chk = check_plan(inst, sp, {vs[vi].id: list(perm)})
                if chk.ok and (shortest is None or chk.distance < shortest):
                    shortest = chk.distance
            if shortest is None:
                ok = False
                break
            tot_d += shortest
            tot_w += sum(inst.point(j).urgency * inst.point(j).waste for j in g)
        if ok and is_better((tot_w, tot_d), best):
            best = (tot_w, tot_d)
    return best


def tiny(seed, n=5, caps=(500, 700), ratio=1.2, dur=200.0):
    return generate_instance(n_points=n, n_isolated=0, usable_roads=2 * n, vehicle_capacities=caps,
                             vehicle_durations=(dur,) * len(caps), demand_ratio=ratio, seed=seed, name=f"t{seed}")


class TestExactnessAgainstIndependentOracle(unittest.TestCase):
    def test_matches_naive_full_enumeration_on_random_tiny_instances(self):
        for seed in range(1, 9):
            inst = tiny(seed)
            sp = ShortestPaths(inst)
            res = solve_exact(inst, sp)
            w, d = naive_optimum(inst, sp)
            self.assertEqual(res.W, w, f"seed {seed}")
            self.assertAlmostEqual(res.D, d, places=6, msg=f"seed {seed}")

    def test_matches_naive_with_three_vehicles(self):
        for seed in (1, 2, 3):
            inst = tiny(seed, n=4, caps=(400, 500, 600), ratio=1.4)
            sp = ShortestPaths(inst)
            res = solve_exact(inst, sp)
            w, d = naive_optimum(inst, sp)
            self.assertEqual((res.W, round(res.D, 6)), (w, round(d, 6)))

    def test_matches_naive_on_edge_case_datasets(self):
        for inst in (edge_no_2opt_gain(), edge_deadline_violation(), edge_oversized()):
            sp = ShortestPaths(inst)
            res = solve_exact(inst, sp)
            w, d = naive_optimum(inst, sp)
            self.assertEqual(res.W, w, inst.name)
            self.assertAlmostEqual(res.D, d, places=6, msg=inst.name)

    def test_pruned_equals_unpruned(self):
        for inst in [tiny(s) for s in range(1, 6)] + [edge_unreachable(), edge_oversized()]:
            sp = ShortestPaths(inst)
            a, b = solve_exact(inst, sp, prune=True), solve_exact(inst, sp, prune=False)
            self.assertEqual(a.W, b.W, inst.name)
            self.assertAlmostEqual(a.D, b.D, places=6, msg=inst.name)

    def test_restricting_to_eligible_points_loses_nothing(self):
        for inst in (edge_unreachable(), edge_oversized(), edge_deadline_violation()):
            sp = ShortestPaths(inst)
            a = solve_exact(inst, sp)
            b = solve_exact(inst, sp, restrict_to_eligible=False)
            self.assertEqual(a.W, b.W, inst.name)
            self.assertAlmostEqual(a.D, b.D, places=6, msg=inst.name)


class TestExactPlanProperties(unittest.TestCase):
    def test_plan_is_feasible_and_values_match_the_checker(self):
        for seed in range(1, 21):
            inst = gap_instance(seed, demand_ratio=1.25)
            sp = ShortestPaths(inst)
            res = solve_exact(inst, sp)
            chk = check_plan(inst, sp, res.plan)
            self.assertTrue(chk.ok, f"seed {seed}: {chk.violations}")
            self.assertEqual(chk.waste_value, res.W)
            self.assertAlmostEqual(chk.distance, res.D, places=6)

    def test_never_beaten_by_any_heuristic(self):
        for ratio in (0.9, 1.25, 1.6):
            for seed in range(1, 21):
                inst = gap_instance(seed, demand_ratio=ratio)
                sp = ShortestPaths(inst)
                ex = solve_exact(inst, sp)
                for m in compare_methods(inst, sp):
                    self.assertFalse(is_better((m.W, m.distance_km), (ex.W, ex.D)),
                                     f"{m.label} beat the exact optimum on {inst.name}")

    def test_prefers_fewer_longer_trips_when_that_is_shorter(self):
        # both points fit one truck: one route (8 km) beats two routes (6 + 8 km)
        inst = _manual("one_route", [(3, 0, 100, 5, 5, 300), (4, 0, 100, 5, 5, 300)],
                       [(0, 1), (1, 2)], (500, 500), (300, 300))
        res = solve_exact(inst)
        self.assertEqual(len(res.plan), 1)
        self.assertAlmostEqual(res.D, 8.0)

    def test_lexicographic_value_beats_distance(self):
        # serving the far urgent point costs distance but W is the first objective
        inst = _manual("lex", [(2, 0, 500, 2, 5, 300), (9, 0, 500, 9, 5, 300)],
                       [(0, 1), (0, 2)], (600,), (300,))
        res = solve_exact(inst)
        self.assertEqual(sorted(j for s in res.plan.values() for j in s), [2])   # urgency 9 x 500 > urgency 2 x 500
        self.assertEqual(res.W, 4500)

    def test_over_demand_and_oversized_cases(self):
        inst = edge_over_demand()
        sp = ShortestPaths(inst)
        ex, g = solve_exact(inst, sp), greedy_construct(inst, sp)
        self.assertFalse(is_better(objective(inst, sp, g.plan()), (ex.W, ex.D)))
        self.assertEqual(ex.W, 8700)
        inst2 = edge_oversized()
        ex2 = solve_exact(inst2)
        self.assertNotIn(3, [j for s in ex2.plan.values() for j in s])

    def test_no_eligible_points_gives_empty_plan(self):
        inst = _manual("none", [(3, 0, 100, 5, 5, 300)], [(0, 1, False)], (500,), (300,))
        res = solve_exact(inst)
        self.assertEqual((res.plan, res.W, res.D, res.n_candidates), ({}, 0, 0.0, 0))

    def test_deterministic(self):
        inst = gap_instance(4)
        self.assertEqual(solve_exact(inst).plan, solve_exact(inst).plan)

    def test_size_limit_is_enforced(self):
        with self.assertRaises(ValueError) as cm:
            solve_exact(generate_instance())                  # 15 candidate points
        self.assertIn("limited to 8", str(cm.exception))
        self.assertEqual(solve_exact(gap_instance(1, n_points=9), max_points=9).n_candidates <= 9, True)


class TestComplexityDerivation(unittest.TestCase):
    """With no binding constraint every ordering is feasible, so the number of enumerated orderings
    must equal sum_{s=1..n} n!/(n-s)!  per vehicle (~ e * n!)."""

    def test_ordering_count_matches_the_formula(self):
        for n in (4, 5, 6):
            for k in (1, 3):
                inst = generate_instance(n_points=n, n_isolated=0, usable_roads=2 * n,
                                         vehicle_capacities=(10 ** 6,) * k, vehicle_durations=(10 ** 6,) * k,
                                         demand_ratio=0.1, seed=3, name="free")
                for p in inst.points:
                    p.deadline = 10 ** 6
                expected = k * sum(math.factorial(n) // math.factorial(n - s) for s in range(1, n + 1))
                self.assertEqual(solve_exact(inst).sequences_explored, expected, f"n={n}, k={k}")


class TestGreedyCounterexampleAndGap(unittest.TestCase):
    def test_counterexample_numbers(self):
        inst = greedy_counterexample()
        sp = ShortestPaths(inst)
        g = greedy_construct(inst, sp)
        ex = solve_exact(inst, sp)
        self.assertEqual(g.collected(), [1])
        self.assertEqual(objective(inst, sp, g.plan())[0], 5400)
        self.assertEqual(sorted(j for s in ex.plan.values() for j in s), [2, 3])
        self.assertEqual(ex.W, 8000)
        self.assertTrue(is_better((ex.W, ex.D), objective(inst, sp, g.plan())))
        # 2-opt cannot repair it: it never changes the served set
        t = improve_plan(inst, sp, g)
        self.assertEqual(objective(inst, sp, t.plan())[0], 5400)

    def test_gap_rows_are_consistent(self):
        inst = gap_instance(2, demand_ratio=1.25)
        ex, rows = gap_rows(inst)
        self.assertEqual(len(rows), 8)
        for r in rows:
            self.assertFalse(r["exact_beaten"])
            self.assertEqual(r["W_exact"], ex.W)
            if r["W"] != ex.W:
                self.assertIsNone(r["D_gap_pct"])                 # distances only comparable at equal W
                self.assertGreater(r["W_gap_pct"], 0)
            if r["optimal"]:
                self.assertAlmostEqual(r["km"], ex.D, places=6)
                self.assertEqual(r["W"], ex.W)


if __name__ == "__main__":
    unittest.main(verbosity=2)
