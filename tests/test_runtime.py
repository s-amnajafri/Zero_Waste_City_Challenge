"""Tests for runtime_experiment.py: the derived complexity formulas are checked against counted work.
(Timings themselves are machine-dependent and are never asserted.)"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.runtime_experiment as rt


class TestFit(unittest.TestCase):
    def test_known_exponents(self):
        xs = [10, 20, 40, 80, 160]
        s, _, r2 = rt.loglog_fit(xs, [3 * x ** 2 for x in xs])
        self.assertAlmostEqual(s, 2.0, places=9)
        self.assertAlmostEqual(r2, 1.0, places=9)
        s3, _, _ = rt.loglog_fit(xs, [0.5 * x ** 3 for x in xs])
        self.assertAlmostEqual(s3, 3.0, places=9)

    def test_exponent_rows_on_synthetic_data(self):
        rows = [dict(n=n, ms_paths=n ** 2 * math.log2(n), ms_greedy=n ** 2, greedy_evals=n ** 2,
                     ms_2opt=n ** 3, twoopt_delta_evals=n ** 3) for n in (10, 20, 40, 80, 160)]
        out = {r["quantity"]: r for r in rt.exponent_rows(rows, "synthetic")}
        self.assertAlmostEqual(out["Greedy time"]["exponent_all_sizes"], 2.0, places=6)
        self.assertAlmostEqual(out["2-opt time"]["exponent_n_ge_40"], 3.0, places=6)
        self.assertTrue(2.0 < out["Dijkstra all-pairs time"]["exponent_all_sizes"] < 2.4)


class TestWorstCaseFormulas(unittest.TestCase):
    def test_greedy_pair_evaluations_equal_the_bound_exactly(self):
        for n, k in ((10, 1), (10, 3), (25, 4), (40, 2)):
            m = rt.measure(rt.free_instance(n, k), repeats=1)
            self.assertEqual(m["served"], n, f"n={n},k={k}: everything must be served")
            self.assertEqual(m["greedy_evals"], k * n * (n + 1) // 2, f"n={n},k={k}")
            self.assertEqual(m["greedy_evals"], m["greedy_bound"])

    def test_two_opt_operation_count_formula_for_a_single_long_route(self):
        # one route of r stops, converged: (moves + 1) scans, each r(r-1)/2 delta evaluations
        for n in (10, 20, 40):
            m = rt.measure(rt.free_instance(n, 4), repeats=1)
            self.assertEqual(m["vehicles_used"], 1)
            r = m["r_max"]
            self.assertEqual(m["twoopt_delta_evals"], (m["twoopt_moves"] + 1) * r * (r - 1) // 2, f"n={n}")

    def test_exact_solver_ordering_counts_match_the_formula(self):
        rows = rt.exact_growth(ns=(4, 5, 6), ks=(1, 3))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["matches"] for r in rows))


class TestRealisticFamily(unittest.TestCase):
    def test_scaled_instance_properties(self):
        for n in (20, 40, 80):
            inst = rt.scaled_instance(n, 4)
            self.assertEqual(len(inst.points), n)
            self.assertEqual(len(inst.unreachable_points()), 1)
            self.assertAlmostEqual(sum(v.capacity for v in inst.vehicles), 300 * n, delta=4)
            m = rt.measure(inst, repeats=1)
            self.assertGreaterEqual(m["served"] / m["eligible"], 0.8, f"n={n}: coverage should stay high")
            self.assertLessEqual(m["greedy_evals"], m["greedy_bound"])
            self.assertGreaterEqual(m["greedy_evals"], 0.9 * m["greedy_bound"])   # the bound is nearly tight

    def test_measure_fields_are_sane(self):
        m = rt.measure(rt.scaled_instance(20, 4), repeats=2)
        for key in ("ms_paths", "ms_greedy", "ms_2opt", "dijkstra_theory", "ns_per_dijkstra_unit"):
            self.assertGreater(m[key], 0, key)
        self.assertEqual(m["V"], m["n"] + 1)

    def test_fleet_sweep_keeps_the_graph_fixed_and_work_grows_with_k(self):
        rows = rt.fleet_sweep(n=40, ks=(1, 2, 4), repeats=1)
        self.assertEqual({(r["V"], r["E"]) for r in rows}, {(rows[0]["V"], rows[0]["E"])})   # same road network
        evals = [r["greedy_evals"] for r in rows]
        self.assertEqual(evals, sorted(evals))
        self.assertTrue(all(r["greedy_evals"] <= r["greedy_bound"] for r in rows))
        self.assertGreater(rows[0]["r_max"], rows[-1]["r_max"])                              # routes get shorter


class TestOutputs(unittest.TestCase):
    def test_plot_is_written_when_matplotlib_is_available(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        free = rt.size_sweep("free", (10, 20), 4, 1)
        scaled = rt.size_sweep("scaled", (10, 20), 4, 1)
        fleet = rt.fleet_sweep(20, (1, 2), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "figs", "r.png")
            self.assertTrue(rt.make_plot(free, scaled, fleet, path))
            self.assertGreater(os.path.getsize(path), 5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
