"""Tests for baselines.py and experiments.py (run from zero_waste/:  python -m unittest discover -s tests -v)"""

import copy
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.experiments as ex
from experiments.baselines import compare_methods, first_fit
from data_generator import _manual, generate_instance
from feasibility import check_plan
from problem import UNREACHABLE, classify_points
from algorithms.shortest_path import ShortestPaths


class TestVariants(unittest.TestCase):
    def setUp(self):
        self.inst = generate_instance()
        self.snapshot = copy.deepcopy(self.inst.to_dict())

    def test_variants_do_not_modify_the_original(self):
        ex.with_fleet(self.inst, (1000, 1000), "f")
        ex.scale_capacity(self.inst, 0.5, "c")
        ex.scale_deadlines(self.inst, 0.5, "d")
        ex.remove_points(self.inst, [1, 2], "r")
        self.assertEqual(self.inst.to_dict(), self.snapshot)

    def test_fleet_capacity_and_deadline_scaling(self):
        f = ex.with_fleet(self.inst, (1200, 1500), "f")
        self.assertEqual([(v.id, v.capacity, v.max_duration) for v in f.vehicles], [(1, 1200, 360.0), (2, 1500, 390.0)])
        c = ex.scale_capacity(self.inst, 0.5, "c")
        self.assertEqual([v.capacity for v in c.vehicles], [400, 500, 600, 750])
        d = ex.scale_deadlines(self.inst, 0.5, "d")
        self.assertAlmostEqual(d.point(9).deadline, self.inst.point(9).deadline * 0.5)

    def test_remove_points_renumbers_and_keeps_names(self):
        r = ex.remove_points(self.inst, [3, 9], "r")
        self.assertEqual(len(r.points), len(self.inst.points) - 2)
        self.assertEqual([p.id for p in r.points], list(range(1, len(r.points) + 1)))
        self.assertNotIn("P03", [p.name for p in r.points])
        self.assertNotIn("P09", [p.name for p in r.points])
        self.assertEqual(r.point(3).name, "P04")            # P04 slid into id 3
        for road in r.roads:
            self.assertTrue(0 <= road.u <= len(r.points) and 0 <= road.v <= len(r.points))
        self.assertLess(len(r.roads), len(self.inst.roads))

    def test_removing_a_hub_can_cut_off_other_points(self):
        chain = _manual("chain", [(2, 0, 100, 5, 5, 300), (4, 0, 100, 5, 5, 300), (6, 0, 100, 5, 5, 300)],
                        [(0, 1), (1, 2), (2, 3)], (500,), (300,))
        r = ex.remove_points(chain, [2], "cut")              # P02 was the only link to P03
        cls = classify_points(r, ShortestPaths(r))
        self.assertEqual({r.point(j).name: c.status for j, c in cls.items()}["P03"], UNREACHABLE)


class TestBaselines(unittest.TestCase):
    def test_first_fit_is_feasible_and_deterministic(self):
        for seed in range(1, 11):
            inst = generate_instance(seed=seed)
            sp = ShortestPaths(inst)
            a, b = first_fit(inst, sp), first_fit(inst, sp)
            self.assertEqual(a.plan(), b.plan())
            self.assertTrue(check_plan(inst, sp, a.plan()).ok, f"seed {seed}")

    def test_first_fit_uses_lowest_vehicle_id_and_point_id_order(self):
        # both points fit vehicle 1, so first-fit gives them to vehicle 1 (not the tighter vehicle 2),
        # appended in point-id order
        inst = _manual("ff", [(3, 0, 400, 5, 5, 300), (0, 3, 100, 5, 5, 300)],
                       [(0, 1), (0, 2)], (1000, 500), (300, 300))
        self.assertEqual(first_fit(inst).plan(), {1: [1, 2]})

    def test_compare_methods_shape_and_feasibility(self):
        for seed in (42, 1, 2, 3):
            ms = compare_methods(generate_instance(seed=seed))
            self.assertEqual(len(ms), 8)
            self.assertTrue(all(m.feasible for m in ms))
            for plain, improved in zip(ms[0::2], ms[1::2]):
                self.assertEqual(improved.label, plain.label + " + 2-opt")
                self.assertEqual((improved.n_collected, improved.W), (plain.n_collected, plain.W))
                self.assertLessEqual(improved.distance_km, plain.distance_km + 1e-9)

    def test_regression_default_beats_first_fit_on_30_seeds(self):
        rows = {r["method"]: r for r in ex.exp_baselines_over_seeds(range(1, 31))}
        ff, default = rows["first-fit (id order)"], rows["greedy: default a=1,b=0.5"]
        self.assertGreater(default["mean_W"], ff["mean_W"])
        self.assertGreater(default["mean_urgent_cov_pct"], ff["mean_urgent_cov_pct"])
        self.assertLess(default["mean_km_per_100kg"], ff["mean_km_per_100kg"])
        self.assertTrue(all(r["all_feasible"] for r in rows.values()))


class TestSweeps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = generate_instance()

    def test_every_sweep_row_is_feasible(self):
        for name, fn, n in (("fleet", ex.exp_fleet, 5), ("capacity", ex.exp_capacity, 5),
                            ("alpha", ex.exp_alpha, 7), ("deadlines", ex.exp_deadlines, 5)):
            rows = fn(self.inst)
            self.assertEqual(len(rows), n, name)
            self.assertTrue(all(r["feasible"] for r in rows), name)

    def test_unit_scale_reproduces_the_main_run(self):
        base = ex.summarize(ex.run_pipeline(self.inst), setting="base")
        for row in (ex.exp_capacity(self.inst, (1.0,))[0], ex.exp_deadlines(self.inst, (1.0,))[0],
                    ex.exp_fleet(self.inst)[2]):               # fleet entry 2 is the default 4-vehicle fleet
            self.assertEqual((row["W"], row["kg"]), (base["W"], base["kg"]))
            self.assertAlmostEqual(row["km_final"], base["km_final"], places=6)

    def test_tighter_deadlines_can_make_points_individually_infeasible(self):
        rows = {r["setting"]: r for r in ex.exp_deadlines(self.inst)}
        self.assertEqual(rows["deadlines x1.0"]["infeasible"], 0)
        self.assertGreaterEqual(rows["deadlines x0.5"]["infeasible"], 1)

    def test_removal_experiment_accounting(self):
        rows, notes = ex.exp_removal(self.inst)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(notes), 2)
        base = ex.run_pipeline(self.inst)
        # W identity: W_after = W_before - W(removed & collected before) + W(newly collected) - W(lost)
        eligible = [p for p in self.inst.points if classify_points(self.inst, base.sp)[p.id].status == "eligible"]
        victims = sorted(eligible, key=lambda p: (-p.urgency, -p.waste, p.id))[:2]
        inst2 = ex.remove_points(self.inst, [p.id for p in victims], "r")
        run2 = ex.run_pipeline(inst2)
        diff = ex.plan_diff(self.inst, base.final.plan(), inst2, run2.final.plan())
        val = lambda inst, name: next(p.urgency * p.waste for p in inst.points if p.name == name)
        before = {self.inst.point(j).name for s in base.final.plan().values() for j in s}
        expected = (base.m_final.W
                    - sum(val(self.inst, v.name) for v in victims if v.name in before)
                    + sum(val(inst2, n) for n in diff["gained"])
                    - sum(val(inst2, n) for n in diff["lost"]))
        self.assertEqual(run2.m_final.W, expected)

    def test_plan_diff(self):
        inst = generate_instance()
        a = {1: [1, 2], 2: [3]}
        b = {1: [1], 2: [3, 2], 3: [4]}
        d = ex.plan_diff(inst, a, inst, b)
        self.assertEqual((d["lost"], d["gained"], d["moved"]), ([], ["P04"], ["P02"]))


class TestOutput(unittest.TestCase):
    def test_csv_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = ex.RESULTS_DIR
            ex.RESULTS_DIR = tmp
            try:
                ex.write_csv("t.csv", [{"a": 1, "b": 2.5}, {"a": 3, "c": "x"}])
            finally:
                ex.RESULTS_DIR = old
            with open(os.path.join(tmp, "t.csv")) as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["a"], "1")
        self.assertEqual(rows[1]["c"], "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
