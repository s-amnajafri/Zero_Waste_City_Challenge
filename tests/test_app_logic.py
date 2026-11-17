"""Tests for app_logic.py (everything the interactive app computes; no Streamlit needed)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import interface.app_logic as A
from feasibility import check_plan
from problem import is_better


def all_scenarios():
    """A grid of scenarios, including extreme slider positions."""
    out = [dict(choice=c) for c in A.DATASET_CHOICES]
    for n in (8, 20, 60):
        for iso in (0, 2):
            for ratio in (0.5, 1.1, 2.0):
                out.append(dict(choice=A.RANDOM_CHOICE, n_points=n, n_isolated=iso, demand_ratio=ratio, seed=n + iso))
    for fleet in A.FLEET_SIZES:
        out.append(dict(choice=A.MAIN_CHOICE, fleet_size=fleet))
    for cap in (0.5, 2.0):
        for dl in (0.5, 2.0):
            out.append(dict(choice=A.MAIN_CHOICE, capacity_scale=cap, deadline_scale=dl))
    return out


class TestBuildInstance(unittest.TestCase):
    def test_every_dataset_choice_builds(self):
        for c in A.DATASET_CHOICES:
            inst = A.build_instance(c)
            self.assertGreater(len(inst.points), 0, c)
            self.assertGreater(len(inst.vehicles), 0, c)

    def test_main_dataset_is_the_documented_one(self):
        inst = A.build_instance(A.MAIN_CHOICE)
        self.assertEqual((len(inst.points), len(inst.roads), len(inst.usable_roads())), (16, 30, 27))
        self.assertEqual([v.capacity for v in inst.vehicles], [800, 1000, 1200, 1500])

    def test_random_instance_respects_controls(self):
        inst = A.build_instance(A.RANDOM_CHOICE, seed=7, n_points=30, n_isolated=2, demand_ratio=1.5)
        self.assertEqual(len(inst.points), 30)
        self.assertEqual(len(inst.unreachable_points()), 2)
        again = A.build_instance(A.RANDOM_CHOICE, seed=7, n_points=30, n_isolated=2, demand_ratio=1.5)
        self.assertEqual(inst.to_dict(), again.to_dict())                       # reproducible
        other = A.build_instance(A.RANDOM_CHOICE, seed=8, n_points=30, n_isolated=2, demand_ratio=1.5)
        self.assertNotEqual(inst.to_dict(), other.to_dict())

    def test_isolated_points_are_clamped_for_tiny_instances(self):
        inst = A.build_instance(A.RANDOM_CHOICE, n_points=8, n_isolated=2)
        self.assertEqual(len(inst.points), 8)

    def test_fleet_and_scales(self):
        base = A.build_instance(A.MAIN_CHOICE)
        f3 = A.build_instance(A.MAIN_CHOICE, fleet_size=3)
        self.assertEqual(len(f3.vehicles), 3)
        c = A.build_instance(A.MAIN_CHOICE, capacity_scale=0.5)
        self.assertEqual([v.capacity for v in c.vehicles], [v.capacity // 2 for v in base.vehicles])
        d = A.build_instance(A.MAIN_CHOICE, deadline_scale=0.5)
        self.assertAlmostEqual(d.point(9).deadline, base.point(9).deadline * 0.5, places=1)

    def test_original_dataset_is_never_modified_by_scenarios(self):
        before = A.build_instance(A.MAIN_CHOICE).to_dict()
        A.build_instance(A.MAIN_CHOICE, fleet_size=2, capacity_scale=0.5, deadline_scale=0.5)
        self.assertEqual(A.build_instance(A.MAIN_CHOICE).to_dict(), before)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            A.build_instance("no such dataset")
        with self.assertRaises(ValueError):
            A.build_instance(A.MAIN_CHOICE, fleet_size=9)


class TestRunAll(unittest.TestCase):
    def test_main_dataset_headline_numbers(self):
        res = A.run_all(A.build_instance(A.MAIN_CHOICE))
        self.assertEqual(res.m_final.W, 26927)
        self.assertAlmostEqual(res.m_final.distance_km, 173.09, places=2)
        self.assertEqual((res.m_final.n_collected, res.m_final.n_reachable), (13, 15))
        self.assertEqual(res.blocked, 1)
        self.assertTrue(res.check.ok)

    def test_every_scenario_ends_in_a_verified_feasible_plan(self):
        for sc in all_scenarios():
            res = A.run_all(A.build_instance(**sc))
            self.assertTrue(res.check.ok, f"{sc}: {res.check.violations}")
            self.assertTrue(res.m_greedy.feasible and res.m_final.feasible, str(sc))
            self.assertLessEqual(res.m_final.distance_km, res.m_greedy.distance_km + 1e-9, str(sc))
            self.assertEqual(res.m_final.W, res.m_greedy.W, str(sc))              # 2-opt never changes W

    def test_alpha_beta_extremes_run(self):
        inst = A.build_instance(A.MAIN_CHOICE)
        for a, b in ((0.0, 0.0), (0.0, 2.0), (4.0, 0.0), (4.0, 2.0)):
            self.assertTrue(A.run_all(inst, a, b).check.ok, (a, b))

    def test_parameters_change_the_plan(self):
        inst = A.build_instance(A.MAIN_CHOICE)
        self.assertNotEqual(A.run_all(inst, 1.0, 0.5).m_final.W, A.run_all(inst, 0.0, 1.0).m_final.W)

    def test_iteration_cap_is_respected(self):
        inst = A.build_instance(A.RANDOM_CHOICE, seed=7, n_points=17, n_isolated=0, demand_ratio=0.8)
        res = A.run_all(inst, max_iter=1)
        self.assertTrue(all(len(i.moves) <= 1 for i in res.final.improvements.values()))
        self.assertTrue(res.check.ok)


class TestTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = A.build_instance(A.MAIN_CHOICE)
        cls.res = A.run_all(cls.inst)

    def test_stop_rows_are_consistent_with_the_plan_and_metrics(self):
        rows = A.stop_rows(self.inst, self.res.final.routes)
        self.assertEqual(len(rows), self.res.m_final.n_collected)
        self.assertAlmostEqual(min(r["Slack (min)"] for r in rows), self.res.m_final.min_slack_min, places=1)
        for vid, r in self.res.final.routes.items():
            mine = [x for x in rows if x["Vehicle"] == f"V{vid}"]
            if mine:
                self.assertEqual(mine[-1]["Load after (kg)"], r.load)
                self.assertEqual([x["Stop"] for x in mine], list(range(1, len(mine) + 1)))

    def test_vehicle_rows_cover_the_fleet_and_show_unused_vehicles(self):
        self.assertEqual(len(A.vehicle_rows(self.res)), len(self.inst.vehicles))
        inst = A.build_instance("Edge: no 2-opt gain")
        rows = A.vehicle_rows(A.run_all(inst))
        self.assertEqual(rows[0]["Stops"], 4)
        inst2 = A.build_instance("Edge: deadline violation")
        self.assertIn("unused", [r["Route"] for r in A.vehicle_rows(A.run_all(inst2))])

    def test_outcome_rows_account_for_every_point_exactly_once(self):
        rows = A.outcome_rows(self.res)
        self.assertEqual([r["Point"] for r in rows], [p.name for p in self.inst.points])
        collected = [r for r in rows if r["Outcome"].startswith("collected")]
        self.assertEqual(len(collected), self.res.m_final.n_collected)
        by = {r["Point"]: r for r in rows}
        self.assertIn("capacity", by["P01"]["Outcome"])
        self.assertIn("capacity", by["P07"]["Outcome"])
        self.assertEqual(by["P16"]["Initial class"], "unreachable")
        self.assertTrue(by["P16"]["Outcome"].startswith("not attempted"))

    def test_outcome_rows_for_infeasible_points(self):
        res = A.run_all(A.build_instance("Edge: oversized point"))
        row = next(r for r in A.outcome_rows(res) if r["Point"] == "P03")
        self.assertEqual(row["Initial class"], "individually infeasible")
        self.assertIn("exceeds the capacity", row["Outcome"])

    def test_metrics_rows(self):
        rows = A.metrics_rows(self.res.m_greedy, self.res.m_final)
        self.assertEqual(len(rows), 15)
        d = {r["Metric"]: r for r in rows}
        self.assertEqual(d["Objective W = sum urgency x waste"]["Greedy + 2-opt"], "26,927")
        self.assertEqual(d["Total distance (km)"]["Greedy"], "175.54")
        self.assertEqual(d["Total distance (km)"]["Greedy + 2-opt"], "173.09")
        self.assertEqual(d["Urgent points served"]["Greedy"], "6 / 6")

    def test_improvement_rows(self):
        rows = A.improvement_rows(self.res)
        v3 = next(r for r in rows if r["Vehicle"] == "V3")
        self.assertAlmostEqual(v3["Saved (km)"], 2.45, places=2)
        v4 = next(r for r in rows if r["Vehicle"] == "V4")
        self.assertEqual(v4["Improving moves blocked by a deadline"], 1)
        self.assertTrue(all(r["Stopped because"] == "no improving feasible move left" for r in rows))

    def test_baseline_rows(self):
        rows = A.baseline_rows(self.inst, self.res.sp)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["Feasible"] == "yes" for r in rows))
        self.assertEqual(rows[-1]["Method"], "greedy: default a=1,b=0.5 + 2-opt")
        self.assertEqual(rows[-1]["W"], 26927)

    def test_rows_to_csv(self):
        rows = A.stop_rows(self.inst, self.res.final.routes)
        text = A.rows_to_csv(rows)
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), len(rows) + 1)
        self.assertTrue(lines[0].startswith("Vehicle,Stop,Point"))
        self.assertEqual(A.rows_to_csv([]), "")


class TestExactAndExport(unittest.TestCase):
    def test_exact_is_skipped_politely_for_large_instances(self):
        ex = A.exact_comparison(A.run_all(A.build_instance(A.MAIN_CHOICE)))
        self.assertEqual(ex["status"], "skipped")
        self.assertIn("15 eligible points", ex["message"])

    def test_exact_on_small_instances_is_consistent_and_never_beaten(self):
        for seed in range(1, 13):
            inst = A.build_instance(A.RANDOM_CHOICE, seed=seed, n_points=8, n_isolated=0, demand_ratio=1.3)
            res = A.run_all(inst)
            ex = A.exact_comparison(res)
            self.assertEqual(ex["status"], "ok")
            self.assertFalse(is_better((res.m_final.W, res.m_final.distance_km), (ex["W_exact"], ex["km_exact"])),
                             f"seed {seed}: pipeline beat the exact optimum")
            self.assertGreaterEqual(ex["w_gap_pct"], 0.0)
            if ex["optimal"]:
                self.assertEqual(res.m_final.W, ex["W_exact"])
            plan = {int(v[1:]): [next(p.id for p in inst.points if p.name == n) for n in names] for v, names in ex["plan"].items()}
            self.assertTrue(check_plan(inst, res.sp, plan).ok)

    def test_export_json_is_valid_and_matches_the_command_line_structure(self):
        inst = A.build_instance(A.MAIN_CHOICE)
        d = A.export_json(inst, 1.0, 0.5, 1000)
        text = json.dumps(d)
        back = json.loads(text)
        self.assertTrue(back["feasible"])
        self.assertEqual(back["metrics_final"]["W"], 26927)
        self.assertIsNone(back["exact"])
        self.assertEqual(back["classification"]["P16"]["status"], "unreachable")
        small = A.build_instance(A.RANDOM_CHOICE, seed=2, n_points=8, n_isolated=0)
        self.assertIsNotNone(A.export_json(small, 1.0, 0.5, 1000, with_exact=True)["exact"])

    def test_export_json_prints_nothing(self):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            A.export_json(A.build_instance(A.MAIN_CHOICE), 1.0, 0.5, 1000)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
