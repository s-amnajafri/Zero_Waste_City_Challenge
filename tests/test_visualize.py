"""Smoke and content tests for visualize.py (skipped automatically if matplotlib is not installed)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import interface.visualize as viz
from data_generator import edge_unreachable, generate_instance
from algorithms.greedy import greedy_construct
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import improve_plan

HAVE = viz.HAVE_MPL


@unittest.skipUnless(HAVE, "matplotlib not installed")
class TestFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = generate_instance()
        cls.sp = ShortestPaths(cls.inst)
        cls.g = greedy_construct(cls.inst, cls.sp)
        cls.t = improve_plan(cls.inst, cls.sp, cls.g)

    def test_before_after_titles_show_the_real_distances(self):
        fig = viz.plot_before_after(self.inst, self.sp, self.g.plan(), self.t.plan())
        titles = [ax.get_title() for ax in fig.axes]
        self.assertEqual(len(fig.axes), 2)
        self.assertIn("175.54", titles[0])
        self.assertIn("173.09", titles[1])

    def test_single_route_figure_and_numbered_stops(self):
        fig = viz.plot_routes(self.inst, self.sp, self.t.plan(), "Greedy + 2-opt")
        ax = fig.axes[0]
        stop_labels = [t.get_text() for t in ax.texts if t.get_text().isdigit()]
        self.assertEqual(len(stop_labels), sum(len(s) for s in self.t.plan().values()))   # one number per stop
        self.assertIn("173.09", ax.get_title())

    def test_unreachable_point_is_drawn_with_its_own_marker(self):
        inst = edge_unreachable()
        sp = ShortestPaths(inst)
        fig, ax = viz.plt.subplots()
        viz.draw_network(ax, inst, sp)
        self.assertTrue(ax.collections)                          # points were drawn
        labels = [t.get_text() for t in ax.texts]
        self.assertIn("P06", labels)
        self.assertIn("Depot", labels)
        viz.plt.close(fig)

    def test_closed_roads_are_dashed_and_open_roads_solid(self):
        fig, ax = viz.plt.subplots()
        viz.draw_network(ax, self.inst, self.sp)
        styles = [l.get_linestyle() for l in ax.lines]
        self.assertEqual(len(ax.lines), len(self.inst.roads))
        self.assertEqual(sum(s != "-" for s in styles), len(self.inst.closed_roads()))
        viz.plt.close(fig)

    def test_block_diagram_has_all_nine_boxes(self):
        fig = viz.block_diagram()
        boxes = [p for p in fig.axes[0].patches if isinstance(p, viz.FancyBboxPatch)]
        self.assertEqual(len(boxes), 9)                          # 6 pipeline stages + 3 benchmark modules

    def test_result_charts_from_rows(self):
        base_rows = [dict(method=m + " + 2-opt", mean_W=1000.0 * i, mean_urgent_cov_pct=50.0 + i,
                          mean_km=100.0 + i, mean_km_per_100kg=4.0 + i) for i, m in enumerate(
                              ["first-fit (id order)", "greedy: distance-only a=0", "greedy: urgency-only b=0",
                               "greedy: default a=1,b=0.5"], 1)]
        self.assertEqual(len(viz.baselines_chart(base_rows).axes), 4)

        mk = lambda n, setting, infeasible=0: dict(setting=setting, vehicles=4, collected=n, urgent_cov=90.0,
                                                    km_final=150.0, infeasible=infeasible)
        fleet = [mk(9, "2 vehicles"), mk(13, "4 vehicles")]
        cap = [mk(8, "capacity x0.6"), mk(13, "capacity x1.0")]
        alpha = [mk(13, "alpha=0.0, beta=0.5"), mk(13, "urgency-only (beta=0)")]
        dl = [mk(13, "deadlines x0.5", infeasible=1), mk(13, "deadlines x1.0")]
        fig = viz.sensitivity_chart(fleet, cap, alpha, dl)
        self.assertEqual(len([a for a in fig.axes if a.get_ylabel().startswith(("points", "urgent"))]), 4)
        notes = [t.get_text() for a in fig.axes for t in a.texts]
        self.assertTrue(any("infeasible" in n for n in notes))   # excluded points are annotated, not hidden

        methods = ["greedy: default a=1,b=0.5 + 2-opt", "greedy: urgency-only b=0 + 2-opt",
                   "greedy: distance-only a=0 + 2-opt", "first-fit (id order) + 2-opt"]
        summary = [dict(demand_ratio=r, method=m, mean_W_gap_pct=5.0 + i) for r in (0.9, 1.25) for i, m in enumerate(methods)]
        self.assertEqual(len(viz.gap_chart(summary).axes), 1)

    def test_save_all_writes_every_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = viz.save_all(tmp, quick=True)
            names = sorted(os.path.basename(p) for p in written)
            self.assertEqual(names, ["baselines_30_seeds.png", "block_diagram.png", "main_final_routes.png",
                                     "main_routes.png", "optimality_gap.png", "over_demand_final_routes.png",
                                     "over_demand_routes.png", "sensitivity.png"])
            for p in written:
                self.assertGreater(os.path.getsize(p), 5000, p)


class TestWithoutMatplotlib(unittest.TestCase):
    def test_clear_error_when_matplotlib_is_missing(self):
        old = viz.HAVE_MPL
        viz.HAVE_MPL = False
        try:
            with self.assertRaises(RuntimeError) as cm:
                viz.block_diagram()
            self.assertIn("matplotlib", str(cm.exception))
        finally:
            viz.HAVE_MPL = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
