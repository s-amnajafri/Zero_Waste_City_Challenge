"""
optimality_gap.py  --  how far are the heuristics from the true optimum?

Compares every construction method (with and without 2-opt) against the exact optimum from
brute_force.py on small instances, under the SAME objective and constraints (problem.py):

    objective = (W, D):  maximise W = sum(urgency x waste), then minimise total distance D.

Gap definitions
---------------
  W gap  (%) = 100 * (W* - W) / W*            primary objective (0 = served the best possible value)
  D gap  (%) = 100 * (D - D*) / D*            ONLY where W = W*, because distances of plans that
                                              serve different points are not comparable
  optimal    = the method's (W, D) equals the exact optimum
  "exact beaten" must always be 0; a non-zero value would mean a bug in one of the two sides.

Also contains greedy_counterexample(): a hand-checkable instance on which Greedy is provably
suboptimal (the greedy-choice property fails for this problem), as required for the analysis.
"""

from __future__ import annotations

import statistics as stats
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.baselines import compare_methods
from experiments.brute_force import ExactResult, solve_exact
from data_generator import Instance, _manual, generate_instance
from experiments.experiments import print_table, write_csv
from algorithms.greedy import greedy_construct
from metrics import Metrics
from problem import EPS, is_better
from algorithms.shortest_path import ShortestPaths


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------
def gap_instance(seed: int, n_points: int = 8, demand_ratio: float = 1.25,
                 capacities: Sequence[int] = (600, 800, 1000), duration: float = 200.0) -> Instance:
    """Small random instance (all points connected) on which exact search is cheap."""
    return generate_instance(n_points=n_points, n_isolated=0, usable_roads=int(2.2 * n_points),
                             vehicle_capacities=capacities, vehicle_durations=(duration,) * len(capacities),
                             demand_ratio=demand_ratio, seed=seed, name=f"gap_s{seed}_r{demand_ratio}")


def greedy_counterexample() -> Instance:
    """
    One truck (1000 kg). Depot at (0,0).
      A (2,0)  600 kg, urgency 9   <- closest and most urgent: Greedy takes it first
      B (0,3)  500 kg, urgency 8
      C (3,3)  500 kg, urgency 8
    Greedy: A scores 0.43, B 0.09, C -0.20, so it picks A; then B and C no longer fit (1100 kg > 1000 kg).
            W = 9 x 600 = 5400.
    Optimum: B + C (exactly 1000 kg): W = 8x500 + 8x500 = 8000.
    The locally best choice blocks the better combination: the greedy-choice property fails.
    """
    pts = [(2, 0, 600, 9, 5, 300), (0, 3, 500, 8, 5, 300), (3, 3, 500, 8, 5, 300)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return _manual("greedy_counterexample", pts, edges, (1000,), (300,))


# --------------------------------------------------------------------------
# Gap computation
# --------------------------------------------------------------------------
def gap_rows(inst: Instance, sp: Optional[ShortestPaths] = None) -> Tuple[ExactResult, List[Dict[str, object]]]:
    sp = sp or ShortestPaths(inst)
    ex = solve_exact(inst, sp)
    rows = []
    for m in compare_methods(inst, sp):
        w_gap = 100.0 * (ex.W - m.W) / ex.W if ex.W > 0 else None
        same_w = (m.W == ex.W)
        d_gap = 100.0 * (m.distance_km - ex.D) / ex.D if (same_w and ex.D > 0) else None
        rows.append(dict(instance=inst.name, method=m.label, W=m.W, W_exact=ex.W, W_gap_pct=w_gap,
                         km=m.distance_km, km_exact=ex.D, D_gap_pct=d_gap,
                         optimal=not is_better((ex.W, ex.D), (m.W, m.distance_km)),
                         exact_beaten=is_better((m.W, m.distance_km), (ex.W, ex.D))))
    return ex, rows


def gap_experiment(seeds: Sequence[int], ratios: Sequence[float], n_points: int = 8
                   ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    detail: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []
    for ratio in ratios:
        per_method: Dict[str, List[Dict[str, object]]] = {}
        for s in seeds:
            _, rows = gap_rows(gap_instance(s, n_points, ratio))
            for r in rows:
                r["demand_ratio"] = ratio
                detail.append(r)
                per_method.setdefault(r["method"], []).append(r)
        for method, rs in per_method.items():
            wg = [r["W_gap_pct"] for r in rs if r["W_gap_pct"] is not None]
            dg = [r["D_gap_pct"] for r in rs if r["D_gap_pct"] is not None]
            summary.append(dict(demand_ratio=ratio, method=method, instances=len(rs),
                                optimal=sum(r["optimal"] for r in rs),
                                mean_W_gap_pct=stats.mean(wg) if wg else None,
                                max_W_gap_pct=max(wg) if wg else None,
                                W_equal=len(dg), mean_D_gap_pct=stats.mean(dg) if dg else None,
                                exact_beaten=sum(r["exact_beaten"] for r in rs)))
    return detail, summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    print("=== 1. Greedy counterexample (hand-checkable) ===")
    inst = greedy_counterexample()
    sp = ShortestPaths(inst)
    g = greedy_construct(inst, sp)
    ex = solve_exact(inst, sp)
    nm = lambda j: inst.point(j).name
    print(f"  Greedy : serves {[nm(j) for j in g.collected()]}  ->  W = {sum(inst.point(j).urgency * inst.point(j).waste for j in g.collected())}")
    print(f"  Exact  : serves {[nm(j) for s in ex.plan.values() for j in s]}  ->  W = {ex.W}")
    print("  Greedy's first pick (A, 600 kg) leaves 400 kg: neither B nor C (500 kg) fits any more.\n")

    detail, summary = gap_experiment(range(1, 41), (0.9, 1.25, 1.6))
    keep = {"first-fit (id order) + 2-opt", "greedy: distance-only a=0 + 2-opt",
            "greedy: urgency-only b=0 + 2-opt", "greedy: default a=1,b=0.5",
            "greedy: default a=1,b=0.5 + 2-opt"}
    shown = [r for r in summary if r["method"] in keep]
    print("=== 2. Optimality gap vs exact optimum: 40 random instances per demand level, 8 points, 3 vehicles ===")
    print_table("Gap summary (demand ratio = total waste / fleet capacity)", shown,
                ["demand_ratio", "method", "optimal", "mean_W_gap_pct", "max_W_gap_pct", "W_equal", "mean_D_gap_pct", "exact_beaten"])
    worst = max((r for r in detail if r["method"] == "greedy: default a=1,b=0.5 + 2-opt" and r["W_gap_pct"] is not None),
                key=lambda r: r["W_gap_pct"])
    print(f"\n  Largest pipeline W gap: {worst['instance']}  W = {worst['W']} vs optimum {worst['W_exact']} "
          f"({worst['W_gap_pct']:.1f}% below)")
    write_csv("optimality_gap_detail.csv", detail)
    write_csv("optimality_gap_summary.csv", summary)
    print("\n  CSV files written to results/")


if __name__ == "__main__":
    main()
