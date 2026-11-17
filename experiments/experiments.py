"""
experiments.py  --  Step 4: baselines, sensitivity analysis and point-removal scenarios

Everything here calls the algorithms; none of the algorithm modules contain experiment code.
Results are printed as tables and written to results/*.csv (reused by the report and the
Streamlit interface later).

Experiments
-----------
  1. Baseline comparison on the main dataset (4 construction methods, each with/without 2-opt)
  2. The same comparison averaged over 30 random instances (robustness: not one lucky seed)
  3. Sensitivity to fleet size            (2..6 vehicles)
  4. Sensitivity to vehicle capacity      (all capacities scaled 0.6x .. 1.5x)
  5. Sensitivity to the urgency weight    (alpha swept, beta fixed; plus urgency-only)
  6. Sensitivity to deadline tightness    (all deadlines scaled 0.5x .. 2x)
  7. Point removal (rubric scenario)      remove the 2 most urgent / the 2 heaviest points

Removal rule: eligible points are ranked by (urgency desc, waste desc, id) for "most urgent"
and by (waste desc, urgency desc, id) for "heaviest"; the top two are removed together with
their roads. Remaining points are renumbered but keep their names.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.baselines import compare_methods
from data_generator import Instance, Point, Road, Vehicle, generate_instance
from algorithms.greedy import GreedyParams, GreedyResult, greedy_construct
from metrics import Metrics, compute_metrics
from problem import ELIGIBLE, Plan, classify_points
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import TwoOptResult, improve_plan

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DURATION_BY_CAPACITY = {800: 300.0, 1000: 330.0, 1200: 360.0, 1500: 390.0}   # as in the main dataset


# --------------------------------------------------------------------------
# Instance variants (the original is never modified)
# --------------------------------------------------------------------------
def make_fleet(capacities: Sequence[int]) -> List[Vehicle]:
    return [Vehicle(i + 1, int(c), DURATION_BY_CAPACITY.get(int(c), 360.0)) for i, c in enumerate(capacities)]


def with_fleet(inst: Instance, capacities: Sequence[int], name: str) -> Instance:
    return replace(inst, name=name, vehicles=make_fleet(capacities))


def scale_capacity(inst: Instance, factor: float, name: str) -> Instance:
    vehicles = [Vehicle(v.id, int(round(v.capacity * factor)), v.max_duration) for v in inst.vehicles]
    return replace(inst, name=name, vehicles=vehicles)


def scale_deadlines(inst: Instance, factor: float, name: str) -> Instance:
    points = [replace(p, deadline=round(p.deadline * factor, 1)) for p in inst.points]
    return replace(inst, name=name, points=points)


def remove_points(inst: Instance, remove_ids: Sequence[int], name: str) -> Instance:
    """Delete points and their incident roads; renumber the rest 1..n' (names are kept)."""
    gone = set(remove_ids)
    keep = [p for p in inst.points if p.id not in gone]
    new_id = {0: 0}
    for k, p in enumerate(keep, start=1):
        new_id[p.id] = k
    points = [replace(p, id=new_id[p.id]) for p in keep]
    roads = [Road(new_id[r.u], new_id[r.v], r.distance, r.accessible)
             for r in inst.roads if r.u in new_id and r.v in new_id]
    return replace(inst, name=name, points=points, roads=roads)


# --------------------------------------------------------------------------
# One full pipeline run
# --------------------------------------------------------------------------
@dataclass
class PipelineRun:
    inst: Instance
    sp: ShortestPaths
    greedy: GreedyResult
    final: TwoOptResult
    m_greedy: Metrics
    m_final: Metrics
    t_paths_ms: float

    @property
    def blocked(self) -> int:
        return sum(i.rejected_deadline for i in self.final.improvements.values())


def run_pipeline(inst: Instance, params: GreedyParams = GreedyParams(), max_iter: int = 1000) -> PipelineRun:
    t0 = time.perf_counter()
    sp = ShortestPaths(inst)
    t_paths = (time.perf_counter() - t0) * 1000
    cls = classify_points(inst, sp)
    g = greedy_construct(inst, sp, params)
    fin = improve_plan(inst, sp, g, max_iter)
    mg = compute_metrics(inst, sp, g.plan(), "greedy", g.runtime_ms, cls)
    mf = compute_metrics(inst, sp, fin.plan(), "greedy+2opt", g.runtime_ms + fin.runtime_ms, cls)
    return PipelineRun(inst, sp, g, fin, mg, mf, t_paths)


def summarize(run: PipelineRun, **extra) -> Dict[str, object]:
    mg, mf = run.m_greedy, run.m_final
    red = 100.0 * (mg.distance_km - mf.distance_km) / mg.distance_km if mg.distance_km > 0 else 0.0
    return dict(extra,
                vehicles=mf.n_vehicles, eligible=mf.n_eligible, unreachable=mf.n_unreachable,
                infeasible=mf.n_infeasible, collected=mf.n_collected, kg=mf.kg_collected, W=mf.W,
                urgent_cov=mf.urgent_coverage_pct, used=mf.vehicles_used,
                fleet_util=mf.fleet_utilization_pct, km_greedy=mg.distance_km, km_final=mf.distance_km,
                reduction_pct=red, min_slack=mf.min_slack_min, blocked=run.blocked,
                feasible=mg.feasible and mf.feasible)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def exp_baselines(inst: Instance) -> List[Dict[str, object]]:
    return [dict(method=m.label, feasible=m.feasible, collected=m.n_collected, kg=m.kg_collected,
                 W=m.W, W_pct=m.W_pct, urgent=f"{m.urgent_collected}/{m.urgent_eligible}",
                 km=m.distance_km, km_per_100kg=m.km_per_100kg, runtime_ms=m.runtime_ms)
            for m in compare_methods(inst)]


def exp_baselines_over_seeds(seeds: Sequence[int]) -> List[Dict[str, object]]:
    acc: Dict[str, List[Metrics]] = {}
    for s in seeds:
        for m in compare_methods(generate_instance(seed=s, name=f"seed{s}")):
            acc.setdefault(m.label, []).append(m)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    rows = []
    for label, ms in acc.items():
        urg = [m.urgent_coverage_pct for m in ms if m.urgent_coverage_pct is not None]
        rows.append(dict(method=label, instances=len(ms), all_feasible=all(m.feasible for m in ms),
                         mean_kg=mean([m.kg_collected for m in ms]), mean_W=mean([m.W for m in ms]),
                         mean_urgent_cov_pct=mean(urg), mean_km=mean([m.distance_km for m in ms]),
                         mean_km_per_100kg=mean([m.km_per_100kg for m in ms if m.km_per_100kg])))
    return rows


FLEETS = {2: (1200, 1500), 3: (1000, 1200, 1500), 4: (800, 1000, 1200, 1500),
          5: (800, 1000, 1000, 1200, 1500), 6: (800, 800, 1000, 1000, 1200, 1500)}


def exp_fleet(inst: Instance) -> List[Dict[str, object]]:
    return [summarize(run_pipeline(with_fleet(inst, caps, f"fleet{m}")), setting=f"{m} vehicles {list(caps)}")
            for m, caps in FLEETS.items()]


def exp_capacity(inst: Instance, factors=(0.6, 0.8, 1.0, 1.2, 1.5)) -> List[Dict[str, object]]:
    return [summarize(run_pipeline(scale_capacity(inst, f, f"cap{f}")), setting=f"capacity x{f}") for f in factors]


def exp_alpha(inst: Instance, alphas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0)) -> List[Dict[str, object]]:
    rows = [summarize(run_pipeline(inst, GreedyParams(a, 0.5)), setting=f"alpha={a}, beta=0.5") for a in alphas]
    rows.append(summarize(run_pipeline(inst, GreedyParams(1.0, 0.0)), setting="urgency-only (beta=0)"))
    return rows


def exp_deadlines(inst: Instance, factors=(0.5, 0.75, 1.0, 1.5, 2.0)) -> List[Dict[str, object]]:
    return [summarize(run_pipeline(scale_deadlines(inst, f, f"dl{f}")), setting=f"deadlines x{f}") for f in factors]


def plan_diff(a: Instance, plan_a: Plan, b: Instance, plan_b: Plan) -> Dict[str, List[str]]:
    """Compare two plans by point NAME (ids change when points are removed)."""
    own_a = {a.point(j).name: v for v, s in plan_a.items() for j in s}
    own_b = {b.point(j).name: v for v, s in plan_b.items() for j in s}
    names_b = {p.name for p in b.points}
    return dict(lost=sorted(n for n in own_a if n in names_b and n not in own_b),
                gained=sorted(n for n in own_b if n not in own_a),
                moved=sorted(n for n in own_a if n in own_b and own_a[n] != own_b[n]))


def exp_removal(inst: Instance) -> Tuple[List[Dict[str, object]], List[str]]:
    base = run_pipeline(inst)
    cls = classify_points(inst, base.sp)
    eligible = [p for p in inst.points if cls[p.id].status == ELIGIBLE]
    scenarios = [
        ("remove 2 most urgent", sorted(eligible, key=lambda p: (-p.urgency, -p.waste, p.id))[:2]),
        ("remove 2 heaviest", sorted(eligible, key=lambda p: (-p.waste, -p.urgency, p.id))[:2]),
    ]
    rows = [summarize(base, setting="baseline (nothing removed)")]
    notes: List[str] = []
    for label, victims in scenarios:
        names = ", ".join(f"{p.name}(u={p.urgency}, {p.waste} kg)" for p in victims)
        inst2 = remove_points(inst, [p.id for p in victims], label)
        run2 = run_pipeline(inst2)
        rows.append(summarize(run2, setting=f"{label}: {names}"))
        d_g = plan_diff(inst, base.greedy.plan(), inst2, run2.greedy.plan())
        d_f = plan_diff(inst, base.final.plan(), inst2, run2.final.plan())
        notes.append(f"{label}: removed {names}\n"
                     f"    Greedy   : lost {d_g['lost'] or 'none'}, newly collected {d_g['gained'] or 'none'}, "
                     f"changed vehicle {d_g['moved'] or 'none'}\n"
                     f"    + 2-opt  : lost {d_f['lost'] or 'none'}, newly collected {d_f['gained'] or 'none'}, "
                     f"changed vehicle {d_f['moved'] or 'none'}")
    return rows, notes


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 100 else f"{v:.2f}"
    return str(v)


def print_table(title: str, rows: List[Dict[str, object]], cols: Sequence[str]) -> None:
    print(f"\n### {title}")
    table = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in table)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def write_csv(name: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(RESULTS_DIR, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


SWEEP_COLS = ["setting", "eligible", "unreachable", "infeasible", "collected", "kg", "W", "urgent_cov",
              "used", "fleet_util", "km_greedy", "km_final", "reduction_pct", "blocked", "min_slack", "feasible"]


def main() -> None:
    inst = generate_instance()
    print("Zero-Waste City Challenge - Step 4 experiments (main dataset: seed 42, 15 reachable points)")

    rows = exp_baselines(inst)
    print_table("1. Baselines on the main dataset", rows,
                ["method", "feasible", "collected", "kg", "W", "W_pct", "urgent", "km", "km_per_100kg"])
    write_csv("baselines_main.csv", rows)

    rows = exp_baselines_over_seeds(range(1, 31))
    print_table("2. Baselines averaged over 30 random instances (seeds 1-30)", rows,
                ["method", "all_feasible", "mean_kg", "mean_W", "mean_urgent_cov_pct", "mean_km", "mean_km_per_100kg"])
    write_csv("baselines_30_seeds.csv", rows)

    for title, fn, fname in (("3. Sensitivity: fleet size", exp_fleet, "sensitivity_fleet.csv"),
                             ("4. Sensitivity: vehicle capacity", exp_capacity, "sensitivity_capacity.csv"),
                             ("5. Sensitivity: urgency weight", exp_alpha, "sensitivity_urgency_weight.csv"),
                             ("6. Sensitivity: deadline tightness", exp_deadlines, "sensitivity_deadlines.csv")):
        rows = fn(inst)
        print_table(title, rows, SWEEP_COLS)
        write_csv(fname, rows)

    rows, notes = exp_removal(inst)
    print_table("7. Point removal (rubric scenario)", rows, SWEEP_COLS)
    print()
    for n in notes:
        print("  " + n)
    write_csv("removal.csv", rows)
    print(f"\nCSV files written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
