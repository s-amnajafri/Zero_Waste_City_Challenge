"""
app_logic.py  --  everything the interactive app computes, with NO Streamlit import

The Streamlit file (app.py) is a thin layer that only draws widgets and shows what this module
returns. Keeping the logic here means it can be unit-tested without a browser or Streamlit.

Functions
---------
  build_instance   scenario controls -> Instance (dataset, fleet, capacity scale, deadline scale)
  run_all          Dijkstra -> classify -> Greedy -> 2-opt -> checker -> metrics
  *_rows           plain list-of-dict tables for the interface (and CSV export)
  exact_comparison optimum on small instances, or a polite "skipped" message
  export_json      the same JSON as `python main.py`
"""

from __future__ import annotations

import contextlib
import csv
import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import main as cli_main
from experiments.baselines import compare_methods
from experiments.brute_force import DEFAULT_MAX_POINTS, solve_exact
from data_generator import Instance, all_edge_cases, generate_instance
from experiments.experiments import FLEETS, scale_capacity, scale_deadlines, with_fleet
from feasibility import PlanCheck, check_plan
from algorithms.greedy import UNCOLLECTED_TEXT, GreedyParams, GreedyResult, VehicleRoute, greedy_construct
from metrics import Metrics, compute_metrics
from problem import ELIGIBLE, INFEASIBLE, REASON_TEXT, UNREACHABLE, PointClass, classify_points, is_better
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import TwoOptResult, improve_plan

EDGE_NAMES = {
    "Edge: unreachable point": "edge_unreachable",
    "Edge: over-demand": "edge_over_demand",
    "Edge: oversized point": "edge_oversized",
    "Edge: deadline violation": "edge_deadline_violation",
    "Edge: no 2-opt gain": "edge_no_2opt_gain",
}
MAIN_CHOICE = "Main dataset (seed 42)"
RANDOM_CHOICE = "Random instance"
DATASET_CHOICES = [MAIN_CHOICE, *EDGE_NAMES, RANDOM_CHOICE]
FLEET_SIZES = sorted(FLEETS)                       # 2..6 vehicles


# --------------------------------------------------------------------------
# Scenario -> instance
# --------------------------------------------------------------------------
def build_instance(choice: str = MAIN_CHOICE, seed: int = 1, n_points: int = 20, n_isolated: int = 1,
                   demand_ratio: float = 1.08, fleet_size: Optional[int] = None,
                   capacity_scale: float = 1.0, deadline_scale: float = 1.0) -> Instance:
    if choice == MAIN_CHOICE:
        inst = generate_instance()
    elif choice in EDGE_NAMES:
        inst = all_edge_cases()[EDGE_NAMES[choice]]
    elif choice == RANDOM_CHOICE:
        n_isolated = max(0, min(int(n_isolated), int(n_points) - 3))
        n_conn = int(n_points) - n_isolated
        inst = generate_instance(n_points=int(n_points), n_isolated=n_isolated,
                                 usable_roads=max(n_conn, int(1.8 * n_conn)), demand_ratio=float(demand_ratio),
                                 seed=int(seed), name=f"random_n{int(n_points)}_seed{int(seed)}")
    else:
        raise ValueError(f"unknown dataset: {choice!r}")
    if fleet_size is not None:
        if fleet_size not in FLEETS:
            raise ValueError(f"fleet size must be one of {FLEET_SIZES}")
        inst = with_fleet(inst, FLEETS[fleet_size], f"{inst.name}_fleet{fleet_size}")
    if abs(capacity_scale - 1.0) > 1e-9:
        inst = scale_capacity(inst, capacity_scale, f"{inst.name}_cap{capacity_scale:g}")
    if abs(deadline_scale - 1.0) > 1e-9:
        inst = scale_deadlines(inst, deadline_scale, f"{inst.name}_dl{deadline_scale:g}")
    return inst


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
@dataclass
class AppResult:
    inst: Instance
    sp: ShortestPaths
    cls: Dict[int, PointClass]
    greedy: GreedyResult
    final: TwoOptResult
    m_greedy: Metrics
    m_final: Metrics
    check: PlanCheck

    @property
    def blocked(self) -> int:
        return sum(i.rejected_deadline for i in self.final.improvements.values())


def run_all(inst: Instance, alpha: float = 1.0, beta: float = 0.5, max_iter: int = 1000) -> AppResult:
    sp = ShortestPaths(inst)
    cls = classify_points(inst, sp)
    g = greedy_construct(inst, sp, GreedyParams(alpha, beta))
    t = improve_plan(inst, sp, g, max_iter)
    mg = compute_metrics(inst, sp, g.plan(), "Greedy", g.runtime_ms, cls)
    mf = compute_metrics(inst, sp, t.plan(), "Greedy + 2-opt", g.runtime_ms + t.runtime_ms, cls)
    return AppResult(inst, sp, cls, g, t, mg, mf, check_plan(inst, sp, t.plan()))


# --------------------------------------------------------------------------
# Tables (plain list-of-dict so they are easy to test, display and export)
# --------------------------------------------------------------------------
def _n(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def stop_rows(inst: Instance, routes: Dict[int, VehicleRoute]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for vid in sorted(routes):
        r = routes[vid]
        load = 0
        for k, (j, a, c) in enumerate(zip(r.stops, r.arrivals, r.completions), start=1):
            p = inst.point(j)
            load += p.waste
            rows.append({"Vehicle": f"V{vid}", "Stop": k, "Point": p.name, "Waste (kg)": p.waste,
                         "Urgency": p.urgency, "Arrive (min)": round(a, 1), "Done (min)": round(c, 1),
                         "Deadline (min)": round(p.deadline, 1), "Slack (min)": round(p.deadline - c, 1),
                         "Load after (kg)": load})
    return rows


def vehicle_rows(res: AppResult) -> List[Dict[str, object]]:
    rows = []
    for v in res.inst.vehicles:
        g, f = res.greedy.routes[v.id], res.final.routes[v.id]
        if not f.stops:
            rows.append({"Vehicle": f"V{v.id}", "Capacity (kg)": v.capacity, "Stops": 0, "Load (kg)": 0, "Load (%)": 0.0,
                         "Route": "unused", "km Greedy": 0.0, "km 2-opt": 0.0, "Return (min)": None, "Limit (min)": v.max_duration})
            continue
        names = " -> ".join(res.inst.point(j).name for j in f.stops)
        rows.append({"Vehicle": f"V{v.id}", "Capacity (kg)": v.capacity, "Stops": len(f.stops), "Load (kg)": f.load,
                     "Load (%)": round(100.0 * f.load / v.capacity, 1), "Route": f"Depot -> {names} -> Depot",
                     "km Greedy": round(g.distance, 2), "km 2-opt": round(f.distance, 2),
                     "Return (min)": round(f.return_time, 1), "Limit (min)": v.max_duration})
    return rows


def outcome_rows(res: AppResult) -> List[Dict[str, object]]:
    owner = {j: vid for vid, r in res.final.routes.items() for j in r.stops}
    rows = []
    for p in res.inst.points:
        c = res.cls[p.id]
        if c.status == UNREACHABLE:
            cls_txt, out = "unreachable", "not attempted: no open-road path from the depot"
        elif c.status == INFEASIBLE:
            cls_txt = "individually infeasible"
            out = "not attempted: " + "; ".join(REASON_TEXT.get(r, r) for r in c.reasons)
        else:
            cls_txt = "eligible"
            if p.id in owner:
                out = f"collected by V{owner[p.id]}"
            else:
                why = res.greedy.uncollected.get(p.id, "")
                out = f"uncollected: {UNCOLLECTED_TEXT.get(why, why)}"
        rows.append({"Point": p.name, "Waste (kg)": p.waste, "Urgency": p.urgency, "Deadline (min)": round(p.deadline, 1),
                     "Initial class": cls_txt, "Outcome": out})
    return rows


def metrics_rows(mg: Metrics, mf: Metrics) -> List[Dict[str, str]]:
    def pair(label, a, b):
        return {"Metric": label, "Greedy": a, "Greedy + 2-opt": b}
    return [
        pair("Points collected (of reachable)", f"{mg.n_collected} / {mg.n_reachable}", f"{mf.n_collected} / {mf.n_reachable}"),
        pair("Waste collected (kg)", f"{mg.kg_collected:,}", f"{mf.kg_collected:,}"),
        pair("Eligible waste not collected (kg)", f"{mg.kg_uncollected_eligible:,}", f"{mf.kg_uncollected_eligible:,}"),
        pair("Individually infeasible waste (kg)", f"{mg.kg_infeasible:,}", f"{mf.kg_infeasible:,}"),
        pair("Unreachable waste (kg)", f"{mg.kg_unreachable:,}", f"{mf.kg_unreachable:,}"),
        pair("Objective W = sum urgency x waste", f"{mg.W:,}", f"{mf.W:,}"),
        pair("W as % of the maximum achievable", _n(mg.W_pct), _n(mf.W_pct)),
        pair("Urgent points served", f"{mg.urgent_collected} / {mg.urgent_eligible}", f"{mf.urgent_collected} / {mf.urgent_eligible}"),
        pair("Total distance (km)", _n(mg.distance_km, 2), _n(mf.distance_km, 2)),
        pair("Distance per 100 kg collected (km)", _n(mg.km_per_100kg, 2), _n(mf.km_per_100kg, 2)),
        pair("Vehicles used", f"{mg.vehicles_used} / {mg.n_vehicles}", f"{mf.vehicles_used} / {mf.n_vehicles}"),
        pair("Fleet utilisation (%)", _n(mg.fleet_utilization_pct), _n(mf.fleet_utilization_pct)),
        pair("Deadline compliance of served points (%)", _n(mg.deadline_compliance_pct), _n(mf.deadline_compliance_pct)),
        pair("Tightest deadline slack (min)", _n(mg.min_slack_min), _n(mf.min_slack_min)),
        pair("Execution time (ms)", _n(mg.runtime_ms, 2), _n(mf.runtime_ms, 2)),
    ]


def improvement_rows(res: AppResult) -> List[Dict[str, object]]:
    rows = []
    for vid, i in sorted(res.final.improvements.items()):
        rows.append({"Vehicle": f"V{vid}", "km before": round(i.before_km, 2), "km after": round(i.after_km, 2),
                     "Saved (km)": round(i.saved_km, 2), "Saved (%)": round(i.reduction_pct, 1), "Moves": len(i.moves),
                     "Stopped because": "no improving feasible move left" if i.converged else "iteration cap reached",
                     "Improving moves blocked by a deadline": i.rejected_deadline})
    return rows


def baseline_rows(inst: Instance, sp: Optional[ShortestPaths] = None) -> List[Dict[str, object]]:
    rows = []
    for m in compare_methods(inst, sp):
        rows.append({"Method": m.label, "Points": f"{m.n_collected} / {m.n_reachable}", "Waste (kg)": m.kg_collected,
                     "W": m.W, "Urgent served": f"{m.urgent_collected} / {m.urgent_eligible}",
                     "Distance (km)": round(m.distance_km, 2), "km per 100 kg": None if m.km_per_100kg is None else round(m.km_per_100kg, 2),
                     "Feasible": "yes" if m.feasible else "NO"})
    return rows


def exact_comparison(res: AppResult) -> Dict[str, object]:
    """Exact optimum of the same objective (<= 8 eligible points), compared with the pipeline."""
    n_elig = sum(c.status == ELIGIBLE for c in res.cls.values())
    if n_elig > DEFAULT_MAX_POINTS:
        return {"status": "skipped",
                "message": f"This scenario has {n_elig} eligible points; exact search is limited to {DEFAULT_MAX_POINTS} "
                           f"because its cost grows factorially. Choose a smaller random instance (8 points) to see the comparison."}
    ex = solve_exact(res.inst, res.sp)
    mf = res.m_final
    w_gap = 100.0 * (ex.W - mf.W) / ex.W if ex.W else 0.0
    d_gap = 100.0 * (mf.distance_km - ex.D) / ex.D if (mf.W == ex.W and ex.D) else None
    nm = lambda j: res.inst.point(j).name
    return {"status": "ok", "W_exact": ex.W, "km_exact": ex.D, "W_pipeline": mf.W, "km_pipeline": mf.distance_km,
            "w_gap_pct": w_gap, "d_gap_pct": d_gap,
            "optimal": not is_better((ex.W, ex.D), (mf.W, mf.distance_km)),
            "orderings": ex.sequences_explored, "runtime_ms": ex.runtime_ms,
            "plan": {f"V{v}": [nm(j) for j in s] for v, s in sorted(ex.plan.items())}}


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_json(inst: Instance, alpha: float, beta: float, max_iter: int, with_exact: bool = False) -> dict:
    """The same dictionary `python main.py` writes (its console output is discarded)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return cli_main.run(inst, GreedyParams(alpha, beta), max_iter, with_exact)


def rows_to_csv(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
