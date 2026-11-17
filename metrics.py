"""
metrics.py  --  evaluation metrics for any plan (Greedy, Greedy + 2-opt, baselines)

Every number is recomputed from the plan by the INDEPENDENT checker (feasibility.py),
not read from an algorithm's own bookkeeping.

Metrics from the proposal
-------------------------
  total travel distance        distance_km
  collection coverage          coverage_pct  = collected points / reachable points
  waste collected              kg_collected, waste_pct = kg collected / reachable kg
  vehicle utilization          fleet_utilization_pct = total load / total fleet capacity
                               used_utilization_pct  = total load / capacity of vehicles used
  deadline compliance          deadline_compliance_pct = served points finished on time / served
                               (100 by construction when the plan is feasible) plus
                               urgent_coverage_pct = urgent eligible points collected / urgent eligible
                               (urgent = urgency >= 8) and min_slack_min = tightest deadline slack
  execution time               runtime_ms (supplied by the caller)
  global objective             W = sum(urgency x waste) and W_pct = W / W achievable by ALL eligible points

Where the waste goes (kept distinct on purpose)
-----------------------------------------------
  kg_unreachable            waste at points with no open-road path
  kg_infeasible             reachable but individually infeasible (oversized / deadline / duration)
  kg_uncollected_eligible   eligible points the plan did not serve (capacity, deadline, duration)
  kg_collected              served
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from data_generator import Instance
from feasibility import check_plan
from problem import (ELIGIBLE, EPS, INFEASIBLE, UNREACHABLE, Plan, PointClass, classify_points)
from algorithms.shortest_path import ShortestPaths

URGENT_THRESHOLD = 8        # urgency >= 8 counts as urgent (these points get the tight deadlines)


@dataclass
class VehicleMetrics:
    vehicle_id: int
    capacity: int
    load: int
    utilization_pct: float
    distance_km: float
    return_time_min: float
    stops: int


@dataclass
class Metrics:
    label: str
    feasible: bool
    n_points: int
    n_reachable: int
    n_vehicles: int
    n_unreachable: int
    n_infeasible: int
    n_eligible: int
    n_collected: int
    n_uncollected_eligible: int
    kg_total: int
    kg_unreachable: int
    kg_infeasible: int
    kg_uncollected_eligible: int
    kg_collected: int
    coverage_pct: float                 # of reachable points
    waste_pct: float                    # of reachable kg
    W: int
    W_pct: float                        # of the W all eligible points would give
    urgent_eligible: int
    urgent_collected: int
    urgent_coverage_pct: Optional[float]
    vehicles_used: int
    fleet_utilization_pct: float
    used_utilization_pct: Optional[float]
    distance_km: float
    km_per_100kg: Optional[float]
    deadline_compliance_pct: Optional[float]
    min_slack_min: Optional[float]
    runtime_ms: Optional[float] = None
    per_vehicle: List[VehicleMetrics] = field(default_factory=list)

    def row(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("per_vehicle")
        return d


def _pct(a: float, b: float) -> Optional[float]:
    return 100.0 * a / b if b else None


def compute_metrics(inst: Instance, sp: ShortestPaths, plan: Plan, label: str = "",
                    runtime_ms: Optional[float] = None,
                    classification: Optional[Dict[int, PointClass]] = None) -> Metrics:
    cls = classification or classify_points(inst, sp)
    chk = check_plan(inst, sp, plan)
    kg = lambda ids: sum(inst.point(j).waste for j in ids)
    ids_of = lambda status: [j for j, c in cls.items() if c.status == status]
    unreachable, infeasible, eligible = ids_of(UNREACHABLE), ids_of(INFEASIBLE), ids_of(ELIGIBLE)
    served = chk.served
    served_set = set(served)
    uncollected = [j for j in eligible if j not in served_set]

    reachable_ids = infeasible + eligible
    kg_reachable = kg(reachable_ids)
    w_all_eligible = sum(inst.point(j).urgency * inst.point(j).waste for j in eligible)

    urgent_elig = [j for j in eligible if inst.point(j).urgency >= URGENT_THRESHOLD]
    urgent_got = [j for j in urgent_elig if j in served_set]

    per_vehicle: List[VehicleMetrics] = []
    on_time, slacks = 0, []
    veh = {v.id: v for v in inst.vehicles}
    for vid, rc in chk.routes.items():
        cap = veh[vid].capacity
        per_vehicle.append(VehicleMetrics(vid, cap, rc.load, 100.0 * rc.load / cap, rc.distance,
                                          rc.return_time, len(rc.stops)))
        for j, done in zip(rc.stops, rc.completions):
            slack = inst.point(j).deadline - done
            slacks.append(slack)
            on_time += slack >= -EPS
    load_used = sum(m.load for m in per_vehicle)
    cap_used = sum(m.capacity for m in per_vehicle)
    kg_collected = kg(served)

    return Metrics(
        label=label, feasible=chk.ok, n_points=len(inst.points),
        n_reachable=len(reachable_ids), n_vehicles=len(inst.vehicles),
        n_unreachable=len(unreachable), n_infeasible=len(infeasible), n_eligible=len(eligible),
        n_collected=len(served), n_uncollected_eligible=len(uncollected),
        kg_total=kg([p.id for p in inst.points]), kg_unreachable=kg(unreachable),
        kg_infeasible=kg(infeasible), kg_uncollected_eligible=kg(uncollected),
        kg_collected=kg_collected,
        coverage_pct=_pct(len(served), len(reachable_ids)) or 0.0,
        waste_pct=_pct(kg_collected, kg_reachable) or 0.0,
        W=chk.waste_value, W_pct=_pct(chk.waste_value, w_all_eligible) or 0.0,
        urgent_eligible=len(urgent_elig), urgent_collected=len(urgent_got),
        urgent_coverage_pct=_pct(len(urgent_got), len(urgent_elig)),
        vehicles_used=len(per_vehicle),
        fleet_utilization_pct=100.0 * load_used / sum(v.capacity for v in inst.vehicles),
        used_utilization_pct=_pct(load_used, cap_used),
        distance_km=chk.distance,
        km_per_100kg=(100.0 * chk.distance / kg_collected) if kg_collected else None,
        deadline_compliance_pct=_pct(on_time, len(served)),
        min_slack_min=min(slacks) if slacks else None,
        runtime_ms=runtime_ms, per_vehicle=per_vehicle)


def format_metrics(m: Metrics) -> str:
    f = lambda x, fmt="{:.1f}": "n/a" if x is None else fmt.format(x)
    pc = lambda x: "n/a" if x is None else f"{x:.1f}%"
    L = [f"--- Metrics: {m.label} ({'feasible' if m.feasible else 'INFEASIBLE'}) ---",
         f"  Distance            : {m.distance_km:.2f} km   ({f(m.km_per_100kg, '{:.2f}')} km per 100 kg collected)",
         f"  Coverage            : {m.n_collected}/{m.n_reachable} reachable points ({m.coverage_pct:.1f}%)",
         f"  Waste collected     : {m.kg_collected} kg ({m.waste_pct:.1f}% of reachable demand)",
         f"  Waste not collected : {m.kg_unreachable} kg unreachable | {m.kg_infeasible} kg individually infeasible"
         f" | {m.kg_uncollected_eligible} kg eligible-but-uncollected",
         f"  Objective W         : {m.W} ({m.W_pct:.1f}% of the W all eligible points would give)",
         f"  Urgent coverage     : {m.urgent_collected}/{m.urgent_eligible} ({pc(m.urgent_coverage_pct)})",
         f"  Deadline compliance : {pc(m.deadline_compliance_pct)} of served points on time,"
         f" tightest slack {f(m.min_slack_min)} min",
         f"  Vehicles            : {m.vehicles_used}/{m.n_vehicles} used,"
         f" fleet utilization {m.fleet_utilization_pct:.1f}%, of used vehicles {pc(m.used_utilization_pct)}"]
    if m.runtime_ms is not None:
        L.append(f"  Execution time      : {m.runtime_ms:.2f} ms")
    return "\n".join(L)
