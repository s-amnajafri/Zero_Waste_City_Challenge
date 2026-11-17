"""
problem.py  --  shared formal definitions for the Zero-Waste City Challenge

Greedy, 2-opt, the brute-force solver and the feasibility checker ALL import
their constraint tolerance, point classification and objective from here, so
the heuristic and the exact solver are guaranteed to optimise the same thing
under the same rules.

Formal model
------------
Nodes 0..n (0 = depot).  d(i,j) = shortest open-road distance, tau(i,j) = d/v*60 min.
Point j: waste q_j, urgency u_j in 1..10, service time s_j, deadline D_j.
Vehicle k: capacity Q_k, maximum route duration T_k.

PLAN = for each vehicle k an ordered list of stops R_k = (j_1, ..., j_r), r >= 0.
Timeline of a non-empty route (vehicle leaves the depot at t = 0):
    arrival_1    = tau(0, j_1)
    completion_i = arrival_i + s_{j_i}
    arrival_{i+1}= completion_i + tau(j_i, j_{i+1})
    return time  = completion_r + tau(j_r, 0)

A plan is FEASIBLE iff
  (C1) capacity    : sum_i q_{j_i} <= Q_k                         for every vehicle k
  (C2) deadline    : completion_i  <= D_{j_i}                     for every stop
  (C3) duration    : return time   <= T_k                         for every vehicle k
  (C4) accessible  : every leg (a,b) has a finite d(a,b) over OPEN roads only
  (C5) uniqueness  : every point appears at most once in the whole plan
  (C6) fleet       : only the given vehicles are used, each at most once

GLOBAL OBJECTIVE (lexicographic)
  1. maximise  W(plan) = sum over served points p of  u_p * q_p     (urgency-weighted waste)
  2. among plans with equal W, minimise  D(plan) = sum_k dist(0, R_k, 0)   (total km)

  W is an integer, so "equal W" is exact.  Distances are floats compared with EPS.
  Greedy chooses WHICH points are served (and so fixes W); 2-opt only reorders
  stops inside routes (W unchanged, D reduced); the brute-force solver searches
  all feasible plans for the lexicographic optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from data_generator import Instance
from algorithms.shortest_path import ShortestPaths

URGENCY_MAX = 10          # urgency scale is 1..10
EPS = 1e-9                # tolerance for floating-point comparisons

Plan = Dict[int, List[int]]   # vehicle id -> ordered list of collection-point ids (no depot)

UNREACHABLE = "unreachable"
INFEASIBLE = "individually_infeasible"
ELIGIBLE = "eligible"

REASON_TEXT = {
    "oversized": "waste exceeds the capacity of every vehicle",
    "deadline": "cannot be completed before its deadline even if served first",
    "route_duration": "a round trip plus service exceeds every vehicle's duration limit",
    "no_vehicle_meets_all_limits": "no single vehicle satisfies capacity and duration together",
    "no open-road path from the depot": "no open-road path from the depot",
}


def leq(a: float, b: float) -> bool:
    """a <= b with floating-point tolerance."""
    return a <= b + EPS


# --------------------------------------------------------------------------
# Initial point classification (pre-screening)
# --------------------------------------------------------------------------
@dataclass
class PointClass:
    status: str                                   # UNREACHABLE / INFEASIBLE / ELIGIBLE
    reasons: List[str] = field(default_factory=list)


def classify_points(inst: Instance, sp: ShortestPaths) -> Dict[int, PointClass]:
    """
    Split points into three disjoint initial classes.

      UNREACHABLE : no open-road path from the depot.
      INFEASIBLE  : reachable, but NO vehicle could serve it even as a one-stop route
                    (checked on d(0,j) only; this is necessary, not sufficient, for
                    route-level feasibility -- Greedy re-checks with real arrival times).
      ELIGIBLE    : passes pre-screening; handed to Greedy.

    Runs in O(n * k): O(1) per point plus one scan of the fleet.
    """
    max_cap = max(v.capacity for v in inst.vehicles)
    max_dur = max(v.max_duration for v in inst.vehicles)
    out: Dict[int, PointClass] = {}
    for p in inst.points:
        if not sp.reachable(0, p.id):
            out[p.id] = PointClass(UNREACHABLE, ["no open-road path from the depot"])
            continue
        done = sp.travel_time(0, p.id) + p.service_time        # completion if served first
        back = sp.travel_time(p.id, 0)
        reasons: List[str] = []
        if p.waste > max_cap:
            reasons.append("oversized")
        if not leq(done, p.deadline):
            reasons.append("deadline")
        if not leq(done + back, max_dur):
            reasons.append("route_duration")
        if not reasons and not any(p.waste <= v.capacity and leq(done + back, v.max_duration)
                                   for v in inst.vehicles):
            reasons.append("no_vehicle_meets_all_limits")
        out[p.id] = PointClass(INFEASIBLE if reasons else ELIGIBLE, reasons)
    return out


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def urgency_weighted_waste(inst: Instance, plan: Plan) -> int:
    """W(plan) = sum over served points of urgency * waste."""
    return sum(inst.point(j).urgency * inst.point(j).waste
               for stops in plan.values() for j in stops)


def plan_distance(sp: ShortestPaths, plan: Plan) -> float:
    """D(plan) = total km over all non-empty routes (depot -> stops -> depot)."""
    return sum(sp.route_distance([0] + list(s) + [0]) for s in plan.values() if s)


def objective(inst: Instance, sp: ShortestPaths, plan: Plan) -> Tuple[int, float]:
    """(W, D).  Better plan = larger W; if W ties, smaller D."""
    return urgency_weighted_waste(inst, plan), plan_distance(sp, plan)


def is_better(a: Tuple[int, float], b: Tuple[int, float]) -> bool:
    """True iff objective a is strictly better than b under the lexicographic rule."""
    if a[0] != b[0]:
        return a[0] > b[0]
    return a[1] < b[1] - EPS
