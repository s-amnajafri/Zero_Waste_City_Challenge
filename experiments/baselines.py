"""
baselines.py  --  reference methods to compare the Greedy + 2-opt pipeline against

A baseline is only informative if it obeys the SAME rules, so every method here uses the
same classification, the same eligibility checks (capacity, completion deadline, duration)
and is scored by the same independent metrics.

  first_fit        naive reference: take eligible points in id order; give each point to the
                   first vehicle (by id) that can feasibly append it. No scoring at all.
                   It shows what the Greedy score is worth.
  distance-only    Greedy with alpha = 0   (pure nearest-neighbour style)
  urgency-only     Greedy with beta  = 0   (pure urgency, ignores distance)
  default          Greedy with alpha = 1, beta = 0.5   (the proposed method)
Each construction can then be improved by 2-opt, which separates the effect of the
construction rule from the effect of the improvement step.

Pseudocode (first_fit)
----------------------
  FIRST_FIT:  for j in eligible points in increasing id:
                  for k in vehicles in increasing id:
                      if E1, E2, E3 hold for (k, j): append j to route_k; break
  Time O(n k) (n eligible points, k vehicles; O(1) per check), space O(n + k).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from data_generator import Instance
from algorithms.greedy import GreedyParams, GreedyResult, VehicleRoute, _State, _evaluate, greedy_construct
from metrics import Metrics, compute_metrics
from problem import ELIGIBLE, Plan, classify_points
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import _rebuild, improve_plan


@dataclass
class ConstructionResult:
    """Common shape for any construction method: routes + the time it took."""
    routes: Dict[int, VehicleRoute]
    runtime_ms: float

    def plan(self) -> Plan:
        return {vid: list(r.stops) for vid, r in self.routes.items() if r.stops}


def first_fit(inst: Instance, sp: Optional[ShortestPaths] = None) -> ConstructionResult:
    t0 = time.perf_counter()
    sp = sp or ShortestPaths(inst)
    cls = classify_points(inst, sp)
    states = {v.id: _State(v) for v in inst.vehicles}
    for j in sorted(pid for pid, c in cls.items() if c.status == ELIGIBLE):
        for vid in sorted(states):
            st = states[vid]
            fail, _, completion = _evaluate(inst, sp, st, j)
            if fail is None:
                st.stops.append(j)
                st.load += inst.point(j).waste
                st.pos, st.clock = j, completion
                break
    routes = {vid: _rebuild(inst, sp, st.v, st.stops) for vid, st in states.items()}
    return ConstructionResult(routes, (time.perf_counter() - t0) * 1000)


def from_greedy(res: GreedyResult) -> ConstructionResult:
    return ConstructionResult(res.routes, res.runtime_ms)


METHODS = [
    ("first-fit (id order)",        lambda inst, sp: first_fit(inst, sp)),
    ("greedy: distance-only a=0",   lambda inst, sp: from_greedy(greedy_construct(inst, sp, GreedyParams(0.0, 1.0)))),
    ("greedy: urgency-only b=0",    lambda inst, sp: from_greedy(greedy_construct(inst, sp, GreedyParams(1.0, 0.0)))),
    ("greedy: default a=1,b=0.5",   lambda inst, sp: from_greedy(greedy_construct(inst, sp, GreedyParams()))),
]


def compare_methods(inst: Instance, sp: Optional[ShortestPaths] = None) -> List[Metrics]:
    """Metrics for every construction method, each with and without 2-opt."""
    sp = sp or ShortestPaths(inst)
    cls = classify_points(inst, sp)
    out: List[Metrics] = []
    for name, build in METHODS:
        built = build(inst, sp)
        out.append(compute_metrics(inst, sp, built.plan(), name, built.runtime_ms, cls))
        improved = improve_plan(inst, sp, built)          # needs only `.routes`
        out.append(compute_metrics(inst, sp, improved.plan(), name + " + 2-opt",
                                   built.runtime_ms + improved.runtime_ms, cls))
    return out
