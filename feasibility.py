"""
feasibility.py  --  independent plan checker

Verifies constraints C1-C6 (see problem.py) for ANY plan, from scratch.
It shares no state with Greedy / 2-opt / brute force: arrival times, loads and
distances are recomputed here, and every leg's distance is rebuilt from the
actual open roads along the shortest path (not read from the distance matrix),
so a bug in the algorithms or in the distance bookkeeping cannot hide.

Pseudocode
----------
  CHECK(plan):
      for each vehicle k with stops R_k:
          flag unknown vehicle / invalid point / point already used elsewhere (C5, C6)
          t <- 0; load <- 0
          for each leg (a, b) along depot -> R_k -> depot:
              path <- shortest open-road path a..b;  none => violation (C4)
              d    <- sum of road weights along path (each hop must be an OPEN road)
              t    <- t + d / speed * 60
              if b is a point: load += q_b;  t <- t + s_b;  need t <= D_b   (C2)
          need load <= Q_k (C1);  need t (return time) <= T_k (C3)

Complexity: O(total stops * path length) time, O(n + roads) space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data_generator import Instance
from problem import Plan, leq
from algorithms.shortest_path import ShortestPaths


@dataclass
class RouteCheck:
    vehicle_id: int
    stops: List[int]
    load: int = 0
    distance: float = 0.0
    arrivals: List[float] = field(default_factory=list)
    completions: List[float] = field(default_factory=list)
    return_time: float = 0.0


@dataclass
class PlanCheck:
    ok: bool
    violations: List[str]
    routes: Dict[int, RouteCheck]
    served: List[int]
    waste_value: int        # W = sum urgency * waste of served points
    distance: float         # D = total km


def check_plan(inst: Instance, sp: ShortestPaths, plan: Plan) -> PlanCheck:
    violations: List[str] = []
    vehicles = {v.id: v for v in inst.vehicles}
    n = len(inst.points)

    open_w: Dict[frozenset, float] = {}
    for r in inst.usable_roads():
        key = frozenset((r.u, r.v))
        open_w[key] = min(open_w.get(key, float("inf")), r.distance)

    def leg_distance(a: int, b: int, where: str) -> Optional[float]:
        path = sp.path(a, b)
        if path is None:
            violations.append(f"C4 {where}: no open-road path {a}->{b}")
            return None
        total = 0.0
        for x, y in zip(path, path[1:]):
            w = open_w.get(frozenset((x, y)))
            if w is None:
                violations.append(f"C4 {where}: leg {a}->{b} uses a closed/missing road {x}-{y}")
                return None
            total += w
        return total

    owner: Dict[int, int] = {}
    checks: Dict[int, RouteCheck] = {}
    served: List[int] = []

    for vid, stops in plan.items():
        where = f"vehicle {vid}"
        if vid not in vehicles:
            violations.append(f"C6 {where}: not in the fleet")
            continue
        stops = list(stops)
        valid = True
        for p in stops:
            if not (isinstance(p, int) and 1 <= p <= n):
                violations.append(f"C5 {where}: invalid point id {p}")
                valid = False
            elif p in owner:
                violations.append(f"C5 {where}: point {p} already served by vehicle {owner[p]}"
                                  if owner[p] != vid else f"C5 {where}: point {p} repeated")
            else:
                owner[p] = vid
        if not valid or not stops:
            continue

        veh = vehicles[vid]
        rc = RouteCheck(vid, stops)
        nodes = [0] + stops + [0]
        t = 0.0
        broken = False
        for a, b in zip(nodes, nodes[1:]):
            d = leg_distance(a, b, where)
            if d is None:
                broken = True
                break
            rc.distance += d
            t += inst.travel_time(d)
            if b != 0:
                pt = inst.point(b)
                rc.load += pt.waste
                rc.arrivals.append(t)
                t += pt.service_time
                rc.completions.append(t)
                if not leq(t, pt.deadline):
                    violations.append(f"C2 {where}: {pt.name} completes at {t:.2f} min "
                                      f"> deadline {pt.deadline:.0f}")
        if broken:
            continue
        rc.return_time = t
        if rc.load > veh.capacity:
            violations.append(f"C1 {where}: load {rc.load} kg > capacity {veh.capacity} kg")
        if not leq(t, veh.max_duration):
            violations.append(f"C3 {where}: returns at {t:.2f} min > limit {veh.max_duration:.0f} min")
        checks[vid] = rc
        served.extend(stops)

    w = sum(inst.point(p).urgency * inst.point(p).waste for p in served)
    dist = sum(rc.distance for rc in checks.values())
    return PlanCheck(not violations, violations, checks, sorted(served), w, dist)
