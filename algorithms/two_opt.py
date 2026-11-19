"""
two_opt.py  --  Algorithm 2: 2-opt local search (intra-route)

SPECIFICATION
=============
Works on ONE vehicle's route at a time and only REORDERS its stops.
  * The served set is unchanged  -> W (urgency-weighted waste) and the load are unchanged,
    so capacity (C1) can never be violated.
  * Only the total distance D can change; it can only go down.

Route stops s_1..s_r with s_0 = s_{r+1} = depot.  A move (a, b), 1 <= a < b <= r,
reverses the segment s_a..s_b.  It removes edges (s_{a-1}, s_a), (s_b, s_{b+1}) and adds
(s_{a-1}, s_b), (s_a, s_{b+1}):

    delta(a,b) = d(s_{a-1}, s_b) + d(s_a, s_{b+1}) - d(s_{a-1}, s_a) - d(s_b, s_{b+1})

Distances are symmetric, so delta is an O(1) table lookup.

A move is ACCEPTED only if
  (i)  it strictly reduces distance:  delta < -EPS   (EPS = 1e-9 km, so rounding noise
       can never cause an endless loop), and
  (ii) the reversed route is still feasible: arrival times of every stop from position a
       onward change, so deadlines (C2) and the duration limit (C3) are re-simulated.
Best-improvement: each iteration applies the single most-reducing feasible move; ties go
to the smallest (a, b), so the result is deterministic.

Stopping: no feasible improving move exists (route is 2-optimal), or a computational cap
(max_iter applied moves per route, default 1000) is reached; the result records which.

Note on C3: the vehicle never waits, so return time = total travel time + total service
time. Reordering leaves service time unchanged and an improving move lowers travel time,
so an improving move can never break the duration limit. The check is kept as a cheap
guard, and the tests confirm the implication. Only deadlines (C2) can block a move.

PSEUDOCODE
==========
  TWO_OPT(route s_1..s_r of vehicle k, max_iter):
      require the input route is feasible
      loop:
          best <- none
          for a in 1..r-1:  for b in a+1..r:
              delta <- d(s[a-1],s[b]) + d(s[a],s[b+1]) - d(s[a-1],s[a]) - d(s[b],s[b+1])
              if delta < -EPS and (best is none or delta < best.delta - EPS):
                  if FEASIBLE(route with s[a..b] reversed):      # re-simulate clock, check C2, C3
                      best <- (delta, a, b)
          if best is none:            return route, converged          # 2-optimal
          if moves applied = max_iter: return route, NOT converged     # cap reached
          reverse s[a..b]; record move

TERMINATION AND GUARANTEE
=========================
  Every applied move lowers the route length by more than EPS, so no ordering can repeat;
  the number of orderings (r!) is finite, hence 2-opt terminates. The result is 2-optimal
  (no single FEASIBLE reversal improves it): a local optimum, NOT a global one.
  A polynomial bound on the number of iterations is not guaranteed (that is why the cap
  exists); the measured number of iterations is reported.

COMPLEXITY  (r = stops on a route, I = applied moves + 1 final scan)
==========
  One scan: r(r-1)/2 delta evaluations, O(1) each            -> O(r^2)
            + F feasibility checks at O(r) each, where F <= r(r-1)/2 counts improving
              candidates that beat the best feasible one so far -> O(F r) <= O(r^3) worst case
  One route : O(I (r^2 + F r)), typically O(I r^2).
  Whole plan: sum over routes; with sum r_k <= n this is O(I n^3) worst case and typically
              O(I sum r_k^2). Greedy yields r ~ n/k, so the r^2 terms stay small.
  Space     : O(r) per route (in-place segment reversal / one candidate copy) -> O(n) overall,
              beyond the shared distance matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from data_generator import Instance, Vehicle
from algorithms.greedy import GreedyResult, VehicleRoute
from problem import EPS, Plan, leq
from algorithms.shortest_path import ShortestPaths

DEFAULT_MAX_ITER = 1000


# --------------------------------------------------------------------------
# Feasibility of a candidate order (re-simulates the clock from the depot)
# a shorter route is not necessarily a feasible route. So, check feasibility after proposed reversed route
# --------------------------------------------------------------------------

def _violation(inst: Instance, sp: ShortestPaths, veh: Vehicle, stops: Sequence[int]) -> Optional[str]:
    """None if the order meets C2 (completion deadlines) and C3 (duration); else the first violated rule."""
    t, pos = 0.0, 0        #initialise time and position at depot
    for j in stops:
        p = inst.point(j)
        t += inst.travel_time(sp.dist(pos, j)) + p.service_time    #update time
        if not leq(t, p.deadline):       #check the deadline
            return "deadline"
        pos = j                          #update position of vehicle
    t += inst.travel_time(sp.dist(pos, 0))     #check return journey
    if not leq(t, veh.max_duration):
        return "route_duration"
    return None              #proposed reversed route is feasible


def _rebuild(inst: Instance, sp: ShortestPaths, veh: Vehicle, stops: List[int]) -> VehicleRoute:
    """Recompute arrivals, completions, load, distance and return time for an order."""
    if not stops:
        return VehicleRoute(veh.id, veh.capacity, veh.max_duration, [], [], [], 0, 0.0, 0.0)
    t, pos, load = 0.0, 0, 0
    arrivals: List[float] = []
    completions: List[float] = []
    for j in stops:
        p = inst.point(j)
        t += inst.travel_time(sp.dist(pos, j))
        arrivals.append(t)
        t += p.service_time
        completions.append(t)
        load += p.waste
        pos = j
    t += inst.travel_time(sp.dist(pos, 0))
    dist = sp.route_distance([0] + list(stops) + [0])
    return VehicleRoute(veh.id, veh.capacity, veh.max_duration, list(stops),
                        arrivals, completions, load, dist, t)


# --------------------------------------------------------------------------
# Result Classes
# --------------------------------------------------------------------------
@dataclass
class RouteImprovement:     #stores the improvement information for one vehicle.
    vehicle_id: int
    before: List[int]
    after: List[int]
    before_km: float
    after_km: float
    moves: List[Tuple[int, int, float]] = field(default_factory=list)   # (a, b, delta)
    converged: bool = True
    rejected_deadline: int = 0          # improving candidates rejected because of a deadline
    rejected_duration: int = 0          # ... because of the duration limit (always 0, see note)
    delta_evaluations: int = 0
    feasibility_checks: int = 0

    @property
    def saved_km(self) -> float:   #Dbefore - Dafter
        return self.before_km - self.after_km

    @property
    def reduction_pct(self) -> float:
        return 100.0 * self.saved_km / self.before_km if self.before_km > 0 else 0.0


@dataclass
class TwoOptResult:
    routes: Dict[int, VehicleRoute]
    improvements: Dict[int, RouteImprovement]
    total_before: float
    total_after: float
    runtime_ms: float

    @property
    def reduction_pct(self) -> float:
        return 100.0 * (self.total_before - self.total_after) / self.total_before if self.total_before > 0 else 0.0

    @property
    def all_converged(self) -> bool:        
        return all(i.converged for i in self.improvements.values())

    def plan(self) -> Plan:
        return {vid: list(r.stops) for vid, r in self.routes.items() if r.stops}


# --------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------
def two_opt_route(inst: Instance, sp: ShortestPaths, veh: Vehicle, stops: Sequence[int],
                  max_iter: int = DEFAULT_MAX_ITER) -> Tuple[List[int], RouteImprovement]:
    """Improve one route; returns (new stop order, statistics)."""
    cur = list(stops)            #current route
    why = _violation(inst, sp, veh, cur)
    if why is not None:
        raise ValueError(f"input route of vehicle {veh.id} is not feasible ({why})")

    r = len(cur)      #number of stops on the route
    before_km = sp.route_distance([0] + cur + [0]) if cur else 0.0     #og distance
    imp = RouteImprovement(veh.id, list(cur), list(cur), before_km, before_km)    #object that'll store improved stats
    d = sp.dist

    #seaeching for improved route by reversing segments of the current route. The outer loop continues until no more feasible improvements can be found or the maximum number of iterations is reached.
    while True:
        nodes = [0] + cur + [0]
        best = None                                   # (delta, a, b, candidate order)
        #check for every possible reversal move of 2-opt
        for a in range(1, r):
            for b in range(a + 1, r + 1):
                imp.delta_evaluations += 1
                delta = (d(nodes[a - 1], nodes[b]) + d(nodes[a], nodes[b + 1])     #delta = change in distance if segment s[a..b] is reversed
                         - d(nodes[a - 1], nodes[a]) - d(nodes[b], nodes[b + 1]))
                if delta >= -EPS:                     # not an improvement (for eg change of -0.0000000001 km is just rounding noise)
                    continue
                if best is not None and delta >= best[0] - EPS:   # not better than current best  (if delta -ve, newer route is shorter else reject)
                    continue
                #create the proposed route (reversing a section of a list using list slicing)
                cand = cur[:a - 1] + cur[a - 1:b][::-1] + cur[b:]
                
                imp.feasibility_checks += 1
                #feasibility check of proposed route
                why = _violation(inst, sp, veh, cand)
                if why == "deadline":
                    imp.rejected_deadline += 1
                    continue
                if why == "route_duration":
                    imp.rejected_duration += 1
                    continue
                best = (delta, a, b, cand)       #save the best feasible move
        if best is None:
            imp.converged = True      #The algorithm finds improvements, applies them, and eventually finds no more feasible improvements. (local optimum reached)
            break
        if len(imp.moves) >= max_iter:
            imp.converged = False     #The algorithm has already applied 1000 improvements, but another feasible improvement still exists. It stops because it has reached its limit (found the improved route but not the local optimum).
            break
        delta, a, b, cur = best
        imp.moves.append((a, b, delta))     #keep appending eac moves inorder to keep track of no. of moves to compare with max iterations

    imp.after = list(cur)
    imp.after_km = sp.route_distance([0] + cur + [0]) if cur else 0.0
    return cur, imp


def improve_plan(inst: Instance, sp: ShortestPaths, greedy: GreedyResult,
                 max_iter: int = DEFAULT_MAX_ITER) -> TwoOptResult:
    """Run 2-opt on every route of a Greedy result."""
    t0 = time.perf_counter()
    vehicles = {v.id: v for v in inst.vehicles}
    routes: Dict[int, VehicleRoute] = {}
    improvements: Dict[int, RouteImprovement] = {}
    before = after = 0.0
    for vid, route in greedy.routes.items():
        veh = vehicles[vid]
        if not route.stops:
            routes[vid] = route
            continue
        new_stops, imp = two_opt_route(inst, sp, veh, route.stops, max_iter)
        routes[vid] = _rebuild(inst, sp, veh, new_stops)
        improvements[vid] = imp
        before += imp.before_km
        after += imp.after_km
    return TwoOptResult(routes, improvements, before, after, (time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def format_report(inst: Instance, res: TwoOptResult) -> str:
    nm = lambda j: inst.point(j).name
    seq = lambda s: " -> ".join(["Depot"] + [nm(j) for j in s] + ["Depot"])
    L = [f"=== 2-opt improvement on '{inst.name}' ==="]
    for vid in sorted(res.improvements):
        i = res.improvements[vid]
        L.append(f"  V{vid}: {i.before_km:7.2f} km -> {i.after_km:7.2f} km   "
                 f"saved {i.saved_km:5.2f} km ({i.reduction_pct:4.1f}%)   "
                 f"moves={len(i.moves)}  {'converged' if i.converged else 'STOPPED AT CAP'}")
        if i.moves:
            L.append(f"        before: {seq(i.before)}")
            L.append(f"        after : {seq(i.after)}")
            L.append("        moves : " + ", ".join(f"reverse positions {a}..{b} ({dl:+.2f} km)"
                                                   for a, b, dl in i.moves))
        if i.rejected_deadline or i.rejected_duration:
            L.append(f"        blocked: {i.rejected_deadline} improving move(s) rejected for a deadline, "
                     f"{i.rejected_duration} for the duration limit")
    L.append(f"Total distance : {res.total_before:.2f} km -> {res.total_after:.2f} km   "
             f"reduction {res.reduction_pct:.2f}%")
    evals = sum(i.delta_evaluations for i in res.improvements.values())
    checks = sum(i.feasibility_checks for i in res.improvements.values())
    L.append(f"Work           : {evals} delta evaluations, {checks} feasibility checks, "
             f"{res.runtime_ms:.2f} ms, {'all routes converged' if res.all_converged else 'cap reached'}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import all_edge_cases, generate_instance
    from feasibility import check_plan
    from algorithms.greedy import greedy_construct
    from problem import objective

    for inst in [generate_instance()] + list(all_edge_cases().values()):
        sp = ShortestPaths(inst)
        g = greedy_construct(inst, sp)
        res = improve_plan(inst, sp, g)
        print(format_report(inst, res))
        chk = check_plan(inst, sp, res.plan())
        W0, _ = objective(inst, sp, g.plan())
        print(f"Independent checker: {'PASS' if chk.ok else 'FAIL'}   "
              f"W before/after = {W0}/{chk.waste_value} (unchanged: {W0 == chk.waste_value})   "
              f"D = {chk.distance:.2f} km")
        print()
