"""
greedy.py  --  Algorithm 1: global greedy route construction

SPECIFICATION
=============
State of vehicle k : position i (last stop, 0 = depot), load L_k, departure time t_k
                     (= completion time at i; 0 at the depot).

Eligibility of pair (k, j) for an unassigned candidate j  (checked in this order):
  E1 capacity   : L_k + q_j <= Q_k
  E2 deadline   : completion c = t_k + tau(i,j) + s_j  <=  D_j        (COMPLETION deadline)
  E3 duration   : c + tau(j,0) <= T_k                                  (can still get home)
  (E0 reachability is implied: candidates are reachable from the depot.)
  E2 uses the vehicle's ACTUAL running clock, so earlier travel and service
  are included; reachability of j from the depot alone is not enough.

Additional distance (detour of inserting j before the return trip):
      extra(i,j) = d(i,j) + d(j,0) - d(i,0)          (= 2 d(0,j) for an empty vehicle)
  Shortest-path distances obey the triangle inequality, so 0 <= extra <= 2 d(0,j).

Priority score (maximised):
      score(i,j) = alpha * (u_j / 10)  -  beta * (extra(i,j) / D_max)
  alpha = 1.0, beta = 0.5 by default.  D_max = largest d(0,p) over reachable points;
  if D_max = 0 the distance term is defined as 0 (no division by zero).
  The score may be negative.  There is NO score threshold: the only stopping
  condition is "no feasible (vehicle, point) pair remains".

GLOBAL selection: at every iteration ALL feasible (vehicle, point) pairs are
evaluated, the best pair is chosen, that vehicle is updated, and the loop repeats.

Deterministic tie-break, applied lexicographically to a pair (k, j):
  1. higher score (compared after rounding to 9 decimals)
  2. higher urgency u_j
  3. smaller additional distance
  4. smaller point id
  5. vehicle best-fit: smaller remaining capacity AFTER loading, Q_k - L_k - q_j
  6. smaller vehicle id
  Rules 2-4 pick between different points; rules 5-6 pick between vehicles for the
  same point.  (If beta > 0 and alpha > 0, rule 3 is rarely decisive; it matters in
  the beta = 0 "pure urgency" sensitivity setting.)
  Best-fit keeps large trucks free for large loads when several vehicles are
  otherwise equally good, e.g. several empty trucks that all see extra = 2 d(0,j).

PSEUDOCODE
==========
  GREEDY(instance, alpha, beta):
      classify points (unreachable / individually infeasible / eligible)      # problem.py
      U <- eligible points;  D_max <- max_p d(0,p) over reachable p
      for each vehicle k: route_k <- [], pos_k <- 0, load_k <- 0, clock_k <- 0
      while U not empty:
          best <- none
          for each vehicle k:
              for each j in U:
                  if E1,E2,E3 hold for (k,j):
                      key <- (-score, -u_j, extra, j, Q_k-load_k-q_j, k)
                      best <- min(best, key)
          if best is none: break                    # no feasible pair left
          (k, j) <- best;  append j to route_k;  update load_k, clock_k, pos_k;  remove j from U
      label every point left in U with the reason it is blocked
      return routes (each closed by the return trip to the depot)

COMPLEXITY  (n = eligible points, k = vehicles; distances come from the O(1)
             lookup table built by ShortestPaths, whose cost is analysed separately:
             O(V (V+E) log V) time, O(V^2) space)
==========
  Each iteration scans <= k * |U| pairs; each pair needs O(1) work (a table lookup,
  a few additions/comparisons, building one 6-tuple, one tuple comparison).
  Each iteration removes one point from U (list removal O(n)), so there are at most
  n iterations that select a pair, plus at most one final scan that finds none.
      pair evaluations <= sum_{t=0}^{n} k (n - t) = k n (n+1) / 2
  Time per iteration: O(k|U|) scan + O(n) removal.
      T(n,k) = O(k n^2)   worst case  (n log n / n^2 extras are dominated)
  Pre-screening O(n k); unmet-reason labelling O(n k); both inside O(k n^2).
  Space: O(n + k) beyond the shared distance matrix (routes, vehicle states, labels).
  The implementation counts pair evaluations so the bound can be checked in tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from data_generator import Instance, Vehicle
from problem import (ELIGIBLE, INFEASIBLE, REASON_TEXT, UNREACHABLE, URGENCY_MAX,
                     Plan, PointClass, classify_points, leq, objective)
from algorithms.shortest_path import INF, ShortestPaths


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GreedyParams:
    alpha: float = 1.0      # weight of urgency  (default)
    beta: float = 0.5       # weight of additional distance (default)


def extra_distance(sp: ShortestPaths, i: int, j: int) -> float:
    """Detour of visiting j after i instead of returning straight to the depot."""
    return max(0.0, sp.dist(i, j) + sp.dist(j, 0) - sp.dist(i, 0))


def priority_score(urgency: float, extra: float, d_max: float,
                   alpha: float = 1.0, beta: float = 0.5) -> float:
    """alpha*(u/10) - beta*(extra/D_max); the distance term is 0 when D_max = 0."""
    dist_term = extra / d_max if d_max > 0 else 0.0
    return alpha * urgency / URGENCY_MAX - beta * dist_term


# --------------------------------------------------------------------------
# Vehicle state and eligibility
# --------------------------------------------------------------------------
@dataclass
class _State:
    v: Vehicle
    stops: List[int] = field(default_factory=list)
    arrivals: List[float] = field(default_factory=list)
    completions: List[float] = field(default_factory=list)
    load: int = 0               #total load collected so far
    pos: int = 0                #vehicle's current location
    clock: float = 0.0          # completion time at `pos`
    dist: float = 0.0           # km driven so far (without the return trip)


_RANK = {"unreachable": -1, "capacity": 0, "deadline": 1, "route_duration": 2}

#higher score doesn't automatically mean a point will be collected because the vehicle may not have enough capacity or may not be able to reach the point before its deadline. 
# So, a vehicle-point combination must also be 'feasible' (E1, E2, E3) 
def _evaluate(inst: Instance, sp: ShortestPaths, st: _State, j: int
              ) -> Tuple[Optional[str], float, float]:
    """(blocking reason or None, arrival, completion) for adding point j to vehicle st."""
    p = inst.point(j)
    #constraint 1: capacity
    if st.load + p.waste > st.v.capacity:                                  # E1
        return "capacity", 0.0, 0.0
    
    #constraint 2: reachability
    d = sp.dist(st.pos, j)
    if d == INF:
        return "unreachable", 0.0, 0.0
    
    #constraint 3: completion deadline
    arrival = st.clock + inst.travel_time(d)            #st.clock is the completion time at the last stop, while travel_time() is time required to reach this new point.
    completion = arrival + p.service_time
    if not leq(completion, p.deadline):                                    # E2
        return "deadline", arrival, completion
    
    #constraint 4: route duration  (a truck should be able to return to depot within its max vehicle duration time)
    if not leq(completion + inst.travel_time(sp.dist(j, 0)), st.v.max_duration):   # E3
        return "route_duration", arrival, completion
    
    return None, arrival, completion


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class VehicleRoute:
    vehicle_id: int
    capacity: int
    max_duration: float
    stops: List[int]
    arrivals: List[float]
    completions: List[float]
    load: int
    distance: float             # km, including the return trip
    return_time: float          # minutes

    @property
    def nodes(self) -> List[int]:
        return [0] + self.stops + [0] if self.stops else []


UNCOLLECTED_TEXT = {
    "capacity": "no vehicle had enough remaining capacity",
    "deadline": "every vehicle with spare capacity would finish it after its deadline",
    "route_duration": "every vehicle that could meet the deadline would exceed its duration limit",
}


@dataclass
class GreedyResult:
    params: GreedyParams
    d_max: float
    classification: Dict[int, PointClass]
    routes: Dict[int, VehicleRoute]
    uncollected: Dict[int, str]                 # eligible points not served -> reason code
    log: List[Tuple[int, int, float, float]]    # (vehicle, point, score, extra) in selection order
    iterations: int
    evaluations: int
    candidates: int
    runtime_ms: float

    def plan(self) -> Plan:
        return {vid: list(r.stops) for vid, r in self.routes.items() if r.stops}

    def collected(self) -> List[int]:
        return sorted(j for r in self.routes.values() for j in r.stops)


# --------------------------------------------------------------------------
# Main Algorithm
# --------------------------------------------------------------------------
def greedy_construct(inst: Instance, sp: Optional[ShortestPaths] = None,
                     params: GreedyParams = GreedyParams()) -> GreedyResult:
    t0 = time.perf_counter()           #to measure execution time; this records teh starting time
    sp = sp or ShortestPaths(inst)
    classification = classify_points(inst, sp)           #classification logic from problem.py
    unassigned = sorted(pid for pid, c in classification.items() if c.status == ELIGIBLE)        #list of eligible points
    n_cand = len(unassigned)
    d_max = max((sp.dist(0, p.id) for p in inst.points if sp.reachable(0, p.id)), default=0.0)

    states = {v.id: _State(v) for v in inst.vehicles}       #stores state of each vehicle
    
    log: List[Tuple[int, int, float, float]] = []     #log records choice made by greedy (like vehicle, point, score etc)
    evaluations = 0                                   #how many vehicle-point pairs were checks; log and evaluate are useful to explain and evaluate alg later

    #selection loop: while there are still unassigned points, find the best feasible vehicle-point pair and assign it to the vehicle. Repeat until no feasible pair remains.
    while unassigned:
        best_key = None     #stores best collection point
        best = None         #stores corresponding vehicle-point combination

        #Examine every vehicle and every remaining point (global greedy approach) to find the best feasible pair (vehicle, point) to serve next.
        for st in states.values():
            for j in unassigned:
                evaluations += 1
                fail, arrival, completion = _evaluate(inst, sp, st, j)      #check feasibility before calculating score
                if fail is not None:               #not removing that point from unassigned because it may be feasible for another vehicle
                    continue
                #calculate the score for feasible vehicle-point combinations
                p = inst.point(j)
                extra = extra_distance(sp, st.pos, j)
                score = priority_score(p.urgency, extra, d_max, params.alpha, params.beta)
                key = (-round(score, 9), -p.urgency, round(extra, 9), j,      #score and urgency are -ve cz Python's normal tuple comparison selects the smaller tuple; (lexicographic order). 
                       st.v.capacity - st.load - p.waste, st.v.id)
                if best_key is None or key < best_key:     #smaller key is better so it is assigned as best_key and the corresponding vehicle-point combination is assigned as best
                    best_key, best = key, (st, j, arrival, completion, extra, score)
        
        if best is None:                       # no feasible (vehicle, point) pair left
            break
        #update state of the selected/winning vehicle and remove the selected point from unassigned
        st, j, arrival, completion, extra, score = best
        st.dist += sp.dist(st.pos, j)
        st.stops.append(j)
        st.arrivals.append(arrival)
        st.completions.append(completion)
        st.load += inst.point(j).waste
        st.pos, st.clock = j, completion
        unassigned.remove(j)       #remove the selected point from unassigned so that it is not considered in future iterations
        log.append((st.v.id, j, score, extra))

    # label every unserved eligible point with the reason it is blocked (final state)
    uncollected: Dict[int, str] = {}
    for j in unassigned:
        reasons = [_evaluate(inst, sp, st, j)[0] for st in states.values()]
        uncollected[j] = max(reasons, key=lambda r: _RANK[r])   # "furthest-along" blocker

    # close routes with the return trip (selection loop tracks the distance travelled to collection points, but does not include the return trip in st.dist.)
    routes: Dict[int, VehicleRoute] = {}
    for vid, st in states.items():
        if st.stops:
            back = sp.dist(st.pos, 0)
            routes[vid] = VehicleRoute(vid, st.v.capacity, st.v.max_duration, st.stops,
                                       st.arrivals, st.completions, st.load,
                                       st.dist + back, st.clock + inst.travel_time(back))
        else:
            routes[vid] = VehicleRoute(vid, st.v.capacity, st.v.max_duration, [], [], [], 0, 0.0, 0.0)

    return GreedyResult(params, d_max, classification, routes, uncollected, log,
                        len(log), evaluations, n_cand, (time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def format_report(inst: Instance, res: GreedyResult, sp: ShortestPaths) -> str:
    L: List[str] = []
    nm = lambda j: inst.point(j).name
    kg = lambda ids: sum(inst.point(j).waste for j in ids)
    by = {UNREACHABLE: [], INFEASIBLE: [], ELIGIBLE: []}
    for pid, c in res.classification.items():
        by[c.status].append(pid)
    collected = res.collected()

    L.append(f"=== Greedy construction on '{inst.name}'  "
             f"(alpha={res.params.alpha}, beta={res.params.beta}, D_max={res.d_max:.2f} km) ===")
    L.append("Initial classification")
    L.append(f"  unreachable             : {[nm(j) for j in by[UNREACHABLE]] or 'none'}  ({kg(by[UNREACHABLE])} kg)")
    inf_txt = "; ".join(f"{nm(j)} [{', '.join(res.classification[j].reasons)}]" for j in by[INFEASIBLE])
    L.append(f"  individually infeasible : {inf_txt or 'none'}  ({kg(by[INFEASIBLE])} kg)")
    L.append(f"  eligible                : {len(by[ELIGIBLE])} points  ({kg(by[ELIGIBLE])} kg)")
    L.append("Routes")
    for vid in sorted(res.routes):
        r = res.routes[vid]
        if not r.stops:
            L.append(f"  V{vid} (cap {r.capacity} kg): unused")
            continue
        L.append(f"  V{vid}: " + " -> ".join(["Depot"] + [nm(j) for j in r.stops] + ["Depot"]))
        L.append(f"      load {r.load}/{r.capacity} kg ({100 * r.load / r.capacity:.0f}%)   "
                 f"distance {r.distance:.2f} km   returns at {r.return_time:.1f}/{r.max_duration:.0f} min")
        L.append("      stops: " + "  ".join(
            f"{nm(j)}[arr {a:.1f}, done {c:.1f}, dl {inst.point(j).deadline:.0f}]"
            for j, a, c in zip(r.stops, r.arrivals, r.completions)))
    L.append("Final outcome")
    L.append(f"  collected   : {len(collected)} of {len(by[ELIGIBLE])} eligible points, {kg(collected)} kg")
    if res.uncollected:
        for j, why in sorted(res.uncollected.items()):
            L.append(f"  uncollected : {nm(j)} ({inst.point(j).waste} kg, urgency {inst.point(j).urgency})"
                     f" - {why}: {UNCOLLECTED_TEXT[why]}")
    else:
        L.append("  uncollected : none (every eligible point was served)")
    W, D = objective(inst, sp, res.plan())
    L.append(f"Objective  : W = sum(urgency x waste) = {W},  D = {D:.2f} km")
    bound = res.candidates * (res.candidates + 1) // 2 * len(inst.vehicles)
    L.append(f"Work       : {res.evaluations} pair evaluations over {res.iterations} selections "
             f"(bound k n(n+1)/2 = {bound}), {res.runtime_ms:.2f} ms")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import all_edge_cases, generate_instance
    from feasibility import check_plan

    cases = [generate_instance()] + list(all_edge_cases().values())
    for inst in cases:
        sp = ShortestPaths(inst)
        res = greedy_construct(inst, sp)
        print(format_report(inst, res, sp))
        chk = check_plan(inst, sp, res.plan())
        print(f"Independent checker: {'PASS' if chk.ok else 'FAIL'}  "
              f"(W={chk.waste_value}, D={chk.distance:.2f} km)"
              + ("" if chk.ok else f"  {chk.violations}"))
        print()
