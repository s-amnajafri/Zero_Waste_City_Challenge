"""
brute_force.py  --  exact solver for SMALL instances (optimality benchmark only)

Purpose: find the true optimum of the global objective defined in problem.py so the
Greedy + 2-opt pipeline can be compared with it (optimality gap). It is NOT part of the
normal pipeline and refuses large instances.

SAME problem, SAME rules
------------------------
  * Objective (lexicographic): maximise W = sum(urgency x waste) of served points, then
    minimise total distance D.  Imported from problem.py (is_better, EPS).
  * Constraints C1-C6 (capacity, completion deadlines, duration, open roads, uniqueness,
    fleet) are exactly those of problem.py. Every plan it returns is re-verified by the
    independent checker in the tests.
  * Only ELIGIBLE points are searched. This loses nothing: an unreachable point, or a point
    that no vehicle can serve even alone, cannot appear in ANY feasible plan.
    (restrict_to_eligible=False searches all points; tests confirm the same optimum.)

Method (exhaustive search, exact)
---------------------------------
  Step 1 - routes of one vehicle.  DFS over every ordered selection of candidate points.
     For each subset S (bitmask) keep the SHORTEST feasible ordering of S for that vehicle.
     Pruning: a prefix that already breaks capacity, a completion deadline, or the duration
     limit (including the return trip) is cut. This is exact: capacity and arrival times only
     grow as stops are added, and by the triangle inequality of shortest-path distances the
     return time of any extension is >= the return time of the prefix.
     (prune=False enumerates every permutation of every subset and tests each with the
     independent checker; tests check that both modes give the same optimum.)
  Step 2 - combine vehicles.  best(vehicle i, used points) = max over feasible subsets S of
     vehicle i disjoint from `used` of  [ W(S) + best(i+1, used + S) ], with memoisation on
     (i, used). Within equal W, smaller distance wins (lexicographic).

Pseudocode
----------
  SOLVE_EXACT(n candidates, k vehicles):
      for each vehicle v:  route[v][mask] <- shortest feasible ordering of mask (DFS, pruned)
      f(i, used) = (0, 0)                                    if i = k
                 = best over masks m in route[i] with m & used = 0 of
                       (W(m) + f(i+1, used|m).W ,  d(route[i][m]) + f(i+1, used|m).D)   # lexicographic
      return f(0, 0)

Complexity  (n = eligible points, k = vehicles)
----------
  Step 1: at most sum_{s<=n} n!/(n-s)! ~ e*n! feasible sequences per vehicle, each extension
          scans n candidates  ->  O(k * n * n!) time worst case (pruning only helps), O(2^n) space
          per vehicle for the best-route table.
  Step 2: (i, used) has at most k * 2^n states, each scans <= 2^n masks -> O(k * 4^n) time,
          O(k * 2^n) space.
  Total   O(k * n * n!  +  k * 4^n).   Compare with Greedy's O(k n^2): this is why exact search
  is only a benchmark. An unstructured enumeration of assignments x orders is even larger:
  (k+1)^n assignments times up to n! orders.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data_generator import Instance, Vehicle
from feasibility import check_plan
from problem import EPS, Plan, classify_points, is_better, leq, ELIGIBLE
from algorithms.shortest_path import INF, ShortestPaths

DEFAULT_MAX_POINTS = 8


@dataclass
class ExactResult:
    plan: Plan
    W: int
    D: float
    n_candidates: int
    sequences_explored: int      # feasible (pruned mode) or tested (unpruned mode) orderings, all vehicles
    states: int                  # memoised (vehicle, used-set) states
    runtime_ms: float


RouteTable = Dict[int, Tuple[float, List[int]]]      # mask -> (distance incl. return, ordering)


def _routes_pruned(inst: Instance, sp: ShortestPaths, veh: Vehicle, cands: List[int]) -> Tuple[RouteTable, int]:
    idx = {pid: i for i, pid in enumerate(cands)}
    best: RouteTable = {0: (0.0, [])}
    explored = 0
    order: List[int] = []

    def dfs(pos: int, t: float, load: int, dist: float, mask: int) -> None:
        nonlocal explored
        for j in cands:
            bit = 1 << idx[j]
            if mask & bit:
                continue
            p = inst.point(j)
            if load + p.waste > veh.capacity:                         # C1
                continue
            d = sp.dist(pos, j)
            if d == INF:                                              # C4
                continue
            t2 = t + inst.travel_time(d) + p.service_time
            if not leq(t2, p.deadline):                               # C2 (completion deadline)
                continue
            back = sp.dist(j, 0)
            if not leq(t2 + inst.travel_time(back), veh.max_duration):   # C3 incl. return trip
                continue
            explored += 1
            dist2, m2 = dist + d, mask | bit
            order.append(j)
            cur = best.get(m2)
            if cur is None or dist2 + back < cur[0] - EPS:
                best[m2] = (dist2 + back, list(order))
            dfs(j, t2, load + p.waste, dist2, m2)
            order.pop()

    dfs(0, 0.0, 0, 0.0, 0)
    return best, explored


def _routes_unpruned(inst: Instance, sp: ShortestPaths, veh: Vehicle, cands: List[int]) -> Tuple[RouteTable, int]:
    """Every permutation of every subset, each judged by the independent checker (slow; n <= ~6)."""
    best: RouteTable = {0: (0.0, [])}
    explored = 0
    for s in range(1, len(cands) + 1):
        for subset in itertools.combinations(range(len(cands)), s):
            mask = sum(1 << i for i in subset)
            for perm in itertools.permutations([cands[i] for i in subset]):
                explored += 1
                chk = check_plan(inst, sp, {veh.id: list(perm)})
                if chk.ok:
                    cur = best.get(mask)
                    if cur is None or chk.distance < cur[0] - EPS:
                        best[mask] = (chk.distance, list(perm))
    return best, explored


def solve_exact(inst: Instance, sp: Optional[ShortestPaths] = None,
                max_points: int = DEFAULT_MAX_POINTS, prune: bool = True,
                restrict_to_eligible: bool = True) -> ExactResult:
    t0 = time.perf_counter()
    sp = sp or ShortestPaths(inst)
    if restrict_to_eligible:
        cls = classify_points(inst, sp)
        cands = sorted(j for j, c in cls.items() if c.status == ELIGIBLE)
    else:
        cands = [p.id for p in inst.points]
    if len(cands) > max_points:
        raise ValueError(f"exact search is limited to {max_points} candidate points "
                         f"(this instance has {len(cands)}); the search grows factorially. "
                         f"Raise max_points only for small experiments.")

    vehicles = sorted(inst.vehicles, key=lambda v: v.id)
    build = _routes_pruned if prune else _routes_unpruned
    tables: List[RouteTable] = []
    explored = 0
    for veh in vehicles:
        table, n = build(inst, sp, veh, cands)
        tables.append(table)
        explored += n

    w_of: Dict[int, int] = {}
    for mask in set().union(*[set(t) for t in tables]) if tables else {0}:
        w_of[mask] = sum(inst.point(cands[i]).urgency * inst.point(cands[i]).waste
                         for i in range(len(cands)) if mask >> i & 1)

    memo: Dict[Tuple[int, int], Tuple[int, float, List[Tuple[int, List[int]]]]] = {}

    def best_from(vi: int, used: int) -> Tuple[int, float, List[Tuple[int, List[int]]]]:
        if vi == len(vehicles):
            return 0, 0.0, []
        key = (vi, used)
        if key in memo:
            return memo[key]
        best: Optional[Tuple[int, float, List[Tuple[int, List[int]]]]] = None
        for mask in sorted(tables[vi]):
            if mask & used:
                continue
            dist, order = tables[vi][mask]
            w_rest, d_rest, plan_rest = best_from(vi + 1, used | mask)
            cand = (w_of[mask] + w_rest, dist + d_rest, [(vehicles[vi].id, order)] + plan_rest)
            if best is None or is_better((cand[0], cand[1]), (best[0], best[1])):
                best = cand
        memo[key] = best
        return best

    W, D, assignment = best_from(0, 0)
    plan: Plan = {vid: order for vid, order in assignment if order}
    return ExactResult(plan, W, D, len(cands), explored, len(memo), (time.perf_counter() - t0) * 1000)


if __name__ == "__main__":
    from data_generator import small_instance
    inst = small_instance(seed=1, n_points=7)
    sp = ShortestPaths(inst)
    res = solve_exact(inst, sp)
    nm = lambda j: inst.point(j).name
    print(f"Exact optimum on '{inst.name}' ({res.n_candidates} candidate points, {len(inst.vehicles)} vehicles)")
    for vid, stops in sorted(res.plan.items()):
        print(f"  V{vid}: Depot -> " + " -> ".join(nm(j) for j in stops) + " -> Depot")
    print(f"  W = {res.W}, D = {res.D:.2f} km | {res.sequences_explored} feasible orderings, "
          f"{res.states} memo states, {res.runtime_ms:.1f} ms")
    chk = check_plan(inst, sp, res.plan)
    print(f"  independent checker: {'PASS' if chk.ok else 'FAIL'}  (W={chk.waste_value}, D={chk.distance:.2f})")
