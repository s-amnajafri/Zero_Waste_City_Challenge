"""
data_generator.py  --  Step 1 of the Zero-Waste City Challenge

Builds simulated urban waste-collection instances:
  * a weighted road network (some roads closed / inaccessible)
  * collection points (waste quantity, urgency, service time, deadline)
  * a heterogeneous vehicle fleet (capacity, max route duration)

Conventions
-----------
  * Node 0 is the depot. Collection points are nodes 1..n.
  * Distances are in km, times in minutes, waste in kg.
  * Travel time = distance / speed * 60  (constant speed, default 30 km/h).
  * Deadline = latest allowed COMPLETION time at a point (minutes after shift start):
    arrival time + service time must be <= deadline.

No algorithm code lives here: this module only creates and describes data.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

SPEED_KMH = 30.0
DEFAULT_CAPACITIES = (800, 1000, 1200, 1500)      # kg
DEFAULT_DURATIONS = (300, 330, 360, 390)          # minutes


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Point:
    id: int
    name: str
    x: float
    y: float
    waste: int            # kg
    urgency: int          # 1..10 (10 = most urgent)
    service_time: float   # minutes
    deadline: float       # minutes (latest COMPLETION = arrival + service)


@dataclass
class Road:
    u: int
    v: int
    distance: float       # km
    accessible: bool = True


@dataclass
class Vehicle:
    id: int
    capacity: int         # kg
    max_duration: float   # minutes (travel + service, depot to depot)


@dataclass
class Instance:
    name: str
    depot: Tuple[float, float]
    points: List[Point]
    roads: List[Road]
    vehicles: List[Vehicle]
    speed_kmh: float = SPEED_KMH
    seed: Optional[int] = None

    # ---- basic helpers ---------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self.points) + 1

    def point(self, pid: int) -> Point:
        return self.points[pid - 1]

    def travel_time(self, distance: float) -> float:
        return distance / self.speed_kmh * 60.0

    def usable_roads(self) -> List[Road]:
        return [r for r in self.roads if r.accessible]

    def closed_roads(self) -> List[Road]:
        return [r for r in self.roads if not r.accessible]

    def adjacency(self, usable_only: bool = True) -> Dict[int, List[Tuple[int, float]]]:
        adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(self.node_count)}
        for r in self.roads:
            if usable_only and not r.accessible:
                continue
            adj[r.u].append((r.v, r.distance))
            adj[r.v].append((r.u, r.distance))
        return adj

    # ---- connectivity / demand summaries ---------------------------------
    def reachable_nodes(self) -> set:
        adj = self.adjacency()
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def unreachable_points(self) -> List[Point]:
        reach = self.reachable_nodes()
        return [p for p in self.points if p.id not in reach]

    def total_demand(self, reachable_only: bool = True) -> int:
        reach = self.reachable_nodes()
        return sum(p.waste for p in self.points if (not reachable_only) or p.id in reach)

    def total_capacity(self) -> int:
        return sum(v.capacity for v in self.vehicles)

    def oversized_points(self) -> List[Point]:
        cap = max(v.capacity for v in self.vehicles)
        return [p for p in self.points if p.waste > cap]

    # ---- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Instance":
        with open(path) as f:
            d = json.load(f)
        return Instance(
            name=d["name"],
            depot=tuple(d["depot"]),
            points=[Point(**p) for p in d["points"]],
            roads=[Road(**r) for r in d["roads"]],
            vehicles=[Vehicle(**v) for v in d["vehicles"]],
            speed_kmh=d["speed_kmh"],
            seed=d.get("seed"),
        )

    # ---- human-readable summary -----------------------------------------
    def summary(self) -> str:
        unreach = self.unreachable_points()
        over = self.oversized_points()
        lines = [
            f"Instance: {self.name}  (seed={self.seed})",
            f"  Collection points : {len(self.points)}  (+ 1 depot)",
            f"  Roads             : {len(self.roads)} total, "
            f"{len(self.usable_roads())} usable, {len(self.closed_roads())} closed",
            f"  Vehicles          : {len(self.vehicles)}  capacities="
            f"{[v.capacity for v in self.vehicles]} kg",
            f"  Fleet capacity    : {self.total_capacity()} kg",
            f"  Reachable demand  : {self.total_demand(True)} kg "
            f"(ratio {self.total_demand(True) / self.total_capacity():.2f} of capacity)",
            f"  Unreachable pts   : {[p.name for p in unreach] or 'none'}",
            f"  Oversized pts     : {[p.name for p in over] or 'none'}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Random generator
# --------------------------------------------------------------------------
def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_instance(
    n_points: int = 16,
    n_isolated: int = 1,
    usable_roads: int = 27,
    vehicle_capacities: Sequence[int] = DEFAULT_CAPACITIES,
    vehicle_durations: Optional[Sequence[float]] = DEFAULT_DURATIONS,
    demand_ratio: float = 1.08,
    seed: int = 42,
    name: str = "main",
    area_km: float = 20.0,
) -> Instance:
    """
    Generate a reproducible random instance.

    n_points       : collection points, INCLUDING the isolated ones.
    n_isolated     : points cut off from the network (3 closed roads each).
    usable_roads   : number of accessible roads among depot + connected points.
    demand_ratio   : reachable demand / total fleet capacity (>1 => over-demand).

    Construction (guarantees the usable network is connected):
      1. Place the depot in the middle and scatter the points.
      2. Build a spanning tree (each node joins one of its 2 nearest tree nodes).
      3. Add extra short roads until `usable_roads` is reached.
      4. Connect each isolated point to its 3 nearest nodes with CLOSED roads.
    Road distance = Euclidean distance x U(1.0, 1.3)  (roads are never straighter
    than the crow flies).
    """
    rng = random.Random(seed)
    n_conn = n_points - n_isolated            # connected points (excluding depot)
    coords: Dict[int, Tuple[float, float]] = {0: (area_km / 2, area_km / 2)}
    for i in range(1, n_points + 1):
        coords[i] = (round(rng.uniform(0, area_km), 2), round(rng.uniform(0, area_km), 2))

    def road_len(u: int, v: int) -> float:
        return round(_euclid(coords[u], coords[v]) * rng.uniform(1.0, 1.3), 2)

    roads: List[Road] = []
    used = set()

    # 2. spanning tree over connected nodes 0..n_conn
    in_tree = [0]
    order = list(range(1, n_conn + 1))
    rng.shuffle(order)
    for v in order:
        nearest = sorted(in_tree, key=lambda t: _euclid(coords[v], coords[t]))[:2]
        u = rng.choice(nearest)
        roads.append(Road(u, v, road_len(u, v)))
        used.add(frozenset((u, v)))
        in_tree.append(v)

    # 3. extra roads (sampled from the shortest candidate pairs)
    n_extra = max(0, usable_roads - len(roads))
    candidates = [
        (a, b)
        for a in range(n_conn + 1)
        for b in range(a + 1, n_conn + 1)
        if frozenset((a, b)) not in used
    ]
    candidates.sort(key=lambda p: _euclid(coords[p[0]], coords[p[1]]))
    pool = candidates[: max(1, 3 * n_extra)]
    for a, b in rng.sample(pool, min(n_extra, len(pool))):
        roads.append(Road(a, b, road_len(a, b)))
        used.add(frozenset((a, b)))

    # 4. isolated points: roads exist but are all closed
    for i in range(n_conn + 1, n_points + 1):
        nearest = sorted(range(n_conn + 1), key=lambda t: _euclid(coords[i], coords[t]))[:3]
        for t in nearest:
            roads.append(Road(t, i, road_len(t, i), accessible=False))

    # vehicles
    if vehicle_durations is None:
        vehicle_durations = [360] * len(vehicle_capacities)
    vehicles = [
        Vehicle(k + 1, int(c), float(d))
        for k, (c, d) in enumerate(zip(vehicle_capacities, vehicle_durations))
    ]
    total_cap = sum(v.capacity for v in vehicles)
    max_cap = max(v.capacity for v in vehicles)

    # waste: scale reachable demand to demand_ratio x fleet capacity
    raw = {i: rng.randint(150, 500) for i in range(1, n_points + 1)}
    reach_sum = sum(raw[i] for i in range(1, n_conn + 1))
    scale = (demand_ratio * total_cap) / reach_sum if reach_sum else 1.0

    points: List[Point] = []
    for i in range(1, n_points + 1):
        waste = raw[i] if i > n_conn else max(50, round(raw[i] * scale))
        waste = min(waste, max_cap)           # oversize is a deliberate edge case
        urgency = rng.randint(1, 10)
        service = rng.randint(5, 20)
        # urgent points get tight deadlines -> urgency vs. route-time conflict
        deadline = rng.randint(60, 150) if urgency >= 8 else rng.randint(240, 420)
        points.append(Point(i, f"P{i:02d}", coords[i][0], coords[i][1],
                            int(waste), urgency, float(service), float(deadline)))

    return Instance(name, coords[0], points, roads, vehicles, SPEED_KMH, seed)


# --------------------------------------------------------------------------
# Hand-built edge-case instances (small, deterministic, easy to reason about)
# --------------------------------------------------------------------------
def _manual(
    name: str,
    points: List[Tuple[float, float, int, int, float, float]],
    edges: List[Tuple],
    capacities: Sequence[int],
    durations: Sequence[float],
) -> Instance:
    """points: (x, y, waste, urgency, service_time, deadline); edges: (u, v[, accessible])."""
    depot = (0.0, 0.0)
    coords = {0: depot}
    pts = []
    for i, (x, y, w, u, s, d) in enumerate(points, start=1):
        coords[i] = (x, y)
        pts.append(Point(i, f"P{i:02d}", x, y, w, u, float(s), float(d)))
    roads = []
    for e in edges:
        u, v = e[0], e[1]
        acc = e[2] if len(e) > 2 else True
        roads.append(Road(u, v, round(_euclid(coords[u], coords[v]), 2), acc))
    vehicles = [Vehicle(k + 1, int(c), float(d)) for k, (c, d) in enumerate(zip(capacities, durations))]
    return Instance(name, depot, pts, roads, vehicles, SPEED_KMH, None)


def edge_unreachable() -> Instance:
    """P06 is cut off: both of its roads are closed. Must be reported, not crash."""
    pts = [(2, 0, 150, 5, 10, 300), (4, 1, 120, 7, 10, 300), (3, 3, 180, 4, 10, 300),
           (1, 3, 100, 9, 10, 200), (6, 4, 140, 3, 10, 300), (9, 9, 200, 10, 10, 300)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (2, 5), (5, 6, False), (3, 6, False)]
    return _manual("edge_unreachable", pts, edges, (500, 600), (300, 300))


def edge_over_demand() -> Instance:
    """Total waste (1800 kg) exceeds fleet capacity (1500 kg): partial coverage expected."""
    pts = [(2, 0, 300, 9, 10, 300), (4, 1, 300, 3, 10, 300), (3, 3, 300, 7, 10, 300),
           (1, 3, 300, 5, 10, 300), (6, 4, 300, 8, 10, 300), (5, 6, 300, 2, 10, 300)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (2, 5), (5, 6), (3, 6)]
    return _manual("edge_over_demand", pts, edges, (400, 500, 600), (300, 300, 300))


def edge_oversized() -> Instance:
    """P03 weighs 900 kg but the biggest truck carries 800 kg: must be excluded."""
    pts = [(2, 0, 200, 6, 10, 300), (4, 1, 250, 5, 10, 300), (3, 3, 900, 10, 10, 300),
           (1, 3, 150, 4, 10, 300), (6, 4, 220, 7, 10, 300)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (2, 5)]
    return _manual("edge_oversized", pts, edges, (600, 800), (300, 300))


def edge_deadline_violation() -> Instance:
    """P02 is ~6 km away (12 min) but its deadline is 5 min: no vehicle can serve it."""
    pts = [(2, 0, 150, 5, 10, 300), (6, 0, 150, 10, 10, 5), (3, 3, 150, 4, 10, 300),
           (1, 3, 150, 6, 10, 300)]
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (0, 4)]
    return _manual("edge_deadline_violation", pts, edges, (600, 600), (300, 300))


def edge_no_2opt_gain() -> Instance:
    """Points on a line, one road chain: the greedy route is already optimal."""
    pts = [(2, 0, 100, 5, 5, 300), (4, 0, 100, 5, 5, 300),
           (6, 0, 100, 5, 5, 300), (8, 0, 100, 5, 5, 300)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    return _manual("edge_no_2opt_gain", pts, edges, (1000,), (300,))


def small_instance(seed: int = 1, n_points: int = 7) -> Instance:
    """Tiny fully-connected-ish instance (6-8 points) for the brute-force optimality gap."""
    return generate_instance(
        n_points=n_points, n_isolated=0, usable_roads=n_points + 5,
        vehicle_capacities=(700, 900), vehicle_durations=(300, 300),
        demand_ratio=0.9, seed=seed, name=f"small_seed{seed}", area_km=10.0,
    )


def all_edge_cases() -> Dict[str, Instance]:
    return {i.name: i for i in (
        edge_unreachable(), edge_over_demand(), edge_oversized(),
        edge_deadline_violation(), edge_no_2opt_gain(),
    )}


# --------------------------------------------------------------------------
# CLI: generate everything into ./data and print summaries
# --------------------------------------------------------------------------
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    main_inst = generate_instance()
    main_inst.save(os.path.join(out, "main_dataset.json"))
    print(main_inst.summary(), "\n")
    for p in main_inst.points:
        print(f"  {p.name}  waste={p.waste:4d}  urgency={p.urgency:2d}  "
              f"service={p.service_time:4.0f}  deadline={p.deadline:4.0f}")
    print()
    for inst in list(all_edge_cases().values()) + [small_instance()]:
        inst.save(os.path.join(out, f"{inst.name}.json"))
        print(inst.summary(), "\n")
    print(f"Saved datasets to {out}")
