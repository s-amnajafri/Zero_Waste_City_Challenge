"""
shortest_path.py  --  Step 2a of the Zero-Waste City Challenge

Supporting procedure for Greedy, 2-opt and the brute-force solver.

Dijkstra's algorithm (adjacency list + binary-heap priority queue) computes
shortest road distances. Closed roads are excluded because the adjacency list
comes from Instance.adjacency(usable_only=True).

Mathematical formulation
------------------------
Graph G = (V, E_open): V = {0..n}, E_open = accessible roads, w(e) > 0 in km.
  d(s, v) = min over s-v paths P of  sum_{e in P} w(e),   d(s, v) = +inf if no path.
Dijkstra invariant: when a node u is popped with key dist[u], dist[u] = d(s, u).
This needs w >= 0 (negative weights are rejected).

Pseudocode (lazy-deletion variant)
----------------------------------
  DIJKSTRA(adj, s):
      dist[v] <- inf for all v;  parent[v] <- None;  dist[s] <- 0
      PQ <- min-heap containing (0, s)
      while PQ not empty:
          (d, u) <- pop-min(PQ)
          if d > dist[u]: continue                 # stale entry, skip
          for (v, w) in adj[u]:
              if d + w < dist[v]:
                  dist[v] <- d + w;  parent[v] <- u;  push(PQ, (dist[v], v))
      return dist, parent

  PATH(parent, s, t):  walk t -> parent[t] -> ... -> s, then reverse.
                       Return None if t was never reached.

Complexity  (V = nodes, E = open roads)
---------------------------------------
  One run : time O((V + E) log V)   -- each edge relaxation pushes at most one
            heap entry, so the heap holds <= E entries; log E <= 2 log V.
            space O(V + E) (adjacency list + dist/parent arrays + heap).
  All-pairs (one run per node, V = n+1):
            time O(V (V + E) log V),  space O(V^2) for the dist and parent matrices.
  Path reconstruction: O(length of path).
"""

from __future__ import annotations

import heapq              #implements min-heap used by dijkstra
import math              
import time               #measures execution time in the demo
from typing import Dict, List, Optional, Sequence, Tuple           

from data_generator import Instance, generate_instance, edge_unreachable

INF = math.inf                          #infinity for node that hasn't yet been reached
#graph is stored as adjacency list
Adjacency = Dict[int, List[Tuple[int, float]]]        #{node_id: list of neighbours stored as tuple (node_id, distance)}


# Single-source Dijkstra
def dijkstra(adj: Adjacency, source: int, n_nodes: Optional[int] = None
             ) -> Tuple[List[float], List[Optional[int]]]:
    """Return (dist, parent) from `source`. Unreached nodes keep dist = inf, parent = None."""
    if n_nodes is None:              #if n_nodes is not provided, we calculate it from adj list
        n_nodes = len(adj)
    if not 0 <= source < n_nodes:
        raise ValueError(f"source {source} is not a node (0..{n_nodes - 1})")

    dist = [INF] * n_nodes     #dist stores shortest distance from the source..initialized with infinity
    parent: List[Optional[int]] = [None] * n_nodes      #parent tracks the previous node in the best-known path
    dist[source] = 0.0                #dist = [0,INF,INF,...] 
    heap = [(0.0, source)]        #heap lets us select the node with smallest known distance (dist) from the source. Stores tuples in te list like (distance, node)

    while heap:
        d, u = heapq.heappop(heap)           #min-heap pops the node u with the smallest distance d from the source
        if d > dist[u]:                      # stale entry (a shorter path was found later) (lazy deletion)
            continue
        for v, w in adj.get(u, ()):             #inspect neighhbours of popped node u
            if w < 0:
                raise ValueError(f"negative edge weight {w} on road {u}-{v}")
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, parent


def reconstruct_path(parent: Sequence[Optional[int]], source: int, target: int
                     ) -> Optional[List[int]]:
    """Node sequence source -> target, or None if target is unreachable."""
    if target == source:
        return [source]
    if parent[target] is None:   #target is unreachable
        return None
    
    path = [target]
    while path[-1] != source:                    #if the last node in the path is not the source, we keep adding its parent to the path
        path.append(parent[path[-1]])
    path.reverse()
    return path

#dijkstra is a single-source shortest path algorithm but we need shortest distances between all pairs of nodes (depot + collection points). So we run dijkstra from each node and store the results in a matrix. 
# All-pairs wrapper used by the rest of the project
class ShortestPaths:
    """Shortest-path oracle over depot + collection points (closed roads excluded)."""

    def __init__(self, instance: Instance):
        self.instance = instance
        self.n = instance.node_count
        adj = instance.adjacency(usable_only=True)
        self._dist: List[List[float]] = []
        self._parent: List[List[Optional[int]]] = []
        #run dijkstra from each node and store the results in a matrix
        for s in range(self.n):
            d, p = dijkstra(adj, s, self.n)
            self._dist.append(d)
            self._parent.append(p)

    # queries 
    def dist(self, i: int, j: int) -> float:
        return self._dist[i][j]

    def reachable(self, i: int, j: int) -> bool:
        return self._dist[i][j] < INF

    def path(self, i: int, j: int) -> Optional[List[int]]:
        return reconstruct_path(self._parent[i], i, j)

    def travel_time(self, i: int, j: int) -> float:        #finds the shortest dist then converts it into time using instances's travel-time calculation
        """Minutes; +inf if unreachable."""
        d = self._dist[i][j]
        return INF if d == INF else self.instance.travel_time(d)

    def route_distance(self, route: Sequence[int]) -> float:
        """Total km of a route given as a node sequence (e.g. [0, 3, 5, 0])."""
        return sum(self._dist[a][b] for a, b in zip(route, route[1:]))        #zip() to create pairs of consecutive nodes in the route

    def expand_route(self, route: Sequence[int]) -> List[int]:
        """Replace each leg by its road-level path (used for drawing maps)."""    #suppose waste was collected from point 3 then point 5; path is 0->3->5->0. But the actual road-level path may be 0->1->3->4->5->2->0. This function finds the actual road-level path.
        full: List[int] = [route[0]]
        for a, b in zip(route, route[1:]):
            leg = self.path(a, b)
            if leg is None:
                raise ValueError(f"no road path between {a} and {b}")
            full.extend(leg[1:])
        return full

    #return list of reachable and unreachable collection points from the depot (0)
    def unreachable_points(self):
        """Collection points with no open-road path from the depot."""
        return [p for p in self.instance.points if not self.reachable(0, p.id)]

    def reachable_points(self):
        return [p for p in self.instance.points if self.reachable(0, p.id)]


# Demo
if __name__ == "__main__":
    inst = generate_instance()
    t0 = time.perf_counter()
    sp = ShortestPaths(inst)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{inst.name}: all-pairs Dijkstra over {sp.n} nodes, "
          f"{len(inst.usable_roads())} open roads, {ms:.2f} ms\n")
    print("Shortest distance from the depot (closed roads excluded):")
    for p in inst.points:
        path = sp.path(0, p.id)
        if path is None:
            print(f"  {p.name}: UNREACHABLE (no open road path)")
        else:
            print(f"  {p.name}: {sp.dist(0, p.id):6.2f} km  "
                  f"({sp.travel_time(0, p.id):5.1f} min)  path {path}")
    print("\nUnreachable points reported:", [p.name for p in sp.unreachable_points()])

    sp2 = ShortestPaths(edge_unreachable())
    print("edge_unreachable ->", [p.name for p in sp2.unreachable_points()],
          "| dist(0, P06) =", sp2.dist(0, 6))

#cd .. ;then py -m algorithms.shortest_path