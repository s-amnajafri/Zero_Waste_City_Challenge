"""Tests for shortest_path.py  (run from zero_waste/:  python -m unittest discover -s tests -v)"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import (Instance, Road, _manual, edge_unreachable,
                            generate_instance)
from algorithms.shortest_path import INF, ShortestPaths, dijkstra, reconstruct_path

EPS = 1e-9


def floyd_warshall(inst: Instance):
    """Independent oracle: O(V^3) all-pairs over open roads."""
    n = inst.node_count
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0.0
    for r in inst.usable_roads():
        d[r.u][r.v] = min(d[r.u][r.v], r.distance)
        d[r.v][r.u] = min(d[r.v][r.u], r.distance)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


class TestDijkstraCore(unittest.TestCase):
    def test_shortcut_beats_direct_edge(self):
        adj = {0: [(1, 10.0), (2, 3.0)], 1: [(0, 10.0), (2, 4.0)], 2: [(0, 3.0), (1, 4.0)]}
        dist, parent = dijkstra(adj, 0)
        self.assertEqual(dist[1], 7.0)
        self.assertEqual(reconstruct_path(parent, 0, 1), [0, 2, 1])

    def test_source_to_itself(self):
        adj = {0: [(1, 2.0)], 1: [(0, 2.0)]}
        dist, parent = dijkstra(adj, 0)
        self.assertEqual(dist[0], 0.0)
        self.assertEqual(reconstruct_path(parent, 0, 0), [0])

    def test_unreachable_node(self):
        adj = {0: [(1, 2.0)], 1: [(0, 2.0)], 2: []}
        dist, parent = dijkstra(adj, 0)
        self.assertEqual(dist[2], INF)
        self.assertIsNone(reconstruct_path(parent, 0, 2))

    def test_parallel_roads_take_the_shorter(self):
        adj = {0: [(1, 5.0), (1, 2.5)], 1: [(0, 5.0), (0, 2.5)]}
        dist, _ = dijkstra(adj, 0)
        self.assertEqual(dist[1], 2.5)

    def test_negative_weight_rejected(self):
        adj = {0: [(1, -1.0)], 1: [(0, -1.0)]}
        with self.assertRaises(ValueError):
            dijkstra(adj, 0)

    def test_invalid_source_rejected(self):
        with self.assertRaises(ValueError):
            dijkstra({0: []}, 5)


class TestClosedRoads(unittest.TestCase):
    def setUp(self):
        # P1 (4,0), P2 (4,3). Open: 0-1 (4 km), 1-2 (3 km). CLOSED shortcut 0-2 (5 km).
        self.inst = _manual("closed_demo",
                            [(4, 0, 100, 5, 5, 300), (4, 3, 100, 5, 5, 300)],
                            [(0, 1), (1, 2), (0, 2, False)], (500,), (300,))

    def test_closed_shortcut_is_ignored(self):
        sp = ShortestPaths(self.inst)
        self.assertAlmostEqual(sp.dist(0, 2), 7.0)       # via P1, not the closed 5 km road
        self.assertEqual(sp.path(0, 2), [0, 1, 2])

    def test_same_road_open_would_be_used(self):
        self.inst.roads[2].accessible = True
        sp = ShortestPaths(self.inst)
        self.assertAlmostEqual(sp.dist(0, 2), 5.0)


class TestUnreachablePoints(unittest.TestCase):
    def test_edge_unreachable_dataset(self):
        sp = ShortestPaths(edge_unreachable())
        self.assertEqual([p.name for p in sp.unreachable_points()], ["P06"])
        self.assertEqual(sp.dist(0, 6), INF)
        self.assertIsNone(sp.path(0, 6))
        self.assertFalse(sp.reachable(0, 6))
        self.assertEqual(sp.travel_time(0, 6), INF)
        self.assertEqual(len(sp.reachable_points()), 5)

    def test_main_dataset_isolated_p16(self):
        sp = ShortestPaths(generate_instance())
        self.assertEqual([p.name for p in sp.unreachable_points()], ["P16"])
        self.assertEqual(len(sp.reachable_points()), 15)


class TestAgainstOracleAndProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.insts = [generate_instance(), generate_instance(seed=7, name="seed7"),
                     generate_instance(n_points=25, n_isolated=2, usable_roads=40,
                                       seed=3, name="bigger")]

    def test_matches_floyd_warshall(self):
        for inst in self.insts:
            sp, fw = ShortestPaths(inst), floyd_warshall(inst)
            for i in range(inst.node_count):
                for j in range(inst.node_count):
                    a, b = sp.dist(i, j), fw[i][j]
                    if math.isinf(b):
                        self.assertEqual(a, INF)
                    else:
                        self.assertAlmostEqual(a, b, places=9)

    #distance from A to B = distance from B to A (symmetric) and triangle inequality holds (distance from A to C <= distance from A to B + distance from B to C)
    def test_symmetry(self):        
        for inst in self.insts:
            sp = ShortestPaths(inst)
            for i in range(inst.node_count):
                for j in range(inst.node_count):
                    self.assertAlmostEqual(sp.dist(i, j), sp.dist(j, i), places=9)

    def test_triangle_inequality(self):
        # Greedy's extra-distance value relies on this (it makes it non-negative).
        inst = self.insts[0]
        sp = ShortestPaths(inst)
        nodes = sorted(inst.reachable_nodes())
        for i in nodes:
            for j in nodes:
                for k in nodes:
                    self.assertLessEqual(sp.dist(i, k), sp.dist(i, j) + sp.dist(j, k) + EPS)

    def test_paths_use_only_open_roads_and_sum_to_distance(self):
        for inst in self.insts:
            sp = ShortestPaths(inst)
            open_w = {}
            for r in inst.usable_roads():
                key = frozenset((r.u, r.v))
                open_w[key] = min(open_w.get(key, INF), r.distance)
            for j in sorted(inst.reachable_nodes()):
                path = sp.path(0, j)
                self.assertEqual((path[0], path[-1]), (0, j))
                total = 0.0
                for a, b in zip(path, path[1:]):
                    self.assertIn(frozenset((a, b)), open_w, "path uses a closed/missing road")
                    total += open_w[frozenset((a, b))]
                self.assertAlmostEqual(total, sp.dist(0, j), places=9)

    def test_route_distance_and_expand_route(self):
        inst = self.insts[0]
        sp = ShortestPaths(inst)
        route = [0, 5, 9, 0]
        self.assertAlmostEqual(sp.route_distance(route),
                               sp.dist(0, 5) + sp.dist(5, 9) + sp.dist(9, 0))
        full = sp.expand_route(route)
        self.assertEqual((full[0], full[-1]), (0, 0))
        for a, b in zip(full, full[1:]):               # every hop is a real open road
            self.assertTrue(any({r.u, r.v} == {a, b} for r in inst.usable_roads()))

    def test_expand_route_rejects_unreachable_leg(self):
        sp = ShortestPaths(edge_unreachable())
        with self.assertRaises(ValueError):
            sp.expand_route([0, 6, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
