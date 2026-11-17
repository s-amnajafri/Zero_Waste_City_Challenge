"""
runtime_experiment.py  --  measured execution time versus the derived complexity bounds

The proposal asks that theoretical complexity be distinguished from experimentally measured
execution time. Three experiments:

  1. Size sweep, WORST-CASE regime ("free"): capacity, duration and deadlines never bind, so
     Greedy serves every point. Then the number of Greedy pair evaluations is EXACTLY
     k n (n+1) / 2 (the bound), which tests the derivation, and Greedy time should grow as n^2.
  2. Size sweep, REALISTIC regime ("scaled"): capacity, duration and deadlines scale with n so
     that the main dataset's tightness (demand ~1.08x capacity, ~85-90% coverage) is kept.
     In the reported runs Greedy still reaches about 98-99% of the bound: points that cannot be
     served stay in the candidate list and are re-scanned every iteration, so the O(k n^2) bound is
     tight whenever coverage is high (it is only far from tight when very few points are served).
  3. Fleet sweep (n fixed, k varies): Greedy pair evaluations are exactly proportional to k.
     2-opt cost falls steeply with k: route length ~ n/k, and the number of improving moves per
     route itself grows with route length, so total cost ~ sum of r^3 ~ n^3 / k^2.
     (A first estimate with a fixed number of moves, n^2 / k, underestimates the fall; the
     measurements showed this.)

Also: growth of the exact solver (orderings enumerated vs the formula k * sum_s n!/(n-s)!).

Method: each stage is timed with time.perf_counter, median of `repeats` runs; the algorithms are
deterministic, so operation counts are identical in every run. Exponents come from a least-squares
line through (log n, log time). Timings depend on the machine; exponents and ratios do not (much).
"""

from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from data_generator import Instance, generate_instance
from experiments.experiments import RESULTS_DIR, print_table, write_csv
from algorithms.greedy import greedy_construct
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import improve_plan

BIG = 10 ** 7


# --------------------------------------------------------------------------
# Instance families
# --------------------------------------------------------------------------
def free_instance(n: int, k: int, seed: int = 5) -> Instance:
    """Worst case: nothing binds, so every point is served and all orderings are feasible."""
    inst = generate_instance(n_points=n, n_isolated=0, usable_roads=2 * n,
                             vehicle_capacities=(BIG,) * k, vehicle_durations=(float(BIG),) * k,
                             demand_ratio=0.1, seed=seed, name=f"free_n{n}_k{k}")
    for p in inst.points:
        p.deadline = float(BIG)
    return inst


def scaled_instance(n: int, k: int = 4, seed: int = 5) -> Instance:
    """Realistic: total capacity ~ 300 n kg, duration ~ 60 + 60 n/k min, deadlines x n/16,
    demand 1.08x capacity, one isolated point. At n=16, k=4 this resembles the main dataset."""
    cap = max(200, int(300 * n / k))
    dur = 60.0 + 60.0 * n / k
    inst = generate_instance(n_points=n, n_isolated=1, usable_roads=2 * n,
                             vehicle_capacities=(cap,) * k, vehicle_durations=(dur,) * k,
                             demand_ratio=1.08, seed=seed, name=f"scaled_n{n}_k{k}")
    inst.points = [replace(p, deadline=round(p.deadline * n / 16.0, 1)) for p in inst.points]
    return inst


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def measure(inst: Instance, repeats: int = 5) -> Dict[str, object]:
    t_paths: List[float] = []
    t_greedy: List[float] = []
    t_2opt: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sp = ShortestPaths(inst)
        t_paths.append((time.perf_counter() - t0) * 1000)
        g = greedy_construct(inst, sp)
        t_greedy.append(g.runtime_ms)
        res = improve_plan(inst, sp, g)
        t_2opt.append(res.runtime_ms)

    n_elig = g.candidates
    k = len(inst.vehicles)
    V = inst.node_count
    E = len(inst.usable_roads())
    imps = list(res.improvements.values())
    r_max = max((len(r.stops) for r in g.routes.values()), default=0)
    delta_evals = sum(i.delta_evaluations for i in imps)
    return dict(
        n=len(inst.points), k=k, V=V, E=E, eligible=n_elig, served=len(g.collected()),
        vehicles_used=sum(1 for r in g.routes.values() if r.stops), r_max=r_max,
        ms_paths=statistics.median(t_paths), ms_greedy=statistics.median(t_greedy),
        ms_2opt=statistics.median(t_2opt),
        greedy_evals=g.evaluations, greedy_bound=k * n_elig * (n_elig + 1) // 2,
        twoopt_delta_evals=delta_evals, twoopt_moves=sum(len(i.moves) for i in imps),
        twoopt_feas_checks=sum(i.feasibility_checks for i in imps),
        dijkstra_theory=V * (V + E) * math.log2(V),
        us_per_greedy_eval=1000 * statistics.median(t_greedy) / g.evaluations if g.evaluations else None,
        ns_per_dijkstra_unit=1e6 * statistics.median(t_paths) / (V * (V + E) * math.log2(V)),
        us_per_delta_eval=1000 * statistics.median(t_2opt) / delta_evals if delta_evals else None)


def loglog_fit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Least-squares line through (ln x, ln y): returns (slope, intercept, R^2)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts)
    return slope, intercept, (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def size_sweep(family: str, sizes: Sequence[int], k: int = 4, repeats: int = 5) -> List[Dict[str, object]]:
    make = free_instance if family == "free" else scaled_instance
    rows = []
    for n in sizes:
        row = measure(make(n, k), repeats)
        row["family"] = family
        rows.append(row)
    return rows


def fleet_sweep(n: int = 120, ks: Sequence[int] = (1, 2, 4, 8, 16), repeats: int = 5) -> List[Dict[str, object]]:
    rows = []
    for k in ks:
        row = measure(scaled_instance(n, k), repeats)
        row["family"] = "scaled"
        rows.append(row)
    return rows


def exponent_rows(rows: List[Dict[str, object]], label: str, n_min: int = 40) -> List[Dict[str, object]]:
    """Growth exponents from log-log fits, over all sizes and over n >= n_min only.
    The second fit is reported because at tiny n fixed Python overhead dominates the timings."""
    out = []
    for name, key, theory in (("Dijkstra all-pairs time", "ms_paths", "n^2 log n (E = O(n)): a little above 2"),
                              ("Greedy time", "ms_greedy", "n^2 (k fixed)"),
                              ("Greedy pair evaluations", "greedy_evals", "n^2 (k fixed)"),
                              ("2-opt time", "ms_2opt", "sum r^2 per scan x scans; ~n^3 if one long route"),
                              ("2-opt delta evaluations", "twoopt_delta_evals", "(moves+1) x sum r(r-1)/2")):
        ns = [r["n"] for r in rows]
        ys = [r[key] for r in rows]
        if not all(y and y > 0 for y in ys):
            continue
        s_all, _, r2_all = loglog_fit(ns, ys)
        big = [(n, y) for n, y in zip(ns, ys) if n >= n_min]
        s_big, _, r2_big = loglog_fit([b[0] for b in big], [b[1] for b in big])
        out.append(dict(family=label, quantity=name, exponent_all_sizes=s_all, r2_all=r2_all,
                        exponent_n_ge_40=s_big, r2_n_ge_40=r2_big, theory=theory))
    return out


# --------------------------------------------------------------------------
# Exact solver growth
# --------------------------------------------------------------------------
def exact_growth(ns: Sequence[int] = (4, 5, 6, 7, 8), ks: Sequence[int] = (1, 3)) -> List[Dict[str, object]]:
    from experiments.brute_force import solve_exact
    rows = []
    for n in ns:
        for k in ks:
            inst = free_instance(n, k, seed=3)
            sp = ShortestPaths(inst)
            res = solve_exact(inst, sp)
            formula = k * sum(math.factorial(n) // math.factorial(n - s) for s in range(1, n + 1))
            rows.append(dict(n=n, k=k, orderings=res.sequences_explored, formula=formula,
                             matches=res.sequences_explored == formula, ms=res.runtime_ms))
    return rows


# --------------------------------------------------------------------------
# Optional plot
# --------------------------------------------------------------------------
def make_plot(free: List[Dict], scaled: List[Dict], fleet: List[Dict], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, grid = plt.subplots(2, 2, figsize=(10.5, 8))
    ax = grid.ravel()
    # (a) time vs n, log-log
    for rows, name, mk in ((free, "worst case (all served)", "o-"), (scaled, "realistic (scaled)", "s--")):
        ns = [r["n"] for r in rows]
        ax[0].loglog(ns, [r["ms_paths"] for r in rows], mk, label=f"Dijkstra all-pairs, {name}")
        ax[0].loglog(ns, [r["ms_greedy"] for r in rows], mk, label=f"Greedy, {name}")
        ax[0].loglog(ns, [r["ms_2opt"] for r in rows], mk, label=f"2-opt, {name}")
    n0, t0 = free[0]["n"], free[0]["ms_greedy"]
    ax[0].loglog([r["n"] for r in free], [t0 * (r["n"] / n0) ** 2 for r in free], "k:", label="reference slope n^2")
    ax[0].set_xlabel("collection points n"); ax[0].set_ylabel("time (ms)"); ax[0].set_title("(a) Time vs n (k = 4)")
    ax[0].legend(fontsize=6.5); ax[0].grid(True, which="both", alpha=0.3)
    # (b) greedy work vs bound
    ns = [r["n"] for r in free]
    ax[1].plot(ns, [r["greedy_evals"] for r in free], "o-", label="measured (worst case)")
    ax[1].plot(ns, [r["greedy_bound"] for r in free], "k:", label="bound k n(n+1)/2")
    ax[1].plot([r["n"] for r in scaled], [r["greedy_evals"] for r in scaled], "s--", label="measured (realistic)")
    ax[1].set_xlabel("collection points n"); ax[1].set_ylabel("Greedy pair evaluations")
    ax[1].set_title("(b) Greedy work vs the bound"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # (c) time vs fleet size
    ks = [r["k"] for r in fleet]
    ax[2].plot(ks, [r["ms_greedy"] for r in fleet], "o-", label="Greedy")
    ax[2].plot(ks, [r["ms_2opt"] for r in fleet], "s-", label="2-opt")
    ax[2].plot(ks, [r["ms_paths"] for r in fleet], "^-", label="Dijkstra (independent of k)")
    ax[2].set_xlabel(f"vehicles k (n = {fleet[0]['n']})"); ax[2].set_ylabel("time (ms)")
    ax[2].set_title("(c) Time vs fleet size"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    # (d) operation counts vs k, log-log, with reference slopes
    g0, d0 = fleet[0]["greedy_evals"], fleet[0]["twoopt_delta_evals"]
    ax[3].loglog(ks, [r["greedy_evals"] for r in fleet], "o-", label="Greedy pair evaluations")
    ax[3].loglog(ks, [r["twoopt_delta_evals"] for r in fleet], "s-", label="2-opt delta evaluations")
    ax[3].loglog(ks, [g0 * k / ks[0] for k in ks], "k:", label="slope +1 (proportional to k)")
    ax[3].loglog(ks, [d0 * (ks[0] / k) ** 2 for k in ks], "k--", label="slope -2 (proportional to 1/k^2)")
    ax[3].set_xlabel(f"vehicles k (n = {fleet[0]['n']})"); ax[3].set_ylabel("operations")
    ax[3].set_title("(d) Operation counts vs fleet size"); ax[3].legend(fontsize=7); ax[3].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(sizes: Sequence[int] = (10, 20, 40, 80, 160, 320), repeats: int = 5) -> None:
    print(f"Runtime experiment (median of {repeats} runs; times in ms; timings are machine-dependent)")
    free = size_sweep("free", sizes, 4, repeats)
    scaled = size_sweep("scaled", sizes, 4, repeats)
    fleet = fleet_sweep(120, (1, 2, 4, 8, 16), repeats)

    cols = ["n", "V", "E", "eligible", "served", "vehicles_used", "r_max", "ms_paths", "ms_greedy", "ms_2opt",
            "greedy_evals", "greedy_bound", "twoopt_delta_evals", "twoopt_moves", "us_per_greedy_eval",
            "ns_per_dijkstra_unit"]
    print_table("1. Size sweep, WORST-CASE regime (nothing binds), k = 4", free, cols)
    print_table("2. Size sweep, REALISTIC regime (constraints scaled with n), k = 4", scaled, cols)
    print_table("3. Fleet sweep, realistic regime, n = 120", fleet,
                ["k", "served", "vehicles_used", "r_max", "ms_paths", "ms_greedy", "ms_2opt",
                 "greedy_evals", "greedy_bound", "twoopt_delta_evals", "twoopt_moves"])

    fits = exponent_rows(free, "worst case") + exponent_rows(scaled, "realistic")
    print_table("4. Measured growth exponents (slope of log quantity vs log n)", fits,
                ["family", "quantity", "exponent_all_sizes", "r2_all", "exponent_n_ge_40", "r2_n_ge_40", "theory"])

    exact = exact_growth()
    print_table("5. Exact solver: orderings enumerated vs formula k * sum n!/(n-s)!", exact,
                ["n", "k", "orderings", "formula", "matches", "ms"])

    write_csv("runtime_size_free.csv", free)
    write_csv("runtime_size_scaled.csv", scaled)
    write_csv("runtime_fleet.csv", fleet)
    write_csv("runtime_exponents.csv", fits)
    write_csv("runtime_exact_growth.csv", exact)
    fig = os.path.join(RESULTS_DIR, "figures", "runtime_scaling.png")
    print("\nFigure written to " + fig if make_plot(free, scaled, fleet, fig)
          else "\nmatplotlib not installed: plot skipped (tables and CSVs are still written)")


if __name__ == "__main__":
    main()
