"""
main.py  --  run the whole Zero-Waste City pipeline in one command

    python main.py                              main dataset (seed 42)
    python main.py data/edge_oversized.json     any saved dataset (see data_generator.py)
    python main.py --exact                      also compare with the exact optimum (<= 8 eligible points)
    python main.py --alpha 1 --beta 0.5         change the Greedy score weights

Pipeline:  shortest paths (Dijkstra) -> classify points -> Greedy construction -> 2-opt improvement
           -> independent feasibility check -> metrics.  Results are also written to results/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os

from experiments.brute_force import DEFAULT_MAX_POINTS, solve_exact
from data_generator import Instance, generate_instance
from feasibility import check_plan
from algorithms.greedy import GreedyParams, greedy_construct
from algorithms.greedy import format_report as greedy_report
from metrics import compute_metrics, format_metrics
from problem import ELIGIBLE, classify_points, is_better
from algorithms.shortest_path import ShortestPaths
from algorithms.two_opt import improve_plan
from algorithms.two_opt import format_report as two_opt_report

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
#step 1: load dataset

def run(inst: Instance, params: GreedyParams, max_iter: int, with_exact: bool) -> dict:       #inst contains out city information liek depot, collection pts, roads etc..
    sp = ShortestPaths(inst)              #step 2: compute shortest paths
    cls = classify_points(inst, sp)            #3 categories hain points ki: unreachable, individually infeasible, eligible.
    greedy = greedy_construct(inst, sp, params)     #step 3: select collection points and construct initial routes
    final = improve_plan(inst, sp, greedy, max_iter)   #step 4: improve the order of stops.

    print(greedy_report(inst, greedy, sp))
    print()
    print(two_opt_report(inst, final))
    chk = check_plan(inst, sp, final.plan())       #step 5: check feasibility of the final plan
    print(f"Independent checker on the final plan: {'PASS' if chk.ok else 'FAIL ' + str(chk.violations)}")
    print()
    #step 6: calculate distance, waste collected, coverage and other metrics
    m_greedy = compute_metrics(inst, sp, greedy.plan(), "Greedy", greedy.runtime_ms, cls)    #performane of initial greedy solution
    m_final = compute_metrics(inst, sp, final.plan(), "Greedy + 2-opt", greedy.runtime_ms + final.runtime_ms, cls)   #performance of final solution after 2-opt
    print(format_metrics(m_final))      #step 7: print reports and write to JSON

    #(optional)EXACT SOLVER (since heuristic algs don't guarantee optimal solution; so benchmark to evaluate; small dataset for this cz it can grow extremely quickly
    exact_info = None
    if with_exact:
        n_elig = sum(c.status == ELIGIBLE for c in cls.values())
        print()
        if n_elig > DEFAULT_MAX_POINTS:
            print(f"Exact comparison skipped: {n_elig} eligible points > limit of {DEFAULT_MAX_POINTS} "
                  f"(exact search grows factorially).")
        else:
            ex = solve_exact(inst, sp)
            w_gap = 100.0 * (ex.W - m_final.W) / ex.W if ex.W else 0.0       #finds % gap bw
            same = m_final.W == ex.W
            d_gap = 100.0 * (m_final.distance_km - ex.D) / ex.D if (same and ex.D) else None      #W is Urgency-weighted waste value (total priotiy-weighted amount of waste collected)
            optimal = not is_better((ex.W, ex.D), (m_final.W, m_final.distance_km))
            print(f"Exact optimum : W = {ex.W}, D = {ex.D:.2f} km  ({ex.sequences_explored} orderings, {ex.runtime_ms:.1f} ms)")
            print(f"Pipeline      : W = {m_final.W}, D = {m_final.distance_km:.2f} km  ->  "
                  f"W gap {w_gap:.1f}%, " + (f"D gap {d_gap:.1f}%" if d_gap is not None else "D gap n/a (different W)")
                  + ("  [OPTIMAL]" if optimal else ""))
            exact_info = dict(W=ex.W, D=ex.D, W_gap_pct=w_gap, D_gap_pct=d_gap, optimal=optimal,
                              plan={str(v): [inst.point(j).name for j in s] for v, s in ex.plan.items()})

    nm = lambda j: inst.point(j).name
    return dict(
        dataset=inst.name, alpha=params.alpha, beta=params.beta, feasible=chk.ok,
        classification={nm(j): dict(status=c.status, reasons=c.reasons) for j, c in cls.items()},
        routes={f"V{vid}": dict(stops=[nm(j) for j in r.stops], load_kg=r.load, capacity_kg=r.capacity,
                                distance_km=round(r.distance, 4), return_time_min=round(r.return_time, 4),
                                arrivals=[round(a, 4) for a in r.arrivals],
                                completions=[round(c, 4) for c in r.completions])
                for vid, r in final.routes.items() if r.stops},
        uncollected={nm(j): why for j, why in greedy.uncollected.items()},
        two_opt={f"V{vid}": dict(before_km=round(i.before_km, 4), after_km=round(i.after_km, 4),
                                 moves=len(i.moves), converged=i.converged,
                                 blocked_by_deadline=i.rejected_deadline)
                 for vid, i in final.improvements.items()},
        metrics_greedy=m_greedy.row(), metrics_final=m_final.row(), exact=exact_info)


#defines command line options
def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-Waste City: Greedy + 2-opt waste-collection planner")
    ap.add_argument("dataset", nargs="?", help="path to a saved dataset (.json); default = main dataset (seed 42)")
    ap.add_argument("--alpha", type=float, default=1.0, help="urgency weight (default 1.0)")
    ap.add_argument("--beta", type=float, default=0.5, help="additional-distance weight (default 0.5)")
    ap.add_argument("--max-iter", type=int, default=1000, help="2-opt iteration cap per route (default 1000)")
    ap.add_argument("--exact", action="store_true", help="also compute the exact optimum (small instances only)")
    args = ap.parse_args()

    #loading the dataset
    inst = Instance.load(args.dataset) if args.dataset else generate_instance()
    print(inst.summary())
    print()

    #running the pipeline
    result = run(inst, GreedyParams(args.alpha, args.beta), args.max_iter, args.exact)

    #saving the result to JSON file (step 7)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{inst.name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)     #python dictionary is converted to JSON and written to file
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
