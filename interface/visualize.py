"""
visualize.py  --  figures for the report and for the Streamlit interface

Every function returns a matplotlib Figure (so the same code serves the report and the app) and
contains no algorithm logic: it only draws results produced by the other modules.

    python visualize.py            write all figures to results/figures/
    python visualize.py --quick    same, with fewer random instances (faster)

Figures
-------
  block_diagram.png          the system pipeline (required for the report)
  main_routes.png            road network + routes before / after 2-opt on the main dataset
  main_final_routes.png      the same, final plan only, as one large map (used in the report)
  over_demand_routes.png     the over-demand edge case (before / after)
  over_demand_final_routes.png  the over-demand edge case, final plan only
  baselines_30_seeds.png     the four construction methods (+ 2-opt) averaged over random instances
  sensitivity.png            fleet size, vehicle capacity, urgency weight, deadline tightness
  optimality_gap.png         heuristics vs the exact optimum
Map legend: circle size = waste (kg); colour = urgency (yellow low, red high); X = unreachable point;
grey line = open road; red dashed line = closed road; numbered discs = stop order on each route.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    HAVE_MPL = True
except ImportError:                                   # the algorithms never need matplotlib
    HAVE_MPL = False

from data_generator import Instance
from problem import Plan, plan_distance
from algorithms.shortest_path import ShortestPaths

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")


def _require() -> None:
    if not HAVE_MPL:
        raise RuntimeError("matplotlib is required for figures: pip install matplotlib")


def _xy(inst: Instance, i: int):
    return inst.depot if i == 0 else (inst.point(i).x, inst.point(i).y)


# --------------------------------------------------------------------------
# Network and routes
# --------------------------------------------------------------------------
def draw_network(ax, inst: Instance, sp: ShortestPaths, label_points: bool = True) -> None:
    _require()
    for r in inst.roads:
        (x1, y1), (x2, y2) = _xy(inst, r.u), _xy(inst, r.v)
        if r.accessible:
            ax.plot([x1, x2], [y1, y2], color="#c9ced6", lw=1.0, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color="#d9534f", lw=1.0, ls=(0, (3, 3)), zorder=1)
    unreachable = {p.id for p in sp.unreachable_points()}
    cmap = plt.get_cmap("YlOrRd")
    for p in inst.points:
        size = 40 + p.waste / 3.0
        if p.id in unreachable:
            ax.scatter(p.x, p.y, s=size, marker="X", c="#444444", zorder=3)
        else:
            ax.scatter(p.x, p.y, s=size, c=[cmap((p.urgency - 1) / 9.0)], edgecolors="#333333",
                       linewidths=0.6, zorder=3)
        if label_points:
            ax.annotate(p.name, (p.x, p.y), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.scatter(*inst.depot, marker="s", s=120, c="black", zorder=4)
    ax.annotate("Depot", inst.depot, textcoords="offset points", xytext=(7, -13), fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_routes(ax, inst: Instance, sp: ShortestPaths, plan: Plan) -> None:
    """Routes follow the real road paths (expand_route), with numbered stop order."""
    _require()
    palette = plt.get_cmap("tab10")
    colour = {v.id: palette(k % 10) for k, v in enumerate(inst.vehicles)}
    for vid in sorted(plan):
        stops = plan[vid]
        if not stops:
            continue
        full = sp.expand_route([0] + list(stops) + [0])
        xs, ys = zip(*[_xy(inst, i) for i in full])
        ax.plot(xs, ys, color=colour[vid], lw=2.4, alpha=0.9, zorder=2)
        for order, j in enumerate(stops, 1):
            x, y = _xy(inst, j)
            ax.text(x, y, str(order), color="white", fontsize=8, ha="center", va="center", fontweight="bold",
                    zorder=5, bbox=dict(boxstyle="circle,pad=0.18", fc=colour[vid], ec="none"))


def _vehicle_legend(fig, inst: Instance, plans: Sequence[Plan]) -> None:
    palette = plt.get_cmap("tab10")
    used = {vid for plan in plans for vid, s in plan.items() if s}
    handles = [Line2D([0], [0], color=palette(k % 10), lw=3, label=f"V{v.id} ({v.capacity} kg)")
               for k, v in enumerate(inst.vehicles) if v.id in used]
    handles += [Line2D([0], [0], color="#c9ced6", lw=1.5, label="open road"),
                Line2D([0], [0], color="#d9534f", lw=1.2, ls=(0, (3, 3)), label="closed road"),
                Line2D([0], [0], marker="X", color="w", markerfacecolor="#444444", markersize=9, label="unreachable")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=9, frameon=False)


def plot_routes(inst: Instance, sp: ShortestPaths, plan: Plan, title: str = ""):
    _require()
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    draw_network(ax, inst, sp)
    draw_routes(ax, inst, sp, plan)
    ax.set_title(f"{title}   {plan_distance(sp, plan):.2f} km".strip(), fontsize=11)
    _vehicle_legend(fig, inst, [plan])
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def plot_before_after(inst: Instance, sp: ShortestPaths, before: Plan, after: Plan,
                      titles=("Greedy", "Greedy + 2-opt")):
    _require()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.9))
    for ax, plan, t in zip(axes, (before, after), titles):
        draw_network(ax, inst, sp)
        draw_routes(ax, inst, sp, plan)
        ax.set_title(f"{t}: {plan_distance(sp, plan):.2f} km", fontsize=11)
    _vehicle_legend(fig, inst, [before, after])
    fig.suptitle(f"Routes on '{inst.name}'   (circle size = waste, colour = urgency, numbers = stop order)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# Block diagram
# --------------------------------------------------------------------------
def block_diagram():
    _require()
    fig, ax = plt.subplots(figsize=(15, 6.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(-1, 103)
    ax.set_ylim(-1, 57)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec="#2f3b4c", lw=1.3))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7.6, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.5, color="#2f3b4c"))
        if label:
            ax.text((x1 + x2) / 2 + 0.8, (y1 + y2) / 2, label, fontsize=7.2, style="italic", va="center")

    w, h, gap = 14.2, 17.5, 2.9
    xs = [1 + i * (w + gap) for i in range(6)]
    yA = 33
    main = [("1. Dataset", "points: waste, urgency,\nservice time, deadline\nroads (some closed)\nfleet: capacity, duration\n(data_generator.py)", "#e8f1fb"),
            ("2. Shortest paths", "Dijkstra, adjacency list\n+ binary heap\nclosed roads excluded\nall-pairs distances\n(shortest_path.py)", "#e8f1fb"),
            ("3. Classification", "unreachable\nindividually infeasible\neligible\nconstraints C1-C6,\nobjective (problem.py)", "#e8f1fb"),
            ("4. Greedy construction", "score = a*urgency/10\n- b*extra/Dmax\nall feasible (vehicle,\npoint) pairs, best wins\n(greedy.py)", "#fdf0d5"),
            ("5. 2-opt improvement", "reverse route segments\naccept only if shorter\nand deadlines still hold\n(two_opt.py)", "#fdf0d5"),
            ("6. Check + metrics", "independent checker\n(feasibility.py)\ndistance, coverage, waste,\nutilisation, W (metrics.py)", "#e3f4e3")]
    for x, (t, b, c) in zip(xs, main):
        box(x, yA, w, h, t, b, c)
    for a, b in zip(xs, xs[1:]):
        arrow(a + w + 0.6, yA + h / 2, b - 0.6, yA + h / 2)

    yB = 6.5
    bench = [(xs[3], "Baselines", "first-fit, distance-only,\nurgency-only\n(baselines.py)", "Greedy"),
             (xs[4], "Exact solver", "<= 8 points: optimum of\nthe same objective\n(brute_force.py)", "2-opt"),
             (xs[5], "Experiments", "sensitivity, runtime,\noptimality gap\n(experiments.py ...)", "Metrics")]
    for x, t, b, _ in bench:
        box(x, yB, w, 15.5, t, b, "#f1e6f7")
    for (x, _, _, _) in bench:
        arrow(x + w / 2, yB + 15.5 + 0.7, x + w / 2, yA - 0.7)
    ax.text(xs[3] - 1, yB + 9, "Evaluation and\nbenchmarks", ha="right", va="center", fontsize=9, style="italic")
    ax.text(51, 54.2, "Zero-Waste City Challenge: system pipeline", ha="center", fontsize=12.5, fontweight="bold")
    ax.text(51, 1.2, "Greedy decides WHICH points are served and by WHICH vehicle; 2-opt only reorders stops "
                     "within a route (the served set and W never change).", ha="center", fontsize=8.8, style="italic")
    return fig


# --------------------------------------------------------------------------
# Result charts
# --------------------------------------------------------------------------
def baselines_chart(rows: List[Dict[str, object]]):
    _require()
    rows = [r for r in rows if str(r["method"]).endswith("+ 2-opt")]
    names = [str(r["method"]).replace(" + 2-opt", "").replace("greedy: ", "").replace(" a=0", "").replace(" b=0", "")
             for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6))
    spec = [("mean_W", "mean objective W (higher is better)"),
            ("mean_urgent_cov_pct", "urgent points served, % (higher is better)"),
            ("mean_km", "mean total distance, km (lower is better)"),
            ("mean_km_per_100kg", "km per 100 kg collected (lower is better)")]
    colours = ["#9aa5b1", "#6baed6", "#fd8d3c", "#31a354"]
    for ax, (key, title) in zip(axes.ravel(), spec):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=colours[:len(rows)])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
        ax.set_title(title, fontsize=10)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}" if v > 500 else f"{v:.1f}", ha="center",
                    va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Construction methods, each followed by 2-opt (average over random instances)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def sensitivity_chart(fleet, capacity, alpha, deadlines):
    _require()
    fig, grid = plt.subplots(2, 2, figsize=(10.5, 7.6))
    axes = grid.ravel()

    def panel(ax, rows, labels, title, bar_key, bar_label, bar_max=None):
        x = range(len(rows))
        ax.bar(x, [r[bar_key] for r in rows], color="#6baed6")
        for i, r in enumerate(rows):                      # make exclusions visible, not hidden in the percentage
            if r.get("infeasible"):
                ax.text(i, r[bar_key] / 2, f"{r['infeasible']} point\ninfeasible", ha="center",
                        va="center", fontsize=7, color="white", fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(bar_label, color="#2b6ca3", fontsize=9)
        if bar_max:
            ax.set_ylim(0, bar_max)
        ax2 = ax.twinx()
        ax2.plot(list(x), [r["km_final"] for r in rows], "o-", color="#d94801")
        ax2.set_ylabel("distance, km", color="#d94801", fontsize=9)
        ax.set_title(title, fontsize=10)

    panel(axes[0], fleet, [str(r["vehicles"]) for r in fleet], "Fleet size (vehicles)", "collected", "points collected", 16)
    panel(axes[1], capacity, [r["setting"].replace("capacity x", "x") for r in capacity],
          "Vehicle capacity (scale factor)", "collected", "points collected", 16)
    panel(axes[2], alpha, [r["setting"].split(",")[0].replace("alpha=", "a=").replace("urgency-only (beta=0)", "b=0")
                           for r in alpha], "Urgency weight (beta = 0.5)", "urgent_cov", "urgent eligible points served (%)", 105)
    panel(axes[3], deadlines, [r["setting"].replace("deadlines x", "x") for r in deadlines],
          "Deadline tightness (scale factor)", "urgent_cov", "urgent eligible points served (%)", 105)
    fig.suptitle("Sensitivity analysis on the main dataset (bars: coverage, line: distance after 2-opt)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def gap_chart(summary: List[Dict[str, object]]):
    _require()
    wanted = ["greedy: default a=1,b=0.5 + 2-opt", "greedy: urgency-only b=0 + 2-opt",
              "greedy: distance-only a=0 + 2-opt", "first-fit (id order) + 2-opt"]
    labels = ["default (a=1, b=0.5)", "urgency-only", "distance-only", "first-fit"]
    ratios = sorted({r["demand_ratio"] for r in summary})
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    width = 0.2
    for i, (m, lab) in enumerate(zip(wanted, labels)):
        vals = [next(r["mean_W_gap_pct"] for r in summary if r["method"] == m and r["demand_ratio"] == ratio)
                for ratio in ratios]
        bars = ax.bar([k + (i - 1.5) * width for k in range(len(ratios))], vals, width, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"demand / capacity = {r}" for r in ratios])
    ax.set_ylabel("mean gap to the exact optimum in W (%)")
    ax.set_title("Optimality gap (8 points, 3 vehicles; exact optimum from brute force)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Write everything
# --------------------------------------------------------------------------
def save_all(outdir: str = FIG_DIR, quick: bool = False) -> List[str]:
    _require()
    from data_generator import edge_over_demand, generate_instance
    from experiments.experiments import (exp_alpha, exp_baselines_over_seeds, exp_capacity, exp_deadlines, exp_fleet)
    from algorithms.greedy import greedy_construct
    from experiments.optimality_gap import gap_experiment
    from algorithms.two_opt import improve_plan

    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []

    def save(fig, name):
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    save(block_diagram(), "block_diagram.png")
    for inst, name, final_name in ((generate_instance(), "main_routes.png", "main_final_routes.png"),
                                   (edge_over_demand(), "over_demand_routes.png", "over_demand_final_routes.png")):
        sp = ShortestPaths(inst)
        g = greedy_construct(inst, sp)
        t = improve_plan(inst, sp, g)
        save(plot_before_after(inst, sp, g.plan(), t.plan()), name)
        save(plot_routes(inst, sp, t.plan(), "Greedy + 2-opt:"), final_name)

    seeds = range(1, 11) if quick else range(1, 31)
    save(baselines_chart(exp_baselines_over_seeds(seeds)), "baselines_30_seeds.png")
    main_inst = generate_instance()
    save(sensitivity_chart(exp_fleet(main_inst), exp_capacity(main_inst), exp_alpha(main_inst),
                           exp_deadlines(main_inst)), "sensitivity.png")
    _, summary = gap_experiment(range(1, 9) if quick else range(1, 41), (0.9, 1.25, 1.6))
    save(gap_chart(summary), "optimality_gap.png")
    return written


if __name__ == "__main__":
    try:
        for p in save_all(quick="--quick" in sys.argv):
            print("wrote", p)
    except RuntimeError as err:                       # e.g. matplotlib not installed
        print(err)
        sys.exit(1)
