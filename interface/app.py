"""
app.py  --  interactive Zero-Waste City planner (Streamlit)

    pip install -r requirements.txt
    streamlit run app.py

A thin interface: every number comes from app_logic.py, which calls the same algorithms as
`python main.py`. Change the scenario in the sidebar and the plan, map and metrics update at once.
"""

import json

import pandas as pd
import streamlit as st

import app_logic as A
import visualize as viz

st.set_page_config(page_title="Zero-Waste City Planner", layout="wide")


def show_table(rows):
    """Display a list of dicts as a table."""
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption("Nothing to show for this scenario.")


# --------------------------------------------------------------------------
# Sidebar: scenario controls
# --------------------------------------------------------------------------
sb = st.sidebar
sb.title("Scenario")
choice = sb.selectbox("Dataset", A.DATASET_CHOICES)

seed, n_points, n_isolated, demand_ratio = 1, 20, 1, 1.1
if choice == A.RANDOM_CHOICE:
    n_points = sb.slider("Collection points", 8, 60, 20)
    n_isolated = sb.slider("Isolated (cut-off) points", 0, 2, 1)
    demand_ratio = sb.slider("Demand / fleet capacity", 0.5, 2.0, 1.1, 0.1)
    seed = sb.number_input("Random seed", min_value=0, max_value=9999, value=1, step=1)

sb.subheader("Fleet and limits")
own_fleet = sb.checkbox("Use the dataset's own fleet", value=True)
fleet_size = None
if not own_fleet:
    fleet_size = sb.slider("Number of vehicles", 2, 6, 4)
cap_scale = sb.slider("Vehicle capacity (scale)", 0.5, 2.0, 1.0, 0.1)
dl_scale = sb.slider("Deadline tightness (scale)", 0.5, 2.0, 1.0, 0.1)

sb.subheader("Greedy score")
alpha = sb.slider("Urgency weight (alpha)", 0.0, 4.0, 1.0, 0.25)
beta = sb.slider("Distance weight (beta)", 0.0, 2.0, 0.5, 0.25)
with sb.expander("Advanced"):
    max_iter = st.number_input("2-opt iteration cap per route", min_value=10, max_value=100000, value=1000, step=100)

# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------
inst = A.build_instance(choice, seed=int(seed), n_points=int(n_points), n_isolated=int(n_isolated),
                        demand_ratio=float(demand_ratio), fleet_size=fleet_size,
                        capacity_scale=float(cap_scale), deadline_scale=float(dl_scale))
res = A.run_all(inst, float(alpha), float(beta), int(max_iter))
mg, mf = res.m_greedy, res.m_final

st.title("Zero-Waste City Challenge")
st.caption("Greedy route construction followed by 2-opt local search. "
           "Change the scenario in the sidebar; everything below updates immediately.")

if res.check.ok:
    st.success("Plan verified by the independent checker: capacity, deadlines, shift limits, open roads and "
               "uniqueness all hold.")
else:
    st.error("The independent checker found violations: " + "; ".join(res.check.violations))

unreachable = [p.name for p in res.sp.unreachable_points()]
if unreachable:
    st.info("Unreachable (no open road from the depot), reported but not planned: " + ", ".join(unreachable))
infeasible = [inst.point(j).name for j, c in res.cls.items() if c.status == "individually_infeasible"]
if infeasible:
    st.warning("Excluded before planning because no vehicle could serve them even alone: " + ", ".join(infeasible))

# --------------------------------------------------------------------------
# Headline numbers
# --------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Points collected", f"{mf.n_collected} / {mf.n_reachable}")
c2.metric("Waste collected", f"{mf.kg_collected:,} kg")
c3.metric("Urgent points served",
          "n/a" if mf.urgent_eligible == 0 else f"{mf.urgent_collected} / {mf.urgent_eligible}")
c4.metric("Objective W (urgency x waste)", f"{mf.W:,}")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Total distance", f"{mf.distance_km:.2f} km", delta=f"{mf.distance_km - mg.distance_km:+.2f} km vs Greedy",
          delta_color="inverse")
d2.metric("Fleet utilisation", f"{mf.fleet_utilization_pct:.1f} %")
d3.metric("Tightest deadline slack", "n/a" if mf.min_slack_min is None else f"{mf.min_slack_min:.1f} min")
d4.metric("Run time (Greedy + 2-opt)", f"{mf.runtime_ms:.1f} ms")

with st.expander("Dataset summary"):
    st.code(inst.summary())

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_map, tab_routes, tab_metrics, tab_methods, tab_exact, tab_about = st.tabs(
    ["Map", "Routes and outcomes", "Metrics", "Compare methods", "Exact optimum", "About"])

with tab_map:
    view = st.radio("View", ["Greedy + 2-opt", "Greedy only (before 2-opt)", "Side by side"], horizontal=True)
    if not viz.HAVE_MPL:
        st.warning("Maps need matplotlib: pip install matplotlib")
    else:
        if view == "Side by side":
            fig = viz.plot_before_after(inst, res.sp, res.greedy.plan(), res.final.plan())
        elif view == "Greedy only (before 2-opt)":
            fig = viz.plot_routes(inst, res.sp, res.greedy.plan(), "Greedy:")
        else:
            fig = viz.plot_routes(inst, res.sp, res.final.plan(), "Greedy + 2-opt:")
        st.pyplot(fig)
        viz.plt.close(fig)
        st.caption("Circle size = waste (kg); colour = urgency (yellow low, red high); X = unreachable point; "
                   "grey = open road; red dashed = closed road; numbered discs = stop order on each route.")

with tab_routes:
    st.subheader("Vehicles")
    show_table(A.vehicle_rows(res))
    st.subheader("What happened to every point")
    show_table(A.outcome_rows(res))
    st.subheader("Stops with times (final plan)")
    stops = A.stop_rows(inst, res.final.routes)
    show_table(stops)

with tab_metrics:
    st.subheader("Greedy versus Greedy + 2-opt")
    show_table(A.metrics_rows(mg, mf))
    st.subheader("What 2-opt did on each route")
    show_table(A.improvement_rows(res))
    if res.blocked:
        st.caption(f"{res.blocked} improving move(s) were rejected because they would have broken a deadline: "
                   "shorter routes are only accepted if every deadline still holds.")

with tab_methods:
    st.subheader("Four construction methods, each with and without 2-opt")
    show_table(A.baseline_rows(inst, res.sp))
    st.caption("first-fit has no scoring; distance-only uses alpha = 0; urgency-only uses beta = 0; "
               "default uses alpha = 1, beta = 0.5. All use identical constraints and metrics.")

with tab_exact:
    ex = A.exact_comparison(res)
    if ex["status"] == "skipped":
        st.info(ex["message"])
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Exact optimum W", f"{ex['W_exact']:,}")
        e2.metric("Pipeline W", f"{ex['W_pipeline']:,}", delta=f"{ex['W_pipeline'] - ex['W_exact']:,}")
        e3.metric("Gap in W", f"{ex['w_gap_pct']:.1f} %")
        st.write("Distance: pipeline " + f"{ex['km_pipeline']:.2f} km versus optimum {ex['km_exact']:.2f} km"
                 + (f" (gap {ex['d_gap_pct']:.1f} %)." if ex["d_gap_pct"] is not None
                    else " (not comparable: the plans serve different points)."))
        if ex["optimal"]:
            st.success("The pipeline found the exact optimum on this instance.")
        else:
            st.warning("The pipeline is not optimal here; the optimum serves:")
            st.code("\n".join(f"{v}: Depot -> {' -> '.join(s)} -> Depot" for v, s in ex["plan"].items()))
        st.caption(f"Exact search enumerated {ex['orderings']:,} feasible orderings in {ex['runtime_ms']:.1f} ms.")

with tab_about:
    st.markdown(
        "**What it does.** Greedy decides *which* points each truck serves, scoring every feasible "
        "(vehicle, point) pair by `alpha x urgency/10 - beta x extra distance / Dmax`. 2-opt then reorders the stops "
        "inside each route to shorten it, accepting a change only if every capacity, deadline and shift limit still holds. "
        "Dijkstra supplies the distances over the open roads.\n\n"
        "**What it does not do.** The problem is NP-hard, so no global optimum is guaranteed. "
        "Use the *Exact optimum* tab on small random instances (8 points) to see how far the plan is from the best possible.\n\n"
        "**Try this.** Set alpha to 0 and watch urgent points get missed; tighten the deadlines and watch points become "
        "infeasible; shrink capacity and see which points are left behind and why.")

st.divider()
st.download_button("Download full results (JSON)",
                   data=json.dumps(A.export_json(inst, float(alpha), float(beta), int(max_iter)), indent=2),
                   file_name=f"{inst.name}.json", mime="application/json")
st.download_button("Download stops (CSV)", data=A.rows_to_csv(stops), file_name=f"{inst.name}_stops.csv",
                   mime="text/csv")
