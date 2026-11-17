# Zero-Waste City Challenge

CT-363 Design and Analysis of Algorithms. A decision-support system that plans waste-collection
routes with two algorithmic paradigms working together:

1. **Greedy construction** decides *which* points are served and by *which* vehicle.
2. **2-opt local search** reorders the stops inside each route to shorten it.

Dijkstra supplies the road distances. An exact brute-force solver (small instances only) measures
how far the heuristics are from the true optimum. Neither heuristic guarantees global optimality
(the problem is NP-hard); the report proves what can be proved (feasibility, 2-opt termination)
and measures the rest.

## Run it

All commands are run from inside this folder.

```
python data_generator.py              # writes the datasets to data/
python main.py                        # whole pipeline on the main dataset (also writes results/main.json)
python main.py data/edge_oversized.json
python main.py data/small_seed1.json --exact     # also compare with the exact optimum
python main.py --alpha 1 --beta 0.5   # Greedy score weights

python experiments.py                 # baselines + sensitivity analysis  -> results/*.csv
python optimality_gap.py              # heuristics vs exact optimum       -> results/*.csv
python runtime_experiment.py          # measured time vs complexity       -> results/*.csv + figure
python visualize.py                   # all report figures                -> results/figures/*.png
python -m unittest discover -s tests -v    # 180 tests

pip install -r requirements.txt       # once, for the interactive app
streamlit run app.py                  # interactive planner in the browser
```

The algorithms, experiments and core tests need only the Python standard library. `visualize.py` and the plot in `runtime_experiment.py` need `matplotlib`; the interactive app needs `streamlit` (which brings `pandas`) and `matplotlib`: `pip install -r requirements.txt`. Without these, the affected tests are skipped and the figure scripts print a clear message; everything else still runs. Developed and tested on Python 3.12.

## Files

| File | What it does |
|---|---|
| `data_generator.py` | Builds datasets: weighted road network (some roads closed), collection points (waste, urgency, service time, deadline), vehicles. Seeded, parameterised, plus 5 hand-built edge cases. |
| `shortest_path.py` | Dijkstra (adjacency list + binary heap), path reconstruction, unreachable handling. Closed roads are excluded. |
| `problem.py` | The shared formal definitions: constraints C1-C6, the lexicographic objective, and the point classification. Everything else imports from here. |
| `greedy.py` | Algorithm 1: global greedy construction. Specification, pseudocode and complexity are in the module docstring. |
| `two_opt.py` | Algorithm 2: 2-opt local search, feasibility-preserving. Specification, pseudocode, termination argument and complexity in the docstring. |
| `feasibility.py` | Independent plan checker: rebuilds loads, arrival times and distances from the road network. |
| `metrics.py` | Evaluation metrics (distance, coverage, waste, utilisation, deadline compliance, W, runtime). |
| `baselines.py` | Reference methods (first-fit, distance-only, urgency-only) for comparison. |
| `experiments.py` | Baseline comparison, sensitivity analysis (fleet, capacity, urgency weight, deadlines), point removal. |
| `brute_force.py` | Exact solver for small instances (optimality benchmark). |
| `optimality_gap.py` | Heuristics vs the exact optimum, and a hand-checkable Greedy counterexample. |
| `runtime_experiment.py` | Measured execution time versus the derived complexity (size sweeps in a worst-case and a realistic regime, fleet sweep, exact-solver growth). |
| `visualize.py` | Route maps (before / after 2-opt), block diagram, and result charts. Returns matplotlib figures, so the same code can serve a GUI. |
| `main.py` | One-command pipeline. |
| `app.py` | Interactive Streamlit planner (thin interface over `app_logic.py`). |
| `app_logic.py` | Everything the app computes (scenario building, tables, exact comparison, export); no Streamlit import, fully unit-tested. |
| `tests/` | 180 unit and end-to-end tests. |

## The model in brief

* Node 0 is the depot; every vehicle starts and ends there. Distances in km, times in minutes,
  waste in kg; travel time = distance / 30 km/h.
* **Constraints:** vehicle capacity; **completion deadline** (arrival + service <= deadline);
  route-duration limit (including the return trip); open roads only; each point served at most once;
  only the available vehicles.
* **Objective (lexicographic):** maximise `W = sum(urgency x waste)` of served points, then minimise
  total distance.
* **Point classes** (reported separately): *unreachable* (no open-road path), *individually
  infeasible* (no vehicle could serve it even alone), *eligible* (given to Greedy). Eligible points
  end up *collected* or *uncollected*, and every uncollected point has a reason
  (capacity / deadline / route duration).
* **Greedy score:** `alpha * urgency/10 - beta * extra_distance / D_max`, with
  `extra_distance = d(i,j) + d(j,0) - d(i,0)`; defaults alpha = 1.0, beta = 0.5. All feasible
  (vehicle, point) pairs are scored at every step, the best pair wins; ties go to higher urgency,
  smaller extra distance, smaller point id, then best-fit vehicle, then smaller vehicle id.
* **2-opt:** best-improvement segment reversals inside one route; a move is accepted only if it
  shortens the route and every completion deadline and the duration limit still hold. Only
  deadlines can block an improving move (the duration limit is implied, since no vehicle waits).

## Interactive app

```
pip install -r requirements.txt
streamlit run app.py
```

The sidebar chooses the dataset (main, the five edge cases, or a random instance with adjustable size, isolated points and demand), the fleet (own fleet or 2-6 vehicles), a capacity scale, a deadline-tightness scale, the Greedy weights alpha and beta, and the 2-opt cap. The plan updates on every change. Tabs: **Map** (Greedy, Greedy + 2-opt or side by side), **Routes and outcomes** (every point with its class and outcome, stops with times and slack), **Metrics** (Greedy against Greedy + 2-opt, and what 2-opt did per route), **Compare methods** (the four baselines, each with and without 2-opt), **Exact optimum** (for random instances of 8 points, the gap to the true optimum) and **About**. Results can be downloaded as JSON (the same file as `python main.py`) and CSV.

All computation lives in `app_logic.py`; `app.py` only draws widgets. The tests run `app.py` end to end against a strict stand-in for Streamlit (`tests/streamlit_stub.py`) that rejects the mistakes which break real apps (mixed int/float slider arguments, duplicate widgets, calls before `set_page_config`, unknown API names). That checks that the script runs and uses the API consistently; it does not replace opening the real app once.

## Complexity (excluding shortest paths)

| Part | Time | Space |
|---|---|---|
| Dijkstra, one source | O((V+E) log V) | O(V+E) |
| Dijkstra, all pairs | O(V (V+E) log V) | O(V^2) |
| Greedy (n points, k vehicles) | O(k n^2) | O(n + k) |
| 2-opt, one route of r stops | O(I (r^2 + F r)), typically O(I r^2) | O(r) |
| Exact search | O(k n n! + k 4^n) | O(k 2^n) |

The derivations are in the module docstrings. The code counts its own work (Greedy pair
evaluations, 2-opt delta evaluations, exact-search orderings) and the tests check those counts
against the formulas.

## Measured runtime (`runtime_experiment.py`, median of 5 runs, k = 4)

* **Greedy:** when nothing binds, pair evaluations equal k n(n+1)/2 *exactly* at every size
  (e.g. 205,440 at n = 320); in a realistic regime they reach 98-99% of the bound, so the
  O(k n^2) bound is tight, not loose. Fitted time exponent for n >= 40: about 2 (1.9-2.0 between runs).
* **Dijkstra all-pairs:** exponent about 1.9-2.1 for n >= 40, consistent with n^2 log n (the log factor is too
  small to separate from timing noise at these sizes); time divided by V(V+E)log V settles near 22 ns.
* **2-opt:** exponent about 2.9-3.1 for n >= 40. For one route of r stops the number of delta
  evaluations is exactly (moves + 1) r(r-1)/2, and the number of moves grows roughly in proportion
  to r. With more vehicles the routes get shorter and the 2-opt cost falls roughly as 1/k^2
  (about 200x from k = 1 to k = 16), while Greedy's evaluation count grows exactly in proportion to k.
* **Exact solver:** orderings enumerated equal k * sum n!/(n-s)! exactly (328,800 at n = 8, k = 3).

Absolute timings depend on the machine, and fitted *timing* exponents vary by about +/-0.1 between runs; the
operation counts are deterministic and are the exact evidence.

## Main dataset results (seed 42)

15 reachable points (4,859 kg) against 4,500 kg of fleet capacity, plus one unreachable point (P16).

* Greedy serves 13 of 15 reachable points (3,986 kg), including all 6 urgent points, in 175.54 km.
  The two points left behind are the lowest-urgency ones (capacity).
* 2-opt reduces the distance to 173.09 km (1.40%); one further improving move was blocked because
  it would have broken a deadline.
* Averaged over 30 random instances, the default weights trade about 5% of W for about 22% less
  distance compared with urgency-only Greedy (see `experiments.py`).
* On small instances (8 points, 3 vehicles) the pipeline is rarely exactly optimal; 2-opt cannot
  fix which vehicle serves which point (see `optimality_gap.py`).

## Limitations

Simulated data and a static network (no traffic or waste-generation forecasting); constant vehicle
speed; Greedy does not try to minimise the number of vehicles used; 2-opt works inside one route
only (no moves between routes); exact search is limited to about 8 eligible points. Heuristic
results can behave non-monotonically when parameters change.
