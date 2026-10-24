# Gazzola3001: F1 2026 overtake probability

MMA3001 Numerical Methods and Machine Learning, individual project.

In 2026 Formula 1 replaced DRS with **Overtake Mode** (extra battery energy
for a car within one second of the car ahead) and **Boost** (maximum engine
plus battery power, usable anywhere). This project estimates the probability
that an attacker completes an overtake, given the race situation: gap, battery
charge, Boost and Overtake Mode use, tyres, weather, where on the lap, drivers,
cars and circuit.

It combines a **physics-based simulator** (the two cars' equations of motion
integrated with RK4, and the exact probability by Gauss-Legendre quadrature)
with a **neural network** (an ensemble of five) that learns the probability
from 0/1 outcomes, as it would from real race data, and is validated against
the simulator's exact answer.

```
$ python -m overtake.predict --condition dry --tyre soft --segment straight --battery 100 --gap 1.0

Scenario
  Track:    dry, 35 C, generic, straight (800 m run)
  Attacker: soft (5 laps), battery 100%, Boost on, Overtake Mode on, skill 85
  Defender: medium (15 laps), battery 60%, Boost on, skill 85
  Gap:      1.00 s

P(overtake) = 44.0%
  without Boost: 0.7%  (Boost adds +43.2 points)

Same situation at other gaps
  0.25 s   85.7%
  0.50 s   77.7%
  0.75 s   63.6%
  1.00 s   44.0%
  1.25 s    8.9%
  1.50 s    3.6%
```

**Full write-up:** [docs/project_documentation.md](docs/project_documentation.md)
(engineering problem, inputs and outputs, method, alternatives, validation,
performance, limitations, AI use). A PDF and Word copy are in `docs/`.

## Results at a glance

| Test set (30,000 held-out attempts) | Neural network (ensemble of 5) | Gradient boosting | Logistic regression |
|---|---:|---:|---:|
| Mean error vs true probability (points) | **2.0** | 3.8 | 15.7 |
| Log loss | **0.257** | 0.267 | 0.411 |
| ROC AUC | **0.948** | 0.944 | 0.853 |
| Calibration error | 0.005 | 0.014 | **0.004** |

Numerical verification: RK4 converges at order 4.0 and Euler at 1.0 against
analytical solutions; the full model's braking-point error at the chosen step
(0.05 s) is 0.059 ms, and the step is 2.5 times inside the first battery
overdraw (0.125 s), the stiffest part of the model; 16-point Gauss-Legendre
quadrature is exact to round-off. The network predicts 20 times faster than
direct simulation, including input checks.

Real-world check against 17 races of the 2026 season (OpenF1 timing data,
1,222 on-track passes after removing the start, pit stops and safety cars):
real top speeds respect the 355 km/h electrical-power limit, real acceleration
matches the car model within about 10 km/h, and the circuit ranking agrees
(rank correlation 0.65); but passes from 0.5-1 s behind are easier in the
simulator than in real races, because real drivers ration battery energy over
the lap (Section 5.3 of the documentation).

![What-if curves](reports/figures/what_if.png)

## Quick start

Requires Python 3.10 or later (tested with 3.13).

```bash
git clone https://github.com/SamGazz10/Gazzola3001.git
cd Gazzola3001
pip install -r requirements.txt

# predict with the trained model in models/
python -m overtake.predict --condition dry --tyre soft --segment straight --battery 100 --gap 1.0
python -m overtake.predict --help          # every option
```

Or set every input with sliders and drop-downs in your browser:

```bash
python -m overtake.app                     # opens http://127.0.0.1:8000; Ctrl+C to stop
```

From Python:

```python
from overtake.predict import OvertakePredictor

predictor = OvertakePredictor.load()
predictor.predict({"track_condition": "dry", "attacker_tyre": "soft",
                   "segment": "straight", "attacker_battery": 100, "gap_s": 1.0})
```

Fields left out (or set to `None`) take the defaults in `overtake/features.py`;
the run length (800 m on a straight, 150 m into a corner) and track temperature
(35 °C dry, 20 °C damp or wet) depend on the segment and weather. A pandas
Series counts as one scenario. Invalid inputs, NaN and unknown field names
raise a `ValueError` naming the field; inputs outside the training range give
a warning.

## Reproducing the results

Everything is generated from fixed random seeds; no data files are needed,
except for the real-data comparison, which downloads 2026 race data from
OpenF1 once (about 4 minutes, internet needed) and caches it in `data/real/`.
The spreadsheet export and the download cache are regenerated rather than
stored in git because of their size.

Times are rough and machine-dependent.

| Step | Command | Time | Output |
|---|---|---|---|
| Train the network and baselines | `python -m overtake.train` | ~4-10 min | `models/overtake_net.pt`, `reports/results/test_metrics.json` |
| All experiments | `python -m overtake.experiments all` | ~30-45 min | `reports/results/*.json`, `reports/figures/*.png` |
| One experiment | `python -m overtake.experiments solver` (or `timestep`, `stability`, `adaptive`, `quadrature`, `speed_trace`, `constants`, `rules`, `selection`, `ablation`, `baselines`, `comparison`, `performance`, `sensitivity`, `real`, `diagram`) | seconds to 10 min | |
| Redraw every figure from the saved results | `python -m overtake.experiments figures` | seconds | `reports/figures/*.png` |
| Comparison with real 2026 races | `python -m overtake.experiments real` | ~5 min first time (download), then ~1 min | `reports/results/real_data.json`, `reports/figures/real_*.png` |
| Tests (89, all offline) | `python -m pytest` | ~2 min | |
| Test report | `python -m pytest --html=reports/test_report.html --self-contained-html` | ~2 min | `reports/test_report.html` |
| HTML API docs | `pdoc overtake --docformat numpy -o docs/api` | seconds | `docs/api/` |
| Walkthrough notebook | `jupyter notebook notebooks/walkthrough.ipynb` | | runs on Google Colab too |
| The network by hand | `jupyter notebook notebooks/network_by_hand.ipynb` | seconds | one prediction recomputed from the saved weights with NumPy, checked against the library |
| Spreadsheet of all simulation data | `python -m overtake.export` | ~2.5 min | `data/overtake_simulations.xlsx` (~65 MB, not in git) |

`baselines`, `comparison`, `performance` and `sensitivity` use the trained
model, so run `python -m overtake.train` first.

To train on real labelled attempts instead of simulated ones:
`python -m overtake.train --data attempts.csv` (the CSV needs every input
column plus `overtake`, 1 if the move stuck and 0 if not).

## Repository structure

```
overtake/                 Python package
  features.py             input schema, defaults, validation, preprocessing
  physics.py              two-car equations of motion; Euler and RK4 integrators
  simulator.py            scenario sampling, outcomes, exact probability (quadrature)
  model.py                the neural network
  train.py                training, metrics, evaluation against alternatives
  baselines.py            logistic regression, gradient-boosted trees
  predict.py              command-line and Python prediction
  app.py                  interactive calculator in the browser (python -m overtake.app)
  experiments.py          verification, model selection, comparison, performance, sensitivity
  export.py               spreadsheet of every simulated attempt and every result
  realdata.py             download of real 2026 race data (OpenF1) and comparison with the simulator
  plots.py                figures
tests/                    pytest tests (physics, simulator, pipeline, experiments, export, real data, notebooks)
notebooks/
  walkthrough.ipynb       the workflow end to end
  network_by_hand.ipynb   the trained network's arithmetic, step by step
models/overtake_net.pt    trained ensemble of five networks (154 kB)
reports/
  figures/                every figure in the documentation
  results/                every number in the documentation, as JSON; files named
                          *_before_*, *_after_*, *_3_seeds or *_min_rule are earlier
                          runs the documentation refers to (Appendix D)
  test_report.html        pytest report
docs/
  project_documentation.md   the report (also as .pdf and .docx)
  api/                    HTML API documentation (pdoc)
  index.html              landing page for GitHub Pages
```

## Documentation

* **Project documentation:** [docs/project_documentation.md](docs/project_documentation.md)
* **API documentation:** generated from the NumPy-style docstrings into
  `docs/api/` (open `docs/api/index.html`). To publish it with GitHub Pages:
  repository **Settings → Pages → Build and deployment → Deploy from a branch**,
  branch `main`, folder `/docs`. The site will then be at
  `https://samgazz10.github.io/Gazzola3001/` (Pages on a private repository
  needs a GitHub plan that includes it, e.g. through the Student Developer Pack).
* **Test report:** `reports/test_report.html`.

## Licence and AI use

MIT licence (see `LICENSE`). AI tools (Claude Code) were used in this project;
see Section 8 of the project documentation for what they were used for and how
the output was checked.
