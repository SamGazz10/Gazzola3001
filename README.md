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
performance, limitations, AI use).

## Results at a glance

| Test set (30,000 held-out attempts) | Neural network (ensemble of 5) | Gradient boosting | Logistic regression |
|---|---:|---:|---:|
| Mean error vs true probability (points) | **2.0** | 3.8 | 15.7 |
| Log loss | **0.257** | 0.267 | 0.411 |
| ROC AUC | **0.948** | 0.944 | 0.853 |
| Calibration error | 0.005 | 0.014 | **0.004** |

Numerical verification: RK4 converges at order 4.0 and Euler at 1.0 against
analytical solutions; the full model's braking-point error at the chosen step
(0.05 s) is 0.059 ms; 16-point Gauss-Legendre quadrature is exact to
round-off.

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

From Python:

```python
from overtake.predict import OvertakePredictor

predictor = OvertakePredictor.load()
predictor.predict({"track_condition": "dry", "attacker_tyre": "soft",
                   "segment": "straight", "attacker_battery": 100, "gap_s": 1.0})
```

Fields left out take the defaults in `overtake/features.py`. Invalid inputs
raise a `ValueError` naming the field; inputs outside the training range give
a warning.

## Reproducing the results

Everything is generated from fixed random seeds; no data files are needed. The
spreadsheet export is regenerated rather than stored in git because of its size.

| Step | Command | Time (4-core CPU) | Output |
|---|---|---|---|
| Train the network and baselines | `python -m overtake.train` | ~4 min | `models/overtake_net.pt`, `reports/results/test_metrics.json` |
| All experiments | `python -m overtake.experiments all` | ~20 min | `reports/results/*.json`, `reports/figures/*.png` |
| One experiment | `python -m overtake.experiments solver` (or `timestep`, `quadrature`, `selection`, `ablation`, `comparison`, `performance`, `sensitivity`) | seconds to 10 min | |
| Tests | `python -m pytest` | ~15 s | |
| Test report | `python -m pytest --html=reports/test_report.html --self-contained-html` | ~15 s | `reports/test_report.html` |
| HTML API docs | `pdoc overtake --docformat numpy -o docs/api` | seconds | `docs/api/` |
| Walkthrough notebook | `jupyter notebook notebooks/walkthrough.ipynb` | | runs on Google Colab too |
| Spreadsheet of all simulation data | `python -m overtake.export` | ~2.5 min | `data/overtake_simulations.xlsx` (~65 MB, not in git) |

`comparison`, `performance` and `sensitivity` use the trained model, so run
`python -m overtake.train` first.

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
  experiments.py          verification, model selection, comparison, performance, sensitivity
  export.py               spreadsheet of every simulated attempt and every result
  plots.py                figures
tests/                    pytest tests (physics, simulator, pipeline)
notebooks/walkthrough.ipynb
models/overtake_net.pt    trained ensemble of five networks (154 kB)
reports/
  figures/                every figure in the documentation
  results/                every number in the documentation, as JSON; files named
                          *_before_* or after a rule are earlier runs the
                          documentation refers to
  test_report.html        pytest report
docs/
  project_documentation.md
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
