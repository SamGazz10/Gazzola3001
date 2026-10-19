# Probability of an F1 overtake under the 2026 rules

MMA3001 Numerical Methods and Machine Learning, individual project. Sam Gazzola.

Repository: <https://github.com/SamGazz10/Gazzola3001>. API documentation:
`docs/api/` (generated from the docstrings with pdoc). Every number below is
produced by code in the repository; the command that regenerates it is given
with each result.

## Summary

In 2026 Formula 1 replaced DRS with an electrical overtaking aid: a car within
one second of the car ahead gets **Overtake Mode** (extra battery energy and a
higher speed limit for electrical power), and either driver can use **Boost**
(maximum engine plus battery power) anywhere. Whether an attack succeeds now
depends on how much battery energy each car has left, not just on the gap.

This project estimates **P(overtake)**, the probability that an attacker
completes a pass, from 19 inputs describing the situation (gap, battery
charge, Boost and Overtake Mode use, tyres, weather, where on the lap, drivers,
cars, circuit). It has two parts:

1. A **physics-based simulator**. The two cars' equations of motion are
   integrated with fourth-order Runge-Kutta (RK4) from the corner exit to the
   braking point; random execution variation then decides the outcome. The
   exact probability is a one-dimensional integral, evaluated by
   Gauss-Legendre quadrature. The simulator generates training data with a
   known ground truth.
2. A **neural network** (an ensemble of five) that learns P(overtake) from
   0/1 outcomes only, as it would have to with real race data, and is
   validated against that ground truth.

On 30,000 held-out attempts the network's probabilities are within
2.0 percentage points of the truth on average (gradient-boosted trees:
3.8; logistic regression: 15.7). For the default scenario (dry, new
softs, full battery, 1 s behind on an 800 m straight, Boost and Overtake
Mode) the simulator gives 47% and the network 44%;
without Boost the chance falls below 1 %.

---

## 1. Engineering problem

**Context.** The 2026 power unit takes roughly half its peak power from the
350 kW MGU-K (the electric motor-generator), but the battery holds only a few
megajoules: at full power it empties in about ten seconds, less than one long
straight. A car that runs out of electrical energy ("clips") loses tens of
km/h at the end of a straight. Overtaking is therefore an energy-management
problem: when to deploy, when to save, and when an attack is worth the energy.
The regulations limit electrical power above 290 km/h; Overtake Mode moves
that limit to 337 km/h and adds 0.5 MJ [1-4].

**Problem.** Given the situation at the exit of the corner before a straight,
estimate the probability that the attacker completes the pass at the end of
that straight (or into the next corner).

**Why it matters.** Teams decide every lap whether to attack or harvest
energy, and drivers decide when to use Boost. A quick, calibrated probability
lets a strategist compare options ("attack now with 60 % battery, or wait a lap
and arrive with 90 %?"). Regulators tune the Overtake Mode parameters to make
overtaking possible but not trivial; the same model shows how sensitive the
outcome is to those parameters.

**Intended use.** Decision support and what-if analysis: evaluating many
situations quickly, and a training pipeline that can be retrained, or
fine-tuned, on real labelled overtaking attempts.

**Limits of the scope.**

* One attempt on one straight; no multi-lap battles or energy strategy across
  a lap.
* Training data are simulated. Battery charge and Boost use are not in public
  timing data, so the simulator stands in for real data. The network can only
  be as right as the simulator, and many simulator constants are engineering
  estimates (Section 3.2).
* Driver ability is a single rating; team orders, safety cars, blue flags and
  damage are not modelled.

## 2. Inputs and outputs

### Inputs

One scenario is one row (a Python dict, a pandas DataFrame row, or a CSV row).

| Input | Type and units | Domain accepted | Training range |
|---|---|---|---|
| `track_condition` | category | `dry`, `damp`, `wet` | all |
| `segment` | category: where the move completes | `straight`, `slow_corner`, `medium_corner`, `fast_corner` | all |
| `circuit` | category | 24 circuits or `generic` | all |
| `attacker_tyre`, `defender_tyre` | category | `soft`, `medium`, `hard`, `intermediate`, `wet` | all |
| `gap_s` | gap at the corner exit, s | 0-3 | 0.05-2.0 |
| `run_length_m` | corner exit to braking point, m | 10-2000 | 250-1300 (straights), 50-300 (corners) |
| `attacker_battery`, `defender_battery` | state of charge, % | 0-100 | 0-100 (15 % at exactly 100) |
| `attacker_tyre_age`, `defender_tyre_age` | laps | 0-80 | 0-40 |
| `track_temp_c` | deg C | 0-70 | 18-55 dry, 10-32 wet |
| `attacker_skill`, `defender_skill` | rating 0-100 | 0-100 | 60-100 |
| `top_speed_delta_kph` | attacker car's straight-line advantage, km/h | -30 to 30 | -15 to 15 |
| `pace_delta_s` | attacker car's lap-time advantage, s/lap | -3 to 3 | -1.5 to 1.5 |
| `attacker_boost`, `defender_boost` | bool | true/false, 1/0 | both |
| `overtake_mode` | bool: within 1 s at the detection point | true/false, 1/0 | both |

### Output

`P(overtake)`, a probability between 0 and 1: the chance that this attempt
succeeds. It is calibrated, so among many situations predicted at 30 %, about
30 % should succeed (verified in Section 5.2). The command-line tool also
reports the probability without Boost and at other gaps.

### Invalid, missing and unsupported inputs

| Case | Handling |
|---|---|
| Unknown category, number outside the domain, non-numeric number, non-boolean flag, missing column | `ValueError` naming the column and the allowed values (`features.validate`); the command line prints it and exits with status 2 |
| Field omitted or `None` when predicting | Default value from `features.DEFAULTS`; `overtake_mode` defaults to `gap_s <= 1` |
| Inside the domain but outside the training range | Accepted; `OvertakePredictor.out_of_range` returns a warning and the command line prints it, because the prediction is extrapolation (Section 5.2 measures the cost) |
| Missing trained model | `FileNotFoundError` telling the user to run `python -m overtake.train` |
| Training CSV without a 0/1 `overtake` column | `ValueError` |

Inputs and outputs of each function and class are documented in their
docstrings and the generated API documentation.

## 3. Computational solution

### 3.1 Overview

```
scenario ──► initial conditions ──► two-car ODE, RK4 ──► deficit D at braking point
                                                              │
                     execution noise ε ~ N(0, σ), completion logit z0
                                                              │
               ┌──────────────────────────────────────────────┴───────────┐
       draw ε, u ──► 0/1 outcome                    Gauss-Legendre ──► exact P
       (training labels)                            (ground truth, validation only)
               │
       neural network learns P from outcomes ──► validated against exact P
```

### 3.2 Physical model of the closing phase (`overtake/physics.py`)

Each car has position x, speed v and battery energy used e:

    dx/dt = v
    m dv/dt = min(F_power, F_traction) - F_drag - F_rolling
    de/dt = P_ers

* `F_power = η (P_ICE + P_ERS) / v`. `P_ERS` is the MGU-K power: 350 kW,
  reduced linearly from 290 km/h to zero at 355 km/h (from 337 km/h with
  Overtake Mode), multiplied by 1 with Boost or 0.55 on the standard energy
  map, and ramped to zero as the available energy runs out.
* `F_traction = μ w_r (m g + ½ ρ C_L A v²)`: the most the driven rear tyres can
  transmit. Below about 180 km/h this limits acceleration, which is how tyre
  compound, age, temperature and wetness enter.
* `F_drag = ½ ρ C_D A v²`. The follower's `C_D A` is cut by up to 25 % in the
  slipstream, decaying with distance (e-folding length 20 m) and fading out
  over one car length as it draws alongside.
* `F_rolling = c_rr (m g + ½ ρ C_L A v²)`.

The cars are coupled through the slipstream, so all six states are integrated
together. Integration stops when the defender reaches the braking point; the
result is the attacker's remaining deficit D (seconds).

| Parameter | Value | Source |
|---|---|---|
| MGU-K power, taper | 350 kW; 290→355 km/h; Overtake Mode 337 km/h | 2026 regulations as reported [1, 2, 4] (some reports give 340/345 km/h) |
| Overtake Mode energy | +0.5 MJ | [1] |
| ICE power | 400 kW | reported target [5] |
| Usable battery energy | 4 MJ at 100 % | estimate; storage capacity reported unchanged from previous rules [6] |
| Mass | 800 kg | estimate: 2026 minimum ~768 kg plus fuel |
| C_D A, C_L A (straight mode) | 1.0 m², 3.0 m² | estimates, checked against top speeds below |
| Tyre μ (new soft, dry), rolling coefficient, efficiency | 1.7, 0.012, 0.95 | estimates |
| Slipstream | 25 % at 0 m, 20 m decay | estimate |

*Plausibility checks* (`python -m overtake.experiments` and `simulator.speed_trace`):
with these values a car reaches about 348 km/h at the end of an 800 m
straight with Overtake Mode and about 330 km/h without it (the power taper
above 290 km/h), about 303 km/h with no electrical power at all, and a car
whose battery empties mid-straight levels off near 300 km/h, consistent
with reported 2026 top speeds and "clipping" behaviour. These checks are
coarse; they do not validate individual parameters.

![Speed traces](../reports/figures/speed_trace.png)

### 3.3 Numerical methods

**RK4 time integration.** The system is a small non-stiff initial value
problem: the fastest physical time scale is the drag response,
`m / (ρ C_D A v) ≈ 8 s` at 80 m/s, so an explicit method is stable for any
step small enough to be accurate. RK4 was chosen over forward Euler because it is fourth order for
four function evaluations per step (Section 4 compares them). The step is
fixed at **dt = 0.05 s** from the time-step study (Section 6.1).

**Locating the braking point.** The defender crosses the braking point
partway through a step. Linear interpolation would be second-order accurate
and spoil RK4's accuracy, so the crossing time is found by Newton iteration on
the cubic Hermite interpolant through the step's end states and derivatives
(fourth-order accurate), and every state is interpolated to that instant.

**Vectorisation.** All scenarios in a chunk of 20,000 are integrated together
as NumPy arrays; a scenario leaves the active set when its defender crosses
the braking point.

### 3.4 Outcome model and exact probability (`overtake/simulator.py`)

The deficit D is increased slightly for caution in the wet and for the
defender's positioning. Driver execution varies from attempt to attempt:
ε ~ N(0, σ), σ = 0.20 s dry (0.24 damp, 0.28 wet, more for lower-rated
attackers). The attacker can lunge if `D - ε` is within 0.15 s; the margin is
`u = ε - (D - 0.15)`, and the lunge sticks with probability
`logistic(z0 + 5 min(u, 0.5))`, where `z0` combines driver ratings, braking
grip, corner type, weather and circuit. These constants are estimates chosen
so that representative scenarios give plausible results (e.g. near 0 % without
Boost against a Boosting defender; about 75 % from 0.5 s behind).

Because the only randomness is ε (plus the final success draw), the exact
probability is

    P = ∫₀^∞ φ_σ(u + D − 0.15) · logistic(z0 + 5 min(u, 0.5)) du.

The integrand has a kink at u = 0.5, so the integral is split there: the tail
`[0.5, ∞)` has a closed form, and `[0, 0.5]` (smooth) uses 16-point
Gauss-Legendre quadrature.

### 3.5 Neural network (`overtake/model.py`, `overtake/train.py`)

* **Inputs.** Each categorical input passes through a learned embedding
  (condition 2, segment 3, circuit 6, tyre 4 dimensions; both tyre columns
  share one tyre embedding). Numeric inputs are standardised with training-set
  statistics; flags are 0/1. Width of the first layer: 33. (Two
  physics-informed inputs were tested and not adopted; Section 5.2.)
* **Architecture.** Two hidden layers of 64 units, SiLU activation, dropout
  0.1, one output logit; sigmoid gives P. 6,589 parameters per network. Chosen from
  four sizes on the validation set by the one-standard-error rule (Section 6.3).
* **Ensemble.** Five such networks are trained identically from seeds 0-4;
  the prediction is the mean of their probabilities. This cut the mean error
  by about 10 % (Section 5.2) at five times the training and
  prediction cost (Section 6.4).
* **Training.** Binary cross-entropy on 0/1 outcomes; AdamW (learning rate
  2×10⁻³, weight decay 10⁻⁴), batch 1024, learning rate halved after 2
  epochs without improvement; early stopping after 5 epochs without
  improvement, restoring the best weights; at most 100 epochs. Seeds are fixed
  so runs repeat.
* **Data.** 300,000 simulated attempts (overtake rate 28%), split
  80 / 10 / 10 into training, validation and test sets.

### 3.6 Implementation and reproducibility

| Module | Role |
|---|---|
| `features.py` | input schema, defaults, validation, preprocessing |
| `physics.py` | equations of motion, Euler and RK4, Hermite crossing |
| `simulator.py` | scenario sampling, outcomes, quadrature, Monte Carlo |
| `model.py`, `train.py` | network, training, metrics |
| `baselines.py` | logistic regression, gradient boosting |
| `predict.py` | command line and Python API |
| `experiments.py`, `plots.py` | every study and figure in this document |
| `export.py` | spreadsheet of all simulation data and results |

`python -m overtake.train` (about 4 minutes on a 4-core CPU) and
`python -m overtake.experiments all` (about 20 minutes) regenerate all
results; `python -m overtake.export` writes every simulated attempt (inputs,
physics results, outcome, true and predicted probabilities), the metric and
input lists and every experiment's results to one spreadsheet; `python -m pytest` runs 51 tests.

## 4. Alternative solutions

**Numerical methods.**

| Choice | Alternative | Evidence | Decision |
|---|---|---|---|
| RK4 | Forward Euler | At dt = 0.05 s Euler's worst error is 1.3e-02 s against 5.9e-05 s for RK4; Euler needs far smaller steps for the same accuracy and costs more overall (Section 6.1) | RK4 |
| Gauss-Legendre (16 nodes) | Composite Simpson, trapezoid, Monte Carlo | 12 nodes already reach 2×10⁻¹⁵; Simpson needs 257 points for 3×10⁻¹¹; Monte Carlo with 1,600 replays has 0.7-point RMS error and is ~200× slower | Gauss-Legendre |

**Models.** All use the same inputs and training data. Results on the test set
(`python -m overtake.experiments comparison`):

| | Neural network (ensemble of 5) | Gradient boosting | Logistic regression | Direct simulation |
|---|---|---|---|---|
| Mean error vs truth (points) | 2.0 | 3.8 | 15.7 | 0 (it is the truth) |
| Log loss | 0.257 | 0.267 | 0.411 | - |
| ROC AUC | 0.948 | 0.944 | 0.853 | - |
| Calibration error | 0.005 | 0.014 | 0.004 | - |
| Training time | 208 s | 4 s | 3 s | none |
| Prediction time (µs per scenario, batch of 10k, incl. input checks) | 4.1 | 8.6 | 3.7 | 95 |
| Model size | 132 kB | 1,139 kB | 4 kB | - |
| Interpretability | low (sensitivity analysis needed) | medium (feature importances) | high (coefficients) | high (physics) |
| Learns from real outcomes | yes | yes | yes | no |

* **Logistic regression** is fast and interpretable but cannot represent
  interactions; for example, battery charge matters far more on a long
  straight than into a corner. Its error is 15.7 points.
* **Gradient-boosted trees** are the strongest standard method for tabular
  data and train 48× faster than the network, but are
  1.9× less accurate here, less well calibrated, and larger. Their
  piecewise-constant predictions approximate the smooth physics less well.
* **Direct simulation** is exact by construction and 23× slower per
  prediction, but it needs the physical model to be right; it cannot learn
  from observed race outcomes. The network is the method that transfers to
  real data.

The neural network was selected because it is the most accurate of the
learnable methods (half the error of gradient boosting) and as well calibrated
as the best of them, at an acceptable cost.

## 5. Validation

### 5.1 Verification of the numerical methods

| What | How | Acceptance | Result | Command |
|---|---|---|---|---|
| RHS + integrators | Special cases with analytical solutions: constant power without drag, `v² = v0² + 2Pt/m`; drag without power, `v = v0/(1 + k v0 t/m)` | relative error < 10⁻⁸ at dt = 0.05 s | passes | `pytest tests/test_physics.py` |
| Order of accuracy | Slope of log error against log dt | Euler 1, RK4 4 (±0.15) | Euler 0.95 and 1.00; RK4 4.05 and 4.00 | `experiments solver` |
| Full model step size | Deficit against dt = 0.001 s reference, 2,000 random scenarios | max error < 1 ms | 5.9e-05 s at dt = 0.05 s | `experiments timestep` |
| Physical constraints and expected behaviour | Energy used ≤ energy available; speeds between 20 m/s and the cut-off speed; P(overtake) moves in the expected direction for 10 inputs; slicks in the wet nearly hopeless | always | passes | `pytest tests/` |
| Quadrature | Against SciPy adaptive `quad` | < 10⁻¹⁰ | passes | `pytest tests/test_simulator.py` |
| Outcome sampling | Monte Carlo (20,000 replays) against exact P, 50 scenarios | within 4.5 standard errors | passes | same |

![Solver convergence](../reports/figures/solver_convergence.png)

![Quadrature and Monte Carlo](../reports/figures/quadrature_study.png)

### 5.2 Validation of the network

**Design.** The network never sees the true probability: it is trained on 0/1
outcomes, as it would be with race data. The validation set (30,000 attempts)
was used for early stopping and to choose the architecture; the test set
(30,000) was used once, for the numbers here. Because the simulator provides
the exact probability of every test scenario, the main metric is the **mean
absolute error against the true probability** (in percentage points). Log
loss, Brier score, ROC AUC and calibration error, which need only outcomes and
would also be available with real data, are reported alongside.

**Results.** Mean error 2.0 points; median 0.7, 90th percentile
5.6, 99th percentile 15.6, worst 58.7. Calibration error
0.005: predicted probabilities match observed frequencies.

![Calibration](../reports/figures/calibration.png)

**Regions.** Error is checked by operating region, because a single average
can hide a poor region:

![Error by region](../reports/figures/method_comparison.png)

**Failure modes.**

* *Rare combinations.* Error is higher where either car's tyres do not suit
  the surface (slicks in the wet, wet-weather tyres in the dry): 2.6 points
  against 1.9 elsewhere. Such cases are 18 % of the test set but 50 % of the
  worst 1 % of errors and 70 % of the worst 0.1 %. They are rare in the
  training data, as in racing.
* *Bias near a full battery, and run-to-run variation.* Sensitivity analysis
  showed the first single network 6 points low at the default scenario
  (41 % against 47 %), the error growing as the attacker's battery rose above
  80 %. Three explanations were tested in turn:
  1. *Too little training data at a full battery*, which was the edge of a
     uniform 0-100 % distribution. Making 15 % of training attempts start full
     (realistic, so kept) did not reduce the error: rejected.
  2. *The network smooths a sharp corner.* Along the battery axis the true
     probability rises steeply and then stops abruptly at about 60 %, where
     the battery first covers the straight. Two physics-informed inputs
     (`features.DERIVED`: each car's battery energy divided by the energy
     needed for full power along the straight, capped at 1) were added to hand
     the network that corner.
  3. *Run-to-run variation.* Retraining with a different random seed moves a
     single network's prediction at any one scenario by several points.

  Explanations 2 and 3 were tested together, five seeds each
  (`experiments ablation`, validation set):

  | | Mean error vs truth (points) | Log loss | Error at default scenario (points) | Worst error along battery axis (points) |
  |---|---|---|---|---|
  | Raw inputs, single network (mean ± sd) | 2.24 ± 0.06 | 0.2595 ± 0.0004 | -3.5 ± 2.5 | 7.8 ± 2.0 |
  | Raw inputs, ensemble of 5 | 2.02 | 0.2585 | -3.5 | 7.4 |
  | Energy-cover inputs, single network (mean ± sd) | 2.18 ± 0.04 | 0.2589 ± 0.0004 | +0.3 ± 2.5 | 12.6 ± 4.3 |
  | Energy-cover inputs, ensemble of 5 | 1.97 | 0.2580 | +0.3 | 12.6 |

  The seed alone moves the default-scenario error by about ±2.5
  points, so the first network's -6 points was partly bad luck: one training
  run is not enough to judge a change. Ensembling cut the mean error by about
  10 % with either inputs by averaging out run-to-run variation, but it
  cannot remove a bias that all runs share. The energy-cover inputs removed the
  bias at the default scenario but made the worst error along the battery axis
  larger: their energy estimate ignores the power taper above 290 km/h, so it
  puts the corner at about 75 % battery where the true corner is near 60 %.
  Their log-loss gain (0.0005) is well within one standard error of the
  validation log loss (about 0.003), so by the same rule as Section 6.3 the
  simpler model was kept: **raw inputs, ensemble of five networks**. Its
  remaining error at the default scenario is -3.5 points.

  The same bias is larger on a wet track. With intermediates on both cars and
  the other inputs at their defaults, the ensemble is 5 points low at a 40 %
  attacker battery and 10.5 points low at 100 % (20 % against 30 %;
  `experiments sensitivity`). Across all 1,569 wet straights on treaded tyres
  in the test set the error averages 2.8 points with no overall bias, so the
  problem is local to high battery charge; wet attempts are about an eighth
  of the training data, which leaves this region thinly covered.
* *Extrapolation.* On straights of 1300-2000 m, longer than any in training,
  mean error rises to 4.4 points (gradient boosting 9.7,
  logistic regression 26.9). The prediction tool warns when an input
  is outside the training range.

**Sensitivities.** The network's response to each input agrees with the
simulator's in direction and closely in size (Section 6.5), so it has
learned the mechanisms, not just the average.

### 5.3 What the validation cannot establish

The validation shows that the numerical methods are implemented correctly and
that the network reproduces the simulator. It does **not** show that the
simulator matches real 2026 racing. Parameters marked as estimates, the
execution-noise model and the circuit factors have not been fitted to real
overtakes, and the plausibility checks in Section 3.2 are coarse. Testing that
would need labelled real attempts with battery and Boost data, which are not
public.

## 6. Optimisation and performance

### 6.1 Time step

![Time-step study](../reports/figures/timestep_study.png)

The first time-step study showed the worst-case error of RK4 shrinking only in
proportion to dt (first order), 1.1 ms at dt = 0.05 s, above the 1 ms limit.
The cause was the slipstream switching off instantly when the attacker drew
alongside, a jump in drag that RK4 cannot integrate at fourth order. Making the
slipstream fade over one car length (also more realistic) cut the worst error
at dt = 0.05 s by more than an order of magnitude: to 0.043 ms on the same
2,000 scenarios, and 5.9e-05 s (0.059 ms) on the final scenario sample (results
before the fix: `reports/results/timestep_before_slipstream_fix.json`). The remaining kinks
(traction limit, power taper, battery running out) keep the observed order
around 2.7 rather than 4. dt = 0.05 s was chosen: its error is
3,402 times smaller than the 0.2 s execution noise, and halving it
doubles the cost for no meaningful gain.

### 6.2 Where the time goes

Profiling (`cProfile`, 20,000 attempts) shows the force evaluation
(`acceleration`, 26 % of run time), the RK4 bookkeeping and the power
taper dominate; the network and data handling are negligible.

| Change | Before | After | Gain |
|---|---|---|---|
| Integrate scenarios together instead of one at a time (200 scenarios) | 10.8 s | 0.23 s | 47× |
| Chunk of 20,000 instead of 1,000 (40,000 attempts) | 14.0 s | 3.1 s | 4.5×; 40,000 gains only 3 % more and needs twice the memory |
| Gauss-Legendre instead of Monte Carlo (1,600 replays) for the exact P | 90 µs | 0.48 µs per scenario | 189×, and exact |

### 6.3 Network size and data requirements

![Model selection](../reports/figures/model_selection.png)

| Hidden layers | Parameters | FLOPs per prediction | Validation log loss ± standard error | Validation error (points) |
|---|---|---|---|---|
| 32 | 1,309 | 2,369 | 0.2671 ± 0.0029 | 3.4 |
| 64-64 (selected) | 6,589 | 13,313 | 0.2594 ± 0.0029 | 2.2 |
| 128-128-64 | 29,373 | 59,649 | 0.2591 ± 0.0029 | 2.3 |
| 256-256-128 | 107,709 | 217,601 | 0.2606 ± 0.0030 | 2.6 |

Selection used validation log loss, the metric that would also be
available with real data; it ranks the networks the same way as the error
against truth. The lowest validation log loss was 128-128-64 (0.2591); 64-64 (0.2594) is within one standard error (0.0029) and needs 4.5× fewer FLOPs, so the one-standard-error rule [14] selects it. This rule was adopted after two runs on different data ranked 64-64 and 128-128-64 in opposite orders by less than 0.001: picking the raw minimum would have chosen on noise (first-run results: `reports/results/selection_before_battery_sampling.json`, `selection_min_rule.json`). Training-set size matters more than network size: the error
falls from 6.8 points with 10,000 attempts to 2.2 with 240,000 and
is still falling. One F1 season produces perhaps a thousand overtaking
attempts (a rough estimate), so a network trained only on real data would be data-starved; the
practical route is to pre-train on simulation and fine-tune on real attempts.

### 6.4 FLOPs, arithmetic intensity and memory

| | Ensemble (5 × 64-64) | Direct simulation |
|---|---|---|
| FLOPs per prediction | 66,565 (counted from layer sizes) | ~68,602 (153 RK4 steps × ~444 FLOPs, hand-counted) |
| Achieved rate | 81 GFLOP/s (batch matrix products) | 0.7 GFLOP/s |
| Time per prediction | 0.82 µs (forward passes only) | 95 µs |

The five-network ensemble needs about as many operations as the simulation
but runs them 118× faster. The difference is arithmetic intensity.
The network's work is dense matrix products: each weight read from memory is
used for every prediction in the batch, so intensity grows from 0.5
FLOP/byte for a single prediction (memory-bound) to 251 at a batch of
1024 (compute-bound, using the CPU's vector units). The simulation performs
many small element-wise NumPy operations, each reading and writing whole
arrays for a few FLOPs, so its intensity is well below 1 FLOP/byte and it is
memory- and overhead-bound. Memory: ensemble 132 kB; the 300,000
attempt dataset 115 MB; ODE state 0.96 MB per chunk.

### 6.5 Sensitivity

![Sensitivity](../reports/figures/sensitivity.png)

One input at a time between plausible extremes around the default scenario
(`experiments sensitivity`). The gap, straight length and the two batteries
matter most, followed by Boost, tyre age and the circuit; driver ratings and
track temperature matter least. The network (bars) and the simulator (ticks)
agree in direction for every input; across the 30 settings the median
difference is 2 points, 87 % are within 5 points, and the largest is 6.5
points (part of it is the 3.5-point bias at the default scenario).
One-at-a-time analysis misses interactions; the what-if curves below
show two important ones: the drop when Overtake Mode is lost above 1 s, and
how Boost's value grows with straight length.

![What-if curves](../reports/figures/what_if.png)

## 7. Limitations and lessons learned

**Limitations.** The model's realism is limited by the simulator: estimated
parameters, a single-straight scope, a simple driver model, and no real
outcome data. The network is accurate on average but less so for rare
combinations, outside its training range, and near a full attacker battery,
where it under-predicts by about 3.5 points in the dry and up to about 10
points in the wet.

**Lessons.**

1. *Convergence studies find modelling errors, not just numerical ones.* The
   first-order convergence of RK4 exposed a discontinuity in the slipstream
   model that a quick visual check would never have shown.
2. *Sensitivity analysis is a validation tool, and hypotheses must be
   tested.* The comparison with the simulator's sensitivities exposed an
   error at the most common operating point. The obvious explanation (too
   little data at a full battery) was wrong. A physics-informed input looked
   like a fix in one training run, but repeated runs showed it moved the error
   rather than removing it, because its physics was too approximate.
3. *One training run is not evidence.* The random seed alone moves a single
   network's prediction by several points. Comparisons were therefore repeated
   over five seeds, and the final model averages five networks.
4. *A known ground truth makes model comparison decisive.* Outcome-only
   metrics (log loss) ranked the models the same way as error against truth,
   which gives some confidence they would also guide selection on real data.
5. *Data, not network size, limits accuracy.* Doubling the network did not
   help; more data did.

**Improvements.** Fit the uncertain constants to real races (speed-trap data
and overtake counts are public); extend to a full lap with energy strategy;
use event location so RK4 keeps fourth order through the kinks; fine-tune the
network on real labelled attempts.

## 8. AI use

### Disclosure (factual record)

Claude Code (Anthropic) was used throughout, in a cloud development session.

| Task | What the AI did | How it was checked |
|---|---|---|
| Rules research | Web searches on the 2026 power-unit, Boost and Overtake Mode rules; flagged conflicting reports (taper end points) | Sources listed in References |
| First version | Wrote a first simulator with hand-tuned formulas and the network | Replaced after reading the project brief: it had no numerical method that could be verified |
| Physics model and numerics | Wrote the two-car ODE model, RK4/Euler, Hermite crossing, quadrature | Analytical-solution tests, convergence studies, Monte Carlo agreement |
| Calibration | Proposed the estimated constants and tuned them for plausibility | Speed traces and representative scenarios checked; constants marked as estimates |
| Network, baselines, experiments, figures, tests, docs | Wrote the code and this documentation draft | 51 automated tests; all numbers regenerated by the code |

Errors made by the AI and caught during the work: a first simulator too crude
for the unit's requirements; a slipstream discontinuity (found by the
time-step study); two wrong diagnoses of the network's error at a full
battery (a data gap, then a physics-informed input that one training run made
look like a fix and five runs did not support); tests with wrong expectations
(wet-track test using slick tyres; Euler order measured at too large a step);
a crash on text input in the prediction tool.

### Reflection

*To be written by Sam: the brief asks for your own reflection, so this section
is deliberately left for you.* The brief asks you to cover: which tools you
used and for what; the approximate level of AI contribution; why you used AI
for those tasks; how you checked the AI-generated material; which important
decisions remained yours; errors, limitations or unhelpful suggestions; and
how AI affected your understanding and workflow.

## References

1. Formula 1, "Explained: the key terms for Formula 1's new-for-2026 rules".
   <https://www.formula1.com/en/latest/article/explained-the-new-key-terms-for-formula-1s-new-for-2026-rules.3T5BU6TC9quGcIpGzoWkY0>
2. Formula 1, "Explained: 2026 power unit regulations".
   <https://www.formula1.com/en/latest/article/explained-2026-power-unit-regulations-fia.68izKQ2tn1voQPWvgLVMXN>
3. ESPN, "Formula 1's new terminology explained: Overtake Mode, Boost, and how it affects 2026 racing".
   <https://www.espn.com/f1/story/_/id/47333501/>
4. Crash.net, "'Manual Override Mode': F1's new overtaking aid - and DRS replacement - explained".
   <https://www.crash.net/f1/news/1050188/1/manuel-override-mode-f1s-new-overtaking-aid-and-drs-replacement-explained>
5. PMW Magazine, "FIA signs off 2026 F1 PU rules with MGU-H dropped and tight constraints on ICE design".
   <https://pmw-magazine.com/news/race-series-news/fia-signs-off-2026-f1-pu-rules-with-mgu-h-dropped-and-tight-constraints-on-ice-design.html>
6. Honda, "2026 Formula 1 Regulations Overview". <https://global.honda/en/F1/features/2026_Commentary/regulations/>
7. FIA, 2026 Formula 1 Technical Regulations, Section C (Power Unit), Issue 4, 2023.
   <https://FIA.com/sites/default/files/fia_2026_formula_1_technical_regulations_pu_-_issue_4_-_2023-10-25.pdf>
8. The Race, "Boost, overtake mode, active aero: key 2026 F1 terms explained".
   <https://www.the-race.com/formula-1/boost-overtake-mode-active-aero-recharge-key-2026-terms-explained>
9. S. C. Chapra and R. P. Canale, *Numerical Methods for Engineers*, McGraw-Hill (Runge-Kutta methods, Gauss quadrature).
10. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization", ICLR 2019 (AdamW).
11. C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks", ICML 2017 (calibration error).
12. F. Pedregosa et al., "Scikit-learn: machine learning in Python", JMLR 12, 2011.
13. A. Paszke et al., "PyTorch: an imperative style, high-performance deep learning library", NeurIPS 2019.
14. T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009, Section 7.10 (one-standard-error rule).
