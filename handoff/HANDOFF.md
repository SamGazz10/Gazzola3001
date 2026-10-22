# Handoff: MMA3001 F1 overtake project (cloud session → local session)

Written 2026-10-22 by the Claude Code cloud session that built this project with Sam Gazzola, so
that a locally hosted Claude Code session can take over without the original conversation.
Read this whole file first, then `handoff/brief.md` (the assessment brief and rubric, verbatim
in substance) and `handoff/audit_findings.md` (an unfinished audit, see Section 7).

**Repository:** <https://github.com/SamGazz10/Gazzola3001> (clone URL is case-insensitive;
`origin` was `https://github.com/SamGazz10/gazzola3001`).
**Work in progress lives on branch `report-polish`** (pushed). `main` is stable and complete
except for the polishing described here. The `handoff/` folder exists only on `report-polish`;
**delete it before merging `report-polish` into `main`** (it is not part of the assessed repo).

---

## 1. What Sam asked for (most recent request first)

1. *"Make me a full document outlining exactly what we are doing/have done in this chat so I can
   give it to a locally hosted session for it to take over."* → this file.
2. The request currently being executed (interrupted by 1):
   > "Go over everything made and check that all parts of the brief are included and done to an
   > exceptional standard. Whatever needs to be changed, let me know and then proceed with fixing
   > it, polish off the 5-10 page project documentation, go through all the code and make sure all
   > code includes inline comments and docstrings for functions, then finally create the
   > presentation for me too with a script and PowerPoint as well as a separate document that
   > explains everything in the project documentation in detail."
   Sam asked for this with "ultracode" on (exhaustive, multi-agent, adversarially checked work;
   token cost not a constraint).
3. Earlier requests (all done): build the project to the brief; merge to main; spreadsheet of all
   simulation data; explain the simulator; compare with real data (Kaggle/OpenF1); explain each
   real-data step "for the interview"; a "network by hand" notebook; merge to main.

## 2. Ground rules that applied in this project (keep them)

* **Sam must write the AI-use Reflection himself** (the brief requires his own reflection, and AI
  is not allowed in the interview). Never write it for him. The report keeps a clearly marked
  placeholder (Section 8, "Reflection") and leaves about 0.45 of page 10 free for it. You may give
  him prompts/notes in the study guide, but not the text.
* **Every number in the documentation must come from code**: `reports/results/*.json` (written by
  `python -m overtake.experiments <name>` / `python -m overtake.train`) or code constants. Never
  hand-edit a number without a source; if you change code that affects results, rerun the
  experiment and update the text.
* **No AI during the presentation/interview**, so everything produced (report, slides, script,
  study guide) must help Sam *understand*, not just read aloud.
* Git: commits authored as `Sam Gazzola <samcgazzola@gmail.com>` (use `--author=...`, and for merges
  `GIT_AUTHOR_NAME=... GIT_AUTHOR_EMAIL=...`). Work on feature branches, merge into `main` with
  `git merge --no-ff`, push both. The rubric rewards "evidence of branching and insightful
  commits", so write descriptive commit messages (what and why). The cloud session also added
  co-author/session trailer lines required by its own environment; follow your local session's
  own attribution rules instead. Do not put AI model identifiers in commits, code or docs.
* Keep code, docstrings, HTML docs (`docs/api/`, pdoc) and the test report
  (`reports/test_report.html`) in sync: regenerate both after code changes (commands in Section 5).
* Writing style used throughout: plain, precise, no marketing language, numbers with units,
  British spelling (tyre, optimisation), NumPy-style docstrings.

## 3. The project in one page

**Engineering problem.** From 2026, F1 replaced DRS with *Overtake Mode* (a car within 1 s of the car
ahead at a detection point gets +0.5 MJ battery energy and full electrical power to 337 km/h instead
of the normal taper 290→355 km/h) and *Boost* (max engine + battery power, usable by either driver).
The project estimates **P(overtake)**, the probability an attack succeeds by the braking point, from
19 inputs (gap, run length, both batteries, Boost flags, Overtake Mode flag, tyres and ages, track
condition and temperature, segment, circuit, driver skills, car top-speed and pace deltas).
Framing for the brief: longitudinal vehicle dynamics + hybrid-powertrain energy management
(mechanical engineering).

**Method.**
1. *Physics simulator* (`overtake/physics.py`, `overtake/simulator.py`): two cars as point masses,
   states x, v, e (energy used) each, coupled through slipstream; drive force
   η(P_ICE+P_E)/v limited by traction μ w_r (mg + ½ρC_LA v²); drag ½ρC_DA k v² with slipstream factor
   k = 1 − 0.25 e^(−Δx/20 m) fading over one car length (5.5 m) when alongside; rolling resistance;
   electrical power P_E = s·P_K(v)·min(1,(E−e)/S) with s = 1 Boost, 0.55 standard map, ramp
   S = 20 kJ. Integrated with **RK4, dt = 0.05 s**, vectorised over chunks of 20,000 scenarios;
   braking-point crossing by Newton iteration on the cubic Hermite interpolant (4 iterations).
   Output: attacker's time deficit D at the braking point.
2. *Outcome model*: execution noise ε ~ N(0, σ), σ = 0.20/0.24/0.28 s (dry/damp/wet, more for low
   skill); lunge range L = 0.15 s; success logit z0 + 5·min(u, 0.5) with u = ε − (D − L).
   **Exact P** = ∫₀^∞ φσ(u + D − L)·logistic(z0 + 5 min(u, 0.5)) du, split at u = 0.5 (closed-form
   tail) with **16-point Gauss-Legendre** on [0, 0.5]. Labels for training are random 0/1 draws.
3. *Neural network* (`overtake/model.py`, `overtake/train.py`): embeddings (circuit 6, tyre 4 shared
   by both tyre columns, condition 2, segment 3) + 11 standardised numerics + 3 flags = 33 inputs →
   64 → 64 (SiLU, dropout 0.1) → 1 logit → sigmoid; 6,589 parameters; BCE, AdamW (lr 2e-3, wd 1e-4),
   batch 1024, ReduceLROnPlateau(patience 2, halves the LR after the 3rd epoch without improvement),
   early stopping patience 5 (max 100 epochs); **ensemble of 5 seeds** (0-4), mean of probabilities.
   Trained on 300,000 simulated attempts (80/10/10 split; 15 % of batteries start full; 28 % succeed).
   Network size chosen by validation log loss with the **one-standard-error rule** (64-64 over
   128-128-64). Physics-informed "energy cover" inputs were tested over 5 seeds and *not* adopted.
4. *Baselines*: logistic regression and histogram gradient boosting (`overtake/baselines.py`);
   direct simulation as a further alternative.
5. *Real-data comparison* (`overtake/realdata.py`): OpenF1 API, 17 races of 2026 before 2026-10-19
   (cut-off fixed in code as `RACES_BEFORE`); position changes classified start / neutralised
   (safety car, VSC, red flag, Overtake Mode disabled — from race-control messages) / pit stop /
   on track; speed traps; one car's telemetry (car 81, Monza); pass rate vs gap; circuits.

**Key results (all from `reports/results/*.json`).**

| Item | Value |
|---|---|
| Test set (30,000), mean abs error vs exact P | NN 2.0 pts; GB 3.8; LR 15.7 |
| Log loss / AUC / calibration error | NN 0.257 / 0.948 / 0.005; GB 0.267 / 0.944 / 0.014; LR 0.411 / 0.853 / 0.004 |
| Error quantiles (NN) | median 0.7, p90 5.6, p99 15.6, max 58.7 points |
| Default scenario | exact 47.4 %, network 44.0 %; without Boost < 1 % |
| Prediction time (comparison.json, incl. input checks) | NN 4.1 µs, GB 8.6, LR 3.6, direct simulation 94 µs → 23× |
| Forward pass only (performance.json) | 0.82 µs vs simulation 100 µs (≈120×); 81 GFLOP/s vs 0.7 |
| FLOPs per prediction | ensemble 66,565 (13,313 per network); simulation ≈68,600 (153 RK4 steps × ≈444) |
| Arithmetic intensity (network) | 0.5 FLOP/byte (batch 1) → 251 (batch 1024) |
| Training time | NN 208 s; GB 4 s; LR 3 s |
| Solver verification | RK4 observed order 4.05/4.00, Euler 0.95/1.00 vs exact solutions |
| Time step | worst deficit error 0.059 ms at dt 0.05 (limit 1 ms); before slipstream fix 1.1 ms |
| Stability (new) | battery ramp λ = sP_K/S = 17.5 s⁻¹ (Boost), RK4 real-axis limit 2.785 → dt < 0.159 s; energy budget first violated at dt 0.125 s (RK4, Boost); Euler limit 0.114 s |
| Quadrature | 12 GL nodes → 2e-15; Simpson 257 points → 3e-11; MC 1,600 replays 0.7-pt RMS, 189× slower |
| Vectorisation / chunking | 47× (200 scenarios); chunk 20,000 vs 1,000: 4.5× |
| Ensemble | single 2.24 ± 0.06 pts → ensemble 2.02 (validation, 5 seeds) |
| Data size | error 6.8 pts at 10k attempts → 2.2 at 240k, still falling |
| Failure modes | tyre mismatch 2.6 vs 1.9 pts (18 % of test, half of worst 1 %); default-scenario bias −3.5 pts (−10.5 wet at full battery); extrapolation 1300-2000 m: 4.4 pts (GB 9.7, LR 26.9) |
| Real data | 4,289 position changes → 962 start, 852 neutralised, 1,253 pit, **1,222 on track**; speed traps: 2 of 19,588 laps > 355 km/h (358); telemetry: standard map best on 115/135 straights, median RMS 9.9 km/h (Boost 21.2, none 23.4); pass rate falls far faster with gap than simulator (0.75-1 s: 0.19 vs 0.65 relative to 0.25-0.5 s); circuits Spearman ρ 0.65 (p 0.006, n 16), 0.58 without Monaco |
| Main lesson | verification passed, but real races show the simulator gives each attack too much energy (drivers ration it across the lap) |

## 4. Repository map (main + report-polish)

```
overtake/            Python package
  features.py        input schema, DEFAULTS, NUMERIC_RANGES, validate, Preprocessor
                     (+ on report-polish: situational_defaults, TRAINED_* ranges)
  physics.py         CarParameters, CarState, acceleration, slipstream, two_car_rhs, euler/rk4, run_to_braking_point
  simulator.py       constants, initial conditions, closing_run, outcome model, overtake_integral (quadrature),
                     probability, Monte Carlo, sample_scenarios, generate_dataset
  model.py           OvertakeNet
  train.py           training, ensemble, metrics, `python -m overtake.train`
  baselines.py       logistic regression, gradient boosting
  predict.py         OvertakePredictor, complete(), CLI `python -m overtake.predict`
  experiments.py     every study: solver, timestep, stability*, quadrature, selection, ablation, comparison,
                     performance, sensitivity, real, diagram*   (* new on report-polish)
  plots.py           every figure (+ workflow_diagram*, stability*, _label_points)
  export.py          spreadsheet of all simulations/results (`python -m overtake.export`, ~65 MB, not in git)
  realdata.py        OpenF1 download/cache (data/real/openf1, not in git) and analysis
tests/               pytest: physics, simulator, pipeline, experiments, export, realdata, notebooks (67 on report-polish)
notebooks/           walkthrough.ipynb, network_by_hand.ipynb (both executed by tests/test_notebooks.py)
models/overtake_net.pt  trained ensemble (154 kB)
reports/results/     JSON for every number; *_before_* files are earlier runs the docs refer to
reports/figures/     PNG for every figure
reports/test_report.html   pytest-html report
docs/project_documentation.md   THE REPORT (on report-polish: condensed 10-page version + appendices)
docs/api/            pdoc HTML;  docs/index.html  GitHub Pages landing page
README.md, LICENSE (MIT, Sam Gazzola), requirements.txt, pyproject.toml (pytest pythonpath)
```

## 5. Commands

| Task | Command | Time (4-core CPU) |
|---|---|---|
| Install | `pip install -r requirements.txt` (Python 3.10+, tested 3.13) | |
| Train network + baselines | `python -m overtake.train` | ~4 min |
| All experiments | `python -m overtake.experiments all` | ~25 min (+4 min OpenF1 download first time) |
| One experiment | `python -m overtake.experiments stability` (or solver, timestep, quadrature, selection, ablation, comparison, performance, sensitivity, real, diagram) | s to 10 min |
| Tests | `python -m pytest -q` | ~25 s |
| Test report | `python -m pytest --html=reports/test_report.html --self-contained-html` | |
| API docs | `rm -rf docs/api && pdoc overtake --docformat numpy -o docs/api` | s |
| Execute a notebook | `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` | s |
| Lint | `python -m pyflakes overtake/ tests/` | |
| Report → PDF (page count) | `python handoff/tools/render_report_pdf.py docs/project_documentation.md /tmp/report.pdf` | s |

`render_report_pdf.py` uses pandoc (gfm + tex math → HTML with print CSS) and Playwright Chromium.
In the cloud the Chromium path was `/opt/pw-browsers/chromium` (hard-coded in `launch(executable_path=...)`);
locally, delete that argument (after `playwright install chromium`) or point it at your Chrome. It prints the
page count and the page on which "References" starts; References and Appendices start on new pages.
Alternatives: `pandoc docs/project_documentation.md -o report.docx --resource-path docs` then LibreOffice
`soffice --headless --convert-to pdf report.docx`.

## 6. What happened in this session (chronological)

1. Built the first version (hand-tuned simulator + network), then rebuilt it to the brief after
   reading `MMA3001_2026_Project_Brief.pdf` (in Sam's Google Drive "3001 Project"): physics ODE with
   RK4/Euler, Hermite crossing, quadrature, verification tests, model selection, baselines,
   performance study (cProfile, FLOPs, arithmetic intensity), sensitivity, docs, pdoc, licence,
   notebook, test report. Merged to `main` (`1202fbc`).
2. Fixes found by our own checks along the way (each is in the docs' "errors caught" list): slipstream
   discontinuity (RK4 dropped to first order; fixed by fading over one car length); battery-edge
   hypothesis rejected; physics-informed inputs not robust over 5 seeds (not adopted); one-SE rule
   adopted after a selection flip between runs; crash on text input; wrong test expectations.
3. Spreadsheet export of every simulated attempt (`overtake/export.py`).
4. Real-data validation with OpenF1 (`overtake/realdata.py`, Section 5.4 of the old docs; branch
   `real-data-validation`, merged `5f30544`). Spot-checking a Monza pass revealed safety-car/red-flag
   position changes counted as passes → added race-control "neutralised" filter (on-track passes
   1,578 → 1,222; circuit ρ 0.56 → 0.65).
5. `notebooks/network_by_hand.ipynb` (NumPy forward pass of the ensemble reproduces the library to
   6e-8; FLOP count; one BCE gradient by hand vs autograd) + `tests/test_notebooks.py` running both
   notebooks. Branch `network-by-hand`, merged (`d4f531a`, current `main`).
6. Advice given to Sam about presenting (Section 9 below), including the stability analysis.
7. Current request (Section 1, item 2) started:
   * A 15-agent audit workflow against the brief (7 brief areas + 8 code-documentation groups,
     each with a verifier, then a completeness critic) was launched, then **stopped after 3 areas**
     to save credits. Results in `handoff/audit_findings.md`.
   * On branch `report-polish` (worktree in the cloud at `/home/user/gz-polish`):
     - commit `13814fe`: new `stability` experiment (+ figure, test), `workflow_diagram` figure
       (+ `diagram` experiment), removed stray root files `hello.py`, `hello-again.py` (Sam's
       early git test files);
     - commit `3930250`: **condensed report** (`docs/project_documentation.md`): Sections 1-8 fit 10
       A4 pages at 11 pt with the print CSS in `render_report_pdf.py`, leaving ~0.45 page for Sam's
       reflection; References on page 11; Appendices A-D (supplementary figures, the full-battery
       bias investigation, real-data details, reproduction commands) after it;
     - the final commit on the branch (with this handoff): **input-handling fixes** from the audit
       (see Section 7.1) — code done and all 67 tests pass, but *new tests for them still need to
       be written*.

## 7. Open work, in order

### 7.1 Finish the input-handling fixes (started; code done, tests and docs not)

Implemented on `report-polish` (`overtake/features.py`, `overtake/predict.py`, `overtake/simulator.py`):
`features.situational_defaults(segment, condition)` (800 m straight / 150 m corner; 35 °C dry / 20 °C
damp-wet) shared by the API and CLI; `features.TRAINED_RUN_LENGTH_M`, `TRAINED_TRACK_TEMP_C` used by both
`simulator.sample_scenarios` (identical values; sampled data verified bit-identical) and
`OvertakePredictor.out_of_range`; `simulator.DETECTION_GAP_SD_S = 0.2`; `predict.complete` rewritten
(`_fill_row` for dict/Series/list: unknown field → `ValueError`; `None` → default; NaN is *not* filled and
is rejected by `validate`; `_fill_columns` vectorised for DataFrames); `as_frame` and `predict` accept a
pandas Series (returns a float); `out_of_range` adds per-segment/per-condition ranges and an Overtake-Mode-
vs-gap consistency warning (on with gap > 1.6 s or off with gap < 0.4 s); CLI `finite_float` type (rejects
nan/inf, exit 2), clean message + exit 1 for a missing model, scenario completed once so the printout
matches the prediction; docstrings for `finite_float`, `parse_args`, `scenario_from_args`, `describe`.

To do:
* Tests in `tests/test_pipeline.py`: API corner default equals explicit 150 m; wet default temp 20 °C;
  Series input returns float equal to dict input; unknown field (`{"gap": 1.0}`) raises
  `ValueError` matching "unknown field"; NaN gap raises matching `gap_s`; `out_of_range` flags
  `{"segment": "slow_corner", "run_length_m": 800}`, wet 45 °C, and `{"gap_s": 2.5, "overtake_mode": True}`;
  CLI `--gap nan` exits 2; CLI missing model exits 1 with the message.
* Update README "Fields left out take the defaults..." and report Section 2 wording to match
  (the condensed report already says None means default; add "NaN is rejected", "unknown field names
  are rejected", "a pandas Series counts as one scenario", and the situational defaults).
* Re-run `python -m overtake.experiments comparison` is NOT needed (prediction path unchanged
  numerically for DataFrames), but double-check `predict_us_per_scenario` is unaffected if you do.

### 7.2 Finish the audit

Run the remaining areas with the prompts in `handoff/tools/audit_workflow_prompts.js` (it is the
cloud workflow script: `CONTEXT`, the per-area prompts `DOC_AREAS`, `CODE_STANDARD` and `CODE_GROUPS`,
the verifier prompt, the critic prompt). Areas still to run: **performance, ai-lessons-refs-writing,
repo-deliverables, numbers** (every number in docs vs JSON), and the **8 code-documentation groups**;
then verify each finding (skeptical verifier: "default to not real if you cannot confirm") and run the
completeness critic. Re-audit the *new* condensed report, not the old one.

### 7.3 Act on the audit findings already in hand (`handoff/audit_findings.md`)

Statuses are marked in that file. The open ones that matter most, with the recommended action:

*Engineering problem / inputs (A-series of the first area)*
* A5: say explicitly in Section 1 that this is a mechanical-engineering problem (longitudinal vehicle
  dynamics, hybrid powertrain energy management, tyre traction, aerodynamics).
* A6: the report claims the model shows rule makers how sensitive overtaking is to the Overtake Mode
  parameters, but no experiment varies them. Either add `experiments rules` (exact P at the default
  scenario for Overtake Mode energy 0-1 MJ and taper start 290-337 km/h; the constants are
  `physics.OVERTAKE_MODE_EXTRA_J` and the taper constants in `physics.py`) — recommended, it also
  makes a good "engineering interpretation" result and slide — or soften the sentence.
* A9-A12, A14: give inputs operational meaning (battery % of 4 MJ; skill anchor 60-100; segment
  classes; list of circuit keys + "use generic"), define the output event precisely, say calibration
  is against the simulator only (Section 5.3 shows real rates are lower at 0.5-1 s), justify scope
  limits, state the rules precisely (taper 290→355; Overtake Mode full power to 337, +0.5 MJ).

*Computational approach / alternatives (second area; IDs A1-A6, B1-B8, C1)*
* A1 (must): the argument "direct simulation cannot learn from real outcomes" is wrong — its few
  outcome constants (σ, L, z0 offsets, circuit passabilities) can be fitted by maximum likelihood to
  aggregate real data (pass-rate curve, per-circuit counts), which the network cannot use. Reframe:
  the network's advantages are speed and flexibility with labelled attempts; the simulator's is that
  it can be calibrated with the aggregate data that exist. Fix Table 4's "Can learn from real
  outcomes" row and Section 4's text (the condensed report still has the old argument).
* A3: document the missing constants, above all the **corner-exit speed** (`simulator.EXIT_SPEED_MPS`
  = 33 m/s × √grip; the most influential estimate), the completion-logit composition, caution and
  positioning terms, driven-axle share 0.55, air density 1.2, car length 5.5 m, dirty-air grip loss,
  grip/degradation tables, sampling distributions. Put a full constants table in an appendix and the
  corner-exit speed in Table 2.
* A4 (must): the plausibility numbers are not reproducible (no experiment calls `simulator.speed_trace`/
  `plots.speed_trace`) and one is wrong: no electrical power gives **288 km/h** at 800 m (303 km/h is
  the asymptotic top speed). Add `experiments speed_traces` writing `speed_trace.json` and the figure;
  fix the sentence in Section 3.1 of the condensed report ("303 km/h with no electrical power").
* A6: `train.py` docstring: ReduceLROnPlateau(patience=2) halves after the **3rd** epoch without a
  relative improvement of 1e-4; early stopping uses an absolute 1e-5.
* B1: put the speed advantage in absolute terms and state the break-even (training ≈208 s ÷ ≈90 µs
  saved per prediction ≈ 2.3 million predictions); say 0/1 training is a deliberate stand-in for real
  data (training on exact P would be a better pure surrogate).
* B2: the "fine-tune on real attempts" route is untested — present it as a proposal, not a finding.
* B3: compare against an adaptive solver with event location (SciPy `solve_ivp` RK45, one scenario at
  a time) — needs a small experiment; auditor measured ≈57 ms/scenario at rtol 1e-6 vs ≈0.08 ms for
  vectorised RK4. Also give Euler's cost to reach 1 ms (≈6.5× RK4) and to match RK4 (≈110×).
* B4: add comparison rows: extrapolation error, worst error (NN 58.7, GB 70.0, LR 98.9), tyre-mismatch
  error, working memory, cost scaling, implementation size.
* B5: Hermite vs linear crossing: auditor found linear gives 0.20 ms vs Hermite 0.059 ms (both < 1 ms).
  Either add a `crossing="linear"` option to the timestep experiment to measure it, or soften the claim.
* B6: the energy-cover inputs improved log loss in 5/5 seeds (paired SE ≈0.0002), so "within one
  standard error" is the wrong yardstick; justify rejection by the worst battery-axis error (12.6 vs
  7.4 pts) and the misplaced corner. Fix Appendix B.
* B7: report baseline settings (GB: ≤500 iterations, early stopping on 10 % split, lr 0.1, 31 leaves;
  LR: L2, C = 1 — verify in `baselines.py`) and show GB was not under-tuned (small tuning experiment on
  the validation set).
* B8: add an explicit assumptions list (straight-line, flat, no wind; same corner-exit speed;
  one deployment level per run; whole battery available for this straight; only the attacker
  slipstreams; neither driver reacts).
* C1: carry the ODE error into P (|dP/dD| ≤ ≈2 per second → ≤ 0.012 points).

*Validation (third area; V-series)* — V1 done (stability). Important open ones:
* **V2 (must):** the time-step study's "observed order ≈2.7" comes from a fit that includes
  dt = 0.2 s, outside the stability limit; over the stable, asymptotic steps the order is ≈2.0 (the
  theoretical order with kinks in the right-hand side). Fix the fit range in
  `experiments.timestep` (`fine = slice(...)`), rerun, and fix the text (condensed report doesn't quote
  2.7 any more but Appendix/figure captions should be checked).
* V3-V7: answer the interview questions explicitly in the report (error budget carried to P; analytic
  check of the energy equation and of the production `run_to_braking_point` path; "most vulnerable
  part" — now in Section 5.1; convergence criteria for the Newton iterations (fixed 4, add a residual
  check) and quadrature).
* V8-V9: acceptance criteria and uncertainty for the network metrics (e.g. bootstrap CI on the 2.0
  points; Bayes-optimal log loss of the exact P as a reference) and pass/fail criteria for the real-data
  checks.
* V10: sensitivity to the *estimated constants* (corner-exit speed, σ, C_DA, battery energy), not only
  to inputs — a small experiment would answer interview Q2.
* V12: the "expected direction" check only at the default scenario; ground truth is non-monotonic in
  the gap in some scenarios (documented failure mode).
* V13: validate paired what-if differences (the intended use is comparing options).
* V15/V17: calibration is against the simulator only; the test set is i.i.d. from the training sampler.

### 7.4 Code documentation pass (Sam explicitly asked for this)

Every module, class, function and method — including private helpers (`_name`), nested functions,
lambdas assigned to names, pytest tests and fixtures — must have a docstring; public ones in NumPy
style with Parameters / Returns / Raises (and Examples for entry points); inline comments must explain
*intent* at non-obvious steps (physics formulas, numerical choices, magic numbers, unit conversions,
vectorisation tricks), never restate code. Known gaps: many test functions have no docstring
(only descriptive names); helpers like `experiments._single_car`, `experiments._write`, `_timed`,
`realdata._times` partly; check every file. Then regenerate pdoc and the test report.

### 7.5 Report (5-10 pages) — draft done on `report-polish`

`docs/project_documentation.md` is the condensed report. Current budget: Sections 1-8 = 10 pages with
~0.45 page free for Sam's reflection; any additions from Section 7.3 must be paid for by cuts or go to
appendices (appendices and references don't count). Known issues: the figures are 10 in wide with 9 pt
fonts, so at full text width they print at about 6 pt — consider regenerating report figures with
larger fonts (`plots.py` rcParams `font.size` 9, titles 11) and re-checking label overlaps. Figure
numbering: main 1-4 (workflow, solver convergence, what-if, real gap), appendix A1-A8, C1-C3. Table 1
column "Accepted (trained)". Test count in the text says 67 — update after adding tests (Section 3.5
and the AI table). Deliverables for Moodle: a PDF (and DOCX so Sam can paste in his reflection).
After finalising, update README (it still describes the old long documentation in places).

### 7.6 Presentation (PowerPoint + script)

Brief: **5-minute presentation + 3 minutes of questions, week 14**, no AI. Must include: problem with
primary input/output; demo of GitHub structure and code documentation; evidence of validation and
testing; chosen approach and why; principal limitations/lessons; Sam's use and evaluation of AI.
Running order agreed with Sam (≈1 slide per 40-60 s):
1. 0:00-0:40 Problem (2026 rules → P(overtake)); show CLI example (1 s, softs, full battery → 44 %;
   without Boost 0.7 %).
2. 0:40-1:40 Approach and why (no public battery data → physics simulator gives labelled data with
   exact answer → network learns from 0/1); `what_if.png` or `workflow.png`.
3. 1:40-2:50 Validation: three layers — verification (`solver_convergence.png`), network vs exact P and
   baselines (`method_comparison.png`), real races (`real_gap.png`); 67+ tests.
4. 2:50-3:40 GitHub demo (README → docs/api → test report → branches/commits); screenshots as backup.
5. 3:40-4:20 Limitations/lessons (slipstream bug found by convergence; battery ramp sets stability;
   verification ≠ validation: energy rationing).
6. 4:20-5:00 AI use and evaluation (Sam's own words).
Backup slides for questions: `timestep_study.png` (+ before-fix), `stability.png`, `speed_trace.png`,
`sensitivity.png`, `real_telemetry.png`, `real_speed_traps.png`, `quadrature_study.png`, `calibration.png`.
Produce: `.pptx` (python-pptx or pptxgenjs; 16:9; large fonts; one message per slide; speaker notes)
and a separate script (~650-700 words for 5 min) plus Q&A prep. Render to images to QA overflow.
Suggested location: a `presentation/` folder (ask Sam whether he wants it in the repo or only locally).

### 7.7 Detailed explanation document ("study guide")

Sam asked for "a separate document that explains everything in the project documentation in detail".
Audience: Sam, preparing for the interview without AI. Explain each section of the report from first
principles: the 2026 rules; the force model and each term; why RK4, what order of accuracy means, how
convergence was measured; stability regions (with the 17.5 s⁻¹ / 2.785 / 0.16 s calculation); the
Hermite crossing; the outcome model and why P is an integral; Gauss-Legendre vs Simpson vs Monte Carlo;
what the network computes (point to the by-hand notebook); training (BCE, gradients, AdamW, early
stopping, one-SE rule, ensembles); every validation metric and why; the real-data pipeline and
filters (Wilson interval, Spearman); performance concepts (profiling, FLOPs, arithmetic intensity,
memory-bound vs compute-bound); limitations and lessons; and model answers to every example question
in the brief. Format: Word/PDF or a doc in Sam's preferred tool (ask him).

### 7.8 Wrap-up

Merge `report-polish` into `main` with `--no-ff` **after deleting `handoff/`**, regenerate
`docs/api` and `reports/test_report.html`, push. Final check: fresh clone → `pip install -r
requirements.txt` → `python -m pytest` passes → `python -m overtake.predict --help` works → README
commands valid. Then give Sam the PDF/DOCX report, the PPTX, the script and the study guide.

## 8. Things that tripped us up (save yourself the time)

* OpenF1: free tier ≈30 requests/min (code spaces requests 2.2 s); 404 means "no results" (treated as
  empty); some 2026 races have no data (Sakhir, Jeddah). Cache in `data/real/openf1/` (gitignored, 82 MB):
  `python -m overtake.experiments real` re-downloads it on a fresh clone (~4 min).
* Timestamps: OpenF1 mixes formats; `realdata._times` parses with `format="ISO8601"` and converts to
  tz-naive UTC — comparing tz-aware with tz-naive raised errors before.
* Race control does not announce every safety car with "OVERTAKE DISABLED"; the neutralised-period
  state machine combines both kinds of message (and red flags end only at OVERTAKE ENABLED).
* `experiments.stability`: `t_end` must be a whole number of every step, or positions are compared at
  different times (an early bug produced nonsense position errors).
* Spreadsheet export: LibreOffice turned TRUE/FALSE into formulas → flags stored as 1/0; float32
  rounding needed care.
* Figures: label overlaps are common; `plots._label_points` places scatter labels automatically;
  after any change, look at the PNG.
* The stop hook in the cloud demanded commit+push before finishing a turn; not relevant locally.

## 9. Interview-prep notes already given to Sam

* Validation = three questions, each against something that can answer it: (1) verification —
  analytical solutions, convergence order, stability, quadrature reference, Monte Carlo, physical
  constraints; (2) network vs exact P on held-out data and baselines; (3) simulator vs real 2026
  races. Key line: verification and network validation passed, yet real data found a wrong
  assumption (energy per attack) — "verification is not validation".
* Stability: RK4 explicit, stable for |λΔt| ≤ 2.785 (real axis). Vehicle speed dynamics |λ| ≈ 0.25 s⁻¹
  (2kv/m + P/(mv²) at 90 m/s); battery ramp |λ| = 350 kW / 20 kJ = 17.5 s⁻¹ → Δt < 0.16 s; chosen 0.05 s;
  empirical onset 0.125 s (energy overdrawn). This ramp is "the part most vulnerable to numerical
  failure"; mitigations: fixed step with margin, energy-budget test, exponential/implicit treatment of
  the energy equation if larger steps were needed.
* Error budget: time stepping ≤ 0.059 ms of deficit vs 200 ms execution noise; quadrature 1e-15;
  Hermite crossing fourth order; model error dominates.
* Most influential inputs (sensitivity): gap, straight length, both batteries, Boost; most influential
  constant (audit V10): corner-exit speed — not yet quantified.
* The "network by hand" notebook is the best way for Sam to be able to explain what the network
  computes: 33 → 64 → 64 → 1, SiLU, sigmoid, mean of 5; 13,313 FLOPs per network.
