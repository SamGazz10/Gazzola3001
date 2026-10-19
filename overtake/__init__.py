"""Probability of an F1 overtake under the 2026 Boost / Overtake Mode rules.

A physics-based simulator of overtaking attempts generates training data with
a known ground-truth probability; a neural network learns the probability from
0/1 outcomes alone, as it would have to with real race data.

Modules, in workflow order:

| Module | Role |
|---|---|
| `overtake.features` | Input schema, defaults, validation, preprocessing |
| `overtake.physics` | Two-car equations of motion, RK4 / Euler integrators |
| `overtake.simulator` | Scenario sampling, outcomes, exact probability by quadrature |
| `overtake.model` | The neural network |
| `overtake.train` | Training, metrics, evaluation against alternatives |
| `overtake.baselines` | Logistic regression and gradient-boosted trees |
| `overtake.predict` | Command-line and Python prediction |
| `overtake.experiments` | Verification, model selection, comparison, performance, sensitivity |
| `overtake.plots` | Figures |

See the README and `docs/project_documentation.md` in the repository for the
full write-up.
"""
