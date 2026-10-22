"""Input schema shared by the simulator, the network and the prediction tool.

Every overtaking attempt is one row with the columns below. The same schema is
used for simulated data and for any real data you collect, so the network does
not care where a row came from.

Inputs
------
| Column | Type | Meaning (units, allowed values) |
|---|---|---|
| `track_condition` | category | `dry`, `damp`, `wet` |
| `segment` | category | where the move is completed: `straight` (into the braking zone at its end), `slow_corner`, `medium_corner`, `fast_corner` |
| `circuit` | category | one of `CIRCUITS`; `generic` if unknown |
| `attacker_tyre` | category | `soft`, `medium`, `hard`, `intermediate`, `wet` |
| `defender_tyre` | category | as above |
| `gap_s` | number | gap to the car ahead at the corner exit, s |
| `run_length_m` | number | distance from that exit to the braking point, m |
| `attacker_battery` | number | state of charge, % |
| `defender_battery` | number | state of charge, % |
| `attacker_tyre_age` | number | laps |
| `defender_tyre_age` | number | laps |
| `track_temp_c` | number | deg C |
| `attacker_skill` | number | racecraft rating, 0-100 |
| `defender_skill` | number | defensive rating, 0-100 |
| `top_speed_delta_kph` | number | attacker car's straight-line speed advantage from lower drag, km/h (negative = slower) |
| `pace_delta_s` | number | attacker car's lap-time advantage, s/lap |
| `attacker_boost` | bool | attacker uses Boost for the attack |
| `defender_boost` | bool | defender uses Boost to defend |
| `overtake_mode` | bool | attacker was within 1 s at the detection point |

Hard limits are in `NUMERIC_RANGES`; `validate` rejects anything outside them,
unknown category values, missing values and non-boolean flags with a
``ValueError`` naming the column. Inputs inside the limits but outside the
training data's range are accepted, and
`overtake.predict.OvertakePredictor.out_of_range` flags them as extrapolation.

Output
------
P(overtake): the probability, between 0 and 1, that the attacker completes the
pass in this attempt.
"""
import numpy as np
import pandas as pd
import torch

from .physics import BATTERY_USABLE_J, OVERTAKE_MODE_EXTRA_J, CarParameters

CONDITIONS = ["dry", "damp", "wet"]
COMPOUNDS = ["soft", "medium", "hard", "intermediate", "wet"]
# Where the attack happens: down a straight into the braking zone, or into/through a corner.
SEGMENTS = ["straight", "slow_corner", "medium_corner", "fast_corner"]
CIRCUITS = [
    "generic", "melbourne", "shanghai", "suzuka", "bahrain", "jeddah", "miami",
    "montreal", "monaco", "barcelona", "spielberg", "silverstone", "spa",
    "hungaroring", "zandvoort", "monza", "madrid", "baku", "singapore", "austin",
    "mexico_city", "interlagos", "las_vegas", "lusail", "yas_marina",
]

VOCABS = {
    "condition": CONDITIONS,
    "segment": SEGMENTS,
    "circuit": CIRCUITS,
    "tyre": COMPOUNDS,
}

# Categorical column -> vocabulary. Both tyre columns share one vocabulary so
# the network learns a single embedding per compound.
CATEGORICAL = {
    "track_condition": "condition",
    "segment": "segment",
    "circuit": "circuit",
    "attacker_tyre": "tyre",
    "defender_tyre": "tyre",
}

NUMERIC = [
    "gap_s",                # gap to the car ahead at the start of the attack (seconds)
    "run_length_m",         # length of the straight/run before the braking point (metres)
    "attacker_battery",     # attacker state of charge (%)
    "defender_battery",     # defender state of charge (%)
    "attacker_tyre_age",    # laps
    "defender_tyre_age",    # laps
    "track_temp_c",
    "attacker_skill",       # racecraft rating 0-100
    "defender_skill",       # defensive rating 0-100
    "top_speed_delta_kph",  # attacker car's straight-line speed advantage, energy aside
    "pace_delta_s",         # attacker car's lap-time advantage (s/lap), tyres aside
]

BINARY = [
    "attacker_boost",   # attacker uses Boost (max engine + battery power) for the attack
    "defender_boost",   # defender uses Boost to defend
    "overtake_mode",    # attacker was within 1 s at the detection point: +0.5 MJ
]

FEATURES = list(CATEGORICAL) + NUMERIC + BINARY
LABEL = "overtake"

# Optional physics-informed inputs, computed from the ones above: the share of
# the run each car's battery could power at full MGU-K output, capped at 1.
# The chance of an overtake stops rising abruptly once a battery covers the
# run, and these inputs were meant to hand the network that corner. They are
# not used by the final model: experiments.ablation found the gain within one
# standard error, and the simple energy estimate puts the corner in the wrong
# place (it ignores the power taper).
DERIVED = ["attacker_energy_cover", "defender_energy_cover"]
RUN_SPEED_MPS = 80.0  # typical average speed along a straight, for the energy estimate

# A typical attack: nearly new softs (5 laps), full battery, 1 s behind on an
# 800 m straight, against a car on 15-lap-old mediums with 60% battery that
# also uses Boost.
DEFAULTS = {
    "track_condition": "dry",
    "segment": "straight",
    "circuit": "generic",
    "attacker_tyre": "soft",
    "defender_tyre": "medium",
    "gap_s": 1.0,
    "run_length_m": 800.0,
    "attacker_battery": 100.0,
    "defender_battery": 60.0,
    "attacker_tyre_age": 5.0,
    "defender_tyre_age": 15.0,
    "track_temp_c": 35.0,
    "attacker_skill": 85.0,
    "defender_skill": 85.0,
    "top_speed_delta_kph": 0.0,
    "pace_delta_s": 0.0,
    "attacker_boost": True,
    "defender_boost": True,
    "overtake_mode": True,
}

# Hard limits used to reject impossible inputs before they reach the network.
NUMERIC_RANGES = {
    "gap_s": (0.0, 3.0),
    "run_length_m": (10.0, 2000.0),
    "attacker_battery": (0.0, 100.0),
    "defender_battery": (0.0, 100.0),
    "attacker_tyre_age": (0.0, 80.0),
    "defender_tyre_age": (0.0, 80.0),
    "track_temp_c": (0.0, 70.0),
    "attacker_skill": (0.0, 100.0),
    "defender_skill": (0.0, 100.0),
    "top_speed_delta_kph": (-30.0, 30.0),
    "pace_delta_s": (-3.0, 3.0),
}

# Defaults that depend on the situation: a run into a corner is short, and a
# damp or wet track is cooler. The Python API and the command line share them
# through `situational_defaults`.
CORNER_RUN_LENGTH_M = 150.0
WET_TRACK_TEMP_C = 20.0

# Narrower ranges that the training data covers within the accepted ones
# (`overtake.simulator.sample_scenarios` draws from exactly these). Predictions
# outside them are extrapolation, and `OvertakePredictor.out_of_range` says so.
TRAINED_RUN_LENGTH_M = {"straight": (250.0, 1300.0), "corner": (50.0, 300.0)}
TRAINED_TRACK_TEMP_C = {"dry": (18.0, 55.0), "wet": (10.0, 32.0)}  # "wet" covers damp and wet


def situational_defaults(segment, condition):
    """Defaults that depend on where the move happens and on the weather.

    Parameters
    ----------
    segment : str
        One of `SEGMENTS`.
    condition : str
        One of `CONDITIONS`.

    Returns
    -------
    dict
        ``run_length_m``: 800 m on a straight, 150 m into a corner;
        ``track_temp_c``: 35 C when dry, 20 C when damp or wet.
    """
    return {"run_length_m": DEFAULTS["run_length_m"] if segment == "straight" else CORNER_RUN_LENGTH_M,
            "track_temp_c": DEFAULTS["track_temp_c"] if condition == "dry" else WET_TRACK_TEMP_C}


def validate(df):
    """Check that a table of scenarios matches the schema.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per scenario. Extra columns are ignored.

    Raises
    ------
    ValueError
        If a column is missing, a category is unknown, a number is missing or
        outside `NUMERIC_RANGES`, or a flag is not true/false. The message
        names the column and the allowed values.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col, vocab in CATEGORICAL.items():
        bad = sorted(set(df[col]) - set(VOCABS[vocab]), key=str)
        if bad:
            raise ValueError(f"{col}: unknown value(s) {bad}; allowed: {VOCABS[vocab]}")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or (values < lo).any() or (values > hi).any():
            raise ValueError(f"{col}: every value must be a number between {lo} and {hi}")
    for col in BINARY:
        if not df[col].isin([True, False]).all():
            raise ValueError(f"{col}: values must be True/False (or 1/0)")


def derived_features(df):
    """Physics-informed inputs computed from a validated table of scenarios.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict of str -> numpy.ndarray
        ``attacker_energy_cover`` and ``defender_energy_cover``: battery energy
        (plus 0.5 MJ with Overtake Mode for the attacker) divided by the energy
        needed to run 350 kW for the whole run at 80 m/s, capped at 1.
    """
    need = CarParameters().mguk_power_w * df["run_length_m"].to_numpy(dtype=float) / RUN_SPEED_MPS
    attacker = (BATTERY_USABLE_J * df["attacker_battery"].to_numpy(dtype=float) / 100
                + OVERTAKE_MODE_EXTRA_J * df["overtake_mode"].to_numpy(dtype=float))
    defender = BATTERY_USABLE_J * df["defender_battery"].to_numpy(dtype=float) / 100
    return {"attacker_energy_cover": np.minimum(attacker / need, 1.0),
            "defender_energy_cover": np.minimum(defender / need, 1.0)}


def with_derived(df):
    """Return ``df`` with the `DERIVED` columns added (``df`` must already be valid)."""
    return df.assign(**derived_features(df))


class Preprocessor:
    """Turns schema rows into the tensors the network takes.

    Categorical columns become vocabulary indices; numeric columns are
    standardised with the training set's mean and standard deviation; binary
    columns become 0/1. Statistics come from the training set only, so
    validation and test data are transformed exactly as unseen data would be.

    Parameters
    ----------
    means, stds, mins, maxs : dict of str -> float, optional
        Per-column training statistics; set by `fit` or restored from a
        checkpoint with `from_state_dict`.
    physics_features : bool, optional
        Add the `DERIVED` energy-cover inputs. Off by default (raw inputs
        only); see `overtake.experiments.ablation`.
    """

    def __init__(self, means=None, stds=None, mins=None, maxs=None, physics_features=False):
        self.physics_features = physics_features
        self.numeric_columns = NUMERIC + (DERIVED if physics_features else [])
        self.means = means
        self.stds = stds
        # Training range of each numeric input, used to flag extrapolation.
        self.mins = mins
        self.maxs = maxs

    def fit(self, df):
        """Record mean, standard deviation and range of each numeric column. Returns self."""
        validate(df)
        df = with_derived(df)
        self.means = {c: float(df[c].mean()) for c in self.numeric_columns}
        self.stds = {c: float(df[c].std()) or 1.0 for c in self.numeric_columns}
        self.mins = {c: float(df[c].min()) for c in self.numeric_columns}
        self.maxs = {c: float(df[c].max()) for c in self.numeric_columns}
        return self

    @property
    def n_inputs(self):
        """Number of numeric and flag inputs the network receives."""
        return len(self.numeric_columns) + len(BINARY)

    def transform(self, df):
        """Convert scenarios to network inputs.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        cat : torch.LongTensor, shape (n, 5)
            Vocabulary index of each categorical column.
        num : torch.FloatTensor, shape (n, `n_inputs`)
            Standardised numeric (and derived) columns followed by the 0/1 flags.

        Raises
        ------
        ValueError
            If ``df`` fails `validate`.
        """
        validate(df)
        df = with_derived(df)
        cat = np.stack(
            [df[col].map({v: i for i, v in enumerate(VOCABS[vocab])}).to_numpy()
             for col, vocab in CATEGORICAL.items()],
            axis=1,
        )
        num = np.stack(
            [(df[c].to_numpy(dtype=np.float32) - self.means[c]) / self.stds[c] for c in self.numeric_columns]
            + [df[c].astype(bool).to_numpy(dtype=np.float32) for c in BINARY],
            axis=1,
        )
        return torch.as_tensor(cat, dtype=torch.long), torch.as_tensor(num, dtype=torch.float32)

    def state_dict(self):
        """Statistics as plain dicts, for saving in a checkpoint."""
        return {"means": self.means, "stds": self.stds, "mins": self.mins, "maxs": self.maxs,
                "physics_features": self.physics_features}

    @classmethod
    def from_state_dict(cls, state):
        """Rebuild a fitted preprocessor from `state_dict` output."""
        return cls(**state)


def as_frame(scenarios):
    """Convert scenarios to a DataFrame with one row per scenario.

    Parameters
    ----------
    scenarios : dict, pandas.Series, list of dict or pandas.DataFrame
        A single scenario (dict or Series, e.g. one DataFrame row), several, or a table.

    Returns
    -------
    pandas.DataFrame
    """
    if isinstance(scenarios, pd.DataFrame):
        return scenarios.reset_index(drop=True)
    if isinstance(scenarios, (dict, pd.Series)):
        scenarios = [dict(scenarios)]
    return pd.DataFrame(list(scenarios))
