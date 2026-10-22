"""Input schema shared by the simulator, the network and the prediction tool.

Every overtaking attempt is one row with the columns below. The same schema is
used for simulated data and for any real data you collect, so the network does
not care where a row came from.

Inputs
------
| Column | Type | Meaning (units, allowed values) |
|---|---|---|
| `track_condition` | category | `dry`, `damp`, `wet` |
| `segment` | category | where the move is completed: `straight`, `slow_corner`, `medium_corner`, `fast_corner` |
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
| `top_speed_delta_kph` | number | attacker car's top-speed advantage from lower drag, km/h (negative = slower) |
| `pace_delta_s` | number | attacker car's lap-time advantage, s/lap |
| `attacker_boost` | bool | attacker uses Boost for the attack |
| `defender_boost` | bool | defender uses Boost to defend |
| `overtake_mode` | bool | attacker was within 1 s at the detection point |

A `straight` attack is completed in the braking zone at the end of the
straight. Flags accept 1/0 as well as True/False.

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
# The 24 circuits in 2026 calendar order, plus "generic" for any other or unknown track.
CIRCUITS = [
    "generic", "melbourne", "shanghai", "suzuka", "bahrain", "jeddah", "miami",
    "montreal", "monaco", "barcelona", "spielberg", "silverstone", "spa",
    "hungaroring", "zandvoort", "monza", "madrid", "baku", "singapore", "austin",
    "mexico_city", "interlagos", "las_vegas", "lusail", "yas_marina",
]

# Vocabulary name -> allowed values. A value's position in its list is its
# embedding index, so changing any list invalidates saved checkpoints:
# reordering silently remaps categories, and adding a value changes the size
# of the embedding table.
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
    "gap_s",                # gap to the car ahead at the corner exit where the attack starts (s)
    "run_length_m",         # distance from that exit to the braking point (m)
    "attacker_battery",     # attacker state of charge (%)
    "defender_battery",     # defender state of charge (%)
    "attacker_tyre_age",    # laps
    "defender_tyre_age",    # laps
    "track_temp_c",         # deg C
    "attacker_skill",       # racecraft rating 0-100
    "defender_skill",       # defensive rating 0-100
    "top_speed_delta_kph",  # attacker car's top-speed advantage from lower drag (km/h), energy aside
    "pace_delta_s",         # attacker car's lap-time advantage (s/lap), tyres aside
]

BINARY = [
    "attacker_boost",   # attacker uses Boost (maximum engine + battery power) for the attack
    "defender_boost",   # defender uses Boost to defend
    "overtake_mode",    # attacker was within 1 s at the detection point: +0.5 MJ
]

FEATURES = list(CATEGORICAL) + NUMERIC + BINARY  # the 19 model inputs
LABEL = "overtake"  # 0/1 outcome column in training data

# Optional physics-informed inputs, computed from the ones above: the share of
# the run each car's battery could power at full MGU-K output, capped at 1.
# The chance of an overtake stops rising abruptly once a battery covers the
# run, and these inputs were meant to hand the network that corner. They are
# not used by the final model: experiments.ablation found the gain within one
# standard error, and the simple energy estimate puts the corner in the wrong
# place (it ignores the power taper).
DERIVED = ["attacker_energy_cover", "defender_energy_cover"]
RUN_SPEED_MPS = 80.0  # typical average speed along a straight, 288 km/h (estimate)

# A typical attack: nearly new softs (5 laps), full battery, 1 s behind on an
# 800 m straight, against a car on 15-lap-old mediums with 60% battery that
# also uses Boost. Dry, 35 C, a generic circuit, equal cars and drivers (both
# rated 85), and the attacker has Overtake Mode.
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

# Hard limits (engineering judgement) that `validate` enforces, to reject
# impossible or out-of-scope inputs before they reach the network. They are
# wider than the training data; see the TRAINED_* ranges below and
# `Preprocessor.mins` / `Preprocessor.maxs`.
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
# through `situational_defaults`. Both are estimates inside the trained ranges below.
CORNER_RUN_LENGTH_M = 150.0
WET_TRACK_TEMP_C = 20.0

# Narrower ranges that the training data covers within the accepted ones
# (`overtake.simulator.sample_scenarios` draws from exactly these). Predictions
# outside them are extrapolation, and `OvertakePredictor.out_of_range` says so.
TRAINED_RUN_LENGTH_M = {"straight": (250.0, 1300.0), "corner": (50.0, 300.0)}
TRAINED_TRACK_TEMP_C = {"dry": (18.0, 55.0), "wet": (10.0, 32.0)}  # "wet" covers damp and wet


def situational_defaults(segment, condition):
    """Defaults that depend on where the move happens and on the weather.

    The arguments are not validated: any segment other than ``"straight"``
    counts as a corner, and any condition other than ``"dry"`` as damp or wet.

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
        One row per scenario. Extra columns (such as the label) are ignored.

    Raises
    ------
    ValueError
        If a column of `FEATURES` is missing, a category is not in its
        vocabulary, a number is missing, non-numeric or outside
        `NUMERIC_RANGES`, or a flag is not True/False (1/0 is accepted). The
        message names the offending column(s) and the allowed values or range.
        Checks run in that order and stop at the first failure.

    Examples
    --------
    >>> validate(as_frame({**DEFAULTS, "gap_s": 5.0}))
    Traceback (most recent call last):
        ...
    ValueError: gap_s: every value must be a number between 0.0 and 3.0
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col, vocab in CATEGORICAL.items():
        # key=str lets the sort handle mixed types, e.g. NaN next to strings.
        bad = sorted(set(df[col]) - set(VOCABS[vocab]), key=str)
        if bad:
            raise ValueError(f"{col}: unknown value(s) {bad}; allowed: {VOCABS[vocab]}")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        # Non-numbers become NaN here, so they fail the same check as missing values.
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or (values < lo).any() or (values > hi).any():
            raise ValueError(f"{col}: every value must be a number between {lo} and {hi}")
    for col in BINARY:
        # 1 == True and 0 == False in this comparison, so 1/0 flags pass too.
        if not df[col].isin([True, False]).all():
            raise ValueError(f"{col}: values must be True/False (or 1/0)")


def derived_features(df):
    """Physics-informed inputs computed from a validated table of scenarios.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios that have passed `validate`. Only ``run_length_m``,
        ``attacker_battery``, ``defender_battery`` and ``overtake_mode`` are read.

    Returns
    -------
    dict of str -> numpy.ndarray
        ``attacker_energy_cover`` and ``defender_energy_cover``, each of shape
        (n,) with values in [0, 1]: battery energy (4 MJ at 100 %, plus 0.5 MJ
        with Overtake Mode for the attacker) divided by the energy needed to
        run 350 kW for the whole run at 80 m/s, capped at 1. At the `DEFAULTS`
        (800 m, so 3.5 MJ needed) the attacker's 4.5 MJ gives 1 and the
        defender's 2.4 MJ gives 0.69.

    Raises
    ------
    KeyError
        If one of the columns above is missing (`validate` catches this first).
    """
    # Energy (J) for full MGU-K power over the time the run takes at RUN_SPEED_MPS.
    need = CarParameters().mguk_power_w * df["run_length_m"].to_numpy(dtype=float) / RUN_SPEED_MPS
    # State of charge (%) -> usable energy (J); Overtake Mode's extra energy goes to the attacker only.
    attacker = (BATTERY_USABLE_J * df["attacker_battery"].to_numpy(dtype=float) / 100
                + OVERTAKE_MODE_EXTRA_J * df["overtake_mode"].to_numpy(dtype=float))
    defender = BATTERY_USABLE_J * df["defender_battery"].to_numpy(dtype=float) / 100
    return {"attacker_energy_cover": np.minimum(attacker / need, 1.0),
            "defender_energy_cover": np.minimum(defender / need, 1.0)}


def with_derived(df):
    """Add the `DERIVED` columns to a table of scenarios.

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios that have passed `validate`.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with ``attacker_energy_cover`` and
        ``defender_energy_cover`` added (see `derived_features`); ``df`` itself
        is not modified.
    """
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

    Attributes
    ----------
    numeric_columns : list of str
        `NUMERIC`, followed by `DERIVED` when ``physics_features`` is set. This
        is the order of the standardised columns in `transform`'s output.
    means, stds, mins, maxs : dict of str -> float or None
        As above; ``None`` until fitted. ``mins`` and ``maxs`` are the training
        range of each numeric input, which
        `overtake.predict.OvertakePredictor.out_of_range` uses to flag
        extrapolation.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame([DEFAULTS, {**DEFAULTS, "gap_s": 0.5, "circuit": "monza"}])
    >>> cat, num = Preprocessor().fit(df).transform(df)
    >>> cat.shape, num.shape
    (torch.Size([2, 5]), torch.Size([2, 14]))
    >>> cat[:, 2]  # circuit: index of "generic" and "monza" in CIRCUITS
    tensor([ 0, 15])
    >>> num[:, 0]  # gap_s, standardised with this two-row "training set"
    tensor([ 0.7071, -0.7071])
    """

    def __init__(self, means=None, stds=None, mins=None, maxs=None, physics_features=False):
        """Store the statistics and choose the numeric columns; see the class docstring."""
        self.physics_features = physics_features
        self.numeric_columns = NUMERIC + (DERIVED if physics_features else [])
        self.means = means
        self.stds = stds
        # Training range of each numeric input, used to flag extrapolation.
        self.mins = mins
        self.maxs = maxs

    def fit(self, df):
        """Record the mean, standard deviation and range of each numeric column.

        Parameters
        ----------
        df : pandas.DataFrame
            The training set. Never pass validation or test data here, or their
            statistics leak into the inputs.

        Returns
        -------
        Preprocessor
            ``self``, now fitted, so calls can be chained.

        Raises
        ------
        ValueError
            If ``df`` fails `validate`.
        """
        validate(df)
        df = with_derived(df)
        self.means = {c: float(df[c].mean()) for c in self.numeric_columns}
        # Sample standard deviation (pandas' ddof=1). A constant column has 0,
        # replaced by 1 so it standardises to 0 instead of dividing by zero.
        self.stds = {c: float(df[c].std()) or 1.0 for c in self.numeric_columns}
        self.mins = {c: float(df[c].min()) for c in self.numeric_columns}
        self.maxs = {c: float(df[c].max()) for c in self.numeric_columns}
        return self

    @property
    def n_inputs(self):
        """Number of numeric and flag inputs the network receives: 14, or 16 with ``physics_features``."""
        return len(self.numeric_columns) + len(BINARY)

    def transform(self, df):
        """Convert scenarios to network inputs.

        Parameters
        ----------
        df : pandas.DataFrame
            Scenarios in the schema of this module; extra columns are ignored.

        Returns
        -------
        cat : torch.LongTensor, shape (n, 5)
            Vocabulary index of each categorical column, in `CATEGORICAL` order.
        num : torch.FloatTensor, shape (n, `n_inputs`)
            Standardised numeric (and derived) columns in `numeric_columns`
            order, followed by the 0/1 flags in `BINARY` order.

        Raises
        ------
        ValueError
            If ``df`` fails `validate`.
        TypeError
            If the preprocessor has not been fitted (its statistics are ``None``).
        """
        validate(df)
        df = with_derived(df)
        # Position in the vocabulary = embedding row. Both tyre columns map
        # through the tyre vocabulary, so they index the same embedding table.
        cat = np.stack(
            [df[col].map({v: i for i, v in enumerate(VOCABS[vocab])}).to_numpy()
             for col, vocab in CATEGORICAL.items()],
            axis=1,
        )
        # Numeric columns use the training statistics; flags are not scaled,
        # and the bool cast turns True/False or 1/0 into 1.0/0.0.
        num = np.stack(
            [(df[c].to_numpy(dtype=np.float32) - self.means[c]) / self.stds[c] for c in self.numeric_columns]
            + [df[c].astype(bool).to_numpy(dtype=np.float32) for c in BINARY],
            axis=1,
        )
        return torch.as_tensor(cat, dtype=torch.long), torch.as_tensor(num, dtype=torch.float32)

    def state_dict(self):
        """Statistics and settings for saving in a checkpoint.

        Returns
        -------
        dict
            ``means``, ``stds``, ``mins``, ``maxs`` (dicts of float) and
            ``physics_features`` (bool): plain Python values, so the
            checkpoint loads with ``torch.load(..., weights_only=True)``.
        """
        return {"means": self.means, "stds": self.stds, "mins": self.mins, "maxs": self.maxs,
                "physics_features": self.physics_features}

    @classmethod
    def from_state_dict(cls, state):
        """Rebuild a fitted preprocessor from a checkpoint.

        Parameters
        ----------
        state : dict
            Output of `state_dict`.

        Returns
        -------
        Preprocessor
            Ready to `transform` without refitting.
        """
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
        One row per scenario with a fresh 0..n-1 index (a DataFrame input is
        copied, not modified). Values are not checked; see `validate`.
    """
    if isinstance(scenarios, pd.DataFrame):
        return scenarios.reset_index(drop=True)
    if isinstance(scenarios, (dict, pd.Series)):
        scenarios = [dict(scenarios)]
    return pd.DataFrame(list(scenarios))
