"""Feature schema shared by the simulator, the network and the prediction tool.

Every overtaking attempt is described by one row with the columns below. The
same schema is used for simulated data and for any real data you collect, so
the network does not care where a row came from.
"""
import numpy as np
import pandas as pd
import torch

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

# A typical attack: fresh softs, full battery, 1 s behind on an 800 m straight,
# against a car on 15-lap-old mediums with 60% battery that also uses Boost.
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
    "run_length_m": (0.0, 2000.0),
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


def validate(df):
    """Raise ValueError if a column is missing or a value is outside the schema."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col, vocab in CATEGORICAL.items():
        bad = sorted(set(df[col]) - set(VOCABS[vocab]))
        if bad:
            raise ValueError(f"{col}: unknown value(s) {bad}; allowed: {VOCABS[vocab]}")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        values = df[col].astype(float)
        if values.isna().any() or (values < lo).any() or (values > hi).any():
            raise ValueError(f"{col}: values must be between {lo} and {hi}")


class Preprocessor:
    """Turns schema rows into the tensors the network takes.

    Categorical columns become vocabulary indices; numeric columns are
    standardised with the training set's mean and standard deviation; binary
    columns become 0/1.
    """

    def __init__(self, means=None, stds=None, mins=None, maxs=None):
        self.means = means
        self.stds = stds
        # Training range of each numeric input, used to flag extrapolation.
        self.mins = mins
        self.maxs = maxs

    def fit(self, df):
        self.means = {c: float(df[c].mean()) for c in NUMERIC}
        self.stds = {c: float(df[c].std()) or 1.0 for c in NUMERIC}
        self.mins = {c: float(df[c].min()) for c in NUMERIC}
        self.maxs = {c: float(df[c].max()) for c in NUMERIC}
        return self

    def transform(self, df):
        validate(df)
        cat = np.stack(
            [df[col].map({v: i for i, v in enumerate(VOCABS[vocab])}).to_numpy()
             for col, vocab in CATEGORICAL.items()],
            axis=1,
        )
        num = np.stack(
            [(df[c].to_numpy(dtype=np.float32) - self.means[c]) / self.stds[c] for c in NUMERIC]
            + [df[c].astype(bool).to_numpy(dtype=np.float32) for c in BINARY],
            axis=1,
        )
        return torch.as_tensor(cat, dtype=torch.long), torch.as_tensor(num, dtype=torch.float32)

    def state_dict(self):
        return {"means": self.means, "stds": self.stds, "mins": self.mins, "maxs": self.maxs}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


def as_frame(scenarios):
    """Accept a dict (one scenario), a list of dicts or a DataFrame."""
    if isinstance(scenarios, pd.DataFrame):
        return scenarios.reset_index(drop=True)
    if isinstance(scenarios, dict):
        scenarios = [scenarios]
    return pd.DataFrame(list(scenarios))
