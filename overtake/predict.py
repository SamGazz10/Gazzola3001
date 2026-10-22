"""Predict the probability of an overtake for a race situation.

    python -m overtake.predict --condition dry --tyre soft --segment straight --battery 100 --gap 1.0

From Python:

    from overtake.predict import OvertakePredictor
    predictor = OvertakePredictor.load()
    predictor.predict({"track_condition": "dry", "attacker_tyre": "soft", "gap_s": 1.0})

Fields you leave out (or set to ``None``) take the values in
`overtake.features.DEFAULTS`, except the run length and track temperature,
which depend on the segment and weather (`overtake.features.situational_defaults`),
and ``overtake_mode``, which defaults to "gap is 1 s or less". Invalid inputs
raise a ``ValueError`` (the command line prints the message and exits with
status 2).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import (CIRCUITS, COMPOUNDS, CONDITIONS, DEFAULTS, FEATURES, NUMERIC, SEGMENTS,
                       TRAINED_RUN_LENGTH_M, TRAINED_TRACK_TEMP_C, Preprocessor, situational_defaults)
from .model import OvertakeNet
from .simulator import DETECTION_GAP_SD_S

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "overtake_net.pt"


class OvertakePredictor:
    """One or more trained networks together with the preprocessing they were trained with.

    With several networks (an ensemble trained from different random seeds)
    the prediction is the mean of their probabilities.

    Parameters
    ----------
    models : overtake.model.OvertakeNet or list of them
        All trained on the same data, so they share one preprocessor.
    preprocessor : overtake.features.Preprocessor
        Fitted on the networks' training data.
    """

    def __init__(self, models, preprocessor):
        self.models = [m.eval() for m in (models if isinstance(models, (list, tuple)) else [models])]
        self.preprocessor = preprocessor

    @property
    def model(self):
        """The first network (all members share the same architecture)."""
        return self.models[0]

    @property
    def n_parameters(self):
        """Trainable parameters across all networks."""
        return sum(p.numel() for m in self.models for p in m.parameters())

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        """Load a checkpoint written by `save`.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Defaults to ``models/overtake_net.pt``.

        Returns
        -------
        OvertakePredictor

        Raises
        ------
        FileNotFoundError
            If there is no checkpoint at ``path``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no trained model at {path}; run `python -m overtake.train` first")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        models = []
        for state in checkpoint["model_states"]:
            model = OvertakeNet(checkpoint["hidden"], checkpoint["dropout"], checkpoint["n_inputs"])
            model.load_state_dict(state)
            models.append(model)
        return cls(models, Preprocessor.from_state_dict(checkpoint["preprocessor"]))

    def save(self, path, **extra):
        """Write weights, architecture and preprocessing statistics to ``path``.

        Extra keyword arguments (e.g. metrics, training history) are stored
        alongside and must be plain Python values.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_states": [m.state_dict() for m in self.models],
            "hidden": self.model.hidden,
            "dropout": self.model.dropout,
            "n_inputs": self.model.n_inputs,
            "preprocessor": self.preprocessor.state_dict(),
            **extra,
        }, path)

    def predict(self, scenarios):
        """Probability of an overtake.

        Parameters
        ----------
        scenarios : dict, pandas.Series, list of dict or pandas.DataFrame
            One or more scenarios in the `overtake.features` schema. Missing
            fields take their default values (see `complete`).

        Returns
        -------
        float or numpy.ndarray
            A float for a single scenario (dict or Series), otherwise one
            probability per row: the mean of the ensemble members' probabilities.

        Raises
        ------
        ValueError
            If a field name is unknown or a value is outside the schema (see
            `overtake.features.validate`).

        Examples
        --------
        >>> predictor = OvertakePredictor.load()
        >>> round(predictor.predict({"gap_s": 0.5, "attacker_battery": 80}), 2)  # doctest: +SKIP
        0.74
        """
        df = complete(scenarios)
        cat, num = self.preprocessor.transform(df)
        with torch.no_grad():
            p = torch.stack([torch.sigmoid(m(cat, num)) for m in self.models]).mean(dim=0).numpy()
        return float(p[0]) if isinstance(scenarios, (dict, pd.Series)) else p

    def out_of_range(self, scenario):
        """List inputs outside the range seen in training, where predictions are extrapolation.

        Parameters
        ----------
        scenario : dict
            One scenario (missing fields take default values).

        Returns
        -------
        list of str
            One message per input outside the training data, empty if none:
            a numeric input more than 2 % of its range outside the training
            minimum-maximum; a run length or track temperature outside the
            range trained for that segment or weather; or Overtake Mode
            inconsistent with the gap (on beyond, or off within, three standard
            deviations of the detection-point noise around 1 s).
        """
        df = complete(scenario)
        row = df.iloc[0]
        warnings = []
        # Run length and temperature were trained on narrower, situation-specific ranges.
        situational = {
            "run_length_m": ("straight" if row["segment"] == "straight" else "corner", TRAINED_RUN_LENGTH_M,
                             "segments"),
            "track_temp_c": ("dry" if row["track_condition"] == "dry" else "wet", TRAINED_TRACK_TEMP_C,
                             "conditions"),
        }
        for col in NUMERIC:
            value = float(row[col])
            if col in situational:
                key, ranges, what = situational[col]
                lo, hi = ranges[key]
                where = f" for {key} {what}" if key != "wet" else " for damp or wet conditions"
            else:
                lo, hi, where = self.preprocessor.mins[col], self.preprocessor.maxs[col], ""
            slack = 0.02 * (hi - lo)
            if not lo - slack <= value <= hi + slack:
                warnings.append(f"{col}={value:g} is outside the training range {lo:g}-{hi:g}{where}")
        # In training, Overtake Mode follows the detection-point gap, which differs from
        # the attack gap by N(0, DETECTION_GAP_SD_S); far from 1 s a mismatch never occurs.
        gap, limit = float(row["gap_s"]), 3 * DETECTION_GAP_SD_S
        if bool(row["overtake_mode"]) and gap > 1.0 + limit:
            warnings.append(f"overtake_mode is on with gap_s={gap:g}; training data has it only within {1 + limit:g} s")
        if not bool(row["overtake_mode"]) and gap < 1.0 - limit:
            warnings.append(f"overtake_mode is off with gap_s={gap:g}; training data always has it on below {1 - limit:g} s")
        return warnings


def complete(scenarios):
    """Return the scenarios as a DataFrame with every field of the schema.

    Omitted fields take their defaults: `overtake.features.DEFAULTS`, except
    ``run_length_m`` and ``track_temp_c`` (`overtake.features.situational_defaults`)
    and ``overtake_mode`` (on when ``gap_s`` is 1 s or less). For dict or Series
    input a field set to ``None`` also means "use the default". Values are not
    checked here: `overtake.features.validate` rejects NaN and other invalid
    values with a message naming the field.

    Parameters
    ----------
    scenarios : dict, pandas.Series, list of dict or pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        One row per scenario.

    Raises
    ------
    ValueError
        If a dict or Series has a field that is not in the schema (usually a
        typo). Extra DataFrame columns, such as a label, are ignored.
    """
    if isinstance(scenarios, pd.DataFrame):
        return _fill_columns(scenarios.reset_index(drop=True).copy())
    rows = [dict(scenarios)] if isinstance(scenarios, (dict, pd.Series)) else [dict(r) for r in scenarios]
    unknown = sorted({key for row in rows for key in row} - set(FEATURES))
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; allowed fields: {FEATURES}")
    return pd.DataFrame([_fill_row(row) for row in rows], columns=FEATURES)


def _fill_row(row):
    """One scenario dict with omitted or ``None`` fields replaced by their defaults."""
    given = {key: value for key, value in row.items() if value is not None}
    segment = given.get("segment", DEFAULTS["segment"])
    condition = given.get("track_condition", DEFAULTS["track_condition"])
    filled = {**DEFAULTS, **situational_defaults(segment, condition), **given}
    if "overtake_mode" not in given:
        # A gap that is not a number gives False here; validate() then rejects the gap itself.
        gap = pd.to_numeric(pd.Series([filled["gap_s"]]), errors="coerce").iloc[0]
        filled["overtake_mode"] = bool(gap <= 1.0)
    return filled


def _fill_columns(df):
    """Add any missing schema columns to a DataFrame, vectorised (existing cells are left as they are)."""
    for col, value in DEFAULTS.items():
        if col in df.columns or col == "overtake_mode":
            continue
        if col in ("run_length_m", "track_temp_c"):
            # DEFAULTS lists segment and track_condition first, so both columns exist by now.
            value = np.array([situational_defaults(seg, cond)[col]
                              for seg, cond in zip(df["segment"], df["track_condition"])], dtype=float)
        df[col] = value
    if "overtake_mode" not in df.columns:
        df["overtake_mode"] = pd.to_numeric(df["gap_s"], errors="coerce") <= 1.0
    return df


def finite_float(text):
    """argparse type for a finite number: ``float()`` alone would accept "nan" and "inf".

    Parameters
    ----------
    text : str

    Returns
    -------
    float

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not a finite number (argparse then exits with status 2).
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number")
    return value


def parse_args(argv=None):
    """Parse command-line options; every option defaults to the default scenario.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` if omitted.

    Returns
    -------
    argparse.Namespace
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--condition", choices=CONDITIONS, default=DEFAULTS["track_condition"])
    p.add_argument("--segment", choices=SEGMENTS, default=DEFAULTS["segment"])
    p.add_argument("--circuit", choices=CIRCUITS, default=DEFAULTS["circuit"])
    p.add_argument("--gap", type=finite_float, default=DEFAULTS["gap_s"], help="seconds to the car ahead")
    p.add_argument("--run-length", type=finite_float,
                   help="metres of straight before the braking point (default 800 on a straight, 150 into a corner)")
    p.add_argument("--track-temp", type=finite_float, help="deg C (default 35 dry, 20 damp/wet)")

    a = p.add_argument_group("attacker")
    a.add_argument("--tyre", choices=COMPOUNDS, default=DEFAULTS["attacker_tyre"])
    a.add_argument("--tyre-age", type=finite_float, default=DEFAULTS["attacker_tyre_age"], help="laps")
    a.add_argument("--battery", type=finite_float, default=DEFAULTS["attacker_battery"],
                   help="state of charge, %% of the 4 MJ usable energy")
    a.add_argument("--skill", type=finite_float, default=DEFAULTS["attacker_skill"],
                   help="racecraft rating 0-100 (grid range about 60-100)")
    a.add_argument("--no-boost", action="store_true", help="attack without Boost")
    a.add_argument("--overtake-mode", choices=["auto", "on", "off"], default="auto",
                   help="auto = on when the gap is 1 s or less")
    a.add_argument("--top-speed-delta", type=finite_float, default=0.0, help="car straight-line advantage, km/h")
    a.add_argument("--pace-delta", type=finite_float, default=0.0, help="car lap-time advantage, s/lap")

    d = p.add_argument_group("defender")
    d.add_argument("--defender-tyre", choices=COMPOUNDS, default=DEFAULTS["defender_tyre"])
    d.add_argument("--defender-tyre-age", type=finite_float, default=DEFAULTS["defender_tyre_age"])
    d.add_argument("--defender-battery", type=finite_float, default=DEFAULTS["defender_battery"])
    d.add_argument("--defender-skill", type=finite_float, default=DEFAULTS["defender_skill"])
    d.add_argument("--defender-no-boost", action="store_true", help="defender does not use Boost")

    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return p.parse_args(argv)


def scenario_from_args(args):
    """Translate parsed options into one scenario dict.

    Options left unset (run length, track temperature) are omitted so that
    `complete` gives them their situational defaults, exactly as in Python.

    Parameters
    ----------
    args : argparse.Namespace
        From `parse_args`.

    Returns
    -------
    dict
    """
    scenario = {
        "track_condition": args.condition,
        "segment": args.segment,
        "circuit": args.circuit,
        "attacker_tyre": args.tyre,
        "defender_tyre": args.defender_tyre,
        "gap_s": args.gap,
        "run_length_m": args.run_length,
        "attacker_battery": args.battery,
        "defender_battery": args.defender_battery,
        "attacker_tyre_age": args.tyre_age,
        "defender_tyre_age": args.defender_tyre_age,
        "track_temp_c": args.track_temp,
        "attacker_skill": args.skill,
        "defender_skill": args.defender_skill,
        "top_speed_delta_kph": args.top_speed_delta,
        "pace_delta_s": args.pace_delta,
        "attacker_boost": not args.no_boost,
        "defender_boost": not args.defender_no_boost,
        "overtake_mode": {"auto": args.gap <= 1.0, "on": True, "off": False}[args.overtake_mode],
    }
    return {key: value for key, value in scenario.items() if value is not None}


def describe(s):
    """Format a completed scenario as the indented summary the command line prints.

    Parameters
    ----------
    s : dict
        A scenario with every field (see `complete`).

    Returns
    -------
    str
    """
    on = {True: "on", False: "off"}
    return "\n".join([
        f"  Track:    {s['track_condition']}, {s['track_temp_c']:g} C, {s['circuit']}, "
        f"{s['segment'].replace('_', ' ')} ({s['run_length_m']:g} m run)",
        f"  Attacker: {s['attacker_tyre']} ({s['attacker_tyre_age']:g} laps), battery {s['attacker_battery']:g}%, "
        f"Boost {on[s['attacker_boost']]}, Overtake Mode {on[s['overtake_mode']]}, skill {s['attacker_skill']:g}",
        f"  Defender: {s['defender_tyre']} ({s['defender_tyre_age']:g} laps), battery {s['defender_battery']:g}%, "
        f"Boost {on[s['defender_boost']]}, skill {s['defender_skill']:g}",
        f"  Gap:      {s['gap_s']:.2f} s",
    ])


def main(argv=None):
    """Command-line entry point; see ``python -m overtake.predict --help``."""
    args = parse_args(argv)
    try:
        predictor = OvertakePredictor.load(args.model)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    try:
        # Fill the situational defaults once, so that what is printed is what is predicted.
        scenario = complete(scenario_from_args(args)).iloc[0].to_dict()
        p = predictor.predict(scenario)
    except ValueError as error:
        print(f"invalid input: {error}", file=sys.stderr)
        raise SystemExit(2)

    print("Scenario")
    print(describe(scenario))
    print(f"\nP(overtake) = {p:.1%}")
    if scenario["attacker_boost"]:
        p_no_boost = predictor.predict({**scenario, "attacker_boost": False})
        print(f"  without Boost: {p_no_boost:.1%}  (Boost adds {100 * (p - p_no_boost):+.1f} points)")
    for warning in predictor.out_of_range(scenario):
        print(f"  warning: {warning}")

    print("\nSame situation at other gaps")
    for gap in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        q = predictor.predict({**scenario, "gap_s": gap,
                               "overtake_mode": gap <= 1.0 if args.overtake_mode == "auto" else scenario["overtake_mode"]})
        print(f"  {gap:4.2f} s  {q:6.1%}")


if __name__ == "__main__":
    main()
