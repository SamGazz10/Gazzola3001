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
status 2; a missing model file exits with status 1).
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

    Attributes
    ----------
    models : list of overtake.model.OvertakeNet
        The networks, in evaluation mode (dropout off).
    preprocessor : overtake.features.Preprocessor
        The shared preprocessing.
    """

    def __init__(self, models, preprocessor):
        """Store the networks in evaluation mode and the shared preprocessor (parameters as for the class)."""
        # Accept one network or an ensemble; eval() turns dropout off so predictions are deterministic.
        self.models = [m.eval() for m in (models if isinstance(models, (list, tuple)) else [models])]
        self.preprocessor = preprocessor

    @property
    def model(self):
        """The first network (all members share the same architecture)."""
        return self.models[0]

    @property
    def n_parameters(self):
        """Trainable parameters (embeddings, weights and biases) summed over all networks."""
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
        KeyError
            If the file is not a checkpoint written by `save`.

        Examples
        --------
        >>> predictor = OvertakePredictor.load()  # doctest: +SKIP
        >>> len(predictor.models), predictor.n_parameters  # doctest: +SKIP
        (5, 32945)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no trained model at {path}; run `python -m overtake.train` first")
        # map_location="cpu": loads on machines without a GPU. weights_only=True: unpickles tensors and
        # plain containers only, never arbitrary code, which is why `save` requires plain extras.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        models = []
        for state in checkpoint["model_states"]:
            model = OvertakeNet(checkpoint["hidden"], checkpoint["dropout"], checkpoint["n_inputs"])
            model.load_state_dict(state)
            models.append(model)
        return cls(models, Preprocessor.from_state_dict(checkpoint["preprocessor"]))

    def save(self, path, **extra):
        """Write weights, architecture and preprocessing statistics to ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            Checkpoint file; missing parent folders are created.
        **extra
            Stored under their own keys (`overtake.train.main` adds
            ``metrics``, ``history`` and ``fit_seconds``). Must be plain
            Python values (such as dict, list, tuple, str, int, float, bool,
            None), because `load` reads with ``torch.load(weights_only=True)``,
            which refuses arbitrary pickled objects.

        Notes
        -----
        Checkpoint keys: ``model_states`` (one state dict per network),
        ``hidden``, ``dropout``, ``n_inputs`` and ``preprocessor``
        (`overtake.features.Preprocessor.state_dict`), plus the ``extra`` keys.
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
        float or numpy.ndarray of float32, shape (n,)
            A float for a single scenario (dict or Series), otherwise one
            probability per row. Either way the value is the mean of the
            ensemble members' probabilities.

        Raises
        ------
        ValueError
            If a field name is unknown or a value is outside the schema (see
            `overtake.features.validate`).

        Examples
        --------
        With the committed model (the values change if it is retrained):

        >>> predictor = OvertakePredictor.load()  # doctest: +SKIP
        >>> round(predictor.predict({"gap_s": 0.5, "attacker_battery": 80}), 2)  # doctest: +SKIP
        0.8
        >>> predictor.predict([{"gap_s": 0.5}, {"gap_s": 1.5}]).round(2)  # doctest: +SKIP
        array([0.78, 0.04], dtype=float32)
        """
        df = complete(scenarios)
        cat, num = self.preprocessor.transform(df)
        # Ensemble: the mean of the members' probabilities (sigmoid of each logit), not of their logits.
        # no_grad because this is inference only.
        with torch.no_grad():
            p = torch.stack([torch.sigmoid(m(cat, num)) for m in self.models]).mean(dim=0).numpy()
        # One dict or Series gives a float; a list or DataFrame gives one value per row.
        return float(p[0]) if isinstance(scenarios, (dict, pd.Series)) else p

    def out_of_range(self, scenario):
        """List inputs outside the range seen in training, where predictions are extrapolation.

        Values are not validated here (`predict` does that), so call this on a
        scenario that `predict` accepts, as `main` does.

        Parameters
        ----------
        scenario : dict or pandas.Series
            One scenario (missing fields take default values, see `complete`).
            Anything else `complete` accepts also works, but only its first
            row is checked.

        Returns
        -------
        list of str
            One message per input outside the training data, empty if none:
            a numeric input more than 2 % of its range outside the training
            minimum-maximum; a run length or track temperature more than 2 %
            outside the range trained for that segment (straight or any
            corner) or weather (dry, or damp and wet); or Overtake Mode
            inconsistent with the gap: on with a gap more than three standard
            deviations of the detection-point noise above 1 s (1.6 s), or off
            with a gap more than three below it (0.4 s).

        Raises
        ------
        ValueError
            If a field name is unknown (see `complete`) or a numeric field
            cannot be read as a number.
        """
        df = complete(scenario)
        # Only the first row is checked; see the docstring.
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
            # Small tolerance (2 % of the range) so values at the edge of the training data are not
            # flagged as extrapolation.
            slack = 0.02 * (hi - lo)
            if not lo - slack <= value <= hi + slack:
                warnings.append(f"{col}={value:g} is outside the training range {lo:g}-{hi:g}{where}")
        # In training, Overtake Mode follows the detection-point gap, which differs from the attack gap
        # by N(0, DETECTION_GAP_SD_S = 0.2 s). Beyond 3 SD from 1 s a mismatch has probability below
        # 0.14 % per attempt: of 300,000 simulated attempts (seed 0), 6 have the mode on above 1.6 s
        # (4 in the training split) and 13 have it off below 0.4 s, so the network has almost no data there.
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
    and ``overtake_mode`` (on when ``gap_s`` is 1 s or less). For dict, Series
    or list-of-dict input a field set to ``None`` also means "use the default";
    in a DataFrame, empty cells (None or NaN) are kept. Values are not checked
    here: `overtake.features.validate` later rejects empty cells and other
    invalid values with a message naming the field.

    Parameters
    ----------
    scenarios : dict, pandas.Series, list of dict or pandas.DataFrame
        One scenario (dict or Series) or several (list of dicts, or a
        DataFrame with one row per scenario) in the `overtake.features`
        schema. A DataFrame is copied, not modified.

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
    """Fill one scenario's omitted or ``None`` fields with their defaults.

    Parameters
    ----------
    row : dict
        Field name -> value; names already checked by `complete`.

    Returns
    -------
    dict
        Every schema field: the given values, the situational run length and
        track temperature for the given segment and weather, and
        ``overtake_mode`` from the gap (on at 1 s or less) unless given.
    """
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
    """Add any missing schema columns to a DataFrame, in place.

    Columns already present are left untouched, including empty cells
    (`overtake.features.validate` rejects those later).

    Parameters
    ----------
    df : pandas.DataFrame
        Scenarios, one per row; modified in place (`complete` passes a copy).

    Returns
    -------
    pandas.DataFrame
        The same object, with every `overtake.features.FEATURES` column.
    """
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
        The option's value as typed on the command line.

    Returns
    -------
    float
        The parsed value.

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

    ``--run-length`` and ``--track-temp`` are left as ``None`` when not given,
    so that `complete` fills the situational value for the segment and
    weather (see `scenario_from_args`).

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` if omitted.

    Returns
    -------
    argparse.Namespace
        One attribute per option (``args.gap``, ``args.defender_battery``, ...).

    Raises
    ------
    SystemExit
        Status 2 for an unknown option or an invalid value (argparse prints
        the usage and the error); status 0 after printing ``--help``.
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
        Schema field -> value. Every field is present except ``run_length_m``
        and ``track_temp_c`` when their options were not given.
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
        # auto applies the 1 s rule, taking the detection-point gap equal to the attack gap.
        "overtake_mode": {"auto": args.gap <= 1.0, "on": True, "off": False}[args.overtake_mode],
    }
    # Unset options are None; dropping them lets `complete` fill the situational defaults.
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
        Four indented lines (track, attacker, defender, gap) joined by
        newlines, as printed by `main`.
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
    """Command-line entry point: print P(overtake) for one scenario.

    Prints the completed scenario, its probability, the change Boost makes
    when the attacker uses it, any extrapolation warnings from
    `OvertakePredictor.out_of_range`, and the same situation at gaps of
    0.25-1.5 s. See ``python -m overtake.predict --help`` for the options.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments without the program name; ``sys.argv[1:]`` if
        omitted.

    Raises
    ------
    SystemExit
        Status 1 if there is no trained model at ``--model``; status 2 if an
        option is malformed (argparse) or the scenario fails
        `overtake.features.validate`.

    Examples
    --------
    With the committed model this prints ``P(overtake) = 19.7%``::

        python -m overtake.predict --segment slow_corner --gap 0.6 --battery 70
    """
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
        # Boost effect: the same scenario with only the attacker's Boost switched off.
        p_no_boost = predictor.predict({**scenario, "attacker_boost": False})
        print(f"  without Boost: {p_no_boost:.1%}  (Boost adds {100 * (p - p_no_boost):+.1f} points)")
    for warning in predictor.out_of_range(scenario):
        print(f"  warning: {warning}")

    print("\nSame situation at other gaps")
    # Vary only the gap. With --overtake-mode auto the mode follows each gap by the 1 s rule; a forced
    # on or off is held fixed.
    for gap in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        q = predictor.predict({**scenario, "gap_s": gap,
                               "overtake_mode": gap <= 1.0 if args.overtake_mode == "auto" else scenario["overtake_mode"]})
        print(f"  {gap:4.2f} s  {q:6.1%}")


if __name__ == "__main__":
    main()
