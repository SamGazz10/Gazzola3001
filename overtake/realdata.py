"""Real 2026 race data from the OpenF1 API, for comparison with the simulator.

    python -m overtake.realdata download      # fetch and cache (about 4 minutes, once)
    python -m overtake.experiments real       # download if needed, analyse, write
                                              # reports/results/real_data.json and four figures

From Python::

    from overtake import realdata
    realdata.download()
    result = realdata.analyse()     # dict described in `analyse`

OpenF1 (https://openf1.org) is a free, unofficial API of Formula 1 timing
data. No public source records battery charge or Boost / Overtake Mode use
(the old DRS channel is empty in 2026), so the real data can test the
simulator only indirectly: speeds, overtakes and gaps.

Responses are cached as JSON under ``data/real/openf1/`` (not in git), so
the download runs once and every later analysis is offline and repeatable.
OpenF1's free tier allows about 30 requests a minute, so requests are spaced
2.2 s apart.

The analysis functions take plain OpenF1 rows (lists of dicts), so they can
be used and tested without the network.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import physics
from .train import ROOT

API = "https://api.openf1.org/v1"
CACHE = ROOT / "data" / "real" / "openf1"
REQUEST_SPACING_S = 2.2
# "session" holds the race's OpenF1 circuit_short_name (matched in `download` and `analyse`), not a session_key.
TELEMETRY = {"session": "Monza", "driver_number": 81}  # one car, one race, for the speed-trace check
RACES_BEFORE = "2026-10-19"  # date of the analysis; fixed so later races do not change the results

# Endpoints needed for every race: what each is used for.
RACE_ENDPOINTS = {
    "overtakes": "position changes between two cars (OpenF1 beta; includes pit-stop swaps)",
    "laps": "lap start times, pit-out laps and speed-trap speeds for every car",
    "intervals": "gap to the car ahead, about every 4 s",
    "pit": "pit stops (lap and time), to discard pit-stop position changes",
    "race_control": "safety car, red flag and Overtake Mode enabled/disabled messages",
}

_last_request = 0.0  # time.monotonic() at the end of the last API call, for the spacing in `fetch`


def fetch(endpoint, **params):
    """Return an OpenF1 response, from the cache if present, otherwise from the API.

    Every response, an empty one included, is saved under `CACHE` and never
    refreshed: a race fetched before OpenF1 had its data stays empty until its
    file is deleted. API calls are spaced at least `REQUEST_SPACING_S` apart,
    and a rate-limited call (HTTP 429) is retried after 15, 30 and 45 s.

    Parameters
    ----------
    endpoint : str
        OpenF1 endpoint name, e.g. ``"laps"``.
    **params
        Query parameters, e.g. ``session_key=11361``. They also name the cache file.

    Returns
    -------
    list of dict
        One dict per OpenF1 record; ``[]`` if OpenF1 answers 404 ("No results found").

    Raises
    ------
    urllib.error.HTTPError
        For an HTTP error other than 404, or a fourth 429 in a row.
    urllib.error.URLError
        If the data is not cached and the API cannot be reached.
    TimeoutError
        If the response stalls for 120 s (the socket timeout) after the request is sent.

    Examples
    --------
    >>> laps = fetch("laps", session_key=11361)   # doctest: +SKIP
    """
    global _last_request
    # Cache file named by the endpoint and the sorted parameters, e.g. laps_session_key-11361.json.
    name = endpoint + "".join(f"_{k}-{v}" for k, v in sorted(params.items())) + ".json"
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text())
    # Spacing runs from the end of the previous API call to the start of this one.
    wait = REQUEST_SPACING_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):  # up to 4 attempts; only a 429 is retried, after 15, 30, then 45 s
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # socket timeout, s (estimate)
                data = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:  # OpenF1 answers 404 "No results found" when there is no data
                data = []
                break
            if error.code != 429 or attempt == 3:  # 429 = rate limited: wait and retry
                raise
            time.sleep(15 * (attempt + 1))
    _last_request = time.monotonic()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def completed_races(year=2026, before=RACES_BEFORE):
    """Grand Prix races (not sprints) of ``year`` dated before ``before`` (YYYY-MM-DD).

    Some may have no data in OpenF1; `download` and `analyse` skip those.

    Parameters
    ----------
    year : int, optional
        Season.
    before : str, optional
        Cut-off date "YYYY-MM-DD", exclusive, compared with the date part of the
        race's ``date_start`` (UTC). Fixed by default (`RACES_BEFORE`) so later
        races do not change the results.

    Returns
    -------
    list of dict
        OpenF1 ``sessions`` records (``session_key``, ``circuit_short_name``,
        ``date_start``, ...), in the order OpenF1 returns them.

    Raises
    ------
    urllib.error.URLError
        If the session list is not cached and OpenF1 cannot be reached (see `fetch`).
    """
    sessions = fetch("sessions", year=year, session_type="Race")
    # Session type "Race" also covers sprints; session_name tells them apart. ISO dates compare as strings.
    return [s for s in sessions if s["session_name"] == "Race" and s["date_start"][:10] < before]


def download(log=print):
    """Fetch and cache everything the comparison needs.

    For every completed race (`completed_races`) with lap data, each of the
    `RACE_ENDPOINTS`; then the telemetry (``car_data``) of car
    ``TELEMETRY["driver_number"]`` at the ``TELEMETRY["session"]`` race.
    Cached responses are not fetched again, so a second run is offline. This
    is what ``python -m overtake.realdata download`` runs.

    Parameters
    ----------
    log : callable, optional
        Receives the progress lines: the race count, one line per race and one
        for the telemetry (default `print`).

    Returns
    -------
    list of dict
        OpenF1 session records of every completed race, including those
        skipped for having no data.

    Raises
    ------
    urllib.error.URLError
        If something is not cached and OpenF1 cannot be reached (see `fetch`
        for the other network errors).
    StopIteration
        If the ``TELEMETRY["session"]`` race is not among the completed races.

    Examples
    --------
    >>> races = download(log=lambda line: None)   # doctest: +SKIP
    """
    races = completed_races()
    log(f"{len(races)} completed races")
    for race in races:
        if not fetch("laps", session_key=race["session_key"]):
            log(f"  {race['circuit_short_name']:18s} no data in OpenF1, skipped")
            continue
        counts = []
        for endpoint in RACE_ENDPOINTS:
            counts.append(f"{endpoint} {len(fetch(endpoint, session_key=race['session_key']))}")
        log(f"  {race['circuit_short_name']:18s} " + ", ".join(counts))
    monza = next(r for r in races if r["circuit_short_name"] == TELEMETRY["session"])
    rows = fetch("car_data", session_key=monza["session_key"], driver_number=TELEMETRY["driver_number"])
    log(f"  telemetry: car {TELEMETRY['driver_number']} at Monza, {len(rows):,} samples")
    return races



# --- Analysis ----------------------------------------------------------------
#
# Each function below turns raw OpenF1 rows into one comparison with the
# simulator. They take plain lists of dicts / DataFrames so they can be
# tested offline with small hand-made inputs.

# OpenF1 circuit names -> the simulator's circuit names (None: not in the model).
CIRCUIT_MAP = {
    "Melbourne": "melbourne", "Shanghai": "shanghai", "Suzuka": "suzuka", "Sakhir": "bahrain",
    "Jeddah": "jeddah", "Miami": "miami", "Montreal": "montreal", "Monte Carlo": "monaco",
    "Catalunya": "barcelona", "Spielberg": "spielberg", "Silverstone": "silverstone",
    "Spa-Francorchamps": "spa", "Hungaroring": "hungaroring", "Zandvoort": "zandvoort", "Monza": "monza",
    "Madring": "madrid", "Baku": "baku", "Singapore": "singapore", "Austin": "austin",
    "Mexico City": "mexico_city", "Interlagos": "interlagos", "Las Vegas": "las_vegas", "Lusail": "lusail",
    "Yas Marina Circuit": "yas_marina", "Kuala Lumpur": None,
}
# Gap-bin edges, s: 0.25 s wide up to 1.5 s, with an edge at 1 s (the Overtake Mode detection gap), then
# 1.5-2 s and 2-3 s, where passes are rare. The simulator samples gaps of 0.05-2 s, so 2-3 s is real data only.
GAP_BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def _times(values):
    """Parse OpenF1 timestamps to timezone-naive UTC (so every comparison uses one convention).

    ``format="ISO8601"`` accepts the mix of ISO 8601 variants OpenF1 returns;
    comparing timezone-aware with naive times would raise. Returns a
    `pandas.Series` of naive UTC times in input order.
    """
    return pd.to_datetime(pd.Series(values), utc=True, format="ISO8601").dt.tz_localize(None)


def lap_index(laps):
    """Start time of every lap of every driver, for looking up which lap a moment falls in.

    Parameters
    ----------
    laps : list of dict
        OpenF1 ``laps`` rows.

    Returns
    -------
    dict of int -> pandas.DataFrame
        Per driver: ``lap_number``, ``start`` (naive UTC), ``pit_out`` (bool), sorted
        by lap. Lap 1 has no recorded start (standing start) and is left out, and
        a driver with no recorded lap start has no entry.
    """
    df = pd.DataFrame(laps)
    df = df[df["date_start"].notna()].copy()
    df["start"] = _times(df["date_start"]).values
    out = {}
    for driver, rows in df.groupby("driver_number"):
        out[int(driver)] = rows.sort_values("lap_number")[["lap_number", "start", "is_pit_out_lap"]].rename(
            columns={"is_pit_out_lap": "pit_out"}).reset_index(drop=True)
    return out


def lap_at(index, driver, when):
    """Lap a driver was on at a given moment.

    Parameters
    ----------
    index : dict of int -> pandas.DataFrame
        Output of `lap_index`.
    driver : int
        Driver number.
    when : pandas.Timestamp or numpy.datetime64
        Moment to look up, timezone-naive UTC (as from `_times`).

    Returns
    -------
    int or None
        Lap number; 1 if ``when`` is before the driver's first recorded lap
        start (normally lap 2's, as lap 1 has none); None if the driver has no
        lap data.
    """
    rows = index.get(int(driver))
    if rows is None or rows.empty:
        return None
    # Index of the last lap that started at or before `when`; -1 means none has, so still lap 1.
    i = rows["start"].searchsorted(when, side="right") - 1
    return 1 if i < 0 else int(rows["lap_number"].iloc[i])


# Race-control messages that start a period without normal racing, and what kind it is.
NEUTRALISING = {"OVERTAKE DISABLED": "overtake mode", "SAFETY CAR DEPLOYED": "safety car",
                "VSC DEPLOYED": "safety car", "RED FLAG - RACE SUSPENDED": "red flag"}


def neutralised_periods(race_control):
    """Time intervals without normal racing with Overtake Mode available, from race-control messages.

    A period starts with OVERTAKE DISABLED, SAFETY CAR DEPLOYED, VSC DEPLOYED
    or RED FLAG and lasts until every such cause has ended: OVERTAKE ENABLED
    ends all of them; TRACK CLEAR also ends a (virtual) safety car, but not a
    red flag, after which the race restarts from the grid. Both kinds of
    message are needed because race control does not announce every safety
    car with OVERTAKE DISABLED.

    Parameters
    ----------
    race_control : list of dict
        OpenF1 ``race_control`` rows for one race.

    Returns
    -------
    list of (pandas.Timestamp, pandas.Timestamp)
        Start and end (naive UTC), in time order; the end is
        ``pandas.Timestamp.max`` if the race finished neutralised.
    """
    rows = sorted(race_control, key=lambda m: m["date"])
    times = _times([m["date"] for m in rows])
    # `active` holds the kinds of neutralisation in force; a period lasts while it is non-empty.
    active, periods, start = set(), [], None
    for m, when in zip(rows, times):
        message = m.get("message") or ""
        if message in NEUTRALISING:
            active.add(NEUTRALISING[message])
        elif message == "OVERTAKE ENABLED":
            active.clear()
        elif message == "TRACK CLEAR":
            active.discard("safety car")
        if active and start is None:
            start = when
        elif not active and start is not None:
            periods.append((start, when))
            start = None
    if start is not None:
        periods.append((start, pd.Timestamp.max))
    return periods


def _pit_laps(pits):
    """Map driver number -> set of laps on which OpenF1 records a pit stop (rows with no lap are skipped)."""
    out = {}
    for p in pits:
        if p.get("lap_number") is not None:
            out.setdefault(int(p["driver_number"]), set()).add(int(p["lap_number"]))
    return out


def classify_overtakes(overtakes, laps, pits, race_control=()):
    """Label every position change with why it happened.

    OpenF1's overtakes are position changes between two cars. Each gets the
    first reason that applies, judged on the overtaking car's lap at the moment
    of the change (`lap_at`):

    * ``"start"``: lap 1, where positions shuffle from a standing start;
    * ``"neutralised"``: under a safety car, virtual safety car or red flag, or
      while race control had disabled Overtake Mode (`neutralised_periods`);
    * ``"pit stop"``: either car pitted on that lap or the lap before, or is
      on a pit-out lap (one car stopping for tyres is not an overtake);
    * ``"on track"``: everything else.

    Not identifiable from the data, so left as on-track passes: passes on cars
    slowing with a problem. A change whose lap is unknown (the overtaking car
    has no lap data) is ``"start"`` only in a race where no change's lap is
    known; otherwise its lap is NaN, which passes neither lap test, so it ends
    up ``"neutralised"`` or ``"on track"``.

    Parameters
    ----------
    overtakes, laps, pits : list of dict
        OpenF1 ``overtakes``, ``laps`` and ``pit`` rows for one race.
    race_control : sequence of dict, optional
        OpenF1 ``race_control`` rows; empty (the default) means no neutralised periods.

    Returns
    -------
    pandas.DataFrame
        The overtakes with ``time`` (naive UTC), ``lap`` (the overtaking car's
        lap) and ``reason`` (one of the four labels) added. With no overtakes,
        an empty table that still has those three columns.
    """
    df = pd.DataFrame(overtakes)
    # Keep the output columns for a race with no overtakes, so callers can still filter on reason.
    if df.empty:
        return df.assign(time=pd.Series(dtype="datetime64[ns]"), lap=pd.Series(dtype=int),
                         reason=pd.Series(dtype=str))
    index, pit_laps = lap_index(laps), _pit_laps(pits)
    periods = neutralised_periods(race_control)
    df["time"] = _times(df["date"]).values
    # The overtaking car's lap at the moment of the change; the pit checks below use it for both cars.
    df["lap"] = [lap_at(index, d, t) for d, t in zip(df["overtaking_driver_number"], df["time"])]

    def pit_related(driver, lap):
        """True if ``driver`` pitted on ``lap`` or ``lap - 1``, or ``lap`` is their pit-out lap."""
        rows = index.get(int(driver))
        out_lap = rows is not None and bool(rows.loc[rows["lap_number"] == lap, "pit_out"].any())
        return out_lap or bool(pit_laps.get(int(driver), set()) & {lap, lap - 1})

    def reason(a, b, lap, when):
        """First label that applies to a change by car ``a`` on car ``b`` on ``lap`` at ``when``."""
        # None: unknown lap. Mixed with known laps the column is float and an unknown lap is NaN,
        # which neither test catches (see `classify_overtakes`).
        if lap is None or lap <= 1:
            return "start"
        if any(lo <= when < hi for lo, hi in periods):
            return "neutralised"
        if pit_related(a, lap) or pit_related(b, lap):
            return "pit stop"
        return "on track"

    df["reason"] = [reason(a, b, lap, t) for a, b, lap, t in
                    zip(df["overtaking_driver_number"], df["overtaken_driver_number"], df["lap"], df["time"])]
    return df


def on_track_overtakes(overtakes, laps, pits, race_control=()):
    """Only the position changes that look like real on-track passes (see `classify_overtakes`).

    Parameters
    ----------
    overtakes, laps, pits : list of dict
        OpenF1 ``overtakes``, ``laps`` and ``pit`` rows for one race.
    race_control : sequence of dict, optional
        OpenF1 ``race_control`` rows; empty (the default) means no neutralised periods.

    Returns
    -------
    pandas.DataFrame
        The overtakes labelled ``"on track"``, with ``time``, ``lap`` and
        ``reason`` columns added and the index reset.
    """
    df = classify_overtakes(overtakes, laps, pits, race_control)
    return df[df["reason"] == "on track"].reset_index(drop=True)


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion (Wilson, 1927).

    Unlike the normal (Wald) interval it behaves well for small ``n`` and for
    rates near 0 or 1: it stays inside [0, 1] (up to rounding) and is not zero
    width when there are no successes.

    Parameters
    ----------
    successes : int
        Number of successes (here passes), 0 <= successes <= n.
    n : int
        Number of trials (here driver-laps).
    z : float, optional
        Standard-normal quantile; the default 1.96 gives a 95 % interval.

    Returns
    -------
    (low, high) : tuple of float
        Bounds as proportions; ``(nan, nan)`` if ``n == 0``.

    Examples
    --------
    >>> [round(float(b), 4) for b in wilson_interval(5, 10)]
    [0.2366, 0.7634]
    """
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    # Roots in p of (p_hat - p)^2 = z^2 p (1 - p) / n; the centre moves from p_hat towards 1/2.
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def laps_by_gap(laps, intervals, pits, passes, race_control=()):
    """One row per driver-lap: gap to the car ahead at the start of the lap, and whether the driver passed.

    The gap is the ``interval`` sample nearest the lap's start (within 10 s);
    the lap counts as a pass if the driver made an on-track overtake before
    their next lap began. Lap 1, pit in/out laps, laps that overlap a
    neutralised period (`neutralised_periods`), the race leader and lapped
    gaps ("+1 LAP") are left out. So is each driver's last recorded lap,
    whose end (the next lap's start) is unknown. A lap on which the car ahead
    pits stays in as a non-pass: only the driver's own pit laps are dropped.

    Parameters
    ----------
    laps, intervals, pits : list of dict
        OpenF1 ``laps``, ``intervals`` and ``pit`` rows for one race.
    passes : pandas.DataFrame
        On-track passes, e.g. from `on_track_overtakes`; only
        ``overtaking_driver_number`` and ``time`` (naive UTC) are used.
    race_control : sequence of dict, optional
        OpenF1 ``race_control`` rows; empty (the default) means no neutralised periods.

    Returns
    -------
    pandas.DataFrame
        Columns ``driver`` (int), ``lap`` (int), ``gap_s`` (gap to the car
        ahead, s) and ``passed`` (bool).

    Raises
    ------
    KeyError
        If ``intervals`` is empty (no ``interval`` column).
    """
    index = lap_index(laps)
    iv = pd.DataFrame(intervals)
    # The leader's interval is None and a lapped car's "+1 LAP": both become NaN and are dropped, as are gaps <= 0.
    iv["gap"] = pd.to_numeric(iv["interval"], errors="coerce")
    iv = iv[iv["gap"].notna() & (iv["gap"] > 0)].copy()
    iv["time"] = _times(iv["date"]).values
    pit_laps, periods = _pit_laps(pits), neutralised_periods(race_control)
    pass_times = {}
    for d, t in zip(passes.get("overtaking_driver_number", []), passes.get("time", [])):
        pass_times.setdefault(int(d), []).append(t)

    rows = []
    for driver, lap_rows in index.items():
        samples = iv[iv["driver_number"] == driver].sort_values("time")
        if samples.empty:
            continue
        times = samples["time"].to_numpy()
        mine = np.array(sorted(pass_times.get(driver, [])), dtype="datetime64[ns]")
        # Lap i runs from its start to lap i + 1's start, so the last recorded lap has no end and is skipped.
        for i in range(len(lap_rows) - 1):
            lap, start, end = int(lap_rows["lap_number"][i]), lap_rows["start"][i], lap_rows["start"][i + 1]
            if lap <= 1 or lap_rows["pit_out"][i] or lap in pit_laps.get(driver, set()):
                continue
            if any(lo < end and start < hi for lo, hi in periods):  # overlaps a neutralised period at all
                continue
            # Nearest interval sample to the lap start, on either side (samples come about every 4 s).
            j = np.searchsorted(times, start.to_datetime64())
            candidates = [k for k in (j - 1, j) if 0 <= k < len(times)]
            k = min(candidates, key=lambda k: abs(times[k] - start.to_datetime64()))
            if abs(times[k] - start.to_datetime64()) > np.timedelta64(10, "s"):  # 10 s: estimate
                continue
            # A pass counts for this lap if it falls in [start, end).
            passed = bool(((mine >= start.to_datetime64()) & (mine < end.to_datetime64())).any())
            rows.append((driver, lap, float(samples["gap"].iloc[k]), passed))
    return pd.DataFrame(rows, columns=["driver", "lap", "gap_s", "passed"])


def rate_by_gap(gap_s, passed, bins=GAP_BINS):
    """Pass rate (with 95 % Wilson interval) in each gap bin.

    Parameters
    ----------
    gap_s : array_like of float
        Gap to the car ahead at the start of each driver-lap, s.
    passed : array_like of bool
        Whether the driver made an on-track pass on that lap.
    bins : sequence of float, optional
        Increasing bin edges, s. Bins are half-open, [low, high); gaps outside
        [bins[0], bins[-1]) are not counted.

    Returns
    -------
    list of dict
        One per bin: ``bin`` (label, e.g. "0.25-0.5"), ``low``/``high`` (edges, s),
        ``n`` (driver-laps), ``passes``, ``rate`` (passes / n) and ``ci`` ([low, high]
        as proportions, `wilson_interval`). For an empty bin ``rate`` is None and
        ``ci`` is ``[None, None]``, which become JSON null (NaN is not valid JSON).
    """
    gap_s, passed = np.asarray(gap_s, dtype=float), np.asarray(passed, dtype=bool)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (gap_s >= lo) & (gap_s < hi)
        n, k = int(mask.sum()), int(passed[mask].sum())
        out.append({"bin": f"{lo:g}-{hi:g}", "low": lo, "high": hi, "n": n, "passes": k,
                    "rate": k / n if n else None, "ci": list(wilson_interval(k, n)) if n else [None, None]})
    return out


def full_throttle_runs(car_data, min_seconds=6.0, min_gain_kph=80):
    """Stretches of telemetry where the car accelerates flat out (straights).

    A run is a sequence of samples with throttle >= 99 % and no brake, lasting
    at least ``min_seconds`` and gaining at least ``min_gain_kph``. Distance
    is the trapezoidal integral of speed over time.

    Parameters
    ----------
    car_data : list of dict
        OpenF1 ``car_data`` rows for one car: ``date`` (ISO 8601), ``speed``
        (km/h), ``throttle`` (%), ``brake`` (0 or 100). Any order; sorted by
        time here.
    min_seconds : float, optional
        Shortest run kept, s, from its first sample to its last.
    min_gain_kph : float, optional
        Smallest rise from the run's first speed to its peak, km/h.

    Returns
    -------
    list of pandas.DataFrame
        Per run, in time order: ``t`` (s from run start), ``x`` (m from run
        start), ``kph``.

    Raises
    ------
    KeyError
        If ``car_data`` is empty (no ``date`` column).
    """
    df = pd.DataFrame(car_data)
    df["time"] = _times(df["date"]).values
    df = df.sort_values("time").reset_index(drop=True)
    flat = (df["throttle"] >= 99) & (df["brake"] == 0)  # 99 % rather than exactly 100 % (estimate)
    runs, start = [], None
    # A trailing False closes a run that lasts to the end of the data.
    for i, on in enumerate(list(flat) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            seg = df.iloc[start:i]
            t = (seg["time"] - seg["time"].iloc[0]).dt.total_seconds().to_numpy()
            v = seg["speed"].to_numpy(dtype=float) / 3.6  # km/h -> m/s for the distance integral
            if t[-1] >= min_seconds and (v.max() - v[0]) * 3.6 >= min_gain_kph:
                # Cumulative trapezoidal rule: distance from the run start, m.
                x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
                runs.append(pd.DataFrame({"t": t, "x": x, "kph": v * 3.6}))
            start = None
    return runs


def simulate_run(v0_kph, duration_s, deploy_share, overtake_mode=False, battery=1.0, dt=0.05):
    """Speed of one car accelerating from ``v0_kph`` for ``duration_s``, from the simulator's physics.

    Integrates the one-car ODE with the simulator's force model
    (`overtake.physics.acceleration`) and fixed-step RK4
    (`overtake.physics.rk4_step`): new soft tyre (``mu_reference``), dry, no
    slipstream, no braking.

    Parameters
    ----------
    v0_kph : float
        Starting speed, km/h.
    duration_s : float
        Length of the run, s; rounded to a whole number of steps.
    deploy_share : float
        Share of the MGU-K limit used: 1 with Boost,
        `overtake.simulator.STANDARD_MAP_SHARE` (0.55) on the standard map, 0
        with no electrical power.
    overtake_mode : bool, optional
        Overtake Mode: the MGU-K taper starts at 337 km/h instead of 290 km/h,
        and 0.5 MJ of extra energy (`overtake.physics.OVERTAKE_MODE_EXTRA_J`).
    battery : float, optional
        Starting charge as a fraction (0-1) of the usable 4 MJ
        (`overtake.physics.BATTERY_USABLE_J`), not a percentage as in the
        simulator's scenarios.
    dt : float, optional
        RK4 step, s; the default equals the simulator's ``TIME_STEP_S``.

    Returns
    -------
    pandas.DataFrame
        One row per step from t = 0 to the rounded duration: ``t`` (s), ``x``
        (distance, m), ``kph`` (speed, km/h).
    """
    car = physics.CarParameters()
    # One scenario: `physics` works on arrays with one entry per scenario.
    state = physics.CarState(mu=np.array([car.mu_reference]), cda_m2=np.array([car.cda_m2]),
                             deploy_share=np.array([float(deploy_share)]),
                             energy_j=np.array([physics.BATTERY_USABLE_J * battery
                                                + physics.OVERTAKE_MODE_EXTRA_J * overtake_mode]),
                             overtake_mode=np.array([overtake_mode]))

    def f(y):
        """Right-hand side for y = [x (m), v (m/s), energy used (J)]: dy/dt = [v, a, MGU-K power]."""
        # One-element slices in, first elements out, as `physics.acceleration` works on arrays.
        accel, ers = physics.acceleration(y[1:2], y[2:3], state, state.cda_m2, car)
        return np.array([y[1], accel[0], ers[0]])

    y, rows = np.array([0.0, v0_kph / 3.6, 0.0]), []  # start at x = 0, km/h -> m/s, no energy used
    # Record the state, then step; the step taken after the last row is not used.
    for i in range(int(round(duration_s / dt)) + 1):
        rows.append((i * dt, y[0], y[1] * 3.6))
        y = physics.rk4_step(f, y, dt)
    return pd.DataFrame(rows, columns=["t", "x", "kph"])


def analyse(log=print):
    """Run every comparison on the OpenF1 data and return the results.

    Reads the cache; `fetch` downloads anything missing. Steps:

    1. Classify each race's position changes (start, neutralised, pit stop,
       on track) and keep the on-track passes.
    2. Pass rate per lap against the gap at the start of the lap, real and
       simulated (the simulator's mean true probability per attempt over its
       own scenario distribution). Both are also expressed relative to the
       0.25-0.5 s bin, because a per-lap rate and a per-attempt probability
       differ in level but can be compared in shape.
    3. Real on-track passes per race lap against the simulator's circuit
       passability (Spearman rank correlation).
    4. Speed-trap readings per race against 355 km/h, where the 2026 MGU-K
       power taper reaches zero (`overtake.physics.TAPER_STANDARD_KPH`).
    5. Full-throttle stretches of one car's telemetry against the RK4 car
       model started from the same speed, with full Boost, the standard map
       and no electric power.

    Parameters
    ----------
    log : callable, optional
        Receives one progress line per race with lap data (default `print`).

    Returns
    -------
    dict
        Written to ``reports/results/real_data.json`` by `overtake.experiments.real`:

        * ``source`` (str): data source and filter.
        * ``races`` (list of dict): per race with lap data, ``race`` (OpenF1
          circuit name), ``circuit`` (simulator name, None if not modelled),
          ``position_changes`` and its split into ``start``, ``neutralised``,
          ``pit_stop`` and ``on_track``, ``race_laps``, ``on_track_per_lap``,
          and ``passability`` for modelled circuits.
        * ``gap`` (dict): ``driver_laps``, ``passes``, ``real`` (`rate_by_gap`
          rows plus ``relative``, the rate divided by the 0.25-0.5 s bin's) and
          ``simulated`` (per bin ``n``, ``mean_p``, the mean exact probability,
          and ``relative`` on the same basis; both None beyond the simulated 2 s).
        * ``circuits`` (dict): ``spearman_rho``, ``p_value``, ``n``, and the
          same two statistics without Monaco (``without_monaco``).
        * ``speed_traps`` (list of dict): per race, ``laps``, ``median``,
          ``p99`` and ``max`` of each lap's fastest speed-trap reading (km/h),
          and ``share_over_355``.
        * ``telemetry`` (dict): ``car`` (`TELEMETRY`), ``runs``, ``best_fit``
          (runs each variant fits best), ``median_rms_kph`` and ``rms_kph``
          (per run, km/h), a ``longest_run`` summary, and ``traces`` (``t``,
          ``x``, ``kph`` lists for the longest real run and each model variant).

    Raises
    ------
    urllib.error.URLError
        If data is not cached and OpenF1 cannot be reached (see `fetch`).
    StopIteration
        If the ``TELEMETRY["session"]`` race is not among the completed races.
    ValueError
        If no race has lap data, or the telemetry has no full-throttle run.
    KeyError
        If a race with lap data has an empty ``intervals`` response, or the
        telemetry response is empty.

    Examples
    --------
    >>> result = analyse(log=lambda line: None)   # doctest: +SKIP
    >>> rho = result["circuits"]["spearman_rho"]  # doctest: +SKIP
    """
    from scipy.stats import spearmanr

    from . import simulator

    races, gap_rows, speed_rows = [], [], []
    for race in completed_races():
        key = race["session_key"]
        laps = fetch("laps", session_key=key)
        if not laps:
            continue
        overtakes, pits = fetch("overtakes", session_key=key), fetch("pit", session_key=key)
        control = fetch("race_control", session_key=key)
        labelled = classify_overtakes(overtakes, laps, pits, control)
        passes = labelled[labelled["reason"] == "on track"]
        counts = labelled["reason"].value_counts()
        race_laps = max(l["lap_number"] for l in laps)  # laps completed by the winner
        races.append({"race": race["circuit_short_name"], "circuit": CIRCUIT_MAP.get(race["circuit_short_name"]),
                      "position_changes": len(overtakes), "start": int(counts.get("start", 0)),
                      "neutralised": int(counts.get("neutralised", 0)), "pit_stop": int(counts.get("pit stop", 0)),
                      "on_track": len(passes), "race_laps": race_laps, "on_track_per_lap": len(passes) / race_laps})
        table = laps_by_gap(laps, fetch("intervals", session_key=key), pits, passes, control)
        gap_rows.append(table)
        # Fastest of each lap's three speed readings (i1, i2: intermediates; st: speed trap), km/h, laps
        # with none dropped; compared with 355 km/h, where MGU-K power reaches zero (physics.TAPER_STANDARD_KPH).
        top = pd.DataFrame(laps)[["i1_speed", "i2_speed", "st_speed"]].max(axis=1).dropna()
        speed_rows.append({"race": race["circuit_short_name"], "laps": int(len(top)), "median": float(top.median()),
                           "p99": float(top.quantile(0.99)), "max": float(top.max()),
                           "share_over_355": float((top > 355).mean())})
        log(f"  {race['circuit_short_name']:18s} {len(passes):4d} on-track passes, {len(table):5d} driver-laps")

    # Gap: real per-lap rate vs the simulator's mean probability per attempt.
    gaps = pd.concat(gap_rows, ignore_index=True)
    real_gap = rate_by_gap(gaps["gap_s"], gaps["passed"])
    # 300,000 attempts: over 30,000 in every bin below 2 s, so each bin's mean probability has a standard
    # error below 0.003 (p lies in [0, 1], so its SD is at most 0.5).
    sim = simulator.generate_dataset(300_000, seed=0)
    sim_gap = []
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        mask = (sim["gap_s"] >= lo) & (sim["gap_s"] < hi)
        sim_gap.append({"bin": f"{lo:g}-{hi:g}", "low": lo, "high": hi, "n": int(mask.sum()),
                        "mean_p": float(sim.loc[mask, "p_true"].mean()) if mask.any() else None})
    # Reference: index 1, the 0.25-0.5 s bin. The simulator samples gaps from 0.05 s, so its 0-0.25 s bin
    # is only partly covered. Every real bin has driver-laps in the shipped data; an empty one (rate None)
    # would make the division below fail.
    ref_real, ref_sim = real_gap[1]["rate"], sim_gap[1]["mean_p"]
    for r, q in zip(real_gap, sim_gap):
        r["relative"] = r["rate"] / ref_real
        q["relative"] = q["mean_p"] / ref_sim if q["n"] else None  # gaps beyond the simulated range

    # Circuits: real passes per lap vs the simulator's passability.
    modelled = [r for r in races if r["circuit"]]
    for r in modelled:
        r["passability"] = simulator.PASSABILITY[r["circuit"]]
    rho, p_value = spearmanr([r["on_track_per_lap"] for r in modelled], [r["passability"] for r in modelled])
    # Monaco is an outlier on both scales; check the agreement does not rest on it alone.
    others = [r for r in modelled if r["circuit"] != "monaco"]
    rho_others, p_others = spearmanr([r["on_track_per_lap"] for r in others], [r["passability"] for r in others])

    # Telemetry: real full-throttle stretches vs the RK4 car model.
    monza = next(r for r in completed_races() if r["circuit_short_name"] == TELEMETRY["session"])
    runs = full_throttle_runs(fetch("car_data", session_key=monza["session_key"],
                                    driver_number=TELEMETRY["driver_number"]))
    variants = {"Boost": 1.0, "Standard map": 0.55, "No electric power": 0.0}  # 0.55: simulator.STANDARD_MAP_SHARE
    errors = {name: [] for name in variants}
    best = {name: 0 for name in variants}
    for run in runs:
        rms = {}
        for name, share in variants.items():
            # Model run from the real run's first speed for its duration, full battery, no Overtake Mode.
            model = simulate_run(run["kph"].iloc[0], run["t"].iloc[-1], share)
            # RMS speed error, km/h, with the model speed interpolated to the telemetry sample times.
            rms[name] = float(np.sqrt(np.mean((np.interp(run["t"], model["t"], model["kph"]) - run["kph"]) ** 2)))
            errors[name].append(rms[name])
        best[min(rms, key=rms.get)] += 1  # credit the variant with the smallest error on this run
    longest = max(runs, key=lambda r: r["x"].iloc[-1])  # longest by distance: the example trace for the figure
    traces = {"real": longest[["t", "x", "kph"]].to_dict("list")}
    for name, share in variants.items():
        traces[name] = simulate_run(longest["kph"].iloc[0], longest["t"].iloc[-1], share).to_dict("list")

    return {
        "source": "OpenF1 (https://openf1.org), 2026 races; position changes filtered to on-track passes",
        "races": races,
        "gap": {"driver_laps": int(len(gaps)), "passes": int(gaps["passed"].sum()), "real": real_gap,
                "simulated": sim_gap},
        "circuits": {"spearman_rho": float(rho), "p_value": float(p_value), "n": len(modelled),
                     "without_monaco": {"spearman_rho": float(rho_others), "p_value": float(p_others)}},
        "speed_traps": speed_rows,
        "telemetry": {"car": TELEMETRY, "runs": len(runs), "best_fit": best,
                      "median_rms_kph": {k: float(np.median(v)) for k, v in errors.items()},
                      "rms_kph": errors,
                      "longest_run": {"length_m": float(longest["x"].iloc[-1]),
                                      "duration_s": float(longest["t"].iloc[-1]),
                                      "start_kph": float(longest["kph"].iloc[0]),
                                      "peak_kph": float(longest["kph"].max()),
                                      "end_kph": float(longest["kph"].iloc[-1]),
                                      "model_end_kph": {k: float(v["kph"][-1]) for k, v in traces.items()
                                                        if k != "real"}},
                      "traces": traces},
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download()
    else:
        print(__doc__)
