"""Offline checks on the real-data comparison, using small hand-made OpenF1-style
rows (no network): lap lookup, neutralised periods, the on-track-pass filter,
gap binning, the confidence interval, telemetry run detection and the cache."""
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from overtake import realdata

START = pd.Timestamp("2026-09-07T13:00:00+00:00")
LAP_S = 90.0
OFFSET = {1: 0.0, 2: 0.5, 3: 1.5}  # drivers cross the line in order 1, 2, 3


def iso(seconds):
    """ISO 8601 time ``seconds`` after the race start, as OpenF1 writes it."""
    return (START + pd.Timedelta(seconds=seconds)).isoformat()


def at(seconds):
    """The same moment as a timezone-naive UTC Timestamp, as realdata stores times."""
    return pd.Timestamp(iso(seconds)).tz_localize(None)


def lap_start(driver, lap):
    """Start time (s after the race start) of ``lap`` for ``driver``, on 90 s laps."""
    return LAP_S * (lap - 1) + OFFSET[driver]


# Six laps for three cars; car 3 pits on lap 3, so lap 4 is its pit-out lap.
LAPS = [{"driver_number": d, "lap_number": n, "date_start": None if n == 1 else iso(lap_start(d, n)),
         "is_pit_out_lap": d == 3 and n == 4} for d in OFFSET for n in range(1, 7)]
PITS = [{"driver_number": 3, "lap_number": 3, "date": iso(lap_start(3, 3) + 80)}]
# Overtake Mode enabled shortly after the start; a safety car from 290 s to 330 s, during lap 4.
RACE_CONTROL = [{"date": iso(-60), "message": "OVERTAKE DISABLED"}, {"date": iso(30), "message": "OVERTAKE ENABLED"},
                {"date": iso(330), "message": "TRACK CLEAR"}, {"date": iso(290), "message": "SAFETY CAR DEPLOYED"}]
OVERTAKES = [
    {"overtaking_driver_number": 2, "overtaken_driver_number": 1, "date": iso(45)},   # lap 1
    {"overtaking_driver_number": 2, "overtaken_driver_number": 1, "date": iso(100)},  # on track
    {"overtaking_driver_number": 1, "overtaken_driver_number": 3, "date": iso(200)},  # car 3 pitting
    {"overtaking_driver_number": 2, "overtaken_driver_number": 1, "date": iso(310)},  # behind the safety car
    {"overtaking_driver_number": 3, "overtaken_driver_number": 2, "date": iso(340)},  # car 3's pit-out lap
    {"overtaking_driver_number": 1, "overtaken_driver_number": 2, "date": iso(380)},  # on track, lap 5
]


def interval(driver, seconds, gap):
    """One OpenF1 interval row: ``driver``'s gap to the car ahead at a moment."""
    return {"driver_number": driver, "date": iso(seconds), "interval": gap}


INTERVALS = (
    [interval(1, lap_start(1, n) + 1, 0.8) for n in range(2, 7)]
    + [interval(1, lap_start(1, n) + 2, None) for n in range(2, 7)]  # leader-style empty readings
    + [interval(2, lap_start(2, 2) + 1, 0.3), interval(2, lap_start(2, 3) + 1, "+1 LAP"),
       interval(2, lap_start(2, 4) + 30, 0.3), interval(2, lap_start(2, 5) + 1, 0.3)]
    + [interval(3, lap_start(3, n) + 1, 1.5) for n in range(2, 7)]
)


def test_lap_at_finds_the_lap_a_moment_falls_in():
    """A moment belongs to the lap that started last before it; an unknown driver gives None."""
    index = realdata.lap_index(LAPS)
    assert realdata.lap_at(index, 1, at(45)) == 1  # before lap 2 starts
    assert realdata.lap_at(index, 2, at(lap_start(2, 3))) == 3
    assert realdata.lap_at(index, 2, at(lap_start(2, 3) - 0.1)) == 2
    assert realdata.lap_at(index, 99, at(100)) is None


def test_neutralised_periods_from_race_control():
    """Overtake-disabled and safety-car periods are read from race control, whatever the message order."""
    assert realdata.neutralised_periods(RACE_CONTROL) == [(at(-60), at(30)), (at(290), at(330))]


def test_red_flag_lasts_until_overtake_mode_returns_and_open_periods_run_to_the_end():
    """A red flag ends only at OVERTAKE ENABLED, and a period still open at the end runs to the race end."""
    messages = [{"date": iso(0), "message": "RED FLAG - RACE SUSPENDED"},
                {"date": iso(100), "message": "TRACK CLEAR"},           # not the restart
                {"date": iso(900), "message": "OVERTAKE ENABLED"},
                {"date": iso(1000), "message": "VSC DEPLOYED"}]         # race ends under it
    assert realdata.neutralised_periods(messages) == [(at(0), at(900)), (at(1000), pd.Timestamp.max)]


def test_every_position_change_gets_a_reason():
    """Each position change is labelled start, neutralised, pit stop or on track; only on-track ones are kept."""
    labelled = realdata.classify_overtakes(OVERTAKES, LAPS, PITS, RACE_CONTROL)
    assert labelled["reason"].tolist() == ["start", "on track", "pit stop", "neutralised", "pit stop", "on track"]
    kept = realdata.on_track_overtakes(OVERTAKES, LAPS, PITS, RACE_CONTROL)
    assert kept["lap"].tolist() == [2, 5]
    assert kept["overtaking_driver_number"].tolist() == [2, 1]


def test_no_overtakes_gives_an_empty_table():
    """A race without position changes gives an empty table with the usual columns, not an error."""
    kept = realdata.on_track_overtakes([], LAPS, PITS)
    assert kept.empty and {"time", "lap", "reason"} <= set(kept.columns)


def test_driver_laps_get_the_gap_at_lap_start_and_whether_they_passed():
    """Each eligible driver-lap gets its lap-start gap and a pass flag; lap 1, pit and neutralised laps drop out."""
    passes = realdata.on_track_overtakes(OVERTAKES, LAPS, PITS, RACE_CONTROL)
    table = realdata.laps_by_gap(LAPS, INTERVALS, PITS, passes, RACE_CONTROL)
    got = set(table.itertuples(index=False, name=None))
    assert got == {
        (1, 2, 0.8, False), (1, 3, 0.8, False), (1, 5, 0.8, True),  # lap 4 had the safety car
        (2, 2, 0.3, True), (2, 5, 0.3, False),  # lap 3 lapped, lap 4 no reading within 10 s
        (3, 2, 1.5, False), (3, 5, 1.5, False),  # laps 3 and 4 are the pit stop
    }


def test_wilson_interval_known_value_and_zero_successes():
    """The Wilson interval matches a textbook value and stays inside [0, 1] with no successes."""
    low, high = realdata.wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    low, high = realdata.wilson_interval(0, 10)
    assert low == pytest.approx(0, abs=1e-12) and 0 < high < 0.35


def test_rate_by_gap_bins_and_counts():
    """Pass rates are counted per gap bin, empty bins have no rate, and each rate lies inside its interval."""
    rows = realdata.rate_by_gap([0.1, 0.2, 0.6, 2.5], [True, False, True, False], bins=[0, 0.5, 1, 3, 4])
    assert [r["n"] for r in rows] == [2, 1, 1, 0]
    assert [r["rate"] for r in rows] == [0.5, 1.0, 0.0, None]
    assert rows[3]["ci"] == [None, None]
    for r in rows[:3]:
        assert r["ci"][0] <= r["rate"] <= r["ci"][1]


def test_full_throttle_runs_finds_the_straight_and_integrates_distance():
    """A long full-throttle stretch is found, a short burst ignored, and distance integrated exactly."""
    t = np.arange(0, 20, 0.25)
    speed = np.where(t <= 10, 200 + 10 * t, 150.0)  # 10 s straight, 200 -> 300 km/h, then a corner
    throttle = np.where(t <= 10, 100, 40)
    throttle[(t > 14) & (t <= 17)] = 100  # a 3 s burst: too short to count
    car_data = [{"date": iso(s), "speed": v, "throttle": th, "brake": 0 if th == 100 else 100}
                for s, v, th in zip(t, speed, throttle)]
    runs = realdata.full_throttle_runs(car_data[::-1])  # order should not matter
    assert len(runs) == 1
    run = runs[0]
    assert run["t"].iloc[0] == 0 and run["t"].iloc[-1] == pytest.approx(10)
    # Speed rises linearly, so the trapezoidal rule is exact: mean speed x time.
    assert run["x"].iloc[-1] == pytest.approx((200 + 300) / 2 / 3.6 * 10)


def test_simulated_runs_order_by_electric_power_and_respect_the_limit():
    """More electrical power gives more speed, the 355 km/h cut-off holds, and an empty battery equals no deployment."""
    ends = {share: realdata.simulate_run(250, 15, share)["kph"].iloc[-1] for share in (1.0, 0.55, 0.0)}
    assert ends[1.0] > ends[0.55] > ends[0.0]
    assert realdata.simulate_run(250, 60, 1.0)["kph"].max() < 355
    empty = realdata.simulate_run(250, 10, 1.0, battery=0.0)["kph"]
    np.testing.assert_allclose(empty, realdata.simulate_run(250, 10, 0.0)["kph"])  # no energy, no boost


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Cache in a temporary folder; any network call fails the test."""
    calls = []

    def no_network(url, timeout=None):
        """Stand-in for urlopen: records the URL and answers 404, as OpenF1 does."""
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "No results found", None, None)

    monkeypatch.setattr(realdata, "CACHE", tmp_path)
    monkeypatch.setattr(realdata, "REQUEST_SPACING_S", 0.0)
    monkeypatch.setattr(realdata.urllib.request, "urlopen", no_network)
    return tmp_path, calls


def test_fetch_reads_the_cache_without_the_network(offline):
    """A cached response is returned without any request."""
    cache, calls = offline
    (cache / "laps_session_key-1.json").write_text(json.dumps([{"lap_number": 2}]))
    assert realdata.fetch("laps", session_key=1) == [{"lap_number": 2}]
    assert calls == []


def test_fetch_treats_404_as_no_data_and_caches_it(offline):
    """A 404 means no data: it returns an empty list and is cached, so the second call makes no request."""
    _, calls = offline
    assert realdata.fetch("laps", session_key=7) == []
    assert realdata.fetch("laps", session_key=7) == []
    assert len(calls) == 1 and "session_key=7" in calls[0]


def test_completed_races_keeps_past_grands_prix_only(offline):
    """Only Grand Prix races held before the cut-off date are kept, not sprints or future races."""
    cache, _ = offline
    sessions = [{"session_name": "Race", "date_start": "2026-03-08T04:00:00+00:00", "session_key": 1},
                {"session_name": "Sprint", "date_start": "2026-03-14T03:00:00+00:00", "session_key": 2},
                {"session_name": "Race", "date_start": "2026-12-06T13:00:00+00:00", "session_key": 3}]
    (cache / "sessions_session_type-Race_year-2026.json").write_text(json.dumps(sessions))
    assert [s["session_key"] for s in realdata.completed_races(before="2026-10-19")] == [1]
