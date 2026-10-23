"""Checks on the interactive calculator (`overtake.app`), using the trained model in models/."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from overtake.app import OTHER_GAPS, evaluate, make_handler
from overtake.features import DEFAULTS
from overtake.predict import OvertakePredictor


@pytest.fixture(scope="module")
def predictor():
    """The trained ensemble shipped in models/."""
    return OvertakePredictor.load()


def page_form(**changes):
    """The default scenario as the page sends it: every value a string, Overtake Mode on "auto"."""
    form = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in DEFAULTS.items()}
    form["overtake_mode"] = "auto"
    return {**form, **changes}


def test_page_inputs_give_the_same_answer_as_the_api(predictor):
    """Strings from the page are converted correctly: the default form predicts the default scenario."""
    result = evaluate(predictor, page_form())
    assert result["p"] == pytest.approx(predictor.predict(DEFAULTS))
    assert result["p_no_boost"] == pytest.approx(predictor.predict({**DEFAULTS, "attacker_boost": False}))
    assert [g for g, _ in result["gaps"]] == list(OTHER_GAPS)


def test_auto_overtake_mode_follows_the_gap(predictor):
    """On "auto", Overtake Mode is off beyond 1 s, as in the rules; "on" keeps it on (with a warning far out)."""
    auto = evaluate(predictor, page_form(gap_s="1.2"))
    manual = evaluate(predictor, page_form(gap_s="1.2", overtake_mode="on"))
    assert auto["p"] == pytest.approx(predictor.predict({**DEFAULTS, "gap_s": 1.2, "overtake_mode": False}))
    assert manual["p"] > auto["p"]


def test_invalid_input_is_reported_not_raised(predictor):
    """A bad value comes back as an error message naming the field, so the page can show it."""
    result = evaluate(predictor, page_form(circuit="imola"))
    assert "circuit" in result["error"]


def test_server_serves_the_page_and_predictions(predictor):
    """End to end over HTTP on a free local port: the page loads and /predict answers with a probability."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(predictor))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        html = urllib.request.urlopen(url).read().decode("utf-8")
        assert "Overtake calculator" in html and 'name="gap_s"' in html
        request = urllib.request.Request(url + "/predict", data=json.dumps(page_form()).encode("utf-8"), method="POST")
        result = json.loads(urllib.request.urlopen(request).read())
        assert 0 <= result["p"] <= 1
    finally:
        server.shutdown()
        server.server_close()
