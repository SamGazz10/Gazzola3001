"""Interactive calculator: set the inputs in a browser page and see P(overtake) update.

    python -m overtake.app              # opens http://127.0.0.1:8000 in your browser
    python -m overtake.app --port 8050 --no-browser

A small local web server (Python standard library only) serves one page with a
control for each of the 19 inputs. Every change sends the inputs to the
server, which asks the trained network (`overtake.predict.OvertakePredictor`)
and returns P(overtake), the same without Boost, the chance at other gaps and
any out-of-range warnings. The server only listens on this computer
(127.0.0.1); stop it with Ctrl+C.
"""
import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .features import BINARY, DEFAULTS, NUMERIC_RANGES, VOCABS
from .predict import DEFAULT_MODEL, OvertakePredictor

# Gaps (s) at which the page shows the same situation, as the command-line tool does.
OTHER_GAPS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)

# Slider settings per numeric input: label, step, and the range the network was trained on (for the hint).
SLIDERS = {
    "gap_s": ("Gap to the car ahead (s)", 0.05, "trained 0.05-2"),
    "run_length_m": ("Corner exit to braking point (m)", 10, "trained 250-1300 on straights, 50-300 into corners"),
    "attacker_battery": ("Attacker battery (% of 4 MJ)", 1, ""),
    "defender_battery": ("Defender battery (% of 4 MJ)", 1, ""),
    "attacker_tyre_age": ("Attacker tyre age (laps)", 1, "trained 0-40"),
    "defender_tyre_age": ("Defender tyre age (laps)", 1, "trained 0-40"),
    "track_temp_c": ("Track temperature (°C)", 1, "trained 18-55 dry, 10-32 damp or wet"),
    "attacker_skill": ("Attacker racecraft (grid about 60-100)", 1, "trained 60-100"),
    "defender_skill": ("Defender racecraft (grid about 60-100)", 1, "trained 60-100"),
    "top_speed_delta_kph": ("Attacker car top-speed advantage (km/h)", 1, "trained ±15"),
    "pace_delta_s": ("Attacker car lap-time advantage (s/lap)", 0.1, "trained ±1.5"),
}
SELECTS = {
    "track_condition": ("Track", "condition"),
    "circuit": ("Circuit", "circuit"),
    "segment": ("Where the move completes", "segment"),
    "attacker_tyre": ("Attacker tyre", "tyre"),
    "defender_tyre": ("Defender tyre", "tyre"),
}


def evaluate(predictor, form):
    """Predict for the inputs sent by the page.

    Parameters
    ----------
    predictor : overtake.predict.OvertakePredictor
    form : dict
        One value per schema field, as the page sends them (numbers may be
        strings), with ``overtake_mode`` one of "auto", "on" or "off";
        "auto" means on when the gap is 1 s or less, as in the rules.

    Returns
    -------
    dict
        ``p`` and ``p_no_boost`` (probabilities), ``gaps`` (list of
        ``[gap_s, p]``), ``warnings`` (list of str); or ``error`` (str) if an
        input is invalid.

    Examples
    --------
    >>> evaluate(OvertakePredictor.load(), {"gap_s": "1.0", "overtake_mode": "auto"})["p"]  # doctest: +SKIP
    0.44
    """
    try:
        scenario = {}
        for key, value in form.items():
            if key == "overtake_mode":
                continue
            if key in BINARY:
                scenario[key] = value in (True, "true", "on", 1, "1")
            elif key in NUMERIC_RANGES:
                scenario[key] = float(value)
            else:
                scenario[key] = value
        mode = form.get("overtake_mode", "auto")
        gap = scenario.get("gap_s", DEFAULTS["gap_s"])

        def with_gap(g):
            """The scenario at gap ``g``, Overtake Mode following the gap unless it was set by hand."""
            return {**scenario, "gap_s": g, "overtake_mode": g <= 1.0 if mode == "auto" else mode == "on"}

        current = with_gap(gap)
        return {
            "p": predictor.predict(current),
            "p_no_boost": predictor.predict({**current, "attacker_boost": False}),
            "gaps": [[g, predictor.predict(with_gap(g))] for g in OTHER_GAPS],
            "warnings": predictor.out_of_range(current),
        }
    except ValueError as error:
        return {"error": str(error)}


def page():
    """The HTML page, with one control per input built from the schema.

    Returns
    -------
    str
    """
    selects = "".join(
        f'<label>{label}<select name="{key}">'
        + "".join(f'<option{" selected" if v == DEFAULTS[key] else ""}>{v}</option>' for v in VOCABS[vocab])
        + "</select></label>"
        for key, (label, vocab) in SELECTS.items())
    sliders = "".join(
        f'<label>{label} <b id="{key}_v">{DEFAULTS[key]:g}</b>'
        f'<input type="range" name="{key}" min="{NUMERIC_RANGES[key][0]:g}" max="{NUMERIC_RANGES[key][1]:g}" '
        f'step="{step}" value="{DEFAULTS[key]:g}"><small>{hint}</small></label>'
        for key, (label, step, hint) in SLIDERS.items())
    return PAGE.replace("{{SELECTS}}", selects).replace("{{SLIDERS}}", sliders)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overtake calculator</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f6f5f1;color:#1b1e23}
 main{max-width:1100px;margin:auto;padding:16px;display:grid;grid-template-columns:1fr 320px;gap:20px}
 h1{font-size:20px;margin:16px 16px 0;max-width:1100px}
 form{display:grid;grid-template-columns:1fr 1fr;gap:6px 18px}
 label{display:flex;flex-direction:column;font-size:13px;padding:4px 0}
 label small{color:#6b6f76}
 select,input[type=range]{margin-top:3px}
 .row{display:flex;gap:16px;grid-column:1/-1;font-size:13px}
 .row label{flex-direction:row;align-items:center;gap:6px}
 aside{background:#fff;border:1px solid #ddd9cf;border-radius:8px;padding:16px;align-self:start;position:sticky;top:12px}
 #p{font-size:48px;font-weight:600;margin:4px 0}
 table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
 td{border-top:1px solid #eee;padding:3px 0}
 #warn{color:#a3361f;font-size:12px}
 @media (max-width:760px){main{grid-template-columns:1fr}form{grid-template-columns:1fr}}
</style></head><body>
<h1>Overtake calculator <small style="font-weight:400;color:#6b6f76">neural network trained on the simulator</small></h1>
<main><form id="f">{{SELECTS}}{{SLIDERS}}
<div class="row"><label><input type="checkbox" name="attacker_boost" checked> Attacker uses Boost</label>
<label><input type="checkbox" name="defender_boost" checked> Defender uses Boost</label>
<label>Overtake Mode <select name="overtake_mode"><option value="auto">auto (gap ≤ 1 s)</option>
<option value="on">on</option><option value="off">off</option></select></label></div></form>
<aside><div>P(overtake)</div><div id="p">-</div><div id="nb"></div>
<table id="gaps"></table><p id="warn"></p></aside></main>
<script>
const f = document.getElementById("f");
async function update() {
  const data = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    data[el.name] = el.type === "checkbox" ? el.checked : el.value;
    const shown = document.getElementById(el.name + "_v");
    if (shown) shown.textContent = el.value;
  }
  const r = await (await fetch("/predict", {method: "POST", body: JSON.stringify(data)})).json();
  const pct = x => (100 * x).toFixed(1) + " %";
  if (r.error) { document.getElementById("warn").textContent = r.error; return; }
  document.getElementById("p").textContent = pct(r.p);
  document.getElementById("nb").textContent = "without Boost: " + pct(r.p_no_boost);
  document.getElementById("gaps").innerHTML = "<tr><td>Same situation at a gap of</td><td></td></tr>" +
    r.gaps.map(([g, p]) => `<tr><td>${g.toFixed(2)} s</td><td>${pct(p)}</td></tr>`).join("");
  document.getElementById("warn").textContent = r.warnings.length ? "Outside the training data: " + r.warnings.join("; ") : "";
}
// Same situational defaults as the command line: 150 m into a corner, 20 °C when damp or wet.
f.segment.addEventListener("change", () => { f.run_length_m.value = f.segment.value === "straight" ? 800 : 150; });
f.track_condition.addEventListener("change", () => { f.track_temp_c.value = f.track_condition.value === "dry" ? 35 : 20; });
f.addEventListener("input", update);
f.addEventListener("change", update);
update();
</script></body></html>"""


def make_handler(predictor):
    """A request handler class that serves the page and answers predictions with ``predictor``.

    Parameters
    ----------
    predictor : overtake.predict.OvertakePredictor

    Returns
    -------
    type
        A subclass of `http.server.BaseHTTPRequestHandler`.
    """
    html = page().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        """GET / returns the page; POST /predict takes the inputs as JSON and returns `evaluate`'s result."""

        def _send(self, body, content_type):
            """Write a 200 response with ``body`` (bytes)."""
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """Serve the page."""
            self._send(html, "text/html; charset=utf-8")

        def do_POST(self):
            """Answer one prediction request."""
            length = int(self.headers.get("Content-Length", 0))
            try:
                form = json.loads(self.rfile.read(length) or b"{}")
                result = evaluate(predictor, form)
            except json.JSONDecodeError:
                result = {"error": "request was not valid JSON"}
            self._send(json.dumps(result).encode("utf-8"), "application/json")

        def log_message(self, *args):
            """Keep the terminal quiet: one line per slider move is noise."""

    return Handler


def main(argv=None):
    """Start the calculator and open it in the default browser.

    Parameters
    ----------
    argv : list of str, optional
        ``--port``, ``--model`` and ``--no-browser``; ``sys.argv[1:]`` if omitted.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)
    try:
        predictor = OvertakePredictor.load(args.model)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(predictor))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Overtake calculator running at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
