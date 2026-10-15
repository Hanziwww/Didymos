"""icutwin on the GPU host (CPU only): Paper P's Fig. 7d, one teaching case's synthetic patient for the internal case
panel. It writes, as JSON to stdout:
* the history series before the decision point (MAP, heart rate, SpO2, lactate; last 48 h);
* the Forecast incidence curves;
* the What-if responses and contrasts;
* the Explain groups.

Twins only (the bank; ``toolkit.figures.case_figure``'s rule). The output is a synthetic patient derived from the
credentialed data: internal, never committed, shown only once the study lead clears it. Run without writing on the host,
the script read from stdin:

    ssh ... 'cd /root/autodl-tmp/icutwin && .venv/bin/python - --objective shock --case 0' \\
        < tools/server/paper_case_data.py > runs/paper/figures/internal/case.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="shock")
    ap.add_argument("--case", type=int, default=0)
    a = ap.parse_args(argv)
    from icutwin.toolkit import inputs
    from icutwin.toolkit.card import SERIES
    from icutwin.toolkit.session import Paths, Session

    s = Session(Paths.default(), backend="hf")
    c = json.loads((s.paths.out / "v5" / a.objective / f"case_{a.case}.json").read_text())
    fc = c["forecast"]
    p = fc["patient"]
    if p["kind"] != "twin":
        raise SystemExit("twins only")
    stay = inputs.load(s, p["ref"]).stay(s.vocab)
    h = float(fc["anchor"]["hours"])
    t0 = stay.icu_in
    series = {}
    for name, codes, unit in SERIES[:4]:
        pts = []
        for t, _r, code, num, _x in stay.timeline:
            th = (t - t0).total_seconds() / 3600
            if code in codes and num is not None and np.isfinite(num) and max(0.0, h - 48) <= th <= h:
                pts.append((th, float(num)))
        series[name] = {"unit": unit, "points": pts}
    wi = c.get("whatif") or {}
    out = {"objective": a.objective, "case": a.case, "dataset": c.get("dataset"), "hours": h, "series": series,
           "incidence": fc["outputs"].get("incidence"),
           "whatif": {k: {"settings": v.get("settings"), "responses": (v.get("outputs") or {}).get("responses")}
                      for k, v in wi.items() if isinstance(v, dict)},
           "explain": {"program": c["explain"]["settings"].get("program"),
                       "groups": c["explain"]["outputs"]["groups"][:10]},
           "internal": "a synthetic patient derived from credentialed data; not for release unless cleared"}
    json.dump(out, sys.stdout, default=float)


if __name__ == "__main__":
    main()
