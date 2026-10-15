"""Twin cards (``docs/toolkit.md``): one self-contained HTML page per toolkit result (inline CSS, SVG and script; no
network request, so a card opens offline on the approved host), and the teaching pages.

* ``render(result)``: the card of a Forecast, What-if, Explain, Similar patients, Cohort or Trial result: what was
  asked, the outputs as tables and charts, the evidence and caveats, the provenance (weights, gate and its scope).
* ``timeline``: a patient's history as small multiples on hours since ICU admission (MAP, heart rate, SpO2, lactate,
  creatinine, and a treatment lane: vasoactive infusions, IMV and CRRT records), the decision point marked; with
  ``after`` the course that followed (a twin's own continuation).
* ``teaching(case, mode)``: a teaching case (``toolkit.teach``), ``learner`` (history, risk sliders, a choice of
  action, a reveal, answers downloaded as JSON) or ``teacher`` (everything, with attributions and prompts);
  ``drill(cases)``: the prognosis drill with a calibration plot of the learner; ``blinded(patient, hours)``: a
  timeline alone (the clinician study's stimulus: nothing says whether the stay is real or synthetic).

Cards of real stays carry individual data: they stay on the approved host. Figures and teaching use twins.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

CSS = """
:root{--bg:#f7f7f5;--panel:#ffffff;--ink:#1d1f23;--muted:#5d6470;--line:#d9dce1;--accent:#2f6fde;--accent2:#d9822b;
--good:#2e8b57;--warn:#b5651d;--bad:#c0392b;--band:rgba(47,111,222,.12);--grid:#eceef1}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--panel:#1d2024;--ink:#e8eaed;
--muted:#a3a9b3;--line:#33373d;--accent:#6ea0ff;--accent2:#f0a35b;--good:#5cc28a;--warn:#e0a060;--bad:#ff7b6b;
--band:rgba(110,160,255,.16);--grid:#2a2d32}}
:root[data-theme="dark"]{--bg:#15171a;--panel:#1d2024;--ink:#e8eaed;--muted:#a3a9b3;--line:#33373d;--accent:#6ea0ff;
--accent2:#f0a35b;--good:#5cc28a;--warn:#e0a060;--bad:#ff7b6b;--band:rgba(110,160,255,.16);--grid:#2a2d32}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif}main{max-width:980px;margin:0 auto;padding:20px 16px 48px}
header h1{font-size:22px;margin:0 0 4px}.meta{color:var(--muted);margin:0 0 10px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
section h2{font-size:16px;margin:0 0 10px}section h3{font-size:14px;margin:14px 0 6px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:5px 8px;
border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12px}
td.num,th.num{text-align:right}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;
padding:1px 9px;margin:0 6px 6px 0;font-size:12px;color:var(--muted)}.badge.ok{border-color:var(--good);
color:var(--good)}.badge.warn{border-color:var(--warn);color:var(--warn)}.badge.bad{border-color:var(--bad);
color:var(--bad)}.banner{background:var(--band);border-left:4px solid var(--accent);padding:8px 12px;
border-radius:6px;margin:10px 0}ul.caveats li{margin:3px 0;color:var(--muted)}.small{font-size:12px;
color:var(--muted)}svg{max-width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px}
svg .axis{stroke:var(--line)}svg .grid{stroke:var(--grid)}.wrap{overflow-x:auto}
.hidden{display:none}button{font:inherit;background:var(--accent);color:#fff;border:0;border-radius:8px;
padding:7px 14px;cursor:pointer;margin:6px 8px 6px 0}button.secondary{background:transparent;color:var(--accent);
border:1px solid var(--accent)}label.q{display:block;margin:8px 0 2px}input[type=range]{width:min(420px,100%)}
textarea{width:100%;min-height:60px;font:inherit;background:var(--bg);color:var(--ink);border:1px solid var(--line);
border-radius:6px;padding:6px}.flag-reproduced{color:var(--good)}.flag-not_reproduced{color:var(--bad)}
.flag-unresolved,.flag-exploratory{color:var(--warn)}
"""

PALETTE = ("#2f6fde", "#d9822b", "#2e8b57", "#9b59b6", "#c0392b", "#7f8c8d")
VASO = ("angiotensin", "dobutamine", "dopamine", "epinephrine", "isoproterenol", "milrinone", "norepinephrine",
        "phenylephrine", "vasopressin")
SERIES = (("MAP", ("VITAL//map//arterial", "VITAL//map//cuff"), "mmHg"), ("Heart rate", ("VITAL//heart_rate",), "/min"),
          ("SpO2", ("VITAL//spo2",), "%"), ("Lactate", ("LAB//lactate//mmol/l//misc",), "mmol/L"),
          ("Creatinine", ("LAB//creatinine//mg/dl//bmp",), "mg/dL"))
LABELS = {"decompensation@1440": "Death within 24 h", "icu_death@-1": "ICU death", "hospital_death@0":
          "In-hospital death", "imv_onset@1440": "New IMV within 24 h", "vasopressor_onset@1440":
          "New vasopressor within 24 h", "rrt_onset@2880": "New RRT within 48 h", "aki_2@2880": "AKI stage 2+ within 48 h",
          "remaining_los_gt_7d": "ICU stay over 7 more days", "hypotension@240": "Hypotension within 4 h",
          "hyperkalemia_top@1440": "Hyperkalaemia within 24 h", "thrombocytopenia@2880": "Thrombocytopenia within 48 h",
          "sepsis_ase@1440": "Sepsis (ASE) within 24 h"}
CAUSE_NAMES = {"death": "Death", "alive": "Alive discharge", "imv": "IMV", "vasopressor": "Vasopressor",
               "rrt": "RRT"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt(x, d: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)) and not pct:
        return f"{int(x):,}"
    if pct:
        return f"{100 * float(x):.{max(d - 2, 0)}f}%"
    return f"{float(x):,.{d}f}"


def label(prog: str) -> str:
    return LABELS.get(prog, prog)


def page(title: str, body: str, script: str = "") -> str:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            f"content=\"width=device-width, initial-scale=1\"><title>{esc(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main>{f'<script>{script}</script>' if script else ''}</body></html>")


def write(text: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def data_tag(obj, tag_id: str) -> str:
    """JSON for the page's script (``</`` escaped so it cannot close the tag)."""
    from icutwin.toolkit.schema import clean

    s = json.dumps(clean(obj)).replace("</", "<\\/")
    return f"<script type=\"application/json\" id=\"{tag_id}\">{s}</script>"


# -- charts -----------------------------------------------------------------------------------------------------------
def _scale(lo, hi, a, b):
    span = (hi - lo) or 1.0
    return lambda x: a + (float(x) - lo) / span * (b - a)


def line_chart(series: dict, xlabel: str = "hours", ylabel: str = "", ymax: float | None = None, width: int = 620,
               height: int = 230, pct: bool = True) -> str:
    """Lines (name -> (xs, ys, dashed)); None or NaN values break a line."""
    xs_all = [x for xs, _ys, _d in series.values() for x in xs]
    ys_all = [y for _xs, ys, _d in series.values() for y in ys if y is not None and np.isfinite(y)]
    if not xs_all or not ys_all:
        return "<p class=\"small\">No values.</p>"
    x0, x1 = min(xs_all), max(xs_all)
    y0, y1 = 0.0, ymax if ymax is not None else max(ys_all) * 1.1 or 1.0
    L, R, T, B = 48, 140, 10, 34
    sx, sy = _scale(x0, x1, L, width - R), _scale(y0, y1, height - B, T)
    out = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\">"]
    for k in range(6):
        yv = y0 + (y1 - y0) * k / 5
        y = sy(yv)
        out.append(f"<line class=\"grid\" x1=\"{L}\" x2=\"{width - R}\" y1=\"{y:.1f}\" y2=\"{y:.1f}\"/>"
                   f"<text x=\"{L - 6}\" y=\"{y + 4:.1f}\" text-anchor=\"end\">"
                   f"{fmt(yv, 0, pct=True) if pct else fmt(yv, 1)}</text>")
    for xv in np.linspace(x0, x1, 7):
        x = sx(xv)
        out.append(f"<text x=\"{x:.1f}\" y=\"{height - B + 16}\" text-anchor=\"middle\">{xv:.0f}</text>")
    out.append(f"<line class=\"axis\" x1=\"{L}\" x2=\"{width - R}\" y1=\"{height - B}\" y2=\"{height - B}\"/>"
               f"<text x=\"{(L + width - R) / 2:.0f}\" y=\"{height - 4}\" text-anchor=\"middle\">{esc(xlabel)}</text>")
    if ylabel:
        out.append(f"<text x=\"12\" y=\"{T + 10}\">{esc(ylabel)}</text>")
    for i, (name, (xs, ys, dashed)) in enumerate(series.items()):
        col = PALETTE[i % len(PALETTE)]
        segs, cur = [], []
        for x, y in zip(xs, ys, strict=True):
            if y is None or not np.isfinite(y):
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(f"{sx(x):.1f},{sy(y):.1f}")
        if cur:
            segs.append(cur)
        dash = " stroke-dasharray=\"4 3\"" if dashed else ""
        for s in segs:
            out.append(f"<polyline fill=\"none\" stroke=\"{col}\" stroke-width=\"2\"{dash} points=\"{' '.join(s)}\"/>")
        ly = T + 14 + 16 * i
        out.append(f"<line x1=\"{width - R + 10}\" x2=\"{width - R + 30}\" y1=\"{ly}\" y2=\"{ly}\" stroke=\"{col}\" "
                   f"stroke-width=\"2\"{dash}/><text x=\"{width - R + 36}\" y=\"{ly + 4}\">{esc(name)}</text>")
    out.append("</svg>")
    return "".join(out)


def interval_chart(rows: list, band: tuple | None = None, width: int = 620, unit: str = "") -> str:
    """Dot-and-whisker rows (label, centre, lo, hi); ``band`` a shaded null band (lo, hi) around zero."""
    vals = [v for _l, c, lo, hi in rows for v in (c, lo, hi) if v is not None and np.isfinite(v)]
    if band:
        vals += [band[0], band[1]]
    if not vals:
        return "<p class=\"small\">No values.</p>"
    lo_v, hi_v = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.08 * ((hi_v - lo_v) or 1.0)
    lo_v, hi_v = lo_v - pad, hi_v + pad
    L, R, rh = 170, 20, 26
    height = rh * len(rows) + 36
    sx = _scale(lo_v, hi_v, L, width - R)
    out = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\">"]
    if band:
        out.append(f"<rect x=\"{sx(band[0]):.1f}\" y=\"6\" width=\"{max(sx(band[1]) - sx(band[0]), 1):.1f}\" "
                   f"height=\"{rh * len(rows)}\" fill=\"var(--band)\"/>")
    out.append(f"<line class=\"axis\" x1=\"{sx(0):.1f}\" x2=\"{sx(0):.1f}\" y1=\"4\" y2=\"{rh * len(rows) + 8}\"/>")
    for i, (lab, c, lo, hi) in enumerate(rows):
        y = 6 + rh * i + rh / 2
        out.append(f"<text x=\"{L - 8}\" y=\"{y + 4:.1f}\" text-anchor=\"end\">{esc(lab)}</text>")
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            out.append(f"<line x1=\"{sx(lo):.1f}\" x2=\"{sx(hi):.1f}\" y1=\"{y:.1f}\" y2=\"{y:.1f}\" "
                       f"stroke=\"{PALETTE[0]}\" stroke-width=\"2\"/>")
        if c is not None and np.isfinite(c):
            out.append(f"<circle cx=\"{sx(c):.1f}\" cy=\"{y:.1f}\" r=\"4.5\" fill=\"{PALETTE[0]}\"/>")
    for xv in np.linspace(lo_v, hi_v, 5):
        out.append(f"<text x=\"{sx(xv):.1f}\" y=\"{height - 8}\" text-anchor=\"middle\">{xv:.3g}{esc(unit)}</text>")
    out.append("</svg>")
    return "".join(out)


def bar_chart(rows: list, width: int = 620) -> str:
    """Horizontal bars (label, value, marker): signed values about zero, an optional marker (a second method)."""
    vals = [v for _l, a, b in rows for v in (a, b) if v is not None and np.isfinite(v)]
    if not vals:
        return "<p class=\"small\">No values.</p>"
    m = max(abs(min(vals)), abs(max(vals))) or 1.0
    L, R, rh = 190, 20, 22
    height = rh * len(rows) + 30
    sx = _scale(-m, m, L, width - R)
    out = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\">",
           f"<line class=\"axis\" x1=\"{sx(0):.1f}\" x2=\"{sx(0):.1f}\" y1=\"2\" y2=\"{rh * len(rows) + 4}\"/>"]
    for i, (lab, a, b) in enumerate(rows):
        y = 4 + rh * i
        out.append(f"<text x=\"{L - 8}\" y=\"{y + 14}\" text-anchor=\"end\">{esc(lab)}</text>")
        if a is not None and np.isfinite(a):
            x0, x1 = sorted((sx(0), sx(a)))
            col = PALETTE[1] if a > 0 else PALETTE[0]
            out.append(f"<rect x=\"{x0:.1f}\" y=\"{y + 3}\" width=\"{max(x1 - x0, 1):.1f}\" height=\"{rh - 8}\" "
                       f"fill=\"{col}\" opacity=\".85\"/>")
        if b is not None and np.isfinite(b):
            out.append(f"<path d=\"M{sx(b):.1f},{y + 2} l5,7 l-5,7 l-5,-7 z\" fill=\"var(--ink)\"/>")
    out.append(f"<text x=\"{sx(-m):.1f}\" y=\"{height - 6}\">lowered the estimate</text>"
               f"<text x=\"{sx(m):.1f}\" y=\"{height - 6}\" text-anchor=\"end\">raised the estimate</text></svg>")
    return "".join(out)


# -- the patient's history ----------------------------------------------------------------------------------------
def _intervals(stay, hours_to: float) -> list:
    """Vasoactive infusion intervals (drug, start h, end h) from MED_CON records, IMV and CRRT ticks."""
    run, out = {}, []
    t0 = stay.icu_in
    for t, _r, code, _n, _x in stay.timeline:
        h = (t - t0).total_seconds() / 3600
        if h > hours_to:
            break
        parts = code.split("//")
        if parts[0] == "MED_CON" and len(parts) == 5 and parts[1] in VASO:
            act = parts[3]
            if act in ("start", "going") and parts[1] not in run:
                run[parts[1]] = h
            elif act == "stop" and parts[1] in run:
                out.append(("vaso", parts[1], run.pop(parts[1]), h))
        elif code == "RESP//device_category//imv":
            out.append(("imv", "IMV", h, h))
        elif code.startswith(("CRRT//", "IHD//")):
            out.append(("rrt", "RRT", h, h))
    out += [("vaso", d, h, hours_to) for d, h in run.items()]
    return out


def timeline(stay, hours: float, after: float = 0.0, start: float | None = None, width: int = 620) -> str:
    """Small multiples of a stay up to ``hours`` (and ``after`` more hours, shaded) on hours since ICU admission."""
    t0 = stay.icu_in
    end = hours + after
    s0 = max(0.0, hours - 48.0) if start is None else start
    pts: dict = {name: [] for name, _c, _u in SERIES}
    for t, _r, code, num, _x in stay.timeline:
        h = (t - t0).total_seconds() / 3600
        if h < s0 or h > end or num is None or not np.isfinite(num):
            continue
        for name, codes, _u in SERIES:
            if code in codes:
                pts[name].append((h, float(num)))
    L, R, ph, gap = 92, 16, 46, 10
    rows = [n for n, _c, _u in SERIES if pts[n]]
    lanes = _intervals(stay, end)
    height = (ph + gap) * (len(rows) + 1) + 30
    sx = _scale(s0, max(end, s0 + 1), L, width - R)
    out = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\">"]
    if after > 0:
        out.append(f"<rect x=\"{sx(hours):.1f}\" y=\"0\" width=\"{sx(end) - sx(hours):.1f}\" height=\"{height - 26}\" "
                   f"fill=\"var(--band)\"/>")
    for i, name in enumerate(rows):
        y0 = i * (ph + gap) + 4
        xs = [p[0] for p in pts[name]]
        ys = [p[1] for p in pts[name]]
        lo, hi = min(ys), max(ys)
        sy = _scale(lo - 0.05 * ((hi - lo) or 1), hi + 0.05 * ((hi - lo) or 1), y0 + ph, y0)
        unit = next(u for n, _c, u in SERIES if n == name)
        out.append(f"<text x=\"{L - 8}\" y=\"{y0 + 14}\" text-anchor=\"end\">{esc(name)}</text>"
                   f"<text x=\"{L - 8}\" y=\"{y0 + 28}\" text-anchor=\"end\">{esc(unit)}</text>"
                   f"<line class=\"grid\" x1=\"{L}\" x2=\"{width - R}\" y1=\"{y0 + ph:.1f}\" y2=\"{y0 + ph:.1f}\"/>"
                   f"<text x=\"{width - R}\" y=\"{y0 + 9}\" text-anchor=\"end\">{hi:.3g}</text>")
        col = PALETTE[i % len(PALETTE)]
        if len(xs) > 1 and name in ("MAP", "Heart rate", "SpO2"):
            out.append(f"<polyline fill=\"none\" stroke=\"{col}\" stroke-width=\"1.5\" points=\""
                       + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys, strict=True)) + "\"/>")
        else:
            out += [f"<circle cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" r=\"2.6\" fill=\"{col}\"/>" for x, y in
                    zip(xs, ys, strict=True)]
    y0 = len(rows) * (ph + gap) + 4
    out.append(f"<text x=\"{L - 8}\" y=\"{y0 + 14}\" text-anchor=\"end\">Treatments</text>")
    drugs = sorted({d for k, d, _a, _b in lanes if k == "vaso"})
    for kind, name, a, b in lanes:
        if b < s0:
            continue
        a = max(a, s0)
        if kind == "vaso":
            yy = y0 + 4 + 8 * (drugs.index(name) % 4)
            out.append(f"<rect x=\"{sx(a):.1f}\" y=\"{yy}\" width=\"{max(sx(b) - sx(a), 2):.1f}\" height=\"6\" "
                       f"fill=\"{PALETTE[1]}\"><title>{esc(name)}</title></rect>")
        else:
            yy = y0 + 38 if kind == "imv" else y0 + 30
            out.append(f"<line x1=\"{sx(a):.1f}\" x2=\"{sx(a):.1f}\" y1=\"{yy - 6}\" y2=\"{yy + 2}\" "
                       f"stroke=\"{PALETTE[2] if kind == 'imv' else PALETTE[3]}\" stroke-width=\"1.5\"/>")
    out.append(f"<line x1=\"{sx(hours):.1f}\" x2=\"{sx(hours):.1f}\" y1=\"0\" y2=\"{height - 26}\" stroke=\"var(--ink)\" "
               f"stroke-dasharray=\"3 3\"/>")
    for xv in np.linspace(s0, max(end, s0 + 1), 7):
        out.append(f"<text x=\"{sx(xv):.1f}\" y=\"{height - 10}\" text-anchor=\"middle\">{xv:.0f} h</text>")
    legend = " ".join(f"{esc(d)}" for d in drugs) or "none"
    out.append("</svg>")
    return ("".join(out) + f"<p class=\"small\">Orange bars: vasoactive infusions ({legend}); green ticks: IMV records; "
            f"purple ticks: RRT records. Dashed line: the decision point" + (" (shaded: what followed)." if after
                                                                              else ".") + "</p>")


# -- result sections ------------------------------------------------------------------------------------------------
def header(r: dict, title: str) -> str:
    p, a, pv = r.get("patient") or {}, r.get("anchor") or {}, r.get("provenance") or {}
    who = (f"{esc(p.get('ref'))} · {esc(p.get('kind'))} · {esc(p.get('dataset') or 'no known dataset')}"
           + (" · site unknown in the prompt" if p.get("unknown_site") else "")) if p else ""
    at = f" · at {fmt(a.get('hours'), 1)} h after ICU admission" if a else ""
    badges = [(f"<span class=\"badge {'ok' if pv.get('gate') == 'PASS' else 'warn'}\">privacy gate "
               f"{esc(pv.get('gate') or 'not checked')}</span>")]
    cs = str((pv.get("gate_not_in_v1") or {}).get("conditional_and_counterfactual_sampling") or "")
    if cs and r.get("tool") in ("forecast", "whatif", "explain"):
        badges.append("<span class=\"badge ok\">prompted sampling: gate PASS (amendment 11 V3)</span>"
                      if cs.startswith("PASS") else
                      f"<span class=\"badge warn\">prompted sampling: {esc(cs)} under the gate</span>")
    if p.get("kind") == "twin":
        badges.append("<span class=\"badge\">synthetic patient (a twin)</span>")
    elif p:
        badges.append("<span class=\"badge bad\">real stay: individual data, approved host only</span>")
    return (f"<header><h1>{esc(title)}</h1><p class=\"meta\">{who}{at}</p>{''.join(badges)}</header>")


def footer(r: dict) -> str:
    ev, pv = r.get("evidence") or {}, r.get("provenance") or {}
    refs = "".join(f"<li>{esc(x)}</li>" for x in ev.get("refs") or [])
    claim = f"<p>{esc(ev.get('claim'))}</p>" if ev.get("claim") else ""
    framing = f"<p class=\"banner\">{esc(ev.get('framing'))}</p>" if ev.get("framing") else ""
    cav = "".join(f"<li>{esc(c)}</li>" for c in r.get("caveats") or [])
    prov = ", ".join(f"{k}: {esc(pv.get(k))}" for k in ("run", "weights_sha256", "gate", "gate_amendment", "backend",
                                                        "git", "created") if pv.get(k))
    return (f"<section><h2>Evidence and caveats</h2>{framing}{claim}<ul>{refs}</ul><h3>Caveats</h3>"
            f"<ul class=\"caveats\">{cav}</ul><p class=\"small\">Provenance: {prov}</p></section>")


def forecast_section(r: dict, show_observed: bool = False) -> str:
    o = r["outputs"]
    risks = o.get("risks") or {}
    keys = [k for k in LABELS if k in risks] + sorted(k for k in risks if k not in LABELS)
    rows = []
    for k in keys:
        x = risks[k]
        obs = f"<td class=\"num\">{fmt(x.get('observed'), 0)}</td>" if show_observed else ""
        rows.append(f"<tr><td>{esc(label(k))}<div class=\"small\">{esc(k)}</div></td><td class=\"num\">"
                    f"{fmt(x.get('risk'), 3, pct=True)}</td><td>{esc(x.get('estimator'))}"
                    f"{' · calibrated' if x.get('calibrated') else ''}</td><td class=\"num\">{fmt(x.get('share'), 3, pct=True)}"
                    f"</td><td class=\"num\">{fmt(x.get('e2'), 3, pct=True)}</td><td class=\"num\">"
                    f"{fmt(x.get('paper_oe'), 2)}</td>{obs}</tr>")
    head = ("<tr><th>Outcome</th><th class=\"num\">Risk</th><th>Estimator</th><th class=\"num\">Share</th>"
            "<th class=\"num\">E2</th><th class=\"num\">Paper O/E</th>"
            + ("<th class=\"num\">Observed</th>" if show_observed else "") + "</tr>")
    inc = o.get("incidence") or {}
    chart = ""
    if inc.get("hours"):
        series = {}
        for c in ("death", "imv", "vasopressor", "rrt"):
            if c in inc:
                series[f"{CAUSE_NAMES[c]} (E2)"] = (inc["hours"], inc[c]["e2"], False)
        chart = line_chart(series, "hours after the prediction time", "cumulative incidence")
    vals = o.get("values") or {}
    vrows = "".join(f"<tr><td>{esc(k)}</td><td class=\"num\">{fmt(v.get('q10'))}</td><td class=\"num\">"
                    f"{fmt(v.get('q50'))}</td><td class=\"num\">{fmt(v.get('q90'))}</td><td class=\"num\">"
                    f"{fmt(v.get('n_valid'), 0)}</td></tr>" for k, v in sorted(vals.items()) if v.get("q50") is not None)
    rl = o.get("remaining_los") or {}
    los = (f"<p>Remaining ICU stay: median {fmt(rl.get('median_hours'), 1)} h; P(over 7 more days) "
           f"{fmt(rl.get('p_gt_7d'), 3, pct=True)}; censored rollouts {fmt(rl.get('censored_share'), 3, pct=True)}.</p>"
           if rl else "")
    return (f"<section><h2>Risks</h2><div class=\"wrap\"><table>{head}{''.join(rows)}</table></div>"
            f"<p class=\"small\">Risk: E2 where the hazards define it, else the share of valid rollouts; recalibrated "
            f"when amendment 11's map exists. Paper O/E: observed over expected of the raw Twin in the paper (above 1: "
            f"under-predicted).</p></section><section><h2>Cumulative incidence (E2)</h2>{chart}</section>"
            + (f"<section><h2>Value forecasts</h2><div class=\"wrap\"><table><tr><th>Program</th><th class=\"num\">"
               f"10th</th><th class=\"num\">median</th><th class=\"num\">90th</th><th class=\"num\">rollouts</th></tr>"
               f"{vrows}</table></div><p class=\"small\">80% intervals cover about 0.70 of real values (F5): too narrow."
               f"</p>{los}</section>" if vrows or los else ""))


def whatif_section(r: dict) -> str:
    o = r["outputs"]
    parts = []
    for resp, x in (o.get("responses") or {}).items():
        if not x.get("pre_specified") and not x["contrasts"]:
            continue
        rows = [(arm, a.get("mean"), a.get("q10"), a.get("q90")) for arm, a in x["arms"].items()]
        crow = [(f"{arm} - withhold", c.get("difference"), (c.get("ci") or [None, None])[0],
                 (c.get("ci") or [None, None])[1]) for arm, c in x["contrasts"].items()]
        band = (-x["aa_band"], x["aa_band"]) if x.get("aa_band") else None
        flag = x.get("evidence") or "exploratory"
        paper = x.get("paper") or {}
        note = (f" Paper (F5): observational {fmt(paper.get('observational'))}, Twin {fmt(paper.get('twin'))}."
                if paper else "")
        parts.append(f"<h3>{esc(resp)} <span class=\"flag-{esc(flag)}\">({esc(flag.replace('_', ' '))})</span></h3>"
                     f"<p class=\"small\">Arms: mean with the 10th to 90th percentile of rollouts.{esc(note)}</p>"
                     f"{interval_chart(rows)}<p class=\"small\">Contrasts with Welch 95% intervals; shaded: the A/A band "
                     f"(+-1.96 standard errors between two withhold arms).</p>{interval_chart(crow, band)}")
    dropped = o.get("dropped") or {}
    drop = ("<p class=\"small\">Rejected by the grammar: " + "; ".join(f"{esc(k)} ({esc(v)})" for k, v in
                                                                         dropped.items()) + "</p>") if dropped else ""
    return f"<section><h2>Responses</h2>{''.join(parts)}{drop}<p class=\"banner\">{esc(o.get('timing'))}</p></section>"


def explain_section(r: dict) -> str:
    o = r["outputs"]
    rows = [(g["group"], g.get("edit"), g.get("knockout")) for g in (o.get("groups") or [])[:14]]
    ag = o.get("agreement_edit_knockout") or {}
    return (f"<section><h2>What moved the estimate</h2><p>Probe risk {fmt(o['probe']['risk'], 3, pct=True)} for "
            f"{esc(label(r['settings']['program']))}. Bars: counterfactual editing (the probe logit's change when the "
            f"group is removed); diamonds: attention knockout.</p>{bar_chart(rows)}<p class=\"small\">Agreement of the "
            f"two methods: Spearman {fmt(ag.get('spearman'))}, top-3 overlap {fmt(ag.get('top_overlap'))} over "
            f"{fmt(ag.get('groups'), 0)} groups.</p></section>")


def similar_section(r: dict) -> str:
    o = r["outputs"]
    rows = []
    for prog, x in o["rates"].items():
        rate = fmt(x.get("rate"), 3, pct=True) if x.get("rate") is not None else esc(x.get("bound") or "–")
        ci = x.get("ci") or [None, None]
        rows.append(f"<tr><td>{esc(label(prog))}</td><td class=\"num\">{rate}</td><td class=\"num\">"
                    f"{fmt(ci[0], 3, pct=True)} to {fmt(ci[1], 3, pct=True)}</td><td class=\"num\">"
                    f"{fmt(x.get('neighbours_with_label'), 0)}</td><td class=\"num\">{fmt(x.get('twin_risk'), 3, pct=True)}"
                    f"</td></tr>")
    return (f"<section><h2>Outcomes of the {o['k']} nearest held-out stays</h2><div class=\"wrap\"><table><tr>"
            f"<th>Outcome</th><th class=\"num\">Rate</th><th class=\"num\">95% interval</th><th class=\"num\">"
            f"With a label</th><th class=\"num\">Twin's risk</th></tr>{''.join(rows)}</table></div><p class=\"small\">"
            f"Counts under {o['min_cell']} suppressed; no neighbour shown. Median similarity "
            f"{fmt(o['similarity']['median'], 3)}.</p></section>")


def cohort_section(r: dict) -> str:
    o = r["outputs"]
    fid = o.get("fidelity") or {}
    rows = []
    if fid:
        for k in ("age", "female", "icu_death", "hosp_death", "ever_imv", "ever_vasopressor", "ever_rrt",
                  "events_per_hour", "labs_per_day"):
            c, q = fid["cohort"].get(k) or {}, fid["real"].get(k) or {}
            cv, qv = c.get("mean", c.get("median")), q.get("mean", q.get("median"))
            rows.append(f"<tr><td>{esc(k)}</td><td class=\"num\">{fmt(cv, 3)}</td><td class=\"num\">{fmt(qv, 3)}</td>"
                        f"</tr>")
    conf = "".join(f"<tr><td>{esc(k)}</td><td class=\"num\">{fmt(v, 3, pct=True)}</td></tr>"
                   for k, v in (o.get("conformance") or {}).items())
    return (f"<section><h2>Cohort</h2><p>{fmt(o.get('n'), 0)} twins ({esc(o.get('mode'))}); per dataset: "
            f"{esc(o.get('datasets'))}. Exported to {esc(o.get('export_dir') or 'nothing')}.</p>"
            f"<h3>Meeting each filter</h3><table>{conf}</table><h3>Profile against real held-out stays (same filters)"
            f"</h3><table><tr><th>Quantity</th><th class=\"num\">Cohort</th><th class=\"num\">Real</th></tr>"
            f"{''.join(rows)}</table></section>")


def trial_section(r: dict) -> str:
    o = r["outputs"]
    rows = []
    for name, s in o["scores"].items():
        adj = s.get("adjustment") or {}
        enr = {e["fraction"]: e for e in s["enrichment"]}
        rows.append(f"<tr><td>{esc(name)}</td><td class=\"num\">{fmt(s.get('auroc'), 3)}</td>"
                    + "".join(f"<td class=\"num\">{fmt((enr.get(q) or {}).get('n_total'), 0)}</td>"
                              for q in (1.0, 0.5, 0.25))
                    + f"<td class=\"num\">{fmt(adj.get('n_ratio'), 2)}</td><td class=\"num\">"
                      f"{fmt(adj.get('type1_adjusted'), 3)}</td></tr>")
    pop = o["population"]
    return (f"<section><h2>{esc(label(o['endpoint']))} at {esc(o['design'])}, relative risk {fmt(o['rr'], 2)}</h2>"
            f"<p>{fmt(pop['anchors'], 0)} eligible held-out anchors; event rate {fmt(pop['event_rate'], 3, pct=True)};"
            f" unenriched {fmt(o['unenriched']['n_total'], 0)} patients for {fmt(o['power'], 2)} power.</p>"
            f"<div class=\"wrap\"><table><tr><th>Score</th><th class=\"num\">AUROC</th><th class=\"num\">N, all</th>"
            f"<th class=\"num\">N, top 50%</th><th class=\"num\">N, top 25%</th><th class=\"num\">N adjusted / "
            f"unadjusted</th><th class=\"num\">Type I (adjusted)</th></tr>{''.join(rows)}</table></div></section>")


SECTIONS = {"forecast": ("Forecast", forecast_section), "whatif": ("What-if", whatif_section),
            "explain": ("Explain", explain_section), "similar": ("Similar patients", similar_section),
            "cohort": ("Cohort synthesizer", cohort_section), "trial": ("Trial designer", trial_section)}


def render(r: dict, session=None) -> str:
    """The card of a toolkit result (with ``session``: the patient's history timeline for patient results)."""
    title, fn = SECTIONS[r["tool"]]
    body = header(r, title)
    p, a = r.get("patient"), r.get("anchor")
    if session is not None and p and a and p.get("ref"):
        from icutwin.toolkit import inputs

        pat = inputs.load(session, p["ref"], unknown_site=p.get("unknown_site"))
        body += f"<section><h2>History</h2>{timeline(pat.stay(session.vocab), a['hours'])}</section>"
    if r["tool"] == "whatif":
        body += f"<p class=\"banner\">{esc((r.get('evidence') or {}).get('framing'))}</p>"
    body += fn(r) + footer(r)
    return page(f"{title} card", body)


def blinded(session, patient, hours: float) -> str:
    """The clinician study's stimulus: the timeline alone, no reference, kind, dataset or outcome."""
    return page("Timeline", f"<section><h2>ICU stay, first {hours:g} hours</h2>"
                            f"{timeline(patient.stay(session.vocab), hours, start=0.0)}</section>")


# -- teaching -------------------------------------------------------------------------------------------------------
TEACH_JS = """
const D=JSON.parse(document.getElementById('case').textContent);
function val(id){const e=document.getElementById(id);return e?Number(e.value)/100:null}
document.querySelectorAll('input[type=range]').forEach(r=>{const o=document.getElementById(r.id+'_v');
 const f=()=>{o.textContent=r.value+'%'};r.addEventListener('input',f);f()});
function answers(){const a={case:D.case,patient:D.patient,time:new Date().toISOString(),risks:{},action:null,why:''};
 D.drill.forEach(k=>{a.risks[k]=val('q_'+k)});const c=document.querySelector('input[name=action]:checked');
 a.action=c?c.value:null;const w=document.getElementById('why');a.why=w?w.value:'';return a}
function reveal(){document.querySelectorAll('.hidden').forEach(e=>e.classList.remove('hidden'));
 const a=answers();let rows='',bl=0,bt=0,n=0;D.drill.forEach(k=>{const r=D.risks[k]||{};const y=r.observed;
  const t=r.risk;const l=a.risks[k];if(y!==null&&y!==undefined&&t!==null&&l!==null){bl+=(l-y)**2;bt+=(t-y)**2;n++}
  rows+=`<tr><td>${D.labels[k]||k}</td><td class=num>${l===null?'–':Math.round(100*l)+'%'}</td><td class=num>${t==null?'–':
  (100*t).toFixed(1)+'%'}</td><td class=num>${y==null?'–':(y?'yes':'no')}</td></tr>`});
 document.getElementById('score').innerHTML=`<table><tr><th>Outcome</th><th class=num>Your estimate</th><th class=num>
 Twin forecast</th><th class=num>What happened (the twin)</th></tr>${rows}</table><p>Brier score: yours ${n?(bl/n).toFixed(3)
 :'–'}, the twin's forecast ${n?(bt/n).toFixed(3):'–'} (lower is better).</p>`}
function download(){const b=new Blob([JSON.stringify(answers(),null,1)],{type:'application/json'});const u=URL.createObjectURL(b);
 const a=document.createElement('a');a.href=u;a.download=`answers_case${D.case}.json`;a.click();URL.revokeObjectURL(u)}
"""


def _case_data(case: dict) -> dict:
    risks = (case["forecast"].get("outputs") or {}).get("risks") or {}
    return {"case": case["case"], "patient": case["patient"], "drill": case["drill"],
            "risks": {k: {"risk": risks.get(k, {}).get("risk"), "observed": risks.get(k, {}).get("observed")}
                      for k in case["drill"]}, "labels": {k: label(k) for k in case["drill"]}}


def teaching(case: dict, session, mode: str = "learner") -> str:
    """A teaching case's page (module docstring)."""
    from icutwin.toolkit import inputs

    pat = inputs.load(session, case["patient"])
    stay = pat.stay(session.vocab)
    h = float(case["hours"])
    teacher = mode == "teacher"
    hide = "" if teacher else " hidden"
    body = [(f"<header><h1>Teaching case {case['case'] + 1}</h1><p class=\"meta\">A synthetic ICU patient (a twin of "
             f"the bank, {esc(case['dataset'])}); decision point {h:.1f} h after ICU admission.</p>"
             f"<span class=\"badge\">synthetic patient</span><span class=\"badge warn\">internal: not for sharing "
             f"until cleared</span></header>"),
            f"<section><h2>The story so far</h2>{timeline(stay, h)}</section>"]
    opts = []
    for opt, wi in (case.get("whatif") or {}).items():
        if "outputs" in wi:
            opts += [f"{opt}:{arm}" for arm in wi["outputs"]["arms"] if arm.startswith("give_")]
    opts += ["withhold"]
    if not teacher:
        sliders = "".join(f"<label class=\"q\" for=\"q_{esc(k)}\">{esc(label(k))}: <b id=\"q_{esc(k)}_v\"></b></label>"
                          f"<input type=\"range\" id=\"q_{esc(k)}\" min=\"0\" max=\"100\" value=\"10\">"
                          for k in case["drill"])
        radios = "".join(f"<label class=\"q\"><input type=\"radio\" name=\"action\" value=\"{esc(o)}\"> "
                         f"{esc(o.replace(':', ': ').replace('_', ' '))}</label>" for o in opts)
        body.append(f"<section><h2>Your assessment</h2>{sliders}<h3>What would you do now?</h3>{radios}"
                    f"<label class=\"q\" for=\"why\">Why?</label><textarea id=\"why\"></textarea>"
                    f"<button onclick=\"reveal()\">Reveal</button><button class=\"secondary\" onclick=\"download()\">"
                    f"Download answers</button><div id=\"score\"></div></section>")
    fc = case["forecast"]
    body.append(f"<div class=\"{hide.strip()}\"><h2>The twin's forecast</h2>{forecast_section(fc, show_observed=True)}")
    for opt, wi in (case.get("whatif") or {}).items():
        if "outputs" in wi:
            body.append(f"<section><h2>What if: {esc(opt)}</h2><p class=\"banner\">Observational simulation within the "
                        f"twin world, not the effect of the action.</p>{whatif_section(wi)}</section>")
    body.append(f"<section><h2>What happened to this twin</h2>{timeline(stay, h, after=24.0)}<p>ICU discharge "
                f"{fmt(case['course'].get('icu_discharge_hours'), 1)} h after the decision point.</p></section></div>")
    if teacher:
        ex = case.get("explain") or {}
        if "outputs" in ex:
            body.append(explain_section(ex))
        top = [g["group"] for g in (ex.get("outputs") or {}).get("groups", [])[:3]]
        body.append("<section><h2>Discussion prompts</h2><ul>"
                    f"<li>Which parts of the history drove the twin's estimate ({esc(', '.join(top) or '–')})? Do they "
                    f"match your reasoning?</li><li>Compare the simulated responses under each option with the A/A band: "
                    f"which differences exceed rollout noise?</li><li>Why can an observational simulation differ from "
                    f"the causal effect of the same action (confounding by indication)?</li><li>The twin's risks against "
                    f"what happened: is one case evidence of calibration?</li></ul></section>")
    from icutwin.toolkit.schema import evidence

    body.append(footer({**fc, "evidence": {"refs": evidence()["teach"]["refs"]},
                        "caveats": evidence()["teach"]["caveats"]}))
    return page(f"Teaching case {case['case'] + 1}", "".join(body) + data_tag(_case_data(case), "case"),
                TEACH_JS if not teacher else "")


DRILL_JS = """
const C=JSON.parse(document.getElementById('cases').textContent);
document.querySelectorAll('input[type=range]').forEach(r=>{const o=document.getElementById(r.id+'_v');
 const f=()=>{o.textContent=r.value+'%'};r.addEventListener('input',f);f()});
function score(){const pts=[];let bl=0,bt=0,n=0;C.forEach(c=>c.drill.forEach(k=>{const e=document.getElementById(`d${c.case}_${k}`);
 const r=c.risks[k]||{};if(!e||r.observed==null)return;const l=Number(e.value)/100;pts.push([l,r.observed,r.risk]);
 bl+=(l-r.observed)**2;if(r.risk!=null){bt+=(r.risk-r.observed)**2};n++}));
 const bins=[0,.1,.2,.4,.6,1];let svg='<svg viewBox="0 0 320 320"><line x1="40" y1="280" x2="300" y2="20" stroke="gray" stroke-dasharray="4 3"/>';
 for(let i=0;i<bins.length-1;i++){const s=pts.filter(p=>p[0]>=bins[i]&&p[0]<(i==bins.length-2?1.01:bins[i+1]));if(!s.length)continue;
  const x=s.reduce((a,p)=>a+p[0],0)/s.length,y=s.reduce((a,p)=>a+p[1],0)/s.length;
  svg+=`<circle cx="${40+260*x}" cy="${280-260*y}" r="${4+Math.sqrt(s.length)*2}" fill="#2f6fde" opacity=".8"/>`}
 svg+='<line x1="40" y1="280" x2="300" y2="280" stroke="gray"/><line x1="40" y1="20" x2="40" y2="280" stroke="gray"/>'+
 '<text x="170" y="306" text-anchor="middle">your estimate</text><text x="10" y="150" transform="rotate(-90 10 150)" text-anchor="middle">observed rate</text></svg>';
 document.getElementById('result').innerHTML=`<p>Brier score over ${n} questions: yours ${n?(bl/n).toFixed(3):'–'}, the twin's forecast ${n?(bt/n).toFixed(3):'–'}.</p>${svg}`}
"""


def drill(cases: list, session) -> str:
    """The prognosis drill: each case's history and risk questions, scored at the end with a calibration plot."""
    from icutwin.toolkit import inputs

    parts = [(f"<header><h1>Prognosis drill</h1><p class=\"meta\">{len(cases)} synthetic ICU patients (twins). "
              f"Estimate each risk, then score your estimates against what happened to the twins.</p>"
              f"<span class=\"badge\">synthetic patients</span><span class=\"badge warn\">internal: not for sharing "
              f"until cleared</span></header>")]
    for c in cases:
        pat = inputs.load(session, c["patient"])
        qs = "".join(f"<label class=\"q\" for=\"d{c['case']}_{esc(k)}\">{esc(label(k))}: <b id=\"d{c['case']}_{esc(k)}_v\">"
                     f"</b></label><input type=\"range\" id=\"d{c['case']}_{esc(k)}\" min=\"0\" max=\"100\" value=\"10\">"
                     for k in c["drill"])
        parts.append(f"<section><h2>Patient {c['case'] + 1} at {float(c['hours']):.1f} h</h2>"
                     f"{timeline(pat.stay(session.vocab), float(c['hours']))}{qs}</section>")
    parts.append("<section><button onclick=\"score()\">Score my estimates</button><div id=\"result\"></div></section>")
    return page("Prognosis drill", "".join(parts) + data_tag([_case_data(c) for c in cases], "cases"), DRILL_JS)
