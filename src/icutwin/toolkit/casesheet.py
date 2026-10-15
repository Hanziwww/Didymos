"""The clinician study's standardized case sheet (``docs/paper/clinician_study_protocol.md`` section 4): one decoded
stay over a window of hours since ICU admission, as HTML with inline SVG.

Rendering parity: real stays and twins come here the same way, decoded from their tokens by the features' decode
(``factory.twins.decode_features``: values drawn within their bins with the vocabulary's rank persistence, the clock
keyed by the stay's id), so neither quantization nor formatting tells them apart. No identifier, dataset name or date
is written: times are hours since ICU admission.

The sheet: the admission (age band, sex, unit, admission source and type, diagnosis group, weight); vital signs as
small multiples (heart rate; systolic, mean and diastolic pressure; respiratory rate; SpO2; temperature);
respiratory support (the device over time, FiO2, PEEP); the treatment lanes (vasoactive, sedative and other
infusions with their rates, intermittent drugs, renal replacement therapy); urine output per hour; and a table of
labs, blood gases and GCS in 4-hour bins. ``after`` hours past ``hours`` are drawn shaded (a course that followed:
module B1); ``end`` marks an ICU discharge within the window (module A).
"""

from __future__ import annotations

import html

import numpy as np

VITALS = (("Heart rate", "/min", (("HR", ("VITAL//heart_rate",)),)),
          ("Blood pressure", "mmHg", (("SBP", ("VITAL//sbp//arterial", "VITAL//sbp//cuff")),
                                      ("MAP", ("VITAL//map//arterial", "VITAL//map//cuff")),
                                      ("DBP", ("VITAL//dbp//arterial", "VITAL//dbp//cuff")))),
          ("Respiratory rate", "/min", (("RR", ("VITAL//respiratory_rate",)),)),
          ("SpO2", "%", (("SpO2", ("VITAL//spo2",)),)),
          ("Temperature", "°C", (("T", ("VITAL//temp_c",)),)))
RESP = (("FiO2", "%", ("RESP//fio2_set",)), ("PEEP", "cmH2O", ("RESP//peep_set", "RESP//peep_obs")))
LABS = (("Sodium", "mmol/L", ("LAB//sodium//mmol/l//bmp",), 0),
        ("Potassium", "mmol/L", ("LAB//potassium//mmol/l//bmp",), 1),
        ("Bicarbonate", "mmol/L", ("LAB//bicarbonate//mmol/l//bmp", "LAB//bicarbonate//mmol/l//blood_gas"), 0),
        ("Creatinine", "mg/dL", ("LAB//creatinine//mg/dl//bmp",), 2),
        ("Lactate", "mmol/L", ("LAB//lactate//mmol/l//misc",), 1),
        ("Glucose", "mg/dL", ("LAB//glucose//mg/dl//poc", "LAB//glucose_serum//mg/dl//bmp"), 0),
        ("Hemoglobin", "g/dL", ("LAB//hemoglobin//g/dl//cbc",), 1),
        ("Platelets", "10³/µL", ("LAB//platelet_count//10^3/ul//cbc",), 0),
        ("White cells", "10³/µL", ("LAB//wbc//10^3/ul//cbc",), 1),
        ("Bilirubin", "mg/dL", ("LAB//bilirubin_total//mg/dl//lft",), 1),
        ("pH (arterial)", "", ("LAB//ph_arterial//(no units)//blood_gas",), 2),
        ("PaO2", "mmHg", ("LAB//po2_arterial//mmhg//blood_gas",), 0),
        ("PaCO2", "mmHg", ("LAB//pco2_arterial//mmhg//blood_gas",), 0))
GCS = ("PA//gcs_eye", "PA//gcs_motor", "PA//gcs_verbal")
URINE = ("OUTPUT//indwelling_urinary_catheter", "OUTPUT//external_urinary_catheter", "OUTPUT//suprapubic_catheter",
         "OUTPUT//urethral", "OUTPUT//urinary_other", "OUTPUT//nephrostomy", "OUTPUT//urostomy")
VASO = ("angiotensin", "dobutamine", "dopamine", "epinephrine", "isoproterenol", "milrinone", "norepinephrine",
        "phenylephrine", "vasopressin")
SEDATION = ("propofol", "midazolam", "dexmedetomidine", "fentanyl", "morphine", "ketamine", "lorazepam",
            "remifentanil", "sufentanil", "hydromorphone", "clonidine")
DEVICES = {"room_air": 0, "nasal_cannula": 1, "face_mask": 2, "trach_collar": 2, "t_piece": 2, "hfnc": 3, "cpap": 4,
           "nippv": 4, "imv": 5}
DEVICE_NAMES = {"room_air": "room air", "nasal_cannula": "nasal cannula", "face_mask": "face mask",
                "trach_collar": "trach collar", "t_piece": "T-piece", "hfnc": "high-flow O2", "cpap": "CPAP",
                "nippv": "NIV", "imv": "invasive ventilation", "other": "other", "UNK": "unknown"}
DEVICE_SHORT = {"room_air": "RA", "nasal_cannula": "NC", "face_mask": "mask", "trach_collar": "trach", "t_piece": "T-p",
                "hfnc": "HFNC", "cpap": "CPAP", "nippv": "NIV", "imv": "IMV", "other": "other", "UNK": "?"}
COLORS = ("#2f6fde", "#d9822b", "#2e8b57", "#9b59b6", "#c0392b")
# Layout (SVG user units). Every text has a zone of its own, so nothing is drawn over anything else:
# the left gutter (names, units, legends; truncated to fit, the full name in a tooltip), the plot [L, W - R] (data
# only), the right margin (the scale's values), a top strip (the treatment-start label, when there is one), and the
# axis rows below. Values written inside the plot (FiO2, PEEP, infusion rates) sit on rows of their own and are
# written only where they change and their estimated width fits; every record keeps a dot, so the charting rhythm
# stays visible.
W, L, R = 900, 150, 46
TOP = 18                                       # the top strip's height, when a sheet has a cut label
SIZE = {"": 10.5, "tt": 10.5, "lg": 10.5, "cut": 10.5, "tick": 9.5, "in": 10.0, "val": 10.0, "rate": 9.0, "dim": 10.5}
BOLD = {"tt", "lg", "cut", "in"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def label(x) -> str:
    return "unknown" if x in (None, "UNK", "unknown") else str(x).replace("_", " ")


def text_width(s, cls: str = "") -> float:
    """A conservative estimate of a text's width in user units (a proportional sans-serif at the class's size)."""
    w = 0.0
    for ch in str(s):
        if ch in "mwMW%@":
            f = 0.88
        elif ch in "ilI.,:;|!'1()[]jftr/- ":
            f = 0.36
        elif ch.isupper():
            f = 0.70
        elif ch.isdigit():
            f = 0.58
        else:
            f = 0.58
        w += f
    return w * SIZE.get(cls, 10.5) * (1.08 if cls in BOLD else 1.0) * 1.06


def fit(s, width: float, cls: str = "") -> tuple[str, bool]:
    """``s`` truncated with an ellipsis to ``width``; and whether it was truncated."""
    s = str(s)
    if text_width(s, cls) <= width:
        return s, False
    k = len(s)
    while k > 1 and text_width(s[:k].rstrip() + "…", cls) > width:
        k -= 1
    return s[:k].rstrip() + "…", True


def _text(x, y, s, cls: str = "", anchor: str = "start", width: float | None = None, fill: str | None = None) -> str:
    shown, cut = fit(s, width, cls) if width else (str(s), False)
    a = "" if anchor == "start" else f" text-anchor=\"{anchor}\""
    c = f" class=\"{cls}\"" if cls else ""
    f = f" style=\"fill:{fill}\"" if fill else ""
    tip = f"<title>{esc(s)}</title>" if cut else ""
    return f"<text x=\"{x:.1f}\" y=\"{y:.1f}\"{a}{c}{f}>{esc(shown)}{tip}</text>"


def _gutter(y, s, cls: str = "", fill: str | None = None) -> str:
    """A name in the left gutter, right-aligned, truncated to the gutter."""
    return _text(L - 8, y, s, cls, "end", L - 14, fill)


def _h(stay, t) -> float:
    return (t - stay.icu_in).total_seconds() / 3600


def _sx(t0: float, t1: float):
    return lambda h: L + (h - t0) / max(t1 - t0, 1e-6) * (W - L - R)


AXIS_H = 36                                    # the axis rows: tick labels, then the axis title


def _axis(t0: float, t1: float, y: float) -> str:
    """The hour axis at ``y``: ticks at least 34 units apart, the title on a row of its own below them."""
    sx = _sx(t0, t1)
    px = (W - L - R) / max(t1 - t0, 1e-6)
    step = next((s for s in (1, 2, 3, 4, 6, 8, 12, 24, 48, 96) if s * px >= 34), 168)
    out = [f"<line class=\"ax\" x1=\"{L}\" x2=\"{W - R}\" y1=\"{y}\" y2=\"{y}\"/>"]
    for k in np.arange(np.ceil(t0 / step) * step, t1 + 1e-9, step):
        out.append(_text(sx(k), y + 13, f"{k:.0f}", "", "middle"))
    out.append(_text(W - R, y + 30, "hours since ICU admission", "", "end"))
    return "".join(out)


def _events(stay, codes, t0, t1) -> list:
    return [(_h(stay, t), float(n)) for t, _r, c, n, _x in stay.timeline
            if c in codes and n is not None and np.isfinite(n) and t0 <= _h(stay, t) <= t1]


def _shade(t0, t1, hours, after, y0, y1) -> str:
    if after <= 0:
        return ""
    sx = _sx(t0, t1)
    return (f"<rect x=\"{sx(hours):.1f}\" y=\"{y0}\" width=\"{sx(hours + after) - sx(hours):.1f}\" height=\"{y1 - y0}\" "
            f"class=\"after\"/>")


def _cut(t0, t1, hours, y0, y1, label_=None) -> str:
    """The dashed line at ``hours`` over the plot, and its label in the top strip (right of the line, or left of it
    when it would not fit before the right edge)."""
    sx = _sx(t0, t1)
    x = sx(hours)
    out = f"<line x1=\"{x:.1f}\" x2=\"{x:.1f}\" y1=\"{y0}\" y2=\"{y1}\" class=\"cutl\"/>"
    if label_:
        w = text_width(label_, "cut")
        if x + 4 + w <= W - 2:
            out += _text(x + 4, 13, label_, "cut")
        else:
            out += _text(x - 4, 13, label_, "cut", "end", x - L)
    return out


def _spread(items: list, lo: float, hi: float, cls: str, gap: float = 5.0) -> list:
    """(x, text) -> the ones that can be written centred at x without touching the previous one or the plot's
    edges, the first of a run of equal texts only."""
    out, right, last = [], -1e9, None
    for x, s in items:
        if s == last:
            continue
        w = text_width(s, cls)
        x = min(max(x, lo + w / 2), hi - w / 2)
        if x - w / 2 < right + gap:
            continue
        out.append((x, s))
        right, last = x + w / 2, s
    return out


def vitals_svg(stay, t0, t1, hours, after, cut_label=None) -> str:
    top = TOP if cut_label else 4
    ph, gap = 56, 10
    bottom = top + len(VITALS) * (ph + gap) - gap
    height = bottom + AXIS_H
    sx = _sx(t0, t1)
    out = [f"<svg viewBox=\"0 0 {W} {height}\" class=\"sheet\">", _shade(t0, t1, hours, after, top, bottom)]
    for i, (title, unit, series) in enumerate(VITALS):
        y0 = top + i * (ph + gap)
        pts = {lab: _events(stay, codes, t0, t1) for lab, codes in series}
        vals = [v for p in pts.values() for _h_, v in p]
        out.append(_gutter(y0 + 13, title, "tt") + _gutter(y0 + 28, unit) +
                   f"<line class=\"gr\" x1=\"{L}\" x2=\"{W - R}\" y1=\"{y0 + ph}\" y2=\"{y0 + ph}\"/>")
        if len(series) > 1:                     # the legend, in the gutter under the unit
            x = L - 8
            for j in reversed(range(len(series))):
                lab = series[j][0]
                out.append(_text(x, y0 + 43, lab, "lg", "end", fill=COLORS[j % len(COLORS)]))
                x -= text_width(lab, "lg") + 6
        if not vals:
            out.append(_text(L + 6, y0 + 30, "not recorded", "dim"))
            continue
        lo, hi = min(vals), max(vals)
        pad = 0.08 * ((hi - lo) or max(abs(hi), 1.0))
        lo, hi = lo - pad, hi + pad
        sy = lambda v, lo=lo, hi=hi, y0=y0: y0 + 4 + (ph - 8) * (1 - (v - lo) / (hi - lo))  # noqa: E731
        dec = 1 if title == "Temperature" else 0
        ticks = [hi - pad] if abs(sy(hi - pad) - sy(lo + pad)) < 12 or f"{lo + pad:.{dec}f}" == f"{hi - pad:.{dec}f}" \
            else [hi - pad, lo + pad]
        for v_ in ticks:                          # the scale, in the right margin
            out.append(_text(W - R + 5, sy(v_) + 3.5, f"{v_:.{dec}f}", "tick"))
        for j, (lab, p) in enumerate(pts.items()):
            if not p:
                continue
            col = COLORS[j % len(COLORS)]
            out.append(f"<polyline fill=\"none\" stroke=\"{col}\" stroke-width=\"1.4\" points=\""
                       + " ".join(f"{sx(h):.1f},{sy(v):.1f}" for h, v in p) + "\"/>")
            out += [f"<circle cx=\"{sx(h):.1f}\" cy=\"{sy(v):.1f}\" r=\"1.8\" fill=\"{col}\"/>" for h, v in p]
    if cut_label or after:
        out.append(_cut(t0, t1, hours, top - 2, bottom, cut_label))
    out.append(_axis(t0, t1, bottom + 2) + "</svg>")
    return "".join(out)


def _infusions(stay, t0, t1) -> dict:
    """drug -> [(start h, end h, [(h, rate, unit)])] of MED_CON records (start/going opens, stop closes)."""
    runs: dict = {}
    open_: dict = {}
    for t, _r, code, num, _x in stay.timeline:
        p = code.split("//")
        if p[0] != "MED_CON" or len(p) != 5:
            continue
        h = _h(stay, t)
        if h > t1:
            break
        drug, unit, act = p[1], p[2], p[3]
        rate = float(num) if num is not None and np.isfinite(num) else None
        if act in ("start", "going"):
            if drug not in open_:
                open_[drug] = [h, None, []]
            open_[drug][2].append((h, rate, unit))
        elif act == "dose_change" and drug in open_:
            open_[drug][2].append((h, rate, unit))
        elif act == "stop" and drug in open_:
            s = open_.pop(drug)
            s[1] = h
            runs.setdefault(drug, []).append(tuple(s))
    for drug, s in open_.items():
        s[1] = t1
        runs.setdefault(drug, []).append(tuple(s))
    return {d: [x for x in rs if x[1] >= t0] for d, rs in runs.items() if any(x[1] >= t0 for x in rs)}


def _fmt_rate(rate, unit) -> str:
    if rate is None:
        return ""
    u = "" if unit in ("UNK", None) else f" {unit}"
    return f"{rate:.2f}{u}" if rate < 1 else (f"{rate:.1f}{u}" if rate < 10 else f"{rate:.0f}{u}")


def _fio2(v: float) -> float:
    """FiO2 in percent (the stores keep a fraction; a value above 1 is already a percentage)."""
    return v * 100 if v <= 1.0 + 1e-9 else v


ROW, VAL_ROW, INF_ROW = 22, 22, 30


def treatments_svg(stay, t0, t1, hours, after, cut_label=None) -> str:
    inf = _infusions(stay, t0, t1)
    order = sorted(inf, key=lambda d: (0 if d in VASO else (1 if d in SEDATION else 2), d))
    ints: dict = {}
    for t, _r, code, num, _x in stay.timeline:
        p = code.split("//")
        h = _h(stay, t)
        if t0 <= h <= t1 and p[0] == "MED_INT" and len(p) >= 2:
            ints.setdefault(p[1], []).append((h, float(num) if num is not None and np.isfinite(num) else None,
                                              p[2] if len(p) > 2 else ""))
    bol = sorted(ints, key=lambda d: -len(ints[d]))[:8]
    rrt = [_h(stay, t) for t, _r, c, _n, _x in stay.timeline if c.startswith(("CRRT//", "IHD//"))
           and t0 <= _h(stay, t) <= t1]
    dev = [(_h(stay, t), c.split("//")[2]) for t, _r, c, _n, _x in stay.timeline
           if c.startswith("RESP//device_category//") and _h(stay, t) <= t1]
    resp = {name: _events(stay, codes, t0, t1) for name, _u, codes in RESP}
    top = TOP if cut_label else 4
    bottom = top + ROW + len(RESP) * VAL_ROW + len(order) * INF_ROW + len(bol) * ROW + (ROW if rrt else 0)
    height = bottom + AXIS_H
    sx = _sx(t0, t1)
    out = [f"<svg viewBox=\"0 0 {W} {height}\" class=\"sheet\">", _shade(t0, t1, hours, after, top, bottom)]
    y = top
    # respiratory support: the device in force over time, named inside its band where the name fits
    out.append(_gutter(y + 14, "Respiratory support", "tt"))
    cur, segs = None, []
    for h, d in dev:
        if h < t0:
            cur = d
            continue
        segs.append((max(t0, h), d))
    if cur is not None:
        segs.insert(0, (t0, cur))
    for k, (h, d) in enumerate(segs):
        h_end = segs[k + 1][0] if k + 1 < len(segs) else t1
        lvl = DEVICES.get(d, 1)
        col = "#c0392b" if d == "imv" else ("#d9822b" if lvl >= 3 else "#9aa3ad")
        name = DEVICE_NAMES.get(d, d)
        width = sx(h_end) - sx(h)
        out.append(f"<rect x=\"{sx(h):.1f}\" y=\"{y + 5}\" width=\"{max(width, 1):.1f}\" height=\"13\" "
                   f"fill=\"{col}\" opacity=\".35\"><title>{esc(name)}</title></rect>")
        for s in (name, DEVICE_SHORT.get(d, "")):
            if s and text_width(s, "in") + 6 <= width:
                out.append(_text(sx(h) + 3, y + 15, s, "in"))
                break
    if not segs:
        out.append(_text(L + 6, y + 15, "not recorded", "dim"))
    y += ROW
    # FiO2 and PEEP: a dot per record; the value written where it changes and fits
    for name, unit, _codes in RESP:
        out.append(_gutter(y + 12, f"{name} ({unit})"))
        pts = [(h, _fio2(v) if name == "FiO2" else v) for h, v in resp[name]]
        out += [f"<circle cx=\"{sx(h):.1f}\" cy=\"{y + 17}\" r=\"1.5\" class=\"rec\"/>" for h, _v in pts]
        for x, s in _spread([(sx(h), f"{v:.0f}") for h, v in pts], L, W - R, "val"):
            out.append(_text(x, y + 12, s, "val", "middle"))
        if not pts:
            out.append(_text(L + 6, y + 12, "—", "dim"))
        y += VAL_ROW
    # infusions: the bar, and the rate on a line of its own above it, written where it changes and fits
    for drug in order:
        cls = "vaso" if drug in VASO else ("sed" if drug in SEDATION else "inf")
        out.append(_gutter(y + 24, label(drug)))
        right, last = -1e9, None
        for a, b, marks in inf[drug]:
            a_, b_ = max(a, t0), min(b, t1)
            out.append(f"<rect x=\"{sx(a_):.1f}\" y=\"{y + 15}\" width=\"{max(sx(b_) - sx(a_), 2):.1f}\" height=\"10\" "
                       f"class=\"{cls}\"/>")
            for h, rate, unit in marks:
                s = _fmt_rate(rate, unit)
                if h < t0 or not s or s == last:
                    continue
                x, w = sx(h), text_width(s, "rate")
                if x < right + 6 or x + w > W - 2:
                    continue
                out.append(_text(x, y + 10, s, "rate"))
                right, last = x + w, s
        y += INF_ROW
    for drug in bol:
        out.append(_gutter(y + 14, label(drug)))
        for h, dose, unit in ints[drug]:
            out.append(f"<path d=\"M{sx(h):.1f},{y + 4} l4,6 l-4,6 l-4,-6 z\" class=\"bolus\"><title>"
                       f"{esc(_fmt_rate(dose, unit))}</title></path>")
        y += ROW
    if rrt:
        out.append(_gutter(y + 14, "Renal replacement"))
        out += [f"<line x1=\"{sx(h):.1f}\" x2=\"{sx(h):.1f}\" y1=\"{y + 4}\" y2=\"{y + 16}\" class=\"rrt\"/>" for h in rrt]
        y += ROW
    if cut_label or after:
        out.append(_cut(t0, t1, hours, top - 2, bottom, cut_label))
    out.append(_axis(t0, t1, bottom + 2) + "</svg>")
    return "".join(out)


def urine_svg(stay, t0, t1, hours, after) -> str:
    ev = _events(stay, URINE, t0, t1)
    height, base, tall = 64, 52, 42
    sx = _sx(t0, t1)
    bins = np.arange(np.floor(t0), np.ceil(t1) + 1e-9, 1.0)
    tot = np.zeros(max(len(bins) - 1, 0))
    for h, v in ev:
        k = int(np.searchsorted(bins, h, side="right") - 1)
        if 0 <= k < len(tot) and v <= 2000:
            tot[k] += v
    out = [f"<svg viewBox=\"0 0 {W} {height}\" class=\"sheet\">", _shade(t0, t1, hours, after, 4, base),
           _gutter(17, "Urine output", "tt"), _gutter(32, "mL per hour")]
    m = max(tot.max() if len(tot) else 0, 1)
    for k, val in enumerate(tot):
        if val <= 0:
            continue
        hh = val / m * tall
        a, b = sx(max(bins[k], t0)), sx(min(bins[k + 1], t1))          # clipped to the plot
        out.append(f"<rect x=\"{a + 1:.1f}\" y=\"{base - hh:.1f}\" width=\"{max(b - a - 2, 1):.1f}\""
                   f" height=\"{hh:.1f}\" class=\"uo\"><title>{val:.0f} mL</title></rect>")
    out.append(_text(W - R + 5, base - tall + 3.5, f"{m:.0f}", "tick") + _text(W - R + 5, base + 3.5, "0", "tick")
               if ev else _text(L + 6, 30, "not recorded", "dim"))
    out.append(f"<line class=\"ax\" x1=\"{L}\" x2=\"{W - R}\" y1=\"{base}\" y2=\"{base}\"/></svg>")
    return "".join(out)


def labs_table(stay, t0, t1, hours, after, step: float = 4.0) -> str:
    edges = list(np.arange(t0, t1 + 1e-9, step))
    if edges[-1] < t1:
        edges.append(t1)
    head = "".join(f"<th class=\"num{' aft' if a >= hours - 1e-9 and after else ''}\">{a:.0f}–{b:.0f} h</th>"
                   for a, b in zip(edges[:-1], edges[1:], strict=False))
    rows = []
    for name, unit, codes, dec in LABS:
        ev = _events(stay, codes, t0, t1)
        if not ev:
            continue
        cells = []
        for a, b in zip(edges[:-1], edges[1:], strict=False):
            inb = [v for h, v in ev if a <= h < b or (b == t1 and h == b)]
            cells.append(f"<td class=\"num\">{inb[-1]:.{dec}f}</td>" if inb else "<td class=\"num dim\">·</td>")
        rows.append(f"<tr><td>{esc(name)}<span class=\"u\"> {esc(unit)}</span></td>{''.join(cells)}</tr>")
    g = {}
    for t, _r, c, n, _x in stay.timeline:
        h = _h(stay, t)
        if c in GCS and n is not None and t0 <= h <= t1:
            g.setdefault(t, {})[c] = float(n)
    gcs = [(_h(stay, t), sum(x.values())) for t, x in sorted(g.items()) if len(x) == 3]
    if gcs:
        cells = []
        for a, b in zip(edges[:-1], edges[1:], strict=False):
            inb = [v for h, v in gcs if a <= h < b]
            cells.append(f"<td class=\"num\">{inb[-1]:.0f}</td>" if inb else "<td class=\"num dim\">·</td>")
        rows.append(f"<tr><td>GCS<span class=\"u\"> total</span></td>{''.join(cells)}</tr>")
    if not rows:
        return "<p class=\"dim\">No laboratory values in this window.</p>"
    return (f"<div class=\"wrap\"><table class=\"labs\"><tr><th>Latest value in each block</th>{head}</tr>"
            f"{''.join(rows)}</table></div>")


def admission(stay) -> str:
    p, h = stay.prefix or {}, stay.header or {}
    age = p.get("age")
    band = f"{int(age // 5 * 5)}–{int(age // 5 * 5) + 4}" if isinstance(age, (int, float)) and np.isfinite(age) else \
        "unknown"
    w = next((n for t, _r, c, n, _x in stay.timeline if c == "VITAL//weight_kg" and n is not None), None)
    items = [("Age", band), ("Sex", label(p.get("sex"))), ("Unit", label(h.get("unit_type"))),
             ("Admitted from", label(h.get("admission_source"))), ("Admission type", label(h.get("admission_type"))),
             ("Diagnosis group", label(h.get("adm_dx_group"))),
             ("Weight", f"{w:.0f} kg" if w is not None and np.isfinite(w) else "unknown")]
    return "<dl class=\"adm\">" + "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in items) + "</dl>"


def sheet(stay, hours: float, start: float | None = None, after: float = 0.0, end: bool = False,
          cut_label: str | None = None) -> str:
    """The case sheet of ``stay`` from ``start`` (default: ICU admission, or 24 h before ``hours`` when later) to
    ``hours`` (+ ``after`` hours shaded)."""
    los = (stay.icu_out - stay.icu_in).total_seconds() / 3600
    t0 = max(0.0, hours - 24.0) if start is None else start
    t1 = hours + after
    if end and los < t1:
        t1 = max(los, t0 + 1.0)
    note = ""
    if end and los <= hours + 1e-9:
        died = any(c == "ICU//discharge//dead" for _t, _r, c, _n, _x in stay.timeline)
        note = (f"<p class=\"endnote\">The ICU stay ends at {los:.1f} h "
                f"({'death in the ICU' if died else 'discharged alive from the ICU'}).</p>")
    return (f"<div class=\"casesheet\">{admission(stay)}{note}"
            f"<h4>Vital signs</h4><div class=\"plot\">{vitals_svg(stay, t0, t1, hours, after, cut_label)}</div>"
            f"<h4>Respiratory support and treatments</h4>"
            f"<div class=\"plot\">{treatments_svg(stay, t0, t1, hours, after, cut_label)}"
            f"{urine_svg(stay, t0, t1, hours, after)}</div>"
            f"<h4>Laboratory values</h4>{labs_table(stay, t0, t1, hours, after)}</div>")


CSS = """
.casesheet h4{margin:14px 0 4px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.casesheet .plot{overflow-x:auto} .casesheet svg.sheet{width:100%;min-width:700px;height:auto;display:block}
.casesheet svg text{font-size:10.5px;fill:var(--muted)} .casesheet svg text.tt{fill:var(--ink);font-weight:600}
.casesheet svg .ax{stroke:var(--line)} .casesheet svg .gr{stroke:var(--grid)} .casesheet svg .tick{font-size:9.5px}
.casesheet svg .lg{font-weight:600} .casesheet svg .in{fill:var(--ink);font-size:10px;font-weight:600}
.casesheet svg .val{fill:var(--ink);font-size:10px} .casesheet svg .rate{font-size:9px;fill:var(--ink)}
.casesheet svg .dim,.casesheet .dim{color:var(--muted);fill:var(--muted)}
.casesheet svg .vaso{fill:#c0392b;opacity:.75} .casesheet svg .sed{fill:#7f8c8d;opacity:.7}
.casesheet svg .inf{fill:#2f6fde;opacity:.6} .casesheet svg .bolus{fill:#2f6fde}
.casesheet svg .rrt{stroke:#9b59b6;stroke-width:2} .casesheet svg .uo{fill:#d9a62b}
.casesheet svg .after{fill:var(--band)} .casesheet svg .cutl{stroke:var(--ink);stroke-dasharray:4 3}
.casesheet svg .cut{fill:var(--ink);font-weight:600} .casesheet svg .rec{fill:var(--muted)}
.casesheet dl.adm{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 6px}
.casesheet dl.adm div{display:flex;gap:6px} .casesheet dt{color:var(--muted)} .casesheet dd{margin:0;font-weight:600}
.casesheet table.labs{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12.5px;min-width:100%}
.casesheet table.labs th,.casesheet table.labs td{padding:3px 7px;border-bottom:1px solid var(--line);text-align:left}
.casesheet table.labs .num{text-align:right} .casesheet table.labs th{white-space:nowrap}
.casesheet table.labs th.aft{background:var(--band)}
.casesheet .u{color:var(--muted);font-size:11px} .casesheet .wrap{overflow-x:auto}
.casesheet .endnote{background:var(--band);padding:6px 10px;border-radius:6px;margin:4px 0}
.casesheet{overflow-wrap:anywhere}
"""


def window_has_data(stay, hours: float, start: float = 0.0) -> int:
    """Events in the window (selection: a sheet with almost nothing on it is not informative)."""
    return sum(1 for t, *_ in stay.timeline if start <= _h(stay, t) <= hours)
