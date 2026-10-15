"""Paper P amendment 10 (``docs/p1_paper_protocol.md``): traditional clinical scores and bedside indicators at a
prediction time, from per-anchor variables (``tools/paper_clinical.py`` extracts them from canonical CLIF).

Every function takes numpy arrays (NaN: not observed) and returns float arrays; an unobserved component scores 0
(normal), as mimic-code and ricu do.

* ``news``, ``mews``, ``qsofa``, ``sirs``: ricu's ``news_score``, ``mews_score``, ``qsofa_score`` and ``sirs_score``
  (``refs/ricu``, the pinned commit) on the latest values. Thresholds are upper-inclusive breaks (ricu ``map_vals``);
  AVPU comes from the GCS total as ricu's ``avpu`` does (3 U, 4-9 P, 10-13 V, 14-15 A). SIRS has no band count.
* ``sapsii``, ``oasis``: mimic-code's ``sapsii`` and ``oasis`` (``refs/mimic-code``, the pinned commit) on the worst
  values of the window, with their probability equations. SAPS II's chronic-disease items are left out: only
  discharge diagnoses record them.
* ``saps3_prob``: SAPS 3's general probability equation (Moreno et al., Intensive Care Med 2005).
* ``kdigo_creatinine``: the creatinine rules of mimic-code's ``kdigo_stages`` (as ``meds.derived.kdigo``);
  ``shock_index``; ``rox`` (Roca et al. 2016).
"""

from __future__ import annotations

import numpy as np

# SAPS II admission type and OASIS elective surgery from CLIF's surgical status (unknown: no points, as mimic-code)
SAPSII_ADMISSION = {"elective_surgery": 0.0, "nonsurgical": 6.0, "emergency_surgery": 8.0}
OASIS_SURGERY = {"elective_surgery": 0.0, "nonsurgical": 6.0, "emergency_surgery": 6.0}


def _a(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def map_vals(x, breaks, points) -> np.ndarray:
    """ricu ``map_vals``: ``points[i]`` for x in (``breaks[i-1]``, ``breaks[i]``] (``points[0]`` up to and including
    ``breaks[0]``, ``points[-1]`` above the last); an unobserved x scores 0."""
    x = _a(x)
    miss = np.isnan(x)
    idx = np.searchsorted(_a(breaks), np.where(miss, 0.0, x), side="left")
    return np.where(miss, 0.0, _a(points)[idx])


def _select(conds, values) -> np.ndarray:
    """The value of the first true condition (SQL ``CASE``), else 0; comparisons with NaN are false."""
    return np.select([np.asarray(c, dtype=bool) for c in conds], [float(v) for v in values], 0.0)


def avpu(gcs) -> np.ndarray:
    """ricu ``avpu``: 0 A (GCS 14-15), 1 V (10-13), 2 P (4-9), 3 U (3); NaN outside 3-15 or unobserved."""
    g = _a(gcs)
    out = np.full(g.shape, np.nan)
    for lo, hi, level in ((2, 3, 3), (3, 9, 2), (9, 13, 1), (13, 15, 0)):
        out[(g > lo) & (g <= hi)] = level
    return out


def news(rr, spo2, supp_o2, temp, sbp, hr, gcs) -> np.ndarray:
    """NEWS (ricu ``news_score``; the NEWS2 SpO2 scale 1): 0-20."""
    a = avpu(gcs)
    return (map_vals(rr, (8, 11, 20, 24), (3, 1, 0, 2, 3)) + map_vals(spo2, (91, 93, 95), (3, 2, 1, 0))
            + np.where(_a(supp_o2) > 0, 2.0, 0.0) + map_vals(temp, (35, 36, 38, 39), (3, 1, 0, 1, 2))
            + map_vals(sbp, (90, 100, 110, 219), (3, 2, 1, 0, 3)) + map_vals(hr, (40, 50, 90, 110, 130), (3, 1, 0, 1, 2, 3))
            + np.where(a > 0, 3.0, 0.0))


def mews(sbp, hr, rr, temp, gcs) -> np.ndarray:
    """MEWS (ricu ``mews_score``): 0-14."""
    return (map_vals(sbp, (70, 80, 100, 199), (3, 2, 1, 0, 2)) + map_vals(hr, (40, 50, 100, 110, 129), (2, 1, 0, 1, 2, 3))
            + map_vals(rr, (9, 14, 20, 29), (2, 0, 1, 2, 3)) + map_vals(temp, (35, 38.4), (2, 0, 2))
            + np.nan_to_num(avpu(gcs), nan=0.0))


def qsofa(rr, sbp, gcs) -> np.ndarray:
    """qSOFA (ricu ``qsofa_score``): respiratory rate >= 22, systolic pressure <= 100, GCS <= 13; 0-3."""
    return (_a(rr) >= 22).astype(float) + (_a(sbp) <= 100).astype(float) + (_a(gcs) <= 13).astype(float)


def sirs(temp, hr, rr, paco2, wbc) -> np.ndarray:
    """SIRS (ricu ``sirs_score``, without bands): temperature < 36 or > 38; heart rate > 90; respiratory rate > 20 or
    PaCO2 < 32; WBC < 4 or > 12; 0-4."""
    t, h, r, p, w = map(_a, (temp, hr, rr, paco2, wbc))
    return (((t < 36) | (t > 38)).astype(float) + (h > 90).astype(float) + ((r > 20) | (p < 32)).astype(float)
            + ((w < 4) | (w > 12)).astype(float))


def sapsii(age, hr_min, hr_max, sbp_min, sbp_max, temp_max, pf_vent_min, urine_ml, bun_max, wbc_min, wbc_max, k_min,
           k_max, na_min, na_max, hco3_min, bili_max, gcs_min, surgical_status) -> np.ndarray:
    """SAPS II points (mimic-code ``sapsii``, without the chronic-disease items). ``pf_vent_min``: the lowest PaO2/FiO2
    under ventilation or CPAP (NaN otherwise); ``urine_ml``: the window's urine output; ``surgical_status``: CLIF's."""
    a, hn, hx, sn, sx, tx, pf, uo, bun, wn, wx, kn, kx, nn, nx, bc, bi, g = map(_a, (
        age, hr_min, hr_max, sbp_min, sbp_max, temp_max, pf_vent_min, urine_ml, bun_max, wbc_min, wbc_max, k_min,
        k_max, na_min, na_max, hco3_min, bili_max, gcs_min))
    adm = np.array([SAPSII_ADMISSION.get(str(s), 0.0) for s in np.asarray(surgical_status, dtype=object)])
    return (_select([a < 40, a < 60, a < 70, a < 75, a < 80, a >= 80], [0, 7, 12, 15, 16, 18])
            + _select([hn < 40, hx >= 160, hx >= 120, hn < 70], [11, 7, 4, 2])
            + _select([sn < 70, sn < 100, sx >= 200], [13, 5, 2])
            + _select([tx >= 39], [3])
            + _select([pf < 100, pf < 200, pf >= 200], [11, 9, 6])
            + _select([uo < 500, uo < 1000], [11, 4])
            + _select([bun < 28, bun < 84, bun >= 84], [0, 6, 10])
            + _select([wn < 1, wx >= 20], [12, 3])
            + _select([kn < 3, kx >= 5], [3, 3])
            + _select([nn < 125, nx >= 145], [5, 1])
            + _select([bc < 15, bc < 20], [6, 3])
            + _select([bi < 4, bi < 6, bi >= 6], [0, 4, 9])
            + _select([g < 3, g < 6, g < 9, g < 11, g < 14], [0, 26, 13, 7, 5])
            + adm)


def sapsii_prob(score) -> np.ndarray:
    s = _a(score)
    return 1.0 / (1.0 + np.exp(-(-7.7631 + 0.0737 * s + 0.9971 * np.log(s + 1.0))))


def oasis(pre_icu_minutes, age, gcs_min, hr_min, hr_max, map_min, map_max, rr_min, rr_max, temp_min, temp_max,
          urine_ml, ventilated, surgical_status) -> np.ndarray:
    """OASIS points (mimic-code ``oasis``). ``ventilated``: invasive ventilation in the window (0/1)."""
    m, a, g, hn, hx, mn, mx, rn, rx, tn, tx, uo, v = map(_a, (
        pre_icu_minutes, age, gcs_min, hr_min, hr_max, map_min, map_max, rr_min, rr_max, temp_min, temp_max, urine_ml,
        ventilated))
    surg = np.array([OASIS_SURGERY.get(str(s), 0.0) for s in np.asarray(surgical_status, dtype=object)])
    return (_select([m < 10.2, m < 297, m < 1440, m < 18708, m >= 18708], [5, 3, 0, 2, 1])
            + _select([a < 24, a <= 53, a <= 77, a <= 89, a >= 90], [0, 3, 6, 9, 7])
            + _select([g <= 7, g < 14, g == 14], [10, 4, 3])
            + _select([hx > 125, hn < 33, (hx >= 107) & (hx <= 125), (hx >= 89) & (hx <= 106)], [6, 4, 3, 1])
            + _select([mn < 20.65, mn < 51, mx > 143.44, (mn >= 51) & (mn < 61.33)], [4, 3, 3, 2])
            + _select([rn < 6, rx > 44, rx > 30, rx > 22, rn < 13], [10, 9, 6, 1, 1])
            + _select([tx > 39.88, (tn >= 33.22) & (tn <= 35.93), (tx >= 33.22) & (tx <= 35.93), tn < 33.22,
                       (tn > 35.93) & (tn <= 36.39), (tx >= 36.89) & (tx <= 39.88)], [6, 4, 4, 3, 2, 2])
            + _select([uo < 671.09, uo > 6896.80, (uo >= 671.09) & (uo <= 1426.99), (uo >= 1427.00) & (uo <= 2544.14)],
                      [10, 8, 5, 1])
            + _select([v == 1], [9])
            + surg)


def oasis_prob(score) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(-6.1746 + 0.1275 * _a(score))))


def saps3_prob(score) -> np.ndarray:
    """SAPS 3's general equation: logit = -32.6659 + ln(score + 20.5958) x 7.3068."""
    return 1.0 / (1.0 + np.exp(-(-32.6659 + np.log(_a(score) + 20.5958) * 7.3068)))


def kdigo_creatinine(creat, low48, low7) -> np.ndarray:
    """The KDIGO creatinine stage of a value against the lowest of the previous 48 h and 7 days (mimic-code
    ``kdigo_stages``, as ``meds.derived.kdigo``); 0 where unobserved."""
    c, l48, l7 = map(_a, (creat, low48, low7))
    return _select([c >= l7 * 3.0, (c >= 4) & ((c >= l48 + 0.3) | (c >= 1.5 * l7)), c >= l7 * 2.0, c >= l48 + 0.3,
                    c >= l7 * 1.5], [3, 3, 2, 1, 1])


def shock_index(hr, sbp) -> np.ndarray:
    h, s = _a(hr), _a(sbp)
    return np.where(s > 0, h / np.where(s > 0, s, 1.0), np.nan)


def rox(spo2, fio2, rr) -> np.ndarray:
    """ROX index (Roca et al. 2016): SpO2 / FiO2 (a fraction) over the respiratory rate; lower is worse."""
    s, f, r = _a(spo2), _a(fio2), _a(rr)
    ok = (f > 0) & (r > 0)
    return np.where(ok, s / np.where(ok, f, 1.0) / np.where(ok, r, 1.0), np.nan)
