"""The label engine (S2, ``factory.processes`` and ``factory.labels``) on hand-built stays with known answers: interval
merging, threshold runs (hold, sustained duration, bin edges), AND / OR, time to event with competing death and
censoring, the horizon reduction, multi-horizon, values in windows and changes, schedules and triggers, ordinal bands.
"""

from __future__ import annotations

import numpy as np

from icutwin.factory import labels as fl
from icutwin.factory import processes as fp
from icutwin.factory.channels import Channels

MAP_EDGES = [55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0]


def channels() -> Channels:
    names = ["LAB//creatinine", "OUTPUT//urine", "VITAL//map"]
    return Channels(names, ["LAB", "OUTPUT", "VITAL"], ["labs", "fluids", "vitals"], ["numeric"] * 3,
                    [[0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 4.0], [10.0, 50.0, 100.0], MAP_EDGES], [True] * 3)


def store():
    stays = [
        {"los": 1440, "end": "alive", "hosp_death": False,
         "events": [(0, "VITAL//map", 70), (60, "VITAL//map", 62), (120, "VITAL//map", 60), (180, "VITAL//map", 64),
                    (240, "VITAL//map", 75), (400, "VITAL//map", 58), (700, "VITAL//map", 59),
                    (100, "LAB//creatinine", 1.0), (700, "LAB//creatinine", 2.2),
                    (30, "OUTPUT//urine", 40), (90, "OUTPUT//urine", 60), (150, "OUTPUT//urine", 20)],
         "derived": [("ep:imv", 300, 900, None), ("kdigo", 100, 100, 0), ("kdigo", 700, 700, 2)]},
        {"los": 500, "end": "dead", "hosp_death": True, "events": [(10, "VITAL//map", 80)], "derived": []},
        {"los": 400, "end": "censored", "hosp_death": None, "events": [], "derived": []},
    ]
    return fp.MemStore(channels(), stays)


def test_merge_and_algebra() -> None:
    st, s, e = fp.merge([0, 0, 0, 1], [10, 30, 50, 0], [40, 45, 60, 5])
    assert st.tolist() == [0, 0, 1] and s.tolist() == [10, 50, 0] and e.tolist() == [45, 60, 5]
    st, s, e = fp.merge([0, 0], [0, 50], [50, 80])               # touching intervals merge
    assert s.tolist() == [0] and e.tolist() == [80]
    a = fp.from_intervals([0], [0], [100])
    b = fp.from_intervals([0], [50], [150])
    both, either = fp.intersect(a, b), fp.union(a, b)
    assert (both.s.tolist(), both.e.tolist()) == ([50], [100]) and both.on_t.tolist() == [50]
    assert (either.s.tolist(), either.e.tolist()) == ([0], [150])
    c, d = fp.from_intervals([0], [0], [50]), fp.from_intervals([0], [50], [80])
    assert len(fp.intersect(c, d).s) == 0 and fp.union(c, d).e.tolist() == [80]
    x, y = fp.from_intervals([0, 1], [0, 0], [10, 10]), fp.from_intervals([1], [5], [20])
    assert fp.intersect(x, y).stay.tolist() == [1]                # stays never mix


def test_threshold_runs() -> None:
    m = store()
    ev = m.channel_events(m.channels.index["VITAL//map"])
    # below the edge 65 (index 2): bins <= 2; on at 60, 120, 180 (stay 0), 400, 700 (stay 0)
    p = fp.threshold(ev, edge=2, direction="below", hold=120)
    assert p.on_stay.tolist() == [0, 0, 0] and p.on_t.tolist() == [60, 400, 700]
    assert p.s.tolist() == [60, 400, 700] and p.e.tolist() == [240, 520, 820]
    q = fp.threshold(ev, edge=2, direction="below", hold=120, sustain=90)       # sustained 90 minutes
    assert q.on_t.tolist() == [180] and (q.s.tolist(), q.e.tolist()) == ([180], [240])
    hi = fp.threshold(ev, edge=4, direction="above", hold=60)                   # >= 75: bins >= 5
    assert hi.on_t.tolist() == [240, 10] and hi.on_stay.tolist() == [0, 1]


def test_time_to_event_and_horizons() -> None:
    m = store()
    imv = fp.episodes(m.derived_events("ep:imv"))
    rows = fl.Rows(np.array([0, 0, 0, 0, 1, 2, 2]), np.array([60, 300, 600, 1000, 100, 100, 450]))
    b = fl.tte(m, imv, rows)
    assert b.y.tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert b.valid.tolist() == [True, False, False, True, True, True, False]
    assert b.reason[1] & fl.REASONS["in_event"] and b.reason[6] & fl.REASONS["not_in_icu"]
    assert b.t_event[0] == 4.0 and b.cause.tolist() == [0, 2, 2, 2, 1, 3, 3]
    assert b.t_resolved[0] == 300
    h4 = fl.at_horizon(b, rows, 240)
    assert h4.y[0] == 1 and h4.valid[0] and h4.t_resolved[0] == 300
    h3 = fl.at_horizon(b, rows, 180)
    assert h3.y[0] == 0 and h3.valid[0] and h3.t_resolved[0] == 240
    h10 = fl.at_horizon(b, rows, 600)
    assert h10.y[3] == 0 and h10.valid[3]                      # discharged alive before the horizon: no ICU event
    assert h10.valid[4] and h10.reason[4] & fl.REASONS["competing"]   # died before the horizon
    assert not h10.valid[5] and h10.reason[5] & fl.REASONS["censored"]  # administratively censored
    assert fl.at_horizon(b, rows, 120).valid[5]                 # followed past a short horizon
    first = fl.tte(m, imv, rows, ("in_icu", "never_before"))
    assert not first.valid[3] and first.reason[3] & fl.REASONS["after_event"]
    y, ok = fl.multi_horizon(b, rows, (60, 240, 600))
    assert y[0].tolist() == [0, 1, 1] and ok[5].tolist() == [True, True, False]


def test_death_and_discharge() -> None:
    m = store()
    rows = fl.Rows(np.array([0, 1, 2]), np.array([100, 100, 100]))
    d = fl.tte(m, fp.ends(m, "death"), rows)
    assert d.y.tolist() == [0, 1, 0] and d.t_event[1] == np.float32(400 / 60) and d.cause.tolist() == [2, 0, 3]
    alive = fl.tte(m, fp.ends(m, "discharge"), rows)
    assert alive.y.tolist() == [1, 0, 0]
    hd = fl.hospital_death(m, rows)
    assert hd.y[:2].tolist() == [0, 1] and not hd.valid[2]


def test_values_and_changes() -> None:
    m = store()
    ev = m.channel_events(m.channels.index["OUTPUT//urine"])
    rows = fl.Rows(np.array([0, 0]), np.array([0, 100]))
    total, cnt = fl.window_values(ev, rows, 0, 120, "sum")
    assert total.tolist() == [100.0, 20.0] and cnt.tolist() == [2, 1]
    for agg, want in (("last", 60), ("first", 40), ("min", 40), ("max", 60), ("mean", 50), ("count", 2)):
        assert fl.window_values(ev, rows.take([0]), 0, 120, agg)[0][0] == want
    cr = m.channel_events(m.channels.index["LAB//creatinine"])
    v = fl.value_target(m, cr, fl.Rows(np.array([0, 0, 1]), np.array([200, 1400, 10])), 0, 600)
    assert v.y[0] == np.float32(2.2) and v.valid.tolist() == [True, False, False]
    assert v.reason[1] & fl.REASONS["censored"] and v.reason[2] & fl.REASONS["no_value"]
    ch = fl.value_target(m, cr, fl.Rows(np.array([0]), np.array([200])), 0, 600, baseline=240)
    assert abs(ch.y[0] - 1.2) < 1e-5 and ch.t_resolved[0] == 800
    # to the end of the stay (b < 0): the worst MAP after t until the discharge, known at the discharge; a stay
    # without a discharge or death (censored) has no end to read to
    mp = m.channel_events(m.channels.index["VITAL//map"])
    end = fl.value_target(m, mp, fl.Rows(np.array([0, 0, 1, 2]), np.array([100, 1000, 0, 0])), 0, -1, "min")
    assert end.y[0] == np.float32(58) and end.t_resolved[0] == 1440 and end.valid.tolist() == [True, False, True, False]
    assert end.reason[1] & fl.REASONS["no_value"] and end.reason[3] & fl.REASONS["censored"]
    assert end.y[2] == np.float32(80) and end.t_resolved[2] == 500


def test_schedules_triggers_bands() -> None:
    m = store()
    r = fl.rolling(m, 60)
    assert r.t[r.stay == 2].tolist() == [60, 120, 180, 240, 300, 360] and r.t[r.stay == 0].max() == 1380
    assert fl.landmark(m, 450).stay.tolist() == [0, 1]
    imv = fp.episodes(m.derived_events("ep:imv"))
    ext = fl.triggers(m, imv, "ends")
    assert (ext.stay.tolist(), ext.t.tolist()) == ([0], [900])
    los = fl.remaining_los(m, fl.Rows(np.array([0, 2]), np.array([240, 100])))
    assert los.y[0] == 20.0 and not los.valid[1]
    k = fl.stage_max(m, m.derived_events("kdigo"), fl.Rows(np.array([0, 0]), np.array([0, 200])), 0, 600)
    assert k.y.tolist() == [0.0, 2.0]
    bd = fl.band(k, [1, 2, 3])
    assert bd.y.tolist() == [0.0, 2.0]


def test_value_targets_skip_events_without_a_value() -> None:
    m = store()
    ev = m.channel_events(m.channels.index["LAB//creatinine"])
    ev = {k: np.r_[x, x[-1:]] for k, x in ev.items()}               # a later event of the channel without a value
    ev["t"][-1], ev["v"][-1] = 800, np.nan
    v = fl.value_target(m, ev, fl.Rows(np.array([0]), np.array([200])), 0, 600)
    assert v.valid[0] and v.y[0] == np.float32(2.2)
