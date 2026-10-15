"""One infusion stream per MEDS code (S4): canonical streams of one drug that land on one code are summed."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from icutwin.clif.conform import connect
from icutwin.meds.streams import restream_infusions

T = lambda h, m=0: datetime(2100, 1, 1, h, m)  # noqa: E731


def _file(path: Path, rows: list[tuple]) -> Path:
    pq.write_table(pa.table({
        "subject_id": pa.array([r[0] for r in rows], pa.int64()),
        "time": pa.array([r[1] for r in rows], pa.timestamp("us")),
        "code": [r[2] for r in rows],
        "numeric_value": pa.array([r[3] for r in rows], pa.float32()),
        "text_value": [r[4] for r in rows],
        "hospitalization_id": pa.array([10] * len(rows), pa.int64())}), path)
    return path


def _events(path: Path) -> list[tuple]:
    t = pq.read_table(path).to_pylist()
    assert all(r["text_value"] is None for r in t)   # the stream id is blanked
    return sorted((r["subject_id"], r["time"].hour, r["time"].minute, r["code"].split("//")[3], r["numeric_value"])
                  for r in t if "insulin" in r["code"] or "norepinephrine" in r["code"] or "propofol" in r["code"])


def _consistent(events: list[tuple]) -> bool:
    running = False
    for *_k, action, _v in events:
        if action in ("dose_change", "stop") and not running:
            return False
        running = action != "stop"
    return True


def test_streams_of_two_units_on_one_code_are_summed(tmp_path: Path) -> None:
    ins = lambda a, s: f"MED_CON//insulin//u/hr//{a}//iv"  # noqa: E731
    rows = [  # stream u/hr: 2 u/hr from 8:00, 1 u/hr from 10:00, stop 12:00; stream u/min (converted): 3 u/hr 9:00-11:00
        (1, T(8), ins("start", 0), 2.0, "u/hr"), (1, T(10), ins("dose_change", 0), 1.0, "u/hr"),
        (1, T(12), ins("stop", 0), 0.0, "u/hr"),
        (1, T(9), ins("start", 0), 3.0, "u/min"), (1, T(11), ins("stop", 0), 0.0, "u/min"),
        # a unit switch at one time: one stream stops as the other starts at the same rate -> no event
        (1, T(13), ins("start", 0), 2.0, "u/hr"), (1, T(14), ins("stop", 0), 0.0, "u/hr"),
        (1, T(14), ins("start", 0), 2.0, "u/min"), (1, T(15), ins("stop", 0), 0.0, "u/min"),
        # one stream, one code: unchanged
        (2, T(8), "MED_CON//propofol//mcg/kg/min//start//iv", 20.0, "mcg/kg/min"),
        (2, T(9), "MED_CON//propofol//mcg/kg/min//stop//iv", 0.0, "mcg/kg/min")]
    path = _file(tmp_path / "MED_CON.parquet", rows)
    report = restream_infusions(connect(), path)
    ev = _events(path)
    assert [e for e in ev if e[0] == 1] == [
        (1, 8, 0, "start", 2.0), (1, 9, 0, "dose_change", 5.0), (1, 10, 0, "dose_change", 4.0),
        (1, 11, 0, "dose_change", 1.0), (1, 12, 0, "stop", 0.0), (1, 13, 0, "start", 2.0), (1, 15, 0, "stop", 0.0)]
    assert [e for e in ev if e[0] == 2] == [(2, 8, 0, "start", 20.0), (2, 9, 0, "stop", 0.0)]
    assert _consistent([e for e in ev if e[0] == 1])
    assert report["drugs_summed"] == 1 and report["events_in"] == 9 and report["events_out"] == 7


def test_a_stream_split_across_codes_stops_where_it_moves(tmp_path: Path) -> None:
    """A weight-based conversion without a weight at first: the stream's events move from one code to another;
    the first code stops there, the second starts; an unknown converted rate is NULL."""
    ne = lambda u, a: f"MED_CON//norepinephrine//{u}//{a}//iv"  # noqa: E731
    rows = [(1, T(8), ne("mcg/min", "start"), 8.0, "mcg/min"),
            (1, T(9), ne("mcg/kg/min", "dose_change"), 0.12, "mcg/min"),
            (1, T(10), ne("mcg/kg/min", "dose_change"), None, "mcg/min"),
            (1, T(11), ne("mcg/kg/min", "stop"), 0.0, "mcg/min")]
    path = _file(tmp_path / "MED_CON.parquet", rows)
    restream_infusions(connect(), path)
    t = sorted((r["time"].hour, r["code"], r["numeric_value"]) for r in pq.read_table(path).to_pylist())
    assert [(h, c.split("//")[2], c.split("//")[3], v if v is None else round(v, 4)) for h, c, v in t] == [
        (8, "mcg/min", "start", 8.0), (9, "mcg/kg/min", "start", 0.12), (9, "mcg/min", "stop", 0.0),
        (10, "mcg/kg/min", "dose_change", None), (11, "mcg/kg/min", "stop", 0.0)]


def test_presence_streams_keep_their_confirmations(tmp_path: Path) -> None:
    ne = lambda a: f"MED_CON//norepinephrine//mcg/kg/min//{a}//iv"  # noqa: E731
    rows = [(1, T(8), ne("start"), None, "mcg/kg/min"), (1, T(9), ne("going"), None, "mcg/kg/min"),
            (1, T(9, 30), ne("start"), None, "mcg/min"), (1, T(10), ne("stop"), 0.0, "mcg/kg/min"),
            (1, T(11), ne("going"), None, "mcg/min"), (1, T(12), ne("stop"), 0.0, "mcg/min")]
    path = _file(tmp_path / "MED_CON.parquet", rows)
    restream_infusions(connect(), path)
    ev = _events(path)
    assert [(h, m, a) for _s, h, m, a, _v in ev] == [(8, 0, "start"), (9, 0, "going"), (11, 0, "going"), (12, 0, "stop")]
    assert _consistent(ev)
