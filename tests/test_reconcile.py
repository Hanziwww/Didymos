"""S3 reconciliation engine on hand-built community events and CLIF rows."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.clif import reconcile as rc
from icutwin.clif.bridges import MERGED_IDENTICAL

T = lambda h, m=0: datetime(2150, 1, 2, h, m)  # noqa: E731


def _community(root: Path, rows: list[tuple]) -> Path:
    """MEDS events: (subject_id, time, code, numeric_value) with float32 values, as the community ETLs write."""
    (root / "data" / "train").mkdir(parents=True)
    pq.write_table(pa.table({
        "subject_id": pa.array([r[0] for r in rows], pa.int64()),
        "time": pa.array([r[1] for r in rows], pa.timestamp("us")),
        "code": pa.array([r[2] for r in rows], pa.string()),
        "numeric_value": pa.array([r[3] for r in rows], pa.float32()),
        "text_value": pa.array([None] * len(rows), pa.string())}), root / "data" / "train" / "0.parquet")
    return root


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("reconcile")
    clif = root / "clif"
    clif.mkdir()
    utc = lambda d: d.replace(tzinfo=UTC)  # noqa: E731
    pd.DataFrame({"patient_id": ["1"], "hospitalization_id": ["10"], "admission_dttm": [utc(datetime(2150, 1, 2))],
                  "discharge_dttm": [utc(datetime(2150, 1, 3))]}).to_parquet(clif / "clif_hospitalization.parquet")
    ours = [  # (time, name, value)
        (T(1), "hr", 80.0),                                   # pairs (float32 community copy)
        (T(2), "k", 7.2), (T(2), "k", 6.9), (T(2), "k", 6.9),  # the community merged the two identical 6.9
        (T(3), "temp_dc", 74.0),                              # unit factor 2 (ours = 2 x community)
        (T(4), "hr", 16.6545),                                # a 5-digit rounding boundary in float32
        (T(5), "hr", 90.0),                                   # only ours, unexplained
    ]
    pd.DataFrame({"hospitalization_id": "10", "recorded_dttm": [utc(t) for t, _, _ in ours],
                  "vital_name": [n for _, n, _ in ours], "vital_value": [v for _, _, v in ours]}
                 ).to_parquet(clif / "clif_vitals.parquet")
    community = _community(root / "community", [
        (1, T(1), "V//hr", 80.0),
        (1, T(2), "V//k", 7.2), (1, T(2), "V//k", 6.9),
        (1, T(3), "V//temp_dc", 37.0),
        (1, T(4), "V//hr", 16.6545),
        (1, T(6), "V//hr", None),                             # explained: no value
        (1, T(7), "V//hr", 70.0),                             # unexplained
        (1, datetime(2150, 1, 9), "V//hr", 60.0),             # outside the hospitalization
    ])
    bridge = rc.Bridge(
        "vitals",
        "SELECT pid, time AS t, split_part(code, '//', 2) AS item, numeric_value AS value FROM ev",
        """SELECT h.patient_id AS pid, x.recorded_dttm AS t, x.vital_name AS item, x.vital_value AS value
           FROM t_vitals x JOIN t_hospitalization h USING (hospitalization_id)""",
        explain_community=(("no_value", "u.value IS NULL"),),
        explain_ours=(MERGED_IDENTICAL,))
    rep = rc.reconcile("demo", community, clif, [bridge], work_dir=root / "work")
    rc.write([rep], root / "qa")
    assert (root / "qa" / "reconciliation.md").exists()
    return rep["tables"]["vitals"]


def test_counts_and_classification(report: dict) -> None:
    assert (report["community"], report["ours"], report["paired"]) == (8, 7, 5)
    assert report["community_out_of_scope"] == 1
    assert report["only_community_explained"] == {"no_value": 1}
    assert report["only_ours_explained"] == {"community_merges_identical_events": 1}
    assert (report["only_community_unexplained"], report["only_ours_unexplained"]) == (1, 1)


def test_values_pair_by_value_before_order(report: dict) -> None:
    # order alone would pair 7.2 with the second 6.9; value pairing leaves the duplicate 6.9 unpaired
    items = {i["item"]: i for i in report["items"]}
    assert items["k"]["values_agree"] == items["k"]["values_compared"] == 2
    assert items["temp_dc"]["factor"] == [2.0]
    assert items["hr"]["values_agree"] == items["hr"]["values_compared"] == 2
    assert report["values_agree"] == 1.0


def test_values_straddling_a_rounding_boundary_pair_by_value(tmp_path: Path) -> None:
    # the community merged our two identical 1.0 rows; its float32 1.2345501 and our 1.23454999 round to 1.2346 and
    # 1.2345 but truncate alike: they pair by value (by order the 1.2345501 would take our second 1.0)
    utc = lambda d: d.replace(tzinfo=UTC)  # noqa: E731
    clif = tmp_path / "clif"
    clif.mkdir()
    pd.DataFrame({"patient_id": ["1"], "hospitalization_id": ["10"], "admission_dttm": [utc(datetime(2150, 1, 2))],
                  "discharge_dttm": [utc(datetime(2150, 1, 3))]}).to_parquet(clif / "clif_hospitalization.parquet")
    pd.DataFrame({"hospitalization_id": "10", "recorded_dttm": [utc(T(1))] * 3, "vital_name": "x",
                  "vital_value": [1.0, 1.0, 1.2345499992370605]}).to_parquet(clif / "clif_vitals.parquet")
    community = _community(tmp_path / "c", [(1, T(1), "V//x", 1.0), (1, T(1), "V//x", 1.23455012)])
    bridge = rc.Bridge(
        "vitals", "SELECT pid, time AS t, split_part(code, '//', 2) AS item, numeric_value AS value FROM ev",
        """SELECT h.patient_id AS pid, x.recorded_dttm AS t, x.vital_name AS item, x.vital_value AS value
           FROM t_vitals x JOIN t_hospitalization h USING (hospitalization_id)""",
        explain_ours=(MERGED_IDENTICAL,))
    m = rc.reconcile("demo", community, clif, [bridge], work_dir=tmp_path / "work")["tables"]["vitals"]
    assert (m["paired"], m["only_ours_unexplained"], m["values_agree"]) == (2, 0, 1.0)
    assert m["only_ours_explained"] == {"community_merges_identical_events": 1}


def _clif_meds(clif: Path, table: str, rows: list[tuple], extra: dict | None = None) -> None:
    utc = lambda d: d.replace(tzinfo=UTC)  # noqa: E731
    clif.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": ["1"], "hospitalization_id": ["10"], "admission_dttm": [utc(datetime(2150, 1, 2))],
                  "discharge_dttm": [utc(datetime(2150, 1, 2, 12))]}).to_parquet(clif / "clif_hospitalization.parquet")
    pd.DataFrame({"hospitalization_id": "10", "admin_dttm": [utc(r[0]) for r in rows], "med_name": [r[1] for r in rows],
                  "med_dose": [r[2] for r in rows], **(extra or {})}).to_parquet(clif / f"clif_{table}.parquet")


def test_a_pair_key_keeps_rows_of_two_orders_apart(tmp_path: Path) -> None:
    """Two orders end at one time; one restarts. Pairing per (time, item) would hand our stop row to either end;
    with the order id as pair key it goes to its own order's end."""
    _clif_meds(tmp_path / "clif", "medication_admin_continuous", [(T(5), "nacl", None), (T(5), "nacl", 26.0)],
               {"med_order_id": ["B", "A"]})
    community = _community(tmp_path / "c", [(1, T(5), "END//nacl//A", None), (1, T(5), "START//nacl//A", 26.0),
                                            (1, T(5), "END//nacl//B", None)])
    bridge = rc.Bridge(
        "medication_admin_continuous",
        """SELECT pid, time AS t, split_part(code, '//', 2) AS item, numeric_value AS value,
                  split_part(code, '//', 3) AS pair_key, split_part(code, '//', 1) AS role FROM ev""",
        """SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item, x.med_dose AS value, x.med_order_id AS pair_key
           FROM t_medication_admin_continuous x JOIN t_hospitalization h USING (hospitalization_id)""",
        explain_community=(("end_at_the_orders_next_start",
                            "u.role = 'END' AND EXISTS (SELECT 1 FROM c_all c WHERE c.pair_key = u.pair_key AND c.role = 'START')"),),
        name="continuous (orders)")
    rep = rc.reconcile("demo", community, tmp_path / "clif", [bridge], work_dir=tmp_path / "work")
    m = rep["tables"]["continuous (orders)"]
    assert (m["paired"], m["only_community_unexplained"], m["only_ours_unexplained"]) == (2, 0, 0)
    assert m["only_community_explained"] == {"end_at_the_orders_next_start": 1}


def _streams(tmp_path: Path, records: list[tuple], ours: list[tuple], **kw) -> dict:
    _clif_meds(tmp_path / "clif", "medication_admin_continuous", ours)
    community = _community(tmp_path / "c", [(1, t0, f"REC//{key}//{t1.isoformat()}", rate) for t0, t1, key, rate in records])
    s = rc.Streams(
        "streams", "medication_admin_continuous",
        """SELECT pid, split_part(code, '//', 2) AS key, time AS t0, CAST(split_part(code, '//', 3) AS TIMESTAMP) AS t1,
                  numeric_value AS rate, CASE WHEN split_part(code, '//', 2) = 'flush' THEN 'flush' END AS reason
           FROM ev""",
        """SELECT h.patient_id AS pid, x.med_name AS key, x.admin_dttm AS t, x.med_dose AS rate
           FROM t_medication_admin_continuous x JOIN t_hospitalization h USING (hospitalization_id)""", **kw)
    return rc.reconcile("demo", community, tmp_path / "clif", [], work_dir=tmp_path / "work", streams=[s])["streams"]["streams"]


def test_streams_compare_summed_records_with_our_rate_steps(tmp_path: Path) -> None:
    # two overlapping syringes (2 + 1 mg/h) against our summed stream; a flush is left out by its reason; the
    # second key differs by one hour of 1 mg/h; the third runs past the discharge (12:00): both sides count to it
    m = _streams(tmp_path,
                 [(T(1), T(3), "ne", 2.0), (T(2), T(4), "ne", 1.0), (T(1), T(2), "flush", 5.0),
                  (T(1), T(3), "ins", 1.0), (T(11), T(14), "pro", 4.0)],
                 [(T(1), "ne", 2.0), (T(2), "ne", 3.0), (T(3), "ne", 1.0), (T(4), "ne", 0.0),
                  (T(1), "ins", 1.0), (T(4), "ins", 0.0), (T(11), "pro", 4.0)])
    assert (m["keys"], m["keys_agree"], m["unexplained"]) == (3, 2, 1)
    assert m["community_records_excluded"] == {"flush": 1}
    by = {i["item"]: i for i in m["keys_by_item"]}
    assert by["ne"]["community_total"] == by["ne"]["our_total"] == (2 * 2 + 1 * 2) * 60   # mg/h x minutes
    assert (by["ins"]["community_total"], by["ins"]["our_total"]) == (120, 180)
    assert by["pro"]["community_total"] == by["pro"]["our_total"] == 4 * 60  # clipped at the discharge


def test_streams_count_a_minute_inside_two_windows_once(tmp_path: Path) -> None:
    # the patient's second hospitalization begins 2 h after the first ends, so its window (from before admission)
    # covers the first one: ours exceeds the community by 1 mg/h for 2 h, a difference of 120 mg/h x min, not 240
    utc = lambda d: d.replace(tzinfo=UTC)  # noqa: E731
    clif = tmp_path / "clif"
    _clif_meds(clif, "medication_admin_continuous", [(T(1), "ins", 2.0), (T(3), "ins", 0.0)])
    pd.DataFrame({"patient_id": ["1", "1"], "hospitalization_id": ["10", "11"],
                  "admission_dttm": [utc(datetime(2150, 1, 2)), utc(datetime(2150, 1, 2, 14))],
                  "discharge_dttm": [utc(datetime(2150, 1, 2, 12)), utc(datetime(2150, 1, 3))]}
                 ).to_parquet(clif / "clif_hospitalization.parquet")
    community = _community(tmp_path / "c", [(1, T(1), f"REC//ins//{T(3).isoformat()}", 1.0)])
    s = rc.Streams("streams", "medication_admin_continuous",
                   "SELECT pid, split_part(code, '//', 2) AS key, time AS t0, CAST(split_part(code, '//', 3) AS TIMESTAMP) AS t1, "
                   "numeric_value AS rate, CAST(NULL AS VARCHAR) AS reason FROM ev",
                   """SELECT h.patient_id AS pid, x.med_name AS key, x.admin_dttm AS t, x.med_dose AS rate
                      FROM t_medication_admin_continuous x JOIN t_hospitalization h USING (hospitalization_id)""",
                   explain=(("one_extra_mg_per_hour_for_2_h", "abs(k.diff - 120) < 1e-9"),))
    m = rc.reconcile("demo", community, clif, [], work_dir=tmp_path / "work", streams=[s])["streams"]["streams"]
    assert (m["integral_community"], m["integral_ours"], m["integral_abs_difference"]) == (120, 240, 120)
    assert (m["unexplained"], m["explained"]) == (0, {"both: one_extra_mg_per_hour_for_2_h": 1})


def test_streams_explain_and_compare_totals(tmp_path: Path) -> None:
    # doses at points in time (totals only): 10 + 20 against a 30-unit rate stream over another interval
    m = _streams(tmp_path / "a", [(T(1), T(1, 1), "dose", 10.0), (T(5), T(5, 1), "dose", 20.0)],
                 [(T(2), "dose", 0.5), (T(3), "dose", 0.0)], totals_only=True)
    assert (m["keys_agree"], m["unexplained"]) == (1, 0)
    m = _streams(tmp_path / "b", [(T(1), T(2), "ins", 1.0)], [(T(1), "ins", 2.0), (T(2), "ins", 0.0)],
                 explain=(("documented", "k.o_total = 2 * k.c_total"),))
    assert (m["keys_agree"], m["unexplained"], m["explained"]) == (0, 0, {"both: documented": 1})
    # a dose recorded after the discharge (12:00), given since 11:00: totals compare over the whole stream, where
    # the window would keep half of our segment and none of the community's dose
    m = _streams(tmp_path / "c", [(T(12, 59), T(13), "dose", 10.0)], [(T(11), "dose", 10 / 120), (T(13), "dose", 0.0)],
                 totals_only=True)
    assert (m["keys_agree"], m["unexplained"], m["community_out_of_scope"]) == (1, 0, 0)
