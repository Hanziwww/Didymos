"""Weight causality gate of the training MEDS (``meds.weights``): a per-kg medication value before the
hospitalization's first weight must be the source's own unit; a converted one used a weight charted later."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from icutwin.meds.weights import weight_causality


def _t(h: int, m: int = 0) -> datetime:
    return datetime(2100, 1, 1, h, m)


def _write(clif: Path, meds: Path, rows: list[tuple]) -> None:
    clif.mkdir(parents=True)
    (meds / "data" / "train").mkdir(parents=True)
    utc = lambda t: t.replace(tzinfo=timezone.utc)  # noqa: E731
    pq.write_table(pa.table({"hospitalization_id": ["7"], "recorded_dttm": [utc(_t(10))],
                             "vital_category": ["weight_kg"], "vital_value": [80.0]}), clif / "clif_vitals.parquet")
    src = [  # (time, category, source unit)
        (_t(5), "norepinephrine", "mcg/kg/min"),     # charted per kg: reported (window-start going below), although
        (_t(6), "norepinephrine", "mcg/min"),        # the same drug's last row before the window is another stream
        (_t(8), "propofol", "mcg/min"),              # unweighted at the source
        (_t(9), "dexmedetomidine", "mcg/kg/hr"),     # charted per kg
        (_t(11), "propofol", "mcg/min"),             # after the weight: converted causally
    ]
    pq.write_table(pa.table({"hospitalization_id": ["7"] * len(src), "admin_dttm": [utc(t) for t, _, _ in src],
                             "med_category": [c for _, c, _ in src], "med_dose_unit": [u for _, _, u in src]}),
                   clif / "clif_medication_admin_continuous.parquet")
    pq.write_table(pa.table({"subject_id": [1] * len(rows), "hospitalization_id": [7] * len(rows),
                             "time": [t for t, _ in rows], "code": [c for _, c in rows],
                             "numeric_value": [1.0] * len(rows)}), meds / "data" / "train" / "0.parquet")


def test_converted_values_before_the_first_weight_are_violations(tmp_path: Path) -> None:
    causal = [(_t(7), "MED_CON//norepinephrine//mcg/kg/min//going//iv"),      # window start, per kg at the source
              (_t(8), "MED_CON//propofol//mcg/min//start//iv"),                 # no weight yet: unweighted unit
              (_t(9), "MED_CON//dexmedetomidine//mcg/kg/hr//start//iv"),
              (_t(11), "MED_CON//propofol//mcg/kg/min//start//iv")]             # a new segment after the weight
    _write(tmp_path / "a" / "clif", tmp_path / "a" / "meds", causal)
    ok = weight_causality(tmp_path / "a" / "clif", tmp_path / "a" / "meds")
    assert ok == {"weighted_events": 3, "before_first_weight_reported": 2, "before_first_weight_converted": 0,
                  "segment_before_first_weight_reported": 2, "segment_before_first_weight_converted": 0,
                  "violations": 0}
    # a first-weight fallback: the 8:00 propofol converted to per kg with the weight charted at 10:00, and its
    # segment carried on per kg after the weight
    fallback = [causal[0], (_t(8), "MED_CON//propofol//mcg/kg/min//start//iv"), causal[2],
                (_t(11), "MED_CON//propofol//mcg/kg/min//dose_change//iv")]
    _write(tmp_path / "b" / "clif", tmp_path / "b" / "meds", fallback)
    bad = weight_causality(tmp_path / "b" / "clif", tmp_path / "b" / "meds")
    assert bad["before_first_weight_converted"] == 1 and bad["segment_before_first_weight_converted"] == 2
    assert bad["violations"] == 3
