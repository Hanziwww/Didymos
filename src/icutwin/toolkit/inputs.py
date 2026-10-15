"""Patients and anchors (``docs/toolkit.md``): a stay as token ids on its own clock, and the stay as known at an hour.

References (``load``):

* ``split:<dataset>/<split>/<stay id>``: a tokenized stay of the model's tokens (``held_out``: a real held-out
  stay; ``tuning`` or ``train`` also read, the latter marked a training member);
* ``bank:<twin id>``: a twin of the bank (``data/bank/<run>_bank``; internal, DUA-derived);
* ``meds:<MEDS dir>::<split>::<stay id>``: a stay of a MEDS dataset (``tokenization.stays.iter_stays``), encoded with
  the model's vocabulary (``tokenization.encode.encode``); a new site is prompted with its dataset and site unknown
  (B2/F9), the default for this kind.

``Patient.at(hours)`` is the paper's prompt (``eval.bench.prompt``: the stay decoded on its clock, keyed by its id,
then ``tokenization.encode.prospective`` at the hour: header fields documented later unknown, no goal or end block)
and its prediction time ``t_min`` (``eval.bench.anchor_time``: the last event, minutes after ICU admission on that
clock), where the label engine reads labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Anchor:
    """A patient as known at ``hours`` after ICU admission."""
    id: str
    patient: Patient
    hours: float
    prompt: list
    t_min: int

    @property
    def dataset(self) -> str | None:
        return self.patient.dataset


@dataclass
class Patient:
    ref: str
    stay_id: str                      # the clock id: the stay decodes on the clock keyed by it
    dataset: str | None               # None: a stay of no dataset the model knows
    ids: list
    kind: str                         # held_out | tuning | train (real stays) | twin | external
    unknown_site: bool = False
    _stay: object = field(default=None, repr=False)

    @property
    def real(self) -> bool:
        return self.kind in ("held_out", "tuning", "train", "external")

    def stay(self, v):
        """The labels' decode of the whole stay (values at their bins' medians, gaps on its clock)."""
        if self._stay is None:
            from icutwin.factory import twins

            self._stay = twins.decode_labels(self.ids, v, self.stay_id)
        return self._stay

    def los_hours(self, v) -> float:
        s = self.stay(v)
        return (s.icu_out - s.icu_in).total_seconds() / 3600

    def at(self, v, hours: float) -> Anchor:
        """The anchor at ``hours`` after ICU admission (module docstring); within the ICU stay."""
        from icutwin.eval import bench
        from icutwin.tokenization.encode import prospective

        los = self.los_hours(v)
        if not 0 <= hours < los:
            raise ValueError(f"{self.ref}: {hours:g} h is outside its ICU stay (0 to {los:.1f} h)")
        p = prospective(self.stay(v), v, float(hours))
        if self.unknown_site:
            p = bench.unknown_site(p, v)
        return Anchor(f"{self.stay_id}@{hours:g}", self, float(hours), [int(x) for x in p],
                      int(bench.anchor_time(p, v, self.stay_id)))


def _row(path: Path, column: str, key: str, cols: list) -> dict:
    import pyarrow.parquet as pq

    typ = pq.read_schema(path).field(column).type
    val = int(key) if str(typ).startswith(("int", "uint")) else key
    t = pq.read_table(path, columns=cols, filters=[(column, "=", val)]).to_pylist()
    if not t:
        raise KeyError(f"{key} not in {path}")
    return t[0]


def load(session, ref: str, unknown_site: bool | None = None) -> Patient:
    """``unknown_site``: prompt with the dataset and site unknown (default: only for MEDS stays)."""
    kind, _, rest = ref.partition(":")
    v = session.vocab
    if kind == "split":
        ds, split, stay = rest.split("/", 2)
        r = _row(session.paths.tokens / ds / f"{split}.parquet", "stay_id", stay, ["stay_id", "ids"])
        return Patient(ref, str(r["stay_id"]), ds, [int(x) for x in r["ids"]], split, bool(unknown_site))
    if kind == "bank":
        stem = rest.rsplit("_", 1)[0]
        files = sorted(session.paths.bank.glob(f"part*/{stem}.parquet"))
        if not files:
            raise KeyError(f"{rest} not in the bank {session.paths.bank}")
        r = _row(files[0], "twin", rest, ["twin", "dataset", "ids"])
        return Patient(ref, str(r["twin"]), r["dataset"], [int(x) for x in r["ids"]], "twin", bool(unknown_site))
    if kind == "meds":
        meds, split, stay = rest.split("::")
        return from_meds(v, Path(meds), split, [stay], unknown_site=unknown_site is not False)[0]
    raise ValueError(f"unknown reference {ref!r} (split:, bank:, meds:)")


def from_meds(v, meds_dir: Path, split: str, stay_ids=None, derived_dir: Path | None = None,
              unknown_site: bool = True) -> list[Patient]:
    """Stays of a MEDS dataset encoded with the model's vocabulary (``stay_ids``: these only)."""
    from icutwin.tokenization.encode import encode
    from icutwin.tokenization.stays import iter_stays

    want = None if stay_ids is None else {str(x) for x in stay_ids}
    out = []
    for s in iter_stays(Path(meds_dir), split, derived_dir):
        if want is not None and str(s.stay_id) not in want:
            continue
        ds = s.dataset if f"PREFIX//dataset//{s.dataset}" in v.index else None
        out.append(Patient(f"meds:{meds_dir}::{split}::{s.stay_id}", str(s.stay_id), ds,
                           [int(x) for x in encode(s, v).whole], "external", unknown_site))
    if want is not None and len(out) < len(want):
        raise KeyError(f"{len(want) - len(out)} stays not in {meds_dir} ({split})")
    return out


def from_ids(ids, stay_id: str, dataset: str | None, kind: str = "twin", unknown_site: bool = False) -> Patient:
    """A patient from token ids at hand (a sampled twin, a test sequence)."""
    return Patient(f"ids:{stay_id}", str(stay_id), dataset, [int(x) for x in ids], kind, unknown_site)
