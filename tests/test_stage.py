"""S1 staging on a synthetic archive: deterministic selection, exact-text filtering, copies, manifests."""

from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from icutwin.stage.spec import DatasetSpec
from icutwin.stage.staging import MANIFEST, patient_selected, stage_dataset

STAYS = (
    "patient_id,stay_id,note\n"
    + "".join(f'P{i},S{i}{j},"stay {j}, of ""P{i}""\nsecond line"\n' for i in range(40) for j in range(2))
)
EVENTS = "stay_id,t,value\n" + "".join(f"S{i}{j},{k},{i * 10 + k}.5\n" for i in range(40) for j in range(2) for k in range(3))
DICT = "﻿itemid,label\n1,heart rate\n2,\"blood pressure, systolic\"\n"


def _make_archive(path: Path) -> None:
    files = {
        "DS-1.0/stays.csv": STAYS,
        "DS-1.0/sub/events.csv": EVENTS,
        "DS-1.0/d_items.csv": DICT,
        "DS-1.0/SHA256SUMS.txt": "deadbeef  stays.csv\n",
        "DS-1.0/code/readme.md": "# bundled code\n",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path.write_bytes(zstandard.ZstdCompressor(level=3).compress(buf.getvalue()))


SPEC = DatasetSpec.model_validate(
    {
        "name": "toy",
        "archive": "toy.tar.zst",
        "strip_prefix": "DS-1.0/",
        "out_compression": "gzip",
        "anchor": {"member": "stays.csv", "patient_col": "patient_id", "derived_keys": {"stay": "stay_id"}},
        "rules": [
            {"pattern": "code/*", "action": "refs"},
            {"pattern": "d_*.csv", "action": "copy"},
            {"pattern": "stays.csv", "action": "filter", "key_col": "patient_id", "key": "patient"},
            {"pattern": "*.csv", "action": "filter", "key_col": "stay_id", "key": "stay"},
            {"pattern": "*SHA256SUMS.txt", "action": "checksums"},
            {"pattern": "*", "action": "copy"},
        ],
    }
)


def _read_gz_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture()
def archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "archives"
    root.mkdir()
    _make_archive(root / "toy.tar.zst")
    return root


def test_selection_is_deterministic_and_salted() -> None:
    a = [patient_selected(f"P{i}", "toy", 0.3, "s1") for i in range(500)]
    assert a == [patient_selected(f"P{i}", "toy", 0.3, "s1") for i in range(500)]
    assert a != [patient_selected(f"P{i}", "toy", 0.3, "s2") for i in range(500)]
    assert 0.2 < sum(a) / len(a) < 0.4


def test_subset_filters_exact_rows(archive_root: Path, tmp_path: Path) -> None:
    out, refs = tmp_path / "out", tmp_path / "refs"
    manifest = stage_dataset(SPEC, archive_root, out, refs, mode="subset", frac=0.5, salt="t")
    chosen = {f"P{i}" for i in range(40) if patient_selected(f"P{i}", "toy", 0.5, "t")}
    assert 0 < len(chosen) < 40

    stays = _read_gz_csv(out / "toy" / "stays.csv.gz")
    expected_stays = [r for r in csv.DictReader(io.StringIO(STAYS)) if r["patient_id"] in chosen]
    assert stays == expected_stays  # values (quoted commas, quotes, newlines) preserved exactly

    chosen_stays = {r["stay_id"] for r in expected_stays}
    events = _read_gz_csv(out / "toy" / "sub" / "events.csv.gz")
    assert events == [r for r in csv.DictReader(io.StringIO(EVENTS)) if r["stay_id"] in chosen_stays]

    # dictionary copied byte-exactly (BOM included), refs extracted, checksums dropped in subset mode
    assert gzip.decompress((out / "toy" / "d_items.csv.gz").read_bytes()) == DICT.encode("utf-8")
    assert (refs / "physionet_bundled" / "toy" / "code" / "readme.md").read_text() == "# bundled code\n"
    assert not (out / "toy" / "SHA256SUMS.txt").exists()

    on_disk = json.loads((out / "toy" / MANIFEST).read_text())
    assert on_disk["selection"]["patient_selected"] == len(chosen)
    rows = {m["member"]: m for m in manifest["members"]}
    assert rows["stays.csv"]["rows_in"] == 80 and rows["stays.csv"]["rows_out"] == len(expected_stays)
    assert rows["sub/events.csv"]["rows_out"] == len(events)


def test_full_mode_copies_everything(archive_root: Path, tmp_path: Path) -> None:
    out, refs = tmp_path / "out", tmp_path / "refs"
    stage_dataset(SPEC, archive_root, out, refs, mode="full", compression="none")
    assert (out / "toy" / "stays.csv").read_bytes() == STAYS.encode("utf-8")
    assert (out / "toy" / "sub" / "events.csv").read_bytes() == EVENTS.encode("utf-8")
    assert (out / "toy" / "SHA256SUMS.txt").exists()


def test_unmatched_member_is_an_error(archive_root: Path, tmp_path: Path) -> None:
    # without the refs rule and the catch-all, code/readme.md matches nothing
    spec = SPEC.model_copy(update={"rules": [r for r in SPEC.rules if r.pattern not in {"*", "code/*"}]})
    with pytest.raises(LookupError):
        stage_dataset(spec, archive_root, tmp_path / "o", tmp_path / "r", mode="subset", frac=0.5, salt="t")


def test_mirror_matches_staged_rows(archive_root: Path, tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from icutwin.stage.mirror import MIRROR_MANIFEST, mirror_dataset

    out, refs = tmp_path / "out", tmp_path / "refs"
    stage_dataset(SPEC, archive_root, out, refs, mode="subset", frac=0.5, salt="t")
    manifest = mirror_dataset(out / "toy", tmp_path / "pq", temp_dir=tmp_path / "duck", threads=2, memory_limit="1GB")
    tables = {t["table"]: t for t in manifest["tables"]}
    assert set(tables) == {"stays.parquet", "sub/events.parquet", "d_items.parquet"}
    assert tables["stays.parquet"]["checked_against"] == "staging"
    assert tables["d_items.parquet"]["checked_against"] == "pyarrow" and tables["d_items.parquet"]["rows"] == 2
    stays = pq.read_table(tmp_path / "pq" / "stays.parquet").to_pylist()
    expected = [r for r in csv.DictReader(io.StringIO(STAYS)) if patient_selected(r["patient_id"], "toy", 0.5, "t")]
    assert [r["note"] for r in stays] == [r["note"] for r in expected]  # embedded newlines survive DuckDB
    assert (tmp_path / "pq" / MIRROR_MANIFEST).exists()


def test_latin1_source_round_trips_bytes(tmp_path: Path) -> None:
    # odd items carry a C1 byte (0x96, an en dash on Windows code pages), which DuckDB's latin-1 decoder rejects
    rows = "admissionid,item\n" + "".join(f"{i},Niet iv Zalven/Crèmes/Druppels {i}{chr(0x96) * (i % 2)}\n"
                                          for i in range(50))
    adm = "patientid,admissionid\n" + "".join(f"P{i},{i}\n" for i in range(50))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, text in {"admissions.csv": adm, "drugitems.csv": rows}.items():
            data = text.encode("latin-1")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    root = tmp_path / "a"
    root.mkdir()
    (root / "l1.tar.zst").write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    spec = DatasetSpec.model_validate({
        "name": "l1", "archive": "l1.tar.zst", "out_compression": "none", "encoding": "latin-1",
        "anchor": {"member": "admissions.csv", "patient_col": "patientid", "derived_keys": {"adm": "admissionid"}},
        "rules": [
            {"pattern": "admissions.csv", "action": "filter", "key_col": "patientid", "key": "patient"},
            {"pattern": "*.csv", "action": "filter", "key_col": "admissionid", "key": "adm"},
        ],
    })
    stage_dataset(spec, root, tmp_path / "o", tmp_path / "r", mode="subset", frac=0.5, salt="t")
    raw = (tmp_path / "o" / "l1" / "drugitems.csv").read_bytes()
    assert b"Cr\xe8mes" in raw and b"Cr\xc3\xa8mes" not in raw  # still Latin-1, not re-encoded as UTF-8
    kept = {f"P{i}" for i in range(50) if patient_selected(f"P{i}", "l1", 0.5, "t")}
    got = list(csv.DictReader(io.StringIO(raw.decode("latin-1"))))
    assert [r["admissionid"] for r in got] == [str(i) for i in range(50) if f"P{i}" in kept]
    assert all(r["item"].startswith("Niet iv Zalven/Crèmes/Druppels ") for r in got)

    from icutwin.stage.mirror import mirror_dataset
    import pyarrow.parquet as pq
    mirror_dataset(tmp_path / "o" / "l1", tmp_path / "pq", temp_dir=tmp_path / "duck", threads=1, memory_limit="1GB")
    items = pq.read_table(tmp_path / "pq" / "drugitems.parquet").column("item").to_pylist()
    assert any(int(r["admissionid"]) % 2 for r in got)              # the C1 byte survives as its code point
    assert items == [f"Niet iv Zalven/Crèmes/Druppels {r['admissionid']}" + chr(0x96) * (int(r["admissionid"]) % 2)
                     for r in got]


def test_filtered_csv_keeps_source_formatting(tmp_path: Path) -> None:
    text = 'id,num,empty,txt\n1,10,,plain\n2,20,"","a, quoted ""x"""\n'
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, body in {"p.csv": "id\n1\n2\n", "t.csv": text}.items():
            data = body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    root = tmp_path / "a"
    root.mkdir()
    (root / "f.tar.zst").write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    spec = DatasetSpec.model_validate({
        "name": "f", "archive": "f.tar.zst", "out_compression": "none",
        "anchor": {"member": "p.csv", "patient_col": "id"},
        "rules": [{"pattern": "*.csv", "action": "filter", "key_col": "id", "key": "patient"}],
    })
    stage_dataset(spec, root, tmp_path / "o", tmp_path / "r", mode="subset", frac=1.0, salt="t")
    # numbers stay bare, missing values stay empty fields, only values that need quotes are quoted
    assert (tmp_path / "o" / "f" / "t.csv").read_text() == 'id,num,empty,txt\n1,10,,plain\n2,20,,"a, quoted ""x"""\n'
