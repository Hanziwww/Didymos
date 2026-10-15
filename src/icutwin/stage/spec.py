"""Per-dataset staging specifications (``icutwin/stage/specs/<name>.yaml``)."""

from __future__ import annotations

import fnmatch
from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Action = Literal["filter", "copy", "skip", "refs", "checksums"]


class AnchorSpec(BaseModel):
    """Table that defines the patient population and any derived row keys."""

    member: str
    patient_col: str
    # key-set name -> column of the anchor table holding that key for selected patients
    derived_keys: dict[str, str] = Field(default_factory=dict)


class Rule(BaseModel):
    """First matching rule (fnmatch on the prefix-stripped member path) decides the action.

    ``filter``    keep rows whose ``key_col`` is in key set ``key`` (subset mode); full copy in full mode
    ``copy``      byte-exact copy (dictionaries, documentation)
    ``skip``      not staged
    ``refs``      extracted to ``refs/physionet_bundled/<dataset>/`` instead of the data layout
    ``checksums`` copied in full mode only (subset files no longer match the published checksums)
    """

    pattern: str
    action: Action
    key_col: str | None = None
    key: str | None = None
    # optional members are skipped unless their tag is listed in the run's ``include``
    optional: str | None = None

    @model_validator(mode="after")
    def _filter_needs_key(self) -> Rule:
        if self.action == "filter" and (self.key_col is None or self.key is None):
            raise ValueError(f"filter rule {self.pattern!r} needs key_col and key")
        return self


class DatasetSpec(BaseModel):
    name: str
    archive: str  # relative to the raw archive root
    strip_prefix: str = ""
    out_compression: Literal["gzip", "none"] = "gzip"
    # text encoding of the source CSVs; latin-1 round-trips every byte exactly
    encoding: Literal["utf-8", "latin-1"] = "utf-8"
    # field delimiter of the source CSVs (RICD's are ';'); subset filtering supports only ',', so a dataset with
    # another delimiter is staged in full mode by byte-exact copies
    delimiter: str = ","
    anchor: AnchorSpec
    rules: list[Rule]

    @model_validator(mode="after")
    def _keys_defined(self) -> DatasetSpec:
        known = {"patient", *self.anchor.derived_keys}
        for rule in self.rules:
            if rule.action == "filter" and rule.key not in known:
                raise ValueError(f"rule {rule.pattern!r} uses undefined key set {rule.key!r}; known {sorted(known)}")
        return self

    def match(self, rel: str) -> Rule:
        for rule in self.rules:
            if fnmatch.fnmatchcase(rel, rule.pattern):
                return rule
        raise LookupError(f"{self.name}: no staging rule matches member {rel!r}")


DATASETS = ("mimic", "eicu", "aumc", "hirid", "sicdb", "nwicu", "inspire")
# inference-only databases for Paper P's transfer analysis (amendment 16a); never in training
EXTERNAL = ("ricd", "zigong")


def stay_dataset(dataset: str | None, stay_id: str | None) -> str | None:
    """A stay's database. An inference-only database is not in the vocabulary (its stays encode the unknown
    dataset token), so a stay decoded from its tokens takes the database from its id ('<dataset>:<id>'); every
    other stay keeps the decoded name (the training databases' stays prompted as unknown stay 'unknown')."""
    head = (stay_id or "").split(":", 1)[0]
    return head if dataset in (None, "", "unknown") and head in EXTERNAL else dataset


def load_spec(name: str) -> DatasetSpec:
    text = resources.files("icutwin.stage").joinpath("specs", f"{name}.yaml").read_text(encoding="utf-8")
    spec = DatasetSpec.model_validate(yaml.safe_load(text))
    if spec.name != name:
        raise ValueError(f"spec file {name}.yaml declares name {spec.name!r}")
    return spec
