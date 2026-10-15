"""Base class for supplements to reference-tool CLIF outputs (see the package docstring)."""

from __future__ import annotations

import json
from pathlib import Path

from icutwin.clif.adapters.base import ClifAdapter

SUPPLEMENT_REPORT = "_supplement_report.json"
TOOL_PROVENANCE = "_provenance.json"  # written by icutwin.clif.external after each reference-tool run


class ClifSupplement(ClifAdapter):
    """An adapter that writes parts ``clif_<table>+icutwin.parquet`` into a reference tool's output directory."""

    part = "icutwin"
    report_name = SUPPLEMENT_REPORT

    def retractions(self) -> dict[str, list[dict]]:
        """Rows of the tool's own tables that this supplement replaces, as ``{table: [{when_regex: {...}}]}``.

        S2b drops them from the tool's file only (never from supplement parts), before the union.
        """
        return {}

    def text_repairs(self) -> dict[str, str]:
        """``{lossy token: token}`` for text the tool decoded with the wrong encoding (S2b repairs the tool's
        text columns token by token)."""
        return {}

    def run(self, tables: tuple[str, ...] | None = None) -> dict:
        tool_run = self._tool_run()
        report = super().run(tables)
        report["tool_run"] = tool_run  # the tool output this supplement belongs to (checked by harmonize)
        report["retract"] = self.retractions()
        report["text_repairs"] = self.text_repairs()
        (self.out_dir / self.report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _tool_run(self) -> dict:
        prov = self.out_dir / TOOL_PROVENANCE
        if not prov.exists():
            raise FileNotFoundError(f"{prov}: run the reference tool before its supplement")
        rec = json.loads(prov.read_text(encoding="utf-8"))
        return {"tool": rec.get("tool"), "commit": rec.get("commit"), "finished_at": rec.get("finished_at")}


def check_supplement_fresh(clif_dir: Path) -> dict | None:
    """Fail if supplement parts are stale; return ``{tool_run, retract}`` of the supplement (None: no parts).

    Parts are stale when the reference tool ran again after the supplement was built, or when a part is not
    listed in the supplement report (a table the current supplement no longer writes).
    """
    parts = sorted(clif_dir.glob("clif_*+*.parquet"))
    if not parts:
        return None
    sup = clif_dir / SUPPLEMENT_REPORT
    if not sup.exists():
        raise ValueError(f"{clif_dir}: supplement parts {[p.name for p in parts]} without {SUPPLEMENT_REPORT}")
    report = json.loads(sup.read_text(encoding="utf-8"))
    listed = {f"clif_{t}+{ClifSupplement.part}" for t in report.get("tables", {})}
    unknown = [p.name for p in parts if p.stem not in listed]
    if unknown:
        raise ValueError(f"{clif_dir}: parts not written by the current supplement: {unknown}")
    built = report.get("tool_run") or {}
    prov = clif_dir / TOOL_PROVENANCE
    if prov.exists():
        current = json.loads(prov.read_text(encoding="utf-8")).get("finished_at")
        if built.get("finished_at") != current:
            raise ValueError(f"{clif_dir}: supplement built against the tool run of {built.get('finished_at')}, "
                             f"but the tool output is from {current}; re-run task=supplement")
    return {"tool_run": built, "retract": report.get("retract") or {}, "text_repairs": report.get("text_repairs") or {}}
