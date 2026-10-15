"""Supplements to reference-tool CLIF outputs (CLIF-MIMIC, CLIF-AmsterdamUMCdb).

A supplement adds what the reference ETL does not produce: the ``ext_icu_stay`` extension table and CLIF 3.0
concepts outside the tool's mapping (e.g. hematocrit, base excess, invasive hemodynamics). Its tables are
written as parts ``clif_<table>+icutwin.parquet`` next to the tool's output, and S2b harmonization unions
them into the canonical table. A supplement never maps an item the tool maps (tested against the tool's
mapping files), so no measurement is emitted twice.
"""

from icutwin.clif.supplements.aumc import AumcSupplement
from icutwin.clif.supplements.base import SUPPLEMENT_REPORT, ClifSupplement, check_supplement_fresh
from icutwin.clif.supplements.mimic import MimicSupplement

SUPPLEMENTS = {"mimic": MimicSupplement, "aumc": AumcSupplement}

__all__ = ["SUPPLEMENTS", "SUPPLEMENT_REPORT", "ClifSupplement", "check_supplement_fresh"]
