"""icutwin on the GPU host: Paper P amendment 11, V5 (teaching cases; ``docs/p1_paper_protocol.md``).

Three pre-specified objectives on the bank, three typical cases each (``toolkit.teach``) and a drill of the nine:
shock (a vasopressor episode; decision one minute before the twin's first norepinephrine start; options norepinephrine
and a fluid bolus), respiratory failure (IMV; before its first IMV record; option IMV) and renal support (RRT; before
its first CRRT record; options CRRT and furosemide), each in the datasets where C1 defined every option's record.
Checks: every case meets its objective (the twin's attributes in the bank's store), every twin shown passes novelty
against the training stays when the reference exists (``runs/toolkit/novelty/<run>.npz``; else the bank's production
filter is cited), every page is self-contained (no external URL, script or stylesheet). Also the case figure
(``toolkit.figures``) of the first shock case. Writes ``runs/toolkit/v5/``.

    python tools/server/toolkit_v5.py --engines 2
"""
import argparse
import json
import re

OBJECTIVES = {
    "shock": ({"filters": {"vasopressor": "yes"}, "datasets": ["mimic", "aumc", "nwicu"]}, "norepinephrine",
              ["norepinephrine", "fluid_bolus"]),
    "respiratory": ({"filters": {"imv": "yes"}, "datasets": ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu",
                                                             "inspire"]}, "imv", ["imv"]),
    "renal": ({"filters": {"rrt": "yes"}, "datasets": ["mimic", "aumc", "eicu", "hirid", "sicdb", "inspire"]}, "crrt",
              ["crrt", "furosemide"]),
}
EXTERNAL = re.compile(r"https?://|<script[^>]+src=|<link[^>]+href=", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", type=int, default=2)
    ap.add_argument("--gpu", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    import numpy as np

    from icutwin.factory import store as fst
    from icutwin.toolkit import card, cohort, figures, teach
    from icutwin.toolkit.session import Paths, Session

    s = Session(Paths.default(), backend="vllm", engines=a.engines, gpu_memory_utilization=a.gpu, workers=a.workers)
    out = s.paths.out / "v5"
    attrs = cohort.attributes(fst.Store(s.paths.bank_store)).set_index("stay_id")
    rep, all_cases = {"objectives": {}}, []
    for name, (obj, trt, opts) in OBJECTIVES.items():
        res = teach.run(s, obj, treatment=trt, options=opts, cases=a.cases, n=a.n, pool=a.pool, pick="typical",
                        seed=0, out_dir=out / name)
        cases = [json.loads((out / name / c["files"][0]).read_text(encoding="utf-8")) for c in res["outputs"]["cases"]]
        all_cases += cases
        meets = [bool(cohort.match(attrs.loc[[c["patient"].split(":", 1)[1]]].reset_index(), obj["filters"])[0])
                 for c in cases]
        pages = [p for c in res["outputs"]["cases"] for p in c["files"][1:]] + ["drill.html"]
        external = {p: bool(EXTERNAL.search((out / name / p).read_text(encoding="utf-8"))) for p in pages}
        rep["objectives"][name] = {"cases": len(cases), "candidates": res["outputs"]["candidates"],
                                   "meets_objective": meets, "external_references": external,
                                   "patients": [c["patient"] for c in cases], "hours": [c["hours"] for c in cases]}
    for i, c in enumerate(all_cases):
        c["case"] = i
    card.write(card.drill(all_cases, s), out / "drill_all.html")
    nov = None
    ref = s.paths.out / "novelty" / f"{s.run}.npz"
    if ref.exists():
        from icutwin.toolkit import inputs

        ids = [inputs.load(s, c["patient"]).ids for c in all_cases]
        nov = cohort.novelty_of(s, ids, [c["dataset"] for c in all_cases])
    rep["novelty"] = nov or "not computed on this host: the twins passed the bank's production novelty filter"
    first = all_cases[0]
    wi = next(iter(first["whatif"].values()))
    ex = first["explain"] if "outputs" in first["explain"] else None
    rep["figure"] = [str(p) for p in figures.case_figure(s, first["forecast"], wi if "outputs" in wi else None, ex,
                                                         out / "figure_case")]
    rep["checks"] = {"meets_objective": bool(np.all([m for o in rep["objectives"].values() for m in o["meets_objective"]])),
                     "self_contained": not any(x for o in rep["objectives"].values()
                                               for x in o["external_references"].values())}
    (out / "v5.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(rep["checks"]), flush=True)


if __name__ == "__main__":
    main()
