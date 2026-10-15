"""The paper's main figures, drawn by the design system in
``tools/paper_figstyle.py`` from the aggregate files under ``runs/paper`` and ``runs/toolkit`` and amendment 13's
statistics (``runs/paper/stats/stats.json``). One module per figure (``tools/paper_fig<k>.py``).

Each figure writes into ``--out`` a vector PDF, an SVG, its source data (``source_data/<figure>.csv``) and its entry in
``manifest.json``: the inputs with their sha256, the git sha, and the design checks' result. Into ``--preview`` go a
600-dpi PNG and its colour-blind and greyscale simulations. Internal figures (a synthetic patient, the clinician
template) go to ``--internal`` only, never to ``--out``.

    python tools/paper_figures.py --root runs/paper --toolkit runs/toolkit --out docs/paper/figures \\
        --preview runs/paper/figures/preview --internal runs/paper/figures/internal [--only fig2]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figstyle as st  # noqa: E402
from paper_figdata import Inputs  # noqa: E402

PUBLIC = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9")
# Supplementary figures in the order of the main figures; S1-S3 carry schematic artwork (GPT-image)
SUPP_FUNCS = {
    "supp1": "data_processing",
    "supp2": "architecture",
    "supp3": "evaluation_sets",
    "supp4": "predictor_matrix",
    "supp5": "ranks",
    "supp6": "futures_curves",
    "supp7": "per_database",
    "supp8": "breadth",
    "supp9": "families_and_labels",
    "supp10": "regression_and_time_to_event",
    "supp11": "sites",
    "supp12": "subgroups",
    "supp13": "recalibration_maps",
    "supp14": "conformance",
    "supp15": "similarity",
    "supp16": "trial_checks",
}
# amendments 16 and 17 (tools/paper_figures_ext.py); their numbers follow the manuscript's first citations
EXT_FUNCS = {"supp_l48x": "l48x", "supp_knockout": "knockout", "supp_transfer": "transfer", "supp_adapt": "adaptation",
             "supp_adapt_cal": "adapt_calibration", "supp_adapt_gate": "adapt_gate"}
SUPPLEMENTARY = tuple(SUPP_FUNCS) + tuple(EXT_FUNCS)
INTERNAL = ("fig7_case", "fig8_template")
FUNCS = {**{f"fig{i}": (f"paper_fig{i}", "figure") for i in range(1, 10)},
         **{k: ("paper_figures_supp", v) for k, v in SUPP_FUNCS.items()},
         **{k: ("paper_figures_ext", v) for k, v in EXT_FUNCS.items()},
         "fig7_case": ("paper_fig7", "case"), "fig8_template": ("paper_fig8", "template")}


def build(name: str, inp: Inputs, out: Path, preview: Path, cvd: bool = True) -> dict:
    import matplotlib.pyplot as plt

    mod_name, fn_name = FUNCS[name]
    fn = getattr(importlib.import_module(mod_name), fn_name, None)
    if fn is None:
        return {}
    cv, sd = fn(inp)
    issues = st.check(cv)
    files = st.save(cv, name, out, preview)
    sd.write(out / "source_data")
    if cvd:
        st.simulate_png(preview / f"{name}.png", preview / "cvd")
    plt.close(cv.fig)
    return {"git": st.git_sha(), "inputs": sd.inputs, "checks": issues or "pass",
            "files": [str(f).replace("\\", "/") for f in files if f.suffix != ".png"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/paper")
    ap.add_argument("--toolkit", default="runs/toolkit")
    ap.add_argument("--out", default="docs/paper/figures")
    ap.add_argument("--preview", default="runs/paper/figures/preview")
    ap.add_argument("--internal", default="runs/paper/figures/internal")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-cvd", action="store_true")
    a = ap.parse_args(argv)
    inp = Inputs(Path(a.root), Path(a.toolkit))
    for names, out in ((PUBLIC, Path(a.out)), (SUPPLEMENTARY, Path(a.out) / "supplementary"),
                       (INTERNAL, Path(a.internal))):
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        done = False
        for name in names:
            if a.only and name not in a.only:
                continue
            try:
                importlib.import_module(FUNCS[name][0])
            except ModuleNotFoundError:
                continue
            m = build(name, inp, out, Path(a.preview), cvd=not a.no_cvd)
            if not m:
                continue
            manifest[name] = m
            done = True
            print(json.dumps({name: manifest[name]["checks"]})[:2000], flush=True)
        if done:
            out.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
