"""Paper P's diagrams: the Mermaid sources of Supplementary Figs. 1-2, and the artwork bundle for redrawing.

    python tools/paper_diagrams.py render [--browser PATH]
        docs/paper/manuscript/diagrams/*.mmd -> docs/paper/figures/supplementary/diagrams/*.svg and manifest.json,
        PNG previews -> runs/paper/figures/diagrams_preview/
    python tools/paper_diagrams.py bundle [--out runs/paper/figures/diagram_bundle]
        the schematic panels (600-dpi crops and vector crops), their verbatim labels and slot sizes, the prompts, the
        Mermaid sources and their renders -> <out>/ and <out>.zip

Rendering is local: a loopback-only HTTP server (127.0.0.1) serves a page that loads mermaid (pinned) from the CDN in
headless Edge or Chrome and posts each rendered SVG back. Nothing is sent to an online renderer. The SVGs carry their
labels as SVG text with the font as a presentation attribute, so vector editors open them as editable artwork.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figstyle as st  # noqa: E402

MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SRC = Path("docs/paper/manuscript/diagrams")
OUT = Path("docs/paper/figures/supplementary/diagrams")
PREVIEW = Path("runs/paper/figures/diagrams_preview")
FIGS = Path("docs/paper/figures")
PNG = Path("runs/paper/figures/preview")             # the 600-dpi previews that paper_figures.py writes
FONT = "Arial, Helvetica, sans-serif"
# The schematic panels: (figure, x, y, width, height) of the artwork area in mm from the figure's top left. The panel
# letter and title stay in the figure code; replacement artwork fills exactly this box.
SLOTS = {
    "fig1a": ("fig1", 0, 6, 180, 62),
    "fig1c": ("fig1", 94, 75, 86, 48),
    "fig4a": ("fig4", 0, 5, 180, 39),
    "fig7a": ("fig7", 0, 6, 180, 38),
    "supp3": ("supp3", 0, 0, 180, 112),
}
PT_PER_MM = 72 / 25.4
DPI = 600


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _figure_file(fig: str, ext: str) -> Path:
    return (FIGS / "supplementary" if fig.startswith("supp") else FIGS) / f"{fig}.{ext}"


# ---- render ------------------------------------------------------------------------------------------------------------

def render(browser: str = EDGE, timeout: float = 120.0) -> dict:
    names = sorted(p.stem for p in SRC.glob("*.mmd"))
    codes = {n: (SRC / f"{n}.mmd").read_text(encoding="utf-8") for n in names}
    page = f"""<!doctype html><html><head><meta charset="utf-8"><script src="{MERMAID}"></script></head><body><script>
const codes = {json.dumps(codes)};
const post = (path, body) => fetch(path, {{method: "POST", body, headers: {{"Content-Type": "text/plain"}}}});
mermaid.initialize({{startOnLoad: false, securityLevel: "strict"}});
(async () => {{
  for (const [name, code] of Object.entries(codes)) {{
    try {{ const {{svg}} = await mermaid.render("d-" + name, code); await post("/save?name=" + name, svg); }}
    catch (e) {{ await post("/error?name=" + name, String(e && (e.message || e))); }}
  }}
  await post("/done", "");
}})();
</script></body></html>""".encode("utf-8")
    received, errors, done = {}, {}, threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            u = urlparse(self.path)
            data = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
            name = (parse_qs(u.query).get("name") or [""])[0]
            if u.path == "/save" and name in codes:
                received[name] = data
            elif u.path == "/error":
                errors[name] = data
            elif u.path == "/done":
                done.set()
            self.send_response(204)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="paper_diagrams_"))
    base = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            f"--user-data-dir={tmp / 'profile'}"]
    try:
        # the launcher may return before the page runs; the page reports back over the loopback server
        subprocess.run(base + ["--virtual-time-budget=60000", f"--screenshot={tmp / 'page.png'}",
                               f"http://127.0.0.1:{srv.server_address[1]}/"], capture_output=True, timeout=60)
        done.wait(timeout=timeout)
    finally:
        srv.shutdown()
    if errors or set(received) != set(names):
        raise SystemExit(f"Mermaid failed: errors {errors}; missing {sorted(set(names) - set(received))}")

    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    rows = {}
    for n in names:
        svg = received[n].replace("&nbsp;", "&#160;")
        svg = svg.replace("<text ", f'<text font-family="{FONT}" ').replace("<text>", f'<text font-family="{FONT}">')
        if "<foreignObject" in svg:
            raise SystemExit(f"{n}: HTML labels in the SVG; set htmlLabels false in its init block")
        ET.fromstring(svg)                                      # well-formed XML
        f = OUT / f"{n}.svg"
        f.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg, encoding="utf-8", newline="\n")
        vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
        rows[n] = {"source": str(SRC / f"{n}.mmd").replace("\\", "/"), "source_sha256": _sha(SRC / f"{n}.mmd"),
                   "svg_sha256": _sha(f), "viewbox": vb}
        _png(base, f, PREVIEW / f"{n}.png", vb[2], vb[3], tmp)
    manifest = {"renderer": f"mermaid 11.4.1 ({MERMAID}), headless {Path(browser).stem}", "git": st.git_sha(),
                "diagrams": rows}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return manifest


def _png(base: list, svg: Path, png: Path, w: float, h: float, tmp: Path, width: int = 2400) -> None:
    wrap = tmp / f"{svg.stem}.html"
    wrap.write_text(f'<!doctype html><html><body style="margin:20px;background:#fff"><img src="{svg.resolve().as_uri()}"'
                    f' style="width:{width - 40}px;display:block"></body></html>', encoding="utf-8")
    png.unlink(missing_ok=True)
    height = int(round((width - 40) * h / w)) + 40
    subprocess.run(base + [f"--window-size={width},{height}", f"--screenshot={png.resolve()}", "--hide-scrollbars",
                           wrap.as_uri()], capture_output=True, timeout=60)
    t0 = time.time()
    while not png.exists() and time.time() - t0 < 60:          # the launcher returns before the shot is written
        time.sleep(0.5)
    time.sleep(1.0)


# ---- bundle ------------------------------------------------------------------------------------------------------------

def labels(fig: str, x: float, y: float, w: float, h: float) -> list:
    """The panel's text labels from the published vector SVG, in reading order: [x_mm, y_mm, text]."""
    ns = "{http://www.w3.org/2000/svg}"
    root = ET.parse(_figure_file(fig, "svg")).getroot()
    out = []
    for el in root.iter(f"{ns}text"):
        s = "".join(el.itertext()).strip()
        if s and el.get("x") is not None and el.get("y") is not None:
            tx, ty = float(el.get("x")) / PT_PER_MM, float(el.get("y")) / PT_PER_MM
            if x <= tx <= x + w and y <= ty <= y + h:
                out.append([round(tx, 1), round(ty, 1), s])
    return sorted(out, key=lambda r: (round(r[1] / 1.5), r[0]))


def _svg_crop(fig: str, x: float, y: float, w: float, h: float, dst: Path) -> None:
    s = _figure_file(fig, "svg").read_text(encoding="utf-8")
    vb = f'viewBox="{x * PT_PER_MM:.3f} {y * PT_PER_MM:.3f} {w * PT_PER_MM:.3f} {h * PT_PER_MM:.3f}"'
    s = re.sub(r'viewBox="[^"]+"', vb, s, count=1)
    s = re.sub(r'(<svg[^>]*?)\swidth="[^"]+"', rf'\1 width="{w:.1f}mm"', s, count=1)
    s = re.sub(r'(<svg[^>]*?)\sheight="[^"]+"', rf'\1 height="{h:.1f}mm"', s, count=1)
    dst.write_text(s, encoding="utf-8", newline="\n")


def bundle(out: Path) -> Path:
    from PIL import Image

    if out.exists():
        shutil.rmtree(out)
    for sub in ("reference", "vector", "figures", "mermaid"):
        (out / sub).mkdir(parents=True)
    px = DPI / 25.4
    rows, lab = [], {}
    for sid, (fig, x, y, w, h) in SLOTS.items():
        im = Image.open(PNG / f"{fig}.png")
        im.crop((round(x * px), round(y * px), round((x + w) * px), round((y + h) * px))).save(
            out / "reference" / f"{sid}.png", dpi=(DPI, DPI))
        _svg_crop(fig, x, y, w, h, out / "vector" / f"{sid}.svg")
        lab[sid] = labels(fig, x, y, w, h)
        rows.append({"id": sid, "figure": fig, "x_mm": x, "y_mm": y, "width_mm": w, "height_mm": h,
                     "aspect": round(w / h, 2), "width_px_600dpi": round(w * px), "height_px_600dpi": round(h * px),
                     "labels": len(lab[sid])})
    for fig in sorted({v[0] for v in SLOTS.values()}):
        for ext in ("pdf", "svg"):
            shutil.copy2(_figure_file(fig, ext), out / "figures" / f"{fig}.{ext}")
    with open(out / "slots.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "labels.json").write_text(json.dumps(lab, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for p in sorted(SRC.glob("*.mmd")):
        shutil.copy2(p, out / "mermaid" / p.name)
        for d, ext in ((OUT, "svg"), (PREVIEW, "png")):
            if (d / f"{p.stem}.{ext}").exists():
                shutil.copy2(d / f"{p.stem}.{ext}", out / "mermaid" / f"{p.stem}.{ext}")
    for name in ("README.md", "prompts.md"):
        shutil.copy2(SRC / name, out / name)
    z = out.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, Path(out.name) / p.relative_to(out))
    return z


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--browser", default=EDGE)
    b = sub.add_parser("bundle")
    b.add_argument("--out", type=Path, default=Path("runs/paper/figures/diagram_bundle"))
    a = ap.parse_args(argv)
    if a.cmd == "render":
        m = render(a.browser)
        print(json.dumps({k: v["viewbox"][2:] for k, v in m["diagrams"].items()}))
    else:
        print(bundle(a.out))


if __name__ == "__main__":
    main()
