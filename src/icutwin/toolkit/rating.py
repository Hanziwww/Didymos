"""The clinician study's rating packet (``docs/paper/clinician_study_protocol.md``): one self-contained HTML file per
rater, opened offline in a browser. No network request, no external resource.

* **Contents:** the rater's items in their order and versions (fixed by the builder, ``tools/clinician_study.py``),
  and the modules in the protocol's order (practice, A, B2 | session break | B1, C, the questionnaire).
* **Blinding by construction:**
  - The file holds no answer: no real-or-synthetic label, no source, no outcome. Item ids are opaque; the key stays
    with the study team.
  - Module C's forecast cards are obfuscated (XOR with a key derived from the packet, then base64). They are decoded
    only once the rater has locked the unaided estimates. This deters casual reading of the source; it is not
    encryption.
* **Progress:** saved in the browser after every answer (``localStorage``, keyed by packet and build). An item
  cannot be revisited once answered.
* **Export:** "Export" downloads the responses as JSON (progress or final), with a checksum; "Resume from a file"
  loads an exported file back (another browser, a cleared cache).
* **What leaves the rater:** only the exported JSON (answers and times), returned to the center lead. The packet is
  deleted when the study ends.

``build_packet(spec)`` renders the page; ``obfuscate`` / ``deobfuscate`` are the card's encoding, ``check`` the
export's checksum (FNV-1a over the responses), the same as the page's script.
"""

from __future__ import annotations

import base64
import hashlib
import json

from icutwin.toolkit.casesheet import CSS as SHEET_CSS

STUDY = "icutwin-clinician-1"

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1f23;--muted:#5d6470;--line:#d9dce1;--grid:#eceef1;--accent:#2f6fde;
--band:rgba(47,111,222,.12);--warn:#b5651d}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--panel:#1d2024;--ink:#e8eaed;
--muted:#a3a9b3;--line:#33373d;--grid:#2a2d32;--accent:#6ea0ff;--band:rgba(110,160,255,.16);--warn:#e0a060}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif}header.top{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid
var(--line);padding:8px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}header.top .grow{flex:1}
.bar{height:6px;background:var(--grid);border-radius:3px;overflow:hidden;min-width:160px}.bar>div{height:100%;
background:var(--accent)}main{max-width:960px;margin:0 auto;padding:16px 16px 80px}section{background:var(--panel);
border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}h1{font-size:21px;margin:0 0 6px}
h2{font-size:17px;margin:0 0 8px}button{font:inherit;background:var(--accent);color:#fff;border:0;border-radius:8px;
padding:8px 16px;cursor:pointer}button[disabled]{opacity:.45;cursor:default}button.sec{background:transparent;
color:var(--accent);border:1px solid var(--accent)}.q{margin:12px 0}.q>.lab{font-weight:600;display:block;
margin-bottom:6px}.opts{display:flex;flex-wrap:wrap;gap:6px}.opts label{border:1px solid var(--line);border-radius:8px;
padding:6px 12px;cursor:pointer;user-select:none}.opts input{margin-right:6px}.ladder{display:inline-flex;
flex-direction:column;max-width:100%}.scale{display:flex;justify-content:space-between;flex-wrap:wrap;gap:2px 14px;
margin-top:3px;color:var(--muted);font-size:12px}textarea{width:100%;min-height:52px;font:inherit;
background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px}input[type=number]{
width:90px;font:inherit;padding:4px 6px;background:var(--bg);color:var(--ink);border:1px solid var(--line);
border-radius:6px}.muted{color:var(--muted)}.warn{color:var(--warn)}.risk{font-size:20px;font-weight:700}
.factor{border-left:3px solid var(--accent);padding:4px 10px;margin:8px 0}.factor ul{margin:4px 0 0 18px;padding:0}
.card{border:2px solid var(--accent);border-radius:10px;padding:12px 14px;margin:12px 0}.card table{border-collapse:
collapse}.card td{padding:3px 12px 3px 0;vertical-align:top}.card td.num{text-align:right;white-space:nowrap}
.card svg{width:100%;max-width:640px;height:auto;display:block}.card svg text{font-size:11px;fill:var(--muted)}
.card svg .grid{stroke:var(--grid)}.card svg .axis{stroke:var(--line)}section,header.top{overflow-wrap:break-word}.grid2{display:grid;
grid-template-columns:1fr auto;gap:6px 14px;align-items:center;max-width:560px}.tag{display:inline-block;
border:1px solid var(--line);border-radius:999px;padding:0 9px;font-size:12px;color:var(--muted)}
"""

JS = r"""
"use strict";
const P = JSON.parse(document.getElementById("packet").textContent);
const KEY = "icutwin-clinician:" + P.packet + ":" + P.build;
const main = document.querySelector("main");
let S = load() || fresh();
function fresh() {
  return {study: P.study, packet: P.packet, build: P.build, consent: null, started: null, updated: null,
          pos: {m: -1, i: 0}, responses: {}, questionnaire: null, draft: null};
}
function load() { try { const x = localStorage.getItem(KEY); return x ? JSON.parse(x) : null; } catch (e) { return null; } }
function save() {
  S.updated = new Date().toISOString();
  try { localStorage.setItem(KEY, JSON.stringify(S)); document.getElementById("saved").textContent = "saved"; }
  catch (e) { document.getElementById("saved").textContent = "not saved in this browser: export regularly"; }
}
function fnv(s) { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return ("00000000" + h.toString(16)).slice(-8); }
function check(o) { return fnv(JSON.stringify(o.responses) + "|" + JSON.stringify(o.questionnaire)); }
function exportJSON(final) {
  const body = Object.assign({}, S, {exported: new Date().toISOString(), final: !!final});
  delete body.draft; body.check = check(body);
  const b = new Blob([JSON.stringify(body, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(b);
  a.download = "clinician_" + P.packet + "_" + (final ? "final" : "progress") + "_" + body.exported.slice(0, 19).replace(/[:T]/g, "-") + ".json";
  document.body.appendChild(a); a.click(); a.remove();
}
function importJSON(file) {
  const r = new FileReader();
  r.onload = () => {
    try {
      const o = JSON.parse(r.result);
      if (o.packet !== P.packet || o.build !== P.build) { alert("This file belongs to another packet."); return; }
      if (o.check !== check(o)) { alert("This file's checksum does not match: it was changed or damaged."); return; }
      delete o.exported; delete o.final; delete o.check; o.draft = null; S = o; save(); render();
    } catch (e) { alert("This file could not be read."); }
  };
  r.readAsText(file);
}
function now() { return Date.now(); }
function esc(s) { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
function counts() {
  let done = 0, total = 0;
  P.modules.forEach(m => { if (m.practice || m.kind === "questionnaire") return; total += m.items.length;
    done += (S.responses[m.id] || []).length; });
  return [done, total];
}
function header() {
  const [d, t] = counts();
  document.getElementById("prog").innerHTML = "<div style='width:" + (t ? 100 * d / t : 0) + "%'></div>";
  document.getElementById("progtxt").textContent = d + " of " + t + " answered";
}
function radios(name, values, labels) {
  return "<div class='opts'>" + values.map((v, k) => "<label><input type='radio' name='" + name + "' value='" + v + "'>" +
    esc(labels ? labels[k] : v) + "</label>").join("") + "</div>";
}
function ladder(name, values, lo, hi) {   // a rating scale with its end labels under its own options
  return "<div class='ladder'>" + radios(name, values) + "<div class='scale'><span>" + esc(lo) + "</span><span>" + esc(hi) +
    "</span></div></div>";
}
function val(name) { const x = document.querySelector("input[name='" + name + "']:checked"); return x ? x.value : null; }
function deobf(b64) {
  const key = new TextEncoder().encode(P.cardkey); const raw = atob(b64); const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i) ^ key[i % key.length];
  return new TextDecoder().decode(out);
}
function render() {
  header(); window.scrollTo(0, 0);
  if (!S.consent) return startScreen();
  const m = P.modules[S.pos.m];
  if (!m) return endScreen();
  if (S.pos.i === -1) return introScreen(m);
  if (m.kind === "questionnaire") return questionnaire(m);
  if (S.pos.i >= m.items.length) return moduleDone(m);
  const it = m.items[S.pos.i];
  ({realism: realism, course: course, factors: factors, reader: reader})[m.kind](m, it);
}
function advance(m, rec) {
  (S.responses[m.id] = S.responses[m.id] || []).push(rec); S.pos.i += 1; S.draft = null; save(); render();
}
function startScreen() {
  const prof = P.profile !== false;                 // the rater's role and years of practice (off in the short pilot)
  main.innerHTML ="<section><h1>" + esc(P.title) + "</h1><p>Packet <b>" + esc(P.packet) + "</b></p>" + P.intro +
    "<div class='q'><label><input type='checkbox' id='c1'> I have read the study information and agree to take part.</label></div>" +
    "<div class='q'><label><input type='checkbox' id='c2'> I hold PhysioNet credentials for MIMIC-IV and eICU-CRD. I will keep this file " +
    "confidential, not share it, and delete it and its exported files when the study team confirms receipt.</label></div>" +
    (prof ? "<div class='q'><span class='lab'>Your role</span>" + radios("role", ["attending", "fellow", "other"],
      ["attending intensivist", "critical care fellow", "other"]) + "</div><div class='q'><span class='lab'>Years of ICU " +
    "practice</span>" + radios("years", ["<2", "2-5", "6-10", ">10"]) + "</div>" : "") +
    "<button id='go' disabled>Start</button></section><section><h2>Continue in another browser</h2><p class='muted'>" +
    "If you exported your progress before, load that file here.</p><input type='file' id='imp' accept='application/json'></section>";
  const ok = () => document.getElementById("go").disabled = !(c1.checked && c2.checked && (!prof || (val("role") && val("years"))));
  const c1 = document.getElementById("c1"), c2 = document.getElementById("c2");
  main.querySelectorAll("input").forEach(x => x.onchange = ok);
  document.getElementById("go").onclick = () => { S.consent = {participation: true, confidentiality: true, at: new Date().toISOString(),
    role: prof ? val("role") : null, years: prof ? val("years") : null}; S.started = S.consent.at; S.pos = {m: 0, i: -1}; save(); render(); };
  document.getElementById("imp").onchange = e => e.target.files[0] && importJSON(e.target.files[0]);
}
function introScreen(m) {
  main.innerHTML = "<section><h1>" + esc(m.title) + "</h1>" + m.intro + "<p class='muted'>" +
    (m.kind === "questionnaire" ? (m.sus.length + m.trust.length) + " questions." :
     m.items.length + " items. An answered item cannot be changed.") + "</p><button id='go'>Begin</button></section>";
  document.getElementById("go").onclick = () => { S.pos.i = 0; save(); render(); };
}
function moduleDone(m) {
  const next = P.modules[S.pos.m + 1];
  main.innerHTML = "<section><h1>" + esc(m.title) + ": done</h1>" + (m.break_after ?
    "<p>This is the end of session 1. Your progress is saved in this browser. Please also export it now as a backup. " +
    "Continue with session 2 on another day, in this browser, or load the exported file in another one.</p>" +
    "<button class='sec' id='exp'>Export progress</button> " : "") +
    (next ? "<button id='go'>Continue to " + esc(next.title) + "</button>" : "<button id='go'>Finish</button>") + "</section>";
  if (m.break_after) document.getElementById("exp").onclick = () => exportJSON(false);
  document.getElementById("go").onclick = () => { S.pos = {m: S.pos.m + 1, i: -1}; save(); render(); };
}
function endScreen() {
  main.innerHTML = "<section><h1>Thank you</h1><p>All modules are complete. Export your results and send the file to your " +
    "center lead. When the study team confirms receipt, delete this packet and the exported files.</p>" +
    "<button id='exp'>Export final results</button></section>";
  document.getElementById("exp").onclick = () => exportJSON(true);
}
function itemHead(m, it) {
  return "<section><div class='muted'>" + esc(m.title) + " · item " + (S.pos.i + 1) + " of " + m.items.length +
    (m.practice ? " · <span class='tag'>practice, not scored</span>" : "") + "</div>" + (it.lead ? "<p>" + it.lead + "</p>" : "") +
    it.html + "</section>";
}
function realism(m, it) {
  const t0 = now();
  main.innerHTML = itemHead(m, it) + "<section><div class='q'><span class='lab'>Is this ICU stay real or synthetic?</span>" +
    radios("kind", ["real", "synthetic"]) + "</div><div class='q'><span class='lab'>How confident are you?</span>" +
    ladder("conf", [1, 2, 3, 4, 5], "1 guessing", "5 certain") + "</div>" +
    "<div class='q'><span class='lab'>What decided it? <span class='muted'>(optional)</span></span><textarea id='cue'></textarea></div>" +
    "<button id='go' disabled>Submit</button></section>";
  const ok = () => document.getElementById("go").disabled = !(val("kind") && val("conf"));
  main.querySelectorAll("input").forEach(x => x.onchange = ok);
  document.getElementById("go").onclick = () => advance(m, {item: it.id, order: S.pos.i, call: val("kind"),
    confidence: +val("conf"), cue: document.getElementById("cue").value, shown: t0, answered: now()});
}
function course(m, it) {
  const t0 = now();
  main.innerHTML = itemHead(m, it) + "<section><div class='q'><span class='lab'>How physiologically plausible is the course " +
    "in the shaded hours, given the treatment?</span>" + ladder("pl", [1, 2, 3, 4, 5], "1 implausible", "5 fully plausible") +
    "</div><div class='q'><span class='lab'>" +
    esc(it.direction) + "</span>" + radios("dir", ["yes", "no", "unsure"]) + "</div><button id='go' disabled>Submit</button></section>";
  const ok = () => document.getElementById("go").disabled = !(val("pl") && val("dir"));
  main.querySelectorAll("input").forEach(x => x.onchange = ok);
  document.getElementById("go").onclick = () => advance(m, {item: it.id, order: S.pos.i, plausibility: +val("pl"),
    direction: val("dir"), shown: t0, answered: now()});
}
function factors(m, it) {
  const t0 = now();
  const fs = it.factors.map((f, k) => "<div class='factor'><b>" + esc(f.name) + "</b><ul>" + f.events.map(e => "<li>" + esc(e) +
    "</li>").join("") + "</ul><label class='muted'><input type='checkbox' id='irr" + k + "'> clinically irrelevant to this risk</label></div>").join("");
  main.innerHTML = itemHead(m, it) + "<section><p>The model's risk of <b>" + esc(it.outcome) + "</b>: <span class='risk'>" +
    esc(it.risk) + "</span></p><p>The three factors the model relied on most:</p>" + fs + "<div class='q'><span class='lab'>" +
    "How relevant are these factors, taken together, to this patient's risk?</span>" +
    ladder("rel", [1, 2, 3, 4, 5], "1 not relevant", "5 highly relevant") + "</div><button id='go' disabled>Submit</button></section>";
  const ok = () => document.getElementById("go").disabled = !val("rel");
  main.querySelectorAll("input[name=rel]").forEach(x => x.onchange = ok);
  document.getElementById("go").onclick = () => advance(m, {item: it.id, order: S.pos.i, relevance: +val("rel"),
    irrelevant: it.factors.map((f, k) => document.getElementById("irr" + k).checked), shown: t0, answered: now()});
}
function numbers(prefix, outs, init) {
  return "<div class='grid2'>" + outs.map(o => "<label for='" + prefix + o.key + "'>" + esc(o.label) + "</label><span><input type='number' " +
    "min='0' max='100' step='1' id='" + prefix + o.key + "' value='" + (init && init[o.key] != null ? init[o.key] : "") + "'> %</span>").join("") + "</div>";
}
function read(prefix, outs) {
  const o = {}; for (const x of outs) { const v = document.getElementById(prefix + x.key).value;
    if (v === "" || isNaN(+v) || +v < 0 || +v > 100) return null; o[x.key] = +v; } return o;
}
function reader(m, it) {
  if (!S.draft || S.draft.item !== it.id) { S.draft = {item: it.id, shown: now(), unaided: null}; save(); }
  const d = S.draft;
  let h = itemHead(m, it) + "<section><h2>Your estimates</h2><p class='muted'>For each outcome, the probability (0-100%).</p>";
  if (!d.unaided) {
    main.innerHTML = h + numbers("u_", it.outcomes, null) + "<p><button id='lock'>Lock my estimates and show the model's forecast</button></p></section>";
    document.getElementById("lock").onclick = () => { const u = read("u_", it.outcomes);
      if (!u) { alert("Enter every probability as a number from 0 to 100."); return; }
      d.unaided = u; d.locked = now(); save(); render(); };
    return;
  }
  main.innerHTML = h + "<p>Locked: " + it.outcomes.map(o => esc(o.label) + " " + d.unaided[o.key] + "%").join("; ") +
    "</p></section><section class='card'>" + deobf(it.card) + "</section><section><h2>Your final estimates</h2>" +
    numbers("a_", it.outcomes, d.unaided) + "<div class='q'><span class='lab'>Did the forecast change your assessment?</span>" +
    radios("chg", ["yes", "no"]) + "</div><button id='go'>Submit</button></section>";
  document.getElementById("go").onclick = () => { const a = read("a_", it.outcomes);
    if (!a || !val("chg")) { alert("Enter every final probability and answer the question."); return; }
    advance(m, {item: it.id, order: S.pos.i, unaided: d.unaided, aided: a, changed: val("chg"), shown: d.shown,
                locked: d.locked, answered: now()}); };
}
function questionnaire(m) {
  const t0 = now();
  const block = (title, qs, n, lo, hi, pre) => "<section><h2>" + esc(title) + "</h2>" + qs.map((q, k) => "<div class='q'><span class='lab'>" +
    esc(q) + "</span>" + ladder(pre + k, Array.from({length: n}, (_, j) => j + 1), lo, hi) + "</div>").join("") + "</section>";
  main.innerHTML = "<section><h1>" + esc(m.title) + "</h1>" + m.intro + "</section>" +
    block("Usability of the forecast card", m.sus, 5, "1 strongly disagree", "5 strongly agree", "s") +
    block("Trust in the forecast", m.trust, 7, "1 not at all", "7 extremely", "t") + "<section><button id='go'>Submit</button></section>";
  document.getElementById("go").onclick = () => {
    const s = m.sus.map((_, k) => val("s" + k)), t = m.trust.map((_, k) => val("t" + k));
    if (s.includes(null) || t.includes(null)) { alert("Please answer every question."); return; }
    S.questionnaire = {sus: s.map(Number), trust: t.map(Number), shown: t0, answered: now()};
    S.pos = {m: S.pos.m + 1, i: -1}; save(); render(); };
}
document.getElementById("exportbtn").onclick = () => exportJSON(false);
render();
"""

SUS = ("I think that I would like to use this forecast frequently.",
       "I found the forecast unnecessarily complex.",
       "I thought the forecast was easy to use.",
       "I think that I would need the support of a technical person to be able to use this forecast.",
       "I found the various parts of the forecast well integrated.",
       "I thought there was too much inconsistency in the forecast.",
       "I would imagine that most clinicians would learn to use this forecast very quickly.",
       "I found the forecast very cumbersome to use.",
       "I felt very confident using the forecast.",
       "I needed to learn a lot of things before I could get going with this forecast.")
TRUST = ("The forecast is reliable.", "I can trust the forecast.", "I am confident in the forecast.",
         "The forecast is dependable.", "I am wary of the forecast.",
         "The forecast could lead to harmful decisions.")


def card_key(packet: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{packet}".encode()).hexdigest()


def obfuscate(text: str, key: str) -> str:
    k, raw = key.encode(), text.encode("utf-8")
    return base64.b64encode(bytes(b ^ k[i % len(k)] for i, b in enumerate(raw))).decode("ascii")


def deobfuscate(b64: str, key: str) -> str:
    k, raw = key.encode(), base64.b64decode(b64)
    return bytes(b ^ k[i % len(k)] for i, b in enumerate(raw)).decode("utf-8")


def fnv(s: str) -> str:
    """FNV-1a over UTF-16 code units, as the page's ``charCodeAt`` loop."""
    raw = s.encode("utf-16-le")
    h = 0x811C9DC5
    for k in range(0, len(raw), 2):
        h ^= raw[k] | (raw[k + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def check(export: dict) -> str:
    """The page's checksum of an exported file (FNV-1a of the responses and questionnaire as the page serialized
    them; JSON with no spaces, keys in insertion order, as JavaScript's ``JSON.stringify``)."""
    dump = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False)  # noqa: E731
    return fnv(dump(export.get("responses")) + "|" + dump(export.get("questionnaire")))


def build_packet(spec: dict) -> str:
    """The page of one packet. ``spec``: title, intro (HTML), packet, build, cardkey, modules (each: id, title, kind
    in realism / course / factors / reader / questionnaire, intro, items, practice, break_after), and optionally
    profile (False: no role or years-of-practice questions at the start; the short pilot)."""
    data = {"study": STUDY, **spec}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width, initial-scale=1\"><title>ICU case review</title>"
            f"<style>{CSS}{SHEET_CSS}</style></head><body><header class=\"top\"><b>ICU case review</b>"
            f"<span class=\"muted\">packet {spec['packet']}</span><span class=\"grow\"></span>"
            "<span class=\"muted\" id=\"progtxt\"></span><div class=\"bar\" id=\"prog\"></div>"
            "<span class=\"muted\" id=\"saved\"></span><button class=\"sec\" id=\"exportbtn\">Export progress</button>"
            f"</header><main></main><script type=\"application/json\" id=\"packet\">{blob}</script>"
            f"<script>{JS}</script></body></html>")
