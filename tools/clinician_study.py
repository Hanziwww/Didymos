"""Paper P's clinician study materials (amendment 12; ``docs/paper/clinician_study_protocol.md``): the items, their
simulations, the self-contained rating packets (``toolkit.rating``) and the key. Runs on the approved host (the 4080
host has every store and the model).

    python tools/clinician_study.py select            # CPU: the items of every module (items.json)
    python tools/clinician_study.py simulate          # GPU: B1's rollouts, B2's factors, C's forecasts
    python tools/clinician_study.py packets           # CPU: case sheets, packets R01-R20 and P01-P02, the key
    python tools/clinician_study.py collect <exported json files>   # CPU: checked responses joined to the key

Everything is written under ``runs/toolkit/clinician`` (git-ignored). ``key.json`` and ``items.json`` never leave
the host; the packets go to the raters (all PhysioNet-credentialed for MIMIC-IV and eICU-CRD; the study lead's decision,
2026-10-10).

Selection (seeded), in this order so that a stay appears in one module only:
* C: 48 A1 24-hour-landmark cases, MIMIC-IV and eICU (half each), 12 ICU deaths and 36 survivors (a case-cohort
  sample). The outcomes asked are ICU death (always) and AKI stage 2 or more within 48 h (when the patient is not
  already in that state at the landmark). A new vasopressor within 24 h was dropped before the pilot: 1 event in
  its 44 eligible cases (``C_ASKED``).
* B2: 20 A1 rolling anchors, 10 for death within 24 h and 10 for a new vasopressor within 24 h (half per dataset), in
  the upper half of the Twin probe's risk, with a valid label and prompts of at most 8,192 tokens.
* B1: 24 F5 starters (MIMIC-IV and eICU; four per treatment: norepinephrine, fluid bolus, furosemide, invasive
  ventilation, CRRT, insulin), cut one minute before the real start; their real futures must align with the prompt.
* A: 80 first-24-hour sheets, 40 real held-out stays (20 per dataset) and 40 bank twins of the same dataset, each
  matched to a real one on the length-of-stay band; and 4 practice sheets (2 real, 2 twins) outside the scored set.

Packets: per rater, each module's items in a random order (seeded by the rater's number); B1 counterbalanced
(version (k + j) mod 2: twin or real course), B2 counterbalanced (method (k + j) mod 3: editing, knockout, random).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

DS = ("mimic", "eicu")
TREATS = ("norepinephrine", "fluid_bolus", "furosemide", "imv", "crrt", "insulin")
DIRECTION = {"norepinephrine": "Does mean arterial pressure respond in the direction you would expect?",
             "fluid_bolus": "Does mean arterial pressure respond in the direction you would expect?",
             "furosemide": "Does urine output respond in the direction you would expect?",
             "imv": "Does oxygenation (SpO2) respond in the direction you would expect?",
             "crrt": "Does potassium respond in the direction you would expect?",
             "insulin": "Does glucose respond in the direction you would expect?"}
STARTED = {"norepinephrine": "Norepinephrine started", "fluid_bolus": "Fluid bolus given",
           "furosemide": "Furosemide given", "imv": "Invasive ventilation started", "crrt": "CRRT started",
           "insulin": "Insulin infusion started"}
C_OUTCOMES = (("icu_death", "icu_death@-1", "Death in the ICU (this admission)"),
              ("vasopressor", "vasopressor_onset@1440", "A new vasopressor within 24 h"),
              ("aki", "aki_2@2880", "AKI stage 2 or more within 48 h"))
# the outcomes raters estimate (2026-10-10, before the pilot): a new vasopressor dropped, 1 event in 44 eligible cases
C_ASKED = ("icu_death", "aki")
B2_PROGRAMS = (("decompensation@1440", "death within 24 h"), ("vasopressor_onset@1440", "a new vasopressor within 24 h"))
METHODS = ("edit", "knockout", "random")


def session():
    from icutwin.toolkit.session import Paths, Session

    return Session(Paths.default(), backend="vllm", engines=1, gpu_memory_utilization=0.85, workers=4)


def out_dir(s) -> Path:
    d = s.paths.out / "clinician"
    d.mkdir(parents=True, exist_ok=True)
    return d


def real_rows(s, split: str, stays: set) -> dict:
    from icutwin.eval import bench

    return {r["stay_id"]: r for r in bench.real_rows(s.paths.tokens, split, stays, datasets=DS)}


# -- select ---------------------------------------------------------------------------------------------------------
def select(s, a) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from icutwin.factory import oracle
    from icutwin.toolkit.inputs import from_ids

    rng = np.random.default_rng(a.seed)
    paper, v = s.paths.paper, s.vocab
    used: set = set()
    # C: the 24-hour landmark, a case-cohort sample
    an = pd.read_parquet(paper / "a1/test_L24/anchors.parquet")
    pr = pd.read_parquet(paper / "a1/test_L24/predictions.parquet")
    lab = {p: g.set_index("anchor")["label"] for p, g in pr[pr["program"].isin([x[1] for x in C_OUTCOMES])]
           .groupby("program")}
    an = an[an["dataset"].isin(DS) & an["anchor"].isin(lab["icu_death@-1"].index)].reset_index(drop=True)
    an["death"] = an["anchor"].map(lab["icu_death@-1"])
    C = []
    for ds in DS:
        for dead, k in ((1.0, a.c_deaths // 2), (0.0, (a.c_cases - a.c_deaths) // 2)):
            pool = an[(an["dataset"] == ds) & (an["death"] == dead)]
            for x in pool.iloc[rng.permutation(len(pool))[:k]].itertuples():
                outs = {}
                for key, prog, _l in C_OUTCOMES:
                    y = lab[prog].get(x.anchor) if prog in lab else None
                    if y is not None and np.isfinite(y):
                        outs[key] = float(y)
                C.append({"anchor": x.anchor, "stay": str(x.twin), "dataset": ds, "hours": 24.0, "labels": outs})
                used.add(str(x.twin))
    # B2: rolling anchors in the upper half of the probe's risk
    ar = pd.read_parquet(paper / "a1/test_R/anchors.parquet").set_index("anchor")
    pp = pd.read_parquet(paper / "a1/test_R/probe_predictions.parquet")
    prr = pd.read_parquet(paper / "a1/test_R/predictions.parquet")
    B2 = []
    for prog, _name in B2_PROGRAMS:
        valid = set(prr[(prr["program"] == prog) & np.isfinite(prr["label"])]["anchor"])
        q = pp[(pp["program"] == prog) & pp["dataset"].isin(DS) & pp["anchor"].isin(valid)].copy()
        q = q[q["anchor"].map(ar["prompt_len"]) <= 8192]
        q = q[q["p"] >= q["p"].median()]
        for ds in DS:
            pool = q[q["dataset"] == ds]
            k = 0
            for i in rng.permutation(len(pool)):
                x = pool.iloc[int(i)]
                stay = str(ar.loc[x["anchor"], "twin"])
                if stay in used:
                    continue
                B2.append({"anchor": x["anchor"], "stay": stay, "dataset": ds, "hours": float(ar.loc[x["anchor"], "hours"]),
                           "program": prog, "probe_risk": float(x["p"])})
                used.add(stay)
                k += 1
                if k == a.b2_per // 2:
                    break
    # B1: F5's starters, the real future aligned with the prompt
    d = pd.read_parquet(paper / "f5/design.parquet")
    d = d[(d["group"] == "starter") & (d["arm"] == "give") & d["dataset"].isin(DS) & d["treatment"].isin(TREATS)]
    rows = real_rows(s, "held_out", set(d["twin"].astype(str)))
    B1 = []
    for t in TREATS:
        pool = d[d["treatment"] == t]
        k = 0
        for i in rng.permutation(len(pool)):
            x = pool.iloc[int(i)]
            stay = str(x["twin"])
            if stay in used or stay not in rows:
                continue
            p = from_ids(rows[stay]["ids"], stay, x["dataset"], "held_out")
            prompt = p.at(v, float(x["hours"])).prompt
            fut = continuation(s, p, prompt)
            if fut is None or len(prompt) > 16000:
                continue
            B1.append({"anchor": x["anchor"], "stay": stay, "dataset": x["dataset"], "hours": float(x["hours"]),
                       "treatment": t, "give_token": int(x["give_token"])})
            used.add(stay)
            k += 1
            if k == a.b1_per:
                break
    # A: real held-out first 24 hours against bank twins matched on dataset and length-of-stay band
    twins: dict = {}
    for f in sorted(s.paths.bank.glob("part*/*.parquet")):
        t = pq.read_table(f, columns=["twin", "dataset", "ids"])
        for r in t.to_pylist():
            if r["dataset"] in DS:
                twins.setdefault(r["dataset"], []).append(r)
    A, practice = [], []
    for ds in DS:
        pool: dict = {}
        for i in rng.permutation(len(twins[ds])):
            r = twins[ds][int(i)]
            pool.setdefault(oracle.los_band(from_ids(r["ids"], r["twin"], ds, "twin").los_hours(v)), []).append(r["twin"])
        t = pq.read_table(s.paths.tokens / ds / "held_out.parquet", columns=["stay_id", "ids"])
        k = 0
        for i in rng.permutation(t.num_rows):
            r = t.slice(int(i), 1).to_pylist()[0]
            stay = str(r["stay_id"])
            if stay in used:
                continue
            band = oracle.los_band(from_ids(r["ids"], stay, ds, "held_out").los_hours(v))
            if not pool.get(band):
                continue
            tw = pool[band].pop()
            pair = [{"kind": "real", "ref": f"split:{ds}/held_out/{stay}", "dataset": ds, "band": int(band)},
                    {"kind": "twin", "ref": f"bank:{tw}", "dataset": ds, "band": int(band)}]
            used.add(stay)
            if k < a.a_real // 2:
                A += pair
            else:
                practice += pair
            k += 1
            if k == a.a_real // 2 + 1:              # one more pair per dataset: the practice sheets
                break
    import secrets

    items = {"seed": a.seed, "salt": secrets.token_hex(8),       # the opaque item ids' secret (host only)
             "A": A, "practice": practice, "B1": B1, "B2": B2, "C": C}
    (out_dir(s) / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: len(x) for k, x in items.items() if isinstance(x, list)}))


def continuation(s, p, prompt: list):
    """The stay's tokens after the prompt (the whole remaining stay; F3's D1 alignment), or None."""
    from icutwin.tokenization.encode import prospective

    v = s.vocab
    tl, cont = v.id("[TL]"), v.id("[CONT]")
    st = p.stay(v)
    last_h = max((e[0] - st.icu_in).total_seconds() for e in st.timeline) / 3600 if st.timeline else 0.0
    full = list(prospective(st, v, last_h + 1.0)) + [v.id("[END]"), v.id("[EOS]")]
    if tl not in prompt or tl not in full or cont in prompt or cont in full:
        return None
    ptl, ftl = prompt[prompt.index(tl) + 1:], full[full.index(tl) + 1:]
    if ftl[:len(ptl)] != ptl:
        return None
    return [int(x) for x in ftl[len(ptl):]]


# -- simulate -------------------------------------------------------------------------------------------------------
def simulate(s, a) -> None:
    from icutwin.eval import attribution as at
    from icutwin.toolkit import explain, forecast
    from icutwin.toolkit.inputs import from_ids
    from icutwin.toolkit.jobs import Job, SimRequest

    out = out_dir(s)
    items = json.loads((out / "items.json").read_text())
    v = s.vocab
    caps24 = json.loads((s.paths.paper / "f5/rollouts_design.json").read_text())["caps"]
    rows = real_rows(s, "held_out", {x["stay"] for k in ("B1", "B2", "C") for x in items[k]})
    pat = {st: from_ids(r["ids"], st, r["dataset"], "held_out") for st, r in rows.items()}
    # B1: 16 give-arm rollouts per item
    job = Job(out / "b1_rollouts")
    if not (job.root / "requests.parquet").exists():
        reqs = [SimRequest(f"b1_{j}", tuple(pat[x["stay"]].at(v, x["hours"]).prompt), a.b1_rollouts, 90_000 + j,
                           int(caps24[x["dataset"]]), give=(x["give_token"],)) for j, x in enumerate(items["B1"])]
        job.write_requests(reqs)
        job.write_spec({"tool": "clinician_b1", "items": len(reqs), "provenance": s.provenance()})
    s.sample(job)
    # B2: editing and knockout on the Twin probe (R design); random groups drawn from the same list
    rng = np.random.default_rng(items["seed"] + 7)
    b2 = []
    for x in items["B2"]:
        p = pat[x["stay"]]
        coef = explain.probe_of(s, x["program"], "R")
        _st, groups, p0, edits = explain.groups_of(s, p, x["hours"])
        names = list(edits)
        lg = s.teacher.probe_logits([p0] + [edits[g] for g in names], coef)
        edit = {g: float(lg[0] - y) for g, y in zip(names, lg[1:], strict=True)}
        positions: dict = {}
        for k, gname in enumerate(at.token_groups(p0, v)):
            if gname is not None:
                positions.setdefault(gname, []).append(k)
        ko = s.teacher.knockout(p0, coef, positions)
        pick = [names[i] for i in rng.permutation(len(names))[:3]]
        b2.append({"anchor": x["anchor"], "logit": float(lg[0]),
                   "edit": sorted(edit, key=lambda g: -edit[g])[:3],
                   "knockout": sorted(ko, key=lambda g: -ko[g])[:3], "random": pick,
                   "edit_scores": edit, "knockout_scores": ko})
    (out / "b2_factors.json").write_text(json.dumps(b2, indent=1), encoding="utf-8", newline="\n")
    # C: the twin's standard forecast at four times A1's 168-hour caps (F1)
    caps168 = json.loads((s.paths.paper / "a1/test_L24/design.json").read_text())["caps"]
    caps = {d: min(int(c * 4), s.context - 1) for d, c in caps168.items()}
    res = forecast.run(s, [pat[x["stay"]] for x in items["C"]], [24.0], n=a.c_rollouts, caps=caps, cap_hours=168.0,
                       cap_mult=4.0, design="L24", job_dir=out / "c_forecast", seed=items["seed"] + 11)
    res = res if isinstance(res, list) else [res]
    (out / "c_forecasts.json").write_text(json.dumps([{"stay": x["stay"], "result": r_} for x, r_ in
                                                      zip(items["C"], res, strict=True)], indent=1),
                                          encoding="utf-8", newline="\n")
    print(json.dumps({"b1_items": len(items["B1"]), "b2_items": len(b2), "c_cases": len(res)}))


# -- packets --------------------------------------------------------------------------------------------------------
def rho(s):
    from icutwin.tokenization.decode import load_persistence

    f = s.paths.tokens / "rank_persistence.json"
    return load_persistence(f, s.vocab) if f.exists() else None


def features(s, ids, stay_id: str, r):
    from icutwin.factory import twins

    return twins.decode_features([int(x) for x in ids], s.vocab, stay_id, 0, r)


def describe(code: str, num) -> str:
    parts = code.split("//")
    name = parts[1] if len(parts) > 1 else parts[0]
    name = re.sub(r"_(set|obs)$", "", name).replace("_", " ")
    unit = parts[2] if len(parts) > 2 and parts[0] in ("LAB",) else ""
    if parts[0] == "MED_CON" and len(parts) == 5:
        return f"{name} {parts[3].replace('_', ' ')}" + (f" {num:.2g} {parts[2]}" if num is not None and np.isfinite(num)
                                                         and parts[2] != "UNK" else "")
    if parts[0] in ("RESP",) and "category" in parts[1]:
        return f"{parts[1].split('_')[0]} {parts[2].replace('_', ' ')}"
    val = f" {num:.3g}" if num is not None and np.isfinite(num) else ""
    return f"{name}{val} {unit}".strip()


FAMILY = {"vitals": "Vital signs", "labs": "Laboratory values", "vasoactives": "Vasoactive drugs", "sedation":
          "Sedation and analgesia", "fluids_output": "Fluids and outputs", "respiratory": "Respiratory support",
          "rrt_mcs": "Renal replacement or circulatory support", "other_meds": "Other medications",
          "assessments": "Assessments", "other": "Other records"}
BLOCK = {"0-6h": "in the last 6 hours", "6-24h": "6 to 24 hours before", ">24h": "more than 24 hours before"}


def factor(stay_lab, stay_feat, groups: dict, g: str, hours: float) -> dict:
    if g == "statics":
        return {"name": "Admission information", "events": ["age, sex, unit, admission source and type, diagnosis group"]}
    fam, _, blk = g.partition("|")
    idx = groups.get(g, [])
    tl = stay_feat.timeline if len(stay_feat.timeline) == len(stay_lab.timeline) else stay_lab.timeline
    ev = []
    for k in sorted(idx, key=lambda k: tl[k][0], reverse=True)[:3]:
        t, _r, code, num, _x = tl[k]
        h = (t - stay_lab.icu_in).total_seconds() / 3600 - hours
        ev.append(f"{describe(code, num)} ({h:+.1f} h)")
    return {"name": f"{FAMILY.get(fam, fam)}, {BLOCK.get(blk, blk)}", "events": ev}


def c_card(res: dict, outcomes: list) -> str:
    from icutwin.toolkit import card

    o = res["outputs"]
    risks = o["risks"]
    rows = []
    for key, prog, label in C_OUTCOMES:
        if key not in outcomes:
            continue
        r = risks.get(prog) or {}
        p = r.get("risk")
        note = "" if r.get("calibrated") else (f" <span class='muted'>(raw; in the paper's validation, observed over "
                                                f"expected was {r['paper_oe']:.2f})</span>" if r.get("paper_oe") else
                                                " <span class='muted'>(raw)</span>")
        rows.append(f"<tr><td>{card.esc(label)}</td><td class='num'><b>{'–' if p is None else f'{100 * p:.0f}%'}</b>"
                    f"</td><td>{note}</td></tr>")
    inc = o.get("incidence") or {}
    chart = card.line_chart({f"{card.CAUSE_NAMES[c]}": (inc["hours"], inc[c]["e2"], False) for c in ("death", "vasopressor")
                             if c in inc}, "", "") if inc.get("hours") else ""   # the axes named in the heading
    return ("<h2>The ICU digital twin's forecast</h2><p class='muted'>From 128 simulated futures of this patient (the "
            "twin model, MUC32_main).</p><table>" + "".join(rows) + "</table><h3>Cumulative probability by hours from now (the next 48 hours)</h3>" + chart +
            "<ul class='muted'><li>Risks of starting a treatment were over-predicted about two-fold in validation "
            "unless recalibrated.</li><li>The forecast is a statistical estimate from similar courses, not an "
            "assessment of this patient.</li></ul>")


INTRO = """<p>You will review ICU case sheets: the patient's admission, vital signs, respiratory support and treatments,
urine output and laboratory values, as hours since ICU admission. The study has two sessions of about 90 minutes:
session 1 is modules A and B2, session 2 is modules B1 and C and a short questionnaire. Your answers are saved in this
browser after every item; you can stop and continue later. An answered item cannot be changed.</p>
<p>Use a recent Chrome, Edge or Firefox on a computer. Export your progress at the end of each session as a backup.</p>"""
MODULE_INTRO = {
    "T": "<p>Four practice sheets to learn the layout and the questions. They are not scored.</p>",
    "A": "<p>Each sheet shows the first 24 hours of an ICU stay (or the whole stay, if shorter). Half of the sheets are "
         "real ICU stays and half are synthetic, generated by a model. For each, say whether you think it is real or "
         "synthetic, and how confident you are. There is no feedback.</p>",
    "B2": "<p>Each sheet shows a patient up to now, a model's risk of an outcome, and the three factors the model relied on "
          "most. Rate how relevant these factors are to this patient's risk, and mark any you find clinically "
          "irrelevant.</p>",
    "B1": "<p>Each sheet shows a patient up to the dashed line, where a treatment was started, and the next 6 hours "
          "(shaded). The shaded course is either the patient's observed course or a model's simulation. Rate how "
          "physiologically plausible the shaded course is, given the treatment.</p>",
    "C": "<p>Each case is a real ICU patient 24 hours after admission. First estimate the probability of each outcome "
         "from the sheet alone, and lock your estimates. Then the ICU digital twin's forecast is shown, and you give "
         "your final estimates.</p>",
    "Q": "<p>A few questions about the forecast you used in module C.</p>"}


LEADS = """ICU case review: instructions for center leads (build {build}, {n} packets)

0. The pilot comes first. P01 and P02 go to two physicians who are not raters, each with pilot_feedback_form.txt.
   Their exports and forms go to the study team, which records any change before the first rater starts. Send no R packet until the study team releases this
   build. A change after the pilot means a new build, with a new build code, and these files are then void.
1. Assign each rater one packet, in number order (the study team tells you which numbers are your center's). Keep
   the list of who has which packet yourself. Never send names to the study team: answers are identified by the
   packet only. Tell the study team which packet numbers your center used; centers are reported as codes only.
   Leave unused packets unused: do not reassign a packet once a rater has opened it.
2. Before sending a packet, confirm that the rater:
   - holds PhysioNet credentials for MIMIC-IV and eICU-CRD (a packet contains renderings of those datasets' ICU
     stays);
   - is a board-certified intensivist, or a critical care fellow with at least two years of ICU practice;
   - did not work on the model or this study.
3. Send each packet through your center's secure file channel. Do not use public file-sharing links. Check that the
   file arrived intact: its SHA-256 is in manifest.json.
4. The rater opens the file in a recent Chrome, Edge or Firefox on a computer and works offline, in two sessions:
   - session 1: modules A and B2, about 90 minutes;
   - session 2: B1, C and the questionnaire, about 100 minutes.
   Progress is saved in the browser. Ask raters to click "Export progress" at the end of session 1, and "Export
   final results" at the end. If a browser loses its progress, the start screen's "Continue in another browser"
   loads the last export.
5. Collect the exported files (clinician_<packet>_final_<date>.json; answers only, no patient data), and the
   progress files of anyone who did not finish. Send them to the study team.
6. When the study team confirms receipt, ask each rater to delete the packet and the exported files, and delete
   your copies.
"""


PILOT = """ICU case review: pilot feedback (packets P01 and P02, build {build})

Work through the whole packet as a rater would, in two sessions. Then answer below and return this file with your
exported results to the center lead. Your answers fix the case sheet, the instructions, the cue codebook and the
timing before the first rater starts (protocol section 4, Pilot).

1. Time: how long did session 1 (practice, A, B2) and session 2 (B1, C, questionnaire) take? Were you rushed or
   tired at any point?

2. Rendering: did any case sheet fail to display, overlap, or show values that looked broken (empty charts,
   impossible numbers, misplaced labels)? Note the module and the item number shown at the top of the page.

3. Instructions: was anything in the instructions or questions unclear, or open to more than one reading?

4. Module A: apart from the clinical content, did anything give away whether a sheet was real or synthetic
   (layout, timing of entries, missing sections, loading)?

5. Module B1: apart from the clinical content, did anything give away whether the shaded course was observed or
   simulated?

6. Module C: was the forecast card clear? Did anything in it confuse you or seem misleading?

7. The cues you relied on in module A, as kinds (e.g. implausible values, charting rhythm, missing documentation,
   treatment logic, timing): they seed the cue codebook.

8. Anything else.
"""


def packets(s, a) -> None:
    from icutwin.eval import attribution as at
    from icutwin.toolkit import casesheet, rating
    from icutwin.toolkit import inputs as tin
    from icutwin.toolkit.jobs import Job
    from icutwin.tokenization.encode import prospective_stay

    out = out_dir(s)
    items = json.loads((out / "items.json").read_text())
    salt, seed = items["salt"], items["seed"]
    v, r = s.vocab, rho(s)

    def oid(module: str, key: str) -> str:
        return hashlib.sha256(f"{salt}|{module}|{key}".encode()).hexdigest()[:10]

    def load(ref: str):
        return tin.load(s, ref)

    key = {"salt": salt, "items": {}, "raters": {}}
    # A and practice: the first 24 hours
    sheets: dict = {}
    for mod in ("A", "practice"):
        for x in items[mod]:
            p = load(x["ref"])
            st = features(s, p.ids, p.stay_id, r)
            i = oid(mod, x["ref"])
            sheets[i] = casesheet.sheet(st, 24.0, start=0.0, end=True)
            key["items"][i] = {"module": mod, **x}
    # B1: the history and the next 6 hours, twin (one of 16 rollouts, drawn) or real
    conts = Job(out / "b1_rollouts").rollouts()
    rng = np.random.default_rng(seed + 3)
    b1 = {}
    for j, x in enumerate(items["B1"]):
        p = load(f"split:{x['dataset']}/held_out/{x['stay']}")
        prompt = p.at(v, x["hours"]).prompt
        k = int(rng.integers(len(conts[f"b1_{j}"])))
        twin_ids = list(prompt) + [int(t) for t in conts[f"b1_{j}"][k]]
        real_ids = list(prompt) + continuation(s, p, prompt)
        i = oid("B1", x["anchor"])
        lead = f"{STARTED[x['treatment']]} at the dashed line."
        b1[i] = {v_: casesheet.sheet(features(s, ids, p.stay_id, r), x["hours"], after=6.0,
                                     cut_label=STARTED[x["treatment"]]) for v_, ids in (("twin", twin_ids),
                                                                                         ("real", real_ids))}
        b1[i]["lead"], b1[i]["direction"] = lead, DIRECTION[x["treatment"]]
        key["items"][i] = {"module": "B1", "rollout_shown": k, **x}
    # B2: the sheet, the probe's risk and three factors per method
    facs = {f["anchor"]: f for f in json.loads((out / "b2_factors.json").read_text())}
    b2 = {}
    for x in items["B2"]:
        p = load(f"split:{x['dataset']}/held_out/{x['stay']}")
        lab = prospective_stay(p.stay(v), x["hours"])
        feat = prospective_stay(features(s, p.ids, p.stay_id, r), x["hours"])
        groups = at.stay_groups(lab, x["hours"])
        i = oid("B2", x["anchor"])
        f = facs[x["anchor"]]
        b2[i] = {"html": casesheet.sheet(feat, x["hours"]), "risk": f"{100 / (1 + np.exp(-f['logit'])):.0f}%",
                 "outcome": dict(B2_PROGRAMS)[x["program"]],
                 **{m: [factor(lab, feat, groups, g, x["hours"]) for g in f[m]] for m in METHODS}}
        key["items"][i] = {"module": "B2", **x, "factors": {m: f[m] for m in METHODS}}
    # C: the first 24 hours, the outcomes asked, the card (obfuscated per packet)
    fc = {x["stay"]: x["result"] for x in json.loads((out / "c_forecasts.json").read_text())}
    # the card rule (protocol section 7): ICU death raw; another outcome recalibrated by amendment 11's adopted V2 map
    # where one exists (AKI's passed: AUROC +0.0004, O/E 0.67 -> 0.82, slope 0.79 -> 0.90), as forecast.outputs_of
    # applies it
    from icutwin.toolkit import recalibration as rc

    maps = {p: m for p, m in rc.load(rc.path_of(s)).items() if p != "icu_death@-1"}
    for res in fc.values():
        for prog, x in res["outputs"]["risks"].items():
            raw_p = x.get("e2") if x.get("e2") is not None and np.isfinite(x["e2"]) else x.get("share")
            p_cal, ok = rc.apply(maps, prog, raw_p)
            x["risk"], x["calibrated"] = (p_cal if ok else raw_p), bool(ok)
    key["c_maps"] = {prog: maps[prog].get("source") for _k, prog, _l in C_OUTCOMES if prog in maps}
    c = {}
    for x in items["C"]:
        p = load(f"split:{x['dataset']}/held_out/{x['stay']}")
        i = oid("C", x["anchor"])
        outs = [k for k, _p, _l in C_OUTCOMES if k in x["labels"] and k in C_ASKED]
        c[i] = {"html": casesheet.sheet(features(s, p.ids, p.stay_id, r), 24.0, start=0.0),
                "outcomes": [{"key": k, "label": lbl} for k, _p, lbl in C_OUTCOMES if k in outs],
                "card": c_card(fc[x["stay"]], outs)}
        key["items"][i] = {"module": "C", **x, "twin": {k: (fc[x["stay"]]["outputs"]["risks"].get(prog) or {}).get("risk")
                                                       for k, prog, _l in C_OUTCOMES}}
    # the comparators of C's key secondary analysis (LightGBM; SAPS II for ICU death)
    import pandas as pd

    bl = pd.read_parquet(s.paths.paper / "a1/test_L24/baseline_predictions.parquet")
    bl = bl[bl["model"] == "lgbm_pooled"].set_index(["anchor", "program"])["p"]
    sc = pd.read_parquet(s.paths.paper / "f10/clinical/scores.parquet")
    sc = sc[sc["design"] == "L24"].set_index("anchor")["sapsii_prob"]
    for i, kx in key["items"].items():
        if kx["module"] == "C":
            kx["lgbm"] = {k: float(bl.get((kx["anchor"], prog), np.nan)) for k, prog, _l in C_OUTCOMES}
            kx["sapsii_prob"] = float(sc.get(kx["anchor"], np.nan))
    # the case-cohort's population (select's frame: the landmark's anchors with an ICU-death label), for the weights
    an = pd.read_parquet(s.paths.paper / "a1/test_L24/anchors.parquet")
    pr = pd.read_parquet(s.paths.paper / "a1/test_L24/predictions.parquet")
    death = pr[pr["program"] == "icu_death@-1"].set_index("anchor")["label"]
    an = an[an["dataset"].isin(DS) & an["anchor"].isin(death.index)]
    key["c_population"] = {f"{ds}|{int(y)}": int(n) for (ds, y), n in
                           an.groupby([an["dataset"], an["anchor"].map(death)]).size().items()}
    raters = [f"R{k:02d}" for k in range(1, a.raters + 1)] + [f"P{k:02d}" for k in range(1, a.pilot + 1)]
    (out / "packets").mkdir(exist_ok=True)
    specs, h = [], hashlib.sha256()
    for k, name in enumerate(raters):
        prng = np.random.default_rng(seed * 1000 + k)
        ck = rating.card_key(name, salt)
        mods = []

        def order(ids):
            return [ids[t] for t in prng.permutation(len(ids))]

        prac = [i for i, x in key["items"].items() if x["module"] == "practice"]
        mods.append({"id": "T", "title": "Practice", "kind": "realism", "practice": True, "intro": MODULE_INTRO["T"],
                     "items": [{"id": i, "html": sheets[i]} for i in order(prac)]})
        ids_a = [i for i, x in key["items"].items() if x["module"] == "A"]
        mods.append({"id": "A", "title": "Module A: real or synthetic", "kind": "realism", "intro": MODULE_INTRO["A"],
                     "items": [{"id": i, "html": sheets[i]} for i in order(ids_a)]})
        ids_b2 = sorted(b2)
        meth = {i: METHODS[(k + j) % 3] for j, i in enumerate(ids_b2)}
        mods.append({"id": "B2", "title": "Module B2: the model's explanations", "kind": "factors",
                     "intro": MODULE_INTRO["B2"], "break_after": True,
                     "items": [{"id": i, "html": b2[i]["html"], "risk": b2[i]["risk"], "outcome": b2[i]["outcome"],
                                "factors": b2[i][meth[i]]} for i in order(ids_b2)]})
        ids_b1 = sorted(b1)
        ver = {i: ("twin", "real")[(k + j) % 2] for j, i in enumerate(ids_b1)}
        mods.append({"id": "B1", "title": "Module B1: courses after a treatment", "kind": "course",
                     "intro": MODULE_INTRO["B1"],
                     "items": [{"id": i, "html": b1[i][ver[i]], "lead": b1[i]["lead"], "direction": b1[i]["direction"]}
                               for i in order(ids_b1)]})
        mods.append({"id": "C", "title": "Module C: prognosis with the twin's forecast", "kind": "reader",
                     "intro": MODULE_INTRO["C"],
                     "items": [{"id": i, "html": c[i]["html"], "outcomes": c[i]["outcomes"],
                                "card": rating.obfuscate(c[i]["card"], ck)} for i in order(sorted(c))]})
        mods.append({"id": "Q", "title": "Questionnaire", "kind": "questionnaire", "intro": MODULE_INTRO["Q"],
                     "items": [], "sus": list(rating.SUS), "trust": list(rating.TRUST)})
        specs.append((k, name, ck, mods, ver, meth))
        h.update(json.dumps([name, mods], sort_keys=True).encode())
    # the build code: a hash of every packet's content and the page's code, so any change makes a new build
    h.update((INTRO + rating.JS + rating.CSS).encode())
    build = h.hexdigest()[:12]
    manifest = {}
    for k, name, ck, mods, ver, meth in specs:
        page = rating.build_packet({"title": "ICU case review: a clinician study", "intro": INTRO, "packet": name,
                                    "build": build, "cardkey": ck, "modules": mods})
        f = out / "packets" / f"icu_case_review_{name}.html"
        f.write_text(page, encoding="utf-8", newline="\n")
        manifest[name] = {"file": f.name, "bytes": f.stat().st_size, "sha256": hashlib.sha256(page.encode()).hexdigest()}
        key["raters"][name] = {"index": k, "b1_version": ver, "b2_method": meth,
                               "order": {m["id"]: [it["id"] for it in m["items"]] for m in mods}}
    key["build"] = build
    (out / "packets" / "README_center_leads.txt").write_text(LEADS.format(build=build, n=len(raters)),
                                                              encoding="utf-8", newline="\n")
    (out / "packets" / "pilot_feedback_form.txt").write_text(PILOT.format(build=build), encoding="utf-8", newline="\n")
    (out / "key.json").write_text(json.dumps(key, indent=1, default=float), encoding="utf-8", newline="\n")
    (out / "packets" / "manifest.json").write_text(json.dumps({"build": build, "packets": manifest}, indent=1),
                                                   encoding="utf-8", newline="\n")
    print(json.dumps({"build": build, "packets": len(manifest),
                      "mb_per_packet": round(np.mean([x["bytes"] for x in manifest.values()]) / 1e6, 1)}))


# -- collect --------------------------------------------------------------------------------------------------------
def join(key: dict, exports: dict) -> tuple[list, list]:
    """Exports (``{file name: parsed JSON}``) -> rows joined to the key, and the problems. Per packet, only the most
    complete export counts (final first, then the most answers, then the latest); the others are listed as
    superseded."""
    from icutwin.toolkit import rating

    rows, problems, best = [], [], {}
    for f, o in exports.items():
        if o.get("build") != key["build"] or o.get("packet") not in key["raters"]:
            problems.append({"file": f, "problem": "another build or packet"})
            continue
        if o.get("check") != rating.check(o):
            problems.append({"file": f, "problem": "checksum mismatch"})
            continue
        rank = (bool(o.get("final")), sum(len(x) for x in o["responses"].values()), o.get("exported") or "")
        old = best.get(o["packet"])
        if old is not None:
            loser = old if rank > old[0] else (rank, f)
            problems.append({"file": loser[1], "problem": "superseded by a more complete export"})
            if rank <= old[0]:
                continue
        best[o["packet"]] = (rank, f)
    for _rank, f in sorted(best.values(), key=lambda x: x[1]):
        o = exports[f]
        rk = key["raters"][o["packet"]]
        for mod, recs in o["responses"].items():
            for rec in recs:
                it = key["items"].get(rec["item"], {})
                extra = {}
                if mod == "B1":
                    extra["version"] = rk["b1_version"].get(rec["item"])
                if mod == "B2":
                    extra["method"] = rk["b2_method"].get(rec["item"])
                rows.append({"rater": o["packet"], "module": mod, **rec, **extra,
                             "truth": {k: it.get(k) for k in ("kind", "dataset", "band", "treatment", "program",
                                                              "labels", "twin", "lgbm", "sapsii_prob") if k in it}})
        if o.get("questionnaire"):
            rows.append({"rater": o["packet"], "module": "Q", **o["questionnaire"]})
        rows.append({"rater": o["packet"], "module": "profile", "role": (o.get("consent") or {}).get("role"),
                     "years": (o.get("consent") or {}).get("years"), "final": bool(o.get("final")), "file": f})
    return rows, problems


def collect(s, a) -> None:
    """Exported files -> checked, joined rows (``responses.json``; host only)."""
    out = out_dir(s)
    key = json.loads((out / "key.json").read_text())
    rows, problems = join(key, {f: json.loads(Path(f).read_text(encoding="utf-8")) for f in a.files})
    (out / "responses.json").write_text(json.dumps({"rows": rows, "problems": problems}, indent=1),
                                        encoding="utf-8", newline="\n")
    print(json.dumps({"rows": len(rows), "problems": problems}))


# -- the short pilot set and the distribution archives --------------------------------------------------------------
PILOT5_INTRO = """<p>A short pilot of the ICU case review: five cases from the study's four tasks, about 10-15 minutes.
Each case sheet shows the patient's admission, vital signs, respiratory support and treatments, urine output and
laboratory values, as hours since ICU admission. Your answers are saved in this browser after every item and are not
analysed: they check the layout, the instructions and the export. An answered item cannot be changed.</p>
<p>Use a recent Chrome, Edge or Firefox on a computer.</p>"""

PILOT5_FORM = """ICU case review: short pilot feedback (five questions; build {build})

Answer below and return this file with your exported results to the study team.

1. Time: how long did the five cases take? Was anything slow to load or to respond?

2. Display: did any case sheet, chart or the forecast card fail to display, overlap, or show values that looked
   broken? Note the module and the item.

3. Instructions: was anything in the instructions or the questions unclear, or open to more than one reading?

4. Giveaways: apart from the clinical content, did anything suggest whether the module A sheets or the module B1
   course were real or generated by a model?

5. Anything else, including the forecast card in module C.
"""

RATER = """ICU case review: packet {packet} (build {build})

1. Unzip this file and open icu_case_review_{packet}.html in a recent Chrome, Edge or Firefox on a computer. It
   works offline: nothing is sent anywhere.
2. {sessions}
   Your progress is saved in this browser after every answer: use the same computer and browser, and do not clear
   the browser's data in between.
3. {export}
   Send the downloaded file (clinician_{packet}_final_<date>.json){form} to {to}.
4. If the browser loses your progress, open the packet again and load your last exported file under "Continue in
   another browser".
5. The packet shows renderings of MIMIC-IV and eICU-CRD ICU stays: keep it confidential and do not share it. When the
   study team confirms receipt, delete the packet and the exported files.
"""
FULL = {"sessions": "Two sessions: session 1 is the practice, module A and module B2 (about 90 minutes); session 2 is "
                    "module B1,\n   module C and a short questionnaire (about 100 minutes).",
        "export": "At the end of session 1, click \"Export progress\"; at the end, click \"Export final results\".",
        "to": "your center lead"}
SHORT = {"sessions": "One sitting of about 10-15 minutes: five cases.",
         "export": "At the end, click \"Export final results\".", "to": "the study team"}

LAYOUT = """Contents of this archive:
- raters/R01.zip ... R{last}.zip: one per rater, in number order. Each holds the rater's packet and instructions, so
  one file goes to one rater.
- pilot/P01.zip, P02.zip: the full pilot (two physicians who are not raters), each with pilot_feedback_form.txt.
- manifest.json: each packet's SHA-256, to check that a file arrived intact.
- distribution_log.csv: a tracking sheet (packet, center code, dates). Never write names in it; it can go to the
  study team as the list of packet numbers your center used.

"""

PILOT5_README = """ICU case review: the short pilot set (build {build}; five questions; packets {first}-{last})

- Five questions, as raters meet them: two module A sheets (real or synthetic, from the practice sheets), and one item
  each of module B2 (the model's explanation), module B1 (a course after a treatment) and module C (prognosis with the
  forecast card). No practice round, no questions about the physician's role or experience, no questionnaire and
  no session break; about 10-15 minutes.
- One packet per pilot physician (S01, S02, ...), never someone who will be a rater. Each zip holds the packet, the
  instructions and pilot5_feedback_form.txt.
- The answers are not analysed. They check the layout, the instructions, the export and anything other than content
  that gives the source away. The session timing needs the full pilot packets (P01, P02).
- Send each zip through a secure file channel; keep the packets confidential and delete them when the study team
  confirms receipt. manifest.json holds each packet's SHA-256.
"""


def packet_data(page: str) -> dict:
    """The JSON a packet page carries (``rating.build_packet``)."""
    m = re.search(r'<script type="application/json" id="packet">(.*?)</script>', page, re.S)
    return json.loads(m.group(1))


def pilot(d: Path, a) -> None:
    """The short pilot set (five questions; packets S01-S0n, host only: it reads the key). From the current build's
    pilot packet P01 and the key: two of module A's practice sheets (one real, one synthetic), the first B2 item in
    P01's order with editing's factors, the first B1 item with the twin's course (not insulin), and the first C case
    asking both outcomes, rendered as in the build; no questionnaire, no session break, no role or experience
    questions. Its own build code; the selection is in ``pilot5/key.json``."""
    from icutwin.toolkit import rating

    key = json.loads((d / "key.json").read_text())
    salt = json.loads((d / "items.json").read_text())["salt"]
    P = packet_data((d / "packets" / "icu_case_review_P01.html").read_text(encoding="utf-8"))
    assert P["build"] == key["build"]
    mods = {m["id"]: m for m in P["modules"]}
    rk, ki = key["raters"]["P01"], key["items"]
    prac = mods["T"]["items"]
    a_items = [next(it for it in prac if ki[it["id"]]["kind"] == kind) for kind in ("real", "twin")]
    b2 = next(it for it in mods["B2"]["items"] if rk["b2_method"][it["id"]] == "edit")
    b1 = next(it for it in mods["B1"]["items"] if rk["b1_version"][it["id"]] == "twin"
              and ki[it["id"]]["treatment"] != "insulin")
    c = next(it for it in mods["C"]["items"] if len(it["outcomes"]) == 2)
    card = rating.deobfuscate(c["card"], P["cardkey"])
    names = [f"S{k:02d}" for k in range(1, a.copies + 1)]
    specs, h = [], hashlib.sha256()
    for k, name in enumerate(names):
        rng = np.random.default_rng(key_seed(salt) + k)
        ck = rating.card_key(name, salt)
        mm = [{"id": "A", "title": "Module A: real or synthetic", "kind": "realism", "intro": MODULE_INTRO["A"],
               "items": [a_items[t] for t in rng.permutation(2)]},
              {**{x: mods["B2"][x] for x in ("id", "title", "kind", "intro")}, "items": [b2]},
              {**{x: mods["B1"][x] for x in ("id", "title", "kind", "intro")}, "items": [b1]},
              {**{x: mods["C"][x] for x in ("id", "title", "kind", "intro")},
               "items": [{**c, "card": rating.obfuscate(card, ck)}]}]
        specs.append((name, ck, mm))
        h.update(json.dumps([name, mm], sort_keys=True).encode())
    h.update((PILOT5_INTRO + "profile off" + rating.JS + rating.CSS).encode())
    build = h.hexdigest()[:12]
    o = d / "pilot5"
    o.mkdir(exist_ok=True)
    manifest = {}
    for name, ck, mm in specs:
        page = rating.build_packet({"title": "ICU case review: short pilot", "intro": PILOT5_INTRO, "packet": name,
                                    "build": build, "cardkey": ck, "modules": mm, "profile": False})
        f = o / f"icu_case_review_{name}.html"
        f.write_text(page, encoding="utf-8", newline="\n")
        manifest[name] = {"file": f.name, "bytes": f.stat().st_size, "sha256": hashlib.sha256(page.encode()).hexdigest()}
    sel = {"A": [it["id"] for it in a_items], "B2": b2["id"], "B1": b1["id"], "C": c["id"]}
    (o / "key.json").write_text(json.dumps({"build": build, "from_build": key["build"], "from_packet": "P01",
                                            "items": {i: ki[i] for v_ in sel.values() for i in
                                                      (v_ if isinstance(v_, list) else [v_])}}, indent=1, default=float),
                                encoding="utf-8", newline="\n")
    (o / "manifest.json").write_text(json.dumps({"build": build, "from_build": key["build"], "packets": manifest},
                                                indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"build": build, "packets": len(manifest), "kb_per_packet":
                      round(np.mean([x["bytes"] for x in manifest.values()]) / 1e3)}))


def key_seed(salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|pilot5".encode()).hexdigest()[:8], 16)


def archive(d: Path, a) -> None:
    """The distribution archives (no key needed): ``archives/icu_case_review_<build>_distribution.zip`` (the raters'
    packets in number order, one zip each with the rater's instructions; the full pilot; the leads' README, the
    manifest and a tracking sheet), and, when the short pilot set exists, ``..._pilot5.zip``."""
    import csv
    import io
    import zipfile

    def zipped(files: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for arc, data in files.items():
                z.writestr(arc, data)
        return buf.getvalue()

    (d / "archives").mkdir(exist_ok=True)
    out = []
    man = json.loads((d / "packets" / "manifest.json").read_text())
    build, pk = man["build"], man["packets"]
    raters = sorted(n for n in pk if n.startswith("R"))
    pilots = sorted(n for n in pk if n.startswith("P"))
    log = io.StringIO()
    w = csv.writer(log, lineterminator="\n")
    w.writerow(["sequence", "packet", "zip", "html_sha256", "center_code", "sent_on", "progress_export_received",
                "final_export_received", "deletion_confirmed"])
    files = {"README_center_leads.txt": LAYOUT.format(last=raters[-1][1:]) + LEADS.format(build=build, n=len(pk)),
             "manifest.json": json.dumps(man, indent=1)}
    form = PILOT.format(build=build)
    for seq, name in enumerate(raters + pilots, 1):
        html = (d / "packets" / pk[name]["file"]).read_bytes()
        assert hashlib.sha256(html).hexdigest() == pk[name]["sha256"], name
        is_pilot = name.startswith("P")
        inner = {pk[name]["file"]: html,
                 "instructions.txt": RATER.format(packet=name, build=build, form=" and the filled-in feedback form"
                                                  if is_pilot else "", **FULL)}
        if is_pilot:
            inner["pilot_feedback_form.txt"] = form
        arc = f"{'pilot' if is_pilot else 'raters'}/{name}.zip"
        files[arc] = zipped(inner)
        w.writerow([seq, name, arc, pk[name]["sha256"], "", "", "", "", ""])
    files["distribution_log.csv"] = log.getvalue()
    f = d / "archives" / f"icu_case_review_{build}_distribution.zip"
    f.write_bytes(zipped(files))
    out.append(f)
    if (d / "pilot5" / "manifest.json").exists():
        m5 = json.loads((d / "pilot5" / "manifest.json").read_text())
        names = sorted(m5["packets"])
        form5 = PILOT5_FORM.format(build=m5["build"])
        files = {"README_short_pilot.txt": PILOT5_README.format(build=m5["build"], first=names[0], last=names[-1]),
                 "manifest.json": json.dumps(m5, indent=1)}
        for name in names:
            html = (d / "pilot5" / m5["packets"][name]["file"]).read_bytes()
            assert hashlib.sha256(html).hexdigest() == m5["packets"][name]["sha256"], name
            files[f"{name}.zip"] = zipped({m5["packets"][name]["file"]: html, "pilot5_feedback_form.txt": form5,
                                           "instructions.txt": RATER.format(packet=name, build=m5["build"],
                                                                            form=" and the filled-in feedback form",
                                                                            **SHORT)})
        f = d / "archives" / f"icu_case_review_{m5['build']}_pilot5.zip"
        f.write_bytes(zipped(files))
        out.append(f)
    print(json.dumps({x.name: {"mb": round(x.stat().st_size / 1e6, 1),
                               "sha256": hashlib.sha256(x.read_bytes()).hexdigest()} for x in out}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("select", "simulate", "packets", "collect", "pilot", "archive"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--a-real", type=int, default=40)
    ap.add_argument("--b1-per", type=int, default=4)
    ap.add_argument("--b2-per", type=int, default=10)
    ap.add_argument("--c-cases", type=int, default=48)
    ap.add_argument("--c-deaths", type=int, default=12)
    ap.add_argument("--c-rollouts", type=int, default=128)
    ap.add_argument("--b1-rollouts", type=int, default=16)
    ap.add_argument("--raters", type=int, default=20)
    ap.add_argument("--pilot", type=int, default=2)
    ap.add_argument("--copies", type=int, default=5, help="pilot: packets in the short pilot set (S01, S02, ...)")
    ap.add_argument("--dir", type=Path, default=Path("runs/toolkit/clinician"), help="pilot, archive: the study's directory")
    a = ap.parse_args(argv)
    if a.step in ("pilot", "archive"):                           # no session: the build's files only
        {"pilot": pilot, "archive": archive}[a.step](a.dir, a)
        return
    s = session()
    {"select": select, "simulate": simulate, "packets": packets, "collect": collect}[a.step](s, a)


if __name__ == "__main__":
    main()
