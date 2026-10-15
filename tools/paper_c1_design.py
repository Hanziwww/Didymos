"""Paper P amendment 7, F5's designs (``docs/p1_paper_protocol.md``): the what-if arms that validate C1.

* (a) the A/A arms: C1's prefixes (``runs/paper/c1/anchors.parquet``) with two further withhold arms (seeds 7 and 8),
  whose difference is C1's read-outs under the null;
* (b) the factual design. Per treatment (``tools/server/paper_whatif.py``'s definitions), up to 200 held-out
  *starters* (the first record of the treatment is a start at least 2.5 h after the ICU admission), cut 1 minute
  before that start; each matched to a held-out *non-starter* of the same dataset at the same ICU hour (or within
  3 h, then 6 h): in the ICU then, no record of the treatment before, no start in the next 6 h. Matching is on the
  propensity of starting within 6 h (LightGBM on A1's features at the hour, trained on the real train store's
  eligible ICU hours, early stopping on the tuning store's), nearest neighbour without replacement within 0.2 SD of
  the logit. Arms: starters *give* (their own start token) and *withhold*; non-starters *give* (C1's given record:
  the dataset's modal start at bin 4) and *withhold*.

Writes ``<out>/design.parquet`` (one row per anchor and arm) and ``<out>/design.json`` (counts, balance).

    python tools/paper_c1_design.py --tokens data/tokens/full_r4 --c1 runs/paper/c1 --out runs/paper/f5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402
from paper_whatif import BINS, TREATMENTS  # noqa: E402

HORIZON_H = 6.0


def first_record(s, fam_codes: set, start_c) -> tuple | None:
    """(hours after the ICU admission, code, is a start) of a decoded stay's first record of the treatment."""
    for t, _rk, code, _n, _x in s.timeline:
        if code in fam_codes or start_c.match(code):
            return (t - s.icu_in).total_seconds() / 3600, code, bool(start_c.match(code))
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", default="data/factory/stores/real_test")
    ap.add_argument("--train", default="data/factory/stores/real_train")
    ap.add_argument("--val", default="data/factory/stores/real_dev")
    ap.add_argument("--per-treatment", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-rows-per-stay", type=int, default=4)
    a = ap.parse_args(argv)
    import pandas as pd

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst
    from icutwin.factory import twins
    from icutwin.generate.engine import treatment_tokens
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = Vocab.load(tokens / "vocab.json")
    code_of = [m.get("code") or "" for m in v.meta]
    rng = np.random.default_rng(a.seed)
    rep: dict = {"git": _sha(), "treatments": {}}
    rows_out = []

    # (a) the A/A arms on C1's prefixes
    c1 = pd.read_parquet(Path(a.c1) / "anchors.parquet")
    for x in c1.itertuples():
        for arm, seed in (("withhold_a", 7), ("withhold_b", 8)):
            rows_out.append({"anchor": f"aa:{x.anchor}", "treatment": x.treatment, "group": "aa", "twin": str(x.twin),
                             "dataset": x.dataset, "hours": float(x.hours), "pair": -1, "arm": arm, "give_token": -1,
                             "seed": seed})
    rep["aa_prefixes"] = int(len(c1))

    # (b) the factual design
    stores = {k: fst.Store(Path(p)) for k, p in (("test", a.test), ("train", a.train), ("val", a.val))}
    split_of = {"test": "held_out", "train": "train", "val": "tuning"}
    decoded = {}
    for k, st in stores.items():
        keep = set(st.stays["stay_id"].astype(str))
        decoded[k] = {r["stay_id"]: (r["dataset"], r["ids"], twins.decode_labels(r["ids"], v, r["stay_id"]))
                      for r in bench.real_rows(tokens, split_of[k], keep)}
        print(json.dumps({"decoded": k, "stays": len(decoded[k])}), flush=True)
    sidx = {k: {s: i for i, s in enumerate(st.stays["stay_id"].astype(str))} for k, st in stores.items()}
    fz = bl.Featurizer(bl.feature_channels(stores["train"], 64)).fit_levels(stores["train"])
    evs = {k: fpr.Evaluator(st) for k, st in stores.items()}

    def feats(k, items):                              # items: (stay id, hours)
        r = fl.Rows(np.array([sidx[k][s] for s, _h in items], dtype=np.int64),
                    np.array([int(h * 60) for _s, h in items], dtype=np.int64))
        return fz(stores[k], r, evs[k])

    for name, (start_re, fam, _dose) in TREATMENTS.items():
        start_c = re.compile(start_re)
        fam_ids = treatment_tokens(v, fam) if fam in ("rrt", "imv") else \
            frozenset(i for i, c in enumerate(code_of) if re.match(fam, c))
        fam_codes = {code_of[i] for i in fam_ids}

        # per stay once: (first record's hours or inf, a start, ICU hours)
        frs = {k: {} for k in decoded}
        for k, dk in decoded.items():
            for sid, (_ds, _ids, s) in dk.items():
                fr = first_record(s, fam_codes, start_c)
                frs[k][sid] = (fr[0] if fr else np.inf, bool(fr and fr[2]), (s.icu_out - s.icu_in).total_seconds() / 3600)

        def status(k, sid, h0):
            """'start' (first record a start in (h0, h0 + 6]), 'free' (no record up to h0 + 6), else None."""
            f0, is_start, los = frs[k][sid]
            if los <= h0:
                return None
            if f0 > h0 + HORIZON_H:
                return "free"
            if f0 > h0 and is_start:
                return "start"
            return None

        # the propensity model: eligible ICU hours of train stays, labelled by a start within 6 h
        def hour_rows(k, per):
            items, ys = [], []
            for sid in decoded[k]:
                los = frs[k][sid][2]
                if los < 3:
                    continue
                for h0 in rng.uniform(2.0, los, size=per):
                    stt = status(k, sid, float(h0))
                    if stt is not None:
                        items.append((sid, float(h0)))
                        ys.append(1.0 if stt == "start" else 0.0)
            return items, np.array(ys)

        it_tr, y_tr = hour_rows("train", a.train_rows_per_stay)
        it_va, y_va = hour_rows("val", 2)
        if y_tr.sum() < 20:
            rep["treatments"][name] = {"skipped": "too few train starts"}
            continue
        model = bl.fit_lgbm(feats("train", it_tr), y_tr, feats("val", it_va), y_va)

        # starters and their own start tokens; C1's given record per dataset (the modal start at bin 4)
        elig, modal = [], {}
        for sid, (ds, ids, s) in decoded["test"].items():
            fr = first_record(s, fam_codes, start_c) if frs["test"][sid][1] else None
            if fr is not None and fr[2] and fr[0] >= 2.5:
                tok = next((int(t) for t in ids if code_of[int(t)] in fam_codes or start_c.match(code_of[int(t)])),
                           None)
                elig.append((sid, ds, fr[0], tok))
                modal.setdefault(ds, []).append(fr[1])
        give_mod = {}
        for ds, codes in modal.items():
            code = max(set(codes), key=codes.count)
            cand = [i for i, c in enumerate(code_of) if c == code]
            with_bin = {v.meta[i].get("bin"): i for i in cand if v.meta[i].get("kind") == "bin"}
            plain = [i for i in cand if v.meta[i].get("kind") == "na"] or cand
            give_mod[ds] = int(with_bin.get(BINS["median"], next(iter(with_bin.values())))) if with_bin else int(plain[0])
        pick = [elig[k] for k in rng.permutation(len(elig))[:a.per_treatment]]
        # candidates: test stays without the treatment by then and no start within 6 h, at the starter's hour
        used, pairs = set(), []
        st_items = [(sid, h - 1 / 60) for sid, _ds, h, _t in pick]
        lp_st = model.predict_proba(feats("test", st_items))[:, 1]
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))  # noqa: E731
        lg_st = logit(lp_st)
        cal_sd = float(np.std(lg_st)) if len(lg_st) > 1 else 1.0
        by_ds: dict = {}
        for sid2, (ds2, _ids2, _s2) in decoded["test"].items():
            by_ds.setdefault(ds2, []).append(sid2)
        for j, (sid, ds, h, tok) in enumerate(pick):
            match = None
            for width in (0.0, 3.0, 6.0):
                cands = []
                for sid2 in by_ds.get(ds, []):
                    if sid2 == sid or sid2 in used:
                        continue
                    hh = h - 1 / 60
                    if width:                          # the nearest hour within the band where the stay qualifies
                        hh = float(np.clip(hh, 2.0, frs["test"][sid2][2] - 1))
                        if abs(hh - (h - 1 / 60)) > width:
                            continue
                    if status("test", sid2, hh) == "free":
                        cands.append((sid2, hh))
                if len(cands) > 300:                   # a random 300 are scored (the features cost)
                    cands = [cands[i] for i in np.sort(rng.choice(len(cands), 300, replace=False))]
                if cands:
                    lg_c = logit(model.predict_proba(feats("test", cands))[:, 1])
                    k_ = int(np.argmin(np.abs(lg_c - lg_st[j])))
                    if abs(lg_c[k_] - lg_st[j]) <= 0.2 * cal_sd:
                        match = (cands[k_][0], cands[k_][1], float(lg_c[k_]))
                        break
            if match is None:
                continue
            used.add(match[0])
            pairs.append((sid, ds, h - 1 / 60, tok, match[0], match[1], float(lg_st[j]), match[2]))
        for p_id, (sid, ds, hs, tok, sid2, hn, _ls, _ln) in enumerate(pairs):
            for grp, s_, h_, give in (("starter", sid, hs, tok), ("nonstarter", sid2, hn, give_mod.get(ds, -1))):
                anchor = f"{name}:{grp}:{s_}@{h_:.3f}"
                for arm, gt in (("give", give), ("withhold", -1)):
                    if arm == "give" and (gt is None or gt < 0):
                        continue
                    rows_out.append({"anchor": anchor, "treatment": name, "group": grp, "twin": s_, "dataset": ds,
                                     "hours": float(h_), "pair": p_id, "arm": arm, "give_token": int(gt), "seed": 7})
        ls = np.array([p[6] for p in pairs])
        ln = np.array([p[7] for p in pairs])
        rep["treatments"][name] = {"eligible_starters": len(elig), "picked": len(pick), "matched": len(pairs),
                                   "train_rows": int(len(y_tr)), "train_starts": int(y_tr.sum()),
                                   "logit_starters_mean": float(ls.mean()) if len(ls) else None,
                                   "logit_nonstarters_mean": float(ln.mean()) if len(ln) else None,
                                   "logit_sd": cal_sd}
        print(json.dumps({name: rep["treatments"][name]}), flush=True)
    pd.DataFrame(rows_out).to_parquet(out / "design.parquet", index=False)
    (out / "design.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
