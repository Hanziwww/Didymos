"""The authenticity audit (round 2, ``docs/m7_authenticity.md``): what tells twins from real stays, and the
pre-registered targets A1-A7.

Views (token sequences derived from a stay; ``views``):
* ``first_hours``: prefix, header and every timeline token before ``hours`` after the ICU admission marker
  (``eval.utility.landmark``); a stay that ends or leaves the ICU earlier keeps its whole timeline; no end block;
* family views: only one family's tokens, in order (vitals, labs, medications, respiratory, timing = intervals);
* ``order``: every token mapped to its family (a coarse sequence: the order of event kinds, not their values);
* ``whole``: every token; ``length_matched`` subsamples both sets to the same counts per decile of the real
  events per stay.
Each view's discriminator AUC is ``eval.fidelity.discriminator_auc``.

Drift and stopping (aggregates): repetition of an event within its last ``k`` events, by position and by hours since
admission; events per hour; the ICU discharge hazard per day; within-stay lag-1 coherence of value bins of the most
charted numeric codes; length tails (events per stay) and the most-repeated event per stay.

What separates the sets (``explain``, a diagnosis without a target): the discriminator on unigrams alone, the tokens
and event-kind bigrams whose share of a stay differs most, tokens per kind and events per time group.

Model checks (GPU; ``model_checks``): per-position NLL of real held-out stays and of the twins under the model; the
distortion monitor (probability mass the model puts outside the generation mask, grammar and support, on real
held-out sequences, by automaton state); NLL of tuning windows at temperatures around 1.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from icutwin.eval import fidelity as fd
from icutwin.eval import utility as ut
from icutwin.eval.forecast import Reader

FAMILIES = {"vitals": ("VITAL//",), "labs": ("LAB//",), "medications": ("MED_CON//", "MED_INT//", "MED_ORD//"),
            "respiratory": ("RESP//",)}
POS_EDGES = (0, 100, 300, 1000, 3000, 10_000, 10**9)          # event index buckets
HOUR_EDGES = (0, 24, 72, 168, 336, 10**9)                       # hours since the ICU admission marker


# -- views ----------------------------------------------------------------------------------------------------------
class Views:
    def __init__(self, v):
        self.v = v
        self.r = Reader(v)
        self.end = v.id("[END]")
        n = len(v.tokens)
        self.family_of = np.zeros(n, dtype=np.int64)
        self.fam_names = ["other", "special", "prefix", "header", "interval", "end", "chunk"]
        code = [(m.get("code") or "") for m in v.meta]
        for i, m in enumerate(v.meta):
            c = m.get("cls")
            if c in ("special", "prefix", "header", "interval", "end", "chunk"):
                self.family_of[i] = self.fam_names.index(c)
            elif c == "event":
                dom = code[i].split("//")[0]
                if dom not in self.fam_names:
                    self.fam_names.append(dom)
                self.family_of[i] = self.fam_names.index(dom)
        self.masks = {name: np.array([any(code[i].startswith(p) for p in pre) and v.meta[i].get("cls") == "event"
                                      for i in range(n)]) for name, pre in FAMILIES.items()}
        self.masks["timing"] = np.array([m.get("cls") == "interval" for m in v.meta])

    def first_hours(self, ids, hours: float) -> np.ndarray:
        ids = np.asarray(ids)
        cut = ut.landmark(ids.tolist(), self.r, hours)
        if cut is None:
            at = np.flatnonzero(ids == self.end)
            cut = int(at[0]) if len(at) else len(ids)
        return ids[:cut]

    def family(self, ids, name: str) -> np.ndarray:
        ids = np.asarray(ids)
        keep = ids[self.masks[name][ids]]
        return keep if len(keep) else np.zeros(1, dtype=ids.dtype)     # an empty view: one padding token

    def order(self, ids) -> np.ndarray:
        return self.family_of[np.asarray(ids)]


def length_matched(twins: list, real: list, voc: fd.Vocabulary, rng: np.random.Generator) -> tuple[list, list]:
    """Both sets subsampled without replacement to the same counts per decile of the real set's events per stay
    (amendment 1 of ``docs/m7_authenticity.md``: resampling with replacement duplicated stays, which then sat in
    both cross-validation folds and inflated the AUC; real against real 0.79). Returns (twins, real)."""
    ev_r = np.array([fd.features(s, voc).events for s in real])
    ev_t = np.array([fd.features(s, voc).events for s in twins])
    edges = np.unique(np.quantile(ev_r, np.linspace(0, 1, 11)))
    bins_r = np.clip(np.searchsorted(edges, ev_r, side="right") - 1, 0, len(edges) - 2)
    bins_t = np.clip(np.searchsorted(edges, ev_t, side="right") - 1, 0, len(edges) - 2)
    out_t, out_r = [], []
    for b in range(len(edges) - 1):
        pool_r, pool_t = np.flatnonzero(bins_r == b), np.flatnonzero(bins_t == b)
        n = min(len(pool_r), len(pool_t))
        out_t += [twins[i] for i in rng.choice(pool_t, size=n, replace=False)]
        out_r += [real[i] for i in rng.choice(pool_r, size=n, replace=False)]
    return out_t, out_r


def view_aucs(twins: list, real: list, v, seed: int = 0) -> dict:
    """The discriminator AUC of every view (module docstring)."""
    vw = Views(v)
    voc = fd.vocabulary(v)
    n = len(v.tokens)
    rng = np.random.default_rng(seed)
    out = {}
    for h in (6, 24, 72):
        out[f"first_{h}h"] = fd.discriminator_auc([vw.first_hours(s, h) for s in real],
                                                  [vw.first_hours(s, h) for s in twins], n, seed)
    out["whole"] = fd.discriminator_auc(real, twins, n, seed)
    lm_t, lm_r = length_matched(twins, real, voc, rng)
    out["length_matched"] = fd.discriminator_auc(lm_r, lm_t, n, seed)
    out["length_matched_stays"] = len(lm_t)
    for name in (*FAMILIES, "timing"):
        out[name] = fd.discriminator_auc([vw.family(s, name) for s in real], [vw.family(s, name) for s in twins], n,
                                         seed)
    k = len(vw.fam_names)
    out["order"] = fd.discriminator_auc([vw.order(s) for s in real], [vw.order(s) for s in twins], k, seed)
    return out


# -- drift, stopping, coherence -----------------------------------------------------------------------------------
def _hours_per_token(ids, r: Reader) -> np.ndarray:
    """Hours since the ICU admission marker at each token (NaN before it)."""
    t, start = 0.0, None
    out = np.full(len(ids), np.nan)
    for k, tok in enumerate(ids):
        tok = int(tok)
        if start is None and tok in r.adm:
            start = t
        if start is not None:
            out[k] = (t - start) / 60
        t += r.minutes.get(tok, 0.0)
    return out


def drift(seqs: list, v, k: int = 20) -> dict:
    """Repetition (an event token equal to one of the stay's previous ``k`` events) by event position and by hours
    since admission; events per hour by hours since admission."""
    r = Reader(v)
    event = np.array([m.get("cls") == "event" for m in v.meta])
    rep_pos = np.zeros((len(POS_EDGES) - 1, 2))
    rep_hour = np.zeros((len(HOUR_EDGES) - 1, 2))
    events_hour = np.zeros(len(HOUR_EDGES) - 1)
    exposure_hour = np.zeros(len(HOUR_EDGES) - 1)
    for s in seqs:
        s = np.asarray(s)
        hrs = _hours_per_token(s, r)
        evi = np.flatnonzero(event[s])
        last = []
        for j, p in enumerate(evi):
            tok = int(s[p])
            rep = tok in last
            last.append(tok)
            if len(last) > k:
                last.pop(0)
            bp = np.searchsorted(POS_EDGES, j, side="right") - 1
            rep_pos[bp] += (rep, 1)
            if not np.isnan(hrs[p]):
                bh = np.searchsorted(HOUR_EDGES, hrs[p], side="right") - 1
                rep_hour[bh] += (rep, 1)
                events_hour[bh] += 1
        span = np.nanmax(hrs) if np.isfinite(hrs).any() else 0.0
        for b in range(len(HOUR_EDGES) - 1):
            exposure_hour[b] += max(0.0, min(span, HOUR_EDGES[b + 1]) - HOUR_EDGES[b])
    ratio = lambda a: [float(x / n) if n else None for x, n in a]  # noqa: E731
    return {"repeat_share_by_event_position": dict(zip(map(str, POS_EDGES[:-1]), ratio(rep_pos))),
            "repeat_share_by_hours": dict(zip(map(str, HOUR_EDGES[:-1]), ratio(rep_hour))),
            "events_per_hour_by_hours": dict(zip(map(str, HOUR_EDGES[:-1]),
                                                 [float(e / x) if x else None for e, x in zip(events_hour, exposure_hour)]))}


def los_hours(seqs: list, v) -> np.ndarray:
    """ICU length of stay (hours, interval medians) of each sequence; NaN without a discharge."""
    r = Reader(v)
    return np.array([np.nan if o["icu_los_hours"] is None else o["icu_los_hours"] for o in map(r.outcomes, seqs)])


def discharge_hazard(los: np.ndarray, days: int = 21) -> list:
    """Per day d (1..days, then all later days together): discharges during the day over stays in the ICU at its
    start."""
    los = los[~np.isnan(los)] / 24
    out = []
    for d in range(days):
        at_risk = np.sum(los >= d)
        out.append(float(np.sum((los >= d) & (los < d + 1)) / at_risk) if at_risk else None)
    late = np.sum(los >= days)
    out.append(float(1 / max(np.mean(los[los >= days] - days), 1e-9)) if late else None)    # per day, beyond
    return out


def coherence(seqs: list, v, codes: list) -> dict:
    """Within-stay lag-1 coherence of value bins: per code, the share of consecutive values (same stay) in the same
    bin and their mean absolute bin difference."""
    voc = fd.vocabulary(v)
    out = {}
    for code in codes:
        same = diff = n = 0
        for s in seqs:
            s = np.asarray(s)
            b = voc.bin[s][[voc.code[t] == code for t in s]]
            b = b[b >= 0]
            if len(b) > 1:
                d = np.abs(np.diff(b))
                same += int((d == 0).sum())
                diff += float(d.sum())
                n += len(d)
        out[code] = {"pairs": n, "same_bin": same / n if n else None, "mean_abs_bin_change": diff / n if n else None}
    return out


def length_and_repeats(seqs: list, v) -> dict:
    voc = fd.vocabulary(v)
    ev = np.array([fd.features(s, voc).events for s in seqs])
    top = []
    for s in seqs:
        s = np.asarray(s)
        e = s[voc.event[s]]
        top.append(Counter(e.tolist()).most_common(1)[0][1] if len(e) else 0)
    q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
    return {"events_p50": q(ev, 50), "events_p90": q(ev, 90), "events_p99": q(ev, 99),
            "top_event_p50": q(top, 50), "top_event_p99": q(top, 99)}


def audit(twins: list, real: list, v, seed: int = 0, coherence_codes: list | None = None) -> dict:
    """Every model-free measure of one dataset (or pooled): views, drift, stopping, coherence, tails."""
    voc = fd.vocabulary(v)
    if coherence_codes is None:
        cnt = Counter(voc.code[t] for s in real for t in np.asarray(s) if voc.bin[t] >= 0)
        coherence_codes = [c for c, _n in cnt.most_common(8)]
    lr, lt = los_hours(real, v), los_hours(twins, v)
    return {"views": view_aucs(twins, real, v, seed),
            "drift": {"real": drift(real, v), "twins": drift(twins, v)},
            "hazard": {"real": discharge_hazard(lr), "twins": discharge_hazard(lt)},
            "coherence": {"real": coherence(real, v, coherence_codes), "twins": coherence(twins, v, coherence_codes)},
            "tails": {"real": length_and_repeats(real, v), "twins": length_and_repeats(twins, v)}}


# -- what separates the sets (aggregates; a diagnosis, not a target) ----------------------------------------------
def _welch(a, b) -> np.ndarray:
    """Welch's t of the column means, ``a`` against ``b`` (rows: stays; dense or sparse)."""
    def mv(x):
        n = x.shape[0]
        m = np.asarray(x.mean(0)).ravel()
        sq = x.multiply(x) if hasattr(x, "multiply") else x * x
        return m, np.maximum((np.asarray(sq.mean(0)).ravel() - m ** 2) * n / max(n - 1, 1), 0.0), n

    (ma, va, na), (mb, vb, nb) = mv(a), mv(b)
    se = np.sqrt(va / na + vb / nb)
    return np.divide(ma - mb, se, out=np.zeros_like(ma), where=se > 0)


def _counts(seqs: list, n: int):
    from scipy import sparse

    rows, cols, vals = [], [], []
    for r, s in enumerate(seqs):
        u, c = np.unique(np.asarray(s, dtype=np.int64), return_counts=True)
        rows.append(np.full(len(u), r))
        cols.append(u)
        vals.append(c)
    return sparse.csr_matrix((np.concatenate(vals).astype(np.float64), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(seqs), n))


def explain(twins: list, real: list, v, hours: float | None = 24, k: int = 15, min_stays: int = 20,
            seed: int = 0) -> dict:
    """What separates the twins from the real stays in a view (the first ``hours``; the whole stay with None), in
    aggregates: the discriminator on token unigrams alone (content without order; against the view's AUC, what
    order adds); the ``k`` tokens whose share of a stay's tokens differs most (Welch's t over stays, among tokens in
    at least ``min_stays`` stays of either set), with tokens per stay and the share of stays holding them; tokens per
    stay of each kind (``Views.fam_names``); the ``k`` most different kind bigrams (share of a stay's bigrams: the
    order of event kinds); events per time group (between interval tokens). Twins first in every pair."""
    from scipy import sparse

    vw, voc, n = Views(v), fd.vocabulary(v), len(v.tokens)
    cut = (lambda s: vw.first_hours(s, hours)) if hours is not None else np.asarray
    ts, rs = [cut(s) for s in twins], [cut(s) for s in real]
    out = {"stays": [len(ts), len(rs)], "unigram_auc": fd.discriminator_auc(rs, ts, n, seed, bigrams=False)}

    ct, cr = _counts(ts, n), _counts(rs, n)
    share = lambda x: sparse.diags(1 / np.maximum(np.asarray(x.sum(1)).ravel(), 1)) @ x  # noqa: E731
    t = _welch(share(ct), share(cr))
    held_t, held_r = np.asarray((ct > 0).sum(0)).ravel(), np.asarray((cr > 0).sum(0)).ravel()
    ok = np.flatnonzero((held_t >= min_stays) | (held_r >= min_stays))
    per_t, per_r = np.asarray(ct.mean(0)).ravel(), np.asarray(cr.mean(0)).ravel()
    r3 = lambda x: round(float(x), 3)  # noqa: E731
    out["tokens"] = [{"token": v.tokens[i], "t": round(float(t[i]), 1), "per_stay": [r3(per_t[i]), r3(per_r[i])],
                      "stays": [r3(held_t[i] / len(ts)), r3(held_r[i] / len(rs))]}
                     for i in ok[np.argsort(-np.abs(t[ok]))][:k]]
    out["max_abs_t"] = {"tokens": round(float(np.abs(t[ok]).max()), 1) if len(ok) else None}

    nk = len(vw.fam_names)
    fam_t = np.array([np.bincount(vw.family_of[s], minlength=nk) for s in ts], dtype=np.float64)
    fam_r = np.array([np.bincount(vw.family_of[s], minlength=nk) for s in rs], dtype=np.float64)
    out["kinds"] = {name: [r3(fam_t[:, j].mean()), r3(fam_r[:, j].mean())] for j, name in enumerate(vw.fam_names)
                    if fam_t[:, j].any() or fam_r[:, j].any()}

    def bigram_shares(seqs):
        rows = []
        for s in seqs:
            f = vw.family_of[np.asarray(s)]
            rows.append(np.bincount(f[:-1] * nk + f[1:], minlength=nk * nk) / max(len(f) - 1, 1))
        return np.array(rows)

    bt, br = bigram_shares(ts), bigram_shares(rs)
    tb = _welch(bt, br)
    okb = np.flatnonzero(((bt > 0).sum(0) >= min_stays) | ((br > 0).sum(0) >= min_stays))
    out["kind_bigrams"] = [{"bigram": f"{vw.fam_names[j // nk]}>{vw.fam_names[j % nk]}", "t": round(float(tb[j]), 1),
                            "share": [r3(bt[:, j].mean()), r3(br[:, j].mean())]}
                           for j in okb[np.argsort(-np.abs(tb[okb]))][:k]]
    out["max_abs_t"]["kind_bigrams"] = round(float(np.abs(tb[okb]).max()), 1) if len(okb) else None

    def groups(seqs):
        sizes = []
        for s in seqs:
            s = np.asarray(s)
            g = np.bincount(np.cumsum(vw.masks["timing"][s])[voc.event[s]])
            sizes.append(g[g > 0])
        g = np.concatenate([*sizes, np.zeros(0, dtype=np.int64)])
        g = g if len(g) else np.zeros(1)
        return {"groups": int(len(g)), "mean": r3(g.mean()), "p50": r3(np.percentile(g, 50)),
                "p90": r3(np.percentile(g, 90)), "single": r3(np.mean(g == 1))}

    out["group_events"] = {"twins": groups(ts), "real": groups(rs)}
    return out


def targets(per_dataset: dict, pooled_quality: dict | None = None) -> dict:
    """The pre-registered targets A1-A6 (A7's parts from the pooled harness and utility when given)."""
    out = {}
    for ds, a in per_dataset.items():
        vw, tr, tt = a["views"], a["tails"]["real"], a["tails"]["twins"]
        ratio = lambda k: tt[k] / tr[k] if tr[k] else None  # noqa: E731
        out[ds] = {
            "A1": vw["first_24h"] <= 0.70,
            "A2": vw["length_matched"] <= 0.80,
            "A3": all(vw[f] <= 0.80 for f in (*FAMILIES, "timing")),
            "A4": all(r is not None and 0.80 <= r <= 1.20 for r in (ratio("events_p90"), ratio("events_p99"))),
            "A5": ratio("top_event_p99") is not None and 0.75 <= ratio("top_event_p99") <= 1.25}
    if pooled_quality is not None:
        pos = pooled_quality["pooled"]["positive_control"]["diversity"]
        cand = pooled_quality["pooled"]["candidates"]["model"]
        d = cand["metrics"]["diversity"]
        replaced = {"discriminator_auc", "length_p99_logratio", "vendi_ratio", "vendi_ratio_by_dataset_min"}
        util = pooled_quality.get("utility", {}).get("sets", {}).get("model", {})
        out["pooled"] = {
            "A6": d["vendi_ratio"] >= 0.95 and d["rare_code_recall"] >= pos["rare_code_recall"] - 0.03,
            "A7_harness": not (set(cand["flags"]) - replaced),
            "A7_utility": all(x.get("utility_ratio") is not None and x["utility_ratio"] >= 0.90 for x in util.values()),
            "A7_grammar": all(r.get("grammar_valid") == 1.0 for r in pooled_quality.get("datasets", {}).values())}
    return out


# -- model checks (GPU) ---------------------------------------------------------------------------------------------
def _autocast(device):
    import torch

    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def token_logprobs(model, ids, device, temperatures=(1.0,)):
    """Per target position (1..len-1): log-probability of the next token at each temperature, and the logits'
    row for the distortion monitor is not kept (memory: one sequence at a time)."""
    import torch

    with torch.no_grad():
        x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)[None]
        with _autocast(device):         # as training and the privacy gate: memory-efficient attention at 32k
            logits = model(input_ids=x).logits[0, :-1].float()
        tgt = x[0, 1:]
        out = {}
        for temp in temperatures:
            lp = torch.log_softmax(logits / temp, -1).gather(1, tgt[:, None])[:, 0]
            out[temp] = lp.cpu().numpy()
    return out


def masked_mass(model, ids, v, t, m, support, device, positions: int = 256, seed: int = 0) -> list:
    """Distortion monitor: at up to ``positions`` sampled positions of a real sequence, the model's probability mass
    outside the generation mask (grammar alone, grammar and the dataset's support) given the automaton state;
    (phase, last, in-group, mass_grammar, mass_support) per position."""
    import torch

    from icutwin.generate.constrain import NONE, generation_mask
    from icutwin.tokenization.fsm import Automaton

    ids = [int(x) for x in ids]
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(np.arange(1, len(ids)), size=min(positions, len(ids) - 1), replace=False).tolist())
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)[None]
        with _autocast(device):
            logits = model(input_ids=x).logits[0]
        probs = torch.softmax(logits.float(), -1)
    a = Automaton(t)
    dataset_token, out = None, []
    for pos, tok in enumerate(ids):
        if pos in pick and a.phase not in ("start", "done") and a.violation is None:
            p = probs[pos - 1, :m.n].cpu().numpy()
            g = generation_mask(a, m, None, NONE, None, None)
            sup = support.mask(dataset_token) if support is not None else None
            gs = generation_mask(a, m, None, NONE, None, sup) if sup is not None else g
            out.append((a.phase, a.last, bool(a.phase == "tl" and a.group_first >= 0),
                        float(p[~g].sum()), float(p[~gs].sum())))
        if a.phase == "prefix" and a.i == 0:
            dataset_token = tok
        if not a.step(tok):
            break
    return out


def model_checks(model, real: list, twins: list, tuning: list, v, t, m, support, device,
                 temps=(0.9, 0.95, 1.0, 1.05, 1.1)) -> dict:
    """Per-position NLL (real held-out against the twins), the distortion monitor and the temperature scan."""
    def by_position(seqs):
        tot, cnt = np.zeros(len(POS_EDGES) - 1), np.zeros(len(POS_EDGES) - 1)
        for s in seqs:
            lp = token_logprobs(model, s, device)[1.0]
            pos = np.arange(1, len(s))
            b = np.searchsorted(POS_EDGES, pos, side="right") - 1
            np.add.at(tot, b, -lp)
            np.add.at(cnt, b, 1)
        return {str(POS_EDGES[i]): (float(tot[i] / cnt[i]) if cnt[i] else None) for i in range(len(cnt))}

    states: dict = {}
    for s in real:
        for phase, last, ingroup, mg, ms in masked_mass(model, s, v, t, m, support, device):
            key = f"{phase}:{last}:{'group' if ingroup else 'open'}" if phase == "tl" else phase
            st = states.setdefault(key, [0, 0.0, 0.0, 0])
            st[0] += 1
            st[1] += mg
            st[2] += ms
            st[3] += int(mg > 0.01)
    scan = {temp: [0.0, 0] for temp in temps}
    for s in tuning:
        lps = token_logprobs(model, s, device, temps)
        for temp in temps:
            scan[temp][0] += float(-lps[temp].sum())
            scan[temp][1] += len(lps[temp])
    return {"nll_by_position": {"real": by_position(real), "twins": by_position(twins)},
            "masked_mass": {k: {"positions": n, "grammar": g / n, "grammar_support": s_ / n, "share_over_1pct": c / n}
                            for k, (n, g, s_, c) in sorted(states.items())},
            "temperature_nll": {str(temp): tot / n for temp, (tot, n) in scan.items()}}


# -- runner ---------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """``audit`` (CPU): per dataset and pooled, the twins against held-out stays and the positive control (real
    training stays) against them, on the sets the pilot draws (``eval.quality.draws``); ``explain`` (CPU): per
    dataset, what separates them in the first 24 h and the whole stay (``explain``; the positive control's unigram
    AUC and largest t beside it); ``model`` (GPU): the model checks on up to ``--n`` stays per dataset; ``targets``:
    A1-A7 from the audit and the pooled pilot report."""
    import argparse
    import json
    from pathlib import Path

    from icutwin.eval.quality import DATASETS, draws
    from icutwin.tokenization.vocab import Vocab

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "explain", "model"):
        g = sub.add_parser(name)
        g.add_argument("--tokens", required=True)
        g.add_argument("--twins", required=True)
        g.add_argument("--out", required=True)
        g.add_argument("--datasets", nargs="+", default=list(DATASETS))
        g.add_argument("--cap", type=int, default=1000)
        g.add_argument("--seed", type=int, default=0)
        if name == "model":
            g.add_argument("--model", required=True)
            g.add_argument("--n", type=int, default=200)
    tg = sub.add_parser("targets")
    tg.add_argument("--audit", required=True)
    tg.add_argument("--quality", required=True)
    tg.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "targets":
        au = json.loads(Path(a.audit).read_text(encoding="utf-8"))
        q = json.loads(Path(a.quality).read_text(encoding="utf-8"))
        res = targets({ds: x["twins"] for ds, x in au.items() if ds != "pooled"}, q)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res))
        return
    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    if a.cmd == "audit":
        report, pooled = {}, ([], [], [])
        for ds, held, train, _pool, syn in draws(tokens, Path(a.twins), a.datasets, a.cap, a.seed):
            report[ds] = {"twins": audit(syn, held, v, a.seed), "positive_control": {"views": view_aucs(train, held, v,
                                                                                                         a.seed)}}
            for lst, x in zip(pooled, (held, train, syn)):
                lst += x
            print(ds, json.dumps({k: round(x, 3) for k, x in report[ds]["twins"]["views"].items()}), flush=True)
        held, train, syn = pooled
        report["pooled"] = {"twins": {"views": view_aucs(syn, held, v, a.seed)},
                            "positive_control": {"views": view_aucs(train, held, v, a.seed)}}
        Path(a.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        return
    if a.cmd == "explain":
        report = {}
        for ds, held, train, _pool, syn in draws(tokens, Path(a.twins), a.datasets, a.cap, a.seed):
            report[ds] = {}
            for name, hours in (("first_24h", 24), ("whole", None)):
                e = explain(syn, held, v, hours, seed=a.seed)
                c = explain(train, held, v, hours, seed=a.seed)
                e["positive_control"] = {"unigram_auc": c["unigram_auc"], "max_abs_t": c["max_abs_t"]}
                report[ds][name] = e
            print(ds, json.dumps({name: {"unigram_auc": round(x["unigram_auc"], 3),
                                         "top": [y["token"] for y in x["tokens"][:5]]}
                                  for name, x in report[ds].items()}), flush=True)
            Path(a.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        return
    import torch

    from icutwin.eval import privacy as pv
    from icutwin.generate.support import Support
    from icutwin.tokenization.fsm import compile_tables, mask_tables

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv.load_model(Path(a.model), dev)
    t = compile_tables(v)
    m = mask_tables(t, len(v.tokens))
    sup = Support.load(v, tokens / "support.json") if (tokens / "support.json").exists() else None
    rng = np.random.default_rng(a.seed)
    report = {}
    for ds, held, _train, _pool, syn in draws(tokens, Path(a.twins), a.datasets, a.cap, a.seed, bare=False):
        tuning = pv.first_windows(tokens, ds, "tuning")
        tuning = [tuning[i] for i in rng.choice(len(tuning), size=min(a.n, len(tuning)), replace=False)]
        report[ds] = model_checks(model, held[:a.n], syn[:a.n], tuning, v, t, m, sup, dev)
        print(ds, json.dumps({"nll_by_position": report[ds]["nll_by_position"],
                              "temperature_nll": report[ds]["temperature_nll"]}, default=float), flush=True)
        Path(a.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
