"""S8 fidelity of generated stays against real ones (plan S8 *Fidelity* and *Diversity*), with the harness controls.

Stays are token sequences (whole first windows); features come from the tokens (no parse, so an unconstrained
sample is read as it is): the dataset, the event codes and their value bins, the stay's length in events and in
minutes (interval tokens at their median gap), in-hospital death (``meds.outcomes``: the expired discharge or
an ICU death).

Metrics of a candidate set C against a real reference set R (held-out stays):
* ``code_jsd``: Jensen-Shannon divergence (base 2) of the event-code mix; ``code_jsd_by_dataset`` the same within
  each dataset (C's stays by their dataset token);
* ``value_jsd``: per code the divergence of its value bins, weighted by R's events (codes with ``min_events`` on both
  sides);
* ``rate_spearman``: Spearman correlation of the codes' rates (events per stay);
* ``length_ks`` / ``los_ks``: Kolmogorov-Smirnov statistics of events per stay and minutes per stay;
* ``death_rate_diff``: |C's in-hospital death rate - R's| among stays with an end block;
* ``bp_violation_rate``: a clinical rule (plan S8 *Fidelity*), measured, not enforced: within a time group and a
  measurement site, diastolic below systolic and MAP between them, on the value bins' medians; the share of
  checked groups that break it (C's, held against the positive control's, which binning alone puts above 0);
* ``discriminator_auc``: 5-fold cross-validated AUC of a logistic regression telling C from R on TF-IDF token
  unigrams and bigrams (0.5: indistinguishable; bigrams see order and pairings, which marginals do not);
* ``precision``, ``recall``, ``density``, ``coverage`` (Naeem et al. 2020, k nearest neighbours) in a truncated SVD
  of the unigram TF-IDF fitted on R.

Controls (plan S8 *Harness controls*): the positive control is a second real set (train stays) and sets each
metric's band; the negative controls are that set corrupted -- shuffled times (interval tokens moved between
groups), shuffled values (value bins permuted among the events of each code), swapped dataset labels, a collapsed
generator (a few stays repeated). A metric that flags no negative control is not trusted (``flags``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from icutwin.meds.outcomes import TokenOutcome
from icutwin.tokenization.vocab import Vocab

DIVERGENCES = ("code_jsd", "value_jsd", "length_ks", "los_ks", "death_rate_diff", "bp_violation_rate",
               "discriminator_auc")
AGREEMENTS = ("rate_spearman", "precision", "recall", "density", "coverage")


@dataclass
class Vocabulary:
    """Token classes of a vocabulary as arrays, for the features."""
    v: Vocab
    event: np.ndarray        # event-first tokens (bins, flags, text, n/a, OOV, systems)
    code: list               # token -> event code ('' elsewhere)
    bin: np.ndarray          # token -> value bin (-1 elsewhere)
    minutes: np.ndarray      # interval token -> median minutes (0 elsewhere)
    dataset_pos: int
    end: int
    outcome: TokenOutcome
    bp: dict                 # blood-pressure bin token -> (kind: sbp | dbp | map, site, its bin's median value)


def vocabulary(v: Vocab) -> Vocabulary:
    from icutwin.model.data import layout

    n = len(v.tokens)
    event = np.array([m.get("cls") == "event" and m.get("kind") is not None for m in v.meta])
    code = [m.get("code", "") if event[i] else "" for i, m in enumerate(v.meta)]
    bins = np.array([m.get("bin", -1) if m.get("kind") == "bin" else -1 for m in v.meta])
    minutes = np.zeros(n)
    sk = v.interval_sketches["*"]
    for i, m in enumerate(v.meta):
        if m.get("cls") == "interval":
            s = sk[m["interval"]]
            minutes[i] = s[len(s) // 2] if s else 0.0
    bp = {}
    for i, m in enumerate(v.meta):
        parts = (m.get("code") or "").split("//")
        if m.get("kind") == "bin" and len(parts) == 3 and parts[0] == "VITAL" and parts[1] in ("sbp", "dbp", "map"):
            sk = v.value_sketches[m["code"]][m["bin"]]
            if sk:
                bp[i] = (parts[1], parts[2], float(sk[len(sk) // 2]))
    return Vocabulary(v, event, code, bins, minutes, layout(v).dataset_pos, v.id("[END]"), TokenOutcome(v), bp)


@dataclass
class Feat:
    dataset: str
    codes: list
    pairs: list              # (code, bin) of value events
    events: int
    minutes: float
    death: bool | None       # in-hospital death; None without an end block or a recorded outcome
    bp_checked: int = 0      # (time group, site) pairs with a systolic and a diastolic value
    bp_violations: int = 0


def features(ids: np.ndarray, voc: Vocabulary) -> Feat:
    ids = np.asarray(ids)
    tok = voc.v.tokens[int(ids[voc.dataset_pos])] if len(ids) > voc.dataset_pos else ""
    ds = tok.split("//")[-1] if tok.startswith("PREFIX//dataset//") else "unknown"
    ev = ids[voc.event[ids]]
    codes = [voc.code[t] for t in ev]
    pairs = [(voc.code[t], int(voc.bin[t])) for t in ev if voc.bin[t] >= 0]
    checked, bad = _bp_rule(ids, voc)
    return Feat(ds, codes, pairs, int(len(ev)), float(voc.minutes[ids].sum()), voc.outcome.death(ids), checked, bad)


def _bp_rule(ids: np.ndarray, voc: Vocabulary) -> tuple[int, int]:
    """(groups checked, groups breaking the rule): per time group and site, diastolic < systolic and
    diastolic <= MAP <= systolic when MAP is there."""
    checked = bad = 0
    group: dict = {}

    def close():
        nonlocal checked, bad
        for vals in group.values():
            if "sbp" in vals and "dbp" in vals:
                checked += 1
                s, d = vals["sbp"], vals["dbp"]
                if d >= s or ("map" in vals and not d <= vals["map"] <= s):
                    bad += 1
        group.clear()

    for t in ids:
        t = int(t)
        if voc.minutes[t] > 0:
            close()
        elif t in voc.bp:
            kind, site, val = voc.bp[t]
            group.setdefault(site, {}).setdefault(kind, val)
    close()
    return checked, bad


def _jsd(a: Counter, b: Counter) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    p = np.array([a.get(k, 0) for k in keys], dtype=np.float64)
    q = np.array([b.get(k, 0) for k in keys], dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return 1.0
    p, q = p / p.sum(), q / q.sum()
    mix = (p + q) / 2

    def kl(x):
        k = x > 0
        return float(np.sum(x[k] * np.log2(x[k] / mix[k])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def _ks(a, b) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic) if len(a) and len(b) else 1.0


def _matrix(seqs: list, n_vocab: int, bigram_bits: int = 18):
    """Token unigram counts and hashed bigram counts, as one sparse matrix."""
    from scipy import sparse

    rows, cols, vals = [], [], []
    for r, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.int64)
        u, c = np.unique(s, return_counts=True)
        big = (s[:-1] * 1_000_003 + s[1:]) % (1 << bigram_bits) + n_vocab
        bu, bc = np.unique(big, return_counts=True)
        rows.append(np.full(len(u) + len(bu), r))
        cols.append(np.r_[u, bu])
        vals.append(np.r_[c, bc])
    return sparse.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(seqs), n_vocab + (1 << bigram_bits)))


def discriminator_auc(real: list, cand: list, n_vocab: int, seed: int = 0, bigrams: bool = True) -> float:
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    x = _matrix(list(real) + list(cand), n_vocab)
    if not bigrams:                     # token content alone, without order
        x = x[:, :n_vocab]
    y = np.r_[np.zeros(len(real)), np.ones(len(cand))]
    clf = make_pipeline(TfidfTransformer(sublinear_tf=True), LogisticRegression(C=1.0, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, x, y, cv=cv, scoring="roc_auc")))


def prdc(real_x: np.ndarray, fake_x: np.ndarray, k: int = 5) -> dict:
    """Precision, recall, density and coverage (Naeem et al. 2020) of embeddings."""
    from sklearn.neighbors import NearestNeighbors

    def radii(x):
        d, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
        return d[:, -1]

    r_real, r_fake = radii(real_x), radii(fake_x)
    f, r = np.asarray(fake_x, dtype=np.float64), np.asarray(real_x, dtype=np.float64)
    d2 = (f * f).sum(1)[:, None] + (r * r).sum(1)[None, :] - 2.0 * (f @ r.T)    # fake x real, without a
    d = np.sqrt(np.maximum(d2, 0))                                                # fake x real x dims array
    inside = d <= r_real[None, :]
    return {"precision": float(inside.any(1).mean()),
            "recall": float((d <= r_fake[:, None]).any(0).mean()),
            "density": float(inside.sum() / (k * len(fake_x))),
            "coverage": float((d.min(0) <= r_real).mean())}


def compare(real: list, cand: list, voc: Vocabulary, min_events: int = 20, svd_dims: int = 32, seed: int = 0) -> dict:
    """The metrics of a candidate set against the real reference set (module docstring)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    fr, fc = [features(s, voc) for s in real], [features(s, voc) for s in cand]
    cr, cc = Counter(c for f in fr for c in f.codes), Counter(c for f in fc for c in f.codes)
    out = {"stays": [len(real), len(cand)], "code_jsd": _jsd(cr, cc)}
    by_ds = {}
    for ds in sorted({f.dataset for f in fr}):
        a = Counter(c for f in fr if f.dataset == ds for c in f.codes)
        b = Counter(c for f in fc if f.dataset == ds for c in f.codes)
        if sum(a.values()) and sum(b.values()):
            by_ds[ds] = _jsd(a, b)
    out["code_jsd_by_dataset"] = by_ds
    out["code_jsd_by_dataset_max"] = max(by_ds.values()) if by_ds else None
    vr, vc = Counter(p for f in fr for p in f.pairs), Counter(p for f in fc for p in f.pairs)
    per_code_r, per_code_c = {}, {}
    for (c, b), n in vr.items():
        per_code_r.setdefault(c, Counter())[b] = n
    for (c, b), n in vc.items():
        per_code_c.setdefault(c, Counter())[b] = n
    num = den = 0.0
    for c, a in per_code_r.items():
        b = per_code_c.get(c)
        if b and sum(a.values()) >= min_events and sum(b.values()) >= min_events:
            num += sum(a.values()) * _jsd(a, b)
            den += sum(a.values())
    out["value_jsd"] = num / den if den else None
    from scipy.stats import spearmanr

    keys = sorted(set(cr) | set(cc))
    rr = np.array([cr.get(k, 0) / len(fr) for k in keys])
    rc = np.array([cc.get(k, 0) / max(len(fc), 1) for k in keys])
    out["rate_spearman"] = float(spearmanr(rr, rc).statistic) if len(keys) > 2 else None
    out["length_ks"] = _ks([f.events for f in fr], [f.events for f in fc])
    out["los_ks"] = _ks([f.minutes for f in fr], [f.minutes for f in fc])
    dr = [f.death for f in fr if f.death is not None]
    dc = [f.death for f in fc if f.death is not None]
    out["death_rate_diff"] = abs(float(np.mean(dc)) - float(np.mean(dr))) if dr and dc else None
    n_chk = sum(f.bp_checked for f in fc)
    out["bp_violation_rate"] = sum(f.bp_violations for f in fc) / n_chk if n_chk else None
    out["discriminator_auc"] = discriminator_auc(real, cand, len(voc.v.tokens), seed)
    x = _matrix(list(real) + list(cand), len(voc.v.tokens))[:, :len(voc.v.tokens)]
    tf = TfidfTransformer(sublinear_tf=True).fit(x[:len(real)])
    svd = TruncatedSVD(svd_dims, random_state=seed).fit(tf.transform(x[:len(real)]))
    er, ec = svd.transform(tf.transform(x[:len(real)])), svd.transform(tf.transform(x[len(real):]))
    out.update(prdc(er, ec))
    return out


# -- controls -----------------------------------------------------------------------------------------------------
def shuffled_times(seqs: list, voc: Vocabulary, rng: np.random.Generator) -> list:
    """Interval tokens moved between a stay's groups (the same tokens, another order of gaps)."""
    out = []
    is_int = voc.minutes > 0
    for s in seqs:
        s = np.array(s)
        at = np.flatnonzero(is_int[s])
        s[at] = s[rng.permutation(at)]
        out.append(s)
    return out


def shuffled_values(seqs: list, voc: Vocabulary, rng: np.random.Generator) -> list:
    """Value bins permuted among the events of each code (every code's bin mix kept, its pairing with the stay
    lost)."""
    v = voc.v
    tok_of = {(m.get("code"), m.get("bin")): i for i, m in enumerate(v.meta) if m.get("kind") == "bin"}
    pos = {}
    for k, s in enumerate(seqs):
        for j in np.flatnonzero(voc.bin[np.asarray(s)] >= 0):
            pos.setdefault(voc.code[s[j]], []).append((k, j, int(voc.bin[s[j]])))
    out = [np.array(s) for s in seqs]
    for c, items in pos.items():
        bins = rng.permutation([b for _k, _j, b in items])
        for (k, j, _b), b in zip(items, bins):
            out[k][j] = tok_of[(c, int(b))]
    return out


def swapped_datasets(seqs: list, voc: Vocabulary, rng: np.random.Generator) -> list:
    v = voc.v
    ds_toks = sorted({int(s[voc.dataset_pos]) for s in seqs if v.tokens[int(s[voc.dataset_pos])].startswith("PREFIX//dataset//")
                      and not v.tokens[int(s[voc.dataset_pos])].endswith("unknown")})
    nxt = {a: ds_toks[(i + 1) % len(ds_toks)] for i, a in enumerate(ds_toks)}
    out = []
    for s in seqs:
        s = np.array(s)
        s[voc.dataset_pos] = nxt.get(int(s[voc.dataset_pos]), s[voc.dataset_pos])
        out.append(s)
    return out


def collapsed(seqs: list, rng: np.random.Generator, distinct: int = 10) -> list:
    keep = rng.choice(len(seqs), size=min(distinct, len(seqs)), replace=False)
    return [np.array(seqs[keep[i % len(keep)]]) for i in range(len(seqs))]


def flags(positive: dict, cand: dict, margin: float = 0.05) -> list:
    """The metrics on which a candidate is worse than the positive control by more than ``margin`` (absolute; the
    by-dataset code divergence by its maximum)."""
    out = []
    for k in (*DIVERGENCES, "code_jsd_by_dataset_max"):
        if positive.get(k) is not None and cand.get(k) is not None and cand[k] > positive[k] + margin:
            out.append(k)
    for k in AGREEMENTS:
        if positive.get(k) is not None and cand.get(k) is not None and cand[k] < positive[k] - margin:
            out.append(k)
    return out


def harness(real_eval: list, real_ref: list, voc: Vocabulary, candidates: dict, seed: int = 0,
            train: list | None = None) -> dict:
    """Every candidate set and every control against ``real_eval``; ``real_ref`` (other real stays, as many) is
    the positive control and the base of the negative controls. With ``train`` (the model's training stays, a
    sample; stays equal to one of ``real_ref`` are left out), the diversity metrics (``eval.diversity``) join each
    report and the copying generator joins the controls."""
    from icutwin.eval import diversity as dv
    from icutwin.generate.bank import seq_hash

    rng = np.random.default_rng(seed)
    controls = {"shuffled_times": shuffled_times(real_ref, voc, rng),
                "shuffled_values": shuffled_values(real_ref, voc, rng),
                "swapped_datasets": swapped_datasets(real_ref, voc, rng),
                "collapsed": collapsed(real_ref, rng)}
    if train is not None:
        ref = {seq_hash(s) for s in real_ref}
        train = [s for s in train if seq_hash(s) not in ref]
        controls["copying"] = dv.copying(train, len(real_ref), rng)

    def score(seqs):
        m = compare(real_eval, seqs, voc, seed=seed)
        if train is not None:
            m["diversity"] = dv.compare(real_eval, seqs, voc, train, seed)
        return m

    def flagged(m):
        return flags(pos, m) + (dv.flags(pos["diversity"], m["diversity"]) if train is not None else [])

    pos = score(real_ref)
    out = {"positive_control": pos, "negative_controls": {}, "candidates": {}}
    for name, seqs in controls.items():
        m = score(seqs)
        out["negative_controls"][name] = {"metrics": m, "flags": flagged(m)}
    for name, seqs in candidates.items():
        m = score(seqs)
        out["candidates"][name] = {"metrics": m, "flags": flagged(m)}
    out["controls_detected"] = all(x["flags"] for x in out["negative_controls"].values())
    return out
