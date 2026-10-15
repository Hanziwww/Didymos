"""S8 utility (plan S8 *Utility*): train on synthetic, test on real (TSTR) against train on real, test on real (TRTR).

Tasks are read off token sequences, the same way for real and generated stays (``eval.forecast.Reader``): a
landmark ``hours`` after the ICU admission marker (interval tokens at their median gaps); the observation window is
every token before it (prefix, header and the timeline so far); stays that end or leave the ICU before the landmark
are out. Outcomes after the landmark: in-hospital death, ICU stay over 7 days, and new IMV, vasopressor or RRT (among
stays without one before the landmark). Model: logistic regression on the window's sublinear TF-IDF token counts
(fitted on the training set), scored by AUROC on held-out stays. TSTR's training set is the candidate's stays, TRTR's
as many real training stays; ``utility_ratio`` = (TSTR AUROC - 0.5) / (TRTR AUROC - 0.5).

Controls: real training stays in the candidate's place (positive: the ratio near 1), and stays whose continuation
after the landmark is swapped between stays (``swapped_outcomes``: history and outcome unrelated, AUROC near 0.5).
"""

from __future__ import annotations

import numpy as np

from icutwin.eval.forecast import Reader

TASKS = ("death", "icu_los_gt_7d", "imv", "vasopressor", "rrt")
NEW_ONSET = ("imv", "vasopressor", "rrt")


def landmark(ids, r: Reader, hours: float = 24.0) -> int | None:
    """Position of the first token at or after ``hours`` past the ICU admission marker (None: the stay ends,
    leaves the ICU or has no admission marker before it)."""
    t, start = 0.0, None
    for k, tok in enumerate(ids):
        tok = int(tok)
        if tok == r.end:
            return None
        if start is None:
            if tok in r.adm:
                start = t
        elif tok in r.dis:
            return None
        m = r.minutes.get(tok, 0.0)
        if m and start is not None and t + m - start >= hours * 60:
            return k
        t += m
    return None


def examples(seqs: list, r: Reader, hours: float = 24.0) -> tuple[list, dict]:
    """The observation windows of the stays still in the ICU at the landmark, and per task the labels (NaN where
    the task does not apply: an unknown outcome, or the event already before the landmark)."""
    windows, labels = [], {k: [] for k in TASKS}
    for ids in seqs:
        ids = [int(x) for x in ids]
        cut = landmark(ids, r, hours)
        if cut is None:
            continue
        o = r.outcomes(ids, after=cut)
        before = set(ids[:cut])
        windows.append(np.asarray(ids[:cut], dtype=np.int64))
        for k in TASKS:
            y = o[k]
            if k in NEW_ONSET and r.sets[k] & before:
                y = None
            labels[k].append(np.nan if y is None else float(y))
    return windows, {k: np.array(v) for k, v in labels.items()}


def _counts(windows: list, n_vocab: int):
    from scipy import sparse

    rows, cols, vals = [], [], []
    for i, w in enumerate(windows):
        u, c = np.unique(w, return_counts=True)
        rows.append(np.full(len(u), i))
        cols.append(u)
        vals.append(c)
    return sparse.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(windows), n_vocab))


def auroc(train: tuple, test: tuple, n_vocab: int, task: str, min_pos: int = 10) -> dict:
    """AUROC on ``test`` of a model fitted on ``train`` (each: windows, labels)."""
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    (wa, la), (wb, lb) = train, test
    ya, yb = la[task], lb[task]
    ka, kb = ~np.isnan(ya), ~np.isnan(yb)
    out = {"train": int(ka.sum()), "test": int(kb.sum()),
           "train_prevalence": float(ya[ka].mean()) if ka.any() else None,
           "test_prevalence": float(yb[kb].mean()) if kb.any() else None, "auroc": None}
    if min(ya[ka].sum(), (1 - ya[ka]).sum(), yb[kb].sum(), (1 - yb[kb]).sum()) < min_pos:
        return out
    xa = _counts([w for w, k in zip(wa, ka) if k], n_vocab)
    xb = _counts([w for w, k in zip(wb, kb) if k], n_vocab)
    tf = TfidfTransformer(sublinear_tf=True).fit(xa)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(tf.transform(xa), ya[ka])
    out["auroc"] = float(roc_auc_score(yb[kb], clf.predict_proba(tf.transform(xb))[:, 1]))
    return out


def swapped_outcomes(seqs: list, r: Reader, rng: np.random.Generator, hours: float = 24.0) -> list:
    """Each stay's window before the landmark joined to another stay's continuation (history and outcome made
    unrelated); stays without a landmark are left as they are."""
    cuts = [landmark([int(x) for x in s], r, hours) for s in seqs]
    have = [i for i, c in enumerate(cuts) if c is not None]
    perm = rng.permutation(have)
    out = [np.asarray(s) for s in seqs]
    for i, j in zip(have, perm):
        out[i] = np.r_[np.asarray(seqs[i])[:cuts[i]], np.asarray(seqs[j])[cuts[j]:]]
    return out


def tstr(real_train: list, real_test: list, candidates: dict, v, hours: float = 24.0, seed: int = 0) -> dict:
    """TRTR (``real_train``) and, per candidate set, TSTR on ``real_test``, per task, with both controls (the
    positive control: ``real_train`` itself in a candidate's place, i.e. the TRTR value; the negative control:
    ``real_train`` with swapped outcomes)."""
    r = Reader(v)
    n = len(v.tokens)
    rng = np.random.default_rng(seed)
    test = examples(real_test, r, hours)
    trtr = examples(real_train, r, hours)
    sets = {"swapped_outcomes": examples(swapped_outcomes(real_train, r, rng, hours), r, hours),
            **{k: examples(s, r, hours) for k, s in candidates.items()}}
    out = {"landmark_hours": hours, "stays": {"test": len(test[0]), "trtr": len(trtr[0]),
                                              **{k: len(s[0]) for k, s in sets.items()}},
           "trtr": {t: auroc(trtr, test, n, t) for t in TASKS}, "sets": {}}
    for name, s in sets.items():
        res = {}
        for t in TASKS:
            a = auroc(s, test, n, t)
            base = out["trtr"][t]["auroc"]
            a["utility_ratio"] = ((a["auroc"] - 0.5) / (base - 0.5) if a["auroc"] is not None and base
                                  and base > 0.5 else None)
            res[t] = a
        out["sets"][name] = res
    return out
