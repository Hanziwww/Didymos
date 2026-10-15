"""Factory episodes as model inputs (plan Phase G, Architecture 1-4). ``Builder(...)(episodes)`` returns a dict of
numpy arrays (``to_torch`` moves it to a device); it runs in the data loader's workers.

Per patient (a context patient's complete trajectory, or a query's trajectory up to its last prediction time):
* **tokens**, each with its time in minutes since ICU admission:
  - events (channel, standardized value ``ev_z``, the source decile, flags);
  - interval starts (with symlog of their value) and stops, each one minute after the change (as the factory cuts
    a query: a state starting at the cut is not yet known there, and one running past it shows no stop);
  - statics at their availability time;
  - **label tokens**: the patient's own label timeline (context ``lab_*``, query ``own_lab_*``), each at its
    ``t_resolved``, so a state at t holds exactly the labels known by t (forecasts carry none: their values are
    events already);
  - **row marks**: one at each row time (a context patient's kept label times, a query's prediction times).
* **patches**: a token belongs to the patch (previous boundary, next boundary], the boundaries being the hourly grid
  and every row time, so the state at a row time t contains every token with time <= t and none after it.
* **rows**:
  - context **train rows** (the state at a label time with its label; per patient at most ``max_rows``, drawn
    uniformly and never by outcome, so the context's base rate is kept);
  - query **test rows** (the state at t_k, with the target there).

Channels: a channel is its global id (zeroed when the episode is anonymized, and in training dropped for 10% of an
episode's channels) plus a random identity vector drawn per episode and channel (``rand_id``), so the model ties a
channel across patients by behaviour, as it must for anonymized episodes and unknown codes. Values that carry units
(regression labels, forecast targets) are standardized per episode from the context's labels (median and IQR) and
squashed (symlog), so units never name a channel; a descriptor threshold is read as z only.
"""

from __future__ import annotations

import numpy as np

KINDS = {"event": 0, "iv_start": 1, "iv_stop": 2, "static": 3, "label": 4, "mark": 5}
LABELS = ("binary", "hosp_death", "ordinal", "regression", "tte", "multihorizon", "forecast")
EVENTS = ("binary", "hosp_death")                         # a 0/1 label at a horizon
PROCESSES = ("tte", "multihorizon")                       # an event process: time, cause, censoring
PATCH_MIN = 60
NO_BIN = 10
MAX_CLASSES = 10
MAX_HORIZONS = 8
F_LAB = 44                                                # label payload width (``label_features``)
F_TASK = 40                                               # task token width (``task_tokens``)
CADENCES = ("rolling60", "rolling360", "rolling1440", "landmark", "trigger", "lab")
POLICIES = ("random", "stratified", "retrieved", "none")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "FC", "FH")
ELIGIBILITY = ("in_icu", "not_in_event", "never_before")
OPS = ("cross", "cross_value", "and", "or")
TASK_KINDS = {"header": 0, "term": 1, "horizon": 2, "forecast_channel": 3}
U_CLIP = 8.0


def symlog(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def robust_stats(x) -> tuple[float, float]:
    """(median, scale) of finite values: the IQR / 1.349, else the standard deviation, else 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float(np.median(x)) if len(x) else 0.0), 1.0
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    scale = (q3 - q1) / 1.349
    if not scale > 0:
        scale = float(np.std(x))
    if not scale > 0:
        scale = max(abs(med) * 0.1, 1.0)
    return float(med), float(scale)


def standardize(x, stats: tuple[float, float]) -> np.ndarray:
    """symlog((x - median) / scale), clipped: a value in a scale-free, bounded space (``heads`` bar grid)."""
    med, scale = stats
    return np.clip(symlog((np.asarray(x, dtype=np.float64) - med) / scale), -U_CLIP, U_CLIP).astype(np.float32)


def label_features(kind: str, d: dict, prefix: str, sl: slice, reg: tuple = (0.0, 1.0)) -> np.ndarray:
    """[n, F_LAB] payload of label rows ``sl`` of a part's label arrays: validity, reason bits, then the type's
    fields (0/1; the class one-hot; the standardized value; the event process: observed, log time, cause; the
    multi-horizon labels and their mask). Column 42 is set by callers (a label token's age)."""
    y = np.asarray(d[f"{prefix}y"][sl], dtype=np.float64)
    valid = np.asarray(d[f"{prefix}valid"][sl], dtype=bool)
    reason = np.asarray(d[f"{prefix}reason"][sl], dtype=np.int64)
    n = len(y)
    out = np.zeros((n, F_LAB), dtype=np.float32)
    out[:, 0] = valid
    for b in range(7):
        out[:, 1 + b] = (reason >> b) & 1
    if kind in EVENTS:
        out[:, 8] = np.where(valid, np.nan_to_num(y), 0.0)
    elif kind == "ordinal":
        cls = np.clip(np.nan_to_num(y), 0, MAX_CLASSES - 1).astype(np.int64)
        out[np.flatnonzero(valid), 9 + cls[valid]] = 1.0
    elif kind == "regression":
        out[:, 19] = np.where(valid, standardize(np.nan_to_num(y), reg), 0.0)
    elif kind in PROCESSES:
        te = np.asarray(d[f"{prefix}t_event"][sl], dtype=np.float64)
        tc = np.asarray(d[f"{prefix}t_censor"][sl], dtype=np.float64)
        cause = np.asarray(d[f"{prefix}cause"][sl], dtype=np.int64)
        ev = (cause == 0) & np.isfinite(te)
        out[:, 20] = ev & valid
        out[:, 21] = np.where(valid, np.log1p(np.clip(np.nan_to_num(np.where(ev, te, tc)), 0, None)), 0.0)
        for c in range(4):
            out[:, 22 + c] = (cause == c) & valid
        if kind == "multihorizon" and f"{prefix}yh" in d:
            yh = np.asarray(d[f"{prefix}yh"][sl], dtype=np.float64)[:, :MAX_HORIZONS]
            mh = np.asarray(d[f"{prefix}mh"][sl], dtype=bool)[:, :MAX_HORIZONS]
            out[:, 26:26 + yh.shape[1]] = np.where(mh, np.nan_to_num(yh), 0.0)
            out[:, 34:34 + mh.shape[1]] = mh
    return out


def _cadence(c) -> int:
    kind = c[0]
    if kind == "rolling":
        return {60: 0, 360: 1, 1440: 2}.get(int(c[1]), 0)
    return {"landmark": 3, "ends": 4, "onsets": 4, "lab": 5}.get(kind, 4)


def _direction(x) -> float:
    return -1.0 if str(x).lower() in ("below", "down", "<", "lt", "fall", "low") else 1.0


class Builder:
    """Episodes → a batch. ``channel_names``: the store's channel table (global ids); ``n_static``: the static key
    vocabulary size (``factory.trajectories.static_vocab``); ``train``: draw row subsets and drop global channel ids
    (off for evaluation, where every row is used unless ``max_rows`` is set)."""

    def __init__(self, channel_names: list, n_static: int, train: bool = True, max_rows: int | None = 64,
                 d_id: int = 32, global_drop: float = 0.1, seed: int = 0):
        self.names = list(channel_names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.n_ch, self.n_static = len(self.names), int(n_static)
        self.train, self.max_rows, self.d_id = train, max_rows, d_id
        self.global_drop, self.seed = global_drop, seed

    def __call__(self, eps: list) -> dict:
        return self.build(eps)

    # -- one episode ---------------------------------------------------------------------------------------------
    def _channels(self, ep: dict) -> tuple:
        """(sorted channel ids the episode uses, their global ids, their random identity vectors, descriptor and
        forecast channel ids in the episode's id space)."""
        task, ctx, q = ep["task"], ep["context"], ep["query"]
        gmap = ep["channels"].get("global_ids")
        anon = gmap is not None
        local = {int(g): k for k, g in enumerate(gmap)} if anon else None
        desc_ids = []
        for term in (task.get("descriptor") or {}).get("terms", []):
            ch = term.get("channel")
            if ch is None:
                desc_ids.append(-1)
            elif anon:
                desc_ids.append(int(ch) if not isinstance(ch, str) else -1)
            else:
                desc_ids.append(self.index.get(ch, -1) if isinstance(ch, str) else int(ch))
        fc_ids = []
        spec = task["program"]["spec"]
        if task["label"] == "forecast":
            for name in spec["target"][1]:
                g = self.index.get(name, -1)
                fc_ids.append(local.get(g, -1) if anon and g >= 0 else g)
        used = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in
                                         (ctx["ev_c"], ctx["iv_c"], q["ev_c"], q["iv_c"],
                                          [i for i in desc_ids + fc_ids if i >= 0])]))
        # each channel's random identity and global-id dropout come from a seed of its own (episode, channel): they
        # never depend on which other channels the episode holds (a channel seen only after t_k changes nothing at
        # t_k), and an episode's channel draws are the same in any batch
        index = int(ep["recipe"].get("index", 0))
        vec = np.empty((len(used), self.d_id), dtype=np.float32)
        drop = np.zeros(len(used), dtype=bool)
        for k, c in enumerate(used.tolist()):
            g = np.random.default_rng([self.seed, index, 23, int(c)])
            vec[k] = g.standard_normal(self.d_id) / np.sqrt(self.d_id)
            drop[k] = self.train and g.random() < self.global_drop
        gid = np.full(len(used), self.n_ch, dtype=np.int64) if anon else np.where(drop, self.n_ch, used)
        return used, gid.astype(np.int64), vec, desc_ids, fc_ids

    def _forecast_stats(self, ctx: dict, n_c: int, mode: str) -> np.ndarray:
        """[2 slots, C, 2] (median, scale) per forecast channel from the context's observed targets (grid mode: all
        steps pooled into slot 0; next mode: slot 0 values, slot 1 delays)."""
        st = np.tile(np.array([0.0, 1.0]), (2, n_c, 1))
        yf, mf = ctx.get("lab_yf"), ctx.get("lab_mf")
        if yf is None or len(yf) == 0:
            return st
        yf, mf = np.asarray(yf, dtype=np.float64), np.asarray(mf, dtype=bool)
        for c in range(n_c):
            if mode == "grid":
                st[0, c] = st[1, c] = robust_stats(yf[:, :, c][mf[:, :, c]])
            else:
                for s in range(min(2, yf.shape[1])):
                    st[s, c] = robust_stats(yf[:, s, c][mf[:, s, c]])
        return st

    def _forecast_u(self, yf, mf, stats, mode: str) -> np.ndarray:
        yf = np.asarray(yf, dtype=np.float64)
        u = np.zeros(yf.shape, dtype=np.float32)
        for c in range(yf.shape[2]):
            for h in range(yf.shape[1]):
                s = 0 if mode == "grid" else min(h, 1)
                u[:, h, c] = standardize(np.nan_to_num(yf[:, h, c]), tuple(stats[s, c]))
        return np.where(np.asarray(mf, dtype=bool), u, 0.0).astype(np.float32)

    def _patient(self, part: dict, k: int, lab_prefix: str, rows_t: np.ndarray, cut: int, label: str,
                 lookup, reg: tuple, use_labels: bool) -> dict:
        """One patient's tokens, patches and the patch of each row time."""
        def sl(name):
            off = np.asarray(part[f"{name}_off"])
            return slice(int(off[k]), int(off[k + 1]))

        e, i, s = sl("ev"), sl("iv"), sl("st")
        parts = []
        # events
        t = np.asarray(part["ev_t"][e], dtype=np.int64)
        if len(t):
            gid, slot = lookup(np.asarray(part["ev_c"][e], dtype=np.int64))
            z = np.asarray(part["ev_z"][e], dtype=np.float64)
            has = np.isfinite(z)
            b = np.asarray(part["ev_bin"][e], dtype=np.int64)
            parts.append(dict(t=t, kind=np.full(len(t), KINDS["event"]), gid=gid, slot=slot, skey=None,
                              val=np.where(has, symlog(np.clip(np.nan_to_num(z), -50, 50)), 0.0), has=has,
                              bin=np.where(b <= 9, b, NO_BIN), flag=np.asarray(part["ev_f"][e], dtype=np.int64)))
        # intervals: a start and a stop, each known one minute after it happens (the factory's cut keeps a state
        # starting at the cut out and clips one running past it, so a change exactly at t is not yet known at t)
        ivs, ive = np.asarray(part["iv_s"][i], dtype=np.int64), np.asarray(part["iv_e"][i], dtype=np.int64)
        ivs, ive = ivs + 1, ive + 1
        if len(ivs):
            gid, slot = lookup(np.asarray(part["iv_c"][i], dtype=np.int64))
            v = np.asarray(part["iv_v"][i], dtype=np.float64)
            hv = np.isfinite(v)
            parts.append(dict(t=ivs, kind=np.full(len(ivs), KINDS["iv_start"]), gid=gid, slot=slot, skey=None,
                              val=np.where(hv, symlog(np.nan_to_num(v)), 0.0), has=hv,
                              bin=np.full(len(ivs), NO_BIN), flag=np.zeros(len(ivs), dtype=np.int64)))
            stop = ive <= cut
            if stop.any():
                parts.append(dict(t=ive[stop], kind=np.full(int(stop.sum()), KINDS["iv_stop"]), gid=gid[stop],
                                  slot=slot[stop], skey=None, val=np.zeros(int(stop.sum())),
                                  has=np.zeros(int(stop.sum()), dtype=bool), bin=np.full(int(stop.sum()), NO_BIN),
                                  flag=np.zeros(int(stop.sum()), dtype=np.int64)))
        # statics at their availability
        sk = np.asarray(part["st_k"][s], dtype=np.int64)
        if len(sk):
            sv = np.asarray(part["st_v"][s], dtype=np.float64)
            parts.append(dict(t=np.maximum(np.asarray(part["st_avail"][s], dtype=np.int64), 0),
                              kind=np.full(len(sk), KINDS["static"]), gid=None, slot=None, skey=sk,
                              val=symlog(sv), has=np.ones(len(sk), dtype=bool), bin=np.full(len(sk), NO_BIN),
                              flag=np.zeros(len(sk), dtype=np.int64)))
        lab_parts = []
        # the patient's own labels, each at its resolution time
        if use_labels and f"{lab_prefix}off" in part:
            lo = np.asarray(part[f"{lab_prefix}off"])
            ls = slice(int(lo[k]), int(lo[k + 1]))
            tr = np.asarray(part[f"{lab_prefix}t_resolved"][ls], dtype=np.int64)
            if len(tr):
                feats = label_features(label, part, lab_prefix, ls, reg)
                lt = np.asarray(part[f"{lab_prefix}t"][ls], dtype=np.int64)
                feats[:, 42] = np.log1p(np.clip(tr - lt, 0, None) / 60.0)
                n = len(tr)
                parts.append(dict(t=tr, kind=np.full(n, KINDS["label"]), gid=None, slot=None, skey=None,
                                  val=np.zeros(n), has=np.zeros(n, dtype=bool), bin=np.full(n, NO_BIN),
                                  flag=np.zeros(n, dtype=np.int64)))
                lab_parts.append((len(parts) - 1, feats))
        # row marks
        rt = np.asarray(rows_t, dtype=np.int64)
        if len(rt):
            n = len(rt)
            parts.append(dict(t=rt, kind=np.full(n, KINDS["mark"]), gid=None, slot=None, skey=None, val=np.zeros(n),
                              has=np.zeros(n, dtype=bool), bin=np.full(n, NO_BIN), flag=np.zeros(n, dtype=np.int64)))
        tok = {}
        n_tot = sum(len(p["t"]) for p in parts)
        for key, fill, dt in (("t", 0, np.int64), ("kind", 0, np.int64), ("gid", self.n_ch, np.int64),
                              ("slot", -1, np.int64), ("skey", self.n_static, np.int64), ("val", 0.0, np.float32),
                              ("has", False, bool), ("bin", NO_BIN, np.int64), ("flag", 0, np.int64)):
            tok[key] = np.concatenate([p[key] if p[key] is not None else np.full(len(p["t"]), fill)
                                       for p in parts]).astype(dt) if parts else np.zeros(0, dtype=dt)
        lab = np.zeros((n_tot, F_LAB), dtype=np.float32)
        starts = np.cumsum([0] + [len(p["t"]) for p in parts])
        for j, feats in lab_parts:
            lab[starts[j]:starts[j + 1]] = feats
        keep = tok["t"] <= cut
        tok = {kk: x[keep] for kk, x in tok.items()}
        lab = lab[keep]
        # patches: the hourly grid and every row time
        tt = tok["t"]
        grid_end = -((-tt) // PATCH_MIN) * PATCH_MIN
        marks = np.unique(rt)
        if len(marks):
            j = np.searchsorted(marks, tt, "left")
            m_next = np.where(j < len(marks), marks[np.minimum(j, len(marks) - 1)], np.iinfo(np.int64).max)
            end = np.minimum(grid_end, m_next)
        else:
            end = grid_end
        order = np.lexsort((tt, end))
        tok = {kk: x[order] for kk, x in tok.items()}
        lab, end = lab[order], end[order]
        ends, tok_patch = np.unique(end, return_inverse=True)
        gap = np.diff(np.r_[ends[:1] - PATCH_MIN, ends]) / 60.0 if len(ends) else np.zeros(0)
        row_patch = np.searchsorted(ends, rt)
        assert len(rt) == 0 or (ends[row_patch] == rt).all(), "a row time without its patch"
        return {"tok": tok, "lab": lab, "tok_patch": tok_patch, "dt": (end - tok["t"]) / float(PATCH_MIN),
                "ends": ends, "gap": gap, "row_patch": row_patch}

    def _episode(self, ep: dict, e: int, acc: dict) -> None:
        task, r = ep["task"], ep["recipe"]
        label = task["label"]
        rng = np.random.default_rng([self.seed, int(r.get("index", 0)), 17])
        used, gid_of, vec, desc_ids, fc_ids = self._channels(ep)
        slot0 = acc["n_slots"]
        acc["rand"].append(vec)
        acc["n_slots"] += len(used)

        def lookup(c):
            p = np.searchsorted(used, c)
            return gid_of[p], slot0 + p

        def lookup_one(c):
            if c is None or c < 0:
                return self.n_ch, -1
            p = int(np.searchsorted(used, c))
            return int(gid_of[p]), slot0 + p

        ctx, q = ep["context"], ep["query"]
        n_ctx = len(ctx["ev_off"]) - 1
        reg = robust_stats(np.asarray(ctx["lab_y"])[np.asarray(ctx["lab_valid"], dtype=bool)]) \
            if label == "regression" else (0.0, 1.0)
        mode = task["program"]["spec"]["target"][3] if label == "forecast" else None
        n_c = len(fc_ids)
        fstats = self._forecast_stats(ctx, n_c, mode) if label == "forecast" else None
        use_labels = label != "forecast"
        lo = np.asarray(ctx["lab_off"])
        for j in range(n_ctx):
            n_rows = int(lo[j + 1] - lo[j])
            if n_rows == 0:
                continue                                  # a context patient without a label time adds no row
            pick = np.arange(n_rows)
            if self.max_rows is not None and n_rows > self.max_rows:
                pick = np.sort(rng.choice(n_rows, size=self.max_rows, replace=False))
            rows_t = np.asarray(ctx["lab_t"], dtype=np.int64)[int(lo[j]) + pick]
            pt = self._patient(ctx, j, "lab_", rows_t, int(ctx["end_t"][j]), label, lookup, reg, use_labels)
            base = self._add_patient(acc, pt, e, role=0)
            idx = slice(int(lo[j]), int(lo[j + 1]))
            feats = label_features(label, ctx, "lab_", idx, reg)[pick]
            acc["tr_patch"].append(base + pt["row_patch"])
            acc["tr_ep"].append(np.full(len(pick), e))
            acc["tr_lab"].append(feats)
            if label == "forecast":
                acc["tr_fc"].append((self._forecast_u(np.asarray(ctx["lab_yf"])[idx][pick],
                                                      np.asarray(ctx["lab_mf"])[idx][pick], fstats, mode),
                                     np.asarray(ctx["lab_mf"], dtype=bool)[idx][pick]))
            else:
                acc["tr_fc"].append(None)
        qo = np.asarray(q["lab_off"])
        for j in range(len(r["queries"])):
            ls = slice(int(qo[j]), int(qo[j + 1]))
            rows_t = np.asarray(q["lab_t"], dtype=np.int64)[ls]
            cut = int(rows_t.max())
            pt = self._patient(q, j, "own_lab_", rows_t, cut, label, lookup, reg, use_labels)
            base = self._add_patient(acc, pt, e, role=1)
            acc["te_patch"].append(base + pt["row_patch"])
            acc["te_ep"].append(np.full(len(rows_t), e))
            acc["te_t"].append(rows_t)
            self._targets(acc, q, ls, label, reg, fstats, mode)
        self._task(acc, ep, e, desc_ids, fc_ids, lookup_one, n_ctx)
        acc["ep_label"].append(LABELS.index(label))
        acc["ep_ncls"].append(len(task.get("bands") or []) + 1 if label == "ordinal" else 0)
        acc["ep_reg"].append(reg)
        acc["ep_hz"].append([b / 60.0 for b in (task.get("bands") or [])][:MAX_HORIZONS]
                            if label == "multihorizon" else [])
        acc["ep_fc"].append([lookup_one(c) for c in fc_ids] if label == "forecast" else [])
        acc["ep_fc_mode"].append(0 if mode in (None, "grid") else 1)
        acc["ep_index"].append(int(r.get("index", 0)))

    def _targets(self, acc: dict, q: dict, ls: slice, label: str, reg: tuple, fstats, mode) -> None:
        valid = np.asarray(q["lab_valid"][ls], dtype=bool)
        y = np.asarray(q["lab_y"][ls], dtype=np.float64)
        n = len(y)
        if label == "regression":
            yy = np.where(valid, standardize(np.nan_to_num(y), reg), np.nan)
        else:
            yy = np.where(valid, y, np.nan)
        acc["te_valid"].append(valid)
        acc["te_y"].append(yy.astype(np.float32))
        if label in PROCESSES:
            te = np.asarray(q["lab_t_event"][ls], dtype=np.float64)
            tc = np.asarray(q["lab_t_censor"][ls], dtype=np.float64)
            cause = np.asarray(q["lab_cause"][ls], dtype=np.int64)
            ev = (cause == 0) & np.isfinite(te)
            comp = np.isin(cause, (1, 2))                    # death or discharge ends follow-up: a competing event
            acc["te_time"].append(np.where(ev, te, tc).astype(np.float32))
            acc["te_cause"].append(np.where(ev, 0, np.where(comp, cause, -1)).astype(np.int64))
        else:
            acc["te_time"].append(np.zeros(n, dtype=np.float32))
            acc["te_cause"].append(np.full(n, -1, dtype=np.int64))
        if label == "multihorizon":
            acc["te_yh"].append(np.asarray(q["lab_yh"][ls], dtype=np.float32)[:, :MAX_HORIZONS])
            acc["te_mh"].append(np.asarray(q["lab_mh"][ls], dtype=bool)[:, :MAX_HORIZONS] & valid[:, None])
        else:
            acc["te_yh"].append(None)
            acc["te_mh"].append(None)
        if label == "forecast":
            mf = np.asarray(q["lab_mf"][ls], dtype=bool) & valid[:, None, None]
            acc["te_fc"].append((self._forecast_u(np.asarray(q["lab_yf"][ls]), mf, fstats, mode), mf))
        else:
            acc["te_fc"].append(None)

    def _task(self, acc: dict, ep: dict, e: int, desc_ids: list, fc_ids: list, lookup_one, n_ctx: int) -> None:
        """Task tokens: a header (label type and encoding, cadence, the context's selection; with a shown
        descriptor also the horizon, family and eligibility), one token per descriptor term, one per
        multi-horizon horizon, one per forecast channel."""
        task, r = ep["task"], ep["recipe"]
        desc = task.get("descriptor")
        shown = desc is not None
        label = task["label"]
        h = np.zeros(F_TASK, dtype=np.float32)
        h[LABELS.index(label)] = 1
        hz = task.get("horizon_min") or 0
        if shown:
            h[7] = np.log1p(max(hz, 0) / 60.0)
            h[8] = float(hz < 0)
        h[9] = (len(task.get("bands") or []) + 1) / MAX_CLASSES if label == "ordinal" else 0.0
        h[10 + _cadence(task["cadence"])] = 1
        h[16], h[17] = float(shown), float(bool(task.get("anonymized")))
        sel = task.get("selection") or {}
        pol = sel.get("policy", r.get("policy"))
        h[18 + (POLICIES.index(pol) if pol in POLICIES else 3)] = 1
        odds = sel.get("case_odds")
        h[22] = np.log(odds) if odds else 0.0
        h[23] = (sel.get("min_cases") or 0) / 10.0
        h[24] = np.log1p(n_ctx)
        if shown:
            fam = desc.get("family")
            if fam in FAMILIES:
                h[25 + FAMILIES.index(fam)] = 1
            for x in desc.get("eligibility") or []:
                if x in ELIGIBILITY:
                    h[34 + ELIGIBILITY.index(x)] = 1
        h[37] = float(n_ctx == 0)
        toks = [(TASK_KINDS["header"], h, (self.n_ch, -1))]
        if shown:
            for term, cid in zip(desc.get("terms", []), desc_ids, strict=True):
                f = np.zeros(F_TASK, dtype=np.float32)
                op = term.get("op", "cross" if "channel" in term else None)
                if op in OPS:
                    f[OPS.index(op)] = 1
                if "direction" in term:
                    f[4] = _direction(term["direction"])
                z = term.get("z")
                if z is not None and np.isfinite(z):
                    f[5], f[6] = float(np.clip(z, -U_CLIP, U_CLIP)), 1.0
                f[7] = np.log1p((term.get("sustain_min") or 0) / 60.0)
                toks.append((TASK_KINDS["term"], f, lookup_one(cid if "channel" in term else -1)))
        if label == "multihorizon":
            for b in (task.get("bands") or [])[:MAX_HORIZONS]:
                f = np.zeros(F_TASK, dtype=np.float32)
                f[0] = np.log1p(b / 60.0)
                toks.append((TASK_KINDS["horizon"], f, (self.n_ch, -1)))
        if label == "forecast":
            _, _, grid, mode = task["program"]["spec"]["target"]
            for cid in fc_ids:
                f = np.zeros(F_TASK, dtype=np.float32)
                f[0 if mode == "grid" else 1] = 1
                f[2] = np.log1p(max(grid) / 60.0)
                f[3] = len(grid) / 24.0
                toks.append((TASK_KINDS["forecast_channel"], f, lookup_one(cid)))
        for kind, f, (g, s) in toks:
            acc["task_kind"].append(kind)
            acc["task_feat"].append(f)
            acc["task_gid"].append(g)
            acc["task_slot"].append(s)
            acc["task_ep"].append(e)

    def _add_patient(self, acc: dict, pt: dict, e: int, role: int) -> int:
        """Append a patient's tokens and patches; returns the global index of its first patch."""
        base = acc["n_patches"]
        tok = pt["tok"]
        for key in ("kind", "gid", "slot", "skey", "val", "has", "bin", "flag"):
            acc[f"tok_{key}"].append(tok[key])
        acc["tok_dt"].append(pt["dt"].astype(np.float32))
        acc["tok_lab"].append(pt["lab"])
        acc["tok_patch"].append(base + pt["tok_patch"])
        acc["patch_end_h"].append((pt["ends"] / 60.0).astype(np.float32))
        acc["patch_gap_h"].append(pt["gap"].astype(np.float32))
        acc["pat_len"].append(len(pt["ends"]))
        acc["pat_ep"].append(e)
        acc["pat_role"].append(role)
        acc["n_patches"] += len(pt["ends"])
        return base

    # -- the batch -----------------------------------------------------------------------------------------------
    def build(self, eps: list) -> dict:
        keys = ("tok_kind", "tok_gid", "tok_slot", "tok_skey", "tok_val", "tok_has", "tok_bin", "tok_flag", "tok_dt",
                "tok_lab", "tok_patch", "patch_end_h", "patch_gap_h", "pat_len", "pat_ep", "pat_role", "tr_patch",
                "tr_ep", "tr_lab", "tr_fc", "te_patch", "te_ep", "te_t", "te_valid", "te_y", "te_time", "te_cause",
                "te_yh", "te_mh", "te_fc", "task_kind", "task_feat", "task_gid", "task_slot", "task_ep", "ep_label",
                "ep_ncls", "ep_reg", "ep_hz", "ep_fc", "ep_fc_mode", "ep_index", "rand")
        acc: dict = {k: [] for k in keys}
        acc["n_slots"], acc["n_patches"] = 0, 0
        for e, ep in enumerate(eps):
            self._episode(ep, e, acc)
        out: dict = {}

        def cat(key, dtype, width=None):
            parts = [x for x in acc[key] if x is not None and len(x)]
            if parts:
                return np.concatenate(parts).astype(dtype)
            return np.zeros((0,) if width is None else (0, *width), dtype=dtype)

        for key, dt in (("tok_kind", np.int64), ("tok_gid", np.int64), ("tok_slot", np.int64), ("tok_skey", np.int64),
                        ("tok_val", np.float32), ("tok_has", np.float32), ("tok_bin", np.int64),
                        ("tok_dt", np.float32), ("tok_patch", np.int64), ("patch_end_h", np.float32),
                        ("patch_gap_h", np.float32), ("tr_patch", np.int64), ("tr_ep", np.int64),
                        ("te_patch", np.int64), ("te_ep", np.int64), ("te_t", np.int64), ("te_valid", bool),
                        ("te_y", np.float32), ("te_time", np.float32), ("te_cause", np.int64)):
            out[key] = cat(key, dt)
        flags = cat("tok_flag", np.int64)
        out["tok_flag"] = ((flags[:, None] >> np.arange(8)) & 1).astype(np.float32)
        out["tok_lab"] = cat("tok_lab", np.float32, (F_LAB,))
        out["tr_lab"] = cat("tr_lab", np.float32, (F_LAB,))
        out["pat_off"] = np.concatenate([[0], np.cumsum(acc["pat_len"])]).astype(np.int64)
        out["pat_ep"] = np.asarray(acc["pat_ep"], dtype=np.int64)
        out["pat_role"] = np.asarray(acc["pat_role"], dtype=np.int64)
        out["patch_pat"] = np.repeat(np.arange(len(acc["pat_len"])), acc["pat_len"]).astype(np.int64)
        # multi-horizon targets, padded to the batch's horizons
        n_te = len(out["te_y"])
        out["te_yh"] = np.zeros((n_te, MAX_HORIZONS), dtype=np.float32)
        out["te_mh"] = np.zeros((n_te, MAX_HORIZONS), dtype=bool)
        at = 0
        for yh, mh, y in zip(acc["te_yh"], acc["te_mh"], acc["te_y"], strict=True):
            if yh is not None:
                out["te_yh"][at:at + len(y), :yh.shape[1]] = np.nan_to_num(yh)
                out["te_mh"][at:at + len(y), :mh.shape[1]] = mh
            at += len(y)
        # forecast targets and training-row values, padded to the batch's steps and channels
        shapes = [x[0].shape[1:] for x in acc["te_fc"] + acc["tr_fc"] if x is not None]
        hf = max((s[0] for s in shapes), default=0)
        cf = max((s[1] for s in shapes), default=0)
        for part, n_rows in (("te", n_te), ("tr", len(out["tr_patch"]))):
            u = np.zeros((n_rows, hf, cf), dtype=np.float32)
            m = np.zeros((n_rows, hf, cf), dtype=bool)
            at = 0
            lens = [len(x) for x in acc["te_y"]] if part == "te" else [len(x) for x in acc["tr_patch"]]
            for x, ln in zip(acc[f"{part}_fc"], lens, strict=True):
                if x is not None:
                    u[at:at + ln, :x[0].shape[1], :x[0].shape[2]] = x[0]
                    m[at:at + ln, :x[1].shape[1], :x[1].shape[2]] = x[1]
                at += ln
            out[f"{part}_yf"], out[f"{part}_mf"] = u, m
        out["task_kind"] = np.asarray(acc["task_kind"], dtype=np.int64)
        out["task_feat"] = np.stack(acc["task_feat"]).astype(np.float32)
        out["task_gid"] = np.asarray(acc["task_gid"], dtype=np.int64)
        out["task_slot"] = np.asarray(acc["task_slot"], dtype=np.int64)
        out["task_ep"] = np.asarray(acc["task_ep"], dtype=np.int64)
        b = len(eps)
        out["ep_label"] = np.asarray(acc["ep_label"], dtype=np.int64)
        out["ep_ncls"] = np.asarray(acc["ep_ncls"], dtype=np.int64)
        out["ep_reg"] = np.asarray(acc["ep_reg"], dtype=np.float64)
        out["ep_hz"] = np.full((b, MAX_HORIZONS), np.nan, dtype=np.float32)
        for e, hz in enumerate(acc["ep_hz"]):
            out["ep_hz"][e, :len(hz)] = hz
        out["ep_fc_gid"] = np.full((b, max(cf, 1)), self.n_ch, dtype=np.int64)
        out["ep_fc_slot"] = np.full((b, max(cf, 1)), -1, dtype=np.int64)
        for e, chans in enumerate(acc["ep_fc"]):
            for c, (g, s) in enumerate(chans):
                out["ep_fc_gid"][e, c], out["ep_fc_slot"][e, c] = g, s
        out["ep_fc_mode"] = np.asarray(acc["ep_fc_mode"], dtype=np.int64)
        out["ep_index"] = np.asarray(acc["ep_index"], dtype=np.int64)
        out["rand_id"] = (np.concatenate(acc["rand"]) if acc["rand"] else np.zeros((0, self.d_id))).astype(np.float32)
        out["n_episodes"] = np.int64(b)
        return out


def to_torch(batch: dict, device=None) -> dict:
    """The batch's arrays as tensors on ``device`` (non-blocking from pinned memory where possible)."""
    import torch

    out = {}
    for k, x in batch.items():
        t = torch.as_tensor(np.asarray(x))
        out[k] = t.to(device, non_blocking=True) if device is not None else t
    return out
