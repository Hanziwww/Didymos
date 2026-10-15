"""The factory's quality suite (plan Phase F, "Acceptance"; S4 measures, S9 accepts): metrics over episodes and stores,
aggregates only. ``tools/factory_report.py`` runs them on the host and writes the factory report.

* ``summarize``: one episode as a row of attributes (program, family group, label, domain, horizon band, cadence,
  site variant, context and queries, realised prevalence per prediction time and per trajectory, censoring,
  positives in the context, prediction times, the stays it used, filter and difficulty).
* ``mix_report`` / ``distribution_report`` / ``program_report``: shares against the configured mix and P*; distinct
  programs, the largest program's share of its family, catalog coverage.
* ``label_functions``: labels of many programs on one fixed panel of patient-times: Cohen's kappa between random pairs
  (median |kappa|, near duplicates with kappa > 0.9) and the effective number of label functions (the Vendi score of
  the centred label vectors).
* ``reuse_report``: patients' uses across episodes (Gini, the largest over the mean), header strata reached,
  datasets against the bank's natural shares.
* ``difficulty``: a probe (ExtraTrees on the episode's tabular snapshot, fitted on the context, scored on the
  queries): AUROC for binary labels, R^2 for values.
* ``context_vendi`` / ``coverage``: diversity of context trajectories and coverage of real trajectories by twins
  (Naeem density and coverage), on bag-of-channel embeddings.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from icutwin.factory import labels as fl


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def shares(values) -> dict:
    c = Counter(values)
    n = sum(c.values())
    return {str(k): v / n for k, v in c.most_common()} if n else {}


def tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def summarize(ep: dict, store) -> dict:
    """An episode's attributes (module docstring)."""
    from icutwin.factory import pool
    from icutwin.factory.episodes import _program

    r = ep["recipe"]
    prog = _program(r["program"])
    ctx, q = ep["context"], ep["query"]
    ok = np.asarray(ctx["lab_valid"], dtype=bool)
    y = np.asarray(ctx["lab_y"], dtype=np.float64)
    off = np.asarray(ctx["lab_off"])
    out = {"index": r["index"], "hash": r["program"]["hash"], "name": r["program"]["name"],
           "family": prog.family, "group": pool.group_of(prog.family), "label": prog.label,
           "domain": pool.domain_of(prog), "horizon_bin": pool.horizon_bin(prog), "cadence": r["cadence"][0],
           "policy": r["policy"], "case_odds": r.get("case_odds"), "min_cases": r.get("min_cases"),
           "as_drawn": bool(r.get("as_drawn")), "site": r["site"], "query_site": r.get("query_site", r["site"]),
           "variant": (r.get("site_variant") or {}).get("kind", "plain"),
           "virtual": (r.get("site_variant") or {}).get("transform"),
           "realism": sorted((r.get("realism") or {}).keys()), "descriptor": r["descriptor"],
           "anonymized": r["anonymize"] is not None, "context": len(r["context"]), "queries": len(r["queries"]),
           "times": [t for qq in r["queries"] for t in qq["times"]],
           "stays": list(r["context"]) + [qq["stay"] for qq in r["queries"]],
           "events": int(len(ctx["ev_t"]) + len(q["ev_t"])), "filter_stage": (r.get("filter") or {}).get("stage"),
           "difficulty_loss": r.get("difficulty_loss"),
           "fn_q": float(prog.target[-1]) if prog.target[0] == "fn" and prog.label == "binary" else None,
           "thinned": float(np.mean((np.asarray(ctx["ev_f"]) & 32) > 0)) if len(ctx["ev_f"]) else 0.0,
           "thinned_trajectories": int(sum(bool(((np.asarray(ctx["ev_f"])[a:b] & 32) > 0).any())
                                           for a, b in zip(ctx["ev_off"][:-1], ctx["ev_off"][1:], strict=True))),
           "thinned_queries": int(sum(bool(((np.asarray(q["ev_f"])[a:b] & 32) > 0).any())
                                      for a, b in zip(q["ev_off"][:-1], q["ev_off"][1:], strict=True))),
           "probe_kind": "auroc" if prog.label in ("binary", "hosp_death", "tte", "multihorizon") else "r2"}
    cens = (np.asarray(ctx["lab_reason"]) & fl.REASONS["censored"]) > 0
    out["censored"] = float(cens.mean()) if len(cens) else None
    if prog.label in ("binary", "hosp_death", "tte", "multihorizon") and ok.any():
        out["prevalence"] = float((y[ok] > 0).mean())
        pos = [bool((y[a:b][ok[a:b]] > 0).any()) for a, b in zip(off[:-1], off[1:])]
        out["per_trajectory"] = float(np.mean(pos)) if pos else None
        out["positive_trajectories"] = int(sum(pos))
    return out


def mix_report(rows: list[dict], cfg: dict) -> dict:
    groups = {g: x["share"] for g, x in cfg["families"].items()}
    have = {r["group"] for r in rows}
    tot = sum(v for g, v in groups.items() if g in have)
    target = {g: v / tot for g, v in groups.items() if g in have}
    real = shares(r["group"] for r in rows)
    out = {"groups": {"realised": real, "target": target, "max_abs_gap": max(abs(real.get(g, 0) - t)
                                                                           for g, t in target.items())}}
    for key, conf in (("variant", cfg["sites"]), ("cadence", None), ("policy", cfg["context_policy"])):
        out[key] = {"realised": shares(r[key] for r in rows), "target": conf}
    out["descriptor_shown"] = {"realised": float(np.mean([r["descriptor"] for r in rows])),
                               "target": cfg["descriptor_shown"]}
    out["anonymized"] = {"realised": float(np.mean([r["anonymized"] for r in rows])), "target": cfg["anonymize"]}
    out["realism_any"] = {"realised": float(np.mean([bool(r["realism"]) for r in rows])),
                          "target": cfg["realism"]["share"]}
    out["zero_context"] = float(np.mean([r["context"] == 0 for r in rows]))
    ev = [r for r in rows if r.get("positive_trajectories") is not None and r["context"]]
    odds = [r["case_odds"] for r in ev if r.get("policy") == "stratified" and r.get("case_odds") is not None]
    out["selection"] = {      # how event tasks' contexts were chosen (the task block's ``selection``)
        "as_drawn": float(np.mean([r.get("as_drawn", False) for r in ev])) if ev else None,
        "no_case": float(np.mean([r["positive_trajectories"] == 0 for r in ev])) if ev else None,
        "stratified_case_odds_p10_p50_p90": [round(float(x), 2) for x in np.quantile(odds, [0.1, 0.5, 0.9])]
        if odds else None}
    return out


def distribution_report(rows: list[dict], targets: dict, prevalence_by_hash: dict | None = None) -> dict:
    """The drawn programs' attributes over episodes against P*, per family group, on the pool's axes
    (``pool.axis_features``: an axis over the labels it applies to): the realised shares, P* there and their total
    variation. A program's prevalence is the pool's (on its panel, ``prevalence_by_hash``), a fresh functional head's
    the one it was drawn with. The clinical group is the anchor (its natural mix), reported, not targeted."""
    import pandas as pd

    from icutwin.factory import pool

    prev = prevalence_by_hash or {}
    out = {}
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g]
        df = pd.DataFrame({"label": [r["label"] for r in sub], "domain": [r["domain"] for r in sub],
                           "horizon_bin": [r["horizon_bin"] for r in sub],
                           "prevalence_bin": [prev.get(r["hash"]) or (pool.prevalence_bin(r["fn_q"])
                                                                      if r.get("fn_q") is not None else "na")
                                              for r in sub]})
        feats = pool.axis_features(df)
        rep_g = {"anchor": g == "clinical"}
        for ax, col in pool.AXES:
            cats = feats[ax]
            on = cats != "na"
            if not on.any():
                continue
            real = shares(cats[on].tolist())
            ps = pool.p_star(targets, col, sorted(set(real) & set(targets["axes"].get(col, {}))))
            rep_g[ax] = {"realised": real, "p_star": ps, "tv": tv(real, ps), "episodes": int(on.sum())}
        out[g] = rep_g
    return out


def program_report(rows: list[dict], catalog_hashes: set) -> dict:
    by_fam: dict = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r["hash"])
    top = {f: max(Counter(h).values()) / len(h) for f, h in by_fam.items()}
    seen = {r["hash"] for r in rows}
    return {"distinct_programs": len(seen), "largest_share_of_family": top,
            "catalog_reached": len(seen & catalog_hashes) / max(len(catalog_hashes), 1)}


def label_functions(store, programs: list, n_rows: int = 2000, n_pairs: int = 5000, seed: int = 0) -> dict:
    """Cohen's kappa between random pairs of binary programs and the effective number of label functions, on one
    fixed panel of ``n_rows`` patient-times (rows where a program does not apply count as negatives: a label function
    is also where it applies)."""
    from icutwin.eval.diversity import vendi
    from icutwin.factory.programs import Evaluator

    rng = np.random.default_rng(seed)
    r = fl.rolling(store, 60)
    r = r.take(np.sort(rng.choice(len(r), size=min(n_rows, len(r)), replace=False)))
    ev = Evaluator(store)
    ys = []
    for p in programs:
        b = ev.evaluate(p, r)
        ys.append(np.where(b.valid, b.y > 0, False))
    ys = np.asarray(ys, dtype=bool)
    keep = ys.any(1) & ~ys.all(1)
    ys = ys[keep]
    k = len(ys)
    if k < 2:
        return {"programs": int(k)}
    pairs = rng.integers(0, k, size=(n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    a, b = ys[pairs[:, 0]].astype(np.float64), ys[pairs[:, 1]].astype(np.float64)
    po = (a == b).mean(1)
    pa, pb = a.mean(1), b.mean(1)
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = np.where(pe < 1, (po - pe) / (1 - pe), 1.0)
    x = ys.astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return {"programs": int(k), "median_abs_kappa": float(np.median(np.abs(kappa))),
            "near_duplicates": float(np.mean(kappa > 0.9)), "effective_label_functions": vendi(x, cap=3000)}


def reuse_report(rows: list[dict], store, split: str = "train") -> dict:
    uses = Counter(s for r in rows for s in r["stays"])
    st = store.stays
    pool_ids = st.loc[st["split"] == split, "stay_id"] if "split" in st else st["stay_id"]
    u = np.array([uses.get(s, 0) for s in pool_ids], dtype=np.float64)
    age = np.digitize(st["age"].astype(float).fillna(-1).to_numpy(), [40, 60, 75])
    strata = (st["dataset"].astype(str) + ":" + age.astype(str) + ":" + st["sex"].astype(str) + ":"
              + st["unit_type"].astype(str) + ":" + st["admission_type"].astype(str) + ":"
              + st["adm_dx_group"].astype(str))
    size = strata.value_counts()
    big = set(size[size >= 20].index)
    used_ids = set(uses)
    reached = {s for s, sid in zip(strata, st["stay_id"]) if s in big and sid in used_ids}
    ds_of = dict(zip(st["stay_id"], st["dataset"], strict=True))      # once (not per patient of every episode)
    ds_use = shares(ds_of.get(s) for r in rows for s in r["stays"])
    return {"gini": gini(u), "max_over_mean": float(u.max() / u.mean()) if u.mean() > 0 else None,
            "unused_share": float((u == 0).mean()), "strata_with_20": len(big),
            "strata_reached": len(reached) / max(len(big), 1), "dataset_use": ds_use}


def difficulty(sampler, ep: dict) -> float | None:
    """The probe's score on an episode: ExtraTrees on the tabular snapshot, fitted on the context's labels, scored on
    the queries' targets (AUROC; R^2 for values); None when either side has too few labels."""
    from icutwin.factory.episodes import _program
    from icutwin.factory.tasks import informative

    prog = _program(ep["recipe"]["program"])
    if prog.label in ("forecast",):
        return None
    idx = sampler.index
    rows_c, y_c, rows_q, y_q = [], [], [], []
    ctx, q = ep["context"], ep["query"]
    for k, sid in enumerate(ep["recipe"]["context"]):
        a, b = ctx["lab_off"][k], ctx["lab_off"][k + 1]
        ok = ctx["lab_valid"][a:b]
        rows_c += [(idx[sid], t) for t in ctx["lab_t"][a:b][ok]]
        y_c += list(ctx["lab_y"][a:b][ok])
    for k, qq in enumerate(ep["recipe"]["queries"]):
        a, b = q["lab_off"][k], q["lab_off"][k + 1]
        ok = q["lab_valid"][a:b]
        rows_q += [(idx[qq["stay"]], t) for t in q["lab_t"][a:b][ok]]
        y_q += list(q["lab_y"][a:b][ok])
    if len(y_c) < 20 or len(y_q) < 5:
        return None
    rc = fl.Rows(np.array([r[0] for r in rows_c]), np.array([r[1] for r in rows_c]))
    rq = fl.Rows(np.array([r[0] for r in rows_q]), np.array([r[1] for r in rows_q]))
    x = np.vstack([sampler.snapshot(prog, rc), sampler.snapshot(prog, rq)])
    y = np.r_[y_c, y_q].astype(np.float64)
    binary = prog.label in ("binary", "hosp_death", "tte", "multihorizon")
    if binary and (len(np.unique(y_c)) < 2 or len(np.unique(y_q)) < 2):
        return None
    if prog.label == "ordinal":
        binary = False
    _ok, score = informative(x, y, len(y_c), "binary" if binary else "regression", margin=0.0)
    return float(score)


def shift_loss(sampler, ep: dict, seed: int = 0) -> dict | None:
    """A cross-site episode's shift for a non-adaptive probe (Acceptance, "Sites and views": 0.05-0.20 AUROC): the
    probe's AUROC on the queries with a context of as many patients from the queries' own site (same program and
    cadence), minus its AUROC with the episode's context from the other site. None for other episodes or when either
    side cannot be scored."""
    from icutwin.factory.episodes import _program
    from icutwin.factory.tasks import informative

    r = ep["recipe"]
    if (r.get("site_variant") or {}).get("kind") != "cross_site":
        return None
    prog = _program(r["program"])
    if prog.label not in ("binary", "hosp_death"):
        return None
    cross = difficulty(sampler, ep)
    if cross is None:
        return None
    idx = sampler.index
    q = ep["query"]
    rows_q, y_q = [], []
    for k, qq in enumerate(r["queries"]):
        a, b = q["lab_off"][k], q["lab_off"][k + 1]
        ok = q["lab_valid"][a:b]
        rows_q += [(idx[qq["stay"]], t) for t in q["lab_t"][a:b][ok]]
        y_q += list(q["lab_y"][a:b][ok])
    rows, b = sampler.site_labels(prog, tuple(r["cadence"]), r["query_site"])
    qs = {idx[qq["stay"]] for qq in r["queries"]}
    cand = np.setdiff1d(np.unique(rows.stay[b.valid]), list(qs))
    if len(cand) < 2 or not r["context"]:
        return None
    pick = np.random.default_rng([seed, r["index"]]).choice(cand, size=min(len(r["context"]), len(cand)),
                                                            replace=False)
    m = np.isin(rows.stay, pick) & b.valid
    y_c = b.y[m].astype(np.float64)
    if len(y_c) < 20 or len(np.unique(y_c)) < 2 or len(np.unique(y_q)) < 2:
        return None
    rq = fl.Rows(np.array([x[0] for x in rows_q]), np.array([x[1] for x in rows_q]))
    x = np.vstack([sampler.snapshot(prog, rows.take(np.flatnonzero(m))), sampler.snapshot(prog, rq)])
    _ok, local = informative(x, np.r_[y_c, np.asarray(y_q, dtype=np.float64)], len(y_c), "binary", margin=0.0)
    return {"cross": float(cross), "local": float(local), "loss": float(local - cross)}


def embed(store, stays) -> np.ndarray:
    """Bag-of-channels embeddings of stays (events per channel group and per channel, log-scaled, L2-normalized)."""
    n_ch = len(store.channels)
    out = np.zeros((len(stays), n_ch), dtype=np.float32)
    for k, i in enumerate(stays):
        c = np.asarray(store.events(int(i))["c"], dtype=np.int64)
        np.add.at(out[k], c, 1.0)
    out = np.log1p(out)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


def context_vendi(store, stays) -> float:
    from icutwin.eval.diversity import vendi

    return vendi(embed(store, stays)) if len(stays) > 1 else float(len(stays))


def context_vendi_ratio(sampler, real_sampler, ep: dict, seed: int = 0) -> float | None:
    """An episode's context Vendi over that of as many real patients with a valid label for the same program at the
    same site (Acceptance, "Within episodes": >= 0.9); None without two context patients or enough real ones."""
    from icutwin.factory.episodes import _program

    r = ep["recipe"]
    nc = len(r["context"])
    if nc < 2 or r["site"] not in set(real_sampler.site_names):
        return None
    prog = _program(r["program"])
    rows, b = real_sampler.site_labels(prog, tuple(r["cadence"]), r["site"])
    cand = np.unique(rows.stay[b.valid])
    if len(cand) < nc:
        return None
    pick = np.random.default_rng([seed, r["index"]]).choice(cand, size=nc, replace=False)
    rv = context_vendi(real_sampler.store, pick)
    return context_vendi(sampler.store, [sampler.index[x] for x in r["context"]]) / rv if rv > 0 else None


def coverage(real_store, real_stays, twin_store, twin_stays, k: int = 5) -> dict:
    """Naeem density and coverage of real trajectories by twins (bag-of-channel embeddings in one channel space)."""
    from icutwin.eval.fidelity import prdc

    names = twin_store.channels.names
    if real_store.channels.names != names:
        raise ValueError("stores with different channel tables")
    return prdc(embed(real_store, real_stays), embed(twin_store, twin_stays), k=k)
