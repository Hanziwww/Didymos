"""Teacher-forced hazard calibration (round 3, ``docs/m7_refinement.md``, step 0a): whether the model's next-token
probabilities of an outcome, summed over the positions where it could come next, match how often it comes in real
held-out stays. Under teacher forcing a calibrated model's expected count equals the observed one (the summed
probabilities are the event's compensator). A model calibrated here but not in its own rollouts drifts as it
generates (compounding errors); one already off here is miscalibrated step by step.

Outcomes, from the ICU admission marker on: the first IMV, vasopressor and RRT event (``forecast.Reader.sets``, the
engine's obligation sets: new onset, so a stay is at risk until its first event and a stay with one before the
admission is left out), the ICU discharge (``Reader.dis``) and the ICU discharge 'dead'. Each prediction is placed in
the ICU day of the token it predicts (hours since the admission marker, interval tokens at their median gaps: 1, 2,
3-7, 8+; ``after_24h`` sums days 2 on, the utility tasks' landmark). Probabilities are the model's own: generation
renormalizes them over the grammar's tokens, which on real stays takes 0.1-0.9% of the mass (round 2's distortion
monitor). Aggregates only.

    python -m icutwin.eval.calibration --tokens <tokens root> --model <hf dir> --out <report.json> [--n 2000]
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

DAYS = (("1", 0.0, 24.0), ("2", 24.0, 48.0), ("3-7", 48.0, 168.0), ("8+", 168.0, np.inf))
TREATMENTS = ("imv", "vasopressor", "rrt")
OUTCOMES = (*TREATMENTS, "icu_discharge", "icu_death")


@dataclass
class Targets:
    """The token sets the measure reads (``targets(reader)``; a toy vocabulary in the tests)."""
    adm: set
    dis: set
    dead: set
    end: int
    sets: dict                                   # treatment -> token set
    minutes: dict = field(default_factory=dict)  # interval token -> minutes (median gap)
    clock: set = field(default_factory=set)      # round 3: clock tokens (the sampler writes them; nothing else can come)


def targets(r) -> Targets:
    dead = {i for i in r.dis if (r.v.meta[i].get("code") or "").endswith("//dead")}
    clock = {i for i, m in enumerate(r.v.meta) if m.get("cls") == "clock"}
    return Targets(set(r.adm), set(r.dis), dead, r.end, dict(r.sets), dict(r.minutes), clock)


def _hours(ids: list, tg: Targets) -> np.ndarray:
    """Hours since the ICU admission marker at each token (NaN before it)."""
    t, start = 0.0, None
    out = np.full(len(ids), np.nan)
    for k, tok in enumerate(ids):
        if start is None and tok in tg.adm:
            start = t
        if start is not None:
            out[k] = (t - start) / 60
        t += tg.minutes.get(tok, 0.0)
    return out


def _day(h: float) -> int | None:
    if not np.isfinite(h):
        return None
    for j, (_name, lo, hi) in enumerate(DAYS):
        if lo <= h < hi:
            return j
    return None


def risk_span(ids, tg: Targets) -> tuple | None:
    """(admission marker position, ICU discharge position or None, the last predicted token inside the ICU)."""
    ids = [int(x) for x in ids]
    n = len(ids)
    adm = next((k for k, t in enumerate(ids) if t in tg.adm), None)
    if adm is None:
        return None
    end = ids.index(tg.end) if tg.end in ids else n
    dis = next((k for k in range(adm + 1, end) if ids[k] in tg.dis), None)
    return adm, dis, (dis if dis is not None else min(end, n - 1))


def stay_hazards(ids, p: dict, tg: Targets, w: np.ndarray | None = None) -> dict:
    """Per outcome: (expected, observed, at risk) by ICU day for one stay. ``p[k][t]``: the model's probability that
    the token after position ``t`` is in outcome k's set. A stay is at risk from the admission marker to its first
    event, its ICU discharge or its timeline's end, whichever comes first: the prediction of every token up to and
    including that one counts (the event could have come instead of it). ``w``: per position weights of a sample of
    the positions (1 / inclusion probability where picked, 0 elsewhere: a Horvitz-Thompson estimate of the sum)."""
    ids = [int(x) for x in ids]
    span = risk_span(ids, tg)
    if span is None:
        return {}
    adm, dis, limit = span
    hours = _hours(ids, tg)
    out = {}
    for k in OUTCOMES:
        if k in TREATMENTS:
            s = tg.sets[k]
            if any(t in s for t in ids[:adm]):
                continue                                         # not new onset in the ICU
            first = next((j for j in range(adm + 1, limit) if ids[j] in s), None)
        elif k == "icu_discharge":
            first = dis
        else:
            first = dis if dis is not None and ids[dis] in tg.dead else None
        last = first if first is not None else limit
        e, o, risk = np.zeros(len(DAYS)), np.zeros(len(DAYS)), np.zeros(len(DAYS))
        for t in range(adm, last):
            if ids[t + 1] in tg.clock:              # the sampler writes the clock block here
                continue
            j = _day(hours[t + 1])
            if j is not None:
                if w is None:
                    e[j] += p[k][t]
                elif w[t] > 0:
                    e[j] += p[k][t] * w[t]
                risk[j] = 1
        if first is not None and _day(hours[first]) is not None:
            o[_day(hours[first])] += 1
        out[k] = (e, o, risk)
    return out


def _oe(o: float, e: float) -> dict:
    """Observed over expected with the exact (Garwood) 95% interval of the observed count."""
    from scipy.stats import chi2

    lo = chi2.ppf(0.025, 2 * o) / 2 if o > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * o + 2) / 2
    return {"observed": float(o), "expected": float(e), "o_e": float(o / e) if e > 0 else None,
            "o_e_ci": [float(lo / e), float(hi / e)] if e > 0 else None}


class Accumulator:
    def __init__(self):
        self.e = {k: np.zeros(len(DAYS)) for k in OUTCOMES}
        self.o = {k: np.zeros(len(DAYS)) for k in OUTCOMES}
        self.risk = {k: np.zeros(len(DAYS)) for k in OUTCOMES}
        self.stays = 0

    def add(self, res: dict) -> None:
        if res:
            self.stays += 1
        for k, (e, o, r) in res.items():
            self.e[k] += e
            self.o[k] += o
            self.risk[k] += r

    def summary(self) -> dict:
        out = {"stays": self.stays}
        for k in OUTCOMES:
            row = {name: {**_oe(self.o[k][j], self.e[k][j]), "at_risk": int(self.risk[k][j])}
                   for j, (name, _lo, _hi) in enumerate(DAYS)}
            row["after_24h"] = _oe(self.o[k][1:].sum(), self.e[k][1:].sum())
            row["all"] = _oe(self.o[k].sum(), self.e[k].sum())
            out[k] = row
        return out


def set_probs(model, ids, index: dict, device) -> dict:
    """Per position t (0..len-2), the model's probability that token t+1 is in each set (``index``: name -> tensor
    of token ids), from one teacher-forced forward pass (bf16 autocast as training)."""
    import torch

    from icutwin.eval.authenticity import _autocast

    with torch.no_grad():
        x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)[None]
        with _autocast(device):
            logits = model(input_ids=x).logits[0, :-1].float()
        lse = torch.logsumexp(logits, -1)
        return {k: torch.exp(torch.logsumexp(logits[:, ix], -1) - lse).cpu().numpy() for k, ix in index.items()}


POSITIONS = (("0-2k", 0, 2048), ("2k-8k", 2048, 8192), ("8k-16k", 8192, 16384), ("16k+", 16384, 10**9))


def at_risk(ids, tg: Targets) -> dict:
    """Per treatment, (positions at risk, position of the first event or None) as ``stay_hazards`` counts them."""
    ids = [int(x) for x in ids]
    span = risk_span(ids, tg)
    if span is None:
        return {}
    adm, _dis, limit = span
    out = {}
    for k in TREATMENTS:
        s = tg.sets[k]
        if any(t in s for t in ids[:adm]):
            continue
        first = next((j for j in range(adm + 1, limit) if ids[j] in s), None)
        pos = np.arange(adm, first if first is not None else limit)
        if tg.clock:                                 # not where the sampler writes the clock block
            pos = pos[~np.isin(np.asarray(ids)[pos + 1], list(tg.clock))]
        out[k] = (pos, first)
    return out


class TokenQA:
    """The treatment QA (round 3, the diagnosis' per-step reading): per treatment, the expected first events by token
    of its set (which tokens carry the model's excess) against the tokens the first events were, and observed over
    expected by the position of the predicted token (a long-context effect, as against the ICU day)."""

    def __init__(self, tg: Targets, v):
        self.tg, self.v = tg, v
        self.tokens = {k: sorted(tg.sets[k]) for k in TREATMENTS}
        self.e_tok = {k: np.zeros(len(t)) for k, t in self.tokens.items()}
        self.o_tok = {k: Counter() for k in TREATMENTS}
        self.e_pos = {k: np.zeros(len(POSITIONS)) for k in TREATMENTS}
        self.o_pos = {k: np.zeros(len(POSITIONS)) for k in TREATMENTS}
        # ICU day x position: does the excess follow time in the ICU or the length of the context?
        self.e_dp = {k: np.zeros((len(DAYS), len(POSITIONS))) for k in TREATMENTS}
        self.o_dp = {k: np.zeros((len(DAYS), len(POSITIONS))) for k in TREATMENTS}

    def add(self, model, ids, device) -> None:
        import torch

        from icutwin.eval.authenticity import _autocast

        risk = at_risk(ids, self.tg)
        if not risk:
            return
        hrs = _hours([int(x) for x in ids], self.tg)
        day_edges = np.array([d[2] for d in DAYS[:-1]])
        pos_edges = np.array([p[2] for p in POSITIONS[:-1]])
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)[None]
            with _autocast(device):
                logits = model(input_ids=x).logits[0, :-1]
            for k, (pos, first) in risk.items():
                if not len(pos):
                    continue
                ix = torch.as_tensor(self.tokens[k], device=device)
                rows = torch.as_tensor(pos, device=device)
                p = torch.softmax(logits[rows].float(), -1)[:, ix]               # positions x set tokens
                self.e_tok[k] += p.sum(0).cpu().numpy()
                per_pos = p.sum(1).cpu().numpy()
                bucket = np.searchsorted([b[2] for b in POSITIONS], pos + 1, side="right")
                np.add.at(self.e_pos[k], np.minimum(bucket, len(POSITIONS) - 1), per_pos)
                d = np.searchsorted(day_edges, hrs[pos + 1], side="right")
                np.add.at(self.e_dp[k], (d, np.searchsorted(pos_edges, pos + 1, side="right")), per_pos)
                if first is not None:
                    self.o_tok[k][int(ids[first])] += 1
                    self.o_pos[k][min(int(np.searchsorted([b[2] for b in POSITIONS], first, side="right")),
                                      len(POSITIONS) - 1)] += 1
                    self.o_dp[k][int(np.searchsorted(day_edges, hrs[first], side="right")),
                                 int(np.searchsorted(pos_edges, first, side="right"))] += 1

    def summary(self, top: int = 8) -> dict:
        out = {}
        for k in TREATMENTS:
            order = np.argsort(-self.e_tok[k])[:top]
            out[k] = {"tokens": [{"token": self.v.tokens[self.tokens[k][i]], "expected": float(self.e_tok[k][i]),
                                  "observed": int(self.o_tok[k][self.tokens[k][i]])} for i in order],
                      "observed_tokens_not_listed": int(sum(c for t, c in self.o_tok[k].items()
                                                            if t not in {self.tokens[k][i] for i in order})),
                      "by_position": {name: _oe(self.o_pos[k][j], self.e_pos[k][j])
                                      for j, (name, _lo, _hi) in enumerate(POSITIONS)},
                      "by_day_position": {dn: {pn: _oe(self.o_dp[k][i, j], self.e_dp[k][i, j])
                                               for j, (pn, _a, _b) in enumerate(POSITIONS)}
                                          for i, (dn, _lo, _hi) in enumerate(DAYS)}}
        return out


def masked_probs(model, ids, index: dict, tables, masks, support, device, pick: np.ndarray) -> dict:
    """Per picked position t: the probability that token t+1 is in each set under the generation mask (the grammar
    and the dataset's support, as the sampler draws; renormalized over the allowed tokens). Returns arrays over all
    positions, NaN where not picked."""
    import torch

    from icutwin.eval.authenticity import _autocast
    from icutwin.generate.constrain import NONE, generation_mask
    from icutwin.tokenization.fsm import Automaton

    ids = [int(x) for x in ids]
    want, rows, mk = set(int(t) for t in pick), [], []
    a, dtok = Automaton(tables), None
    for pos, tok in enumerate(ids):
        if pos - 1 in want and a.phase not in ("start", "done") and a.violation is None:
            sup = support.mask(dtok) if support is not None and dtok is not None else None
            rows.append(pos - 1)
            mk.append(generation_mask(a, masks, None, NONE, None, sup))
        if a.phase == "prefix" and a.i == 0:
            dtok = tok
        if not a.step(tok):
            break
    out = {k: np.full(len(ids) - 1, np.nan) for k in index}
    if not rows:
        return out
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=device)[None]
        with _autocast(device):
            logits = model(input_ids=x).logits[0]
        r = torch.as_tensor(rows, device=device)
        allowed = torch.as_tensor(np.stack(mk), device=device)
        probs = torch.softmax(logits[r, :allowed.shape[1]].float(), -1) * allowed
        den = probs.sum(-1).clamp_min(1e-30)
        for k, ix in index.items():
            out[k][rows] = (probs[:, ix].sum(-1) / den).cpu().numpy()
    return out


def main(argv: list[str] | None = None) -> None:
    """Raw: the model's own probabilities at every position at risk. ``--masked``: as the sampler draws (grammar and
    support), at a ``--share`` of the positions at risk picked at random (Horvitz-Thompson: each weighted by
    1 / share), beside the raw ones."""
    import argparse
    import json
    from pathlib import Path

    import torch

    from icutwin.eval import privacy as pv
    from icutwin.eval.forecast import DATASETS, Reader
    from icutwin.tokenization.vocab import Vocab

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--share", type=float, default=0.25)
    ap.add_argument("--qa", action="store_true", help="the treatment QA (TokenQA) instead of the calibration")
    a = ap.parse_args(argv)
    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    tg = targets(Reader(v))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv.load_model(Path(a.model), dev)
    tensor = lambda s: torch.as_tensor(sorted(s), dtype=torch.long, device=dev)  # noqa: E731
    index = {**{k: tensor(tg.sets[k]) for k in TREATMENTS}, "icu_discharge": tensor(tg.dis), "icu_death": tensor(tg.dead)}
    if a.masked:
        from icutwin.generate.support import Support
        from icutwin.tokenization.fsm import compile_tables, mask_tables

        tables = compile_tables(v)
        masks = mask_tables(tables, len(v.tokens))
        support = Support.load(v, tokens / "support.json") if (tokens / "support.json").exists() else None
    rng = np.random.default_rng(a.seed)
    report = {}
    if a.qa:
        for ds in a.datasets:
            held = pv.first_windows(tokens, ds, "held_out")
            qa = TokenQA(tg, v)
            for i in np.sort(rng.choice(len(held), size=min(a.n, len(held)), replace=False)):
                qa.add(model, held[i], dev)
            report[ds] = qa.summary()
            print(ds, json.dumps({k: {n: x["o_e"] for n, x in report[ds][k]["by_position"].items()} for k in TREATMENTS}),
                  flush=True)
            Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        return
    for ds in a.datasets:
        held = pv.first_windows(tokens, ds, "held_out")
        acc, acc_m = Accumulator(), Accumulator()
        for i in np.sort(rng.choice(len(held), size=min(a.n, len(held)), replace=False)):
            acc.add(stay_hazards(held[i], set_probs(model, held[i], index, dev), tg))
            span = risk_span(held[i], tg)
            if a.masked and span is not None:
                at_risk = np.arange(span[0], span[2])
                if tg.clock:
                    at_risk = at_risk[~np.isin(np.asarray(held[i])[at_risk + 1], list(tg.clock))]
                pick = at_risk[rng.random(len(at_risk)) < a.share]
                w = np.zeros(len(held[i]) - 1)
                w[pick] = 1 / a.share
                acc_m.add(stay_hazards(held[i], masked_probs(model, held[i], index, tables, masks, support, dev, pick),
                                       tg, w))
        report[ds] = acc.summary()
        if a.masked:
            report[ds]["masked"] = {**acc_m.summary(), "share": a.share}
        print(ds, json.dumps({k: report[ds][k]["after_24h"]["o_e"] for k in OUTCOMES}),
              json.dumps({k: report[ds]["masked"][k]["after_24h"]["o_e"] for k in OUTCOMES}) if a.masked else "",
              flush=True)
        Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
