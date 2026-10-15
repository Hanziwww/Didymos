"""The factory's channel vocabulary (plan Phase F, Factory design 1): one channel per event code of the token
vocabulary, except that a continuous infusion's actions (start, going, dose change, verify, other, stop) fold into
one channel per infusion key (drug, unit, route) and travel as a flag. Each channel has:
* its domain (the code's first part) and group (``configs/factory/channel_groups.yaml``);
* its kind: ``numeric`` (decile bins: ``edges``), ``flag`` (presence), ``text`` (a text value), ``na`` (none), or
  ``state`` (``STATE_CHANNELS``: CRRT and MCS active, the tokenizer's state rule, as intervals only);
* whether it is an input (the "outcome" group never is) and physiological (a target of threshold, value and
  forecast programs).

Channel ids are positions in a list sorted by name, so the vocabulary alone fixes them. Normalization statistics
(``stats``: median, p05, p95 per numeric channel) come from a store's training twins (``store.channel_stats``) and
are saved with that store, never in the repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from icutwin.tokenization.vocab import Vocab, infusion_key

ACTIONS = ("start", "going", "dose_change", "verify", "other", "stop")
STATE_CHANNELS = ("STATE//CRRT//active", "STATE//MCS//active")
GROUPS_YAML = Path(__file__).resolve().parents[1] / "configs" / "factory" / "channel_groups.yaml"


def load_groups(path: Path = GROUPS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass
class Channels:
    names: list                       # channel id -> name (a code, or an infusion key)
    domain: list
    group: list
    kind: list                        # numeric | flag | text | na
    edges: list                       # numeric channels: the code's decile edges (an infusion: its key's), else []
    physiological: list
    stats: dict = field(default_factory=dict)      # name -> {"median", "p05", "p95"} (a store's training twins)

    def __post_init__(self):
        self.index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def of(self, code: str) -> tuple[int, int]:
        """(channel id, action: 1 + its position in ``ACTIONS`` for an infusion's code, else 0) of an event code;
        (-1, 0) for a code outside the vocabulary."""
        k = infusion_key(code)
        if k is not None and k[0] in self.index:
            return self.index[k[0]], 1 + ACTIONS.index(k[1]) if k[1] in ACTIONS else 0
        return self.index.get(code, -1), 0

    def is_input(self, cid: int) -> bool:
        return self.group[cid] not in ("outcome",)

    @classmethod
    def from_vocab(cls, v: Vocab, groups: dict | None = None) -> Channels:
        groups = groups or load_groups()
        dom2grp = {d: g for g, ds in groups["groups"].items() for d in ds}
        phys = set(groups["physiological"])
        rows: dict = {}
        for code, info in v.codes.items():
            k = infusion_key(code)
            name = k[0] if k is not None else code
            dom = name.split("//", 1)[0]
            kind = info.get("kind") or "na"
            kind = {"bin": "numeric", "oov": "na", "system": "na"}.get(kind, kind)
            edges = list(info.get("edges") or [])
            if k is not None and k[0] in v.state_keys:
                edges = list(v.state_keys[k[0]].get("edges") or edges)
            prev = rows.get(name)
            if prev is not None:                       # an infusion key's actions: numeric if any action has a rate
                kind = "numeric" if "numeric" in (kind, prev[1]) else prev[1]
                edges = edges or prev[2]
            rows[name] = (dom, kind, edges)
        for name in STATE_CHANNELS:                    # the tokenizer's 'charted within 24 hours' states
            rows[name] = (name.split("//")[1], "state", [])
        names = sorted(rows)
        grp = [dom2grp.get(rows[n][0], "other") for n in names]
        return cls(names, [rows[n][0] for n in names], grp, [rows[n][1] for n in names], [rows[n][2] for n in names],
                   [g in phys and rows[n][1] == "numeric" for n, g in zip(names, grp)])

    def to_json(self) -> dict:
        return {"names": self.names, "domain": self.domain, "group": self.group, "kind": self.kind,
                "edges": self.edges, "physiological": self.physiological, "stats": self.stats}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Channels:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["names"], d["domain"], d["group"], d["kind"], d["edges"], d["physiological"], d.get("stats", {}))
