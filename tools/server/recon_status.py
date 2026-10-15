"""icutwin on the AutoDL host: the reconciliation status per dataset from the parts in qa/full/reconcile_parts
(aggregates: community events, paired, explained and unexplained rows per side, stream keys)."""
import json
import sys
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin/data/qa/full/reconcile_parts")
for ds in sys.argv[1:] or ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]:
    p = R / ds / "reconciliation.json"
    if not p.exists():
        print(ds, "no part")
        continue
    rep = [r for r in json.loads(p.read_text(encoding="utf-8")) if r["dataset"] == ds][0]
    tot = {"community": 0, "paired": 0, "community_out_of_scope": 0, "c_explained": 0, "c_unexplained": 0, "ours": 0,
           "o_explained": 0, "o_unexplained": 0}
    for name, m in rep["tables"].items():
        if "skipped" in m:
            continue
        tot["community"] += m["community"]
        tot["ours"] += m["ours"]
        tot["paired"] += m["paired"]
        tot["community_out_of_scope"] += m["community_out_of_scope"]
        tot["c_explained"] += sum(m["only_community_explained"].values())
        tot["o_explained"] += sum(m["only_ours_explained"].values())
        tot["c_unexplained"] += m["only_community_unexplained"]
        tot["o_unexplained"] += m["only_ours_unexplained"]
        if m["only_community_unexplained"] or m["only_ours_unexplained"]:
            print(f"  {ds} {name}: unexplained community {m['only_community_unexplained']}, ours {m['only_ours_unexplained']}")
    st = {}
    for name, m in (rep.get("streams") or {}).items():
        if "skipped" in m:
            continue
        st[name] = {k: m[k] for k in ("keys", "keys_agree", "explained", "unexplained")}
    print(ds, json.dumps(tot), json.dumps(st))
