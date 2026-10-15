"""icutwin on the AutoDL host: a CPU check of goal-first sampling's registers on real tokens (vllm-env or .venv): the
vLLM processor's module imports, the goal plan covers every goal token, and an automaton with its obligations fed a
real stay preceded by that stay's own goal block (drawn goals that the stay keeps) is never left with an obligation
the stay breaks: no forbidden token is ever allowed-and-taken, and every required set is met by the end.
    python check_goalfirst.py <tokens root> [n stays]"""
import sys
from pathlib import Path


from icutwin.generate.constrain import Conditions, Obligations, generation_mask
from icutwin.generate.engine import goal_plan
from icutwin.tokenization.fsm import Automaton, compile_tables, mask_tables
from icutwin.tokenization.vocab import Vocab

try:
    import icutwin.generate.vllm_grammar  # noqa: F401  (vllm-env only)
    print("vllm_grammar imports")
except ImportError as e:
    print("vllm_grammar not importable here:", e)

tokens = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
v = Vocab.load(tokens / "vocab.json")
t = compile_tables(v)
m = mask_tables(t, len(v.tokens))
plan = goal_plan(v)
print("goal tokens", len(t.goal), "binding", len(set(plan.require_icu) | set(plan.forbid_icu) | set(plan.require_end)
                                                  | set(plan.forbid)))
for ds in (sys.argv[3:] or ["mimic", "aumc", "hirid"]):
    import pyarrow.parquet as pq

    tab = pq.read_table(tokens / ds / "held_out.parquet", columns=["goal_ids", "windows"]).slice(0, n)
    goals, wins = tab.column("goal_ids").to_pylist(), tab.column("windows").to_pylist()
    bad = taken = 0
    for g, w in zip(goals, wins):
        ids = list(w[0])
        if not g or v.id("[EOS]") not in ids:
            continue                                     # whole stays in one window only
        g = list(g)                                      # the stay encoded with its goal block
        if v.id("[GOAL]") not in g:
            continue
        block = g[g.index(v.id("[GOAL]")):g.index(v.id("[/GOAL]")) + 1]      # [GOAL] ... [/GOAL], every field
        tl = ids.index(v.id("[TL]"))
        seq = ids[:tl] + block + ids[tl:]
        a, obl = Automaton(t), Obligations(Conditions(draw_goals=True), plan)
        for pos, tok in enumerate(seq):
            if pos and a.phase not in ("start", "done"):
                mask = generation_mask(a, m, None, Conditions(draw_goals=True), obl, None)
                if not mask[tok]:
                    bad += 1
                    print(ds, "masked real token", v.tokens[tok], "at", pos, "phase", a.phase)
                    break
            if not a.step(tok):
                print(ds, "grammar rejects", a.violation)
                bad += 1
                break
            obl.feed(tok)
        else:
            taken += 1
            if obl.open_timeline() or obl.open_end():
                bad += 1
                print(ds, "obligation left open")
    print(ds, "stays checked", taken, "problems", bad)
