"""Round 3's clock blocks (spec ``clock``) on the 2% subset's stored tokens, as the full data gets them (the
vocabulary extended, the shards through ``insert_clock``; skipped without the subset): ids unchanged; the blocks
removable, idempotent and accepted; each block where the ICU day changes, with the day and the treatment states an
independent replay of the tokens gives; the mask forces the block; the sampler writes it; the loss skips it; and
every state with a block due still completes within the reserve."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from icutwin.tokenization.spec import load_spec

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


@pytest.fixture(scope="module")
def setup():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.eval import privacy as pv
    from icutwin.tokenization.fsm import compile_tables, insert_clock, mask_tables
    from icutwin.tokenization.vocab import Vocab, extend_clock

    old = Vocab.load(T / "vocab.json")
    v = extend_clock(old)
    t = compile_tables(v)
    seqs = []
    for ds in ("aumc", "hirid", "sicdb", "mimic", "eicu", "nwicu", "inspire"):
        seqs += [[int(x) for x in s] for s in pv.first_windows(T, ds, "held_out")[:20]]
    return old, v, t, mask_tables(t, len(v.tokens)), seqs, [insert_clock(s, t) for s in seqs]


def test_harness_validity_reads_twins_as_written(setup, tmp_path) -> None:
    """The harness checks grammar validity on each twin as the sampler wrote it (with its clock
    blocks); the measures' bare reading of a twin with a block fails the clock grammar."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.eval.quality import grammar_validity
    from icutwin.tokenization.fsm import accepts

    _old, v, t, _m, _seqs, new = setup
    new = [s for s in new if s[-1] == v.id("[EOS]")]          # whole stays, as twins are (cut windows continue)
    clock = {i for i, m in enumerate(v.meta) if m.get("cls") == "clock"}
    with_block = [s for s in new if clock & set(s)]
    assert with_block
    pq.write_table(pa.table({"ids": pa.array(new, type=pa.list_(pa.int32()))}), tmp_path / "x.parquet")
    assert grammar_validity(tmp_path / "x.parquet", t) == 1.0
    assert not any(accepts(t, [x for x in s if x not in clock])[0] for s in with_block)


def test_extension_keeps_every_id(setup) -> None:
    from icutwin.tokenization.vocab import extend_clock

    old, v, t, _m, _seqs, _new = setup
    assert v.tokens[:len(old.tokens)] == old.tokens and v.meta[:len(old.meta)] == old.meta
    assert all(m["cls"] == "clock" for m in v.meta[len(old.tokens):]) and extend_clock(v) is v
    c = load_spec()["clock"]
    assert len(v.tokens) - len(old.tokens) == 1 + (c["max_day"] - 1) + len(c["treatments"]) * len(c["states"])
    assert t.clock is not None and len(t.clock.days) == c["max_day"] - 1


def _replay(ids: list, v) -> list:
    """Independent of the automaton: per block, (the ICU day at the interval before it, the states from the
    tokens before it)."""
    c = load_spec()["clock"]
    sk = v.interval_sketches["*"]
    minutes = {i: (sk[m["interval"]][len(sk[m["interval"]]) // 2] if sk[m["interval"]] else 0.0)
               for i, m in enumerate(v.meta) if m.get("cls") == "interval"}
    t, adm, device, running, rrt_t, ever, out = 0.0, None, None, set(), None, set(), []
    for pos, tok in enumerate(ids):
        m = v.meta[tok]
        if m.get("role") == "marker":
            states = {}
            for k in c["treatments"]:
                if k == "imv":
                    active = device == c["imv_device"]
                elif k == "vasopressor":
                    active = any(key.split("//")[1] in c["vasoactives"] for key in running)
                else:
                    active = rrt_t is not None and t - rrt_t <= c["rrt_active_hours"] * 60
                states[k] = "active" if active else "ended" if k in ever else "never"
            out.append((int((t - adm) // 1440) + 1, states))
            continue
        code = (m.get("code") or "") if m.get("cls") == "event" else ""
        if code == "ICU//admission" and adm is None:
            adm = t
        if code.startswith(c["device_prefix"]):
            device = code[len(c["device_prefix"]):]
            if device == c["imv_device"]:
                ever.add("imv")
        if "key" in m and m.get("cls") == "event":
            if m["action"] in ("start", "going"):
                running.add(m["key"])
                if m["key"].split("//")[1] in c["vasoactives"]:
                    ever.add("vasopressor")
            elif m["action"] == "stop":
                running.discard(m["key"])
        if code.split("//", 1)[0] in c["rrt_domains"]:
            rrt_t = t
            ever.add("rrt")
        t += minutes.get(tok, 0.0)
    return out


def test_blocks_are_where_and_what_the_tokens_say(setup) -> None:
    from icutwin.tokenization.fsm import accepts, insert_clock
    from icutwin.tokenization.parser import parse

    _old, v, t, _m, seqs, new = setup
    c, blocks = t.clock, 0
    for s, x in zip(seqs, new):
        assert [tok for tok in x if tok not in c.tokens] == s                      # removable
        assert insert_clock(x, t) == x and accepts(t, x, whole=x[-1] == v.id("[EOS]"))[0]
        p = parse(x, v, whole=x[-1] == v.id("[EOS]"))
        marks = [i for i, tok in enumerate(x) if tok == c.marker]
        blocks += len(marks)
        assert len(p.clock) == len(marks)
        for i in marks:                                                             # after an interval, whole
            assert x[i - 1] in t.interval and x[i + 1] in set(c.days.values())
        replay = _replay(x, v)
        for (day, states), (pday, pstates) in zip(replay, p.clock):
            assert pday == min(day, c.max_day) and pstates == states, (day, states, pday, pstates)
        days = [d for d, _s in replay]
        assert days == sorted(set(days)) and all(d >= 2 for d in days)              # one block per new day
    assert blocks > 50


def test_a_block_after_each_repeated_week(setup) -> None:
    """Two weeks without an event: each 7-day token opens a new day, so each gets its block (found on the full
    SICdb tokens: the parser read one block per run of intervals)."""
    from icutwin.tokenization.fsm import accepts, insert_clock
    from icutwin.tokenization.parser import parse

    _old, v, t, _m, seqs, _new = setup
    s = next(x for x in seqs if any(tok in t.clock.adm for tok in x) and x[-1] == v.id("[EOS]"))
    adm = next(i for i, tok in enumerate(s) if tok in t.clock.adm)
    k = next(i for i in range(adm + 1, len(s)) if s[i] in t.interval and s[i] != t.week)
    gap = s[:k] + [t.week, t.week] + s[k + 1:]                   # the gap after the admission becomes two weeks
    x = insert_clock(gap, t)
    assert accepts(t, x)[0] and [tok for tok in x if tok not in t.clock.tokens] == gap
    weeks = [i for i, tok in enumerate(x) if tok == t.week]
    assert all(x[i + 1] == t.clock.marker for i in weeks[:2])    # a block after each of the two weeks
    assert len(parse(x, v).clock) == x.count(t.clock.marker)


def test_the_mask_forces_the_block_and_the_loss_skips_it(setup) -> None:
    from icutwin.model.data import IGNORE, labels_for, layout
    from icutwin.tokenization.fsm import Automaton, allowed

    _old, v, t, m, _seqs, new = setup
    lay = layout(v)
    for x in new[:30]:
        a = Automaton(t)
        for tok in x:
            ok = allowed(a, m)
            assert ok[tok]
            if a.pending:
                assert ok.sum() == 1
            a.step(tok)
        labels = labels_for(np.asarray(x), lay)
        assert (labels[lay.clock[np.asarray(x)]] == IGNORE).all()


def test_states_with_a_block_due_complete_within_the_reserve(setup) -> None:
    from icutwin.generate import support as gs
    from icutwin.generate.constrain import END_RESERVE, NONE, generation_mask
    from icutwin.tokenization.fsm import Automaton

    _old, v, t, m, _seqs, new = setup
    sup = gs.Support(v, gs.compute(T, ["aumc", "hirid", "sicdb", "mimic", "eicu", "nwicu", "inspire"]))
    checked = 0
    for x in new:
        a, dtok = Automaton(t), None
        for tok in x:
            if a.phase == "prefix" and a.i == 0:
                dtok = tok
            a.step(tok)
            if a.pending and a.pending[0] == t.clock.marker:                       # a block just became due
                b, left = copy.deepcopy(a), END_RESERVE
                while b.phase != "done":
                    ok = np.flatnonzero(generation_mask(b, m, left, NONE, None, sup.mask(dtok)))
                    assert len(ok), b.phase
                    pick = int(ok[-1]) if v.id("[EOS]") not in set(ok) else v.id("[EOS]")
                    assert b.step(pick)
                    left -= 1
                    assert left >= 0
                checked += 1
    assert checked > 20


def test_the_sampler_writes_the_blocks(setup) -> None:
    import torch
    from transformers import GenerationConfig, LogitsProcessorList, Qwen3ForCausalLM

    from icutwin.generate.constrain import GrammarProcessor
    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.fsm import accepts, insert_clock

    _old, v, t, m, _seqs, new = setup
    x = next(s for s in new if t.clock.marker in s)
    adm = next(i for i, tok in enumerate(x) if tok in t.clock.adm)
    prompt = x[:adm + 1]                                         # the stay up to its ICU admission marker
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"))).eval()
    with torch.no_grad():
        model.lm_head.weight[sorted(t.interval)] += 0.5         # a tiny model that moves time along
    proc = GrammarProcessor(t, m, v.id("[PAD]"), max_length=len(prompt) + 400)
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=400,
                           eos_token_id=v.id("[EOS]"), pad_token_id=v.id("[PAD]"))
    ids = torch.tensor([prompt] * 6)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                             logits_processor=LogitsProcessorList([proc]))
    written = 0
    for row in out.tolist():
        row = [tok for tok in row if tok != v.id("[PAD]")]
        assert accepts(t, row, whole=row[-1] == v.id("[EOS]"))[0] and insert_clock(row, t) == row
        written += row.count(t.clock.marker)
    assert written > 0 and proc.violations == 0
