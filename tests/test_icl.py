"""The in-context model (plan Phase G, step G2): batches, causality, invariances, heads, robustness and a tiny overfit,
on a store built from real held-out token sequences standing in for twins (skipped without them). CPU, float32."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from icutwin.factory import contracts, episodes, store as fst
from icutwin.icl import batch as B, heads as H, infer, model as M

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"
ATOL = 1e-5


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "eicu" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.factory import pool, programs as fpr
    from icutwin.factory.trajectories import static_vocab
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu", "aumc"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 60).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "train"} for r in t]
    root = tmp_path_factory.mktemp("icl")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=90)
    st = fst.Store(root / "s")
    st.channels.stats = fst.channel_stats(st)
    cat = fpr.catalog(st.channels)
    progs = cat + pool.fc_candidates(st.channels, 6, 0) + pool.mh_candidates(cat, 6, 0)
    s = episodes.Sampler(st, v, stage=2, seed=3, programs=progs)
    eps, kinds = [], {}
    for i in range(600):
        try:
            ep = s.episode(i)
        except episodes.NoEpisode:
            continue
        lab = ep["task"]["label"]
        if kinds.get(lab, 0) < 3:
            kinds[lab] = kinds.get(lab, 0) + 1
            eps.append(ep)
        if len(kinds) >= 7 and min(kinds.values()) >= 3:
            break
    n_static = len(static_vocab(v))
    torch.manual_seed(0)
    model = M.TrajPFN(M.config("tiny", len(st.channels.names), n_static)).eval()
    return {"eps": eps, "names": st.channels.names, "n_static": n_static, "model": model}


def _builder(setup, train=False, max_rows=None):
    return B.Builder(setup["names"], setup["n_static"], train=train, max_rows=max_rows,
                     d_id=setup["model"].cfg.d_id)


def _run(setup, eps, **kw):
    return infer.predict(setup["model"], _builder(setup, **kw), eps)


def _rows(res, e: int) -> tuple[np.ndarray, torch.Tensor]:
    """(prediction times, final states) of episode ``e``'s test rows."""
    m = res["batch"]["te_ep"] == e
    return res["batch"]["te_t"][m].numpy(), res["z"][m]


def _queries_unthinned(ep) -> bool:
    return not (np.asarray(ep["query"]["ev_f"]) & 32).any()


def test_label_types_and_batch_layout(setup) -> None:
    eps = setup["eps"]
    assert {e["task"]["label"] for e in eps} >= {"binary", "ordinal", "regression", "tte", "forecast", "multihorizon"}
    b = _builder(setup, train=True, max_rows=8)(eps)
    for ep in eps:
        assert not contracts.episode_failures(ep)
    # every row's state is the patch ending exactly at its time; no token after its patch's end
    assert np.allclose(b["patch_end_h"][b["te_patch"]], b["te_t"] / 60.0)
    assert (b["tok_dt"] >= 0).all()
    # a query's last patch ends at its last prediction time
    q = np.flatnonzero(b["pat_role"] == 1)
    off = b["pat_off"]
    last_ends = b["patch_end_h"][off[q + 1] - 1]
    by_pat = {}
    for p, t in zip(b["patch_pat"][b["te_patch"]], b["te_t"], strict=True):
        by_pat[p] = max(by_pat.get(p, -np.inf), t / 60.0)
    assert np.allclose(last_ends, [by_pat[p] for p in q])
    # at most 8 train rows per context patient, drawn among its label times
    per = np.bincount(b["patch_pat"][b["tr_patch"]])
    assert per.max() <= 8


def test_output_at_t_equals_the_episode_cut_at_t(setup) -> None:
    """Causality, by definition: a query's output at t_k equals its output when the episode is cut at t_k."""
    checked = 0
    for ep in setup["eps"]:
        if not _queries_unthinned(ep):
            continue
        times = ep["recipe"]["queries"][0]["times"]
        if len(times) < 2:
            continue
        tk = int(times[len(times) // 2 - 1])
        full = _run(setup, [ep])
        cut = _run(setup, [infer.truncate_query(ep, 0, tk)])
        tf, zf = _rows(full, 0)
        tc, zc = _rows(cut, 0)
        q_full = full["batch"]["patch_pat"][full["batch"]["te_patch"]]
        first_query = q_full[0]
        mask_f = (q_full == first_query) & (torch.as_tensor(tf) <= tk)
        q_cut = cut["batch"]["patch_pat"][cut["batch"]["te_patch"]]
        mask_c = q_cut == q_cut[0]
        torch.testing.assert_close(zf[mask_f], zc[mask_c], atol=ATOL, rtol=0)
        checked += 1
    assert checked >= 5


def test_a_state_changing_exactly_at_t_is_not_yet_known_at_t(setup) -> None:
    """A state that starts (or stops) exactly at t_k: the factory's query cut at t_k leaves the start out and clips
    the stop, so the full episode's output at t_k must not see either."""
    checked = 0
    for ep in setup["eps"]:
        q = ep["query"]
        a, b = int(q["iv_off"][0]), int(q["iv_off"][1])
        times = ep["recipe"]["queries"][0]["times"]
        if b - a < 2 or len(times) < 2 or not _queries_unthinned(ep):
            continue
        tk = int(times[0])
        ep = copy.deepcopy(ep)
        q = ep["query"]
        iv_s, iv_e = np.asarray(q["iv_s"]).copy(), np.asarray(q["iv_e"]).copy()
        iv_s[a] = tk                                    # one state starts exactly at t_k ...
        iv_e[a] = max(int(iv_e[a]), tk + 60)
        iv_s[a + 1] = min(int(iv_s[a + 1]), tk - 30)    # ... another stops exactly there
        iv_e[a + 1] = tk
        q["iv_s"], q["iv_e"] = iv_s.astype(np.int32), iv_e.astype(np.int32)
        full = _run(setup, [ep])
        cut = _run(setup, [infer.truncate_query(ep, 0, tk)])
        pat = full["batch"]["patch_pat"][full["batch"]["te_patch"]]
        tf, zf = _rows(full, 0)
        mask = (pat == pat[0]) & (torch.as_tensor(tf) <= tk)
        pc = cut["batch"]["patch_pat"][cut["batch"]["te_patch"]]
        torch.testing.assert_close(zf[mask], _rows(cut, 0)[1][pc == pc[0]], atol=ATOL, rtol=0)
        checked += 1
    assert checked >= 2


def test_later_values_and_unresolved_labels_do_not_reach_earlier_rows(setup) -> None:
    checked = 0
    for ep in setup["eps"]:
        times = ep["recipe"]["queries"][0]["times"]
        if len(times) < 3 or ep["task"]["label"] == "forecast":
            continue
        tk = int(times[0])
        base = _run(setup, [ep])
        ep2 = copy.deepcopy(ep)
        q = ep2["query"]
        a, b = int(q["ev_off"][0]), int(q["ev_off"][1])
        later = np.zeros(len(q["ev_t"]), dtype=bool)
        later[a:b] = np.asarray(q["ev_t"][a:b]) > tk
        q["ev_z"] = np.where(later, np.asarray(q["ev_z"]) + 3.0, q["ev_z"]).astype(np.float32)
        lo, hi = int(q["own_lab_off"][0]), int(q["own_lab_off"][1])
        unres = np.zeros(len(q["own_lab_t_resolved"]), dtype=bool)
        unres[lo:hi] = np.asarray(q["own_lab_t_resolved"][lo:hi]) > tk
        q["own_lab_y"] = np.where(unres & np.asarray(q["own_lab_valid"], dtype=bool),
                                  1 - np.nan_to_num(np.asarray(q["own_lab_y"])), q["own_lab_y"]).astype(np.float32)
        pert = _run(setup, [ep2])
        t0, z0 = _rows(base, 0)
        t1, z1 = _rows(pert, 0)
        first = (base["batch"]["patch_pat"][base["batch"]["te_patch"]] ==
                 base["batch"]["patch_pat"][base["batch"]["te_patch"]][0]).numpy()
        early = first & (t0 <= tk)
        torch.testing.assert_close(z0[early], z1[early], atol=ATOL, rtol=0)
        if later.any() and (first & (t0 > tk)).any():
            assert not torch.allclose(z0[first & (t0 > tk)], z1[first & (t0 > tk)], atol=ATOL)
        checked += 1
    assert checked >= 5


def test_context_labels_reach_queries_only_through_the_rows(setup) -> None:
    """A context label changes the queries' predictions, never their own temporal states."""
    ep = next(e for e in setup["eps"] if e["task"]["label"] == "binary" and len(e["recipe"]["context"]) >= 2)
    base = _run(setup, [ep])
    ep2 = copy.deepcopy(ep)
    c = ep2["context"]
    ok = np.asarray(c["lab_valid"], dtype=bool)
    c["lab_y"] = np.where(ok, 1 - np.nan_to_num(np.asarray(c["lab_y"])), c["lab_y"]).astype(np.float32)
    pert = _run(setup, [ep2])
    qp = base["batch"]["pat_role"] == 1
    qpatch = torch.isin(base["batch"]["patch_pat"], torch.nonzero(qp).squeeze(1))
    torch.testing.assert_close(base["h"][qpatch], pert["h"][qpatch], atol=ATOL, rtol=0)
    assert not torch.allclose(base["z"], pert["z"], atol=1e-4)


def _permute_context(ep: dict, perm: list) -> dict:
    ep = copy.deepcopy(ep)
    c = ep["context"]
    groups = dict(contracts._GROUPS)
    groups["lab"] = contracts._LABEL_FIELDS + tuple(x for x in ("lab_yh", "lab_mh", "lab_yf", "lab_mf") if x in c)
    for g, keys in groups.items():
        off = np.asarray(c[f"{g}_off"])
        idx = np.concatenate([np.arange(off[k], off[k + 1]) for k in perm]).astype(np.int64)
        for key in keys:
            c[key] = np.asarray(c[key])[idx]
        c[f"{g}_off"] = np.concatenate([[0], np.cumsum([off[k + 1] - off[k] for k in perm])]).astype(np.int64)
    c["end_t"] = np.asarray(c["end_t"])[perm]
    c["end_kind"] = [c["end_kind"][k] for k in perm]
    ep["recipe"]["context"] = [ep["recipe"]["context"][k] for k in perm]
    return ep


def test_context_order_does_not_matter(setup) -> None:
    for ep in [e for e in setup["eps"] if len(e["recipe"]["context"]) >= 3][:6]:
        n = len(ep["recipe"]["context"])
        perm = list(np.random.default_rng(1).permutation(n))
        a, b = _run(setup, [ep]), _run(setup, [_permute_context(ep, perm)])
        torch.testing.assert_close(a["z"], b["z"], atol=ATOL, rtol=0)


def test_a_packed_batch_equals_its_episodes_alone(setup) -> None:
    eps = setup["eps"][:5]
    together = _run(setup, eps)
    for e, ep in enumerate(eps):
        alone = _run(setup, [ep])
        torch.testing.assert_close(_rows(together, e)[1], _rows(alone, 0)[1], atol=ATOL, rtol=0)


def test_anonymized_channels_are_known_only_by_behaviour(setup) -> None:
    """Relabelling an anonymized episode's local channel ids (with each channel's random identity kept) changes
    nothing: no channel id carries meaning of its own."""
    ep = copy.deepcopy(next(e for e in setup["eps"] if e["task"]["label"] != "forecast"))
    used = np.unique(np.concatenate([ep[p][k] for p in ("context", "query") for k in ("ev_c", "iv_c")]))
    perm = np.random.default_rng(2).permutation(len(used))
    loc = {int(c): int(k) for k, c in enumerate(used)}
    ep["channels"] = {"global_ids": used.copy(), "n": len(used)}
    ep["task"]["anonymized"] = True
    ep["task"]["descriptor"] = None
    for p in ("context", "query"):
        for k in ("ev_c", "iv_c"):
            ep[p][k] = np.array([loc[int(x)] for x in ep[p][k]], dtype=np.int32)
    ep2 = copy.deepcopy(ep)
    for p in ("context", "query"):
        for k in ("ev_c", "iv_c"):
            ep2[p][k] = perm[np.asarray(ep2[p][k])].astype(np.int32)
    bld = _builder(setup)
    b1, b2 = bld([ep]), bld([ep2])
    # give channel perm[c] in the second batch the identity channel c had in the first
    b2["rand_id"][perm] = b1["rand_id"][np.arange(len(used))]
    m = setup["model"]
    with torch.no_grad():
        z1 = m(B.to_torch(b1))["z"]
        z2 = m(B.to_torch(b2))["z"]
    torch.testing.assert_close(z1, z2, atol=ATOL, rtol=0)


def test_channel_identities_do_not_depend_on_the_other_channels(setup) -> None:
    ep = setup["eps"][0]
    bld = _builder(setup)
    b1 = bld([ep])
    ep2 = copy.deepcopy(ep)
    q = ep2["query"]
    a, b = int(q["ev_off"][0]), int(q["ev_off"][1])
    keep = np.ones(len(q["ev_t"]), dtype=bool)
    first_channel = int(np.asarray(q["ev_c"][a:b])[0]) if b > a else None
    if first_channel is not None:
        keep[a:b] &= np.asarray(q["ev_c"][a:b]) != first_channel
    for k in ("ev_t", "ev_c", "ev_v", "ev_z", "ev_bin", "ev_f"):
        q[k] = np.asarray(q[k])[keep]
    q["ev_off"] = np.concatenate([[0], np.cumsum(np.diff(q["ev_off"]) - np.array(
        [int((~keep[int(q0):int(q1)]).sum()) for q0, q1 in zip(ep["query"]["ev_off"][:-1], ep["query"]["ev_off"][1:],
                                                                 strict=True)]))])
    b2 = bld([ep2])
    v1 = {int(g): b1["rand_id"][s] for g, s in zip(b1["tok_gid"], b1["tok_slot"], strict=True) if s >= 0}
    v2 = {int(g): b2["rand_id"][s] for g, s in zip(b2["tok_gid"], b2["tok_slot"], strict=True) if s >= 0}
    for g in set(v1) & set(v2):
        np.testing.assert_array_equal(v1[g], v2[g])


def test_context_subsets_and_training_pieces(setup) -> None:
    from icutwin.icl import eval as E, train as TR

    ep = next(e for e in setup["eps"] if len(e["recipe"]["context"]) >= 4)
    sub = infer.select_context(ep, [3, 1])
    assert sub["recipe"]["context"] == [ep["recipe"]["context"][3], ep["recipe"]["context"][1]]
    assert len(sub["context"]["ev_off"]) == 3 and not contracts.episode_failures(sub)
    empty = infer.select_context(ep, [])
    assert empty["recipe"]["context"] == [] and len(empty["context"]["ev_t"]) == 0
    # every parameter is in exactly one optimizer group; the schedule warms up, then decays to its floor
    m = M.TrajPFN(M.config("tiny", len(setup["names"]), setup["n_static"]))
    cfg = TR.TrainConfig(store="", pool=None, vocab="", run_dir="", warmup_steps=10, lr=1e-3, min_lr_ratio=0.1)
    opt = TR.optimizer(m, cfg)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in m.parameters())
    lrs = [TR.lr_at(s, 100, cfg) for s in range(100)]
    assert lrs[0] < lrs[9] == pytest.approx(1e-3) and lrs[-1] == pytest.approx(1e-4, rel=0.05)
    assert E.auroc([0.1, 0.9, 0.8], [0, 1, 1]) == 1.0 and E.auroc([0.5], [1]) is None
    m_, lo, hi = E.bootstrap_mean(np.r_[np.ones(50), -np.ones(50)] * 0.0 - 1.0)
    assert m_ == -1.0 and lo == hi == -1.0


# -- heads ----------------------------------------------------------------------------------------------------------
def test_hazard_likelihood_against_brute_force() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, H.N_BINS, H.OUTCOMES)
    p = torch.softmax(logits, -1)
    time = torch.tensor([0.3, 5.0, 40.0, 200.0, 12.0])
    cause = torch.tensor([0, 1, -1, 2, 0])
    nll = H.hazard_nll(logits, time, cause)
    for i in range(5):
        j = int(H.hazard_bins(time[i:i + 1])[0])
        surv = float(torch.prod(p[i, :j, 0]))
        lik = surv * float(p[i, j, 1 + int(cause[i])]) if cause[i] >= 0 else surv
        assert abs(float(nll[i]) + np.log(lik)) < 1e-4
    # incidence of the event plus the other causes plus survival is one at the grid's end
    cif = H.cumulative_incidence(logits, torch.full((5, 1), 2 * H.GRID_HOURS[-1]))
    surv_end = torch.prod(p[..., 0], -1)
    other = sum(torch.sum(torch.cumprod(torch.cat([torch.ones(5, 1), p[:, :-1, 0]], 1), -1) * p[..., c], -1)
                for c in (2, 3))
    torch.testing.assert_close(cif[:, 0] + other + surv_end, torch.ones(5), atol=1e-4, rtol=0)


def test_bar_distribution_is_a_density() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 64)
    e = H.bar_edges(64)
    centres = (e[:-1] + e[1:]) / 2
    dens = torch.exp(-H.bar_nll(logits[:, None, :].expand(3, 64, 64), centres.expand(3, 64)))
    torch.testing.assert_close((dens * (e[1] - e[0])).sum(1), torch.ones(3), atol=1e-4, rtol=0)
    point = torch.full((1, 64), -50.0)
    point[0, 10] = 50.0
    assert float(H.bar_crps(torch.softmax(point, -1), centres[10:11])) < 1e-3


# -- robustness and learning ------------------------------------------------------------------------------------------
def test_degenerate_episodes_run_finitely(setup) -> None:
    ep = copy.deepcopy(next(e for e in setup["eps"] if e["task"]["label"] == "binary"))
    zero = copy.deepcopy(ep)                       # no context, descriptor shown
    for k, x in list(zero["context"].items()):
        if k.endswith("_off"):
            zero["context"][k] = np.zeros(1, dtype=np.int64)
        elif isinstance(x, np.ndarray):
            zero["context"][k] = x[:0]
    zero["context"]["end_kind"] = []
    zero["recipe"]["context"] = []
    invalid = copy.deepcopy(ep)                    # every label invalid
    for p, pre in (("context", "lab_"), ("query", "lab_"), ("query", "own_lab_")):
        invalid[p][f"{pre}valid"] = np.zeros_like(invalid[p][f"{pre}valid"])
        invalid[p][f"{pre}y"] = np.full_like(invalid[p][f"{pre}y"], np.nan)
    fc = copy.deepcopy(next(e for e in setup["eps"] if e["task"]["label"] == "forecast"))
    fc["query"]["lab_mf"] = np.zeros_like(fc["query"]["lab_mf"])
    m = setup["model"]
    b = B.to_torch(_builder(setup, train=True, max_rows=16)([zero, invalid, fc]))
    m.train()
    loss, res = m.loss(b)
    loss.backward()
    m.zero_grad()
    m.eval()
    assert torch.isfinite(loss)
    assert torch.isfinite(res["z"]).all()


def test_tiny_model_fits_a_few_episodes(setup) -> None:
    """Plumbing and capacity: a tiny model's loss on 6 fixed episodes falls by half in 60 steps."""
    torch.manual_seed(0)
    eps = [e for e in setup["eps"] if e["task"]["label"] in ("binary", "ordinal", "regression", "tte")][:6]
    m = M.TrajPFN(M.config("tiny", len(setup["names"]), setup["n_static"]))
    b = B.to_torch(_builder(setup, train=False, max_rows=16)(eps))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss, _r = m.loss(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
    assert float(loss) < 0.5 * first, (first, float(loss))
