"""Paper P amendment 7: the hazard estimators (``eval.hazards``) on a toy grammar and a simulated process."""

import numpy as np

from icutwin.eval import hazards as hz


def toy() -> hz.Grammar:
    # 0 pad, 1 [TL], 2 [END], 3-4 intervals, 5-9 events (6 IMV, 8 alive discharge, 9 dead), 10 clock
    n = 11
    vec = lambda ids: np.isin(np.arange(n), ids)  # noqa: E731
    return hz.Grammar(vec([3, 4]), vec([5, 6, 7, 8, 9]), vec([10]), vec([5]), 1, 2,
                      {"death": np.array([9]), "alive": np.array([8]), "imv": np.array([6]),
                       "vasopressor": np.array([], dtype=int), "rrt": np.array([], dtype=int)})


def test_group_first_and_set_hazard():
    g = toy()
    seq = np.array([1, 5, 7, 3, 6, 10, 4, 9])
    assert hz.group_first(seq, g).tolist() == [-1, 5, 7, -1, 6, 6, -1, 9]
    p = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    toks = np.array([6, 9])
    # after an interval both may come; after event 7 only 9; after event 9 only 9 itself (equal ids allowed)
    assert np.allclose(hz.set_hazard(p, toks, np.array([-1, 7, 9])), [0.3, 0.2, 0.2])


def test_summarize_first_events_and_increments():
    g = toy()
    # prompt [TL] 5 | continuation 3 6 3 7 4 9 [END]: IMV at the second group, death at the fourth
    seq = np.array([1, 5, 3, 6, 3, 7, 4, 9, 2])
    hours = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=float)
    edges = np.arange(0, 5, dtype=float)
    haz = {c: np.full(len(seq), 0.01) for c in hz.CAUSES}
    p = hz.summarize(seq, 1, haz, hours, edges, g)
    assert p.end_kind == hz.END_DEAD and p.end_hour == 3.0
    assert p.first["imv"] == 1.0 and p.first["death"] == 3.0 and np.isinf(p.first["alive"])
    # death at risk over positions 1..6 (predicting tokens 2..7): hours 0 (none), 1, 1, 2, 2, 3 -> bins 0, 0, 1, 1, 2
    assert np.allclose(p.inc["death"], [0.02, 0.02, 0.01, 0.0])
    # IMV at risk until its first event (position 3, predicted at 2): positions 1, 2 -> hour 0 none, hour 1 bin 0
    assert np.allclose(p.inc["imv"], [0.01, 0, 0, 0])
    assert np.allclose(p.comp["imv"], [0.02, 0, 0, 0])


def test_estimators_unbiased_and_hybrid_less_variable():
    """A discrete-time process with path-dependent hazards: E1 and E2 both estimate the cumulative incidence the
    paths' own indicator gives (large sample), and E2 varies less across small ensembles."""
    rng = np.random.default_rng(0)
    edges = np.arange(0, 49, dtype=float)

    def ensemble(n):
        first = np.full(n, np.inf)
        end_hour = np.full(n, np.inf)
        end_kind = np.full(n, hz.END_CENSORED)
        inc_d = np.zeros((n, 48))
        inc_a = np.zeros((n, 48))
        for r in range(n):
            sick = rng.random() < 0.3
            for step in range(4 * 48):           # 4 steps per hour
                hour = (step + 1) / 4
                q = (0.01 if sick else 0.001) * (1 + 0.5 * rng.random())
                a = 0.004
                b = min(int(np.ceil(hour)) - 1, 47)
                inc_d[r, b] += q
                inc_a[r, b] += a
                u = rng.random()
                if u < q:
                    first[r], end_hour[r], end_kind[r] = hour, hour, hz.END_DEAD
                    break
                if u < q + a:
                    end_hour[r], end_kind[r] = hour, hz.END_ALIVE
                    break
                if hour >= 40 and rng.random() < 0.02:     # a cap: censored late
                    end_hour[r] = hour
                    break
            else:
                end_hour[r] = 48.0
        return first, end_hour, end_kind, inc_d, inc_a

    big = ensemble(4000)
    truth = np.mean(big[0] <= 24)
    est = [hz.estimate("death", 24.0, edges, *ensemble(32)) for _ in range(60)]
    e1 = np.array([x["e1"] for x in est])
    e2 = np.array([x["e2"] for x in est])
    assert abs(e1.mean() - truth) < 0.03 and abs(e2.mean() - truth) < 0.03
    assert e2.std() < 0.8 * e1.std()
    # the self-check: summed compensators match the sampled deaths
    f, eh, ek, d, _a = big
    assert abs((ek == hz.END_DEAD).sum() / d.sum() - 1) < 0.1


def test_e0_complete_case():
    first = np.array([5.0, np.inf, np.inf, np.inf])
    end_hour = np.array([5.0, 30.0, 10.0, 12.0])
    end_kind = np.array([hz.END_DEAD, hz.END_ALIVE, hz.END_ALIVE, hz.END_CENSORED])
    # known: the death, the discharge after 24 h, the discharge at 10 h; the rollout censored at 12 h is not
    assert hz.e0(first, end_hour, end_kind, 24.0) == 1 / 3


def test_clock_vectorized_equals_loop():
    """The decode clock's vectorized draws equal the token-by-token reference (same rng order, same values)."""
    import random

    class V:
        meta = [{"cls": "event"}] * 5 + [{"cls": "interval", "interval": k} for k in range(4)]
        interval_sketches = {"*": [[1.0, 2.0, 4.0], [5.0, 9.0, 15.0, 30.0], None, [60.0, 120.0]],
                             "mimic": [None, [6.0, 10.0, 20.0], None, None]}

    ck = hz.Clock(V(), "mimic")
    seq = np.random.default_rng(1).integers(0, 9, size=500)
    a = ck.minutes(seq, random.Random("clock:0:x"), 3)
    b = ck.minutes_loop(seq, random.Random("clock:0:x"), 3)
    assert np.array_equal(a, b)


def test_running_state():
    """Infusions run from a 'begins' action until an 'ends' one; a dose change changes nothing."""
    n = 12
    nz = hz.Normalizer(np.arange(n), np.array([], dtype=int), -1, -1, ((), (), ()), np.zeros(n, bool),
                       np.zeros(n, int), np.array([5, 6]), np.array([0, 0]), 2,
                       np.array([-1, -1, -1, -1, 0, 0, 0, 1, -1, -1, -1, -1]),
                       np.array([-1, -1, -1, -1, 1, -1, 0, 1, -1, -1, -1, -1]))
    # tokens: 4 starts key 0, 5 changes its dose, 6 stops it, 7 starts key 1
    seq = np.array([1, 4, 2, 5, 7, 6, 3])
    r = hz.running_state(seq, np.arange(len(seq)), nz)
    assert r[:, 0].tolist() == [False, True, True, True, True, False, False]
    assert r[:, 1].tolist() == [False, False, False, False, True, True, True]
