from types import SimpleNamespace

from ragcmp.cache import Cache
from ragcmp.rerankers.base import order_by_scores
from ragcmp.rerankers.bge import BGEReranker
from ragcmp.rerankers.jev import JevReranker
from ragcmp.rerankers.vanilla import VanillaReranker

CANDS = [{"id": c, "text": f"text {c}", "score": 0.0} for c in "abcd"]
CFG = {
    "rerank": {
        "bge": {"model": "fake-bge", "batch_size": 8, "max_length": 128},
        "jev": {"model": "fake-jev", "instructions": "Does `passages[{i}]` help answer `query`?"},
    }
}


def test_order_by_scores_stable_ties():
    out = order_by_scores(CANDS, [0.1, 0.9, 0.1, 0.5])
    assert [c["id"] for c in out] == ["b", "d", "a", "c"]


def test_vanilla_keeps_order():
    assert [c["id"] for c in VanillaReranker().rerank("q", CANDS).ranked] == list("abcd")


class FakeCE:
    def __init__(self):
        self.calls = 0

    def predict(self, pairs, **kw):
        self.calls += 1
        table = {"a": 0.0, "b": 0.2, "c": 0.9, "d": 0.5}
        return [table[p[1][-1]] for p in pairs]


def test_bge_ranks_and_caches(tmp_path):
    ce = FakeCE()
    r = BGEReranker(CFG, Cache(tmp_path / "c.sqlite"), model=ce)
    res = r.rerank("q", CANDS)
    assert [c["id"] for c in res.ranked] == ["c", "d", "b", "a"] and not res.cached
    assert r.rerank("q", CANDS).cached and ce.calls == 1


class FakeJevClient:
    def __init__(self, scores):
        self.scores = scores
        self.last = None

    def system_one(self, state, questions):
        self.last = (state, questions)
        answers = {k: SimpleNamespace(noul=self.scores[int(k[1:])]) for k in questions}
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=123), model="jev-x")


def test_jev_request_shape_ranking_and_cache(tmp_path):
    client = FakeJevClient([0.1, 0.95, 0.3, 0.7])
    r = JevReranker(CFG, Cache(tmp_path / "c.sqlite"), client=client)
    res = r.rerank("who?", CANDS)
    state, questions = client.last
    assert state == {"query": "who?", "passages": [c["text"] for c in CANDS]}
    assert list(questions) == ["p0", "p1", "p2", "p3"]
    assert "passages[2]" in questions["p2"].instructions
    assert [c["id"] for c in res.ranked] == ["b", "d", "c", "a"]
    assert res.input_tokens == 123 and res.extra["model"] == "jev-x"
    client.scores = [1, 0, 0, 0]
    assert [c["id"] for c in r.rerank("who?", CANDS).ranked][0] == "b"  # served from cache


def test_jev_reads_openrouter_cost_from_raw_body():
    from ragcmp.rerankers.jev import _billed_cost

    raw = SimpleNamespace(json=lambda: {"usage": {"input_tokens": 10, "cost": 4.2e-07}})
    resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=10), raw_http_response=raw)
    assert _billed_cost(resp) == 4.2e-07
    assert _billed_cost(SimpleNamespace(usage=SimpleNamespace(), raw_http_response=None)) is None
