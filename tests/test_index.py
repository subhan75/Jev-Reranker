from types import SimpleNamespace

from ragcmp.cache import Cache
from ragcmp.index import Embedder, build_index, retrieve

VOCAB = ["apple", "banana", "car", "dog", "engine"]


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        data = [SimpleNamespace(embedding=[float(w in t.lower()) + 1e-3 for w in VOCAB]) for t in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=len(input)))


def _cfg(tmp_path):
    return {
        "embedding": {"model": "fake", "batch_size": 2},
        "retrieval": {"collection": "test_col", "top_k": 2},
        "paths": {"chroma_dir": tmp_path / "chroma"},
    }


def test_build_and_retrieve(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeEmbeddings()
    emb = Embedder(cfg, Cache(tmp_path / "c.sqlite"), client=SimpleNamespace(embeddings=fake))
    corpus = [
        {"id": "A", "text": "apple banana"},
        {"id": "C", "text": "car engine"},
        {"id": "D", "text": "dog"},
    ]
    assert build_index(cfg, corpus, emb) == 3
    assert build_index(cfg, corpus, emb) == 3  # idempotent
    hits = retrieve(cfg, [{"id": "q", "question": "which car engine"}], emb)
    assert [h["id"] for h in hits["q"]][0] == "C"
    assert len(hits["q"]) == 2


def test_embedder_uses_cache(tmp_path):
    cfg = _cfg(tmp_path)
    fake = FakeEmbeddings()
    emb = Embedder(cfg, Cache(tmp_path / "c.sqlite"), client=SimpleNamespace(embeddings=fake))
    emb.embed(["apple", "dog"])
    n = fake.calls
    emb.embed(["apple", "dog"])
    assert fake.calls == n
