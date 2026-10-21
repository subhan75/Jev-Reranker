"""OpenAI embeddings (cached) -> Chroma; first-stage dense retrieval."""
from __future__ import annotations

import chromadb
from openai import OpenAI
from tqdm import tqdm

from ragcmp.cache import Cache


class Embedder:
    def __init__(self, cfg: dict, cache: Cache, client: OpenAI | None = None):
        self.model = cfg["embedding"]["model"]
        self.batch_size = cfg["embedding"]["batch_size"]
        self.cache = cache
        self.client = client or OpenAI()
        self.tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(self._key(t))
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=[texts[i] for i in batch])
            self.tokens += resp.usage.total_tokens
            for i, d in zip(batch, resp.data):
                out[i] = d.embedding
                self.cache.set(self._key(texts[i]), d.embedding)
        return out  # type: ignore[return-value]

    def _key(self, text: str) -> str:
        from ragcmp.cache import make_key
        return make_key("emb:" + self.model, text)


def get_collection(cfg: dict):
    client = chromadb.PersistentClient(path=str(cfg["paths"]["chroma_dir"]))
    return client.get_or_create_collection(
        name=cfg["retrieval"]["collection"], metadata={"hnsw:space": "cosine"}
    )


def build_index(cfg: dict, corpus: list[dict], embedder: Embedder, chunk: int = 512) -> int:
    col = get_collection(cfg)
    existing = set(col.get(include=[])["ids"]) if col.count() else set()
    rows = [c for c in corpus if c["id"] not in existing]
    for start in tqdm(range(0, len(rows), chunk), desc="index"):
        batch = rows[start : start + chunk]
        embs = embedder.embed([c["text"] for c in batch])
        col.add(ids=[c["id"] for c in batch], documents=[c["text"] for c in batch], embeddings=embs)
    return col.count()


def retrieve(cfg: dict, questions: list[dict], embedder: Embedder) -> dict[str, list[dict]]:
    """Return {qid: [{id, text, score}, ...top_k]} in vector order."""
    col = get_collection(cfg)
    k = cfg["retrieval"]["top_k"]
    q_embs = embedder.embed([q["question"] for q in questions])
    res = col.query(query_embeddings=q_embs, n_results=k, include=["documents", "distances"])
    out = {}
    for q, ids, docs, dists in zip(questions, res["ids"], res["documents"], res["distances"]):
        out[q["id"]] = [{"id": i, "text": d, "score": 1.0 - dist} for i, d, dist in zip(ids, docs, dists)]
    return out
