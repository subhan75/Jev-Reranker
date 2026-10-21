"""BAAI/bge-reranker-base cross-encoder, run locally."""
from __future__ import annotations

import time

from ragcmp.cache import Cache
from ragcmp.rerankers.base import Reranker, RerankResult, order_by_scores


class BGEReranker(Reranker):
    name = "bge"

    def __init__(self, cfg: dict, cache: Cache, model=None):
        c = cfg["rerank"]["bge"]
        self.model_name = c["model"]
        self.batch_size = c["batch_size"]
        self.max_length = c["max_length"]
        self.cache = cache
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def _score(self, query: str, candidates: list[dict]) -> dict:
        model = self.model  # load outside the timed region
        t0 = time.perf_counter()
        scores = model.predict(
            [(query, c["text"]) for c in candidates], batch_size=self.batch_size, show_progress_bar=False
        )
        return {"scores": [float(s) for s in scores], "latency_s": time.perf_counter() - t0}

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        payload = {"q": query, "ids": [c["id"] for c in candidates], "max_length": self.max_length}
        val, cached = self.cache.get_or_compute(
            "bge:" + self.model_name, payload, lambda: self._score(query, candidates)
        )
        return RerankResult(
            ranked=order_by_scores(candidates, val["scores"]), latency_s=val["latency_s"], cached=cached
        )
