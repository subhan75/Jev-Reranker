"""Jev (TypeSafe System One) as a pointwise reranker: one request per query,
one Noul question per candidate passage, ranked by the returned probability."""
from __future__ import annotations

import time

from ragcmp.cache import Cache
from ragcmp.rerankers.base import Reranker, RerankResult, order_by_scores


class JevReranker(Reranker):
    name = "jev"

    def __init__(self, cfg: dict, cache: Cache, client=None):
        c = cfg["rerank"]["jev"]
        self.cfg = cfg
        self.model = c["model"]
        self.instructions = " ".join(c["instructions"].split())
        self.cache = cache
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from ragcmp.clients import jev_client
            self._client = jev_client(self.cfg)
        return self._client

    def build_request(self, query: str, candidates: list[dict]) -> tuple[dict, dict]:
        from typesafe_sdk import Noul

        state = {"query": query, "passages": [c["text"] for c in candidates]}
        questions = {
            f"p{i}": Noul(instructions=self.instructions.format(i=i)) for i in range(len(candidates))
        }
        return state, questions

    def _score(self, query: str, candidates: list[dict]) -> dict:
        state, questions = self.build_request(query, candidates)
        t0 = time.perf_counter()
        resp = self.client.system_one(state=state, questions=questions)
        latency = time.perf_counter() - t0
        scores = [float(resp.answers[f"p{i}"].noul) for i in range(len(candidates))]
        cost = getattr(resp.usage, "cost", None)  # added by OpenRouter
        return {
            "scores": scores,
            "latency_s": latency,
            "input_tokens": int(resp.usage.input_tokens),
            "cost_usd": float(cost) if cost is not None else None,
            "model": resp.model,
        }

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        payload = {
            "q": query,
            "ids": [c["id"] for c in candidates],
            "instructions": self.instructions,
        }
        val, cached = self.cache.get_or_compute(
            "jev:" + self.model, payload, lambda: self._score(query, candidates)
        )
        return RerankResult(
            ranked=order_by_scores(candidates, val["scores"]),
            latency_s=val["latency_s"],
            input_tokens=val["input_tokens"],
            cached=cached,
            extra={"model": val["model"], "cost_usd": val.get("cost_usd")},
        )
