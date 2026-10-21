"""Common reranker interface."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RerankResult:
    ranked: list[dict]           # candidates (id, text, score) in new order
    latency_s: float = 0.0       # measured on the original (uncached) call
    input_tokens: int = 0
    cached: bool = False
    extra: dict = field(default_factory=dict)


class Reranker:
    name = "base"

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        raise NotImplementedError


def order_by_scores(candidates: list[dict], scores: list[float]) -> list[dict]:
    """Stable sort by score desc; ties keep first-stage order."""
    paired = [dict(c, score=float(s)) for c, s in zip(candidates, scores)]
    return [p for _, p in sorted(enumerate(paired), key=lambda t: (-t[1]["score"], t[0]))]
