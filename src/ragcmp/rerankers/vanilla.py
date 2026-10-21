from ragcmp.rerankers.base import Reranker, RerankResult


class VanillaReranker(Reranker):
    """Control: keep the dense-retrieval order."""

    name = "vanilla"

    def rerank(self, query: str, candidates: list[dict]) -> RerankResult:
        return RerankResult(ranked=list(candidates))
