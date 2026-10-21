from ragcmp.rerankers.base import Reranker, RerankResult

__all__ = ["Reranker", "RerankResult", "make_reranker"]


def make_reranker(name: str, cfg: dict, cache) -> Reranker:
    if name == "vanilla":
        from ragcmp.rerankers.vanilla import VanillaReranker
        return VanillaReranker()
    if name == "bge":
        from ragcmp.rerankers.bge import BGEReranker
        return BGEReranker(cfg, cache)
    if name == "jev":
        from ragcmp.rerankers.jev import JevReranker
        return JevReranker(cfg, cache)
    raise ValueError(f"unknown reranker: {name}")
