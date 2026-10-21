"""Run retrieval -> rerank -> (generate) for a split and write per-question records."""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from ragcmp.cache import Cache
from ragcmp.data import load_corpus, load_split
from ragcmp.index import Embedder, build_index, retrieve
from ragcmp.metrics import exact_match, f1_score, retrieval_metrics
from ragcmp.rerankers import make_reranker

VARIANTS = ("vanilla", "bge", "jev")


def open_cache(cfg: dict) -> Cache:
    return Cache(cfg["paths"]["cache_dir"] / "cache.sqlite")


def index_corpus(cfg: dict) -> dict:
    cache = open_cache(cfg)
    emb = Embedder(cfg, cache)
    n = build_index(cfg, load_corpus(cfg), emb)
    cache.close()
    return {"indexed": n, "embedding_tokens_this_run": emb.tokens}


def first_stage(cfg: dict, questions: list[dict], cache: Cache) -> dict[str, list[dict]]:
    """Dense top-k for every question; saved to disk so all variants share identical candidates."""
    path = cfg["paths"]["data_dir"] / "processed" / "retrieval.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [q for q in questions if q["id"] not in saved]
    if missing:
        saved.update(retrieve(cfg, missing, Embedder(cfg, cache)))
        path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    return {q["id"]: saved[q["id"]] for q in questions}


def run_split(cfg: dict, split: str, variants=VARIANTS, limit: int | None = None,
              generate: bool = True) -> dict[str, Path]:
    questions = load_split(cfg, split)[:limit]
    cache = open_cache(cfg)
    cands = first_stage(cfg, questions, cache)
    final_k = cfg["rerank"]["final_k"]
    pool_k = cfg["retrieval"]["top_k"]
    gen = None
    if generate:
        from ragcmp.generate import Generator
        gen = Generator(cfg, cache)

    outputs = {}
    for name in variants:
        rr = make_reranker(name, cfg, cache)
        path = cfg["paths"]["results_dir"] / f"{split}_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for q in tqdm(questions, desc=f"{split}:{name}"):
                res = rr.rerank(q["question"], cands[q["id"]])
                ranked_ids = [c["id"] for c in res.ranked]
                rec = {
                    "qid": q["id"], "type": q["type"], "level": q["level"], "gold": q["gold"],
                    "ranked": ranked_ids,
                    **retrieval_metrics(ranked_ids, q["gold"], pool_k=pool_k),
                    "rerank_latency_s": res.latency_s,
                    "rerank_input_tokens": res.input_tokens,
                    "rerank_cached": res.cached,
                }
                if gen is not None:
                    g, g_cached = gen.answer(q["question"], res.ranked[:final_k])
                    rec.update({
                        "answer": q["answer"], "pred": g["answer"],
                        "em": exact_match(g["answer"], q["answer"]),
                        "f1": f1_score(g["answer"], q["answer"]),
                        "gen_input_tokens": g["input_tokens"],
                        "gen_output_tokens": g["output_tokens"],
                        "gen_latency_s": g["latency_s"],
                        "gen_cached": g_cached,
                    })
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        outputs[name] = path
    cache.close()
    return outputs
