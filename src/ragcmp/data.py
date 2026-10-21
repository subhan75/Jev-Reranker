"""Sample HotpotQA (distractor) questions and build a pooled paragraph corpus.

Outputs (data/processed/):
  questions_dev.jsonl, questions_test.jsonl  -> {id, question, answer, type, level, gold: [titles]}
  corpus.jsonl                               -> {id: title, text}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"


def paragraph_text(title: str, sentences: list[str]) -> str:
    return f"{title}: " + "".join(sentences).strip()


def gold_titles(example: dict) -> list[str]:
    seen: list[str] = []
    for t in example["supporting_facts"]["title"]:
        if t not in seen:
            seen.append(t)
    return seen


def sample_indices(n_total: int, n_dev: int, n_test: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idx = rng.sample(range(n_total), n_dev + n_test)
    return idx[:n_dev], idx[n_dev:]


def to_question(ex: dict) -> dict:
    return {
        "id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex["type"],
        "level": ex["level"],
        "gold": gold_titles(ex),
    }


def build_corpus(examples: list[dict]) -> list[dict]:
    corpus: dict[str, str] = {}
    for ex in examples:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            corpus.setdefault(title, paragraph_text(title, sents))
    return [{"id": t, "text": txt} for t, txt in corpus.items()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(cfg: dict) -> dict:
    out = cfg["paths"]["data_dir"] / "processed"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    dev_idx, test_idx = sample_indices(len(ds), cfg["n_dev"], cfg["n_test"], cfg["seed"])
    dev = [ds[i] for i in dev_idx]
    test = [ds[i] for i in test_idx]
    corpus = build_corpus(dev + test)
    write_jsonl(out / "questions_dev.jsonl", [to_question(e) for e in dev])
    write_jsonl(out / "questions_test.jsonl", [to_question(e) for e in test])
    write_jsonl(out / "corpus.jsonl", corpus)
    ids = {c["id"] for c in corpus}
    missing = sum(1 for e in dev + test for g in gold_titles(e) if g not in ids)
    return {"dev": len(dev), "test": len(test), "corpus": len(corpus), "gold_missing": missing}


def load_split(cfg: dict, split: str) -> list[dict]:
    return read_jsonl(cfg["paths"]["data_dir"] / "processed" / f"questions_{split}.jsonl")


def load_corpus(cfg: dict) -> list[dict]:
    return read_jsonl(cfg["paths"]["data_dir"] / "processed" / "corpus.jsonl")
