"""Retrieval metrics, HotpotQA answer EM/F1, and paired significance tests."""
from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np
from scipy.stats import binomtest

# ---------- retrieval ----------


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold) if gold else 0.0


def all_gold_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return float(set(gold) <= set(ranked[:k]))


def mrr(ranked: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, pid in enumerate(ranked, start=1):
        if pid in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], start=1) if pid in gold_set)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(ranked: list[str], gold: list[str], ks=(1, 3, 5), pool_k: int | None = None) -> dict:
    m = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}
    m["mrr"] = mrr(ranked, gold)
    m["ndcg@5"] = ndcg_at_k(ranked, gold, 5)
    m["both_gold@5"] = all_gold_at_k(ranked, gold, 5)
    if pool_k:
        m[f"recall@{pool_k}"] = recall_at_k(ranked, gold, pool_k)
    return m


# ---------- answers (HotpotQA official evaluation) ----------


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (np_ in special or ng in special) and np_ != ng:
        return 0.0
    p_toks, g_toks = np_.split(), ng.split()
    common = Counter(p_toks) & Counter(g_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p_toks), same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# ---------- statistics ----------


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> dict:
    """Mean difference (a - b) with a 95% percentile CI and two-sided p-value."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    diff = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"delta": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p": float(min(p, 1.0))}


def mcnemar(a: list[float], b: list[float]) -> dict:
    """Exact McNemar test on paired binary outcomes."""
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_only + c_only
    p = binomtest(b_only, n, 0.5).pvalue if n else 1.0
    return {"a_only": b_only, "b_only": c_only, "p": float(p)}
