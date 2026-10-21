"""Aggregate per-question records into a summary table with paired significance tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ragcmp.data import read_jsonl
from ragcmp.metrics import mcnemar, paired_bootstrap

RETRIEVAL = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "both_gold@5"]
ANSWER = ["em", "f1"]
BINARY = ["both_gold@5", "em"]
COMPARISONS = [("jev", "bge"), ("bge", "vanilla"), ("jev", "vanilla")]


def _pct(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def _billed(recs: list[dict], key: str) -> float | None:
    """Sum of provider-reported cost, or None unless every record has one."""
    vals = [r.get(key) for r in recs]
    return float(sum(vals)) if vals and all(v is not None for v in vals) else None


def summarize_variant(recs: list[dict], cfg: dict) -> dict:
    pool_key = f"recall@{cfg['retrieval']['top_k']}"
    metrics = [m for m in [pool_key] + RETRIEVAL + ANSWER if m in recs[0]]
    s = {"n": len(recs), **{m: float(np.mean([r[m] for r in recs])) for m in metrics}}
    lat = [r["rerank_latency_s"] for r in recs]
    s["rerank_p50_s"], s["rerank_p95_s"] = _pct(lat, 50), _pct(lat, 95)
    jev_tokens = sum(r["rerank_input_tokens"] for r in recs)
    billed = _billed(recs, "rerank_cost_usd") if jev_tokens else None
    s["rerank_cost_usd"] = billed if billed is not None else jev_tokens / 1e6 * cfg["pricing"]["jev_per_m"]
    if "gen_input_tokens" in recs[0]:
        g = cfg["generation"]
        tin = sum(r["gen_input_tokens"] for r in recs)
        tout = sum(r["gen_output_tokens"] for r in recs)
        s["ctx_tokens_mean"] = tin / len(recs)
        billed = _billed(recs, "gen_cost_usd")
        s["gen_cost_usd"] = billed if billed is not None else tin / 1e6 * g["price_in"] + tout / 1e6 * g["price_out"]
        s["gen_p50_s"] = _pct([r["gen_latency_s"] for r in recs], 50)
    return s


def compare(a: list[dict], b: list[dict]) -> dict:
    assert [r["qid"] for r in a] == [r["qid"] for r in b], "variants must share question order"
    out = {}
    for m in RETRIEVAL + ANSWER:
        if m not in a[0]:
            continue
        out[m] = paired_bootstrap([r[m] for r in a], [r[m] for r in b])
        if m in BINARY:
            out[m]["mcnemar_p"] = mcnemar([r[m] for r in a], [r[m] for r in b])["p"]
    return out


def by_type(recs: list[dict], metric: str) -> dict:
    types = sorted({r["type"] for r in recs})
    return {t: float(np.mean([r[metric] for r in recs if r["type"] == t])) for t in types}


def build_report(cfg: dict, split: str) -> Path:
    rdir: Path = cfg["paths"]["results_dir"]
    data = {v: read_jsonl(rdir / f"{split}_{v}.jsonl") for v in ("vanilla", "bge", "jev")
            if (rdir / f"{split}_{v}.jsonl").exists()}
    summary = {v: summarize_variant(recs, cfg) for v, recs in data.items()}
    comps = {f"{a}-{b}": compare(data[a], data[b]) for a, b in COMPARISONS if a in data and b in data}

    # CSV: one row per variant
    cols = sorted({k for s in summary.values() for k in s})
    with open(rdir / f"{split}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + cols)
        for v, s in summary.items():
            w.writerow([v] + [round(s.get(c, float("nan")), 4) for c in cols])
    (rdir / f"{split}_comparisons.json").write_text(json.dumps(comps, indent=2), encoding="utf-8")

    # Markdown
    pool_key = f"recall@{cfg['retrieval']['top_k']}"
    show = [pool_key] + RETRIEVAL + [m for m in ANSWER if any(m in s for s in summary.values())]
    lines = [f"# Results: {split} split (n={next(iter(summary.values()))['n']})", "",
             "## Quality", "", "| variant | " + " | ".join(show) + " |",
             "|---|" + "---|" * len(show)]
    for v, s in summary.items():
        lines.append(f"| {v} | " + " | ".join(f"{s.get(m, float('nan')):.3f}" for m in show) + " |")

    lines += ["", "## Paired differences (95% bootstrap CI)", "",
              "| comparison | metric | delta | 95% CI | p (bootstrap) | p (McNemar) |",
              "|---|---|---|---|---|---|"]
    for name, mets in comps.items():
        for m, r in mets.items():
            mc = f"{r['mcnemar_p']:.4f}" if "mcnemar_p" in r else ""
            lines.append(f"| {name} | {m} | {r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
                         f"| {r['p']:.4f} | {mc} |")

    lines += ["", "## By question type (both_gold@5)", "", "| variant | " +
              " | ".join(by_type(next(iter(data.values())), "both_gold@5")) + " |"]
    lines.append("|---|" + "---|" * len(by_type(next(iter(data.values())), "both_gold@5")))
    for v, recs in data.items():
        lines.append(f"| {v} | " + " | ".join(f"{x:.3f}" for x in by_type(recs, "both_gold@5").values()) + " |")

    lines += ["", "## Cost and latency", "",
              "| variant | rerank p50 (s) | rerank p95 (s) | rerank $ | ctx tokens/q | gen $ | gen p50 (s) |",
              "|---|---|---|---|---|---|---|"]
    for v, s in summary.items():
        lines.append(f"| {v} | {s['rerank_p50_s']:.3f} | {s['rerank_p95_s']:.3f} | {s['rerank_cost_usd']:.4f} "
                     f"| {s.get('ctx_tokens_mean', 0):.0f} | {s.get('gen_cost_usd', 0):.4f} "
                     f"| {s.get('gen_p50_s', 0):.3f} |")
    out = rdir / f"{split}_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
