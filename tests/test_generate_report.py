import json
import random
from types import SimpleNamespace

from ragcmp.cache import Cache
from ragcmp.generate import Generator, build_input
from ragcmp.report import build_report

CFG_GEN = {"generation": {"model": "fake", "reasoning_effort": "none", "max_output_tokens": 16}}


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(output_text=" Paris \n", model="fake-1",
                               usage=SimpleNamespace(input_tokens=50, output_tokens=2))


def test_build_input_numbers_passages():
    s = build_input("Q?", [{"text": "a"}, {"text": "b"}])
    assert "[1] a" in s and "[2] b" in s and s.endswith("Question: Q?\nAnswer:")


def test_generator_params_and_cache(tmp_path):
    fake = FakeResponses()
    g = Generator(CFG_GEN, Cache(tmp_path / "c.sqlite"), client=SimpleNamespace(responses=fake))
    out, cached = g.answer("Q?", [{"text": "a"}])
    assert out["answer"] == "Paris" and not cached
    assert fake.calls[0]["reasoning"] == {"effort": "none"}
    assert fake.calls[0]["max_output_tokens"] == 16
    _, cached2 = g.answer("Q?", [{"text": "a"}])
    assert cached2 and len(fake.calls) == 1


def _records(variant_quality: float, n: int = 40, seed: int = 0):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        hit = float(rng.random() < variant_quality)
        recs.append({
            "qid": f"q{i}", "type": "bridge" if i % 2 else "comparison", "level": "hard",
            "recall@30": 1.0, "recall@1": hit, "recall@3": hit, "recall@5": hit, "mrr": hit,
            "ndcg@5": hit, "both_gold@5": hit, "rerank_latency_s": 0.1, "rerank_input_tokens": 1000,
            "em": hit, "f1": hit, "gen_input_tokens": 500, "gen_output_tokens": 3, "gen_latency_s": 0.2,
        })
    return recs


def test_report_end_to_end(tmp_path):
    cfg = {
        "paths": {"results_dir": tmp_path},
        "retrieval": {"top_k": 30},
        "pricing": {"jev_per_m": 0.042},
        "generation": {"price_in": 0.1, "price_out": 0.5},
    }
    for seed, (v, qual) in enumerate([("vanilla", 0.3), ("bge", 0.6), ("jev", 0.7)]):
        with open(tmp_path / f"dev_{v}.jsonl", "w") as f:
            for r in _records(qual, seed=seed):
                f.write(json.dumps(r) + "\n")
    md = build_report(cfg, "dev").read_text()
    assert "| jev-bge | both_gold@5 |" in md
    assert "## Cost and latency" in md
    assert (tmp_path / "dev_summary.csv").exists()
    comps = json.loads((tmp_path / "dev_comparisons.json").read_text())
    assert set(comps) == {"jev-bge", "bge-vanilla", "jev-vanilla"}
