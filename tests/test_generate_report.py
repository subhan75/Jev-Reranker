import json
import random
from types import SimpleNamespace

from ragcmp.cache import Cache
from ragcmp.generate import Generator, build_input
from ragcmp.report import build_report

CFG_GEN = {"generation": {"model": "fake", "reasoning_effort": "none", "max_output_tokens": 16}}


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(message=SimpleNamespace(content=" Paris \n"))
        return SimpleNamespace(choices=[msg], model="fake-1",
                               usage=SimpleNamespace(prompt_tokens=50, completion_tokens=2, cost=0.00001))


def test_build_input_numbers_passages():
    s = build_input("Q?", [{"text": "a"}, {"text": "b"}])
    assert "[1] a" in s and "[2] b" in s and s.endswith("Question: Q?\nAnswer:")


def test_generator_params_and_cache(tmp_path):
    fake = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    g = Generator(CFG_GEN, Cache(tmp_path / "c.sqlite"), client=client)
    out, cached = g.answer("Q?", [{"text": "a"}])
    assert out["answer"] == "Paris" and not cached
    assert out["input_tokens"] == 50 and out["cost_usd"] == 0.00001
    call = fake.calls[0]
    assert call["extra_body"] == {"reasoning": {"effort": "none"}}
    assert call["max_tokens"] == 16
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    _, cached2 = g.answer("Q?", [{"text": "a"}])
    assert cached2 and len(fake.calls) == 1


def test_generator_retries_on_429_and_throttles(tmp_path, monkeypatch):
    import httpx
    from openai import RateLimitError

    sleeps = []
    monkeypatch.setattr("ragcmp.generate.time.sleep", lambda s: sleeps.append(s))
    resp429 = httpx.Response(429, request=httpx.Request("POST", "http://x"))

    class Flaky(FakeCompletions):
        def create(self, **kw):
            if len(self.calls) == 0:
                self.calls.append(kw)
                raise RateLimitError("rate limited", response=resp429, body=None)
            return super().create(**kw)

    fake = Flaky()
    cfg = {"generation": {**CFG_GEN["generation"], "max_rpm": 20, "retry_wait_s": 5}}
    g = Generator(cfg, Cache(tmp_path / "c.sqlite"), client=SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    out, _ = g.answer("Q?", [{"text": "a"}])
    assert out["answer"] == "Paris" and len(fake.calls) == 2
    assert 5 in sleeps  # backed off once after the 429
    assert g.min_interval == 3.0


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
