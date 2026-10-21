"""Answer generation over the top-k reranked passages via OpenRouter chat completions (cached)."""
from __future__ import annotations

import time

from openai import RateLimitError

from ragcmp.cache import Cache

INSTRUCTIONS = (
    "You answer multi-hop questions using only the provided passages. "
    "Reply with the shortest possible answer span (a name, date, number, short phrase, or yes/no), "
    "with no explanation and no full sentence. "
    "If the passages do not contain the answer, reply exactly: unknown"
)


def build_input(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    return f"Passages:\n{ctx}\n\nQuestion: {question}\nAnswer:"


class Generator:
    def __init__(self, cfg: dict, cache: Cache, client=None):
        g = cfg["generation"]
        self.cfg = cfg
        self.model = g["model"]
        self.reasoning_effort = g.get("reasoning_effort")
        self.max_output_tokens = g["max_output_tokens"]
        self.cache = cache
        self._client = client
        # OpenRouter limits new accounts (20 req/min on gpt-6-luna); pace calls and retry on 429.
        self.min_interval = 60.0 / g["max_rpm"] if g.get("max_rpm") else 0.0
        self.max_retries = g.get("max_retries", 6)
        self.retry_wait_s = g.get("retry_wait_s", 15.0)
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _create_with_retry(self, **kwargs):
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_wait_s * (attempt + 1))

    @property
    def client(self):
        if self._client is None:
            from ragcmp.clients import openai_client
            self._client = openai_client(self.cfg)
        return self._client

    def _call(self, prompt: str) -> dict:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_output_tokens,
        )
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
        t0 = time.perf_counter()
        resp = self._create_with_retry(**kwargs)
        cost = getattr(resp.usage, "cost", None)
        return {
            "answer": (resp.choices[0].message.content or "").strip(),
            "latency_s": time.perf_counter() - t0,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "cost_usd": float(cost) if cost is not None else None,
            "model": resp.model,
        }

    def answer(self, question: str, passages: list[dict]) -> tuple[dict, bool]:
        prompt = build_input(question, passages)
        payload = {"instructions": INSTRUCTIONS, "input": prompt, "effort": self.reasoning_effort,
                   "max_out": self.max_output_tokens}
        return self.cache.get_or_compute("gen:" + self.model, payload, lambda: self._call(prompt))
