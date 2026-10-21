"""API clients. Everything goes through OpenRouter with a single OPENROUTER_API_KEY."""
from __future__ import annotations

import os


def _api_key(cfg: dict) -> str:
    env = cfg["provider"]["api_key_env"]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"{env} is not set. Add it to .env (see .env.example).")
    return key


def openai_client(cfg: dict):
    """OpenAI SDK pointed at OpenRouter's OpenAI-compatible API (chat + embeddings)."""
    from openai import OpenAI

    p = cfg["provider"]
    return OpenAI(
        base_url=p["base_url"],
        api_key=_api_key(cfg),
        default_headers={"X-OpenRouter-Title": p.get("app_title", "ragcmp")},
    )


def jev_client(cfg: dict):
    """TypeSafe SDK pointed at OpenRouter; the SDK appends /v1/systemone to base_url."""
    from typesafe_sdk import TypeSafeClient

    return TypeSafeClient(
        api_key=_api_key(cfg),
        base_url=cfg["provider"]["jev_base_url"],
        model=cfg["rerank"]["jev"]["model"],
    )
