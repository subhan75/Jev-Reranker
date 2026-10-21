import pytest

from ragcmp.clients import jev_client, openai_client

CFG = {
    "provider": {
        "api_key_env": "RAGCMP_TEST_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "jev_base_url": "https://openrouter.ai/api",
        "app_title": "ragcmp",
    },
    "rerank": {"jev": {"model": "typesafe/jev-1.14"}},
}


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("RAGCMP_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RAGCMP_TEST_KEY"):
        openai_client(CFG)


def test_openai_client_points_at_openrouter(monkeypatch):
    monkeypatch.setenv("RAGCMP_TEST_KEY", "sk-or-test")
    c = openai_client(CFG)
    assert str(c.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert c.api_key == "sk-or-test"


def test_jev_client_constructs(monkeypatch):
    monkeypatch.setenv("RAGCMP_TEST_KEY", "sk-or-test")
    assert jev_client(CFG) is not None
