"""Tiny persistent JSON cache in SQLite, keyed by sha256 of (namespace, payload)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable


def make_key(namespace: str, payload: Any) -> str:
    blob = json.dumps([namespace, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self.conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, json.dumps(value)))
        self.conn.commit()

    def get_or_compute(self, namespace: str, payload: Any, fn: Callable[[], Any]) -> tuple[Any, bool]:
        """Return (value, cached)."""
        key = make_key(namespace, payload)
        value = self.get(key)
        if value is not None:
            self.hits += 1
            return value, True
        self.misses += 1
        value = fn()
        self.set(key, value)
        return value, False

    def close(self) -> None:
        self.conn.close()
