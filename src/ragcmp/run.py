"""CLI entry point: python -m ragcmp.run <command> [--split dev|test] [--limit N]"""
from __future__ import annotations

import argparse
import json

from ragcmp.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare"])
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    cfg = load_config()

    if args.command == "prepare":
        from ragcmp.data import prepare
        print(json.dumps(prepare(cfg), indent=2))


if __name__ == "__main__":
    main()
