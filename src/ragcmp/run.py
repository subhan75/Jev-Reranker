"""CLI entry point.

  python -m ragcmp.run prepare                        # sample HotpotQA, build corpus
  python -m ragcmp.run index                          # embed corpus into Chroma
  python -m ragcmp.run eval --split dev --limit 5     # retrieve, rerank, generate, write records
  python -m ragcmp.run report --split dev             # summary tables + significance tests
  python -m ragcmp.run all --split test               # index + eval + report
"""
from __future__ import annotations

import argparse
import json

from ragcmp.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "index", "eval", "report", "all"])
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--variants", default="vanilla,bge,jev")
    p.add_argument("--no-generate", action="store_true", help="retrieval metrics only")
    args = p.parse_args()
    cfg = load_config()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.command == "prepare":
        from ragcmp.data import prepare
        print(json.dumps(prepare(cfg), indent=2))
    if args.command in ("index", "all"):
        from ragcmp.pipeline import index_corpus
        print(json.dumps(index_corpus(cfg), indent=2))
    if args.command in ("eval", "all"):
        from ragcmp.pipeline import run_split
        out = run_split(cfg, args.split, variants, args.limit, generate=not args.no_generate)
        print("wrote:", ", ".join(str(p) for p in out.values()))
    if args.command in ("report", "eval", "all"):
        from ragcmp.report import build_report
        path = build_report(cfg, args.split)
        print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
