"""Command-line entry point.

    python -m src.ask ingest data/sample/clinical_handbook_sample.md
    python -m src.ask ask "How is a panic episode assessed?"
    python -m src.ask selftest        # fully offline, no Ollama required
"""
from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import replace
from typing import List

from .config import CONFIG
from .ingest import ingest
from .llm import REFUSAL, answer
from .retrieve import retrieve

SAMPLE_DOC = "data/sample/clinical_handbook_sample.md"


def _expand(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if not matched:
            print(f"warning: no files matched {pattern!r}", file=sys.stderr)
        paths.extend(matched)
    return sorted(set(paths))


def cmd_ingest(args) -> int:
    paths = _expand(args.paths)
    if not paths:
        print("error: no input documents found.", file=sys.stderr)
        return 1
    count = ingest(paths, CONFIG)
    print(f"Indexed {count} chunks from {len(paths)} document(s) into "
          f"collection {CONFIG.collection!r} at {CONFIG.chroma_path!r}.")
    return 0


def _print_answer(text: str, sources) -> None:
    print("\n" + text + "\n")
    if sources:
        print("Sources:")
        for src in sources:
            print(f"  [{src['tag']}] {src['section_path']} "
                  f"({src['source']}, score {src['score']})")


def cmd_ask(args) -> int:
    contexts = retrieve(args.question, CONFIG)
    text, sources = answer(args.question, contexts, CONFIG)
    _print_answer(text, sources)
    return 0


def cmd_selftest(_args) -> int:
    """End-to-end smoke test with the deterministic offline embedder."""
    cfg = replace(CONFIG, offline=True, chroma_path="./.chroma_selftest",
                  collection="selftest")
    n = ingest([SAMPLE_DOC], cfg)
    assert n > 0, "ingest produced no chunks"

    # A question that should resolve to the panic-assessment section.
    contexts = retrieve("how do you assess a panic episode", cfg)
    assert contexts, "retrieval returned nothing"
    top_path = contexts[0]["meta"]["section_path"].lower()
    assert "panic" in top_path, f"unexpected top section: {top_path!r}"

    text, sources = answer("how do you assess a panic episode", contexts, cfg)
    assert sources and sources[0]["tag"] == "S1", "answer did not cite a source"

    # A question with no support in the corpus must be refused, not guessed.
    off_q = "what is the capital of France"
    off_topic = retrieve(off_q, cfg)
    off_text, _ = answer(off_q, off_topic, cfg)
    assert off_text == REFUSAL, f"off-topic query was not refused: {off_text!r}"

    print(f"PASS: indexed {n} chunks")
    print(f"PASS: top section for panic query -> {contexts[0]['meta']['section_path']}")
    print(f"PASS: grounded answer cited {len(sources)} source(s)")
    print("PASS: off-topic query refused instead of hallucinating")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index one or more documents")
    p_ingest.add_argument("paths", nargs="+", help="files or globs to ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a grounded, cited question")
    p_ask.add_argument("question", help="natural-language question")
    p_ask.set_defaults(func=cmd_ask)

    p_self = sub.add_parser("selftest", help="run an offline end-to-end check")
    p_self.set_defaults(func=cmd_selftest)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
