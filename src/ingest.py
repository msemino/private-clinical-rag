"""Ingest documents into a local ChromaDB collection.

The design choices that actually matter:

* **Hierarchical chunking.** Documents are split along their heading structure,
  so every chunk carries its full section path (e.g.
  ``"Anxiety Presentations > Panic Episodes > Assessment"``). A chunk that would
  lose its parent context is discarded rather than indexed half-blind.
* **Metadata enrichment.** Each chunk gets a "DNI" — source file, heading,
  section path, ordinal — enabling surgical metadata filters *before* the LLM
  ever sees the query.
* **Privacy by design.** Embeddings and storage are fully local. No bytes leave
  the machine.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence

import chromadb

from .embeddings import get_embedder

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MAX_CHARS = 1200


@dataclass
class Chunk:
    text: str
    source: str
    section_path: str
    heading: str
    ordinal: int


def _split_text(body: str, max_chars: int) -> List[str]:
    """Split a section body into <=max_chars pieces along paragraph boundaries."""
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    pieces: List[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def parse_markdown(path: str, max_chars: int = _MAX_CHARS) -> List[Chunk]:
    """Parse a Markdown file into context-preserving chunks."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    source = os.path.basename(path)
    stack: List[tuple[int, str]] = []  # (heading level, title)
    buffer: List[str] = []
    chunks: List[Chunk] = []
    ordinal = 0

    def flush() -> None:
        nonlocal buffer, ordinal
        body = "\n".join(buffer).strip()
        buffer = []
        if not body or not stack:
            # No body, or body with no heading parent -> drop (loses context).
            return
        section_path = " > ".join(title for _, title in stack)
        heading = stack[-1][1]
        for piece in _split_text(body, max_chars):
            chunks.append(Chunk(piece, source, section_path, heading, ordinal))
            ordinal += 1

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return chunks


def ingest(paths: Sequence[str], config) -> int:
    """Embed and store all chunks from ``paths``. Idempotent: rebuilds the
    collection from scratch so re-ingesting never duplicates documents."""
    embedder = get_embedder(config)
    client = chromadb.PersistentClient(path=config.chroma_path)

    try:
        client.delete_collection(config.collection)
    except Exception:
        pass
    collection = client.create_collection(
        config.collection, metadata={"hnsw:space": "cosine"}
    )

    chunks: List[Chunk] = []
    for path in paths:
        chunks.extend(parse_markdown(path))
    if not chunks:
        raise SystemExit("No chunks produced — check that your documents have headings.")

    embeddings = embedder.embed([c.text for c in chunks])
    collection.add(
        ids=[f"{c.source}::{c.ordinal}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": c.source,
                "section_path": c.section_path,
                "heading": c.heading,
                "ordinal": c.ordinal,
            }
            for c in chunks
        ],
    )
    return len(chunks)
