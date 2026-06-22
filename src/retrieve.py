"""Retrieval = vector search + a transparent re-ranking pass.

Vector similarity alone is noisy. We pull ``top_k`` candidates from ChromaDB,
then re-rank with a second, explainable signal that blends:

* cosine similarity from the embedder,
* lexical overlap between the question and the chunk,
* a small boost when the question hits the chunk's *heading*.

No heavyweight cross-encoder dependency — the re-ranker is auditable and runs
in microseconds. Swap in a cross-encoder later if your domain needs it; the
interface stays the same.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import chromadb

from .embeddings import get_embedder, tokenize


def _rerank(question: str, hits: List[Dict]) -> List[Dict]:
    q_tokens = set(tokenize(question))
    for hit in hits:
        c_tokens = set(tokenize(hit["text"]))
        overlap = len(q_tokens & c_tokens) / (len(q_tokens) or 1)
        heading_tokens = set(tokenize(hit["meta"].get("heading", "")))
        heading_boost = 0.10 if (q_tokens & heading_tokens) else 0.0
        hit["score"] = 0.65 * hit["vec_score"] + 0.35 * overlap + heading_boost
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def retrieve(
    question: str, config, where: Optional[Dict] = None
) -> List[Dict]:
    """Return the ``final_k`` best-scoring chunks for ``question``.

    ``where`` is an optional ChromaDB metadata filter, e.g.
    ``{"source": "handbook.md"}`` — applied *before* ranking.
    """
    embedder = get_embedder(config)
    client = chromadb.PersistentClient(path=config.chroma_path)
    collection = client.get_collection(config.collection)

    q_emb = embedder.embed([question])[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=config.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "meta": meta, "vec_score": 1.0 - dist})

    return _rerank(question, hits)[: config.final_k]
