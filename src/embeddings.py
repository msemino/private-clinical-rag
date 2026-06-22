"""Embedding providers.

Two backends, one interface:

* ``OllamaEmbedder`` — production. Calls a local Ollama instance
  (default model ``nomic-embed-text``). Embeddings are computed on hardware you
  own; no bytes leave the machine.
* ``HashEmbedder`` — offline. A deterministic bag-of-words embedder used by the
  self-test and CI so the whole pipeline can run end-to-end with zero external
  services. It is intentionally simple — good enough to prove retrieval works,
  not a substitute for a real embedding model.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

import requests

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small bilingual stopword set. Dropping these sharpens both the offline
# bag-of-words embedder and the lexical re-ranking signal, so that off-topic
# queries score low enough to hit the refusal path instead of matching on
# filler words like "what / is / the / of".
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from how in into is it its of on or that the
    this to was what when where which who why with you your do does did done
    el la los las un una unos unas de del y o que en con por para como es son
    cual cuales donde cuando quien porque se su sus al lo
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer (stopwords removed) shared by the
    offline embedder and re-ranking."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashEmbedder:
    """Deterministic, offline embedder. Hashes tokens into a fixed-dim vector."""

    def __init__(self, dim: int = 4096) -> None:
        # A wide vector keeps token collisions rare, so genuinely off-topic
        # queries score near zero and reach the refusal path.
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        return _l2_normalize(vec)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class OllamaEmbedder:
    """Local embeddings via Ollama's ``/api/embeddings`` endpoint."""

    def __init__(self, host: str, model: str) -> None:
        self.host = host.rstrip("/")
        self.model = model

    def _embed_one(self, text: str) -> List[float]:
        resp = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


def get_embedder(config) -> "HashEmbedder | OllamaEmbedder":
    if config.offline:
        return HashEmbedder()
    return OllamaEmbedder(config.ollama_host, config.embed_model)
