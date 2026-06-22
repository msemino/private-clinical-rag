"""Runtime configuration, driven entirely by environment variables.

Every value is local by default. Nothing here points at a third-party API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in {"1", "true", "True", "yes"}


@dataclass(frozen=True)
class Config:
    chroma_path: str = os.getenv("CHROMA_PATH", "./.chroma")
    collection: str = os.getenv("CHROMA_COLLECTION", "clinical_manual")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    top_k: int = int(os.getenv("TOP_K", "8"))
    final_k: int = int(os.getenv("FINAL_K", "4"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.15"))
    offline: bool = _flag("OFFLINE_EMBEDDINGS")


CONFIG = Config()
