"""Answer generation with citations and a hard refusal path.

Two rules keep this honest in a high-stakes domain:

1. **Answer only from retrieved context.** The prompt forbids outside knowledge
   and requires inline ``[S1]``, ``[S2]`` citations.
2. **Refuse instead of guessing.** If the best retrieved chunk scores below
   ``MIN_SCORE``, we return a refusal rather than let the model hallucinate.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import requests

PROMPT_TEMPLATE = """You are a careful reference assistant for a clinical manual.
Answer the QUESTION using ONLY the CONTEXT below. Do not use outside knowledge.
Cite every claim with the matching [S#] tag. If the context does not contain the
answer, reply exactly: "I can't find this in the provided manual."
Do not give personalized medical advice; you summarize the manual only.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with [S#] citations):"""

REFUSAL = "I can't find this in the provided manual."


def _ollama_chat(prompt: str, config) -> str:
    resp = requests.post(
        f"{config.ollama_host.rstrip('/')}/api/generate",
        json={"model": config.llm_model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def answer(question: str, contexts: List[Dict], config) -> Tuple[str, List[Dict]]:
    """Return ``(answer_text, cited_sources)``."""
    if not contexts or contexts[0]["score"] < config.min_score:
        return REFUSAL, []

    sources: List[Dict] = []
    blocks: List[str] = []
    for i, ctx in enumerate(contexts, 1):
        tag = f"S{i}"
        sources.append({"tag": tag, **ctx["meta"], "score": round(ctx["score"], 3)})
        blocks.append(f"[{tag}] ({ctx['meta']['section_path']})\n{ctx['text']}")
    context_text = "\n\n".join(blocks)

    if config.offline:
        # No LLM available offline: return an extractive answer from the top hit,
        # still citing its source so the contract (grounded + cited) holds.
        top = contexts[0]
        text = (
            f"(offline extractive mode) From {top['meta']['section_path']} [S1]:\n\n"
            f"{top['text']}"
        )
        return text, sources

    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    return _ollama_chat(prompt, config), sources
