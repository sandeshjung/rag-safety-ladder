"""Answer generation, with citations gated on config.require_citations
(the ladder's rung D axis).

When citations are required, the model is told to cite every claim inline
using the excerpt's chunk_id in square brackets, e.g. "Staff accrue 27 days
of leave [a36-employment-break-policy-april-26-0006]." We parse those
bracketed IDs back out rather than trust the model to also reproduce a
human-readable label — chunk_id is a stable token the model is unlikely to
mangle, and we resolve it to Chunk.cite_label ourselves for display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from refuses_to_lie.config import RunConfig
from refuses_to_lie.llm_client import call_gemini
from refuses_to_lie.retrieval import Hit

# A bracket holding one or more chunk ids, comma-separated. Ids are matched
# loosely on purpose: GOV.UK doc ids carry spaces and parentheses
# ("Print Statutory Sick Pay (SSP) - GOV.UK-0001"), and the model often
# groups sources as "[id-0001, id-0002]". Version 1 of this parser allowed
# only [\w.-] and one id per bracket, so every GOV.UK citation and every
# grouped one was silently dropped -- unverified at rung E and invisible to
# the confidence score. Whatever is extracted is still checked against the
# ids actually in context before it counts.
_CITATION_BLOCK_RE = re.compile(r"\[([^\[\]]*?-\d{4}(?:\s*,\s*[^\[\]]*?-\d{4})*)\]")


def citation_ids(text: str) -> list[str]:
    """Every chunk id cited in `text`, in order, duplicates kept."""
    return [
        part.strip()
        for block in _CITATION_BLOCK_RE.findall(text)
        for part in block.split(",")
        if part.strip()
    ]


ABSTAIN_PHRASE = "I don't have enough information in the provided documents to answer this."

_PROMPT_WITH_CITATIONS = """You are answering a question using ONLY the excerpts below, \
taken from policy documents. Do not use outside knowledge.

Rules:
- Every factual claim you make must end with a citation in square brackets using the \
excerpt's ID, e.g. "Staff accrue 27 days of leave [{example_id}]."
- If the excerpts do not contain enough information to answer, reply with exactly: \
"{abstain_phrase}"
- Do not guess or fill gaps with plausible-sounding text.

Excerpts:
{excerpts}

Question: {question}

Answer:"""

_PROMPT_WITHOUT_CITATIONS = """You are answering a question using ONLY the excerpts below, \
taken from policy documents. Do not use outside knowledge.

If the excerpts do not contain enough information to answer, reply with exactly: \
"{abstain_phrase}"
Do not guess or fill gaps with plausible-sounding text.

Excerpts:
{excerpts}

Question: {question}

Answer:"""


@dataclass
class Citation:
    chunk_id: str
    cite_label: str


@dataclass
class GeneratedAnswer:
    question: str
    text: str
    citations: list[Citation]
    context_chunk_ids: list[str]

    @property
    def abstained(self) -> bool:
        return self.text.strip() == ABSTAIN_PHRASE


def format_excerpts(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.chunk.chunk_id}] ({h.chunk.cite_label})\n{h.chunk.text}" for h in hits
    )


def _extract_citations(text: str, hits_by_id: dict[str, Hit]) -> list[Citation]:
    seen: list[str] = []
    for chunk_id in citation_ids(text):
        if chunk_id in hits_by_id and chunk_id not in seen:
            seen.append(chunk_id)
    return [
        Citation(chunk_id=cid, cite_label=hits_by_id[cid].chunk.cite_label) for cid in seen
    ]


def generate_answer(
    question: str,
    hits: list[Hit],
    config: RunConfig,
    cache_key: str | None = None,
) -> GeneratedAnswer:
    if not hits:
        raise ValueError("generate_answer requires at least one retrieved chunk")

    hits_by_id = {h.chunk.chunk_id: h for h in hits}
    excerpts = format_excerpts(hits)

    if config.require_citations:
        prompt = _PROMPT_WITH_CITATIONS.format(
            example_id=hits[0].chunk.chunk_id,
            abstain_phrase=ABSTAIN_PHRASE,
            excerpts=excerpts,
            question=question,
        )
    else:
        prompt = _PROMPT_WITHOUT_CITATIONS.format(
            abstain_phrase=ABSTAIN_PHRASE,
            excerpts=excerpts,
            question=question,
        )

    text = call_gemini(
        prompt,
        model=config.generator_model,
        temperature=config.temperature,
        cache_key=cache_key,
    )
    citations = _extract_citations(text, hits_by_id) if config.require_citations else []
    return GeneratedAnswer(
        question=question,
        text=text,
        citations=citations,
        context_chunk_ids=list(hits_by_id.keys()),
    )
