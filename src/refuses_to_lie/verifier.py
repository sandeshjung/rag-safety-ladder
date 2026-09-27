"""Groundedness verification: for each cited claim in a generated answer,
check whether the excerpt it cites actually supports it.

Uses a different model (Groq) from the generator (Gemini) to answer a
focused SUPPORTED/UNSUPPORTED/PARTIAL question per (claim, cited excerpt)
pair — checking the generator's own citations against the source text it
says backs them, not re-answering the question from scratch.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from refuses_to_lie.config import RunConfig
from refuses_to_lie.generation import GeneratedAnswer, citation_ids
from refuses_to_lie.llm_client import call_groq
from refuses_to_lie.retrieval import Hit

# Parser version 1's claim pattern. It only recognised ids made of
# [\w.-], one per bracket, so claims citing GOV.UK pages (spaces and
# parentheses in the id) or grouped sources were never verified. Kept only
# so claims it did see keep their cached verdicts -- see _cache_key.
_LEGACY_CLAIM_RE = re.compile(r"([^.\n]*\[([\w.\-]+-\d{4})\][^.\n]*\.)")

# Sentence boundary: a full stop followed by whitespace, or a line break.
# A bare "." inside "£116.75", "GOV.UK" or "B1.1" is not a boundary.
_SENTENCE_BREAK_RE = re.compile(r"(?<=\.)\s+")

# Longer/more specific labels first: "SUPPORTED" is a substring of
# "UNSUPPORTED", so scanning in the other order would match "SUPPORTED"
# against an "UNSUPPORTED" response and get every unsupported claim backwards.
_VERDICT_CHECK_ORDER = ("UNSUPPORTED", "PARTIAL", "SUPPORTED")

_VERIFY_PROMPT = """Source excerpt:
{source}

Claim: "{claim}"

Does the source excerpt support this claim? Reply with exactly one word: \
SUPPORTED, UNSUPPORTED, or PARTIAL."""


@dataclass
class ClaimVerdict:
    claim_text: str
    chunk_id: str
    # SUPPORTED, UNSUPPORTED, PARTIAL, or UNKNOWN (cited chunk_id wasn't in context)
    verdict: str


@dataclass
class VerifiedAnswer:
    answer: GeneratedAnswer
    claim_verdicts: list[ClaimVerdict]

    @property
    def groundedness(self) -> float:
        """Fraction of claims marked SUPPORTED. 1.0 for an answer with no
        citation-bearing claims at all (nothing to contradict)."""
        if not self.claim_verdicts:
            return 1.0
        supported = sum(1 for v in self.claim_verdicts if v.verdict == "SUPPORTED")
        return supported / len(self.claim_verdicts)

    @property
    def all_supported(self) -> bool:
        return all(v.verdict == "SUPPORTED" for v in self.claim_verdicts)


def _split_claims(text: str) -> list[tuple[str, str]]:
    """(claim_sentence, chunk_id) pairs: one per source a sentence cites.

    A sentence citing two sources yields two pairs, each checked against
    its own excerpt. A citation standing alone after a full stop belongs to
    the sentence before it.
    """
    claims: list[tuple[str, list[str]]] = []
    previous = ""
    for line in text.split("\n"):
        for sentence in _SENTENCE_BREAK_RE.split(line):
            sentence = sentence.strip()
            ids = list(dict.fromkeys(citation_ids(sentence)))
            if ids and not _CITATION_ONLY_RE.sub("", sentence).strip(" .") and previous:
                if claims and claims[-1][0] == previous:
                    claims[-1][1].extend(i for i in ids if i not in claims[-1][1])
                else:
                    claims.append((previous, ids))
            elif ids:
                claims.append((sentence, ids))
            if sentence:
                previous = sentence
    return [(sentence, chunk_id) for sentence, ids in claims for chunk_id in ids]


_CITATION_ONLY_RE = re.compile(r"\[[^\[\]]*-\d{4}[^\[\]]*\]")


def _cache_key(
    prefix: str, claim: str, chunk_id: str, legacy: dict[tuple[str, str], int]
) -> str:
    """Content-addressed, so a cached verdict can never answer a different claim.

    Version 1 keyed verdicts by position ("claim3"). Positions shift once
    the parser finds more claims, and a positional key would then replay
    one claim's verdict for another. A claim the old parser saw at the same
    position, word for word, keeps its old key and cached verdict.
    """
    if (claim, chunk_id) in legacy:
        return f"{prefix}-claim{legacy[(claim, chunk_id)]}"
    digest = hashlib.sha256(f"{chunk_id}\n{claim}".encode()).hexdigest()[:12]
    return f"{prefix}-c{digest}"


def _parse_verdict(response: str) -> str:
    upper = response.strip().upper()
    for label in _VERDICT_CHECK_ORDER:
        if label in upper:
            return label
    return "UNKNOWN"


def verify_answer(
    answer: GeneratedAnswer,
    hits: list[Hit],
    config: RunConfig,
    cache_key_prefix: str | None = None,
) -> VerifiedAnswer:
    hits_by_id = {h.chunk.chunk_id: h for h in hits}
    verdicts: list[ClaimVerdict] = []
    legacy = {
        (m.group(1).strip(), m.group(2)): i
        for i, m in enumerate(_LEGACY_CLAIM_RE.finditer(answer.text))
    }

    for claim_text, chunk_id in _split_claims(answer.text):
        hit = hits_by_id.get(chunk_id)
        if hit is None:
            verdicts.append(ClaimVerdict(claim_text, chunk_id, "UNKNOWN"))
            continue

        prompt = _VERIFY_PROMPT.format(source=hit.chunk.text, claim=claim_text)
        cache_key = (
            _cache_key(cache_key_prefix, claim_text, chunk_id, legacy)
            if cache_key_prefix
            else None
        )
        response = call_groq(
            prompt, model=config.verifier_model, temperature=0.0, cache_key=cache_key
        )
        verdicts.append(ClaimVerdict(claim_text, chunk_id, _parse_verdict(response)))

    return VerifiedAnswer(answer=answer, claim_verdicts=verdicts)


def drop_unsupported_claims(verified: VerifiedAnswer) -> str:
    """Rung E's "drop_unsupported" verifier action: strip any sentence
    whose cited claim wasn't SUPPORTED, leaving the rest of the answer
    intact."""
    text = verified.answer.text
    # A sentence citing several sources stays if any one of them supports it.
    supported = {v.claim_text for v in verified.claim_verdicts if v.verdict == "SUPPORTED"}
    for claim in dict.fromkeys(v.claim_text for v in verified.claim_verdicts):
        if claim not in supported:
            text = text.replace(claim, "").strip()
    return text
