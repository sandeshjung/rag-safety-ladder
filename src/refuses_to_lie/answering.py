"""The whole system in one readable pass: question in, Answer out.

Every other module here does one job well — retrieval ranks chunks,
generation writes text, verifier checks citations, confidence scores
them — but until now nothing showed how they fit together, so reading any
one of them told you very little about what actually happens to a
question. This module is that missing seam.

It is also the only place that reads the RunConfig ladder axes. Each rung
turns on one more stage:

    A  dense retrieval, answer, done
    B  hybrid retrieval (BM25 + dense via RRF)
    C  + reranking
    D  + mandatory inline citations
    E  + groundedness verification, dropping unsupported claims
    F  + calibrated abstention on the composite confidence signal

and two extension rungs, built from what the A-F grid measured:

    G  abstention gated on answerability instead of the composite
    H  + only registered documents may reach the context
    I  + a superseded document is dropped when its replacement is present
    J  + ten passages in the context instead of six

Because the rungs are cumulative, a config is a set of flags rather than a
branch, and adding a rung means adding a stage here rather than forking
the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from refuses_to_lie.answerability import AnswerabilitySignal, check_answerability
from refuses_to_lie.confidence import (
    AgreementResult,
    ConfidenceSignal,
    compute_confidence,
    sample_agreement,
    should_abstain,
)
from refuses_to_lie.config import RunConfig
from refuses_to_lie.generation import (
    ABSTAIN_PHRASE,
    Citation,
    GeneratedAnswer,
    generate_answer,
)
from refuses_to_lie.rerank import Reranker, get_reranker
from refuses_to_lie.retrieval import Hit, Index
from refuses_to_lie.verifier import (
    ClaimVerdict,
    VerifiedAnswer,
    drop_unsupported_claims,
    verify_answer,
)


@dataclass
class Answer:
    """What one config produced for one question, with the intermediate
    signals kept so the eval harness can score *why* an answer looked the
    way it did, not just what it said."""

    question: str
    config_id: str
    text: str
    hits: list[Hit]
    citations: list[Citation]
    verdicts: list[ClaimVerdict]
    confidence: ConfidenceSignal | AnswerabilitySignal | None
    abstained: bool

    @property
    def cited_labels(self) -> list[str]:
        return [c.cite_label for c in self.citations]


def retrieve_context(
    index: Index,
    question: str,
    config: RunConfig,
    reranker: Reranker | None = None,
    trusted_docs: frozenset[str] | None = None,
    superseded_docs: frozenset[str] | None = None,
) -> list[Hit]:
    """Rungs A-C: retrieve a wide shortlist, optionally rerank it, then
    narrow to the excerpts that actually go in the prompt.

    The two-stage width matters for rung C specifically. Reranking can only
    reorder what the first stage returned, so retrieving top_k_retrieve and
    narrowing to top_k_context afterwards is what gives the cross-encoder
    room to promote a chunk the fast rankers put at rank 15. Narrowing
    first would leave it nothing to do.
    """
    if config.retrieval == "hybrid":
        hits = index.search_hybrid(question, k=config.top_k_retrieve, rrf_k=config.rrf_k)
    else:
        hits = index.search_dense(question, k=config.top_k_retrieve)

    if config.rerank:
        hits = (reranker or get_reranker(config.reranker_model)).rerank(question, hits)

    if config.require_provenance:
        if trusted_docs is None:
            # Refuse rather than run unprotected: a provenance rung that
            # silently skipped its check would report a defence it never had.
            raise ValueError("require_provenance=True needs trusted_docs from the register")
        # Filter before narrowing, so the context backfills from the wider
        # shortlist with registered documents instead of just shrinking.
        hits = [h for h in hits if h.chunk.doc_id in trusted_docs]

    if config.exclude_superseded:
        if superseded_docs is None:
            raise ValueError("exclude_superseded=True needs superseded_docs from the versions")
        # Same place and reason as provenance: the current version's chunks
        # backfill from the shortlist rather than the context shrinking.
        hits = [h for h in hits if h.chunk.doc_id not in superseded_docs]

    return hits[: config.top_k_context]


def _draft_answer(
    question: str, hits: list[Hit], config: RunConfig, cache_key: str | None
) -> tuple[GeneratedAnswer, AgreementResult | None]:
    """Rung F needs several samples to measure agreement; every other rung
    needs one. Reuse the first sample as the answer rather than paying for
    an extra generation that would say the same thing."""
    if not config.abstain:
        return generate_answer(question, hits, config, cache_key=cache_key), None

    agreement = sample_agreement(question, hits, config, cache_key_prefix=cache_key)
    return agreement.samples[0], agreement


def _refusal(
    question: str,
    config: RunConfig,
    hits: list[Hit],
    confidence: AnswerabilitySignal | None = None,
) -> Answer:
    return Answer(
        question=question,
        config_id=config.id,
        text=ABSTAIN_PHRASE,
        hits=hits,
        citations=[],
        verdicts=[],
        confidence=confidence,
        abstained=True,
    )


def answer_question(
    question: str,
    index: Index,
    config: RunConfig,
    cache_key: str | None = None,
    reranker: Reranker | None = None,
    trusted_docs: frozenset[str] | None = None,
    superseded_docs: frozenset[str] | None = None,
) -> Answer:
    uses_composite = config.abstain and config.confidence_source == "composite"
    if uses_composite and config.verifier == "off":
        raise ValueError(
            "abstain=True needs a verifier: groundedness is one of the three inputs "
            "to the confidence signal, and without it the score is not comparable "
            "to a threshold calibrated with it."
        )

    hits = retrieve_context(
        index,
        question,
        config,
        reranker=reranker,
        trusted_docs=trusted_docs,
        superseded_docs=superseded_docs,
    )
    if not hits:
        # Provenance can empty the context entirely. Nothing trustworthy to
        # answer from is the textbook case for refusing.
        return _refusal(question, config, hits)

    gate: AnswerabilitySignal | None = None
    if config.abstain and config.confidence_source == "answerability":
        # Asked before generating, so a refusal costs no generator call --
        # the constrained resource on this project.
        gate_key = f"{cache_key}-answerability" if cache_key else None
        gate = check_answerability(question, hits, config, cache_key=gate_key)
        if should_abstain(gate, config.abstain_threshold):
            return _refusal(question, config, hits, confidence=gate)
        draft = generate_answer(question, hits, config, cache_key=cache_key)
        agreement = None
    else:
        draft, agreement = _draft_answer(question, hits, config, cache_key)

    verified: VerifiedAnswer | None = None
    if config.verifier != "off":
        verified = verify_answer(draft, hits, config, cache_key_prefix=cache_key)

    text = draft.text
    if verified is not None and config.verifier == "drop_unsupported":
        text = drop_unsupported_claims(verified)

    abstained = draft.abstained
    confidence: ConfidenceSignal | AnswerabilitySignal | None = gate
    if uses_composite:
        # Both are non-None here: agreement comes from _draft_answer whenever
        # abstain is set, and verified from the guard at the top of this function.
        assert agreement is not None and verified is not None
        confidence = compute_confidence(hits, verified, agreement)
        if should_abstain(confidence, config.abstain_threshold):
            abstained = True
            text = ABSTAIN_PHRASE

    return Answer(
        question=question,
        config_id=config.id,
        text=text,
        hits=hits,
        citations=draft.citations,
        verdicts=verified.claim_verdicts if verified else [],
        confidence=confidence,
        abstained=abstained,
    )
