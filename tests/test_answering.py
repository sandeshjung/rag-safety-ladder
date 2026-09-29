from pathlib import Path

import pytest

from refuses_to_lie.answering import answer_question, retrieve_context
from refuses_to_lie.config import LADDER, A, B, C, E, F
from refuses_to_lie.corpus import chunk_document, load_body_pages
from refuses_to_lie.retrieval import Index

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


@pytest.fixture(scope="module")
def index() -> Index:
    chunks = chunk_document(
        "a36", load_body_pages(CORPUS / "a36-employment-break-policy-april-26.pdf")
    )
    return Index(chunks)


def test_retrieval_axis_selects_a_different_ranker(index: Index):
    # A and B differ only on the retrieval axis, so if the orchestrator were
    # ignoring config.retrieval this would silently return identical hits and
    # the A->B rung of the ablation would measure nothing.
    question = "how long must someone work here before taking a career break"
    dense = retrieve_context(index, question, A)
    hybrid = retrieve_context(index, question, B)

    assert [h.chunk.chunk_id for h in dense] != [h.chunk.chunk_id for h in hybrid] or [
        h.score for h in dense
    ] != [h.score for h in hybrid]


def test_context_is_narrowed_to_top_k_context(index: Index):
    hits = retrieve_context(index, "career break eligibility", A)
    assert len(hits) <= A.top_k_context
    assert A.top_k_context < A.top_k_retrieve


def test_rerank_rung_reorders_without_changing_the_candidate_set(index: Index):
    # Reranking must reorder the shortlist, not fetch from the corpus again:
    # rung C can only improve on what rung B already surfaced, and that
    # ceiling is what makes B->C a meaningful thing to measure separately.
    question = "how long must someone work here before taking a career break"
    before = retrieve_context(index, question, B)
    after = retrieve_context(index, question, C)

    shortlist = {h.chunk.chunk_id for h in index.search_hybrid(question, k=B.top_k_retrieve)}
    assert {h.chunk.chunk_id for h in after} <= shortlist
    assert len(after) == len(before)
    # Scores come from the cross-encoder now, so they are on a different
    # scale entirely (unbounded logits, often negative) rather than RRF sums.
    assert [h.score for h in after] != [h.score for h in before]
    assert after == sorted(after, key=lambda h: h.score, reverse=True)


def test_abstention_without_a_verifier_is_rejected(index: Index):
    from dataclasses import replace

    broken = replace(F, verifier="off")
    with pytest.raises(ValueError, match="verifier"):
        answer_question("anything", index, broken)


def test_ladder_rungs_turn_stages_on_cumulatively():
    # The orchestrator reads these flags directly, so this pins the shape it
    # depends on: once a stage is on it stays on for every later rung.
    assert A.retrieval == "dense" and B.retrieval == "hybrid"
    assert not A.require_citations and LADDER[3].require_citations
    assert E.verifier == "drop_unsupported" and F.verifier == "drop_unsupported"
    assert not E.abstain and F.abstain
    assert F.require_citations, "abstention measures citation agreement, so D must stay on"


# --- extension rungs G and H -----------------------------------------------
# These stub the two LLM calls, so they verify the orchestration -- what is
# called, when, and what reaches the context -- without spending quota.


class _Calls:
    def __init__(self) -> None:
        self.generate = 0
        self.gate = 0


def _stub_llms(monkeypatch, verdict: str) -> _Calls:
    from refuses_to_lie import answering
    from refuses_to_lie.answerability import SCORES, AnswerabilitySignal
    from refuses_to_lie.generation import GeneratedAnswer

    calls = _Calls()

    def fake_gate(question, hits, config, cache_key=None):
        calls.gate += 1
        return AnswerabilitySignal(verdict, SCORES.get(verdict, 0.0))

    def fake_generate(question, hits, config, cache_key=None):
        calls.generate += 1
        return GeneratedAnswer(question, "An answer.", [], [h.chunk.chunk_id for h in hits])

    monkeypatch.setattr(answering, "check_answerability", fake_gate)
    monkeypatch.setattr(answering, "generate_answer", fake_generate)
    monkeypatch.setattr(answering, "verify_answer", lambda draft, hits, config, **_: None)
    return calls


def test_gate_refusal_costs_no_generation(index: Index, monkeypatch):
    # The point of asking before generating: a "no" spends nothing on the
    # generator, which is the rate-limited provider.
    from dataclasses import replace

    from refuses_to_lie.config import G

    calls = _stub_llms(monkeypatch, "UNANSWERABLE")
    answer = answer_question("anything", index, replace(G, verifier="off"))

    assert answer.abstained
    assert calls.gate == 1
    assert calls.generate == 0
    assert answer.confidence is not None and answer.confidence.composite == 0.0


def test_partial_context_is_refused_at_rung_gs_threshold(index: Index, monkeypatch):
    # The near-miss case: the topic is covered, the specific point is not.
    from dataclasses import replace

    from refuses_to_lie.config import G

    calls = _stub_llms(monkeypatch, "PARTIAL")
    answer = answer_question("anything", index, replace(G, verifier="off"))
    assert answer.abstained and calls.generate == 0


def test_answerable_context_is_generated_once_not_sampled(index: Index, monkeypatch):
    # Rung F samples three times to measure agreement; G drops agreement,
    # so it must not keep paying for it.
    from dataclasses import replace

    from refuses_to_lie.config import G

    calls = _stub_llms(monkeypatch, "ANSWERABLE")
    answer = answer_question("anything", index, replace(G, verifier="off"))

    assert not answer.abstained
    assert calls.generate == 1


def test_answerability_rung_does_not_require_a_verifier(index: Index, monkeypatch):
    # The verifier requirement exists because groundedness feeds the
    # composite; the answerability gate does not use it.
    from dataclasses import replace

    from refuses_to_lie.config import G

    _stub_llms(monkeypatch, "ANSWERABLE")
    answer_question("anything", index, replace(G, verifier="off"))  # must not raise


def test_provenance_filter_keeps_only_registered_documents(index: Index):
    from dataclasses import replace

    trusted = frozenset({"a36"})
    open_hits = retrieve_context(index, "career break eligibility", B)
    guarded = retrieve_context(
        index,
        "career break eligibility",
        replace(B, require_provenance=True),
        trusted_docs=trusted,
    )
    assert {h.chunk.doc_id for h in guarded} <= trusted
    assert len(guarded) == len(open_hits)  # a36 is registered, so nothing is lost


def test_an_unregistered_corpus_leaves_nothing_to_answer_from(index: Index, monkeypatch):
    # Every hit filtered out must mean a refusal, reached without asking
    # either model -- not a crash inside generate_answer.
    from dataclasses import replace

    from refuses_to_lie.config import H

    calls = _stub_llms(monkeypatch, "ANSWERABLE")
    answer = answer_question(
        "career break eligibility", index, replace(H, verifier="off"), trusted_docs=frozenset()
    )
    assert answer.abstained
    assert answer.hits == []
    assert calls.gate == 0 and calls.generate == 0


def test_provenance_without_a_register_refuses_to_run(index: Index):
    # A provenance rung that silently skipped its check would report a
    # defence it never applied.
    from dataclasses import replace

    with pytest.raises(ValueError, match="trusted_docs"):
        retrieve_context(index, "anything", replace(B, require_provenance=True))


def test_extension_rungs_build_on_the_full_system():
    from refuses_to_lie.config import G, H

    assert G.confidence_source == "answerability" and G.abstain
    assert H.require_provenance and H.confidence_source == "answerability"
    assert not F.require_provenance and F.confidence_source == "composite"


def test_superseded_filter_drops_old_versions_and_backfills(index: Index):
    from dataclasses import replace

    open_hits = retrieve_context(index, "career break eligibility", B)
    stale = frozenset({"a36"})
    filtered = retrieve_context(
        index,
        "career break eligibility",
        replace(B, exclude_superseded=True),
        superseded_docs=stale,
    )
    assert open_hits and not filtered  # the only document here is marked stale
    kept = retrieve_context(
        index,
        "career break eligibility",
        replace(B, exclude_superseded=True),
        superseded_docs=frozenset({"some-other-doc"}),
    )
    assert [h.chunk.chunk_id for h in kept] == [h.chunk.chunk_id for h in open_hits]


def test_superseded_filter_without_a_version_record_refuses_to_run(index: Index):
    from dataclasses import replace

    with pytest.raises(ValueError, match="superseded_docs"):
        retrieve_context(index, "anything", replace(B, exclude_superseded=True))


def test_prefer_current_rung_builds_on_provenance():
    from refuses_to_lie.config import H, I

    assert I.exclude_superseded and I.require_provenance
    assert not H.exclude_superseded


def test_wider_context_rung_changes_only_the_context_size():
    from dataclasses import fields

    from refuses_to_lie.config import I, J

    changed = {f.name for f in fields(I) if getattr(I, f.name) != getattr(J, f.name)}
    assert changed == {"id", "label", "top_k_context"}
    assert J.top_k_context > I.top_k_context
