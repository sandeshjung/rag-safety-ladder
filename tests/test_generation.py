from pathlib import Path

import pytest

from refuses_to_lie.config import A, D
from refuses_to_lie.corpus import chunk_document, load_body_pages
from refuses_to_lie.generation import GeneratedAnswer, generate_answer
from refuses_to_lie.retrieval import Index

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


@pytest.fixture(scope="module")
def index() -> Index:
    chunks = chunk_document(
        "UHN-PO-HR24",
        load_body_pages(CORPUS / "uhn-annual-leave-for-agenda-for-change-staff000100pdf.pdf"),
    )
    return Index(chunks)


@pytest.mark.live
def test_generate_answer_with_citations(index: Index):
    hits = index.search_hybrid("Can unused annual leave be carried into the next year?", k=4)

    answer = generate_answer(
        "Can unused annual leave be carried into the next year?",
        hits,
        D,
        cache_key="test-generation-with-citations",
    )

    assert answer.text.strip()
    assert answer.citations
    context_ids = set(answer.context_chunk_ids)
    for citation in answer.citations:
        assert citation.chunk_id in context_ids
        assert citation.cite_label.startswith("UHN-PO-HR24")


@pytest.mark.live
def test_generate_answer_without_citations_extracts_none(index: Index):
    hits = index.search_hybrid("Can unused annual leave be carried into the next year?", k=4)

    answer = generate_answer(
        "Can unused annual leave be carried into the next year?",
        hits,
        A,
        cache_key="test-generation-without-citations",
    )

    assert answer.text.strip()
    assert answer.citations == []


def test_generate_answer_requires_at_least_one_hit(index: Index):
    with pytest.raises(ValueError):
        generate_answer("anything", [], D)


def test_abstained_property():
    abstained = GeneratedAnswer(
        question="q",
        text="I don't have enough information in the provided documents to answer this.",
        citations=[],
        context_chunk_ids=[],
    )
    answered = GeneratedAnswer(
        question="q", text="Yes, per section 4.8.", citations=[], context_chunk_ids=[]
    )

    assert abstained.abstained is True
    assert answered.abstained is False


# --- citation parser v2 ------------------------------------------------------


def test_citation_ids_reads_gov_uk_ids_with_spaces_and_parentheses():
    # Version 1 allowed only [\w.-] and silently dropped every one of these.
    from refuses_to_lie.generation import citation_ids

    text = "SSP is £123.25 a week [Print Statutory Sick Pay (SSP) - GOV.UK-0002]."
    assert citation_ids(text) == ["Print Statutory Sick Pay (SSP) - GOV.UK-0002"]


def test_citation_ids_splits_a_grouped_citation():
    from refuses_to_lie.generation import citation_ids

    text = "It is 52 weeks [UHN-PO-HR10-0017, Print Maternity pay and leave - GOV.UK-0003]."
    assert citation_ids(text) == [
        "UHN-PO-HR10-0017",
        "Print Maternity pay and leave - GOV.UK-0003",
    ]


def test_citation_ids_ignores_brackets_that_are_not_citations():
    from refuses_to_lie.generation import citation_ids

    assert citation_ids("Band 5 [see note] and [2025] figures.") == []


def test_extracted_citations_keep_only_ids_that_were_in_context():
    from types import SimpleNamespace

    from refuses_to_lie.generation import _extract_citations

    hit = SimpleNamespace(chunk=SimpleNamespace(cite_label="SSP page"))
    hits = {"Print Statutory Sick Pay (SSP) - GOV.UK-0002": hit}
    text = "£123.25 [Print Statutory Sick Pay (SSP) - GOV.UK-0002, invented-doc-0009]."
    cited = _extract_citations(text, hits)  # type: ignore[arg-type]
    assert [c.chunk_id for c in cited] == ["Print Statutory Sick Pay (SSP) - GOV.UK-0002"]
