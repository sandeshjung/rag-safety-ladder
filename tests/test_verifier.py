from pathlib import Path

import pytest

from refuses_to_lie.config import D
from refuses_to_lie.corpus import chunk_document, load_body_pages
from refuses_to_lie.generation import GeneratedAnswer
from refuses_to_lie.retrieval import Index
from refuses_to_lie.verifier import (
    ClaimVerdict,
    VerifiedAnswer,
    _parse_verdict,
    drop_unsupported_claims,
    verify_answer,
)

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


@pytest.fixture(scope="module")
def index() -> Index:
    chunks = chunk_document(
        "UHN-PO-HR24",
        load_body_pages(CORPUS / "uhn-annual-leave-for-agenda-for-change-staff000100pdf.pdf"),
    )
    return Index(chunks)


def test_parse_verdict_distinguishes_supported_from_unsupported():
    # Regression test: "SUPPORTED" is a substring of "UNSUPPORTED", so a
    # naive scan that checks "SUPPORTED" before "UNSUPPORTED" silently
    # flips every unsupported claim to supported. Caught via a real model
    # response that was correctly "UNSUPPORTED" but got parsed as SUPPORTED.
    assert _parse_verdict("SUPPORTED") == "SUPPORTED"
    assert _parse_verdict("UNSUPPORTED") == "UNSUPPORTED"
    assert _parse_verdict("The claim is UNSUPPORTED by the excerpt.") == "UNSUPPORTED"
    assert _parse_verdict("PARTIAL") == "PARTIAL"
    assert _parse_verdict("garbage response") == "UNKNOWN"


@pytest.mark.live
def test_verify_answer_flags_an_unsupported_claim(index: Index):
    hits = index.search_hybrid("Can unused annual leave be carried into the next year?", k=4)
    target_chunk_id = hits[0].chunk.chunk_id

    false_answer = GeneratedAnswer(
        question="Can unused annual leave be carried into the next year?",
        text=(
            "All employees are entitled to unlimited carry-over of annual leave "
            f"with no restrictions [{target_chunk_id}]."
        ),
        citations=[],
        context_chunk_ids=[h.chunk.chunk_id for h in hits],
    )

    verified = verify_answer(
        false_answer, hits, D, cache_key_prefix="test-verifier-unsupported"
    )

    assert len(verified.claim_verdicts) == 1
    assert verified.claim_verdicts[0].verdict == "UNSUPPORTED"
    assert verified.groundedness == 0.0
    assert verified.all_supported is False


@pytest.mark.live
def test_verify_answer_confirms_a_supported_claim(index: Index):
    hits = index.search_hybrid("Can unused annual leave be carried into the next year?", k=4)
    target_chunk_id = hits[0].chunk.chunk_id

    true_answer = GeneratedAnswer(
        question="Can unused annual leave be carried into the next year?",
        text=(
            "Annual leave can only be carried over where there are statutory "
            f"obligations or rules to do so [{target_chunk_id}]."
        ),
        citations=[],
        context_chunk_ids=[h.chunk.chunk_id for h in hits],
    )

    verified = verify_answer(true_answer, hits, D, cache_key_prefix="test-verifier-supported")

    assert len(verified.claim_verdicts) == 1
    assert verified.claim_verdicts[0].verdict == "SUPPORTED"
    assert verified.groundedness == 1.0
    assert verified.all_supported is True


def test_groundedness_with_no_claims_is_1():
    empty = VerifiedAnswer(
        answer=GeneratedAnswer(
            question="q", text="no citations here", citations=[], context_chunk_ids=[]
        ),
        claim_verdicts=[],
    )
    assert empty.groundedness == 1.0


def test_drop_unsupported_claims_strips_only_the_bad_sentence():
    answer = GeneratedAnswer(
        question="q",
        text=(
            "Staff accrue 27 days of leave [doc-0001]. "
            "All leave rolls over indefinitely with no cap [doc-0002]."
        ),
        citations=[],
        context_chunk_ids=["doc-0001", "doc-0002"],
    )
    verified = VerifiedAnswer(
        answer=answer,
        claim_verdicts=[
            ClaimVerdict("Staff accrue 27 days of leave [doc-0001].", "doc-0001", "SUPPORTED"),
            ClaimVerdict(
                "All leave rolls over indefinitely with no cap [doc-0002].",
                "doc-0002",
                "UNSUPPORTED",
            ),
        ],
    )

    result = drop_unsupported_claims(verified)

    assert "27 days" in result
    assert "no cap" not in result


# --- citation parser v2 ------------------------------------------------------


def test_claims_citing_gov_uk_pages_are_verified():
    from refuses_to_lie.verifier import _split_claims

    text = "You get £123.25 a week [Print Statutory Sick Pay (SSP) - GOV.UK-0002]."
    assert _split_claims(text) == [(text, "Print Statutory Sick Pay (SSP) - GOV.UK-0002")]


def test_a_sentence_citing_two_sources_is_checked_against_each():
    from refuses_to_lie.verifier import _split_claims

    text = "Leave is 52 weeks [doc-a-0001, doc-b-0002]."
    assert _split_claims(text) == [(text, "doc-a-0001"), (text, "doc-b-0002")]


def test_a_decimal_is_not_a_sentence_boundary():
    from refuses_to_lie.verifier import _split_claims

    text = "The rate is £116.75 per week [doc-0001]. It is paid weekly [doc-0002]."
    claims = _split_claims(text)
    assert claims[0] == ("The rate is £116.75 per week [doc-0001].", "doc-0001")
    assert claims[1] == ("It is paid weekly [doc-0002].", "doc-0002")


def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before():
    from refuses_to_lie.verifier import _split_claims

    text = "Probation lasts six months. [doc-0001]"
    assert _split_claims(text) == [("Probation lasts six months.", "doc-0001")]


def test_cached_verdicts_are_reused_only_for_the_same_claim_text():
    # Version 1 keyed verdicts by position. Once more claims are found the
    # positions shift, and a positional key would replay one claim's verdict
    # for another.
    from refuses_to_lie.verifier import _cache_key

    legacy = {("Old claim [doc-0001].", "doc-0001"): 0}
    assert _cache_key("p", "Old claim [doc-0001].", "doc-0001", legacy) == "p-claim0"
    new = _cache_key("p", "New claim [doc-0002].", "doc-0002", legacy)
    assert new.startswith("p-c") and new != "p-claim0"
    assert new != _cache_key("p", "New claim [doc-0002].", "doc-0003", legacy)


def test_a_sentence_supported_by_one_of_its_sources_is_kept():
    text = "Leave is 52 weeks [doc-a-0001, doc-b-0002]."
    verified = VerifiedAnswer(
        answer=GeneratedAnswer(question="q", text=text, citations=[], context_chunk_ids=[]),
        claim_verdicts=[
            ClaimVerdict(text, "doc-a-0001", "SUPPORTED"),
            ClaimVerdict(text, "doc-b-0002", "UNSUPPORTED"),
        ],
    )
    assert drop_unsupported_claims(verified) == text
