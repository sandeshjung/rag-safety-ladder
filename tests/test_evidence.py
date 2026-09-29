import json
from pathlib import Path

import pytest

from refuses_to_lie.pipeline import load_corpus_chunks

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / "eval" / "evidence.json").read_text())
LABELS = json.loads((ROOT / "eval" / "wrong_answer_labels.json").read_text())
QUESTIONS = {q["id"]: q for q in json.loads((ROOT / "eval" / "questions.json").read_text())}


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


@pytest.fixture(scope="module")
def chunks_by_doc() -> dict[str, list[str]]:
    by_doc: dict[str, list[str]] = {}
    corpus = ROOT / "corpus"
    for chunk in load_corpus_chunks(corpus / "employer", corpus / "statutory"):
        by_doc.setdefault(chunk.doc_id, []).append(_norm(chunk.text))
    return by_doc


def test_every_evidence_phrasing_is_in_an_expected_document(chunks_by_doc):
    # A quote that matches nothing would report the passage as missing from
    # every context, and blame retrieval for a typo.
    for qid, parts in EVIDENCE.items():
        if qid.startswith("_"):
            continue
        texts = [t for doc in QUESTIONS[qid]["expected_doc_ids"] for t in chunks_by_doc[doc]]
        for part in parts:
            for phrasing in [part] if isinstance(part, str) else part:
                assert any(_norm(phrasing) in t for t in texts), f"{qid}: {phrasing!r}"


def test_evidence_covers_only_answerable_questions():
    for qid in EVIDENCE:
        if not qid.startswith("_"):
            assert QUESTIONS[qid]["expected_answerable"], qid


def test_every_label_uses_a_defined_cause():
    for label in LABELS["labels"]:
        assert label["cause"] in LABELS["causes"], label
        assert label["question_id"] in QUESTIONS, label
