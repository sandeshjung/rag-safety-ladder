import json
from pathlib import Path

from refuses_to_lie.provenance import trusted_doc_ids
from refuses_to_lie.versions import load_versions, superseded_doc_ids

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ROOT / "corpus" / "versions.json"
REGISTER = ROOT / "corpus" / "register.json"


def test_every_version_record_names_registered_documents():
    # A misspelt id would make rung I silently keep the stale copy it was
    # built to drop.
    registered = trusted_doc_ids(REGISTER)
    for old, new in load_versions(VERSIONS).items():
        assert old in registered, old
        assert new in registered, new


def test_a_replacement_is_never_itself_superseded():
    # Dropping only the first link of a chain would leave the middle
    # version in the context as though it were current.
    versions = load_versions(VERSIONS)
    assert not set(versions.values()) & set(versions)


def test_no_eval_question_depends_on_a_superseded_document():
    # If one did, rung I would lose it by design and its score would
    # measure the eval set, not the filter.
    stale = set(load_versions(VERSIONS))
    for question in json.loads((ROOT / "eval" / "questions.json").read_text()):
        assert not set(question.get("expected_doc_ids") or []) & stale, question["id"]


def test_a_superseded_document_stays_when_its_replacement_is_missing(tmp_path: Path):
    path = tmp_path / "versions.json"
    path.write_text(json.dumps({"old-a": "new-a", "old-b": "new-b"}))
    assert superseded_doc_ids(path, frozenset({"old-a", "new-a", "old-b"})) == {"old-a"}
