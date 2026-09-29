import json
from dataclasses import replace
from pathlib import Path

from refuses_to_lie.config import LADDER, A, B
from refuses_to_lie.grid import (
    cache_key,
    config_fingerprint,
    load_completed,
    pending_work,
    stratified_sample,
)

EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def _row(
    config_id: str,
    question_id: str,
    fingerprint: str,
    error: str | None = None,
    includes_injected: bool = True,
    **extra: object,
) -> str:
    row = {
        "config_id": config_id,
        "question_id": question_id,
        "config_fingerprint": fingerprint,
        "corpus_includes_injected": includes_injected,
        **extra,
    }
    if error:
        row["error"] = error
    return json.dumps(row)


def test_fingerprint_changes_when_a_substantive_field_changes():
    # If this did not hold, editing a model name would silently reuse answers
    # produced by the old one and the reported number would mix two systems.
    assert config_fingerprint(A) != config_fingerprint(replace(A, generator_model="other"))
    assert config_fingerprint(A) != config_fingerprint(replace(A, top_k_context=3))
    assert config_fingerprint(A) != config_fingerprint(B)


def test_fingerprint_ignores_cosmetic_label_changes():
    # Rewording a human-readable label should not throw away hours of work.
    assert config_fingerprint(A) == config_fingerprint(replace(A, label="reworded"))


def test_cache_key_is_stable_and_config_sensitive():
    assert cache_key(A, "AS-001") == cache_key(A, "AS-001")
    assert cache_key(A, "AS-001") != cache_key(B, "AS-001")
    assert cache_key(A, "AS-001") != cache_key(A, "AS-002")


def test_completed_rows_are_skipped(tmp_path: Path):
    path = tmp_path / "grid.jsonl"
    fp = config_fingerprint(A)
    path.write_text(_row("A", "AS-001", fp) + "\n")

    assert load_completed(path) == {("A", "AS-001", fp, "inj")}


def test_failed_rows_are_retried_not_treated_as_done(tmp_path: Path):
    # A transient 429 must not become a permanent hole in the grid.
    path = tmp_path / "grid.jsonl"
    fp = config_fingerprint(A)
    path.write_text(
        _row("A", "AS-001", fp) + "\n" + _row("A", "AS-002", fp, error="APIError: 429") + "\n"
    )

    completed = load_completed(path)
    assert ("A", "AS-001", fp, "inj") in completed
    assert ("A", "AS-002", fp, "inj") not in completed

    questions = [{"id": "AS-001", "category": "x"}, {"id": "AS-002", "category": "x"}]
    todo = pending_work(questions, [A], completed)
    assert [q["id"] for q, _ in todo] == ["AS-002"]


def test_changing_the_config_invalidates_previous_rows(tmp_path: Path):
    path = tmp_path / "grid.jsonl"
    path.write_text(_row("A", "AS-001", config_fingerprint(A)) + "\n")
    completed = load_completed(path)

    changed = replace(A, generator_model="a-different-model")
    questions = [{"id": "AS-001", "category": "x"}]

    assert pending_work(questions, [A], completed) == []
    assert len(pending_work(questions, [changed], completed)) == 1


def test_missing_results_file_means_everything_is_pending(tmp_path: Path):
    assert load_completed(tmp_path / "absent.jsonl") == set()


def test_stratified_sample_covers_every_category():
    # A slice used to validate the harness is only useful if it exercises
    # abstention and injection, not just whatever a shuffle surfaced.
    questions = json.loads(EVAL_FILE.read_text())
    all_categories = {q["category"] for q in questions}

    sample = stratified_sample(questions, 40)

    assert {q["category"] for q in sample} == all_categories
    assert len(sample) <= 40
    assert len({q["id"] for q in sample}) == len(sample), "sample must not repeat questions"


def test_stratified_sample_is_deterministic():
    questions = json.loads(EVAL_FILE.read_text())
    assert [q["id"] for q in stratified_sample(questions, 30)] == [
        q["id"] for q in stratified_sample(questions, 30)
    ]


def test_pending_work_covers_the_whole_grid_when_nothing_is_done():
    questions = [{"id": f"Q{i}", "category": "x"} for i in range(5)]
    todo = pending_work(questions, list(LADDER), set())
    assert len(todo) == 5 * len(LADDER)


def test_a_clean_corpus_run_does_not_reuse_contaminated_rows(tmp_path: Path):
    # The injected corpus changes what the model saw, so a row produced
    # with it says nothing about a clean baseline. Treating it as done
    # would leave the baseline silently made of contaminated answers.
    path = tmp_path / "grid.jsonl"
    fp = config_fingerprint(A)
    path.write_text(_row("A", "AS-001", fp, includes_injected=True) + "\n")

    completed = load_completed(path)
    questions = [{"id": "AS-001", "category": "x"}]

    assert pending_work(questions, [A], completed, includes_injected=True) == []
    assert len(pending_work(questions, [A], completed, includes_injected=False)) == 1


def test_cache_key_separates_the_two_corpora():
    # Same question and config against a different corpus is a different
    # prompt, so it must not collide in the on-disk LLM cache.
    assert cache_key(A, "AS-001", True) != cache_key(A, "AS-001", False)


# Fingerprints the committed A-F results were produced under. If this test
# fails, a change to RunConfig or to config_fingerprint has just marked every
# existing grid row stale -- days of rate-limited quota. Add new fields to
# ADDED_LATER_FIELDS instead of changing these values.
RECORDED_FINGERPRINTS = {
    "A": "a61b52ce27",
    "B": "73489207b5",
    "C": "4130bd632e",
    "D": "2f4cb2653e",
    "E": "513df9cff4",
    "F": "1c3589d9fa",
}


def test_existing_ladder_fingerprints_are_unchanged():
    for config in LADDER:
        assert config_fingerprint(config) == RECORDED_FINGERPRINTS[config.id], config.id


def test_extension_fingerprints_are_unchanged():
    # Same protection for the rungs added after the ladder: a new field must
    # not mark G's and H's completed rows stale.
    from refuses_to_lie.config import G, H

    assert config_fingerprint(G) == "53260558c5"
    assert config_fingerprint(H) == "47f1b2ce91"


def test_a_later_field_at_its_default_does_not_move_the_fingerprint():
    # The whole point of ADDED_LATER_FIELDS: a capability the old rows never
    # used must not invalidate them just by existing.
    assert config_fingerprint(replace(A, require_provenance=False)) == config_fingerprint(A)
    assert config_fingerprint(replace(A, exclude_superseded=False)) == config_fingerprint(A)


def test_a_later_field_turned_on_does_move_the_fingerprint():
    assert config_fingerprint(replace(A, require_provenance=True)) != config_fingerprint(A)
    assert config_fingerprint(replace(A, exclude_superseded=True)) != config_fingerprint(A)
    assert config_fingerprint(replace(A, confidence_source="answerability")) != (
        config_fingerprint(A)
    )


def test_extension_rungs_do_not_collide_with_the_ladder():
    from refuses_to_lie.config import ALL_CONFIGS

    prints = [config_fingerprint(c) for c in ALL_CONFIGS]
    assert len(set(prints)) == len(prints)


def test_citing_rows_from_the_old_parser_are_not_done(tmp_path: Path):
    # Parser v1 dropped GOV.UK and grouped citations, so its rows on citing
    # rungs must be re-run; rungs that never cite are unaffected.
    from refuses_to_lie.config import D
    from refuses_to_lie.grid import PARSER_VERSION

    path = tmp_path / "grid.jsonl"
    fp_a, fp_d = config_fingerprint(A), config_fingerprint(D)
    path.write_text(
        _row("A", "AS-001", fp_a)
        + "\n"
        + _row("D", "AS-001", fp_d)
        + "\n"
        + _row("D", "AS-002", fp_d, parser_version=PARSER_VERSION)
        + "\n"
    )
    done = load_completed(path, frozenset({"D"}))
    assert ("A", "AS-001", fp_a, "inj") in done
    assert ("D", "AS-001", fp_d, "inj") not in done
    assert ("D", "AS-002", fp_d, "inj") in done
