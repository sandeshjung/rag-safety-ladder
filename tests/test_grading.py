from refuses_to_lie.grading import (
    UNGRADED,
    Grade,
    _parse_grade,
    accuracy,
    grade_cache_key,
    gradeable,
)


def _row(**extra) -> dict:
    row = {
        "question_id": "AS-001",
        "config_id": "A",
        "question": "How much annual leave?",
        "expected_answer_summary": "27 days plus 8 public holidays.",
        "answer": "27 days plus 8 public holidays [DOC-1-0001].",
        "expected_answerable": True,
        "abstained": False,
    }
    row.update(extra)
    return row


def test_incorrect_is_parsed_before_correct():
    # "INCORRECT" contains "CORRECT", so checking in the wrong order reads
    # every wrong answer as right -- silently halving the reported error.
    assert _parse_grade("INCORRECT") == "INCORRECT"
    assert _parse_grade("CORRECT") == "CORRECT"
    assert _parse_grade("The answer is INCORRECT.") == "INCORRECT"


def test_partial_is_parsed():
    assert _parse_grade("PARTIAL") == "PARTIAL"


def test_an_unparseable_reply_is_not_silently_treated_as_correct():
    assert _parse_grade("I'm not sure how to grade this") == UNGRADED


def test_partial_counts_as_wrong():
    # A half-right entitlement answer is still actionable and still wrong.
    assert Grade("Q", "A", "PARTIAL", "text").is_wrong
    assert Grade("Q", "A", "INCORRECT", "text").is_wrong
    assert not Grade("Q", "A", "CORRECT", "text").is_wrong


def test_abstentions_are_never_graded():
    # Refusing is the behaviour the project rewards; grading it for
    # correctness would score the system's best moments as failures.
    assert not gradeable(_row(abstained=True))


def test_unanswerable_questions_are_not_graded():
    # There is no reference answer to compare against; error_floor covers it.
    assert not gradeable(_row(expected_answerable=False))


def test_failed_rows_are_not_graded():
    assert not gradeable(_row(error="APIError: 429"))


def test_rows_without_a_reference_answer_are_not_graded():
    assert not gradeable(_row(expected_answer_summary=""))


def test_a_normal_answered_row_is_gradeable():
    assert gradeable(_row())


def test_identical_answers_share_a_cache_key():
    # Later rungs often leave the text untouched; grading it twice would
    # pay for the same judgement on every rung of the ladder.
    same = grade_cache_key(_row(config_id="A"))
    assert same == grade_cache_key(_row(config_id="F"))


def test_a_changed_answer_gets_a_new_cache_key():
    assert grade_cache_key(_row()) != grade_cache_key(_row(answer="25 days."))


def test_accuracy_excludes_ungraded_from_both_sides():
    # A parsing failure must not inflate either correctness or error.
    grades = [
        Grade("Q1", "A", "CORRECT", ""),
        Grade("Q2", "A", "INCORRECT", ""),
        Grade("Q3", "A", UNGRADED, ""),
    ]
    correct, wrong = accuracy(grades)
    assert correct == 0.5
    assert wrong == 0.5


def test_accuracy_of_nothing_is_zero_not_a_crash():
    assert accuracy([]) == (0.0, 0.0)


# --- judge v2 ----------------------------------------------------------------


def test_false_premise_answers_are_graded_against_the_premise_note():
    # These have no reference answer; the right response is to reject the
    # premise, so they must be gradeable against the note that explains it.
    row = _row(
        expected_answerable=False,
        expected_answer_summary="",
        category="false_premise",
        notes="The policy requires the opposite.",
    )
    assert gradeable(row)


def test_a_false_premise_row_without_its_note_is_not_graded():
    row = _row(expected_answerable=False, expected_answer_summary="", category="false_premise")
    assert not gradeable(row)


def test_a_false_premise_abstention_is_still_not_graded():
    row = _row(expected_answerable=False, category="false_premise", notes="n", abstained=True)
    assert not gradeable(row)


def test_false_premise_rows_use_the_premise_prompt(monkeypatch):
    from refuses_to_lie import grading

    sent = {}

    def fake_groq(prompt, **_):
        sent["prompt"] = prompt
        return "CORRECT"

    monkeypatch.setattr(grading, "call_groq", fake_groq)
    row = _row(
        category="false_premise", notes="Badges must be returned.", expected_answerable=False
    )
    grade = grading.grade_row(row, model="m")

    assert "FALSE premise" in sent["prompt"]
    assert "Badges must be returned." in sent["prompt"]
    assert grade.judge_version == grading.JUDGE_VERSION


def test_the_judge_version_is_part_of_the_cache_key():
    # Grades are cached by answer text. Without the version in the key, a
    # revised rubric would replay every old verdict from disk.
    from refuses_to_lie.grading import JUDGE_VERSION

    assert f"-{JUDGE_VERSION}-" in grade_cache_key(_row())


def test_load_grades_keeps_only_the_requested_judge_version(tmp_path):
    import json

    from refuses_to_lie.grading import JUDGE_VERSION, load_grades

    path = tmp_path / "grades.jsonl"
    legacy = {"question_id": "Q1", "config_id": "A", "verdict": "PARTIAL", "answer": "x"}
    current = {
        **legacy,
        "question_id": "Q2",
        "verdict": "CORRECT",
        "judge_version": JUDGE_VERSION,
    }
    path.write_text(json.dumps(legacy) + "\n" + json.dumps(current) + "\n")

    assert set(load_grades(path)) == {("A", "Q2")}
    # Rows written before versioning existed are treated as v1, not as current.
    assert set(load_grades(path, version="v1")) == {("A", "Q1")}


def test_a_regrade_supersedes_the_earlier_grade_of_the_same_row(tmp_path):
    import json

    from refuses_to_lie.grading import JUDGE_VERSION, load_grades

    path = tmp_path / "grades.jsonl"
    base = {
        "question_id": "Q1",
        "config_id": "A",
        "answer": "x",
        "judge_version": JUDGE_VERSION,
    }
    path.write_text(
        json.dumps({**base, "verdict": "PARTIAL"})
        + "\n"
        + json.dumps({**base, "verdict": "CORRECT"})
    )
    assert load_grades(path)[("A", "Q1")].verdict == "CORRECT"


def test_a_grade_does_not_carry_over_to_changed_answer_text():
    # A re-run can keep a row's (config, question) key and change its text;
    # the old grade must not be reported as a grade of the new answer.
    from refuses_to_lie.grading import grade_for

    grades = {("E", "AS-001"): Grade("AS-001", "E", "CORRECT", "old text")}
    assert grade_for(grades, _row(config_id="E", answer="old text")).verdict == "CORRECT"
    assert grade_for(grades, _row(config_id="E", answer="new text")) is None
