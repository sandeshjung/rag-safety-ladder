"""Judged correctness: was the answer actually right?

The grid's own metrics can only prove an answer wrong when the question
had no answer to begin with, or when an adversarial document was cited.
Neither catches the failure the project is really about -- a fluent,
correctly-cited answer carrying the wrong number. Deciding that needs a
reader, so this module asks one.

Three deliberate choices:

  - The judge is the VERIFIER's model, not the generator's. A model
    grading its own output agrees with itself, and the resulting number
    would measure self-consistency rather than correctness.
  - It grades against `expected_answer_summary` from the eval set, which
    was written from the source documents, rather than re-deriving the
    answer from the corpus. The judge's job is comparison, not retrieval.
  - Abstentions are never graded. Refusing is not a wrong answer, and
    feeding refusals to a correctness judge would bury the project's whole
    premise under a pile of INCORRECTs.

Grades are cached per (question, answer text), so re-running after adding
a rung costs only the new rows, and editing an answer invalidates just
that grade.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from refuses_to_lie.llm_client import call_groq

# "INCORRECT" contains "CORRECT", so the longer label has to be tested
# first or every wrong answer reads as right, silently halving the
# reported error rate -- the same trap the verifier's verdict parsing has.
_GRADE_CHECK_ORDER = ("INCORRECT", "PARTIAL", "CORRECT")
UNGRADED = "UNGRADED"

# Bump when either prompt changes. Grades are cached by answer text, so
# without this a revised rubric would silently hand back the old verdicts.
#   v1  original rubric
#   v2  grades facts not completeness; separates hedging from error; grades
#       false-premise answers on whether they reject the premise
JUDGE_VERSION = "v2"
LEGACY_VERSION = "v1"  # grades written before versioning existed

# v1 said "less detailed is CORRECT" and, two sentences later, "wrong OR
# MISSING on others is PARTIAL". An answer omitting a detail satisfies
# both, and in a hand audit the judge resolved it towards PARTIAL,
# inflating error. v2 grades what the answer says, not what it leaves out.
_GRADE_PROMPT = """You are grading a question-answering system against a \
reference answer written from the source documents.

Question: {question}

Reference answer: {reference}

System answer: {answer}

Grade the facts the system answer STATES, not how complete it is. Ignore \
wording, length, formatting and citation markers.

- CORRECT: every figure, duration, condition or entitlement the answer states \
agrees with the reference, and it answers the question asked. Leaving out \
details the reference includes is still CORRECT.
- PARTIAL: the answer gives the correct fact but ALSO presents a conflicting \
figure or condition as possibly applying (for example "£129, or £123 according \
to another document"), or it answers only one part of a question that asks \
for several things.
- INCORRECT: the answer states a figure, duration, condition or entitlement \
that contradicts the reference, attaches a correct figure to the wrong thing, \
or does not answer the question asked.

Reply with exactly one word: CORRECT, INCORRECT, or PARTIAL."""

# False-premise questions have no answer to compare against: the right
# response is to reject the premise. The first cut of the metrics counted
# ANY answer to them as an error, and every one of the six answered at
# rung D turned out to be a correct, cited rejection of the premise.
_PREMISE_PROMPT = """The question below rests on a FALSE premise. The note \
explains, from the source documents, why it is false.

Question: {question}

Why the premise is false: {reference}

System answer: {answer}

- CORRECT: the answer rejects or corrects the premise -- it says the assumption \
is wrong and gives the actual position, or plainly declines to accept it.
- PARTIAL: the answer hedges, neither accepting nor clearly rejecting the premise.
- INCORRECT: the answer accepts the premise, explains or justifies it, or \
answers as if it were true.

Reply with exactly one word: CORRECT, INCORRECT, or PARTIAL."""


@dataclass
class Grade:
    question_id: str
    config_id: str
    verdict: str
    answer: str
    # Rows written before versioning existed carry no field; they were v1.
    judge_version: str = LEGACY_VERSION

    @property
    def is_wrong(self) -> bool:
        """PARTIAL counts as wrong.

        A half-right answer about parental leave entitlement is still an
        answer somebody could act on and be wrong. Counting it as a pass
        would flatter the system in exactly the place this project cares
        about most.
        """
        return self.verdict in ("INCORRECT", "PARTIAL")


def _parse_grade(response: str) -> str:
    upper = response.upper()
    for verdict in _GRADE_CHECK_ORDER:
        if verdict in upper:
            return verdict
    return UNGRADED


FALSE_PREMISE = "false_premise"


def is_false_premise(row: dict) -> bool:
    return row.get("category") == FALSE_PREMISE


def gradeable(row: dict) -> bool:
    """Answers with something to be graded against.

    Two kinds qualify: an answer to an answerable question, graded against
    its reference, and an answer to a false-premise question, graded on
    whether it rejected the premise. An abstention has no factual content,
    and a question the corpus simply cannot answer is covered by the grid's
    own `error_floor` -- answering it at all is the error.
    """
    if "error" in row or row.get("abstained", False) or not row.get("answer"):
        return False
    if is_false_premise(row):
        return bool(row.get("notes"))
    return bool(row.get("expected_answerable")) and bool(row.get("expected_answer_summary"))


def grade_cache_key(row: dict) -> str:
    """Keyed on judge version and answer text, not on the config.

    Two rungs that produced identical text get one grade and one API call,
    which on a ladder where later rungs often leave the answer untouched is
    most of them. The version is in the key so a revised rubric regrades
    instead of replaying the old verdict from disk.
    """
    digest = hashlib.sha256(row["answer"].encode()).hexdigest()[:12]
    return f"grade-{JUDGE_VERSION}-{row['question_id']}-{digest}"


def grade_row(row: dict, model: str) -> Grade:
    if is_false_premise(row):
        template, reference = _PREMISE_PROMPT, row["notes"]
    else:
        template, reference = _GRADE_PROMPT, row["expected_answer_summary"]
    prompt = template.format(
        question=row["question"], reference=reference, answer=row["answer"]
    )
    response = call_groq(prompt, model=model, temperature=0.0, cache_key=grade_cache_key(row))
    return Grade(
        question_id=row["question_id"],
        config_id=row["config_id"],
        verdict=_parse_grade(response),
        answer=row["answer"],
        judge_version=JUDGE_VERSION,
    )


def load_all_grades(path: Path, version: str = JUDGE_VERSION) -> list[Grade]:
    """Every grade from one judge version, including superseded ones."""
    if not path.exists():
        return []
    grades = [
        Grade(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()
    ]
    return [g for g in grades if g.judge_version == version]


def grade_for(grades: dict[tuple[str, str], Grade], row: dict) -> Grade | None:
    """The grade for this row's answer, or None if the text has changed since.

    Grades are keyed by (config, question), but a row re-run under a fixed
    parser can keep its key and change its text -- rung E drops a sentence
    it now verifies. A grade of the old text would then be reported as a
    grade of the new one.
    """
    grade = grades.get((row["config_id"], row["question_id"]))
    return grade if grade is not None and grade.answer == row.get("answer") else None


def load_grades(path: Path, version: str = JUDGE_VERSION) -> dict[tuple[str, str], Grade]:
    """Grades from one judge version, keyed by (config_id, question_id).

    Filtering by version matters because the grades file is append-only:
    after a rubric change it holds both generations, and mixing them would
    report a number no single judge produced. Later rows win, so a regrade
    supersedes an earlier grade of the same row.
    """
    if not path.exists():
        return {}
    grades: dict[tuple[str, str], Grade] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        grade = Grade(**json.loads(line))
        if grade.judge_version == version:
            grades[(grade.config_id, grade.question_id)] = grade
    return grades


def accuracy(grades: list[Grade]) -> tuple[float, float]:
    """(correct rate, wrong rate) over graded answers.

    They do not sum to 1: a grade the judge did not return cleanly is
    counted in neither, so a parsing failure cannot silently inflate
    either side.
    """
    scored = [g for g in grades if g.verdict != UNGRADED]
    if not scored:
        return 0.0, 0.0
    correct = sum(1 for g in scored if g.verdict == "CORRECT")
    wrong = sum(1 for g in scored if g.is_wrong)
    return correct / len(scored), wrong / len(scored)
