"""Grade the grid's answers for factual correctness.

Turns the grid's error FLOOR into a real error rate. The floor only counts
answers that were provably unwarranted; this counts the ones that were
confidently wrong -- the failure the project is named after.

Runs on Groq, deliberately: it is a different model from the generator, so
it is not grading its own homework.

Do NOT run this while a grid run is in flight. Groq is also the verifier's
provider, so rungs E and F spend the same per-minute budget, and a grading
sweep alongside them rate-limits the grid. Learned the hard way: 64
grading calls took out a rung E row mid-run.

Usage:
  uv run python scripts/grade_answers.py --results results/clean.jsonl
  uv run python scripts/grade_answers.py --configs A,F
  uv run python scripts/grade_answers.py --results results/frontier-injection.jsonl \
      --include-external
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from refuses_to_lie.analysis import (
    attach_expectations,
    drop_stale,
    load_rows,
    wilson_halfwidth,
)
from refuses_to_lie.config import ALL_CONFIGS, A
from refuses_to_lie.grading import (
    JUDGE_VERSION,
    Grade,
    accuracy,
    grade_for,
    grade_row,
    gradeable,
    load_grades,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "questions.json"
DEFAULT_RESULTS = ROOT / "results" / "clean.jsonl"
AUDIT_SHEET = ROOT / "eval" / "judge_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--grades", type=Path, default=None)
    parser.add_argument("--configs", default="", help="comma-separated rung ids")
    parser.add_argument("--model", default=A.verifier_model)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="grade only the rows in eval/judge_audit.json, so the judge check comes first",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="also grade rows from outside the ladder (fingerprint starting 'external:')",
    )
    args = parser.parse_args()

    grades_path = args.grades or args.results.with_name(args.results.stem + "-grades.jsonl")

    loaded = load_rows(args.results)
    rows, _ = drop_stale(loaded, ALL_CONFIGS)
    if args.include_external:
        # Rows produced outside the ladder (another generator fed the same
        # prompts) carry an "external:" fingerprint. They are opt-in so a
        # ladder report can never average them in by accident.
        rows += [
            r for r in loaded if str(r.get("config_fingerprint", "")).startswith("external:")
        ]
    rows = attach_expectations(rows, json.loads(EVAL_FILE.read_text()))
    if args.configs:
        wanted = {c.strip().upper() for c in args.configs.split(",")}
        rows = [r for r in rows if r["config_id"] in wanted]

    if args.audit:
        sheet = json.loads(AUDIT_SHEET.read_text())
        audited = {(e["config_id"], e["question_id"]) for e in sheet}
        rows = [r for r in rows if (r["config_id"], r["question_id"]) in audited]

    to_grade = [r for r in rows if gradeable(r)]
    # Only this judge version counts as done: a rubric change regrades.
    # A grade counts only for the exact text it graded, so an answer that
    # changed on a re-run is graded again. Unchanged text costs nothing: the
    # judge's cache is keyed by the answer's digest.
    existing = load_grades(grades_path)
    current = [grade_for(existing, r) for r in to_grade]
    grades = [g for g in current if g is not None]
    pending = [r for r, g in zip(to_grade, current, strict=True) if g is None]

    print(
        f"{len(rows)} rows | {len(to_grade)} gradeable "
        f"(answered, answerable, has reference)\n"
        f"judge {JUDGE_VERSION}: {len(grades)} already graded, "
        f"{len(pending)} to grade -> {grades_path}\n"
    )

    grades_path.parent.mkdir(parents=True, exist_ok=True)
    with grades_path.open("a") as out:
        for i, row in enumerate(pending, start=1):
            grade = grade_row(row, args.model)
            out.write(json.dumps(grade.__dict__) + "\n")
            out.flush()
            grades.append(grade)
            if i % 10 == 0 or i == len(pending):
                print(
                    f"[{i}/{len(pending)}] {grade.config_id} {grade.question_id} "
                    f"{grade.verdict}",
                    flush=True,
                )

    by_config: dict[str, list[Grade]] = defaultdict(list)
    for grade in grades:
        by_config[grade.config_id].append(grade)

    print(f"\n{'rung':<6}{'graded':>8}{'correct':>18}{'wrong':>8}")
    print("-" * 42)
    for config_id in sorted(by_config):
        group = by_config[config_id]
        correct, wrong = accuracy(group)
        halfwidth = wilson_halfwidth(correct, len(group)) * 100
        print(f"{config_id:<6}{len(group):>8}{correct:>10.1%} ±{halfwidth:4.1f}{wrong:>9.1%}")
    print(
        "\nwrong = INCORRECT or PARTIAL. A half-right answer about an "
        "entitlement\n        is still one somebody could act on and be wrong."
    )


if __name__ == "__main__":
    main()
