"""Measure what abstain_threshold should be, instead of guessing it.

Reads rung F rows (the only ones that compute a confidence signal), joins
them to the correctness grades, and reports two things:

  1. whether the confidence score separates right answers from wrong ones
  2. what coverage and error rate each threshold would have produced

Read them in that order. If the score does not separate, the sweep is
choosing between equally uninformed lines and the answer is to rebuild the
signal, not to move the threshold.

Usage:
  uv run python scripts/calibrate_threshold.py
  uv run python scripts/calibrate_threshold.py --results results/injection.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from refuses_to_lie.analysis import (
    attach_expectations,
    drop_stale,
    load_rows,
    must_refuse,
)
from refuses_to_lie.calibration import COMPONENTS, Scored, best_threshold, separation, sweep
from refuses_to_lie.config import ALL_CONFIGS, F, RunConfig
from refuses_to_lie.grading import grade_for, load_grades

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "questions.json"
DEFAULT_RESULTS = ROOT / "results" / "clean.jsonl"


def load_scored(
    results: Path, grades_path: Path, questions: list[dict], rung: RunConfig
) -> list[Scored]:
    rows, _ = drop_stale(load_rows(results), ALL_CONFIGS)
    rows = attach_expectations(rows, questions)

    grades = load_grades(grades_path)

    scored = []
    for row in rows:
        if row["config_id"] != rung.id or "error" in row or not row.get("confidence"):
            continue
        grade = grade_for(grades, row)
        verdict = grade.verdict if grade else None
        # An answer to a question the corpus cannot support is wrong however
        # fluent it was, and it will never carry a grade -- there is no
        # reference to grade it against.
        wrong = must_refuse(row) or verdict in ("INCORRECT", "PARTIAL")
        confidence = row["confidence"]
        scored.append(
            Scored(
                question_id=row["question_id"],
                composite=confidence["composite"],
                components={c: confidence[c] for c in COMPONENTS if c in confidence},
                wrong=wrong,
                abstained=row["abstained"],
            )
        )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--grades", type=Path, default=None)
    parser.add_argument("--max-error", type=float, default=0.10)
    parser.add_argument("--config", default=F.id, help="rung to calibrate (F or G)")
    args = parser.parse_args()

    grades_path = args.grades or args.results.with_name(args.results.stem + "-grades.jsonl")
    questions = json.loads(EVAL_FILE.read_text())
    by_id = {c.id: c for c in ALL_CONFIGS}
    if args.config.upper() not in by_id:
        parser.error(f"unknown rung {args.config!r}")
    rung = by_id[args.config.upper()]
    scored = load_scored(args.results, grades_path, questions, rung)

    if not scored:
        parser.error(f"no rung-{rung.id} rows with a confidence signal in {args.results}")

    answered = [s for s in scored if not s.abstained]
    print(
        f"{len(scored)} rung-{rung.id} rows | {len(answered)} answered | "
        f"{sum(1 for s in answered if s.wrong)} of those wrong\n"
    )

    print("does the signal separate right from wrong?")
    print("-" * 62)
    print(f"{'component':<16}{'right':>9}{'wrong':>9}{'gap':>9}")
    for component, stats in separation(scored).items():
        print(
            f"{component:<16}{stats['right_mean']:>9.3f}{stats['wrong_mean']:>9.3f}"
            f"{stats['gap']:>+9.3f}"
        )
    print(
        "\nA gap near zero means the component carries no information about\n"
        "correctness, and no threshold on it can help."
    )

    points = sweep(scored, applied_threshold=rung.abstain_threshold)
    print(f"\nthreshold sweep (only at or above the {rung.abstain_threshold} that ran)")
    print("-" * 62)
    for point in points:
        print("  " + point.summary)
    print(
        "\nBelow that threshold the data is CENSORED, not missing: those rows\n"
        "were abstained and their answer text replaced, so there is nothing\n"
        "to score. Sweeping downward would be inventing numbers."
    )

    pick = best_threshold(points, args.max_error)
    print(f"\nhighest coverage with error <= {args.max_error:.0%}:")
    print(f"  {pick.summary}" if pick else "  unreachable at any threshold measured")


if __name__ == "__main__":
    main()
