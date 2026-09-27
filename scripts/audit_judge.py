"""Check the correctness judge against human labels.

Every accuracy figure in this project is an LLM's opinion. This measures how
far to trust it, in two steps:

  sample  write a BLIND labelling sheet: question, reference and answer, with
          no judge verdict on it. A labeller who can see the judge's grade
          anchors to it, and the agreement figure then measures deference
          rather than accuracy.
  score   compare the judge (current version) against whatever labels have
          been filled in, and report agreement, Cohen's kappa, and which
          direction the judge errs in.

Labels are grouped by the `labeller` field and reported separately. The
rows labelled during the audit that motivated judge v2 are a DEVELOPMENT
set: they shaped the rubric, so agreement on them shows v2 fixed those
cases, not that it generalises. Only independently labelled rows are a test.

Usage:
  uv run python scripts/audit_judge.py sample --n 40
  uv run python scripts/audit_judge.py score
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from refuses_to_lie.analysis import attach_expectations, drop_stale, load_rows
from refuses_to_lie.config import ALL_CONFIGS
from refuses_to_lie.grading import JUDGE_VERSION, gradeable, load_all_grades

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "results" / "clean.jsonl"
GRADES = ROOT / "results" / "clean-grades.jsonl"
SHEET = ROOT / "eval" / "judge_audit.json"

VERDICTS = ("CORRECT", "PARTIAL", "INCORRECT")
# How far each verdict is from "fine to act on", for describing disagreement.
SEVERITY = {"CORRECT": 0, "PARTIAL": 1, "INCORRECT": 2}


def _load_rows() -> list[dict]:
    rows, _ = drop_stale(load_rows(RESULTS), ALL_CONFIGS)
    return attach_expectations(rows, json.loads(EVAL_FILE.read_text()))


def sample(n: int, seed: int) -> None:
    existing = json.loads(SHEET.read_text()) if SHEET.exists() else []
    taken = {(e["config_id"], e["question_id"]) for e in existing}
    # One row per question: labelling the same question at six rungs would
    # spend the labeller's effort on near-duplicate answers.
    by_question: dict[str, dict] = {}
    for row in _load_rows():
        if gradeable(row) and (row["config_id"], row["question_id"]) not in taken:
            by_question.setdefault(row["question_id"], row)
    candidates = sorted(by_question.values(), key=lambda r: r["question_id"])
    picked = random.Random(seed).sample(candidates, min(n, len(candidates)))

    for row in picked:
        premise = row.get("category") == "false_premise"
        existing.append(
            {
                "config_id": row["config_id"],
                "question_id": row["question_id"],
                "category": row.get("category", ""),
                "question": row["question"],
                "reference": row["notes"] if premise else row["expected_answer_summary"],
                "grading_rule": (
                    "false premise: CORRECT if it rejects the premise"
                    if premise
                    else "facts stated, not completeness"
                ),
                "answer": row["answer"],
                "human_verdict": "",
                "labeller": "",
                "note": "",
            }
        )
    SHEET.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    where = SHEET.relative_to(ROOT)
    print(f"added {len(picked)} blank rows -> {where} ({len(existing)} total)")
    print(f"fill human_verdict with one of {', '.join(VERDICTS)} and set labeller")


def _kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa: agreement beyond what the label frequencies predict.

    Raw agreement flatters a judge on a lopsided set -- one that answered
    CORRECT to everything would agree ~65% of the time here by default.
    """
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    human, judge = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(human[v] * judge[v] for v in VERDICTS) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def score() -> None:
    if not SHEET.exists():
        raise SystemExit("no sheet yet - run `sample` first")
    labelled = [e for e in json.loads(SHEET.read_text()) if e["human_verdict"] in VERDICTS]
    # Matched on the exact text the labeller read. If a row has since been
    # re-run and its answer changed, the judge's grade of the labelled text
    # is still the right comparison -- it is the same answer both saw.
    by_text = {(g.config_id, g.question_id, g.answer): g for g in load_all_grades(GRADES)}

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing = 0
    for entry in labelled:
        grade = by_text.get((entry["config_id"], entry["question_id"], entry["answer"]))
        if grade is None:
            missing += 1
            continue
        labeller = entry["labeller"] or "unattributed"
        groups[labeller].append((entry["human_verdict"], grade.verdict))

    print(f"judge {JUDGE_VERSION} vs human labels")
    if missing:
        print(f"({missing} labelled row(s) have no {JUDGE_VERSION} grade yet - regrade first)")
    for labeller, pairs in sorted(groups.items()):
        agree = sum(1 for a, b in pairs if a == b)
        harsher = sum(1 for a, b in pairs if SEVERITY[b] > SEVERITY[a])
        lenient = sum(1 for a, b in pairs if SEVERITY[b] < SEVERITY[a])
        print(f"\n{labeller}: {len(pairs)} rows")
        print(f"  agreement  {agree}/{len(pairs)} = {agree / len(pairs):.0%}")
        print(f"  kappa      {_kappa(pairs):+.2f}")
        print(f"  judge harsher than human: {harsher}   more lenient: {lenient}")
        print(f"  {'human \\ judge':<16}" + "".join(f"{v:>11}" for v in VERDICTS))
        for human in VERDICTS:
            counts = Counter(b for a, b in pairs if a == human)
            print(f"  {human:<16}" + "".join(f"{counts[v]:>11}" for v in VERDICTS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sampler = sub.add_parser("sample", help="append blind rows to label")
    sampler.add_argument("--n", type=int, default=40)
    sampler.add_argument("--seed", type=int, default=0)
    sub.add_parser("score", help="judge agreement with the labels filled in")
    args = parser.parse_args()

    if args.command == "sample":
        sample(args.n, args.seed)
    else:
        score()


if __name__ == "__main__":
    main()
