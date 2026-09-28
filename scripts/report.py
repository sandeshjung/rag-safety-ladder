"""Every headline number in one place, recomputed from results/.

Each figure the writeup quotes is produced here, from the recorded rows and
the current judge's grades, so filling in the numbers is one command and a
figure cannot drift from the data behind it. Grades still in progress show
up as "graded x/y" rather than as a silently smaller denominator.

Sections:
  1. the ladder on the clean corpus: coverage, error floor, accuracy
  2. planted documents: citation of fakes, obedience, accuracy
  3. same prompts, frontier generator (rows with an "external:" fingerprint)
  4. whether the confidence signals predict correctness
  5. why answers were wrong: retrieval, and superseded documents

Usage:
  uv run python scripts/report.py
  uv run python scripts/report.py --out report.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Callable
from pathlib import Path

from refuses_to_lie.analysis import (
    answered,
    attach_expectations,
    cited_doc_ids,
    cites_injection,
    drop_stale,
    load_rows,
    must_refuse,
    retrieved_injection,
    score_config,
    wilson_halfwidth,
)
from refuses_to_lie.calibration import COMPONENTS
from refuses_to_lie.config import ALL_CONFIGS
from refuses_to_lie.grading import Grade, grade_for, gradeable, load_grades
from refuses_to_lie.injection import obeyed_signatures

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EVAL_FILE = ROOT / "eval" / "questions.json"

REGISTER = ROOT / "corpus" / "register.json"

# An older copy kept alongside the current one: the 2025 GOV.UK snapshots
# and the policy whose file is marked SUPERSEDED. Read from the register's
# file names, because doc ids come from cover sheets ("B1.1") and do not
# always carry the marker.
_OLD_COPY = re.compile(r"SUPERSEDED|\s2025\s?-")


def superseded_doc_ids() -> set[str]:
    register = json.loads(REGISTER.read_text())
    return {doc_id for doc_id, entry in register.items() if _OLD_COPY.search(entry["file"])}


EXTERNAL = "external:"


def pct(rate: float, n: int) -> str:
    if not n:
        return "n/a"
    return f"{rate:.1%} ±{wilson_halfwidth(rate, n) * 100:.1f}"


def rows_and_grades(name: str) -> tuple[list[dict], dict[tuple[str, str], Grade]]:
    """Current-config rows plus external ones, joined to the eval set."""
    path = RESULTS / f"{name}.jsonl"
    if not path.exists():
        return [], {}
    loaded = load_rows(path)
    rows, _ = drop_stale(loaded, ALL_CONFIGS)
    rows += [r for r in loaded if str(r.get("config_fingerprint", "")).startswith(EXTERNAL)]
    rows = attach_expectations(rows, json.loads(EVAL_FILE.read_text()))
    return rows, load_grades(RESULTS / f"{name}-grades.jsonl")


def by_config(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["config_id"], []).append(row)
    return dict(sorted(grouped.items()))


def accuracy_cell(rows: list[dict], grades: dict[tuple[str, str], Grade]) -> str:
    """Share of graded answers that are CORRECT, with how much is graded yet."""
    due = [r for r in rows if "error" not in r and gradeable(r)]
    graded = [g for g in (grade_for(grades, r) for r in due) if g is not None]
    counted = [g for g in graded if g.verdict in ("CORRECT", "PARTIAL", "INCORRECT")]
    if not counted:
        return f"graded 0/{len(due)}"
    rate = sum(1 for g in counted if g.verdict == "CORRECT") / len(counted)
    suffix = "" if len(graded) == len(due) else f" (graded {len(graded)}/{len(due)})"
    return pct(rate, len(counted)) + suffix


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def ladder(rows: list[dict], grades: dict[tuple[str, str], Grade]) -> str:
    body = []
    for config_id, group in by_config(rows).items():
        if config_id.startswith("D-"):
            continue
        score = score_config(group)
        body.append(
            [
                config_id,
                str(score.n_ok),
                pct(score.coverage, score.n_ok),
                pct(score.error_floor, score.n_answered),
                accuracy_cell(group, grades),
            ]
        )
    return table(["rung", "rows", "coverage", "error floor", "accuracy"], body)


def injection(
    rows: list[dict],
    grades: dict[tuple[str, str], Grade],
    clean_rows: list[dict],
    clean_grades: dict[tuple[str, str], Grade],
) -> str:
    clean_pi = [r for r in clean_rows if r.get("category") == "prompt_injection"]
    body = []
    for config_id, group in by_config(rows).items():
        ok = [r for r in group if "error" not in r]
        given = [r for r in ok if answered(r)]
        cites = sum(1 for r in given if cites_injection(r))
        only = sum(
            1
            for r in given
            if cited_doc_ids(r) and all(d.startswith("INJ-") for d in cited_doc_ids(r))
        )
        obeyed = sum(1 for r in ok if obeyed_signatures(r.get("answer", "")))
        same_clean = [r for r in clean_pi if r["config_id"] == config_id]
        body.append(
            [
                config_id,
                str(len(ok)),
                f"{sum(1 for r in ok if retrieved_injection(r))}/{len(ok)}",
                f"{cites}/{len(ok)}",
                pct(cites / len(given), len(given)) if given else "n/a",
                str(only),
                str(obeyed),
                accuracy_cell(group, grades),
                accuracy_cell(same_clean, clean_grades) if same_clean else "not run",
            ]
        )
    return table(
        [
            "rung",
            "rows",
            "fake retrieved",
            "cited a fake (of all)",
            "cited a fake (of answers)",
            "cite only fakes",
            "obeyed",
            "accuracy (fakes present)",
            "accuracy, same questions clean",
        ],
        body,
    )


def frontier(rows: list[dict], grades: dict[tuple[str, str], Grade], base: str) -> str:
    """The recorded rung beside an external generator given its exact prompts.

    Refusals are split by whether refusing was right, because the two mean
    opposite things: refusing a must-refuse question is the pass, refusing
    an answerable one is lost coverage.
    """
    external = [r for r in rows if str(r.get("config_fingerprint", "")).startswith(EXTERNAL)]
    if not external:
        return "_no external rows_"
    asked = {r["question_id"] for r in external}
    body = []
    for label, group in (
        (base, [r for r in rows if r["config_id"] == base and r["question_id"] in asked]),
        (external[0]["config_id"], external),
    ):
        given = [r for r in group if answered(r)]
        refuse = [r for r in group if must_refuse(r)]
        answerable = [r for r in group if not must_refuse(r)]
        body.append(
            [
                label,
                str(len(group)),
                f"{sum(1 for r in answerable if not answered(r))}/{len(answerable)}",
                f"{sum(1 for r in refuse if answered(r))}/{len(refuse)}",
                str(sum(1 for r in given if cites_injection(r))),
                accuracy_cell(group, grades),
            ]
        )
    return table(
        [
            "generator",
            "rows",
            "refused an answerable question",
            "answered a must-refuse question",
            "cited a fake",
            "accuracy",
        ],
        body,
    )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def confidence(rows: list[dict], grades: dict[tuple[str, str], Grade], rung: str) -> str:
    """Correlation of each confidence component with being right.

    Positive means the signal points the right way. Only graded answers
    count: an answer with no verdict says nothing about correctness.
    """
    pairs: list[tuple[dict, bool]] = []
    for row in rows:
        if row["config_id"] != rung or not answered(row) or not row.get("confidence"):
            continue
        # Same rule as calibrate_threshold: answering a question the corpus
        # cannot support is wrong, and such an answer never carries a grade.
        grade = grade_for(grades, row)
        if must_refuse(row):
            pairs.append((row["confidence"], False))
        elif grade and grade.verdict in ("CORRECT", "PARTIAL", "INCORRECT"):
            pairs.append((row["confidence"], grade.verdict == "CORRECT"))
    body = []
    for component in COMPONENTS:
        present = [(c[component], ok) for c, ok in pairs if component in c]
        r = _pearson([v for v, _ in present], [1.0 if ok else 0.0 for _, ok in present])
        if r is not None:
            body.append([component, f"{r:+.3f}", str(len(present))])
    return table(["component", "correlation with correct", "n"], body) if body else "_no data_"


def gate_outcomes(rows: list[dict], grades: dict[tuple[str, str], Grade], rung: str) -> str:
    """What happened at each answerability score, for a gated rung.

    A correlation says nothing here: the gate only lets 1.0 through, so
    every answer it allows carries the same score. What matters is how the
    questions landed at each score -- which it stopped, which it let
    through, and whether those were right.
    """
    gated = [
        r
        for r in rows
        if r["config_id"] == rung
        and "error" not in r
        and "answerability" in (r.get("confidence") or {})
    ]
    if not gated:
        return "_no data_"
    body = []
    for score in sorted({r["confidence"]["answerability"] for r in gated}, reverse=True):
        group = [r for r in gated if r["confidence"]["answerability"] == score]
        given = [r for r in group if answered(r)]
        graded = [g for g in (grade_for(grades, r) for r in given) if g is not None]
        body.append(
            [
                f"{score:.1f}",
                str(len(group)),
                str(sum(1 for r in group if must_refuse(r))),
                str(len(given)),
                str(sum(1 for r in given if must_refuse(r))),
                str(sum(1 for g in graded if g.verdict == "CORRECT")),
                str(sum(1 for g in graded if g.is_wrong)),
            ]
        )
    return table(
        [
            "answerability",
            "questions",
            "of them must-refuse",
            "answered",
            "answered a must-refuse",
            "graded correct",
            "graded wrong",
        ],
        body,
    )


def why_wrong(rows: list[dict], grades: dict[tuple[str, str], Grade]) -> str:
    wrong = [
        r
        for r in rows
        if not r["config_id"].startswith("D-")
        and (grade := grade_for(grades, r)) is not None
        and grade.is_wrong
    ]
    if not wrong:
        return "_no wrong answers graded yet_"
    with_truth = [r for r in wrong if r.get("expected_doc_ids")]
    retrieved = sum(
        1 for r in with_truth if set(r["retrieved_doc_ids"]) & set(r["expected_doc_ids"])
    )
    old_copies = superseded_doc_ids()
    stale = sum(1 for r in wrong if cited_doc_ids(r) & old_copies)
    return table(
        ["of wrong answers", "share", "n"],
        [
            [
                "right document was retrieved",
                pct(retrieved / len(with_truth), len(with_truth)),
                str(len(with_truth)),
            ],
            [
                "cited a superseded document",
                pct(stale / len(wrong), len(wrong)),
                str(len(wrong)),
            ],
        ],
    )


def build() -> str:
    clean, clean_grades = rows_and_grades("clean")
    inj, inj_grades = rows_and_grades("injection")
    f_inj, f_inj_grades = rows_and_grades("frontier-injection")
    f_clean, f_clean_grades = rows_and_grades("frontier-clean")

    def section(title: str, render: Callable[[], str]) -> str:
        return f"## {title}\n\n{render()}\n"

    parts = [
        "# refuses-to-lie: results\n",
        "Accuracy is the share of graded answers the judge marked CORRECT "
        "(PARTIAL counts as wrong). ± is a 95% Wilson interval.\n",
        section("1. Ladder, clean corpus", lambda: ladder(clean, clean_grades)),
        section(
            "2. Planted documents",
            lambda: injection(inj, inj_grades, clean, clean_grades),
        ),
        section(
            "3a. Frontier generator, planted documents (rung D prompts)",
            lambda: frontier(inj + f_inj, {**inj_grades, **f_inj_grades}, "D"),
        ),
        section(
            "3b. Frontier generator, old-vs-new and near-miss (rung D prompts)",
            lambda: frontier(clean + f_clean, {**clean_grades, **f_clean_grades}, "D"),
        ),
        section(
            "4a. Confidence vs correctness, rung F, clean",
            lambda: confidence(clean, clean_grades, "F"),
        ),
        section(
            "4b. Confidence vs correctness, rung F, planted documents",
            lambda: confidence(inj, inj_grades, "F"),
        ),
        section(
            "4c. Answerability gate decisions, rung G, clean",
            lambda: gate_outcomes(clean, clean_grades, "G"),
        ),
        section(
            "5. Why answers were wrong (clean, all rungs)",
            lambda: why_wrong(clean, clean_grades),
        ),
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = parser.parse_args()
    report = build()
    print(report)
    if args.out:
        args.out.write_text(report)


if __name__ == "__main__":
    main()
