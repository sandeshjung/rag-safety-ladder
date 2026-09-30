"""A small UI for looking at what each rung actually said.

Two modes:

- Browse results replays the recorded rows in results/. It makes no model
  calls, so it needs no API keys, costs no quota and shows exactly the
  answers the published numbers were computed from.
- Ask live runs a question of your own through any rung. It calls Gemini and
  Groq, so it needs keys in .env and spends free-tier quota.

Run:  uv run --group ui streamlit run scripts/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from report import build as build_report  # noqa: E402
from report import rows_and_grades  # noqa: E402
from run_grid import REGISTER, VERSIONS, load_injected_chunks  # noqa: E402

from refuses_to_lie.analysis import INJECTION_DOC_PREFIX, answered, cited_doc_ids  # noqa: E402
from refuses_to_lie.config import ALL_CONFIGS  # noqa: E402
from refuses_to_lie.corpus import Chunk  # noqa: E402
from refuses_to_lie.grading import grade_for, gradeable  # noqa: E402
from refuses_to_lie.pipeline import load_corpus_chunks  # noqa: E402
from refuses_to_lie.versions import load_versions  # noqa: E402

EVAL_FILE = ROOT / "eval" / "questions.json"
CONFIGS = {c.id: c for c in ALL_CONFIGS}

RESULT_SETS = {
    "clean": "Clean library (132-question sample)",
    "injection": "Planted fake documents (25 questions)",
    "frontier-injection": "Claude Opus 5.5 on rung D prompts, fake documents",
    "frontier-clean": "Claude Opus 5.5 on rung D prompts, old-vs-new and near-miss",
}

CATEGORY_NAMES = {
    "answerable_single": "answerable, one document",
    "answerable_synthesis": "answerable, several documents",
    "contradictory_stale": "old vs new versions",
    "false_premise": "false premise",
    "near_miss": "near miss (not in the documents)",
    "out_of_scope": "out of scope",
    "prompt_injection": "planted fake document",
}

VERDICT_BADGE = {
    "CORRECT": ":green-badge[graded correct]",
    "PARTIAL": ":orange-badge[graded partial]",
    "INCORRECT": ":red-badge[graded incorrect]",
}


@st.cache_data
def questions() -> dict[str, dict]:
    return {q["id"]: q for q in json.loads(EVAL_FILE.read_text())}


@st.cache_resource
def chunks() -> dict[str, Chunk]:
    corpus = ROOT / "corpus"
    loaded = load_corpus_chunks(corpus / "employer", corpus / "statutory")
    loaded += load_injected_chunks(corpus / "injected")
    return {c.chunk_id: c for c in loaded}


@st.cache_data
def superseded() -> dict[str, str]:
    return load_versions(VERSIONS)


@st.cache_data
def results(name: str) -> tuple[list[dict], dict]:
    rows, grades = rows_and_grades(name)
    if name.startswith("frontier-"):
        # The frontier files hold only Claude's answers; put Gemini's rung D
        # answer to the same prompt beside each one.
        base_rows, base_grades = rows_and_grades(name.removeprefix("frontier-"))
        asked = {r["question_id"] for r in rows}
        rows = [r for r in base_rows if r["config_id"] == "D" and r["question_id"] in asked]
        rows += rows_and_grades(name)[0]
        grades = {**base_grades, **grades}
    return rows, grades


@st.cache_data
def report_markdown() -> str:
    return build_report()


def rung_name(config_id: str) -> str:
    if config_id in CONFIGS:
        return f"{config_id} · {CONFIGS[config_id].label}"
    if config_id == "D-opus":
        return "D with Claude Opus 5.5 as the generator"
    return config_id


def doc_badges(doc_id: str) -> str:
    if doc_id.startswith(INJECTION_DOC_PREFIX):
        return " :red-badge[planted fake]"
    if doc_id in superseded():
        return " :orange-badge[superseded]"
    return ""


def show_passages(chunk_ids: list[str], cited: set[str]) -> None:
    by_id = chunks()
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        chunk = by_id.get(chunk_id)
        doc_id = chunk.doc_id if chunk else chunk_id
        mark = " :blue-badge[cited]" if chunk_id in cited else ""
        with st.expander(f"{rank}. {doc_id}", expanded=False):
            st.markdown(f"`{chunk_id}`{doc_badges(doc_id)}{mark}")
            st.text(chunk.text if chunk else "(passage not in the current corpus)")


def show_row(row: dict, grades: dict) -> None:
    if "error" in row:
        st.error(f"This run failed: {row['error']}")
        return
    badges = [":gray-badge[refused]" if not answered(row) else ":blue-badge[answered]"]
    grade = grade_for(grades, row) if gradeable(row) else None
    if grade is not None:
        badges.append(VERDICT_BADGE.get(grade.verdict, f":gray-badge[{grade.verdict}]"))
    if any(d.startswith(INJECTION_DOC_PREFIX) for d in cited_doc_ids(row)):
        badges.append(":red-badge[cited a planted fake]")
    confidence = row.get("confidence") or {}
    score = confidence.get("composite")
    if isinstance(score, int | float):
        badges.append(f":gray-badge[confidence {score:.2f}]")
    st.markdown(" ".join(badges))
    st.markdown(row.get("answer") or "_(no answer text)_")

    removed = [v for v in row.get("verdicts", []) if v.get("verdict") != "SUPPORTED"]
    if removed:
        with st.expander(f"Verifier flagged {len(removed)} claim(s)"):
            for v in removed:
                st.markdown(f"- **{v['verdict']}** · {v['claim_text']} `{v['chunk_id']}`")

    cited = {c["chunk_id"] for c in row.get("citations", [])}
    st.markdown("**Passages the model was shown**")
    show_passages(row.get("retrieved_chunk_ids", []), cited)


def browse() -> None:
    # The published sets first, then any file a reader's own runs wrote.
    found = sorted(
        p.stem for p in (ROOT / "results").glob("*.jsonl") if not p.stem.endswith("-grades")
    )
    names = [n for n in RESULT_SETS if n in found] + [n for n in found if n not in RESULT_SETS]
    name = st.sidebar.selectbox(
        "Result set", names, format_func=lambda n: RESULT_SETS.get(n, f"results/{n}.jsonl")
    )
    rows, grades = results(name)
    if not rows:
        st.info(f"results/{name}.jsonl is empty or missing.")
        return

    qs = questions()
    by_question: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["config_id"]] = row

    categories = sorted({qs[q]["category"] for q in by_question})
    category = st.sidebar.selectbox(
        "Question type",
        ["all", *categories],
        format_func=lambda c: "all" if c == "all" else CATEGORY_NAMES.get(c, c),
    )
    ids = sorted(q for q in by_question if category in ("all", qs[q]["category"]))
    qid = st.sidebar.selectbox(
        f"Question ({len(ids)})", ids, format_func=lambda q: f"{q} · {qs[q]['question'][:60]}"
    )

    question = qs[qid]
    st.subheader(question["question"])
    st.markdown(
        f":gray-badge[{qid}] :gray-badge[{CATEGORY_NAMES.get(question['category'])}] "
        + (
            ":green-badge[answerable]"
            if question["expected_answerable"]
            else ":orange-badge[should not be answered as asked]"
        )
    )
    with st.expander("Reference answer and notes", expanded=True):
        st.markdown(f"**Expected:** {question['expected_answer_summary']}")
        st.markdown(f"**Source:** {', '.join(question['expected_doc_ids']) or 'none'}")
        if question.get("injection_behaviour"):
            captured = question["injection_behaviour"]
            st.markdown(f"**What a captured answer looks like:** {captured}")

    runs = by_question[qid]
    rungs = st.multiselect(
        "Rungs to compare",
        list(runs),
        default=[r for r in ("A", "F", "G", "I") if r in runs] or list(runs)[:3],
        format_func=rung_name,
    )
    for config_id, column in zip(rungs, st.columns(max(len(rungs), 1)), strict=False):
        with column:
            st.markdown(f"#### {rung_name(config_id)}")
            show_row(runs[config_id], grades)


def scorecard() -> None:
    st.markdown(
        "Every number here is recomputed from the rows in `results/` by "
        "`scripts/report.py`; it is the same table the write-up quotes."
    )
    st.markdown(report_markdown())


@st.cache_resource
def live_index(include_injected: bool):  # type: ignore[no-untyped-def]
    from refuses_to_lie.retrieval import Index

    loaded = [c for c in chunks().values() if not c.doc_id.startswith(INJECTION_DOC_PREFIX)]
    if include_injected:
        loaded = list(chunks().values())
    return Index(loaded)


def ask_live() -> None:
    from refuses_to_lie.answering import answer_question
    from refuses_to_lie.provenance import trusted_doc_ids
    from refuses_to_lie.rerank import get_reranker
    from refuses_to_lie.versions import superseded_doc_ids

    missing = [k for k in ("GOOGLE_API_KEY", "GROQ_API_KEY") if not os.environ.get(k)]
    if missing:
        st.warning(
            f"Live mode needs {' and '.join(missing)} in .env (see .env.example). "
            "Browse results works without any keys."
        )
    st.caption("Calls Gemini (answers) and Groq (verifier and gate), so it spends quota.")

    question = st.text_input(
        "Your question",
        "How much annual leave does a full-time member of staff get on appointment?",
    )
    config_id = st.selectbox(
        "Rung", list(CONFIGS), index=list(CONFIGS).index("I"), format_func=rung_name
    )
    include_injected = st.checkbox("Put the 25 planted fake documents in the library")
    if not st.button("Ask", type="primary", disabled=bool(missing)):
        return

    config = CONFIGS[config_id]
    trusted = trusted_doc_ids(REGISTER)
    with st.spinner("Searching, answering and checking..."):
        answer = answer_question(
            question,
            live_index(include_injected),
            config,
            reranker=get_reranker() if config.rerank else None,
            trusted_docs=trusted if config.require_provenance else None,
            superseded_docs=(
                superseded_doc_ids(VERSIONS, trusted) if config.exclude_superseded else None
            ),
        )
    row = {
        "answer": answer.text,
        "abstained": answer.abstained,
        "retrieved_doc_ids": [h.chunk.doc_id for h in answer.hits],
        "retrieved_chunk_ids": [h.chunk.chunk_id for h in answer.hits],
        "citations": [{"chunk_id": c.chunk_id} for c in answer.citations],
        "verdicts": [vars(v) for v in answer.verdicts],
        "confidence": (
            {"composite": answer.confidence.composite} if answer.confidence else None
        ),
    }
    show_row(row, {})


def main() -> None:
    st.set_page_config(page_title="rag-safety-ladder", layout="wide")
    st.sidebar.title("rag-safety-ladder")
    mode = st.sidebar.radio("Mode", ["Browse results", "Scorecard", "Ask live"])
    {"Browse results": browse, "Scorecard": scorecard, "Ask live": ask_live}[mode]()


main()
