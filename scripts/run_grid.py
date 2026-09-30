"""Run the eval grid: every question against every ladder config.

Resumable, not restartable. Results are appended to a JSONL file one row at
a time, and a rerun skips any (config, question) pair already present. A
rate limit, a crash or a Ctrl-C three hours in costs you the row in flight
and nothing else — which matters because a full grid is thousands of LLM
calls against free-tier providers.

Two layers of resumability stack here: this file skips completed rows, and
llm_client caches individual provider responses to disk. So even a row that
failed halfway through (say the verifier died after generation succeeded)
replays the generation from cache rather than paying for it twice.

Config changes invalidate results automatically. Each row records a
fingerprint of the config's substantive fields, and rows whose fingerprint
no longer matches are treated as not-done, so editing a model name or
top_k silently reusing stale answers is not possible.

Usage:
  uv run python scripts/run_grid.py --sample 40          # stratified slice
  uv run python scripts/run_grid.py --configs A,B        # some rungs
  uv run python scripts/run_grid.py --category prompt_injection
  uv run python scripts/run_grid.py                      # the full grid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from refuses_to_lie.answering import answer_question
from refuses_to_lie.config import ALL_CONFIGS, LADDER, RunConfig
from refuses_to_lie.grid import (
    PARSER_VERSION,
    cache_key,
    config_fingerprint,
    load_completed,
    pending_work,
    stratified_sample,
)
from refuses_to_lie.pipeline import load_corpus_chunks, load_injected_chunks
from refuses_to_lie.provenance import trusted_doc_ids
from refuses_to_lie.rerank import Reranker, get_reranker
from refuses_to_lie.retrieval import Index
from refuses_to_lie.versions import superseded_doc_ids

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "questions.json"
DEFAULT_RESULTS = ROOT / "results" / "grid.jsonl"
REGISTER = ROOT / "corpus" / "register.json"
VERSIONS = ROOT / "corpus" / "versions.json"


def run_one(
    question: dict,
    config: RunConfig,
    index: Index,
    reranker: Reranker | None,
    includes_injected: bool,
    trusted_docs: frozenset[str] | None = None,
    superseded_docs: frozenset[str] | None = None,
) -> dict:
    started = time.monotonic()
    row: dict = {
        "question_id": question["id"],
        "category": question["category"],
        "config_id": config.id,
        "config_fingerprint": config_fingerprint(config),
        "parser_version": PARSER_VERSION,
        "expected_answerable": question["expected_answerable"],
        "corpus_includes_injected": includes_injected,
    }
    try:
        answer = answer_question(
            question["question"],
            index,
            config,
            cache_key=cache_key(config, question["id"], includes_injected),
            reranker=reranker,
            trusted_docs=trusted_docs,
            superseded_docs=superseded_docs,
        )
    except Exception as exc:
        # A failed row must not kill the grid: record it and move on, so a
        # single malformed response does not cost hours of completed work.
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_s"] = round(time.monotonic() - started, 2)
        return row

    row.update(
        {
            "answer": answer.text,
            "abstained": answer.abstained,
            "retrieved_doc_ids": [h.chunk.doc_id for h in answer.hits],
            "retrieved_chunk_ids": [h.chunk.chunk_id for h in answer.hits],
            "citations": [
                {"chunk_id": c.chunk_id, "cite_label": c.cite_label} for c in answer.citations
            ],
            "verdicts": [asdict(v) for v in answer.verdicts],
            "confidence": (
                {**asdict(answer.confidence), "composite": answer.confidence.composite}
                if answer.confidence
                else None
            ),
            "elapsed_s": round(time.monotonic() - started, 2),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="stratified subset size")
    parser.add_argument(
        "--category",
        default="",
        help="restrict to one eval category, e.g. prompt_injection",
    )
    parser.add_argument("--configs", default="", help="comma-separated rung ids, e.g. A,B")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=0,
        help="stop after this many failed rows in a row (0 = never stop early)",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--no-injected",
        action="store_true",
        help="exclude the adversarial corpus (prompt_injection questions need it)",
    )
    args = parser.parse_args()

    # The default is the published A-F ladder only, so existing commands
    # keep doing exactly what they did; the extension rungs are opt-in.
    configs = list(LADDER)
    if args.configs:
        wanted = {c.strip().upper() for c in args.configs.split(",")}
        configs = [c for c in ALL_CONFIGS if c.id in wanted]
        if not configs:
            parser.error(f"no ladder configs matched {sorted(wanted)}")

    questions = json.loads(EVAL_FILE.read_text())
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        if not questions:
            parser.error(f"no questions in category {args.category!r}")
    if args.sample:
        questions = stratified_sample(questions, args.sample)

    include_injected = not args.no_injected
    chunks = load_corpus_chunks(ROOT / "corpus" / "employer", ROOT / "corpus" / "statutory")
    if include_injected:
        chunks += load_injected_chunks(ROOT / "corpus" / "injected")

    index = Index(chunks)
    reranker = get_reranker() if any(c.rerank for c in configs) else None
    # Loaded from the committed register, not rebuilt from the corpus
    # directories: rebuilding would register whatever happens to be on disk,
    # including anything an attacker managed to drop there.
    trusted_docs = (
        trusted_doc_ids(REGISTER) if any(c.require_provenance for c in configs) else None
    )
    superseded_docs = (
        superseded_doc_ids(VERSIONS, trusted_doc_ids(REGISTER))
        if any(c.exclude_superseded for c in configs)
        else None
    )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(
        args.results, frozenset(c.id for c in ALL_CONFIGS if c.require_citations)
    )

    todo = pending_work(questions, configs, completed, include_injected)
    total = len(questions) * len(configs)
    print(
        f"{len(chunks)} chunks (injected={'yes' if include_injected else 'no'}) | "
        f"{len(questions)} questions x {len(configs)} configs = {total} rows\n"
        f"{total - len(todo)} already done, {len(todo)} to run -> {args.results}\n"
    )

    ran = 0
    failures = 0
    streak = 0
    stopped_early = False
    with args.results.open("a") as out:
        for i, (question, config) in enumerate(todo, start=1):
            row = run_one(
                question,
                config,
                index,
                reranker,
                include_injected,
                trusted_docs,
                superseded_docs,
            )
            out.write(json.dumps(row) + "\n")
            out.flush()  # one row at a time: a kill -9 loses nothing already written
            ran += 1

            if "error" in row:
                failures += 1
                streak += 1
                print(
                    f"[{i}/{len(todo)}] {config.id} {question['id']} FAILED {row['error']}",
                    flush=True,
                )
            else:
                streak = 0
                if i % 10 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)}] {config.id} {question['id']} ok", flush=True)

            # A run of consecutive failures almost always means a provider's
            # quota is gone. Every further row would spend minutes in retries
            # and still fail, so stop and let the next attempt resume instead.
            if args.max_consecutive_failures and streak >= args.max_consecutive_failures:
                stopped_early = True
                print(f"\nstopping early: {streak} consecutive failures", flush=True)
                break

    print(f"\ndone: {ran - failures} ok, {failures} failed, {len(todo) - ran} not run")
    if failures or stopped_early:
        print("rerun the same command to retry the failed and unrun rows", file=sys.stderr)
        # Non-zero so a supervisor can tell "finished" from "needs another go".
        sys.exit(1)


if __name__ == "__main__":
    main()
