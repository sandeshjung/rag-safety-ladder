"""Bookkeeping for a resumable eval grid.

The grid is thousands of LLM calls against rate-limited free tiers, so it
will be interrupted — by a 429, a crash, or someone pressing Ctrl-C. The
rules that make that survivable live here rather than in the CLI, because
getting them subtly wrong is the difference between resuming a run and
silently reusing results that no longer mean anything:

  - a row counts as done only if it SUCCEEDED, so a transient failure is
    retried rather than baked in as a permanent hole in the grid
  - every row carries a fingerprint of the config that produced it, so
    editing a model name or a top_k invalidates those rows automatically
    instead of mixing settings within one reported number
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

from refuses_to_lie.config import RunConfig

# Cosmetic: changing a label should not invalidate completed work.
NON_SUBSTANTIVE_FIELDS = frozenset({"label"})

# Fields added to RunConfig after results were collected. Hashing every
# field would mean adding ANY new capability changes every config's
# fingerprint and marks every existing row stale -- days of free-tier quota
# thrown away for a feature those rows never used. So a field listed here
# enters the hash only when set away from its default; at its default it
# is, by definition, the behaviour the old rows already ran.
ADDED_LATER_FIELDS = frozenset({"confidence_source", "require_provenance"})

# Version of the citation parser (generation.citation_ids and the verifier's
# claim splitter), recorded on every row. Version 1 dropped every citation
# to a GOV.UK page and every grouped citation, so on the rungs that cite,
# its rows measured a half-working verifier and confidence score. Those
# rows are neither done nor scored. It is not part of the fingerprint on
# purpose: the fingerprint is also the LLM cache key, and keeping it lets a
# re-run reuse the cached drafts, so the parser is the only thing that
# changes between the old rows and the new.
PARSER_VERSION = 2


def is_current_parse(row: dict, citing_config_ids: frozenset[str]) -> bool:
    """False for a citing rung's row parsed by an older citation parser."""
    if row["config_id"] not in citing_config_ids:
        return True
    return row.get("parser_version", 1) >= PARSER_VERSION


def config_fingerprint(config: RunConfig) -> str:
    """Short hash over everything about a config that could change output."""
    substantive = {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if f.name not in NON_SUBSTANTIVE_FIELDS
        and not (f.name in ADDED_LATER_FIELDS and getattr(config, f.name) == f.default)
    }
    blob = json.dumps(substantive, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def corpus_tag(includes_injected: bool) -> str:
    """Which corpus a row was produced against.

    Part of both keys below, because the corpus is an input to the answer
    exactly as much as the config is. Without it, a clean-baseline run
    would skip every row a contaminated run had already done, and worse,
    replay that run's cached generations -- answers written while looking
    at adversarial documents -- as though they were clean results.
    """
    return "inj" if includes_injected else "clean"


def cache_key(config: RunConfig, question_id: str, includes_injected: bool = True) -> str:
    """Stable per (config, corpus, question); invalidated by config changes."""
    fingerprint = config_fingerprint(config)
    return f"{config.id}-{fingerprint}-{corpus_tag(includes_injected)}-{question_id}"


Completion = tuple[str, str, str, str]


def load_completed(
    path: Path, citing_config_ids: frozenset[str] = frozenset()
) -> set[Completion]:
    """(config_id, question_id, fingerprint, corpus) rows that succeeded.

    Failed rows stay in the file as a record of what went wrong but are not
    treated as done, so rerunning the same command retries exactly them.
    So do rows from `citing_config_ids` written by an older citation parser.
    """
    if not path.exists():
        return set()
    done: set[Completion] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row or not is_current_parse(row, citing_config_ids):
            continue
        done.add(
            (
                row["config_id"],
                row["question_id"],
                row["config_fingerprint"],
                corpus_tag(row.get("corpus_includes_injected", True)),
            )
        )
    return done


def pending_work(
    questions: list[dict],
    configs: list[RunConfig],
    completed: set[Completion],
    includes_injected: bool = True,
) -> list[tuple[dict, RunConfig]]:
    tag = corpus_tag(includes_injected)
    return [
        (question, config)
        for config in configs
        for question in questions
        if (config.id, question["id"], config_fingerprint(config), tag) not in completed
    ]


def stratified_sample(questions: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Sample proportionally across categories.

    A validation slice is only useful if it exercises the hard paths, so
    taking a plain random sample risks a slice with no abstention cases or
    no injection cases in it at all.
    """
    by_category: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        by_category[question["category"]].append(question)

    rng = random.Random(seed)
    picked: list[dict] = []
    for _, group in sorted(by_category.items()):
        share = max(1, round(n * len(group) / len(questions)))
        picked.extend(rng.sample(group, min(share, len(group))))
    rng.shuffle(picked)
    return picked[:n]
