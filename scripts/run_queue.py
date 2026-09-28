"""Work through the remaining runs, waiting out provider quotas as they come.

Both free tiers now bind: Gemini at 500 requests a day, Groq at 200,000
tokens a day on a rolling window. Rather than someone checking back and
re-running commands by hand, this runs each stage in priority order and,
whenever one stops short, waits and tries it again. Every stage is
resumable -- completed rows and grades are skipped, the LLM cache replays
finished calls -- so a retry pays only for what is genuinely missing.

A stage is complete when its command exits 0. run_grid exits non-zero when
any row failed or it stopped early; grade_answers exits non-zero when a
grading call gives out. Stages run strictly in order, so a later stage never
spends a quota an earlier, more important one is waiting for.

Runs detached so it outlives the terminal:
  nohup uv run python scripts/run_queue.py >/dev/null 2>&1 &
Watch:  tail -f .cache/queue.log
Stop:   kill $(cat .cache/queue.pid)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / ".cache" / "queue.log"
PIDFILE = ROOT / ".cache" / "queue.pid"

WAIT_MINUTES = 20
# About a week of 20-minute retries per stage before giving up on it.
MAX_ATTEMPTS = 500

GRADE = ["uv", "run", "python", "scripts/grade_answers.py"]
GRID = ["uv", "run", "python", "scripts/run_grid.py", "--max-consecutive-failures", "3"]
LADDER = "A,B,C,D,E,F"

CLEAN = ["--no-injected", "--results", "results/clean.jsonl"]
INJECTION = ["--category", "prompt_injection", "--results", "results/injection.jsonl"]

# Priority order. Citation parser v2 first: the citing rungs' rows are
# re-run so every grade after this point is of a final answer. The re-runs
# reuse cached drafts (same fingerprint, same cache key), so they cost
# verifier calls, not generator quota. Then grading, then the frontier
# comparison (cheap, and the write-up needs it), then clean G, the most
# expensive stage. The clean grid is the union of the 40- and 120-question
# stratified samples, so both are re-run.
STAGES: list[tuple[str, list[str]]] = [
    (
        "re-run clean D-F, parser v2 (40 slice)",
        [*GRID, "--sample", "40", *CLEAN, "--configs", "D,E,F"],
    ),
    (
        "re-run clean D-F, parser v2 (120 slice)",
        [*GRID, "--sample", "120", *CLEAN, "--configs", "D,E,F"],
    ),
    ("re-run injection D-H, parser v2", [*GRID, *INJECTION, "--configs", "D,E,F,G,H"]),
    ("regrade clean A-F (judge v2)", [*GRADE, "--configs", LADDER]),
    (
        "regrade injection A-H (judge v2)",
        [*GRADE, "--results", "results/injection.jsonl", "--configs", f"{LADDER},G,H"],
    ),
    (
        "grade frontier comparison, planted documents",
        [*GRADE, "--results", "results/frontier-injection.jsonl", "--include-external"],
    ),
    (
        "grade frontier comparison, old-vs-new and near-miss",
        [*GRADE, "--results", "results/frontier-clean.jsonl", "--include-external"],
    ),
    ("run clean G (40 slice)", [*GRID, "--sample", "40", *CLEAN, "--configs", "G"]),
    ("run clean G (120 slice)", [*GRID, "--sample", "120", *CLEAN, "--configs", "G"]),
    ("grade clean G (judge v2)", [*GRADE, "--configs", "G"]),
]

_child: subprocess.Popen[bytes] | None = None


def log(message: str) -> None:
    with LOG.open("a") as out:
        out.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")


def _already_running() -> bool:
    if not PIDFILE.exists():
        return False
    try:
        os.kill(int(PIDFILE.read_text().strip()), 0)
    except (ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def _stop(signum: int, _frame: object) -> None:
    """Take the running stage down with the queue.

    Without this, killing the queue leaves its child grid run going, still
    spending quota, with nothing left to supervise it.
    """
    if _child is not None and _child.poll() is None:
        _child.terminate()
    log(f"stopped by signal {signum}")
    PIDFILE.unlink(missing_ok=True)
    sys.exit(0)


def run_stage(name: str, command: list[str]) -> bool:
    global _child
    env = {**os.environ, "GEMINI_RPM": os.environ.get("GEMINI_RPM", "12")}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"START  {name}  (attempt {attempt})")
        with LOG.open("a") as out:
            # stdin from /dev/null: once the launching terminal closes, an
            # inherited stdin is revoked and every child dies at startup.
            _child = subprocess.Popen(
                command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out
            )
            code = _child.wait()
        _child = None
        if code == 0:
            log(f"DONE   {name}")
            return True
        log(f"WAIT   {name} stopped short (exit {code}); retrying in {WAIT_MINUTES} min")
        time.sleep(WAIT_MINUTES * 60)
    log(f"GIVE UP  {name} after {MAX_ATTEMPTS} attempts")
    return False


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if _already_running():
        sys.exit(f"queue already running (pid {PIDFILE.read_text().strip()})")
    PIDFILE.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log(f"queue started with {len(STAGES)} stages")
    try:
        for name, command in STAGES:
            if not run_stage(name, command):
                log("queue halted: a stage never completed")
                return
        log("ALL STAGES COMPLETE")
    finally:
        PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
