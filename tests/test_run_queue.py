import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("run_queue", ROOT / "scripts" / "run_queue.py")
run_queue = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_queue)


def test_every_stage_only_passes_flags_its_script_accepts():
    # A mistyped flag makes a stage exit non-zero on every attempt, and the
    # queue would then spend a week retrying it while every later stage
    # waited. Catch that here rather than in the log on day three.
    for name, command in run_queue.STAGES:
        script = ROOT / next(part for part in command if part.startswith("scripts/"))
        assert script.exists(), f"{name}: {script} does not exist"
        accepted = set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', script.read_text()))
        for flag in (part for part in command if part.startswith("--")):
            assert flag in accepted, f"{name}: {script.name} has no {flag} option"


def test_parser_reruns_finish_before_any_grading():
    # Grading an answer the parser re-run is about to change spends judge
    # quota on text that will not be reported.
    names = [name for name, _ in run_queue.STAGES]
    last_rerun = max(i for i, n in enumerate(names) if n.startswith("re-run "))
    first_grade = min(i for i, n in enumerate(names) if n.startswith(("grade", "regrade")))
    assert last_rerun < first_grade


def test_grid_stages_stop_early_instead_of_grinding_through_a_dead_quota():
    for name, command in run_queue.STAGES:
        if "scripts/run_grid.py" in command:
            assert "--max-consecutive-failures" in command, name
