"""Which documents have been replaced by a newer version.

The clean grid's wrong answers were almost never a retrieval miss: the
right document was in the context for nearly all of them. About a quarter
cited a superseded document instead -- an archived GOV.UK page quoting last
year's rate, or the old numbering of a policy that has since been reissued.
Both copies are genuine and registered, so provenance cannot tell them
apart, and the model has no reliable way to know which one is in force.

Document control does know. The version record (corpus/versions.json) maps
each superseded document to the one that replaced it, and rung I drops a
superseded document from the context whenever its replacement is
registered. A superseded document whose replacement is missing stays in:
an out-of-date answer the reader can check is better than no source at all.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_versions(path: Path) -> dict[str, str]:
    """Superseded doc_id -> the doc_id that replaced it."""
    versions: dict[str, str] = json.loads(path.read_text())
    return versions


def superseded_doc_ids(path: Path, registered: frozenset[str]) -> frozenset[str]:
    """Superseded documents whose replacement is registered, so safe to drop."""
    versions = load_versions(path)
    return frozenset(old for old, new in versions.items() if {old, new} <= registered)
