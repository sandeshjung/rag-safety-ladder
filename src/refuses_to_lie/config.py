from dataclasses import dataclass, replace
from typing import Literal

Retrieval = Literal["dense", "hybrid"]
VerifierAction = Literal["off", "annotate", "drop_unsupported"]
ConfidenceSource = Literal["composite", "answerability"]


@dataclass(frozen=True)
class RunConfig:
    id: str
    label: str

    # Ladder axis
    retrieval: Retrieval = "dense"
    rerank: bool = False
    require_citations: bool = False
    verifier: VerifierAction = "off"
    abstain: bool = False

    # held constant across the whole ladder
    chunk_tokens: int = 400
    chunk_overlap: int = 64
    top_k_retrieve: int = 20
    top_k_context: int = 6
    rrf_k: int = 60
    temperature: float = 0.2
    agreement_samples: int = 3
    # Only consulted when abstain=True. Uncalibrated: picking a threshold
    # that corresponds to a real error rate is a separate step that needs
    # the eval grid to have been run first.
    abstain_threshold: float = 0.5
    # gemini-3.6-flash allows 20 requests a DAY on the free tier, which is
    # ~100 days for one full grid. The lite model has a workable daily
    # quota; the generator staying on Google also keeps it on a different
    # provider from the Groq verifier, which is the point of verifying.
    generator_model: str = "gemini-3.5-flash-lite"
    verifier_model: str = "openai/gpt-oss-120b"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- added after the A-F grid was collected ---------------------------
    # These enter the config fingerprint only when set away from their
    # default (grid.ADDED_LATER_FIELDS), so adding them leaves every existing
    # A-F row valid instead of silently marking days of results stale.
    #
    # The composite (retrieval margin + groundedness + agreement) measured
    # as anti-correlated with correctness: wrong answers scored higher.
    # "answerability" replaces it with one question-conditioned check.
    confidence_source: ConfidenceSource = "composite"
    # Injection won by source capture, not instruction-following, so the
    # defence is to stop unregistered documents reaching the context at all.
    require_provenance: bool = False


A = RunConfig(id="A", label="dense retrieval only, no abstention")
B = replace(A, id="B", label="+ hybrid retrieval (BM25 + dense, RRF)", retrieval="hybrid")
C = replace(B, id="C", label="+ cross-encoder reranker", rerank=True)
D = replace(C, id="D", label="+ mandatory inline citations", require_citations=True)
E = replace(D, id="E", label="+ groundedness verifier", verifier="drop_unsupported")
F = replace(E, id="F", label="+ threshold abstention (full system)", abstain=True)

LADDER = (A, B, C, D, E, F)

# Rungs built from what the A-F grid measured. Kept out of LADDER so the
# published ladder -- and every command that runs it by default -- is
# unchanged; run them explicitly with --configs G,H.
G = replace(
    F,
    id="G",
    label="+ answerability gate replaces composite confidence",
    confidence_source="answerability",
    # Answerability scores 1.0 / 0.5 / 0.0, so 0.75 means: answer only when
    # the context is judged to fully contain the answer.
    abstain_threshold=0.75,
)
H = replace(G, id="H", label="+ provenance register", require_provenance=True)

EXTENSIONS = (G, H)
ALL_CONFIGS = LADDER + EXTENSIONS
