"""Thin, retrying, disk-cached clients for the two LLM providers this
project uses: Google Gemini (generator) and Groq (verifier).

Both free tiers rate-limit, so calls are defended four ways. They are
paced to stay under the published per-minute limit rather than discovering
it by failing; every request carries a timeout, so a stalled connection
raises instead of parking the run forever; on a 429, 5xx or transport
error they back off, honouring the delay the provider asks for instead of
guessing one; and when given a cache_key they write the result to disk the
moment it arrives and skip the call entirely if that key is already there.

Together that is the "resumable, not restartable" property an overnight
eval run needs: it survives a rate limit without losing progress, and a
rerun pays only for what genuinely has not been done.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import httpx
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

DEFAULT_CACHE_DIR = Path(".cache/llm")
# Generous enough for a slow generation, short enough that a dead socket
# surfaces as a retryable error instead of parking the run.
REQUEST_TIMEOUT_MS = 120_000
GROQ_TIMEOUT_S = REQUEST_TIMEOUT_MS / 1000
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_google_client: genai.Client | None = None

# Keys supplied for one caller only -- a visitor to the hosted app using
# their own quota. A ContextVar rather than os.environ: the environment is
# shared by every session in the process, so a key put there would be spent
# on (and readable by) whoever asks next. Set only inside `api_keys`, never
# written to disk, and never used to build the shared cached client.
_session_keys: ContextVar[dict[str, str] | None] = ContextVar("session_keys", default=None)


@contextmanager
def api_keys(google: str, groq: str) -> Iterator[None]:
    """Use these keys for the calls made inside the block, then forget them."""
    token = _session_keys.set({"GOOGLE_API_KEY": google, "GROQ_API_KEY": groq})
    try:
        yield
    finally:
        _session_keys.reset(token)


def _session_key(name: str) -> str | None:
    keys = _session_keys.get()
    return keys[name] if keys else None


def _new_google_client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        # Without this the client waits forever on a stalled connection.
        # A hang is worse than a failure here: backoff only retries what
        # raises, so one dead socket silently parks an overnight grid
        # run rather than costing it a row. Cost us 50 idle minutes.
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def _get_google_client() -> genai.Client:
    global _google_client
    session_key = _session_key("GOOGLE_API_KEY")
    if session_key:
        # Built per call and dropped after it, so a visitor's key is never
        # kept in the module-level client another session would reuse.
        return _new_google_client(session_key)
    if _google_client is None:
        _google_client = _new_google_client(os.environ["GOOGLE_API_KEY"])
    return _google_client


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    # httpx, not requests, is what google-genai speaks, so its transport
    # failures arrive as a different exception tree entirely. A server
    # hanging up mid-request is the textbook retryable error, and missing
    # it here cost a grid row to "Server disconnected without sending a
    # response" -- a failure that would have succeeded on the next try.
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, requests.exceptions.ConnectionError | requests.exceptions.Timeout)


_RETRY_AFTER_PATTERNS = (
    re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'"),
    re.compile(r"retry in (\d+(?:\.\d+)?)s"),
)
MAX_BACKOFF_SECONDS = 120.0


def _server_retry_delay(exc: Exception) -> float | None:
    """How long the provider asked us to wait, if it said.

    Both providers volunteer this, and guessing instead of reading it is
    what turns a rate limit into a failed row: Gemini's free tier allows 5
    requests a minute and replies "retryDelay: 19s", so an exponential
    schedule that tops out below 19s burns every attempt inside the window
    it was told to sit out.
    """
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        header = exc.response.headers.get("Retry-After")
        if header:
            try:
                return float(header)
            except (TypeError, ValueError):
                # Retry-After also has an HTTP-date form, and a stubbed
                # response may hand back something stranger still. Neither
                # is worth parsing: fall through to the exponential schedule.
                pass

    for pattern in _RETRY_AFTER_PATTERNS:
        match = pattern.search(str(exc))
        if match:
            return float(match.group(1))
    return None


def with_backoff(fn, max_retries: int = 6, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1 or not _is_retryable(exc):
                raise
            delay = max(base_delay * (2**attempt), _server_retry_delay(exc) or 0.0)
            time.sleep(min(delay, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5))
    raise AssertionError("unreachable")


class RateLimiter:
    """Keep a minimum gap between calls to one provider.

    Backoff alone recovers from a rate limit; it does not avoid one. On a
    5-requests-per-minute tier that means roughly one wasted request in
    every five, each costing a full retry window. Pacing up front turns a
    jagged hit-wait-retry loop into a steady drip, which over a grid of
    thousands of calls is the difference between hours and most of a day.

    Deliberately per-process and approximate: the grid runs in one process,
    so anything more elaborate would be machinery without a user.
    """

    def __init__(self, requests_per_minute: float) -> None:
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        # None rather than 0.0: there is no gap to honour before the first
        # call, and a zero here would stall it for a full interval.
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            gap = time.monotonic() - self._last_call
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
        self._last_call = time.monotonic()


def _limiter_from_env(var: str, default_rpm: float) -> RateLimiter:
    """Free-tier limits are the default; a paid key can raise them."""
    return RateLimiter(float(os.environ.get(var, default_rpm)))


# Gemini's free tier allows 5 generate_content requests a minute; Groq's is
# more generous. Both are overridable for anyone running with a paid key.
_gemini_limiter = _limiter_from_env("GEMINI_RPM", 5)
_groq_limiter = _limiter_from_env("GROQ_RPM", 28)


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key.replace('/', '_')}.json"


def _read_cache(cache_dir: Path, key: str) -> str | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    result: str = json.loads(path.read_text())["text"]
    return result


def _write_cache(cache_dir: Path, key: str, text: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, key).write_text(json.dumps({"text": text}))


def _cached_call(send: Callable[[], str], cache_key: str | None, cache_dir: Path) -> str:
    """Read-through cache around a provider call, with backoff on the miss.

    Every provider shares this so the resumability policy lives in exactly
    one place: a cache hit never reaches the network, and a result is
    written the moment it arrives rather than at the end of a batch.
    """
    if cache_key is not None:
        cached = _read_cache(cache_dir, cache_key)
        if cached is not None:
            return cached
    text = with_backoff(send)
    if cache_key is not None:
        _write_cache(cache_dir, cache_key, text)
    return text


def call_gemini(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    cache_key: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> str:
    def send() -> str:
        # Inside send() so a cache hit is never throttled: replaying a
        # finished run from disk should cost nothing. The limiter paces the
        # environment's key; a caller's own key has its own limits, and
        # backoff covers them.
        if not _session_key("GOOGLE_API_KEY"):
            _gemini_limiter.wait()
        # Held in a local for the whole call: a per-session client has no
        # other reference, and once collected it closes its connection
        # mid-request ("the client has been closed").
        client = _get_google_client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return response.text or ""

    return _cached_call(send, cache_key, cache_dir)


def call_groq(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    cache_key: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> str:
    def send() -> str:
        session_key = _session_key("GROQ_API_KEY")
        if not session_key:
            _groq_limiter.wait()
        api_key = session_key or os.environ["GROQ_API_KEY"]
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            timeout=GROQ_TIMEOUT_S,
        )
        response.raise_for_status()
        result: str = response.json()["choices"][0]["message"]["content"]
        return result

    return _cached_call(send, cache_key, cache_dir)
