from unittest.mock import Mock

import httpx
import pytest
import requests

from refuses_to_lie import llm_client
from refuses_to_lie.config import A
from refuses_to_lie.llm_client import (
    MAX_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_MS,
    RateLimiter,
    _server_retry_delay,
    call_gemini,
    call_groq,
    with_backoff,
)


def _http_error(
    status_code: int, retry_after: str | None = None
) -> requests.exceptions.HTTPError:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.headers = {"Retry-After": retry_after} if retry_after else {}
    return requests.exceptions.HTTPError(response=response)


def test_with_backoff_retries_transient_failures_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429)
        return "ok"

    result = with_backoff(flaky, max_retries=5, base_delay=0.001)

    assert result == "ok"
    assert len(calls) == 3


def test_with_backoff_gives_up_after_max_retries():
    calls = []

    def always_fails():
        calls.append(1)
        raise _http_error(503)

    with pytest.raises(requests.exceptions.HTTPError):
        with_backoff(always_fails, max_retries=3, base_delay=0.001)

    assert len(calls) == 3


def test_with_backoff_does_not_retry_non_retryable_errors():
    calls = []

    def bad_request():
        calls.append(1)
        raise _http_error(400)

    with pytest.raises(requests.exceptions.HTTPError):
        with_backoff(bad_request, max_retries=5, base_delay=0.001)

    # A 400 (bad request) is never going to succeed on retry, unlike a 429 or
    # 503 — retrying it would just burn through the rate-limit budget.
    assert len(calls) == 1


@pytest.mark.live
def test_call_gemini_writes_and_reuses_cache(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    result = call_gemini(
        "Reply with exactly the single word: pineapple",
        model=A.generator_model,
        cache_key="test-gemini-pineapple",
        cache_dir=cache_dir,
    )
    assert "pineapple" in result.lower()
    assert (cache_dir / "test-gemini-pineapple.json").exists()

    # Second call with the same key must come from disk, not the network —
    # corrupt the cache dir's write path to prove no live call happens.
    cached_result = call_gemini(
        "Reply with exactly the single word: pineapple",
        model="nonexistent-model-would-error",
        cache_key="test-gemini-pineapple",
        cache_dir=cache_dir,
    )
    assert cached_result == result


@pytest.mark.live
def test_call_groq_writes_and_reuses_cache(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    result = call_groq(
        "Reply with exactly the single word: mango",
        model=A.verifier_model,
        cache_key="test-groq-mango",
        cache_dir=cache_dir,
    )
    assert "mango" in result.lower()
    assert (cache_dir / "test-groq-mango.json").exists()

    cached_result = call_groq(
        "Reply with exactly the single word: mango",
        model="nonexistent-model-would-error",
        cache_key="test-groq-mango",
        cache_dir=cache_dir,
    )
    assert cached_result == result


GEMINI_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota... Please retry in 19.141830479s.', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '19s'}]}}"
)


def test_server_retry_delay_is_read_from_a_gemini_429():
    # The regression that made the first grid run fail: Gemini's free tier
    # allows 5 requests a minute and says so in the error. An exponential
    # schedule topping out below that delay burns every retry inside the
    # window it was told to wait, and the row fails for no good reason.
    assert _server_retry_delay(Exception(GEMINI_429)) == 19.0


def test_server_retry_delay_is_read_from_a_retry_after_header():
    assert _server_retry_delay(_http_error(429, retry_after="30")) == 30.0


def test_server_retry_delay_tolerates_an_http_date_header():
    dated = _http_error(429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
    assert _server_retry_delay(dated) is None


def test_server_retry_delay_is_none_when_the_provider_says_nothing():
    assert _server_retry_delay(_http_error(503)) is None


def test_backoff_waits_at_least_as_long_as_the_server_asked(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("refuses_to_lie.llm_client.time.sleep", slept.append)

    calls = []

    def rate_limited():
        calls.append(1)
        if len(calls) < 2:
            raise _http_error(429, retry_after="19")
        return "ok"

    assert with_backoff(rate_limited, base_delay=0.001) == "ok"
    assert slept and slept[0] >= 19.0


def test_backoff_never_sleeps_longer_than_the_cap(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("refuses_to_lie.llm_client.time.sleep", slept.append)

    calls = []

    def absurd_delay():
        calls.append(1)
        if len(calls) < 2:
            raise _http_error(429, retry_after="99999")
        return "ok"

    assert with_backoff(absurd_delay, base_delay=0.001) == "ok"
    assert slept[0] <= MAX_BACKOFF_SECONDS + 1


def test_rate_limiter_spaces_calls_by_the_configured_interval(monkeypatch):
    # Pacing is what keeps a 5-per-minute tier from being discovered by
    # failing; without it roughly one request in five is wasted.
    slept: list[float] = []
    clock = [0.0]
    monkeypatch.setattr("refuses_to_lie.llm_client.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("refuses_to_lie.llm_client.time.sleep", slept.append)

    limiter = RateLimiter(requests_per_minute=5)
    limiter.wait()  # first call is immediate
    assert slept == []

    clock[0] = 2.0
    limiter.wait()
    assert slept == [10.0]  # 12s interval, 2s already elapsed


def test_rate_limiter_with_no_limit_never_sleeps(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("refuses_to_lie.llm_client.time.sleep", slept.append)

    limiter = RateLimiter(requests_per_minute=0)
    limiter.wait()
    limiter.wait()
    assert slept == []


def test_httpx_transport_errors_are_retried(monkeypatch):
    # google-genai speaks httpx, so a server hanging up mid-request does
    # not arrive as a requests exception. Missing this cost a real grid row.
    monkeypatch.setattr("refuses_to_lie.llm_client.time.sleep", lambda _: None)
    calls = []

    def disconnects_once():
        calls.append(1)
        if len(calls) < 2:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return "ok"

    assert with_backoff(disconnects_once, base_delay=0.001) == "ok"
    assert len(calls) == 2


def test_the_google_client_is_built_with_a_request_timeout(monkeypatch):
    # A hang is worse than a failure: backoff only retries what raises, so
    # an untimed request parks a multi-day grid run instead of costing it a
    # single row. This silently idled a real run for 50 minutes.
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("refuses_to_lie.llm_client.genai.Client", FakeClient)
    monkeypatch.setattr("refuses_to_lie.llm_client._google_client", None)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    llm_client._get_google_client()

    assert captured["http_options"].timeout == REQUEST_TIMEOUT_MS
    assert REQUEST_TIMEOUT_MS > 0


def test_session_keys_override_the_environment_only_inside_the_block(monkeypatch):
    # The hosted app runs every visitor in one process. A visitor's key must
    # reach their own calls and nobody else's, and be gone afterwards.
    sent = []

    def fake_post(url, headers, json, timeout):
        sent.append(headers["Authorization"])
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        return response

    monkeypatch.setattr("refuses_to_lie.llm_client.requests.post", fake_post)
    monkeypatch.setattr("refuses_to_lie.llm_client._groq_limiter.wait", lambda: None)
    monkeypatch.setenv("GROQ_API_KEY", "owner-key")

    with llm_client.api_keys(google="visitor-google", groq="visitor-groq"):
        call_groq("hi", model="m")
    call_groq("hi", model="m")

    assert sent == ["Bearer visitor-groq", "Bearer owner-key"]


def test_a_session_key_never_becomes_the_shared_google_client(monkeypatch):
    built = []

    class FakeClient:
        def __init__(self, **kwargs):
            built.append(kwargs["api_key"])

    monkeypatch.setattr("refuses_to_lie.llm_client.genai.Client", FakeClient)
    monkeypatch.setattr("refuses_to_lie.llm_client._google_client", None)
    monkeypatch.setenv("GOOGLE_API_KEY", "owner-key")

    with llm_client.api_keys(google="visitor-google", groq="visitor-groq"):
        llm_client._get_google_client()
    assert llm_client._google_client is None

    llm_client._get_google_client()
    assert built == ["visitor-google", "owner-key"]


def test_session_keys_are_forgotten_even_when_the_call_fails():
    with pytest.raises(RuntimeError), llm_client.api_keys(google="g", groq="q"):
        raise RuntimeError("provider down")
    assert llm_client._session_key("GROQ_API_KEY") is None
