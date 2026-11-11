import asyncio
import json
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import main  # noqa: E402
from google.genai import types  # noqa: E402


def setup_module(_):
    main.init_db()


def test_history_starts_with_user_turn():
    main.save_pair("a", "hello there", "2026-11-01T10:00:00+00:00", "hi", "2026-11-01T10:00:02+00:00")
    assert main.load_recent("a", 1) == []  # would be the assistant row only
    assert [r["role"] for r in main.load_recent("a", 2)] == ["user", "assistant"]


def test_recall_finds_old_message_outside_recent_window():
    main.save_pair("b", "Tell me about the Mars rover Perseverance", "2026-11-01T09:00:00+00:00", "It landed in 2021.", "2026-11-01T09:00:03+00:00")
    for i in range(15):
        main.save_pair("b", f"filler question {i}", "2026-11-02T09:00:00+00:00", f"filler answer {i}", "2026-11-02T09:00:01+00:00")
    recent = main.load_recent("b", 10)
    found = main.recall("b", "what did I ask about Perseverance?", {r["id"] for r in recent})
    assert any("Perseverance" in r["content"] for r in found)


def test_sessions_are_isolated():
    main.save_pair("c", "secret apples", "2026-11-01T09:00:00+00:00", "ok", "2026-11-01T09:00:01+00:00")
    assert main.recall("d", "apples", set()) == []


def test_search_normalises_results_and_handles_failure(monkeypatch):
    monkeypatch.setattr(main, "_search_sync", lambda q, r, k="web": [{"title": "T", "href": "https://x.test", "body": "B"}])
    results, error = asyncio.run(main.do_search({"query": "x"}))
    assert error is None and results[0]["url"] == "https://x.test"

    # news results use "url" instead of "href" and carry a date and source
    monkeypatch.setattr(main, "_search_sync", lambda q, r, k="web": [{"title": "H", "url": "https://n.test", "body": "B", "date": "2026-11-11", "source": "Outlet"}])
    results, error = asyncio.run(main.do_search({"query": "x", "kind": "news"}))
    assert results[0]["source"] == "Outlet" and results[0]["date"] == "2026-11-11"

    def boom(q, r, k="web"):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(main, "_search_sync", boom)
    results, error = asyncio.run(main.do_search({"query": "y"}))
    assert results == [] and error.startswith("Search failed")


def test_citation_numbers_are_shared_across_searches():
    numbers = {}
    a = [{"title": "A", "url": "https://a.test", "body": "", "date": "", "source": ""}]
    b = [
        {"title": "A again", "url": "https://a.test", "body": "", "date": "", "source": ""},
        {"title": "B", "url": "https://b.test", "body": "", "date": "", "source": ""},
    ]
    text_a = main.format_results(a, None, numbers)
    text_b = main.format_results(b, None, numbers)
    assert text_a.startswith("[1] A")
    assert "[1] A again" in text_b and "[2] B" in text_b  # same page keeps the same number
    assert main.format_results([], "No results.", numbers) == "No results."


def test_times_are_shown_in_the_users_timezone():
    assert main.fmt("2026-11-11T08:08:00+00:00", "Asia/Kolkata") == "2026-11-11 13:38 IST"
    assert main.fmt("2026-11-11T08:08:00+00:00", "Not/AZone") == "2026-11-11 08:08 UTC"


def collect(session, message):
    async def go():
        return [json.loads(e[6:]) async for e in main.run_agent(session, message)]
    return asyncio.run(go())


def test_agent_loop_searches_then_answers_and_saves(monkeypatch):
    """Fake model: round 1 asks for a search, round 2 answers. No network used."""
    rounds = {"n": 0}

    async def fake_stream(system, contents, allow_tools):
        rounds["n"] += 1
        if rounds["n"] == 1:
            call = types.Part(function_call=types.FunctionCall(name="web_search", args={"query": "rupee rate"}))
            yield ("parts", [call])
        else:
            last = contents[-1].parts[0].function_response.response["result"]
            assert "https://x.test" in last  # search results were fed back
            yield ("text", "The rate is X [src](https://x.test).")
            yield ("parts", [types.Part(text="The rate is X [src](https://x.test).")])

    monkeypatch.setattr(main, "stream_model", fake_stream)
    monkeypatch.setattr(main, "_search_sync", lambda q, r, k="web": [{"title": "T", "href": "https://x.test", "body": "B"}])
    events = collect("loop", "rupee rate today?")
    kinds = [e["type"] for e in events]
    assert kinds == ["reset", "status", "token", "sources", "done"]
    saved = main.load_recent("loop", 10)
    assert [r["role"] for r in saved] == ["user", "assistant"]
    assert saved[0]["content"] == "rupee rate today?"  # stored without the timestamp prefix


def test_failed_turn_is_not_saved(monkeypatch):
    async def broken(system, contents, allow_tools):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
        yield  # pragma: no cover

    monkeypatch.setattr(main, "stream_model", broken)
    events = collect("fail", "hello")
    assert events[-1]["type"] == "error" and "rate limit" in events[-1]["text"].lower()
    assert main.load_recent("fail", 10) == []


class FakeModels:
    def __init__(self, failing, error="503 UNAVAILABLE high demand"):
        self.calls, self.failing, self.error = [], failing, error

    async def generate_content_stream(self, model, contents, config):
        self.calls.append(model)
        if model in self.failing:
            raise RuntimeError(self.error)

        async def gen():
            part = types.Part(text="ok")
            yield type("C", (), {"candidates": [type("D", (), {"content": type("E", (), {"parts": [part]})()})()]})()

        return gen()


def run_stream(monkeypatch, fake):
    monkeypatch.setattr(main, "get_client", lambda: type("Cl", (), {"aio": type("A", (), {"models": fake})()})())
    monkeypatch.setattr(main, "MODEL", "primary")
    monkeypatch.setattr(main, "FALLBACK_MODELS", ["backup"])
    monkeypatch.setattr(main, "RETRY_DELAY", 0)

    async def go():
        return [x async for x in main.stream_model("sys", [], True)]

    return asyncio.run(go())


def test_overloaded_model_is_retried_then_falls_back(monkeypatch):
    fake = FakeModels(failing={"primary"})
    out = run_stream(monkeypatch, fake)
    assert fake.calls == ["primary", "primary", "backup"]
    assert out[0] == ("text", "ok")


def test_non_retryable_error_is_raised_immediately(monkeypatch):
    fake = FakeModels(failing={"primary"}, error="400 INVALID_ARGUMENT bad request")
    try:
        run_stream(monkeypatch, fake)
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert fake.calls == ["primary"]
