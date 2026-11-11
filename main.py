"""Chat agent with one tool (web search). See design.md for the reasoning."""
import asyncio
import json
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ddgs import DDGS
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DB_PATH = os.getenv("DB_PATH", "memory.db")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-3.6-flash").split(",")
    if m.strip()
]
RETRY_DELAY = 1.0       # seconds to wait after a 503 before retrying
MAX_SEARCH_ROUNDS = 2   # the model may search up to 2 times, then must answer
HISTORY_MESSAGES = 20   # recent messages sent verbatim
RECALL_LIMIT = 5        # older messages pulled in by keyword match
SEARCH_TIMEOUT = 8      # seconds
CACHE_TTL = 600         # seconds

_client = None
_has_fts = False
_cache = {}

SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the web. Use for anything recent, changing, or specific "
                "(news, prices, people's current roles, docs, statistics) or when unsure. "
                "Do not use for stable general knowledge."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="Short, specific search query."),
                    "kind": types.Schema(
                        type=types.Type.STRING,
                        enum=["web", "news"],
                        description="Use 'news' for headlines and recent events. Default 'web'.",
                    ),
                    "recency": types.Schema(
                        type=types.Type.STRING,
                        enum=["d", "w", "m", "y"],
                        description="Optional: limit to past day, week, month or year.",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


def get_client():
    global _client
    if _client is None:
        # v1beta is required for function calling (the tool-use feature this agent relies on).
        _client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))  # reads GEMINI_API_KEY
    return _client


# ---------- memory (SQLite) ----------

@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    global _has_fts
    with connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL)"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id)")
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content)")
            _has_fts = True
        except sqlite3.OperationalError:
            _has_fts = False  # recall falls back to recent history only


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt(iso: str, tz: str = "UTC") -> str:
    """Format a stored UTC timestamp in the user's own timezone, e.g. 2026-11-11 13:38 IST."""
    try:
        zone = ZoneInfo(tz)
    except Exception:  # unknown zone, or no tz database on this machine
        zone = timezone.utc
    dt = datetime.fromisoformat(iso).astimezone(zone)
    return dt.strftime("%Y-%m-%d %H:%M ") + (dt.tzname() or "UTC")


def save_pair(session_id, user_text, user_ts, assistant_text, assistant_ts):
    with connect() as conn:
        for role, content, ts in (
            ("user", user_text, user_ts),
            ("assistant", assistant_text, assistant_ts),
        ):
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, role, content, ts),
            )
            if _has_fts:
                conn.execute(
                    "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
                    (cur.lastrowid, content),
                )


def load_recent(session_id, limit):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    rows = [dict(r) for r in reversed(rows)]
    while rows and rows[0]["role"] != "user":  # API needs the first message to be a user turn
        rows.pop(0)
    return rows


def recall(session_id, query, exclude_ids):
    """Keyword-match older messages so the agent can answer 'what did I ask about X?'"""
    tokens = list(dict.fromkeys(re.findall(r"[a-z0-9]{4,}", query.lower())))[:8]
    if not tokens or not _has_fts:
        return []
    match = " OR ".join(f'"{t}"' for t in tokens)
    exclude = list(exclude_ids) or [0]
    marks = ",".join("?" * len(exclude))
    try:
        with connect() as conn:
            rows = conn.execute(
                f"""SELECT m.* FROM messages_fts f JOIN messages m ON m.id = f.rowid
                    WHERE messages_fts MATCH ? AND m.session_id = ? AND m.id NOT IN ({marks})
                    ORDER BY rank LIMIT ?""",
                (match, session_id, *exclude, RECALL_LIMIT),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return sorted((dict(r) for r in rows), key=lambda r: r["id"])


# ---------- the one tool: web search ----------

def _search_sync(query, recency, kind="web"):
    key = (query.strip().lower(), recency, kind)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    kwargs = {"max_results": 5}
    if recency in ("d", "w", "m", "y"):
        kwargs["timelimit"] = recency
    ddgs = DDGS()
    results = ddgs.news(query, **kwargs) if kind == "news" else ddgs.text(query, **kwargs)
    _cache[key] = (time.time(), results)
    return results


async def do_search(tool_input):
    """Returns (results, error). Each result: {title, url, body, date, source}."""
    query = str(tool_input.get("query", "")).strip()[:200]
    if not query:
        return [], "Empty query."
    kind = "news" if tool_input.get("kind") == "news" else "web"
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_search_sync, query, tool_input.get("recency"), kind),
            timeout=SEARCH_TIMEOUT,
        )
    except Exception as exc:  # timeout, rate limit, network
        return [], f"Search failed ({type(exc).__name__}). Say so, and answer only from what you know."
    results = []
    for r in raw or []:
        url = r.get("href") or r.get("url") or ""
        if url:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": url,
                    "body": (r.get("body") or "")[:400],
                    "date": r.get("date", ""),
                    "source": r.get("source", ""),
                }
            )
    if not results:
        return [], "No results. Say so rather than guessing."
    return results, None


def format_results(results, error, numbers):
    """Text for the model. `numbers` maps url -> n for the whole turn, so [n] means the
    same page across several searches and the same page always keeps the same number."""
    if error:
        return error
    blocks = []
    for r in results:
        r["n"] = numbers.setdefault(r["url"], len(numbers) + 1)
        extra = " | ".join(x for x in (r["source"], r["date"]) if x)
        blocks.append(f"[{r['n']}] {r['title']}" + (f" ({extra})" if extra else "") + f"\n{r['url']}\n{r['body']}")
    return "\n\n".join(blocks)


# ---------- agent loop ----------

def build_system(now_ts, older, tz="UTC"):
    memory = ""
    if older:
        rows = "\n".join(f"- {fmt(r['created_at'], tz)} {r['role']}: {r['content'][:300]}" for r in older)
        memory = f"\nEarlier messages from this chat that may be relevant:\n{rows}\n"
    return f"""You are a fast, careful assistant with one tool: web_search.
Current date and time: {fmt(now_ts, tz)}.

Accuracy:
- Answer stable general knowledge directly, with no search.
- Search for anything recent, changing or specific. Prefer one focused search; search again only if the results are not enough. If you need several independent searches, request them together.
- For headlines, breaking news or recent events use kind="news". When asked for headlines, report the actual headlines (with outlet and date when given), not just a list of news sites.
- State facts that came from search only as the results support them. Cite with the result numbers, like [2] or [1, 3]. Numbers match the numbered results you were given.
- If results conflict or are thin, say so. Never invent sources, numbers or quotes.
- Do not write any text before calling the tool.

Style: concise and direct. Lead with the answer.

Memory: each user message begins with a [timestamp] that is metadata. Never repeat it. When asked what the user asked before, or when, answer from the conversation and the earlier messages below, and give the time exactly as written there (it is already in the user's timezone). If it is not there, say you do not have it.
{memory}"""


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def retryable(exc):
    """Overload (503), quota (429) or an unknown model name (404): safe to try another model."""
    text = f"{type(exc).__name__} {exc}"
    return any(k in text for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND"))


async def stream_model(system, contents, allow_tools):
    """Yields ("text", chunk) as tokens arrive, then ("parts", all_parts) at the end.

    If the main model is overloaded, retry it once, then fall back to the other models.
    Falling back is only safe before any text has been sent to the user, so a failure
    after tokens have started streaming is raised instead.
    """
    def make_config(model):
        extra = {}
        if model.startswith("gemini-2.5-flash"):
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=0)  # faster replies
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=[SEARCH_TOOL],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO" if allow_tools else "NONE")
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=4096,
            **extra,
        )

    last_exc = None
    for model in [MODEL, *FALLBACK_MODELS]:
        for _attempt in range(2 if model == MODEL else 1):
            started, parts = False, []
            try:
                stream = await get_client().aio.models.generate_content_stream(
                    model=model, contents=contents, config=make_config(model)
                )
                async for chunk in stream:
                    if not chunk.candidates:
                        continue
                    content = chunk.candidates[0].content
                    if not content or not content.parts:
                        continue
                    for part in content.parts:
                        parts.append(part)
                        if part.text and not part.thought:
                            started = True
                            yield ("text", part.text)
                yield ("parts", parts)
                return
            except Exception as exc:
                if started or not retryable(exc):
                    raise
                last_exc = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    break  # quota or bad model name: do not retry the same model
    raise last_exc


def friendly_error(exc):
    text = f"{type(exc).__name__}: {exc}"
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "Free-tier rate limit reached. Wait a minute and try again."
    if "503" in text or "UNAVAILABLE" in text:
        return "Gemini is overloaded right now (all models tried). Wait a moment and try again."
    if "API key" in text or "401" in text or "403" in text or "PERMISSION_DENIED" in text:
        return "The Gemini API key was rejected. Check GEMINI_API_KEY."
    return text


async def run_agent(session_id: str, user_text: str, tz: str = "UTC"):
    user_ts = now_iso()
    recent = load_recent(session_id, HISTORY_MESSAGES)
    older = recall(session_id, user_text, {r["id"] for r in recent})
    contents = [
        types.Content(
            role="user" if r["role"] == "user" else "model",
            parts=[
                types.Part(
                    text=f"[{fmt(r['created_at'], tz)}] {r['content']}" if r["role"] == "user" else r["content"]
                )
            ],
        )
        for r in recent
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=f"[{fmt(user_ts, tz)}] {user_text}")]))
    system = build_system(user_ts, older, tz)

    numbers, catalog, answer = {}, {}, ""
    try:
        for round_no in range(MAX_SEARCH_ROUNDS + 1):
            answer, parts = "", []
            # On the last round function calling is switched off so the model must answer.
            async for kind, value in stream_model(system, contents, round_no < MAX_SEARCH_ROUNDS):
                if kind == "text":
                    answer += value
                    yield sse({"type": "token", "text": value})
                else:
                    parts = value
            calls = [p.function_call for p in parts if p.function_call]
            if not calls:
                break
            yield sse({"type": "reset"})
            yield sse({"type": "status", "text": "Searching: " + "; ".join(str((c.args or {}).get("query", "")) for c in calls)})
            results = await asyncio.gather(*(do_search(c.args or {}) for c in calls))
            texts = []
            for found, error in results:
                texts.append(format_results(found, error, numbers))
                for r in found:
                    catalog.setdefault(r["n"], {"n": r["n"], "title": r["title"], "url": r["url"]})
            contents.append(types.Content(role="model", parts=parts))  # keeps any thought signatures intact
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(name=c.name, response={"result": t})
                        for c, t in zip(calls, texts)
                    ],
                )
            )

        if not answer.strip():
            yield sse({"type": "error", "text": "The model returned an empty reply. Please try again."})
            return

        # Show only the sources the answer actually cites (or the top few if it cites none).
        cited = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer) for n in re.findall(r"\d+", group)}
        items = [catalog[n] for n in sorted(catalog) if n in cited] or [catalog[n] for n in sorted(catalog)][:3]
        if items:
            yield sse({"type": "sources", "items": items})
        save_pair(session_id, user_text, user_ts, answer, now_iso())
        yield sse({"type": "done"})
    except Exception as exc:
        yield sse({"type": "error", "text": friendly_error(exc)})


# ---------- HTTP ----------

@asynccontextmanager
async def lifespan(_):
    init_db()
    yield


app = FastAPI(title="chat agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)
    tz: str = Field(default="UTC", max_length=64)  # the browser's timezone, e.g. Asia/Kolkata


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        run_agent(req.session_id, req.message.strip(), req.tz),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history")
def history(session_id: str):
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
