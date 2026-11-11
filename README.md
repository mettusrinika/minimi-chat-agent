# Chat agent with web search

A small chat agent with one tool (web search). It streams answers, cites sources, and remembers what you asked and when. See [design.md](design.md) for the design and trade-offs.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here      # Windows PowerShell: $env:GEMINI_API_KEY="..."
uvicorn main:app --reload
```

Open http://localhost:8000.

Get a free key at https://aistudio.google.com/apikey (no billing needed).

## Try it

- "What are the latest headlines about the Indian rupee?" (searches)
- "What is a hash map?" (answers without searching)
- "What did I ask you first, and when?" (uses memory)
- Restart the server and ask again: history is kept in `memory.db`.

## Test

```bash
pytest
```

## Config

`GEMINI_MODEL` (default `gemini-2.5-flash`), `GEMINI_FALLBACK_MODELS` (comma-separated, used when the main model is overloaded) and `DB_PATH` (default `memory.db`) can be set as environment variables.
