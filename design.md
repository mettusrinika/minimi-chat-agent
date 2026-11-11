# Design

A chat agent with one tool, web search. The goals from the brief are: answer fast, get things right, and remember what was asked and when. This note explains how each goal shaped the design and what I chose not to build.

## Architecture

```
Browser (static page)
   |  POST /chat  (server-sent events)
   v
FastAPI  ->  run_agent()
   |            1. load recent messages + keyword-recall older ones (SQLite)
   |            2. stream the model's reply to the browser
   |            3. if it asks for web_search: run searches in parallel, feed results back (max 2 rounds)
   |            4. on success, save the user + assistant messages with timestamps
   v
SQLite (messages + FTS5 index)        DuckDuckGo search (via ddgs, no key)
```

Single process, no queue, no framework beyond FastAPI. The agent loop is about 50 lines so it can be read in one sitting.

## Fast

- **Streaming.** Tokens go to the browser as they are generated, so the first words appear quickly even when the full answer is long.
- **Search only when needed.** The system prompt tells the model to answer stable knowledge directly. Skipping a search saves a full network round trip.
- **Parallel searches.** If the model needs several independent lookups, it requests them in one turn and they run concurrently.
- **Bounded work.** At most 2 search rounds, then the tool is switched off (function-calling mode `NONE`) so the model must answer. Each search has an 8 second timeout. Only the top 5 results are used, each trimmed to 400 characters, which keeps the prompt small.
- **Thinking switched off** on Gemini 2.5 Flash models, which cuts latency on simple questions. Search results are the grounding, so extra reasoning time buys little here.
- **Short-lived cache.** Identical queries within 10 minutes reuse results.

## Reliable

- **Overload handling.** If the main model returns a 503 (high demand), the agent retries it once, then falls back to other models (`GEMINI_FALLBACK_MODELS`). A quota error (429) or an unknown model name moves straight to the next model. Fallback happens only before any text has streamed, so the user never sees a half answer followed by a different one.
- **Clear errors.** If every model fails, the page shows a plain message (overloaded, rate-limited or bad key), and the failed turn is not saved to memory.

## Correct

- **News vs web search.** The one tool takes a `kind` (web or news). Headline questions use news results, which carry outlet and date, and the prompt tells the model to report real headlines, not just list news sites.
- **Stable citations.** Results are numbered across the whole turn, and the same page keeps the same number, so [2] always means the same source. Only the sources the answer actually cites are shown under it.
- **Grounding.** The model is told to state searched facts only as the results support them and to cite with links. Sources are also shown under the answer.
- **Honest failure.** If a search fails, times out or returns nothing, the tool result tells the model to say so instead of guessing. Conflicting or thin results should be reported as such.
- **Current date in the prompt**, so "latest" and "this year" questions search with the right date.
- **Failed turns are not saved.** A message is stored only after a successful answer, so an error never leaves a half-finished turn in the memory.

## Remembers what you asked and when

- Every message is stored in SQLite with a UTC timestamp. Times shown to the model and the user are converted to the browser's timezone, so "when did I ask?" matches the clock on the user's screen.
- The last 20 messages go to the model verbatim. Each user message carries a `[timestamp]` prefix so the model can answer "when did I ask that?". The prompt marks the prefix as metadata so it is not repeated back.
- For older messages, an FTS5 keyword index finds the few most relevant ones for the current question and adds them to the system prompt with their times. This keeps the prompt small while still reaching far back.
- Memory is per session id. History survives restarts because it lives in the database.

## Decisions and trade-offs

- **Memory is not a second tool.** The brief allows one tool, so recall happens before the model is called, in code, not through a tool call.
- **Keyword recall, not embeddings.** Simpler, no extra service or cost, and exact words are usually enough for "what did I ask about X". The cost is that a paraphrased question can miss. With more time I would add embeddings.
- **Gemini free tier.** The model is Gemini (`gemini-2.5-flash` by default) through the free Google AI Studio tier, so the project runs with no billing. The cost is rate limits: heavy use returns a 429, which the app reports as a clear message instead of failing silently. Free-tier requests may also be used by Google to improve its products, so private data should not be sent through it.
- **DuckDuckGo via `ddgs`.** It needs no API key, which makes the project easy to run. Result quality and rate limits are weaker than a paid search API. The search function is isolated in `do_search`, so swapping providers is a small change.
- **Snippets only.** I do not fetch full pages. That keeps latency low but means long or subtle details can be missed.
- **Fixed history window (20 messages).** Simple and predictable. Very long single messages are not summarised.

## Not built

Auth, rate limiting, multi-user deployment, summarising old history, embeddings, and fetching full pages. Each is a reasonable next step but adds latency or complexity that the brief does not ask for.

## Tests

`pytest` covers: history always starting with a user turn, recall of a message outside the recent window, session isolation, search formatting and failure handling, the full agent loop (search round, then answer, then save) using a fake model, and that a failed turn is not saved. The live model and live search are not covered by tests; see the README for how to try them.
