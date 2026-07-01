# llm-search-demo — Developer Reference

YouTube transcript **semantic vs LLM-indexed search** demo. Everything except the transcript fetch runs in-browser via transformers.js.

---

## Project layout

```
llm-search-demo/
├── server/
│   ├── main.py            # FastAPI server — single endpoint: POST /api/transcript
│   ├── requirements.txt   # fastapi, uvicorn[standard], youtube-transcript-api
│   └── .venv/             # gitignored; re-create below
├── web/
│   ├── index.html         # Full UI: model picker, prompt settings, cache manager, progress, results
│   ├── style.css          # All styles
│   ├── app.js             # Main app logic, worker wiring, IndexedDB cache
│   ├── worker.js          # Arm A: embedding worker (raw chunks → index → search)
│   └── llm-worker.js      # LLM worker (shared by Arm B and Arm C as two instances)
├── .gitignore
├── README.md
└── CLAUDE.md              # This file
```

---

## How to run

```bash
# Web — any static server works. Text file / paste tabs need nothing else.
python -m http.server 3000 --directory web

# Server — only needed for the YouTube URL tab
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload          # listens on http://localhost:8000
```

`SERVER_URL` in `web/app.js` line 1 must match the server address (default `http://localhost:8000`).

**YouTube URL tab is local-only:** YouTube blocks transcript fetches from cloud/hosted IPs, so this tab (now the third tab, `web/index.html`) only works when the transcript server runs on the same machine as the browser. It carries a `.tab-note` explaining this and linking to the repo. Default active tab is now `file` (see `activeTab` init in `app.js`), not `youtube`.

---

## Three-arm architecture

| Arm | What gets embedded | Fusion |
|-----|--------------------|--------|
| **A** | Raw text chunks (cosine sim only) | — |
| **B** | LLM-generated 2-3 sentence summary per chunk | RRF(rank\_A, rank\_B) |
| **C** | LLM-generated questions derived from Arm B summaries | RRF(rank\_A, rank\_C) |

**RRF formula:** `score = 1/(60 + rank_A) + 1/(60 + rank_llm)`

### B → C pipelined dependency

Arm C builds from Arm B's summaries (not raw chunks) and starts as soon as `C_START_THRESHOLD = 3` summaries are ready, running in parallel with the remainder of B. The `llmWorkerC` instance receives summaries as a stream via `add-summary` messages while B is still running.

**Cache miss path (`loadBtn`):**
- Starts A and B only
- C status: `"Arm C will start after 3 Arm B summaries are ready…"`
- No `llmWorkerC.postMessage` here at all

**During B's `b-entry` events:**
- `_partialSummaries` is pushed unconditionally (outside the cache-guard)
- When `_partialSummaries.length >= C_START_THRESHOLD` and `!_cStarted && !_cDone`: start C with `{ type: "build-b", summaries: _partialSummaries.slice(_pendingCStart), totalSummaries: progressB.max, startIndex: _pendingCStart, ... }`, set `_cStarted = true`
- While `_cStarted && !_cDone`: forward each new summary with `{ type: "add-summary", output }`

**`b-complete` handler:**
- If `!_cStarted` (transcript shorter than threshold): start C now with all summaries
- Always send `{ type: "summaries-done" }` so C's queue drains and exits

**Cache hit, B already complete:** start C immediately + send `summaries-done` in the same tick; set `_cStarted = true`

**Cache hit, B resuming:** threshold logic in `b-entry` fires naturally as B produces new entries

### Async queue in `llm-worker.js`

`buildArmFromSummaries` uses a pull-queue to receive summaries from `app.js`:

```js
_summaryQueue    // string[] — buffered summary texts
_summariesDone   // bool — set by summaries-done message
_summaryWaiter   // resolve fn — set when dequeueSummary is waiting

async function dequeueSummary() {
  if (_summaryQueue.length > 0) return _summaryQueue.shift();
  if (_summariesDone) return null;
  await new Promise(r => { _summaryWaiter = r; }); // yields event loop
  return _summaryQueue.length > 0 ? _summaryQueue.shift() : null;
}
```

Because `buildArmFromSummaries` is `async` and the worker's `onmessage` handler is not awaited by the runtime, each `await` inside the function yields the event loop — allowing `add-summary` and `summaries-done` messages to be processed between chunk generations.

`onmessage` resets `_summaryQueue / _summariesDone / _summaryWaiter` at the start of every `build-b` message.

---

## Chunking

Defined in both `worker.js` and `llm-worker.js` (identical logic):
- `CHUNK_SIZE = 750`, `CHUNK_OVERLAP = 150`
- Timestamps preserved via `timeMap` built from YouTube `segments` array
- Each chunk: `{ text, startTime }`

---

## Workers

### `worker.js` (searchWorker — Arm A)

| Message type | Payload | Effect |
|---|---|---|
| `build` | `transcript, segments, partialIndexA?, startIndex?, embeddingModel` | Embed chunks; emits `chunk-a`, `progress-a`, `arm-a-complete` |
| `restore` | `indexA, summaries, questions, metrics, embeddingModel` | Load full cache; emits `restore-complete` |
| `add-b` | `index, output, embedding` | Slot Arm B entry into in-memory index |
| `add-c` | `index, output, embedding` | Slot Arm C entry into in-memory index |
| `get-index-a` | — | Emits `index-a-data` with current Arm A data |
| `get-index` | — | Emits `index-data` with full A+B+C state for final cache write |
| `search` | `query` | Embed query, cosine + RRF; emits `search-results` |
| `rescore` | `queryEmbedding` | Re-run search with cached embedding (after B/C complete) |

### `llm-worker.js` (llmWorkerB and llmWorkerC — two instances)

**Messages in:**

| type | When sent | Effect |
|---|---|---|
| `build-b` | Once per run | Start building; `summaries` array → `buildArmFromSummaries`, else → `buildArm` |
| `add-summary` | After C starts, once per B entry | Enqueue next summary text for C to process |
| `summaries-done` | When B finishes | Signal queue end; C exits loop after draining |

**Messages out:** `status-b`, `download-progress-b`, `download-ready-b`, `progress-b`, `b-entry`, `b-complete`, `error-b`

`llmWorkerB` only ever receives `build-b`. `llmWorkerC` receives all three.

---

## IndexedDB cache

- DB: `llm-search-demo` (v1), object store: `indexes`, keyPath: `hash`
- **Cache key:** `SHA-256(transcript + "\0" + embeddingModel + "\0" + llmSource).slice(0, 16)` — one entry per (content, embedding model, LLM) combo; switching models routes to a different entry, the old one is preserved and hit again if you switch back
- `llmSource`: `"server:<url>:<model>"` for server mode, model ID string for local mode

Stored fields per entry:

```
hash, label, videoId, embeddingModel, llmSource,
indexA,       // [{text, startTime, embedding}]  — Arm A
summaries,    // [{output, outputEmbedding}]      — Arm B
questions,    // [{output, outputEmbedding}]      — Arm C
totalChunks,
promptB, promptC,
metrics: { a, b, c }
```

**Partial writes:** every chunk writes immediately via `chunk-a` / `b-entry` / `b-entry` (C) handlers — progress survives page refresh mid-build.

**Concurrent write safety:** A and B start at the same time (cache miss path). The `chunk-a` handler uses `_partialSummaries`/`_partialQuestions` (not `null`) so that a late A-chunk write never stomps on B or C progress already written to the cache. `_partialQuestions.push` in the C `b-entry` handler is unconditional (outside the cache guard), matching B's pattern — this ensures the tracking array stays accurate even if the save guard fires false.

**`_partialChunksA` in cache-hit path:** set synchronously from `cached.indexA` before any workers start (line ~819), so `_partialChunksA.length > 0` is always true by the time C fires its first `b-entry`.

**Prompt invalidation:** if `promptB`/`promptC` changed within a key, the affected arm's `startIndex` resets to 0 and its cached data is discarded. The key itself is unchanged.

**Cache manager UI:** `<details id="cache-manager">` opens lazily, renders all entries with label / models / chunk count / A-B-C status, and a trash icon per row.

---

## Model picker

### Embedding models

| Value | Size | Notes |
|---|---|---|
| `Xenova/all-MiniLM-L6-v2` | 23 MB | Default |
| `Xenova/all-MiniLM-L12-v2` | 33 MB | Slightly better |
| `Xenova/all-mpnet-base-v2` | 86 MB | Higher quality |
| `Xenova/bge-small-en-v1.5` | 33 MB | Strong retrieval |
| `Xenova/bge-base-en-v1.5` | 109 MB | Best retrieval |

### LLM modes

**In-browser:** transformers.js v3 (`@huggingface/transformers@3` via CDN), `pipeline("text-generation", model)`

| Model | Size |
|---|---|
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360 MB (default) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7 GB |
| `Qwen/Qwen2.5-0.5B-Instruct` | 494 MB |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 GB |

**Local server:** OpenAI-compatible `/v1/chat/completions`. Presets: Ollama (11434), LM Studio (1234), llama.cpp (8080). Test button hits `/v1/models`. Only generation goes to server — embedding always runs in-browser.

---

## LLM processing

**Local (`processChunkLocal`):**
```js
generator(prompt, {
  max_new_tokens: 150,
  temperature: 0.3,
  do_sample: false,
  repetition_penalty: 1.3,   // prevents token repetition loops in small models
  return_full_text: false,
})
// safe access: (out?.[0]?.generated_text ?? "").trim()
```

**Server (`processChunkServer`):**
```js
POST ${serverUrl}/v1/chat/completions
{ model, messages: [{role:"user", content: prompt}], max_tokens: 80, temperature: 0.3, stream: false }
```

**Output cleaning** strips all leading repeated label prefixes:
```js
raw.replace(/^(?:(?:Summary|Questions?|Answer):\s*)+/i, "").trim()
```
The `+` quantifier handles the `"Questions: Questions: Questions:…"` repetition-loop artifact from small models before `repetition_penalty` was added.

**Note:** `max_tokens: 80` on the server path may truncate multi-line question output — increase to 150+ if needed.

---

## Default prompts

**Arm B** (`#prompt-b` / `PROMPTS.summary`):
```
Summarize this transcript excerpt in 2-3 sentences:

{chunk}

Summary:
```

**Arm C** (`#prompt-c` / `PROMPTS.questions`):
```
{chunk}

Questions this summary answers:
1.
```

`{chunk}` is replaced with the **Arm B summary text**, not a raw transcript chunk.

Content-first layout is intentional: small models perform better when the content appears before the instruction. The `1.` anchor forces the model to start a numbered list immediately, preventing it from generating meta-text or instruction-like output (a common failure mode for sub-1B models).

---

## Key `app.js` state variables

```js
_pendingCacheKey      // hash of entry being built; null after final save
_pendingCacheMeta     // { transcript, segments, videoId, label, embeddingModel, llmSource }
_partialChunksA       // growing [{text, startTime, embedding}] for incremental cache writes
_partialSummaries     // growing [{output, outputEmbedding}] — also streamed to llmWorkerC
_partialQuestions     // growing [{output, outputEmbedding}]
_pendingMetricsB/C    // held from b-complete until final cache save
_bDone / _cDone       // true once arm finishes or was fully cached
_cStarted             // true once llmWorkerC has been started (prevents double-start)
_pendingCStart        // C's startIndex — set in loadBtn, used when C starts from b-entry or b-complete
_pendingLlmConfig     // llmConfig snapshot — set in loadBtn, consumed when C starts
_armAStart/BStart/CStart  // Date.now() timestamps for elapsed-time display in progress bars
C_START_THRESHOLD     // const = 3 — number of B summaries before C starts
```

---

## Open items

- **Larger chunks** — increase `CHUNK_SIZE` to 1200–1500 in both `worker.js` and `llm-worker.js`
- **Editable embedding instruction prefix** — expose prefix field (useful for E5/GTE models)
- **Timing breakdown per chunk** — show avg ms/chunk for LLM vs embed separately in metrics
- **Query latency display** — show per-arm search time in results panel
- **Server `max_tokens`** — still at 80; bump to 150+ to avoid truncating multi-line question output from Arm C
