// LLM worker: runs for both Arm B and Arm C (two separate instances).
// Accepts a promptTemplate with {chunk} placeholder; embeds the LLM output.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

const CHUNK_SIZE = 750;
const CHUNK_OVERLAP = 150;

let embeddingModelId = "Xenova/all-MiniLM-L6-v2";
let llmModelId = "HuggingFaceTB/SmolLM2-360M-Instruct";
let llmMode = "local"; // "local" | "server"
let serverUrl = "";
let serverModel = "";
let serverKey = "";

let embedder = null;
let generator = null;

function post(msg) { self.postMessage(msg); }

function chunkText(transcript, segments, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  let fullText = transcript;
  const timeMap = [];
  if (segments?.length) {
    fullText = "";
    for (const seg of segments) {
      timeMap.push({ offset: fullText.length, start: seg.start });
      fullText += seg.text.trim() + " ";
    }
  }
  const chunks = [];
  const step = size - overlap;
  let pos = 0;
  while (pos < fullText.length) {
    const end = Math.min(pos + size, fullText.length);
    let startTime = null;
    if (timeMap.length) {
      for (let i = timeMap.length - 1; i >= 0; i--) {
        if (timeMap[i].offset <= pos) { startTime = timeMap[i].start; break; }
      }
    }
    chunks.push({ text: fullText.slice(pos, end).trim(), startTime });
    if (end >= fullText.length) break;
    pos += step;
  }
  return chunks;
}

async function embed(text) {
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

async function processChunkLocal(chunkText, promptTemplate) {
  const prompt = promptTemplate.replace("{chunk}", chunkText);
  const out = await generator(prompt, { max_new_tokens: 150, temperature: 0.3, do_sample: false, repetition_penalty: 1.3, return_full_text: false });
  const raw = (out?.[0]?.generated_text ?? "").trim();
  const clean = raw.replace(/^(?:(?:Summary|Questions? this summary answers|Questions?|Answer):\s*)+/i, "").trim();
  return clean || chunkText.slice(0, 120).trim();
}

async function processChunkServer(chunkText, promptTemplate) {
  const prompt = promptTemplate.replace("{chunk}", chunkText);
  const headers = { "Content-Type": "application/json" };
  if (serverKey) headers["Authorization"] = `Bearer ${serverKey}`;
  const res = await fetch(`${serverUrl}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: serverModel,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 80,
      temperature: 0.3,
      stream: false,
    }),
  });
  if (!res.ok) throw new Error(`Server error: ${res.status} ${res.statusText}`);
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content?.trim() ?? "";
  return text || chunkText.slice(0, 120).trim();
}

function snapHeap() {
  return self.performance?.memory?.usedJSHeapSize ?? null;
}

// transformers.js dispatches status:"download" for every file load — cache hit or real
// network fetch alike — so it can't be used to tell them apart. Instead, treat a file as a
// real download only once its progress events have spanned more than this long: cache reads
// (even for large files) resolve in a single burst well under this, real fetches don't.
const DOWNLOAD_MS_THRESHOLD = 500;

function makeProgressCallback() {
  const fileProgress = {};
  let firstProgressAt = null;
  let isNetworkDownload = false;
  return (event) => {
    if (event.status === "progress" && event.total) {
      if (firstProgressAt === null) firstProgressAt = Date.now();
      if (!isNetworkDownload && Date.now() - firstProgressAt > DOWNLOAD_MS_THRESHOLD) isNetworkDownload = true;
      fileProgress[event.file] = { loaded: event.loaded, total: event.total };
      const loaded = Object.values(fileProgress).reduce((s, f) => s + f.loaded, 0);
      const total = Object.values(fileProgress).reduce((s, f) => s + f.total, 0);
      post({ type: "download-progress-b", loaded, total, phase: isNetworkDownload ? "downloading" : "from-cache" });
    } else if (event.status === "done" && fileProgress[event.file]) {
      fileProgress[event.file].loaded = fileProgress[event.file].total;
      const total = Object.values(fileProgress).reduce((s, f) => s + f.total, 0);
      post({ type: "download-progress-b", loaded: total, total, phase: "done" });
    } else if (event.status === "loading") {
      post({ type: "download-progress-b", loaded: null, total: null, phase: "loading", loadProgress: event.progress ?? 0 });
    }
  };
}

// transformers.js swallows Cache Storage write failures (e.g. QuotaExceededError) down to a
// console.warn — the model then silently never persists and re-downloads every load. Surface
// storage pressure to the user before that happens instead of failing invisibly.
async function checkStorageQuota() {
  if (!self.navigator?.storage?.estimate) return null;
  try {
    const { usage, quota } = await self.navigator.storage.estimate();
    if (quota && usage / quota > 0.9) return { usage, quota };
  } catch { /* not available in this browser/context */ }
  return null;
}

async function loadPipeline(task, model, retries = 3) {
  const pressure = await checkStorageQuota();
  if (pressure) {
    const pct = Math.round((pressure.usage / pressure.quota) * 100);
    post({ type: "status-b", message: `Warning: browser storage is ${pct}% full — the model cache may fail to save and re-download every time. Delete old cached indexes to free space.` });
  }
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const result = await pipeline(task, model, { progress_callback: makeProgressCallback() });
      if (typeof result !== "function") throw new Error(`pipeline() returned non-callable`);
      post({ type: "download-ready-b" });
      return result;
    } catch (err) {
      if (attempt === retries) throw err;
      post({ type: "status-b", message: `Download interrupted — retrying (${attempt}/${retries - 1})…` });
      await new Promise((r) => setTimeout(r, 1500 * attempt));
    }
  }
}

async function buildArm(transcript, segments, startIndex, promptTemplate) {
  const chunks = chunkText(transcript, segments);
  post({ type: "progress-b", value: startIndex, max: chunks.length });

  if (!embedder) {
    post({ type: "status-b", message: "Loading embedding model…" });
    embedder = await loadPipeline("feature-extraction", embeddingModelId);
  }
  if (llmMode === "local" && !generator) {
    post({ type: "status-b", message: "Loading LLM model…" });
    generator = await loadPipeline("text-generation", llmModelId);
  }
  post({ type: "status-b", message: "" });

  const heapBefore = snapHeap();
  const tStart = performance.now();
  let tProcess = 0, tEmbed = 0;

  for (let i = startIndex; i < chunks.length; i++) {
    const t0 = performance.now();
    const output = llmMode === "server"
      ? await processChunkServer(chunks[i].text, promptTemplate)
      : await processChunkLocal(chunks[i].text, promptTemplate);
    tProcess += performance.now() - t0;

    const t1 = performance.now();
    const embedding = await embed(output);
    tEmbed += performance.now() - t1;

    post({ type: "b-entry", index: i, output, embedding });
    post({ type: "progress-b", value: i + 1, max: chunks.length });
  }

  const heapAfter = snapHeap();
  post({
    type: "b-complete",
    metrics: {
      total: performance.now() - tStart,
      process: tProcess,
      embed: tEmbed,
      heapDelta: heapBefore !== null ? heapAfter - heapBefore : null,
      chunks: chunks.length,
    },
  });
}

// ── Summary queue (used by buildArmFromSummaries to receive streamed summaries) ─
let _summaryQueue = [];
let _summariesDone = false;
let _summaryWaiter = null;

function enqueueSummary(output) {
  _summaryQueue.push(output);
  if (_summaryWaiter) { const r = _summaryWaiter; _summaryWaiter = null; r(); }
}

function markSummariesDone() {
  _summariesDone = true;
  if (_summaryWaiter) { const r = _summaryWaiter; _summaryWaiter = null; r(); }
}

async function dequeueSummary() {
  if (_summaryQueue.length > 0) return _summaryQueue.shift();
  if (_summariesDone) return null;
  await new Promise(r => { _summaryWaiter = r; });
  return _summaryQueue.length > 0 ? _summaryQueue.shift() : null;
}

async function buildArmFromSummaries(initialSummaries, totalSummaries, startIndex, promptTemplate) {
  for (const s of initialSummaries) _summaryQueue.push(s.output);

  post({ type: "progress-b", value: startIndex, max: totalSummaries });

  if (!embedder) {
    post({ type: "status-b", message: "Loading embedding model…" });
    embedder = await loadPipeline("feature-extraction", embeddingModelId);
  }
  if (llmMode === "local" && !generator) {
    post({ type: "status-b", message: "Loading LLM model…" });
    generator = await loadPipeline("text-generation", llmModelId);
  }
  post({ type: "status-b", message: "" });

  const heapBefore = snapHeap();
  const tStart = performance.now();
  let tProcess = 0, tEmbed = 0;
  let i = startIndex;

  while (true) {
    const summaryText = await dequeueSummary();
    if (summaryText === null) break;

    const t0 = performance.now();
    const output = llmMode === "server"
      ? await processChunkServer(summaryText, promptTemplate)
      : await processChunkLocal(summaryText, promptTemplate);
    tProcess += performance.now() - t0;

    const t1 = performance.now();
    const embedding = await embed(output);
    tEmbed += performance.now() - t1;

    post({ type: "b-entry", index: i, output, embedding });
    post({ type: "progress-b", value: i + 1, max: totalSummaries });
    i++;
  }

  const heapAfter = snapHeap();
  post({
    type: "b-complete",
    metrics: {
      total: performance.now() - tStart,
      process: tProcess,
      embed: tEmbed,
      heapDelta: heapBefore !== null ? heapAfter - heapBefore : null,
      chunks: i,
    },
  });
}

self.onmessage = async (e) => {
  if (e.data.type === "build-b") {
    // Reset summary queue for this run
    _summaryQueue = [];
    _summariesDone = false;
    _summaryWaiter = null;

    // Apply model config (reset pipelines if model changed)
    const newEmbeddingModel = e.data.embeddingModel ?? embeddingModelId;
    if (newEmbeddingModel !== embeddingModelId) {
      embeddingModelId = newEmbeddingModel;
      embedder = null;
    }

    const newLlmMode = e.data.llmMode ?? "local";
    const newLlmModel = e.data.llmModel ?? llmModelId;
    if (newLlmMode !== llmMode || newLlmModel !== llmModelId) {
      llmMode = newLlmMode;
      llmModelId = newLlmModel;
      generator = null;
    }

    serverUrl = e.data.serverUrl ?? serverUrl;
    serverModel = e.data.serverModel ?? serverModel;
    serverKey = e.data.serverKey ?? serverKey;

    try {
      if (Array.isArray(e.data.summaries)) {
        await buildArmFromSummaries(e.data.summaries, e.data.totalSummaries ?? e.data.summaries.length, e.data.startIndex ?? 0, e.data.promptTemplate);
      } else {
        await buildArm(e.data.transcript, e.data.segments, e.data.startIndex ?? 0, e.data.promptTemplate);
      }
    } catch (err) {
      post({ type: "error-b", message: err.message });
    }
  } else if (e.data.type === "add-summary") {
    enqueueSummary(e.data.output);
  } else if (e.data.type === "summaries-done") {
    markSummariesDone();
  }
};
