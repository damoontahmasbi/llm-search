// Search worker: embedding model, Arm A index, all search queries.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

const CHUNK_SIZE = 750;
const CHUNK_OVERLAP = 150;
let embeddingModelId = "Xenova/all-MiniLM-L6-v2";
const RRF_K = 60;

let embedder = null;
let indexA = [];
let indexB = []; // Arm B: LLM output embeddings (summaries by default)
let indexC = []; // Arm C: LLM output embeddings (questions by default)

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

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

async function embed(text) {
  if (!embedder) {
    post({ type: "status", message: "Loading embedding model…" });
    embedder = await loadPipeline("feature-extraction", embeddingModelId);
  }
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

function snapHeap() {
  return self.performance?.memory?.usedJSHeapSize ?? null;
}

function makeProgressCallback() {
  const fileProgress = {};
  let isNetworkDownload = false;
  return (event) => {
    if (event.status === "download") {
      isNetworkDownload = true;
    } else if (event.status === "progress" && event.total) {
      fileProgress[event.file] = { loaded: event.loaded, total: event.total };
      const loaded = Object.values(fileProgress).reduce((s, f) => s + f.loaded, 0);
      const total = Object.values(fileProgress).reduce((s, f) => s + f.total, 0);
      post({ type: "download-progress", loaded, total, phase: isNetworkDownload ? "downloading" : "from-cache" });
    } else if (event.status === "done" && fileProgress[event.file]) {
      fileProgress[event.file].loaded = fileProgress[event.file].total;
      const total = Object.values(fileProgress).reduce((s, f) => s + f.total, 0);
      post({ type: "download-progress", loaded: total, total, phase: "done" });
    } else if (event.status === "loading") {
      post({ type: "download-progress", loaded: null, total: null, phase: "loading", loadProgress: event.progress ?? 0 });
    }
  };
}

async function loadPipeline(task, model, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const result = await pipeline(task, model, { progress_callback: makeProgressCallback() });
      if (typeof result !== "function") throw new Error(`pipeline() returned a non-callable (${typeof result}) — try clearing browser cache`);
      post({ type: "download-ready" });
      return result;
    } catch (err) {
      if (attempt === retries) throw err;
      post({ type: "status", message: `Download interrupted — retrying (${attempt}/${retries - 1})…` });
      await new Promise((r) => setTimeout(r, 1500 * attempt));
    }
  }
}

async function buildArmA(transcript, segments, partialIndexA = [], startIndex = 0) {
  const chunks = chunkText(transcript, segments);
  indexA = [...partialIndexA];
  indexB = [];
  indexC = [];

  post({ type: "progress-a", value: startIndex, max: chunks.length });

  if (!embedder) {
    post({ type: "status", message: "Loading embedding model…" });
    embedder = await loadPipeline("feature-extraction", embeddingModelId);
  }

  const heapBefore = snapHeap();
  const tStart = performance.now();
  for (let i = startIndex; i < chunks.length; i++) {
    const embedding = await embed(chunks[i].text);
    indexA.push({ text: chunks[i].text, startTime: chunks[i].startTime, embedding });
    post({ type: "chunk-a", text: chunks[i].text, startTime: chunks[i].startTime, embedding });
    post({ type: "progress-a", value: i + 1, max: chunks.length });
  }
  const heapAfter = snapHeap();

  post({
    type: "arm-a-complete",
    metrics: { total: performance.now() - tStart, heapDelta: heapBefore !== null ? heapAfter - heapBefore : null, chunks: chunks.length },
  });
}

function rrfScore(rankA, rankB, k = RRF_K) {
  return 1 / (k + rankA) + 1 / (k + rankB);
}

function scoreSearch(queryEmbedding) {
  // Rank all chunks by direct cosine similarity (used by Arm A and as first term for B/C RRF)
  const chunkSims = indexA.map((e, i) => ({ i, sim: cosineSimilarity(queryEmbedding, e.embedding) }));
  chunkSims.sort((a, b) => b.sim - a.sim);
  const chunkRank = new Array(indexA.length);
  chunkSims.forEach((entry, r) => { chunkRank[entry.i] = r; });

  const scoredA = chunkSims.slice(0, 5).map(({ i, sim }) => ({
    text: indexA[i].text, startTime: indexA[i].startTime, score: sim,
  }));

  // Arm B: RRF(chunk_rank, llm_output_rank)
  let scoredB = null;
  if (indexB.length > 0) {
    const bSims = indexB.map((e, i) => ({ i, sim: cosineSimilarity(queryEmbedding, e.outputEmbedding) }));
    bSims.sort((a, b) => b.sim - a.sim);
    const bRank = new Array(indexB.length);
    bSims.forEach((entry, r) => { bRank[entry.i] = r; });
    scoredB = indexB.map((e, i) => ({
      text: e.text, startTime: e.startTime, annotation: e.output,
      score: rrfScore(chunkRank[i] ?? indexA.length, bRank[i] ?? indexB.length),
    })).sort((a, b) => b.score - a.score).slice(0, 5);
  }

  // Arm C: RRF(chunk_rank, llm_output_rank)
  let scoredC = null;
  if (indexC.length > 0) {
    const cSims = indexC.map((e, i) => ({ i, sim: cosineSimilarity(queryEmbedding, e.outputEmbedding) }));
    cSims.sort((a, b) => b.sim - a.sim);
    const cRank = new Array(indexC.length);
    cSims.forEach((entry, r) => { cRank[entry.i] = r; });
    scoredC = indexC.map((e, i) => ({
      text: e.text, startTime: e.startTime, annotation: e.output,
      score: rrfScore(chunkRank[i] ?? indexA.length, cRank[i] ?? indexC.length),
    })).sort((a, b) => b.score - a.score).slice(0, 5);
  }

  return { scoredA, scoredB, scoredC, bReady: indexB.length === indexA.length, cReady: indexC.length === indexA.length };
}

self.onmessage = async (e) => {
  const { type } = e.data;
  try {
    if (type === "build") {
      if (e.data.embeddingModel && e.data.embeddingModel !== embeddingModelId) {
        embeddingModelId = e.data.embeddingModel;
        embedder = null;
      }
      await buildArmA(e.data.transcript, e.data.segments, e.data.partialIndexA ?? [], e.data.startIndex ?? 0);
    } else if (type === "restore") {
      if (e.data.embeddingModel && e.data.embeddingModel !== embeddingModelId) {
        embeddingModelId = e.data.embeddingModel;
        embedder = null;
      }
      indexA = e.data.indexA ?? [];
      indexB = [];
      indexC = [];
      const summaries = Array.isArray(e.data.summaries) ? e.data.summaries : [];
      const questions = Array.isArray(e.data.questions) ? e.data.questions : [];
      indexB = summaries.map((s, i) => ({
        text: indexA[i]?.text ?? "", startTime: indexA[i]?.startTime ?? null,
        output: s.output, outputEmbedding: s.outputEmbedding,
      }));
      indexC = questions.map((q, i) => ({
        text: indexA[i]?.text ?? "", startTime: indexA[i]?.startTime ?? null,
        output: q.output, outputEmbedding: q.outputEmbedding,
      }));
      const bComplete = summaries.length === indexA.length && summaries.length > 0;
      const cComplete = questions.length === indexA.length && questions.length > 0;
      post({ type: "arm-a-complete", metrics: e.data.metrics?.a ?? null, fromCache: true });
      post({ type: "restore-complete", bComplete, cComplete });
      if (!embedder) {
        loadPipeline("feature-extraction", embeddingModelId).then((p) => { embedder = p; }).catch(() => {});
      }
    } else if (type === "get-index-a") {
      post({ type: "index-a-data", indexA: indexA.map((e) => ({ text: e.text, startTime: e.startTime, embedding: e.embedding })) });
    } else if (type === "get-index") {
      post({
        type: "index-data",
        indexA: indexA.map((e) => ({ text: e.text, startTime: e.startTime, embedding: e.embedding })),
        summaries: indexB.map((e) => ({ output: e.output, outputEmbedding: e.outputEmbedding })),
        questions: indexC.map((e) => ({ output: e.output, outputEmbedding: e.outputEmbedding })),
      });
    } else if (type === "search") {
      const queryEmbedding = await embed(e.data.query);
      post({ type: "search-results", ...scoreSearch(queryEmbedding), queryEmbedding });
    } else if (type === "rescore") {
      post({ type: "search-results", ...scoreSearch(e.data.queryEmbedding), queryEmbedding: e.data.queryEmbedding });
    } else if (type === "add-b") {
      const src = indexA[e.data.index];
      if (!src) return;
      indexB.push({ text: src.text, startTime: src.startTime, output: e.data.output, outputEmbedding: e.data.embedding });
    } else if (type === "add-c") {
      const src = indexA[e.data.index];
      if (!src) return;
      indexC.push({ text: src.text, startTime: src.startTime, output: e.data.output, outputEmbedding: e.data.embedding });
    }
  } catch (err) {
    post({ type: "error", message: err.message });
  }
};
