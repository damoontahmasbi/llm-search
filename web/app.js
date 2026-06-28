const DEFAULT_TRANSCRIPT_URL = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? "http://localhost:8000"
  : "";

// ── Theme ─────────────────────────────────────────────────────────────────────
(function () {
  const stored = localStorage.getItem("theme");
  const dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.classList.toggle("dark", dark);
})();

const PROMPTS = {
  summary: `Summarize this transcript excerpt in 2-3 sentences:\n\n{chunk}\n\nSummary:`,
  questions: `{chunk}\n\nQuestions this summary answers:\n1.`,
};

const SERVER_PRESETS = {
  ollama: "http://localhost:11434",
  lmstudio: "http://localhost:1234",
  llamacpp: "http://localhost:8080",
  custom: "",
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const urlInput = document.getElementById("url-input");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const pasteInput = document.getElementById("paste-input");
const loadBtn = document.getElementById("load-btn");
const loadStatus = document.getElementById("load-status");
const downloadBtn = document.getElementById("download-btn");
const loadStatusB = document.getElementById("load-status-b");
const loadStatusC = document.getElementById("load-status-c");
const indexProgress = document.getElementById("index-progress");
const progressA = document.getElementById("progress-a");
const progressAText = document.getElementById("progress-a-text");
const progressB = document.getElementById("progress-b");
const progressBText = document.getElementById("progress-b-text");
const progressC = document.getElementById("progress-c");
const progressCText = document.getElementById("progress-c-text");
const downloadProgress = document.getElementById("download-progress");
const downloadBar = document.getElementById("download-bar");
const downloadText = document.getElementById("download-text");
const downloadProgressB = document.getElementById("download-progress-b");
const downloadBarB = document.getElementById("download-bar-b");
const downloadTextB = document.getElementById("download-text-b");
const downloadProgressC = document.getElementById("download-progress-c");
const downloadBarC = document.getElementById("download-bar-c");
const downloadTextC = document.getElementById("download-text-c");
const searchSection = document.getElementById("search-section");
const queryInput = document.getElementById("query-input");
const searchBtn = document.getElementById("search-btn");
const resultsSection = document.getElementById("results-section");
const resultsA = document.getElementById("results-a");
const resultsB = document.getElementById("results-b");
const resultsC = document.getElementById("results-c");
const promptBEl = document.getElementById("prompt-b");
const promptCEl = document.getElementById("prompt-c");
const presetBEl = document.getElementById("preset-b");
const presetCEl = document.getElementById("preset-c");
const transcriptUrlInput = document.getElementById("transcript-url");

// ── Workers ──────────────────────────────────────────────────────────────────
const searchWorker = new Worker("worker.js", { type: "module" });
const llmWorkerB = new Worker("llm-worker.js", { type: "module" });
const llmWorkerC = new Worker("llm-worker.js", { type: "module" });

searchWorker.onerror = (e) => { setStatus(`Worker error: ${e.message}`, true); _building = false; loadBtn.disabled = false; };
llmWorkerB.onerror = (e) => { setStatusB(`Arm B error: ${e.message}`, true); _bDone = true; tryEnableLoad(); };
llmWorkerC.onerror = (e) => { setStatusC(`Arm C error: ${e.message}`, true); _cDone = true; tryEnableLoad(); };

// ── State ─────────────────────────────────────────────────────────────────────
let lastQueryEmbedding = null;
let metricsAData = null;
let activeTab = "youtube";
let currentVideoId = null;
let _downloadData = null;
let _armAStart = null;
let _armBStart = null;
let _armCStart = null;
let _bDone = false;
let _cDone = false;

// ── IndexedDB cache ───────────────────────────────────────────────────────────
let _db = null;
function openDb() {
  if (_db) return Promise.resolve(_db);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("llm-search-demo", 1);
    req.onupgradeneeded = (e) => e.target.result.createObjectStore("indexes", { keyPath: "hash" });
    req.onsuccess = (e) => { _db = e.target.result; resolve(_db); };
    req.onerror = () => reject(req.error);
  });
}

async function hashText(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

async function getCached(hash) {
  try {
    const db = await openDb();
    return await new Promise((resolve) => {
      const req = db.transaction("indexes").objectStore("indexes").get(hash);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => resolve(null);
    });
  } catch { return null; }
}

async function saveCache(hash, data) {
  try {
    const db = await openDb();
    await new Promise((resolve) => {
      const req = db.transaction("indexes", "readwrite").objectStore("indexes").put({ hash, ...data });
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
    });
  } catch { /* fail silently */ }
}

async function getAllCached() {
  try {
    const db = await openDb();
    return await new Promise((resolve) => {
      const req = db.transaction("indexes").objectStore("indexes").getAll();
      req.onsuccess = () => resolve(req.result ?? []);
      req.onerror = () => resolve([]);
    });
  } catch { return []; }
}

async function deleteCache(hash) {
  try {
    const db = await openDb();
    await new Promise((resolve) => {
      const req = db.transaction("indexes", "readwrite").objectStore("indexes").delete(hash);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
    });
  } catch { /* fail silently */ }
}
// ─────────────────────────────────────────────────────────────────────────────

const C_START_THRESHOLD = 3; // start Arm C after B has this many summaries ready

let _pendingCacheKey = null;
let _pendingCacheMeta = null;
let _pendingMetricsB = null;
let _pendingMetricsC = null;
let _partialChunksA = [];
let _partialSummaries = [];
let _partialQuestions = [];
let _pendingCStart = 0;
let _pendingLlmConfig = null;
let _cStarted = false;
let _activePromptB = "";
let _activePromptC = "";
let _building = false;

// ── Model settings ────────────────────────────────────────────────────────────
function getModelSettings() {
  const embeddingModel = document.querySelector('input[name="embed-model"]:checked')?.value
    ?? "Xenova/all-MiniLM-L6-v2";
  const llmMode = document.querySelector(".mode-btn.active")?.dataset.mode ?? "local";
  const llmModel = llmMode === "local"
    ? (document.querySelector('input[name="llm-model"]:checked')?.value ?? "HuggingFaceTB/SmolLM2-360M-Instruct")
    : "";
  const serverUrl = document.getElementById("server-url")?.value.trim() ?? "";
  const serverModel = document.getElementById("server-model-name")?.value.trim() ?? "";
  const serverKey = document.getElementById("server-key")?.value.trim() ?? "";
  return { embeddingModel, llmMode, llmModel, serverUrl, serverModel, serverKey };
}

function llmSourceKey(settings) {
  return settings.llmMode === "server"
    ? `server:${settings.serverUrl}:${settings.serverModel}`
    : settings.llmModel;
}

// Mode toggle (in-browser / local server)
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const isServer = btn.dataset.mode === "server";
    document.getElementById("llm-local-panel").classList.toggle("hidden", isServer);
    document.getElementById("llm-server-panel").classList.toggle("hidden", !isServer);
  });
});

// Server preset auto-fill
document.getElementById("server-preset").addEventListener("change", (e) => {
  const url = SERVER_PRESETS[e.target.value];
  if (url !== undefined) document.getElementById("server-url").value = url;
});
document.getElementById("server-url").addEventListener("input", () => {
  document.getElementById("server-preset").value = "custom";
});

// Test connection
document.getElementById("server-test-btn").addEventListener("click", async () => {
  const resultEl = document.getElementById("server-test-result");
  const url = document.getElementById("server-url").value.trim();
  const key = document.getElementById("server-key").value.trim();
  if (!url) {
    resultEl.className = "server-test-result err";
    resultEl.textContent = "Enter a server URL first.";
    return;
  }
  resultEl.className = "server-test-result";
  resultEl.textContent = "Testing…";
  try {
    const headers = { "Content-Type": "application/json", ...(key ? { Authorization: `Bearer ${key}` } : {}) };
    const res = await fetch(`${url}/v1/models`, { headers });
    if (res.ok) {
      const data = await res.json();
      const ids = (data.data ?? []).map((m) => m.id).join(", ");
      resultEl.textContent = ids ? `Connected — models: ${ids}` : "Connected.";
      resultEl.className = "server-test-result ok";
    } else {
      resultEl.textContent = `Server returned ${res.status}.`;
      resultEl.className = "server-test-result err";
    }
  } catch (err) {
    resultEl.textContent = `Failed: ${err.message}`;
    resultEl.className = "server-test-result err";
  }
});

// ── Cache manager ─────────────────────────────────────────────────────────────
const TRASH_SVG = `<svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor" aria-hidden="true"><path d="M6.5 1h3a.5.5 0 0 1 .5.5v1H6v-1a.5.5 0 0 1 .5-.5M11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3A1.5 1.5 0 0 0 5 1.5v1H1.5a.5.5 0 0 0 0 1h.538l.853 10.66A2 2 0 0 0 4.885 16h6.23a2 2 0 0 0 1.994-1.84l.853-10.66h.538a.5.5 0 0 0 0-1zm1.958 1-.846 10.58a1 1 0 0 1-.997.92h-6.23a1 1 0 0 1-.997-.92L3.042 3.5zm-7.487 1a.5.5 0 0 1 .528.47l.5 8.5a.5.5 0 0 1-.998.06L5 5.03a.5.5 0 0 1 .47-.53Zm5.058 0a.5.5 0 0 1 .47.53l-.5 8.5a.5.5 0 1 1-.998-.06l.5-8.5a.5.5 0 0 1 .528-.47M8 4.5a.5.5 0 0 1 .5.5v8.5a.5.5 0 0 1-1 0V5a.5.5 0 0 1 .5-.5"/></svg>`;

function formatLlmSource(src) {
  if (!src) return "unknown";
  if (src.startsWith("server:")) {
    const parts = src.split(":");
    return `server · ${parts[parts.length - 1] || "?"}`;
  }
  return src.split("/").pop();
}

async function renderCacheManager() {
  const list = document.getElementById("cache-list");
  list.textContent = "Loading…";
  const entries = await getAllCached();
  list.innerHTML = "";

  if (entries.length === 0) {
    const p = document.createElement("p");
    p.className = "cache-empty";
    p.textContent = "No cached indexes.";
    list.appendChild(p);
    return;
  }

  for (const entry of entries) {
    const row = document.createElement("div");
    row.className = "cache-entry";

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "cache-delete-btn";
    deleteBtn.disabled = _building;
    deleteBtn.title = _building ? "Cannot delete while indexing is in progress" : "Delete this cached index";
    deleteBtn.innerHTML = TRASH_SVG;
    deleteBtn.addEventListener("click", async () => {
      if (_building) return;
      await deleteCache(entry.hash);
      row.remove();
      if (!list.querySelector(".cache-entry")) {
        const p = document.createElement("p");
        p.className = "cache-empty";
        p.textContent = "No cached indexes.";
        list.appendChild(p);
      }
    });

    const label = document.createElement("span");
    label.className = "cache-entry-label";
    label.title = entry.label ?? "";
    label.textContent = entry.label
      ?? (entry.videoId ? `youtube.com/watch?v=${entry.videoId}` : "Transcript");

    const models = document.createElement("span");
    models.className = "cache-entry-models";
    const embedShort = (entry.embeddingModel ?? "—").split("/").pop();
    const llmShort = formatLlmSource(entry.llmSource ?? "");
    models.textContent = `${embedShort} · ${llmShort}`;

    const n = entry.totalChunks ?? entry.indexA?.length ?? 0;
    const chunks = document.createElement("span");
    chunks.className = "cache-entry-chunks";
    chunks.textContent = `${n} chunks`;

    const status = document.createElement("span");
    status.className = "cache-entry-status";
    const bDone = (entry.summaries?.length ?? 0) === n && n > 0;
    const cDone = (entry.questions?.length ?? 0) === n && n > 0;
    status.textContent = `A:${n > 0 ? "✓" : "—"} B:${bDone ? "✓" : "—"} C:${cDone ? "✓" : "—"}`;

    row.append(deleteBtn, label, models, chunks, status);
    list.appendChild(row);
  }
}

document.getElementById("cache-manager").addEventListener("toggle", (e) => {
  if (e.target.open) renderCacheManager();
});

// ── Preset selects ────────────────────────────────────────────────────────────
presetBEl.addEventListener("change", () => {
  if (presetBEl.value !== "custom") promptBEl.value = PROMPTS[presetBEl.value];
});
promptBEl.addEventListener("input", () => { presetBEl.value = "custom"; });

presetCEl.addEventListener("change", () => {
  if (presetCEl.value !== "custom") promptCEl.value = PROMPTS[presetCEl.value];
});
promptCEl.addEventListener("input", () => { presetCEl.value = "custom"; });

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    document.getElementById(`tab-${activeTab}`).classList.remove("hidden");
  });
});

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name ?? "No file chosen";
});

document.getElementById("theme-btn").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  localStorage.setItem("theme", dark ? "dark" : "light");
});

transcriptUrlInput.value = localStorage.getItem("transcriptUrl") ?? DEFAULT_TRANSCRIPT_URL;
transcriptUrlInput.addEventListener("input", () => {
  localStorage.setItem("transcriptUrl", transcriptUrlInput.value.trim() || DEFAULT_TRANSCRIPT_URL);
});

function offerDownload(text, name) {
  _downloadData = { text, name };
  downloadBtn.disabled = false;
}

downloadBtn.addEventListener("click", (e) => {
  e.preventDefault();
  if (!_downloadData?.text) return;
  const blob = new Blob([_downloadData.text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${_downloadData.name ?? "transcript"}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

function setStatus(message, isError = false) {
  loadStatus.textContent = message;
  loadStatus.classList.toggle("error", isError);
}
function setStatusB(message, isError = false) {
  loadStatusB.textContent = message;
  loadStatusB.classList.toggle("error", isError);
}
function setStatusC(message, isError = false) {
  loadStatusC.textContent = message;
  loadStatusC.classList.toggle("error", isError);
}

function tryEnableLoad() {
  if (_bDone && _cDone) {
    loadBtn.disabled = false;
    _building = false;
    const cm = document.getElementById("cache-manager");
    if (cm.open) renderCacheManager();
  }
}

function fmtTime(seconds) {
  const s = Math.floor(seconds);
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function handleDownloadProgress({ loaded, total, phase, loadProgress }, bar, text, container) {
  if (phase === "downloading" && total) {
    bar.value = Math.round((loaded / total) * 100);
    text.textContent = `Downloading ${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`;
    container.classList.remove("hidden");
  } else if (phase === "from-cache" && total) {
    bar.value = Math.round((loaded / total) * 100);
    text.textContent = `Reading from cache ${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`;
    container.classList.remove("hidden");
  } else if (phase === "done") {
    bar.value = 100;
    text.textContent = `${(total / 1048576).toFixed(1)} MB — loading into memory…`;
  } else if (phase === "loading") {
    text.textContent = `Loading into memory… ${Math.round(loadProgress ?? 0)}%`;
  }
}

function renderResults(container, results, placeholder, annotationLabel) {
  container.innerHTML = "";
  if (!results) {
    const msg = document.createElement("p");
    msg.className = "arm-pending";
    msg.textContent = placeholder ?? "Still indexing…";
    container.appendChild(msg);
    return;
  }
  for (const result of results) {
    const card = document.createElement("div");
    card.className = "result-card";

    const meta = document.createElement("div");
    meta.className = "result-meta";

    const score = document.createElement("span");
    score.className = "result-score";
    score.textContent = `score: ${result.score.toFixed(3)}`;
    meta.appendChild(score);

    if (result.startTime != null) {
      const ts = document.createElement("a");
      ts.className = "result-timestamp";
      const t = fmtTime(result.startTime);
      if (currentVideoId) {
        ts.href = `https://www.youtube.com/watch?v=${currentVideoId}&t=${Math.floor(result.startTime)}s`;
        ts.target = "_blank";
        ts.rel = "noopener noreferrer";
      }
      ts.textContent = t;
      meta.appendChild(ts);
    }
    card.appendChild(meta);

    if (result.annotation && annotationLabel) {
      const ann = document.createElement("p");
      ann.className = "result-summary";
      ann.textContent = `${annotationLabel}: "${result.annotation}"`;
      card.appendChild(ann);
    }

    const text = document.createElement("p");
    text.className = "result-text";
    text.textContent = result.text;
    card.appendChild(text);

    container.appendChild(card);
  }
}

// ── Search worker messages ────────────────────────────────────────────────────
searchWorker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      setStatus(msg.message);
      break;
    case "download-progress":
      handleDownloadProgress(msg, downloadBar, downloadText, downloadProgress);
      break;
    case "download-ready":
      downloadProgress.classList.add("hidden");
      downloadText.textContent = "";
      break;
    case "progress-a":
      if (msg.value === 1) _armAStart = Date.now();
      progressA.value = msg.value;
      progressA.max = msg.max;
      progressAText.textContent = _armAStart
        ? `${msg.value}/${msg.max} · ${((Date.now() - _armAStart) / 1000).toFixed(1)}s`
        : `${msg.value}/${msg.max}`;
      break;
    case "arm-a-complete":
      metricsAData = msg.metrics;
      if (msg.fromCache) {
        progressA.value = progressA.max = 1;
        progressAText.textContent = "from cache";
      } else if (_armAStart) {
        const t = ((Date.now() - _armAStart) / 1000).toFixed(1);
        progressAText.textContent = `${progressA.max}/${progressA.max} · ${t}s`;
        setStatus("Arm A ready — Arms B & C indexing in background.");
        if (_pendingCacheKey) searchWorker.postMessage({ type: "get-index-a" });
      }
      searchSection.classList.remove("hidden");
      break;
    case "restore-complete":
      setStatus("Loaded from cache.");
      indexProgress.classList.remove("hidden");
      progressA.value = progressA.max = 1;
      progressAText.textContent = "from cache";
      if (msg.bComplete) {
        progressB.value = progressB.max = 1;
        progressBText.textContent = "from cache";
        setStatusB("Arm B loaded from cache.");
      }
      if (msg.cComplete) {
        progressC.value = progressC.max = 1;
        progressCText.textContent = "from cache";
        setStatusC("Arm C loaded from cache.");
      }
      tryEnableLoad();
      break;
    case "chunk-a":
      if (_pendingCacheKey) {
        _partialChunksA.push({ text: msg.text, startTime: msg.startTime, embedding: msg.embedding });
        saveCache(_pendingCacheKey, {
          ..._pendingCacheMeta,
          indexA: _partialChunksA,
          totalChunks: progressA.max,
          summaries: _partialSummaries,
          questions: _partialQuestions,
          promptB: _activePromptB,
          promptC: _activePromptC,
          metrics: { a: null, b: null, c: null },
        });
      }
      break;
    case "index-a-data":
      if (_pendingCacheKey) {
        _partialChunksA = msg.indexA;
        saveCache(_pendingCacheKey, {
          ..._pendingCacheMeta,
          indexA: msg.indexA,
          totalChunks: msg.indexA.length,
          summaries: _partialSummaries,
          questions: _partialQuestions,
          promptB: _activePromptB,
          promptC: _activePromptC,
          metrics: { a: metricsAData, b: null, c: null },
        });
      }
      break;
    case "index-data":
      if (_pendingCacheKey) {
        saveCache(_pendingCacheKey, {
          ..._pendingCacheMeta,
          indexA: msg.indexA,
          totalChunks: msg.indexA.length,
          summaries: msg.summaries,
          questions: msg.questions,
          promptB: _activePromptB,
          promptC: _activePromptC,
          metrics: { a: metricsAData, b: _pendingMetricsB, c: _pendingMetricsC },
        });
        _pendingCacheKey = null;
        _pendingCacheMeta = null;
        _pendingMetricsB = null;
        _pendingMetricsC = null;
      }
      break;
    case "search-results":
      lastQueryEmbedding = msg.queryEmbedding;
      renderResults(resultsA, msg.scoredA, null, null);
      renderResults(resultsB, msg.scoredB, "Arm B still indexing — search again once ready.", "Summary");
      renderResults(resultsC, msg.scoredC, "Arm C still indexing — search again once ready.", "Questions");
      resultsSection.classList.remove("hidden");
      searchBtn.disabled = false;
      break;
    case "error":
      downloadProgress.classList.add("hidden");
      setStatus(msg.message, true);
      _building = false;
      loadBtn.disabled = false;
      searchBtn.disabled = false;
      break;
  }
};

// ── LLM worker B messages ─────────────────────────────────────────────────────
llmWorkerB.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "status-b":
      setStatusB(msg.message);
      break;
    case "download-progress-b":
      handleDownloadProgress(msg, downloadBarB, downloadTextB, downloadProgressB);
      break;
    case "download-ready-b":
      downloadProgressB.classList.add("hidden");
      downloadTextB.textContent = "";
      setStatusB("");
      break;
    case "progress-b":
      if (!_armBStart && msg.value > 0) _armBStart = Date.now();
      progressB.value = msg.value;
      progressB.max = msg.max;
      progressBText.textContent = _armBStart
        ? `${msg.value}/${msg.max} · ${((Date.now() - _armBStart) / 1000).toFixed(1)}s`
        : `${msg.value}/${msg.max}`;
      break;
    case "b-entry":
      searchWorker.postMessage({ type: "add-b", index: msg.index, output: msg.output, embedding: msg.embedding });
      _partialSummaries.push({ output: msg.output, outputEmbedding: msg.embedding });
      if (_pendingCacheKey && _partialChunksA.length > 0) {
        saveCache(_pendingCacheKey, {
          ..._pendingCacheMeta,
          indexA: _partialChunksA,
          totalChunks: _partialChunksA.length,
          summaries: _partialSummaries,
          questions: _partialQuestions,
          promptB: _activePromptB,
          promptC: _activePromptC,
          metrics: { a: metricsAData, b: null, c: null },
        });
      }
      if (!_cStarted && !_cDone && _pendingLlmConfig && _partialSummaries.length >= C_START_THRESHOLD) {
        _cStarted = true;
        setStatusC("Building Arm C from summaries…");
        indexProgress.classList.remove("hidden");
        llmWorkerC.postMessage({ type: "build-b", summaries: _partialSummaries.slice(_pendingCStart), totalSummaries: progressB.max, startIndex: _pendingCStart, promptTemplate: _activePromptC, ..._pendingLlmConfig });
      } else if (_cStarted && !_cDone) {
        llmWorkerC.postMessage({ type: "add-summary", output: msg.output });
      }
      break;
    case "b-complete":
      _bDone = true;
      _pendingMetricsB = msg.metrics;
      if (_armBStart) {
        progressBText.textContent = `${progressB.max}/${progressB.max} · ${((Date.now() - _armBStart) / 1000).toFixed(1)}s`;
      }
      setStatusB("Arm B ready.");
      if (!_cDone && _pendingLlmConfig) {
        if (!_cStarted) {
          // B finished before the threshold (short transcript) — start C now
          _cStarted = true;
          setStatusC("Building Arm C from summaries…");
          indexProgress.classList.remove("hidden");
          llmWorkerC.postMessage({ type: "build-b", summaries: _partialSummaries.slice(_pendingCStart), totalSummaries: _partialSummaries.length, startIndex: _pendingCStart, promptTemplate: _activePromptC, ..._pendingLlmConfig });
        }
        llmWorkerC.postMessage({ type: "summaries-done" });
      }
      if (_bDone && _cDone && _pendingCacheKey) searchWorker.postMessage({ type: "get-index" });
      tryEnableLoad();
      if (lastQueryEmbedding) searchWorker.postMessage({ type: "rescore", queryEmbedding: lastQueryEmbedding });
      break;
    case "error-b":
      downloadProgressB.classList.add("hidden");
      setStatusB(msg.message, true);
      _bDone = true;
      tryEnableLoad();
      break;
  }
};

// ── LLM worker C messages ─────────────────────────────────────────────────────
llmWorkerC.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "status-b":
      setStatusC(msg.message);
      break;
    case "download-progress-b":
      handleDownloadProgress(msg, downloadBarC, downloadTextC, downloadProgressC);
      break;
    case "download-ready-b":
      downloadProgressC.classList.add("hidden");
      downloadTextC.textContent = "";
      setStatusC("");
      break;
    case "progress-b":
      if (!_armCStart && msg.value > 0) _armCStart = Date.now();
      progressC.value = msg.value;
      progressC.max = msg.max;
      progressCText.textContent = _armCStart
        ? `${msg.value}/${msg.max} · ${((Date.now() - _armCStart) / 1000).toFixed(1)}s`
        : `${msg.value}/${msg.max}`;
      break;
    case "b-entry":
      searchWorker.postMessage({ type: "add-c", index: msg.index, output: msg.output, embedding: msg.embedding });
      _partialQuestions.push({ output: msg.output, outputEmbedding: msg.embedding });
      if (_pendingCacheKey && _partialChunksA.length > 0) {
        saveCache(_pendingCacheKey, {
          ..._pendingCacheMeta,
          indexA: _partialChunksA,
          totalChunks: _partialChunksA.length,
          summaries: _partialSummaries,
          questions: _partialQuestions,
          promptB: _activePromptB,
          promptC: _activePromptC,
          metrics: { a: metricsAData, b: null, c: null },
        });
      }
      break;
    case "b-complete":
      _cDone = true;
      _pendingMetricsC = msg.metrics;
      if (_armCStart) {
        progressCText.textContent = `${progressC.max}/${progressC.max} · ${((Date.now() - _armCStart) / 1000).toFixed(1)}s`;
      }
      setStatusC("Arm C ready.");
      if (_bDone && _cDone && _pendingCacheKey) searchWorker.postMessage({ type: "get-index" });
      tryEnableLoad();
      if (lastQueryEmbedding) searchWorker.postMessage({ type: "rescore", queryEmbedding: lastQueryEmbedding });
      break;
    case "error-b":
      downloadProgressC.classList.add("hidden");
      setStatusC(msg.message, true);
      _cDone = true;
      tryEnableLoad();
      break;
  }
};

// ── Transcript fetching ───────────────────────────────────────────────────────
async function fetchTranscript(url) {
  const serverUrl = transcriptUrlInput.value.trim() || DEFAULT_TRANSCRIPT_URL;
  const res = await fetch(`${serverUrl}/api/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  return res.json();
}

async function getTranscript() {
  if (activeTab === "youtube") {
    const url = urlInput.value.trim();
    if (!url) return null;
    const data = await fetchTranscript(url);
    if (data.error) { setStatus(data.error, true); return null; }
    offerDownload(data.transcript, `transcript-${data.videoId ?? "video"}`);
    return { text: data.transcript, segments: data.segments ?? null, videoId: data.videoId ?? null, label: `${data.transcript.length} chars` };
  }
  if (activeTab === "file") {
    const file = fileInput.files[0];
    if (!file) { setStatus("No file selected.", true); return null; }
    const text = await file.text();
    if (!text.trim()) { setStatus("File is empty.", true); return null; }
    return { text, segments: null, videoId: null, label: file.name };
  }
  if (activeTab === "paste") {
    const text = pasteInput.value.trim();
    if (!text) { setStatus("Nothing pasted.", true); return null; }
    offerDownload(text, "pasted-text");
    return { text, segments: null, videoId: null, label: `${text.length} chars` };
  }
  return null;
}

// ── Load & Index ──────────────────────────────────────────────────────────────
loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  _building = true;
  resultsSection.classList.add("hidden");
  searchSection.classList.add("hidden");
  indexProgress.classList.add("hidden");
  downloadProgress.classList.add("hidden");
  downloadProgressB.classList.add("hidden");
  downloadProgressC.classList.add("hidden");
  downloadBtn.disabled = true;
  lastQueryEmbedding = null;
  metricsAData = null;
  _pendingCacheKey = null;
  _pendingCacheMeta = null;
  _pendingMetricsB = null;
  _pendingMetricsC = null;
  _partialChunksA = [];
  _partialSummaries = [];
  _partialQuestions = [];
  _armAStart = null;
  _armBStart = null;
  _armCStart = null;
  _bDone = false;
  _cDone = false;
  _pendingCStart = 0;
  _pendingLlmConfig = null;
  _cStarted = false;
  _activePromptB = promptBEl.value;
  _activePromptC = promptCEl.value;
  setStatus("Loading…");
  setStatusB("");
  setStatusC("");

  const modelSettings = getModelSettings();
  const llmSrc = llmSourceKey(modelSettings);

  // Helper to build the worker config object for llm workers
  const llmConfig = {
    embeddingModel: modelSettings.embeddingModel,
    llmMode: modelSettings.llmMode,
    llmModel: modelSettings.llmModel,
    serverUrl: modelSettings.serverUrl,
    serverModel: modelSettings.serverModel,
    serverKey: modelSettings.serverKey,
  };
  _pendingLlmConfig = llmConfig;

  try {
    const result = await getTranscript();
    if (!result) { _building = false; loadBtn.disabled = false; return; }

    currentVideoId = result.videoId;
    const entryLabel = result.videoId ? `youtube.com/watch?v=${result.videoId}` : result.label;
    const hash = await hashText(result.text + "\0" + modelSettings.embeddingModel + "\0" + llmSrc);
    const cached = await getCached(hash);

    if (cached) {
      setStatus("Checking cache…");
      const cachedSummaries = Array.isArray(cached.summaries) ? cached.summaries : [];
      const cachedQuestions = Array.isArray(cached.questions) ? cached.questions : [];
      const totalChunks = cached.totalChunks ?? cached.indexA?.length ?? 0;

      const currentPromptB = promptBEl.value;
      const currentPromptC = promptCEl.value;
      const promptBMatch = !cached.promptB || cached.promptB === currentPromptB;
      const promptCMatch = !cached.promptC || cached.promptC === currentPromptC;

      const armAComplete = (cached.indexA?.length ?? 0) === totalChunks && totalChunks > 0;
      const armBComplete = promptBMatch && cachedSummaries.length === totalChunks && totalChunks > 0;
      const armCComplete = promptCMatch && cachedQuestions.length === totalChunks && totalChunks > 0;

      const bStart = promptBMatch ? cachedSummaries.length : 0;
      const cStart = promptCMatch ? cachedQuestions.length : 0;

      _partialChunksA = cached.indexA ? [...cached.indexA] : [];
      _partialSummaries = promptBMatch ? [...cachedSummaries] : [];
      _partialQuestions = promptCMatch ? [...cachedQuestions] : [];
      _bDone = armBComplete;
      _cDone = armCComplete;

      const makeMeta = () => ({
        transcript: result.text, segments: result.segments, videoId: result.videoId,
        label: entryLabel, embeddingModel: modelSettings.embeddingModel, llmSource: llmSrc,
      });

      if (armAComplete) {
        searchWorker.postMessage({
          type: "restore",
          indexA: cached.indexA,
          summaries: promptBMatch ? cachedSummaries : [],
          questions: promptCMatch ? cachedQuestions : [],
          metrics: cached.metrics ?? null,
          embeddingModel: modelSettings.embeddingModel,
        });
      } else {
        const aStart = cached.indexA?.length ?? 0;
        setStatus(`Resuming Arm A from chunk ${aStart + 1} / ${totalChunks}…`);
        indexProgress.classList.remove("hidden");
        _pendingCacheKey = hash;
        _pendingCacheMeta = makeMeta();
        searchWorker.postMessage({
          type: "build",
          transcript: result.text,
          segments: result.segments,
          partialIndexA: cached.indexA ?? [],
          startIndex: aStart,
          embeddingModel: modelSettings.embeddingModel,
        });
      }

      if (!armBComplete) {
        if (!_pendingCacheKey) { _pendingCacheKey = hash; _pendingCacheMeta = makeMeta(); }
        setStatusB(bStart > 0 ? `Resuming Arm B from chunk ${bStart + 1} / ${totalChunks}…` : "Building Arm B…");
        if (armAComplete) indexProgress.classList.remove("hidden");
        llmWorkerB.postMessage({ type: "build-b", transcript: result.text, segments: result.segments, startIndex: bStart, promptTemplate: currentPromptB, ...llmConfig });
      }

      if (!armCComplete) {
        if (!_pendingCacheKey) { _pendingCacheKey = hash; _pendingCacheMeta = makeMeta(); }
        _pendingCStart = cStart;
        if (armBComplete) {
          _cStarted = true;
          setStatusC(cStart > 0 ? `Resuming Arm C from chunk ${cStart + 1} / ${totalChunks}…` : "Building Arm C from summaries…");
          if (armAComplete) indexProgress.classList.remove("hidden");
          llmWorkerC.postMessage({ type: "build-b", summaries: _partialSummaries.slice(cStart), totalSummaries: _partialSummaries.length, startIndex: cStart, promptTemplate: currentPromptC, ...llmConfig });
          llmWorkerC.postMessage({ type: "summaries-done" });
        } else {
          setStatusC(`Arm C will start after ${C_START_THRESHOLD} Arm B summaries are ready…`);
        }
      }

      return;
    }

    // Cache miss — build from scratch
    _pendingCacheKey = hash;
    _pendingCacheMeta = {
      transcript: result.text, segments: result.segments, videoId: result.videoId,
      label: entryLabel, embeddingModel: modelSettings.embeddingModel, llmSource: llmSrc,
    };

    setStatus(`Loaded (${result.label}). Building Arm A…`);
    setStatusB("Arm B will start after models load…");
    setStatusC("Arm C will start after Arm B completes…");
    indexProgress.classList.remove("hidden");

    searchWorker.postMessage({ type: "build", transcript: result.text, segments: result.segments, embeddingModel: modelSettings.embeddingModel });
    llmWorkerB.postMessage({ type: "build-b", transcript: result.text, segments: result.segments, promptTemplate: _activePromptB, ...llmConfig });

  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
    _building = false;
    loadBtn.disabled = false;
  }
});

// ── Search ────────────────────────────────────────────────────────────────────
searchBtn.addEventListener("click", () => {
  const query = queryInput.value.trim();
  if (!query) return;
  searchBtn.disabled = true;
  searchWorker.postMessage({ type: "search", query });
});

queryInput.addEventListener("keydown", (e) => { if (e.key === "Enter") searchBtn.click(); });
urlInput.addEventListener("keydown", (e) => { if (e.key === "Enter") loadBtn.click(); });
pasteInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) loadBtn.click(); });
