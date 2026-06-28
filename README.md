# Semantic vs LLM-Indexed Search Demo

Compare three transcript search strategies side by side in the browser. Paste a YouTube URL (or a text file, or plain text), and all three arms index and search it live.

- **Arm A — Semantic**: each transcript chunk is embedded directly; search ranks by cosine similarity to the query.
- **Arm B — LLM summaries + RRF**: each chunk is summarized in 2-3 sentences by a small LLM; search fuses chunk similarity and summary similarity via Reciprocal Rank Fusion.
- **Arm C — Questions + RRF**: questions are generated from Arm B's summaries; search fuses chunk similarity and question similarity via RRF. Arm C starts streaming as soon as Arm B has a few summaries ready.

Only the YouTube transcript fetch runs on a server. Chunking, embedding, LLM inference, and search all run client-side via [transformers.js](https://huggingface.co/docs/transformers.js). Indexes are cached in IndexedDB so re-opening the same video is instant.

## Setup

### 1. Start the transcript server

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Serve the frontend

In a separate terminal:

```bash
python3 -m http.server 3000 --directory web
```

Open `http://localhost:3000`.

## Usage

1. Paste a public YouTube URL (or switch to the file/paste tab) and click **Load & Index**.
2. Arm A finishes first. Arms B and C run in parallel — C starts after B produces its first few summaries.
3. Search at any point. Results refresh automatically as each arm finishes.
4. Try a query that paraphrases a topic rather than quoting the transcript verbatim — that's where B and C surface matches A misses.

## Options

- **Model settings**: choose from 5 embedding models and 4 in-browser LLM models, or point to a local server (Ollama, LM Studio, llama.cpp) for generation.
- **Prompt settings**: edit the Arm B (summary) and Arm C (questions) prompts before indexing.
- **Cached indexes**: each (video, embedding model, LLM) combination is cached separately. Open "Cached Indexes" to see what's stored and delete individual entries.

## Notes

- First load downloads the embedding model (~23 MB) and the LLM (~360 MB for the default SmolLM2-360M); both are cached by the browser afterward.
- WebGPU accelerates LLM inference in Chrome; falls back to WASM automatically.
- Text files and pasted text work without the server running.
