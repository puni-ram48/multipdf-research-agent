# 🛠️ System Optimizations & Engineering Trade-Offs

This document details the step-by-step engineering challenges, debugging processes, and performance optimizations implemented while building the **MultiPDF Research Agent**.

---

## 📌 Milestone Summary

1. **LLM Provider Transition & Rate-Limit Resilience**
2. **Structure-Aware Parsing & False-Positive Table Reduction**
3. **Ingestion Latency Reduction (4–5 mins ➔ 1–2 mins)**
4. **Incremental Multi-PDF Indexing (Session Persistence)**

---

## 1. LLM Provider Transition & Rate-Limit Resilience

### 🚨 Problem
Initial prototypes relied on Google's Gemini API free tier. During multi-turn conversational testing and chunk extraction evaluations, requests frequently hit strict rate limits (`HTTP 429: Resource Exhausted`), causing UI freezes and failing tests.

### 💡 Root Cause
Free-tier API quotas and burst-rate limits could not support continuous retrieval evaluation and concurrent prompt construction during rapid user queries.

### 🛠️ Fix & Optimization
* **Provider Switch**: Migrated the inference pipeline to **Groq Cloud (LPUs)** utilizing `openai/gpt-oss-120b` / `llama-3.3-70b-versatile` for hardware-accelerated token generation.
* **Resilience Layer**: Wrapped API calls inside exponential backoff retry logic using the `tenacity` library to gracefully retry transient `503` or `429` status codes:
  ```python
  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min=2, max=10),
      retry=retry_if_exception_type(Exception)
  )
  def _call_llm_with_retry(self, messages):
      return self.llm.invoke(messages)

```

---

## 2. Structure-Aware Parsing & False-Positive Table Reduction

### 🚨 Problem

When parsing multi-column academic papers (e.g., IEEE/ACM 2-column layouts), the UI sidebar reported inflated table counts (e.g., detecting **35 tables** on a paper containing only **5 real tables**).

### 💡 Root Cause

1. **Layout Grid Misclassification**: Standard layout extractors misclassified aligned author metadata, copyright notice boxes, and indented bullet points as individual micro-tables.
2. **Naive Metric Counting**: Table counts were measured using raw Markdown pipe divider strings (`docling_md.count("|---")`), counting sub-header splits and multi-page table chunks multiple times.

### 🛠️ Fix & Optimization

Implemented a heuristic filtering function (`count_real_tables`) inside `PDFParserService` that validates Markdown table structures before incrementing metrics:

* **Row/Column Heuristic**: Requires a minimum of 3 contiguous pipe-delimited lines (`| Header |`, `|---|`, `| Data |`).
* **Metadata Filtering**: Automatically ignores non-table grids containing metadata phrases (`ACM acknowledges`, `Copyright`, `arXiv:`, `Department of`).

---

## 3. Ingestion Pipeline Latency Reduction

### 🚨 Problem

Uploading a 13-page academic PDF initially took **4 to 5 minutes** to process before becoming queryable.

### 💡 Root Cause

1. **Model Cold-Start**: Docling layout models and HuggingFace embedding weights were being loaded from disk on demand during the first upload request.
2. **Unnecessary OCR Overhead**: Docling defaulted to running full-page Optical Character Recognition (OCR) on every page, taking 10–20 seconds per page on CPU hardware.

### 🛠️ Fix & Optimization

* **Lifespan Model Pre-Warming**: Pre-loaded `BAAI/bge-large-en-v1.5` embeddings and Docling converter models into RAM during FastAPI's startup event (`lifespan`), eliminating cold-start delays.
* **Adaptive Hybrid Parser**: Built an inspection function (`is_scanned_or_handwritten`) that checks for digital text layers in `<5ms`. Standard digital papers bypass OCR (`do_ocr = False`) with multi-threaded CPU acceleration (`OMP_NUM_THREADS = 8`), while handwritten/scanned files automatically route to OCR.

### 📊 Benchmark Results

| Document Type | Pages | Processing Time (Before) | Processing Time (After) | Improvement |
| --- | --- | --- | --- | --- |
| Short Paper | 3 – 4 pages | ~2 minutes | **~45 – 60 seconds** | **~2× Faster** |
| Long Benchmark Paper | 13 pages | ~4.5 minutes | **~1.5 – 2 minutes** | **~2.5× Faster** |

---

## 4. Incremental Multi-PDF Indexing

### 🚨 Problem

When users uploaded paper #1 and later uploaded paper #2 to the same chat session, the system wiped the existing vector store and re-indexed both papers from scratch, wasting processing time.

### 💡 Root Cause

The `build_index` function cleared the session index directory (`faiss_{session_id}`) on every `POST /api/v1/ingest` request.

### 🛠️ Fix & Optimization

Updated `VectorStoreService.build_index` to inspect if a session index already exists. If found, it appends only the new document chunks using `FAISS.add_documents()`:

```python
if os.path.exists(os.path.join(index_path, "index.faiss")):
    existing_vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    existing_vector_store.add_documents(chunks)
    existing_vector_store.save_local(index_path)

```

* **Result**: Uploading additional PDFs to an active session indexes only the newly added file, reducing incremental upload times to seconds.

