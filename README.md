# pdf-qa-mcp

A local MCP server that answers natural-language questions grounded in a corpus of PDFs. Retrieval is hybrid (lexical BM25 + dense k-NN with reciprocal-rank fusion) over OpenSearch; answers are synthesized by Amazon Bedrock (Claude) with optional Ollama fallback, returned with per-source page citations.

Built for a take-home spec: *"Build a working MCP server that exposes a Q&A tool over the provided PDF documents."*

---

## 1. Setup — from scratch

### Prerequisites
- macOS or Linux, **Python 3.10+**
- ~3 GB free disk (Python deps + BGE embedder)
- One of: **Docker / Colima / OrbStack** (for local OpenSearch), **OpenSearch tarball**, or an **AWS-managed OpenSearch domain**
- For the LLM: AWS account with **Bedrock** Claude access, or **Ollama** running locally (works fully offline)

### Step-by-step

```bash
# 1. Clone and enter the project
git clone <repo-url> mcp-pdf-qa
cd mcp-pdf-qa

# 2. Create a venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"     # dev extras pull pytest

# 3. Bring up OpenSearch (one of the three options below)

# 3a. Local Docker (recommended for first-time reviewers)
docker compose up -d
# wait ~30 s, verify:
curl -s http://localhost:9200/_cluster/health | python3 -m json.tool

# 3b. OR — Local tarball (no Docker, no VM, k-NN bundled)
# curl -O https://artifacts.opensearch.org/releases/bundle/opensearch/2.15.0/opensearch-2.15.0-darwin-arm64.tar.gz
# tar -xzf opensearch-2.15.0-darwin-arm64.tar.gz
# echo "plugins.security.disabled: true" >> opensearch-2.15.0/config/opensearch.yml
# opensearch-2.15.0/bin/opensearch &

# 3c. OR — AWS-managed domain — set OS_HOST + OS_USE_AWS_AUTH=true in step 4.

# 4. Configure the LLM and OS (copy the template, then edit)
cp .env.example .env
# At minimum, decide:
#   LLM_PROVIDER=bedrock  (default — needs AWS creds with Bedrock access)
#   LLM_PROVIDER=ollama   (no cloud — needs `ollama serve` + `ollama pull llama3.2:3b`)

# 5. (If using Bedrock) make sure AWS creds are loaded
aws sts get-caller-identity     # should not error

# 6. Smoke-test the full stack end-to-end
set -a && source .env && set +a
python scripts/smoke_test.py "What is this corpus about?"

# 7. Run the MCP server (eagerly ingests data/*.pdf on startup, then serves on stdio)
python -m pdf_qa
# or: pdf-qa-server  (console script installed via pyproject.toml)
```

The repo ships with **5 representative PDFs** in `data/`:
- `P19-1598.pdf` — ACL 2019 NLP paper
- `NASDAQ_TSLA_2020.pdf` — Tesla 10-K (449 pages — the stress test)
- `FBS_INL_Public.pdf` — US State Department INL Bureau report
- `USCOURTS-laed-2_16-md-02740-84.pdf` — US court order
- `2023555917.pdf` — Library-of-Congress legal-comparison report

Total ~4 MB, ~1500 chunks indexed. To run against a different corpus, point `DATA_DIR` at any directory of PDFs.

### Hooking up to an MCP client (Kiro / Claude Desktop / Cursor)

Add to the client's MCP config (e.g. `~/.kiro/settings/mcp.json`):

```json
{
  "mcpServers": {
    "pdf-qa": {
      "command": "/abs/path/to/mcp-pdf-qa/.venv/bin/python",
      "args": ["-m", "pdf_qa"],
      "cwd": "/abs/path/to/mcp-pdf-qa",
      "env": {
        "LLM_PROVIDER": "bedrock",
        "LLM_FALLBACK": "ollama",
        "BEDROCK_MODEL_ID": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2:3b",
        "OS_HOST": "localhost",
        "OS_PORT": "9200",
        "OS_USE_SSL": "false",
        "DATA_DIR": "/abs/path/to/mcp-pdf-qa/data"
      },
      "timeout": 120000
    }
  }
}
```

Restart the client and the 7 `pdf-qa` tools become available.

---

## 2. Architecture

```
┌──────────────────────┐    JSON-RPC over stdio     ┌────────────────────────┐
│   MCP Client         │ ─────────────────────────▶ │   pdf-qa MCP Server    │
│   (Kiro / Claude /   │ ◀───────────────────────── │   (Python, on host)    │
│    Custom client)    │                            └───────────┬────────────┘
└──────────────────────┘                                        │
                  ┌─────────────────────────────────────────────┴───────────────┐
                  ▼                                                              ▼
   ┌──────────────────────────────┐                              ┌──────────────────────────────┐
   │  OpenSearch                  │                              │  LLM                         │
   │  ──────────────────────────  │                              │  ──────────────────────────  │
   │  pdf_qa_chunks               │                              │  Amazon Bedrock              │
   │   - text  (BM25)             │                              │   (Claude, primary)          │
   │   - embedding (knn_vector)   │                              │  Ollama                      │
   │  pdf_qa_pages                │                              │   (llama3.2:3b, fallback)    │
   │   - text (raw page, BM25)    │                              └──────────────────────────────┘
   └──────────────────────────────┘
```

### Inside the server

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       pdf-qa MCP Server                                    │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │ Tool layer (FastMCP)                                                 │   │
│ │  query_documents · keyword_search · count_occurrences                │   │
│ │  list_documents · ingest_document · ingest_all · get_document_meta…  │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
│ ┌─────────────────────────────┐   ┌────────────────────────────────────┐   │
│ │ Retrieval (RAG)             │   │ Counting (literal)                 │   │
│ │  BM25 + kNN + RRF → top-k   │   │  Per-page exact match              │   │
│ └─────────────────────────────┘   └────────────────────────────────────┘   │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │ Storage (OpenSearch client; SigV4 / basic / no-auth supported)       │   │
│ │  pdf_qa_chunks  ←  used by retrieval                                 │   │
│ │  pdf_qa_pages   ←  used by counting                                  │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │ Ingestion pipeline                                                   │   │
│ │  PDF → blocks (PyMuPDF) → ≤400-tok chunks (60-tok overlap)           │   │
│ │      → embeddings (BGE-base-en-v1.5, 768-dim) → bulk write           │   │
│ │  PDF → per-page raw text (rawdict per-glyph) → bulk write            │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
│ ┌──────────────────────────────────────────────────────────────────────┐   │
│ │ LLM (Bedrock primary, Ollama fallback via _FallbackClient)           │   │
│ └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Two indices, two read paths

| | `pdf_qa_chunks` (RAG) | `pdf_qa_pages` (counting) |
|---|---|---|
| Granularity | ≤400-token chunks | one record per page |
| Overlap | 60 tokens (helps recall) | none |
| Extractor | `fitz.get_text("text")` | `fitz.rawdict` (per-glyph) |
| Embeddings | yes (768-dim BGE) | no |
| BM25 indexed | yes | yes |
| Used by | `query_documents` | `keyword_search`, `count_occurrences` |

**Why two indices?** Chunk overlap is good for retrieval recall (context bridges chunk boundaries) but inflates literal counts (the same mention appears in two adjacent chunks). Adding a non-overlapping per-page index keeps retrieval quality unchanged while making counting accurate. The pages-index extractor uses PyMuPDF's `rawdict` mode — per-glyph extraction that handles span-broken terms (footnote markers, citations, line wraps) so counts match what Acrobat's Find reports.

### Query flow

```
"What was Tesla's 2020 revenue?"
   │
   ▼
[ embed question ] ──┬──▶ [ BM25 search on chunks ]  ─┐
                     │                                 ├──▶ [ RRF fuse ] ──▶ top-5
                     └──▶ [ kNN search on chunks ]   ─┘                          │
                                                                                 ▼
                                                              [ build prompt + call LLM ]
                                                                                 │
                                                                                 ▼
                                              { answer, sources: [{doc, page, snippet, score}] }
```

---

## 3. MCP tools (7)

### `query_documents` — primary Q&A (RAG)

Embed a question, retrieve top-k chunks via hybrid BM25⊕kNN with RRF fusion, prompt the LLM, return a grounded answer with per-source citations.

| Input | Type | Default | |
|---|---|---|---|
| `question` | string | required | natural-language question |
| `doc_filter` | list[str] \| null | null | restrict to these doc_ids or names |
| `top_k` | int | 0 (uses `TOP_K=5`) | chunks to retrieve |

**Output**
```json
{
  "answer": "Tesla's total revenue for 2020 was $31,536 million... [NASDAQ_TSLA_2020.pdf p.61]",
  "sources": [
    {"doc_id": "...", "doc_name": "NASDAQ_TSLA_2020.pdf", "page": 61, "kind": "text",
     "score": 0.029, "snippet": "Revenue by source ... Year Ended December 31, 2020 ..."}
  ],
  "retrieved_chunks_count": 5,
  "llm_provider": "bedrock"
}
```

**Examples**
- `query_documents("What was Tesla's 2020 revenue?")`
- `query_documents("Compare INL focus areas to the Sanofi case allegations.", top_k=8)`
- `query_documents("What is Linked WikiText-2?", doc_filter=["P19-1598.pdf"])`

### `keyword_search` — literal / regex search

Counts exact (or regex) matches across the **pages index**. Overlap-free, page-tagged. Returns total occurrences plus snippet hits.

| Input | Type | Default | |
|---|---|---|---|
| `term` | string | required | phrase or regex |
| `doc_filter` | list[str] \| null | null | scope to these docs |
| `regex` | bool | false | treat as regex |
| `case_sensitive` | bool | false | |
| `max_hits` | int | 10 | sample snippets to return |

**Output**
```json
{
  "term": "WikiText-2", "case_sensitive": false,
  "total_occurrences": 31, "matching_chunks": 5,
  "docs_matched": ["P19-1598.pdf"],
  "hits": [{"doc_name": "P19-1598.pdf", "page": 5, "score": 4.2, "snippet": "...<<WikiText-2>>..."}]
}
```

### `count_occurrences` — Acrobat-style precise count

Re-reads one PDF via PyMuPDF `rawdict` (per-glyph) for maximum accuracy — matches what Acrobat's Find shows. Use this for "how many times does X appear in doc Y?".

| Input | Type | Default | |
|---|---|---|---|
| `doc_name` | string | required | exact name from `list_documents` |
| `term` | string | required | search term |
| `case_sensitive` | bool | false | |
| `whole_word` | bool | true | require word boundaries |

### `list_documents` — corpus inventory

```json
{ "total": 5, "indexed": 5, "documents": [
    {"doc_id": "...", "name": "NASDAQ_TSLA_2020.pdf", "indexed": true, "chunks": 1293, ...},
    ...
]}
```

### `ingest_document` — ingest one PDF

| Input | Type | Default | |
|---|---|---|---|
| `doc_id_or_name` | string | required | doc_id, file name, or absolute path |
| `force` | bool | false | re-index even if already indexed |

### `ingest_all` — bulk ingest

Ingests every discovered, not-yet-indexed PDF. With `force=true`, re-ingests everything.

### `get_document_metadata` — per-doc facts

Returns `pages` (true page count), `word_count`, `title`, `authors`, `last_author`, `first_page_excerpt`, plus index status.

---

## 4. Technology choices

Per spec section 4.1, the stack was chosen freely. Each pick traded off cost vs. capability.

| Layer | Choice | Why |
|---|---|---|
| **MCP framework** | `mcp>=1.2.0` SDK + **FastMCP** wrapper | Official Anthropic SDK; FastMCP lets you register tools with `@mcp.tool()` and get JSON-Schema introspection automatically. Lower boilerplate than raw `Server`. |
| **PDF parsing** | **PyMuPDF (fitz)** | Fast, native page numbers, supports both `text` mode (for chunks/retrieval) and `rawdict` mode (for per-glyph counting). 5–10× faster than `pdfplumber` on the Tesla 10-K. |
| **Embeddings** | **`BAAI/bge-base-en-v1.5`** via `sentence-transformers` | 768-dim, ~63.5 MTEB, ~440 MB one-time download. Runs locally on Apple Silicon MPS. No API cost. Quality on par with OpenAI `text-embedding-3-small` for retrieval. |
| **Vector store** | **OpenSearch 2.15+** with k-NN plugin (HNSW + cosine) | Single backend for BM25 *and* k-NN — hybrid retrieval with RRF in one engine. Supports SigV4 / basic / no-auth, so the same code runs against local Docker, a tarball install, or an AWS-managed domain. |
| **LLM** | **Amazon Bedrock — Claude Haiku 4.5** primary, **Ollama llama3.2:3b** fallback | Bedrock for production-grade quality; Ollama as zero-cred safety net for AWS-token expiry. Auto-fallback wired so a single `LLM_FALLBACK=ollama` setting catches any Bedrock failure. |
| **Language** | **Python 3.10+** | Required by the suggested PDF / embedding / MCP SDK ecosystem. |

---

## 5. Vibe coding — how this was built

**Tool:** Kiro CLI with Anthropic Claude as the underlying model. Used end-to-end — initial scaffolding, the dual-index refactor, debugging the count bug, writing this README.

### What worked

- **AI as a debugger.** When `keyword_search("WikiText-2")` returned 32 vs Acrobat's 31, I had Kiro write a side-by-side diagnostic comparing PyMuPDF `text` / `blocks` / `rawdict` + pdfplumber + pypdf, printing each match's surrounding context. It pinpointed the 3 missing mentions in seconds — footnote-marker-broken spans that the default text extractor stitched with a phantom space. Without it I'd have read the PDF page-by-page in Preview.
- **AI for "tedious-but-known" boilerplate.** Bedrock `InvokeModel` JSON shape, OpenSearch k-NN mapping, the `_FallbackClient` wrapper, the env var wiring, the migration script for the pages index. Hand-writing them would have been slower with no quality gain.
- **Cross-file coherence.** The dual-index refactor touched schema + client + pipeline + search + config + .env in one pass. Kiro kept dataclass field names and import paths consistent across files — the cross-cutting work I'd otherwise babysit.

### Where I overrode the AI

- **The chunk-overlap counting bug.** First instinct from Kiro was to *tweak the chunker* (bigger chunks / word-aware overlap). I argued that wouldn't fix the root cause — overlap *is* what duplicates the content — and pushed for a second, overlap-free index used only for counting. That decoupled the read paths and cleanly fixed it.
- **Silent fallbacks.** Initial draft of the Bedrock→Ollama failover swallowed the primary's exception. I had it log at WARNING so failures aren't invisible.
- **Trimming scope.** Multiple times I deleted code Kiro would have happily kept — three unused LLM providers, the old `examples/` directory, the migration script after the migration was done. Deletion is part of the work.

### Overall view

AI tooling pays off most when the human does the **taste work** — picking primitives, knowing when *not* to use a framework — and uses the assistant for the **labor** of agreed-on code. It pays off least when the human delegates architectural decisions: the result is plausible-looking code that's mis-shaped for the actual problem. The two design choices Kiro got wrong on its own (chunker fix for the count bug, swallowing the fallback exception) were both cases where it solved the symptom without questioning the model. The human's job is to question the model.

---

## 6. Configuration reference

All config is read from environment variables. See `.env.example`.

| Variable                 | Default                                       | Purpose                                                         |
| --------------------------| -----------------------------------------------| -----------------------------------------------------------------|
| `LLM_PROVIDER`           | `bedrock`                                     | `bedrock` or `ollama`                                           |
| `LLM_FALLBACK`           | empty                                         | Optional. Set to `ollama` for auto-fallback on Bedrock failures |
| `AWS_REGION`             | `us-east-1`                                   | Bedrock + OpenSearch (when SigV4) region                        |
| `BEDROCK_MODEL_ID`       | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Any Bedrock-hosted Claude model id                              |
| `OLLAMA_HOST`            | `http://localhost:11434`                      |                                                                 |
| `OLLAMA_MODEL`           | `llama3.2:3b`                                 | Recommended; alternatives: `llama3.1:8b`, `qwen2.5:7b`          |
| `OS_HOST`                | `localhost`                                   | OpenSearch endpoint                                             |
| `OS_PORT`                | `9200`                                        |                                                                 |
| `OS_USE_SSL`             | `false`                                       | true for AWS-managed                                            |
| `OS_USE_AWS_AUTH`        | `false`                                       | true to use SigV4 (boto3 credentials)                           |
| `OS_USER`, `OS_PASSWORD` | empty                                         | basic auth (when not using SigV4)                               |
| `OS_INDEX`               | `pdf_qa_chunks`                               | retrieval index                                                 |
| `OS_PAGES_INDEX`         | `pdf_qa_pages`                                | counting index                                                  |
| `EMBEDDING_MODEL`        | `BAAI/bge-base-en-v1.5`                       | any sentence-transformers model                                 |
| `EMBEDDING_DEVICE`       | `auto`                                        | `auto` / `mps` / `cuda` / `cpu`                                 |
| `DATA_DIR`               | `./data`                                      | directory of PDFs                                               |
| `CHUNK_TOKENS`           | `400`                                         |                                                                 |
| `CHUNK_OVERLAP`          | `60`                                          |                                                                 |
| `TOP_K`                  | `5`                                           | retrieved chunks per query                                      |

---

## 7. Example interaction log

All transcripts captured live from the running MCP server, Bedrock Claude Haiku 4.5, gamma OpenSearch cluster.

### Example 1 — single-document factual query

**Tool:** `query_documents`
**Input:** `{"question": "What was Tesla's total revenue for 2020?"}`

**Output:**
```json
{
  "answer": "Tesla's total revenue for 2020 was $31,536 million (or approximately $31.5 billion). [NASDAQ_TSLA_2020.pdf p.61]",
  "retrieved_chunks_count": 5,
  "llm_provider": "bedrock",
  "sources": [
    {"doc_name": "NASDAQ_TSLA_2020.pdf", "page": 43, "score": 0.031,
     "snippet": "Cost of automotive leasing revenue increased $104 million..."},
    {"doc_name": "NASDAQ_TSLA_2020.pdf", "page": 41, "score": 0.030,
     "snippet": "2020 compared to 2019 Automotive sales revenue increased $6.23 billion..."},
    {"doc_name": "NASDAQ_TSLA_2020.pdf", "page": 62, "score": 0.029,
     "snippet": "Deferred revenue on automotive sales..."},
    {"doc_name": "NASDAQ_TSLA_2020.pdf", "page": 61, "score": 0.029,
     "snippet": "Revenue by source ... Year Ended December 31, 2020 ..."},
    {"doc_name": "NASDAQ_TSLA_2020.pdf", "page": 41, "score": 0.029,
     "snippet": "...combined average selling price of Model 3 and Model Y..."}
  ]
}
```

### Example 2 — academic/NLP question with multi-page citation

**Tool:** `query_documents`
**Input:** `{"question": "What is the primary challenge addressed by Linked WikiText-2?"}`

**Output:**
```json
{
  "answer": "The primary challenge addressed by Linked WikiText-2 is that 'training data is hard to obtain' for incorporating factual knowledge into language models [P19-1598.pdf p.4]. Specifically, standard language modeling corpora consist only of text without describing which entities or facts each token refers to, while relation extraction datasets provide disjoint sentences that lack sufficient context for training powerful language models [P19-1598.pdf p.4]. Linked WikiText-2 solves this by providing articles from WikiText-2 linked to the Wikidata knowledge graph, enabling the development of language models that can dynamically incorporate facts from a knowledge graph guided by discourse [P19-1598.pdf p.4].",
  "retrieved_chunks_count": 5,
  "llm_provider": "bedrock",
  "sources": [
    {"doc_name": "P19-1598.pdf", "page": 5, "score": 0.033, "snippet": "...Linked WikiText-2 Corpus Statistics..."},
    {"doc_name": "P19-1598.pdf", "page": 4, "score": 0.032, "snippet": "...alias vocabulary, we use a copy mechanism..."},
    {"doc_name": "P19-1598.pdf", "page": 4, "score": 0.032, "snippet": "...coverage of facts expressed in the text..."}
  ]
}
```

### Example 3 — multi-document synthesis

**Tool:** `query_documents`
**Input:** `{"question": "What topics do these documents cover?", "top_k": 8}`

**Output (abbreviated):**
```
Based on the provided excerpts, these documents cover the following topics:

1. **Natural Language Processing and Knowledge Graphs** [P19-1598.pdf p.4][P19-1598.pdf p.8]:
   Entity linking, relation annotations, knowledge graph construction, and neural text
   generation methods.

2. **Constitutional Law and Legislative Procedures** [2023555917.pdf p.4][2023555917.pdf p.6]
   [2023555917.pdf p.7][2023555917.pdf p.16][2023555917.pdf p.19]: Supermajority requirements,
   presidential veto procedures, constitutional amendments, and legislative voting thresholds
   across Bulgaria, Cabo Verde, Poland, Portugal, and China.

3. **Corporate Finance and Risk Management** [NASDAQ_TSLA_2020.pdf p.21]: Warranty reserves,
   insurance coverage strategies, and business risk management for a technology/automotive
   company.
```

Retrieved chunks: 8, drawn from **3 different documents** (P19-1598, 2023555917, NASDAQ_TSLA_2020). Confirms multi-document context handling.

### Example 4 — precise occurrence counting (the bug fix in action)

**Tool:** `keyword_search`
**Input:** `{"term": "WikiText-2", "doc_filter": ["P19-1598.pdf"]}`

**Output:**
```json
{
  "term": "WikiText-2",
  "case_sensitive": false,
  "total_occurrences": 31,
  "matching_chunks": 5,
  "docs_matched": ["P19-1598.pdf"],
  "hits": [
    {"doc_name": "P19-1598.pdf", "page": 5, "snippet": "...<<WikiText-2>> Corpus Statistics..."},
    {"doc_name": "P19-1598.pdf", "page": 4, "snippet": "...Linked <<WikiText-2>> dataset..."},
    {"doc_name": "P19-1598.pdf", "page": 1, "snippet": "...the popular <<WikiText-2>> bench-mark..."},
    {"doc_name": "P19-1598.pdf", "page": 9, "snippet": "...Linked <<WikiText-2>> is freely avail-able..."},
    {"doc_name": "P19-1598.pdf", "page": 6, "snippet": "...Differences from <<WikiText-2>>..."}
  ]
}
```

`total_occurrences = 31` matches Acrobat Find exactly. Pre-fix this returned **32** due to chunk-overlap double counting; the dual-index design + `rawdict` extraction in `pdf_qa_pages` brought it to ground truth. See *Architecture* and the *Vibe coding* section.

### Example 5 — document metadata

**Tool:** `get_document_metadata`
**Input:** `{"doc_id_or_name": "NASDAQ_TSLA_2020.pdf"}`

**Output:**
```json
{
  "doc_id": "440f4b5ad9c2fd1e",
  "name": "NASDAQ_TSLA_2020.pdf",
  "size_kb": 2597.4,
  "pages": 449,
  "chunks": 1293,
  "indexed": true,
  "word_count": 247511,
  "first_page_excerpt": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-K..."
}
```

### Example 6 — server handshake (MCP protocol compliance)

`initialize` response from a fresh `python -m pdf_qa` boot:
```json
{
  "protocolVersion": "2024-11-05",
  "serverInfo":      {"name": "pdf-qa", "version": "1.27.1"},
  "capabilities":    ["experimental", "prompts", "resources", "tools"]
}
```

`tools/list` response:
```
- query_documents:       Answer a natural-language question grounded in the indexed PDFs.
- keyword_search:        Search every indexed chunk for exact (or regex) matches of `term`.
- count_occurrences:     Count exact occurrences of a term in a document by re-reading the PDF.
- list_documents:        List every PDF the server knows about, with indexed status.
- ingest_document:       Ingest a discovered PDF on demand.
- ingest_all:            Ingest every discovered PDF that is not yet indexed.
- get_document_metadata: Return metadata (pages, chunks, size, indexed status) for one PDF.
```

---

## 8. Testing

```bash
# Pure unit tests (no Docker, no OpenSearch, no LLM)
pytest tests/test_chunker.py tests/test_pdf_extract.py

# End-to-end smoke test (requires OpenSearch reachable + LLM creds)
python scripts/smoke_test.py "What is this corpus about?"
```

---

## 9. Repository layout

```
mcp-pdf-qa/
├── README.md                    # this file
├── pyproject.toml               # deps + console scripts
├── docker-compose.yml           # single-node OpenSearch (security disabled)
├── .env.example                 # config template
│
├── src/pdf_qa/
│   ├── __main__.py              # `python -m pdf_qa`
│   ├── server.py                # FastMCP + 7 @mcp.tool registrations
│   ├── config.py                # Settings dataclass (env-driven)
│   ├── manifest.py              # in-memory doc registry
│   ├── prompts.py               # SYSTEM_PROMPT + build_user_prompt
│   ├── ingest/   pdf_extract.py · chunker.py · embed.py · pipeline.py
│   ├── store/    schema.py · client.py · search.py
│   ├── llm/      base.py · bedrock.py · ollama.py · __init__.py  (factory + fallback wrapper)
│   └── tools/    query.py · keyword.py · list_docs.py · ingest.py · metadata.py · _state.py
│
├── scripts/
│   ├── ingest_all.py            # CLI bulk ingestion (`pdf-qa-ingest`)
│   └── smoke_test.py            # OS + embedder + LLM end-to-end check
│
├── tests/
│   ├── test_chunker.py          # unit
│   └── test_pdf_extract.py      # uses bundled PDF
│
└── data/                        # 5 sample PDFs
```

---

## 10. Known limitations

- **Figures/charts not transcribed.** Multimodal content (line graphs, donut charts) misses because there's no VLM pass over figure regions. Future: a Bedrock Claude vision pass during ingestion that stores figure descriptions as extra chunks.
- **Tables rely on PyMuPDF auto-detection.** Works on clean financial tables; less so on borderless layouts. `pdfplumber` could be added as a fallback.
- **No conversational memory.** The MCP server is stateless across calls. Multi-turn refinement is the agent's responsibility.
- **English-only embeddings.** BGE-base-en-v1.5 is monolingual. For multilingual, swap to `BAAI/bge-m3`.
- **No production hardening.** No auth, no rate-limiting, no observability beyond stderr logs. Out of scope per the take-home spec.

---

License: MIT.
