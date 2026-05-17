# Evaluation Dataset Summary

## Overview

The evaluation dataset lives in `AbhijjjPlayground/mcp-ask-pdf/data/*/` with one JSONL file per PDF document.

| Metric | Value |
|--------|-------|
| Total JSONL files | 229 |
| Total questions | 1,102 |
| Avg questions/doc | 4.8 |
| Median questions/doc | 4 |
| Min / Max per doc | 1 / 16 |

## Question Type Distribution

| Type | Count | % | Description |
|------|-------|---|-------------|
| `text-only` | 412 | 37% | Answer found in plain text passages (no tables/figures needed) |
| `meta-data` | 258 | 23% | Requires document metadata: page numbers, authors, word counts, term occurrences |
| `multimodal-t` | 220 | 20% | Answer requires reading a **table** in the PDF |
| `unanswerable` | 117 | 11% | The document does NOT contain the answer — tests refusal behavior |
| `multimodal-f` | 88 | 8% | Answer requires interpreting a **figure/chart** in the PDF |
| `una-web` | 7 | 1% | Answer requires external web lookup (e.g., Google Scholar citations) |

## Meta-data Question Patterns (258 total)

| Pattern | Count | Example |
|---------|-------|---------|
| Page location | 85 | "On which page does the paper introduce X?" |
| Other metadata | 75 | "What is the filing date?" / "What court issued this?" |
| Count (general) | 39 | "How many tables are in the document?" |
| Term occurrence count | 20 | "How many times does the paper mention WikiText-2?" |
| Document stats | 14 | "How many pages/words does the document have?" |
| Title | 13 | "What is the title of the document?" |
| Author | 12 | "Who is the last author?" |

## Document Types (229 PDFs)

| Category | Count | Examples |
|----------|-------|----------|
| Other (reports, patents, misc) | 147 | Law Library reports, CRS reports, patents |
| Academic papers | 43 | ACL/EMNLP/NAACL papers (P19-*, D18-*, N18-*, etc.) |
| Government documents | 27 | State Dept FBS, budget justifications |
| SEC filings | 7 | Tesla 10-K, other annual reports |
| Legal/court documents | 5 | USCOURTS MDL orders |

## Tool Routing by Question Type

Based on the question types, the optimal tool routing for the MCP server is:

| Question Type | Primary Tool | Fallback |
|---------------|-------------|----------|
| `text-only` | `query_documents` (semantic RAG) | — |
| `multimodal-t` | `query_documents` (tables are chunked as markdown) | — |
| `multimodal-f` | `query_documents` | Likely to fail — figures aren't extracted |
| `meta-data` (page location) | `query_documents` (page refs in citations) | `keyword_search` |
| `meta-data` (term count) | `keyword_search` (exact count on pages index) | — |
| `meta-data` (author/title) | `get_document_metadata` (direct PDF read) | — |
| `meta-data` (word/page count) | `get_document_metadata` | Needs enhancement |
| `unanswerable` | `query_documents` (should refuse gracefully) | — |
| `una-web` | Out of scope (requires web access) | — |

## Current Coverage Assessment

| Type | Supported? | Notes |
|------|-----------|-------|
| `text-only` | ✅ Full | Hybrid retrieval (BM25 + k-NN) handles well |
| `multimodal-t` | ✅ Good | Tables extracted as markdown via `find_tables()` |
| `multimodal-f` | ❌ Partial | Figures not extracted; only captions may be indexed |
| `meta-data` | ✅ Mostly | Page location, term count, author, title all work. Word count needs a new tool. |
| `unanswerable` | ✅ Good | LLM correctly refuses when sources don't contain answer |
| `una-web` | ❌ None | Would need web search integration |

## JSONL Schema

Each line in a `*_qa.jsonl` file is a JSON object:

```json
{
  "question": "What is the primary challenge addressed by...",
  "answer": "The primary challenge is...",
  "type": "text-only | multimodal-t | multimodal-f | meta-data | unanswerable | una-web",
  "evidence": "Direct quote or description of where the answer is found (empty for meta-data)"
}
```

## Gaps & Recommendations

1. **Word/page count tool** — Add a `count_words` tool that reads the full PDF and counts tokens. Currently `keyword_search` can count specific terms but not total words.

2. **Figure understanding** — 88 questions (8%) require figure interpretation. Would need vision model integration or figure-caption extraction improvements.

3. **Web-augmented answers** — 7 questions require external web lookup. Out of scope for a PDF-only system but could be addressed with a web search tool.

4. **Multi-hop reasoning** — Some `multimodal-t` questions require calculations across table cells (e.g., "percentage increase from 2017 to 2020"). The LLM handles this when both data points are in retrieved chunks, but fails when data spans documents or years not in the filing.
