# Architecture & Design Decisions

## Overview

domesday is a RAG (retrieval-augmented generation) system designed for a specific use case: small teams accumulating tacit project knowledge — dataset caveats, experimental findings, pipeline gotchas — and making it queryable through Claude.

This document explains the architectural choices and their rationale.

## Design principles

**Zero-friction input.** Adding knowledge must be as easy as pasting text into a box. No forms, no required metadata, no taxonomies. If it's harder than pasting into Slack, people won't use it.

**Append-mostly, living document.** Optimized for the common case: adding new snippets. Editing and deleting are supported but secondary. Edits create new versions rather than mutating in place.

**Retrieval over organization.** Users don't file things into folders. The system handles chunking, embedding, and retrieval. Search quality is the product.

**Transparent sourcing.** Every generated answer cites which snippets it drew from with snippet IDs, so users can verify claims and build trust in the system.

**Modular backends.** Every component is behind a Protocol interface. Swap storage, embedding, or generation by changing a config file. This supports running the embedding pipeline on a GPU machine while serving queries from a lightweight server.

## Component architecture

```
┌──────────────────────────────────────────────────────┐
│                     INPUT LAYER                      │
│                                                      │
│  Web UI (paste box)  │  CLI tool  │  MCP ingest API  │
└──────────┬───────────┴─────┬──────┴────────┬─────────┘
           │                 │               │
           ▼                 ▼               ▼
┌──────────────────────────────────────────────────────┐
│                PROCESSING PIPELINE                   │
│                                                      │
│  1. Store raw snippet + metadata (who, when, tags)   │
│  2. Auto-classify type (prose, code, table, ref)     │
│  3. Chunk (if long) with overlap                     │
│  4. Generate embeddings (voyage-3 or similar)        │
│  5. Extract/suggest tags (optional, LLM-assisted)    │
│  6. Write to vector store + document store           │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                  STORAGE LAYER                       │
│                                                      │
│  Document store         │  Vector store              │
│  (Postgres / SQLite)    │  (pgvector / Chroma /      │
│  - raw text             │   Qdrant / Pinecone)       │
│  - author, timestamp    │  - embeddings              │
│  - edit history         │  - chunk ↔ snippet links   │
│  - optional tags        │                            │
└──────────┬──────────────┴──────────────┬─────────────┘
           │                             │
           ▼                             ▼         
┌──────────────────────────────────────────────────────┐
│                 RETRIEVAL + GENERATION               │
│                                                      │
│  Query → embed → vector search → rerank → context    │
│  Context + query → Claude (sonnet/opus) → response   │
│  Response includes inline citations [snippet #id]    │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                   ACCESS LAYER                       │
│                                                      │
│      Web UI     |     CLI     │     MCP server       │
└──────────────────────────────────────────────────────┘
```

File layout:

```
domesday/
├── pyproject.toml
├── domesday.toml                       # config (copy and edit)
├── README.md
├── domesday/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py                   # Snippet, Chunk, SearchResult, RAGResponse
│   │   ├── protocols.py                # DocumentStore, VectorStore, Embedder,
│   │   │                               # Generator, Chunker, Reranker
│   │   └── pipeline.py                 # Orchestrator: add, ingest, search, ask
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py             # DocumentStore → SQLite
│   │   └── chroma_store.py             # VectorStore → ChromaDB
│   ├── embedders.py                    # Voyage, OpenAI, sentence-transformers
│   ├── generators.py                   # Claude via Anthropic API
│   ├── chunking.py                     # Prose/code-aware text splitting
│   ├── config.py                       # toml + env config → wired Pipeline
│   ├── cli.py                          # domes CLI commands
│   ├── mcp_server.py                   # MCP tool definitions (7 tools)
│   └── eval/
│       ├── __init__.py
│       ├── models.py                   # Eval metrics (precision, recall, MRR)
│       ├── runner.py                   # Eval runner, sweeps, interactive REPL
│       └── llm_judge.py                # Haiku judge + LLM reranker
├── tests/
│   ├── __init__.py
│   └── fixtures/
│       ├── __init__.py
│       └── test_corpus.py              # 30 synthetic snippets + 21 eval queries
└── docs/
    ├── architecture.md                 # this file
    └── evaluation.md                   # eval framework guide
```

## Protocol interfaces

Six protocols define the swappable boundaries. All live in `core/protocols.py`.

**DocumentStore** — CRUD for snippet metadata and raw text. Current implementation: SQLite via aiosqlite. Future: S3 (JSON/parquet), Postgres.

**VectorStore** — Stores chunk embeddings and performs similarity search. Current implementation: ChromaDB (local persistent mode). Future: pgvector, Qdrant, Pinecone.

**Embedder** — Converts text to dense vectors. Three implementations provided: Voyage AI (`voyage-3-large`), OpenAI (`text-embedding-3-small`), and local sentence-transformers (`all-MiniLM-L6-v2`). The local option is free and works offline, useful for development and testing. Voyage and OpenAI offer better quality, especially for mixed code/prose content.

**Generator** — Produces answers from retrieved context. Current implementation: Claude via the Anthropic API. The system prompt instructs Claude to cite snippets by ID, note contradictions, and explicitly decline to answer when context is insufficient.

**Chunker** — Splits snippets into embeddable pieces. Current implementation: `SimpleChunker` with prose and code strategies. Short snippets (under ~400 tokens, the common case) are not chunked at all. Prose splits on paragraph boundaries with overlap. Code splits on blank-line boundaries.

**Reranker** (optional) — Post-retrieval relevance filtering. Uses Haiku to score each retrieved chunk's actual relevance to the query and discard false positives from embedding similarity. Adds latency and cost, so it's off by default.

## Data model

The fundamental unit is a **Snippet**: a piece of text with metadata (project, author, timestamp, tags, type). Every snippet belongs to exactly one **project**, which provides namespace isolation so a single domesday instance can serve multiple independent knowledge bases.

Snippets are immutable-ish — editing creates a new snippet with `parent_id` pointing to the original, and the original is marked `is_active = False`. This preserves history while keeping retrieval clean.

Snippets are split into **Chunks** for embedding. Each chunk links back to its parent snippet and carries the project in its vector store metadata, enabling project-scoped similarity search at the Chroma level (not just post-filtering).

```
Snippet (stored in DocumentStore)
  ├── id, project, raw_text, summary, author, created_at, tags
  ├── snippet_type: prose | code | table | reference | correction
  ├── parent_id: points to superseded snippet (if revision)
  └── is_active: false if superseded or deleted

Chunk (stored in VectorStore, with project in metadata)
  ├── id, snippet_id, chunk_index
  ├── text (the chunk content)
  └── embedding (vector, stored in ChromaDB)
```

## Multi-project support

A single domesday deployment supports multiple independent projects. This is implemented as a first-class `project` field on every snippet and chunk, with filtering at every layer:

**DocumentStore:** The `project` column is indexed, and all list/count queries accept an optional project filter. The `list_projects()` method returns all distinct project names.

**VectorStore:** Project is stored in Chroma metadata on each chunk. Similarity search accepts a `where` filter on project, so vector search is scoped at the Chroma level — not post-filtered in Python. This means project-scoped queries only search the relevant vectors, not the entire collection.

**Pipeline:** All operations (`add_snippet`, `ingest_*`, `search`, `ask`) accept an optional `project` parameter. If not provided, they use `default_project` from config. The special value `"__all__"` disables project filtering for cross-project queries.

**CLI:** The `--project` / `-p` flag can be set per-command or at the top level (`domes -p myproject <command>`). The `--all-projects` flag enables cross-project queries for search, list, and stats.

**MCP:** All tools accept an optional `project` argument. The default is set by `DOMESDAY_DEFAULT_PROJECT` in the server environment.

**Why not separate databases?** A single database with project filtering is simpler to manage, back up, and deploy. It also enables cross-project search (`__all__`) when needed — for instance, finding all timing-related caveats across every project. Separate databases would make this impossible without a federation layer.

## Retrieval pipeline

When a user asks a question, the pipeline:

1. **Embeds the query** using the configured Embedder.
2. **Vector similarity search** in ChromaDB returns the top-k most similar chunks by cosine distance.
3. **Score threshold filtering** discards results below `min_score` (default 0.3, configurable). This prevents the system from returning confidently-presented but irrelevant results when no relevant content exists. Without this, ChromaDB always returns k results even if the best match is barely related.
4. **Deduplication** — if multiple chunks from the same snippet are retrieved, they're consolidated.
5. **Optional LLM reranking** — Haiku scores each chunk's relevance to the specific query and filters out false positives. This catches cases where embedding similarity is high but actual relevance is low. Adds ~200ms and one Haiku call per query.
6. **Context assembly** — retrieved snippets are formatted with metadata (author, date, tags, ID) into a structured context block.
7. **Generation** — Claude receives the context block and question with a system prompt that instructs it to cite sources, note contradictions, and abstain when context is insufficient.

## Ingest pipeline

Adding content:

1. **Raw text stored immediately** in the DocumentStore. Nothing is lost even if subsequent steps fail.
2. **Type detection** determines whether the content is prose, code, a table, or a reference (currently by file extension for bulk ingest, defaulting to prose for manual adds).
3. **Chunking** splits long content. Short snippets (the common case for pasted factoids) pass through as a single chunk.
4. **Embedding** via the configured Embedder. Batched automatically.
5. **Indexing** in the VectorStore.

For bulk ingest (`domes ingest ./folder/`), the system recursively processes `.md`, `.txt`, `.py`, `.json`, `.csv`, `.yaml`, `.yml` files. Markdown files are split on `---` or `## ` boundaries into multiple snippets.

## Collaboration model

Multiple people contribute to the same knowledge base. Each snippet records its author. There's no locking or access control in v1 — two people can add contradictory information, and the retrieval/generation layer surfaces both, noting the conflict. A human resolves it by adding a correction or editing.

Edits create new versions (new snippet with `parent_id` → original). The original is deactivated and excluded from retrieval but preserved for history. The new version is re-chunked and re-embedded.

## Score threshold rationale

Vector similarity search always returns the top-k nearest neighbors, even if the best match is barely related to the query. This means a question about lunch restaurants would return neuroscience snippets with high confidence if that's all the knowledge base contains.

The `min_score` threshold (cosine similarity, 0.0–1.0) filters out results below a cutoff. The default of 0.3 is a starting point — the right value depends on the embedding model and content. The evaluation framework includes negative queries (questions with no relevant content) specifically to help tune this threshold.

The LLM reranker is the more precise solution but adds cost. The recommended approach: start with just the score threshold, use the eval framework to find a good value, and enable the reranker only if you're still getting too many false positives.

## MCP integration

The MCP server (`mcp_server.py`) wraps the same Pipeline used by the CLI. It exposes six tools: `search_knowledge`, `add_snippet`, `get_snippet`, `list_recent`, `list_projects`, and `ask`. All tools that operate on content accept an optional `project` parameter; if omitted, the server's `DOMESDAY_DEFAULT_PROJECT` environment variable (or the config default) is used.

This makes the knowledge base available from any MCP-compatible client — Claude Desktop, Cursor, VS Code, or custom integrations.

Two transport modes: local stdio (for single-machine use, launched as a subprocess) and remote SSE (for team access over the network). For multi-project setups, you can either configure one MCP server per project (each with a different `DOMESDAY_DEFAULT_PROJECT`) or use a single server and specify the project in each tool call.

The `add_snippet` tool is worth highlighting: it means team members can contribute to the knowledge base from within their editor or Claude conversation without switching to a separate tool.

## Deployment options

**Local single-user** (simplest): SQLite + ChromaDB on local disk. Everything runs in a single Python process. Install with `pip install -e .`, run with `domes`.

**Local multi-user**: Same stack, but with the MCP server running as a persistent process that multiple clients connect to.

**Split architecture** (GPU + S3): Run the ingest worker (embedding) on a GPU machine, write to S3 (document store) and a hosted vector DB (Pinecone, Qdrant Cloud). The query server reads from both and is stateless. The Protocol abstraction makes this a config change, not a code change.

**Cloud light**: Deploy on Fly.io or Railway with managed Postgres + a hosted vector DB. Minimal ops for team access with a web UI.

## The "just use a big prompt" alternative

For knowledge bases under ~100 short snippets, you can skip the entire RAG pipeline and concatenate all snippets into a Claude Project system prompt or a large context window. Claude sees everything at once, can cross-reference freely, and there are no retrieval failures.

RAG earns its keep when:
- The knowledge base exceeds the context window (~1,000+ snippets).
- Query volume makes sending 200K tokens per query expensive.
- You need structured provenance ("this answer came from snippet #47, added by Sarah on Tuesday").
- Precision matters more than recall — retrieval + reranking can outperform "hope the model attends to the right paragraph in a 200K prompt" for needle-in-a-haystack queries.

## Open questions

**Embedding model choice.** Voyage-3-large is strong for mixed code/prose. OpenAI's text-embedding-3-large is also good. Local models (nomic-embed-text, all-MiniLM) are free but lower quality. The eval framework exists specifically to benchmark these on your actual content.

**Chunking mixed content.** When a snippet contains both prose and a code block, should they stay together or split? Keeping together preserves context; splitting may improve retrieval precision for code-specific queries. Currently they stay together for short snippets and split for long ones.

**Auto-enrichment.** Auto-tagging and auto-summarization (via Haiku) on ingest would improve organization and retrieval. Not yet implemented — start with the core retrieval loop and add enrichment when you can measure whether it helps.

**Versioning granularity.** Full snippet replacement (current) vs. inline diff tracking. Full replacement is simpler and sufficient for short snippets.

**Access control.** Projects provide namespace isolation but not security — any user can query any project. True per-user or per-project access control (e.g., restricting who can read sensitive projects) would require authentication and authorization layers. Skipped for v1.