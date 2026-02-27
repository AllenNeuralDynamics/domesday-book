# domesday-book

A shared knowledge base that keeps AI tools informed of your team's project-specific information.

[![PyPI](https://img.shields.io/pypi/v/domesday.svg?label=PyPI&color=blue)](https://pypi.org/project/domesday/)
[![Python version](https://img.shields.io/pypi/pyversions/domesday)](https://pypi.org/project/domesday/)

[![Coverage](https://img.shields.io/codecov/c/github/AllenNeuralDynamics/domesday-book?logo=codecov)](https://app.codecov.io/github/AllenNeuralDynamics/domesday-book)
[![mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/AllenNeuralDynamics/domesday-book/publish.yaml?label=CI/CD&logo=github)](https://github.com/AllenNeuralDynamics/domesday-book/actions/workflows/publish.yaml)
[![GitHub issues](https://img.shields.io/github/issues/AllenNeuralDynamics/domesday-book?logo=github)](https://github.com/AllenNeuralDynamics/domesday-book/issues)

> *The natives call this book "Domesday" ... concerning the matters contained in the book, its word cannot be denied or set aside.*  
>[wikipedia.org/wiki/Domesday_Book](https://en.wikipedia.org/wiki/Domesday_Book)

## Status
🚧 *Ongoing development!* 🚧

Core functionality implemented as a working protoype. 
See [Roadmap](docs/roadmap.md) for next steps.

## Why this exists
Research teams accumulate critical tacit knowledge — processing caveats, data access optimizations, troubleshooting tips — that lives in Teams conversations, scattered notes, and people's heads. This is information AI tools (and other people) need access to.

We need a system where:

- Adding knowledge is as easy as pasting a text snippet into a box
- The system automatically processes new entries
- Multiple team members can contribute and curate entries
- The knowledge base is queryable by AI tools like Claude, giving answers with citations to the original snippets

## Quickstart

```bash
# Install
uv tool install domesday[voyage,mcp]

# Set API keys
export VOYAGE_API_KEY=voy-...
export ANTHROPIC_API_KEY=sk-ant-...

# Add a snippet to a project
domes -p vbo add "The VBO dataset has an off-by-one error in timestamps before 2023-06-01."

# Bulk ingest a folder
domes -p vbo ingest ./project-notes/ --author ben

# Semantic search within a project to find matching snippets (retrieval only)
domes -p vbo search "VBO timestamp issues"

# Ask a question (retrieve matching snippets → LLM generates answer with citations)
domes -p vbo ask "What are the known caveats with the VBO dataset?"

# Actual answer from Claude Sonnet 4.6
# **VBO Dataset Timestamp Error**: The VBO dataset has an **off-by-one error in timestamps** for any data dated **before 2023-06-01**. [snippet-1fffb1]  

# Search across all projects
domes search "timestamp bugs" --all-projects

# Browse and inspect
domes projects             # list all projects with snippet counts
domes -p vbo list          # recent snippets in a project
domes stats --all-projects # stats across everything
```

## Development
```
# clone repo and install:
uv sync --all-extras
```

## How it works

```
Add snippet (paste/CLI/MCP)
  → Store raw text + metadata (SQLite)
  → Chunk (prose/code-aware, ~400 tokens)
  → Embed (Voyage / OpenAI / local model)
  → Index (ChromaDB vector store)

Ask a question (CLI/MCP/API)
  → Embed query
  → Vector similarity search (cosine, with score threshold)
  → [Optional] LLM reranker filters irrelevant results
  → Format context with author, date, tags
  → Generate answer via Claude with inline citations
```

Every backend is behind a Protocol interface — swap storage, embedding, or generation by changing config. See [Architecture](docs/architecture.md) for details.

## Projects

A single domesday instance can hold multiple projects. Each snippet belongs to exactly one project. Queries are scoped to a project by default, preventing cross-contamination between unrelated knowledge bases.

```bash
# Set a default project in config
# domesday.toml: default_project = "vbo"

# Or specify per-command (--project / -p goes before the subcommand)
domes -p vbo add "some caveat"
domes -p ephys-rig add "different caveat"

# Search within a project
domes -p vbo search "timing issues"

# Search across everything
domes search "timing issues" --all-projects

# See what projects exist
domes projects

# Rename a project
domes rename-project old-name new-name
```

The `--project` flag (or `-p`) can also be set at the top level, applying to all subcommands:

```bash
domes -p vbo add "some caveat"
domes -p vbo search "timing"
domes -p vbo ask "what are the known issues?"
```

For MCP, pass the project in tool arguments, or set `DOMESDAY_DEFAULT_PROJECT` in the server environment.

## Configuration

Place `domesday.toml` in your project root:

```toml
data_dir = "./data"
default_project = "main"      # used when --project is not specified

[embedder]
backend = "voyage"             # voyage | openai | local
model = "voyage-4-large"

[generator]
backend = "claude"
model = "claude-sonnet-4-6"

[chunker]
max_tokens = 400
overlap_tokens = 50

[retrieval]
min_score = 0.3               # cosine similarity threshold

[reranker]
enabled = false               # LLM-based relevance filtering (adds latency)
model = "claude-haiku-4-5"
relevance_threshold = 0.5
```

Environment variables override config: `DOMESDAY_DATA_DIR`, `DOMESDAY_EMBEDDER_BACKEND`, `DOMESDAY_EMBEDDER_MODEL`, `DOMESDAY_GENERATOR_MODEL`.

## CLI reference

All commands accept `--project` / `-p` to scope to a specific project. This can also be set at the top level: `domes -p myproject <command>`.

Use `--verbose` / `-v` for INFO-level logs or `--debug` / `-d` for full DEBUG output:

```bash
domes -v search "timestamp issues"     # see search flow
domes -d ingest ./notes/               # see every chunk and embedding call
```

| Command | Description |
|---------|-------------|
| `domes add "text"` | Add a snippet (also accepts `--file`, stdin, or opens `$EDITOR`) |
| `domes add --author ben --tags "vbo,bug"` | Add with metadata |
| `domes -p myproject ingest ./folder/` | Bulk ingest files into a project |
| `domes search "query"` | Semantic search within the current project |
| `domes search "query" --all-projects` | Search across all projects |
| `domes ask "question"` | Retrieve relevant snippets then generate an answer with citations |
| `domes ask "question" --show-sources` | Also print which snippets were used |
| `domes list` | Show recent snippets in current project |
| `domes list --all-projects` | Show recent snippets across all projects |
| `domes projects` | List all projects with snippet counts |
| `domes rename-project old new` | Rename a project across all stores |
| `domes stats` | Show stats for current project |
| `domes stats --all-projects` | Show stats across all projects |

## MCP integration

domesday exposes itself as an MCP server, making the knowledge base available from Claude Desktop, Cursor, VS Code, or any MCP-compatible client.

**Local (stdio) — add to `claude_desktop_config.json`:**

```json
{
  "mcpServers": {
    "domesday": {
      "command": "python",
      "args": ["-m", "domesday.mcp_server"],
      "env": {
        "DOMESDAY_DATA_DIR": "/absolute/path/to/data",
        "DOMESDAY_DEFAULT_PROJECT": "vbo",
        "VOYAGE_API_KEY": "voy-...",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

**Remote (SSE) — for team access:**

```json
{
  "mcpServers": {
    "domesday": {
      "url": "https://your-server.internal:8080/mcp/sse"
    }
  }
}
```

**Available MCP tools:**

| Tool | Description |
|------|-------------|
| `search_knowledge(query, project?, n_results?, tags?)` | Semantic search over snippets |
| `add_snippet(text, project?, author?, tags?)` | Add new knowledge from any client |
| `get_snippet(snippet_id)` | Retrieve a snippet by full or short (8-char) ID |
| `list_recent(n?, project?, author?)` | Browse recent additions |
| `list_projects()` | List all projects with snippet counts |
| `rename_project(old_name, new_name)` | Rename a project across all stores |
| `ask(question, project?, n_context?)` | Retrieve relevant context and generate an answer with citations |

All tools accept an optional `project` parameter. Pass `"__all__"` to search across all projects.

## Evaluation

domesday includes an evaluation framework for measuring retrieval quality and generation faithfulness. See [Evaluation](docs/evaluation.md) for full details.

```bash
# Run retrieval eval against test corpus
python -m domesday.eval.runner

# Also judge generation quality with Haiku
python -m domesday.eval.runner --judge

# Parameter sweep (min_score, k, chunk size, overlap)
python -m domesday.eval.runner --sweep --quick

# Interactive: inspect individual queries and results
python -m domesday.eval.runner -i
```

## Project structure

```
domesday/
├── pyproject.toml
├── domesday.toml
├── domesday/
│   ├── core/
│   │   ├── models.py           # Snippet, Chunk, SearchResult, RAGResponse
│   │   ├── protocols.py        # Swappable interfaces for all backends
│   │   └── pipeline.py         # Orchestrator: add, ingest, search, ask
│   ├── stores/
│   │   ├── sqlite_store.py     # DocumentStore → SQLite
│   │   └── chroma_store.py     # VectorStore → ChromaDB
│   ├── embedders.py            # Voyage, OpenAI, sentence-transformers
│   ├── generators.py           # Claude via Anthropic API
│   ├── chunking.py             # Prose/code-aware text splitting
│   ├── config.py               # defaults + parsing from file/env
│   ├── cli.py                  # CLI commands
│   ├── mcp_server.py           # MCP tool definitions
│   └── eval/
│       ├── models.py           # Eval metrics (precision, recall, MRR)
│       ├── runner.py           # Eval runner + parameter sweeps
│       └── llm_judge.py        # Haiku-based quality scoring + reranker
├── tests/
│   └── fixtures/
│       └── test_corpus.py      # 30 synthetic snippets + 21 eval queries
└── docs/
    ├── architecture.md
    └── evaluation.md
```

## Further reading

- [Architecture & design decisions](docs/architecture.md)
- [Evaluation framework](docs/evaluation.md)