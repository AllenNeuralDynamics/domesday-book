# domesday — TODO & Next Steps

## Phase 0: Get it running

- [x] Create the repo structure and copy all files in
- [x] `uv sync`
- [x] Set `ANTHROPIC_API_KEY` (needed for generation only)
- [x] Set `VOYAGE_API_KEY` if you want to use Voyage embeddings instead of local — but local works fine to start
- [x] Set `embedder.backend` in `domesday.toml` — `local` for no-API-key embeddings, `voyage` for higher quality via API
- [x] `domes add "test snippet"` — confirm the full ingest pipeline runs (SQLite write → chunk → embed → Chroma index)
- [x] `domes search "test"` — confirm retrieval works end-to-end
- [x] `domes ask "what do you know?"` — confirm generation with citations works
- [x] `domes stats` — verify counts make sense

## Phase 1: Evaluate with the test corpus

- [ ] `python -m domesday.eval.runner` — baseline retrieval metrics with default params
- [ ] `python -m domesday.eval.runner -i` — interactive mode, poke around, try free-form queries, run `eval 0` through `eval 20` to inspect individual results
- [ ] Look at the score distributions: what scores do relevant results get vs. irrelevant? Is there a clean separation or a messy overlap? This tells you whether `min_score` can do the job alone
- [ ] `python -m domesday.eval.runner --sweep --quick` — find a better `min_score` threshold
- [ ] `python -m domesday.eval.runner --judge` — run the Haiku judge, check faithfulness and abstention scores
- [ ] Try a negative query in interactive mode — does the system correctly return nothing?
- [ ] Try a synthesis query — does it pull from multiple relevant snippets?

## Phase 2: Test with real data

- [ ] Gather a folder of real project notes (markdown, text files, whatever you have)
- [ ] `domes -p vbo ingest ./my-notes/ --author ben` — bulk ingest
- [ ] `domes -p vbo list` — sanity check what got ingested, are the snippets sensible or did splitting go wrong?
- [ ] `domes stats` — check snippet and chunk counts, do they feel right for the amount of content?
- [ ] Try 5-10 real questions you know the answer to — does it find the right stuff?
- [ ] Try 2-3 questions you know have no answer in the data — does it correctly abstain or does it hallucinate?
- [ ] If chunking looks wrong (splitting mid-thought, or keeping unrelated content together), experiment with `chunk_max_tokens` and `chunk_overlap` in config

## Phase 3: MCP integration

- [ ] Add domesday to `claude_desktop_config.json` (see README for the config block)
- [ ] Open Claude Desktop, try `search_knowledge` and `ask` tools — verify they work
- [ ] Try `add_snippet` from within a Claude conversation — test the "contribute knowledge from wherever you are" workflow
- [ ] Try in Cursor or VS Code if you use them — same MCP config pattern
- [ ] Test with `DOMESDAY_DEFAULT_PROJECT` set to your actual project name

## Phase 4: Multi-project

- [ ] Create a second project: `domes -p ephys add "some ephys note"`
- [ ] Verify project isolation: `domes -p vbo search "ephys thing"` should NOT find it
- [ ] Verify cross-project: `domes search "ephys thing" --all-projects` SHOULD find it
- [ ] `domes projects` — check the listing looks right
- [ ] `domes rename-project` — test a rename and verify search still works after

## Phase 5: Embedding model comparison

- [ ] Run the eval suite with Voyage (`backend: voyage`, `model: voyage-3-large`) — record metrics
- [ ] Switch to OpenAI (`backend: openai`, `model: text-embedding-3-small`) — re-ingest test corpus into a fresh data dir, run eval, compare
- [ ] If you have GPU access, try the local model (`backend: local`, `model: all-MiniLM-L6-v2`) — useful for offline/free usage even if quality is lower
- [ ] Compare: which model gives the cleanest separation between relevant and irrelevant scores? Which handles your code+prose mix best?

## Phase 6: Reranker evaluation

- [ ] Enable the reranker in config (`reranker: enabled: true`)
- [ ] Run the eval suite — compare precision and specificity with/without reranker
- [ ] Try the reranker in interactive mode on queries that previously had false positives
- [ ] Measure the latency difference — is the extra Haiku call worth it for your query patterns?
- [ ] Tune `relevance_threshold` (0.3 → 0.7 range) and see how it affects negative accuracy

## Explore & extend (no particular order)

- [ ] **Auto-tagging on ingest**: add a Haiku call during ingest to suggest tags automatically. Measure whether it improves retrieval (it might not — embeddings already capture semantics)
- [ ] **Auto-summarization**: generate a one-line summary per snippet on ingest, prepend it to the chunk text before embedding. Hypothesis: richer embedding = better retrieval
- [ ] **Duplicate detection**: on ingest, search for high-similarity existing snippets (>0.9) and warn. Prevents the knowledge base from accumulating near-identical entries
- [ ] **Web UI**: a simple paste-box + chat interface. Could be a single-file FastAPI + HTMX app, or a React app. The pipeline is already async so it'll serve well
- [ ] **S3 document store**: implement the `S3DocumentStore` backend for the GPU-in-cloud + shared-storage architecture. JSON or parquet files keyed by snippet ID
- [ ] **pgvector backend**: implement `PgVectorStore` for multi-user concurrency and production-grade durability
- [ ] **Ingest from URLs**: `domes add --url https://...` that fetches the page, extracts text, and ingests
- [ ] **Slack integration**: a Slack bot or slash command that adds snippets from messages. Low-friction capture from where conversations already happen
- [ ] **Export/import for migration**: `domes export --project vbo --format json > backup.json` and `domes import backup.json --project vbo`. Should export raw snippets with all metadata (author, tags, timestamps, project, parent_id, active status) in a portable JSON format — no embeddings, since those are regenerated on import with whatever embedder the target instance uses. This decouples the data from the storage backend entirely: migrate SQLite → Postgres, switch embedding models, move to a new machine, or share a knowledge base with a collaborator. Also useful as a backup strategy. Consider also supporting JSONL (one snippet per line) for streaming large exports.
- [ ] **Export as markdown**: `domes export --project vbo --format markdown` — dump as a structured document for use as a Claude Projects system prompt, documentation, or human-readable archive
- [ ] **Reindex command**: `domes reindex` — re-embed all active snippets with the current embedding model. Needed when switching embedding models (clears the Chroma collection and rebuilds from the document store). The data is safe in SQLite; only the vectors are regenerated
- [ ] **Per-project embedding models**: allow different projects to use different embedding models and Chroma collections. Would look something like:
  ```toml
  [projects.vbo.embedder]
  backend = "local"
  model = "all-MiniLM-L6-v2"

  [projects.ephys.embedder]
  backend = "voyage"
  model = "voyage-3-large"
  ```
  Each project gets its own Chroma collection (auto-named e.g. `domesday_vbo`, `domesday_ephys`). The Pipeline would resolve the right embedder + collection per project. **Workaround today**: use separate `collection_name` values and swap config files, or auto-name the collection after the model (e.g. `domesday_all-MiniLM-L6-v2`)
- [ ] **Usage analytics**: track which snippets get cited in `ask` responses. After a month, check: are there snippets that are never retrieved? (Maybe poorly embedded or poorly written.) Are some cited constantly? (Maybe they should be promoted or expanded.)

## Known rough edges to fix

- [ ] The `Snippet.from_dict` type handling is loose — the `tags` field does `list(d.get("tags", []))` which doesn't validate contents are strings
- [ ] The eval runner's sweep mode doesn't properly re-ingest with different chunk configs (it overrides the chunker but uses already-embedded chunks). For a true chunk-size sweep, it needs separate data dirs per config — the code has a TODO for this
- [ ] `ChromaVectorStore.search` will error if the collection is empty (`min(k, 0)` = 0 results requested). Add a guard
- [ ] No tests yet beyond the eval framework — add unit tests for chunker, SQLite store, and pipeline
- [ ] The `vec_store.rename_project` call in `Pipeline.rename_project` uses `type: ignore` because the Protocol doesn't include it — either add it to the VectorStore protocol or handle it more gracefully