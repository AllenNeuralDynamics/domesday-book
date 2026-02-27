"""Tests for domesday.core.pipeline — Pipeline orchestration with in-memory mocks."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Sequence, Iterable
from pathlib import Path

import pytest

from domesday.chunking import SimpleChunker
from domesday.core import models, protocols
from domesday.core.pipeline import Pipeline


# ---------------------------------------------------------------------------
# In-memory mock implementations of all protocols
# ---------------------------------------------------------------------------


class InMemoryDocStore(protocols.DocumentStore):
    """Minimal in-memory DocumentStore for testing."""

    def __init__(self) -> None:
        self._store: dict[str, models.Snippet] = {}

    async def add(self, snippet: models.Snippet) -> models.Snippet:
        self._store[snippet.id] = snippet
        return snippet

    async def get(self, snippet_id: str) -> models.Snippet | None:
        return self._store.get(snippet_id)

    async def update(self, snippet: models.Snippet) -> models.Snippet:
        self._store[snippet.id] = snippet
        return snippet

    async def deactivate(self, snippet_id: str) -> None:
        s = self._store.get(snippet_id)
        if s is not None:
            # frozen dataclass → rebuild with is_active=False
            import dataclasses

            self._store[snippet_id] = dataclasses.replace(s, is_active=False)

    async def list_recent(
        self,
        n: int = 20,
        *,
        project: str | None = None,
        author: str | None = None,
        tags: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> list[models.Snippet]:
        results = list(self._store.values())
        if active_only:
            results = [s for s in results if s.is_active]
        if project:
            results = [s for s in results if s.project == project]
        return results[:n]

    async def get_all_active(
        self, *, project: str | None = None
    ) -> list[models.Snippet]:
        results = [s for s in self._store.values() if s.is_active]
        if project:
            results = [s for s in results if s.project == project]
        return results

    async def list_projects(self) -> list[str]:
        return sorted({s.project for s in self._store.values() if s.is_active})

    async def rename_project(self, old_name: str, new_name: str) -> int:
        import dataclasses

        count = 0
        for sid, s in list(self._store.items()):
            if s.project == old_name:
                self._store[sid] = dataclasses.replace(s, project=new_name)
                count += 1
        return count

    async def count(
        self, *, active_only: bool = True, project: str | None = None
    ) -> int:
        results = list(self._store.values())
        if active_only:
            results = [s for s in results if s.is_active]
        if project:
            results = [s for s in results if s.project == project]
        return len(results)


class InMemoryVecStore(protocols.VectorStore):
    """Minimal in-memory VectorStore for testing."""

    def __init__(self) -> None:
        # chunk_id → (snippet_id, project, embedding)
        self._chunks: dict[str, tuple[str, str, list[float]]] = {}

    def initialize(self) -> None:
        pass

    @property
    def dimension(self) -> int:
        return 4

    async def add_chunks(
        self,
        *,
        chunks: Iterable[models.Chunk],
        embeddings: Iterable[Sequence[float]],
        embedding_model: str = "test-embedder",
        project: str = "test-project",
    ) -> None:
        for chunk, emb in zip(chunks, embeddings):
            self._chunks[chunk.id] = (chunk.snippet_id, project, emb)

    async def search(
        self,
        query_embedding: Iterable[float],
        k: int | None = 10,
        *,
        project: str | None = None,
        filter_tags: Sequence[str] | None = None,
        embedding_model: str | None = None,
    ) -> list[tuple[str, str, float]]:
        """Returns top-k (chunk_id, snippet_id, score) using dot-product similarity."""
        results: list[tuple[str, str, float]] = []
        for chunk_id, (snippet_id, proj, emb) in self._chunks.items():
            if project is not None and proj != project:
                continue
            score = sum(a * b for a, b in zip(query_embedding, emb))
            results.append((chunk_id, snippet_id, float(score)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    async def delete_by_snippet(self, snippet_id: str) -> None:
        self._chunks = {
            cid: val for cid, val in self._chunks.items() if val[0] != snippet_id
        }

    def count(self) -> int:
        return len(self._chunks)

    async def rename_project(self, old_name: str, new_name: str) -> None:
        for chunk_id, (snippet_id, proj, emb) in list(self._chunks.items()):
            if proj == old_name:
                self._chunks[chunk_id] = (snippet_id, new_name, emb)


class FixedEmbedder(protocols.Embedder):
    """Returns a constant embedding vector for all inputs."""

    def __init__(self, dim: int = 4, value: float = 0.5) -> None:
        self._dim = dim
        self._value = value

    @property
    def model(self) -> str:
        return "test-embedder"

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[self._value] * self._dim for _ in texts]


class EchoGenerator(protocols.Generator):
    """Returns a canned RAGResponse for testing."""

    async def generate(
        self,
        query: str,
        context: Sequence[models.SearchResult],
        *,
        system_prompt: str | None = None,
    ) -> models.RAGResponse:
        return models.RAGResponse(
            answer=f"Answer to: {query}",
            sources=list(context),
            query=query,
        )

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_pipeline(
    *,
    default_project: str = "test",
    reranker=None,
) -> tuple[Pipeline, InMemoryDocStore, InMemoryVecStore]:
    doc_store = InMemoryDocStore()
    vec_store = InMemoryVecStore()
    embedder = FixedEmbedder()
    generator = EchoGenerator()
    chunker = SimpleChunker(max_tokens=400)
    pipeline = Pipeline(
        doc_store=doc_store,
        vec_store=vec_store,
        embedder=embedder,
        generator=generator,
        chunker=chunker,
        default_project=default_project,
        reranker=reranker,
    )
    return pipeline, doc_store, vec_store


# ---------------------------------------------------------------------------
# _split_file
# ---------------------------------------------------------------------------


def test_split_file_custom_delimiter() -> None:
    text = "section one\n---\nsection two\n---\nsection three"
    parts = Pipeline._split_file(text, ".txt", delimiter="\n---\n")
    assert parts == ["section one", "section two", "section three"]


def test_split_file_md_hr() -> None:
    text = "intro\n\n---\n\nbody\n\n---\n\nfooter"
    parts = Pipeline._split_file(text, ".md", delimiter=None)
    assert len(parts) == 3


def test_split_file_md_headings() -> None:
    text = "# Title\n\nIntro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    parts = Pipeline._split_file(text, ".md", delimiter=None)
    assert len(parts) >= 2
    assert any("Section A" in p for p in parts)


def test_split_file_non_md_returns_whole_text() -> None:
    text = "line1\nline2\nline3"
    parts = Pipeline._split_file(text, ".txt", delimiter=None)
    assert parts == [text]


# ---------------------------------------------------------------------------
# add_snippet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_snippet_stores_in_doc_store() -> None:
    pipeline, doc_store, _ = make_pipeline()
    snippet = await pipeline.add_snippet("Hello, world!", project="proj")
    assert await doc_store.get(snippet.id) is not None
    assert snippet.raw_text == "Hello, world!"
    assert snippet.project == "proj"


@pytest.mark.asyncio
async def test_add_snippet_indexes_chunks_in_vec_store() -> None:
    pipeline, _, vec_store = make_pipeline()
    await pipeline.add_snippet("A short note.", project="proj")
    assert vec_store.count() >= 1
    

@pytest.mark.asyncio
async def test_add_snippet_respects_author() -> None:
    pipeline, _, _ = make_pipeline()
    snippet = await pipeline.add_snippet("Note.", project="proj", author="alice")
    assert snippet.author == "alice"


@pytest.mark.asyncio
async def test_add_snippet_respects_tags() -> None:
    pipeline, _, _ = make_pipeline()
    snippet = await pipeline.add_snippet("Note.", project="proj", tags=["foo", "bar"])
    assert snippet.tags == ["foo", "bar"]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_results() -> None:
    pipeline, _, _ = make_pipeline()
    await pipeline.add_snippet("VBO timestamp bug.", project="p")
    results = await pipeline.search("timestamp", project="p")
    assert len(results) >= 1
    assert all(isinstance(r, models.SearchResult) for r in results)


@pytest.mark.asyncio
async def test_search_filters_by_project() -> None:
    pipeline, _, _ = make_pipeline()
    await pipeline.add_snippet("Snippet in project A.", project="projA")
    await pipeline.add_snippet("Snippet in project B.", project="projB")
    results = await pipeline.search("snippet", project="projA")
    assert all(r.snippet.project == "projA" for r in results)


@pytest.mark.asyncio
async def test_search_min_score_filters_low_scores() -> None:
    """With a very high min_score threshold, no results should pass."""
    pipeline, _, _ = make_pipeline()
    await pipeline.add_snippet("Some text.", project="p")
    # FixedEmbedder gives dot product = 4 * 0.5 * 0.5 = 1.0; use >1 to filter all
    results = await pipeline.search("Some text.", project="p", min_score=99.0)
    assert results == []


@pytest.mark.asyncio
async def test_search_all_projects() -> None:
    pipeline, _, _ = make_pipeline()
    await pipeline.add_snippet("Note A.", project="alpha")
    await pipeline.add_snippet("Note B.", project="beta")
    results = await pipeline.search("Note", project="all")
    projects_found = {r.snippet.project for r in results}
    assert "alpha" in projects_found
    assert "beta" in projects_found


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_returns_rag_response() -> None:
    pipeline, _, _ = make_pipeline()
    await pipeline.add_snippet("The answer is 42.", project="proj")
    response = await pipeline.ask("What is the answer?", project="proj")
    assert isinstance(response, models.RAGResponse)
    assert response.answer.startswith("Answer to:")
    assert response.query == "What is the answer?"


@pytest.mark.asyncio
async def test_ask_sources_contain_snippets() -> None:
    pipeline, _, _ = make_pipeline(default_project="p")
    await pipeline.add_snippet("Relevant info.", project="p")
    response = await pipeline.ask("info?", project="p")
    assert isinstance(response.sources, list)


# ---------------------------------------------------------------------------
# revise_snippet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revise_snippet_deactivates_old() -> None:
    pipeline, doc_store, _ = make_pipeline()
    original = await pipeline.add_snippet("Original text.", project="proj")
    await pipeline.revise_snippet(original.id, "Updated text.")
    old = await doc_store.get(original.id)
    assert old is not None
    assert old.is_active is False


@pytest.mark.asyncio
async def test_revise_snippet_creates_new_active_snippet() -> None:
    pipeline, doc_store, _ = make_pipeline()
    original = await pipeline.add_snippet("Original text.", project="proj")
    revised = await pipeline.revise_snippet(original.id, "Revised text.")
    assert revised.id != original.id
    assert revised.raw_text == "Revised text."
    assert revised.is_active is True


@pytest.mark.asyncio
async def test_revise_snippet_removes_old_chunks() -> None:
    pipeline, _, vec_store = make_pipeline()
    original = await pipeline.add_snippet("Original text.", project="proj")
    original_chunk_count = vec_store.count()
    await pipeline.revise_snippet(original.id, "Revised text.")
    # Old chunks removed, new chunks added; net count may be same or different
    # but the old snippet_id should not remain in the vec store
    for _chunk_id, (snippet_id, _proj, _emb) in vec_store._chunks.items():
        assert snippet_id != original.id


@pytest.mark.asyncio
async def test_revise_nonexistent_snippet_raises() -> None:
    pipeline, _, _ = make_pipeline()
    with pytest.raises(ValueError, match="not found"):
        await pipeline.revise_snippet("nonexistent-id", "New text.")


# ---------------------------------------------------------------------------
# list_projects / rename_project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_projects() -> None:
    pipeline, _, _ = make_pipeline()
    await pipeline.add_snippet("Note in alpha.", project="alpha")
    await pipeline.add_snippet("Note in beta.", project="beta")
    projects = await pipeline.list_projects()
    assert "alpha" in projects
    assert "beta" in projects


@pytest.mark.asyncio
async def test_rename_project() -> None:
    pipeline, doc_store, vec_store = make_pipeline()
    await pipeline.add_snippet("Note.", project="old")
    count = await pipeline.rename_project("old", "new")
    assert count >= 1
    projects = await pipeline.list_projects()
    assert "new" in projects
    assert "old" not in projects


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_file_txt() -> None:
    pipeline, doc_store, _ = make_pipeline()
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write("Text file content for ingestion.")
        tmp_path = Path(f.name)
    try:
        snippets = await pipeline.ingest_file(tmp_path, project="files")
        assert len(snippets) == 1
        assert snippets[0].raw_text == "Text file content for ingestion."
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_ingest_file_md_splits_on_hr() -> None:
    pipeline, doc_store, _ = make_pipeline()
    content = "Section one content.\n\n---\n\nSection two content.\n\n---\n\nSection three content."
    with tempfile.NamedTemporaryFile(
        suffix=".md", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = Path(f.name)
    try:
        snippets = await pipeline.ingest_file(tmp_path, project="md")
        assert len(snippets) == 3
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_ingest_file_py_uses_code_type() -> None:
    pipeline, doc_store, _ = make_pipeline()
    code = "def foo():\n    return 42\n"
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = Path(f.name)
    try:
        snippets = await pipeline.ingest_file(tmp_path, project="proj")
        assert len(snippets) == 1
        assert snippets[0].snippet_type == models.SnippetType.CODE
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_directory_processes_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Content A.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Content B.", encoding="utf-8")
    (tmp_path / "skip.jpg").write_bytes(b"\xff\xd8")  # not a text extension

    pipeline, doc_store, _ = make_pipeline()
    result = await pipeline.ingest_directory(tmp_path, project="bulk")
    assert result.snippets_created == 2
    assert result.files_processed == 2
    assert result.ok is True


@pytest.mark.asyncio
async def test_ingest_directory_records_errors(tmp_path: Path) -> None:
    # Create a file that will fail to read as utf-8
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\x80\x81\x82")

    pipeline, _, _ = make_pipeline()
    result = await pipeline.ingest_directory(tmp_path, project="err")
    assert len(result.errors) == 1
    assert result.ok is False


@pytest.mark.asyncio
async def test_ingest_directory_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    (sub / "nested.txt").write_text("nested", encoding="utf-8")

    pipeline, _, _ = make_pipeline()
    result = await pipeline.ingest_directory(tmp_path, project="rec", recursive=True)
    assert result.snippets_created == 2

    result_flat = await pipeline.ingest_directory(tmp_path, project="flat", recursive=False)
    assert result_flat.snippets_created == 1


# ---------------------------------------------------------------------------
# search — inactive snippet handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_skips_inactive_snippets() -> None:
    pipeline, doc_store, _ = make_pipeline()
    snippet = await pipeline.add_snippet("Will be deactivated.", project="p")
    await doc_store.deactivate(snippet.id)
    # Chunks still in vec_store, but search should skip inactive snippets
    results = await pipeline.search("deactivated", project="p")
    assert all(r.snippet.id != snippet.id for r in results)


# ---------------------------------------------------------------------------
# search — with reranker
# ---------------------------------------------------------------------------


class HalfReranker:
    """Reranker that drops the bottom half of results."""

    async def rerank(
        self, query: str, results: list[models.SearchResult]
    ) -> list[models.SearchResult]:
        return results[: len(results) // 2] if len(results) > 1 else results


@pytest.mark.asyncio
async def test_search_with_reranker() -> None:
    pipeline, _, _ = make_pipeline(reranker=HalfReranker())
    await pipeline.add_snippet("Note one.", project="p")
    await pipeline.add_snippet("Note two.", project="p")
    results = await pipeline.search("Note", project="p")
    # HalfReranker should reduce results
    assert len(results) == 1


# ---------------------------------------------------------------------------
# revise — preserves metadata from original
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revise_snippet_preserves_original_metadata() -> None:
    pipeline, doc_store, _ = make_pipeline()
    original = await pipeline.add_snippet(
        "Original.",
        project="proj",
        tags=["important"],
        source_file="notes.md",
        snippet_type=models.SnippetType.REFERENCE,
    )
    revised = await pipeline.revise_snippet(original.id, "Revised.")
    assert revised.project == "proj"
    assert revised.tags == ["important"]
    assert revised.source_file == "notes.md"
    assert revised.snippet_type == models.SnippetType.REFERENCE


# ---------------------------------------------------------------------------
# ask — with no matching snippets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_with_no_snippets_returns_response() -> None:
    """ask() still returns a RAGResponse even when the knowledge base is empty."""
    pipeline, _, _ = make_pipeline()
    response = await pipeline.ask("Anything?", project="empty")
    assert isinstance(response, models.RAGResponse)
    assert response.sources == []
