"""Integration tests for ChromaVectorStore against a real temp Chroma instance."""

from __future__ import annotations

from pathlib import Path

import pytest

from domesday.core.models import Chunk
from domesday.stores.chroma_store import ChromaVectorStore


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    s = ChromaVectorStore(path=tmp_path / "chroma", collection_name="test")
    s.initialize()
    return s


def _make_chunks(snippet_id: str, n: int = 1) -> list[Chunk]:
    return [
        Chunk(snippet_id=snippet_id, chunk_index=i, text=f"chunk {i}")
        for i in range(n)
    ]


DIM = 4  # embedding dimension for tests


# ---------------------------------------------------------------------------
# Basic add + search
# ---------------------------------------------------------------------------


async def test_add_and_search(store: ChromaVectorStore) -> None:
    chunks = _make_chunks("snip-1", n=2)
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    await store.add_chunks(
        chunks=chunks, embeddings=embeddings, embedding_model="test", project="proj"
    )
    assert store.count() == 2

    results = await store.search([1.0, 0.0, 0.0, 0.0], k=2, project="proj")
    assert len(results) == 2
    # First result should be closest to query
    chunk_id, snippet_id, score = results[0]
    assert snippet_id == "snip-1"
    assert score > results[1][2]  # higher score for more similar chunk


async def test_search_filters_by_project(store: ChromaVectorStore) -> None:
    chunks_a = _make_chunks("snip-a")
    chunks_b = _make_chunks("snip-b")
    emb = [[0.5] * DIM]
    await store.add_chunks(
        chunks=chunks_a, embeddings=emb, embedding_model="test", project="alpha"
    )
    await store.add_chunks(
        chunks=chunks_b, embeddings=emb, embedding_model="test", project="beta"
    )
    results = await store.search([0.5] * DIM, k=10, project="alpha")
    snippet_ids = {r[1] for r in results}
    assert "snip-a" in snippet_ids
    assert "snip-b" not in snippet_ids


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def test_delete_by_snippet_removes_chunks(store: ChromaVectorStore) -> None:
    chunks = _make_chunks("snip-del", n=3)
    embeddings = [[float(i)] * DIM for i in range(3)]
    await store.add_chunks(
        chunks=chunks, embeddings=embeddings, embedding_model="test", project="proj"
    )
    assert store.count() == 3

    await store.delete_by_snippet("snip-del")
    assert store.count() == 0


# ---------------------------------------------------------------------------
# Rename project
# ---------------------------------------------------------------------------


async def test_rename_project(store: ChromaVectorStore) -> None:
    chunks = _make_chunks("snip-rn", n=2)
    embeddings = [[0.5] * DIM, [0.5] * DIM]
    await store.add_chunks(
        chunks=chunks, embeddings=embeddings, embedding_model="test", project="old"
    )

    count = await store.rename_project("old", "new")
    assert count == 2

    # Should find chunks under "new" but not "old"
    results_new = await store.search([0.5] * DIM, k=10, project="new")
    assert len(results_new) == 2
    results_old = await store.search([0.5] * DIM, k=10, project="old")
    assert len(results_old) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_add_empty_chunks_is_noop(store: ChromaVectorStore) -> None:
    await store.add_chunks(
        chunks=[], embeddings=[], embedding_model="test", project="proj"
    )
    assert store.count() == 0


def test_collection_property_raises_before_initialize() -> None:
    store = ChromaVectorStore(path="unused", collection_name="test")
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = store.collection


async def test_rename_nonexistent_project_returns_zero(
    store: ChromaVectorStore,
) -> None:
    count = await store.rename_project("ghost", "new")
    assert count == 0
