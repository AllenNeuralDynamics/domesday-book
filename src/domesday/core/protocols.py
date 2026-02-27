"""Protocol definitions for swappable backends.

Each protocol defines a minimal interface. Implementations live in
domesday/stores/, domesday/embedders/, domesday/generators/.
Swap backends by changing config — nothing else touches these contracts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domesday.core import models


# ---------------------------------------------------------------------------
# Document storage (metadata + raw text)
# ---------------------------------------------------------------------------


@runtime_checkable
class DocumentStore(Protocol):
    """Persistent storage for snippet metadata and raw text."""

    async def add(self, snippet: models.Snippet) -> models.Snippet: ...

    async def get(self, snippet_id: str) -> models.Snippet | None: ...

    async def update(self, snippet: models.Snippet) -> models.Snippet: ...

    async def deactivate(self, snippet_id: str) -> None:
        """Soft-delete: marks a snippet as inactive (superseded or removed)."""
        ...

    async def list_recent(
        self,
        n: int = 20,
        *,
        project: str | None = None,
        author: str | None = None,
        tags: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> list[models.Snippet]: ...

    async def get_all_active(self, *, project: str | None = None) -> list[models.Snippet]:
        """Return all active snippets, optionally filtered by project."""
        ...

    async def list_projects(self) -> list[str]:
        """Return all distinct project names with active snippets."""
        ...

    async def rename_project(self, old_name: str, new_name: str) -> int:
        """Rename all snippets from old_name to new_name. Returns count updated."""
        ...


# ---------------------------------------------------------------------------
# Vector storage (embeddings + similarity search)
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and supports similarity search."""

    async def add_chunks(
        self,
        chunks: Sequence[models.Chunk],
        embeddings: Sequence[list[float]],
        *,
        project: str = "default",
    ) -> None: ...

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        *,
        project: str | None = None,
        filter_tags: Sequence[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Returns list of (chunk_id, snippet_id, score)."""
        ...

    async def delete_by_snippet(self, snippet_id: str) -> None:
        """Remove all chunks for a given snippet (used on edit/deactivate)."""
        ...


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Converts text to dense vector embeddings."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more texts. Batching is handled internally."""
        ...


# ---------------------------------------------------------------------------
# Text generation (RAG answer generation)
# ---------------------------------------------------------------------------


@runtime_checkable
class Generator(Protocol):
    """Generates answers from retrieved context using an LLM."""

    async def generate(
        self,
        query: str,
        context: Sequence[models.SearchResult],
        *,
        system_prompt: str | None = None,
    ) -> models.RAGResponse: ...


# ---------------------------------------------------------------------------
# Chunker (text splitting strategy)
# ---------------------------------------------------------------------------


@runtime_checkable
class Chunker(Protocol):
    """Splits a snippet's text into chunks for embedding."""

    def chunk(self, snippet: models.Snippet) -> list[models.Chunk]: ...


# ---------------------------------------------------------------------------
# Reranker (optional post-retrieval relevance filtering)
# ---------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    """Reranks/filters search results by relevance to the query."""

    async def rerank(
        self,
        query: str,
        results: Sequence[models.SearchResult],
    ) -> list[models.SearchResult]: ...
