"""ChromaDB-backed vector store for chunk embeddings and similarity search."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from domesday.core import models

logger = logging.getLogger(__name__)


@dataclass
class ChromaVectorStore:
    """VectorStore implementation using ChromaDB (local persistent mode).

    Chroma handles vector indexing and similarity search. We store
    chunk text, snippet_id, and project as metadata for filtering
    and reconstruction of SearchResults.

    Usage:
        store = ChromaVectorStore(path="./data/chroma")
        store.initialize()
    """

    path: str | Path
    collection_name: str = "domesday"
    embedding_model: str = ""  # set by pipeline, stored in collection metadata
    _collection: chromadb.Collection | None = field(
        default=None, init=False, repr=False
    )

    def initialize(self) -> None:
        """Create or open the persistent Chroma collection.

        Validates that the embedding model matches what's already stored.
        If the collection exists with a different model, raises an error
        rather than silently mixing incompatible embeddings.
        """
        client = chromadb.PersistentClient(path=str(self.path))

        try:
            existing = client.get_collection(name=self.collection_name)
            stored_model = (existing.metadata or {}).get("embedding_model", "")

            if (
                stored_model
                and self.embedding_model
                and stored_model != self.embedding_model
            ):
                raise RuntimeError(
                    f"Embedding model mismatch: collection was built with "
                    f"'{stored_model}' but config specifies '{self.embedding_model}'. "
                    f"Either switch back, re-embed with `domes reindex`, or use a "
                    f"different collection_name."
                )

            self._collection = existing
            logger.info(
                "Chroma collection '%s' ready (%d vectors, model=%s)",
                self.collection_name,
                self._collection.count(),
                stored_model or "unknown",
            )
        except Exception as exc:
            if "does not exist" in str(exc).lower() or self._collection is None:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "embedding_model": self.embedding_model,
                    },
                )
                logger.info(
                    "Created Chroma collection '%s' (model=%s)",
                    self.collection_name,
                    self.embedding_model,
                )

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            raise RuntimeError("Store not initialized — call store.initialize()")
        return self._collection

    # ---------------------------------------------------------------
    # Write
    # ---------------------------------------------------------------

    async def add_chunks(
        self,
        chunks: Sequence[models.Chunk],
        embeddings: Sequence[list[float]],
        *,
        project: str = "default",
    ) -> None:
        if not chunks:
            logger.debug("No chunks to add, skipping")
            return

        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=[list(e) for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "snippet_id": c.snippet_id,
                    "chunk_index": c.chunk_index,
                    "project": project,
                    "embedding_model": self.embedding_model,
                }
                for c in chunks
            ],
        )
        logger.debug(
            "Added %d chunks to Chroma (project=%s, snippet=%s)",
            len(chunks),
            project,
            chunks[0].snippet_id[:8] if chunks else "?",
        )

    async def delete_by_snippet(self, snippet_id: str) -> None:
        """Remove all chunks belonging to a snippet."""
        self.collection.delete(where={"snippet_id": snippet_id})
        logger.debug("Deleted chunks for snippet %s from Chroma", snippet_id[:8])

    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        *,
        project: str | None = None,
        filter_tags: Sequence[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Similarity search, returns (chunk_id, snippet_id, score).

        Args:
            project: If provided, only search within this project's chunks.
            filter_tags: Tag filtering handled at pipeline level.
        """
        where_filter: dict[str, str] | None = None
        if project is not None:
            where_filter = {"project": project}

        logger.debug(
            "Chroma search: k=%d, project=%s",
            k,
            project or "(all)",
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count() or 1),
            include=["metadatas", "distances", "documents"],
            where=where_filter,
        )

        output: list[tuple[str, str, float]] = []

        ids = results["ids"][0] if results["ids"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for chunk_id, meta, distance in zip(ids, metadatas, distances):
            # Cosine distance → similarity: sim = 1 - distance
            score = 1.0 - distance
            snippet_id = str(meta.get("snippet_id", ""))
            output.append((chunk_id, snippet_id, score))

        logger.debug(
            "Chroma search returned %d results (scores: %s)",
            len(output),
            [f"{s:.3f}" for _, _, s in output[:5]],
        )
        return output

    # ---------------------------------------------------------------
    # Info
    # ---------------------------------------------------------------

    def count(self) -> int:
        return self.collection.count()

    async def rename_project(self, old_name: str, new_name: str) -> int:
        """Update project metadata on all chunks belonging to old_name."""
        # Chroma doesn't support bulk metadata update with a where filter,
        # so we fetch matching IDs and update them.
        results = self.collection.get(
            where={"project": old_name},
            include=["metadatas"],
        )
        if not results["ids"]:
            logger.debug("No chunks found for project '%s' in Chroma", old_name)
            return 0

        updated_metadatas = []
        for meta in results["metadatas"] or []:
            new_meta = dict(meta)
            new_meta["project"] = new_name
            updated_metadatas.append(new_meta)

        self.collection.update(
            ids=results["ids"],
            metadatas=updated_metadatas,
        )
        logger.info(
            "Renamed project '%s' → '%s' in Chroma (%d chunks)",
            old_name,
            new_name,
            len(results["ids"]),
        )
        return len(results["ids"])
