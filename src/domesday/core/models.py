"""Core data models for domesday."""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum  # noqa


class SnippetType(StrEnum):
    """Classification of a knowledge snippet."""

    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    REFERENCE = "reference"
    CORRECTION = "correction"


@dataclass(frozen=False, slots=True)
class Snippet:
    """A single knowledge entry contributed by a user.

    This is the fundamental unit of the knowledge base. Snippets are
    immutable-ish: edits create a new snippet with parent_id pointing
    to the original, and the original is marked inactive.
    """

    raw_text: str
    project: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    summary: str = ""
    snippet_type: SnippetType = SnippetType.PROSE
    author: str = "anonymous"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = field(default_factory=list)
    source_file: str | None = None
    parent_id: str | None = None
    is_active: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "project": self.project,
            "raw_text": self.raw_text,
            "summary": self.summary,
            "snippet_type": self.snippet_type.value,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "source_file": self.source_file,
            "parent_id": self.parent_id,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        return cls(
            id=str(d["id"]),
            project=str(d.get("project", "default")),
            raw_text=str(d["raw_text"]),
            summary=str(d.get("summary", "")),
            snippet_type=SnippetType(str(d.get("snippet_type", "prose"))),
            author=str(d.get("author", "anonymous")),
            created_at=datetime.fromisoformat(str(d["created_at"])),
            updated_at=datetime.fromisoformat(str(d["updated_at"])),
            tags=list(d.get("tags", [])),  # type: ignore[call-overload]
            source_file=str(d["source_file"]) if d.get("source_file") else None,
            parent_id=str(d["parent_id"]) if d.get("parent_id") else None,
            is_active=bool(d.get("is_active", True)),
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of a snippet, prepared for embedding and retrieval.

    Short snippets (<500 tokens) produce a single chunk.
    Longer ones are split with overlap.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    snippet_id: str = ""
    chunk_index: int = 0
    text: str = ""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single result from vector similarity search."""

    snippet: Snippet
    chunk_text: str
    score: float
    chunk_id: str = ""


@dataclass(slots=True)
class RAGResponse:
    """A generated answer grounded in retrieved snippets."""

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    model: str = ""
    query: str = ""

    @property
    def cited_snippet_ids(self) -> list[str]:
        """Unique snippet IDs referenced in this response."""
        return list(dict.fromkeys(s.snippet.id for s in self.sources))


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Summary of a bulk ingest operation."""

    snippets_created: int = 0
    chunks_created: int = 0
    files_processed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0
