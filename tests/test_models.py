"""Tests for core data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domesday.core.models import Chunk, IngestResult, RAGResponse, SearchResult, Snippet, SnippetType


def test_snippet_round_trip_preserves_all_fields() -> None:
    """to_dict → from_dict preserves every field including optional ones."""
    s = Snippet(
        raw_text="some text",
        project="test",
        author="tester",
        tags=["a", "b"],
        source_file="notes.md",
        snippet_type=SnippetType.CODE,
        parent_id="parent-123",
        is_active=False,
    )
    d = s.to_dict()
    s2 = Snippet.from_dict(d)
    assert s2.id == s.id
    assert s2.raw_text == s.raw_text
    assert s2.project == s.project
    assert s2.tags == s.tags
    assert s2.source_file == s.source_file
    assert s2.snippet_type == SnippetType.CODE
    assert s2.parent_id == "parent-123"
    assert s2.is_active is False
    assert s2.author == "tester"
    assert s2.created_at == s.created_at
    assert s2.updated_at == s.updated_at


def test_snippet_from_dict_handles_missing_optional_fields() -> None:
    """from_dict fills in sensible defaults when optional fields are absent."""
    minimal = {
        "id": "abc-123",
        "raw_text": "hello",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    s = Snippet.from_dict(minimal)
    assert s.project == "default"
    assert s.author == "anonymous"
    assert s.snippet_type == SnippetType.PROSE
    assert s.source_file is None
    assert s.parent_id is None
    assert s.is_active is True


def test_ingest_result_ok() -> None:
    result = IngestResult(snippets_created=2, chunks_created=5)
    assert result.ok is True

    result_with_errors = IngestResult(errors=["something went wrong"])
    assert result_with_errors.ok is False


def test_rag_response_cited_ids_deduplicates() -> None:
    """cited_snippet_ids deduplicates when same snippet appears in multiple sources."""
    s1 = Snippet(raw_text="a", project="test")
    s2 = Snippet(raw_text="b", project="test")
    r1 = SearchResult(snippet=s1, chunk_text="a chunk 1", score=0.9)
    r2 = SearchResult(snippet=s1, chunk_text="a chunk 2", score=0.85)
    r3 = SearchResult(snippet=s2, chunk_text="b", score=0.8)
    response = RAGResponse(answer="ans", sources=[r1, r2, r3])
    assert response.cited_snippet_ids == [s1.id, s2.id]
