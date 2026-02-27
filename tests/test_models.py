"""Basic tests for core data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domesday.core.models import Chunk, IngestResult, RAGResponse, SearchResult, Snippet, SnippetType


def test_snippet_defaults() -> None:
    s = Snippet(raw_text="hello")
    assert s.project == "default"
    assert s.author == "anonymous"
    assert s.snippet_type == SnippetType.PROSE
    assert s.is_active is True
    assert isinstance(s.id, str) and len(s.id) == 36


def test_snippet_round_trip() -> None:
    s = Snippet(
        raw_text="some text",
        project="test",
        author="tester",
        tags=["a", "b"],
    )
    d = s.to_dict()
    s2 = Snippet.from_dict(d)
    assert s2.id == s.id
    assert s2.raw_text == s.raw_text
    assert s2.project == s.project
    assert s2.tags == s.tags


def test_snippet_type_values() -> None:
    assert SnippetType.PROSE == "prose"
    assert SnippetType.CODE == "code"
    assert SnippetType.TABLE == "table"


def test_chunk_defaults() -> None:
    c = Chunk()
    assert isinstance(c.id, str)
    assert c.chunk_index == 0
    assert c.text == ""


def test_ingest_result_ok() -> None:
    result = IngestResult(snippets_created=2, chunks_created=5)
    assert result.ok is True

    result_with_errors = IngestResult(errors=["something went wrong"])
    assert result_with_errors.ok is False


def test_rag_response_cited_ids() -> None:
    s1 = Snippet(raw_text="a")
    s2 = Snippet(raw_text="b")
    r1 = SearchResult(snippet=s1, chunk_text="a", score=0.9)
    r2 = SearchResult(snippet=s2, chunk_text="b", score=0.8)
    response = RAGResponse(answer="ans", sources=[r1, r2])
    assert response.cited_snippet_ids == [s1.id, s2.id]
