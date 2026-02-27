"""Tests for domesday.chunking — SimpleChunker."""

from __future__ import annotations

import pytest

from domesday.chunking import SimpleChunker
from domesday.core.models import Chunk, Snippet, SnippetType


# ---------------------------------------------------------------------------
# SimpleChunker — empty / short input
# ---------------------------------------------------------------------------


def test_chunk_empty_snippet_returns_empty_list() -> None:
    chunker = SimpleChunker()
    snippet = Snippet(raw_text="", project="test")
    assert chunker.chunk(snippet) == []


def test_chunk_whitespace_only_returns_empty_list() -> None:
    chunker = SimpleChunker()
    snippet = Snippet(raw_text="   \n\t  ", project="test")
    assert chunker.chunk(snippet) == []


def test_chunk_short_snippet_produces_single_chunk() -> None:
    chunker = SimpleChunker(max_tokens=400)
    snippet = Snippet(raw_text="A short prose note.", project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) == 1
    assert chunks[0].text == "A short prose note."
    assert chunks[0].snippet_id == snippet.id
    assert chunks[0].chunk_index == 0


# ---------------------------------------------------------------------------
# SimpleChunker — prose splitting
# ---------------------------------------------------------------------------


def test_chunk_prose_covers_all_paragraphs() -> None:
    """Every original paragraph appears in at least one chunk."""
    chunker = SimpleChunker(max_tokens=20, overlap_tokens=0)
    paragraphs = [f"Paragraph {i} with enough words to matter." for i in range(10)]
    long_text = "\n\n".join(paragraphs)
    snippet = Snippet(raw_text=long_text, snippet_type=SnippetType.PROSE, project="test")
    chunks = chunker.chunk(snippet)
    combined = "\n\n".join(c.text for c in chunks)
    for para in paragraphs:
        assert para in combined


def test_chunk_prose_text_within_budget() -> None:
    """Each chunk's character count should not wildly exceed the budget."""
    max_tokens = 20
    chunker = SimpleChunker(max_tokens=max_tokens, overlap_tokens=0)
    para = "word " * 30  # ~30 tokens per paragraph
    long_text = (para + "\n\n") * 10
    snippet = Snippet(raw_text=long_text, snippet_type=SnippetType.PROSE, project="test")
    chunks = chunker.chunk(snippet)
    max_chars = max_tokens * 4 * 3  # generous headroom for single oversized paragraphs
    assert all(len(c.text) <= max_chars for c in chunks)


def test_chunk_prose_overlap_carries_content() -> None:
    """With overlap enabled, the end of chunk N appears at the start of chunk N+1."""
    chunker = SimpleChunker(max_tokens=15, overlap_tokens=10)
    paragraphs = [f"Paragraph number {i} here." for i in range(20)]
    long_text = "\n\n".join(paragraphs)
    snippet = Snippet(raw_text=long_text, snippet_type=SnippetType.PROSE, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) >= 3, "Need multiple chunks to test overlap"
    # Check that at least one chunk pair shares overlapping content
    found_overlap = False
    for a, b in zip(chunks, chunks[1:]):
        # The last paragraph of chunk a should appear in chunk b
        last_para_a = a.text.split("\n\n")[-1]
        if last_para_a in b.text:
            found_overlap = True
            break
    assert found_overlap, "Expected overlapping content between consecutive chunks"


def test_chunk_indices_sequential() -> None:
    chunker = SimpleChunker(max_tokens=10, overlap_tokens=0)
    long_text = ("Hello world. " * 5 + "\n\n") * 10
    snippet = Snippet(raw_text=long_text, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_snippet_id_propagated() -> None:
    chunker = SimpleChunker(max_tokens=10)
    long_text = "word " * 50
    snippet = Snippet(raw_text=long_text, project="test")
    chunks = chunker.chunk(snippet)
    assert all(c.snippet_id == snippet.id for c in chunks)


# ---------------------------------------------------------------------------
# SimpleChunker — code splitting
# ---------------------------------------------------------------------------


def test_chunk_code_single_block_no_split() -> None:
    chunker = SimpleChunker(max_tokens=400)
    code = "def foo():\n    return 42\n"
    snippet = Snippet(raw_text=code, snippet_type=SnippetType.CODE, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) == 1
    assert chunks[0].text == code.strip()


def test_chunk_code_splits_on_blank_lines() -> None:
    """Code snippets split on blank-line boundaries between blocks."""
    chunker = SimpleChunker(max_tokens=10, overlap_tokens=0)
    block = "x = " + "1 + " * 15 + "0\n"
    code = (block + "\n\n") * 10
    snippet = Snippet(raw_text=code, snippet_type=SnippetType.CODE, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) > 1
    assert all(c.snippet_id == snippet.id for c in chunks)


def test_chunk_code_has_no_overlap() -> None:
    """Code chunker does not carry overlap — concatenated chunks reconstruct original."""
    chunker = SimpleChunker(max_tokens=10, overlap_tokens=50)  # overlap param ignored for code
    block = "result = " + "value + " * 15 + "0\n"
    code = (block + "\n\n") * 8
    snippet = Snippet(raw_text=code, snippet_type=SnippetType.CODE, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) > 1
    # Code chunks should have no duplicated content
    reconstructed = "\n\n".join(c.text for c in chunks)
    assert reconstructed == code.strip()
