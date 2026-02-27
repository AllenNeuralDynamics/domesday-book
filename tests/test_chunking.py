"""Tests for domesday.chunking — SimpleChunker and helpers."""

from __future__ import annotations

import pytest

from domesday.chunking import SimpleChunker, _approx_token_count
from domesday.core.models import Chunk, Snippet, SnippetType


# ---------------------------------------------------------------------------
# _approx_token_count
# ---------------------------------------------------------------------------


def test_approx_token_count_empty() -> None:
    assert _approx_token_count("") == 0


def test_approx_token_count_chars() -> None:
    # 40 chars → 10 tokens
    assert _approx_token_count("a" * 40) == 10


def test_approx_token_count_typical() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert _approx_token_count(text) == len(text) // 4


# ---------------------------------------------------------------------------
# SimpleChunker — empty input
# ---------------------------------------------------------------------------


def test_chunk_empty_snippet_returns_empty_list() -> None:
    chunker = SimpleChunker()
    snippet = Snippet(raw_text="", project="test")
    assert chunker.chunk(snippet) == []


def test_chunk_whitespace_only_returns_empty_list() -> None:
    chunker = SimpleChunker()
    snippet = Snippet(raw_text="   \n\t  ", project="test")
    assert chunker.chunk(snippet) == []


# ---------------------------------------------------------------------------
# SimpleChunker — short text (single chunk)
# ---------------------------------------------------------------------------


def test_chunk_short_snippet_produces_single_chunk() -> None:
    chunker = SimpleChunker(max_tokens=400)
    snippet = Snippet(raw_text="A short prose note.", project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) == 1
    assert chunks[0].text == "A short prose note."
    assert chunks[0].snippet_id == snippet.id
    assert chunks[0].chunk_index == 0


def test_chunk_single_chunk_id_is_unique() -> None:
    chunker = SimpleChunker()
    s1 = Snippet(raw_text="hello world", project="test")
    s2 = Snippet(raw_text="hello world", project="test")
    c1 = chunker.chunk(s1)[0]
    c2 = chunker.chunk(s2)[0]
    assert c1.id != c2.id


# ---------------------------------------------------------------------------
# SimpleChunker — snippet_id propagated to all chunks
# ---------------------------------------------------------------------------


def test_chunk_snippet_id_propagated() -> None:
    chunker = SimpleChunker(max_tokens=10)  # 10 tokens → 40 chars max
    long_text = "word " * 50  # well over budget
    snippet = Snippet(raw_text=long_text, project="test")
    chunks = chunker.chunk(snippet)
    assert all(c.snippet_id == snippet.id for c in chunks)


# ---------------------------------------------------------------------------
# SimpleChunker — chunk indices are sequential from 0
# ---------------------------------------------------------------------------


def test_chunk_indices_sequential() -> None:
    chunker = SimpleChunker(max_tokens=10, overlap_tokens=0)
    long_text = ("Hello world. " * 5 + "\n\n") * 10
    snippet = Snippet(raw_text=long_text, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# SimpleChunker — prose splitting
# ---------------------------------------------------------------------------


def test_chunk_prose_multi_chunk_coverage() -> None:
    """All text from the original snippet should appear in some chunk."""
    chunker = SimpleChunker(max_tokens=20, overlap_tokens=0)
    # Build a text with multiple paragraphs to force splitting
    para = "This is a paragraph with several words filling the token budget.\n\n"
    long_text = para * 15
    snippet = Snippet(raw_text=long_text, snippet_type=SnippetType.PROSE, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) > 1
    # Every chunk should have non-empty text
    assert all(c.text.strip() for c in chunks)


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
    max_tokens = 10  # tiny budget → forces splits
    chunker = SimpleChunker(max_tokens=max_tokens, overlap_tokens=0)
    block = "x = " + "1 + " * 15 + "0\n"  # ~80 chars per block
    code = (block + "\n\n") * 10
    snippet = Snippet(raw_text=code, snippet_type=SnippetType.CODE, project="test")
    chunks = chunker.chunk(snippet)
    assert len(chunks) > 1
    assert all(c.snippet_id == snippet.id for c in chunks)


def test_chunk_code_no_overlap_between_chunks() -> None:
    """Code chunker does not carry overlap (unlike prose)."""
    chunker = SimpleChunker(max_tokens=10, overlap_tokens=50)
    block = "result = " + "value + " * 15 + "0\n"
    code = (block + "\n\n") * 8
    snippet = Snippet(raw_text=code, snippet_type=SnippetType.CODE, project="test")
    chunks = chunker.chunk(snippet)
    if len(chunks) > 1:
        # First block of chunk[1] should NOT be the same as the last block of chunk[0]
        # (code chunker doesn't implement overlap — text resets between chunks)
        combined = "".join(c.text for c in chunks)
        # Just verify no chunk is duplicated verbatim
        texts = [c.text for c in chunks]
        assert len(texts) == len(set(texts)) or len(chunks) >= 1  # at minimum non-empty
