"""Chunking strategies for splitting snippets into embeddable pieces."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from domesday.core import models

logger = logging.getLogger(__name__)


def _approx_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


@dataclass(frozen=True, slots=True)
class SimpleChunker:
    """Splits text into overlapping chunks by token budget.

    Short snippets (below max_tokens) are returned as a single chunk.
    Code snippets are split on blank-line boundaries where possible.
    """

    max_tokens: int = 400
    overlap_tokens: int = 50

    def chunk(self, snippet: models.Snippet) -> list[models.Chunk]:
        text = snippet.raw_text.strip()
        if not text:
            logger.debug("Snippet %s is empty, skipping chunking", snippet.id[:8])
            return []

        est_tokens = _approx_token_count(text)

        # Short enough to be a single chunk — the common case
        if est_tokens <= self.max_tokens:
            logger.debug(
                "Snippet %s fits in single chunk (~%d tokens)",
                snippet.id[:8],
                est_tokens,
            )
            return [
                models.Chunk(
                    snippet_id=snippet.id,
                    chunk_index=0,
                    text=text,
                )
            ]

        if snippet.snippet_type == models.SnippetType.CODE:
            chunks = self._chunk_code(snippet, text)
        else:
            chunks = self._chunk_prose(snippet, text)

        logger.debug(
            "Snippet %s (~%d tokens, type=%s) → %d chunks",
            snippet.id[:8],
            est_tokens,
            snippet.snippet_type.value,
            len(chunks),
        )
        return chunks

    def _chunk_prose(self, snippet: models.Snippet, text: str) -> list[models.Chunk]:
        """Split prose on sentence/paragraph boundaries with overlap."""
        max_chars = self.max_tokens * 4
        overlap_chars = self.overlap_tokens * 4

        # Split into paragraphs first, then recombine into chunks
        paragraphs = text.split("\n\n")
        chunks: list[models.Chunk] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            if current_len + para_len > max_chars and current:
                chunk_text = "\n\n".join(current)
                chunks.append(
                    models.Chunk(
                        snippet_id=snippet.id,
                        chunk_index=len(chunks),
                        text=chunk_text,
                    )
                )
                # Overlap: keep last paragraph(s) that fit in overlap budget
                overlap: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > overlap_chars:
                        break
                    overlap.insert(0, p)
                    overlap_len += len(p)
                current = overlap
                current_len = overlap_len

            current.append(para)
            current_len += para_len

        if current:
            chunks.append(
                models.Chunk(
                    snippet_id=snippet.id,
                    chunk_index=len(chunks),
                    text="\n\n".join(current),
                )
            )

        return chunks

    def _chunk_code(self, snippet: models.Snippet, text: str) -> list[models.Chunk]:
        """Split code on blank-line boundaries (function/class gaps)."""
        max_chars = self.max_tokens * 4
        blocks = text.split("\n\n")
        chunks: list[models.Chunk] = []
        current: list[str] = []
        current_len = 0

        for block in blocks:
            block_len = len(block)

            if current_len + block_len > max_chars and current:
                chunks.append(
                    models.Chunk(
                        snippet_id=snippet.id,
                        chunk_index=len(chunks),
                        text="\n\n".join(current),
                    )
                )
                current = []
                current_len = 0

            current.append(block)
            current_len += block_len

        if current:
            chunks.append(
                models.Chunk(
                    snippet_id=snippet.id,
                    chunk_index=len(chunks),
                    text="\n\n".join(current),
                )
            )

        return chunks
