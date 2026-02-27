"""LLM generators for RAG answer synthesis."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import anthropic

from domesday.core import models

logger = logging.getLogger(__name__)


def _format_context(results: Sequence[models.SearchResult]) -> str:
    """Format retrieved snippets into a context block for the prompt."""
    if not results:
        return "(No relevant snippets found in the knowledge base.)"

    sections: list[str] = []
    for r in results:
        short_id = r.snippet.id[:8]
        meta_parts = [f"author={r.snippet.author}"]
        meta_parts.append(f"date={r.snippet.created_at.strftime('%Y-%m-%d')}")
        if r.snippet.tags:
            meta_parts.append(f"tags={','.join(r.snippet.tags)}")
        if r.snippet.source_file:
            meta_parts.append(f"source={r.snippet.source_file}")
        meta = " | ".join(meta_parts)

        sections.append(f"[snippet-{short_id}] ({meta})\n{r.snippet.raw_text}")

    return "\n\n---\n\n".join(sections)


@dataclass(frozen=True, slots=True)
class ClaudeGenerator:
    """Generator using the Anthropic Messages API.

    Env: ANTHROPIC_API_KEY
    """

    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.2  # low temp for factual grounding

    async def generate(
        self,
        query: str,
        context: Sequence[models.SearchResult],
        *,
        system_prompt: str | None = None,
    ) -> models.RAGResponse:
        client = anthropic.AsyncAnthropic()

        context_block = _format_context(context)

        logger.debug(
            "Generating answer: model=%s, %d context snippets, query='%s'",
            self.model,
            len(context),
            query[:80],
        )

        user_message = (
            f"<context>\n{context_block}\n</context>\n\n"
            f"<question>\n{query}\n</question>"
        )

        message = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = "".join(
            block.text for block in message.content if block.type == "text"
        )

        logger.info(
            "Generated answer: %d chars, input_tokens=%d, output_tokens=%d",
            len(answer_text),
            message.usage.input_tokens,
            message.usage.output_tokens,
        )

        return models.RAGResponse(
            answer=answer_text,
            sources=list(context),
            model=self.model,
            query=query,
        )
