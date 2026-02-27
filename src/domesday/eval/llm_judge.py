"""LLM-as-judge for evaluating RAG response quality.

Uses a fast, cheap model (Haiku) to score generated answers on dimensions
that pure retrieval metrics can't capture: faithfulness, citation accuracy,
abstention, and synthesis quality.

Also provides a reranker for filtering retrieved chunks at query time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections.abc import Sequence

import anthropic

from domesday.core import models

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Response quality judgments
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgmentScores:
    """Scores assigned by the LLM judge to a RAG response."""

    faithfulness: float       # 0–1: does the answer stick to what's in the snippets?
    citation_accuracy: float  # 0–1: do cited IDs actually support the claims?
    abstention: float         # 0–1: did it correctly say "I don't know" when appropriate?
    synthesis: float          # 0–1: for multi-source answers, coherent combination?
    relevance: float          # 0–1: does the answer actually address the question?
    explanation: str = ""     # free-text reasoning from the judge

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        return (
            self.faithfulness * 0.30
            + self.citation_accuracy * 0.20
            + self.abstention * 0.20
            + self.synthesis * 0.15
            + self.relevance * 0.15
        )


JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for a RAG (retrieval-augmented generation) system.
You will be given a question, the retrieved context snippets, and the system's
generated answer. Score the answer on several dimensions.

Respond ONLY with a JSON object (no markdown, no preamble):
{
    "faithfulness": <0.0-1.0>,
    "citation_accuracy": <0.0-1.0>,
    "abstention": <0.0-1.0>,
    "synthesis": <0.0-1.0>,
    "relevance": <0.0-1.0>,
    "explanation": "<brief reasoning>"
}

Scoring guide:
- faithfulness: 1.0 = every claim is supported by the provided snippets.
  0.0 = the answer fabricates information not in any snippet.
  Penalize hallucinated details even if they sound plausible.
- citation_accuracy: 1.0 = every cited snippet ID actually supports the claim
  it's attached to. 0.0 = citations are wrong or fabricated.
  If no citations are present but should be, score 0.
- abstention: 1.0 = when the snippets don't contain the answer, the system
  clearly says so. 0.0 = the system makes up an answer from nothing.
  If snippets DO contain the answer, score 1.0 (abstention not needed).
- synthesis: 1.0 = information from multiple snippets is combined coherently
  and contradictions are noted. 0.0 = disjointed or contradictory presentation.
  For single-source answers, score 1.0.
- relevance: 1.0 = the answer directly addresses what was asked.
  0.0 = the answer is off-topic despite having relevant context.
"""


@dataclass(frozen=True, slots=True)
class LLMJudge:
    """Uses a fast LLM to evaluate RAG response quality."""

    model: str = JUDGE_MODEL
    max_tokens: int = 512

    async def judge(
        self,
        query: str,
        response: models.RAGResponse,
        *,
        is_negative: bool = False,
    ) -> JudgmentScores:
        """Score a single RAG response."""
        client = anthropic.AsyncAnthropic()

        # Format the context that was available to the generator
        context_parts: list[str] = []
        for r in response.sources:
            sid = r.snippet.id[:8]
            context_parts.append(
                f"[snippet-{sid}] ({r.snippet.author}, "
                f"{r.snippet.created_at.strftime('%Y-%m-%d')})\n"
                f"{r.snippet.raw_text}"
            )
        context_block = "\n---\n".join(context_parts) if context_parts else "(no snippets retrieved)"

        user_message = (
            f"<question>{query}</question>\n\n"
            f"<retrieved_context>\n{context_block}\n</retrieved_context>\n\n"
            f"<generated_answer>\n{response.answer}\n</generated_answer>\n\n"
            f"<metadata>\n"
            f"is_negative_query: {is_negative}\n"
            f"(If is_negative_query is true, the knowledge base has NO relevant "
            f"information. The system should have abstained or said it doesn't know.)\n"
            f"</metadata>"
        )

        message = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = "".join(
            block.text for block in message.content if block.type == "text"
        )

        try:
            scores = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            logger.warning("Judge returned invalid JSON: %s", raw_text[:200])
            return JudgmentScores(
                faithfulness=0.0,
                citation_accuracy=0.0,
                abstention=0.0,
                synthesis=0.0,
                relevance=0.0,
                explanation=f"PARSE ERROR: {raw_text[:200]}",
            )

        return JudgmentScores(
            faithfulness=float(scores.get("faithfulness", 0)),
            citation_accuracy=float(scores.get("citation_accuracy", 0)),
            abstention=float(scores.get("abstention", 0)),
            synthesis=float(scores.get("synthesis", 0)),
            relevance=float(scores.get("relevance", 0)),
            explanation=str(scores.get("explanation", "")),
        )

    async def judge_batch(
        self,
        items: Sequence[tuple[str, models.RAGResponse, bool]],
    ) -> list[JudgmentScores]:
        """Judge multiple responses concurrently.

        Args:
            items: list of (query, response, is_negative) tuples.
        """
        import asyncio

        tasks = [
            self.judge(query, response, is_negative=is_neg)
            for query, response, is_neg in items
        ]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Reranker: use LLM to filter/reorder retrieved chunks
# ---------------------------------------------------------------------------


RERANK_SYSTEM_PROMPT = """\
You are a relevance judge for a retrieval system. Given a query and a list of
text chunks, score each chunk's relevance to the query.

Respond ONLY with a JSON array of objects (no markdown, no preamble):
[
    {"index": 0, "relevant": true, "score": 0.9},
    {"index": 1, "relevant": false, "score": 0.1},
    ...
]

- "relevant": true if this chunk contains information useful for answering the query
- "score": 0.0–1.0 confidence in relevance
Be strict: if a chunk is vaguely related but doesn't actually help answer the
specific question, mark it as not relevant.
"""


@dataclass(frozen=True, slots=True)
class LLMReranker:
    """Uses a fast LLM to rerank/filter retrieved chunks by relevance."""

    model: str = JUDGE_MODEL
    max_tokens: int = 1024
    relevance_threshold: float = 0.5

    async def rerank(
        self,
        query: str,
        results: Sequence[models.SearchResult],
    ) -> list[models.SearchResult]:
        """Rerank results by LLM-judged relevance, filtering out irrelevant ones."""
        if not results:
            return []

        client = anthropic.AsyncAnthropic()

        chunks_block = "\n\n".join(
            f"[{i}] {r.snippet.raw_text[:500]}"
            for i, r in enumerate(results)
        )

        user_message = (
            f"<query>{query}</query>\n\n"
            f"<chunks>\n{chunks_block}\n</chunks>"
        )

        message = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=RERANK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = "".join(
            block.text for block in message.content if block.type == "text"
        )

        try:
            judgments = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            logger.warning("Reranker returned invalid JSON, returning original order")
            return list(results)

        # Build reranked list
        scored: list[tuple[int, float, bool]] = []
        for j in judgments:
            idx = int(j.get("index", -1))
            score = float(j.get("score", 0))
            relevant = bool(j.get("relevant", False))
            if 0 <= idx < len(results):
                scored.append((idx, score, relevant))

        # Sort by LLM score descending, filter by threshold
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: list[models.SearchResult] = []
        for idx, score, relevant in scored:
            if not relevant or score < self.relevance_threshold:
                continue
            original = results[idx]
            # Replace the vector similarity score with the LLM relevance score
            reranked.append(
                models.SearchResult(
                    snippet=original.snippet,
                    chunk_text=original.chunk_text,
                    score=score,
                    chunk_id=original.chunk_id,
                )
            )

        return reranked