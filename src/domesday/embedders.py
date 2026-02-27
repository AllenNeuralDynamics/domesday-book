"""Embedder implementations.

Each class conforms to the Embedder protocol: a `dimension` property
and an async `embed(texts)` method that returns vectors.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Model → dimension lookup
VOYAGE_DIMENSIONS: dict[str, int] = {
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
}

OPENAI_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

# Max batch sizes per API
_VOYAGE_MAX_BATCH = 128
_OPENAI_MAX_BATCH = 2048


@dataclass(frozen=True, slots=True)
class VoyageEmbedder:
    """Embedder using Voyage AI's embedding API.

    Requires: `pip install voyageai`
    Env: VOYAGE_API_KEY
    """

    model: str = "voyage-3-large"

    @property
    def dimension(self) -> int:
        return VOYAGE_DIMENSIONS.get(self.model, 1024)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import voyageai

        client = voyageai.AsyncClient()
        all_embeddings: list[list[float]] = []

        # Batch to stay within API limits
        for i in range(0, len(texts), _VOYAGE_MAX_BATCH):
            batch = list(texts[i : i + _VOYAGE_MAX_BATCH])
            logger.debug(
                "Voyage embed batch %d–%d of %d texts (model=%s)",
                i,
                i + len(batch),
                len(texts),
                self.model,
            )
            result = await client.embed(
                texts=batch,
                model=self.model,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)  # type: ignore[arg-type]

        logger.info(
            "Voyage embedded %d texts → %d vectors (dim=%d)",
            len(texts),
            len(all_embeddings),
            self.dimension,
        )
        return all_embeddings


@dataclass(frozen=True, slots=True)
class OpenAIEmbedder:
    """Embedder using OpenAI's embedding API.

    Requires: `pip install openai`
    Env: OPENAI_API_KEY
    """

    model: str = "text-embedding-3-small"

    @property
    def dimension(self) -> int:
        return OPENAI_DIMENSIONS.get(self.model, 1536)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            import openai  # noqa
        except ImportError:
            raise ImportError(
                "OpenAI embedder requires the 'openai' package. "
                "Install with `uv add domesday[openai]` or choose a different embedder."
            )

        client = openai.AsyncOpenAI()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _OPENAI_MAX_BATCH):
            batch = list(texts[i : i + _OPENAI_MAX_BATCH])
            logger.debug(
                "OpenAI embed batch %d–%d of %d texts (model=%s)",
                i,
                i + len(batch),
                len(texts),
                self.model,
            )
            response = await client.embeddings.create(
                input=batch,
                model=self.model,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        logger.info(
            "OpenAI embedded %d texts → %d vectors (dim=%d)",
            len(texts),
            len(all_embeddings),
            self.dimension,
        )
        return all_embeddings


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Local embedder using sentence-transformers (runs on GPU if available).

    Requires: `pip install sentence-transformers`
    No API key needed — runs locally.
    """

    model: str = "all-MiniLM-L6-v2"
    _dimension: int = 384  # default for MiniLM
    _model_instance: object | None = field(default=None, init=False, repr=False)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> object:
        if self._model_instance is None:
            import sentence_transformers

            self._model_instance = sentence_transformers.SentenceTransformer(self.model)
            # Update dimension from the actual model
            self._dimension = self._model_instance.get_sentence_embedding_dimension()
            logger.info("Loaded local model '%s' (dim=%d)", self.model, self._dimension)
        return self._model_instance

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._get_model()
        logger.debug("Local embed: %d texts (model=%s)", len(texts), self.model)
        embeddings = model.encode(list(texts), show_progress_bar=False)  # type: ignore[attr-defined]
        logger.info(
            "Local embedded %d texts → %d vectors (dim=%d)",
            len(texts),
            len(embeddings),
            self._dimension,
        )
        return [e.tolist() for e in embeddings]
