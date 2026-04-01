"""Vector embedding management for memory retrieval.

Supports three backends (auto-fallback):
1. OpenAI-compatible API embeddings (text-embedding-3-small)
2. sentence-transformers local model
3. TF-IDF bag-of-words (zero-dependency fallback)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Dimension for TF-IDF fallback
_TFIDF_DIM = 256


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash_token(token: str, dim: int = _TFIDF_DIM) -> int:
    """Hash a token to a dimension index."""
    h = hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()
    return int(h, 16) % dim


class EmbeddingProvider:
    """Manages embedding generation with automatic fallback."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_base_url: str = "",
        api_key: str = "",
        cache_dir: Path | None = None,
    ) -> None:
        self._model = model
        self._api_base_url = api_base_url
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._backend: str | None = None
        self._dim: int = _TFIDF_DIM
        self._local_model: Any = None

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        if self._backend is None:
            self._detect_backend()
        return self._backend  # type: ignore[return-value]

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        if self._backend is None:
            self._detect_backend()
        return self._dim

    def _detect_backend(self) -> None:
        """Auto-detect the best available embedding backend."""
        # 1. Try OpenAI API
        if self._api_base_url and self._api_key:
            self._backend = "api"
            self._dim = 1536  # text-embedding-3-small default
            logger.info("Embedding backend: OpenAI API (%s)", self._model)
            return

        # 2. Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._backend = "sentence_transformers"
            self._dim = 384
            logger.info("Embedding backend: sentence-transformers (all-MiniLM-L6-v2)")
            return
        except ImportError:
            pass

        # 3. Fallback to TF-IDF
        self._backend = "tfidf"
        self._dim = _TFIDF_DIM
        logger.info("Embedding backend: TF-IDF fallback (dim=%d)", self._dim)

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a text string.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if self._backend is None:
            self._detect_backend()

        if self._backend == "api":
            return self._embed_api(text)
        elif self._backend == "sentence_transformers":
            return self._embed_local(text)
        else:
            return self._embed_tfidf(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if self._backend is None:
            self._detect_backend()

        if self._backend == "sentence_transformers" and self._local_model is not None:
            embeddings = self._local_model.encode(texts)
            return [e.tolist() for e in embeddings]

        return [self.embed(t) for t in texts]

    def _embed_api(self, text: str) -> list[float]:
        """Get embedding from OpenAI-compatible API."""
        import urllib.request

        url = f"{self._api_base_url.rstrip('/')}/embeddings"
        payload = json.dumps({
            "input": text[:8000],
            "model": self._model,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["data"][0]["embedding"]
        except Exception as exc:
            logger.warning("API embedding failed, falling back to TF-IDF: %s", exc)
            return self._embed_tfidf(text)

    def _embed_local(self, text: str) -> list[float]:
        """Get embedding from local sentence-transformers model."""
        if self._local_model is None:
            return self._embed_tfidf(text)
        embedding = self._local_model.encode(text)
        return embedding.tolist()

    def _embed_tfidf(self, text: str) -> list[float]:
        """Generate TF-IDF-style bag-of-words embedding (zero-dependency fallback)."""
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        vec = [0.0] * _TFIDF_DIM
        for token in tokens:
            idx = _hash_token(token, _TFIDF_DIM)
            vec[idx] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec
