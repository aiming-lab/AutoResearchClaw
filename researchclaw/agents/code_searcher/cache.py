"""Disk-based cache for code search results.

Caches search results by domain + topic hash with a configurable TTL
(default 30 days). This avoids redundant GitHub API calls for similar
topics within the same domain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "code_search_cache"
_DEFAULT_TTL_DAYS = 30


class SearchCache:
    """Disk-based cache for code search results.

    Cache structure::

        code_search_cache/
          {domain_id}/
            {topic_hash}.json
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_days: int = _DEFAULT_TTL_DAYS,
    ) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._ttl_sec = ttl_days * 86400

    def get(self, domain_id: str, topic: str) -> dict[str, Any] | None:
        """Get cached result if it exists and is not expired."""
        cache_path = self._cache_path(domain_id, topic)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            timestamp = data.get("_cached_at", 0)
            if time.time() - timestamp > self._ttl_sec:
                logger.debug("Cache expired for %s/%s", domain_id, topic[:40])
                cache_path.unlink(missing_ok=True)
                return None
            logger.info("Cache hit for %s/%s", domain_id, topic[:40])
            return data
        except Exception:
            logger.warning("Failed to read cache", exc_info=True)
            return None

    def put(self, domain_id: str, topic: str, data: dict[str, Any]) -> None:
        """Store a result in the cache."""
        cache_path = self._cache_path(domain_id, topic)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data["_cached_at"] = time.time()
        data["_domain_id"] = domain_id
        data["_topic_hash"] = self._topic_hash(topic)

        try:
            cache_path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            logger.debug("Cached result for %s/%s", domain_id, topic[:40])
        except Exception:
            logger.warning("Failed to write cache", exc_info=True)

    def clear(self, domain_id: str | None = None) -> int:
        """Clear cache. Returns number of entries removed."""
        count = 0
        if domain_id:
            domain_dir = self._cache_dir / domain_id
            if domain_dir.is_dir():
                for f in domain_dir.glob("*.json"):
                    f.unlink()
                    count += 1
        else:
            if self._cache_dir.is_dir():
                for f in self._cache_dir.rglob("*.json"):
                    f.unlink()
                    count += 1
        return count

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        total = 0
        expired = 0
        by_domain: dict[str, int] = {}

        if not self._cache_dir.is_dir():
            return {"total": 0, "expired": 0}

        for f in self._cache_dir.rglob("*.json"):
            total += 1
            domain = f.parent.name
            by_domain[domain] = by_domain.get(domain, 0) + 1
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if time.time() - data.get("_cached_at", 0) > self._ttl_sec:
                    expired += 1
            except Exception:
                pass

        return {"total": total, "expired": expired, **by_domain}

    def _cache_path(self, domain_id: str, topic: str) -> Path:
        return self._cache_dir / domain_id / f"{self._topic_hash(topic)}.json"

    @staticmethod
    def _topic_hash(topic: str) -> str:
        return hashlib.sha256(topic.lower().strip().encode()).hexdigest()[:16]
