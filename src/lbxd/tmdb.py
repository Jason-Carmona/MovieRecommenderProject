"""Film metadata from TMDB: posters, runtimes, directors, synopses.

The recommender produces MovieLens ids, which are useless to a web page. This
module turns them into something a browser can render. MovieLens `links.csv`
already gave us the TMDB id for every film, so no searching is involved — it is
a direct lookup.

Three decisions drive the whole design, and each is a response to a real
failure mode:

**Everything is cached on disk.** The same popular films appear in thousands of
users' recommendation lists, so an uncached service would ask TMDB for
*Parasite* forever. The cache turns a 20-film page from 20 network round trips
into zero on the second visit. SQLite is the store: one file, atomic writes,
safe for concurrent readers, and in the standard library — no extra dependency
for something this small.

**Misses are cached too.** A TMDB id from `links.csv` can be wrong or deleted,
and an uncached 404 means re-requesting a film that will never exist, on every
single page view. A negative cache entry costs one row and stops that cold.

**Metadata failure must never fail a recommendation.** The film list is the
product; a poster is decoration. Every fetch is individually wrapped, and
anything that fails comes back as `None` so the API can still return the
ranking with empty poster fields. A service that 500s because TMDB had a bad
minute is worse than one that shows grey boxes.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

API_BASE = "https://api.themoviedb.org/3"

# The documented way to get this is TMDB's /configuration endpoint. It has been
# the same string for over a decade, and hardcoding it saves a round trip on
# every cold start. If posters ever 404 everywhere at once, check here first.
IMAGE_BASE = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w500"      # ~500px wide: sharp on a card, small enough to be fast
BACKDROP_SIZE = "w1280"

CACHE_TTL = 30 * 24 * 3600     # a month; runtimes never change, posters rarely
DEFAULT_CONCURRENCY = 8
REQUEST_TIMEOUT = 10.0
MAX_RETRIES = 3


@dataclass
class Metadata:
    """What a web page needs to render one film."""

    tmdb_id: int
    title: str | None = None
    overview: str | None = None
    poster_url: str | None = None
    backdrop_url: str | None = None
    runtime: int | None = None
    director: str | None = None
    tmdb_rating: float | None = None
    release_date: str | None = None
    genres: list[str] | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _image_url(path: str | None, size: str) -> str | None:
    return f"{IMAGE_BASE}/{size}{path}" if path else None


def _parse(tmdb_id: int, payload: dict) -> Metadata:
    """Pull the handful of fields we use out of TMDB's large response."""
    crew = (payload.get("credits") or {}).get("crew") or []
    directors = [c.get("name") for c in crew if c.get("job") == "Director"]

    return Metadata(
        tmdb_id=tmdb_id,
        title=payload.get("title"),
        overview=payload.get("overview") or None,
        poster_url=_image_url(payload.get("poster_path"), POSTER_SIZE),
        backdrop_url=_image_url(payload.get("backdrop_path"), BACKDROP_SIZE),
        runtime=payload.get("runtime") or None,
        # Co-directed films are common enough to matter (the Coens, the
        # Dardennes, Everything Everywhere), so join rather than take the first.
        director=" & ".join(d for d in directors if d) or None,
        tmdb_rating=payload.get("vote_average") or None,
        release_date=payload.get("release_date") or None,
        genres=[g["name"] for g in payload.get("genres") or [] if g.get("name")],
    )


class MetadataCache:
    """SQLite-backed store for TMDB responses, including the misses.

    `missing = 1` rows are negative cache entries: TMDB told us this id does not
    exist, and we should not ask again until the TTL expires.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS metadata (
                   tmdb_id  INTEGER PRIMARY KEY,
                   payload  TEXT,
                   missing  INTEGER NOT NULL DEFAULT 0,
                   fetched  REAL    NOT NULL
               )"""
        )
        self._db.commit()

    def get_many(self, tmdb_ids: list[int]) -> dict[int, Metadata | None]:
        """Fresh cache entries only. A present key with a None value is a known
        miss; an absent key means "not cached, go and ask"."""
        if not tmdb_ids:
            return {}
        cutoff = time.time() - CACHE_TTL
        placeholders = ",".join("?" * len(tmdb_ids))
        rows = self._db.execute(
            f"SELECT tmdb_id, payload, missing FROM metadata "
            f"WHERE tmdb_id IN ({placeholders}) AND fetched > ?",
            (*tmdb_ids, cutoff),
        ).fetchall()

        out: dict[int, Metadata | None] = {}
        for tmdb_id, payload, missing in rows:
            out[tmdb_id] = None if missing else Metadata(**json.loads(payload))
        return out

    def put_many(self, entries: dict[int, Metadata | None]) -> None:
        """One transaction for the whole batch, rather than one per film."""
        now = time.time()
        self._db.executemany(
            "INSERT OR REPLACE INTO metadata (tmdb_id, payload, missing, fetched) "
            "VALUES (?, ?, ?, ?)",
            [
                (tmdb_id, json.dumps(meta.as_dict()) if meta else None,
                 0 if meta else 1, now)
                for tmdb_id, meta in entries.items()
            ],
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class TMDBClient:
    """Fetches metadata, concurrently, through the cache.

    Credentials: TMDB has two schemes. A v3 API key goes in the query string; a
    v4 read token goes in an Authorization header. Both are still supported by
    the API, and people arrive with whichever one their account page showed
    them, so accept either rather than making that someone's first bug.
    """

    def __init__(
        self,
        api_key: str | None = None,
        read_token: str | None = None,
        cache: MetadataCache | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.read_token = read_token
        self.cache = cache
        # A semaphore, not a sleep-based rate limiter: 20 sequential requests at
        # 150 ms each is 3 seconds of page load, while 8 at a time is under 400
        # ms. The cap is there to stay a polite client, not to go slowly.
        self.semaphore = asyncio.Semaphore(concurrency)
        self._transport = transport      # tests inject a fake here

    @property
    def enabled(self) -> bool:
        """False when no credentials are configured.

        The service still works without TMDB — it returns titles and years and
        null posters. Metadata is an enhancement, not a dependency, and the
        whole app refusing to boot over a missing optional key would be wrong.
        """
        return bool(self.api_key or self.read_token)

    def _headers(self) -> dict[str, str]:
        if self.read_token:
            return {"Authorization": f"Bearer {self.read_token}"}
        return {}

    def _params(self) -> dict[str, str]:
        params = {"append_to_response": "credits"}
        if not self.read_token and self.api_key:
            params["api_key"] = self.api_key
        return params

    async def _fetch_one(
        self, client: httpx.AsyncClient, tmdb_id: int
    ) -> Metadata | None:
        """One film. Returns None for "no metadata", never raises."""
        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await client.get(
                        f"{API_BASE}/movie/{tmdb_id}", params=self._params()
                    )
                except (httpx.TimeoutException, httpx.TransportError):
                    if attempt == MAX_RETRIES - 1:
                        return None
                    await asyncio.sleep(0.5 * 2 ** attempt)   # exponential backoff
                    continue

                if response.status_code == 200:
                    try:
                        return _parse(tmdb_id, response.json())
                    except (ValueError, KeyError, TypeError):
                        return None

                if response.status_code == 429:
                    # TMDB tells us how long to wait; guessing is how you get
                    # rate limited harder.
                    wait = float(response.headers.get("Retry-After", 1))
                    await asyncio.sleep(min(wait, 10.0))
                    continue

                if response.status_code == 404:
                    return None          # cached as a miss by the caller

                if 500 <= response.status_code < 600:
                    await asyncio.sleep(0.5 * 2 ** attempt)
                    continue

                return None              # 401/403: bad key, retrying won't help
            return None

    async def fetch_many(self, tmdb_ids: list[int]) -> dict[int, Metadata | None]:
        """Metadata for a list of TMDB ids. Cache first, network for the rest."""
        wanted = [int(i) for i in dict.fromkeys(tmdb_ids) if i]   # dedupe, keep order
        if not wanted:
            return {}

        cached = self.cache.get_many(wanted) if self.cache else {}
        missing = [i for i in wanted if i not in cached]
        if not missing or not self.enabled:
            return cached

        async with httpx.AsyncClient(
            headers=self._headers(),
            timeout=REQUEST_TIMEOUT,
            transport=self._transport,
        ) as client:
            results = await asyncio.gather(
                *(self._fetch_one(client, i) for i in missing)
            )

        fetched = dict(zip(missing, results))
        if self.cache:
            self.cache.put_many(fetched)
        return {**cached, **fetched}
